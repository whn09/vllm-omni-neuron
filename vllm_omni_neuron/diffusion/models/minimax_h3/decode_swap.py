# SPDX-License-Identifier: Apache-2.0
"""Make room for the video decoder by evicting only part of the DiT, from a pinned host copy.

`MINIMAX_H3_SWAP_FOR_DECODE=1` (small instances) moves the whole DiT to the host before the decode and
back after it, copying ~10 GB per core down and up per request. Two observations make most of that
unnecessary:

* The DiT's weights never change between requests, so the device-to-host copy is redundant: keep a
  permanent host copy, and eviction becomes dropping the device storage.
* The decoder needs only ~5 GB per core next to a DiT that already fills ~21 of 24 GiB, so evicting a
  tail of the block stack is enough.

`MINIMAX_H3_EVICT_BLOCKS` selects how many trailing blocks to evict (``all`` for the whole DiT, which is
the same footprint as the full swap, minus the device-to-host copy). The tail rather than the head,
so a prefetch can overlap the next request's first blocks.

`MINIMAX_H3_RELOAD=after_decode` (default) copies the evicted blocks back right after the decode, as the
full swap does; `MINIMAX_H3_RELOAD=next_request` defers the copy to a background thread started when the
next request begins, which overlaps it with the host-side prompt encode, and joins it before the first
evicted block runs.

Correctness: parameters are replaced the way `Module.to` replaces them across device types (a new
`Parameter` per slot), which the per-block `torch.compile` graphs already tolerate under the full swap.
The PDD output heads are not in the evicted tail.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import torch
from torch import nn

logger = logging.getLogger(__name__)

EVICT_ENV = "MINIMAX_H3_EVICT_BLOCKS"
RELOAD_ENV = "MINIMAX_H3_RELOAD"


def evict_blocks_setting() -> str | None:
    value = os.environ.get(EVICT_ENV, "").strip().lower()
    return value or None


class TailEvictor:
    """Evict the last ``k`` transformer blocks to a pinned host copy and bring them back."""

    def __init__(self, transformer: nn.Module, setting: str, device: torch.device):
        if setting == "all_module":
            # The whole of ``transformer`` (any module, e.g. the video decoder).
            self.modules = [transformer]
            self.first_block = 0
        elif setting == "all":
            self.modules = [transformer]
            self.first_block = 0
        else:
            blocks = transformer.transformer_blocks
            k = int(setting)
            if not 0 < k <= len(blocks):
                raise ValueError(f"{EVICT_ENV}={setting}: need 1..{len(blocks)} or 'all'.")
            self.first_block = len(blocks) - k
            self.modules = list(blocks[self.first_block :])
        self.device = device
        self.reload_mode = os.environ.get(RELOAD_ENV, "after_decode").strip().lower()
        if self.reload_mode not in {"after_decode", "next_request"}:
            raise ValueError(f"{RELOAD_ENV}={self.reload_mode}: need after_decode or next_request.")
        self._host: dict[int, torch.Tensor] = {}
        self._slots_list: list[tuple[nn.Module, str, bool]] = []
        self._resident = True
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.last_reload_seconds = 0.0
        self.last_wait_seconds = 0.0
        self.host_bytes = 0

    def _slots(self) -> list[tuple[nn.Module, str, bool]]:
        """``(owner, name, is_parameter)`` for every parameter and buffer of the evicted modules."""
        slots, seen = [], set()
        for module in self.modules:
            for owner in module.modules():
                for name, value in owner._parameters.items():
                    if value is not None and (id(owner), name) not in seen:
                        seen.add((id(owner), name))
                        slots.append((owner, name, True))
                for name, value in owner._buffers.items():
                    if value is not None and (id(owner), name) not in seen:
                        seen.add((id(owner), name))
                        slots.append((owner, name, False))
        return slots

    def _set(self, owner: nn.Module, name: str, is_parameter: bool, tensor: torch.Tensor) -> None:
        # A Neuron tensor and a CPU tensor have different TensorImpl types, so `param.data = ...`
        # across them is refused; replace the Parameter, which is what `Module.to` itself does.
        if is_parameter:
            owner._parameters[name] = nn.Parameter(tensor, requires_grad=False)
        else:
            owner._buffers[name] = tensor

    @torch.no_grad()
    def snapshot(self) -> None:
        """Take the permanent host copy. Call once, with the modules where they normally live."""
        self._slots_list = self._slots()
        pin = os.environ.get("MINIMAX_H3_EVICT_PIN", "0") == "1"
        for owner, name, is_parameter in self._slots_list:
            value = owner._parameters[name] if is_parameter else owner._buffers[name]
            host = value.detach().to("cpu").clone()
            if pin:
                try:
                    host = host.pin_memory()
                except RuntimeError:
                    pass
            self._host[(id(owner), name)] = host
        self.host_bytes = sum(h.numel() * h.element_size() for h in self._host.values())
        logger.info(
            "MiniMax-H3 decode swap: %d tensors from block %d on (%.2f GB per rank) kept on the host.",
            len(self._slots_list),
            self.first_block,
            self.host_bytes / 1e9,
        )

    @torch.no_grad()
    def evict(self) -> None:
        """Drop the device storage. No device-to-host copy: the host copy is reused."""
        self.wait()
        if not self._slots_list:
            self.snapshot()
        for owner, name, is_parameter in self._slots_list:
            self._set(owner, name, is_parameter, self._host[(id(owner), name)])
        self._resident = False

    @torch.no_grad()
    def _reload(self) -> None:
        started = time.perf_counter()
        try:
            for owner, name, is_parameter in self._slots_list:
                self._set(owner, name, is_parameter, self._host[(id(owner), name)].to(self.device))
            self._resident = True
        except BaseException as error:  # surfaced by `wait`
            self._error = error
        self.last_reload_seconds = time.perf_counter() - started

    def reload(self) -> None:
        """Copy the tail back now (blocking)."""
        if not self._resident:
            self._reload()
            self._raise()

    def start_prefetch(self) -> None:
        """Start the reload in a background thread (no-op if resident or already running)."""
        if self._resident or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._reload, name="h3-dit-reload", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        """Join a running prefetch; reload synchronously if none ran."""
        started = time.perf_counter()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if not self._resident:
            self._reload()
        self.last_wait_seconds = time.perf_counter() - started
        self._raise()

    def _raise(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise error
