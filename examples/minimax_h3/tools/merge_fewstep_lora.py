# SPDX-License-Identifier: Apache-2.0
"""Fold a few-step distillation LoRA for MiniMax-H3 into a copy of the checkpoint, offline.

Two supported formats:

* ``lora``: a plain LoRA in diffusers/PEFT keys (``<module>.lora_A[.default].weight`` and
  ``lora_B``). Example: ``lightx2v/Minimax-h3-Turbo``,
  ``minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors``, a 4-step model to run with
  ``--steps 5`` and ``flow_shift: 6.0``::

      python merge_fewstep_lora.py lora <MiniMax-H3> <lora.safetensors> <out_dir>

* ``pdd``: Parallel Decoding Distillation, ``alibaba-pai/MiniMax-H3-Acc-LoRAs``
  (``lora_down``/``lora_up`` plus 32 interval output heads). An 8-step model to run with
  ``--steps 9`` and ``MINIMAX_H3_PDD_HEADS=<out_dir>/pdd_heads.safetensors``::

      python merge_fewstep_lora.py pdd <MiniMax-H3> <MiniMax-H3-FL2VA-Acc-8Step.safetensors> <out_dir>

``W += (alpha / rank) * B @ A`` is computed in fp32 and cast back to each tensor's dtype. Shapes and
dtypes are unchanged, so the compiled graphs, and their NEFF cache, are the same as the base model's.
Every other component of the base model is symlinked, so ``<out_dir>`` is a drop-in ``--model-path``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _link_components(base: str, out: str) -> None:
    os.makedirs(os.path.join(out, "transformer"), exist_ok=True)
    for name in os.listdir(base):
        if name != "transformer" and not os.path.exists(os.path.join(out, name)):
            os.symlink(os.path.abspath(os.path.join(base, name)), os.path.join(out, name))


def _merge(base: str, out: str, pairs: dict[str, tuple[str, str]], lora, scale: float) -> int:
    """Rewrite every transformer shard with ``W += scale * B @ A`` for each LoRA target module."""
    src = os.path.join(base, "transformer")
    index = json.load(open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json")))
    for name in os.listdir(src):
        if not name.endswith(".safetensors"):
            shutil.copy(os.path.join(src, name), os.path.join(out, "transformer", name))
    merged = 0
    for shard in sorted(set(index["weight_map"].values())):
        start = time.time()
        tensors = {}
        with safe_open(os.path.join(src, shard), "pt") as handle:
            shard_meta = handle.metadata()
            for key in handle.keys():
                weight = handle.get_tensor(key)
                module = key[: -len(".weight")] if key.endswith(".weight") else None
                if module in pairs:
                    down, up = pairs[module]
                    delta = lora.get_tensor(up).float() @ lora.get_tensor(down).float()
                    weight = (weight.float() + scale * delta).to(weight.dtype)
                    merged += 1
                tensors[key] = weight.contiguous()
        save_file(tensors, os.path.join(out, "transformer", shard), metadata=shard_meta)
        print(f"{shard}: {time.time() - start:.0f}s, merged so far {merged}", flush=True)
    return merged


def merge_lora(base: str, lora_path: str, out: str) -> None:
    _link_components(base, out)
    lora = safe_open(lora_path, "pt")
    pairs = {}
    for key in lora.keys():
        match = re.match(r"(?:transformer\.|diffusion_model\.)?(.*)\.lora_A(?:\.default)?(?:\.weight)?$", key)
        if match:
            pairs[match.group(1)] = (key, key.replace("lora_A", "lora_B"))
    if not pairs:
        raise SystemExit("no lora_A/lora_B pairs found; is this a PDD file? use `pdd`")
    rank = lora.get_slice(next(iter(pairs.values()))[0]).get_shape()[0]
    scale = float((lora.metadata() or {}).get("alpha", rank)) / rank
    print(f"{len(pairs)} LoRA pairs, rank {rank}, scale {scale}")
    merged = _merge(base, out, pairs, lora, scale)
    if merged != len(pairs):
        raise SystemExit(f"merged {merged}/{len(pairs)}: some LoRA targets were not found")
    print(f"merged {merged}/{len(pairs)}")


def merge_pdd(base: str, lora_path: str, out: str) -> None:
    _link_components(base, out)
    lora = safe_open(lora_path, "pt")
    meta = lora.metadata()
    scale = float(meta["lora_alpha"]) / int(meta["lora_rank"])
    heads = {
        key: lora.get_tensor(key)
        for key in ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias")
    }
    save_file(heads, os.path.join(out, "pdd_heads.safetensors"), metadata=meta)
    pairs = {
        key[: -len(".lora_down")]: (key, key[: -len(".lora_down")] + ".lora_up")
        for key in lora.keys()
        if key.endswith(".lora_down")
    }
    merged = _merge(base, out, pairs, lora, scale)
    if merged != len(pairs):
        raise SystemExit(f"merged {merged}/{len(pairs)}: some LoRA targets were not found")
    print(f"merged {merged}/{len(pairs)} LoRA pairs, scale {scale}; heads -> {out}/pdd_heads.safetensors")


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ("lora", "pdd"):
        raise SystemExit(__doc__)
    torch.set_num_threads(os.cpu_count() or 1)
    {"lora": merge_lora, "pdd": merge_pdd}[sys.argv[1]](*sys.argv[2:])
