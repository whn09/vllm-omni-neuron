# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's rectified-flow Euler scheduler, ported off `ConfigMixin`.

Same arithmetic as `diffusers.MiniMaxH3Scheduler`, including the two things that make it
incompatible with a stock flow-match scheduler:

1. **The velocity points at the data**, so the ``x0`` estimate is ``x_t + sigma * v`` — a
   plus where every other flow-match scheduler in diffusers has a minus.
2. **`step` reads sigma from two sources.** The ``x0`` estimate uses ``1 - timestep``,
   while the Euler ratio uses the sigma grid. For small sigmas the float32 round trip
   ``1 - (1 - sigma)`` is not exact, and MiniMax-H3 sampled its release with these two
   spellings, so unifying them changes the output.

The timestep convention is ``t = 1 - sigma`` on ``[0, 1]`` with ``t = 1`` clean, which is
also why `scale_noise` is ``t * sample + (1 - t) * noise`` rather than the usual
``sigma``-weighted form.

MiniMax-H3 uses **two instances per request**, ``shift=12.0`` for video and ``shift=3.0``
for audio, stepping the same packed sequence at two different noise levels.

The schedule is built and kept **on the host**. It is only ever indexed by a Python `int`
and consumed as a scalar, so a device-resident grid would put that indexing into eager
Neuron ops for nothing — which is also why there is no ``device`` argument, unlike the
reference.
"""

import torch

__all__ = ["NeuronMiniMaxH3Scheduler"]


class NeuronMiniMaxH3Scheduler:
    """Rectified-flow Euler sampling for MiniMax-H3.

    Args:
        shift: The exponential sigma shift. 12.0 for video, 3.0 for audio.
    """

    order = 1

    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)
        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        """Position in the schedule, or `None` before the first `step`."""
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_shift(self, shift: float) -> None:
        """Override the sigma shift. Call before `set_timesteps`."""
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)

    def set_begin_index(self, begin_index: int = 0) -> None:
        """Start from a fixed schedule position instead of looking the timestep up."""
        self._begin_index = begin_index

    def set_timesteps(
        self, num_inference_steps: int | None = None, sigmas=None
    ) -> None:
        """Build the sigma / timestep schedule.

        The grid is ``linspace(1, 0, num_inference_steps)`` pushed through the exponential
        shift, with consecutive duplicates collapsed — at a large shift the tail of the
        grid can round to the same float32. The terminal zero is part of the requested
        count, so ``num_inference_steps`` sigmas drive ``num_inference_steps - 1`` forwards.

        Args:
            num_inference_steps: Number of sigmas, at least 2. Ignored if ``sigmas`` is given.
            sigmas: An explicit schedule: at least two strictly decreasing values ending
                exactly at 0.0.
        """
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "`set_timesteps` requires either an explicit `sigmas` schedule or "
                    f"`num_inference_steps` >= 2, got {num_inference_steps}."
                )
            base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
            sigmas = self._shift * base / (1 + (self._shift - 1) * base)
            sigmas = torch.unique_consecutive(sigmas)
        else:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            if (
                sigmas.numel() < 2
                or not bool((sigmas[1:] < sigmas[:-1]).all())
                or sigmas[-1].item() != 0.0
            ):
                raise ValueError(
                    "`sigmas` must hold at least two strictly decreasing values ending at 0.0."
                )

        self.sigmas = sigmas
        # t = 1 - sigma, and t = 1 is clean. The terminal sigma has no model evaluation.
        self.timesteps = 1.0 - sigmas[:-1]
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep) -> int:
        """Position of ``timestep`` in the schedule."""
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before stepping.")
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError(
                "Passed `timestep` is not in `self.timesteps`; pass one of "
                "`scheduler.timesteps`."
            )
        return int(indices[0].item())

    def scale_noise(
        self, sample: torch.Tensor, timestep, noise: torch.Tensor
    ) -> torch.Tensor:
        """``x_t = t * x_0 + (1 - t) * noise`` — the forward process.

        Used to noise conditioning anchors, where ``t`` is a noise-augmentation level
        rather than a schedule entry, so it is taken at face value instead of looked up.
        """
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        return_dict: bool = False,
    ):
        """One Euler step, ``eta = 0``.

        Args:
            model_output: The predicted velocity.
            timestep: The current timestep, a float from `timesteps` — *not* an index.
            sample: The current noisy sample.
            return_dict: Accepted for signature compatibility; a tuple is always returned.

        Returns:
            ``(prev_sample,)``, in ``sample``'s dtype. Float16/bfloat16 samples are
            blended in float32 first: the Euler ratio approaches 0 near the end of the
            schedule and eats the mantissa.
        """
        del return_dict
        if self.sigmas is None:
            raise ValueError("Call `set_timesteps` before stepping.")
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()
        ):
            raise ValueError(
                "Integer indices are not timesteps; pass one of `scheduler.timesteps`."
            )

        if self._step_index is None:
            self._step_index = (
                self.index_for_timestep(timestep)
                if self._begin_index is None
                else self._begin_index
            )
        if self._step_index + 1 >= self.sigmas.numel():
            raise ValueError(
                f"The schedule has {self.num_inference_steps} steps and all of them have "
                "been taken; call `set_timesteps` again."
            )

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        # Sigma source 1: from the timestep, for the x0 estimate.
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        # A plus: MiniMax-H3's velocity points at the data, not away from it.
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = (
            torch.float32
            if sample.dtype in (torch.float16, torch.bfloat16)
            else sample.dtype
        )
        # Sigma source 2: from the grid, for the Euler ratio. See the module docstring.
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(
            device=sample.device, dtype=compute_dtype
        )
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(
            dtype=compute_dtype
        )
        prev_sample = prev_sample.to(dtype=sample.dtype)

        self._step_index += 1
        return (prev_sample,)
