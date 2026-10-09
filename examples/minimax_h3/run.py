# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 text-to-video-and-audio on Neuron via the Omni entrypoint.

Usage:
    python examples/minimax_h3/run.py --model-path /path/to/MiniMax-H3            # 64 cores
    python examples/minimax_h3/run.py --model-path ... --dev                      # quick smoke
    python examples/minimax_h3/run.py --model-path ... --tensor-parallel-size 8 --ring-degree 4
    python examples/minimax_h3/run.py --model-path ... --image first.png [--last-image last.png]  # fl2va
    python examples/minimax_h3/run.py --model-path ... --reference image:a.png --reference video:b.mp4  # ref2va
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import wave

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import yaml
from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

parser = argparse.ArgumentParser(description="MiniMax-H3 t2va on Neuron")
parser.add_argument("--model-path", type=str, default="MiniMaxAI/MiniMax-H3")
parser.add_argument("--dev", action="store_true", help="Small canvas, 2 steps, 2 layers.")
parser.add_argument("--tensor-parallel-size", type=int, default=8)
parser.add_argument("--ring-degree", type=int, default=8, help="Context-parallel degree.")
parser.add_argument("--num-layers", type=int, default=None, help="Override the DiT depth.")
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--num-frames", type=int, default=None)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stage-config", type=str, default=None)
parser.add_argument("--output", type=str, default="minimax_h3.mp4")
parser.add_argument("--first-device", type=int, default=0, help="First NeuronCore of the stage.")
parser.add_argument("--image", type=str, default=None, help="fl2va: keyframe the video starts from.")
parser.add_argument("--last-image", type=str, default=None, help="fl2va: keyframe the video ends on.")
parser.add_argument(
    "--reference",
    type=str,
    action="append",
    default=None,
    help="ref2va, in the order the model reads them: image:PATH, video:PATH (with its soundtrack) or audio:PATH.",
)
parser.add_argument("--repeat", type=int, default=1, help="Send the request N times, timing each.")
parser.add_argument(
    "--prompt",
    type=str,
    action="append",
    default=None,
    help="May be repeated: the prompts are sent in order, each --repeat times, in one process.",
)
DEFAULT_PROMPT = (
    "A street busker plays a bright melody on a violin under warm evening light, "
    "passers-by slowing to listen, shallow depth of field, cinematic"
)
args = parser.parse_args()

world_size = args.tensor_parallel_size * args.ring_degree
env_profiles.apply(env_profiles.MINIMAX_H3, env_profiles.thread_limits(world_size))


def _stage_config() -> str:
    """The stage yaml, with its parallel layout and device range rewritten from the flags."""
    path = args.stage_config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax_h3_stage.yaml")
    with open(path) as handle:
        config = yaml.safe_load(handle)
    stage = config["stage_args"][0]
    stage["runtime"]["devices"] = f"{args.first_device}-{args.first_device + world_size - 1}"
    parallel = stage["engine_args"].setdefault("parallel_config", {})
    parallel["tensor_parallel_size"] = args.tensor_parallel_size
    parallel["ring_degree"] = args.ring_degree
    out = os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", f".minimax_h3_stage_{args.first_device}.yaml")
    with open(out, "w") as handle:
        yaml.safe_dump(config, handle)
    return out


FPS = 24


def _frames(video):
    """``(T, H, W, 3)`` float in [0, 1] from either output layout vllm-omni hands back."""
    import numpy as np

    video = video.detach().cpu().float().numpy() if hasattr(video, "detach") else np.asarray(video)
    if video.ndim == 5:
        video = video[0]
    if video.shape[0] == 3 and video.shape[-1] != 3:
        video = np.transpose(video, (1, 2, 3, 0))
    return np.clip(video.astype(np.float32), 0.0, 1.0)


def write_outputs(video, audio, sampling_rate: int, path: str) -> None:
    """Mux the frames and the stereo waveform into one ``.mp4``."""
    import numpy as np
    from diffusers.utils import export_to_video

    frames = _frames(video)
    waveform = audio.detach().cpu().float().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    waveform = waveform.reshape(-1, waveform.shape[-1]).T  # (samples, channels)
    samples = (np.clip(waveform, -1.0, 1.0) * 32767.0).round().astype(np.int16)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as directory:
        silent, wav = os.path.join(directory, "video.mp4"), os.path.join(directory, "audio.wav")
        export_to_video(list(frames), silent, fps=FPS)
        with wave.open(wav, "wb") as handle:
            handle.setnchannels(samples.shape[1])
            handle.setsampwidth(2)
            handle.setframerate(sampling_rate)
            handle.writeframes(samples.tobytes())
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-shortest", path],
            check=True,
        )
    print(f"Wrote {frames.shape[0]} frames ({frames.shape[2]}x{frames.shape[1]}) and "
          f"{samples.shape[0] / sampling_rate:.2f} s of audio to {path}")


def load_references(specs):
    """``kind:path`` specs -> the request's ``references`` list, in order.

    Paths, not decoded media: the request is copied to every worker, and each rank decodes the
    files itself (see `references`).
    """
    references = []
    for spec in specs:
        kind, _, path = spec.partition(":")
        if kind not in ("image", "video", "audio") or not path:
            raise SystemExit(f"--reference must be image:PATH, video:PATH or audio:PATH, got {spec!r}.")
        references.append({"type": kind, "path": os.path.abspath(path)})
    return references


def main():
    model_config = {}
    if args.reference:
        model_config["task"] = "ref2va"
    if args.num_layers is not None or args.dev:
        model_config["num_layers"] = args.num_layers or 2

    # A cold compile of the DiT exceeds vllm-omni's 600 s worker handshake.
    timeout = int(os.environ.get("MINIMAX_H3_HANDSHAKE_TIMEOUT_S", "7200"))
    import vllm_omni.diffusion.stage_diffusion_proc as stage_proc

    stage_proc._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(stage_proc, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)

    omni = Omni(
        model=args.model_path,
        stage_configs_path=_stage_config(),
        stage_init_timeout=timeout,
        init_timeout=timeout,
        model_config=model_config,
    )

    # The released recipe: 1344x768, 124 frames (5.17 s at 24 fps), 50 steps. With a keyframe
    # the canvas follows its aspect ratio unless --height/--width are given.
    multi_modal_data = {}
    if args.image:
        multi_modal_data["image"] = Image.open(args.image).convert("RGB")
    if args.last_image:
        multi_modal_data["last_image"] = Image.open(args.last_image).convert("RGB")
    if args.reference:
        multi_modal_data["references"] = load_references(args.reference)
    if args.dev:
        height, width, num_frames, steps = 384, 704, 124, 3
    else:
        # With a keyframe the canvas follows its aspect ratio; ref2va defaults to 16:9.
        keyframed = "image" in multi_modal_data or "last_image" in multi_modal_data
        height, width, num_frames, steps = (None, None, 124, 50) if keyframed else (768, 1344, 124, 50)
    params = OmniDiffusionSamplingParams(
        height=args.height or height,
        width=args.width or width,
        num_frames=args.num_frames or num_frames,
        num_inference_steps=args.steps or steps,
        seed=args.seed,
    )

    prompts = args.prompt or [DEFAULT_PROMPT]
    for prompt_index, prompt in enumerate(prompts):
        for index in range(args.repeat):
            started = time.perf_counter()
            inputs = {"prompt": prompt}
            if multi_modal_data:
                inputs["multi_modal_data"] = multi_modal_data
            result = omni.generate(inputs, params)
            print(f"[prompt {prompt_index + 1}/{len(prompts)} request {index + 1}/{args.repeat}] "
                  f"{time.perf_counter() - started:.2f} s")

    output = result[0].request_output
    audio = (output.multimodal_output or {}).get("audio")
    if not output.images or audio is None:
        print(f"No video+audio to write (images={len(output.images or [])}, audio={audio is not None}).")
        return
    write_outputs(output.images[0], audio, (output.custom_output or {}).get("sampling_rate", 32000), args.output)


if __name__ == "__main__":
    main()
