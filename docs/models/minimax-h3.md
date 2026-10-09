# MiniMax-H3 Model Card

<!-- meta: description: Model card for MiniMax-H3 on AWS Trainium with the vLLM Omni Neuron
plugin — joint text-to-video-and-audio generation, supported configurations, parallelism
(TP x CP), accuracy against the diffusers reference, performance, and known limitations. -->
<!-- meta: keywords: MiniMax-H3, model card, text-to-video, text-to-audio, joint audio-video
generation, diffusion, vLLM, vLLM Omni, Neuron, Trainium, trn2, BF16, tensor parallelism,
context parallelism, VAE tiling -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-09 -->

## Introduction

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) generates a video and its soundtrack
together. A 33B-parameter DiT runs 50 blocks of full self-attention over one packed sequence that
holds the prompt, the audio latents and the video latents; a Qwen3-VL-32B model conditions it, a
2.4B-parameter ViT decoder turns its video latents into frames and a BigVGAN decoder turns its
audio latents into a stereo waveform.

MiniMax-H3 text-to-video-and-audio (`t2va`), first/last-keyframe conditioning (`fl2va`) and
image/video/audio reference conditioning (`ref2va`) are supported for inference with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) on AWS Trainium2 (`trn2`).

| Model | HuggingFace | Hardware | Precision |
|-------|-------------|----------|-----------|
| MiniMax-H3 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Trn2 | BF16 DiT, FP16 video decoder, FP32 audio decoder |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-video-and-audio (`t2va`) | ✅ |
| | 1344x768, 124 frames (5.2 s at 24 fps) | ✅ |
| | First / last keyframe (`fl2va`) | ✅ |
| | Image, video and audio references (`ref2va`) | ✅ |
| | Prompt-length buckets (one graph per bucket) | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | Context Parallelism (CP) | ✅ |
| **Performance** | Spatial Tiling (VAE) | ✅ |
| | Temporal Chunking (VAE) | ✅ |
| **Compilation** | torch.compile | ✅ |

### Where each component runs

| Component | Placement |
|---|---|
| Qwen3-VL conditioner | Host, rank 0; the embeddings are broadcast to every rank. About 1 s for a text prompt. |
| Video encoder (`fl2va` keyframes, `ref2va` images and videos) | NeuronCores; the tiles are split over all ranks. |
| Audio encoder (`ref2va` soundtracks) | Host, rank 0. |
| DiT | NeuronCores, TP x CP. |
| Video decoder | NeuronCores, replicated; the tiles of the clip are split over all ranks. |
| Audio decoder | Host (default, ~1 s), or NeuronCores with `model_config.audio_vae_device: neuron`. |

### Recommended configuration

MiniMax-H3 has 56 attention heads, so `tensor_parallel_size` must divide 56; use 8. Beyond 8
NeuronCores, `ring_degree` shards the packed sequence (context parallelism):
`tensor_parallel_size=8` x `ring_degree=8` uses all 64 cores of a `trn2.48xlarge`, as in
[`minimax_h3_stage.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/examples/minimax_h3/minimax_h3_stage.yaml).

Context-parallel attention all-gathers keys and values and runs flash attention locally; it does
not use the const-max ring-attention kernel, whose softmax bound is too loose for MiniMax-H3's
QK-norm scales (see `vllm_omni_neuron/diffusion/models/minimax_h3/attention.py`).

## Usage

```bash
python examples/minimax_h3/run.py \
  --model-path MiniMaxAI/MiniMax-H3 \
  --tensor-parallel-size 8 --ring-degree 8 \
  --output minimax_h3.mp4
```

`run.py` defaults to the released recipe (1344x768, 124 frames, 50 steps) and writes an `.mp4`
with the soundtrack muxed in. `--dev` runs a 704x384, 3-step, 2-layer smoke test.

`fl2va` passes the keyframes as `multi_modal_data={"image": ..., "last_image": ...}` (either may be
left out); the canvas follows the first keyframe's aspect ratio unless `height`/`width` are given:

```bash
python examples/minimax_h3/run.py --model-path MiniMaxAI/MiniMax-H3 --image first.png --last-image last.png
```

`ref2va` uses the checkpoint's second transformer partition (`transformer_ref/`), so it runs on a
stage of its own: set `model_config: {task: ref2va}` under the stage's `engine_args` (`run.py` does
this when given `--reference`). A request passes `multi_modal_data={"references": [...]}`, in the
order the model should read them; each entry is `{"type": "image", "image": PIL.Image}`,
`{"type": "video", "frames": (T, H, W, 3) uint8, "fps": ..., "audio": (C, N) | None,
"sample_rate": ...}` or `{"type": "audio", "audio": (C, N), "sample_rate": ...}`:

```bash
python examples/minimax_h3/run.py --model-path MiniMaxAI/MiniMax-H3 \
  --reference image:character.png --reference video:motion.mp4 --reference audio:voice.wav
```

### Prompt-length buckets

The DiT graph specializes on the packed sequence length, which includes the prompt. The prompt is
left-padded to the smallest length that makes the sequence a multiple of
`model_config.text_bucket_align` (default 512), and the padding is masked out of attention with the
kernel's KV bounds, so every prompt up to that length shares one graph and the sequence stays
512-aligned. `text_bucket_align: 1` compiles a graph per exact prompt length.

## Accuracy

DiT latents after two denoising steps, against the diffusers reference run on CPU in FP32. At
704x384 the diffusers reference itself run in BF16 sets the floor; at 1344x768 (a 178-token prompt)
the same diffusers model run on NeuronCores is shown for comparison.

| Canvas | Configuration | Video rel. L2 | Video cosine | Audio rel. L2 |
|---|---|---|---|---|
| 704x384 | 8 cores (TP=8) | 7.8% | 0.9970 | 3.3% |
| 704x384 | 32 cores (TP=8 x CP=4) | 7.3% | 0.9974 | 2.8% |
| 704x384 | 64 cores (TP=8 x CP=8) | 6.4% | 0.9980 | 3.3% |
| 704x384 | Diffusers on CPU, BF16 (floor) | 9.1% | 0.9959 | 3.2% |
| 1344x768 | 64 cores (TP=8 x CP=8) | 5.5% | 0.9986 | 7.4% |
| 1344x768 | Diffusers on 64 NeuronCores, BF16 | 23.2% | 0.9730 | 11.0% |

`fl2va` (first and last keyframe) and `ref2va` (an image, a 1-second video with its soundtrack and a
2-second audio clip; an 8243-token presentation), 704x384, same protocol:

| Task | Configuration | Video rel. L2 | Audio rel. L2 |
|---|---|---|---|
| `fl2va` | 8 cores (TP=8) | 10.9% | 5.4% |
| `fl2va` | Diffusers on CPU, BF16 (floor) | 10.0% | 5.8% |
| `ref2va` | 16 cores (TP=8 x CP=2) | 13.8% | 8.1% |
| `ref2va` | Diffusers on CPU, BF16 (floor) | 12.4% | 10.1% |

Prompt buckets are exact: a 29-token prompt padded to its 58-row bucket and the same prompt
unpadded land at 9.55% and 9.56% from the FP32 reference.

Over a full 50-step generation, small numerical differences change the sample — the same scene
and motion, with drifting camera framing — so frame-level metrics against another implementation
are not a correctness test; the two-step comparison above is.

The decoders, given the same latents as the diffusers decoders (1344x768, 124 frames):

| Decoder | Against diffusers in FP32 |
|---|---|
| Video, FP16 on NeuronCores | PSNR 68.0 dB mean, 67.3 dB worst frame |
| Audio, FP32 on the host | SNR 123 dB |

## Performance

1344x768, 124 frames (5.2 s with audio), 50 steps, one `trn2.48xlarge`. Each row is the second of
two identical requests in one process (the first also pays one-time graph loading), measured end to
end at the Omni entrypoint.

| NeuronCores | Configuration | DiT s/step | Text encoder | DiT (49 steps) | Video decoder | Audio decoder | Request |
|---|---|---|---|---|---|---|---|
| 64 | TP=8 x CP=8 | 1.570 | 1.7 s | 79.2 s | 13.0 s | 1.0 s | **97.5 s** |
| 32 | TP=8 x CP=4 | 2.969 | 2.8 s | 146.8 s | 13.2 s | 1.0 s | **165.6 s** |
| 16 | TP=8 x CP=2 | 5.770 | 2.7 s | 284.0 s | 14.0 s | 0.9 s | **303.3 s** |
| 8 | TP=8 | 11.340 | 2.6 s | 557.3 s | 15.8 s | 0.9 s | **578.4 s** |
| 4 | TP=4 | — | — | — | — | — | does not fit (see below) |

The DiT scales close to linearly: 64 cores are 7.2x faster than 8.

A cold start compiles the DiT graph for the request's resolution, frame count and prompt length:
roughly 12 minutes at 64 cores and longer at fewer cores, once per geometry, then cached.

## Known limitations

- **One device is not enough at 1344x768.** At `tensor_parallel_size=4` the DiT's compiled graph
  needs 29.7 GB of HBM per core against 24 GB.
- **Each bucket is its own graph.** Prompts are grouped into length buckets (see above), but a prompt
  past its bucket, a new canvas or frame count, and every `ref2va` reference set (whose geometry
  enters the sequence) compile a new DiT graph.
- **`ref2va` prompts are long.** Every image and video reference becomes a Qwen3-VL vision block in
  the prompt — an image at its 2048-pixel short edge and a 1-second video are thousands of tokens
  each — and its latents become as many DiT rows again.
- **Cold compilation is per rank.** Every rank compiles its own NEFFs; a cold 64-core start runs
  64 compiles at once and needs on the order of 1 TB of host memory.
- **Batch size is limited to one.**
