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

MiniMax-H3 text-to-video-and-audio (`t2va`) is supported for inference with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) on AWS Trainium2 (`trn2`).

| Model | HuggingFace | Hardware | Precision |
|-------|-------------|----------|-----------|
| MiniMax-H3 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Trn2 | BF16 DiT, FP16 video decoder, FP32 audio decoder |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-video-and-audio (`t2va`) | ✅ |
| | 1344x768, 124 frames (5.2 s at 24 fps) | ✅ |
| | Keyframe (`fl2va`) / reference (`ref2va`) conditioning | Not yet |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | Context Parallelism (CP) | ✅ |
| **Performance** | Spatial Tiling (VAE) | ✅ |
| | Temporal Chunking (VAE) | ✅ |
| **Compilation** | torch.compile | ✅ |

### Where each component runs

| Component | Placement |
|---|---|
| Qwen3-VL conditioner | Host, rank 0; the embeddings are broadcast to every rank. About 1 s per request. |
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
- **Each prompt length is its own graph.** The packed sequence includes the prompt tokens, and the
  attention takes no mask, so a new prompt length compiles a new DiT graph.
- **Cold compilation is per rank.** Every rank compiles its own NEFFs; a cold 64-core start runs
  64 compiles at once and needs on the order of 1 TB of host memory.
- **Batch size is limited to one.**
