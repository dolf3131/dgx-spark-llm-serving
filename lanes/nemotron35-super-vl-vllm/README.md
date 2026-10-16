# Nemotron 3.5 Super VL 120B-A12B on one DGX Spark: NVFP4 + LiLiCorr on vLLM

**Status: prepared on 2026-10-16, not booted yet.** The weights are downloaded and checksummed, the image is pulled and its contents checked,
and the start script is NVIDIA's DGX Spark recipe. Nothing below is a measurement of ours on this model. The lane needs the whole Spark, so
the first boot waits until the Qwen lane can be stopped for a test.

## TL;DR

| | |
|---|---|
| Hardware | 1x DGX Spark (GB10, 121 GiB unified memory) |
| Target | [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) @ `82d93699`, 88 shards, 70 GiB (~75 GB) |
| Drafter | [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr) @ `7c7221ae`, one 1.3 GB file |
| Engine | vLLM `0.32.0rc1.dev190+g3f1bcfd82`, image `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2` (arm64, 10.7 GB) |
| Serving | the card's Spark recipe: one sequence, `--gpu-memory-utilization 0.75` (~91 GiB), LiLiCorr drafts of 7 tokens (`--speculative_config.method dflash`), FlashInfer Mamba backend, fp16 Mamba state with stochastic rounding, `qwen3_coder` tool parser, `nemotron_v3` reasoning parser |
| Context | no `--max-model-len` in the recipe, so the model's 1,048,576 tokens; only 8 of 88 layers are attention (2 KV heads x 128), about 8 KiB of bf16 KV per token, 8 GiB for 1M tokens |
| Thinking | on by default in the chat template at full effort; `medium_effort` and `enable_thinking=False` are opt-in per request |
| Measured here | nothing yet |

## What is in this directory

| | |
|---|---|
| [scripts/download.sh](scripts/download.sh) | both repositories at the pinned revisions, then `hf cache verify` against the Hub |
| [scripts/start.sh](scripts/start.sh) | the `docker run`: the card's flags unchanged, plus the environment wrapping listed in its header |

## What was checked (2026-10-16)

- **Weights.** `hf cache verify` at the pinned revisions passed: 114 files of the target, 5 of the drafter.
- **The engine reaches the card's configuration.** The image is built from vLLM commit `3f1bcfd` (2026-10-15). At that commit the model registry
  maps the target's `NemotronH_Omni_Reasoning_V3` to `nano_nemotron_vl` and the drafter's `LiLiCorrDraftModel` to `lilicorr.py`, and
  `docs/features/speculative_decoding/lilicorr.md` documents the drafter (it needs Model Runner V2, which vLLM selects by itself). Inside the
  image: `lilicorr.py`, `nano_nemotron_vl.py`, `nemotron_h.py` and `nemotron_h_mtp.py` are present, the `nemotron_v3` reasoning parser and
  the `qwen3_coder` tool parser are registered, and every flag of the recipe exists as a config field.
- **Draft length.** The drafter's `block_size` is 8, and vLLM's LiLiCorr document asks for `num_speculative_tokens = block_size - 1`: the
  card's 7 matches. The drafter carries its own NVFP4 `lm_head` (`has_own_lm_head: true`) and reads target layers 1, 22, 43, 64 and 85.
- **The start script** passes the same vLLM arguments as the card (compared programmatically) and prints the full command with `DRY_RUN=1`.

## Reproduce

1. **Prerequisites.** A DGX Spark with nothing else holding GPU memory, Docker with the NVIDIA runtime, the `hf` CLI, about 80 GB free.
2. **Weights.** `scripts/download.sh` (or `MODELS_DIR=... scripts/download.sh`).
3. **Image.** `docker pull vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2`.
4. **Start.** `scripts/start.sh`, then `docker logs -f nemotron-vl` and `curl -s localhost:8000/v1/models`. The API model name is
   `nemotron-3.5-super-vl`; the card's [API client examples](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4#api-client)
   cover text, images, video and tool calls.

## Traps found while preparing

- **`v0.32.0` is not on Docker Hub** (checked 2026-10-16). The card names it as the alternative to the nightly; use the nightly.
- **The card's Quick Start breaks when pasted into bash**: the line `--speculative_config.num_speculative_tokens 7 \` is followed by a tab, so the
  backslash escapes the tab instead of the newline, the command ends there, and the remaining lines run as separate commands. `start.sh` passes
  the arguments as an array.
- **`vllm serve --help` fails in a container without a GPU** ("Failed to infer device type"). We checked the flags in the installed source
  instead, to avoid attaching the GPU next to a running server. The `No module named 'vllm._C'` line printed on the way is harmless: in this
  version the module is `_C_stable_libtorch.abi3.so`.
- **Page cache counts as used memory on GB10.** Right after the download most of the 75 GB sits in page cache, and CUDA reports it as taken, so
  vLLM would fail its `0.75` free-memory check. `start.sh` drops the pack's page cache first.
- **The download is slow to finish on Wi-Fi.** About 12 MB/s here (the image took 126 minutes, the weights about two hours), and the 75 GB of
  writes evicted the page cache of the server running next to it, which probably made its next restart slower (262 s instead of ~100 s).

## What others report (2026-10-16, the day after release)

No DGX Spark token rates for this model were published anywhere we looked: the NVIDIA forum thread has only the announcement, the NVFP4 and
LiLiCorr repositories on Hugging Face have no discussions, and we found no community reviews.

- **Artificial Analysis Intelligence Index v4.3.2**: 26 for Nemotron 3.5 Super VL. On the same scale the previous Nemotron 3 Super 120B A12B
  scores 13, Nemotron 3 Ultra 550B 23, and Qwen3.8-Flash-Next (the model of the [EXL3 lane](../exl3-sparkfold/README.md)) 40. The index is measured
  on the providers' own deployments, not on these quantized packs, and Artificial Analysis lists no output speed for Super VL yet.
- **Speed claims (NVIDIA)**: LiLiCorr gives "up to 2.3x" faster generation than no speculation on DGX Spark. The drafter card reports a mean
  accepted length of 4.74 at temperature 0 and 4.36 at temperature 1 with 7 drafts, on SPEED-Bench at 64 concurrent requests, which is not the
  single-stream case of this recipe. The "600+ tok/s" in NVIDIA's announcement is four B200 GPUs behind Baseten's endpoint.
- **NVFP4 against BF16 (the card's own table)**: most scores are within a point, but Terminal-Bench 2.1 66.7 to 60.8, WebVoyager 77.7 to 66.3,
  DeepSWE 24.7 to 18.2 and MMLU-ProX 76.9 to 73.8.
- **Languages**: the target card lists English, French, German, Italian, Japanese, Spanish and Chinese as post-training languages; Korean appears
  only in the pre-training crawl. The drafter card lists Korean among supported languages.

Reference point for decode, from a different model: the previous Nemotron 3 Super 120B-A12B NVFP4 (the same 120B-A12B size) ran on this
Spark in vLLM at 20.2 tok/s on code and 16.9 on prose in our first setup, and at about 29-30 on code and 25 on prose with vLLM 0.25-0.26
(2026-08-09). By parameter count a Super VL decode token reads roughly 6.8 GB of weights (12B active at ~4.5 bits) against 5.28 GB for the
EXL3 lane's pack, so without drafting it should be slower per step; how much LiLiCorr recovers is what the first test measures.

## Not measured yet

Boot time and memory headroom, decode in English and Korean with the [EXL3 lane's tools](../exl3-sparkfold/tools/), prefill by prompt length,
LiLiCorr acceptance at one stream, tool calling, Korean output quality, image input.

## Licence

The scripts are free to use. The weights are not part of this repository; both checkpoints are under NVIDIA's
[OpenMDW-1.1](https://openmdw.ai/license/1-1/) licence.
