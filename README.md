# LLM serving on a single DGX Spark

Large models served on one NVIDIA DGX Spark (GB10, 121 GiB of unified memory), written down as it was done: what ran, what it measured, and
what did not work. Each lane is one model on one engine with one checkpoint, kept side by side; only one fits in memory at a time.

This repository was called `qwen3.8-flash-next-dgx-spark` until 2026-10-16, when it started to hold more than one model. Old links and clones
are redirected by GitHub, and the paths of the existing lanes did not change.

| Model | Lane | Engine | Checkpoint | Status |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (125B total, 6B active) | [`lanes/exl3-sparkfold`](lanes/exl3-sparkfold/README.md) | TensorFold 0.6.5 (Python) + SparkFold + 9 patches | turboderp's EXL3 4.05 bpw pack, 99.94 GiB | **served since 2026-10-09**, this build since 2026-10-16 |
| Qwen3.8-Flash-Next | [`lanes/vllm-nvfp4`](lanes/vllm-nvfp4/README.md) | vLLM, an image carrying unmerged PRs, plus two local patches | NVIDIA's NVFP4, 123.6 GiB, n-gram table paged to SSD swap | snapshot of 2026-09-06, **not served now**, left as written |
| Nemotron 3.5 Super VL (120B total, 12B active) | [`lanes/nemotron35-super-vl-vllm`](lanes/nemotron35-super-vl-vllm/README.md) | vLLM nightly `3f1bcfd`, NVIDIA's DGX Spark recipe | NVIDIA's NVFP4 (~75 GB) + the LiLiCorr drafter (1.3 GB) | **prepared 2026-10-16, not booted yet**: no measurements |

Which one to read:

- You want the setup that is running, its memory behaviour on a shared-memory box, the NVMe spill of idle conversations, the Korean draft
  vocabulary and the patches: the EXL3 lane.
- You want the vLLM route for Flash-Next, the PLE-offload hang at TP=1 and the skinny-GEMM patch: the vLLM NVFP4 lane. Its traps are about
  vLLM and do not depend on which lane is served.
- You want to try Nemotron 3.5 Super VL on a Spark: the Nemotron lane has the card's recipe, the checks made before booting, the traps found
  and what has been published about it so far.

## Layout

```
lanes/
  exl3-sparkfold/            Qwen3.8-Flash-Next: README, patches/ (+ apply.sh), scripts/start.sh, docs/, results/, tools/
  vllm-nvfp4/                Qwen3.8-Flash-Next: README (the 2026-09-06 guide), scripts/
  nemotron35-super-vl-vllm/  Nemotron 3.5 Super VL: README, scripts/download.sh, scripts/start.sh
```

On 2026-10-14 the guide that used to be this page moved to `lanes/vllm-nvfp4/README.md` and `scripts/` moved to `lanes/vllm-nvfp4/scripts/`; its
text is unchanged apart from a note at the top. Links to its old section anchors now belong under that path.

## Licence

The scripts of every lane are free to use. The EXL3 lane's patches are Apache-2.0 derivatives of TensorFold and SparkFold
([patches/LICENSE](lanes/exl3-sparkfold/patches/LICENSE)). The model weights are not part of this repository and keep their own licences
(Qwen's `qwen-community-1.0`, NVIDIA's OpenMDW-1.1).
