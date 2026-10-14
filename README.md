# Qwen3.8-Flash-Next on a single DGX Spark

Serving Qwen3.8-Flash-Next (125B total, 6B active, 262,144 native context) on one NVIDIA DGX Spark (GB10, 121 GiB
of unified memory), written down as it was done: what ran, what it measured, and what did not work. There are two lanes. They are
different engines on different checkpoints, kept side by side.

| Lane | Engine | Checkpoint | Status |
|---|---|---|---|
| [`lanes/exl3-sparkfold`](lanes/exl3-sparkfold/README.md) | TensorFold 0.6.5 (Python) + SparkFold + 7 patches | turboderp's EXL3 4.05 bpw pack, 99.94 GiB | **served since 2026-10-09**, this build since 2026-10-11 |
| [`lanes/vllm-nvfp4`](lanes/vllm-nvfp4/README.md) | vLLM, an image carrying unmerged PRs, plus two local patches | NVIDIA's NVFP4, 123.6 GiB, n-gram table paged to SSD swap | snapshot of 2026-09-06, **not served now**, left as written |

Which one to read:

- You want the setup that is running, its memory behaviour on a shared-memory box, the NVMe spill of idle conversations, and the patches: the EXL3 lane.
- You want the vLLM route, the PLE-offload hang at TP=1 and the skinny-GEMM patch: the vLLM lane. Its traps are about vLLM and do not depend on which lane is served.

## Layout

```
lanes/
  exl3-sparkfold/     README, patches/ (+ apply.sh), scripts/start.sh, docs/, results/, tools/
  vllm-nvfp4/         README (the 2026-09-06 guide), scripts/
```

On 2026-10-14 the guide that used to be this page moved to `lanes/vllm-nvfp4/README.md` and `scripts/` moved to `lanes/vllm-nvfp4/scripts/`; its
text is unchanged apart from a note at the top. Links to its old section anchors now belong under that path.

## Licence

The vLLM lane's scripts are free to use, as its README says. The EXL3 lane's patches are Apache-2.0 derivatives of TensorFold and SparkFold
([patches/LICENSE](lanes/exl3-sparkfold/patches/LICENSE)); its scripts and tools are free to use. The model weights are not part of this
repository and keep their own licences.
