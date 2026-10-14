# Qwen3.8-Flash-Next on one DGX Spark: EXL3 4.05 bpw on TensorFold / SparkFold

What one Spark serves here since 2026-10-09 (this build since 2026-10-11): turboderp's EXL3 4.05 bpw pack of Qwen3.8-Flash-Next on the
TensorFold 0.6.5 Python engine, with [SparkFold](https://github.com/grearjake-star/TensorFold)'s GB10 work merged in and seven patches of
ours on top. Five concurrent streams of 262,144 tokens each, image input, MTP drafting, replies equal to plain decoding.

The engine is a snapshot: TensorFold froze its Python engine (see [Newer upstream work](#newer-upstream-work)), so this lane is pinned
to a commit and does not follow a release line. Measurements are dated; read [Limits](#limits-and-open-items) before relying on any of them.

## TL;DR

| | |
|---|---|
| Hardware | 1x DGX Spark (GB10, 121 GiB unified memory) |
| Checkpoint | [`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), branch `4.05bpw_h6_ng6`, 99.94 GiB: trunk 62.9 GiB (on the GPU), consolidated n-gram table 36.36 GiB (memory-mapped), vision sidecar 0.52 GiB, MTP head included |
| Per decoded token | 5.28 GB read (by header arithmetic), against 7.09 GB for an NVFP4 QAD build we served before (a different checkpoint from the one the vLLM lane uses) |
| Engine | TensorFold 0.6.5 (Python) + SparkFold `806b345` + [7 patches](patches/), source mounted over the image `tensorfold-qwen38:v0.3.6.3` |
| Serving | `--parallel 5 --context 262144 --kv-dtype int8 --mtp-drafts 6 --mtp-confidence 0.60 --thinking --vision`, memory floor 6 GiB, NVMe spill on |
| Decode | single stream, 600-token windows: English 57.4-57.9, Korean 41.8-43.3 tok/s; 128-token windows on short prompts: median 71 tok/s |
| Cold prefill | 1.8-2.1k tok/s from 1.9k to 49k prompt tokens (real English/Korean text) |
| Resuming a 100k-token conversation after an eviction | 40-50 s from scratch, 0.8-2.3 s from NVMe (controlled replay) |
| Cold start | about 90 s once the extensions are built; the first start compiles three extensions (about three minutes) |

Quality: the pack's card reports a KLD of 0.0067 against BF16 at 4.05 bpw (turboderp's measurement on an English-leaning trace). We have not
measured quality ourselves.

## What is in this directory

| | |
|---|---|
| [patches/](patches/) | the 7 patches, `apply.sh` to build the tree, licence and authorship notes |
| [scripts/start.sh](scripts/start.sh) | the `docker run` for this lane, with profiles and knobs |
| [docs/](docs/) | [the memory gate](docs/memory-gate.md), [NVMe spill](docs/nvme-spill.md), [quoted special tokens](docs/quoted-special-tokens.md) |
| [results/](results/) | dated measurements |
| [tools/](tools/) | decode benchmarks, and the replays behind the memory-gate and spill numbers |

## Reproduce

1. **Prerequisites.** A DGX Spark, Docker with the NVIDIA runtime, git, about 110 GiB free for the pack plus room for caches and the spill
   directory (48 GiB at most).
2. **The pack.**
   ```bash
   HF_HUB_DISABLE_XET=1 hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6 \
       --local-dir ~/models/qwen3.8-flash-next-exl3-4.05bpw
   ```
   Ours was 27 files, 99.94 GiB; the SHA-256 of all 14 LFS files matched the Hub.
3. **The image.** Clone [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
   and run its `scripts/prepare.sh`: step 2 builds (or pulls) `tensorfold-qwen38:v0.3.6.3`, TensorFold v0.3.6.3 plus their patches on NVIDIA's
   PyTorch 26.07 container. Steps 3 and 4 of that script fetch that wrapper's own MLX checkpoint, which this lane does not use. The image only
   supplies PyTorch and the dependencies: the TensorFold source mounted in step 4 replaces the code inside it. Ours carries the label
   `tf.patches=c1f5d72f8d16`.
4. **The patched tree.** `patches/apply.sh ~/tensorfold-src` (SparkFold at `806b345` plus the seven patches).
5. **Vision (optional).** Convert the pack's `vision_k6.safetensors` once with `python -m tensorfold.vision.exl3_convert` (run it in the image with
   `PYTHONPATH` at the tree; see the module's `--help` for its arguments) and keep the result as `$CACHE/vision-f16-v3.safetensors`
   (898 MB). Or start with `VISION=0`. We ran this conversion once, on 2026-10-09.
6. **Start.**
   ```bash
   TF_SRC=~/tensorfold-src ./scripts/start.sh        # port 8888 on 127.0.0.1; see the header of start.sh for every variable
   curl -s localhost:8888/health
   ```
   The startup log should show `lone-stream graph slot: 65536 rows`, `spill: N stored states adopted`, the n-gram lock line (about 20-25 GiB of
   pages locked) and `serving qwen3.8-flash-next ... loaded in ~90s`.

The server has no API key. `start.sh` publishes the port on 127.0.0.1; `BIND=` (empty) publishes it on every interface, which is what a
shared box needs and what you must not do on an untrusted network.

## Settings the lane runs (`PROFILE=spark`)

| Knob | Value | Meaning |
|---|---|---|
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 6 | memory floor of the stream-cache gate (stock: max(4, RAM/10) = 12.2): see [the memory gate](docs/memory-gate.md) |
| `TENSORFOLD_PIN_EXTRA_GIB` | -6.2 | keeps the n-gram lock near 20-25 GiB although the floor moved (lock room = budget - estimate) |
| `TF_SPILL_DIR` / `TF_SPILL_GIB` / `TF_SPILL_MIN` | /cache/spill / 48 / 16384 | [NVMe spill](docs/nvme-spill.md); `SPILL=0` turns it off |
| `TF_SOLO_ROWS` | 65536 | lone-stream graph slot pre-sized to 64K rows, buckets captured at start (1.1 GiB) |
| `TF_KEEP` / `TF_FIRST_PASS` / `TF_PASS_MIN` | 16 / 1024 / 1024 | kept states per pool, prompt pass sizes beside live streams |
| `TENSORFOLD_PREFILL_ROWS` / `TF_EXL3_MOE_WINDOW` | 4096 / 4096 | prompt pieces and routed-expert windows |

Left at the tree's defaults: `TF_DRAFT_COST` 0.06, `TF_MULTI_GRAPHS`, fused routing, stage batch, decode and n-gram read-ahead; `TF_NGRAM_LOCK`
unset; `TF_EXL3_HC_FP8` off (it changes output bits). `PROFILE=safe` pins the draft cost and shared-round graphs back to the old policy.

## Measurements

| What | Result | Where |
|---|---|---|
| Cold prefill, real text, 1.9k-49k tokens | 1,830-2,067 tok/s (earlier local build: 1,341-1,374) | [results/2026-10-11](results/2026-10-11-sparkfold-port.md) |
| Decode, English / Korean | 57.4 / 43.3 tok/s (earlier build: 42.6 / 33.6 that day, 59.6 / 31.2 the day before: no gain established) | same |
| Three long conversations taking turns | stock floor 7 of 8 turns re-prefilled, floor 6 GiB 0 of 9 | [memory gate](docs/memory-gate.md) |
| Spill, 98k-token conversation | 51.9 s cold, 1.0-2.3 s read back, identical replies | [NVMe spill](docs/nvme-spill.md) |
| `TF_DRAFT_COST` 0.04 / 0.08, `TF_JOINT_PRICE=1` | no gain at one stream; joint pricing -5% at three | [results/2026-10-14](results/2026-10-14-draft-knobs.md) |
| Prefill and decode by context depth (0 / 4k / 8k) and streams (1 / 2 / 4), llama-benchy | prefill 1.4-1.9k tok/s on a 2k-token piece (1.7-1.9k at 4-8k depth); decode 51-60 tok/s at one stream, 99.5 at four (depth 0); the last two rows overlapped another job | [results/2026-10-14 perf](results/2026-10-14-perf-by-length-and-depth.md) |
| Tool calling, tool-eval-bench (97 scenarios) | 90/100 merged (88 for the full run, structured output fixed by installing `xgrammar`); 3 safety-critical failures | [results/2026-10-14 tool-eval](results/2026-10-14-tool-eval.md) |

Cold prefill here is not faster than the 2,719 tok/s at 30k tokens that the vLLM NVFP4 lane in this repository measured with its own script.
The two were not measured with the same script or on the same checkpoint, so do not subtract them.

## Limits and open items

- **Frozen engine.** Nothing more merges into TensorFold's Python engine. This lane gets SparkFold's fixes if we merge them, and nothing else.
- **No API key, no restart policy.** The container runs with `--restart no` (a crash stays down) and at the default `oom_score_adj`, so
  `earlyoom` can pick it first; the memory floor of 6 GiB narrows the margin to that kill line ([memory gate](docs/memory-gate.md)).
- **An unexplained slowdown.** After 19 hours of uptime the earlier local build prefilled at 439-1,062 tok/s instead of 1,341-1,374. Not page
  faults, swap, clocks, core placement or n-gram residency. We did not measure the current build after a comparable uptime.
- **Spill has not been shown on real traffic**, only in controlled replays ([NVMe spill](docs/nvme-spill.md)).
- **Other jobs move the numbers.** The CPU and GPU share one memory, and a CPU-heavy job next to the server overlapped the last rows of the
  2026-10-14 perf run. An earlier run that afternoon, on the container before `xgrammar` was added, had five of six deeper rows at 964-1,358 tok/s
  against 1,724-1,903 on the idle rerun; we did not record what else was running then and have not found the cause. Run benchmarks on an idle machine.
- **A restart deletes the server's log.** `start.sh` removes the old container; save `docker logs` first if you want the evidence.
- **Not measured:** quality of this pack on Korean or on long tool contexts, long-context decode against other engines, any other GB10.

## Newer upstream work

State on 2026-10-14, not applied here:

- **SparkFold 0.6.5+spark.2** (`793b334`): a fresh request on a full pool takes a cheap or the least recently used idle slot, instead of the one
  with the oldest kept entry, so short helper requests stop evicting a long agent conversation. A trial merge into this lane's tree had no
  conflicts. SparkFold reports 377K to 173K re-processed tokens in a replay and 84-94K-token follow-ups resuming from cache.
- **SparkFold branch `h2c`** (unreleased): pinning of the most used system block, re-warming it while idle, a bulk n-gram read-ahead at admission.
  Merging it conflicts in `multi.py` and changes scheduler behaviour.
- **TensorFold itself.** The Python engine is frozen at 0.6.6 and being retired ([issue #286](https://github.com/ashhart/TensorFold/issues/286)).
  Its replacement is a native engine written in Zig (v1.0.x) that drives Metal and CUDA with no Python in the serving path. On `main` the CUDA
  family is Nemotron 3.5 only; Qwen3.8-Flash-Next on CUDA lives on the `zig-flashnext` branch, which
  [MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) serves with an INT4-AutoRound checkpoint (5
  experts per token), not EXL3. EXL3 operators for the Zig CUDA engine are open pull requests
  ([#740](https://github.com/ashhart/TensorFold/pull/740), [#741](https://github.com/ashhart/TensorFold/pull/741)) that no model uses yet, so
  this lane's pack cannot run on the Zig engine today; the port of Flash Next to `main` is tracked in
  [issue #472](https://github.com/ashhart/TensorFold/issues/472).

## Credits

[TensorFold](https://github.com/ashhart/TensorFold) (ashhart and contributors) is the engine; [SparkFold](https://github.com/grearjake-star/TensorFold)
(grearjake-star) is the GB10 fork this lane builds on, and its CHANGELOG credits the contributors whose changes it carries.
[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) provides the image recipe. The EXL3 pack is
[turboderp](https://huggingface.co/turboderp)'s, quantized with ExLlamaV3; the weights are the Qwen team's, under their `qwen-community-1.0`
licence, and nothing here redistributes them.
