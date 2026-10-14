# Prefill and decode by context depth and concurrency, measured 2026-10-14

llama-benchy (run through [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), MIT, commit `46b7865c2`, `--perf-only`) against
the deployed configuration of this lane: TensorFold 0.6.5 + SparkFold + the [7 patches](../patches/), `--parallel 5 --kv-dtype int8
--mtp-drafts 6`, memory floor 6 GiB, NVMe spill on. The image is the one in the README plus `xgrammar` (structured outputs); the engine code is
unchanged.

```
tool-eval-bench bench --perf-only --base-url http://127.0.0.1:8888 --seed 42 \
  --pp 2048 --tg 128 --depth 0,4096,8192 --concurrency 1,2,4 --benchy-runs 3 \
  --tokenizer <the pack directory>
```

Each point is a 2,048-token prompt appended to a context of `depth` tokens, 128 generated tokens, three runs. The server was idle when the run
started (no other CPU-heavy process, load average 0.08). llama-benchy cuts the prompts from the text of a Project Gutenberg book (English).

| context depth | streams | prefill, tok/s | decode, tok/s (sum over streams) |
|---|---:|---:|---:|
| 0 | 1 | 1,366 * | 53.9 |
| 0 | 2 | 1,463 | 82.0 |
| 0 | 4 | 1,626 | 99.5 |
| 4,096 | 1 | 1,724 * | 59.7 |
| 4,096 | 2 | 1,857 | 80.4 |
| 4,096 | 4 | 1,820 | 51.1 |
| 8,192 | 1 | 1,881 * | 50.7 |
| 8,192 | 2 | 1,903 † | 40.5 † |
| 8,192 | 4 | 1,742 † | 28.4 † |

\* With one stream, llama-benchy estimates prefill from the time to the first token, because the server's first chunk can arrive before the
first content token (llama-benchy issue 33). Treat those figures as estimates. Its generation figure with several streams is the total over
the streams (its own "total" throughput), which is why the depth-0 decode column rises from one to four streams.

† **Another job started during these rows.** A CPU-heavy numerical job on the same machine (about 8 cores) began 75 s before the run ended, which
covers the last row and possibly the end of the one above it. Those two may be slower than they would otherwise have been; read them with care.
Judging by the elapsed times, the other rows ran before it started. The CPU and GPU of a DGX Spark share one memory, so a job next to the server can change its speed; we now
list other busy processes before a benchmark and record them with the result.

What this shows: at 4k to 8k tokens of context, prefill of a new 2k-token piece runs at 1.7-1.9k tok/s, in line with the 1.8-2.1k tok/s the
[2026-10-11 cold-prefill runs](2026-10-11-sparkfold-port.md) measured on real text from 1.9k to 49k tokens. The one-stream decode here
(51-60 tok/s) is lower than the 71 tok/s median of the [short-prompt probe](2026-10-14-draft-knobs.md); the two use different prompts, and we did
not look into the gap.

**Not stable between runs.** An earlier run the same afternoon, on the container before `xgrammar` was added, gave the same depth-0 prefill
(1,426 tok/s) but 964-1,358 tok/s in five of the six deeper rows. We did not record what else was running then and have not
found the cause, so treat any single run as one sample, and compare runs only when the machine was otherwise idle.

Not covered: contexts beyond 8k, five streams, Korean text, agent-shaped requests.
