# SparkFold merged into the local build, measured 2026-10-11

One DGX Spark, the serving container stopped for the lab, the same scripts for every row. Terms:

- **TF-opt**: TensorFold 0.6.5 plus our own EXL3 prompt-path MoE kernels, the special-token escape and the lock knob (the build served before this one).
- **TF-spark**: TF-opt plus SparkFold `spark-exl3` @ `806b345` (the tree [../patches](../patches) rebuilds, minus the NVMe spill).
- **profile safe / spark**: [../scripts/start.sh](../scripts/start.sh) profiles; `safe` pins SparkFold's new decoding policy back to the old one and uses 2048-row pieces.

## Cold prefill, real English/Korean text, tok/s at 1.9k / 7.8k / 25k / 49k prompt tokens (nothing cached)

| build | 1.9k | 7.8k | 25k | 49k |
|---|---|---|---|---|
| TF-opt, live lane after 19 h of uptime | 439 | 596 | 812 | 1,062 |
| TF-opt, fresh start | 1,341 | 1,374 | 1,364 | 1,346 |
| TF-spark, profile safe (floor 12.2 GiB) | 1,811 | 1,894 | 1,532 | 1,808 |
| TF-spark, profile spark + floor 6 GiB (deployed) | 1,830 | 2,067 | 2,054 | 2,061 |
| same, unlocked n-gram pages evicted (ballast, 24 GiB `MemAvailable`) | 830 | 1,478 | 1,858 | 1,981 |
| deployed + NVMe spill on | 1,834 | 1,975 | 2,072 | 1,982 |

The 19-hour row is a slowdown we saw and did not explain (not page faults, swap, clocks, core placement or n-gram residency, which was 94.6%).
The prompts came from our own notes and are not published; [../tools](../tools) uses public text instead, so absolute numbers will differ.

## Decode, single stream, tok/s

[../tools/bench_en.py](../tools/bench_en.py): greedy, 600 tokens, mean of 4 prompts, thinking on. [../tools/bench_ko.py](../tools/bench_ko.py): server-default
sampling, mean of 3 Korean prompts.

| build | English | Korean |
|---|---|---|
| TF-opt, live after 19 h | 43.1 | 33.4 |
| TF-opt, fresh | 42.6 | 33.6 |
| TF-spark, profile safe | 59.0 | 42.2 |
| TF-spark, deployed | 57.4 | 43.3 |
| deployed under memory pressure | 58.8 | 42.4 |
| deployed + NVMe spill | 57.9 | 41.8 |

**The decode gain over TF-opt is not established.** TF-opt measured 59.6 / 31.2 on 2026-10-10 with the same code, so the 42.6 above is a low
draw. What the table supports is "no regression against its best run". Concurrency (a different script, aggregate tok/s): 1 stream 87-92,
2 streams 69-106, 4 streams 102-115. Decode at 105k cached context: 49.6 (first) and 53.8 tok/s (cached reruns), 400-token prose.

## Kernels and the end-to-end prefill

Real layer 1, real routing, 1024-row windows, bit oracle: stock grouping 24.0 ms, our earlier kernel 9.88, SparkFold's prompt kernel 7.97. At 2048
rows 16.3 vs 11.9, at 4096 rows 28.5 vs 18.9. All bit-identical to stock. End to end (4 layers + MTP head, 8192 real tokens): stock 6,082 tok/s,
TF-opt 11,995, TF-spark 16,238 with 2048-row pieces and 17,504 with 4096. The streams were bit-identical to a stock oracle.

## Conversations taking turns

[../tools/two_conv.py](../tools/two_conv.py): three conversations (122k, 99k, 86k tokens) take turns, `MemAvailable` ~16 GiB. At the stock 12.2 GiB
floor 7 of 8 later turns re-prefilled from zero (409 s lost); at 6 GiB 0 of 9 did. See [../docs/memory-gate.md](../docs/memory-gate.md). With
spill: [../docs/nvme-spill.md](../docs/nvme-spill.md).

## GPU tests

With the server stopped, every test file passes on the merged tree and on SparkFold alone, except SparkFold's own `test_flashnext_kernels.py`
(6 failures, `forward._readout_plain` missing).

## State after this run

Profile spark, floor 6 GiB, `PIN_EXTRA` -6.2; 23.0 GiB of the n-gram table locked at that start (18-25 GiB depending on `MemAvailable` at start).
A smoke test covered a Korean thinking reply, a tool call, quoted special tokens and an image.
