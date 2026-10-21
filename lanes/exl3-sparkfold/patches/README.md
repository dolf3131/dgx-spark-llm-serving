# Patches

Eighteen patches on top of [SparkFold](https://github.com/grearjake-star/TensorFold) `spark-exl3` at commit `806b345`
(TensorFold 0.6.5, commit `609ca41`, plus SparkFold's 89 commits for the GB10 and EXL3).

```bash
./apply.sh ~/tensorfold-src      # downloads SparkFold, checks out 806b345, applies the patches
```

`apply.sh` fetches SparkFold from GitHub unless `SPARKFOLD_URL` names a local clone. The tree it builds is the tree `../scripts/start.sh`
mounts as `TF_SRC`. Checked on 2026-10-14 against a local SparkFold clone: the rebuilt tree is identical (`git diff` empty) to the one the
numbers in [../results](../results) were measured on. The download path from GitHub itself was not run. Re-checked on 2026-10-16 with all nine patches: identical to the tree served since that day.
Re-checked on 2026-10-21 with all eighteen: identical to the tree served since that day except two opt-in draft lists we did not publish
(`draft_vocab_id48k.txt`, `draft_vocab_ko_seen.txt`, built from our own model outputs; nothing reads them unless `TF_DRAFT_VOCAB` names them).

| # | Patch | What and why |
|---|---|---|
| 0001 | carry-over of local fixes | (a) reserved token strings quoted in text (`<\|im_end\|>`, `<think>`, `<tool_call>`, `<\|image_pad\|>`, 14 in all) are written with a zero-width space going in and restored coming out, so a fetched page can no longer cut a reply, reopen reasoning, fake a turn or return HTTP 400 ([docs](../docs/quoted-special-tokens.md)); (b) a reply that opens `<tool_call>` twice is read as one call, and a call to an undeclared tool logs a warning; (c) `TENSORFOLD_PIN_EXTRA_GIB` moves the n-gram lock room (negative = lock less). Applied as one commit: it was made on an earlier local tree. |
| 0002 | prefix-miss reason log | one line when a prompt of 8192 tokens or more resumes from nothing: "diverges at token N of the M-token kept state" (the client changed the prompt) or "no kept state" (evicted). `TF_MISS_NOTE=0` turns it off. Also prints the lone-stream graph state at startup. |
| 0003 | memory-gate eviction log | names the gate's numbers when a kept conversation is evicted to grow a cache ([docs](../docs/memory-gate.md)). |
| 0004 | docs | `SPARK_PORT.md`: the 2026-10-11 measurements. |
| 0005 | NVMe spill | new `spill.py` and its hooks: idle conversations are written to NVMe instead of being dropped ([docs](../docs/nvme-spill.md)). Off unless `TF_SPILL_DIR` is set. |
| 0006 | eviction note in `prefixes` | host-test decoder stubs are built from a fixed method list, so helpers called from `_evict_kept` must be module-level. |
| 0007 | docs | `SPARK_PORT.md`: the spill section. |
| 0008 | tool-call block end | a `</tool_call>` ends a call only after `</function>` or `</parameter>` (a JSON body: after its `}`), the rule of vLLM's Qwen parser and [veloGB10](https://github.com/sf-stav/veloGB10) 0.7.4, so a `</tool_call>` the model writes inside an argument stays text instead of cutting the call. Applied to the end-of-reply parser, the streamed content and the single-call filter; a streaming reply waits, a finished one falls back to the first close. Complements 0001, which covers only tags copied from the prompt. Known limit: an argument value still ends at a quoted `</parameter>`. |
| 0009 | Korean draft vocabulary | `draft_vocab_ko.txt`: the stock MTP draft list plus the 6,641 Hangul tokens it lacks (Korean token coverage 47.5% to 98.7%), and `tools/draft_vocab_ko.py` that builds it. Opt-in with `TF_DRAFT_VOCAB` (`../scripts/start.sh` sets it). Speed only: Korean draft acceptance 0.32-0.49 to 0.51-0.61, English replies byte-identical ([results](../results/2026-10-16-ko-draft-vocab-and-decode-profile.md)). |
| 0010 | hyper-connection decode tile | per-shape Triton tile for the fp16 hyper-connection faces at decode (down `(64,128,4,2)`, up `(16,64,2,6)`): only the tile width, K step, warps and stages change, never the K slices or their order, so the bits stay (`tests/cuda/test_flashnext_f16_tile.py`). About 0.2 ms a decode round. `TF_F16_TILE=0` keeps the old tile. |
| 0011 | round profile (+ an experiment 0013 removes) | `TF_ROUND_PROFILE` (off by default) times the host side of a lone-stream round. Also adds `TF_ABSORB_FIRST`, which 0013 removes again (no net gain). |
| 0012 | round profile, finer | splits stage+launch into n-gram ids, pinned-buffer wait, n-gram rows and the graph launch. |
| 0013 | n-gram read-ahead with `MADV_POPULATE_READ` | an experiment kept in the series so later patches apply: 0015 reverts it (no change on the lane: most of the table is locked). |
| 0014 | emit early | a lone-stream round hands its accepted tokens to the reply before it builds the next draft chain, so the HTTP streaming thread no longer waits on the GIL behind the drafts (`TF_EMIT_EARLY`, default on). With 0010: about +1% single-stream decode, greedy replies identical. |
| 0015 | revert of 0013 | |
| 0016 | n-gram gather in whole rows | the first prompt piece's n-gram rows were read byte by byte from the 36 GiB map, page fault after page fault: 0.6 s of idle GPU before a new conversation's first kernel. Now whole rows from int16 views of the same maps, over up to `TF_NGRAM_GATHER_THREADS` (default 16) threads; same bytes (8 new test cases). First turns of 3-4.5k tokens 2.45 to 2.08 s (-15%), ~1k-token follow-ups -16%. |
| 0017 | decode GEMVs: activations a k step ahead | the EXL3 dense linear and grouped expert kernels loaded each k step's activation fragments right before its mma (the weights were already a step ahead); now one step ahead too, same values and order. The dense linear also builds with `__launch_bounds__(warps * 32, 1)` (212 instead of 167 registers at four warps: fewer, faster waves for the split-K shapes). Kernels at 2 rows: out/o projections -8%, z -5%, k/v/indexer -18%, MTP fc -8%. Decode +2.0% (14 identical replies). Extensions renamed so trees never share a build. |
| 0018 | decode MoE: shared expert beside the router | in a decode window the shared expert needs only its input, so it now streams on a side stream while the router (17 programs, most SMs idle), top-k and routed experts run; the routed experts' gate/up and down epilogues are fused into the grouped kernels (the last program of a column block finishes it), an expert's column blocks are grid neighbours, and windows of at most 8 members use half the reduction buffer. Same arithmetic in the same order: 96 deployed-vs-new comparisons, a new cut-model test (verify windows 1-64 rows, split on/off), the draft tests with and without CUDA graphs. MoE block -4 to -7%, decode +1.75% (14 identical replies). `TF_MOE_SPLIT=0` keeps the old path. |

Tests on the patched tree: with 0001-0007 the CPU suite gave 3,604 passed, 55 skipped, 0 failed (SparkFold alone: 3,567 passed). With all nine
the suite passed again (exit 0; `test_warm_starts_host::test_invalidation` and `test_server_descriptors` each failed once in one run and passed on
rerun, unrelated to these patches), including the new `tests/test_tool_block_end.py` and a whole-server case in `test_cuda_server_tool_slips.py`. The GPU tests were run file by
file with the serving container stopped; every file passes on the patched tree and on SparkFold alone, except SparkFold's own
`test_flashnext_kernels.py` (6 failures, `forward._readout_plain` is missing; same on SparkFold alone). With a server running, most growth and
graph tests fail because the memory gate has no room: that is the environment, not the code. With all eighteen (2026-10-21), the EXL3
linear, expert, tall-tile and fused-route GPU files pass (201 passed, 46 skipped; run beside the serving container, which these files fit),
and the CPU suite on the eighteen-patch tree gave 3,642 passed, 39 skipped and 6 failed: the six are n-gram lock tests that need
`mlock`, and they pass (25 passed in their files) when the container runs with `--ulimit memlock=-1`, as `scripts/start.sh` does.

## Licence and authorship

These patches modify TensorFold and SparkFold, which are Apache-2.0 ([LICENSE](LICENSE) is their licence text). The patches are offered under
the same licence, and the files they touch keep the notices of the projects they come from. The `From:` header of each patch carries a GitHub
noreply address.
