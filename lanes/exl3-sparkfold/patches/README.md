# Patches

Nine patches on top of [SparkFold](https://github.com/grearjake-star/TensorFold) `spark-exl3` at commit `806b345`
(TensorFold 0.6.5, commit `609ca41`, plus SparkFold's 89 commits for the GB10 and EXL3).

```bash
./apply.sh ~/tensorfold-src      # downloads SparkFold, checks out 806b345, applies the patches
```

`apply.sh` fetches SparkFold from GitHub unless `SPARKFOLD_URL` names a local clone. The tree it builds is the tree `../scripts/start.sh`
mounts as `TF_SRC`. Checked on 2026-10-14 against a local SparkFold clone: the rebuilt tree is identical (`git diff` empty) to the one the
numbers in [../results](../results) were measured on. The download path from GitHub itself was not run. Re-checked on 2026-10-16 with all nine patches: identical to the tree served since that day.

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

Tests on the patched tree: with 0001-0007 the CPU suite gave 3,604 passed, 55 skipped, 0 failed (SparkFold alone: 3,567 passed). With all nine
the suite passed again (exit 0; `test_warm_starts_host::test_invalidation` and `test_server_descriptors` each failed once in one run and passed on
rerun, unrelated to these patches), including the new `tests/test_tool_block_end.py` and a whole-server case in `test_cuda_server_tool_slips.py`. The GPU tests were run file by
file with the serving container stopped; every file passes on the patched tree and on SparkFold alone, except SparkFold's own
`test_flashnext_kernels.py` (6 failures, `forward._readout_plain` is missing; same on SparkFold alone). With a server running, most growth and
graph tests fail because the memory gate has no room: that is the environment, not the code.

## Licence and authorship

These patches modify TensorFold and SparkFold, which are Apache-2.0 ([LICENSE](LICENSE) is their licence text). The patches are offered under
the same licence, and the files they touch keep the notices of the projects they come from. The `From:` header of each patch carries a GitHub
noreply address.
