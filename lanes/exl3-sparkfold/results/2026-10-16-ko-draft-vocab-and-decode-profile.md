# Korean draft vocabulary and the decode round, measured 2026-10-16

Two questions: why is Korean decode slower than English with the same MTP head, and where does a decode round spend its time.
The answer to the first is now patch [0009](../patches/0009-draft_vocab_ko.txt-the-default-draft-list-plus-the-6.patch); the second
found no single waste.

## Korean draft vocabulary (patch 0009)

**Cause.** The engine drafts with the checkpoint's MTP head over a restricted list of token ids, `families/qwen4_exp/cuda/draft_vocab.txt`
(79,591 ids), built from the CPython standard library plus every id below 65,536. It covers 93.9% of English prose tokens and 99.3-99.5% of
Python, Rust, CUDA and Markdown tokens, but only 47.5% of Korean tokens: 166 of the tokenizer's 6,793 Hangul tokens are in it. A draft can
only propose ids in the list, so most Korean drafts are lost before verification.

**Change.** `draft_vocab_ko.txt` = the stock list plus the 6,641 Hangul tokens it lacks (86,232 ids; Korean coverage 98.7%), built by
`tools/draft_vocab_ko.py` in the patched tree and selected with `TF_DRAFT_VOCAB=draft_vocab_ko.txt` (`scripts/start.sh` sets it by default,
`DRAFT_VOCAB=` turns it off). Drafts are verified against the full model, so the list changes speed only.

**Method.** Cold boots of the lane, server otherwise idle, `tools/bench_en.py` (4 English prompts, greedy, 600 tokens) and
`tools/bench_ko.py` (3 Korean prompts, server sampling defaults, thinking on, 600 tokens). Rates from the server's per-request done lines.
A = the 2026-10-11 tree (7 patches), booted twice (under the two profiler modes below); B = the same tree plus patches 0008 and 0009 with the
Korean list.

| | A, boot 1 | A, boot 2 | B |
|---|---|---|---|
| English greedy reply hashes (LRU / CODE-2 / PROSE / CODE-3) | ca09add1 8ccd1503 676d89af 231c22da | same | same |
| English tok/s | 60.5 / 63.6 / 50.9 / 60.0 | 61.7 / 63.7 / 50.9 / 57.8 | 62.1 / 63.9 / 50.9 / 60.3 |
| Korean draft acceptance (HIST / SCI / ADV) | 0.32 / 0.46 / 0.45 | 0.32 / 0.49 / 0.39 | **0.51 / 0.61 / 0.57** |
| Korean tokens per round | 1.44 / 1.85 / 1.78 | 1.44 / 1.94 / 1.71 | 1.76 / 2.51 / 1.85 |
| Korean tok/s | 36.8 / 44.7 / 43.1 | 36.6 / 46.4 / 41.5 | 45.1 / 57.3 / 44.5 |

English replies are byte-identical across all three boots, so the patches leave English output untouched. Sampled Korean replies differ
between any two boots, including two boots of the same code, so the per-prompt Korean tok/s compare different texts; the rise in draft
acceptance is the robust signal. On live traffic before the change the client clock gave English 56.8 and Korean 41.8 tok/s on average.

`TF_DRAFT_VOCAB` is part of the NVMe spill signature, so switching it discards the stored spill states once.

## The decode round (nsys)

Two profiled boots of A on another port, spill off, no other traffic: one with whole-graph timing (1,976 rounds, short contexts), one with
node-level timing inside the CUDA graphs.

- **Round**: median 39.7 ms (mean 43.0) = main verify graph 34.3 ms (85%) + MTP draft graphs 2.6 ms (8%, ~2.5 drafts of ~1.23 ms) + GPU idle
  between graphs 2.4 ms (6.5%); eager kernels and copies are negligible.
- **Inside the main graph** (smallest bucket, 2 rows, 1,612 kernels, 34.1 ms): `linear_kernel` (EXL3 dense layers and `lm_head`, 2.67 GB)
  12.35 ms, about 216 GB/s; `grouped_kernel` (routed experts) 10.13 ms; `_f16_mm` (the fp16 hyper-connection faces, 1.28 GB) 6.95 ms, about
  184 GB/s; `chain_kernel` 1.27 ms; router 0.77 ms; the other ~1,000 small kernels about 2.6 ms.
- **Per extra verify row**: about +2.5-3 ms (buckets 34.1 / 36.5 / 39.4 / 41.9 / 44.4 / 47.5 ms).

Reading: the large kernels already run at roughly 77-91% of the ~238 GB/s this box sustains, and the distance to the bandwidth floor is spread
over kernel efficiency, small kernels, host gaps and draft steps. Not applied: `TF_EXL3_HC_FP8=1` (MXFP8 copies of the hyper-connection
faces, which changes output bits), a device-side draft chain (removes host gaps, at most a few percent, engine work), fusing small kernels.

## Caveats

- One boot per arm. A 75 GB model download (NVMe writes, about 6% of one CPU) ran next to every run, and it probably explains why the B boot
  took 262 s instead of the usual ~100 s (the pack was read back from disk).
- Short contexts only; Korean was measured on three prompts.
