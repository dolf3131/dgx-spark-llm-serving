# Decode kernels (patches 0010-0018) and the bench suite, measured 2026-10-17 to 2026-10-21

Between the 2026-10-16 build and the 2026-10-21 build the lane gained nine patches. All of them keep every output bit: the same English greedy
replies (hashes `ca09add1` `8ccd1503` `676d89af` `231c22da`) come out of every build, and each kernel change was checked bit for bit against the
code it replaces. On the four English bench prompts, single stream, 600 tokens:

| build | LRU | CODE-2 | PROSE | CODE-3 |
|---|---|---|---|---|
| 2026-10-16 (patches 0001-0009) | 62.1 | 63.9 | 50.9 | 60.3 |
| 2026-10-21 (patches 0001-0018) | 65.9 | 67.8 | 53.9 | 63.9 |
| change | +6.1% | +6.1% | +5.9% | +6.0% |

The two rows are separate boots five days apart (decode tok/s from the server's per-request line). The step-by-step A/Bs below are the
reliable part: each compares two boots back to back on the same prompts.

## What each step did

| Patches | Change | Measured on the live lane | How the bits were checked |
|---|---|---|---|
| 0010, 0014 | hyper-connection decode tile; tokens handed to the reply before the next draft chain (the HTTP thread was waiting on the GIL) | about +1% decode | greedy hashes, and a test that every tile gives the old tile's bits |
| 0016 | n-gram rows gathered whole, over 16 threads, instead of byte by byte from the 36 GiB map | first turns of 3-4.5k tokens 2.45 to 2.08 s (-15%), ~1k-token follow-ups -16%; decode unchanged | gathered bytes equal to the byte-wise gather (8 cases) |
| 0017 | activations of the EXL3 decode GEMVs loaded a k step ahead; the dense kernel built with `min blocks 1` | +1.5 to +3.0%, mean +2.0% over 14 identical replies | 306 linear + 48 expert comparisons, old vs new build, on the pack's tensors |
| 0018 | decode MoE block: shared expert on a side stream beside the router, expert epilogues fused, grid and reduction buffer reshaped | +1.3 to +3.8%, mean +1.75% over 14 identical replies | 96 comparisons, plus a cut-model test of verify windows of 1-64 rows and the draft tests with and without CUDA graphs |

The A/Bs for 0017 and 0018 used the four English prompts greedy and three Korean prompts, greedy and sampled with a fixed seed (the engine's
sampling is exact per seed and position, so builds that keep the bits sample the same tokens): all 14 replies had the same hash, token count and
round count in both arms, so the tok/s ratios compare identical work.

## Why these and not others: the bandwidth ceiling

A two-row verify round on this lane reads about 6.3 GB (dense layers, hyper-connection faces, the experts the window routes to, the
draft head), so the useful question was how close each kernel gets to the memory bandwidth, and what that bandwidth is.

- **The ceiling is about 250-255 GB/s, not 238 and not 270.** A plain streaming kernel over hundreds of MB of distinct data reads 245-259
  GB/s. The 2026-10-16 write-up used 238 (measured on small tensors); a later 270-284 came from a test that re-read the same buffers and so
  was partly served from the 24 MiB L2. At 252 GB/s the main verify graph's floor is about 25 ms against the 34 ms it took on 2026-10-16.
- **Loads in flight are not the limit.** We built a bit-exact version of the dense EXL3 decode kernel that streams its weights through a
  `cp.async` ring in shared memory, 2 to 6 steps deep. It was 20-35% slower. A microbenchmark of the memory path alone (each warp walking its
  own contiguous K range, as the kernel does) saturates at 245-253 GB/s whether it keeps one step in registers, several in a `cp.async` ring
  or several in a 1-D TMA (`cp.async.bulk`) ring. The deployed kernel's one step ahead is already enough; the stall a profiler shows
  (long scoreboard, ~85%) is just what a memory-bound kernel at low occupancy looks like.
- **What was left in the kernels** was the activation fragments: loaded right before each step's tensor-core instruction, so their L1/L2
  latency sat on every step. Moving them a step ahead (0017) is what the per-kernel numbers below show.

Old and new build of the kernel in one process, 2 rows, CUDA graphs over eight layers' weights (GB/s of the weights read):

| kernel | before 0017 | after 0017 |
|---|---|---|
| `in_proj_qkv` (DeltaNet, 19.7 MB) | 224 | 228 |
| `q_proj` (attention, 23.6 MB) | 229 | 233 |
| `in_proj_z`, split-K (11.8 MB) | 207 | 218 |
| `out_proj` / `o_proj`, split-K (11.8 MB) | 200-201 | 217-218 |
| `k_proj` / `v_proj` (1 MB) | 126 | 156 |
| `lm_head` (477 MB) | 246 | 245 |

The hyper-connection faces (fp16, 6.6 MB) run at 226-237 GB/s with 0010's tile, against 242 for a read-only pass over the same tensors.

## The MoE block (0018)

A decode window's MoE block was seven kernels on one stream: router (bf16, 17 programs), top-k and grouping (one block), input rotation,
grouped gate|up, its epilogue, grouped down, its epilogue. At 2 rows (18.4 distinct experts, 49 MB of weights) it took about 254 µs a layer
in the profiled graph against a floor of about 194 µs.

- The shared expert does not depend on the routing, but it ran with the routed experts after the router. It now runs on a side stream from the
  moment its input exists, in the time the router leaves most of the GPU idle; the routed experts no longer include it. Its slot keeps its
  pick and weight, and the two paths write disjoint rows of the output.
- The gate|up epilogue (sum the K splits, rotate, SwiGLU, rotate for down) runs in the last program to finish each column block; the down
  epilogue (one split) runs on the warps' sums directly. Two launches fewer, same arithmetic in the same order.
- An expert's column blocks are grid neighbours, so its weights stream together; windows of at most 8 members per expert use an 8-row
  reduction buffer (half the shared memory).

MoE block alone, real router and expert tables: 1 row -7.0%, 2 rows -6.5%, 3 rows -5.4%, 4 rows -5.1%, 7 rows -4.2%. Tried and dropped (no
gain, all bit-exact): prefetch depth 2 in the expert kernel, other register budgets, the router at 16 experts a program, the shared expert
split off on the same stream (+7.5%: more launches). Removing the trellis decode or the tensor-core instructions from the expert kernel (a
timing-only test) gains 0-3%, so the remaining 5-10% gap to a plain memory read of the same shape is not compute.

## Not done

- The router, its top-k and the input rotation still run one after another before the routed experts start (about 16-26 µs a layer).
- A persistent kernel for a whole layer: the kernels themselves are near the ceiling, so what is left is spread over kernel boundaries,
  small kernels and the recurrent DeltaNet step; we have not built it.
- A warp-granular "one launch for every projection that reads the same input" kernel was bit-exact and helped the attention projections
  (-13 to -17%) until 0017 made the small projections fast on their own; after that it gained nothing and was dropped.

## The bench suite on the 2026-10-21 build

The same three tools as on 2026-10-14 ([perf](2026-10-14-perf-by-length-and-depth.md), [tool-eval](2026-10-14-tool-eval.md)), run back to back
on the served configuration (`scripts/start.sh` defaults, 18 patches, image with `xgrammar`), the server otherwise idle and no other busy process
on the machine (load average under 1.3 throughout). The 2026-10-14 runs were on the build of that day (7 patches).

**sgbench** (the GB10 community bench: 5 workloads x 2 runs, temperature 0, server-default thinking on, tok/s = completion tokens / whole
request time, so time to first token and reasoning tokens count). N = 2 and 4 use 512 tokens for every workload and divide by the slowest
stream's time.

| Workload | N=1 | N=2 | N=4 | 2026-10-14: N=1 / 2 / 4 |
|---|---:|---:|---:|---|
| Q&A | 47.4 * | 75.9 | 107.0 | 59.5 / 70.6 / 104.9 |
| Code | 67.3 | 91.1 | 124.5 | 66.3 / 86.0 / 120.2 |
| JSON | 97.2 | 120.0 | 163.2 | 95.4 / 119.1 / 159.5 |
| Math | 63.1 | 103.0 | 163.8 | 62.1 / 91.7 / 166.0 |
| LongCode | 59.9 | 75.8 | 113.0 | 58.5 / 73.6 / 109.6 |
| **Token-weighted overall** | **66.9** | **92.1** | **131.1** | 65.9 / 88.1 / 127.9 |

\* One of the two Q&A runs (69 tokens) took 1.8 s instead of 1.1 s with the same reply: the first single-stream request after the
multi-stream runs decoded in a slot whose CUDA graphs had not been captured yet (see the README's limits). The N=1 column is from a rerun
at 12:21; the first N=1 run, 19 minutes after a restart, had the same effect on four of its five second-round requests (overall 65.3). The
other replies of a workload match each other within 2%.

**tool-eval-bench** (97 scenarios including Hard Mode, temperature 0, seed 42, thinking on, `reasoning_effort` medium, one scenario at a time):
**90/100, 175/194 points**, deployability 81, responsiveness 60 (median turn 2.3 s), 14 minutes. This is the first full run on the image with
`xgrammar`: it lands on the 175/194 that the 2026-10-14 merge of two runs estimated. Structured output 10/12 (TC-68 fails: the model adds a
note after the JSON); Hard Mode 44/56. The same three safety-critical failures as before: TC-74 sends an email in the same turn as the
calendar call it depends on, TC-82 sends without a confirmed current recipient, TC-92 discloses another tenant's data.

**llama-benchy** (through tool-eval-bench `--perf-only`, a 2,048-token prompt on top of `depth` tokens of context, 128 generated tokens, three
runs; synthetic English text from one book):

| depth | streams | prefill tok/s | decode tok/s (sum over streams) | 2026-10-14: prefill / decode |
|---:|---:|---:|---:|---|
| 0 | 1 | 1,703 * | 61.7 | 1,366 * / 53.9 |
| 0 | 2 | 1,838 | 89.5 | 1,463 / 82.0 |
| 0 | 4 | 1,976 | 112.2 | 1,626 / 99.5 |
| 4,096 | 1 | 1,755 * | 58.1 | 1,724 * / 59.7 |
| 4,096 | 2 | 1,972 | 87.8 | 1,857 / 80.4 |
| 4,096 | 4 | 1,854 | 52.5 | 1,820 / 51.1 |
| 8,192 | 1 | 1,893 * | 62.6 | 1,881 * / 50.7 |
| 8,192 | 2 | 1,898 | 42.8 | 1,903 / 40.5 (another job ran) |
| 8,192 | 4 | 1,819 | 30.0 | 1,742 / 28.4 (another job ran) |

\* With one stream, prefill is estimated from the time to the first token (llama-benchy issue 33). The depth-0 prefill gain is at least partly
patch 0016 (no stall on the first prompt piece). Each row is one run of this tool on each day and we did not measure its run-to-run
spread on the new build, so read single rows with care. The fall of four-stream decode at 4k-8k depth was there on 2026-10-14 too.

## Caveats

- One boot per arm in each A/B, short prompts, single stream. Multi-stream throughput is in the bench suite above.
- Each bench-suite figure is one run on one day; 2026-10-14 and 2026-10-21 differ in build and in what the machine had run before.
- Kernel tables are from a lab process sharing the GPU with the idle server, timed in CUDA graphs over many layers' weights (so the weights
  come from DRAM, not L2).
