# The memory gate and the cold re-prefills

On a DGX Spark the CPU and the GPU share 121 GiB, so the engine decides how much host memory a conversation's cache may take. The
rule in TensorFold 0.6.5: a stream cache may grow only if host `MemAvailable` stays above a floor of `max(4 GiB, RAM / 10)` (12.2 GiB here)
plus 2 GiB. When the check fails, the engine drops an idle conversation's whole kept state. A Spark that is serving this model idles at about
15 GiB of `MemAvailable`, so two long conversations evict each other: every turn of each is a cold prefill.

## What we saw

In a live log of ten hours (153 requests of 3,000 prompt tokens or more), 56 requests with nothing cached and 30,000 prompt tokens or more
took 4,659 s of the 4,835 s spent prefilling (one of the 56 was a benchmark probe). Two conversations of about 140k and 105k tokens alternated and every turn missed for about 70
minutes, after they had been hitting the cache a quarter of an hour earlier. A single conversation on its own hit almost every turn.

Timing and the code pointed at the gate; a controlled replay confirmed it
([tools/two_conv.py](../tools/two_conv.py), three conversations of 122k, 99k and 86k tokens taking turns, 16-token replies, `MemAvailable`
held near 16 GiB with [tools/ballast.py](../tools/ballast.py)):

| memory floor | later turns that re-prefilled from zero |
|---|---|
| 12.2 GiB (stock) | 7 of 8 (409 s of prefill lost; the run was cut by a restart before its last request) |
| 6 GiB (`TENSORFOLD_MEMORY_RESERVE_GIB=6`) | 0 of 9 (cached = prompt - 23 tokens, prefill 0.1 s) |

The new log lines name the cause: `memory gate evicted a 122429-token kept state to grow a cache by 1.95 GiB: free over the floor ...`, and
`prefix miss: ... diverges at token N` when the client changed the prompt instead.

## What we run

`TENSORFOLD_MEMORY_RESERVE_GIB=6` and `TENSORFOLD_PIN_EXTRA_GIB=-6.2`. The second keeps the n-gram table's locked share near 20-25 GiB: the
lock room is "budget minus startup estimate" and the budget follows the floor.

## Costs and cautions

- With three long conversations resident, `MemAvailable` sits at about 8-9 GiB. With `earlyoom -m 3,1` as configured on this box (SIGTERM
  at 3% of RAM = 3.6 GiB, SIGKILL at 1% = 1.2 GiB), the margin to the SIGTERM line drops from about 11 GiB to about 5 GiB. In the replay, a ballast process that overshot was killed by `earlyoom`; it had been given
  `oom_score_adj` 1000, so the server survived. The server itself runs at the default score, so any sudden consumer while `MemAvailable` is
  inflated can make `earlyoom` pick it.
- `MemAvailable` is inflated by locked (mlocked) file pages until reclaim notices they cannot be evicted, then it falls within seconds. Count
  headroom as `MemFree + (buff/cache - VmLck of the server)`.
- Do not lock the whole n-gram table (36 GiB): `earlyoom` killed the server that way (`TENSORFOLD_PIN_EXTRA_GIB=12`). Keep the stock budget of
  about 25 GiB or less.
- One machine, one workload. The 6 GiB floor is a measured fit for this box, not a general constant.

## Newer upstream work

SparkFold 0.6.5+spark.2 changes which idle slot a fresh request takes. Before, slots were ranked by their oldest kept entry, so a
conversation that started long ago looked like the best victim however recently it ran, and short helper requests (titles, tool summaries)
took its slot. That is a second, independent cause of cold re-prefills. It is not in the patched tree (the pin is `806b345`); a trial merge
into a scratch copy on 2026-10-14 had no conflicts, and nothing was run with it.
