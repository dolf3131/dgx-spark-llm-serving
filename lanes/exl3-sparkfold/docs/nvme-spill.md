# NVMe spill of idle conversations

`TF_SPILL_DIR` (patch 0005). Off unless set; `../scripts/start.sh` sets it by default (`SPILL=0` turns it off).

## What it does

- **Write.** When the memory gate, a fresh request taking a busy pool's slot, or a fork claim would drop an idle conversation's kept state,
  its longest state (attention rows below its position for every attention layer and the MTP head, the DeltaNet snapshot, the MTP tail) is first
  written to `TF_SPILL_DIR` with `O_DIRECT`. Only states of `TF_SPILL_MIN` tokens or more (16384) are written. The directory is capped at
  `TF_SPILL_GIB` (48 in `start.sh`), least recently used first; a longer state of the same conversation replaces the shorter file.
- **Read.** A prompt that extends a stored state reads it back through the memory gate and resumes from it. The bytes are the ones
  `copy_prefix` would have resumed, so the reply is the one a fresh prefill gives. Any failure (no room, unreadable file) falls back to
  prefilling, as before.
- **Restart.** Files are adopted at startup when the signature matches (model files, KV format, geometry, numerics knobs, TensorFold version);
  anything else in the directory is deleted. The startup line says `spill: N stored states adopted (X GiB), M stale files deleted`.
- **Privacy.** The files hold the conversations' token ids and attention rows. The directory is mode 0700 and the files 0600, owned by the
  container's user (root): treat it as private. To wipe it with the server stopped:
  `docker run --rm --user 0:0 -v "$CACHE":/c --entrypoint sh "$IMAGE" -c 'rm -f /c/spill/*'`.

## Measured (real model, greedy, thinking off)

[tools/spill_check.py](../tools/spill_check.py): three documents (98k, 83k, 81k tokens) read cold, then asked again identically and as follow-ups.

| configuration | cold | identical request again | follow-up | replies differing from the cold read |
|---|---|---|---|---|
| stock floor 12.2 GiB, `MemAvailable` ~16 GiB (the thrash configuration) | 51.9 / 42.6 / 41.4 s | 2.3 / 1.2 / 1.2 s, then 1.0 / 1.1 / 1.0 | 1.3 / 1.9 / 1.9 s | 0 |
| floor 6 GiB with 9 GiB of ballast (`MemAvailable` 9.6 GiB): the gate still evicts, spill rescues | | 1.8 / 1.2 / 1.4 s, then 1.1 / 1.2 / 0.8 | 1.4 / 1.4 / 0.9 s | 0 |

Without spill, the same stock-floor configuration re-read 40-68 s per turn. After `docker restart`: `3 stored states adopted (4.8 GiB)`, the
follow-up of document A read back in 1.9 s (98,541 of 98,542 tokens cached), same reply hash as before the restart. I/O: a 100k-token state
is 1.5-1.8 GiB, written in 0.30-0.40 s (4.5-5.1 GiB/s) and read in 0.26-0.45 s (3.9-5.9 GiB/s); on this NVMe, `dd` with `O_DIRECT` writes
5.2 GB/s and reads 7.2 GB/s, buffered writes 0.95 GB/s (why `O_DIRECT`). Prefill and decode speed with spill on are unchanged.

Tests, in the patched tree: `tests/test_flashnext_spill_file.py` (CPU, 10 tests) and `tests/cuda/test_flashnext_spill.py` (GPU, 6 tests:
bf16 and int8 caches, a tiny model, reply equal to a fresh prefill, failure fallback, restart).

## Limits

- Only the newest state of a conversation is kept. A request that goes back (regenerate, edit an earlier message, the identical request after
  a follow-up) misses on disk as it would in memory.
- One writer thread stalls the other streams for about 0.4 s per eviction; the stall grows with the state size.
- Two-rank (tensor-parallel) and image states are not spilled.
- **Not shown on real traffic.** The numbers above come from controlled replays. In the days after it was switched on we have no usable
  record of how often it fired on real requests (the server's logs were lost in a restart); the directory held five stored states, one of them
  written after the first hours of deployment.
  Treat it as a safety net that is cheap when it is not needed, not as a measured gain on your workload.
- It rescues evictions; SparkFold 0.6.5+spark.2's slot choice (see [memory-gate.md](memory-gate.md)) prevents a class of them in the first place.
