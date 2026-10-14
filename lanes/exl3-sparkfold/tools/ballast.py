#!/usr/bin/env python3
"""Hold anonymous memory so the box's MemAvailable matches a steady state of about 15 GiB. usage: ballast.py <GiB>; ends on SIGTERM.
Prefers to be the OOM victim (oom_score_adj 1000) and exits by itself when MemAvailable < 8 GiB."""
import os, signal, sys, time
target = float(sys.argv[1])          # stop growing when MemAvailable <= target GiB
open("/proc/self/oom_score_adj", "w").write("1000")
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 2**20
buf = []
while len(buf) < int(target * 4):                 # 256 MiB pieces, touched
    buf.append(bytearray(os.urandom(1)) * (256 * 2**20))
print(f"holding {len(buf) / 4:.2f} GiB, MemAvailable now {avail():.1f} GiB", flush=True)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
while True:
    time.sleep(2)
    if avail() < 8: print("MemAvailable < 8 GiB, releasing", flush=True); sys.exit(0)
