"""ATTACK 5: the headline capacity claim, on a REAL 12 GB card.

The report's table -- "on a 12 GB card the shipped streaming code cannot
double-buffer a tile bigger than 336x336, and with this work it reaches
416x416, 1.53x the cells" -- was not measured on a 12 GB card.  It was
measured on an idle RTX 4090 with the CuPy pool capped at 9.740 GiB, a cap
derived as "a 5070's 11.940 GiB minus the measured 2.06-2.18 GiB of non-pool
footprint".

Two things make that emulation worth checking rather than believing:

* the non-pool term the cap is built from was measured as 2.06-2.18 GiB, but
  the ceiling trials themselves printed 2.44-2.60 GiB on the same idle card.
  Half a gigabyte of optimism in the cap is roughly one 16-cell step of the
  bisection, and the winning 416^2 row held pool_total 9.379 GiB against a
  9.740 cap -- inside that margin.
* the non-pool term is architecture-dependent.  It is dominated by NVRTC
  module images, and a 5070 is sm_120 while the emulator is sm_89.

So this runs the four decisive points on the actual card, with NO pool cap
at all, so the only thing that can refuse a configuration is the device.
Each point is a fresh subprocess (a retry inside a poisoned pool measures
fragmentation, not capacity) and each forces a radiation firing and prints
its cadence counters, so a trial that quietly skipped the largest transient
in the configuration cannot be mistaken for one that survived it.
"""
import subprocess
import sys

import cupy as cp

free, total = cp.cuda.runtime.memGetInfo()
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(f"{name}  {free / 2**30:.2f} GiB free of {total / 2**30:.2f}  "
      "(NO pool cap: the device is the only limit)")
if free / total < 0.90:
    print("*** card is not idle; a capacity ceiling measured next to another "
          "tenant is that tenant's ceiling.  Refusing.")
    raise SystemExit(2)
print("=" * 78)

#: ``(label, n, argv-extra, expectation)``.  The expectation is the report's
#: claim; a row that contradicts it is the interesting one.
POINTS = (
    ("as shipped   nbuffers=2", 336, [], "FIT"),
    ("as shipped   nbuffers=2", 352, [], "NO FIT"),
    ("shared+chunks nbuffers=2", 416,
     ["--share", "--rrtmgp-column-chunk", "1024",
      "--mynn-column-chunk", "4096"], "FIT"),
    ("shared+chunks nbuffers=2", 432,
     ["--share", "--rrtmgp-column-chunk", "1024",
      "--mynn-column-chunk", "4096"], "NO FIT"),
)

results = []
for label, n, extra, expect in POINTS:
    argv = [sys.executable, "-u", "-m", "tilestream.vram_probe", "trial",
            "--rung", "full+MYNN+Noah-MP", "--nx", str(n), "--nz", "49",
            "--nbuffers", "2", "--steps", "2"] + extra
    proc = subprocess.run(argv, capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    tail = lines[-1] if lines else proc.stderr[-300:]
    got = {0: "FIT", 3: "NO FIT"}.get(proc.returncode,
                                      f"BROKE(exit {proc.returncode})")
    ok = "  " if got == expect else "<<"
    print(f"{ok} {label} {n}^2 -> {got:22s} (report says {expect})")
    print(f"     {tail}")
    if got.startswith("BROKE"):
        print("     " + proc.stderr.strip().splitlines()[-1][:200])
    results.append((label, n, expect, got))

print("=" * 78)
bad = [r for r in results if r[3] != r[2]]
if bad:
    print(f"{len(bad)} of {len(results)} points DISAGREE with the report:")
    for label, n, expect, got in bad:
        print(f"  * {label} {n}^2: report {expect}, real card {got}")
else:
    print("every point on the real card agrees with the emulated table")
