"""Build the probe input deck (FP32 bit patterns, one row per case).

Rows cover the full physical snow effective-radius range plus the exact
branch boundaries of the WRF coupling: the 10 um floor, the 130 um cap
(.gt. 130 predicate), and the 130/sqrt(0.99) crossover where the
MIN(0.99, (130/re_s)^2) clamp changes hands.  When the Phase-A production
radii dump (radii_prefix.npz, written by repro_radii.py from this lane's
scratch) is present, every distinct snow radius the Thompson kernel
actually produced on the real column deck is appended, so the fixture is
anchored in real model states, not synthetic values alone.

No fixture row is fabricated: this script only chooses INPUTS; every
output bit in the fixture comes from the compiled, unmodified WRF
statements.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = r"C:\Users\drew\.claude\jobs\fea7a141\tmp\cloud_seams"

f32 = np.float32


def bits(x):
    return np.float32(x).view(np.uint32)


def main():
    re_m = []
    # Branch boundaries and range endpoints (metres).
    for um in (5.0, 9.99, 10.0, 10.01, 25.0, 50.0, 100.0, 125.0,
               129.0, 129.99, 130.0, 130.00001, 130.5, 130.65,
               130.6539, 130.654, 131.0, 150.0, 200.0, 300.0,
               500.0, 750.0, 999.0):
        re_m.append(f32(f32(um) * f32(1.0e-6)))
    # ULP ladders around both predicates, generated in metre space so the
    # m->um conversion rounding is part of what the fixture pins down.
    for centre_um in (130.0, 130.6539):
        v = f32(f32(centre_um) * f32(1.0e-6))
        lo = hi = v
        for _ in range(6):
            lo = np.nextafter(lo, f32(0.0), dtype=f32)
            hi = np.nextafter(hi, f32(1.0), dtype=f32)
            re_m.extend((lo, hi))
    # Real production radii (metres) from the Thompson kernel on the
    # Phase-A deck, if available.
    dump = os.path.join(SCRATCH, "radii_prefix.npz")
    if os.path.exists(dump):
        d = np.load(dump)
        vals = np.unique(d["effs"].astype(f32).ravel())
        re_m.extend(vals.tolist())
        print(f"appended {vals.size} distinct production snow radii")

    re_m = np.unique(np.asarray(re_m, dtype=f32))
    # Fixed physical layer (qs kg/kg, pdel mb, cldfrac) plus variations so
    # the path outputs exercise the discount against several masses.
    combos = [(2.5e-3, 15.0, 0.8), (1.0e-4, 5.0, 0.05), (8.0e-3, 30.0, 1.0)]
    lines = []
    for qs, pdel, cf in combos:
        for r in re_m:
            lines.append(f"{bits(r):08X} {bits(qs):08X} "
                         f"{bits(pdel):08X} {bits(cf):08X}")
    path = os.path.join(HERE, "probe_inputs.txt")
    with open(path, "w", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} rows ({re_m.size} radii x {len(combos)} "
          f"layer states) to {path}")


if __name__ == "__main__":
    main()
