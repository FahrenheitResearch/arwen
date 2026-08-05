#!/usr/bin/env python3
"""Add a capping inversion to a WRF `input_sounding` (P1, capped family).

`input_sounding.arwen_cbl` carries a 3 K km-1 stable lapse and no capping
inversion.  A dry CBL does not need one -- it is capped by its own entrainment
-- but a condensing CBL is not: latent heating carries plumes through a weak
lapse indefinitely, and the measured consequence was a cloud deck pressed
against the model lid for 82 % of a 2 h run
(`wrf-moist-matched-first-light-2026-08-04.md`).

This inserts the inversion and touches nothing else: the level list, the winds,
the surface line and the vapour column are copied through, and the profile
above the inversion is the ORIGINAL profile offset by the inversion strength,
so the free-tropospheric lapse rate is unchanged.  Give the moisture to
`make_moist_sounding.py` afterwards; the two tools are separate so a capped-dry
and a capped-moist sounding differ in the vapour column and in nothing else.

It does NOT have a default inversion.  Base height, depth and strength are a
case definition, ratified and documented, not a constant hidden in a script.

Usage:
  add_capping_inversion.py <src> <out> --base-m Z --depth-m D --strength-k K

  --base-m      height of the inversion base, metres
  --depth-m     depth over which theta ramps by --strength-k, metres
  --strength-k  potential-temperature jump across the inversion, K
"""
import sys


def parse(path):
    rows = [ln.split() for ln in open(path).read().strip().splitlines()]
    surf = [float(x) for x in rows[0]]
    lev = [[float(x) for x in r] for r in rows[1:]]
    return surf, lev


def main():
    argv = sys.argv
    src, out = argv[1], argv[2]

    def opt(name):
        if name not in argv:
            raise SystemExit("%s is required; this tool has no default "
                             "inversion and must not have one." % name)
        return float(argv[argv.index(name) + 1])

    base = opt("--base-m")
    depth = opt("--depth-m")
    strength = opt("--strength-k")

    surf, lev = parse(src)
    top = base + depth
    for r in lev:
        z = r[0]
        if z <= base:
            add = 0.0
        elif z >= top:
            add = strength
        else:
            add = strength * (z - base) / depth
        r[1] += add

    lines = ["%10.2f %10.2f %10.2f" % (surf[0], surf[1], surf[2])]
    for r in lev:
        lines.append("%10.2f %10.2f %10.2f %10.2f %10.2f"
                     % (r[0], r[1], r[2], r[3], r[4]))
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    # report the realised jump on the model's own level list, which is what
    # actually gets initialised -- not the requested one.
    below = [r for r in lev if r[0] <= base]
    above = [r for r in lev if r[0] >= top]
    print("wrote %s: %d levels" % (out, len(lev)))
    print("  inversion base %.1f m, top %.1f m, depth %.1f m, strength %.2f K"
          % (base, top, depth, strength))
    if below and above:
        print("  theta at the last level <= base : %.3f K (z = %.1f)"
              % (below[-1][1], below[-1][0]))
        print("  theta at the first level >= top : %.3f K (z = %.1f)"
              % (above[0][1], above[0][0]))
        print("  realised jump across the ramp   : %.3f K over %.1f m"
              % (above[0][1] - below[-1][1], above[0][0] - below[-1][0]))


if __name__ == "__main__":
    main()
