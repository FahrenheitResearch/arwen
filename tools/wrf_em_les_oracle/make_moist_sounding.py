#!/usr/bin/env python3
"""Build a moist WRF `input_sounding` from a dry one (P1).

The dry matched asset `input_sounding.arwen_cbl` carries qv == 0 at every
level, so no moist arm can use it, and the two shipped moist em_les soundings
top out at 2025 m -- below the matched case's ztop of 2400 m -- and carry
their own thermodynamics.  This tool adds a vapour column to an existing
sounding and changes nothing else: the theta profile, the wind columns, the
level list and the surface pressure are copied through byte for byte where
they are not the vapour field.

It does NOT pick the vapour profile.  Which moist sounding defines the P1
case is a case definition with physics consequences (whether the LCL falls
inside the CBL, whether the case rains, how much latent heat is available),
and it is ratified by the owner from measurements, not chosen by a script.
This tool exists so that once it IS ratified the asset is reproducible from
committed inputs rather than hand-edited.

Usage:
  make_moist_sounding.py <dry_sounding> <out> --qv-ml G --qv-free G
                         [--transition-m Z] [--qv-surface G]

  --qv-ml         mixing ratio in the mixed layer, g/kg
  --qv-free       mixing ratio above the transition, g/kg
  --transition-m  height of the ML/free-atmosphere vapour step, metres
                  (default: the height at which the input theta profile
                  first departs from its surface value by 0.05 K -- i.e. the
                  sounding's own mixed-layer top, not a number typed in)
  --qv-surface    the surface-line mixing ratio, g/kg (default: --qv-ml)

The step is applied as a single level-to-level jump at the first level above
--transition-m, matching the shape WRF's own shipped moist em_les soundings
use (10 g/kg below the inversion, 4 g/kg above).
"""
import sys


def parse(path):
    rows = [ln.split() for ln in open(path).read().strip().splitlines()]
    surf = [float(x) for x in rows[0]]
    lev = [[float(x) for x in r] for r in rows[1:]]
    return surf, lev


def ml_top(lev, tol=0.05):
    th0 = lev[0][1]
    for r in lev:
        if abs(r[1] - th0) > tol:
            return r[0]
    return lev[-1][0]


def main():
    argv = sys.argv
    src, out = argv[1], argv[2]

    def opt(name, default=None, cast=float):
        if name in argv:
            return cast(argv[argv.index(name) + 1])
        return default

    qv_ml = opt("--qv-ml")
    qv_free = opt("--qv-free")
    if qv_ml is None or qv_free is None:
        raise SystemExit("--qv-ml and --qv-free are required; this tool does "
                         "not have a default vapour profile and must not.")
    surf, lev = parse(src)
    ztr = opt("--transition-m", ml_top(lev))
    qv_sfc = opt("--qv-surface", qv_ml)

    surf[2] = qv_sfc
    for r in lev:
        r[2] = qv_ml if r[0] <= ztr + 1e-9 else qv_free

    lines = ["%10.2f %10.2f %10.2f" % (surf[0], surf[1], surf[2])]
    for r in lev:
        lines.append("%10.2f %10.2f %10.2f %10.2f %10.2f"
                     % (r[0], r[1], r[2], r[3], r[4]))
    with open(out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s: %d levels, qv %.2f g/kg below %.1f m, %.2f above, "
          "surface line %.2f g/kg"
          % (out, len(lev), qv_ml, ztr, qv_free, qv_sfc))


if __name__ == "__main__":
    main()
