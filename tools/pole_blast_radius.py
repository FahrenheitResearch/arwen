"""How much ground the pole-clearance refusal actually covers.

The refusal that stops a root domain whose footprint encloses the
projection pole is stated in two places -- the companion doors
(``gpuwm.domain_wizard``) and plan review
(``gpuwm.experiment.build_experiment``) -- and both ask one function,
``gpuwm.static.projection.footprint_contains_pole``.  This probe sweeps
that function so the refusal's reach is a measured number rather than an
assumption: three projections x six centre latitudes x three
resolutions, on the antimeridian, at one root size.

The rows matter because the limit is NOT a "centred on the pole" special
case and NOT a latitude band: an ordinary wide Arctic domain reaches it
while a narrower one at the same centre does not.

Run it from a checkout:

    PYTHONPATH=. python tools/pole_blast_radius.py
"""

from __future__ import annotations

import sys

from gpuwm.static.projection import (POLE_CLEARANCE_CELLS,
                                     footprint_contains_pole)

#: Root size every row is measured at, in mass points.
NX, NY = 200, 180

#: The sweep axes.  ``truelat1``/``truelat2`` are each projection's
#: ordinary pair, not a tuned one: the question is what a user who picks
#: the obvious projection for a latitude gets.
PROJECTIONS = (("lambert", 30.0, 60.0), ("polar", 60.0, 60.0),
               ("mercator", 0.0, 0.0))
REF_LATS = (45.0, 55.0, 65.0, 70.0, 75.0, 80.0)
DX_M = (12000.0, 30000.0, 45000.0)


def rows():
    """One (map_proj, ref_lat, dx_m, encloses_pole) row per configuration."""
    for map_proj, truelat1, truelat2 in PROJECTIONS:
        for ref_lat in REF_LATS:
            for dx_m in DX_M:
                projection = {"map_proj": map_proj, "ref_lat": ref_lat,
                              "ref_lon": 180.0, "truelat1": truelat1,
                              "truelat2": truelat2, "stand_lon": 180.0}
                yield (map_proj, ref_lat, dx_m,
                       footprint_contains_pole(projection, NX, NY, dx_m))


def main() -> int:
    print(f"root {NX} x {NY} mass points on the antimeridian, "
          f"{POLE_CLEARANCE_CELLS:g} cells of pole clearance")
    print(f"{'map_proj':<10}{'ref_lat':>9}{'dx_km':>8}{'extent_km':>11}"
          f"  footprint")
    refused = 0
    total = 0
    for map_proj, ref_lat, dx_m, encloses in rows():
        total += 1
        refused += bool(encloses)
        print(f"{map_proj:<10}{ref_lat:>9.1f}{dx_m / 1000:>8.0f}"
              f"{max(NX, NY) * dx_m / 1000:>11.0f}"
              f"  {'encloses the pole' if encloses else 'clears the pole'}")
    print(f"{refused} of {total} configurations enclose the projection "
          f"pole and are refused at both the doors and plan review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
