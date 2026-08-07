"""Two prepared cases, one grid: measure what is the same and what is not.

``build_case_inputs.py`` proves the two arms agree on the CONFIGURATION
-- the domain identity the prepared cache compares by strict equality,
and the eta ladder that identity does not cover.  It cannot prove they
agree on the ARRAYS, because the HRRR case's arrays do not exist until
it is prepared.  This closes that, after the fact, with numbers.

Two things are measured, and only one of them is expected to be zero.

**Static geography must be identical.**  Both cases build their static
fields from the same WPS geography on the same host for the same Lambert
grid, so terrain, land use, map factors and Coriolis should agree
bit-for-bit.  If they do not, the arms are not integrating the same
lower boundary and "everything else identical" is not true.  A nonzero
maximum absolute difference is reported per field and the exit code
says so.

**The initial layer-interface heights are NOT expected to be identical,
and the size of the difference is a stated caveat of this A/B.**  The
observations are gridded once, onto the GFS arm's georeference
trajectory, and reused byte-identically by every arm -- which is what
makes the comparison an observation-for-observation one.  The price is
that a radar gate is placed in the layer the GFS column put it in, and
the HRRR column's layers sit at slightly different heights because the
hydrostatic integration ran over a different atmosphere.  The number
that matters is that height difference expressed as a FRACTION OF THE
LOCAL LAYER DEPTH: below a fraction of a layer the placement is
unchanged for almost every gate; approaching one layer it is not, and
the caveat becomes a finding.

Reads ``.npy``/``.npz`` with numpy.  No GPU, no model, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "gpuwm-da.background-ab-case-parity.v1"

_STANDARD_GRAVITY = 9.81

#: Static fields whose equality is what "the same lower boundary" means.
#: Not every array in the file: the list is the ones the dynamics, the
#: surface scheme and the georeference actually read.
STATIC_FIELDS = ("HGT_M", "LANDMASK", "LU_INDEX", "MAPFAC_M", "MAPFAC_U",
                 "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA", "SOILTEMP",
                 "SNOALB", "TMN")


def prepared_cache_dir(prepared_root: Path) -> Path:
    """The cache directory, whichever of the two layouts this root is.

    A GFS or ERA5 bundle keeps it at ``prepared-cache``; the certified
    native HRRR preparation keeps it at ``native/prepared-cache`` and the
    forecast front door learned that layout rather than renaming a
    certified output.  The candidate list is taken from the front door's
    own map, so a third layout cannot be right there and wrong here.
    """

    from gpuwm.prepared_single_domain_forecast import HRRR_BUNDLE_PATHS

    candidates = (prepared_root / "prepared-cache",
                  prepared_root / HRRR_BUNDLE_PATHS["prepared_cache"])
    for candidate in candidates:
        if (candidate / "header.json").is_file():
            return candidate
    raise SystemExit(
        f"{prepared_root} carries no prepared cache in either known "
        f"layout ({', '.join(str(c) for c in candidates)})")


def cache_array(prepared_root: Path, key: str) -> np.ndarray:
    cache = prepared_cache_dir(prepared_root)
    header = json.loads((cache / "header.json").read_text(encoding="utf-8"))
    try:
        entry = header["arrays"][key]
    except KeyError:
        raise SystemExit(
            f"{cache / 'header.json'} has no array {key!r}; this is not a "
            "prepared cache of the shape this A/B was designed against"
        ) from None
    return np.asarray(np.load(cache / entry["file"]), np.float64)


def layer_interface_heights(prepared_root: Path) -> np.ndarray:
    """``z_w`` at the initial time, from the prepared cache's own arrays."""

    return ((cache_array(prepared_root, "state/php")
             + cache_array(prepared_root, "base/phb")) / _STANDARD_GRAVITY)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.check_case_parity",
        description=__doc__.splitlines()[0])
    parser.add_argument("--reference-prepared-root", type=Path, required=True,
                        help="the arm the observations were gridded on")
    parser.add_argument("--other-prepared-root", action="append",
                        required=True, metavar="NAME=ROOT",
                        help="repeatable; the name is carried into the "
                             "receipt so a reader can tell which arm a "
                             "row belongs to")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    reference_static = np.load(
        args.reference_prepared_root / "native-static.npz")
    reference_z = layer_interface_heights(args.reference_prepared_root)
    # Layer depth at mass points, from the reference column: the scale a
    # height difference has to be judged against.
    depth = np.diff(reference_z, axis=0)

    entries = []
    static_ok = True
    for raw in args.other_prepared_root:
        name, _, path = raw.partition("=")
        if not name or not path:
            parser.error(f"--other-prepared-root must be NAME=ROOT: {raw!r}")
        root = Path(path)
        other_static = np.load(root / "native-static.npz")
        static = {}
        for field in STATIC_FIELDS:
            if field not in reference_static.files:
                continue
            if field not in other_static.files:
                static[field] = {"present": False}
                static_ok = False
                continue
            delta = float(np.max(np.abs(
                np.asarray(reference_static[field], np.float64)
                - np.asarray(other_static[field], np.float64))))
            static[field] = {"present": True, "max_abs_delta": delta,
                             "identical": delta == 0.0}
            static_ok = static_ok and delta == 0.0

        other_z = layer_interface_heights(root)
        if other_z.shape != reference_z.shape:
            raise SystemExit(
                f"{name}: z_w shape {other_z.shape} != reference "
                f"{reference_z.shape}; these are not the same grid")
        height_delta = np.abs(other_z - reference_z)
        # Interfaces 1..nz-1 are the ones a gate can be moved across;
        # interface 0 is the terrain surface and is shared by definition.
        interior = height_delta[1:-1]
        fraction = interior / np.maximum(depth[:-1], 1e-9)
        entries.append({
            "arm": name,
            "prepared_root": str(root),
            "static": static,
            "initial_z_w": {
                "max_abs_delta_m": float(np.max(interior)),
                "mean_abs_delta_m": float(np.mean(interior)),
                "max_fraction_of_layer_depth": float(np.max(fraction)),
                "mean_fraction_of_layer_depth": float(np.mean(fraction)),
                "p99_fraction_of_layer_depth": float(
                    np.percentile(fraction, 99.0)),
            },
        })
        print(f"{name}: static "
              + ("IDENTICAL" if all(
                  item.get("identical") for item in static.values())
                 else "DIFFERS")
              + f"; initial z_w max |d| {np.max(interior):8.2f} m "
                f"= {np.max(fraction):.3f} layer "
                f"(mean {np.mean(fraction):.4f} layer)")

    payload = {
        "schema": SCHEMA,
        "reference_prepared_root": str(args.reference_prepared_root),
        "static_fields_compared": list(STATIC_FIELDS),
        "static_identical_everywhere": static_ok,
        "arms": entries,
        "why": (
            "static geography identical is a REQUIREMENT -- it is what "
            "'the same lower boundary' means.  A nonzero initial z_w "
            "difference is EXPECTED and is the stated price of gridding "
            "the observations once, on one arm's georeference, and "
            "reusing them byte-identically across arms"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    if not static_ok:
        print("\nREFUSED: the arms do not share a static geography, so a "
              "skill difference between them is not attributable to the "
              "background alone.")
        return 1
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
