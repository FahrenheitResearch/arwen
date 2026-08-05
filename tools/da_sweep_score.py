"""Score a cycled run's free-forecast legs with the gallery's own FSS.

**This is the scorer of record.**  It reproduces the published gallery
numbers bit for bit, and the HRRR comparison, the verification ladder
and the sweep arms are all keyed to it.  Any other scorer in this tree
is a *caller*: it imports these constants and this neighborhood
derivation rather than restating them, and adds only what it reports.
``tools/ens_sweep/score_free_forecast.py`` is the one such caller --
it exists for the per-member distribution that an ensemble-size axis
needs and that this module does not emit.

That rule is not bookkeeping.  The two scorers once carried their own
copies of the same four constants, independently written against the
same published anchor.  They agreed at 3 km and only at 3 km: a
hard-coded half-width of 4 cells is the 27 km box at that spacing and a
13.5 km box at 1.5 km, so the copy silently meant a different metric
under the same name.  :func:`half_width_cells` exists so the derivation
is written down once.

The rolling verifier in ``tools/da_nowcast.py`` grades a *front-door case
directory*.  A sweep arm that drives ``tools/da_cycle_prepared.py``
straight at an already-prepared case has no such directory -- it has a
composites folder and a pile of observation files -- and yet its numbers
have to land on the same axis as the ones already published, or the sweep
answers nothing.

So this module does not re-implement the metric.  It imports the very
constants and the very function the renderer calls
(:func:`gpuwm.verify.field_metrics.fss_distance`, ``FSS_BOX_KM``,
``FSS_THRESHOLD_DBZ``, ``MISSING_OBS_FILL_DBZ``,
``COLUMN_THRESHOLD_DBZ``) and reproduces
``NowcastRender.verify_numbers`` step for step.  If the renderer's
constants ever move, this follows them; if the renderer cannot be
imported at all, the fallback literals are used AND the fact is recorded
in ``constants_source``, so a reader can always tell which happened.

The neighborhood convention is the renderer's and is stated here because
it is the single number a comparison against published skill turns on:
``FSS_BOX_KM`` is the *side length of a square box*, and the half-width
handed to the boxcar is ``round(FSS_BOX_KM / 2 / dx_km)`` cells, so the
scored neighborhood is ``2 * half_width + 1`` cells ACROSS.  At 3 km that
is a 9-cell, 27 km-wide box.  It is not a radius.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gpuwm.verify.field_metrics import fss_distance

#: Fallback values, used only when the renderer will not import.  They are
#: duplicated deliberately rather than defaulted silently: a sweep that
#: scored itself against different constants than the gallery would be
#: worse than one that refused to score at all, so the choice is recorded.
_FALLBACK = {"FSS_BOX_KM": 27.0,
             "FSS_THRESHOLD_DBZ": 30.0,
             "MISSING_OBS_FILL_DBZ": -35.0,
             "COLUMN_THRESHOLD_DBZ": 35.0}

LEG_NAME = re.compile(r"^leg(\d+)_(.+)\.npz$")
# A nest composite lands in the SAME composites/ directory as the parent
# members, named leg{NN}_{name}_d{GG}.npz by
# tools/da_nowcast_render.py:nest_composite_path.  It is a fine-grid view
# OF one member, not an extra member.  Without this filter "leg00_3_d02"
# parses out as a member named "3_d02" and is averaged into the ensemble
# mean this file scores -- every FSS number moves, and nothing raises.
# The committed live-fire-3 composites predate the nest, so the scorer's
# own regression fixtures would not have caught it either.
NEST_SUFFIX = re.compile(r"_d\d+$")


def metric_constants() -> tuple[dict, str]:
    """The renderer's constants, or the recorded fallback."""

    try:
        from tools import da_nowcast_render as render
        return ({k: getattr(render, k) for k in _FALLBACK},
                "tools.da_nowcast_render")
    except Exception as error:            # pragma: no cover - import guard
        return dict(_FALLBACK), f"fallback literals ({error.__class__.__name__}: {error})"


def half_width_cells(dx_km: float, const: dict) -> int:
    """Neighborhood half-width in cells for a grid of spacing ``dx_km``.

    The box is a fixed distance -- ``FSS_BOX_KM`` across -- so the cell
    count has to be derived from the spacing, never written down.  A
    scorer that hard-codes the 3 km answer (4) silently means a 13.5 km
    box at 1.5 km and a 54 km box at 6 km, which is a different metric
    reported under the same name.  Any second scorer calls this rather
    than restating the constant.
    """

    return max(1, round(const["FSS_BOX_KM"] / 2.0 / dx_km))


def load_composite(composites: Path, leg: int, name: str) -> np.ndarray:
    with np.load(composites / f"leg{leg:02d}_{name}.npz") as handle:
        return np.asarray(handle["refl_colmax"], float)


def member_names(composites: Path, leg: int) -> list[str]:
    """Every member composite present for ``leg``, control excluded."""

    names = []
    for path in composites.glob(f"leg{leg:02d}_*.npz"):
        match = LEG_NAME.match(path.name)
        if (match and match.group(2) != "control"
                and not NEST_SUFFIX.search(match.group(2))):
            names.append(match.group(2))
    # Numeric member ids sort numerically; anything else sorts as text, so
    # a mixed set still has a deterministic order.
    return sorted(names, key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else n))


def score_leg(*, composites: Path, obs_path: Path, leg: int, dx_km: float,
              const: dict) -> dict:
    """``verify_numbers`` for one leg, step for step."""

    import netCDF4

    with netCDF4.Dataset(str(obs_path)) as ds:
        z = np.asarray(ds["z_obs"][:], float)
        zmask = np.asarray(ds["z_mask"][:]).astype(bool)
        obs_valid = ds.getncattr("valid_time")

    echo2d = zmask.any(axis=0)
    obs_comp = np.where(zmask, z, -np.inf).max(axis=0)
    obs_comp = np.where(np.isfinite(obs_comp), obs_comp,
                        const["MISSING_OBS_FILL_DBZ"])

    names = member_names(composites, leg)
    if not names:
        raise SystemExit(f"no member composites for leg {leg} in {composites}")
    fcst = np.mean([load_composite(composites, leg, n) for n in names], axis=0)
    ctrl = load_composite(composites, leg, "control")

    if fcst.shape != obs_comp.shape:
        raise SystemExit(
            f"leg {leg}: composite {fcst.shape} and observation "
            f"{obs_comp.shape} are different grids; these are not the "
            "same case and scoring them together would be meaningless")

    half_width = half_width_cells(dx_km, const)
    threshold = const["FSS_THRESHOLD_DBZ"]
    column = const["COLUMN_THRESHOLD_DBZ"]
    return {
        "leg": leg,
        "members_scored": len(names),
        "obs_valid_time": obs_valid,
        "obs_cols_gt35": int(((z * zmask).max(axis=0) >= column).sum()),
        "fcst_cols_gt35_in_echo": int((fcst >= column)[echo2d].sum()),
        "control_cols_gt35_in_echo": int((ctrl >= column)[echo2d].sum()),
        "fss30_fcst": round(1.0 - fss_distance(
            fcst, obs_comp, threshold=threshold, half_width=half_width), 4),
        "fss30_control": round(1.0 - fss_distance(
            ctrl, obs_comp, threshold=threshold, half_width=half_width), 4),
        "fss_half_width_cells": half_width,
        "fss_box_cells_across": 2 * half_width + 1,
        "fss_box_km_across": round((2 * half_width + 1) * dx_km, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_sweep_score",
        description=__doc__.splitlines()[0])
    parser.add_argument("--composites", type=Path, required=True,
                        help="directory of legNN_<name>.npz column maxima")
    parser.add_argument("--obs-dir", type=Path, required=True,
                        help="directory of verification radar-grid files")
    parser.add_argument("--obs-glob", default="*verify*.nc",
                        help="pattern selecting the verification files, "
                             "in valid-time order (default *verify*.nc)")
    parser.add_argument("--first-free-leg", type=int, required=True,
                        help="leg index of the first free-forecast leg "
                             "(6 for the six-cycle demo shape)")
    parser.add_argument("--dx-km", type=float, required=True)
    parser.add_argument("--label", required=True,
                        help="arm name, carried into the receipt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    const, source = metric_constants()
    obs_files = sorted(args.obs_dir.glob(args.obs_glob))
    if not obs_files:
        raise SystemExit(f"no observation files matched {args.obs_glob} "
                         f"in {args.obs_dir}")

    frames = []
    for offset, obs_path in enumerate(obs_files):
        frames.append(score_leg(
            composites=args.composites, obs_path=obs_path,
            leg=args.first_free_leg + offset, dx_km=args.dx_km,
            const=const))

    payload = {
        "schema": "gpuwm-da.sweep-score.v1",
        "label": args.label,
        "dx_km": args.dx_km,
        "constants_source": source,
        "constants": const,
        "neighborhood_convention":
            "FSS_BOX_KM is a square SIDE LENGTH, not a radius; the scored "
            "box is (2*half_width+1) cells across",
        "composites": str(args.composites),
        "obs_dir": str(args.obs_dir),
        "frames": frames,
        "fss30_fcst_mean": round(
            float(np.mean([f["fss30_fcst"] for f in frames])), 4),
        "fss30_control_mean": round(
            float(np.mean([f["fss30_control"] for f in frames])), 4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for frame in frames:
        print(f"leg {frame['leg']:2d}  obs {frame['obs_valid_time']}  "
              f"FSS30 fcst {frame['fss30_fcst']:.4f}  "
              f"ctrl {frame['fss30_control']:.4f}")
    print(f"mean FSS30 fcst {payload['fss30_fcst_mean']:.4f}  "
          f"ctrl {payload['fss30_control_mean']:.4f}  "
          f"[constants from {source}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
