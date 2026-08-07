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

**Two things a single flattering number hides, both reported here.**

*One scale is not a result.*  FSS rises monotonically with neighborhood
size and reaches 1 when the box covers the domain, so a single box is a
point on a curve chosen in advance.  ``--neighborhood-km`` repeats and
produces ``neighborhood_curve``; the published box is always scored
whatever else is asked for, and its numbers keep the flat key names the
existing receipts use, so nothing that reads this file today changes.

*The scored field is the ensemble MEAN.*  Averaging ten members'
column-max reflectivity smooths the field before the metric's own boxcar
smooths it again, which flatters the score relative to any member the
model could actually produce.  ``per_member`` scores each member's own
composite at the published box and reports the spread, so the mean's
number is never read without the distribution it came from.  The two
answer different questions and neither replaces the other.
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


def _dirs(composites) -> list[Path]:
    """Accept one directory or several, uniformly.

    A cycled run that carries its ensemble across process boundaries --
    which is how a cadence that follows the radar is driven -- writes one
    composites directory per process.  The legs are globally numbered by
    ``--leg-number-offset``, so the union of those directories is exactly
    the one directory a single-process run would have written.
    """

    return [Path(c) for c in (composites if isinstance(composites, (list,
                                                                    tuple))
                              else [composites])]


def load_composite(composites, leg: int, name: str) -> np.ndarray:
    wanted = f"leg{leg:02d}_{name}.npz"
    for directory in _dirs(composites):
        path = directory / wanted
        if path.is_file():
            with np.load(path) as handle:
                return np.asarray(handle["refl_colmax"], float)
    raise SystemExit(
        f"no {wanted} in " + ", ".join(str(d) for d in _dirs(composites)))


def member_names(composites, leg: int) -> list[str]:
    """Every member composite present for ``leg``, control excluded.

    A leg appearing in two directories is a duplicate, not two members:
    the name is what identifies a trajectory, so the set is taken by name.
    """

    names: set[str] = set()
    for directory in _dirs(composites):
        for path in directory.glob(f"leg{leg:02d}_*.npz"):
            match = LEG_NAME.match(path.name)
            if (match and match.group(2) != "control"
                    and not NEST_SUFFIX.search(match.group(2))):
                names.add(match.group(2))
    names = list(names)
    # Numeric member ids sort numerically; anything else sorts as text, so
    # a mixed set still has a deterministic order.
    return sorted(names, key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else n))


def _half_width_for_box(box_km: float, dx_km: float) -> int:
    """The renderer's own conversion from a box SIDE to a boxcar half-width.

    Takes the box explicitly so the neighborhood curve can score boxes
    other than the published one.  :func:`half_width_cells` is the
    published-box spelling and the one other scorers import.
    """

    return max(1, round(float(box_km) / 2.0 / float(dx_km)))


def half_width_cells(dx_km: float, const: dict) -> int:
    """Neighborhood half-width in cells for a grid of spacing ``dx_km``.

    The box is a fixed distance -- ``FSS_BOX_KM`` across -- so the cell
    count has to be derived from the spacing, never written down.  A
    scorer that hard-codes the 3 km answer (4) silently means a 13.5 km
    box at 1.5 km and a 54 km box at 6 km, which is a different metric
    reported under the same name.  Any second scorer calls this rather
    than restating the constant.
    """

    return _half_width_for_box(const["FSS_BOX_KM"], dx_km)


def _fss(field, truth, *, threshold, half_width) -> float:
    return round(1.0 - fss_distance(field, truth, threshold=threshold,
                                    half_width=half_width), 4)


def score_leg(*, composites: Path, obs_path: Path, leg: int, dx_km: float,
              const: dict, neighborhoods_km=None) -> dict:
    """``verify_numbers`` for one leg, step for step, plus the two views
    a single flattered number leaves out.

    ``neighborhoods_km`` is the curve to trace.  The renderer's own
    ``FSS_BOX_KM`` is always scored and always fills the flat keys, so a
    reader written against the published receipts is unaffected by what
    else was asked for.
    """

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
    members = [load_composite(composites, leg, n) for n in names]
    fcst = np.mean(members, axis=0)
    ctrl = load_composite(composites, leg, "control")

    if fcst.shape != obs_comp.shape:
        raise SystemExit(
            f"leg {leg}: composite {fcst.shape} and observation "
            f"{obs_comp.shape} are different grids; these are not the "
            "same case and scoring them together would be meaningless")

    published = float(const["FSS_BOX_KM"])
    half_width = half_width_cells(dx_km, const)
    threshold = const["FSS_THRESHOLD_DBZ"]
    column = const["COLUMN_THRESHOLD_DBZ"]

    # -- the curve.  The published box is in it whatever else was asked --
    wanted = list(neighborhoods_km or ())
    if not any(abs(float(k) - published) < 1e-9 for k in wanted):
        wanted.append(published)
    # A half-width is an integer cell count, so two requested boxes can
    # round onto the same one -- at dx = 3 km, 9 km and 15 km are both 2
    # cells.  They are scored once and the collapse is recorded, because
    # two identical rows under different labels read as two measurements.
    by_half_width: dict[int, list[float]] = {}
    for box_km in sorted(float(k) for k in wanted):
        by_half_width.setdefault(_half_width_for_box(box_km, dx_km),
                                 []).append(round(box_km, 3))
    curve = []
    for hw in sorted(by_half_width):
        requested = by_half_width[hw]
        curve.append({
            "box_km_requested": requested,
            "half_width_cells": hw,
            "box_cells_across": 2 * hw + 1,
            # What was actually scored: the honest label, which is not
            # always what was asked for.
            "box_km_across": round((2 * hw + 1) * dx_km, 3),
            "fss30_fcst": _fss(fcst, obs_comp, threshold=threshold,
                               half_width=hw),
            "fss30_control": _fss(ctrl, obs_comp, threshold=threshold,
                                  half_width=hw),
        })

    # -- the ensemble mean is a field no member produced; score them too --
    per_member = [_fss(member, obs_comp, threshold=threshold,
                       half_width=half_width) for member in members]

    return {
        "leg": leg,
        "members_scored": len(names),
        "obs_valid_time": obs_valid,
        "obs_cols_gt35": int(((z * zmask).max(axis=0) >= column).sum()),
        "fcst_cols_gt35_in_echo": int((fcst >= column)[echo2d].sum()),
        "control_cols_gt35_in_echo": int((ctrl >= column)[echo2d].sum()),
        "fss30_fcst": _fss(fcst, obs_comp, threshold=threshold,
                           half_width=half_width),
        "fss30_control": _fss(ctrl, obs_comp, threshold=threshold,
                              half_width=half_width),
        "fss_half_width_cells": half_width,
        "fss_box_cells_across": 2 * half_width + 1,
        "fss_box_km_across": round((2 * half_width + 1) * dx_km, 3),
        "per_member": {
            "scored_field": ("each member's own column-max reflectivity, "
                             "at the published box"),
            "member_names": list(names),
            "fss30": per_member,
            "mean": round(float(np.mean(per_member)), 4),
            "min": round(float(np.min(per_member)), 4),
            "max": round(float(np.max(per_member)), 4),
            "stdev": round(float(np.std(per_member, ddof=1)), 4)
                     if len(per_member) > 1 else 0.0,
            "note": ("fss30_fcst above scores the MEAN of these members' "
                     "fields, which is smoother than any of them and "
                     "therefore scores higher; the gap between "
                     "per_member.mean and fss30_fcst is that smoothing"),
        },
        "neighborhood_curve": curve,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_sweep_score",
        description=__doc__.splitlines()[0])
    parser.add_argument("--composites", type=Path, required=True,
                        action="append", default=[],
                        help="directory of legNN_<name>.npz column maxima. "
                             "Repeatable: a run whose ensemble crossed "
                             "process boundaries writes one per process, "
                             "and the legs are globally numbered, so the "
                             "union is what a single process would have "
                             "written")
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
    parser.add_argument("--neighborhood-km", type=float, action="append",
                        default=[], metavar="KM",
                        help="repeatable: also score at this square box "
                             "SIDE length, so the receipt carries an "
                             "FSS-versus-scale curve instead of one point "
                             "on it. The renderer's own box is always "
                             "scored and always fills the flat keys")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for box_km in args.neighborhood_km:
        if not np.isfinite(box_km) or box_km <= 0.0:
            raise SystemExit(
                f"--neighborhood-km {box_km} is not a box side length; it "
                "must be finite and positive")

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
            const=const, neighborhoods_km=args.neighborhood_km))

    # The curve, averaged over frames: one row per scale, so "FSS rises
    # with the box" is visible as a shape rather than asserted.
    boxes = [row["box_km_across"] for row in frames[0]["neighborhood_curve"]]
    curve_mean = []
    for index, box_km in enumerate(boxes):
        curve_mean.append({
            "box_km_across": box_km,
            "box_cells_across":
                frames[0]["neighborhood_curve"][index]["box_cells_across"],
            "fss30_fcst_mean": round(float(np.mean(
                [f["neighborhood_curve"][index]["fss30_fcst"]
                 for f in frames])), 4),
            "fss30_control_mean": round(float(np.mean(
                [f["neighborhood_curve"][index]["fss30_control"]
                 for f in frames])), 4),
        })

    per_member_means = [f["per_member"]["mean"] for f in frames]
    payload = {
        "schema": "gpuwm-da.sweep-score.v1",
        "label": args.label,
        "dx_km": args.dx_km,
        "constants_source": source,
        "constants": const,
        "neighborhood_convention":
            "FSS_BOX_KM is a square SIDE LENGTH, not a radius; the scored "
            "box is (2*half_width+1) cells across",
        "scored_field":
            "fss30_fcst scores the arithmetic mean over members of each "
            "member's column-max reflectivity -- one deterministic map, "
            "not an ensemble FSS; per_member carries each member's own "
            "score at the same box",
        "truth_smoothing":
            "gpuwm.verify.field_metrics.fss_distance applies the same "
            "boxcar to the observation as to the forecast (Roberts & Lean "
            "smoothed-truth form), which scores higher than a "
            "binary-truth FSS at the same box",
        "composites": [str(c) for c in args.composites],
        "obs_dir": str(args.obs_dir),
        "frames": frames,
        "fss30_fcst_mean": round(
            float(np.mean([f["fss30_fcst"] for f in frames])), 4),
        "fss30_control_mean": round(
            float(np.mean([f["fss30_control"] for f in frames])), 4),
        "fss30_per_member_mean": round(float(np.mean(per_member_means)), 4),
        "fss30_per_member_spread": {
            "min": round(float(np.min([f["per_member"]["min"]
                                       for f in frames])), 4),
            "max": round(float(np.max([f["per_member"]["max"]
                                       for f in frames])), 4),
        },
        "neighborhood_curve_mean": curve_mean,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for frame in frames:
        member = frame["per_member"]
        print(f"leg {frame['leg']:2d}  obs {frame['obs_valid_time']}  "
              f"FSS30 mean-field {frame['fss30_fcst']:.4f}  "
              f"per-member {member['mean']:.4f} "
              f"[{member['min']:.4f}-{member['max']:.4f}]  "
              f"ctrl {frame['fss30_control']:.4f}")
    print(f"mean FSS30 mean-field {payload['fss30_fcst_mean']:.4f}  "
          f"per-member {payload['fss30_per_member_mean']:.4f}  "
          f"ctrl {payload['fss30_control_mean']:.4f}  "
          f"[constants from {source}]")
    if len(curve_mean) > 1:
        print("FSS vs neighborhood (square side, km):")
        for row in curve_mean:
            print(f"  {row['box_km_across']:8.1f} km  "
                  f"({row['box_cells_across']:3d} cells)  "
                  f"fcst {row['fss30_fcst_mean']:.4f}  "
                  f"ctrl {row['fss30_control_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
