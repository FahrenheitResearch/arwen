"""Score one arm of a reflectivity A/B on more than the headline number.

:mod:`tools.da_sweep_score` is the axis of record: it reproduces
``NowcastRender.verify_numbers`` step for step and its numbers land on
the same scale as everything already published.  This module does not
replace it and does not re-derive it.  It **calls** it -- every leg goes
through :func:`tools.da_sweep_score.score_leg` and the result is copied
into the payload verbatim under ``published_axis`` -- and then adds the
diagnostics an A/B needs and a single skill score cannot carry.

Four additions, and each one exists because a specific way of being
wrong is invisible without it.

**A per-member statistic beside the mean-field score.**  The published
number scores the ENSEMBLE MEAN: member composites are averaged into one
deterministic dBZ map and that map is scored.  Averaging is a smoother,
so the mean-field score is a function of how the ensemble is *shaped*
and not only of how skilful its members are -- an arm that spreads its
members further can move the mean-field number with no member getting
better, and an arm that collapses them can move it with no member
getting worse.  Assimilating reflectivity changes ensemble structure by
construction, so on this comparison the mean-field number cannot be read
alone.  Every member is therefore scored on its own and the
distribution is reported next to the mean field.

**Counts outside the observed echo.**  ``score_leg`` reports
``fcst_cols_gt35_in_echo`` -- forecast cores counted only where the
observation found echo somewhere in the column.  That is the right
denominator for "did the run make enough storm where storm was", which
is the failure the velocity-only arm has.  It is also structurally blind
to a core invented where nothing was observed, because such a core is
outside the mask and is never counted.  A run that fixes its
under-production by growing spurious convection would look like a pure
win.  So the whole-domain count and the outside-the-echo count are both
reported.

**A false-alarm side.**  Bias (forecast count over observed count) says
nothing about placement: a run can have exactly the right number of
cores and put every one in the wrong place.  A 2x2 contingency table at
the same 35 dBZ column threshold gives POD, FAR, CSI and frequency bias
over the whole domain, so a rise in skill that was bought with false
alarms is visible as CSI falling while bias rises.  This is also the
only view in which suppression -- the thing clear-air observations are
for -- can show up at all: suppression's whole effect is on
``false_alarms`` and on cores standing in observed-clear columns, and
neither term appears in a bias-only or in-echo-only view.

**A curve, not a point.**  FSS at one neighborhood is a scalar with a
free parameter baked into it.  A displacement error smaller than the box
is invisible; a gain that appears only at the widest box is a statement
about displacement tolerance, not about placement.  The score is
therefore evaluated across a ladder of half-widths with the published
one (4 cells at 3 km) among them, so the shape of the gain can be read.

The neighborhood convention is ``da_sweep_score``'s and is restated
because it is the number a comparison against published skill turns on:
the half-width is a count of cells, the scored box is
``2 * half_width + 1`` cells ACROSS, and it is a square SIDE LENGTH, not
a radius.  ``fss_distance`` smooths BOTH fields -- forecast and truth --
to that box before comparing them.

No case names belong here.  Directories, legs and labels are arguments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpuwm.verify.field_metrics import fss_distance
from tools.da_sweep_score import (load_composite, member_names,
                                  metric_constants, score_leg)

#: Neighborhood half-widths, in cells, the FSS curve is evaluated at.
#: 4 is the published one at 3 km (a 9-cell, 27 km box across) and is in
#: the ladder rather than beside it so the curve and the headline are the
#: same computation.
DEFAULT_HALF_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)

#: Column-max threshold, dBZ, for the count and contingency diagnostics.
#: Deliberately the same 35 dBZ the cycling receipts already report, and
#: deliberately NOT the 30 dBZ the FSS uses: the two questions are
#: different (is there a convective core here, versus is the echo field
#: in the right place) and collapsing them onto one threshold would hide
#: which one moved.
CORE_THRESHOLD_DBZ = 35.0


def observed_fields(obs_path: Path, fill_dbz: float):
    """The truth composite and the 2-D echo footprint.

    Built exactly as :func:`tools.da_sweep_score.score_leg` builds them,
    because a contingency table scored against a differently-constructed
    truth field than the FSS would not be describing the same run.
    """

    import netCDF4

    with netCDF4.Dataset(str(obs_path)) as dataset:
        z = np.asarray(dataset["z_obs"][:], float)
        zmask = np.asarray(dataset["z_mask"][:]).astype(bool)
        valid = dataset.getncattr("valid_time")
        z0mask = (np.asarray(dataset["z0_mask"][:]).astype(bool)
                  if "z0_mask" in dataset.variables else None)

    echo2d = zmask.any(axis=0)
    comp = np.where(zmask, z, -np.inf).max(axis=0)
    comp = np.where(np.isfinite(comp), comp, fill_dbz)
    clear2d = None if z0mask is None else z0mask.any(axis=0)
    return comp, echo2d, clear2d, valid


def contingency(forecast: np.ndarray, truth: np.ndarray,
                threshold: float) -> dict:
    """2x2 table at ``threshold``, over the whole domain.

    Whole-domain rather than in-echo: a false alarm is by definition a
    core where the observation has none, so restricting the count to
    cells the observation called echo would remove exactly the events
    this table exists to count.
    """

    f = forecast >= threshold
    o = truth >= threshold
    hits = int((f & o).sum())
    misses = int((~f & o).sum())
    false_alarms = int((f & ~o).sum())
    correct_negatives = int((~f & ~o).sum())

    def ratio(numerator: int, denominator: int):
        # None, never 0.0: a rate with an empty denominator is undefined,
        # and reporting it as zero would read as perfect or as total
        # failure depending on which rate it is.
        return None if denominator == 0 else round(numerator / denominator, 4)

    return {
        "threshold_dbz": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "pod": ratio(hits, hits + misses),
        "far": ratio(false_alarms, hits + false_alarms),
        "csi": ratio(hits, hits + misses + false_alarms),
        "frequency_bias": ratio(hits + false_alarms, hits + misses),
    }


def fss_curve(field: np.ndarray, truth: np.ndarray, *, threshold: float,
              half_widths, dx_km: float) -> list[dict]:
    curve = []
    for half_width in half_widths:
        curve.append({
            "half_width_cells": int(half_width),
            "box_cells_across": 2 * int(half_width) + 1,
            "box_km_across": round((2 * int(half_width) + 1) * dx_km, 3),
            "fss": round(1.0 - fss_distance(
                field, truth, threshold=threshold,
                half_width=int(half_width)), 4),
        })
    return curve


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 4),
        "min": round(float(array.min()), 4),
        "p25": round(float(np.percentile(array, 25)), 4),
        "median": round(float(np.median(array)), 4),
        "p75": round(float(np.percentile(array, 75)), 4),
        "max": round(float(array.max()), 4),
    }


def score_leg_extended(*, composites: Path, obs_path: Path, leg: int,
                       dx_km: float, const: dict, half_widths) -> dict:
    published = score_leg(composites=composites, obs_path=obs_path, leg=leg,
                          dx_km=dx_km, const=const)
    truth, echo2d, clear2d, _valid = observed_fields(
        obs_path, const["MISSING_OBS_FILL_DBZ"])

    names = member_names(composites, leg)
    members = [load_composite(composites, leg, name) for name in names]
    mean_field = np.mean(members, axis=0)
    control = load_composite(composites, leg, "control")

    threshold = const["FSS_THRESHOLD_DBZ"]
    published_half_width = published["fss_half_width_cells"]

    member_fss = [round(1.0 - fss_distance(
        member, truth, threshold=threshold,
        half_width=published_half_width), 4) for member in members]
    member_cores = [int((member >= CORE_THRESHOLD_DBZ).sum())
                    for member in members]
    member_cores_outside = [
        int(((member >= CORE_THRESHOLD_DBZ) & ~echo2d).sum())
        for member in members]

    core = CORE_THRESHOLD_DBZ
    record = {
        "leg": leg,
        "published_axis": published,
        "member_names": names,
        # The mean-field score is `published_axis.fss30_fcst`; this is the
        # same quantity per member, so the two can be read together and a
        # gain that lives only in the averaging is visible as the
        # mean-field number moving while the distribution does not.
        "member_fss30": member_fss,
        "member_fss30_stats": describe(member_fss),
        "columns_gt35": {
            "observed": int((truth >= core).sum()),
            "observed_in_echo": published["obs_cols_gt35"],
            "mean_field_all": int((mean_field >= core).sum()),
            "mean_field_in_echo": published["fcst_cols_gt35_in_echo"],
            "mean_field_outside_echo":
                int(((mean_field >= core) & ~echo2d).sum()),
            "control_all": int((control >= core).sum()),
            "control_in_echo": published["control_cols_gt35_in_echo"],
            "control_outside_echo":
                int(((control >= core) & ~echo2d).sum()),
            "member_all": member_cores,
            "member_all_stats": describe([float(v) for v in member_cores]),
            "member_outside_echo": member_cores_outside,
            "member_outside_echo_stats":
                describe([float(v) for v in member_cores_outside]),
        },
        "contingency_mean_field": contingency(mean_field, truth, core),
        "contingency_control": contingency(control, truth, core),
        "contingency_member_mean": {
            key: (None if any(contingency(m, truth, core)[key] is None
                              for m in members)
                  else round(float(np.mean([contingency(m, truth, core)[key]
                                            for m in members])), 4))
            for key in ("pod", "far", "csi", "frequency_bias")
        },
        "fss_curve_mean_field": fss_curve(
            mean_field, truth, threshold=threshold,
            half_widths=half_widths, dx_km=dx_km),
        "fss_curve_control": fss_curve(
            control, truth, threshold=threshold,
            half_widths=half_widths, dx_km=dx_km),
    }

    # Suppression's own view: cores standing where the radar established
    # clear air.  Reported as None -- not as zero -- when the observation
    # file carries no clear-air assessment, because "none was found" and
    # "none was looked for" are different statements and only one of them
    # is evidence about the forecast.
    if clear2d is None:
        record["observed_clear_air"] = {
            "assessed": False,
            "note": "the observation file carries no clear-air assessment; "
                    "cores standing in observed-clear columns cannot be "
                    "counted and are not reported as zero",
        }
    else:
        record["observed_clear_air"] = {
            "assessed": True,
            "observed_clear_columns": int(clear2d.sum()),
            "mean_field_cores_in_observed_clear":
                int(((mean_field >= core) & clear2d).sum()),
            "control_cores_in_observed_clear":
                int(((control >= core) & clear2d).sum()),
            "member_cores_in_observed_clear":
                [int(((m >= core) & clear2d).sum()) for m in members],
        }
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_ab_score",
        description=__doc__.splitlines()[0])
    parser.add_argument("--composites", type=Path, required=True)
    parser.add_argument("--obs-dir", type=Path, required=True)
    parser.add_argument("--obs-glob", default="*verify*.nc")
    parser.add_argument("--first-free-leg", type=int, required=True)
    parser.add_argument("--dx-km", type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--half-widths", default=",".join(
        str(h) for h in DEFAULT_HALF_WIDTHS),
        help="comma-separated neighborhood half-widths in cells")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    half_widths = tuple(int(part) for part in args.half_widths.split(",")
                        if part.strip())
    const, source = metric_constants()

    obs_files = sorted(args.obs_dir.glob(args.obs_glob))
    if not obs_files:
        raise SystemExit(f"no observation files matched {args.obs_glob} "
                         f"in {args.obs_dir}")

    frames = [score_leg_extended(
        composites=args.composites, obs_path=path,
        leg=args.first_free_leg + offset, dx_km=args.dx_km, const=const,
        half_widths=half_widths)
        for offset, path in enumerate(obs_files)]

    published = [frame["published_axis"] for frame in frames]
    payload = {
        "schema": "gpuwm-da.ab-score.v1",
        "label": args.label,
        "dx_km": args.dx_km,
        "constants_source": source,
        "constants": const,
        "core_threshold_dbz": CORE_THRESHOLD_DBZ,
        "half_widths_cells": list(half_widths),
        "neighborhood_convention":
            "FSS_BOX_KM is a square SIDE LENGTH, not a radius; the scored "
            "box is (2*half_width+1) cells across, and fss_distance "
            "smooths BOTH the forecast and the truth field to it",
        "scored_field_convention":
            "the published FSS scores the arithmetic mean over members of "
            "each member's column-max reflectivity -- one deterministic "
            "map, not an ensemble/extended FSS; member_fss30 scores each "
            "member's own map at the same neighborhood",
        "composites": str(args.composites),
        "obs_dir": str(args.obs_dir),
        "frames": frames,
        "summary": {
            "fss30_fcst_mean": round(float(np.mean(
                [f["fss30_fcst"] for f in published])), 4),
            "fss30_control_mean": round(float(np.mean(
                [f["fss30_control"] for f in published])), 4),
            "member_fss30_mean_of_means": round(float(np.mean(
                [f["member_fss30_stats"]["mean"] for f in frames])), 4),
            "csi35_mean_field_mean": round(float(np.mean(
                [f["contingency_mean_field"]["csi"] or 0.0
                 for f in frames])), 4),
            "far35_mean_field_mean": round(float(np.mean(
                [f["contingency_mean_field"]["far"] or 0.0
                 for f in frames])), 4),
            "bias35_mean_field_mean": round(float(np.mean(
                [f["contingency_mean_field"]["frequency_bias"] or 0.0
                 for f in frames])), 4),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    for frame in frames:
        pub = frame["published_axis"]
        stats = frame["member_fss30_stats"]
        cols = frame["columns_gt35"]
        table = frame["contingency_mean_field"]
        print(f"leg {frame['leg']:2d}  FSS30 mean-field {pub['fss30_fcst']:.4f}"
              f"  member {stats['mean']:.4f} [{stats['min']:.4f},"
              f"{stats['max']:.4f}]  cols>=35 {cols['mean_field_all']:5d}"
              f" (obs {cols['observed']:5d}, outside-echo "
              f"{cols['mean_field_outside_echo']:5d})  CSI {table['csi']}"
              f"  FAR {table['far']}  bias {table['frequency_bias']}")
    print(f"[{args.label}] mean FSS30 {payload['summary']['fss30_fcst_mean']:.4f}"
          f"  mean member FSS30 "
          f"{payload['summary']['member_fss30_mean_of_means']:.4f}"
          f"  [constants from {source}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
