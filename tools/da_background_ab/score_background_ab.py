"""Does a convection-allowing background improve WaH's skill, and how?

Four questions, four sections, and a verdict that can come out NO.

**1. Skill, as a curve rather than a number.**  A single 27 km
neighborhood cannot separate "the storms are in better places" from
"the storms are in the same places and the metric cannot see the
difference", so the comparison is an FSS ladder over neighborhood size,
mean over the six verification times.  The method is the one
``tools/ens_sweep/score_resolution.py`` built for the 3 km / 1.5 km
resolution comparison -- ladder over half-widths at a fixed physical
distance, reported as a curve -- reused here.  Its family A/B/C split is
NOT reused and does not arise: every arm of this A/B integrates the same
132x132x49 grid (proved before the run by
``tools/da_background_ab/build_case_inputs.py``), so there is one common
grid, no reduction operator, and nothing for a reduction-operator
sensitivity to be sensitive to.

The 27 km rung (half_width 4, nine cells ACROSS -- a square SIDE, not a
radius) is the published one and is marked as such, so a reader can find
the number that lands on the same axis as
``evidence/da-demo/live-fire-3``.  The scored field at that rung is the
ENSEMBLE MEAN, because that is what the published figure scored; the
per-member distribution is reported beside it, never in place of it.

**2. The controls, which matter more than usual here.**  Every arm
carries a never-analysed control trajectory.  The GFS control on this
case grew from 519 to 1,162 storm columns against 2,800-4,000 observed:
it starts with no storms and never gets many.  An HRRR control starts
with storms already in it.  So a large part of any HRRR advantage may be
the background's own forecast rather than the assimilation, and this
section is where that is separated instead of being absorbed into a
headline.

**3. Spread, and the trap.**  A lagged or better-centred background
produces smaller innovations, so the filter needs less spread to explain
them -- and less spread makes the ensemble mean a smoother field, which
FSS rewards.  An arm that wins on FSS while collapsing is not a better
forecast system; it is a worse-calibrated one that happens to score
higher.  The prior spread trajectory across the six analyses, and the
observation-space consistency ratio
``innovation_rms^2 / (spread^2 + obs_error^2)``, are reported for every
arm so that reading is available rather than buried.

**4. The mechanism.**  If the first guess is genuinely better, the
filter has less work to do, and the analysis increments shrink.  That is
the mechanistic signature, and it is checked directly from each run's own
``mean_increment_rms``.  Increments that do NOT shrink while FSS improves
mean the improvement came from somewhere other than a better-centred
background -- most likely from condensate the background carried and the
DA never had to create -- and the verdict says so.

Everything except the FSS ladder is read out of each arm's
``cycle-report.json``.  No arm writes anything extra to be scored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.da_sweep_score import metric_constants                # noqa: E402

SCHEMA = "gpuwm-da.background-ab-score.v1"

#: Ladder rungs, in cells.  At dx = 3 km these are 3, 9, 15, 21, 27, 39
#: and 51 km across.  The 27 km rung is the published neighborhood.
HALF_WIDTHS = (0, 1, 2, 3, 4, 6, 8)
PUBLISHED_HALF_WIDTH = 4

#: Columns whose composite reflectivity reaches this are "storm columns".
COLUMN_THRESHOLD_DBZ = 35.0

LEG_NAME = re.compile(r"^leg(\d+)_(.+)\.npz$")


def observed(path: Path, fill_dbz: float) -> tuple[np.ndarray, np.ndarray]:
    """``(composite dBZ, echo mask)`` from a gpuwm-obs.radar-grid.v1 file."""

    import netCDF4

    with netCDF4.Dataset(str(path)) as handle:
        z_obs = np.asarray(handle.variables["z_obs"][:], np.float64)
        z_mask = np.asarray(handle.variables["z_mask"][:]).astype(bool)
        valid = handle.getncattr("valid_time")
    filled = np.where(z_mask, z_obs, fill_dbz)
    return filled.max(axis=0), z_mask.any(axis=0), valid


def member_names(composites: Path, leg: int) -> list[str]:
    names = []
    for path in composites.glob(f"leg{leg:02d}_*.npz"):
        match = LEG_NAME.match(path.name)
        if match and match.group(2) != "control":
            names.append(match.group(2))
    return sorted(names,
                  key=lambda n: (not n.isdigit(),
                                 int(n) if n.isdigit() else n))


def composite(composites: Path, leg: int, name: str) -> np.ndarray:
    with np.load(composites / f"leg{leg:02d}_{name}.npz") as handle:
        return np.asarray(handle["refl_colmax"], np.float64)


def ladder(field, truth, *, threshold: float, dx_km: float) -> list[dict]:
    from gpuwm.verify.field_metrics import fss_distance

    return [{
        "half_width": half_width,
        "neighborhood_km": round((2 * half_width + 1) * dx_km, 1),
        "published_rung": half_width == PUBLISHED_HALF_WIDTH,
        "fss": round(1.0 - fss_distance(field, truth, threshold=threshold,
                                        half_width=half_width), 4),
    } for half_width in HALF_WIDTHS]


def score_arm(*, name: str, cycle_out: Path, obs_paths: list[Path],
              first_free_leg: int, dx_km: float, const: dict) -> dict:
    composites = cycle_out / "composites"
    threshold = const["FSS_THRESHOLD_DBZ"]
    frames = []
    for offset, obs_path in enumerate(obs_paths):
        leg = first_free_leg + offset
        truth, echo, valid = observed(obs_path,
                                      const["MISSING_OBS_FILL_DBZ"])
        names = member_names(composites, leg)
        if not names:
            raise SystemExit(f"{name}: no member composites for leg {leg}")
        stack = np.stack([composite(composites, leg, n) for n in names])
        if stack.shape[1:] != truth.shape:
            raise SystemExit(
                f"{name}: composite {stack.shape[1:]} and observation "
                f"{truth.shape} are different grids; these are not the same "
                "case and scoring them together would be meaningless")
        mean = stack.mean(axis=0)
        control = composite(composites, leg, "control")
        per_member = [ladder(member, truth, threshold=threshold,
                             dx_km=dx_km) for member in stack]
        published = HALF_WIDTHS.index(PUBLISHED_HALF_WIDTH)
        member_published = [rungs[published]["fss"] for rungs in per_member]
        frames.append({
            "leg": leg,
            "lead_minutes": 15 * (offset + 1),
            "obs_valid_time": valid,
            "obs_cols_gt35": int((truth >= COLUMN_THRESHOLD_DBZ).sum()),
            "cols_gt35_in_echo": {
                "ensemble_mean": int((mean >= COLUMN_THRESHOLD_DBZ)[echo].sum()),
                "control": int((control >= COLUMN_THRESHOLD_DBZ)[echo].sum()),
            },
            "ensemble_mean": ladder(mean, truth, threshold=threshold,
                                    dx_km=dx_km),
            "control": ladder(control, truth, threshold=threshold,
                              dx_km=dx_km),
            # Reported BESIDE the ensemble mean, never instead of it: the
            # published figure is the mean, and a per-member number that
            # replaced it would not land on the same axis.
            "member_fss_at_published_rung": {
                "members": len(names),
                "mean": round(float(np.mean(member_published)), 4),
                "min": round(float(np.min(member_published)), 4),
                "max": round(float(np.max(member_published)), 4),
                "all": member_published,
            },
        })

    def curve(key: str) -> list[dict]:
        rows = []
        for position, half_width in enumerate(HALF_WIDTHS):
            values = [frame[key][position]["fss"] for frame in frames]
            rows.append({
                "half_width": half_width,
                "neighborhood_km": frames[0][key][position]["neighborhood_km"],
                "published_rung": half_width == PUBLISHED_HALF_WIDTH,
                "fss_mean_over_leads": round(float(np.mean(values)), 4),
            })
        return rows

    report = json.loads(
        (cycle_out / "cycle-report.json").read_text(encoding="utf-8"))
    analyses = [leg["analysis"] for leg in report["legs"]
                if (leg.get("analysis") or {}).get("applied")]
    cycles = []
    for index, analysis in enumerate(analyses):
        filt = analysis["filter"]
        innovation = (analysis.get("innovations") or [{}])[0]
        spread = float(innovation.get("ensemble_spread_mean", float("nan")))
        obs_error = float(innovation.get("obs_error_mean", float("nan")))
        innovation_rms = float(innovation.get("innovation_rms", float("nan")))
        denominator = spread ** 2 + obs_error ** 2
        cycles.append({
            "cycle": index,
            "prior_spread": filt["prior_spread"],
            "posterior_spread": filt["posterior_spread"],
            "mean_increment_rms": filt["mean_increment_rms"],
            "observations": innovation.get("observations"),
            "innovation_rms_ms": innovation_rms,
            "obs_space_spread_ms": spread,
            "obs_error_ms": obs_error,
            # >1 means the ensemble plus its stated observation error
            # cannot explain the innovations it is seeing: the classic
            # under-dispersion diagnostic, in the units it is stated in.
            "consistency_ratio": (round(innovation_rms ** 2 / denominator, 3)
                                  if denominator > 0 else None),
            "control_innovation_rms_ms": (
                analysis.get("control_vr") or {}).get("innovation_rms_ms"),
        })

    return {
        "arm": name,
        "cycle_out": str(cycle_out),
        "frames": frames,
        "curve_ensemble_mean": curve("ensemble_mean"),
        "curve_control": curve("control"),
        "cycles": cycles,
        "total_wall_seconds": report.get("total_wall_seconds"),
        "background": report.get("background"),
    }


def _at_published(curve: list[dict]) -> float:
    for row in curve:
        if row["published_rung"]:
            return row["fss_mean_over_leads"]
    raise KeyError("no published rung in the curve")


def verdict(gfs: dict, hrrr: dict) -> dict:
    """The falsification test, applied to the numbers rather than to a story.

    Three clauses, each of which can independently make the answer NO,
    and each of which is a REASON rather than a threshold pulled from
    nowhere.
    """

    ensemble_delta = [
        round(h["fss_mean_over_leads"] - g["fss_mean_over_leads"], 4)
        for g, h in zip(gfs["curve_ensemble_mean"],
                        hrrr["curve_ensemble_mean"])]
    control_delta = [
        round(h["fss_mean_over_leads"] - g["fss_mean_over_leads"], 4)
        for g, h in zip(gfs["curve_control"], hrrr["curve_control"])]

    improved = [delta > 0.0 for delta in ensemble_delta]
    # CLAUSE 1 -- no skill difference anywhere on the ladder.
    no_improvement = not any(improved)
    # CLAUSE 2 -- the whole difference is the background's own forecast:
    # the never-analysed control gains at least as much as the analysed
    # ensemble, at every rung.  Assimilation added nothing on top.
    background_only = all(c >= e for c, e in zip(control_delta,
                                                 ensemble_delta))
    # CLAUSE 3 -- the gain is bought by a collapsing ensemble.  Compared
    # at the LAST analysis, where cycling has had its full effect.
    def last(arm, key):
        return arm["cycles"][-1][key] if arm["cycles"] else None

    gfs_ratio = last(gfs, "consistency_ratio")
    hrrr_ratio = last(hrrr, "consistency_ratio")
    under_dispersive = (
        gfs_ratio is not None and hrrr_ratio is not None
        and hrrr_ratio > gfs_ratio)

    def increment_norm(arm):
        if not arm["cycles"]:
            return None
        values = [sum(cycle["mean_increment_rms"].values())
                  for cycle in arm["cycles"]]
        return round(float(np.mean(values)), 5)

    gfs_increment = increment_norm(gfs)
    hrrr_increment = increment_norm(hrrr)
    increments_shrank = (
        gfs_increment is not None and hrrr_increment is not None
        and hrrr_increment < gfs_increment)

    if no_improvement:
        answer = "FALSIFIED: no rung of the ladder improved"
    elif background_only:
        answer = ("FALSIFIED as a statement about WaH: the never-analysed "
                  "control gained at least as much as the analysed "
                  "ensemble at every rung, so what improved is the "
                  "background's own forecast, not the assimilation")
    elif under_dispersive:
        answer = ("NOT SUPPORTED as stated: FSS improved while the "
                  "ensemble became LESS able to explain its own "
                  "innovations, which is what a collapsing ensemble does "
                  "to a mean-field score.  Report this as under-dispersion")
    elif not increments_shrank:
        answer = ("SUPPORTED, but NOT by the mechanism claimed: skill "
                  "improved and the analysis increments did not shrink, so "
                  "the first guess was not closer to the observations.  The "
                  "gain came from something the background carried -- "
                  "condensate is the candidate -- not from a better-centred "
                  "state")
    else:
        answer = ("SUPPORTED: skill improved beyond the control's own "
                  "gain, the ensemble did not become less consistent, and "
                  "the analysis increments shrank")

    return {
        "answer": answer,
        "delta_fss_ensemble_mean": ensemble_delta,
        "delta_fss_control": control_delta,
        "neighborhood_km": [row["neighborhood_km"]
                            for row in gfs["curve_ensemble_mean"]],
        "published_rung_km": (2 * PUBLISHED_HALF_WIDTH + 1) * 3.0,
        "delta_at_published_rung": round(
            _at_published(hrrr["curve_ensemble_mean"])
            - _at_published(gfs["curve_ensemble_mean"]), 4),
        "clauses": {
            "no_improvement_anywhere": no_improvement,
            "explained_entirely_by_the_control": background_only,
            "hrrr_more_under_dispersive_at_the_last_analysis":
                under_dispersive,
            "analysis_increments_shrank": increments_shrank,
        },
        "consistency_ratio_last_analysis": {"gfs": gfs_ratio,
                                            "hrrr": hrrr_ratio},
        "mean_increment_rms_over_cycles": {"gfs": gfs_increment,
                                           "hrrr": hrrr_increment},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.score_background_ab",
        description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True,
                        metavar="NAME=CYCLE_OUT",
                        help="repeatable; the GFS arm must be named gfs "
                             "and at least one hrrr arm must be present")
    parser.add_argument("--obs", type=Path, action="append", required=True,
                        help="verification radar-grid file per free leg, "
                             "in leg order")
    parser.add_argument("--first-free-leg", type=int, default=6)
    parser.add_argument("--dx-km", type=float, required=True)
    parser.add_argument("--baseline-arm", default="gfs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    const, source = metric_constants()
    arms = {}
    for entry in args.arm:
        name, _, path = entry.partition("=")
        if not name or not path:
            parser.error(f"--arm must be NAME=CYCLE_OUT, got {entry!r}")
        arms[name] = score_arm(
            name=name, cycle_out=Path(path), obs_paths=list(args.obs),
            first_free_leg=args.first_free_leg, dx_km=args.dx_km,
            const=const)
    if args.baseline_arm not in arms:
        parser.error(f"--baseline-arm {args.baseline_arm!r} is not among "
                     f"{sorted(arms)}")

    baseline = arms[args.baseline_arm]
    verdicts = {name: verdict(baseline, arm) for name, arm in arms.items()
                if name != args.baseline_arm}

    payload = {
        "schema": SCHEMA,
        "constants_source": source,
        "constants": const,
        "dx_km": args.dx_km,
        "neighborhood_convention": (
            "FSS_BOX_KM is a square SIDE LENGTH, not a radius; a rung of "
            "half_width h scores a box (2h+1) cells across.  The FSS here "
            "smooths BOTH the forecast and the truth field, as the shipped "
            "metric does"),
        "ladder_method": (
            "reused from tools/ens_sweep/score_resolution.py, which built "
            "the neighborhood ladder for the 3 km / 1.5 km resolution "
            "comparison.  Its family A/B/C split does not arise here: every "
            "arm integrates the same grid, proved before the run by "
            "tools/da_background_ab/build_case_inputs.py"),
        "baseline_arm": args.baseline_arm,
        "arms": arms,
        "verdicts": verdicts,
        "caveats": [
            "Every arm is a single draw.  No arm is repeated, so none of "
            "these numbers carries an error bar, and the no-ECC dual-run "
            "screen is not applied to any of them.",
            "One case, one radar, radial velocity only, one microphysics "
            "scheme, one 90-minute forecast window.",
            "The observations are byte-identical across arms and were "
            "gridded onto the GFS arm's georeference trajectory.  The "
            "arms share a horizontal grid and a terrain field exactly; "
            "they differ in the hydrostatic column, so an observation "
            "sits in a slightly different model layer for the HRRR arms "
            "than the layer its own first guess would have put it in.",
            "The perturbation amplitudes were tuned against a storm-free "
            "GFS first guess and are not retuned here.  An unbalanced "
            "150 km wind perturbation is a cruder instrument on a "
            "background that already contains sharp convection.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n",
                        encoding="utf-8", newline="\n")

    print(f"{'nbhd km':>9}", end="")
    for name in arms:
        print(f" {name:>14}", end="")
    print()
    for position, row in enumerate(baseline["curve_ensemble_mean"]):
        mark = "*" if row["published_rung"] else " "
        print(f"{row['neighborhood_km']:>8.1f}{mark}", end="")
        for arm in arms.values():
            print(f" {arm['curve_ensemble_mean'][position]['fss_mean_over_leads']:>14.4f}",
                  end="")
        print()
    print("\ncontrol (never analysed), same ladder")
    for position, row in enumerate(baseline["curve_control"]):
        mark = "*" if row["published_rung"] else " "
        print(f"{row['neighborhood_km']:>8.1f}{mark}", end="")
        for arm in arms.values():
            print(f" {arm['curve_control'][position]['fss_mean_over_leads']:>14.4f}",
                  end="")
        print()
    for name, entry in verdicts.items():
        print(f"\n{name} vs {args.baseline_arm}: {entry['answer']}")
    print(f"\n{args.out}  [* = the published 27 km rung; constants from "
          f"{source}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
