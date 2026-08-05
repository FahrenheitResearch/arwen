"""Score a cycled run's free forecast with the rolling verifier's metric.

FSS(30 dBZ, 27 km neighborhood) against the observed composite at each
forecast valid time, plus >=35 dBZ column counts inside the echo mask --
the same two numbers ``evidence/da-demo/live-fire-3/
verification-addendum.json`` carries, computed by the same shipped
function (:func:`gpuwm.verify.field_metrics.fss_distance`) so an
ensemble-size sweep lands directly beside the N=10 baseline.

The scored forecast field is the ENSEMBLE-MEAN leg-end column-max
reflectivity.  That is not an assumption: this script reproduces the
addendum's published frames bit-for-bit from the round-3 composites --
FSS 0.7274 / 1899 columns at +15 min and 0.7557 / 2146 at +30, with the
control at 0.2360 and 0.2719 -- and only the ensemble mean does so
(member 0 alone gives 0.7341 / 2062 and 0.7625 / 2296).  Member 0 and
the per-member distribution are reported beside it, never in place of
it.

The mean is the reason ensemble size can move this score at all: with N
members the scored field is an average of N composites, so N enters the
metric directly rather than only through the analysis.

Missing observations inside the domain are filled at -35 dBZ, as the
addendum records.

This is a *caller* of the scorer of record, not a second scorer.  It
exists for what it reports rather than for how it measures: the
per-member FSS distribution and member 0 beside the published
ensemble-mean number, which is the companion statistic an ensemble-size
axis needs and which ``tools/da_sweep_score.py`` does not emit.  Every
constant and the neighborhood derivation come from that module; nothing
metric-defining is written down twice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# This file lives in tools/ens_sweep/, which is not a package (its two
# siblings import it by bare name, having been cd'd into it).  The repo
# root is three levels up and has to be on the path before the scorer of
# record can be imported by its dotted name.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.da_sweep_score import half_width_cells, metric_constants  # noqa: E402

#: The metric definitions, taken from the scorer of record rather than
#: restated.  ``tools/da_sweep_score.py`` is the scorer of record: it
#: reproduces the published gallery numbers, and the HRRR comparison,
#: the verification ladder and the sweep arms are all keyed to it.  It
#: sources these from ``tools/da_nowcast_render.py`` and records which
#: it used, so this module inherits both the values and the provenance.
#:
#: They were duplicated here once -- four literals, independently
#: written against the same published anchor.  They agreed at 3 km and
#: only at 3 km: ``HALF_WIDTH_CELLS = 4`` is the 27 km box only at that
#: spacing, so the copy quietly meant a different metric on any other
#: grid while reporting it under the same name.  A duplicated constant
#: is a silent divergence waiting to happen, so there is now one
#: source and this file is a caller of it.
_CONST, CONSTANTS_SOURCE = metric_constants()

THRESHOLD_DBZ = _CONST["FSS_THRESHOLD_DBZ"]
COLUMN_THRESHOLD_DBZ = _CONST["COLUMN_THRESHOLD_DBZ"]
MISSING_OBS_FILL_DBZ = _CONST["MISSING_OBS_FILL_DBZ"]
FSS_BOX_KM = _CONST["FSS_BOX_KM"]

#: The grid this sweep runs on.  Named, because the neighborhood is
#: derived from it rather than assumed: see :func:`half_width_cells`.
DX_KM = 3.0
HALF_WIDTH_CELLS = half_width_cells(DX_KM, _CONST)


def observed_composite(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(composite dBZ, echo mask)`` from a gpuwm-obs.radar-grid.v1 file."""
    import netCDF4 as nc

    with nc.Dataset(path) as handle:
        z_obs = np.asarray(handle.variables["z_obs"][:], np.float64)
        z_mask = np.asarray(handle.variables["z_mask"][:]).astype(bool)
    filled = np.where(z_mask, z_obs, MISSING_OBS_FILL_DBZ)
    return filled.max(axis=0), z_mask.any(axis=0)


def score_frame(forecast: np.ndarray, observed: np.ndarray,
                echo: np.ndarray) -> dict:
    from gpuwm.verify.field_metrics import fss_distance

    distance = fss_distance(forecast, observed, threshold=THRESHOLD_DBZ,
                            half_width=HALF_WIDTH_CELLS)
    return {
        "fss30_27km": round(1.0 - distance, 4),
        "cols_gt35_in_echo": int((forecast >= COLUMN_THRESHOLD_DBZ)[echo].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--composites", type=Path, required=True,
                        help="<cycle out>/composites")
    parser.add_argument("--members", type=int, required=True)
    parser.add_argument("--first-free-leg", type=int, default=6)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--obs", type=Path, action="append", required=True,
                        help="verification obs per free leg, in leg order")
    parser.add_argument("--valid", action="append", default=[],
                        help="valid-time label per frame, in leg order")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.obs) != args.frames:
        parser.error(f"--obs given {len(args.obs)} time(s), --frames "
                     f"{args.frames}: one verification volume per frame")

    frames = []
    for index in range(args.frames):
        leg = args.first_free_leg + index
        observed, echo = observed_composite(args.obs[index])
        record: dict = {
            "leg": leg,
            "lead_minutes": 15 * (index + 1),
            "valid": args.valid[index] if index < len(args.valid) else None,
            "obs_cols_gt35": int((observed >= COLUMN_THRESHOLD_DBZ).sum()),
        }
        stack = []
        for member in range(args.members):
            field = np.load(args.composites / f"leg{leg:02d}_{member}.npz")
            stack.append(np.asarray(field["refl_colmax"], np.float64))
        stack_arr = np.stack(stack)

        # THE published metric, verified to reproduce the addendum.
        record["ensemble_mean"] = score_frame(stack_arr.mean(axis=0),
                                              observed, echo)
        # Reported beside it, never in place of it.
        record["member0"] = score_frame(stack_arr[0], observed, echo)
        per_member = [score_frame(stack_arr[m], observed, echo)["fss30_27km"]
                      for m in range(args.members)]
        record["member_fss"] = {
            "mean": round(float(np.mean(per_member)), 4),
            "min": round(float(np.min(per_member)), 4),
            "max": round(float(np.max(per_member)), 4),
            "all": per_member,
        }
        control_path = args.composites / f"leg{leg:02d}_control.npz"
        if control_path.is_file():
            control = np.asarray(np.load(control_path)["refl_colmax"],
                                 np.float64)
            record["control"] = score_frame(control, observed, echo)
        frames.append(record)

    payload = {
        "schema": "gpuwm-da.ensemble-sweep-score.v1",
        "definitions": {
            "fss": ("gpuwm.verify.field_metrics.fss_distance, FSS = 1 - "
                    f"distance, threshold {THRESHOLD_DBZ:g} dBZ, half "
                    f"width {HALF_WIDTH_CELLS} cells "
                    f"({(2 * HALF_WIDTH_CELLS + 1) * DX_KM:g} km box at "
                    f"{DX_KM:g} km dx)"),
            "cols_gt35": ("columns whose composite reflectivity reaches "
                          f"{COLUMN_THRESHOLD_DBZ:g} dBZ, counted inside "
                          "the observed echo mask"),
            "missing_obs_fill_dbz": MISSING_OBS_FILL_DBZ,
            "constants_source": CONSTANTS_SOURCE,
            "scorer_of_record": "tools/da_sweep_score.py",
            "scored_field": "ensemble-MEAN leg-end column-max "
                            "reflectivity; verified to reproduce the "
                            "round-3 verification addendum exactly "
                            "(0.7274/1899 at +15, 0.7557/2146 at +30, "
                            "control 0.2360/0.2719). member0 and the "
                            "per-member spread of scores are reported "
                            "beside it, never in place of it",
        },
        "members": args.members,
        "frames": frames,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"members": args.members, "frames": [
        {"lead_minutes": f["lead_minutes"],
         "fss": f["ensemble_mean"]["fss30_27km"],
         "cols35": f["ensemble_mean"]["cols_gt35_in_echo"],
         "fss_control": f.get("control", {}).get("fss30_27km"),
         "fss_member0": f["member0"]["fss30_27km"],
         "fss_member_mean": f["member_fss"]["mean"],
         "cols35_obs": f["obs_cols_gt35"]}
        for f in frames]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
