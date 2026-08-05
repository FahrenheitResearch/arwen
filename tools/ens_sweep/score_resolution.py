"""Does 1.5 km beat 3 km, and at which SCALES?

A single 27 km neighborhood cannot answer that.  27 km is nine coarse
cells across; almost everything halving dx buys lives below it, so a
null measured only there would be a statement about the metric as much
as about resolution.  So the comparison is a CURVE over neighborhood
size, not a pair of scalars, with the fixed-physical-distance rule held
at every rung.

Three families, because no single common grid can see every scale:

FAMILY A -- common 3 km grid (the headline).
    The 1.5 km composite is reduced to the 3 km grid by 2x2 block MAX --
    max, not mean, because the field is a column-max reflectivity
    composite and a composite of a composite is a max -- and both runs
    are scored against the same observed 3 km composite at half-widths
    0,1,2,3,4,6,8 = 3,9,15,21,27,39,51 km.  Every rung is exact on this
    grid for both runs.  This family CANNOT see below 3 km: the
    reduction destroys exactly that information, by design.

FAMILY B -- common 1.5 km grid (what family A cannot see).
    The 3 km composite is replicated 2x onto the fine lattice -- nearest
    neighbour, which adds no information, and that is the point: it is
    what the coarse run actually knows, expressed where the fine run
    lives.  Both are scored against the 1.5 km observed composite at
    1.5,4.5,7.5,13.5,25.5 km.  This is the only family that can say
    whether the fine run resolves structure below 3 km at all.

FAMILY C -- sensitivity: family A with block MEAN instead of block MAX.
    Block max can only raise a cell's value, so it hands the fine run
    more threshold exceedances, and these forecasts already UNDER-produce
    echo against truth (about 2000 columns against 2800 observed).  That
    bias runs in the fine run's favour.  The mean-reduced number beside
    it shows whether the reduction operator, rather than the resolution,
    is carrying the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from score_free_forecast import (COLUMN_THRESHOLD_DBZ, THRESHOLD_DBZ,
                                 observed_composite)

#: Box edge is 2h+1 cells; the km follows from that grid's spacing.
COARSE_HALF_WIDTHS = (0, 1, 2, 3, 4, 6, 8)      # 3..51 km at 3 km dx
FINE_HALF_WIDTHS = (0, 1, 2, 4, 8)              # 1.5..25.5 km at 1.5 km dx


def fss_at(field: np.ndarray, observed: np.ndarray, half_width: int) -> float:
    from gpuwm.verify.field_metrics import fss_distance

    return 1.0 - fss_distance(field, observed, threshold=THRESHOLD_DBZ,
                              half_width=half_width)


def block_reduce(field: np.ndarray, factor: int, how: str) -> np.ndarray:
    ny, nx = field.shape
    if ny % factor or nx % factor:
        raise ValueError(f"{field.shape} is not divisible by {factor}")
    blocks = field.reshape(ny // factor, factor, nx // factor, factor)
    return blocks.max(axis=(1, 3)) if how == "max" else blocks.mean(axis=(1, 3))


def replicate(field: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upsample: adds no information, by construction."""
    return np.repeat(np.repeat(field, factor, axis=0), factor, axis=1)


def mean_composite(composites: Path, leg: int, members: int) -> np.ndarray:
    return np.mean([
        np.asarray(np.load(composites / f"leg{leg:02d}_{m}.npz")
                   ["refl_colmax"], np.float64)
        for m in range(members)], axis=0)


def ladder(field: np.ndarray, observed: np.ndarray, half_widths,
           dx_km: float) -> list[dict]:
    return [{"half_width": h,
             "neighborhood_km": round((2 * h + 1) * dx_km, 1),
             "fss": round(fss_at(field, observed, h), 4)}
            for h in half_widths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fine-composites", type=Path, required=True)
    parser.add_argument("--coarse-composites", type=Path, required=True)
    parser.add_argument("--members", type=int, required=True)
    parser.add_argument("--refine-factor", type=int, default=2)
    parser.add_argument("--first-free-leg", type=int, default=6)
    parser.add_argument("--obs-coarse", type=Path, action="append",
                        required=True)
    parser.add_argument("--obs-fine", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    factor = args.refine_factor
    frames = []
    for index, obs_coarse in enumerate(args.obs_coarse):
        leg = args.first_free_leg + index
        observed_c, echo_c = observed_composite(obs_coarse)
        coarse = mean_composite(args.coarse_composites, leg, args.members)
        fine = mean_composite(args.fine_composites, leg, args.members)

        fine_max = block_reduce(fine, factor, "max")
        if fine_max.shape != coarse.shape:
            raise SystemExit(
                f"leg {leg}: reduced fine grid {fine_max.shape} != 3 km grid "
                f"{coarse.shape}; the runs do not share a footprint and no "
                "honest FSS comparison is possible")
        fine_mean = block_reduce(fine, factor, "mean")

        record: dict = {
            "leg": leg,
            "lead_minutes": 15 * (index + 1),
            "obs_cols_gt35_3km": int(
                (observed_c >= COLUMN_THRESHOLD_DBZ).sum()),
            "cols_gt35": {
                "coarse_3km": int(
                    (coarse >= COLUMN_THRESHOLD_DBZ)[echo_c].sum()),
                "fine_blockmax_3km": int(
                    (fine_max >= COLUMN_THRESHOLD_DBZ)[echo_c].sum()),
                "fine_blockmean_3km": int(
                    (fine_mean >= COLUMN_THRESHOLD_DBZ)[echo_c].sum()),
            },
            "family_a_common_3km": {
                "coarse": ladder(coarse, observed_c, COARSE_HALF_WIDTHS, 3.0),
                "fine_blockmax": ladder(fine_max, observed_c,
                                        COARSE_HALF_WIDTHS, 3.0),
            },
            "family_c_sensitivity_blockmean": {
                "fine_blockmean": ladder(fine_mean, observed_c,
                                         COARSE_HALF_WIDTHS, 3.0),
            },
        }

        if index < len(args.obs_fine):
            observed_f, _ = observed_composite(args.obs_fine[index])
            if observed_f.shape == fine.shape:
                record["family_b_common_1p5km"] = {
                    "fine": ladder(fine, observed_f, FINE_HALF_WIDTHS, 1.5),
                    "coarse_replicated": ladder(replicate(coarse, factor),
                                                observed_f, FINE_HALF_WIDTHS,
                                                1.5),
                    "obs_cols_gt35_1p5km": int(
                        (observed_f >= COLUMN_THRESHOLD_DBZ).sum()),
                }
        frames.append(record)

    def curve(family: str, key: str, half_widths, dx_km: float):
        rows = []
        for position, half_width in enumerate(half_widths):
            values = [f[family][key][position]["fss"] for f in frames
                      if family in f]
            if values:
                rows.append({
                    "half_width": half_width,
                    "neighborhood_km": round((2 * half_width + 1) * dx_km, 1),
                    "fss_mean_over_leads": round(float(np.mean(values)), 4),
                })
        return rows

    summary = {
        "family_a_coarse": curve("family_a_common_3km", "coarse",
                                 COARSE_HALF_WIDTHS, 3.0),
        "family_a_fine_blockmax": curve("family_a_common_3km",
                                        "fine_blockmax",
                                        COARSE_HALF_WIDTHS, 3.0),
        "family_c_fine_blockmean": curve("family_c_sensitivity_blockmean",
                                         "fine_blockmean",
                                         COARSE_HALF_WIDTHS, 3.0),
        "family_b_fine": curve("family_b_common_1p5km", "fine",
                               FINE_HALF_WIDTHS, 1.5),
        "family_b_coarse_replicated": curve("family_b_common_1p5km",
                                            "coarse_replicated",
                                            FINE_HALF_WIDTHS, 1.5),
    }
    payload = {
        "schema": "gpuwm-da.resolution-neighborhood-ladder.v1",
        "members": args.members,
        "families": {
            "a": "common 3 km grid; fine reduced by 2x2 block MAX; headline; "
                 "blind below 3 km by construction",
            "b": "common 1.5 km grid; coarse replicated 2x, which adds no "
                 "information; the only family that can see below 3 km",
            "c": "family A with block MEAN; a sensitivity on the reduction "
                 "operator, which under max biases toward the fine run",
        },
        "frames": frames,
        "summary": summary,
        "resolvable_scales": {
            "family_a": "3 km and coarser; below 3 km is destroyed by the "
                        "reduction, and this family cannot speak to it",
            "family_b": "1.5 km and coarser, but scored against a 1.5 km "
                        "observed composite that is a superob of the same "
                        "radar volume and carries the radar's own resolution "
                        "limit -- the beam is about 1.7 km wide at 100 km "
                        "range, so the truth field is not meaningfully finer "
                        "than the fine grid and the 1.5 km rung is at the "
                        "edge of what the observations can adjudicate",
        },
        "caveat": "one case, one radar, radial velocity only, one "
                  "microphysics scheme, one 90-minute forecast window",
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("FAMILY A -- common 3 km grid (headline)")
    print(f"{'nbhd km':>9} {'3 km FSS':>10} {'1.5 km FSS':>11} "
          f"{'delta':>8} {'mean-reduced':>13}")
    for row_c, row_f, row_m in zip(summary["family_a_coarse"],
                                   summary["family_a_fine_blockmax"],
                                   summary["family_c_fine_blockmean"]):
        delta = row_f["fss_mean_over_leads"] - row_c["fss_mean_over_leads"]
        print(f"{row_c['neighborhood_km']:>9.1f} "
              f"{row_c['fss_mean_over_leads']:>10.4f} "
              f"{row_f['fss_mean_over_leads']:>11.4f} {delta:>+8.4f} "
              f"{row_m['fss_mean_over_leads']:>13.4f}")

    if summary["family_b_fine"]:
        print("\nFAMILY B -- common 1.5 km grid (sees below 3 km)")
        print(f"{'nbhd km':>9} {'1.5 km FSS':>11} {'3 km repl':>10} "
              f"{'delta':>8}")
        for row_f, row_c in zip(summary["family_b_fine"],
                                summary["family_b_coarse_replicated"]):
            delta = row_f["fss_mean_over_leads"] - row_c["fss_mean_over_leads"]
            print(f"{row_f['neighborhood_km']:>9.1f} "
                  f"{row_f['fss_mean_over_leads']:>11.4f} "
                  f"{row_c['fss_mean_over_leads']:>10.4f} {delta:>+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
