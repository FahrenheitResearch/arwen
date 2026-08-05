"""Separate the two reasons FSS moves when the ensemble grows.

The published metric scores the ensemble-MEAN composite.  That makes a
raw skill-versus-N curve confounded, because growing N does two
different things at once:

*   **Analysis quality.**  A larger ensemble gives the LETKF a better
    sampled background covariance, so every member is drawn from a
    better-corrected state.  This is the thing the owner is asking
    about.
*   **Averaging depth.**  The scored field is a mean of N composites.
    Averaging more fields smooths the forecast, and a smoothed forecast
    scores better under a neighborhood metric whose truth field is also
    smoothed.  This costs a member each and buys nothing a forecaster
    can use.

The two are separable because they can be varied independently in post-
processing:

*   Hold the ANALYSIS fixed and vary the DEPTH -- inside one N-member
    run, score the mean of a k-member subset for k = 1..N.  Every subset
    came from the same analysis, so the entire rise with k is the
    smoothing artifact.
*   Hold the DEPTH fixed and vary the ANALYSIS -- score the mean of
    exactly k members drawn from the N = 10, 20, 36, 64 runs.  The
    averaging is identical, so the difference is analysis quality alone.

Depth-1 (the mean over members of each member's own FSS) is the
statistic that never averages away with N, and it is reported for every
N with its across-member spread.

Subsets are drawn at random with a fixed seed and averaged over several
draws, so a curve is not an artefact of which members happen to be
first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from score_free_forecast import (COLUMN_THRESHOLD_DBZ, HALF_WIDTH_CELLS,
                                 THRESHOLD_DBZ, observed_composite)


def fss(field: np.ndarray, observed: np.ndarray) -> float:
    from gpuwm.verify.field_metrics import fss_distance

    return 1.0 - fss_distance(field, observed, threshold=THRESHOLD_DBZ,
                              half_width=HALF_WIDTH_CELLS)


def load_stack(composites: Path, leg: int, members: int) -> np.ndarray:
    return np.stack([
        np.asarray(np.load(composites / f"leg{leg:02d}_{m}.npz")
                   ["refl_colmax"], np.float64)
        for m in range(members)])


def subset_fss(stack: np.ndarray, observed: np.ndarray, depth: int,
               draws: int, rng: np.random.Generator) -> dict:
    """Mean FSS of a depth-member mean, over several random subsets."""
    members = stack.shape[0]
    if depth >= members:
        return {"depth": depth, "draws": 1,
                "fss_mean": round(fss(stack.mean(axis=0), observed), 4),
                "fss_std": 0.0}
    if depth == 1:
        # The end of the curve that matters most, and the one place an
        # exhaustive answer is free: score every member, no sampling.
        scores = [fss(stack[m], observed) for m in range(members)]
        return {"depth": 1, "draws": members, "exhaustive": True,
                "fss_mean": round(float(np.mean(scores)), 4),
                "fss_std": round(float(np.std(scores)), 4)}
    scores = []
    for _ in range(draws):
        pick = rng.choice(members, size=depth, replace=False)
        scores.append(fss(stack[pick].mean(axis=0), observed))
    return {"depth": depth, "draws": draws,
            "fss_mean": round(float(np.mean(scores)), 4),
            "fss_std": round(float(np.std(scores)), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--members", type=int, action="append", required=True)
    parser.add_argument("--obs", type=Path, action="append", required=True)
    parser.add_argument("--first-free-leg", type=int, default=6)
    parser.add_argument("--fixed-depth", type=int, default=10,
                        help="averaging depth held constant across N when "
                             "isolating analysis quality (default 10, the "
                             "ensemble size in production today)")
    parser.add_argument("--draws", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = [{"lead_minutes": 15 * (i + 1), "obs": path}
              for i, path in enumerate(args.obs)]
    for frame in frames:
        frame["observed"], _ = observed_composite(frame["obs"])

    runs = {}
    for members in args.members:
        composites = args.root / f"cycle-n{members}" / "composites"
        if not composites.is_dir():
            continue
        runs[members] = composites
    if not runs:
        raise SystemExit("no composites found under --root")

    payload: dict = {
        "schema": "gpuwm-da.ensemble-skill-decomposition.v1",
        "why": "the published metric scores the ensemble-MEAN composite, so "
               "a raw FSS-versus-N curve mixes analysis quality with the "
               "smoothing that comes free from averaging more fields; these "
               "two families vary one of those at a time",
        "fixed_depth": args.fixed_depth,
        "per_member": {},
        "depth_curve_fixed_analysis": {},
        "analysis_curve_fixed_depth": {},
        "published_mean_field": {},
    }

    rng_seed = args.seed
    for members, composites in sorted(runs.items()):
        per_member_by_lead, published, depth_rows = [], [], {}
        for index, frame in enumerate(frames):
            leg = args.first_free_leg + index
            stack = load_stack(composites, leg, members)
            observed = frame["observed"]

            # Depth 1 -- the statistic that does not average away.
            singles = [fss(stack[m], observed) for m in range(members)]
            per_member_by_lead.append({
                "lead_minutes": frame["lead_minutes"],
                "fss_member_mean": round(float(np.mean(singles)), 4),
                "fss_member_std": round(float(np.std(singles)), 4),
                "fss_member_min": round(float(np.min(singles)), 4),
                "fss_member_max": round(float(np.max(singles)), 4),
            })

            # The published number, for continuity.
            published.append({
                "lead_minutes": frame["lead_minutes"],
                "fss": round(fss(stack.mean(axis=0), observed), 4),
                "cols_gt35": int((stack.mean(axis=0)
                                  >= COLUMN_THRESHOLD_DBZ).sum()),
            })

            # Depth curve, analysis held fixed at this N.
            rng = np.random.default_rng(rng_seed + leg)
            depths = [d for d in (1, 2, 4, 8, 10, 16, 20, 32, 36, 48, 64)
                      if d <= members]
            if members not in depths:
                depths.append(members)
            depth_rows[frame["lead_minutes"]] = [
                subset_fss(stack, observed, d, args.draws, rng)
                for d in depths]

            # Analysis curve, depth held fixed across N.
            if members >= args.fixed_depth:
                rng2 = np.random.default_rng(rng_seed + 7919 + leg)
                row = subset_fss(stack, observed, args.fixed_depth,
                                 args.draws, rng2)
                payload["analysis_curve_fixed_depth"].setdefault(
                    str(members), []).append(
                        {"lead_minutes": frame["lead_minutes"], **row})

        payload["per_member"][str(members)] = per_member_by_lead
        payload["published_mean_field"][str(members)] = published
        payload["depth_curve_fixed_analysis"][str(members)] = depth_rows

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- the two curves, printed side by side --------------------------
    def lead_avg(rows, key):
        return round(float(np.mean([r[key] for r in rows])), 4)

    print("\nDEPTH-1 (per-member FSS, mean over members; no averaging "
          "artifact possible)")
    print(f"{'N':>5} {'FSS mean':>10} {'across-member std':>18}")
    for members in sorted(runs):
        rows = payload["per_member"][str(members)]
        print(f"{members:>5} {lead_avg(rows, 'fss_member_mean'):>10.4f} "
              f"{lead_avg(rows, 'fss_member_std'):>18.4f}")

    print(f"\nANALYSIS QUALITY (averaging depth held at "
          f"{args.fixed_depth} for every N)")
    print(f"{'N':>5} {'FSS':>10}")
    for members in sorted(runs):
        rows = payload["analysis_curve_fixed_depth"].get(str(members))
        if rows:
            print(f"{members:>5} {lead_avg(rows, 'fss_mean'):>10.4f}")

    print("\nPUBLISHED (mean of all N; depth and analysis both move)")
    print(f"{'N':>5} {'FSS':>10}")
    for members in sorted(runs):
        rows = payload["published_mean_field"][str(members)]
        print(f"{members:>5} {lead_avg(rows, 'fss'):>10.4f}")

    largest = max(runs)
    print(f"\nSMOOTHING ARTIFACT ALONE (one analysis, N={largest}, "
          f"averaging depth varied)")
    rows = payload["depth_curve_fixed_analysis"][str(largest)]
    depths = [r["depth"] for r in next(iter(rows.values()))]
    print(f"{'depth':>7} {'FSS':>10}")
    for position, depth in enumerate(depths):
        values = [rows[lead][position]["fss_mean"] for lead in rows]
        print(f"{depth:>7} {float(np.mean(values)):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
