"""Render analysis charts from verified campaign scores, without weather fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Inside the wheel these are tools.<name>; the bare form is what a
# run by path resolves (`python tools/<this file>`), where tools/ is
# sys.path[0].
try:
    from tools.ensemble_calibration_score import sha256
except ImportError:
    from ensemble_calibration_score import sha256


def plot_scores(summary, out):
    if summary.get("schema") != "gpuwm-ensemble-campaign-scores.v1" or summary.get("status") != "complete":
        raise ValueError("analysis charts require a complete native campaign score summary")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    rows = summary["products"]
    files = []
    quantities = ["temperature_2m", "wind_speed_10m", "precipitation_accumulation"]
    titles = {"temperature_2m": "2 m temperature", "wind_speed_10m": "10 m sustained wind",
              "precipitation_accumulation": "Forecast-start precipitation"}
    for case in sorted({row["case_id"] for row in rows}):
        selected = [row for row in rows if row["case_id"] == case]
        recipes = list(dict.fromkeys(row["recipe"] for row in selected))
        colors = {recipe: plt.get_cmap("tab20")(index % 20) for index, recipe in enumerate(recipes)}
        lookup = {(row["quantity"], row["recipe"]): row for row in selected}
        split = "held out" if selected[0]["held_out"] else "training"
        subtitle = f"{case} | {split} | leads {summary['lead_hours'][0]} to {summary['lead_hours'][-1]} h"

        def finish(figure, name, caption):
            figure.suptitle(subtitle, fontsize=12)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            for suffix in ("png", "svg"):
                path = out / f"{case}-{name}.{suffix}"
                figure.savefig(path, dpi=180)
                files.append({"path": str(path), "sha256": sha256(path), "caption": caption})
            plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(17, max(5, 0.42 * len(recipes) + 2)))
        for axis, quantity in zip(axes, quantities):
            units = next(row["units"] for row in selected if row["quantity"] == quantity)
            y = list(range(len(recipes)))
            crps = [lookup[(quantity, recipe)]["scores"]["crps"] for recipe in recipes]
            rmse = [lookup[(quantity, recipe)]["scores"]["ensemble_mean_rmse"] for recipe in recipes]
            axis.barh([v - 0.19 for v in y], crps, height=0.36, label="Empirical CRPS", color="#276a9d")
            axis.barh([v + 0.19 for v in y], rmse, height=0.36, label="Ensemble-mean RMSE", color="#d19a45")
            axis.set_yticks(y, recipes)
            axis.invert_yaxis()
            axis.set_xlabel(units)
            axis.set_title(titles[quantity])
            axis.grid(axis="x", alpha=0.2)
        axes[0].legend(loc="lower right", fontsize=9)
        finish(figure, "errors", "Empirical CRPS and ensemble-mean RMSE on identical observed rows. Lower is better; quantities use different units.")

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, quantity in zip(axes, quantities):
            limit = 0.
            units = lookup[(quantity, recipes[0])]["units"]
            for recipe in recipes:
                row = lookup[(quantity, recipe)]
                rmse, spread = row["scores"]["ensemble_mean_rmse"], row["scores"]["rms_member_sample_spread"]
                if spread is None:
                    continue
                limit = max(limit, rmse, spread)
                axis.scatter(rmse, spread, color=colors[recipe], label=f"{recipe} (N={row['members']})")
            axis.plot([0, limit * 1.05], [0, limit * 1.05], color="0.55", linestyle="--", linewidth=1)
            axis.set_title(titles[quantity])
            axis.set_xlabel(f"Ensemble-mean RMSE ({units})")
            axis.set_ylabel(f"RMS member sample spread ({units})")
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7, loc="best")
        finish(figure, "spread-skill", "Raw sample spread versus ensemble-mean RMSE. The dashed line marks equality; N=1 has undefined sample spread and is omitted. Finite-ensemble-corrected ratios remain in the score receipts.")

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, quantity in zip(axes, quantities):
            for recipe in recipes:
                row = lookup[(quantity, recipe)]
                if row["members"] < 2:
                    continue
                weights = row["scores"]["rank_weight"]
                total = sum(weights)
                if total <= 0:
                    raise ValueError("rank weights are empty for a completed ensemble")
                bins = len(weights)
                axis.plot([(i + 0.5) / bins for i in range(bins)], [weight * bins / total for weight in weights],
                          color=colors[recipe], label=f"{recipe} (N={row['members']})", marker=".")
            axis.axhline(1, color="0.55", linestyle="--", linewidth=1)
            axis.set_title(titles[quantity])
            axis.set_xlabel("Normalized rank bin")
            axis.set_ylabel("Weight relative to a uniform rank histogram")
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7, loc="best")
        finish(figure, "ranks", "Fractional-tie rank histograms normalized to uniform density, with each ensemble's N retained. N=1 is omitted; the number of bins is N+1.")

        for quantity in quantities:
            thresholds = lookup[(quantity, recipes[0])]["scores"]["threshold_scores"]
            figure, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5), squeeze=False)
            for index, threshold in enumerate(thresholds):
                axis = axes[0, index]
                for recipe in recipes:
                    row = lookup[(quantity, recipe)]
                    term = row["scores"]["threshold_scores"][index]
                    if term["threshold"] != threshold["threshold"]:
                        raise ValueError("reliability chart received different production thresholds")
                    bins = term["reliability"]
                    axis.plot([value["probability"] for value in bins], [value["observed_frequency"] for value in bins],
                              color=colors[recipe], marker="o", markersize=4, label=recipe)
                units = lookup[(quantity, recipes[0])]["units"]
                axis.set_title(f"{titles[quantity]} >= {threshold['threshold']:g} {units}")
                axis.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1)
                axis.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Forecast probability", ylabel="Observed frequency")
                axis.grid(alpha=0.2)
            axes[0, 0].legend(fontsize=7, loc="best")
            finish(figure, f"{quantity}-reliability", "Reliability at actual production thresholds. Points are exact forecast-probability bins; empty bins are absent. Bin weights and Brier scores are retained in the native score receipts.")
    return {"schema": "gpuwm-ensemble-analysis-charts.v1", "files": files,
            "scope": "analysis charts from matched-pair scores; no weather fields or inferred qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.scores.read_text(encoding="utf-8"))
    result = plot_scores(summary, args.out)
    result["scores"] = {"path": str(args.scores), "sha256": sha256(args.scores)}
    (args.out / "analysis-charts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "captions.txt").write_text("\n".join(f"{Path(row['path']).name}: {row['caption']}" for row in result["files"] if row["path"].endswith(".png")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
