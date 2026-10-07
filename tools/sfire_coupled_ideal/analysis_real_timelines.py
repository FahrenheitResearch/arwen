"""Analysis charts from native Rust metrics, with observed initialization separated."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("evidence", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows, sources, observations = {}, {}, None
    for arm in ("age900", "age0"):
        names = [arm + "-timeline.json", arm + "-observed.json", "a63-" + arm + "-run-receipt.json"]
        timeline, observed, run = [json.loads((a.evidence / name).read_text()) for name in names]
        sources.update({name: sha(a.evidence / name) for name in names})
        assert run["status"] == "PASS" and run["experiment"]["run_seconds"] == 7200
        assert len(timeline["frames"]) == 25
        rows[arm] = []
        for f in timeline["frames"]:
            source = f["fields"]["SFIRE_SMOKE_FUEL_SOURCE"]
            row = dict(arm=arm, seconds=f["time_seconds"], fire_area_m2=f["fire_area_m2"],
                       nominal_grid_perimeter_m=f["perimeter_m"], sensible_power_w=f["sensible_power_w"],
                       latent_power_w=f["latent_power_w"],
                       stored_cumulative_bulk_source_kg=source["mean"] * source["words"],
                       native_smoke_max_g_kg_air=f["fields"]["fire_smoke"]["max"],
                       aq_bulk_smoke_max_ug_kg_dry=f["fields"]["FIRE_SMOKE"]["max"])
            assert all(math.isfinite(v) for k, v in row.items() if k != "arm")
            rows[arm].append(row)
        assert rows[arm][0]["seconds"] == 0 and rows[arm][-1]["seconds"] == 7200
        expected = run["chem_ledger"]["d03"]["sources"]["emission.sfire"]["fire_smoke"]["expected_fuel_source_kg"]
        assert math.isclose(rows[arm][-1]["stored_cumulative_bulk_source_kg"], expected, rel_tol=1e-6)
        start = datetime.fromisoformat(run["experiment"]["start_time"]).replace(tzinfo=timezone.utc).timestamp()
        parsed = [(v["observed_utc_epoch_ms"] * .001 - start, v["observed_area_m2"])
                  for v in observed["comparisons"]]
        if observations is None:
            observations = parsed
        else:
            assert observations == parsed
    path = a.output / "real-fire-timelines.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows["age900"][0]))
        w.writeheader()
        for arm in rows:
            w.writerows(rows[arm])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    choices = (("fire_area_m2", 1e-6, "Native FIRE_AREA (km²)"),
               ("nominal_grid_perimeter_m", .001, "Perimeter on nominal WRF grid (km)"),
               ("sensible_power_w", 1e-9, "Stored sensible heat power (GW)"),
               ("stored_cumulative_bulk_source_kg", .001, "Stored cumulative bulk smoke source (tonnes)"))
    for ax, (key, scale, title) in zip(axes.flat, choices):
        for arm, color, label in (("age900", "#b23a3a", "Assumed burn age 900 s"),
                                   ("age0", "#255f9c", "Assumed burn age 0 s")):
            ax.plot([v["seconds"] / 60 for v in rows[arm]], [v[key] * scale for v in rows[arm]],
                    color=color, label=label)
        ax.set(xlabel="Elapsed model minutes", ylabel=title, xlim=(0, 120))
        ax.grid(alpha=.25)
    axes[0, 0].scatter([observations[0][0] / 60], [observations[0][1] * 1e-6],
                       color="#777777", marker="x", s=60, label="Assimilated initial polygon")
    axes[0, 0].scatter([v[0] / 60 for v in observations[1:]], [v[1] * 1e-6 for v in observations[1:]],
                       color="#111111", marker="o", label="Later observed polygons (evaluation)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Coupled real fire: both age assumptions underpredict the later observed growth\n"
                 "a63f43f5 science snapshot | 3 nests | 50 m fire grid | bulk smoke is not PM2.5", fontsize=12)
    png = a.output / "real-fire-timelines.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    caption = ("Native Rust NetCDF summaries of both complete 7200-second runs; 25 leaf frames each. "
               "The first observed polygon initializes the model and is excluded from skill. "
               "The two later observations are evaluated at nearest actual UTC outputs, with offsets +105.031 and -74.449 seconds. "
               "FIRE_AREA uses nominal WRF fire-cell area. The contour-distance scores use the separate EPSG:5070 geometry comparator. "
               "Heat power is stored-history sampling; integrated heat uses the 300-second history trapezoid, not every dynamics step. "
               "The bulk source curve sums stored cell diagnostics; the full precision fuel and geometry source totals and ledgers are in the run receipts. "
               "No PM2.5, suppression, or forecast skill claim is made. Missing DNW/C1H/C2H leave atmospheric mass metrics explicitly null.\n")
    (a.output / "REAL-CAPTIONS.txt").write_text(caption, encoding="utf-8")
    qa = dict(schema="sfire-real-analysis-qa-v1", source_sha256=sources, arms=list(rows),
              samples_per_arm=25, start_seconds=0, end_seconds=7200, finite_values=True,
              weather_field_maps=False, source_receipt_agreement_rtol=1e-6,
              artifacts={q.name: {"bytes": q.stat().st_size, "sha256": sha(q)}
                         for q in (path, png, a.output / "REAL-CAPTIONS.txt")})
    (a.output / "real-analysis-qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa))


if __name__ == "__main__":
    main()
