"""POOL_SLACK_FRACTION reduced re-fit, post-#310, on the RTX 3080.

SCOPE, stated up front: this is a REDUCED version of the 2026-08-19
six-forecast calibration -- the legacy-RRTMG lane only, at the
calibration's own two grid sizes, 2 h instead of 6 h, plus the two
whole-forecast parents this lane already ran. It is enough to say
whether the calibrated 0.20 still describes the legacy lane's pool
retention on a post-#310 build; it is not enough to re-derive a new
coefficient. The full protocol is in named-follow-ups.md.

The quantity is BASELINE-FREE by construction: pool_total_peak /
alloc_estimate is in-process CuPy-pool accounting, so the desktop
compositor sharing this card cannot contaminate it -- unlike the
machine-wide device peak, which on this desktop carries a ~2.5-3.0 GB
baseline and must not be read as a residual.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: The 2026-08-19 calibration's own legacy rows (pre-#310), from
#: docs/public/receipts/wddm/rtx3080-wddm-calibration-20260819.json.
CALIBRATION_LEGACY = {
    "t60x48": {"alloc_estimate_gib": 2.941, "pool_total_peak_gib": 3.732},
    "t110x88": {"alloc_estimate_gib": 3.164, "pool_total_peak_gib": 4.644},
}

harvest = subprocess.run(
    [sys.executable, "tools/vram_residue_harvest.py", "--scan",
     "runs/f4-downscale"],
    cwd=REPO, capture_output=True, text=True)
if harvest.returncode != 0:
    sys.exit(f"harvest failed: {harvest.stderr[-2000:]}")
scan = json.loads(harvest.stdout)

rows = []
for run in scan["runs"]:
    est = run["alloc_estimate_bytes"]
    rows.append({
        "run": run["model"],
        "grid": run["grid"],
        "alloc_estimate_gib": round(est / 2**30, 3),
        "pool_total_peak_gib": round(run["pool_total_peak_bytes"] / 2**30, 3),
        "pool_over_estimate": round(run["pool_total_peak_bytes"] / est, 4),
        "measured_slack_fraction": round(
            max(0.0, run["pool_total_peak_bytes"] - est) / est, 4),
    })

pre = {name: {
    "pool_over_estimate": round(v["pool_total_peak_gib"]
                                / v["alloc_estimate_gib"], 4),
    "measured_slack_fraction": round(
        (v["pool_total_peak_gib"] - v["alloc_estimate_gib"])
        / v["alloc_estimate_gib"], 4)} for name, v in
    CALIBRATION_LEGACY.items()}

report = {
    "measurement": "POOL_SLACK_FRACTION reduced re-fit, 2026-08-26",
    "scope": ("legacy-RRTMG lane at the calibration's own two grid sizes "
              "plus this lane's two whole-forecast parents; 2 h runs on "
              "the RTX 3080 10 GiB, post-#310 build. NOT the full "
              "six-forecast 6 h protocol -- see named-follow-ups.md"),
    "quantity": ("pool_total_peak / alloc_estimate, in-process CuPy pool "
                 "accounting (baseline-free; the machine-wide device peak "
                 "on this desktop carries a 2.5-3.0 GB compositor baseline "
                 "and is NOT used as a residual here)"),
    "shipped_constant": 0.20,
    "pre_310_calibration_legacy_rows": pre,
    "post_310_measured": rows,
}
print(json.dumps(report, indent=2))
(HERE / "slack_refit_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
