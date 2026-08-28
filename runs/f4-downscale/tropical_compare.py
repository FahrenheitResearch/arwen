"""TROPICAL_ROOT_TIME_STEP_S settling measurement, both arms.

The motivating case re-run under the CORRECTED v1.1 monitor (co-located
|w|/dz), on the RTX 3080: the wizard's own tropical Mercator emission at
14.6 N, 6 h GFS, one arm on the halved clock the guard imposes (30 s) and
one on the un-halved 5 s/km clock (60 s). Everything else identical --
same TOML, same staged GFS, same card, same build; the only edit between
the arms is the root time_step.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load(arm):
    report = HERE / f"tropical-{arm}-run" / "run" / "report.json"
    if not report.is_file():
        return {"arm": arm, "status": "DID NOT PRODUCE A REPORT",
                "note": "the run stopped before writing its receipt"}
    d = json.loads(report.read_text(encoding="utf-8"))
    health = d.get("health") or {}
    final = health.get("final_stability") or {}
    hist = health.get("history") or []
    timing = d.get("timing_seconds") or {}
    return {
        "arm": arm,
        "status": d.get("status"),
        "run_seconds_simulated": d.get("run_seconds"),
        "forecast_wall_seconds": (timing.get("forecast")
                                  if isinstance(timing, dict) else None),
        "final_vertical_cfl": final.get("vertical_cfl"),
        "final_horizontal_cfl": final.get("horizontal_cfl"),
        "final_interior_w_max": final.get("interior_w_max"),
        "final_w_max": final.get("w_max"),
        "nan": final.get("nan"),
        "peak_vertical_cfl_over_history": max(
            [h.get("cfl") for h in hist if isinstance(h.get("cfl"), (int, float))],
            default=None),
        "peak_w_max_over_history": max(
            [h.get("w_max") for h in hist
             if isinstance(h.get("w_max"), (int, float))], default=None),
        "history_samples": len(hist),
    }


arms = [load("30s"), load("60s")]
report = {
    "measurement": "TROPICAL_ROOT_TIME_STEP_S settling, 2026-08-26",
    "case": ("wizard emission at 14.6 N / 120.98 E, Mercator, 386x308 at "
             "12 km, morrison rte-rrtmgp, 6 h GFS 2026-08-26T00, "
             "RTX 3080 10 GiB"),
    "instrument": ("the run's own v1.1 stability monitor -- co-located "
                   "|w|/dz, the corrected form; the instrument whose "
                   "earlier version exaggerated the motivating event by "
                   "pairing the global w maximum with the unrelated "
                   "thinnest layer"),
    "shipped_constant_s": 30,
    "unhalved_clock_s": 60,
    "arms": arms,
}
print(json.dumps(report, indent=2))
(HERE / "tropical_compare_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
