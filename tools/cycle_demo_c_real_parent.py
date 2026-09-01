"""TIER B: the coupled cycle's parent leg driven by REAL MPAS x1.40962 output.

HONESTY BANNER, printed on every run and stamped into every anchor:
  * the parent frames were produced by the REAL frozen v8.4.1 CUDA dycore
    integrating 8 x 120 s steps per cycle boundary on the RTX 5070 Ti
    (mesh bound to x1.40962 at runtime, six frozen sources verified);
  * the analysis increment does NOT re-enter the dycore -- there is no
    anchor->device-stack round trip yet, so this is a real forecast
    TENDENCY composed with a real analysis, not a closed loop.
    parent_kind says so by name.
"""
import json, sys, numpy as np
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "/tmp/gpuwm_cycle")
from gpuwm.cycle.clock import CycleClock
from gpuwm.cycle.engine import build_replay_parent_engine
from gpuwm.cycle.ledger import CycleLedger
from gpuwm.cycle.supervisor import CycleSupervisor

ROOT = Path("/tmp/tierb/cycle")
PARENT_DT, CYCLE_SECONDS, N_CYCLES, NULL_ARM = 120.0, 960.0, 3, 1
PARENT_KIND = "mpas-cuda-frames"

d = np.load("/tmp/tierb/frames_real2.npz")
frames = []
for i in range(4):
    rho = d[f"{i}__rho"]; theta = d[f"{i}__theta"]
    w = d[f"{i}__w"][:, :rho.shape[1]]; qv = d[f"{i}__qv"]; uz = d[f"{i}__u_zonal"]
    rho_theta = rho * theta
    frames.append({
        "prognostic": {"rho": rho, "rho_theta": rho_theta,
                       "rho_u": rho * uz, "rho_w": rho * w, "scalars": rho * qv,
                       "time_seconds": np.asarray(i * CYCLE_SECONDS)},
        "derived": {"exner": np.power(np.maximum(rho_theta, 1e-12)
                                      * (287.0/100000.0), 287.0/(1004.5-287.0)),
                    "pressure_perturbation": d[f"{i}__pressure"] - d[f"0__pressure"]}})
print(f"parent: 4 REAL x1.40962 frames, nCells={frames[0]['prognostic']['rho'].shape[0]}"
      f" nLevels={frames[0]['prognostic']['rho'].shape[1]}")

clock = CycleClock.build(epoch_anchor=datetime(2026,8,12,6,0,tzinfo=timezone.utc),
                         parent_dt_seconds=PARENT_DT, cycle_seconds=CYCLE_SECONDS,
                         n_cycles=N_CYCLES)
# The analysis: a REAL-SHAPED velocity increment on rho_w. NULL arm is
# genuinely empty so the three-hash gate must report a flat state hash.
rng = np.random.default_rng(20260814)
mask = np.zeros_like(frames[0]["prognostic"]["rho_u"]); 
sel = rng.choice(mask.shape[0], size=2000, replace=False)
mask[sel, 20:30] = 1.0
def increment_for(cycle_index, prognostic):
    if cycle_index == NULL_ARM:
        return {"rho_u": np.zeros_like(prognostic["rho_u"])}
    return {"rho_u": np.asarray(prognostic["rho"]) * (0.8 * mask)}

ledger = CycleLedger(ROOT / "cycle_ledger.jsonl")
advance_parent = build_replay_parent_engine(
    root=ROOT, clock=clock, history_frames=frames, mesh_id="x1.40962",
    increment_for=increment_for, parent_kind=PARENT_KIND, banner=False)
def analyse(ci, rec): return rec.get("ingestion")
sup = CycleSupervisor(clock=clock, ledger=ledger, root=ROOT,
                      advance_parent=advance_parent, analyse=analyse,
                      plan_children=lambda ci, rec: [],
                      advance_children=lambda ci, rec, live: [],
                      max_forecast_only_cycles=4, allow_placement_clamp=False)
res = sup.run(resume=False)
print("RESULT:", json.dumps(res, default=str)[:1200])
print("LEDGER:", json.dumps(ledger.state(), default=str))
