"""What actually limits dt at dx = 1 m: the sound step, or the wind?

The CBL probe box is quiescent -- horizontal advective CFL ~1e-4 -- so a dt
limit measured on it is the ACOUSTIC limit and is irrelevant to a tornado.
This sweep adds a uniform mean wind (a Galilean shift on a doubly periodic
box: no physics changes, the Courant number does) and asks three separate
questions:

A  at 100 m/s, how small must dt be?
B  at 100 m/s and dt = 6 ms, does any config knob rescue it?
C  at dt = 6 ms, at what wind speed does it break?

Every trial reports the step it failed at, so a slow instability is not
recorded as a pass.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import cupy as cp

# The probe is the sibling file, loaded by location rather than imported so
# this runs the same whether or not ``tools`` is on the path.
_PROBE = Path(__file__).resolve().with_name("les1m_probe.py")
sys.path.insert(0, str(_PROBE.parent.parent))
_spec = importlib.util.spec_from_file_location("probe", str(_PROBE))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

from gpuwm.core.dycore import run_steps          # noqa: E402

NSTEP = 1500
ROWS = []


def trial(U, dt, label, **over):
    cfg = probe.cfg_for(64, 64, 48, 1.0, 64.0, dt, 3, **over)
    st = probe.build(cfg, 20.0, mean_wind=U)
    done, fail = 0, None
    h = probe.cfl_numbers(st, cfg)
    while done < NSTEP:
        try:
            run_steps(st, cfg, 50)
        except Exception as exc:                   # noqa: BLE001
            fail, h = done, h
            label += f" raised {type(exc).__name__}"
            break
        done += 50
        h = probe.cfl_numbers(st, cfg)
        # Failure = non-finite, or a perturbation that has grown past
        # 50 m/s on top of the imposed mean.  Both are unambiguous.
        if h["nan"] or not math.isfinite(h["u_max"]) or h["u_max"] > U + 50:
            fail = done
            break
    row = {"label": label, "U": U, "dt": dt,
           "cfl_adv": U * dt / 1.0,
           "cfl_sound": h["cfl_sound_horiz"],
           "failed_at_step": fail, "steps": done,
           "u_max": h["u_max"], "w_max": h["w_max"],
           "ok": fail is None}
    ROWS.append(row)
    print(f"{label:26s} U={U:5.0f} dt={dt * 1000:6.2f}ms "
          f"cfl_adv={U * dt:5.3f} cfl_snd={h['cfl_sound_horiz']:5.3f} -> "
          f"{'FAIL@' + str(fail) if fail else 'OK  ' + str(done):>10s} "
          f"u_max={h['u_max']:.5g}", flush=True)
    del st
    cp.get_default_memory_pool().free_all_blocks()
    return fail is None


print("== A: U=100 m/s, dt ladder (base CBL/LES config) ==", flush=True)
for dt in (0.006, 0.004, 0.003, 0.002, 0.0015, 0.001):
    trial(100.0, dt, "base")

print("== B: U=100 m/s, dt=6 ms, stabilisers ==", flush=True)
trial(100.0, 0.006, "emdiv=0.01", emdiv=0.01)
trial(100.0, 0.006, "h_sca_adv_order=5", h_sca_adv_order=5)
trial(100.0, 0.006, "emdiv+adv5", emdiv=0.01, h_sca_adv_order=5)
trial(100.0, 0.006, "diff_6th_opt=2", diff_6th_opt=2, diff_6th_factor=0.12)
trial(100.0, 0.006, "ts_sound=6", time_step_sound=6)
trial(100.0, 0.006, "epssm=0.5", epssm=0.5)
trial(100.0, 0.006, "smdiv=0.5", smdiv=0.5)

print("== C: dt=6 ms, wind ladder (base) ==", flush=True)
for U in (40.0, 50.0, 60.0, 70.0, 85.0):
    trial(U, 0.006, "base")

print("== D: best stabiliser, dt ladder at 100 m/s ==", flush=True)
for dt in (0.006, 0.005, 0.004, 0.003):
    trial(100.0, dt, "emdiv+adv5+d6", emdiv=0.01, h_sca_adv_order=5,
          diff_6th_opt=2, diff_6th_factor=0.12)

with open("/tmp/claude-1000/-home-drew-bowecho-dea/"
          "12456cae-783d-4a37-9cd5-d2db7c7bd8da/scratchpad/out/"
          "cflsweep.json", "w") as fh:
    json.dump({"nstep": NSTEP, "rows": ROWS}, fh, indent=1, default=float)
print("DONE", flush=True)
