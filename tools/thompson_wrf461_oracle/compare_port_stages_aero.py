"""Where the port's mp=28 call first leaves WRF v4.6.1, stage by stage.

Runs the production adapter on the committed column fixtures with hooks at
the five boundaries instrument_stages_aero.py dumps in WRF, forms the port's
working value of every species there (state, or entry + accumulator*dt for
the accumulated ones) and compares it with WRF's ``X1d + Xten*DT`` in float32
ulps of the larger magnitude.  Values after the sedimentation stage are not
comparable for qr, nr, qi and ni by construction while the port folds the
terminal rain and ice bounds into its fallout kernels.

usage, from a source tree root with a card:
    python tools/thompson_wrf461_oracle/compare_port_stages_aero.py         <STAGE_DUMP build dir>/intermediates [fixture ...]
"""
import csv
import sys

import cupy as cp
import numpy as np

sys.path.insert(0, "tests")
import test_thompson_aerosol_adapter as T  # noqa: E402
import gpuwm.core.microphysics_aerosol as MA  # noqa: E402
from gpuwm.core import thompson_aerosol_sat as S, thompson_aerosol_sed as D  # noqa: E402
from gpuwm.core import thompson_aerosol_warm as Wm  # noqa: E402

f32 = np.float32
ORACLE = sys.argv[1]
FIXTURES = sys.argv[2:] or list(T._FIXTURES)
SPECIES = ("qc", "qr", "nr", "qi", "ni", "qs", "qg", "t", "qv", "nc")
TEN = {"qc": "qcten", "qr": "qrten", "nr": "nrten", "qi": "qiten", "ni": "niten",
       "qs": "qsten", "qg": "qgten", "t": "tten", "qv": "qvten", "nc": "ncten"}
ONE = {"qc": "qc1d", "qr": "qr1d", "nr": "nr1d", "qi": "qi1d", "ni": "ni1d",
       "qs": "qs1d", "qg": "qg1d", "t": "t1d", "qv": "qv1d", "nc": "nc1d"}


def wrf_stages(name, dt):
    rows = list(csv.DictReader(open(f"{ORACLE}/{name}-stages.csv")))
    out = {}
    for r in rows:
        st = out.setdefault(r["stage"], {})
        k = int(r["k"]) - 1
        for s in SPECIES:
            one = f32(float(r[ONE[s]]))
            ten = f32(float(r[TEN[s]]))
            st.setdefault(s, {})[k] = f32(one + f32(ten * f32(dt)))
    return {st: {s: np.array([v[k] for k in sorted(v)], f32) for s, v in d.items()}
            for st, d in out.items()}


def port_snapshot(state, dt):
    def col(a):
        return cp.asnumpy(a).reshape(a.shape[0], -1)[:, 0].astype(f32)
    snap = {"qr": col(state.qr), "nr": col(state.nr), "qi": col(state.qi), "ni": col(state.ni),
            "qs": col(state.qs), "qg": col(state.qg), "qv": col(state.qv),
            "t": col(state._scratch["mp_thompson_temperature"])}
    for s, slot in (("qc", "mp_thompson_aero_qcten"), ("nc", "mp_thompson_aero_ncten")):
        entry = col(getattr(state, s))
        if slot in state._scratch:
            snap[s] = f32(entry + f32(col(state._scratch[slot]) * f32(dt)))
        else:
            snap[s] = entry
    for s, slot in (("qr", "mp_thompson_aero_qrten"), ("nr", "mp_thompson_aero_nrten"),
                    ("qi", "mp_thompson_aero_qiten"), ("ni", "mp_thompson_aero_niten")):
        if slot in state._scratch:
            snap[s] = f32(snap[s] + f32(col(state._scratch[slot]) * f32(dt)))
    return snap


def ulps(a, b):
    a = a.astype(f32)
    b = b.astype(f32)
    scale = np.maximum(np.abs(a), np.abs(b))
    u = np.spacing(np.maximum(scale, f32(1e-38)).astype(f32))
    return np.where(a == b, 0.0, np.abs(a.astype(np.float64) - b.astype(np.float64)) / u)


def run(name):
    stages = {}
    state_box = {}
    hooks = [("A-sources", Wm, "launch_ncten_balance", "after"),
             ("B-condensation", S, "launch_aerosol_saturation_adjust", "after"),
             ("C-rainevap", S, "launch_aerosol_rain_evaporation", "after"),
             ("D-sedimentation", D, "launch_aa_final_phase_cleanup", "before"),
             ("E-cleanup", D, "launch_aa_final_phase_cleanup", "after")]
    originals = {}
    for stage, mod, attr, when in hooks:
        originals.setdefault((mod, attr), getattr(mod, attr))
    by_attr = {}
    for stage, mod, attr, when in hooks:
        by_attr.setdefault((mod, attr), []).append((stage, when))
    for (mod, attr), lst in by_attr.items():
        orig = originals[(mod, attr)]

        def wrapped(*a, _orig=orig, _lst=lst, **k):
            for stage, when in _lst:
                if when == "before":
                    stages[stage] = port_snapshot(state_box["s"], state_box["dt"])
            r = _orig(*a, **k)
            for stage, when in _lst:
                if when == "after":
                    stages[stage] = port_snapshot(state_box["s"], state_box["dt"])
            return r
        setattr(mod, attr, wrapped)
    try:
        state, cfg, dt, before, after, surface, report = T._build_case(cp, name)
        state_box.update(s=state, dt=dt)
        MA._apply_thompson_aerosol(state, cfg, dt, refl_10cm_due=True)
    finally:
        for (mod, attr), orig in originals.items():
            setattr(mod, attr, orig)
    wrf = wrf_stages(name, dt)
    lines = []
    for stage in ("A-sources", "B-condensation", "C-rainevap", "D-sedimentation", "E-cleanup"):
        if stage not in wrf or stage not in stages:
            continue
        cells = []
        for s in SPECIES:
            u = ulps(stages[stage][s], wrf[stage][s])
            if u.max() > 0:
                k = int(u.argmax())
                cells.append(f"{s}:{int((u > 0).sum())}lv max {u.max():.3g}ulp@k{k}")
        lines.append(f"  {stage:16s} " + ("; ".join(cells) if cells else "bitwise"))
    print(name)
    print("\n".join(lines))


for name in FIXTURES:
    run(name)
