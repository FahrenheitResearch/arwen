"""Replay source scheduling, moisture physics and fine fuel weighting."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json

import numpy as np

FIELDS = ("rain_old", "t2_old", "q2_old", "psfc_old", "fmc_gc", "fmep",
          "fmc_equi", "fmc_lag", "rh_fire")


def load(name, fixture_root=None):
    root = Path(__file__).parent / "fixtures" if fixture_root is None else Path(fixture_root)
    receipt = json.loads((root / "receipt.json").read_text())
    path = root / (name + ".npz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["cases"][name]["sha256"]:
        raise ValueError("Moisture driver corpus differs from the native receipt")
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def replay(case, fixture_root=None):
    import cupy as cp
    from gpuwm.core.sfire import FireOptions, FireState
    from gpuwm.core.sfire_phys import FuelTable
    from gpuwm.core.sfire_moisture import MoistureState
    from gpuwm.core.sfire_coupler import FireCoupler
    from tools.sfire_wrf471_oracle.fixture import words
    first = load(f"moisture_driver/corrected_{case}_1", fixture_root)
    table = FuelTable()
    table.moisture["moisture_classes"] = int(first["active_classes"])
    opt = FireOptions(fmoist_run=bool(first["fmoist_run"]),
        fmoist_interp=bool(first["fmoist_interp"]), fmoist_only=bool(first["fmoist_only"]),
        fmoist_freq=int(first["fmoist_freq"]), fmoist_dt=float(first["fmoist_dt"]), fire_fmc_read=0)
    moisture = MoistureState(**{key: cp.asarray(first[key + "_in"]) for key in FIELDS})
    static = cp.zeros(first["nfuel_cat"].shape, cp.float32)
    state = FireState.from_static(first["nfuel_cat"], static, static, static, 30., 20.,
        fmc_g=first["fmc_g_in"], options=opt, table=table, moisture_state=moisture)
    adapter = SimpleNamespace(coarse_shape=first["t2"].shape,
        fine_shape=tuple(n-2 for n in first["nfuel_cat"].shape), sr_x=2, sr_y=3)
    def interpolate(classes):
        return cp.stack([FireCoupler.interpolate(adapter, field) for field in classes])
    result = []
    for step in range(1, 11):
        expected = load(f"moisture_driver/corrected_{case}_{step}", fixture_root)
        state.time_seconds = float(np.float32((step-1)*float(expected["atmosphere_dt"])))
        state.step_count = step-1
        surface = {key: cp.asarray(expected[key]) for key in ("rainc", "rainnc", "t2", "q2", "psfc")}
        before = {key: cp.asnumpy(state.data[key]).tobytes() for key in
                  ("fmc_g", "fgip", "ischap", "betafl", "bbb", "fuel_time", "phiwc", "r_0", "iboros")}
        state._update_moisture(surface, interpolate)
        grades = {key: words(cp.asnumpy(getattr(moisture, key)), expected[key+"_out"]) for key in FIELDS}
        grades["fmc_g"] = words(cp.asnumpy(state.data["fmc_g"][1:-1, 1:-1]), expected["fmc_g_out"][1:-1, 1:-1])
        grades["clocks"] = words(np.array([state.moisture_lasttime, state.moisture_nexttime], np.float32),
            np.array([expected["lasttime_out"], expected["nexttime_out"]], np.float32))
        if state.moisture_initialized != bool(expected["initialized"]):
            raise AssertionError("first actual moisture initialization differs from native control")
        if opt.fmoist_only and before != {key: cp.asnumpy(state.data[key]).tobytes() for key in before}:
            raise AssertionError("moisture-only mode changed inactive fine fuel parameters")
        result.append(grades)
    return result


def grade(destination):
    import cupy as cp
    cases = {str(case): replay(case) for case in range(1, 13)}
    all_grades = [g for sequence in cases.values() for step in sequence for g in step.values()]
    summary = dict(cases=12, scheduled_steps=120,
        graded_words=sum(g["words"] for g in all_grades),
        different_words=sum(g["different_words"] for g in all_grades),
        max_ulp=max(g["max_ulp"] for g in all_grades))
    original = load("moisture_driver/original_5_3")
    corrected = load("moisture_driver/corrected_5_3")
    assert int(original["initialize"]) == 0 and int(corrected["initialize"]) == 1
    assert not np.array_equal(original["fmc_gc_out"], corrected["fmc_gc_out"])
    receipt = dict(reference="byte-extracted WRF scheduling with labeled interval-clock and first-actual-advance corrections; original WRF moisture and interpolation routines",
        device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        summary=summary, source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ("gpuwm/core/sfire.py", "gpuwm/core/sfire_moisture.py", "gpuwm/core/kernels/sfire_moisture.cu",
             "gpuwm/core/sfire_coupler.py", "gpuwm/core/kernels/sfire_coupling.cu")}, cases=cases)
    Path(destination).write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(summary))
    if summary["different_words"]:
        raise SystemExit("moisture driver replay differs from the corrected native control")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("destination")
    grade(p.parse_args().destination)
