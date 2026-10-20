"""Hydro cost meter on a real card with the actual parent LSM (no hydrologic claim)."""
from datetime import datetime
import json
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.config import RunConfig
from gpuwm.core.diagnostics import update_diagnostics
from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.core.moist import init_moist_balanced
from gpuwm.core.physics import initialize_physics
from gpuwm.hydro import ResidentHydroDriver, Stage, cost, forecast_products, runtime


class Spin:
    """A GPU stage with known work per launch, deliberately not a runoff module."""
    restart_identity = {"instrumentation": "hydro-door-cost-meter/v1"}

    def __init__(self, n=1 << 20, rounds=64):
        self.arrays = {"x": cp.zeros(n, cp.float32)}
        self.kernel = cp.RawKernel(
            'extern "C" __global__ void spin(float* x, int n, int rounds) {'
            ' int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;'
            ' float v = x[i]; for (int r = 0; r < rounds; ++r) v = v*0.999f + 1.0f; x[i] = v; }',
            "spin")
        self.n, self.rounds = n, rounds

    def launch(self, dt, *, stream=None):
        self.kernel(((self.n + 255) // 256,), (256,),
                    (self.arrays["x"], np.int32(self.n), np.int32(self.rounds)), stream=stream)


def _physics(monkeypatch):
    cfg = RunConfig(nx=4, ny=2, nz=12, dx=2000., dy=2000., ztop=8000., dt=10., run_seconds=0.,
                    time_step_sound=4, moist=True, mp_physics=0, sf_sfclay_physics=1,
                    sf_surface_physics=4, bl_pbl_physics=0, bldt=1., grid_id=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298. + .004*np.asarray(z, np.float64),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                               lambda z: .01*np.exp(-np.asarray(z, np.float64)/2400.))
    spin = Spin()
    module = ModuleType("gpuwm.hydro._door_cost_test")
    module.prepare = lambda **kw: ResidentHydroDriver(
        device_id=kw["device_id"], interval_seconds=60., identity={"test": "cost meter"},
        before=[Stage("spin_before", spin, spin.launch, "GPU: 1 thread per element, 4096 blocks")],
        after=[Stage("spin_after", spin, spin.launch, "GPU: 1 thread per element, 4096 blocks")])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with runtime.settings_scope({"enabled": True, "driver_factory": module.__name__ + ":prepare"}):
        physics = initialize_physics(state, cfg, soil_temperature=285., soil_moisture=.31,
                                     liquid_moisture=.31, swdown=450., glw=310.,
                                     noahmp_start_time=datetime(2026, 7, 1, 18),
                                     noahmp_latitude=40., noahmp_longitude=-100.)
    return state, cfg, physics


def test_cost_meter_on_actual_parent_records_every_lsm_call(monkeypatch, tmp_path):
    state, cfg, physics = _physics(monkeypatch)
    assert isinstance(physics.hydro_cost, cost.HydroCostMeter)
    steps = 60  # 600 s of parent time; LSM cadence 60 s -> 10 hydro calls plus the first
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    for step in range(steps):
        state.elapsed_seconds = step * cfg.dt
        update_diagnostics(state)
        physics.compute(state, cfg)
    cp.cuda.get_current_stream().synchronize()
    wall = time.perf_counter() - started
    block = cost.receipt(physics, forecast_seconds=wall, atmosphere_steps=steps)
    assert block["lsm_calls"] == physics.hydro.steps == block["sampled_calls"]
    assert block["unsampled_calls"] == 0
    assert block["hydro_ms_total"] > 0
    assert block["hydro_ms_before_lsm_total"] > 0 and block["hydro_ms_advance_total"] > 0
    assert [p["stage"] for p in block["parallelism"]] == ["spin_before", "spin_after"]
    assert block["status"] in ("PASS", "DEFECT")
    products = forecast_products.publish(physics, {}, outdir=tmp_path, dx_m=cfg.dx)
    assert products["status"] == "gap"
    (tmp_path / "hydro-cost.json").write_text(json.dumps(block, indent=2))
    print("HYDRO_COST_RECEIPT", json.dumps(block))
