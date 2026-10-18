"""The analysis door's worker on the card: the increment it forces lands whole, a zero
increment steps the same bytes as no analysis, and a forcing rebuilt at a resume continues
the unbroken leg bit for bit.

The column is the IAU suite's balanced parent over a hill (tests/test_da_iau_forcing_gpu.py),
physics off and periodic, so the coupled theta integral is conserved by the transport and
what it gains is what the forcing put in.  The whole forecast through ``gpuwm run --wrfinput
--analysis`` on a real case is measured outside this suite (it needs a real.exe pair).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.gpu]

VALID = datetime(2022, 6, 13, 23)
WINDOW = 300.0


def _host(array):
    import cupy as cp

    return cp.asnumpy(array).astype(np.float64)


def _plan(tmp_path: Path, spread: dict, at_once: dict | None = None, *, window: float = WINDOW,
          start: str = "iau") -> dict:
    from gpuwm.da import analysis_door as door

    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    split = work / door.SPLIT
    sha = door._write_split(split, at_once or {}, spread)
    return {"schema": door.PLAN_SCHEMA, "start": start, "valid_time": VALID.strftime("%Y-%m-%d_%H:%M:%S"),
            "bundle": {"manifest_sha256": "0" * 64, "grid": {"id": "gpu-test", "horizontal_sha256": "1" * 64}},
            "iau": {"on": True, "mode": "3d", "fields": "all", "window_seconds": window},
            "work": str(work), "record": {}, "run_directory": str(tmp_path),
            "forcing": {"path": str(split), "sha256": sha}, "derived": {}}


def _attach(tmp_path: Path, state, run, plan: dict, *, restart=None):
    from gpuwm.da import analysis_door as door

    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    worker = door.AnalysisInitialization(SimpleNamespace(lateral_boundaries=None), plan,
                                         outdir=tmp_path / "out")
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(run=run))
    worker.after_restore(model=SimpleNamespace(root=node), nodes={1: node}, start_time=VALID,
                         run_seconds=3600.0, restart=restart)
    return worker, SimpleNamespace(root=node)


def _zeros(state) -> dict:
    return {"U": np.zeros(state.u.shape), "V": np.zeros(state.v.shape), "W": np.zeros(state.w.shape),
            "T": np.zeros(state.thp.shape), "PH": np.zeros(state.php.shape),
            "QVAPOR": np.zeros(state.qv.shape)}


def test_the_doors_forcing_lands_whole(tmp_path):
    from test_da_iau_forcing_gpu import DT, _blob, _coupled_theta, _integrate, _run_cfg, _state

    from gpuwm.da import iau

    run = _run_cfg()
    control = _state(run)
    _integrate(control, run, steps=40)
    state = _state(run)
    blob = _host(_blob(state))
    spread = _zeros(state)
    spread["T"] = blob
    mu = _host(state.mub2d + state.mup)
    chm = _host(state.c1h)[:, None, None] * mu[None] + _host(state.c2h)[:, None, None]
    expected = float((chm * (-_host(state.dnw))[:, None, None] * blob).sum())
    worker, model = _attach(tmp_path, state, run, _plan(tmp_path, spread))
    forcing = getattr(state, iau.IAU_STATE_ATTRIBUTE)
    _integrate(state, run, steps=40)
    added = _coupled_theta(state) - _coupled_theta(control)
    print("door forcing added", added, "expected", expected)
    assert abs(added - expected) < 0.01 * abs(expected)
    block = worker.run_report_blocks(model)["analysis_bundle"]["iau_forcing"]
    assert abs(block["applied_fraction"] - 1.0) < 1e-12
    assert block["steps_forced"] == int(WINDOW / DT)
    start = json.loads((tmp_path / "out" / "forecast-start.json").read_text())
    assert start["start"] == "iau" and start["start_seconds"] == 0.0


def test_a_zero_increment_steps_the_same_bytes_as_no_analysis(tmp_path):
    from test_da_iau_forcing_gpu import _integrate, _run_cfg, _state

    from gpuwm.da import iau
    from gpuwm.ensemble.state_sha import live_state_sha256

    run = _run_cfg()
    plain = _state(run)
    _integrate(plain, run, steps=30)
    forced = _state(run)
    _attach(tmp_path, forced, run, _plan(tmp_path, _zeros(forced)))
    forcing = getattr(forced, iau.IAU_STATE_ATTRIBUTE)
    _integrate(forced, run, steps=30)
    iau.detach(forced)
    assert forcing.steps_forced == int(WINDOW / run.dt)
    assert live_state_sha256(plain) == live_state_sha256(forced)


def test_a_forcing_rebuilt_at_a_resume_continues_the_unbroken_leg_bit_for_bit(tmp_path):
    from test_da_iau_forcing_gpu import _blob, _integrate, _run_cfg, _state

    from gpuwm.da import iau
    from gpuwm.ensemble.state_sha import live_state_sha256

    run = _run_cfg()
    unbroken = _state(run)
    spread = _zeros(unbroken)
    spread["T"] = _host(_blob(unbroken))
    spread["U"] = np.full(unbroken.u.shape, 0.5)
    plan = _plan(tmp_path / "a", spread)
    _attach(tmp_path / "a", unbroken, run, plan)
    _integrate(unbroken, run, steps=40)
    iau.detach(unbroken)

    resumed = _state(run)
    _attach(tmp_path / "b", resumed, run, plan)
    _integrate(resumed, run, steps=9)
    iau.detach(resumed)                        # the checkpoint holds no forcing
    checkpoint = tmp_path / "ckpt.gpuwmrst"
    checkpoint.write_bytes(b"inside the window")
    worker, model = _attach(tmp_path / "c", resumed, run, plan, restart=checkpoint)
    rebuilt = getattr(resumed, iau.IAU_STATE_ATTRIBUTE)
    assert rebuilt.applied_fraction == pytest.approx(9 * run.dt / WINDOW, abs=1e-12)
    assert worker.run_report_blocks(model)["analysis_bundle"]["applied_now"]["applied"] is False
    _integrate(resumed, run, steps=31)
    iau.detach(resumed)
    assert abs(rebuilt.applied_fraction - 1.0) < 1e-12
    assert live_state_sha256(unbroken) == live_state_sha256(resumed)
