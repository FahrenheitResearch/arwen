"""Short GPU integrations for the newly admitted MYNN surface pairings."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu

import cupy as cp


def _digest_array(digest, name, value):
    host = np.ascontiguousarray(cp.asnumpy(value))
    digest.update(name.encode("utf-8"))
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(np.asarray(host.shape, np.int64).tobytes())
    digest.update(host.tobytes())


@requires_gpu
@pytest.mark.parametrize("land_surface", [3, 4], ids=["ruc", "noahmp"])
def test_wsm6_mynn_surface_pairing_runs_three_finite_steps(
        land_surface, record_property):
    """Exercise WSM6 + MYNN/MYNN + each LSM through the real dycore driver."""
    from gpuwm.core.dycore import step

    if land_surface == 3:
        from test_ruc_runtime import _build
    else:
        from test_noahmp_runtime import _build

    state, cfg, driver = _build(
        nx=4, ny=2, nz=12, water_columns=0, mp_physics=6,
        sf_sfclay_physics=5, bl_pbl_physics=5)
    steps = 3
    for _ in range(steps):
        step(state, cfg)
        cp.cuda.get_current_stream().synchronize()
        for name, value in state.__dict__.items():
            if isinstance(value, cp.ndarray):
                assert bool(cp.all(cp.isfinite(value)).item()), f"state/{name}"
        for name, value in driver.fields.items():
            assert bool(cp.all(cp.isfinite(value)).item()), f"fields/{name}"

    census = (
        driver.last_ruc_census
        if land_surface == 3 else driver.last_noahmp_census
    )
    assert census is not None
    assert census["land"] == cfg.nx * cfg.ny
    assert driver.scheme_dispatch["sf_sfclay_physics"] == "_run_sfclay"
    assert driver.scheme_dispatch["bl_pbl_physics"] == "_run_mynn_pbl"
    assert driver.scheme_dispatch["sf_surface_physics"] == (
        "_run_ruc" if land_surface == 3 else "_run_noahmp")
    assert float(state.elapsed_seconds) == steps * cfg.dt

    digest = hashlib.sha256()
    for name in sorted(state.__dict__):
        value = getattr(state, name)
        if isinstance(value, cp.ndarray):
            _digest_array(digest, f"state/{name}", value)
    for name in (
            "ust", "hfx", "qfx", "lh", "chs", "chs2", "cqs2",
            "flhc", "flqc", "t2", "q2", "th2"):
        _digest_array(digest, f"fields/{name}", driver.fields[name])
    receipt = {
        "schema": "gpuwm-mynn-surface-pairing-integration-v1",
        "selectors": {
            "mp_physics": 6,
            "sf_sfclay_physics": 5,
            "bl_pbl_physics": 5,
            "sf_surface_physics": land_surface,
        },
        "steps": steps,
        "elapsed_seconds": float(state.elapsed_seconds),
        "census": census,
        "finite": True,
        "sha256": digest.hexdigest(),
    }
    record_property("surface_pairing_receipt", json.dumps(
        receipt, sort_keys=True))
    assert receipt["sha256"] != "0" * 64
