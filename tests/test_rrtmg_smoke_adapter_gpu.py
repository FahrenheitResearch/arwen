"""Prescribed smoke reaches surface SW through the actual legacy adapter.

These supplied profiles are numerical test inputs, not a smoke analysis or
an observation score. The atmospheric column comes from the SW oracle deck.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

START = datetime(2026, 10, 2, 21)


def _manifest(tmp_path, nz, lat, lon, layer_aod, p_top):
    from gpuwm.core.rrtmg_smoke_manifest import SCHEMA

    folder = tmp_path / ("zero" if layer_aod == 0 else "positive")
    folder.mkdir()
    def member(name, value, units):
        path = folder / (name + ".f32le")
        value = np.asarray(value, dtype="<f4")
        value.tofile(path)
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dtype": "<f4", "shape": list(value.shape), "units": units}
    eta = np.linspace(1.0, 0.0, nz + 1).tolist()
    profile = np.full((nz, *lat.shape), layer_aod, np.float32)
    value = member("aod", profile, "1")
    text = {"schema": SCHEMA, "quantity": "layer_aod", "units": "1",
            "shape": list(profile.shape), "vertical_order": "bottom_to_top",
            "time_interpolation": "linear", "start_time": "2026-10-02T21:00:00Z",
            "geometry": {"latitude": member("lat", lat, "degrees_north"),
                         "longitude": member("lon", lon, "degrees_east")},
            "provenance": {"vertical": {"eta_levels": eta, "hybrid_opt": 2,
                                        "etac": .2, "p_top": p_top}},
            "frames": [{"valid_time": "2026-10-02T21:00:00Z", "value": value},
                       {"valid_time": "2026-10-02T23:00:00Z", "value": value}]}
    path = folder / "smoke.json"
    path.write_text(json.dumps(text), encoding="utf-8")
    return path, eta


def test_actual_shortwave_adapter_zero_identity_and_positive_smoke_response(tmp_path):
    import cupy as cp
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    from gpuwm.core.rrtmg_smoke_manifest import BoundSmokeManifest
    from test_rrtmg_legacy_wiring import _bundle, _env_from, _call, profile

    ny = nx = 2
    lat = np.array([[40., 40.], [40.02, 40.02]], np.float32)
    lon = np.array([[-100., -99.98], [-100., -99.98]], np.float32)
    bundle = _bundle(profile.__wrapped__(), ny * nx, seed=20261004)
    env = _env_from(bundle, np.arange(ny * nx), ny, nx, lat.ravel(), lon.ravel())
    for name in ("qc", "qr", "qi", "qs", "qg"):
        getattr(env.state, name).fill(cp.float32(0))
    env.state.nwfa = cp.full_like(env.state.qv, cp.float32(1.e8))
    env.state.nifa = cp.full_like(env.state.qv, cp.float32(1.e4))
    env.state.elapsed_seconds = 0.0
    # The SW-only adapter retains the driver's previously computed LW flux.
    env.fields["glw"] = cp.full((ny, nx), cp.float32(300.))
    env.cfg.mp_physics = 28
    env.cfg.aer_opt = 3
    env.cfg.o3input = 0
    env.cfg.swint_opt = 0
    env.cfg.rrtmg_smoke_manifest = ""
    zero_path, eta = _manifest(tmp_path, env.nz, lat, lon, 0., env.p_top)
    positive_path, _ = _manifest(tmp_path, env.nz, lat, lon, .005, env.p_top)

    def adapter(path=None):
        provider = None
        if path is not None:
            provider = BoundSmokeManifest(path, START, lat, lon, env.nz)
            provider.require_vertical(eta, 2, .2, env.p_top)
            provider.require_coverage(7200)
        return RRTMGLegacyRadiation(START, lat, lon, p_top=env.p_top,
            o3input=0, longwave=False, shortwave=True, aer_opt=3, smoke_provider=provider)

    # Repeat constructions/calls to expose hidden retained state.
    passes = []
    for repetition in range(2):
        env.cfg.rrtmg_smoke_manifest = ""
        baseline = cp.asnumpy(_call(adapter(), env).swdown)
        env.cfg.rrtmg_smoke_manifest = str(zero_path)
        zero = cp.asnumpy(_call(adapter(zero_path), env).swdown)
        np.testing.assert_array_equal(zero.view(np.uint32), baseline.view(np.uint32))
        env.cfg.rrtmg_smoke_manifest = str(positive_path)
        positive = cp.asnumpy(_call(adapter(positive_path), env).swdown)
        assert np.isfinite(positive).all()
        assert (baseline > 0).all(), "the numerical fixture must be sunlit"
        assert (positive < baseline).all(), "positive prescribed AOD must reduce clear-sky surface SW"
        assert (baseline - positive).min() > 1., "the response must exceed float32 rounding"
        passes.append({"pass": repetition + 1,
            "baseline_swdown_w_m2": baseline.tolist(),
            "supplied_zero_swdown_w_m2": zero.tolist(),
            "supplied_positive_swdown_w_m2": positive.tolist(),
            "reduction_w_m2": (baseline - positive).tolist(),
            "zero_profile_exact_float32_words": True})
    receipt_path = os.environ.get("WOOF_SMOKE_ADAPTER_RECEIPT", "")
    if receipt_path:
        path = Path(receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": "sw-excess-smoke-adapter-numerical-v1",
            "case_start_utc": "2026-10-02T21:00:00Z", "grid_shape": [env.nz, ny, nx],
            "aer_opt": 3, "o3input": 0, "positive_layer_aod_550nm": .005,
            "source_identity": "oracle atmosphere with numerical prescribed AOD profiles",
            "claim": "adapter wiring and numerical response, no forecast or observation skill",
            "passes": passes}, indent=2) + "\n", encoding="utf-8")
