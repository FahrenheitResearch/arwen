"""Active classic collision columns and the retained rain concentration owner."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import thompson

FIXTURE = Path(__file__).parent / "fixtures/thompson-active-collision.json"


def test_rain_density_history_rejects_overlapping_output():
    arrays = [np.zeros((4, 1, 1), np.float32) for _ in range(6)]
    with pytest.raises(ValueError, match="source_density must not alias reference_density"):
        thompson.launch_rain_evaporation(
            *arrays[:5], 10., reference_density=arrays[5], source_density=arrays[5])


@pytest.fixture
def cp_backend():
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("device verification is disabled")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("device verification needs CUDA")
    return cp


@pytest.mark.gpu
@pytest.mark.parametrize("evaporating", [False, True])
def test_rain_density_history_follows_actual_evaporation(cp_backend, evaporating):
    cp = cp_backend
    temperature = np.float32(274.15)
    qv = np.float32(.004 if evaporating else .006)
    rho = np.float32(.622) * np.float32(80000.) / (
        np.float32(287.04) * temperature * (qv + np.float32(.622)))
    values = [.0003, 30000., temperature, 80000., qv]
    arrays = [cp.full((4, 1, 1), v, dtype=cp.float32) for v in values]
    source = cp.full_like(arrays[0], rho * np.float32(.99))
    density = cp.zeros_like(arrays[0])
    rain_before = arrays[0].copy()
    thompson.launch_rain_evaporation(
        *arrays, 10., reference_density=density, source_density=source)
    expected = np.float32(rho if evaporating else rho * np.float32(.99))
    np.testing.assert_allclose(cp.asnumpy(density), expected, rtol=3e-7, atol=0.)
    if evaporating:
        assert bool(cp.all(arrays[0] < rain_before))
    else:
        assert bool(cp.array_equal(arrays[0], rain_before))


@pytest.mark.gpu
def test_equal_density_history_preserves_isolated_evaporation(cp_backend):
    cp = cp_backend
    values = [.0003, 30000., 274.15, 80000., .004]
    a = [cp.full((4, 1, 1), v, dtype=cp.float32) for v in values]
    b = [v.copy() for v in a]
    density_a, density_b = cp.zeros_like(a[0]), cp.zeros_like(a[0])
    thompson.launch_rain_evaporation(*a, 10., reference_density=density_a)
    thompson.launch_rain_evaporation(
        *b, 10., reference_density=density_b, source_density=density_a.copy())
    for actual, expected in zip((*b, density_b), (*a, density_a), strict=True):
        assert bool(cp.array_equal(actual, expected))


@pytest.mark.gpu
def test_positive_cloud_condensation_suppresses_same_call_rain_evaporation(cp_backend):
    cp = cp_backend
    a = [cp.full((4, 1, 1), v, dtype=cp.float32)
         for v in (274.15, 80000., .0051506985910236835 * 1.1, .001)]
    density, marker = cp.empty_like(a[0]), cp.empty_like(a[0])
    thompson.launch_cloud_saturation_adjust(
        *a, reference_density=density, condensation_marker=marker)
    assert bool(cp.all(marker == 1.))
    temperature, pressure, vapor, cloud = a
    rain = cp.full_like(cloud, .0003)
    number = cp.full_like(cloud, 30000.)
    before = [v.copy() for v in (rain, number, temperature, vapor)]
    thompson.launch_rain_evaporation(
        rain, number, temperature, pressure, vapor, 10.,
        source_density=density, reference_density=cp.empty_like(density),
        condensation_marker=marker)
    for actual, expected in zip((rain, number, temperature, vapor), before, strict=True):
        assert bool(cp.array_equal(actual, expected))


def _column(cp, case):
    from gpuwm.config import RunConfig
    from gpuwm.core import constants
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.state import init_theta_perturbation

    cfg = RunConfig(nx=16, ny=12, nz=4, dx=1000., dy=1000., ztop=1175.,
                    dt=case["dt_s"], run_seconds=20., time_step_sound=4,
                    moist=True, mp_physics=8)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_theta_perturbation(
        cfg, coord, base, lambda x, z: np.zeros((cfg.nz, 1, 1)))
    profile = lambda n: np.asarray([r[n] for r in case["before"]], np.float32)
    state.thb.fill(0.)
    state.thp[...] = cp.asarray(profile("theta_k"))[:, None, None]
    state.p[...] = cp.asarray(profile("p_pa"))[:, None, None]
    state.php.fill(0.)
    state.w.fill(0.)
    z = np.concatenate(([0.], np.cumsum(profile("dz_m")))).astype(np.float32)
    state.phb[...] = cp.asarray(z * np.float32(constants.G)).reshape(state.phb.shape)
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni"):
        key = name + "_per_kg" if name in ("nr", "ni") else name
        getattr(state, name)[...] = cp.asarray(profile(key))[:, None, None]
    initialize_physics(state, cfg, landmask=1.)
    return state, cfg


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["saturated", "subsaturated", "supersaturated"])
def test_active_collision_column_matches_full_reference_and_heating(cp_backend, name):
    cp = cp_backend
    from gpuwm.core import microphysics

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = next(c for c in fixture["cases"] if c["name"] == name)
    results = []
    for due in (False, True):
        state, cfg = _column(cp, case)
        before_theta = state.thp.copy()
        result = microphysics._apply_thompson(state, cfg, cfg.dt, refl_10cm_due=due)
        cp.cuda.Stream.null.synchronize()
        expected_heat = (state._scratch["mp_th"] - before_theta) / np.float32(cfg.dt)
        assert bool(cp.array_equal(state.h_diabatic, expected_heat))
        assert bool(cp.all(cp.abs(state.h_diabatic) < cfg.mp_tend_lim))
        observed = {}
        for field in ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "thp", "h_diabatic"):
            actual = cp.asnumpy(getattr(state, field))[:, 0, 0]
            assert np.isfinite(actual).all()
            observed[field] = actual
            if field == "h_diabatic":
                continue
            key = "theta_k" if field == "thp" else field + "_per_kg" if field in ("nr", "ni") else field
            expected = np.asarray([r[key] for r in case["after"]], np.float32)
            if field == "thp":
                np.testing.assert_allclose(actual, expected, rtol=0., atol=2 * np.spacing(expected).max())
            elif field in ("nr", "ni"):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=.03)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-11)
        for field in ("rainncv", "snowncv", "graupelncv"):
            actual = float(cp.asnumpy(getattr(result, field))[0, 0])
            expected = case["surface"][field + "_mm"]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-8)
            observed[field] = actual
        assert (state.physics.refl_10cm is not None) == due
        results.append(observed)
    for field in results[0]:
        np.testing.assert_array_equal(results[0][field], results[1][field])
