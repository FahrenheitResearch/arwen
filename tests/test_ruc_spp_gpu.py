"""Device/native equality and actual mixed-surface RUC runtime SPP coupling."""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from test_ruc_spp import ORACLE, RECORDS
from tools.ruc_spp_wrf_oracle.validate import assess, read_csv, replay

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["stage"] + "-" + r["label"])
def test_device_whole_stage_matches_native_overlay(record):
    oracle = read_csv(ORACLE / record["file"])
    actual = replay(record["stage"], oracle, record["mode"], gpu=True)
    # The four dry-canopy fluxes retain the same two-ULP cosine/libm
    # difference as the unperturbed reference. All prognostic fields,
    # conductivity diagnostics and complete snow columns remain bitwise.
    assess(record["stage"], oracle, actual)


def test_device_hydraulic_operator_is_bitwise_and_pattern_is_read_only():
    import cupy as cp
    from gpuwm.core.ruc_spp import hydraulic_spp_device
    source = np.geomspace(1e-30, 1e-3, 108, dtype=np.float32).reshape(9, 3, 4)
    pattern2d = np.linspace(-1, .9, 12, dtype=np.float32).reshape(3, 4)
    pattern = cp.broadcast_to(cp.asarray(pattern2d), source.shape)
    hydro = cp.asarray(source)
    diagnostic = cp.zeros_like(hydro)
    hydraulic_spp_device(hydro, pattern, diagnostic)
    expected = source * (np.float32(1) + pattern2d)
    np.testing.assert_array_equal(cp.asnumpy(hydro).view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(cp.asnumpy(diagnostic).view(np.uint32), (source * pattern2d).view(np.uint32))
    cp.testing.assert_array_equal(pattern, cp.broadcast_to(cp.asarray(pattern2d), source.shape))


def test_hydraulic_operator_preserves_subnormal_inputs_and_results():
    import cupy as cp
    from gpuwm.core.ruc_spp import hydraulic_spp_device
    tiny = np.finfo(np.float32).tiny
    source = np.broadcast_to(np.asarray([
        np.nextafter(np.float32(0), np.float32(1)), tiny * np.float32(.5),
        tiny, np.float32(1e-37)], np.float32)[:, None], (4, 4)).copy()
    pattern = np.broadcast_to(np.asarray([-.9, -.3, .3, .9], np.float32), source.shape).copy()
    hydro = cp.asarray(source)
    diagnostic = cp.zeros_like(hydro)
    hydraulic_spp_device(hydro, cp.asarray(pattern), diagnostic)
    np.testing.assert_array_equal(cp.asnumpy(hydro).view(np.uint32),
                                  (source * (np.float32(1) + pattern)).view(np.uint32))
    np.testing.assert_array_equal(cp.asnumpy(diagnostic).view(np.uint32),
                                  (source * pattern).view(np.uint32))


@pytest.mark.parametrize("nzs", [6, 9])
def test_runtime_zero_identity_nonzero_response_and_unmodified_water_ice(nzs):
    import cupy as cp
    import ruc_fused_fixture as fixture
    from gpuwm.core.ruc_runtime import ruc_lsm_step
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    from test_ruc_lsm_fused import _copy, _equal

    _, _, driver, atmosphere, cold = fixture.build(24, 20, 20, nzs, "mixed", 11)
    off = _copy(driver.fields)
    zero = _copy(driver.fields)
    perturbed = _copy(driver.fields)
    pattern2d = cp.asarray(np.linspace(-.3, .3, 24 * 20, dtype=np.float32).reshape(20, 24))
    pattern = cp.broadcast_to(pattern2d, (nzs, 20, 24))
    before_pattern = pattern.copy()
    diagnostic = cp.zeros((nzs, 20, 24), cp.float32)
    for k in range(1, 4):
        for target, mode, current_pattern in ((off, 0, None), (zero, 1, cp.zeros_like(pattern)),
                                               (perturbed, 1, pattern)):
            fixture.forcing(SimpleNamespace(fields=target), k, 11, (20, 24), cold)
            ruc_lsm_step(target, atmosphere, params=driver.ruc_params,
                         precipitation=SurfacePrecipitationForcing.from_fields(target),
                         dt=12, itimestep=k, mosaic_lu=0, mosaic_soil=0, flag_sm_adj=0,
                         spp_lsm=mode, pattern_spp_lsm=current_pattern,
                         field_sf=diagnostic if target is perturbed else None)
        _equal(off, zero)
    cp.testing.assert_array_equal(pattern, before_pattern)
    assert bool(cp.any(perturbed["smois"] != zero["smois"]))
    assert bool(cp.any(diagnostic != 0))
    # ruc_lsm_step runs with its default lakemodel=0 (2.8.5 made the CLM
    # lake selectable). Lake-masked land cells are then RUC land columns and
    # take the soil perturbation as WRF's operator does; only water and sea
    # ice bypass it. With the lake model on they would be lake columns.
    bypass = (off["xland"] >= cp.float32(1.5)) | (off["xice"] > cp.float32(.02))
    for name in ("smois", "sh2o", "tsk", "snow", "hfx", "qfx"):
        cp.testing.assert_array_equal(perturbed[name][..., bypass], zero[name][..., bypass])
    for value in perturbed.values():
        if isinstance(value, cp.ndarray) and value.dtype.kind == "f":
            assert bool(cp.all(cp.isfinite(value)))


def test_physics_driver_allocates_binds_and_consumes_spp_pattern(monkeypatch):
    import cupy as cp
    import test_ruc_runtime as fixture
    from gpuwm.core import physics
    original_config = fixture.RunConfig
    monkeypatch.setattr(fixture, "RunConfig", lambda **kwargs: original_config(**kwargs, spp_lsm=1))
    state, cfg, driver = fixture._build(nx=8, ny=6, water_columns=2)
    assert driver.fields["field_sf"].shape == (9, 6, 8)
    pattern = cp.broadcast_to(cp.full((6, 8), .3, cp.float32), (9, 6, 8))
    driver.bind_spp_patterns({"lsm": pattern})
    atmosphere = physics._prepare_atmosphere(state)
    driver._run_sfclay(atmosphere, cfg)
    original = physics.ruc_lsm_step
    seen = []
    def observed(*args, **kwargs):
        assert kwargs["spp_lsm"] == 1
        assert kwargs["pattern_spp_lsm"] is pattern
        assert kwargs["field_sf"] is driver.fields["field_sf"]
        seen.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(physics, "ruc_lsm_step", observed)
    driver._run_ruc(atmosphere, cfg, 1)
    assert seen == [True]
    assert bool(cp.any(driver.fields["field_sf"] != 0))
