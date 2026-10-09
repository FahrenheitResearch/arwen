"""The gsd_41 MYNN cycled start (WRF &time_control cycling, audit P18/A10).

The operational WRF 3.9 fork (NOAA-EMC/HRRR v4.1.21 module_bl_mynn.F
:4004-4031, :4105-4112) starts a cycled forecast from the input QKE unless
its lowest-level maximum is below 0.0002, and never zeroes the input QC_BL
and CLDFRA_BL: the first radiation call merges them (radiation runs before
the PBL) and the first cloud decay limits how fast they fall.  A cold start
zeroes all three.  These tests bind each branch of that decision on the
driver fixture's five columns.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

import cupy as cp

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

from test_mynn_pbl import _driver_step

pytestmark = pytest.mark.gpu

_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
            "qke", "tsq", "qsq", "cov", "el", "sh", "sm", "qc_bl",
            "cldfra_bl", "exch_h", "exch_m", "pblh", "rmol")


def _values(*, qke=None, qc_bl=None, cldfra_bl=None):
    _, values, initflag, delt = _driver_step(1)
    assert initflag == 1
    out = {name: np.ascontiguousarray(np.asarray(value))
           for name, value in values.items()}
    sqv = out["sqv"].astype(np.float64)
    out["qv"] = (sqv / (1.0 - sqv)).astype(np.float32)
    out["qc"] = (out["sqc"] / (1.0 - sqv)).astype(np.float32)
    out["qi"] = (out["sqi"] / (1.0 - sqv)).astype(np.float32)
    for name, value in (("qke", qke), ("qc_bl", qc_bl),
                        ("cldfra_bl", cldfra_bl)):
        if value is not None:
            out[name] = np.ascontiguousarray(
                np.broadcast_to(np.float32(value), out[name].shape)
                if np.ndim(value) == 0 else np.asarray(value, np.float32))
    return out, delt


def _run(values, delt, **options):
    device = {name: cp.asarray(value) for name, value in values.items()}
    out = mynn_bl_driver_cuda(device, initflag=1, delt=delt,
                              flag_qs=True, bl_mynn_version="gsd_41",
                              **options)
    return {name: cp.asnumpy(out[name]).copy() for name in _OUTPUTS
            if name in out}


def _identical(left, right):
    for name in left:
        np.testing.assert_array_equal(
            np.ascontiguousarray(left[name]).view(np.uint32),
            np.ascontiguousarray(right[name]).view(np.uint32), err_msg=name)


def _differs(left, right, name):
    return not np.array_equal(left[name], right[name])


@requires_gpu
def test_cycled_start_without_carried_state_is_the_cold_start_bitwise():
    """A cycled start whose input carries no turbulence or subgrid cloud
    (the HRRR route's GRIB start today) is the cold start word for word."""
    values, delt = _values(qke=0.0, qc_bl=0.0, cldfra_bl=0.0)
    _identical(_run(values, delt),
               _run(values, delt, cycling=True))
    _identical(_run(values, delt),
               _run(values, delt, cycling=True, initialize_qke=True))


@requires_gpu
def test_cold_start_ignores_the_input_turbulence():
    """The cold start seeds QKE from u* whatever the input carries.  (The
    input subgrid cloud is read once, by the first PBL-height profile, in
    the fork too: module_bl_mynn.F:4084-4096 reads QC_BL and CLDFRA_BL
    before any cycling test, so a cold start is not independent of it.)"""
    values, delt = _values(qke=0.0)
    carried, _ = _values(qke=0.9)
    _identical(_run(values, delt), _run(carried, delt))


@requires_gpu
def test_the_qke_threshold_decides_the_seed():
    """Below 0.0002 at the lowest level the fork seeds as a cold start;
    at or above it the carried QKE is the first call's turbulence."""
    cold, delt = _values(qke=0.0, qc_bl=0.0, cldfra_bl=0.0)
    low, _ = _values(qke=1.9e-4, qc_bl=0.0, cldfra_bl=0.0)
    high, _ = _values(qke=0.9, qc_bl=0.0, cldfra_bl=0.0)
    reference = _run(cold, delt)
    _identical(reference, _run(low, delt, cycling=True))
    carried = _run(high, delt, cycling=True)
    assert _differs(reference, carried, "qke")
    assert _differs(reference, carried, "exch_h")
    # The decision is the domain's, handed in by the runtime: forcing the
    # seed reproduces the cold start even with a carried field present.
    _identical(reference, _run(high, delt, cycling=True,
                               initialize_qke=True))
    # A carried QKE moves the step; the outputs stay finite.
    for name, value in carried.items():
        assert np.isfinite(value).all(), name


@requires_gpu
@pytest.mark.parametrize("seed", (
    np.nextafter(np.float32(0.0002), np.float32(0.0)),
    np.float32(0.0002),
    np.nextafter(np.float32(0.0002), np.float32(1.0)),
))
def test_cycled_seed_decision_compares_the_fortran_float32_threshold(seed):
    values, delt = _values(qke=seed, qc_bl=0.0, cldfra_bl=0.0)
    expected_seed = bool(seed < np.float32(0.0002))
    _identical(_run(values, delt, cycling=True),
               _run(values, delt, cycling=True, initialize_qke=expected_seed))


@requires_gpu
@pytest.mark.parametrize("generation", ("wrf_461", "gsd_41"))
@pytest.mark.parametrize("chunk", (1, 17))
def test_runtime_threshold_decision_is_shared_by_every_column_piece(monkeypatch, chunk, generation):
    import test_mynn_pbl_runtime as fixture
    import gpuwm.core.mynn_pbl_runtime as runtime
    import gpuwm.core.mynn_pbl_scratch as scratch
    from gpuwm.core.dycore import step
    import sys

    # The width memo and published widths are process state. Keep this
    # deliberately narrow run from changing a later test's column population.
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    monkeypatch.setattr(scratch, "_TILE_WALKED", {})
    monkeypatch.setattr(scratch, "_PINNED", None)
    for module_name in scratch._CHUNK_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, "MYNN_PBL_COLUMN_CHUNK",
                                module.MYNN_PBL_COLUMN_CHUNK)
    monkeypatch.setenv("GPUWM_MYNN_COLUMN_CHUNK", str(chunk))

    original_config = fixture.RunConfig
    monkeypatch.setattr(fixture, "RunConfig", lambda **kw: original_config(
        **kw, bl_mynn_version=generation, bl_mynn_mixlength=2, cycling=True))
    state, cfg, driver = fixture._build()
    threshold = np.float32(0.0002)
    driver.fields["qke"][0].fill(np.nextafter(threshold, np.float32(0)))
    driver.fields["qke"][0, -1, -1] = threshold
    seen = []
    original = runtime.mynn_bl_driver_cuda

    def observe(values, **kwargs):
        seen.append(kwargs["initialize_qke"])
        return original(values, **kwargs)

    monkeypatch.setattr(runtime, "mynn_bl_driver_cuda", observe)
    step(state, cfg)
    assert len(seen) > 1 and all(value is False for value in seen)


@requires_gpu
def test_carried_subgrid_cloud_reaches_the_first_cloud_decay():
    """The cycled start keeps QC_BL and CLDFRA_BL for the first call's
    cloud decay; the cold start zeroes them first."""
    values, delt = _values(qke=0.0)
    carried, _ = _values(qke=0.0, qc_bl=3.0e-4, cldfra_bl=0.6)
    cycled_empty = _run(values, delt, cycling=True)
    cycled_cloud = _run(carried, delt, cycling=True)
    assert _differs(cycled_empty, cycled_cloud, "cldfra_bl")
    # The fork's decay (module_bl_mynn.F:4690-4713) never lets the carried
    # 0.6 fall by more than 0.25*delt/ts_decay in this call, with
    # ts_decay = MIN(1800, 3*dx/MAX(|V|, 1)).
    speed = np.hypot(carried["u"].astype(np.float64),
                     carried["v"].astype(np.float64))
    ts_decay = np.minimum(1800.0, 3.0 * carried["dx"].astype(np.float64)[:, None]
                          / np.maximum(speed, 1.0))
    floor = 0.6 - 0.25 * float(delt) / ts_decay - 1.0e-6
    assert (cycled_cloud["cldfra_bl"] >= floor).all()
    assert (cycled_empty["cldfra_bl"] < floor).any()


@requires_gpu
def test_seed_override_requires_a_cycled_start():
    values, delt = _values()
    device = {name: cp.asarray(value) for name, value in values.items()}
    with pytest.raises(ValueError, match="initialize_qke"):
        mynn_bl_driver_cuda(device, initflag=1, delt=delt, flag_qs=True,
                            bl_mynn_version="gsd_41", initialize_qke=False)


def _cfg(**overrides):
    values = dict(nx=8, ny=8, nz=30, dx=3000.0, dy=3000.0, ztop=16000.0,
                  dt=20.0, run_seconds=60.0, bl_pbl_physics=5,
                  sf_sfclay_physics=5, moist=True, mp_physics=8)
    values.update(overrides)
    return RunConfig(**values)


def test_run_config_admits_cycling_for_both_generations():
    validate_run_config(_cfg())
    validate_run_config(_cfg(bl_mynn_version="gsd_41", cycling=True))
    validate_run_config(_cfg(cycling=True))
    with pytest.raises(TypeError, match="cycling"):
        validate_run_config(_cfg(bl_mynn_version="gsd_41", cycling=1))
