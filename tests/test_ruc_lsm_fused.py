"""Whole-call bitwise checks for the full-width RUC driver kernels."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


def _harness():
    path = Path(__file__).with_name("ruc_fused_fixture.py")
    spec = importlib.util.spec_from_file_location("ruc_bench_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(width, nzs, scenario):
    import cupy as cp
    bench = _harness()
    _, _, driver, atmosphere, cold = bench.build(97, 61, 20, nzs, scenario, 11)
    index = cp.arange(width) % (97 * 61)

    def resize(array):
        if isinstance(array, cp.ndarray) and array.shape[-2:] == (61, 97):
            return cp.ascontiguousarray(array.reshape(array.shape[:-2] + (-1,))[..., index]
                                       .reshape(array.shape[:-2] + (1, width)))
        return array.copy() if hasattr(array, "copy") else array

    fields = {name: resize(array) for name, array in driver.fields.items()}
    atmosphere = {name: resize(array) for name, array in atmosphere.items()}
    cold = cold.reshape(-1)[np.arange(width) % (97 * 61)].reshape(1, width)
    return bench, SimpleNamespace(fields=fields, ruc_params=driver.ruc_params), atmosphere, cold


def _copy(fields):
    return {name: array.copy() if hasattr(array, "copy") else array
            for name, array in fields.items()}


def _equal(left, right):
    import cupy as cp
    for name in left:
        if isinstance(left[name], cp.ndarray):
            np.testing.assert_array_equal(cp.asnumpy(left[name]).view(np.uint8),
                                          cp.asnumpy(right[name]).view(np.uint8),
                                          err_msg=name)


def _call(function, fields, atmosphere, params, k):
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    return function(fields, atmosphere, params=params,
                    precipitation=SurfacePrecipitationForcing.from_fields(fields),
                    dt=12.0, itimestep=k, mosaic_lu=0, mosaic_soil=0,
                    flag_sm_adj=0, spp_lsm=0)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [9, 6])
@pytest.mark.parametrize("scenario", ["mixed", "warm"])
@pytest.mark.parametrize("width", [1, 3, 17, 5917, 100000])
def test_every_field_after_every_call(width, nzs, scenario):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    bench, driver, atmosphere, cold = _case(width, nzs, scenario)
    reference = _copy(driver.fields)
    if width >= 5917 and scenario == "mixed":
        # Two fixed fractions supplement the randomized mixed fixture.
        for fields in (driver.fields, reference):
            fields["xice"][0, 0:2] = cp.asarray([0.55, 0.85], dtype=cp.float32)
    for k in range(1, 8):
        bench.forcing(driver, k, 11, (1, width), cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, (1, width), cold)
        actual = _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k)
        expected = _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, k)
        assert actual == expected
        _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
def test_negative_control_detects_one_word():
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    reference = _copy(driver.fields)
    reference["psfc"].view(cp.uint32)[0, 0] ^= cp.uint32(1)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 1)
    _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, 1)
    with pytest.raises(AssertionError):
        _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name,value", [("psfc", np.nan), ("psfc", 0.0),
                                       ("cqs2", np.nan), ("chs2", np.nan),
                                       ("psfc", np.asarray(0x7fc01234, dtype=np.uint32).view(np.float32)[()])])
def test_diagnostic_nan_words_match_numpy(name, value):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    driver.fields[name][0, 0] = cp.float32(value)
    reference = _copy(driver.fields)
    _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, 2)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 2)
    _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("reverse", [False, True])
def test_diagnostic_signed_zero_ties(reverse):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    driver.fields["lakemask"][0, 0] = 1.0
    driver.fields["xice"][0, 0] = 0.0
    driver.fields["tsk"][0, 0] = cp.float32(0.0 if reverse else -0.0)
    driver.fields["qsfc"][0, 0] = cp.float32(0.0 if reverse else -0.0)
    driver.fields["chs2"][0, 0] = cp.float32(1.0e-7)
    driver.fields["cqs2"][0, 0] = cp.float32(1.0e-7)
    atmosphere["temperature"][0, 0, 0] = cp.float32(-0.0 if reverse else 0.0)
    atmosphere["qv"][0, 0, 0] = cp.float32(-0.0 if reverse else 0.0)
    reference = _copy(driver.fields)
    _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, 2)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 2)
    _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["input", "category", "output", "driver_output",
                                  "qsn_overflow", "epilogue_qsn", "surface_derived_nan", "precedence"])
def test_refusal_matches_and_never_commits(kind, monkeypatch):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    # These first columns are bare land in the warm fixture.
    driver.fields["xice"].fill(0)
    if kind in ("input", "precedence"):
        driver.fields["smois"][0, 0, 0] = cp.nan
    if kind in ("category", "precedence"):
        driver.fields["ivgtyp"][0, 1] = 100
    if kind == "output":
        # qkms is built inside the driver as FLQC/RHO/MAVAIL.  Finite zero
        # inputs produce a NaN, which sfctmp admits before any leaf runs.
        driver.fields["flqc"][0, 0] = 0.0
        driver.fields["mavail"][0, 0] = 0.0
    if kind == "surface_derived_nan":
        driver.fields["ivgtyp"][0, 0] = 1
        driver.fields["shdmin"][0, 0] = -cp.finfo(cp.float32).max
        driver.fields["shdmax"][0, 0] = cp.finfo(cp.float32).max
        driver.fields["vegfra"][0, 0] = cp.finfo(cp.float32).max
    k = 2
    if kind == "qsn_overflow":
        # The prologue qsn refusal must precede the overflowing TSNAV and
        # every sfctmp or driver-output check in this lake column.
        k = 1
        driver.fields["lakemask"][0, 0] = 1.0
        driver.fields["tsk"][0, 0] = cp.finfo(cp.float32).max
        driver.fields["tslb"][0, 0, 0] = cp.finfo(cp.float32).max
    if kind == "epilogue_qsn":
        driver.fields["xland"][0, 0] = 2.0
        atmosphere["temperature"][0, 0, 0] = cp.finfo(cp.float32).max
    if kind == "driver_output":
        from dataclasses import replace
        from gpuwm.core import ruc
        original = ruc.ruc_surface_temperature_step
        def large_finite_flux(*args, **kwargs):
            result = original(*args, **kwargs)
            return replace(result, eeta=cp.full_like(result.eeta, cp.finfo(cp.float32).max))
        monkeypatch.setattr(ruc, "ruc_surface_temperature_step", large_finite_flux)
        # The fused path runs sfctmp as kernels, not through the host
        # function, so the same overflow is injected at its hook.
        from gpuwm.core import ruc_fused
        fused = ruc_fused._RUC_SFCTMP_FULL_WIDTH
        def large_finite_flux_fused(*args, **kwargs):
            result = fused(*args, **kwargs)
            result["eeta"] = cp.full_like(result["eeta"], cp.finfo(cp.float32).max)
            return result
        monkeypatch.setattr(ruc_fused, "_RUC_SFCTMP_FULL_WIDTH", large_finite_flux_fused)
    reference = _copy(driver.fields)
    before = _copy(driver.fields)
    with pytest.raises(Exception) as expected:
        _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, k)
    with pytest.raises(type(expected.value)) as actual:
        _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k)
    assert str(actual.value) == str(expected.value)
    _equal(driver.fields, before)
    _equal(reference, before)


@pytest.mark.gpu
@requires_gpu
def test_old_surface_leaf_differs_on_derived_nan():
    import cupy as cp
    from gpuwm.core import ruc, ruc_gpu
    maximum = cp.finfo(cp.float32).max
    args = (cp.ones(1, dtype=cp.int32), cp.ones(1, dtype=cp.int32),
            cp.full(1, -maximum, dtype=cp.float32),
            cp.full(1, maximum, dtype=cp.float32),
            cp.full(1, maximum, dtype=cp.float32),
            cp.full(1, 0.1, dtype=cp.float32), cp.full(1, 2.0, dtype=cp.float32))
    driver = ruc.ruc_surface_parameters(*args, arrays=ruc_gpu.RUC_DEVICE_ARRAYS)
    leaf = ruc_gpu.ruc_surface_parameters_cuda(*args)
    assert bool(cp.isnan(driver.lai).all())
    assert bool(cp.isfinite(leaf.lai).all())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
@pytest.mark.parametrize("width", [1, 17])
@pytest.mark.parametrize("surface", ["water", "lake"])
def test_no_sfctmp_columns(width, nzs, surface):
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    bench, driver, atmosphere, cold = _case(width, nzs, "warm")
    driver.fields["xice"].fill(0)
    driver.fields["xland"].fill(2 if surface == "water" else 1)
    driver.fields["lakemask"].fill(1 if surface == "lake" else 0)
    reference = _copy(driver.fields)
    for k in range(1, 8):
        bench.forcing(driver, k, 11, (1, width), cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, (1, width), cold)
        assert _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k) == (
            _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, k))
        _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
def test_the_forecast_call_is_six_kernels_and_no_orchestration_reads(nzs, monkeypatch):
    """The speed property itself, held so it cannot quietly regress.

    The array orchestration this path replaced launched about 1,800 kernels
    per call on a production-shaped grid and made 17 to 41 blocking reads
    (admission batch flushes, dispatch-arm index conversions and leaf
    checks).  The fused call is six kernels and reads the host only for the
    SFCDIAGS power pair's input and the one flag slab at the end.
    """
    import cupy as cp
    from gpuwm.core import ruc, ruc_fused, ruc_tier, ruc_validation
    from gpuwm.core.ruc_runtime import ruc_lsm_step
    width = 5917
    bench, driver, atmosphere, cold = _case(width, nzs, "mixed")
    bench.forcing(driver, 1, 11, (1, width), cold)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 1)
    launched = []
    original = ruc_tier.ruc_fused_kernel

    def counting(func, levels):
        kernel = original(func, levels)

        def launch(*args):
            launched.append(func)
            return kernel(*args)
        return launch

    def forbidden(*args, **kwargs):
        raise AssertionError("the fused RUC call took an orchestration read")

    monkeypatch.setattr(ruc_tier, "ruc_fused_kernel", counting)
    monkeypatch.setattr(ruc_fused, "ruc_fused_kernel", counting)
    monkeypatch.setattr(ruc_validation, "_host_list", forbidden)
    monkeypatch.setattr(ruc, "_selected", forbidden)
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    bench.forcing(driver, 2, 11, (1, width), cold)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 2)
    assert launched == ["ruc_driver_prologue", "ruc_sfctmp_stage0",
                        "ruc_sfctmp_stage1", "ruc_sfctmp_stage2",
                        "ruc_driver_epilogue", "ruc_driver_commit"], launched


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
def test_t2_and_th2_do_not_reach_the_hosts_numpy_power(nzs, monkeypatch):
    # A145.  The call's two host powers (SFCDIAGS_RUCLSM's exner factors)
    # were NumPy's float32 power, the host's own function, so an AVX-512
    # product box wrote other T2 and TH2 words.  Here every np.power answer
    # moves one ULP, as another host's would, and no field may move.
    from gpuwm.core.ruc_runtime import ruc_lsm_step
    bench, driver, atmosphere, cold = _case(5917, nzs, "mixed")
    moved = _copy(driver.fields)
    real = np.power

    def another_hosts_power(*args, **kwargs):
        answer = np.asarray(real(*args, **kwargs))
        return np.nextafter(answer, np.asarray(np.inf, dtype=answer.dtype))

    for k in range(1, 4):
        bench.forcing(driver, k, 11, (1, 5917), cold)
        bench.forcing(SimpleNamespace(fields=moved), k, 11, (1, 5917), cold)
        _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k)
        with monkeypatch.context() as patch:
            patch.setattr(np, "power", another_hosts_power)
            assert np.power(np.float32(1.5), np.float32(0.3)) != real(
                np.float32(1.5), np.float32(0.3))
            _call(ruc_lsm_step, moved, atmosphere, driver.ruc_params, k)
        _equal(driver.fields, moved)
