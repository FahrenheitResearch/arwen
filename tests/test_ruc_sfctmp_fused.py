"""The full-width fused sfctmp against the retained device oracle.

``ruc_sfctmp_full_width_fused`` (three kernels, one thread per column) is
graded word for word against ``ruc_sfctmp_full_width_reference`` (the
resident array orchestration gathered to the run columns and scattered back)
on inputs captured from the driver on the fixture grid, over successive
calls with state carried forward, both soil geometries and widths from one
column to 100,000; refusals must name the reference's first failure.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core import ruc  # noqa: E402
from gpuwm.core.ruc_gpu import (  # noqa: E402
    RUC_SFCTMP_FLAGS_SIZE, ruc_sfctmp_full_width_fused,
    ruc_sfctmp_full_width_reference, ruc_sfctmp_raise_from_flags,
)


def _bench():
    path = Path(__file__).with_name('ruc_fused_fixture.py')
    spec = importlib.util.spec_from_file_location('sfctmp_bench', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(monkeypatch, scenario, nzs):
    from gpuwm.core.ruc_runtime import _ruc_lsm_step_reference
    bench = _bench()
    recorded = []
    original = ruc.ruc_surface_temperature_step

    def capture(values, **kw):
        keywords = {name: kw[name] for name in (
            'delt', 'conflx', 'ivgtyp', 'iland', 'nroot', 'ilnb', 'isice',
            'c1sn', 'c2sn', 'isncovr_opt', 'mminlu', 'parameters')}
        recorded.append(({name: value.copy() for name, value in values.items()}, keywords))
        return original(values, **kw)

    with monkeypatch.context() as scoped:
        scoped.setattr(ruc, 'ruc_surface_temperature_step', capture)
        _, cfg, driver, atmosphere, cold = bench.build(97, 61, 20, nzs, scenario, 11)
        bench.forcing(driver, 1, 11, (61, 97), cold)
        bench.call(driver, atmosphere, 1, _ruc_lsm_step_reference)
    return recorded[0]


def _resize(values, keywords, width):
    count = values['snhei'].size
    index = cp.arange(width, dtype=cp.int64) % count
    result = {name: cp.ascontiguousarray(value[..., index]) for name, value in values.items()}
    kw = dict(keywords)
    for name in ('conflx', 'ivgtyp', 'iland', 'nroot', 'ilnb'):
        kw[name] = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(kw[name]), (count,))[index])
    return result, kw


def _assert_words(left, right, run):
    assert set(left) == set(right)
    for name in left:
        a = cp.asnumpy(left[name][..., run]).view(np.uint32)
        b = cp.asnumpy(right[name][..., run]).view(np.uint32)
        assert np.array_equal(a, b), name


@pytest.mark.parametrize('nzs', [9, 6])
@pytest.mark.parametrize('scenario', ['mixed', 'warm'])
@pytest.mark.parametrize('width', [1, 37, 5917, 99991, 100000])
def test_successive_full_width_words(monkeypatch, nzs, scenario, width):
    values, keywords = _resize(*_capture(monkeypatch, scenario, nzs), width)
    run = cp.arange(width) % 5 != 1
    flags = cp.empty(RUC_SFCTMP_FLAGS_SIZE, dtype=cp.uint64)
    for step in range(6):
        reference = ruc_sfctmp_full_width_reference(values, run=run, **keywords)
        result = ruc_sfctmp_full_width_fused(values, run=run, flags=flags, **keywords)
        ruc_sfctmp_raise_from_flags(flags)
        _assert_words(reference, result, run)
        for name in values:
            if name in result:
                values[name][..., run] = result[name][..., run]
        keywords['iland'][run] = result['iland'][run]
        keywords['ilnb'][run] = result['ilnb'][run]


def test_negative_control_and_refusal(monkeypatch):
    values, keywords = _resize(*_capture(monkeypatch, 'warm', 9), 37)
    run = cp.ones(37, dtype=cp.bool_)
    baseline = ruc_sfctmp_full_width_reference(values, run=run, **keywords)
    changed = {name: value.copy() for name, value in values.items()}
    changed['tabs'].view(cp.uint32)[0] ^= cp.uint32(1)
    result = ruc_sfctmp_full_width_fused(changed, run=run, **keywords)
    with pytest.raises(AssertionError):
        _assert_words(baseline, result, run)
    changed['dqm'][0] = cp.nan
    with pytest.raises(ValueError) as reference_error:
        ruc_sfctmp_full_width_reference(changed, run=run, **keywords)
    flags = cp.empty(RUC_SFCTMP_FLAGS_SIZE, dtype=cp.uint64)
    ruc_sfctmp_full_width_fused(changed, run=run, flags=flags, **keywords)
    with pytest.raises(type(reference_error.value)) as fused_error:
        ruc_sfctmp_raise_from_flags(flags)
    assert str(fused_error.value) == str(reference_error.value)


@pytest.mark.parametrize('nzs', [9, 6])
def test_driver_carries_fused_state(monkeypatch, nzs):
    from gpuwm.core.ruc_runtime import _ruc_lsm_step_reference
    bench = _bench()
    original = ruc.ruc_surface_temperature_step

    def fused(values, **kw):
        reference = original(values, **kw)
        keys = {name: kw[name] for name in (
            'delt', 'conflx', 'ivgtyp', 'iland', 'nroot', 'ilnb', 'isice',
            'c1sn', 'c2sn', 'isncovr_opt', 'mminlu', 'parameters')}
        # The fused stages compile the SOILPROP lineage the driver was
        # handed (the runtime default, wrf_45), as the reference runs it.
        keys['soilprop'] = kw['soilprop']
        run = cp.ones(values['snhei'].size, dtype=cp.bool_)
        result = ruc_sfctmp_full_width_fused(values, run=run, **keys)
        _assert_words({name: getattr(reference, name) for name in result}, result, run)
        return ruc.RucSurfaceTemperatureStep(**result)

    monkeypatch.setattr(ruc, 'ruc_surface_temperature_step', fused)
    _, cfg, driver, atmosphere, cold = bench.build(97, 61, 20, nzs, 'mixed', 11)
    for step in range(1, 13):
        bench.forcing(driver, step, 11, (61, 97), cold)
        bench.call(driver, atmosphere, step, _ruc_lsm_step_reference)


@pytest.mark.parametrize('failure', [
    'dqm_zero', 'psis_zero', 'bclh_zero', 'sat_zero', 'rho_zero',
    'mavail_negative', 'ksat_negative', 'ref_equal_wilt', 'nroot_zero',
    'nroot_high', 'ivgtyp_low', 'iland_high', 'isice_high', 'density_nan',
    'nroot_float', 'ivgtyp_float', 'iland_float', 'ilnb_float',
    'early_nan_late_type', 'early_flux_late_delt', 'early_profile_late_shape',
    'leaf_nan_rnet',
])
def test_ordered_refusals_leave_inputs_unchanged(monkeypatch, failure):
    values, keywords = _resize(*_capture(monkeypatch, 'warm', 9), 37)
    run = cp.ones(37, dtype=cp.bool_)
    if failure.endswith('_zero') and failure[:-5] in values:
        values[failure[:-5]][0] = 0
    elif failure == 'mavail_negative':
        values['mavail'][0] = -1
    elif failure == 'ksat_negative':
        values['ksat'][0] = -1
    elif failure == 'ref_equal_wilt':
        values['ref'][0] = values['wilt'][0]
    elif failure == 'nroot_zero':
        keywords['nroot'][0] = 0
    elif failure == 'nroot_high':
        keywords['nroot'][0] = 9
    elif failure == 'ivgtyp_low':
        keywords['ivgtyp'][0] = 0
    elif failure == 'iland_high':
        keywords['iland'][0] = 999
    elif failure == 'isice_high':
        keywords['isice'] = 999
    elif failure == 'density_nan':
        keywords['c1sn'] = float('nan')
    elif failure.endswith('_float'):
        name = failure[:-6]
        keywords[name] = keywords[name].astype(cp.float32)
    elif failure == 'early_nan_late_type':
        values['qvatm'][0] = cp.nan
        keywords['nroot'] = keywords['nroot'].astype(cp.float32)
    elif failure == 'early_flux_late_delt':
        keywords['conflx'][0] = cp.nan
        keywords['delt'] = -1
    elif failure == 'early_profile_late_shape':
        values['soilm1d'][0, 0] = cp.nan
        values['ts1d'] = cp.empty((9, 36), dtype=cp.float32)
    elif failure == 'leaf_nan_rnet':
        values['soilt'][0] = cp.float32(1.0e38)
        values['emiss'][0] = cp.float32(0)
    else:
        raise AssertionError(failure)
    before = {name: cp.asnumpy(array).view(np.uint32).copy() for name, array in values.items()}
    with pytest.raises((ValueError, TypeError)) as reference_error:
        ruc_sfctmp_full_width_reference(values, run=run, **keywords)
    if failure == 'leaf_nan_rnet':
        assert str(reference_error.value) == 'rnet must be finite'
    flags = cp.empty(RUC_SFCTMP_FLAGS_SIZE, dtype=cp.uint64)
    with monkeypatch.context() as scoped:
        scoped.setattr(cp, 'asnumpy', lambda *a, **k: (_ for _ in ()).throw(AssertionError('host read inside deferred function')))
        ruc_sfctmp_full_width_fused(values, run=run, flags=flags, **keywords)
    with pytest.raises(type(reference_error.value)) as fused_error:
        ruc_sfctmp_raise_from_flags(flags)
    assert str(fused_error.value) == str(reference_error.value)
    for name, array in values.items():
        assert np.array_equal(cp.asnumpy(array).view(np.uint32), before[name]), name
