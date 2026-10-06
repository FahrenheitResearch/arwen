"""Cache identity checks for the resident RUC reference launchers."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from conftest import requires_gpu


def _changed_bundle(bundle, **row_changes):
    vegetation = bundle.vegetation_for("MODIFIED_IGBP_MODIS_NOAH")
    rows = (replace(vegetation.rows[0], **row_changes), *vegetation.rows[1:])
    tables = dict(bundle.vegetation)
    tables[vegetation.name] = replace(vegetation, rows=rows)
    return replace(bundle, vegetation=MappingProxyType(tables))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("resident", [False, True])
def test_snow_stage_forwards_default_and_changed_bundle(monkeypatch, resident):
    from gpuwm.core import ruc_gpu as gpu
    from types import SimpleNamespace

    calls = []
    result = SimpleNamespace(**{
        name: object() for name in gpu.RucSnowPreparation.__dataclass_fields__})

    def launch(*args, **kwargs):
        calls.append(kwargs["parameters"])
        return result

    monkeypatch.setattr(gpu, "ruc_snow_preparation_cuda", launch)
    monkeypatch.setattr(gpu.cp, "asnumpy", lambda value: value)
    bundle = gpu.load_ruc_parameters()
    changed = _changed_bundle(bundle, z0=bundle.vegetation_for(
        "MODIFIED_IGBP_MODIS_NOAH").rows[0].z0 + 0.01)
    call = (gpu._resident_snow_prep() if resident
            else gpu._host_facing_snow_prep())
    for candidate in (None, bundle, changed):
        actual = call({}, delt=12.0, ivgtyp=None, iland=None, bundle=candidate)
        if resident:
            assert actual is result
        else:
            for name in gpu.RucSnowPreparation.__dataclass_fields__:
                assert getattr(actual, name) is getattr(result, name)
    assert calls == [None, bundle, changed]


@pytest.mark.gpu
@requires_gpu
def test_snow_bundle_tables_share_default_and_isolate_consumed_columns():
    from gpuwm.core import ruc_gpu as gpu

    gpu._default_parameter_bundle.cache_clear()
    bundle = gpu.load_ruc_parameters()
    dataset = "MODIFIED_IGBP_MODIS_NOAH"
    table = bundle.vegetation_for(dataset)
    try:
        default = gpu._snow_preparation_tables_for(None, dataset)
        assert gpu._snow_preparation_tables_for(bundle, dataset) is default
        assert gpu._snow_preparation_tables_for(
            gpu.load_ruc_parameters(), dataset) is default
        assert gpu._default_parameter_bundle.cache_info().misses == 1
        for column, position, value in (("z0", 0, 0.123), ("lemi", 1, 0.75)):
            changed = _changed_bundle(bundle, **{column: value})
            selected = gpu._snow_preparation_tables_for(changed, dataset)
            assert selected is not default
            assert gpu._snow_preparation_tables_for(changed, dataset) is selected
            assert gpu._snow_preparation_tables_for(
                _changed_bundle(bundle, **{column: value}), dataset) is selected
            assert float(selected[position][0]) == np.float32(value)
            other = 1 - position
            np.testing.assert_array_equal(gpu.cp.asnumpy(selected[other]),
                                          gpu.cp.asnumpy(default[other]))
            assert selected[2:] == default[2:]
        scalars = dict(table.scalars, URBAN=10)
        tables = dict(bundle.vegetation)
        tables[table.name] = replace(table, scalars=MappingProxyType(scalars))
        changed = replace(bundle, vegetation=MappingProxyType(tables))
        selected = gpu._snow_preparation_tables_for(changed, dataset)
        assert selected is not default
        assert selected[2] == 10
        assert selected[3] == default[3]
        for position in (0, 1):
            np.testing.assert_array_equal(gpu.cp.asnumpy(selected[position]),
                                          gpu.cp.asnumpy(default[position]))
        assert gpu._snow_preparation_tables_for(bundle, dataset) is default
    finally:
        gpu._default_parameter_bundle.cache_clear()


@pytest.mark.gpu
@requires_gpu
def test_bundle_tables_key_consumed_values_and_device(monkeypatch):
    from gpuwm.core import ruc_gpu as gpu

    bundle = gpu.load_ruc_parameters()
    device = [0]
    uploads = []

    def upload(bundle, dataset):
        result = object()
        uploads.append(result)
        return result

    from types import SimpleNamespace
    monkeypatch.setattr(gpu.cp.cuda, "Device",
                        lambda: SimpleNamespace(id=device[0]))
    monkeypatch.setattr(gpu, "_upload_tables", upload)
    monkeypatch.setattr(gpu, "_BUNDLE_DEVICE_TABLES", {})
    dataset = "MODIFIED_IGBP_MODIS_NOAH"
    first = gpu._bundle_device_tables(bundle, dataset)
    assert gpu._bundle_device_tables(bundle, dataset) is first
    assert gpu._bundle_device_tables(gpu.load_ruc_parameters(), dataset) is first
    changed = _changed_bundle(bundle, z0=0.123)
    assert gpu._bundle_device_tables(changed, dataset) is not first
    device[0] = 1
    assert gpu._bundle_device_tables(bundle, dataset) is not first
    device[0] = 0
    assert gpu._bundle_device_tables(bundle, "USGS") is not first
    assert len(uploads) == 4


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
def test_cached_soil_geometry_is_bitwise_identical(nzs):
    import cupy as cp
    from gpuwm.core import ruc_gpu as gpu

    device = int(cp.cuda.runtime.getDevice())
    zs, _ = gpu.ruc_soil_geometry(nzs)
    actual = gpu._device_soil_half_levels(device, nzs)
    assert gpu._device_soil_half_levels(device, nzs) is actual
    np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),
                                  gpu.ruc_zshalf(zs).view(np.uint32))


@pytest.mark.gpu
@requires_gpu
def test_cached_tables_match_uncached_upload_and_changed_bundle():
    import cupy as cp
    from gpuwm.core import ruc_gpu as gpu

    dataset = "MODIFIED_IGBP_MODIS_NOAH"
    bundle = gpu.load_ruc_parameters()
    for candidate in (bundle, _changed_bundle(bundle, z0=0.123)):
        cached = gpu._bundle_device_tables(candidate, dataset)
        plain = gpu._upload_tables(candidate, dataset)
        assert cached[1:] == plain[1:]
        for name in gpu._RucDeviceTables.__dataclass_fields__:
            left, right = getattr(cached[0], name), getattr(plain[0], name)
            if isinstance(left, cp.ndarray):
                np.testing.assert_array_equal(cp.asnumpy(left).view(np.uint32),
                                              cp.asnumpy(right).view(np.uint32))
            else:
                assert left == right
