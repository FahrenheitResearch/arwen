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
def test_resident_snow_guard_parses_default_only_once(monkeypatch):
    from gpuwm.core import ruc_gpu as gpu

    gpu._default_parameter_bundle.cache_clear()
    original = gpu.load_ruc_parameters
    calls = []

    def load():
        calls.append(None)
        return original()

    sentinel = object()
    monkeypatch.setattr(gpu, "load_ruc_parameters", load)
    monkeypatch.setattr(gpu, "ruc_snow_preparation_cuda",
                        lambda *args, **kwargs: sentinel)
    bundle = original()
    call = gpu._resident_snow_prep()
    try:
        for _ in range(3):
            assert call({}, delt=12.0, ivgtyp=None, iland=None,
                        bundle=bundle) is sentinel
        assert len(calls) == 1
        changed = _changed_bundle(bundle, z0=bundle.vegetation_for(
            "MODIFIED_IGBP_MODIS_NOAH").rows[0].z0 + 0.01)
        with pytest.raises(ValueError) as error:
            call({}, delt=12.0, ivgtyp=None, iland=None, bundle=changed)
        assert str(error.value) == (
            "the RUC CUDA snow-preparation stage indexes device "
            "tables built from the default parameter bundle; the "
            "supplied bundle differs in z0tbl, lemitbl or URBAN")
        assert len(calls) == 1
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

    monkeypatch.setattr(gpu.cp.cuda.runtime, "getDevice", lambda: device[0])
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
