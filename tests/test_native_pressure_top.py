"""Native mass-level coverage and the serialized upper pressure endpoint.

The native analysis at 2026-10-02 21Z publishes its highest mass pressure
as 1731.475406848..1731.475537920 Pa.  Its decimal eta ladder and 1500 Pa
interface give 1731.475 Pa.  The old operator refused this same-grid start.
The bounded endpoint correction is a declared divergence from WRF real's
strict above-source test.  Its evaluated value is WRF's own endpoint value;
no additional atmospheric layer or value extrapolation is introduced.
"""

from types import SimpleNamespace
from pathlib import Path
import copy
import json

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
from gpuwm.mapped_direct import _source_top_pressure_pa
from gpuwm.mapped_source import load_mapping
from gpuwm.verify.npref import np_wrf_real_vert_interp


def _column():
    source = np.asarray(
        [98000., 70000., 30000., 5000., 2209.201782784, 1731.475537920],
        dtype=np.float32)[:, None, None]
    values = np.asarray([289., 270., 238., 216., 218., 219.],
                        dtype=np.float32)[:, None, None]
    target = np.asarray([95000., 60000., 20000., 4000., 1731.475],
                        dtype=np.float32)[:, None, None]
    surface_p = np.asarray([[100000.]], dtype=np.float32)
    surface_value = np.asarray([[291.]], dtype=np.float32)
    return source, values, target, surface_p, surface_value


def _cpu():
    try:
        return CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")


@pytest.mark.parametrize("logp", [False, True])
@pytest.mark.parametrize("extrap", ["constant", "temperature"])
def test_serialized_native_top_is_the_wrf_endpoint_without_extrapolation(logp, extrap):
    source, values, target, surface_p, surface_value = _column()
    assert target[-1, 0, 0] < source[-1, 0, 0]
    snapped = target.copy()
    snapped[-1] = source[-1]
    # The unmodified WRF reference proves the original failure and supplies
    # the authority after this declared coordinate-only correction.
    with pytest.raises(ValueError, match="above source top"):
        np_wrf_real_vert_interp(values, surface_value, source, surface_p, target,
                                interp_in_logp=logp, extrap=extrap)
    authority = np_wrf_real_vert_interp(
        values, surface_value, source, surface_p, snapped,
        interp_in_logp=logp, extrap=extrap)
    cpu = _cpu()
    got = cpu.wrf_vertical_interpolate(
        values, surface_value, source, surface_p, target,
        interp_in_logp=logp, extrap=extrap, workers=1)
    endpoint = cpu.wrf_vertical_interpolate(
        values, surface_value, source, surface_p, snapped,
        interp_in_logp=logp, extrap=extrap, workers=4)
    np.testing.assert_array_equal(got, endpoint)
    np.testing.assert_allclose(got, authority, rtol=3.e-6, atol=3.e-5)
    assert got[-1, 0, 0] == values[-1, 0, 0]


@pytest.mark.parametrize("pressure", [1731.465, 1500., 1000.])
def test_a_target_physically_above_the_mass_column_still_fails(pressure):
    source, values, target, surface_p, surface_value = _column()
    target[-1] = pressure
    with pytest.raises(ValueError, match="above the source top"):
        _cpu().wrf_vertical_interpolate(
            values, surface_value, source, surface_p, target)


@requires_gpu
@pytest.mark.parametrize("logp", [False, True])
def test_prepared_cuda_and_cpu_use_identical_endpoint_words(logp):
    import cupy as cp
    source, values, target, surface_p, surface_value = _column()
    expected = _cpu().wrf_vertical_interpolate(
        values, surface_value, source, surface_p, target, interp_in_logp=logp)
    backend = resolve_preprocess_backend("cuda")
    plan = backend.prepare_wrf_vertical(source, surface_p, target)
    got = plan.apply(values, surface_value, interp_in_logp=logp)
    np.testing.assert_array_equal(cp.asnumpy(got), expected)
    target[-1] = 1731.465
    with pytest.raises(ValueError, match="above source top"):
        backend.prepare_wrf_vertical(source, surface_p, target)


def test_native_mapping_distinguishes_model_interface_from_highest_mass():
    snapshots = [SimpleNamespace(levels_hpa=np.asarray([17.314755, 22.092]))]
    assert _source_top_pressure_pa(snapshots) == pytest.approx(1731.4755)
    contract = {"coordinates": {"vertical": {
        "kind": "model_level", "model_top_pressure_pa": 1500.}}}
    assert _source_top_pressure_pa(snapshots, mapping_contract=contract) == 1500.
    contract["coordinates"]["vertical"]["model_top_pressure_pa"] = 5000.
    with pytest.raises(ValueError, match="above its decoded mass levels"):
        _source_top_pressure_pa(snapshots, mapping_contract=contract)


def test_native_model_top_metadata_is_validated_before_decode():
    path = Path(__file__).resolve().parents[1] / "gpuwm" / "authorities" / "rw-wps-hrrr-native-grib2.mapping.json"
    mapping = json.loads(path.read_text())
    assert load_mapping(path)["coordinates"]["vertical"]["model_top_pressure_pa"] == 1500.
    for kind, top in (("pressure", 1500.), ("model_level", 0.), ("model_level", -1.)):
        changed = copy.deepcopy(mapping)
        changed["coordinates"]["vertical"].update(kind=kind, model_top_pressure_pa=top)
        with pytest.raises(ValueError, match="model_top_pressure_pa"):
            load_mapping(path, _raw=changed)
