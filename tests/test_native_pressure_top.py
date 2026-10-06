"""Native mass-level coverage and the serialized upper pressure endpoint.

The native analysis at 2026-10-02 21Z publishes its highest mass pressure
as 1731.475406848..1731.475537920 Pa.  Its decimal eta ladder and 1500 Pa
interface give 1731.475 Pa.  The old operator refused this same-grid start.
The bounded endpoint correction is a declared divergence from WRF real's
strict above-source test.  Its evaluated value is WRF's own endpoint value;
no additional atmospheric layer or value extrapolation is introduced.

The gap is the top half-layer's water vapour: the native level is a full
pressure, the target its dry pressure, and integ_moist leaves the top level
undried.  2025-03-14 06Z decodes to 1731.475830078125 Pa (FP32), 4.9e-7
relative above the target, past the first four-epsilon (2^-21) bound, so
that start was refused.  The bound is now 2^-16 (``WRF_TOP_ENDPOINT_RTOL``).
"""

from types import SimpleNamespace
from pathlib import Path
import copy
import json

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.grid import make_vertical_coord
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
from gpuwm.ingest.preprocess_backend import (
    VERTICAL_ENDPOINT_RELATIVE_TOLERANCE, ParallelCpuPreprocessBackend,
    resolve_preprocess_backend)
from gpuwm.ingest.real import _integrate_moisture, finalize_vertical_coord
from gpuwm.ingest.vert import WRF_TOP_ENDPOINT_RTOL
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


@pytest.mark.parametrize("pressure", [1731.40, 1731.475537920 - 50., 1500., 1000.])
def test_a_target_physically_above_the_mass_column_still_fails(pressure):
    source, values, target, surface_p, surface_value = _column()
    target[-1] = pressure
    with pytest.raises(ValueError, match="above the source top") as refused:
        _cpu().wrf_vertical_interpolate(
            values, surface_value, source, surface_p, target)
    # The refusal names what it prevents: values invented above the data.
    assert "extrapolated past the top of the source analysis" in str(refused.value)


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
    target[-1] = 1731.40
    with pytest.raises(ValueError, match="above source top"):
        backend.prepare_wrf_vertical(source, surface_p, target)


#: A 50-level operational native ladder above a 1500 Pa lid (hybrid_opt 2,
#: etac 0.2).  Its top mass level is purely isobaric: c3h = 0, so the dry
#: target is 1500 + 0.00235 * 98500 = 1731.475 Pa in every column.
_NATIVE_ETA = (
    1.0, 0.998, 0.994, 0.987, 0.975, 0.959, 0.939, 0.916, 0.892, 0.865,
    0.835, 0.802, 0.766, 0.727, 0.685, 0.64, 0.592, 0.542, 0.497, 0.4565,
    0.4205, 0.3877, 0.3582, 0.3317, 0.3078, 0.2863, 0.267, 0.2496, 0.2329,
    0.2188, 0.2047, 0.1906, 0.1765, 0.1624, 0.1483, 0.1342, 0.1201, 0.106,
    0.0919, 0.0778, 0.0657, 0.0568, 0.0486, 0.0409, 0.0337, 0.0271, 0.0209,
    0.0151, 0.0097, 0.0047, 0.0)
_LID = 1500.0
_DRY_TOP = 1731.475
_DECODED_2025_TOP = 1731.475830078125  # FP32 maximum, 2025-03-14 06Z


def _cpu_plan_backend():
    _cpu()
    return ParallelCpuPreprocessBackend(workers=2)


def _moist_native_columns(top_pressures):
    """Native columns whose top level is a full pressure, one per column.

    The columns also differ in surface pressure, so the dry-pressure
    adjustment (dry mass = psfc - intq - p_top) differs between them.
    """
    lower = np.asarray([97000., 85000., 70000., 50000., 30000., 20000.,
                        10000., 5000., 2209.2018])
    temperature = np.asarray(
        [285., 278., 268., 250., 228., 218., 212., 215., 220., 225.])
    qv_profile = np.asarray(
        [8e-3, 5e-3, 3e-3, 1e-3, 1e-4, 2e-5, 5e-6, 4e-6, 3.5e-6, 3.3e-6])
    ncol = len(top_pressures)
    pressure = np.empty((lower.size + 1, 1, ncol))
    pressure[:-1] = lower[:, None, None]
    pressure[-1, 0, :] = top_pressures
    t = np.broadcast_to(temperature[:, None, None], pressure.shape).copy()
    q = np.broadcast_to(qv_profile[:, None, None], pressure.shape).copy()
    psfc = np.asarray([[100000., 86000., 101500.][i % 3] for i in range(ncol)])[None]
    tsfc = np.full((1, ncol), 288.)
    qsfc = np.full((1, ncol), 9e-3)
    zsfc = np.zeros((1, ncol))
    # Hypsometric heights above a sea-level reference; strictly increasing.
    height = np.empty(pressure.shape)
    height[0] = 287.0 * 0.5 * (288. + t[0]) / 9.81 * np.log(101325. / pressure[0])
    for k in range(1, pressure.shape[0]):
        height[k] = height[k - 1] + 287.0 * 0.5 * (t[k - 1] + t[k]) / 9.81 * np.log(
            pressure[k - 1] / pressure[k])
    source_pd, intq, order = _integrate_moisture(
        q, pressure, t, height, psfc, tsfc, qsfc, zsfc)
    assert list(order) == list(range(pressure.shape[0]))
    coord = make_vertical_coord(
        len(_NATIVE_ETA) - 1, hybrid_opt=2, etac=0.2,
        eta_levels=np.asarray(_NATIVE_ETA))
    finalize_vertical_coord(coord, _LID)
    dry_mass = psfc - intq - _LID
    dry_target = (coord.c3h[:, None, None] * dry_mass[None]
                  + coord.c4h[:, None, None] + _LID)
    values = ((300. - 2. * np.arange(pressure.shape[0]))[:, None, None]
              * np.ones((1, 1, ncol)))
    return dict(full=pressure, dry=source_pd, surface_full=psfc,
                surface_dry=psfc - intq, target=dry_target, values=values,
                surface_value=tsfc, coord=coord)


def test_the_colocation_bound_is_one_power_of_two_in_every_operator():
    assert WRF_TOP_ENDPOINT_RTOL == VERTICAL_ENDPOINT_RELATIVE_TOLERANCE == 2.0 ** -16
    root = Path(__file__).resolve().parents[1]
    rust = (root / "tools/grib1_bridge/src/lib.rs").read_text()
    cuda = (root / "gpuwm/core/kernels/vert_interp.cu").read_text()
    assert "TOP_COLOCATION_RTOL: f32 = 1.52587890625e-5;" in rust
    assert "__fmul_rn(1.52587890625e-5f, fabsf(top_pressure))" in cuda
    assert float(np.float32(1.52587890625e-5)) == 2.0 ** -16


@pytest.mark.parametrize("qv_top", [0.0, 1.0e-6, 3.34e-6, 1.0e-5])
@pytest.mark.parametrize("dried", [False, True],
                         ids=["full-source", "dry-adjusted-source"])
def test_a_moist_native_top_meets_its_dry_target_before_and_after_dry_adjustment(
        qv_top, dried):
    tops = [_DRY_TOP + (_DRY_TOP - _LID) * qv_top / (1.0 - qv_top),
            _DECODED_2025_TOP, 1731.4755859375]
    case = _moist_native_columns(tops)
    target = case["target"]
    # The target top is the dry, purely isobaric level in every column,
    # whatever each column's dry mass is.
    assert case["coord"].c3h[-1] == 0.0
    np.testing.assert_array_equal(np.float32(target[-1]), np.float32(_DRY_TOP))
    # integ_moist leaves the top level undried: pd(top) == p(top).
    np.testing.assert_array_equal(case["dry"][-1], case["full"][-1])
    source = case["dry"] if dried else case["full"]
    surface = case["surface_dry"] if dried else case["surface_full"]
    source32, target32 = np.float32(source), np.float32(target)
    gap = (source32[-1] - target32[-1]) / source32[-1]
    # The decoded 2025 column is past the old four-epsilon bound, which
    # refused it, and inside the new one.
    assert gap.max() > 4.0 * np.finfo(np.float32).eps
    assert gap.max() <= WRF_TOP_ENDPOINT_RTOL
    expected_top = np.float32(case["values"][-1])
    plan = _cpu_plan_backend().prepare_wrf_vertical(
        source32, np.float32(surface), target32)
    for logp in (False, True):
        planned = plan.apply(np.float32(case["values"]),
                             np.float32(case["surface_value"]),
                             interp_in_logp=logp, extrap="temperature")
        direct = _cpu().wrf_vertical_interpolate(
            case["values"], case["surface_value"], source32, surface, target32,
            interp_in_logp=logp, extrap="temperature", workers=1)
        np.testing.assert_array_equal(planned, direct)
        assert np.isfinite(direct).all()
        # Co-located with the source endpoint: the top value is the top
        # source value, not an extrapolation beyond it.
        np.testing.assert_allclose(direct[-1], expected_top, rtol=2e-6, atol=0)


def test_a_target_top_below_the_source_top_interpolates_inside_the_column():
    # Source top at exactly 1731.475 Pa and a target top at 1731.5 Pa: the
    # target lies below the source top and is ordinary interpolation.
    source, values, target, surface_p, surface_value = _column()
    source[-1] = 1731.475
    target[-1] = 1731.5
    for logp in (False, True):
        got = _cpu().wrf_vertical_interpolate(
            values, surface_value, source, surface_p, target,
            interp_in_logp=logp, workers=1)
        assert np.isfinite(got).all()


@pytest.mark.parametrize("dried", [False, True],
                         ids=["full-source", "dry-adjusted-source"])
def test_a_target_50_pa_above_the_native_top_is_refused_with_its_breakage(dried):
    case = _moist_native_columns([_DECODED_2025_TOP] * 3)
    target = case["target"].copy()
    target[-1] = _DECODED_2025_TOP - 50.0
    source = case["dry"] if dried else case["full"]
    surface = case["surface_dry"] if dried else case["surface_full"]
    plan = _cpu_plan_backend().prepare_wrf_vertical(
        np.float32(source), np.float32(surface), np.float32(target))
    with pytest.raises(ValueError, match="above the source top") as refused:
        plan.apply(np.float32(case["values"]), np.float32(case["surface_value"]),
                   interp_in_logp=True, extrap="temperature")
    assert "extrapolated past the top of the source analysis" in str(refused.value)
    with pytest.raises(ValueError, match="above the source top"):
        _cpu().wrf_vertical_interpolate(
            case["values"], case["surface_value"], source, surface, target,
            interp_in_logp=False, extrap="constant", workers=1)


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
