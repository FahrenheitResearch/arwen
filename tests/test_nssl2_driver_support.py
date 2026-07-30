"""GPU gates for the unified pre-GS NSSL option-18 driver pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).parents[1]
_ORACLE = (_ROOT / "gpuwm" / "data" / "nssl2" / "driver-oracle" /
           "driver-support.csv")
_VOLUME_NAMES = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qndrop_per_kg", "qnr_per_kg", "qni_per_kg", "qns_per_kg",
    "qng_per_kg", "qnh_per_kg", "qnn_per_kg",
    "qvolg_m3_per_kg", "qvolh_m3_per_kg",
)
_SURFACE_NAMES = (
    "rainnc", "rainncv", "snownc", "snowncv",
    "graupelnc", "graupelncv", "hailnc", "hailncv", "sr",
)


def _volume(rows: list[dict[str, str]], name: str) -> np.ndarray:
    nz = max(int(row["k"]) for row in rows)
    nx = max(int(row["column"]) for row in rows)
    result = np.empty((nz, 1, nx), dtype=np.float32)
    for row in rows:
        result[int(row["k"]) - 1, 0, int(row["column"]) - 1] = float(
            row[name])
    return result


def _surface(rows: list[dict[str, str]], name: str) -> np.ndarray:
    nx = max(int(row["column"]) for row in rows)
    result = np.empty((1, nx), dtype=np.float32)
    for row in rows:
        if int(row["k"]) == 1:
            result[0, int(row["column"]) - 1] = float(row[name])
    return result


def test_driver_support_matches_compiled_official_wrf_pipeline():
    import cupy as cp

    from gpuwm.core.nssl2_driver_support import (
        gather_initialize_and_sediment,
        reduce_nssl2_precipitation,
        scatter_nssl2_driver_workspace,
    )

    with _ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 3 * 6 * 14

    saw_first_step = False
    saw_initialized = False
    saw_uninitialized = False
    saw_all_kf_rates = False
    saw_cfl_above_one = False
    saw_variable_density = False
    saw_precipitation = False

    for run_id in (1, 2, 3):
        before = [row for row in rows
                  if int(row["run"]) == run_id and row["phase"] == "before"]
        after = [row for row in rows
                 if int(row["run"]) == run_id and row["phase"] == "after"]
        assert len(before) == len(after) == 6 * 14

        density_host = _volume(before, "rho_kg_m3")
        depth_host = _volume(before, "dz_m")
        density = cp.asarray(density_host)
        depth = cp.asarray(depth_host)
        temperature = cp.asarray(_volume(before, "temperature_k"))
        fields = [cp.asarray(_volume(before, name)) for name in _VOLUME_NAMES]
        registry_before = [field.copy() for field in fields]
        surfaces = [cp.asarray(_surface(before, name))
                    for name in _SURFACE_NAMES]
        rates = [cp.asarray(_volume(before, name)) for name in (
            "qrcuten", "qscuten", "qicuten", "qccuten")]
        step = float(before[0]["dt_s"])
        first_step = bool(int(before[0]["first_step"]))
        cu_used = bool(int(before[0]["cu_used"]))

        workspace = gather_initialize_and_sediment(
            density, depth, *fields, step,
            temperature_k=temperature,
            first_step=first_step, cu_used=cu_used,
            qrcuten=rates[0], qscuten=rates[1],
            qicuten=rates[2], qccuten=rates[3])
        cp.cuda.Stream.null.synchronize()

        # The production seam is durable: no Registry field is changed before
        # future GS/NUCOND/diagnostic phases and the one explicit final scatter.
        for actual, expected in zip(fields, registry_before):
            cp.testing.assert_array_equal(actual, expected)
        for index, name in enumerate(_VOLUME_NAMES[7:], start=7):
            expected_internal = _volume(after, name) * density_host
            internal_rtol = 1.0e-3 if name == "qng_per_kg" else 5.0e-5
            np.testing.assert_allclose(
                cp.asnumpy(workspace.state[index]), expected_internal,
                rtol=internal_rtol,
                atol=5.0 if name == "qng_per_kg"
                else (1.0 if index < 14 else 4.0e-15),
                err_msg=f"run={run_id} field={name}")

        reduce_nssl2_precipitation(workspace, *surfaces)
        scatter_nssl2_driver_workspace(workspace, density, *fields)
        cp.cuda.Stream.null.synchronize()

        # Fixed from the already admitted per-category official-WRF gates.
        for index, (actual, name) in enumerate(zip(fields, _VOLUME_NAMES)):
            if index < 7:
                rtol, atol = 5.0e-5, 4.0e-12
            elif index < 14:
                rtol = 1.0e-3 if name == "qng_per_kg" else 4.0e-5
                atol = 5.0 if name == "qng_per_kg" else 1.0
            else:
                rtol, atol = 5.0e-5, 4.0e-15
            np.testing.assert_allclose(
                cp.asnumpy(actual), _volume(after, name),
                rtol=rtol, atol=atol)

        for actual, name in zip(surfaces, _SURFACE_NAMES):
            if name == "sr":
                rtol, atol = 3.0e-5, 3.0e-7
            elif name.endswith("ncv"):
                rtol, atol = 2.0e-5, 4.0e-10
            else:
                rtol, atol = 5.0e-7, 5.0e-7
            np.testing.assert_allclose(
                cp.asnumpy(actual), _surface(after, name),
                rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            cp.asnumpy(workspace.ice_surface_export),
            _surface(after, "ice_surface_export"),
            rtol=2.0e-5, atol=4.0e-10)

        # Independent column conservation in concentration-space fallout.
        water_before = sum(_volume(before, name)
                           for name in _VOLUME_NAMES[:7])
        water_after = sum(cp.asnumpy(field) for field in fields[:7])
        atmospheric_before = np.sum(
            water_before * density_host * depth_host, axis=0)
        atmospheric_after = np.sum(
            water_after * density_host * depth_host, axis=0)
        surface_export = (cp.asnumpy(surfaces[1])
                          + cp.asnumpy(workspace.ice_surface_export)
                          + cp.asnumpy(workspace.cloud_surface_export))
        np.testing.assert_allclose(
            atmospheric_after + surface_export, atmospheric_before,
            rtol=4.0e-6, atol=4.0e-7)

        total = cp.asnumpy(surfaces[1])
        frozen = sum(cp.asnumpy(surfaces[index]) for index in (3, 5, 7))
        np.testing.assert_allclose(
            cp.asnumpy(surfaces[8]), frozen / (total + 1.0e-12),
            rtol=2.0e-6, atol=3.0e-7)

        saw_first_step |= first_step
        saw_initialized |= bool(np.any(_volume(before, "qnr_per_kg") > 0.0))
        saw_uninitialized |= bool(np.any(
            (_volume(before, "qr") > 1.0e-8)
            & (_volume(before, "qnr_per_kg") == 0.0)))
        kf_active = np.stack([cp.asnumpy(value) for value in rates]) > 0.0
        saw_all_kf_rates |= cu_used and bool(np.all(np.any(
            kf_active, axis=(1, 2, 3))))
        saw_cfl_above_one |= bool(step / np.min(depth_host) > 1.0)
        graupel_density = np.divide(
            _volume(before, "qg"), _volume(before, "qvolg_m3_per_kg"),
            out=np.zeros_like(depth_host),
            where=_volume(before, "qvolg_m3_per_kg") > 0.0)
        saw_variable_density |= bool(
            np.ptp(graupel_density[graupel_density > 0.0]) > 100.0)
        saw_precipitation |= bool(np.any(cp.asnumpy(surfaces[1]) > 0.0))

        # The truly empty first column is an exact no-op apart from ncv/SR reset.
        for field, name in zip(fields[:7], _VOLUME_NAMES[:7]):
            cp.testing.assert_array_equal(
                field[:, :, 0], cp.asarray(_volume(after, name)[:, :, 0]))

    assert saw_first_step
    assert saw_initialized
    assert saw_uninitialized
    assert saw_all_kf_rates
    assert saw_cfl_above_one
    assert saw_variable_density
    assert saw_precipitation


def test_driver_support_validation_and_empty_noop():
    import cupy as cp

    from gpuwm.core.nssl2_driver_support import launch_nssl2_driver_support

    shape = (4, 2, 3)
    density = cp.ones(shape, dtype=cp.float32)
    depth = cp.full(shape, cp.float32(100.0))
    temperature = cp.full(shape, cp.float32(260.0))
    fields = [cp.zeros(shape, dtype=cp.float32) for _ in range(16)]
    surfaces = [cp.zeros((2, 3), dtype=cp.float32) for _ in range(9)]
    fields[0].fill(cp.float32(0.01))
    before = [field.copy() for field in fields]

    result = launch_nssl2_driver_support(
        density, depth, *fields, *surfaces, 1.0,
        temperature_k=temperature)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)
    for surface in surfaces:
        cp.testing.assert_array_equal(surface, 0.0)
    cp.testing.assert_array_equal(result.ice_surface_export, 0.0)
    cp.testing.assert_array_equal(result.cloud_surface_export, 0.0)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_nssl2_driver_support(
                density, depth, *fields, *surfaces, step,
                temperature_k=temperature)
    with pytest.raises(TypeError, match="first_step must be bool"):
        launch_nssl2_driver_support(
            density, depth, *fields, *surfaces, 1.0,
            temperature_k=temperature, first_step=1)
    with pytest.raises(TypeError, match="cu_used must be bool"):
        launch_nssl2_driver_support(
            density, depth, *fields, *surfaces, 1.0,
            temperature_k=temperature, cu_used=1)

    bad_fields = list(fields)
    bad_fields[-1] = cp.zeros((24,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_nssl2_driver_support(
            density, depth, *bad_fields, *surfaces, 1.0,
            temperature_k=temperature)

    bad_surfaces = list(surfaces)
    bad_surfaces[-1] = cp.zeros((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_nssl2_driver_support(
            density, depth, *fields, *bad_surfaces, 1.0,
            temperature_k=temperature)

    too_shallow = [cp.zeros((1, 2, 3), dtype=cp.float32)
                   for _ in range(18)]
    too_shallow[0].fill(1.0)
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        launch_nssl2_driver_support(
            *too_shallow, *surfaces, 1.0,
            temperature_k=too_shallow[0])


def test_driver_source_keeps_internal_moments_in_concentration_space():
    source = (_ROOT / "gpuwm" / "core" / "kernels" /
              "nssl2_driver_support.cu").read_text(encoding="utf-8")

    assert "NSSL2_DRIVER_FIELD_COUNT = 16" in source
    assert "float graupel_volume = qvolg[idx] * rho" in source
    assert "float hail_volume = qvolh[idx] * rho" in source
    assert "number[k] = qnr[idx];" in source
    assert "number[k] = qns[idx];" in source
    assert "number[k] = qni[idx];" in source
    assert "number[k] = qnx[idx];" in source
    assert "volume[k] = qvolx[idx];" in source
    assert "qnr[idx] = number[k] / rho[k]" not in source
    assert "qvolx[idx] = volume[k] / rho[k]" not in source
    assert (
        "qnr[idx] = state[(size_t)NSSL2_NR * n + idx] / air_density[idx]"
        in source
    )
