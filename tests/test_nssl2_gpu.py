"""GPU gates for independently admitted WRF NSSL option-18 slices."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

_ORACLE = (Path(__file__).parents[1] / "gpuwm" / "data" / "nssl2" /
           "oracle" / "effective-radius.csv")


def test_effective_radius_matches_compiled_official_wrf_routine():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_effective_radius

    with _ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    def device(name):
        return cp.asarray(np.asarray(
            [float(row[name]) for row in rows], dtype=np.float32))

    density = device("rho_kg_m3")
    qc = device("qc")
    qndrop = device("nc_per_kg")
    qi = device("qi")
    qni = device("ni_per_kg")
    qs = device("qs")
    qns = device("ns_per_kg")
    outputs = [cp.empty_like(density) for _ in range(3)]

    launch_effective_radius(
        density, qc, qndrop, qi, qni, qs, qns, *outputs)
    cp.cuda.Stream.null.synchronize()

    # GNU Fortran's real exponentiation and CUDA libdevice powf are distinct
    # correctly-rounded implementations.  This bound is measured across all
    # active, background, lower-clamped, and upper-clamped fixture cases.
    for actual, name in zip(
            outputs, ("re_cloud_m", "re_ice_m", "re_snow_m")):
        expected = np.asarray(
            [float(row[name]) for row in rows], dtype=np.float32)
        np.testing.assert_allclose(
            cp.asnumpy(actual), expected, rtol=2.0e-6, atol=2.0e-12)


def test_effective_radius_background_and_launcher_validation():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_effective_radius

    fields = [cp.zeros((3, 2), dtype=cp.float32) for _ in range(10)]
    fields[0].fill(1.0)
    launch_effective_radius(*fields)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(fields[7], cp.float32(2.51e-6))
    cp.testing.assert_array_equal(fields[8], cp.float32(10.01e-6))
    cp.testing.assert_array_equal(fields[9], cp.float32(25.0e-6))

    bad_shape = list(fields)
    bad_shape[-1] = cp.zeros((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_effective_radius(*bad_shape)

    bad_dtype = list(fields)
    bad_dtype[2] = cp.zeros((3, 2), dtype=cp.float64)
    with pytest.raises(TypeError, match="float32"):
        launch_effective_radius(*bad_dtype)


def test_mass_only_initial_state_matches_compiled_official_wrf_routine():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_initial_state

    fixture = _ORACLE.with_name("initial-state.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = [row for row in rows if row["phase"] == "before"]
    after = [row for row in rows if row["phase"] == "after"]
    assert len(before) == len(after) == 48

    names = (
        "rho_kg_m3", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
        "nc_per_kg", "nr_per_kg", "ni_per_kg", "ns_per_kg",
        "ng_per_kg", "nh_per_kg", "qnn_per_kg",
        "qvolg_m3_per_kg", "qvolh_m3_per_kg",
    )

    def host(source, name):
        return np.asarray(
            [float(row[name]) for row in source], dtype=np.float32)

    fields = [cp.asarray(host(before, name)) for name in names]
    launch_initial_state(*fields)
    cp.cuda.Stream.null.synchronize()

    for actual, name in zip(fields[1:], names[1:]):
        expected = host(after, name)
        np.testing.assert_allclose(
            cp.asnumpy(actual), expected, rtol=2.0e-6, atol=2.0e-12)

    before_water = sum(host(before, name) for name in names[1:8])
    after_water = sum(cp.asnumpy(field) for field in fields[1:8])
    np.testing.assert_allclose(
        after_water, before_water, rtol=0.0, atol=2.0e-9)


def test_initial_state_launcher_rejects_mismatched_fields():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_initial_state

    fields = [cp.zeros((2, 3), dtype=cp.float32) for _ in range(17)]
    fields[0].fill(1.0)
    bad_shape = list(fields)
    bad_shape[-1] = cp.zeros((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_initial_state(*bad_shape)

    bad_dtype = list(fields)
    bad_dtype[10] = cp.zeros((2, 3), dtype=cp.float64)
    with pytest.raises(TypeError, match="float32"):
        launch_initial_state(*bad_dtype)


def test_rain_self_collection_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_self_collection

    fixture = _ORACLE.with_name("self-collection.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    # WRF's process routine accepts one scalar dt for a slab.  Group the
    # oracle cells by their exact FP32 step and launch each production-shaped
    # batch once.
    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        density = device("rho_kg_m3")
        rain = device("qr_before")
        number = device("nr_before_per_kg")
        rain_before = rain.copy()
        launch_rain_self_collection(density, rain, number, float(step))
        cp.cuda.Stream.null.synchronize()

        cp.testing.assert_array_equal(rain, rain_before)
        expected = np.asarray(
            [float(row["nr_after_per_kg"]) for row in batch],
            dtype=np.float32)
        np.testing.assert_allclose(
            cp.asnumpy(number), expected, rtol=3.0e-6, atol=2.0e-5)


def test_rain_self_collection_launcher_validation():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_self_collection

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(3)]
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_rain_self_collection(*fields, step)

    bad_shape = list(fields)
    bad_shape[2] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_rain_self_collection(*bad_shape, 1.0)


def test_snow_aggregation_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_snow_aggregation

    fixture = _ORACLE.with_name("snow-aggregation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_active_sink = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        density = device("rho_kg_m3")
        temperature = device("temperature_k")
        snow = device("qs_before")
        number = device("qns_before_per_kg")
        snow_before = snow.copy()
        number_before = cp.asnumpy(number).copy()

        launch_snow_aggregation(
            density, temperature, snow, number, float(step))
        cp.cuda.Stream.null.synchronize()

        cp.testing.assert_array_equal(snow, snow_before)
        expected = np.asarray(
            [float(row["qns_after_per_kg"]) for row in batch],
            dtype=np.float32)
        np.testing.assert_allclose(
            cp.asnumpy(number), expected, rtol=5.0e-6, atol=3.0e-4)
        active = np.asarray(
            [float(row["qs_before"]) > 1.0e-13
             and float(row["temperature_k"]) < 273.15
             and float(row["temperature_k"]) > 258.15
             for row in batch])
        if np.any(active):
            saw_active_sink |= bool(np.any(
                cp.asnumpy(number)[active] < number_before[active]))

    assert saw_active_sink


def test_snow_aggregation_validation_and_native_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_snow_aggregation

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(4)]
    fields[0].fill(cp.float32(1.0))
    fields[1].fill(cp.float32(268.0))
    fields[2].fill(cp.float32(2.5e-4))
    mean_volume = cp.float32(0.523599 * (1.0e-3)**3)
    fields[3].fill(cp.float32(2.5e-4) /
                   (cp.float32(100.0) * mean_volume))
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_snow_aggregation(*fields, step)

    bad_shape = list(fields)
    bad_shape[1] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_snow_aggregation(*bad_shape, 1.0)

    # WRF's efficiency is strictly zero at the melting point.  Use an
    # in-bounds distribution so no independent moment bound is triggered.
    fields[1].fill(cp.float32(273.15))
    before = fields[3].copy()
    launch_snow_aggregation(*fields, 10.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(fields[3], before)

    # The driver excludes mass at the strict 1e-13 kg/kg gate from its
    # gathered process slab, preserving the Registry number bitwise.
    fields[2].fill(cp.float32(1.0e-13))
    fields[3].fill(cp.float32(1234.0))
    before = fields[3].copy()
    launch_snow_aggregation(*fields, 10.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(fields[3], before)


def test_ice_deposition_conversion_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_ice_deposition_conversion

    fixture = _ORACLE.with_name("ice-deposition-conversion.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_deposition = False
    saw_conversion = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        vapor = device("qv_before")
        ice = device("qi_before")
        ice_number = device("qni_before_per_kg")
        snow = device("qs_before")
        snow_number = device("qns_before_per_kg")
        water_before = vapor + ice + snow

        launch_ice_deposition_conversion(
            theta, density, pressure, exner, vapor, ice, ice_number,
            snow, snow_number, float(step))
        cp.cuda.Stream.null.synchronize()

        expected_fields = (
            (theta, "theta_after_k", 4.0e-6, 4.0e-5),
            (vapor, "qv_after", 6.0e-6, 3.0e-10),
            (ice, "qi_after", 8.0e-6, 3.0e-10),
            (ice_number, "qni_after_per_kg", 1.0e-5, 2.0e-2),
            (snow, "qs_after", 8.0e-6, 3.0e-10),
            (snow_number, "qns_after_per_kg", 1.0e-5, 2.0e-2),
        )
        for actual, name, rtol, atol in expected_fields:
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        cp.testing.assert_allclose(
            vapor + ice + snow, water_before, rtol=2.0e-6, atol=3.0e-10)
        active = np.asarray(
            [float(row["qi_before"]) > 1.0e-13 for row in batch])
        if np.any(active):
            saw_deposition |= bool(np.any(
                cp.asnumpy(vapor)[active]
                < np.asarray([float(row["qv_before"])
                              for row in batch], dtype=np.float32)[active]))
            saw_conversion |= bool(np.any(cp.asnumpy(snow)[active] > 0.0))

    assert saw_deposition
    assert saw_conversion


def test_ice_deposition_conversion_validation_and_threshold_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_ice_deposition_conversion

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(9)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(80000.0))
    fields[3].fill(cp.float32(1.0))
    table_index = int((250.0 - 163.15) / 0.002 + 1.5)
    table_temperature = np.float32(163.15 + (table_index - 1) * 0.002)
    saturation = np.float32(380.0 / 80000.0) * np.exp(
        np.float32(21.87455)
        * (table_temperature - np.float32(273.15))
        / (table_temperature - np.float32(7.66)), dtype=np.float32)
    fields[4].fill(cp.float32(1.05) * saturation)
    fields[5].fill(cp.float32(2.5e-5))
    ice_mass = np.float32((80.0e-6 / 0.1871) ** (1.0 / 0.3429))
    fields[6].fill(cp.float32(2.5e-5) / ice_mass)
    fields[7].fill(cp.float32(0.0))
    fields[8].fill(cp.float32(0.0))

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_ice_deposition_conversion(*fields, step)

    bad_shape = list(fields)
    bad_shape[2] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_ice_deposition_conversion(*bad_shape, 1.0)

    # Below 100 microns deposition grows cloud ice without making snow.
    launch_ice_deposition_conversion(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    assert bool(cp.all(fields[5] > cp.float32(2.5e-5)))
    cp.testing.assert_array_equal(fields[7], cp.zeros_like(fields[7]))
    cp.testing.assert_array_equal(fields[8], cp.zeros_like(fields[8]))

    # At the strict ice-mass gate the native gathered slab clears number and
    # leaves mass/vapor/temperature otherwise unchanged.
    fields[5].fill(cp.float32(1.0e-13))
    fields[6].fill(cp.float32(1234.0))
    held = [field.copy() for field in (fields[0], fields[4], fields[5])]
    launch_ice_deposition_conversion(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip((fields[0], fields[4], fields[5]), held):
        cp.testing.assert_array_equal(actual, expected)
    cp.testing.assert_array_equal(fields[6], cp.zeros_like(fields[6]))


def test_frozen_vapor_exchange_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_frozen_vapor_exchange

    fixture = _ORACLE.with_name("frozen-vapor-exchange.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_deposition = False
    saw_sublimation = False
    saw_snow_exchange = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        vapor = device("qv_before")
        ice = device("qi_before")
        ice_number = device("qni_before_per_kg")
        snow = device("qs_before")
        snow_number = device("qns_before_per_kg")
        vapor_before = vapor.copy()
        snow_before = snow.copy()
        water_before = vapor + ice + snow

        launch_frozen_vapor_exchange(
            theta, density, pressure, exner, vapor, ice, ice_number,
            snow, snow_number, float(step))
        cp.cuda.Stream.null.synchronize()

        expected_fields = (
            (theta, "theta_after_k", 6.0e-6, 6.0e-5),
            (vapor, "qv_after", 2.0e-5, 6.0e-10),
            (ice, "qi_after", 3.0e-5, 6.0e-10),
            (ice_number, "qni_after_per_kg", 4.0e-5, 5.0e-2),
            (snow, "qs_after", 3.0e-5, 6.0e-10),
            (snow_number, "qns_after_per_kg", 4.0e-5, 5.0e-2),
        )
        for actual, name, rtol, atol in expected_fields:
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        cp.testing.assert_allclose(
            vapor + ice + snow, water_before, rtol=3.0e-6, atol=6.0e-10)
        saw_deposition |= bool(cp.any(vapor < vapor_before))
        saw_sublimation |= bool(cp.any(vapor > vapor_before))
        saw_snow_exchange |= bool(cp.any(snow != snow_before))

    assert saw_deposition
    assert saw_sublimation
    assert saw_snow_exchange


def test_frozen_vapor_exchange_validation():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_frozen_vapor_exchange

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(9)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(80000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(1.0e-3))
    fields[5].fill(cp.float32(1.0e-5))
    fields[6].fill(cp.float32(1.0e6))
    fields[7].fill(cp.float32(1.0e-5))
    fields[8].fill(cp.float32(1.0e5))

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_frozen_vapor_exchange(*fields, step)

    bad_shape = list(fields)
    bad_shape[2] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_frozen_vapor_exchange(*bad_shape, 1.0)


def test_graupel_hail_vapor_exchange_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_graupel_hail_vapor_exchange

    fixture = _ORACLE.with_name("graupel-hail-vapor-exchange.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_deposition = False
    saw_sublimation = False
    saw_graupel = False
    saw_hail = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        vapor = device("qv_before")
        graupel = device("qg_before")
        graupel_number = device("qng_before_per_kg")
        graupel_volume = device("qvolg_before_m3_per_kg")
        hail = device("qh_before")
        hail_number = device("qnh_before_per_kg")
        hail_volume = device("qvolh_before_m3_per_kg")
        vapor_before = vapor.copy()
        graupel_before = graupel.copy()
        hail_before = hail.copy()
        water_before = vapor + graupel + hail

        launch_graupel_hail_vapor_exchange(
            theta, density, pressure, exner, vapor,
            graupel, graupel_number, graupel_volume,
            hail, hail_number, hail_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 8.0e-6, 8.0e-5),
                (vapor, "qv_after", 4.0e-5, 8.0e-10),
                (graupel, "qg_after", 5.0e-5, 8.0e-10),
                (graupel_number, "qng_after_per_kg", 8.0e-5, 8.0e-3),
                (graupel_volume, "qvolg_after_m3_per_kg",
                 8.0e-5, 8.0e-14),
                (hail, "qh_after", 5.0e-5, 8.0e-10),
                (hail_number, "qnh_after_per_kg", 8.0e-5, 8.0e-3),
                (hail_volume, "qvolh_after_m3_per_kg",
                 8.0e-5, 8.0e-14)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        cp.testing.assert_allclose(
            vapor + graupel + hail, water_before,
            rtol=4.0e-6, atol=8.0e-10)
        saw_deposition |= bool(cp.any(vapor < vapor_before))
        saw_sublimation |= bool(cp.any(vapor > vapor_before))
        saw_graupel |= bool(cp.any(graupel != graupel_before))
        saw_hail |= bool(cp.any(hail != hail_before))

    assert saw_deposition
    assert saw_sublimation
    assert saw_graupel
    assert saw_hail


def test_graupel_hail_vapor_exchange_validation():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_graupel_hail_vapor_exchange

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(11)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(80000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(1.0e-3))
    fields[5].fill(cp.float32(1.0e-5))
    fields[6].fill(cp.float32(1.0e2))
    fields[7].fill(cp.float32(1.0e-5) / cp.float32(500.0))
    fields[8].fill(cp.float32(1.0e-5))
    fields[9].fill(cp.float32(1.0e2))
    fields[10].fill(cp.float32(1.0e-5) / cp.float32(900.0))

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_graupel_hail_vapor_exchange(*fields, step)

    bad_shape = list(fields)
    bad_shape[2] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_graupel_hail_vapor_exchange(*bad_shape, 1.0)


def test_bigg_rain_freezing_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_bigg_rain_freezing

    fixture = _ORACLE.with_name("bigg-rain-freezing.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_noop = False
    saw_partial_freezing = False
    saw_near_complete_freezing = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        exner = device("exner")
        temperature = device("temperature_k")
        rain = device("qr_before")
        rain_number = device("qnr_before_per_kg")
        graupel = device("qg_before")
        graupel_number = device("qng_before_per_kg")
        graupel_volume = device("qvolg_before_m3_per_kg")
        rain_before = rain.copy()
        graupel_before = graupel.copy()
        water_before = rain + graupel

        launch_bigg_rain_freezing(
            theta, density, exner, temperature, rain, rain_number,
            graupel, graupel_number, graupel_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 5.0e-6, 5.0e-5),
                (rain, "qr_after", 8.0e-6, 4.0e-10),
                (rain_number, "qnr_after_per_kg", 1.0e-5, 8.0e-5),
                (graupel, "qg_after", 8.0e-6, 4.0e-10),
                (graupel_number, "qng_after_per_kg", 1.0e-5, 8.0e-3),
                (graupel_volume, "qvolg_after_m3_per_kg",
                 1.0e-5, 8.0e-14)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        # The separate FP32 sink/source updates may round their reconstructed
        # sum by one representable value at the largest combined-water case.
        # Keep this invariant tied to arithmetic precision rather than to an
        # absolute threshold that changes meaning as fixture mass expands.
        cp.testing.assert_array_max_ulp(
            rain + graupel, water_before, maxulp=1)
        frozen_fraction = cp.where(
            rain_before > 0.0,
            (graupel - graupel_before) / rain_before,
            0.0)
        saw_noop |= bool(cp.any(frozen_fraction == 0.0))
        saw_partial_freezing |= bool(cp.any(
            (frozen_fraction > 0.01) & (frozen_fraction < 0.90)))
        saw_near_complete_freezing |= bool(cp.any(frozen_fraction > 0.98))

    assert saw_noop
    assert saw_partial_freezing
    assert saw_near_complete_freezing


def test_bigg_rain_freezing_validation():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_bigg_rain_freezing

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(9)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(1.0))
    fields[3].fill(cp.float32(250.0))
    fields[4].fill(cp.float32(1.0e-3))
    fields[5].fill(cp.float32(1.0e2))
    fields[6].fill(0.0)
    fields[7].fill(0.0)
    fields[8].fill(0.0)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_bigg_rain_freezing(*fields, step)

    bad_shape = list(fields)
    bad_shape[2] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_bigg_rain_freezing(*bad_shape, 1.0)


def test_warm_autoconversion_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_warm_autoconversion

    fixture = _ORACLE.with_name("warm-autoconversion.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        density = device("rho_kg_m3")
        temperature = cp.full_like(density, cp.float32(300.0))
        cloud = device("qc_before")
        rain = device("qr_before")
        cloud_number = device("nc_before_per_kg")
        rain_number = device("nr_before_per_kg")
        water_before = cloud + rain

        launch_warm_autoconversion(
            density, temperature, cloud, rain, cloud_number, rain_number,
            float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (cloud, "qc_after", 3.0e-6, 2.0e-11),
                (rain, "qr_after", 3.0e-6, 2.0e-13),
                (cloud_number, "nc_after_per_kg", 4.0e-6, 2.0e-2),
                (rain_number, "nr_after_per_kg", 4.0e-6, 2.0e-5)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        cp.testing.assert_allclose(
            cloud + rain, water_before, rtol=0.0, atol=2.0e-10)


def test_warm_autoconversion_launcher_validation_and_cold_gate():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_warm_autoconversion

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(6)]
    fields[2].fill(cp.float32(1.0e-4))
    fields[3].fill(0.0)
    fields[4].fill(cp.float32(1.0e7))
    fields[5].fill(0.0)
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_warm_autoconversion(*fields, step)

    bad_shape = list(fields)
    bad_shape[1] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_warm_autoconversion(*bad_shape, 1.0)

    # Below tfrh+4, the mass-transfer branch is closed.  Use an in-bounds
    # cloud distribution so the native final moment limiter is also inert.
    fields[1].fill(cp.float32(230.0))
    before = [value.copy() for value in fields[2:]]
    launch_warm_autoconversion(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(fields[2], before[0])
    cp.testing.assert_array_equal(fields[3], before[1])
    cp.testing.assert_array_equal(fields[4], before[2])
    cp.testing.assert_array_equal(fields[5], before[3])


def test_rain_cloud_accretion_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_cloud_accretion

    fixture = _ORACLE.with_name("rain-cloud-accretion.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        density = device("rho_kg_m3")
        cloud = device("qc_before")
        rain = device("qr_before")
        cloud_number = device("nc_before_per_kg")
        rain_number = device("nr_before_per_kg")
        water_before = cloud + rain

        launch_rain_cloud_accretion(
            density, cloud, rain, cloud_number, rain_number, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (cloud, "qc_after_accretion", 4.0e-6, 3.0e-11),
                (rain, "qr_after_accretion", 4.0e-6, 3.0e-11),
                (cloud_number, "nc_after_accretion_per_kg", 5.0e-6, 3.0e-2),
                (rain_number, "nr_after_accretion_per_kg", 5.0e-6, 3.0e-3)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)
        cp.testing.assert_allclose(
            cloud + rain, water_before, rtol=0.0, atol=3.0e-10)


def test_rain_cloud_accretion_launcher_validation():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_cloud_accretion

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(5)]
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_rain_cloud_accretion(*fields, step)

    bad_shape = list(fields)
    bad_shape[3] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_rain_cloud_accretion(*bad_shape, 1.0)


def test_clear_air_activation_matches_official_wrf_nucond():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_clear_air_activation

    fixture = _ORACLE.with_name("clear-air-activation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        velocity = device("vertical_velocity_m_s")
        vapor = device("qv_before")
        cloud = device("qc_before")
        cloud_number = device("nc_before_per_kg")
        ccn = device("qnn_before_per_kg")
        water_before = vapor + cloud

        launch_clear_air_activation(
            theta, density, pressure, exner, velocity,
            vapor, cloud, cloud_number, ccn, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 5.0e-6, 5.0e-5),
                (vapor, "qv_after", 8.0e-6, 3.0e-10),
                (cloud, "qc_after", 2.5e-3, 1.0e-9),
                (cloud_number, "nc_after_per_kg", 3.0e-4, 2.6e4),
                (ccn, "qnn_after_per_kg", 3.0e-5, 64.0)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        # Compare nonzero activation tendencies independently of the much
        # larger base state, and enforce mass conservation in the CUDA path.
        for actual, before_name, after_name, rtol, atol in (
                (theta, "theta_before_k", "theta_after_k", 4.0e-3, 3.5e-5),
                (vapor, "qv_before", "qv_after", 3.0e-3, 3.0e-9),
                (cloud, "qc_before", "qc_after", 3.0e-3, 3.0e-9),
                (cloud_number, "nc_before_per_kg", "nc_after_per_kg",
                 3.0e-4, 2.6e4)):
            before = np.asarray(
                [float(row[before_name]) for row in batch], dtype=np.float32)
            expected_after = np.asarray(
                [float(row[after_name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual) - before, expected_after - before,
                rtol=rtol, atol=atol)
        cp.testing.assert_allclose(
            vapor + cloud, water_before, rtol=0.0, atol=3.0e-9)


def test_clear_air_activation_launcher_validation_and_native_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_clear_air_activation

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(9)]
    fields[0].fill(cp.float32(300.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(100000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(1.0))
    fields[5].fill(cp.float32(0.001))
    fields[6].fill(0.0)
    fields[7].fill(0.0)
    fields[8].fill(cp.float32(408163264.0))
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_clear_air_activation(*fields, step)

    bad_shape = list(fields)
    bad_shape[4] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_clear_air_activation(*bad_shape, 1.0)

    # Subsaturation and existing cloud each close this isolated branch.
    before = [value.copy() for value in fields]
    launch_clear_air_activation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)

    fields[5].fill(cp.float32(0.1))
    fields[6].fill(cp.float32(1.0e-4))
    before = [value.copy() for value in fields]
    launch_clear_air_activation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_cloudy_water_adjustment_matches_official_wrf_nucond():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_cloudy_water_adjustment

    fixture = _ORACLE.with_name("cloudy-water-adjustment.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        vapor = device("qv_before")
        cloud = device("qc_before")
        cloud_number = device("nc_before_per_kg")
        ccn = device("qnn_before_per_kg")
        water_before = vapor + cloud

        launch_cloudy_water_adjustment(
            theta, density, pressure, exner,
            vapor, cloud, cloud_number, ccn, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 5.0e-6, 5.0e-5),
                (vapor, "qv_after", 1.0e-5, 5.0e-10),
                (cloud, "qc_after", 1.0e-5, 5.0e-10),
                (cloud_number, "nc_after_per_kg", 3.0e-5, 64.0),
                (ccn, "qnn_after_per_kg", 3.0e-5, 64.0)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        # Total-state agreement can hide compensating transfer errors.  Check
        # the independently dynamic mass and heat increments as well.
        for actual, before_name, after_name, rtol, atol in (
                (theta, "theta_before_k", "theta_after_k", 3.0e-3, 4.0e-5),
                (vapor, "qv_before", "qv_after", 3.0e-3, 3.0e-9),
                (cloud, "qc_before", "qc_after", 3.0e-3, 3.0e-9),
                (cloud_number, "nc_before_per_kg", "nc_after_per_kg",
                 3.0e-4, 2048.0)):
            before = np.asarray(
                [float(row[before_name]) for row in batch], dtype=np.float32)
            expected_after = np.asarray(
                [float(row[after_name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual) - before, expected_after - before,
                rtol=rtol, atol=atol)
        cp.testing.assert_allclose(
            vapor + cloud, water_before, rtol=0.0, atol=3.0e-9)


def test_cloudy_water_adjustment_launcher_and_fail_closed_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_cloudy_water_adjustment

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(8)]
    fields[0].fill(cp.float32(300.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(100000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(0.001))
    fields[5].fill(0.0)
    fields[6].fill(cp.float32(2.0e8))
    fields[7].fill(cp.float32(408163264.0))
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_cloudy_water_adjustment(*fields, step)

    bad_shape = list(fields)
    bad_shape[2] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_cloudy_water_adjustment(*bad_shape, 1.0)

    # Clear air belongs to the adjacent activation slice.
    before = [value.copy() for value in fields]
    launch_cloudy_water_adjustment(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)

    # Existing cloud above the 0.5-percent native interior-renucleation gate
    # remains bitwise untouched until that coupled number process is admitted.
    fields[4].fill(cp.float32(0.1))
    fields[5].fill(cp.float32(1.0e-4))
    before = [value.copy() for value in fields]
    launch_cloudy_water_adjustment(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_cloud_interior_renucleation_matches_official_wrf_nucond():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_cloud_interior_renucleation

    fixture = _ORACLE.with_name("cloud-interior-renucleation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_renucleation = False
    for step, batch in by_step.items():
        def device(name):
            center = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            host = np.repeat(center[None, :, None], 3, axis=0)
            return cp.asarray(host)

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        velocity = device("w_m_s")
        vapor = device("qv_before")
        cloud = device("qc_before")
        cloud_number = device("nc_before_per_kg")
        ccn = device("qnn_before_per_kg")
        water_before = vapor[1, :, 0] + cloud[1, :, 0]

        launch_cloud_interior_renucleation(
            theta, density, pressure, exner, velocity,
            vapor, cloud, cloud_number, ccn, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta[1, :, 0], "theta_after_k", 7.0e-6, 8.0e-5),
                (vapor[1, :, 0], "qv_after", 2.0e-5, 8.0e-10),
                (cloud[1, :, 0], "qc_after", 2.0e-5, 8.0e-10),
                (cloud_number[1, :, 0], "nc_after_per_kg",
                 5.0e-5, 2048.0),
                (ccn[1, :, 0], "qnn_after_per_kg", 5.0e-5, 2048.0)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        expected_ccn_before = np.asarray(
            [float(row["qnn_before_per_kg"]) for row in batch],
            dtype=np.float32)
        expected_ccn_after = np.asarray(
            [float(row["qnn_after_per_kg"]) for row in batch],
            dtype=np.float32)
        actual_ccn_delta = cp.asnumpy(ccn[1, :, 0]) - expected_ccn_before
        np.testing.assert_allclose(
            actual_ccn_delta, expected_ccn_after - expected_ccn_before,
            rtol=8.0e-4, atol=2048.0)
        saw_renucleation |= bool(np.any(
            expected_ccn_after < expected_ccn_before))
        cp.testing.assert_allclose(
            vapor[1, :, 0] + cloud[1, :, 0], water_before,
            rtol=0.0, atol=5.0e-9)

    assert saw_renucleation


def test_cloud_interior_renucleation_validation_and_native_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_cloud_interior_renucleation

    shape = (3, 2, 2)
    fields = [cp.ones(shape, dtype=cp.float32) for _ in range(9)]
    fields[0].fill(cp.float32(280.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(100000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(5.0))
    fields[5].fill(cp.float32(0.001))
    fields[6].fill(cp.float32(2.0e-4))
    fields[7].fill(cp.float32(1.0e8))
    fields[8].fill(cp.float32(3.0e8))

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_cloud_interior_renucleation(*fields, step)

    two_dimensional = [value[0].copy() for value in fields]
    with pytest.raises(ValueError, match="must be 3-D"):
        launch_cloud_interior_renucleation(*two_dimensional, 1.0)
    shallow = [value[:2].copy() for value in fields]
    with pytest.raises(ValueError, match="nz >= 3"):
        launch_cloud_interior_renucleation(*shallow, 1.0)

    # Weak/undersaturation belongs to the adjacent water-adjustment slice.
    before = [value.copy() for value in fields]
    launch_cloud_interior_renucleation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)

    # The separate QVEXCESS maximum-supersaturation branch stays fail-closed.
    fields[5].fill(cp.float32(0.1))
    before = [value.copy() for value in fields]
    launch_cloud_interior_renucleation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)

    # At admitted supersaturation, WRF treats the bottom updraft as an inflow
    # boundary: condensation proceeds there, but CCN renucleation starts only
    # above it.
    temperature = np.float32(280.0)
    table_index = int((float(temperature) - 163.15) / 0.002 + 1.5)
    table_temperature = np.float32(163.15 + (table_index - 1) * 0.002)
    saturation = np.float32(380.0 / 100000.0) * np.exp(
        np.float32(17.2693882)
        * (table_temperature - np.float32(273.15))
        / (table_temperature - np.float32(35.86)), dtype=np.float32)
    fields[5].fill(np.float32(1.02) * saturation)
    fields[6].fill(cp.float32(2.0e-4))
    fields[7].fill(cp.float32(1.0e8))
    fields[8].fill(cp.float32(3.0e8))
    ccn_before = fields[8].copy()
    launch_cloud_interior_renucleation(*fields, 10.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(fields[8][0], ccn_before[0])
    assert bool(cp.all(fields[8][1:] < ccn_before[1:]))


def test_primary_ice_nucleation_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_primary_ice_nucleation

    fixture = _ORACLE.with_name("primary-ice-nucleation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_active = False
    saw_vapor_limited = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        velocity = device("w_m_s")
        dz = device("dz_m")
        nuclei_minus = device("nuclei_minus_m3")
        nuclei_center = device("nuclei_center_m3")
        nuclei_plus = device("nuclei_plus_m3")
        vapor = device("qv_before")
        ice = device("qi_before")
        number = device("qni_before_per_kg")
        water_before = vapor + ice

        launch_primary_ice_nucleation(
            theta, density, pressure, exner, velocity, dz,
            nuclei_minus, nuclei_center, nuclei_plus,
            vapor, ice, number, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 2.0e-6, 4.0e-5),
                (vapor, "qv_after", 2.0e-6, 2.0e-10),
                (ice, "qi_after", 8.0e-6, 2.0e-14),
                (number, "qni_after_per_kg", 8.0e-6, 5.0e-3)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        expected_ice = np.asarray(
            [float(row["qi_after"]) for row in batch], dtype=np.float32)
        expected_number = np.asarray(
            [float(row["qni_after_per_kg"]) for row in batch],
            dtype=np.float32)
        saw_active |= bool(np.any(expected_ice > 0.0))
        # The high-gradient, cold case is bounded below its raw source by
        # vapor availability; retain that independently dynamic oracle gate.
        saw_vapor_limited |= bool(np.any(
            (expected_number > 1.0e5) & (expected_number < 1.0e6)))
        cp.testing.assert_allclose(
            vapor + ice, water_before, rtol=0.0, atol=2.0e-10)

    assert saw_active
    assert saw_vapor_limited


def test_primary_ice_nucleation_launcher_and_native_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_primary_ice_nucleation

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(12)]
    fields[0].fill(cp.float32(280.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(100000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(1.0))
    fields[5].fill(cp.float32(100.0))
    fields[6].fill(cp.float32(1000.0))
    fields[7].fill(cp.float32(2000.0))
    fields[8].fill(cp.float32(3000.0))
    fields[9].fill(cp.float32(0.001))
    fields[10].fill(0.0)
    fields[11].fill(0.0)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_primary_ice_nucleation(*fields, step)

    bad_shape = list(fields)
    bad_shape[8] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_primary_ice_nucleation(*bad_shape, 1.0)

    # Warm cells are outside the strict 268.15-K option-1 gate.
    before = [value.copy() for value in fields]
    launch_primary_ice_nucleation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)

    # A cold cell with a reversed nuclei gradient is also a native no-op.
    fields[0].fill(cp.float32(250.0))
    fields[6].fill(cp.float32(3000.0))
    fields[8].fill(cp.float32(1000.0))
    before = [value.copy() for value in fields]
    launch_primary_ice_nucleation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_ice_cloud_riming_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_ice_cloud_riming

    fixture = _ORACLE.with_name("ice-cloud-riming.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_active = False
    saw_depletion_cap = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        exner = device("exner")
        cloud = device("qc_before")
        cloud_number = device("qnc_before_per_kg")
        ice = device("qi_before")
        ice_number = device("qni_before_per_kg")
        water_before = cloud + ice

        launch_ice_cloud_riming(
            theta, density, exner, cloud, cloud_number,
            ice, ice_number, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 2.0e-6, 4.0e-5),
                (cloud, "qc_after", 1.0e-5, 2.0e-9),
                (cloud_number, "qnc_after_per_kg", 1.0e-5, 32.0),
                (ice, "qi_after", 1.0e-5, 2.0e-9),
                (ice_number, "qni_after_per_kg", 1.0e-6, 1.0)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        expected_cloud_before = np.asarray(
            [float(row["qc_before"]) for row in batch], dtype=np.float32)
        expected_cloud_after = np.asarray(
            [float(row["qc_after"]) for row in batch], dtype=np.float32)
        expected_loss = expected_cloud_before - expected_cloud_after
        saw_active |= bool(np.any(expected_loss > 0.0))
        saw_depletion_cap |= bool(np.any(np.isclose(
            expected_loss, np.float32(0.1) * expected_cloud_before,
            rtol=2.0e-5, atol=2.0e-10)))
        cp.testing.assert_allclose(
            cloud + ice, water_before, rtol=0.0, atol=3.0e-9)

    assert saw_active
    assert saw_depletion_cap


def test_ice_cloud_riming_launcher_and_strict_native_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_ice_cloud_riming

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(7)]
    fields[0].fill(cp.float32(258.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(1.0))
    fields[3].fill(cp.float32(5.0e-4))
    cloud_mass = np.float32(
        1000.0 * np.pi / 6.0 * (14.0e-6) ** 3)
    fields[4].fill(np.float32(5.0e-4) / cloud_mass)
    fields[5].fill(cp.float32(2.0e-4))
    ice_mass = np.float32((80.0e-6 / 0.1871) ** (1.0 / 0.3429))
    fields[6].fill(np.float32(2.0e-4) / ice_mass)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_ice_cloud_riming(*fields, step)

    bad_shape = list(fields)
    bad_shape[4] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_ice_cloud_riming(*bad_shape, 1.0)

    # A 14-micron droplet is below the strict 15-micron native gate.
    before = [value.copy() for value in fields]
    launch_ice_cloud_riming(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)

    # WRF disables this collection efficiency at exactly freezing.
    active_cloud_mass = np.float32(
        1000.0 * np.pi / 6.0 * (20.0e-6) ** 3)
    fields[4].fill(np.float32(5.0e-4) / active_cloud_mass)
    fields[0].fill(cp.float32(273.15))
    before = [value.copy() for value in fields]
    launch_ice_cloud_riming(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_snow_cloud_riming_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_snow_cloud_riming

    fixture = _ORACLE.with_name("snow-cloud-riming.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_active = False
    saw_depletion_cap = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        exner = device("exner")
        cloud = device("qc_before")
        cloud_number = device("qnc_before_per_kg")
        snow = device("qs_before")
        snow_number = device("qns_before_per_kg")
        water_before = cloud + snow

        launch_snow_cloud_riming(
            theta, density, exner, cloud, cloud_number,
            snow, snow_number, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 2.0e-6, 4.0e-5),
                (cloud, "qc_after", 1.0e-5, 2.0e-9),
                (cloud_number, "qnc_after_per_kg", 1.0e-5, 32.0),
                (snow, "qs_after", 1.0e-5, 2.0e-9),
                (snow_number, "qns_after_per_kg", 1.0e-6, 32.0)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        expected_cloud_before = np.asarray(
            [float(row["qc_before"]) for row in batch], dtype=np.float32)
        expected_cloud_after = np.asarray(
            [float(row["qc_after"]) for row in batch], dtype=np.float32)
        expected_loss = expected_cloud_before - expected_cloud_after
        saw_active |= bool(np.any(expected_loss > 0.0))
        saw_depletion_cap |= bool(np.any(np.isclose(
            expected_loss, np.float32(0.1) * expected_cloud_before,
            rtol=2.0e-5, atol=2.0e-10)))
        cp.testing.assert_allclose(
            cloud + snow, water_before, rtol=0.0, atol=3.0e-9)

    assert saw_active
    assert saw_depletion_cap


def test_snow_cloud_riming_launcher_validation_and_empty_gate():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_snow_cloud_riming

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(7)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(1.0))
    fields[3].fill(cp.float32(5.0e-4))
    cloud_mass = np.float32(
        1000.0 * np.pi / 6.0 * (20.0e-6) ** 3)
    fields[4].fill(np.float32(5.0e-4) / cloud_mass)
    fields[5].fill(cp.float32(0.0))
    fields[6].fill(cp.float32(0.0))

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_snow_cloud_riming(*fields, step)

    bad_shape = list(fields)
    bad_shape[4] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_snow_cloud_riming(*bad_shape, 1.0)

    before = [value.copy() for value in fields]
    launch_snow_cloud_riming(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_graupel_cloud_riming_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_graupel_cloud_riming

    fixture = _ORACLE.with_name("graupel-cloud-riming.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_active = False
    saw_depletion_cap = False
    saw_volume_growth = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        exner = device("exner")
        cloud = device("qc_before")
        cloud_number = device("qnc_before_per_kg")
        graupel = device("qg_before")
        graupel_number = device("qng_before_per_kg")
        graupel_volume = device("qvolg_before_m3_per_kg")
        water_before = cloud + graupel
        volume_before = graupel_volume.copy()

        launch_graupel_cloud_riming(
            theta, density, exner, cloud, cloud_number,
            graupel, graupel_number, graupel_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 2.0e-6, 4.0e-5),
                (cloud, "qc_after", 1.0e-5, 2.0e-9),
                (cloud_number, "qnc_after_per_kg", 1.0e-5, 32.0),
                (graupel, "qg_after", 1.0e-5, 2.0e-9),
                (graupel_number, "qng_after_per_kg", 1.0e-6, 1.0),
                (graupel_volume, "qvolg_after_m3_per_kg", 1.0e-5,
                 2.0e-12)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        expected_cloud_before = np.asarray(
            [float(row["qc_before"]) for row in batch], dtype=np.float32)
        expected_cloud_after = np.asarray(
            [float(row["qc_after"]) for row in batch], dtype=np.float32)
        expected_loss = expected_cloud_before - expected_cloud_after
        saw_active |= bool(np.any(expected_loss > 0.0))
        saw_depletion_cap |= bool(np.any(np.isclose(
            expected_loss, np.float32(0.5) * expected_cloud_before,
            rtol=2.0e-5, atol=2.0e-10)))
        saw_volume_growth |= bool(cp.any(graupel_volume > volume_before))
        cp.testing.assert_allclose(
            cloud + graupel, water_before, rtol=0.0, atol=3.0e-9)

    assert saw_active
    assert saw_depletion_cap
    assert saw_volume_growth


def test_graupel_cloud_riming_launcher_validation_and_empty_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_graupel_cloud_riming

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(8)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(1.0))
    fields[3].fill(cp.float32(5.0e-4))
    cloud_mass = np.float32(
        1000.0 * np.pi / 6.0 * (20.0e-6) ** 3)
    fields[4].fill(np.float32(5.0e-4) / cloud_mass)
    fields[5].fill(cp.float32(0.0))
    fields[6].fill(cp.float32(0.0))
    fields[7].fill(cp.float32(0.0))

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_graupel_cloud_riming(*fields, step)

    bad_shape = list(fields)
    bad_shape[7] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_graupel_cloud_riming(*bad_shape, 1.0)

    before = [value.copy() for value in fields]
    launch_graupel_cloud_riming(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_hail_cloud_riming_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_hail_cloud_riming

    fixture = _ORACLE.with_name("hail-cloud-riming.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_active = False
    saw_depletion_cap = False
    saw_volume_growth = False
    saw_depth_cap_case = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        exner = device("exner")
        dz = device("cell_depth_m")
        cloud = device("qc_before")
        cloud_number = device("qnc_before_per_kg")
        hail = device("qh_before")
        hail_number = device("qnh_before_per_kg")
        hail_volume = device("qvolh_before_m3_per_kg")
        water_before = cloud + hail
        volume_before = hail_volume.copy()

        launch_hail_cloud_riming(
            theta, density, exner, dz, cloud, cloud_number,
            hail, hail_number, hail_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        # Fixed before the first GPU comparison; do not tune to the fixture.
        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 2.0e-6, 4.0e-5),
                (cloud, "qc_after", 1.0e-5, 2.0e-9),
                (cloud_number, "qnc_after_per_kg", 1.0e-5, 32.0),
                (hail, "qh_after", 1.0e-5, 2.0e-9),
                (hail_number, "qnh_after_per_kg", 1.0e-6, 1.0),
                (hail_volume, "qvolh_after_m3_per_kg", 1.0e-5,
                 2.0e-12)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        expected_cloud_before = np.asarray(
            [float(row["qc_before"]) for row in batch], dtype=np.float32)
        expected_cloud_after = np.asarray(
            [float(row["qc_after"]) for row in batch], dtype=np.float32)
        expected_loss = expected_cloud_before - expected_cloud_after
        saw_active |= bool(np.any(expected_loss > 0.0))
        saw_depletion_cap |= bool(np.any(np.isclose(
            expected_loss, np.float32(0.5) * expected_cloud_before,
            rtol=2.0e-5, atol=2.0e-10)))
        saw_volume_growth |= bool(cp.any(hail_volume > volume_before))
        # The 40-mm, 60-s oracle row has an uncapped native speed above
        # 38 m/s but dz / dt = 22.5 m/s, so it exercises WRF's qhlacw cap.
        saw_depth_cap_case |= any(
            float(row["hail_diameter_m"]) >= 39.9e-3
            and float(row["dt_s"]) >= 60.0
            and float(row["cell_depth_m"]) / float(row["dt_s"]) <= 22.5
            and loss > 0.0
            for row, loss in zip(batch, expected_loss))
        cp.testing.assert_allclose(
            cloud + hail, water_before, rtol=0.0, atol=3.0e-9)

    assert saw_active
    assert saw_depletion_cap
    assert saw_volume_growth
    assert saw_depth_cap_case


def test_hail_cloud_riming_launcher_validation_and_empty_gates():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_hail_cloud_riming

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(9)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(1.0))
    fields[3].fill(cp.float32(1000.0))
    fields[4].fill(cp.float32(5.0e-4))
    cloud_mass = np.float32(
        1000.0 * np.pi / 6.0 * (20.0e-6) ** 3)
    fields[5].fill(np.float32(5.0e-4) / cloud_mass)
    fields[6].fill(cp.float32(0.0))
    fields[7].fill(cp.float32(0.0))
    fields[8].fill(cp.float32(0.0))

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_hail_cloud_riming(*fields, step)

    bad_shape = list(fields)
    bad_shape[8] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_hail_cloud_riming(*bad_shape, 1.0)

    before = [value.copy() for value in fields]
    launch_hail_cloud_riming(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_rain_ice_collection_freezing_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_ice_collection_freezing

    fixture = _ORACLE.with_name("rain-ice-collection-freezing.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_rain_cap = False
    saw_ice_cap = False
    saw_volume_growth = False
    saw_legacy_mass_only_fallback = False
    saw_warm_craci_number_collection = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        temperature = device("temperature_k")
        vapor = device("qv_before")
        rain = device("qr_before")
        rain_number = device("qnr_before_per_kg")
        ice = device("qi_before")
        ice_number = device("qni_before_per_kg")
        graupel = device("qg_before")
        graupel_number = device("qng_before_per_kg")
        graupel_volume = device("qvolg_before_m3_per_kg")
        rain_before = rain.copy()
        rain_number_before = rain_number.copy()
        ice_before = ice.copy()
        ice_number_before = ice_number.copy()
        graupel_number_before = graupel_number.copy()
        volume_before = graupel_volume.copy()
        water_before = rain + ice + graupel

        launch_rain_ice_collection_freezing(
            theta, density, pressure, exner, temperature, vapor,
            rain, rain_number, ice, ice_number,
            graupel, graupel_number, graupel_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        # Fixed before the first GPU comparison; do not tune to the fixture.
        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 2.0e-6, 4.0e-5),
                (vapor, "qv_after", 0.0, 1.0e-10),
                (rain, "qr_after", 2.0e-5, 2.0e-9),
                (rain_number, "qnr_after_per_kg", 2.0e-5, 64.0),
                (ice, "qi_after", 2.0e-5, 2.0e-9),
                (ice_number, "qni_after_per_kg", 2.0e-5, 256.0),
                (graupel, "qg_after", 2.0e-5, 2.0e-9),
                (graupel_number, "qng_after_per_kg", 2.0e-5, 2.0),
                (graupel_volume, "qvolg_after_m3_per_kg",
                 2.0e-5, 2.0e-12)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        rain_loss = rain_before - rain
        ice_loss = ice_before - ice
        saw_rain_cap |= bool(cp.any(cp.isclose(
            rain_loss, cp.float32(0.1) * rain_before,
            rtol=2.0e-5, atol=2.0e-9)))
        saw_ice_cap |= bool(cp.any(cp.isclose(
            ice_loss, cp.float32(0.1) * ice_before,
            rtol=2.0e-5, atol=2.0e-9)))
        saw_volume_growth |= bool(cp.any(graupel_volume > volume_before))

        warm = temperature > cp.float32(270.15)
        legacy = warm & (rain_loss > 0.0)
        saw_legacy_mass_only_fallback |= bool(cp.any(
            legacy
            & (rain_number == rain_number_before)
            & (graupel_number == graupel_number_before)))
        saw_warm_craci_number_collection |= bool(cp.any(
            legacy & (ice == ice_before) & (ice_number < ice_number_before)))
        cp.testing.assert_allclose(
            rain + ice + graupel, water_before, rtol=0.0, atol=5.0e-9)

    assert saw_rain_cap
    assert saw_ice_cap
    assert saw_volume_growth
    assert saw_legacy_mass_only_fallback
    assert saw_warm_craci_number_collection


def test_rain_ice_collection_freezing_validation_and_empty_gate():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_ice_collection_freezing

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(13)]
    fields[0].fill(cp.float32(265.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(80000.0))
    fields[3].fill(cp.float32(0.95))
    fields[4].fill(cp.float32(265.0))
    fields[5].fill(cp.float32(2.0e-3))
    for field in fields[6:]:
        field.fill(0.0)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_rain_ice_collection_freezing(*fields, step)

    bad_shape = list(fields)
    bad_shape[9] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_rain_ice_collection_freezing(*bad_shape, 1.0)

    before = [value.copy() for value in fields]
    launch_rain_ice_collection_freezing(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_frozen_cross_collection_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_frozen_cross_collection

    fixture = _ORACLE.with_name("frozen-cross-collection.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 80

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_rain_mass_cap = False
    saw_rain_number_cap = False
    saw_ice_mass_cap = False
    saw_ice_number_cap = False
    saw_snow_mass_cap = False
    saw_snow_number_cap = False
    saw_snow_taper = False
    saw_graupel_volume_growth = False
    saw_hail_volume_growth = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        case = np.asarray([int(row["case"]) for row in batch])
        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        exner = device("exner")
        temperature = device("temperature_k")
        dz = cp.full_like(density, cp.float32(1000.0))
        cloud = device("qc_before")
        rain = device("qr_before")
        rain_number = device("qnr_before_per_kg")
        ice = device("qi_before")
        ice_number = device("qni_before_per_kg")
        snow = device("qs_before")
        snow_number = device("qns_before_per_kg")
        graupel = device("qg_before")
        graupel_number = device("qng_before_per_kg")
        graupel_volume = device("qvolg_before_m3_per_kg")
        hail = device("qh_before")
        hail_number = device("qnh_before_per_kg")
        hail_volume = device("qvolh_before_m3_per_kg")

        theta_before = theta.copy()
        cloud_before = cloud.copy()
        rain_before = rain.copy()
        rain_number_before = rain_number.copy()
        ice_before = ice.copy()
        ice_number_before = ice_number.copy()
        snow_before = snow.copy()
        snow_number_before = snow_number.copy()
        graupel_volume_before = graupel_volume.copy()
        hail_volume_before = hail_volume.copy()
        water_before = (
            rain.astype(cp.float64) + ice.astype(cp.float64)
            + snow.astype(cp.float64) + graupel.astype(cp.float64)
            + hail.astype(cp.float64))

        launch_frozen_cross_collection(
            theta, density, exner, temperature, dz, cloud,
            rain, rain_number, ice, ice_number, snow, snow_number,
            graupel, graupel_number, graupel_volume,
            hail, hail_number, hail_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        # Fixed before the first GPU comparison; do not tune to the fixture.
        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 3.0e-6, 5.0e-5),
                (rain, "qr_after", 3.0e-5, 3.0e-9),
                (rain_number, "qnr_after_per_kg", 3.0e-5, 64.0),
                (ice, "qi_after", 3.0e-5, 3.0e-9),
                (ice_number, "qni_after_per_kg", 3.0e-5, 256.0),
                (snow, "qs_after", 3.0e-5, 3.0e-9),
                (snow_number, "qns_after_per_kg", 3.0e-5, 64.0),
                (graupel, "qg_after", 3.0e-5, 3.0e-9),
                (graupel_number, "qng_after_per_kg", 3.0e-5, 2.0),
                (graupel_volume, "qvolg_after_m3_per_kg",
                 3.0e-5, 3.0e-12),
                (hail, "qh_after", 3.0e-5, 3.0e-9),
                (hail_number, "qnh_after_per_kg", 3.0e-5, 2.0),
                (hail_volume, "qvolh_after_m3_per_kg",
                 3.0e-5, 3.0e-12)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        cp.testing.assert_array_equal(cloud, cloud_before)
        cp.testing.assert_allclose(
            rain.astype(cp.float64) + ice.astype(cp.float64)
            + snow.astype(cp.float64) + graupel.astype(cp.float64)
            + hail.astype(cp.float64),
            water_before, rtol=0.0, atol=6.0e-9)

        no_rain = rain_before == 0.0
        cp.testing.assert_allclose(
            theta[no_rain], theta_before[no_rain],
            rtol=0.0, atol=5.0e-5)
        rain_loss = rain_before - rain
        ice_loss = ice_before - ice
        snow_loss = snow_before - snow
        rain_cap = cp.asarray(case == 18)
        ice_cap = cp.asarray(case == 16)
        snow_cap = cp.asarray(case == 17)
        saw_rain_mass_cap |= bool(cp.any(rain_cap & cp.isclose(
            rain_loss, cp.float32(0.2) * rain_before,
            rtol=3.0e-5, atol=3.0e-9)))
        saw_rain_number_cap |= bool(cp.any(rain_cap & cp.isclose(
            rain_number_before - rain_number,
            cp.float32(0.2) * rain_number_before,
            rtol=3.0e-5, atol=64.0)))
        saw_ice_mass_cap |= bool(cp.any(ice_cap & cp.isclose(
            ice_loss, cp.float32(0.3) * ice_before,
            rtol=3.0e-5, atol=3.0e-9)))
        saw_ice_number_cap |= bool(cp.any(ice_cap & cp.isclose(
            ice_number_before - ice_number,
            cp.float32(0.3) * ice_number_before,
            rtol=3.0e-5, atol=256.0)))
        saw_snow_mass_cap |= bool(cp.any(snow_cap & cp.isclose(
            snow_loss, cp.float32(0.2) * snow_before,
            rtol=3.0e-5, atol=3.0e-9)))
        saw_snow_number_cap |= bool(cp.any(snow_cap & cp.isclose(
            snow_number_before - snow_number,
            cp.float32(0.2) * snow_number_before,
            rtol=3.0e-5, atol=64.0)))
        taper = cp.asarray(case == 5)
        saw_snow_taper |= bool(cp.any(taper & (snow_loss > 0.0)))
        saw_graupel_volume_growth |= bool(cp.any(
            graupel_volume > graupel_volume_before))
        saw_hail_volume_growth |= bool(cp.any(
            hail_volume > hail_volume_before))

    assert saw_rain_mass_cap
    assert saw_rain_number_cap
    assert saw_ice_mass_cap
    assert saw_ice_number_cap
    assert saw_snow_mass_cap
    assert saw_snow_number_cap
    assert saw_snow_taper
    assert saw_graupel_volume_growth
    assert saw_hail_volume_growth


def test_frozen_cross_collection_validation_and_empty_gate():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_frozen_cross_collection

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(18)]
    fields[0].fill(cp.float32(240.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(1.0))
    fields[3].fill(cp.float32(240.0))
    fields[4].fill(cp.float32(1000.0))
    for field in fields[5:]:
        field.fill(0.0)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_frozen_cross_collection(*fields, step)

    bad_shape = list(fields)
    bad_shape[11] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_frozen_cross_collection(*bad_shape, 1.0)

    before = [value.copy() for value in fields]
    launch_frozen_cross_collection(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_melting_liquid_shedding_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_melting_liquid_shedding

    fixture = _ORACLE.with_name("melting-liquid-shedding.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_snow_cap = False
    saw_graupel_cap = False
    saw_wet_shedding = False
    saw_warm_shedding = False
    saw_melt_soaking = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        case = np.asarray([int(row["case"]) for row in batch])
        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        temperature = device("temperature_k")
        vapor = device("qv_before")
        depth = device("cell_depth_m")
        cloud = device("qc_before")
        cloud_number = device("qnc_before_per_kg")
        rain = device("qr_before")
        rain_number = device("qnr_before_per_kg")
        snow = device("qs_before")
        snow_number = device("qns_before_per_kg")
        graupel = device("qg_before")
        graupel_number = device("qng_before_per_kg")
        graupel_volume = device("qvolg_before_m3_per_kg")
        hail = device("qh_before")
        hail_number = device("qnh_before_per_kg")
        hail_volume = device("qvolh_before_m3_per_kg")

        theta_before = theta.copy()
        vapor_before = vapor.copy()
        cloud_before = cloud.copy()
        rain_before = rain.copy()
        snow_before = snow.copy()
        graupel_before = graupel.copy()
        graupel_volume_before = graupel_volume.copy()
        water_before = (
            cloud.astype(cp.float64) + rain.astype(cp.float64)
            + snow.astype(cp.float64) + graupel.astype(cp.float64)
            + hail.astype(cp.float64))

        launch_melting_liquid_shedding(
            theta, density, pressure, exner, temperature, vapor, depth,
            cloud, cloud_number, rain, rain_number, snow, snow_number,
            graupel, graupel_number, graupel_volume,
            hail, hail_number, hail_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        # Fixed before the first GPU comparison; do not tune to the fixture.
        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 3.0e-6, 1.0e-4),
                (cloud, "qc_after", 5.0e-5, 5.0e-9),
                (cloud_number, "qnc_after_per_kg", 5.0e-5, 64.0),
                (rain, "qr_after", 5.0e-5, 5.0e-9),
                (rain_number, "qnr_after_per_kg", 5.0e-5, 64.0),
                (snow, "qs_after", 5.0e-5, 5.0e-9),
                (snow_number, "qns_after_per_kg", 5.0e-5, 64.0),
                (graupel, "qg_after", 5.0e-5, 5.0e-9),
                (graupel_number, "qng_after_per_kg", 5.0e-5, 4.0),
                (graupel_volume, "qvolg_after_m3_per_kg",
                 5.0e-5, 5.0e-12),
                (hail, "qh_after", 5.0e-5, 5.0e-9),
                (hail_number, "qnh_after_per_kg", 5.0e-5, 4.0),
                (hail_volume, "qvolh_after_m3_per_kg",
                 5.0e-5, 5.0e-12)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        cp.testing.assert_array_equal(vapor, vapor_before)
        cp.testing.assert_allclose(
            cloud.astype(cp.float64) + rain.astype(cp.float64)
            + snow.astype(cp.float64) + graupel.astype(cp.float64)
            + hail.astype(cp.float64),
            water_before, rtol=0.0, atol=8.0e-9)

        snow_cap = cp.asarray(case == 13)
        graupel_cap = cp.asarray(case == 14)
        saw_snow_cap |= bool(cp.any(snow_cap & cp.isclose(
            snow_before - snow, cp.float32(0.7) * snow_before,
            rtol=5.0e-5, atol=5.0e-9)))
        saw_graupel_cap |= bool(cp.any(graupel_cap & cp.isclose(
            graupel_before - graupel,
            cp.float32(0.95) * graupel_before,
            rtol=5.0e-5, atol=5.0e-9)))
        wet_case = cp.asarray(case == 5)
        saw_wet_shedding |= bool(cp.any(
            wet_case & (rain > rain_before) & (cloud < cloud_before)))
        warm_case = cp.asarray(case == 10)
        saw_warm_shedding |= bool(cp.any(
            warm_case & (rain > rain_before) & (cloud < cloud_before)))
        soak_case = cp.asarray(case == 14)
        saw_melt_soaking |= bool(cp.any(
            soak_case & (graupel_volume < graupel_volume_before)))

        exact_freezing = cp.asarray(case == 9)
        cp.testing.assert_allclose(
            theta[exact_freezing], theta_before[exact_freezing],
            rtol=0.0, atol=1.0e-4)

    assert saw_snow_cap
    assert saw_graupel_cap
    assert saw_wet_shedding
    assert saw_warm_shedding
    assert saw_melt_soaking


def test_melting_liquid_shedding_validation_and_empty_gate():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_melting_liquid_shedding

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(19)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(100000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(250.0))
    fields[5].fill(cp.float32(0.0))
    fields[6].fill(cp.float32(1000.0))
    for field in fields[7:]:
        field.fill(0.0)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_melting_liquid_shedding(*fields, step)

    bad_shape = list(fields)
    bad_shape[15] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_melting_liquid_shedding(*bad_shape, 1.0)

    before = [value.copy() for value in fields]
    launch_melting_liquid_shedding(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_secondary_ice_conversions_match_official_wrf_processes():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_secondary_ice_conversions

    fixture = _ORACLE.with_name("secondary-ice-conversions.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 152

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    saw_contact = False
    saw_homogeneous = False
    saw_hm_graupel = False
    saw_hm_hail = False
    saw_ice_to_graupel = False
    saw_snow_to_graupel = False
    saw_graupel_to_hail = False
    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        case = np.asarray([int(row["case"]) for row in batch])
        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        temperature = device("temperature_k")
        vapor = device("qv_before")
        depth = device("cell_depth_m")
        cloud = device("qc_before")
        cloud_number = device("qnc_before_per_kg")
        rain = device("qr_before")
        rain_number = device("qnr_before_per_kg")
        ice = device("qi_before")
        ice_number = device("qni_before_per_kg")
        snow = device("qs_before")
        snow_number = device("qns_before_per_kg")
        graupel = device("qg_before")
        graupel_number = device("qng_before_per_kg")
        graupel_volume = device("qvolg_before_m3_per_kg")
        hail = device("qh_before")
        hail_number = device("qnh_before_per_kg")
        hail_volume = device("qvolh_before_m3_per_kg")

        before = [value.copy() for value in (
            theta, vapor, cloud, cloud_number, rain, rain_number,
            ice, ice_number, snow, snow_number,
            graupel, graupel_number, graupel_volume,
            hail, hail_number, hail_volume)]
        water_before = (
            vapor.astype(cp.float64) + cloud.astype(cp.float64)
            + rain.astype(cp.float64) + ice.astype(cp.float64)
            + snow.astype(cp.float64) + graupel.astype(cp.float64)
            + hail.astype(cp.float64))

        launch_secondary_ice_conversions(
            theta, density, pressure, exner, temperature, vapor, depth,
            cloud, cloud_number, rain, rain_number, ice, ice_number,
            snow, snow_number, graupel, graupel_number, graupel_volume,
            hail, hail_number, hail_volume, float(step))
        cp.cuda.Stream.null.synchronize()

        # Fixed before the first GPU comparison; do not tune to the fixture.
        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 3.0e-6, 1.0e-4),
                (cloud, "qc_after", 5.0e-5, 5.0e-9),
                (cloud_number, "qnc_after_per_kg", 5.0e-5, 64.0),
                (rain, "qr_after", 0.0, 1.0e-12),
                (rain_number, "qnr_after_per_kg", 0.0, 1.0e-8),
                (ice, "qi_after", 5.0e-5, 5.0e-9),
                (ice_number, "qni_after_per_kg", 5.0e-5, 64.0),
                (snow, "qs_after", 5.0e-5, 5.0e-9),
                (snow_number, "qns_after_per_kg", 5.0e-5, 64.0),
                (graupel, "qg_after", 5.0e-5, 5.0e-9),
                (graupel_number, "qng_after_per_kg", 5.0e-5, 4.0),
                (graupel_volume, "qvolg_after_m3_per_kg",
                 5.0e-5, 5.0e-12),
                (hail, "qh_after", 5.0e-5, 5.0e-9),
                (hail_number, "qnh_after_per_kg", 5.0e-5, 4.0),
                (hail_volume, "qvolh_after_m3_per_kg",
                 5.0e-5, 5.0e-12)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        cp.testing.assert_array_equal(vapor, before[1])
        cp.testing.assert_allclose(
            vapor.astype(cp.float64) + cloud.astype(cp.float64)
            + rain.astype(cp.float64) + ice.astype(cp.float64)
            + snow.astype(cp.float64) + graupel.astype(cp.float64)
            + hail.astype(cp.float64),
            water_before, rtol=0.0, atol=8.0e-9)

        contact = cp.asarray((case >= 2) & (case <= 5))
        homogeneous = cp.asarray((case >= 6) & (case <= 8))
        hm_graupel = cp.asarray((case >= 13) & (case <= 17))
        hm_hail = cp.asarray(case == 20)
        ice_to_graupel = cp.asarray(case == 25)
        snow_to_graupel = cp.asarray(case == 26)
        graupel_to_hail = cp.asarray(
            (case >= 29) & (case <= 34) & (case != 31) & (case != 33))
        saw_contact |= bool(cp.any(contact & (ice_number > before[7])))
        saw_homogeneous |= bool(cp.any(homogeneous & (cloud < before[2])))
        saw_hm_graupel |= bool(cp.any(hm_graupel & (graupel < before[10])))
        saw_hm_hail |= bool(cp.any(hm_hail & (hail < before[13])))
        saw_ice_to_graupel |= bool(cp.any(
            ice_to_graupel & (ice < before[6]) & (graupel > before[10])))
        saw_snow_to_graupel |= bool(cp.any(
            snow_to_graupel & (snow < before[8]) & (graupel > before[10])))
        saw_graupel_to_hail |= bool(cp.any(
            graupel_to_hail & (graupel < before[10]) & (hail > before[13])))

        reverse_off = cp.asarray(case == 37)
        for actual, original in zip(
                (graupel, graupel_number, graupel_volume,
                 hail, hail_number, hail_volume),
                before[10:16]):
            cp.testing.assert_array_equal(actual[reverse_off], original[reverse_off])

    assert saw_contact
    assert saw_homogeneous
    assert saw_hm_graupel
    assert saw_hm_hail
    assert saw_ice_to_graupel
    assert saw_snow_to_graupel
    assert saw_graupel_to_hail


def test_secondary_ice_conversions_validation_and_empty_gate():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_secondary_ice_conversions

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(21)]
    fields[0].fill(cp.float32(250.0))
    fields[1].fill(cp.float32(1.0))
    fields[2].fill(cp.float32(100000.0))
    fields[3].fill(cp.float32(1.0))
    fields[4].fill(cp.float32(250.0))
    fields[5].fill(cp.float32(0.0))
    fields[6].fill(cp.float32(1000.0))
    for field in fields[7:]:
        field.fill(0.0)

    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_secondary_ice_conversions(*fields, step)

    bad_shape = list(fields)
    bad_shape[17] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_secondary_ice_conversions(*bad_shape, 1.0)

    before = [value.copy() for value in fields]
    launch_secondary_ice_conversions(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)


def test_rain_evaporation_matches_official_wrf_process():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_evaporation

    fixture = _ORACLE.with_name("rain-evaporation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48

    by_step: dict[np.float32, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(np.float32(row["dt_s"]), []).append(row)

    for step, batch in by_step.items():
        def device(name):
            return cp.asarray(np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32))

        theta = device("theta_before_k")
        density = device("rho_kg_m3")
        pressure = device("pressure_pa")
        exner = device("exner")
        vapor = device("qv_before")
        rain = device("qr_before")
        rain_number = device("nr_before_per_kg")
        water_before = vapor + rain

        launch_rain_evaporation(
            theta, density, pressure, exner, vapor, rain, rain_number,
            float(step))
        cp.cuda.Stream.null.synchronize()

        for actual, name, rtol, atol in (
                (theta, "theta_after_k", 4.0e-6, 4.0e-5),
                (vapor, "qv_after", 6.0e-6, 3.0e-10),
                (rain, "qr_after", 6.0e-6, 3.0e-10),
                (rain_number, "nr_after_per_kg", 8.0e-6, 3.0e-3)):
            expected = np.asarray(
                [float(row[name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=rtol, atol=atol)

        # Total-state comparisons can hide a small but dynamically important
        # evaporation tendency.  Compare each delta independently as well.
        for actual, before_name, after_name, rtol, atol in (
                (theta, "theta_before_k", "theta_after_k", 3.0e-3, 3.2e-5),
                (vapor, "qv_before", "qv_after", 2.0e-3, 2.0e-9),
                (rain, "qr_before", "qr_after", 2.0e-3, 2.0e-9),
                (rain_number, "nr_before_per_kg", "nr_after_per_kg",
                 3.0e-4, 5.0e-3)):
            before = np.asarray(
                [float(row[before_name]) for row in batch], dtype=np.float32)
            expected_after = np.asarray(
                [float(row[after_name]) for row in batch], dtype=np.float32)
            np.testing.assert_allclose(
                cp.asnumpy(actual) - before, expected_after - before,
                rtol=rtol, atol=atol)
        cp.testing.assert_allclose(
            vapor + rain, water_before, rtol=0.0, atol=2.0e-9)


def test_rain_evaporation_launcher_validation_and_empty_gate():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_evaporation

    fields = [cp.ones((2, 3), dtype=cp.float32) for _ in range(7)]
    fields[2].fill(cp.float32(100000.0))
    fields[3].fill(1.0)
    fields[4].fill(cp.float32(0.001))
    fields[5].fill(0.0)
    fields[6].fill(0.0)
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_rain_evaporation(*fields, step)

    bad_shape = list(fields)
    bad_shape[1] = cp.ones((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_rain_evaporation(*bad_shape, 1.0)

    before = [value.copy() for value in (fields[0], *fields[4:])]
    launch_rain_evaporation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip((fields[0], *fields[4:]), before):
        cp.testing.assert_array_equal(actual, expected)

    # Default rcond=2 permits evaporation only in this process routine.
    # Supersaturated vapor must not condense onto existing rain.
    fields[0].fill(cp.float32(300.0))
    fields[4].fill(cp.float32(0.1))
    fields[5].fill(cp.float32(1.0e-4))
    mean_volume = cp.float32(0.523599 * (0.5e-3)**3)
    fields[6].fill(cp.float32(1.0e-4) / (cp.float32(1000.0) * mean_volume))
    before = [value.copy() for value in (fields[0], *fields[4:])]
    launch_rain_evaporation(*fields, 1.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip((fields[0], *fields[4:]), before):
        cp.testing.assert_array_equal(actual, expected)


def test_rain_sedimentation_matches_official_wrf_columns():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_sedimentation

    fixture = _ORACLE.with_name("rain-sedimentation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 6 * 12

    by_case: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case"]), []).append(row)

    for batch in by_case.values():
        nz = max(int(row["k"]) for row in batch)
        nx = max(int(row["column"]) for row in batch)

        def volume(name):
            result = np.empty((nz, 1, nx), dtype=np.float32)
            for row in batch:
                result[int(row["k"]) - 1, 0,
                       int(row["column"]) - 1] = float(row[name])
            return result

        density_host = volume("rho_kg_m3")
        depth_host = volume("dz_m")
        rain_before = volume("qr_before")
        number_before = volume("nr_before_per_kg")
        density = cp.asarray(density_host)
        depth = cp.asarray(depth_host)
        rain = cp.asarray(rain_before)
        number = cp.asarray(number_before)
        rainnc = cp.full((1, nx), cp.float32(2.0))
        rainncv = cp.full((1, nx), cp.float32(-1.0))
        step = float(batch[0]["dt_s"])

        launch_rain_sedimentation(
            density, rain, number, depth, rainnc, rainncv, step)
        cp.cuda.Stream.null.synchronize()

        np.testing.assert_allclose(
            cp.asnumpy(rain), volume("qr_after"),
            rtol=3.0e-5, atol=3.0e-12)
        np.testing.assert_allclose(
            cp.asnumpy(number), volume("nr_after_per_kg"),
            rtol=1.5e-5, atol=3.0e-4)
        expected_surface = np.empty((1, nx), dtype=np.float32)
        for row in batch:
            if int(row["k"]) == 1:
                expected_surface[0, int(row["column"]) - 1] = float(
                    row["rainncv_mm"])
        np.testing.assert_allclose(
            cp.asnumpy(rainncv), expected_surface,
            rtol=8.0e-6, atol=3.0e-10)
        np.testing.assert_allclose(
            cp.asnumpy(rainnc), 2.0 + expected_surface,
            rtol=3.0e-7, atol=3.0e-7)

        # First-order fallout is conservative in rho*dz*q plus surface
        # export.  Check the production kernel independently of the oracle.
        water_before = np.sum(
            rain_before * density_host * depth_host, axis=0)
        water_after = np.sum(
            cp.asnumpy(rain) * density_host * depth_host, axis=0)
        np.testing.assert_allclose(
            water_after + cp.asnumpy(rainncv), water_before,
            rtol=2.0e-6, atol=2.0e-7)


def test_rain_sedimentation_launcher_validation_and_empty_column():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_rain_sedimentation

    volumes = [cp.ones((4, 2, 3), dtype=cp.float32) for _ in range(4)]
    volumes[1].fill(0.0)
    volumes[2].fill(0.0)
    surfaces = [cp.zeros((2, 3), dtype=cp.float32) for _ in range(2)]
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_rain_sedimentation(*volumes, *surfaces, step)

    with pytest.raises(ValueError, match="shape"):
        launch_rain_sedimentation(
            *volumes, surfaces[0], cp.zeros((6,), dtype=cp.float32), 1.0)
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        too_shallow = [cp.ones((1, 2, 3), dtype=cp.float32)
                       for _ in range(4)]
        launch_rain_sedimentation(*too_shallow, *surfaces, 1.0)

    launch_rain_sedimentation(*volumes, *surfaces, 1.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(volumes[1], 0.0)
    cp.testing.assert_array_equal(volumes[2], 0.0)
    cp.testing.assert_array_equal(surfaces[1], 0.0)


def test_snow_sedimentation_matches_official_wrf_columns():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_snow_sedimentation

    fixture = _ORACLE.with_name("snow-sedimentation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 6 * 12

    by_case: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case"]), []).append(row)

    for batch in by_case.values():
        nz = max(int(row["k"]) for row in batch)
        nx = max(int(row["column"]) for row in batch)

        def volume(name):
            result = np.empty((nz, 1, nx), dtype=np.float32)
            for row in batch:
                result[int(row["k"]) - 1, 0,
                       int(row["column"]) - 1] = float(row[name])
            return result

        density_host = volume("rho_kg_m3")
        depth_host = volume("dz_m")
        snow_before = volume("qs_before")
        number_before = volume("ns_before_per_kg")
        density = cp.asarray(density_host)
        depth = cp.asarray(depth_host)
        snow = cp.asarray(snow_before)
        number = cp.asarray(number_before)
        snownc = cp.full((1, nx), cp.float32(2.0))
        snowncv = cp.full((1, nx), cp.float32(-1.0))
        step = float(batch[0]["dt_s"])

        launch_snow_sedimentation(
            density, snow, number, depth, snownc, snowncv, step)
        cp.cuda.Stream.null.synchronize()

        np.testing.assert_allclose(
            cp.asnumpy(snow), volume("qs_after"),
            rtol=3.0e-5, atol=3.0e-12)
        np.testing.assert_allclose(
            cp.asnumpy(number), volume("ns_after_per_kg"),
            rtol=1.5e-5, atol=3.0e-4)
        expected_surface = np.empty((1, nx), dtype=np.float32)
        for row in batch:
            if int(row["k"]) == 1:
                expected_surface[0, int(row["column"]) - 1] = float(
                    row["snowncv_kg_m2"])
        np.testing.assert_allclose(
            cp.asnumpy(snowncv), expected_surface,
            rtol=8.0e-6, atol=3.0e-10)
        np.testing.assert_allclose(
            cp.asnumpy(snownc), 2.0 + expected_surface,
            rtol=3.0e-7, atol=3.0e-7)

        snow_before_column = np.sum(
            snow_before * density_host * depth_host, axis=0)
        snow_after_column = np.sum(
            cp.asnumpy(snow) * density_host * depth_host, axis=0)
        np.testing.assert_allclose(
            snow_after_column + cp.asnumpy(snowncv), snow_before_column,
            rtol=2.0e-6, atol=2.0e-7)


def test_snow_sedimentation_launcher_validation_and_empty_column():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_snow_sedimentation

    volumes = [cp.ones((4, 2, 3), dtype=cp.float32) for _ in range(4)]
    volumes[1].fill(0.0)
    volumes[2].fill(0.0)
    surfaces = [cp.zeros((2, 3), dtype=cp.float32) for _ in range(2)]
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_snow_sedimentation(*volumes, *surfaces, step)

    with pytest.raises(ValueError, match="shape"):
        launch_snow_sedimentation(
            *volumes, surfaces[0], cp.zeros((6,), dtype=cp.float32), 1.0)
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        too_shallow = [cp.ones((1, 2, 3), dtype=cp.float32)
                       for _ in range(4)]
        launch_snow_sedimentation(*too_shallow, *surfaces, 1.0)

    launch_snow_sedimentation(*volumes, *surfaces, 1.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(volumes[1], 0.0)
    cp.testing.assert_array_equal(volumes[2], 0.0)
    cp.testing.assert_array_equal(surfaces[1], 0.0)


def test_ice_sedimentation_matches_official_wrf_columns():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_ice_sedimentation

    fixture = _ORACLE.with_name("ice-sedimentation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 6 * 12

    by_case: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case"]), []).append(row)

    for batch in by_case.values():
        nz = max(int(row["k"]) for row in batch)
        nx = max(int(row["column"]) for row in batch)

        def volume(name):
            result = np.empty((nz, 1, nx), dtype=np.float32)
            for row in batch:
                result[int(row["k"]) - 1, 0,
                       int(row["column"]) - 1] = float(row[name])
            return result

        density_host = volume("rho_kg_m3")
        depth_host = volume("dz_m")
        ice_before = volume("qi_before")
        number_before = volume("ni_before_per_kg")
        density = cp.asarray(density_host)
        depth = cp.asarray(depth_host)
        ice = cp.asarray(ice_before)
        number = cp.asarray(number_before)
        icenc = cp.full((1, nx), cp.float32(2.0))
        icencv = cp.full((1, nx), cp.float32(-1.0))
        step = float(batch[0]["dt_s"])

        launch_ice_sedimentation(
            density, ice, number, depth, icenc, icencv, step)
        cp.cuda.Stream.null.synchronize()

        np.testing.assert_allclose(
            cp.asnumpy(ice), volume("qi_after"),
            rtol=3.0e-5, atol=3.0e-12)
        np.testing.assert_allclose(
            cp.asnumpy(number), volume("ni_after_per_kg"),
            rtol=1.5e-5, atol=3.0e-4)
        expected_surface = np.empty((1, nx), dtype=np.float32)
        for row in batch:
            if int(row["k"]) == 1:
                expected_surface[0, int(row["column"]) - 1] = float(
                    row["icencv_kg_m2"])
        np.testing.assert_allclose(
            cp.asnumpy(icencv), expected_surface,
            rtol=8.0e-6, atol=3.0e-10)
        np.testing.assert_allclose(
            cp.asnumpy(icenc), 2.0 + expected_surface,
            rtol=3.0e-7, atol=3.0e-7)

        ice_before_column = np.sum(
            ice_before * density_host * depth_host, axis=0)
        ice_after_column = np.sum(
            cp.asnumpy(ice) * density_host * depth_host, axis=0)
        np.testing.assert_allclose(
            ice_after_column + cp.asnumpy(icencv), ice_before_column,
            rtol=2.0e-6, atol=2.0e-7)


def test_ice_sedimentation_launcher_validation_and_empty_column():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_ice_sedimentation

    volumes = [cp.ones((4, 2, 3), dtype=cp.float32) for _ in range(4)]
    volumes[1].fill(0.0)
    volumes[2].fill(0.0)
    surfaces = [cp.zeros((2, 3), dtype=cp.float32) for _ in range(2)]
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_ice_sedimentation(*volumes, *surfaces, step)

    with pytest.raises(ValueError, match="shape"):
        launch_ice_sedimentation(
            *volumes, surfaces[0], cp.zeros((6,), dtype=cp.float32), 1.0)
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        too_shallow = [cp.ones((1, 2, 3), dtype=cp.float32)
                       for _ in range(4)]
        launch_ice_sedimentation(*too_shallow, *surfaces, 1.0)

    launch_ice_sedimentation(*volumes, *surfaces, 1.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(volumes[1], 0.0)
    cp.testing.assert_array_equal(volumes[2], 0.0)
    cp.testing.assert_array_equal(surfaces[1], 0.0)


def test_graupel_sedimentation_matches_official_wrf_columns():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_graupel_sedimentation

    fixture = _ORACLE.with_name("graupel-sedimentation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 6 * 12

    by_case: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case"]), []).append(row)

    for batch in by_case.values():
        nz = max(int(row["k"]) for row in batch)
        nx = max(int(row["column"]) for row in batch)

        def volume(name):
            result = np.empty((nz, 1, nx), dtype=np.float32)
            for row in batch:
                result[int(row["k"]) - 1, 0,
                       int(row["column"]) - 1] = float(row[name])
            return result

        density_host = volume("rho_kg_m3")
        depth_host = volume("dz_m")
        graupel_before = volume("qg_before")
        number_before = volume("ng_before_per_kg")
        graupel_volume_before = volume("qvolg_before_m3_per_kg")
        density = cp.asarray(density_host)
        depth = cp.asarray(depth_host)
        graupel = cp.asarray(graupel_before)
        number = cp.asarray(number_before)
        graupel_volume = cp.asarray(graupel_volume_before)
        graupelnc = cp.full((1, nx), cp.float32(2.0))
        graupelncv = cp.full((1, nx), cp.float32(-1.0))
        step = float(batch[0]["dt_s"])

        launch_graupel_sedimentation(
            density, graupel, number, graupel_volume, depth,
            graupelnc, graupelncv, step)
        cp.cuda.Stream.null.synchronize()

        np.testing.assert_allclose(
            cp.asnumpy(graupel), volume("qg_after"),
            rtol=4.0e-5, atol=3.0e-12)
        np.testing.assert_allclose(
            cp.asnumpy(number), volume("ng_after_per_kg"),
            rtol=3.0e-5, atol=3.0e-4)
        np.testing.assert_allclose(
            cp.asnumpy(graupel_volume),
            volume("qvolg_after_m3_per_kg"),
            rtol=4.0e-5, atol=3.0e-15)
        expected_surface = np.empty((1, nx), dtype=np.float32)
        for row in batch:
            if int(row["k"]) == 1:
                expected_surface[0, int(row["column"]) - 1] = float(
                    row["graupelncv_kg_m2"])
        np.testing.assert_allclose(
            cp.asnumpy(graupelncv), expected_surface,
            rtol=1.5e-5, atol=3.0e-10)
        np.testing.assert_allclose(
            cp.asnumpy(graupelnc), 2.0 + expected_surface,
            rtol=3.0e-7, atol=3.0e-7)

        graupel_before_column = np.sum(
            graupel_before * density_host * depth_host, axis=0)
        graupel_after_column = np.sum(
            cp.asnumpy(graupel) * density_host * depth_host, axis=0)
        np.testing.assert_allclose(
            graupel_after_column + cp.asnumpy(graupelncv),
            graupel_before_column, rtol=2.0e-6, atol=2.0e-7)
        assert bool(cp.all(cp.isfinite(graupel)))
        assert bool(cp.all(cp.isfinite(number)))
        assert bool(cp.all(cp.isfinite(graupel_volume)))
        assert bool(cp.all(graupel >= 0.0))
        assert bool(cp.all(number >= 0.0))
        assert bool(cp.all(graupel_volume >= 0.0))


def test_graupel_sedimentation_launcher_validation_and_empty_column():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_graupel_sedimentation

    volumes = [cp.ones((4, 2, 3), dtype=cp.float32) for _ in range(5)]
    volumes[1].fill(0.0)
    volumes[2].fill(0.0)
    volumes[3].fill(0.0)
    surfaces = [cp.zeros((2, 3), dtype=cp.float32) for _ in range(2)]
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_graupel_sedimentation(*volumes, *surfaces, step)

    with pytest.raises(ValueError, match="shape"):
        launch_graupel_sedimentation(
            *volumes, surfaces[0], cp.zeros((6,), dtype=cp.float32), 1.0)
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        too_shallow = [cp.ones((1, 2, 3), dtype=cp.float32)
                       for _ in range(5)]
        launch_graupel_sedimentation(*too_shallow, *surfaces, 1.0)

    launch_graupel_sedimentation(*volumes, *surfaces, 1.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(volumes[1], 0.0)
    cp.testing.assert_array_equal(volumes[2], 0.0)
    cp.testing.assert_array_equal(volumes[3], 0.0)
    cp.testing.assert_array_equal(surfaces[1], 0.0)


def test_hail_sedimentation_matches_official_wrf_columns():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_hail_sedimentation

    fixture = _ORACLE.with_name("hail-sedimentation.csv")
    with fixture.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 6 * 12

    by_case: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case"]), []).append(row)

    for batch in by_case.values():
        nz = max(int(row["k"]) for row in batch)
        nx = max(int(row["column"]) for row in batch)

        def volume(name):
            result = np.empty((nz, 1, nx), dtype=np.float32)
            for row in batch:
                result[int(row["k"]) - 1, 0,
                       int(row["column"]) - 1] = float(row[name])
            return result

        density_host = volume("rho_kg_m3")
        depth_host = volume("dz_m")
        hail_before = volume("qh_before")
        number_before = volume("nh_before_per_kg")
        hail_volume_before = volume("qvolh_before_m3_per_kg")
        density = cp.asarray(density_host)
        depth = cp.asarray(depth_host)
        hail = cp.asarray(hail_before)
        number = cp.asarray(number_before)
        hail_volume = cp.asarray(hail_volume_before)
        hailnc = cp.full((1, nx), cp.float32(2.0))
        hailncv = cp.full((1, nx), cp.float32(-1.0))
        step = float(batch[0]["dt_s"])

        launch_hail_sedimentation(
            density, hail, number, hail_volume, depth,
            hailnc, hailncv, step)
        cp.cuda.Stream.null.synchronize()

        np.testing.assert_allclose(
            cp.asnumpy(hail), volume("qh_after"),
            rtol=4.0e-5, atol=3.0e-12)
        np.testing.assert_allclose(
            cp.asnumpy(number), volume("nh_after_per_kg"),
            rtol=3.0e-5, atol=3.0e-4)
        np.testing.assert_allclose(
            cp.asnumpy(hail_volume), volume("qvolh_after_m3_per_kg"),
            rtol=4.0e-5, atol=3.0e-15)
        expected_surface = np.empty((1, nx), dtype=np.float32)
        for row in batch:
            if int(row["k"]) == 1:
                expected_surface[0, int(row["column"]) - 1] = float(
                    row["hailncv_kg_m2"])
        np.testing.assert_allclose(
            cp.asnumpy(hailncv), expected_surface,
            rtol=1.5e-5, atol=3.0e-10)
        np.testing.assert_allclose(
            cp.asnumpy(hailnc), 2.0 + expected_surface,
            rtol=3.0e-7, atol=3.0e-7)

        hail_before_column = np.sum(
            hail_before * density_host * depth_host, axis=0)
        hail_after_column = np.sum(
            cp.asnumpy(hail) * density_host * depth_host, axis=0)
        np.testing.assert_allclose(
            hail_after_column + cp.asnumpy(hailncv), hail_before_column,
            rtol=2.0e-6, atol=2.0e-7)
        assert bool(cp.all(cp.isfinite(hail)))
        assert bool(cp.all(cp.isfinite(number)))
        assert bool(cp.all(cp.isfinite(hail_volume)))
        assert bool(cp.all(hail >= 0.0))
        assert bool(cp.all(number >= 0.0))
        assert bool(cp.all(hail_volume >= 0.0))


def test_hail_sedimentation_launcher_validation_and_empty_column():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_hail_sedimentation

    volumes = [cp.ones((4, 2, 3), dtype=cp.float32) for _ in range(5)]
    volumes[1].fill(0.0)
    volumes[2].fill(0.0)
    volumes[3].fill(0.0)
    surfaces = [cp.zeros((2, 3), dtype=cp.float32) for _ in range(2)]
    for step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            launch_hail_sedimentation(*volumes, *surfaces, step)

    with pytest.raises(ValueError, match="shape"):
        launch_hail_sedimentation(
            *volumes, surfaces[0], cp.zeros((6,), dtype=cp.float32), 1.0)
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        too_shallow = [cp.ones((1, 2, 3), dtype=cp.float32)
                       for _ in range(5)]
        launch_hail_sedimentation(*too_shallow, *surfaces, 1.0)

    launch_hail_sedimentation(*volumes, *surfaces, 1.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(volumes[1], 0.0)
    cp.testing.assert_array_equal(volumes[2], 0.0)
    cp.testing.assert_array_equal(volumes[3], 0.0)
    cp.testing.assert_array_equal(surfaces[1], 0.0)


def test_mp18_domain_state_allocates_exact_runtime_package_and_binding():
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.nssl2_contract import DEFAULT_RESTART_FIELDS
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.state import DomainState

    cfg = RunConfig(
        nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
        dt=1.0, run_seconds=10.0, moist=True, mp_physics=18)
    state = DomainState(cfg)
    for name in DEFAULT_RESTART_FIELDS:
        value = getattr(state, name)
        assert value.shape == (4, 6, 8)
        assert value.dtype == cp.float32
    for name in DEFAULT_RESTART_FIELDS[3:]:
        assert getattr(state, name + "0").shape == (4, 6, 8)
    cp.testing.assert_array_equal(state.effc, cp.float32(2.51))
    cp.testing.assert_array_equal(state.effi, cp.float32(10.01))
    cp.testing.assert_array_equal(state.effs, cp.float32(25.0))
    cp.testing.assert_array_equal(state.qnn, cp.float32(408163264.0))
    driver = initialize_physics(state, cfg)
    binding = driver.nssl2_binding
    assert binding.state is state
    assert binding.workspace.state is state._scratch["nssl2_driver_state"]
    assert binding.workspace.category_surface_export is \
        state._scratch["nssl2_driver_surface_export"]
    assert binding.workspace.ignored_accumulator is \
        state._scratch["nssl2_driver_ignored_accumulator"]
    binding.validate(state, cfg.dt)


def test_mp18_synthetic_gpu_step_executes_real_selector():
    """Exercise selector -> fused GS/NUCOND -> acceptance on the GPU."""
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(
        nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0,
        dt=0.1, run_seconds=0.1, moist=True, mp_physics=18,
        h_sca_adv_order=2, time_step_sound=2)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.001 * np.asarray(z),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.005 * np.exp(-np.asarray(z) / 3000.0))
    state.qh[2:4, 3:5, 3:5] = cp.float32(1.0e-6)
    state.qvolh[2:4, 3:5, 3:5] = cp.float32(1.0e-6 / 900.0)
    driver = initialize_physics(state, cfg)
    step(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert state.elapsed_seconds == pytest.approx(cfg.dt)
    assert driver.microphysics_updates == 1
    for name in (
            "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
            "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg",
            "qvolh", "thp", "h_diabatic", "effc", "effi", "effs"):
        assert bool(cp.isfinite(getattr(state, name)).all()), name


def test_fused_gs_enforces_nonnegative_hydromass_writeback():
    """Reproduce real d01 residues and preserve positive trace mass."""
    import cupy as cp

    from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
    from gpuwm.core.nssl2_fused_gs import launch_fused_gs

    shape = (2, 1, 2)
    state = cp.zeros((16, *shape), dtype=cp.float32)
    workspace = NSSL2DriverWorkspace(
        state,
        cp.zeros((5, *shape[1:]), dtype=cp.float32),
        shape,
    )
    workspace.field("qv").fill(cp.float32(1.0e-3))
    workspace.field("qs").ravel()[:3] = cp.asarray(
        [-1.0842021724855044e-19, 5.0e-17, -1.0e-10],
        dtype=cp.float32,
    )
    workspace.field("qg").ravel()[:3] = cp.asarray(
        [-1.2815462124664279e-13, 5.0e-13, -1.0e-9],
        dtype=cp.float32,
    )

    full_theta = cp.full(shape, 300.0, dtype=cp.float32)
    air_density = cp.ones(shape, dtype=cp.float32)
    pressure_pa = cp.full(shape, 80000.0, dtype=cp.float32)
    exner = cp.ones(shape, dtype=cp.float32)
    vertical_velocity = cp.zeros(
        (shape[0] + 1, *shape[1:]), dtype=cp.float32)
    temperature_k = cp.empty(shape, dtype=cp.float32)
    primary_ice_target_m3 = cp.empty(shape, dtype=cp.float32)
    dz = cp.full(shape, 1000.0, dtype=cp.float32)

    launch_fused_gs(
        workspace,
        full_theta,
        air_density,
        pressure_pa,
        exner,
        vertical_velocity,
        temperature_k,
        primary_ice_target_m3,
        dz,
        1.0,
    )
    cp.cuda.Stream.null.synchronize()

    actual_snow = cp.asnumpy(workspace.field("qs")).ravel()
    assert actual_snow[0] == np.float32(0.0)
    assert actual_snow[1] == np.float32(5.0e-17)
    assert actual_snow[2] == np.float32(0.0)

    actual_graupel = cp.asnumpy(workspace.field("qg")).ravel()
    assert actual_graupel[0] == np.float32(0.0)
    assert actual_graupel[1] == np.float32(5.0e-13)
    assert actual_graupel[2] == np.float32(0.0)
