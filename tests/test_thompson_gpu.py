"""GPU gates for independently admitted classic Thompson slices."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.gpu
_ORACLE = Path(__file__).parents[1] / "gpuwm" / "data" / "thompson" / "oracle"


@pytest.mark.parametrize(
    "scenario",
    ("warm", "mixed", "ice", "condense", "rain-sed", "ice-sed",
     "cloud-sed", "cloud-condense-sed", "cloud-condense-nofall",
     "cloud-rain-condense-sed", "cloud-rain-condense-nofall",
     "condense-fall-attempt",
     "snow-sed", "graupel-sed", "warm-auto",
     "rain-self", "warm-accrete", "ice-auto", "rain-evap", "snow-subl",
     "graupel-subl", "ice-dep", "ice-nuc", "snow-dep",
     "graupel-dep", "snow-melt", "graupel-melt"))
def test_effective_radius_matches_direct_wrf_mp_gt_driver(scenario):
    import cupy as cp

    from gpuwm.core.thompson import launch_effective_radius

    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["phase"] == "after"]
    assert len(rows) == 24

    def device(name):
        return cp.asarray(
            np.asarray([float(row[name]) for row in rows], dtype=np.float32))

    temperature = device("temp_k")
    pressure = device("p_pa")
    qv = device("qv")
    qc = device("qc")
    qi = device("qi")
    ni = device("ni_per_kg")
    qs = device("qs")
    effc = cp.empty_like(temperature)
    effi = cp.empty_like(temperature)
    effs = cp.empty_like(temperature)

    launch_effective_radius(
        temperature, pressure, qv, qc, qi, ni, qs, effc, effi, effs)
    cp.cuda.Stream.null.synchronize()

    for output, name in ((effc, "effc_m"), (effi, "effi_m"),
                         (effs, "effs_m")):
        # The oracle stores WRF mp_gt_driver's radii in METRES.  The state
        # contract is microns, so the kernel applies WRF's own driver-side
        # conversion re*1.E6 (module_ra_rrtmg_lw.F:12184,12203,12242) as
        # its final multiply; the expectation applies the identical FP32
        # conversion to the oracle metres.  atol scales by the same 1e6
        # (units), rtol is unchanged -- the gate is not widened.
        expected = np.asarray(
            [float(row[name]) for row in rows],
            dtype=np.float32) * np.float32(1.0e6)
        # Snow radii use two non-integer powers.  CUDA libdevice and GNU
        # libm differ by up to 3.25e-6 relative on these vectors while the
        # cloud/ice algebra is much tighter; this bound is measured and
        # explicit rather than an exact-bit claim across math libraries.
        np.testing.assert_allclose(
            cp.asnumpy(output), expected, rtol=5.0e-6, atol=2.0e-5)


def test_effective_radius_launcher_rejects_shape_and_dtype():
    import cupy as cp

    from gpuwm.core.thompson import launch_effective_radius

    good = [cp.zeros((3, 2), dtype=cp.float32) for _ in range(10)]
    bad_shape = list(good)
    bad_shape[-1] = cp.zeros((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_effective_radius(*bad_shape)

    bad_dtype = list(good)
    bad_dtype[2] = cp.zeros((3, 2), dtype=cp.float64)
    with pytest.raises(TypeError, match="float32"):
        launch_effective_radius(*bad_dtype)


def test_warm_saturation_adjust_matches_isolated_wrf_process():
    import cupy as cp

    from gpuwm.core.thompson import launch_warm_saturation_adjust

    with (_ORACLE / "condense-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    temperature = cp.asarray(host(before, "temp_k"))
    pressure = cp.asarray(host(before, "p_pa"))
    qv = cp.asarray(host(before, "qv"))
    qc = cp.asarray(host(before, "qc"))
    launch_warm_saturation_adjust(temperature, pressure, qv, qc)
    cp.cuda.Stream.null.synchronize()

    actual_t = cp.asnumpy(temperature)
    actual_qv = cp.asnumpy(qv)
    actual_qc = cp.asnumpy(qc)
    np.testing.assert_allclose(
        actual_t, host(after, "temp_k"), rtol=2.0e-6, atol=2.0e-5)
    np.testing.assert_allclose(
        actual_qv, host(after, "qv"), rtol=2.0e-6, atol=2.0e-9)
    np.testing.assert_allclose(
        actual_qc, host(after, "qc"), rtol=3.0e-6, atol=2.0e-9)
    np.testing.assert_allclose(
        actual_qv + actual_qc,
        host(before, "qv") + host(before, "qc"), rtol=0.0, atol=2.0e-9)
    assert np.count_nonzero(actual_qc) == 5


def test_rain_sedimentation_matches_isolated_wrf_column_and_budget():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_sedimentation

    with (_ORACLE / "rain-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)

    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(qr[:, 0, 0]), host(after, "qr"),
        rtol=4.0e-6, atol=2.0e-11)
    np.testing.assert_allclose(
        cp.asnumpy(nr[:, 0, 0]), host(after, "nr_per_kg"),
        rtol=5.0e-6, atol=2.0)
    with (_ORACLE / "rain-sed-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    expected_rain = np.float32(float(surface["rainncv_mm"]))
    np.testing.assert_allclose(
        cp.asnumpy(rainncv), expected_rain, rtol=5.0e-6, atol=2.0e-9)
    np.testing.assert_array_equal(cp.asnumpy(rainnc), cp.asnumpy(rainncv))

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    water_before = np.sum(
        rho * host(before, "qr") * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * cp.asnumpy(qr[:, 0, 0]) * host(before, "dz_m"),
        dtype=np.float64)
    assert water_before - water_after == pytest.approx(
        float(cp.asnumpy(rainncv)[0, 0]), abs=2.0e-7)


def test_rain_sedimentation_launcher_rejects_bad_geometry_and_dt():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_sedimentation

    fields = [cp.zeros((3, 2, 2), dtype=cp.float32) for _ in range(6)]
    surface = [cp.zeros((2, 2), dtype=cp.float32) for _ in range(2)]
    with pytest.raises(ValueError, match="positive"):
        launch_rain_sedimentation(*fields, *surface, 0.0)
    bad_surface = cp.zeros((4,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_rain_sedimentation(*fields, bad_surface, surface[1], 1.0)


def test_rain_sedimentation_generic_depth_dry_column_is_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_sedimentation

    shape = (80, 2, 3)
    qr = cp.zeros(shape, dtype=cp.float32)
    nr = cp.zeros_like(qr)
    temperature = cp.full(shape, 280.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.005, dtype=cp.float32)
    dz = cp.full(shape, 500.0, dtype=cp.float32)
    rainnc = cp.zeros(shape[1:], dtype=cp.float32)
    rainncv = cp.full(shape[1:], np.nan, dtype=cp.float32)

    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0)
    cp.cuda.Stream.null.synchronize()
    assert not bool(cp.any(qr))
    assert not bool(cp.any(nr))
    assert not bool(cp.any(rainnc))
    assert not bool(cp.any(rainncv))


def test_ice_sedimentation_matches_isolated_wrf_column_and_budget():
    import cupy as cp

    from gpuwm.core.thompson import launch_ice_sedimentation

    with (_ORACLE / "ice-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)

    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(qi[:, 0, 0]), host(after, "qi"),
        rtol=4.0e-6, atol=2.0e-13)
    np.testing.assert_allclose(
        cp.asnumpy(ni[:, 0, 0]), host(after, "ni_per_kg"),
        rtol=5.0e-6, atol=1.0)
    with (_ORACLE / "ice-sed-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    expected = np.float32(float(surface["rainncv_mm"]))
    for field in (rainnc, rainncv, snownc, snowncv):
        np.testing.assert_allclose(
            cp.asnumpy(field), expected, rtol=5.0e-6, atol=2.0e-12)

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    water_before = np.sum(
        rho * host(before, "qi") * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * cp.asnumpy(qi[:, 0, 0]) * host(before, "dz_m"),
        dtype=np.float64)
    assert water_before - water_after == pytest.approx(
        float(cp.asnumpy(rainncv)[0, 0]), abs=2.0e-10)


def test_ice_sedimentation_generic_depth_dry_column_is_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_ice_sedimentation

    shape = (80, 2, 3)
    qi = cp.zeros(shape, dtype=cp.float32)
    ni = cp.zeros_like(qi)
    temperature = cp.full(shape, 260.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.001, dtype=cp.float32)
    dz = cp.full(shape, 500.0, dtype=cp.float32)
    surfaces = [cp.zeros(shape[1:], dtype=cp.float32) for _ in range(4)]
    surfaces[1].fill(np.nan)
    surfaces[3].fill(np.nan)

    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()
    assert not bool(cp.any(qi))
    assert not bool(cp.any(ni))
    assert all(not bool(cp.any(field)) for field in surfaces)


def test_cloud_sedimentation_matches_isolated_wrf_column_and_budget():
    import cupy as cp

    from gpuwm.core.thompson import launch_cloud_sedimentation

    with (_ORACLE / "cloud-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qc_held = qc.copy()
    vertical_velocity = cp.zeros_like(qc)
    dz = volume("dz_m")

    launch_cloud_sedimentation(
        qc, temperature, pressure, qv, vertical_velocity, dz, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(qc[:, 0, 0]), host(after, "qc"),
        rtol=4.0e-6, atol=2.0e-13)

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    launch_cloud_sedimentation(
        qc_held, temperature, pressure, qv, vertical_velocity, dz, 10.0,
        reference_density=cp.asarray(rho[:, None, None]))
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(qc_held[:, 0, 0]), host(after, "qc"),
        rtol=4.0e-6, atol=2.0e-13)
    water_before = np.sum(
        rho * host(before, "qc") * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * cp.asnumpy(qc[:, 0, 0]) * host(before, "dz_m"),
        dtype=np.float64)
    wrf_after = np.sum(
        rho * host(after, "qc") * host(before, "dz_m"),
        dtype=np.float64)
    # Preserve the official implementation's measured FP32 roundoff budget
    # rather than asserting unrealizable mathematical exactness.
    assert water_before - water_after == pytest.approx(
        water_before - wrf_after, abs=2.0e-13)


def test_cloud_sedimentation_generic_depth_dry_column_is_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_cloud_sedimentation

    shape = (80, 2, 3)
    qc = cp.zeros(shape, dtype=cp.float32)
    temperature = cp.full(shape, 280.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.005, dtype=cp.float32)
    vertical_velocity = cp.zeros(shape, dtype=cp.float32)
    dz = cp.full(shape, 500.0, dtype=cp.float32)

    launch_cloud_sedimentation(
        qc, temperature, pressure, qv, vertical_velocity, dz, 10.0)
    cp.cuda.Stream.null.synchronize()
    assert not bool(cp.any(qc))


def test_cloud_adjust_plus_held_density_fallout_matches_wrf_column():
    """Pin WRF's distinct pre/post-adjustment cloud density semantics."""
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cloud_saturation_adjust,
        launch_cloud_sedimentation,
    )

    with (_ORACLE / "cloud-condense-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before, after = rows[:24], rows[24:]
    with (_ORACLE / "cloud-condense-nofall-column.csv").open(
            newline="", encoding="ascii") as stream:
        nofall_rows = list(csv.DictReader(stream))
    nofall_after = nofall_rows[24:]

    def host(rows_, name):
        return np.asarray(
            [float(row[name]) for row in rows_], dtype=np.float32)

    def volume(rows_, name):
        return cp.asarray(host(rows_, name)[:, None, None])

    # Capture WRF's pre-adjustment density from the common incoming state.
    incoming_temperature = volume(before, "temp_k")
    pressure = volume(before, "p_pa")
    incoming_qv = volume(before, "qv")
    incoming_qc = volume(before, "qc")
    reference_density = cp.empty_like(incoming_qc)

    launch_cloud_saturation_adjust(
        incoming_temperature, pressure, incoming_qv, incoming_qc,
        reference_density=reference_density)
    # Start fallout from the matched official-WRF no-fallout member.  This
    # removes the small independent CUDA/gfortran saturation-rounding delta
    # and gates only the sedimentation increment.
    temperature = volume(nofall_after, "temp_k")
    qv = volume(nofall_after, "qv")
    qc = volume(nofall_after, "qc")
    vertical_velocity = volume(before, "w_m_s")
    dz = volume(before, "dz_m")
    current_density_qc = qc.copy()
    launch_cloud_sedimentation(
        qc, temperature, pressure, qv, vertical_velocity, dz, 10.0,
        reference_density=reference_density)
    # This deliberately wrong composition is the old adapter candidate:
    # rebuilding cloud mass from post-adjustment density must remain
    # distinguishable from the official-WRF oracle.
    launch_cloud_sedimentation(
        current_density_qc, temperature, pressure, qv,
        vertical_velocity, dz, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        host(nofall_after, "temp_k"), host(after, "temp_k"))
    np.testing.assert_array_equal(
        host(nofall_after, "qv"), host(after, "qv"))
    np.testing.assert_allclose(
        cp.asnumpy(qc[:, 0, 0]), host(after, "qc"),
        rtol=5.0e-6, atol=5.0e-13)
    held_error = np.sum(np.abs(
        cp.asnumpy(qc[:, 0, 0]) - host(after, "qc")), dtype=np.float64)
    current_error = np.sum(np.abs(
        cp.asnumpy(current_density_qc[:, 0, 0]) - host(after, "qc")),
        dtype=np.float64)
    assert held_error < current_error


def test_hydrometeor_column_mask_reports_post_source_activity():
    import cupy as cp

    from gpuwm.core.thompson import launch_hydrometeor_column_mask

    mixing_ratio = cp.zeros((4, 2, 3), dtype=cp.float32)
    mixing_ratio[1, 0, 2] = cp.float32(2.0e-12)
    mixing_ratio[3, 1, 0] = cp.float32(1.0e-12)
    active_columns = cp.full((2, 3), cp.nan, dtype=cp.float32)
    launch_hydrometeor_column_mask(mixing_ratio, active_columns)
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(active_columns),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32))


def test_rain_active_cloud_fallout_refreshes_rhof_like_wrf():
    """Pin WRF's column-held ANY(L_qr) fall-speed-density branch."""
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cloud_saturation_adjust,
        launch_cloud_sedimentation,
    )

    def read(scenario):
        with (_ORACLE / f"{scenario}-column.csv").open(
                newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream))
        return rows[:24], rows[24:]

    before, after = read("cloud-rain-condense-sed")
    _, nofall_after = read("cloud-rain-condense-nofall")

    def host(rows_, name):
        return np.asarray(
            [float(row[name]) for row in rows_], dtype=np.float32)

    def volume(rows_, name):
        return cp.asarray(host(rows_, name)[:, None, None])

    incoming_temperature = volume(before, "temp_k")
    pressure = volume(before, "p_pa")
    incoming_qv = volume(before, "qv")
    incoming_qc = volume(before, "qc")
    reference_density = cp.empty_like(incoming_qc)
    launch_cloud_saturation_adjust(
        incoming_temperature, pressure, incoming_qv, incoming_qc,
        reference_density=reference_density)

    temperature = volume(nofall_after, "temp_k")
    qv = volume(nofall_after, "qv")
    qc = volume(nofall_after, "qc")
    held_rhof_qc = qc.copy()
    vertical_velocity = volume(before, "w_m_s")
    dz = volume(before, "dz_m")
    rain_active_columns = cp.ones((1, 1), dtype=cp.float32)
    launch_cloud_sedimentation(
        qc, temperature, pressure, qv, vertical_velocity, dz, 10.0,
        reference_density=reference_density,
        rain_active_columns=rain_active_columns)
    # Omitting the held rain activity reproduces WRF's no-rain RHOF branch
    # and must be measurably worse for this official rain-active pair.
    launch_cloud_sedimentation(
        held_rhof_qc, temperature, pressure, qv,
        vertical_velocity, dz, 10.0,
        reference_density=reference_density)
    cp.cuda.Stream.null.synchronize()

    expected = host(after, "qc")
    np.testing.assert_allclose(
        cp.asnumpy(qc[:, 0, 0]), expected,
        rtol=5.0e-6, atol=5.0e-13)
    refreshed_error = np.sum(np.abs(
        cp.asnumpy(qc[:, 0, 0]) - expected), dtype=np.float64)
    held_error = np.sum(np.abs(
        cp.asnumpy(held_rhof_qc[:, 0, 0]) - expected), dtype=np.float64)
    assert refreshed_error < held_error


def test_fresh_condensation_obeys_wrf_held_cloud_column_guard():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cloud_saturation_adjust,
        launch_cloud_sedimentation,
        launch_hydrometeor_column_mask,
    )

    def read(scenario):
        with (_ORACLE / f"{scenario}-column.csv").open(
                newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream))
        return rows[:24], rows[24:]

    before, after = read("condense-fall-attempt")
    _, nofall_after = read("condense")

    def host(rows_, name):
        return np.asarray(
            [float(row[name]) for row in rows_], dtype=np.float32)

    def volume(rows_, name):
        return cp.asarray(host(rows_, name)[:, None, None])

    incoming_temperature = volume(before, "temp_k")
    pressure = volume(before, "p_pa")
    incoming_qv = volume(before, "qv")
    incoming_qc = volume(before, "qc")
    incoming_qr = volume(before, "qr")
    reference_density = cp.empty_like(incoming_qc)
    launch_cloud_saturation_adjust(
        incoming_temperature, pressure, incoming_qv, incoming_qc,
        reference_density=reference_density)

    qc = volume(nofall_after, "qc")
    unguarded_qc = qc.copy()
    temperature = volume(nofall_after, "temp_k")
    qv = volume(nofall_after, "qv")
    vertical_velocity = volume(before, "w_m_s")
    dz = volume(before, "dz_m")
    rain_active_columns = cp.empty((1, 1), dtype=cp.float32)
    cloud_active_columns = cp.empty((1, 1), dtype=cp.float32)
    launch_hydrometeor_column_mask(incoming_qr, rain_active_columns)
    launch_hydrometeor_column_mask(
        volume(before, "qc"), cloud_active_columns)
    launch_cloud_sedimentation(
        qc, temperature, pressure, qv, vertical_velocity, dz, 10.0,
        reference_density=reference_density,
        rain_active_columns=rain_active_columns,
        cloud_active_columns=cloud_active_columns)
    launch_cloud_sedimentation(
        unguarded_qc, temperature, pressure, qv,
        vertical_velocity, dz, 10.0,
        reference_density=reference_density)
    cp.cuda.Stream.null.synchronize()

    expected = host(after, "qc")
    np.testing.assert_array_equal(
        expected, host(nofall_after, "qc"))
    np.testing.assert_array_equal(cp.asnumpy(qc[:, 0, 0]), expected)
    assert not np.array_equal(cp.asnumpy(unguarded_qc[:, 0, 0]), expected)


def test_snow_sedimentation_matches_isolated_wrf_column_and_budget():
    import cupy as cp

    from gpuwm.core.thompson import launch_snow_sedimentation

    with (_ORACLE / "snow-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qs = volume("qs")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(qs[:, 0, 0]), host(after, "qs"))
    with (_ORACLE / "snow-sed-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "snownc_mm", "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=5.0e-7, atol=1.0e-12)

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    water_before = np.sum(
        rho * host(before, "qs") * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * cp.asnumpy(qs[:, 0, 0]) * host(before, "dz_m"),
        dtype=np.float64)
    assert water_before - water_after == pytest.approx(
        float(cp.asnumpy(surfaces[1])[0, 0]), abs=1.0e-8)


def test_snow_sedimentation_generic_depth_dry_column_is_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_snow_sedimentation

    shape = (80, 2, 3)
    qs = cp.zeros(shape, dtype=cp.float32)
    temperature = cp.full(shape, 260.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.001, dtype=cp.float32)
    dz = cp.full(shape, 500.0, dtype=cp.float32)
    surfaces = [cp.zeros(shape[1:], dtype=cp.float32) for _ in range(4)]
    surfaces[1].fill(np.nan)
    surfaces[3].fill(np.nan)

    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()
    assert not bool(cp.any(qs))
    assert all(not bool(cp.any(field)) for field in surfaces)


def test_snow_rain_velocity_blend_requires_same_call_snow_melt_marker():
    import cupy as cp

    from gpuwm.core.thompson import launch_snow_sedimentation

    shape = (4, 1, 1)
    temperature = cp.full(shape, 275.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.002, dtype=cp.float32)
    dz = cp.full(shape, 100.0, dtype=cp.float32)
    qr = cp.full(shape, 2.0e-3, dtype=cp.float32)
    nr = cp.full(shape, 5.0e4, dtype=cp.float32)
    initial_qs = cp.asarray(
        [1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4],
        dtype=cp.float32)[:, None, None]

    def run(marker):
        qs = initial_qs.copy()
        surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]
        kwargs = {}
        if marker is not None:
            kwargs = {
                "snow_melt_marker": marker,
                "melt_rain_qr": qr,
                "melt_rain_nr": nr,
            }
        launch_snow_sedimentation(
            qs, temperature, pressure, qv, dz, *surfaces, 60.0, **kwargs)
        return qs, surfaces

    baseline_qs, baseline_surfaces = run(None)
    held_no_melt_qs, held_no_melt_surfaces = run(cp.zeros_like(initial_qs))
    held_melt_qs, held_melt_surfaces = run(cp.ones_like(initial_qs))
    cp.cuda.Stream.null.synchronize()

    cp.testing.assert_array_equal(held_no_melt_qs, baseline_qs)
    for actual, expected in zip(
            held_no_melt_surfaces, baseline_surfaces, strict=True):
        cp.testing.assert_array_equal(actual, expected)
    assert not bool(cp.all(held_melt_qs == baseline_qs))
    assert not bool(cp.all(
        held_melt_surfaces[1] == baseline_surfaces[1]))


def test_graupel_sedimentation_matches_isolated_wrf_column_and_budget():
    import cupy as cp

    from gpuwm.core.thompson import launch_graupel_sedimentation

    with (_ORACLE / "graupel-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qg = volume("qg")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(qg[:, 0, 0]), host(after, "qg"),
        rtol=2.0e-7, atol=2.0e-11)
    with (_ORACLE / "graupel-sed-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "graupelnc_mm", "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=5.0e-7, atol=1.0e-10)

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    water_before = np.sum(
        rho * host(before, "qg") * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * cp.asnumpy(qg[:, 0, 0]) * host(before, "dz_m"),
        dtype=np.float64)
    assert water_before - water_after == pytest.approx(
        float(cp.asnumpy(surfaces[1])[0, 0]), abs=5.0e-8)


def test_graupel_sedimentation_generic_depth_dry_column_is_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_graupel_sedimentation

    shape = (80, 2, 3)
    qg = cp.zeros(shape, dtype=cp.float32)
    temperature = cp.full(shape, 260.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.001, dtype=cp.float32)
    dz = cp.full(shape, 500.0, dtype=cp.float32)
    surfaces = [cp.zeros(shape[1:], dtype=cp.float32) for _ in range(4)]
    surfaces[1].fill(np.nan)
    surfaces[3].fill(np.nan)

    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()
    assert not bool(cp.any(qg))
    assert all(not bool(cp.any(field)) for field in surfaces)


def test_graupel_fallout_column_mask_matches_classic_wrf_held_any_rule():
    import cupy as cp

    from gpuwm.core.thompson import launch_graupel_fallout_column_mask

    shape = (3, 2, 2)
    entry_active = cp.zeros(shape, dtype=cp.float32)
    qg = cp.zeros(shape, dtype=cp.float32)
    active_columns = cp.empty(shape[1:], dtype=cp.float32)

    # Entry-active and still present: the column sediments.
    entry_active[0, 0, 0] = cp.float32(1.0)
    qg[0, 0, 0] = cp.float32(2.0e-8)
    # Its only entry-active level was consumed.  New graupel at another
    # level cannot reactivate classic WRF's held L_qg column guard.
    entry_active[0, 0, 1] = cp.float32(1.0)
    qg[1, 0, 1] = cp.float32(2.0e-8)
    # Newly created graupel alone is entry-inactive.
    qg[1, 1, 0] = cp.float32(2.0e-8)
    # One surviving entry-active level enables every level in the column.
    entry_active[2, 1, 1] = cp.float32(1.0)
    qg[2, 1, 1] = cp.float32(2.0e-8)
    qg[0, 1, 1] = cp.float32(3.0e-8)

    launch_graupel_fallout_column_mask(
        entry_active, qg, active_columns)
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(active_columns),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_graupel_sedimentation_column_mask_preserves_inactive_column():
    import cupy as cp

    from gpuwm.core.thompson import launch_graupel_sedimentation

    with (_ORACLE / "graupel-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray(
            [float(row[name]) for row in rows_], dtype=np.float32)

    def two_columns(name):
        values = host(before, name)[:, None, None]
        return cp.asarray(np.broadcast_to(values, (24, 1, 2)))

    temperature = two_columns("temp_k")
    pressure = two_columns("p_pa")
    qv = two_columns("qv")
    qg = two_columns("qg").copy()
    qg_before = qg.copy()
    dz = two_columns("dz_m")
    reference_density = cp.float32(0.622) * pressure / (
        cp.float32(287.04) * temperature * (qv + cp.float32(0.622)))
    surfaces = [cp.zeros((1, 2), dtype=cp.float32) for _ in range(4)]
    active_columns = cp.asarray([[1.0, 0.0]], dtype=cp.float32)

    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz, *surfaces, 10.0,
        reference_density=reference_density,
        active_columns=active_columns)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(qg[:, 0, 0]), host(after, "qg"),
        rtol=2.0e-7, atol=2.0e-11)
    np.testing.assert_array_equal(
        cp.asnumpy(qg[:, 0, 1]), cp.asnumpy(qg_before[:, 0, 1]))
    with (_ORACLE / "graupel-sed-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "graupelnc_mm", "graupelncv_mm")):
        expected = np.float32(float(surface[name]))
        np.testing.assert_allclose(
            cp.asnumpy(field[:, 0]), expected,
            rtol=5.0e-7, atol=1.0e-10)
        np.testing.assert_array_equal(
            cp.asnumpy(field[:, 1]), np.zeros(1, dtype=np.float32))


def test_classic_graupel_number_shadow_matches_wrf_refl_without_feedback():
    """Output-due private ng follows WRF fallout but cannot change qg."""
    import cupy as cp

    from gpuwm.core.refl import launch_refl10cm_thompson
    from gpuwm.core.thompson import (
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
        launch_graupel_sedimentation,
    )

    with (_ORACLE / "graupel-sed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before, after = rows[:24], rows[24:]

    def host(rows_, name):
        return np.asarray(
            [float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    dz = volume("dz_m")
    qg_plain = volume("qg")
    qg_shadowed = qg_plain.copy()
    reference_density = cp.float32(0.622) * pressure / (
        cp.float32(287.04) * temperature * (qv + cp.float32(0.622)))
    active = cp.ones((1, 1), dtype=cp.float32)
    surfaces_plain = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]
    surfaces_shadow = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]
    shadow = cp.empty_like(qg_shadowed)

    launch_graupel_sedimentation(
        qg_plain, temperature, pressure, qv, dz, *surfaces_plain, 10.0,
        reference_density=reference_density, active_columns=active)
    launch_classic_graupel_number_init(
        qg_shadowed, temperature, pressure, qv, shadow)
    launch_graupel_sedimentation(
        qg_shadowed, temperature, pressure, qv, dz, *surfaces_shadow, 10.0,
        reference_density=reference_density, active_columns=active,
        graupel_number_shadow=shadow)
    launch_classic_graupel_number_finalize(
        qg_shadowed, temperature, pressure, qv, shadow)

    zeros = cp.zeros_like(qg_shadowed)
    refl = cp.empty_like(qg_shadowed)
    launch_refl10cm_thompson(
        qv, zeros, zeros, zeros, qg_shadowed, shadow,
        temperature, pressure, refl)
    cp.cuda.Stream.null.synchronize()

    # The diagnostic-only shadow must not perturb the trajectory or surface
    # budget, including FP32 rounding.
    np.testing.assert_array_equal(cp.asnumpy(qg_shadowed), cp.asnumpy(qg_plain))
    for shadowed, plain in zip(
            surfaces_shadow, surfaces_plain, strict=True):
        np.testing.assert_array_equal(cp.asnumpy(shadowed), cp.asnumpy(plain))

    expected = host(after, "refl_dbz")
    got = cp.asnumpy(refl[:, 0, 0])
    assert np.isfinite(got).all() and got.min() >= -35.0
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2.0e-2)


def test_warm_autoconversion_plus_rain_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_rain_sedimentation,
        launch_warm_autoconversion,
    )

    with (_ORACLE / "warm-auto-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)

    launch_warm_autoconversion(
        qc, qr, nr, temperature, pressure, qv, 10.0)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(qc[:, 0, 0]), host(after, "qc"))
    np.testing.assert_allclose(
        cp.asnumpy(qr[:, 0, 0]), host(after, "qr"),
        rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(
        cp.asnumpy(nr[:, 0, 0]), host(after, "nr_per_kg"),
        rtol=2.0e-6, atol=1.0e-4)
    with (_ORACLE / "warm-auto-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    expected_rain = np.float32(float(surface["rainncv_mm"]))
    np.testing.assert_allclose(
        cp.asnumpy(rainncv), expected_rain, rtol=3.0e-6, atol=1.0e-13)
    np.testing.assert_array_equal(cp.asnumpy(rainnc), cp.asnumpy(rainncv))

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    water_before = np.sum(
        rho * (host(before, "qc") + host(before, "qr"))
        * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * (cp.asnumpy(qc[:, 0, 0]) + cp.asnumpy(qr[:, 0, 0]))
        * host(before, "dz_m"), dtype=np.float64)
    wrf_after = np.sum(
        rho * (host(after, "qc") + host(after, "qr"))
        * host(before, "dz_m"), dtype=np.float64)
    # The tiny rain signal is summed beside O(1e-3) cloud water; compare
    # CUDA with WRF's measured FP32 budget instead of exact arithmetic.
    assert water_before - water_after == pytest.approx(
        water_before - wrf_after, abs=1.0e-9)


def test_warm_autoconversion_subthreshold_cloud_is_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_warm_autoconversion

    shape = (80, 2, 3)
    qc = cp.full(shape, 1.0e-7, dtype=cp.float32)
    qr = cp.zeros_like(qc)
    nr = cp.zeros_like(qc)
    temperature = cp.full(shape, 285.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.005, dtype=cp.float32)
    qc_before = qc.copy()

    launch_warm_autoconversion(
        qc, qr, nr, temperature, pressure, qv, 10.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(qc, qc_before)
    assert not bool(cp.any(qr))
    assert not bool(cp.any(nr))


def test_rain_self_collection_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_rain_sedimentation,
        launch_rain_self_collection,
    )

    with (_ORACLE / "rain-self-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)

    launch_rain_self_collection(qr, nr, temperature, pressure, qv, 10.0)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(qr[:, 0, 0]), host(after, "qr"))
    np.testing.assert_allclose(
        cp.asnumpy(nr[:, 0, 0]), host(after, "nr_per_kg"),
        rtol=5.0e-7, atol=2.0e-2)
    with (_ORACLE / "rain-self-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    expected_rain = np.float32(float(surface["rainncv_mm"]))
    np.testing.assert_array_equal(cp.asnumpy(rainncv), expected_rain)
    np.testing.assert_array_equal(cp.asnumpy(rainnc), cp.asnumpy(rainncv))


def test_rain_self_collection_below_d0r_is_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_self_collection

    shape = (4, 2, 3)
    qr = cp.full(shape, 3.0e-4, dtype=cp.float32)
    diameter = np.float32(45.0e-6)
    nr_value = (np.float32(3.0e-4)
                * (np.float32(3.672) / diameter) ** np.float32(3.0)
                / (np.float32(np.pi) * np.float32(1000.0)))
    nr = cp.full(shape, nr_value, dtype=cp.float32)
    temperature = cp.full(shape, 285.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.005, dtype=cp.float32)
    nr_before = nr.copy()

    launch_rain_self_collection(qr, nr, temperature, pressure, qv, 10.0)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(nr, nr_before)


def test_rain_evaporation_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_rain_evaporation,
        launch_rain_sedimentation,
    )

    with (_ORACLE / "rain-evap-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)

    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0,
        reference_density=reference_density)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(temperature[:, 0, 0]), host(after, "temp_k"))
    np.testing.assert_array_equal(
        cp.asnumpy(qv[:, 0, 0]), host(after, "qv"))
    np.testing.assert_allclose(
        cp.asnumpy(qr[:, 0, 0]), host(after, "qr"),
        rtol=4.0e-6, atol=2.0e-11)
    np.testing.assert_allclose(
        cp.asnumpy(nr[:, 0, 0]), host(after, "nr_per_kg"),
        rtol=4.0e-6, atol=4.0)
    with (_ORACLE / "rain-evap-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    expected_rain = np.float32(float(surface["rainncv_mm"]))
    np.testing.assert_allclose(
        cp.asnumpy(rainncv), expected_rain, rtol=2.0e-6, atol=2.0e-10)
    np.testing.assert_array_equal(cp.asnumpy(rainnc), cp.asnumpy(rainncv))

    actual_vapor_gain = np.sum(
        cp.asnumpy(qv[:, 0, 0]) - host(before, "qv"), dtype=np.float64)
    expected_vapor_gain = np.sum(
        host(after, "qv") - host(before, "qv"), dtype=np.float64)
    assert actual_vapor_gain == pytest.approx(expected_vapor_gain, abs=2.0e-9)


def test_rain_evaporation_saturated_and_rainless_cells_are_exact_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_evaporation

    shape = (4, 2, 3)
    qr = cp.full(shape, 3.0e-4, dtype=cp.float32)
    qr[0] = 0.0
    nr = cp.full(shape, 3.0e5, dtype=cp.float32)
    nr[0] = 0.0
    temperature = cp.full(shape, 285.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.02, dtype=cp.float32)
    fields_before = [field.copy() for field in (qr, nr, temperature, qv)]

    launch_rain_evaporation(qr, nr, temperature, pressure, qv, 10.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip((qr, nr, temperature, qv), fields_before):
        cp.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="positive"):
        launch_rain_evaporation(qr, nr, temperature, pressure, qv, 0.0)


def test_cold_rain_evaporation_with_zero_melt_marker_conserves_water():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_evaporation

    shape = (4, 1, 1)
    qr = cp.full(shape, 5.0e-4, dtype=cp.float32)
    nr = cp.full(shape, 1.0e5, dtype=cp.float32)
    temperature = cp.full(shape, 262.0, dtype=cp.float32)
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 5.0e-4, dtype=cp.float32)
    reference_density = cp.empty_like(qr)
    graupel_melt_marker = cp.zeros_like(qr)
    qr_before = qr.copy()
    qv_before = qv.copy()
    temperature_before = temperature.copy()

    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density,
        graupel_melt_marker=graupel_melt_marker)
    cp.cuda.Stream.null.synchronize()

    assert bool(cp.all(qv > qv_before))
    assert bool(cp.all(qr < qr_before))
    assert bool(cp.all(temperature < temperature_before))
    cp.testing.assert_allclose(
        qv + qr, qv_before + qr_before, rtol=0.0, atol=1.0e-10)
    assert bool(cp.all(cp.isfinite(reference_density)))
    assert bool(cp.all(reference_density > 0.0))
    cp.testing.assert_array_equal(graupel_melt_marker, 0.0)


def test_rain_evaporation_rejects_cupy_view_of_density_as_marker():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_evaporation

    shape = (2, 1, 1)
    fields = [cp.zeros(shape, dtype=cp.float32) for _ in range(5)]
    reference_density = cp.zeros(shape, dtype=cp.float32)
    marker_view = reference_density.view()
    assert marker_view is not reference_density
    assert marker_view.data.ptr == reference_density.data.ptr
    with pytest.raises(ValueError, match="must not alias reference_density"):
        launch_rain_evaporation(
            *fields, 10.0,
            reference_density=reference_density,
            graupel_melt_marker=marker_view)


def test_snow_sublimation_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_snow_sedimentation,
        launch_snow_sublimation,
    )

    with (_ORACLE / "snow-subl-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qs = volume("qs")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_snow_sublimation(qs, temperature, pressure, qv, 10.0)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(temperature[:, 0, 0]), host(after, "temp_k"))
    np.testing.assert_array_equal(
        cp.asnumpy(qv[:, 0, 0]), host(after, "qv"))
    np.testing.assert_allclose(
        cp.asnumpy(qs[:, 0, 0]), host(after, "qs"),
        rtol=2.0e-7, atol=2.0e-11)
    with (_ORACLE / "snow-subl-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "snownc_mm", "snowncv_mm")):
        np.testing.assert_array_equal(
            cp.asnumpy(field), np.float32(float(surface[name])))


def test_snow_deposition_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_snow_sedimentation,
        launch_snow_vapor_exchange,
    )

    with (_ORACLE / "snow-dep-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qs = volume("qs")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_snow_vapor_exchange(qs, temperature, pressure, qv, 10.0)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 2.0e-6, 2.0e-5),
            (qv, "qv", 3.0e-6, 2.0e-10),
            (qs, "qs", 8.0e-6, 2.0e-11)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol, err_msg=name)
    with (_ORACLE / "snow-dep-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "snownc_mm", "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=8.0e-6, atol=2.0e-12)


def test_snow_vapor_exchange_warm_and_snowless_cells_are_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_snow_sublimation

    shape = (2, 2, 2)
    qs = cp.full(shape, 2.0e-4, dtype=cp.float32)
    qs[0] = 0.0
    temperature = cp.full(shape, 260.0, dtype=cp.float32)
    temperature[1] = 280.0
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.02, dtype=cp.float32)
    fields_before = [field.copy() for field in (qs, temperature, qv)]

    launch_snow_sublimation(qs, temperature, pressure, qv, 10.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip((qs, temperature, qv), fields_before):
        cp.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="positive"):
        launch_snow_sublimation(qs, temperature, pressure, qv, 0.0)


def test_graupel_sublimation_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_graupel_sedimentation,
        launch_graupel_sublimation,
    )

    with (_ORACLE / "graupel-subl-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qg = volume("qg")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_graupel_sublimation(qg, temperature, pressure, qv, 10.0)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(temperature[:, 0, 0]), host(after, "temp_k"))
    np.testing.assert_array_equal(
        cp.asnumpy(qv[:, 0, 0]), host(after, "qv"))
    np.testing.assert_allclose(
        cp.asnumpy(qg[:, 0, 0]), host(after, "qg"),
        rtol=2.0e-7, atol=2.0e-11)
    with (_ORACLE / "graupel-subl-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "graupelnc_mm", "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=2.0e-7, atol=1.0e-10)


def test_supersaturated_graupel_does_not_deposit_and_fallout_matches_wrf():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_graupel_sedimentation,
        launch_graupel_sublimation,
    )

    with (_ORACLE / "graupel-dep-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qg = volume("qg")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_graupel_sublimation(qg, temperature, pressure, qv, 10.0)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()

    # Classic Thompson intentionally permits snow deposition but only
    # graupel sublimation; supersaturated graupel must not consume vapor.
    np.testing.assert_array_equal(
        cp.asnumpy(temperature[:, 0, 0]), host(after, "temp_k"))
    np.testing.assert_array_equal(
        cp.asnumpy(qv[:, 0, 0]), host(after, "qv"))
    np.testing.assert_allclose(
        cp.asnumpy(qg[:, 0, 0]), host(after, "qg"),
        rtol=8.0e-6, atol=2.0e-11)
    with (_ORACLE / "graupel-dep-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "graupelnc_mm", "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=8.0e-6, atol=2.0e-10)


def test_saturated_snow_melting_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_rain_sedimentation,
        launch_snow_melting,
        launch_snow_sedimentation,
        launch_warm_saturation_adjust,
    )

    with (_ORACLE / "snow-melt-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qs = volume("qs")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)

    qs_before_melt = qs.copy()
    launch_snow_melting(
        qs, qr, nr, temperature, pressure, qv, 10.0)
    snow_melt_marker = (qs < qs_before_melt).astype(cp.float32)
    launch_warm_saturation_adjust(temperature, pressure, qv, qc)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        snow_melt_marker=snow_melt_marker,
        melt_rain_qr=qr, melt_rain_nr=nr)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 8.0e-6, 1.0e-9),
            (qr, "qr", 1.2e-5, 2.0e-11),
            (nr, "nr_per_kg", 1.2e-5, 3.0e-2),
            (qs, "qs", 1.2e-5, 2.0e-11)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)
    with (_ORACLE / "snow-melt-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in (
            (rainnc, "rainnc_mm"), (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"), (snowncv, "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=3.0e-10)


def test_saturated_graupel_melting_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_graupel_melting,
        launch_graupel_sedimentation,
        launch_rain_sedimentation,
        launch_warm_saturation_adjust,
    )

    with (_ORACLE / "graupel-melt-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qg = volume("qg")
    dz = volume("dz_m")
    graupel_number = cp.zeros_like(qg)
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)

    launch_graupel_melting(
        qg, qr, nr, graupel_number,
        temperature, pressure, qv, 10.0)
    launch_warm_saturation_adjust(temperature, pressure, qv, qc)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, 10.0,
        graupel_number=graupel_number, accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 8.0e-6, 1.0e-9),
            (qr, "qr", 1.2e-5, 2.0e-11),
            (nr, "nr_per_kg", 1.2e-5, 3.0e-2),
            (qg, "qg", 1.2e-5, 5.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)
    with (_ORACLE / "graupel-melt-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in (
            (rainnc, "rainnc_mm"), (rainncv, "rainncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=3.0e-10)


def test_graupel_sublimation_saturated_warm_and_graupelless_cells_are_noop():
    import cupy as cp

    from gpuwm.core.thompson import launch_graupel_sublimation

    shape = (3, 2, 2)
    qg = cp.full(shape, 2.0e-4, dtype=cp.float32)
    qg[0] = 0.0
    temperature = cp.full(shape, 260.0, dtype=cp.float32)
    temperature[1] = 280.0
    pressure = cp.full(shape, 80000.0, dtype=cp.float32)
    qv = cp.full(shape, 0.02, dtype=cp.float32)
    fields_before = [field.copy() for field in (qg, temperature, qv)]

    launch_graupel_sublimation(qg, temperature, pressure, qv, 10.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip((qg, temperature, qv), fields_before):
        cp.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="positive"):
        launch_graupel_sublimation(qg, temperature, pressure, qv, 0.0)


def test_warm_rain_collection_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_rain_sedimentation,
        launch_warm_rain_collection,
    )

    with (_ORACLE / "warm-accrete-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Exact entries exercised by this direct column, extracted from the
    # hash-pinned canonical t_Efrw(100,100) FP64 asset.  The production API
    # accepts the full validated table; zero entries are irrelevant here.
    table_host = np.zeros((100, 100), dtype=np.float64, order="F")
    table_host[49, 3] = 0.09128311276435852
    table_host[49, 4] = 0.22680087387561798
    collision_efficiency = cp.asarray(table_host, order="F")

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)

    launch_warm_rain_collection(
        qc, qr, nr, temperature, pressure, qv,
        collision_efficiency, 10.0)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(qc[:, 0, 0]), host(after, "qc"),
        rtol=8.0e-6, atol=2.0e-11)
    np.testing.assert_allclose(
        cp.asnumpy(qr[:, 0, 0]), host(after, "qr"),
        rtol=8.0e-6, atol=2.0e-11)
    np.testing.assert_allclose(
        cp.asnumpy(nr[:, 0, 0]), host(after, "nr_per_kg"),
        rtol=8.0e-6, atol=2.0e-2)
    with (_ORACLE / "warm-accrete-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    expected_rain = np.float32(float(surface["rainncv_mm"]))
    np.testing.assert_allclose(
        cp.asnumpy(rainncv), expected_rain, rtol=8.0e-6, atol=2.0e-9)
    np.testing.assert_array_equal(cp.asnumpy(rainnc), cp.asnumpy(rainncv))

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    water_before = np.sum(
        rho * (host(before, "qc") + host(before, "qr"))
        * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * (cp.asnumpy(qc[:, 0, 0]) + cp.asnumpy(qr[:, 0, 0]))
        * host(before, "dz_m"), dtype=np.float64)
    wrf_after = np.sum(
        rho * (host(after, "qc") + host(after, "qr"))
        * host(before, "dz_m"), dtype=np.float64)
    assert water_before - water_after == pytest.approx(
        water_before - wrf_after, abs=4.0e-8)


def test_warm_rain_collection_rejects_noncanonical_table_layout():
    import cupy as cp

    from gpuwm.core.thompson import launch_warm_rain_collection

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(6)]
    with pytest.raises(TypeError, match="float64"):
        launch_warm_rain_collection(
            *fields, cp.zeros((100, 100), dtype=cp.float32), 10.0)
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_warm_rain_collection(
            *fields, cp.zeros((100, 100), dtype=cp.float64), 10.0)


def test_fused_warm_process_network_matches_simultaneous_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_rain_sedimentation,
        launch_warm_process_network,
    )

    with (_ORACLE / "warm-overlap-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Only this rain row and these cloud-diameter bins are touched by the
    # hash-pinned simultaneous WRF oracle.  Values are exact entries from the
    # canonical t_Efrw(100,100) FP64 table.
    table_host = np.zeros((100, 100), dtype=np.float64, order="F")
    for cloud_bin, value in {
            2: 0.0,
            4: 0.09128311276435852,
            9: 0.5825913548469543,
            15: 0.8137867450714111,
            22: 0.898698091506958,
            23: 0.9053093791007996,
            27: 0.9252923727035522,
            28: 0.9291030168533325,
            30: 0.9357151985168457,
            }.items():
        table_host[49, cloud_bin - 1] = value
    collision_efficiency = cp.asarray(table_host, order="F")

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)

    launch_warm_process_network(
        qc, qr, nr, temperature, pressure, qv,
        collision_efficiency, 10.0)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 0.0, 3.0e-5),
            (qv, "qv", 0.0, 2.0e-10),
            (qc, "qc", 1.0e-5, 3.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-11),
            (nr, "nr_per_kg", 1.0e-5, 3.0e-2)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "warm-overlap-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    expected_rain = np.float32(float(surface["rainncv_mm"]))
    np.testing.assert_allclose(
        cp.asnumpy(rainncv), expected_rain, rtol=1.0e-5, atol=3.0e-9)
    np.testing.assert_array_equal(cp.asnumpy(rainnc), cp.asnumpy(rainncv))

    rho = np.float32(0.622) * host(before, "p_pa") / (
        np.float32(287.04) * host(before, "temp_k")
        * (host(before, "qv") + np.float32(0.622)))
    water_before = np.sum(
        rho * (host(before, "qc") + host(before, "qr"))
        * host(before, "dz_m"), dtype=np.float64)
    water_after = np.sum(
        rho * (cp.asnumpy(qc[:, 0, 0]) + cp.asnumpy(qr[:, 0, 0]))
        * host(before, "dz_m"), dtype=np.float64)
    # Summing the FP32 category transfer beside O(1e-3) retained cloud water
    # exposes a 9.36e-8 kg m-2 cancellation residual on this official column.
    assert water_before - water_after == pytest.approx(
        float(cp.asnumpy(rainncv)[0, 0]), abs=1.2e-7)


def test_fused_warm_process_network_caps_once_and_avoids_process_ordering():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_warm_autoconversion,
        launch_warm_process_network,
        launch_warm_rain_collection,
    )

    temperature = cp.asarray([285.0], dtype=cp.float32)
    pressure = cp.asarray([80000.0], dtype=cp.float32)
    qv = cp.asarray([0.008], dtype=cp.float32)
    qc0 = cp.asarray([1.0e-3], dtype=cp.float32)
    qr0 = cp.asarray([3.0e-4], dtype=cp.float32)
    diameter = np.float32(500.0e-6)
    nr0 = cp.asarray([
        np.float32(3.0e-4) * (np.float32(3.672) / diameter) ** 3
        / (np.float32(np.pi) * np.float32(1000.0))], dtype=cp.float32)
    table = cp.asarray(
        np.full((100, 100), 0.95, dtype=np.float64, order="F"), order="F")
    dt = 1.0e6

    qc, qr, nr = qc0.copy(), qr0.copy(), nr0.copy()
    launch_warm_process_network(
        qc, qr, nr, temperature, pressure, qv, table, dt)

    # The two legacy isolated launch orders intentionally disagree in this
    # source-limited cell because the second launch sees provisional state.
    qc_auto, qr_auto, nr_auto = qc0.copy(), qr0.copy(), nr0.copy()
    launch_warm_autoconversion(
        qc_auto, qr_auto, nr_auto, temperature, pressure, qv, dt)
    launch_warm_rain_collection(
        qc_auto, qr_auto, nr_auto, temperature, pressure, qv, table, dt)
    qc_collect, qr_collect, nr_collect = qc0.copy(), qr0.copy(), nr0.copy()
    launch_warm_rain_collection(
        qc_collect, qr_collect, nr_collect,
        temperature, pressure, qv, table, dt)
    launch_warm_autoconversion(
        qc_collect, qr_collect, nr_collect, temperature, pressure, qv, dt)
    cp.cuda.Stream.null.synchronize()

    assert float(cp.asnumpy(qc)[0]) >= 0.0
    assert float(cp.asnumpy(qc)[0]) <= 2.0e-10
    np.testing.assert_allclose(
        cp.asnumpy(qc + qr), cp.asnumpy(qc0 + qr0),
        rtol=0.0, atol=2.0e-10)
    assert float(cp.asnumpy(nr)[0]) >= 0.0
    assert not bool(cp.all(nr_auto == nr_collect))
    assert not bool(cp.all(nr == nr_auto))
    assert not bool(cp.all(nr == nr_collect))

    # Repeating the fused calculation from the same immutable input is exact;
    # no hidden process ordering exists inside the public launch.
    qc_repeat, qr_repeat, nr_repeat = qc0.copy(), qr0.copy(), nr0.copy()
    launch_warm_process_network(
        qc_repeat, qr_repeat, nr_repeat,
        temperature, pressure, qv, table, dt)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(qc_repeat, qc)
    cp.testing.assert_array_equal(qr_repeat, qr)
    cp.testing.assert_array_equal(nr_repeat, nr)


def test_ice_deposition_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_ice_deposition,
        launch_ice_sedimentation,
        launch_snow_sedimentation,
    )

    with (_ORACLE / "ice-dep-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Every bin exercised by this 29-micron column is exactly 1.0 in the
    # hash-pinned canonical tpi_ide asset.  Production accepts the complete
    # validated FP64 Fortran-ordered table.
    partition = cp.ones((64, 55), dtype=cp.float64, order="F")
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_ice_deposition(
        qi, ni, qs, temperature, pressure, qv, partition, 10.0)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz, *surfaces, 10.0)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz, *surfaces, 10.0,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 2.0e-6, 2.0e-5),
            (qv, "qv", 3.0e-6, 2.0e-10),
            (qi, "qi", 8.0e-6, 2.0e-12),
            (ni, "ni_per_kg", 8.0e-6, 2.0e-2),
            (qs, "qs", 0.0, 1.0e-14)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)
    with (_ORACLE / "ice-dep-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "snownc_mm", "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=8.0e-6, atol=2.0e-15)


def test_ice_deposition_rejects_noncanonical_table_layout():
    import cupy as cp

    from gpuwm.core.thompson import launch_ice_deposition

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(6)]
    with pytest.raises(TypeError, match="float64"):
        launch_ice_deposition(
            *fields, cp.zeros((64, 55), dtype=cp.float32, order="F"), 10.0)
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_ice_deposition(
            *fields, cp.zeros((64, 55), dtype=cp.float64), 10.0)
    with pytest.raises(ValueError, match="positive"):
        launch_ice_deposition(
            *fields, cp.zeros((64, 55), dtype=cp.float64, order="F"), 0.0)


def test_frozen_vapor_network_and_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_frozen_vapor_network,
        launch_graupel_sedimentation,
        launch_ice_sedimentation,
        launch_snow_sedimentation,
    )

    scenario = "frozen-vapor-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Sublimation does not consult the positive-deposition partition, but the
    # production launcher still requires its canonical shape/layout.
    partition = cp.ones((64, 55), dtype=cp.float64, order="F")
    inactive_auto = cp.zeros_like(partition, order="F")
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    qr = cp.zeros_like(qg)
    nr = cp.zeros_like(qg)
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)

    launch_frozen_vapor_network(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        partition, inactive_auto, inactive_auto, 10.0)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        accumulate_surface=True)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, 10.0,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qi, "qi", 1.0e-5, 3.0e-12),
            (ni, "ni_per_kg", 1.0e-5, 3.0),
            (qs, "qs", 1.0e-5, 3.0e-12),
            (qg, "qg", 1.0e-5, 3.0e-12)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_frozen_vapor_network_rejects_bad_table_and_timestep():
    import cupy as cp

    from gpuwm.core.thompson import launch_frozen_vapor_network

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(9)]
    canonical = cp.zeros((64, 55), dtype=cp.float64, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_frozen_vapor_network(
            *fields, cp.zeros((64, 55), dtype=cp.float32, order="F"),
            canonical, canonical, 10.0)
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_frozen_vapor_network(
            *fields, canonical,
            cp.zeros((64, 55), dtype=cp.float64), canonical, 10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        launch_frozen_vapor_network(
            *fields, canonical, canonical, canonical, 0.0)


def test_source_entry_zero_rain_number_uses_wrf_local_one_mm_fallback():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_frozen_vapor_network,
        launch_warm_frozen_source_network,
    )

    # This is the failure shape captured in the 12:00-12:24 diagnostic:
    # advection supplies resolved rain mass to a cell whose prognostic rain
    # number is still zero.  Using only R2 in the source distribution implies
    # a ~100-mm MVD, overflows the breakup exponential, and the final bound
    # then leaves the cell at the opposite 37.5-micron/max-number edge.
    shape = (1, 1, 1)
    qr_value = np.float32(1.1054e-7)
    dt = 60.0
    inactive_ice_table = cp.zeros(
        (64, 55), dtype=cp.float64, order="F")

    def state(temperature_value):
        zeros = [cp.zeros(shape, dtype=cp.float32) for _ in range(7)]
        qc, qi, ni, qs, qg, nr, graupel_number = zeros
        qr = cp.full(shape, qr_value, dtype=cp.float32)
        temperature = cp.full(
            shape, temperature_value, dtype=cp.float32)
        pressure = cp.full(shape, 56860.0, dtype=cp.float32)
        qv = cp.full(shape, 1.0e-4, dtype=cp.float32)
        return (qc, qi, ni, qs, qg, qr, nr, graupel_number,
                temperature, pressure, qv)

    cold = state(np.float32(262.0))
    (cold_qc, cold_qi, cold_ni, cold_qs, cold_qg, cold_qr, cold_nr,
     cold_graupel_number, cold_temperature, cold_pressure, cold_qv) = cold
    # Supplying the transient graupel number selects the same expanded fused
    # cold kernel used by production, without enabling unrelated table rates.
    launch_frozen_vapor_network(
        cold_qi, cold_ni, cold_qs, cold_qg, cold_qr, cold_nr,
        cold_temperature, cold_pressure, cold_qv,
        inactive_ice_table, inactive_ice_table, inactive_ice_table, dt,
        graupel_number_shadow=cold_graupel_number)

    warm = state(np.float32(275.0))
    (warm_qc, _warm_qi, _warm_ni, warm_qs, warm_qg, warm_qr, warm_nr,
     warm_graupel_number, warm_temperature, warm_pressure, warm_qv) = warm
    rain_cloud_efficiency = cp.zeros(
        (100, 100), dtype=cp.float64, order="F")
    snow_cloud_efficiency = cp.zeros_like(
        rain_cloud_efficiency, order="F")
    rain_snow_table = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float64, order="F")
    rain_graupel_table = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float64, order="F")
    graupel_melt_marker = cp.ones(shape, dtype=cp.float32)
    snow_melt_marker = cp.zeros(shape, dtype=cp.float32)
    launch_warm_frozen_source_network(
        warm_qc, warm_qr, warm_nr, warm_qs, warm_qg,
        warm_graupel_number, graupel_melt_marker, snow_melt_marker,
        warm_temperature, warm_pressure, warm_qv,
        rain_cloud_efficiency, snow_cloud_efficiency,
        (rain_snow_table,) * 12, (rain_graupel_table,) * 5, dt)
    cp.cuda.Stream.null.synchronize()

    # WRF's 1-mm fallback is local to source-rate diagnosis.  Because the
    # prognostic nr remains zero and self-collection is a sink, its later
    # mass/number balance starts from R2 and lands at the 2.5-mm/min-number
    # edge.  Expecting 1 mm here would incorrectly mutate prognostic nr.
    am_r = np.float32(np.pi * 1000.0 / 6.0)
    lambda_2p5mm = np.float32(3.672 / 2.5e-3)
    expected_nr = np.float32(
        (1.0 / 6.0) * qr_value / am_r * lambda_2p5mm ** 3)
    lambda_37p5um = np.float32(3.672 / 37.5e-6)
    buggy_max_nr = np.float32(
        (1.0 / 6.0) * qr_value / am_r * lambda_37p5um ** 3)

    for qr, nr in ((cold_qr, cold_nr), (warm_qr, warm_nr)):
        actual_qr = float(cp.asnumpy(qr)[0, 0, 0])
        actual_nr = float(cp.asnumpy(nr)[0, 0, 0])
        assert np.isfinite(actual_nr)
        np.testing.assert_allclose(actual_qr, qr_value, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            actual_nr, expected_nr, rtol=6.0e-6, atol=1.0e-7)
        actual_lambda = (float(am_r) * 6.0 * actual_nr / actual_qr) ** (1.0 / 3.0)
        np.testing.assert_allclose(
            3.672 / actual_lambda, 2.5e-3, rtol=6.0e-6, atol=0.0)
        assert float(buggy_max_nr) / actual_nr > 2.0e5


def test_frozen_vapor_nucleation_autoconversion_shared_caps_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_frozen_vapor_network,
        launch_ice_sedimentation,
        launch_snow_sedimentation,
    )

    scenario = "frozen-vapor-nucleation-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]
    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    dt = float(surface["dt_s"])
    assert dt == 50.0

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Exact canonical entries read by this 200-micron direct-WRF vector.
    # Sparse focused fixtures keep the gate small; production still requires
    # all three complete FP64 Fortran-order tables.
    partition_host = np.ones((64, 55), dtype=np.float64, order="F")
    auto_mass_host = np.zeros_like(partition_host, order="F")
    auto_number_host = np.zeros_like(partition_host, order="F")
    sparse_entries = (
        ((18, 11), (0.9840137958526611,
                    1.3983460565800533e-09, 0.06650428917526063)),
        ((15, 10), (0.9826064109802246,
                    1.0363572205145063e-09, 0.04891872200478731)),
        ((11, 7), (0.9804539680480957,
                   4.802247865008119e-10, 0.022425780383867253)),
        ((7, 1), (0.9782599210739136,
                  1.3738342869490775e-10, 0.006350920990242831)),
        ((0, 0), (0.9988777041435242,
                  1.9396260193494513e-12, 0.00010857258774438814)),
    )
    for index, (partition_value, mass_value, number_value) in sparse_entries:
        partition_host[index] = partition_value
        auto_mass_host[index] = mass_value
        auto_number_host[index] = number_value
    partition = cp.asarray(partition_host, order="F")
    auto_mass = cp.asarray(auto_mass_host, order="F")
    auto_number = cp.asarray(auto_number_host, order="F")
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    qr = cp.zeros_like(qg)
    nr = cp.zeros_like(qg)
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)

    launch_frozen_vapor_network(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        partition, auto_mass, auto_number, dt)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 4.0e-6, 4.0e-5),
            (qv, "qv", 6.0e-6, 4.0e-10),
            (qi, "qi", 1.2e-5, 4.0e-12),
            (ni, "ni_per_kg", 1.2e-5, 6.0),
            (qs, "qs", 1.2e-5, 4.0e-10),
            (qg, "qg", 0.0, 0.0)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=4.0e-10)
    assert float(surface["graupelnc_mm"]) == 0.0
    assert float(surface["graupelncv_mm"]) == 0.0

    # The adversarial source step forces WRF's cap in these layers. Sequential
    # nucleation/autoconversion launches would consume provisional vapor and
    # ice distributions and not retain this exact 0.1-percent residual excess.
    initial_qv = host(before, "qv")
    final_qv = host(after, "qv")
    final_ice_rh = final_qv / (initial_qv / np.float32(1.30))
    np.testing.assert_allclose(
        final_ice_rh[2:5], np.float32(1.0003),
        rtol=0.0, atol=2.0e-7)


def test_cold_ice_rain_shared_caps_and_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_frozen_vapor_network,
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
    )

    scenario = "cold-ice-rain-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]
    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    dt = float(surface["dt_s"])
    assert dt == 50.0

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Exact canonical entries reached by the 200-micron incoming ice
    # distribution. The sparse focused fixtures are not production-table
    # substitutes; production still validates and uploads all three tables.
    partition_host = np.ones((64, 55), dtype=np.float64, order="F")
    auto_mass_host = np.zeros_like(partition_host, order="F")
    auto_number_host = np.zeros_like(partition_host, order="F")
    sparse_entries = (
        ((31, 27), (0.9692050814628601,
                    1.073202837455873e-07, 4.787649907861354)),
        ((29, 26), (0.9840137958526611,
                    4.19503816974016e-08, 1.995128675257819)),
        ((27, 20), (0.9840137958526611,
                    1.3983460565800532e-08, 0.6650428917526064)),
        ((20, 18), (0.9867487549781799,
                    3.6883666452398997e-09, 0.17825591881526304)),
        ((14, 9), (0.959938645362854,
                   1.5116325003456848e-09, 0.0654697936689139)),
        ((6, 1), (0.9826064109802246,
                  1.036357220514506e-10, 0.00489187220047873)),
        ((0, 0), (0.9988777041435242,
                  1.9396260193494513e-12, 0.00010857258774438814)),
    )
    for index, (partition_value, mass_value, number_value) in sparse_entries:
        partition_host[index] = partition_value
        auto_mass_host[index] = mass_value
        auto_number_host[index] = number_value
    partition = cp.asarray(partition_host, order="F")
    auto_mass = cp.asarray(auto_mass_host, order="F")
    auto_number = cp.asarray(auto_number_host, order="F")

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)

    launch_frozen_vapor_network(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        partition, auto_mass, auto_number, dt)
    # WRF's ordinary post-source rain evaporation remains in the direct-call
    # trajectory and also preserves the density used by both entry-time
    # fallout paths.
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, dt,
        reference_density=reference_density)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=reference_density)
    # Snow and graupel are created during this call and therefore remain
    # entry-inactive for sedimentation. Incoming rain is eligible.
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, dt, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 4.0e-6, 4.0e-5),
            (qv, "qv", 6.0e-6, 4.0e-10),
            (qr, "qr", 1.2e-5, 4.0e-12),
            (nr, "nr_per_kg", 1.2e-5, 6.0),
            (qi, "qi", 1.2e-5, 4.0e-12),
            (ni, "ni_per_kg", 1.2e-5, 6.0),
            (qs, "qs", 1.2e-5, 4.0e-10),
            (qg, "qg", 1.2e-5, 4.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=4.0e-10)
    assert float(surface["graupelnc_mm"]) == 0.0
    assert float(surface["graupelncv_mm"]) == 0.0


def test_full_cold_rain_frozen_source_order_matches_wrf_column():
    import cupy as cp
    from types import MappingProxyType

    from gpuwm.core.refl import launch_refl10cm_thompson
    from gpuwm.core.thompson import (
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
        launch_frozen_vapor_network_from_owner,
        launch_graupel_sedimentation,
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
    )
    from gpuwm.core.thompson_runtime import (
        RAIN_FREEZING_TABLE_NAMES,
        RAIN_GRAUPEL_TABLE_NAMES,
        RAIN_SNOW_TABLE_NAMES,
        DeviceClassicTableSet,
    )

    scenario = "cold-full-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]
    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    dt = float(surface["dt_s"])
    assert dt == 50.0

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    def sparse_flat_tables(shape, entries, count):
        hosts = [
            np.zeros(shape, dtype=np.float64, order="F")
            for _ in range(count)
        ]
        for flat_index, values in entries:
            for table, value in zip(hosts, values, strict=True):
                table.ravel(order="F")[flat_index] = value
        return tuple(cp.asarray(table, order="F") for table in hosts)

    partition_host = np.ones((64, 55), dtype=np.float64, order="F")
    auto_mass_host = np.zeros_like(partition_host, order="F")
    auto_number_host = np.zeros_like(partition_host, order="F")
    ice_entries = (
        ((18, 11), (0.9840137958526611,
                    1.3983460565800533e-09, 0.06650428917526063)),
        ((15, 10), (0.9826064109802246,
                    1.0363572205145063e-09, 0.04891872200478731)),
        ((11, 7), (0.9804539680480957,
                   4.802247865008119e-10, 0.022425780383867253)),
        ((7, 1), (0.9782599210739136,
                  1.3738342869490775e-10, 0.006350920990242831)),
        ((0, 0), (0.9988777041435242,
                  1.9396260193494513e-12, 0.00010857258774438814)),
    )
    for index, (partition_value, mass_value, number_value) in ice_entries:
        partition_host[index] = partition_value
        auto_mass_host[index] = mass_value
        auto_number_host[index] = number_value
    partition = cp.asarray(partition_host, order="F")
    auto_mass = cp.asarray(auto_mass_host, order="F")
    auto_number = cp.asarray(auto_number_host, order="F")

    freezing_entries = (
        (1707533, (1.1637489187645824e-05, 2765.860087874303,
                   0.0002883626508654246, 1496.2689256113558)),
        (1707495, (6.219319722033612e-06, 1434.3357143759665,
                   0.00019378079387359336, 877.1899753952558)),
        (1707420, (4.451519467612516e-06, 1073.7770192156147,
                   9.554735910761533e-05, 539.2847541881341)),
        (1707230, (1.7897470191423407e-06, 421.8815589781589,
                   4.821047383678161e-05, 238.08411722196865)),
        (1707152, (5.254067101379029e-07, 129.19236627146046,
                   9.474275653762982e-06, 59.118482483722)),
        (1707144, (3.1389723544495207e-07, 92.4519581123868,
                   1.6852086961899636e-06, 19.84821769043297)),
        (1707143, (2.3564835066769214e-07, 77.20462257960456,
                   7.634299501986699e-07, 11.259479127943294)),
    )
    rain_freezing_tables = sparse_flat_tables(
        (37, 37, 45, 55), freezing_entries, 4)

    rain_snow_entries = (
        (250042, (1.6837419581804296e-05, 0.0003000000142492354,
                  3.0060894397621922e-06, 2.5101512774745658e-06,
                  0.0, 0.0, 1.0014391825207814e-07,
                  4.4138098990103546e-06, 728.4029688568966,
                  62.70272344914709, 0.0, 82.85403721359158)),
        (237386, (7.70406288057629e-06, 0.00019999999494757503,
                  1.0907312872990817e-06, 9.247008073079382e-07,
                  0.0, 0.0, 3.457345595313417e-08,
                  1.4398679983391612e-06, 347.80327453869086,
                  23.952216589930906, 0.0, 29.15493702601655)),
        (224397, (2.253526702939162e-06, 8.55604826506699e-05,
                  3.633814854299629e-07, 3.0924545162723066e-07,
                  0.0, 0.0, 1.3611471776125844e-08,
                  5.434417708718266e-07, 111.05923328807023,
                  9.179388723650572, 0.0, 12.42381172421413)),
        (161125, (3.31064537340766e-07, 1.743945053193794e-05,
                  3.557374084512338e-08, 3.1263798185775425e-08,
                  0.0, 0.0, 1.3528626903931716e-09,
                  4.662401066179559e-08, 18.83171317933403,
                  1.1027062307761348, 0.0, 1.3572620260808852)),
        (111168, (1.7435489846362147e-08, 7.628056686516403e-07,
                  1.9212897587949495e-09, 1.718787199054147e-09,
                  0.0, 0.0, 9.079138471372549e-11,
                  2.7829471013726224e-09, 1.1601105055731609,
                  0.07712351031965813, 0.0, 0.10458106200734094)),
        (12593, (4.2643767037074043e-10, 8.261137441023362e-09,
                 8.227712892078539e-11, 7.411121400293738e-11,
                 0.0, 0.0, 6.66519152055641e-12,
                 1.8264611063262096e-10, 0.03517079066586766,
                 0.0048292981748589165, 0.0, 0.009687516529781268)),
    )
    rain_snow_tables = sparse_flat_tables(
        (37, 9, 37, 37), rain_snow_entries, 12)

    rain_graupel_entries = (
        (1032947, (1.737609020685299e-07, 3.303892641003907e-08,
                   6.091020327012273e-08, 0.06838930809162569,
                   3.3301776919863086)),
        (980888, (7.65519147373558e-08, 1.555420558543834e-08,
                  1.86561245468251e-08, 0.03871040384068884,
                  1.2872997901245293)),
        (927497, (3.5519786863298714e-08, 6.920995103290605e-09,
                  9.71775244816401e-09, 0.019249012838508422,
                  0.735414490795451)),
        (667128, (1.1808937804227983e-08, 3.666520731363542e-09,
                  8.721663261155263e-10, 0.006960845717607241,
                  0.059726041562087676)),
        (461701, (7.097231274158658e-10, 1.8189270229440415e-10,
                  7.101087746246616e-11, 0.000795559363525502,
                  0.010026165671118796)),
        (56141, (1.5140399734947955e-11, 3.7903396602583446e-12,
                 1.4960162068700205e-12, 4.956215597775042e-05,
                 0.0005896004604246597)),
    )
    rain_graupel_tables = sparse_flat_tables(
        (37, 37, 1, 37, 37), rain_graupel_entries, 5)
    table_arrays = {
        "tpi_ide": partition,
        "tps_iaus": auto_mass,
        "tni_iaus": auto_number,
        **dict(zip(RAIN_SNOW_TABLE_NAMES, rain_snow_tables, strict=True)),
        **dict(zip(
            RAIN_GRAUPEL_TABLE_NAMES, rain_graupel_tables, strict=True)),
        **dict(zip(
            RAIN_FREEZING_TABLE_NAMES, rain_freezing_tables, strict=True)),
        # The focused cold-rain call below does not request qc, so the cloud
        # group is not dereferenced.  Keep lightweight sentinel arrays in the
        # synthetic owner while still exercising the expanded owner contract.
        "t_Efrw": partition,
        "tpi_qcfz": partition,
        "tni_qcfz": partition,
    }
    table_owner = DeviceClassicTableSet(
        root=_ORACLE,
        arrays=MappingProxyType(table_arrays),
        payload_bytes=sum(value.nbytes for value in table_arrays.values()),
        array_sha256=MappingProxyType({}),
        identity_json="{}",
        identity_sha256="fixture-identity",
        device_id=int(cp.cuda.Device().id),
        upload_seconds=0.0,
        verification_seconds=0.0,
        roundtrip_verified=True,
    )

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)
    reference_temperature = cp.empty_like(qr)
    graupel_number_shadow = cp.empty_like(qg)
    qg_fallout_columns = cp.ones((1, 1), dtype=cp.float32)
    refl = cp.empty_like(qg)

    launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, graupel_number_shadow)
    launch_frozen_vapor_network_from_owner(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        table_owner, dt,
        graupel_number_shadow=graupel_number_shadow)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, dt,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=reference_density)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=reference_density,
        reference_temperature=reference_temperature,
        accumulate_surface=True)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt,
        reference_density=reference_density,
        active_columns=qg_fallout_columns,
        graupel_number_shadow=graupel_number_shadow,
        accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, dt, reference_density=reference_density,
        accumulate_surface=True)
    launch_classic_graupel_number_finalize(
        qg, temperature, pressure, qv, graupel_number_shadow)
    launch_refl10cm_thompson(
        qv, qr, nr, qs, qg, graupel_number_shadow,
        temperature, pressure, refl)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 5.0e-6, 5.0e-5),
            (qv, "qv", 8.0e-6, 5.0e-10),
            # WRF leaves a single 2**-35 rain-mixing-ratio cancellation
            # residue in this adversarial all-group column; CUDA rounds that
            # physically exhausted bin to zero.
            (qr, "qr", 1.5e-5, 5.0e-11),
            (nr, "nr_per_kg", 1.5e-5, 8.0),
            (qi, "qi", 1.5e-5, 5.0e-12),
            (ni, "ni_per_kg", 1.5e-5, 8.0),
            (qs, "qs", 1.5e-5, 5.0e-10),
            (qg, "qg", 1.5e-5, 5.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    for actual, name in (
            (rainnc, "rainnc_mm"), (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"), (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.5e-5, atol=5.0e-10)
    np.testing.assert_allclose(
        cp.asnumpy(refl[:, 0, 0]), host(after, "refl_dbz"),
        rtol=0.0, atol=5.0e-2)


def test_full_cloud_rain_frozen_source_order_matches_wrf_column():
    import cupy as cp
    from types import MappingProxyType

    from gpuwm.core.refl import launch_refl10cm_thompson
    from gpuwm.core.thompson import (
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
        launch_cloud_saturation_adjust,
        launch_frozen_vapor_network_from_owner,
        launch_graupel_fallout_column_mask,
        launch_graupel_sedimentation,
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
    )
    from gpuwm.core.thompson_runtime import (
        RAIN_FREEZING_TABLE_NAMES,
        RAIN_GRAUPEL_TABLE_NAMES,
        RAIN_SNOW_TABLE_NAMES,
        DeviceClassicTableSet,
    )

    scenario = "cold-cloud-rain-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]
    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    dt = float(surface["dt_s"])
    assert dt == 50.0

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    def sparse_flat_tables(shape, entries, count):
        hosts = [
            np.zeros(shape, dtype=np.float64, order="F")
            for _ in range(count)
        ]
        for flat_index, values in entries:
            for table, value in zip(hosts, values, strict=True):
                table.ravel(order="F")[flat_index] = value
        return tuple(cp.asarray(table, order="F") for table in hosts)

    ice_entries = (
        (0, (0.9988777041435242,
             1.9396260193494513e-12, 0.00010857258774438814)),
        (71, (0.9782599210739136,
              1.3738342869490775e-10, 0.006350920990242831)),
        (459, (0.9804539680480957,
               4.802247865008119e-10, 0.022425780383867253)),
        (655, (0.9826064109802246,
               1.0363572205145063e-09, 0.04891872200478731)),
        (722, (0.9840137958526611,
               1.3983460565800533e-09, 0.06650428917526063)),
    )
    partition, auto_mass, auto_number = sparse_flat_tables(
        (64, 55), ice_entries, 3)

    rain_freezing_entries = (
        (1707143, (2.3564835066769214e-07, 77.20462257960456,
                   7.634299501986699e-07, 11.259479127943294)),
        (1707144, (3.1389723544495207e-07, 92.4519581123868,
                   1.6852086961899636e-06, 19.84821769043297)),
        (1707152, (5.254067101379029e-07, 129.19236627146046,
                   9.474275653762982e-06, 59.118482483722)),
        (1707230, (1.7897470191423407e-06, 421.8815589781589,
                   4.821047383678161e-05, 238.08411722196865)),
        (1707420, (4.451519467612516e-06, 1073.7770192156147,
                   9.554735910761533e-05, 539.2847541881341)),
        (1707495, (6.219319722033612e-06, 1434.3357143759665,
                   0.00019378079387359336, 877.1899753952558)),
        (1707533, (1.1637489187645824e-05, 2765.860087874303,
                   0.0002883626508654246, 1496.2689256113558)),
    )
    rain_freezing_tables = sparse_flat_tables(
        (37, 37, 45, 55), rain_freezing_entries, 4)

    efficiency_host = np.zeros((100, 100), dtype=np.float64, order="F")
    for flat_index, value in (
            (465, 0.2440161556005478),
            (765, 0.5285485982894897),
            (1265, 0.7743796706199646),
            (1765, 0.866192638874054),
            (2265, 0.9078192114830017),
            (2465, 0.9183456897735596)):
        efficiency_host.ravel(order="F")[flat_index] = value
    rain_cloud_efficiency = cp.asarray(efficiency_host, order="F")

    cloud_freezing_entries = (
        (4616308, (2.6940401262332944e-09, 53559.56982915825)),
        (4616315, (6.717408792055978e-08, 267424.3388722158)),
        (4616322, (1.3448398760270702e-06, 1196386.0302252292)),
        (4616324, (6.525313212406292e-06, 2634578.718132887)),
        (4616326, (2.5303775308354418e-05, 5184477.120453546)),
        (4616328, (5.525765771810393e-05, 7654764.681865499)),
    )
    cloud_freezing_tables = sparse_flat_tables(
        (37, 100, 45, 55), cloud_freezing_entries, 2)

    # This vector has no incoming snow or graupel, so the two collision-table
    # families are inactive.  Keep canonical-shape zero fixtures so the
    # simultaneous owner/ABI contract remains exercised.
    rain_snow_zero = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float64, order="F")
    rain_graupel_zero = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float64, order="F")
    table_arrays = {
        "tpi_ide": partition,
        "tps_iaus": auto_mass,
        "tni_iaus": auto_number,
        **dict.fromkeys(RAIN_SNOW_TABLE_NAMES, rain_snow_zero),
        **dict.fromkeys(RAIN_GRAUPEL_TABLE_NAMES, rain_graupel_zero),
        **dict(zip(
            RAIN_FREEZING_TABLE_NAMES, rain_freezing_tables, strict=True)),
        "t_Efrw": rain_cloud_efficiency,
        "tpi_qcfz": cloud_freezing_tables[0],
        "tni_qcfz": cloud_freezing_tables[1],
    }
    table_owner = DeviceClassicTableSet(
        root=_ORACLE,
        arrays=MappingProxyType(table_arrays),
        payload_bytes=sum(value.nbytes for value in table_arrays.values()),
        array_sha256=MappingProxyType({}),
        identity_json="{}",
        identity_sha256="fixture-identity",
        device_id=int(cp.cuda.Device().id),
        upload_seconds=0.0,
        verification_seconds=0.0,
        roundtrip_verified=True,
    )

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    frozen_reference_density = cp.empty_like(qi)
    frozen_reference_temperature = cp.empty_like(qi)
    rain_reference_density = cp.empty_like(qr)
    qg_entry_active = (qg > cp.float32(1.0e-12)).astype(cp.float32)
    qg_fallout_columns = cp.empty((1, 1), dtype=cp.float32)
    graupel_number_shadow = cp.empty_like(qg)
    snow_velocity_boost = cp.empty_like(qs)
    refl = cp.empty_like(qg)

    launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, graupel_number_shadow)
    launch_frozen_vapor_network_from_owner(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        table_owner, dt, qc=qc,
        graupel_number_shadow=graupel_number_shadow,
        snow_velocity_boost=snow_velocity_boost)
    launch_graupel_fallout_column_mask(
        qg_entry_active, qg, qg_fallout_columns)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, dt,
        reference_density=rain_reference_density)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=frozen_reference_density)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature,
        velocity_boost=snow_velocity_boost,
        accumulate_surface=True)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt,
        reference_density=frozen_reference_density,
        active_columns=qg_fallout_columns,
        graupel_number_shadow=graupel_number_shadow,
        accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, dt,
        reference_density=rain_reference_density,
        accumulate_surface=True)
    launch_classic_graupel_number_finalize(
        qg, temperature, pressure, qv, graupel_number_shadow)
    launch_refl10cm_thompson(
        qv, qr, nr, qs, qg, graupel_number_shadow,
        temperature, pressure, refl)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 5.0e-6, 5.0e-5),
            (qv, "qv", 8.0e-6, 5.0e-10),
            (qc, "qc", 2.0e-5, 5.0e-10),
            (qr, "qr", 1.5e-5, 5.0e-11),
            (nr, "nr_per_kg", 1.5e-5, 8.0),
            (qi, "qi", 1.5e-5, 5.0e-12),
            (ni, "ni_per_kg", 1.5e-5, 8.0),
            (qs, "qs", 1.5e-5, 5.0e-10),
            (qg, "qg", 1.5e-5, 5.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    for actual, name in (
            (rainnc, "rainnc_mm"), (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"), (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.5e-5, atol=5.0e-10)
    np.testing.assert_allclose(
        cp.asnumpy(refl[:, 0, 0]), host(after, "refl_dbz"),
        rtol=0.0, atol=5.0e-2)


def test_cold_cloud_source_network_and_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cloud_saturation_adjust,
        launch_cold_cloud_source_network,
        launch_graupel_sedimentation,
        launch_ice_sedimentation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
    )

    scenario = "cold-cloud-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # This vector has no incoming rain, so rain/cloud accretion is inactive.
    # The production launch still requires the canonical table contract.
    rain_cloud_efficiency = cp.zeros(
        (100, 100), dtype=cp.float64, order="F")

    # Exact entries read by this hash-pinned direct-WRF column from the
    # canonical freezeH2O.dat cloud-freezing tables.  Sparse fixtures keep
    # the focused gate small without relaxing the production table contract.
    table_shape = (37, 100, 45, 55)
    mass_host = np.zeros(table_shape, dtype=np.float64, order="F")
    number_host = np.zeros_like(mass_host, order="F")
    sparse_entries = (
        ((25, 66, 5, 28),
         (5.670055310651233e-17, 6.439005390113394e-06)),
        ((28, 66, 5, 28),
         (1.1571266318003071e-16, 9.198666116190871e-06)),
        ((24, 66, 5, 28),
         (4.165808796328629e-17, 5.519061062978681e-06)),
        ((20, 66, 5, 28),
         (4.628439926724867e-18, 1.8388152125462158e-06)),
        ((16, 66, 5, 28),
         (5.666899820622346e-19, 6.421414274889123e-07)),
        ((10, 66, 5, 28),
         (1.1409282079190139e-20, 8.919280480431747e-08)),
        ((1, 66, 5, 28),
         (9.294170944119446e-23, 4.421527044728035e-09)),
    )
    for one_based_index, (mass, number) in sparse_entries:
        index = tuple(value - 1 for value in one_based_index)
        mass_host[index] = mass
        number_host[index] = number
    cloud_to_ice_mass = cp.asarray(mass_host, order="F")
    cloud_to_ice_number = cp.asarray(number_host, order="F")

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qc)
    reference_temperature = cp.empty_like(qc)

    launch_cold_cloud_source_network(
        qc, qr, nr, qi, ni, qs, qg, temperature, pressure, qv,
        rain_cloud_efficiency, cloud_to_ice_mass, cloud_to_ice_number,
        10.0)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature,
        accumulate_surface=True)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 4.0e-6, 4.0e-5),
            (qv, "qv", 6.0e-6, 4.0e-10),
            (qc, "qc", 1.2e-5, 6.0e-10),
            (qr, "qr", 1.2e-5, 4.0e-12),
            (nr, "nr_per_kg", 1.2e-5, 6.0),
            (qi, "qi", 1.2e-5, 4.0e-12),
            (ni, "ni_per_kg", 1.2e-5, 6.0),
            (qs, "qs", 1.2e-5, 4.0e-10),
            (qg, "qg", 1.2e-5, 4.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=4.0e-12)


def test_cold_graupel_cloud_collection_uses_wrf_threshold_and_joint_cap():
    import cupy as cp

    from gpuwm.core.thompson import launch_cold_cloud_source_network

    # Cell 0 forces raw graupel collection above the entire cloud-water
    # budget.  The freezing table independently requests that entire budget,
    # so WRF's one joint cap must allocate more than half of the cloud to the
    # larger, *uncapped* graupel rate.  An erroneous individual rc/dt cap on
    # graupel makes the two rates equal and leaves the graupel share below
    # one half once the small autoconversion sink is included.
    shape = (3, 1, 1)
    dt = np.float32(100.0)
    temperature_host = np.float32(260.0)
    pressure_host = np.float32(80000.0)
    qv_host = np.float32(0.002)
    rho = np.float32(
        np.float32(0.622) * pressure_host
        / (np.float32(287.04) * temperature_host
           * np.float32(qv_host + np.float32(0.622))))
    cloud_mass = np.float32(1.0e-3)

    temperature = cp.full(shape, temperature_host, dtype=cp.float32)
    pressure = cp.full(shape, pressure_host, dtype=cp.float32)
    qv = cp.full(shape, qv_host, dtype=cp.float32)
    qc = cp.full(shape, np.float32(cloud_mass / rho), dtype=cp.float32)
    qr = cp.zeros(shape, dtype=cp.float32)
    nr = cp.zeros(shape, dtype=cp.float32)
    qi = cp.zeros(shape, dtype=cp.float32)
    ni = cp.zeros(shape, dtype=cp.float32)
    qs = cp.zeros(shape, dtype=cp.float32)
    qg_mass = np.asarray(
        [1.0e-2, 0.99e-6, 1.01e-6], dtype=np.float32)
    qg = cp.asarray((qg_mass / rho)[:, None, None], dtype=cp.float32)

    rain_cloud_efficiency = cp.zeros(
        (100, 100), dtype=cp.float64, order="F")
    table_shape = (37, 100, 45, 55)
    # Every addressed mass record exceeds cloud_mass, so the table process
    # is independently bounded to cloud_mass/dt.  A zero number table keeps
    # this mass-allocation test free of an unrelated ice-number source.
    cloud_to_ice_mass = cp.ones(
        table_shape, dtype=cp.float64, order="F")
    cloud_to_ice_number = cp.zeros_like(cloud_to_ice_mass, order="F")

    qc_before = qc.copy()
    qr_before = qr.copy()
    qi_before = qi.copy()
    qg_before = qg.copy()
    launch_cold_cloud_source_network(
        qc, qr, nr, qi, ni, qs, qg, temperature, pressure, qv,
        rain_cloud_efficiency, cloud_to_ice_mass, cloud_to_ice_number,
        float(dt))
    cp.cuda.Stream.null.synchronize()

    cloud_before_mass = cp.asnumpy(qc_before[:, 0, 0]) * rho
    cloud_after_mass = cp.asnumpy(qc[:, 0, 0]) * rho
    rain_gain = cp.asnumpy((qr - qr_before)[:, 0, 0]) * rho
    ice_gain = cp.asnumpy((qi - qi_before)[:, 0, 0]) * rho
    graupel_gain = cp.asnumpy((qg - qg_before)[:, 0, 0]) * rho

    np.testing.assert_allclose(
        cloud_after_mass, 0.0, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(
        rain_gain + ice_gain + graupel_gain,
        cloud_before_mass - cloud_after_mass,
        rtol=3.0e-5, atol=2.0e-10)
    assert graupel_gain[0] > np.float32(0.58) * cloud_before_mass[0]
    assert np.float32(0.34) * cloud_before_mass[0] < ice_gain[0] \
        < np.float32(0.42) * cloud_before_mass[0]
    assert rain_gain[0] < np.float32(0.02) * cloud_before_mass[0]

    # WRF enters prg_gcw only at rg >= 1.e-6 kg m-3.  These cells differ
    # solely across that threshold: the sub-threshold qg must be bitwise held,
    # while the super-threshold cell receives a resolved positive transfer.
    assert qg_mass[1] < np.float32(1.0e-6) < qg_mass[2]
    np.testing.assert_array_equal(graupel_gain[1], np.float32(0.0))
    assert graupel_gain[2] > np.float32(1.0e-8)


def test_cold_cloud_source_network_rejects_bad_tables_and_timestep():
    import cupy as cp

    from gpuwm.core.thompson import launch_cold_cloud_source_network

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(10)]
    efficiency = cp.zeros((100, 100), dtype=cp.float64, order="F")
    freezing = cp.zeros(
        (37, 100, 45, 55), dtype=cp.float64, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_cold_cloud_source_network(
            *fields, cp.zeros((100, 100), dtype=cp.float32, order="F"),
            freezing, freezing, 10.0)
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_cold_cloud_source_network(
            *fields, efficiency,
            cp.zeros((37, 100, 45, 55), dtype=cp.float64),
            freezing, 10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        launch_cold_cloud_source_network(
            *fields, efficiency, freezing, freezing, float("nan"))


def test_ice_nucleation_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_ice_nucleation,
        launch_ice_sedimentation,
    )

    with (_ORACLE / "ice-nuc-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_ice_nucleation(qi, ni, temperature, pressure, qv, 10.0)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz, *surfaces, 10.0)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 2.0e-6, 2.0e-5),
            (qv, "qv", 3.0e-6, 2.0e-10),
            (qi, "qi", 8.0e-6, 2.0e-12),
            (ni, "ni_per_kg", 8.0e-6, 2.0e-2)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)
    with (_ORACLE / "ice-nuc-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "snownc_mm", "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=8.0e-6, atol=2.0e-15)


def test_ice_nucleation_rejects_bad_dt():
    import cupy as cp

    from gpuwm.core.thompson import launch_ice_nucleation

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(5)]
    with pytest.raises(ValueError, match="positive"):
        launch_ice_nucleation(*fields, 0.0)


def test_rain_freezing_network_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_freezing,
        launch_rain_sedimentation,
    )

    with (_ORACLE / "rain-freeze-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Only five rain-mass bins are exercised by this hash-pinned column.
    # Populate those exact entries from the canonical WRF-v4.6.1 FP64
    # freezeH2O.dat asset while retaining production table shapes/layout.
    table_shape = (37, 37, 45, 55)
    table_hosts = [np.zeros(table_shape, dtype=np.float64, order="F")
                   for _ in range(4)]
    sparse_entries = {
        4: (2.903450965056925e-06, 13382.932790268971,
            1.0616349448173607e-10, 0.004679806528733971),
        11: (1.7865994298969045e-05, 48104.600519023676,
             5.761270659401057e-08, 2.3288149788678725),
        18: (8.31181207021196e-05, 121902.44986207546,
             3.731609726336162e-06, 133.32415824762236),
        20: (0.00017495940530277157, 184958.91703563268,
             2.1319604154352404e-05, 697.3137707913168),
        21: (0.0002491045657123386, 224700.4842725426,
             4.689575873756551e-05, 1455.6627751760439),
    }
    for one_based_mass_bin, values in sparse_entries.items():
        for table, value in zip(table_hosts, values):
            table[one_based_mass_bin - 1, 36, 42, 27] = value
    tables = [cp.asarray(table, order="F") for table in table_hosts]

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)

    launch_rain_freezing(
        qr, nr, qi, ni, qg, temperature, pressure, qv, *tables, 10.0)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density)
    # Classic WRF retains the entry-time L_qg mask through this call.  Since
    # the oracle enters with no graupel, rain-freezing products become
    # sedimentation-eligible only on the next microphysics invocation.
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 5.0),
            (qi, "qi", 1.0e-5, 3.0e-12),
            (ni, "ni_per_kg", 1.0e-5, 5.0),
            (qg, "qg", 1.0e-5, 3.0e-12)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "rain-freeze-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_rain_freezing_rejects_noncanonical_table_layout():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_freezing

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(8)]
    canonical = cp.zeros((37, 37, 45, 55), dtype=cp.float64, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_rain_freezing(
            *fields,
            cp.zeros((37, 37, 45, 55), dtype=cp.float32, order="F"),
            canonical, canonical, canonical, 10.0)
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_rain_freezing(
            *fields, canonical,
            cp.zeros((37, 37, 45, 55), dtype=cp.float64),
            canonical, canonical, 10.0)


def test_cloud_freezing_evaporation_network_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cloud_freezing,
        launch_cloud_saturation_adjust,
        launch_ice_sedimentation,
        launch_rain_sedimentation,
        launch_warm_autoconversion,
    )

    with (_ORACLE / "cloud-freeze-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    table_shape = (37, 100, 45, 55)
    mass_host = np.zeros(table_shape, dtype=np.float64, order="F")
    number_host = np.zeros_like(mass_host, order="F")
    sparse_entries = {
        4: (2.6940401262332944e-09, 53559.56982915825),
        11: (6.717408792055978e-08, 267424.3388722158),
        18: (1.3448398760270702e-06, 1196386.0302252292),
        20: (6.525313212406292e-06, 2634578.718132887),
        21: (1.445374000186087e-05, 3919784.0527078197),
    }
    for one_based_mass_bin, (mass, number) in sparse_entries.items():
        mass_host[one_based_mass_bin - 1, 65, 32, 27] = mass
        number_host[one_based_mass_bin - 1, 65, 32, 27] = number
    cloud_to_ice_mass = cp.asarray(mass_host, order="F")
    cloud_to_ice_number = cp.asarray(number_host, order="F")

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qc)

    # WRF diagnoses both process rates from the incoming cloud.  The tiny
    # autoconversion transfer does not cross a freezing-table mass bin, so
    # applying it first preserves their exact simultaneous result.
    launch_warm_autoconversion(
        qc, qr, nr, temperature, pressure, qv, 10.0)
    launch_cloud_freezing(
        qc, qi, ni, temperature, pressure, qv,
        cloud_to_ice_mass, cloud_to_ice_number, 10.0)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=reference_density)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            # The final residual cloud is the subtraction of two O(1e-4)
            # FP32 states; retain WRF's measured sub-5e-10 cancellation
            # budget instead of a misleading relative-only constraint.
            (qc, "qc", 1.0e-5, 5.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 2.0e-3),
            (qi, "qi", 1.0e-5, 3.0e-12),
            (ni, "ni_per_kg", 1.0e-5, 5.0)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "cloud-freeze-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_cloud_freezing_rejects_noncanonical_table_layout():
    import cupy as cp

    from gpuwm.core.thompson import launch_cloud_freezing

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(6)]
    canonical = cp.zeros((37, 100, 45, 55), dtype=cp.float64, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_cloud_freezing(
            *fields,
            cp.zeros((37, 100, 45, 55), dtype=cp.float32, order="F"),
            canonical, 10.0)
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_cloud_freezing(
            *fields, canonical,
            cp.zeros((37, 100, 45, 55), dtype=cp.float64), 10.0)


def test_graupel_riming_evaporation_network_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cloud_saturation_adjust,
        launch_graupel_cloud_riming,
        launch_graupel_sedimentation,
        launch_ice_sedimentation,
        launch_rain_sedimentation,
        launch_warm_autoconversion,
    )

    with (_ORACLE / "graupel-rime-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qc)

    launch_warm_autoconversion(
        qc, qr, nr, temperature, pressure, qv, 10.0)
    launch_graupel_cloud_riming(
        qc, qg, qi, ni, temperature, pressure, qv, 10.0)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=reference_density)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 1.0e-5, 5.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 2.0e-3),
            (qi, "qi", 1.0e-5, 3.0e-12),
            (ni, "ni_per_kg", 1.0e-5, 5.0),
            (qg, "qg", 1.0e-5, 3.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "graupel-rime-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_graupel_riming_rejects_invalid_timestep_and_shape():
    import cupy as cp

    from gpuwm.core.thompson import launch_graupel_cloud_riming

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(7)]
    with pytest.raises(ValueError, match="finite and positive"):
        launch_graupel_cloud_riming(*fields, 0.0)
    with pytest.raises(ValueError, match="must have shape"):
        launch_graupel_cloud_riming(
            *fields[:-1], cp.zeros((2, 2, 3), dtype=cp.float32), 10.0)


def test_snow_riming_evaporation_network_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cloud_saturation_adjust,
        launch_rain_sedimentation,
        launch_snow_cloud_riming,
        launch_snow_sedimentation,
        launch_warm_autoconversion,
    )

    with (_ORACLE / "snow-rime-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qs = volume("qs")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qc)
    reference_temperature = cp.empty_like(qc)

    launch_warm_autoconversion(
        qc, qr, nr, temperature, pressure, qv, 10.0)
    launch_snow_cloud_riming(
        qc, qs, temperature, pressure, qv, 10.0)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 1.0e-5, 5.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 2.0e-3),
            (qs, "qs", 1.0e-5, 3.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "snow-rime-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_snow_riming_rejects_invalid_timestep_and_shape():
    import cupy as cp

    from gpuwm.core.thompson import launch_snow_cloud_riming

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(5)]
    with pytest.raises(ValueError, match="finite and positive"):
        launch_snow_cloud_riming(*fields, float("nan"))
    with pytest.raises(ValueError, match="must have shape"):
        launch_snow_cloud_riming(
            *fields[:-1], cp.zeros((2, 2, 3), dtype=cp.float32), 10.0)


def test_snow_rime_conversion_network_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_classic_graupel_number_init,
        launch_cloud_saturation_adjust,
        launch_frozen_vapor_network,
        launch_rain_sedimentation,
        launch_snow_rime_conversion,
        launch_snow_sedimentation,
        launch_warm_autoconversion,
    )

    with (_ORACLE / "snow-rime-convert-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    velocity_boost = cp.empty_like(qs)
    reference_density = cp.empty_like(qs)
    reference_temperature = cp.empty_like(qs)
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)

    launch_warm_autoconversion(
        qc, qr, nr, temperature, pressure, qv, 10.0)
    launch_snow_rime_conversion(
        qc, qs, qg, temperature, pressure, qv, velocity_boost, 10.0)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature,
        velocity_boost=velocity_boost)
    # Conversion-created graupel has a false entry-time L_qg mask and does
    # not sediment until the next WRF call.
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 1.0e-5, 5.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 5.0),
            (qs, "qs", 1.0e-5, 3.0e-10),
            (qg, "qg", 1.0e-5, 3.0e-12)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    expected_boost = np.ones(24, dtype=np.float32)
    expected_boost[:6] = np.float32(1.4920001)
    np.testing.assert_allclose(
        cp.asnumpy(velocity_boost[:, 0, 0]), expected_boost,
        rtol=0.0, atol=2.0e-7)
    with (_ORACLE / "snow-rime-convert-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)
    assert float(surface["graupelnc_mm"]) == 0.0
    assert float(surface["graupelncv_mm"]) == 0.0

    # Exercise the production fused cold-source ABI on the same direct-WRF
    # vector.  Zero tables are authoritative here because the case has no
    # incoming cloud ice/rain and therefore reads none of those records.
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    velocity_boost = cp.empty_like(qs)
    graupel_number_shadow = cp.empty_like(qg)
    reference_density = cp.empty_like(qs)
    reference_temperature = cp.empty_like(qs)
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    ice_table = cp.zeros((64, 55), dtype=cp.float64, order="F")
    cloud_efficiency = cp.zeros(
        (100, 100), dtype=cp.float64, order="F")
    cloud_freezing = cp.zeros(
        (37, 100, 45, 55), dtype=cp.float64, order="F")

    launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, graupel_number_shadow)
    launch_frozen_vapor_network(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        ice_table, ice_table, ice_table, 10.0,
        qc=qc, rain_cloud_efficiency=cloud_efficiency,
        cloud_freezing_tables=(cloud_freezing, cloud_freezing),
        graupel_number_shadow=graupel_number_shadow,
        snow_velocity_boost=velocity_boost)
    cp.cuda.Stream.null.synchronize()

    # Direct analytic png_scw oracle. With zero incoming qg and no other
    # graupel source, qg after the fused source is exactly prg_scw*dt/rho;
    # WRF defines png_scw=prg_scw*smo(0)/rs.  The number shadow therefore
    # equals qg*smo(0)/rs before its post-fallout bounds are applied.
    temp0 = host(before, "temp_k")
    qv0 = host(before, "qv")
    pressure0 = host(before, "p_pa")
    rho0 = np.float32(0.622) * pressure0 / (
        np.float32(287.04) * temp0 * (qv0 + np.float32(0.622)))
    snow_mass0 = host(before, "qs") * rho0
    tc0 = np.minimum(np.float32(-0.1), temp0 - np.float32(273.15))
    sa = np.asarray((
        5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
        0.31255, 0.000204, 0.003199, 0.0, -0.015952),
        dtype=np.float32)
    sb = np.asarray((
        0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
        0.060366, 0.000079, 0.000594, 0.0, -0.003577),
        dtype=np.float32)
    loga0 = sa[0] + sa[1] * tc0 + sa[4] * tc0 * tc0 \
        + sa[8] * tc0 * tc0 * tc0
    b0 = sb[0] + sb[1] * tc0 + sb[4] * tc0 * tc0 \
        + sb[8] * tc0 * tc0 * tc0
    moment0 = np.power(np.float32(10.0), loga0) * np.power(
        snow_mass0 / np.float32(0.069), b0)
    source_qg = cp.asnumpy(qg[:, 0, 0])
    expected_number = np.where(
        source_qg > 0.0,
        source_qg * moment0 / np.maximum(snow_mass0, np.float32(1.0e-30)),
        np.float32(0.0))
    np.testing.assert_allclose(
        cp.asnumpy(graupel_number_shadow[:, 0, 0]), expected_number,
        rtol=2.5e-5, atol=2.0e-3)

    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature,
        velocity_boost=velocity_boost)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 1.0e-5, 5.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 5.0),
            (qs, "qs", 1.0e-5, 3.0e-10),
            (qg, "qg", 1.0e-5, 3.0e-12)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        cp.asnumpy(velocity_boost[:, 0, 0]), expected_boost,
        rtol=0.0, atol=2.0e-7)


def test_snow_rime_conversion_rejects_invalid_composition_contract():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_snow_rime_conversion,
        launch_snow_sedimentation,
    )

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(7)]
    with pytest.raises(ValueError, match="finite and positive"):
        launch_snow_rime_conversion(*fields, 0.0)
    with pytest.raises(ValueError, match="must have shape"):
        launch_snow_rime_conversion(
            *fields[:-1], cp.zeros((2, 2, 3), dtype=cp.float32), 10.0)

    volume_fields = [
        cp.zeros((2, 1, 1), dtype=cp.float32) for _ in range(5)
    ]
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]
    with pytest.raises(ValueError, match="requires reference_temperature"):
        launch_snow_sedimentation(
            *volume_fields, *surfaces, 10.0,
            velocity_boost=cp.ones((2, 1, 1), dtype=cp.float32))


def test_snow_ice_collection_and_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_ice_sedimentation,
        launch_snow_ice_collection,
        launch_snow_sedimentation,
    )

    with (_ORACLE / "snow-ice-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)

    launch_snow_ice_collection(
        qi, ni, qs, temperature, pressure, qv, 10.0)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 0.0, 0.0),
            (qv, "qv", 0.0, 0.0),
            (qi, "qi", 1.0e-5, 3.0e-12),
            (ni, "ni_per_kg", 1.0e-5, 5.0),
            (qs, "qs", 1.0e-5, 3.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "snow-ice-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_snow_ice_collection_rejects_invalid_timestep_and_shape():
    import cupy as cp

    from gpuwm.core.thompson import launch_snow_ice_collection

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(6)]
    with pytest.raises(ValueError, match="finite and positive"):
        launch_snow_ice_collection(*fields, -1.0)
    with pytest.raises(ValueError, match="must have shape"):
        launch_snow_ice_collection(
            *fields[:-1], cp.zeros((2, 2, 3), dtype=cp.float32), 10.0)


def test_rain_ice_collection_and_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_ice_collection,
        launch_rain_sedimentation,
    )

    with (_ORACLE / "rain-ice-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)

    launch_rain_ice_collection(
        qr, nr, qi, ni, qg, temperature, pressure, qv, 10.0)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density)
    # Collision products enter with L_qg false and are not eligible for
    # graupel fallout until the next WRF microphysics invocation.
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 5.0),
            (qi, "qi", 1.0e-5, 3.0e-12),
            (ni, "ni_per_kg", 1.0e-5, 5.0),
            (qg, "qg", 1.0e-5, 3.0e-12)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "rain-ice-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_rain_ice_collection_rejects_invalid_timestep_and_shape():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_ice_collection

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(8)]
    with pytest.raises(ValueError, match="finite and positive"):
        launch_rain_ice_collection(*fields, float("nan"))
    with pytest.raises(ValueError, match="must have shape"):
        launch_rain_ice_collection(
            *fields[:-1], cp.zeros((2, 2, 3), dtype=cp.float32), 10.0)


def test_rain_snow_collection_and_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_rain_snow_collection,
        launch_snow_sedimentation,
    )

    with (_ORACLE / "rain-snow-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Exact entries reached by the seven active levels of this direct WRF
    # column.  They are extracted from the hash-pinned canonical
    # qr_acr_qsV2.dat asset; production accepts all twelve complete tables.
    sparse_entries = (
        ((19, 1, 20, 20), (
            9.87532202545891e-08, 5.953287972263036e-07,
            2.208143430291573e-07, 1.2867691260365802e-07,
            0.0, 0.0, 2.019440607867497e-08,
            2.5029334230552037e-06, 2.7589255153456795,
            2.07174723249674, 0.0, 13.321829230428476)),
        ((20, 1, 21, 21), (
            2.389863725651701e-07, 1.3352172469596492e-06,
            8.952049618560682e-07, 4.405227329200509e-07,
            0.0, 0.0, 8.150805439515519e-08,
            1.4421270168146614e-05, 5.916051298481054,
            5.476180470021909, 0.0, 46.60521611987131)),
        ((20, 1, 20, 21), (
            2.943198381089359e-07, 1.8135978830826924e-06,
            9.24813713654374e-07, 4.7069393748953727e-07,
            0.0, 0.0, 6.349498005857823e-08,
            1.0551746052582787e-05, 6.504305686711739,
            5.221841101302199, 0.0, 33.93240905952725)),
        ((19, 1, 19, 19), (
            4.937661012729455e-08, 2.976643986131518e-07,
            1.1040717151457865e-07, 6.433845630182901e-08,
            0.0, 0.0, 1.0097203039337485e-08,
            1.2514667115276018e-06, 1.3794627576728398,
            1.03587361624837, 0.0, 6.660914615214238)),
        ((17, 1, 16, 16), (
            2.9274803186760153e-08, 1.8348570249992735e-07,
            5.581210393965682e-08, 3.4032362061841657e-08,
            0.0, 0.0, 5.091779625978452e-09,
            5.649472295299869e-07, 0.8702195351722725,
            0.5998100118986998, 0.0, 3.5341135595206055)),
        ((12, 1, 10, 11), (
            4.422629689489238e-09, 4.573341402962764e-08,
            3.1641234437483367e-09, 2.333794719222374e-09,
            0.0, 0.0, 2.024592293754066e-10,
            1.3142084930918329e-08, 0.1691315593650582,
            0.0555441746580144, 0.0, 0.1652768757524083)),
        ((9, 1, 3, 3), (
            1.575756941530423e-10, 1.658459371002552e-09,
            7.498173871635878e-11, 6.151482996632881e-11,
            0.0, 0.0, 7.020179924610058e-12,
            2.9272790940072794e-10, 0.009874503207460989,
            0.0027991557211046437, 0.0, 0.008459955334495376)),
    )
    table_hosts = [
        np.zeros((37, 9, 37, 37), dtype=np.float64, order="F")
        for _ in range(12)
    ]
    for one_based_index, values in sparse_entries:
        index = tuple(value - 1 for value in one_based_index)
        for table, value in zip(table_hosts, values, strict=True):
            table[index] = value
    tables = tuple(cp.asarray(table, order="F") for table in table_hosts)

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)
    reference_temperature = cp.empty_like(qr)

    launch_rain_snow_collection(
        qr, nr, qs, qg, temperature, pressure, qv, tables, 10.0)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    # Collision products enter with L_qg false, so graupel does not fall
    # until the next WRF microphysics invocation.
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 5.0),
            (qs, "qs", 1.0e-5, 3.0e-10),
            (qg, "qg", 1.0e-5, 3.0e-12)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "rain-snow-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)
    assert float(surface["graupelnc_mm"]) == 0.0
    assert float(surface["graupelncv_mm"]) == 0.0


def test_rain_snow_collection_rejects_noncanonical_tables():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_snow_collection

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(7)]
    table = cp.zeros((37, 9, 37, 37), dtype=cp.float64, order="F")
    tables = (table,) * 12
    with pytest.raises(ValueError, match="twelve|12"):
        launch_rain_snow_collection(*fields, tables[:-1], 10.0)
    bad_dtype = list(tables)
    bad_dtype[0] = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float32, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_rain_snow_collection(*fields, bad_dtype, 10.0)
    bad_layout = list(tables)
    bad_layout[0] = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float64, order="C")
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_rain_snow_collection(*fields, bad_layout, 10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        launch_rain_snow_collection(*fields, tables, 0.0)


def test_rain_graupel_collection_and_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_graupel_sedimentation,
        launch_rain_evaporation,
        launch_rain_graupel_collection,
        launch_rain_sedimentation,
    )

    with (_ORACLE / "rain-graupel-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Exact entries reached by the seven active levels.  The fourth index
    # includes WRF-v4.6.1's observed four-slab classic-mp8 density alias.
    sparse_entries = (
        ((19, 19, 5, 20, 20), (
            2.046240565630309e-08, 2.6926901952091645e-09,
            2.8760191662703972e-08, 0.03017018819507457,
            7.249724881318835)),
        ((19, 20, 5, 21, 21), (
            4.5376215440873715e-08, 5.735751677028559e-09,
            8.861937053824464e-08, 0.05299800193669387,
            18.442432023026445)),
        ((19, 20, 5, 20, 21), (
            5.0274062554536646e-08, 6.470869664651708e-09,
            8.709867173143655e-08, 0.054450480820851736,
            16.656244283557406)),
        ((19, 19, 5, 19, 19), (
            7.204407043806906e-09, 8.957053584306925e-10,
            1.5258304685206622e-08, 0.013625804539537559,
            5.03595196896582)),
        ((19, 17, 5, 16, 16), (
            6.843192121309458e-09, 9.283754285854457e-10,
            7.839581332683137e-09, 0.010242868384790238,
            1.964881053185283)),
        ((19, 12, 5, 10, 11), (
            1.4397065050265788e-09, 2.2584606798532013e-10,
            7.332403048982663e-10, 0.002456961401039669,
            0.19101370458913253)),
        ((16, 9, 5, 3, 3), (
            1.052060257384068e-10, 1.9137686015832566e-11,
            2.8417313543521983e-11, 0.00021250754338861418,
            0.008067907620002177)),
    )
    table_hosts = [
        np.zeros((37, 37, 1, 37, 37), dtype=np.float64, order="F")
        for _ in range(5)
    ]
    for one_based_index, values in sparse_entries:
        g1, graupel, density, rain_intercept, rain = one_based_index
        assert density == 5
        index = (g1 - 1, graupel - 1, 0,
                 rain_intercept - 1 + density - 1, rain - 1)
        for table, value in zip(table_hosts, values, strict=True):
            table[index] = value
    tables = tuple(cp.asarray(table, order="F") for table in table_hosts)

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)

    launch_rain_graupel_collection(
        qr, nr, qg, temperature, pressure, qv, tables, 10.0)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, 10.0,
        reference_density=reference_density)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 5.0),
            (qg, "qg", 1.0e-5, 3.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / "rain-graupel-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)
    assert float(surface["snownc_mm"]) == 0.0
    assert float(surface["snowncv_mm"]) == 0.0


def test_rain_graupel_collection_rejects_noncanonical_tables():
    import cupy as cp

    from gpuwm.core.thompson import launch_rain_graupel_collection

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(6)]
    table = cp.zeros((37, 37, 1, 37, 37), dtype=cp.float64, order="F")
    tables = (table,) * 5
    with pytest.raises(ValueError, match="five|5"):
        launch_rain_graupel_collection(*fields, tables[:-1], 10.0)
    bad_dtype = list(tables)
    bad_dtype[0] = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float32, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_rain_graupel_collection(*fields, bad_dtype, 10.0)
    bad_layout = list(tables)
    bad_layout[0] = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float64, order="C")
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_rain_graupel_collection(*fields, bad_layout, 10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        launch_rain_graupel_collection(*fields, tables, -1.0)


def test_cold_rain_snow_graupel_network_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cold_rain_snow_graupel_network,
        launch_graupel_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
    )

    scenario = "rain-snow-graupel-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # Exact canonical qr_acr_qsV2 entries reached by the seven active levels.
    rain_snow_entries = (
        ((19, 1, 20, 20), (
            9.87532202545891e-08, 5.953287972263036e-07,
            2.208143430291573e-07, 1.2867691260365802e-07,
            0.0, 0.0, 2.019440607867497e-08,
            2.5029334230552037e-06, 2.7589255153456795,
            2.07174723249674, 0.0, 13.321829230428476)),
        ((20, 1, 21, 21), (
            2.389863725651701e-07, 1.3352172469596492e-06,
            8.952049618560682e-07, 4.405227329200509e-07,
            0.0, 0.0, 8.150805439515519e-08,
            1.4421270168146614e-05, 5.916051298481054,
            5.476180470021909, 0.0, 46.60521611987131)),
        ((20, 1, 20, 21), (
            2.943198381089359e-07, 1.8135978830826924e-06,
            9.24813713654374e-07, 4.7069393748953727e-07,
            0.0, 0.0, 6.349498005857823e-08,
            1.0551746052582787e-05, 6.504305686711739,
            5.221841101302199, 0.0, 33.93240905952725)),
        ((19, 1, 19, 19), (
            4.937661012729455e-08, 2.976643986131518e-07,
            1.1040717151457865e-07, 6.433845630182901e-08,
            0.0, 0.0, 1.0097203039337485e-08,
            1.2514667115276018e-06, 1.3794627576728398,
            1.03587361624837, 0.0, 6.660914615214238)),
        ((17, 1, 16, 16), (
            2.9274803186760153e-08, 1.8348570249992735e-07,
            5.581210393965682e-08, 3.4032362061841657e-08,
            0.0, 0.0, 5.091779625978452e-09,
            5.649472295299869e-07, 0.8702195351722725,
            0.5998100118986998, 0.0, 3.5341135595206055)),
        ((12, 1, 10, 11), (
            4.422629689489238e-09, 4.573341402962764e-08,
            3.1641234437483367e-09, 2.333794719222374e-09,
            0.0, 0.0, 2.024592293754066e-10,
            1.3142084930918329e-08, 0.1691315593650582,
            0.0555441746580144, 0.0, 0.1652768757524083)),
        ((9, 1, 3, 3), (
            1.575756941530423e-10, 1.658459371002552e-09,
            7.498173871635878e-11, 6.151482996632881e-11,
            0.0, 0.0, 7.020179924610058e-12,
            2.9272790940072794e-10, 0.009874503207460989,
            0.0027991557211046437, 0.0, 0.008459955334495376)),
    )
    rain_snow_hosts = [
        np.zeros((37, 9, 37, 37), dtype=np.float64, order="F")
        for _ in range(12)
    ]
    for one_based_index, values in rain_snow_entries:
        index = tuple(value - 1 for value in one_based_index)
        for table, value in zip(rain_snow_hosts, values, strict=True):
            table[index] = value
    rain_snow_tables = tuple(
        cp.asarray(table, order="F") for table in rain_snow_hosts)

    # Exact canonical qr_acr_qg_V4 entries reached by those same levels.
    # The density coordinate records WRF-v4.6.1's four-slab legacy alias.
    rain_graupel_entries = (
        ((19, 19, 5, 20, 20), (
            2.046240565630309e-08, 2.6926901952091645e-09,
            2.8760191662703972e-08, 0.03017018819507457,
            7.249724881318835)),
        ((19, 20, 5, 21, 21), (
            4.5376215440873715e-08, 5.735751677028559e-09,
            8.861937053824464e-08, 0.05299800193669387,
            18.442432023026445)),
        ((19, 20, 5, 20, 21), (
            5.0274062554536646e-08, 6.470869664651708e-09,
            8.709867173143655e-08, 0.054450480820851736,
            16.656244283557406)),
        ((19, 19, 5, 19, 19), (
            7.204407043806906e-09, 8.957053584306925e-10,
            1.5258304685206622e-08, 0.013625804539537559,
            5.03595196896582)),
        ((19, 17, 5, 16, 16), (
            6.843192121309458e-09, 9.283754285854457e-10,
            7.839581332683137e-09, 0.010242868384790238,
            1.964881053185283)),
        ((19, 12, 5, 10, 11), (
            1.4397065050265788e-09, 2.2584606798532013e-10,
            7.332403048982663e-10, 0.002456961401039669,
            0.19101370458913253)),
        ((16, 9, 5, 3, 3), (
            1.052060257384068e-10, 1.9137686015832566e-11,
            2.8417313543521983e-11, 0.00021250754338861418,
            0.008067907620002177)),
    )
    rain_graupel_hosts = [
        np.zeros((37, 37, 1, 37, 37), dtype=np.float64, order="F")
        for _ in range(5)
    ]
    for one_based_index, values in rain_graupel_entries:
        g1, graupel, density, rain_intercept, rain = one_based_index
        assert density == 5
        index = (g1 - 1, graupel - 1, 0,
                 rain_intercept - 1 + density - 1, rain - 1)
        for table, value in zip(
                rain_graupel_hosts, values, strict=True):
            table[index] = value
    rain_graupel_tables = tuple(
        cp.asarray(table, order="F") for table in rain_graupel_hosts)

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)
    reference_temperature = cp.empty_like(qr)

    launch_cold_rain_snow_graupel_network(
        qr, nr, qs, qg, temperature, pressure, qv,
        rain_snow_tables, rain_graupel_tables, 10.0)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density,
        reference_temperature=reference_temperature)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-12),
            (nr, "nr_per_kg", 1.0e-5, 5.0),
            (qs, "qs", 1.0e-5, 3.0e-10),
            (qg, "qg", 1.0e-5, 3.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.0e-5, atol=3.0e-12)


def test_cold_rain_network_applies_shared_cap_once_not_sequentially():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cold_rain_snow_graupel_network,
        launch_rain_graupel_collection,
        launch_rain_snow_collection,
    )

    # Constant synthetic tables deliberately make each independently bounded
    # collision family consume about 70% of the incoming rain.  Their sum
    # therefore activates the grouped WRF rain bound, while either old
    # sequential order exposes its second process to a provisional state.
    rain_snow_rates = (
        0.0, 4.0e-6, 0.0, 4.0e-6,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    rain_snow_tables = tuple(
        cp.full((37, 9, 37, 37), value, dtype=cp.float64, order="F")
        for value in rain_snow_rates
    )
    rain_graupel_rates = (0.0, 8.0e-6, 0.0, 0.0, 0.0)
    rain_graupel_tables = tuple(
        cp.full((37, 37, 1, 37, 37), value,
                dtype=cp.float64, order="F")
        for value in rain_graupel_rates
    )

    shape = (1,)
    base = {
        "qr": cp.full(shape, 1.0e-4, dtype=cp.float32),
        "nr": cp.full(
            shape,
            1.0e-4 * (3.672 / 500.0e-6) ** 3
            / (np.pi * 1000.0),
            dtype=cp.float32),
        "qs": cp.full(shape, 8.0e-5, dtype=cp.float32),
        "qg": cp.full(shape, 8.0e-5, dtype=cp.float32),
        "temperature": cp.full(shape, 270.0, dtype=cp.float32),
        "pressure": cp.full(shape, 90000.0, dtype=cp.float32),
        "qv": cp.full(shape, 0.003, dtype=cp.float32),
    }

    def fresh_state():
        return {name: value.copy() for name, value in base.items()}

    def run_fused():
        state = fresh_state()
        launch_cold_rain_snow_graupel_network(
            state["qr"], state["nr"], state["qs"], state["qg"],
            state["temperature"], state["pressure"], state["qv"],
            rain_snow_tables, rain_graupel_tables, 10.0)
        return state

    fused_a = run_fused()
    fused_b = run_fused()
    for name in ("qr", "nr", "qs", "qg", "temperature"):
        np.testing.assert_array_equal(
            cp.asnumpy(fused_a[name]), cp.asnumpy(fused_b[name]))
        assert bool(cp.isfinite(fused_a[name]).all())
        assert float(fused_a[name].min()) >= 0.0
    assert float(fused_a["qr"][0]) <= 2.0e-11

    snow_then_graupel = fresh_state()
    launch_rain_snow_collection(
        snow_then_graupel["qr"], snow_then_graupel["nr"],
        snow_then_graupel["qs"], snow_then_graupel["qg"],
        snow_then_graupel["temperature"],
        snow_then_graupel["pressure"], snow_then_graupel["qv"],
        rain_snow_tables, 10.0)
    launch_rain_graupel_collection(
        snow_then_graupel["qr"], snow_then_graupel["nr"],
        snow_then_graupel["qg"], snow_then_graupel["temperature"],
        snow_then_graupel["pressure"], snow_then_graupel["qv"],
        rain_graupel_tables, 10.0)

    graupel_then_snow = fresh_state()
    launch_rain_graupel_collection(
        graupel_then_snow["qr"], graupel_then_snow["nr"],
        graupel_then_snow["qg"], graupel_then_snow["temperature"],
        graupel_then_snow["pressure"], graupel_then_snow["qv"],
        rain_graupel_tables, 10.0)
    launch_rain_snow_collection(
        graupel_then_snow["qr"], graupel_then_snow["nr"],
        graupel_then_snow["qs"], graupel_then_snow["qg"],
        graupel_then_snow["temperature"],
        graupel_then_snow["pressure"], graupel_then_snow["qv"],
        rain_snow_tables, 10.0)
    cp.cuda.Stream.null.synchronize()

    # Both legacy orders consume the rain, but they assign materially
    # different graupel because the second kernel sees provisional rain.
    assert not np.isclose(
        float(snow_then_graupel["qg"][0]),
        float(graupel_then_snow["qg"][0]),
        rtol=0.0, atol=1.0e-10)
    assert not np.isclose(
        float(fused_a["qg"][0]),
        float(snow_then_graupel["qg"][0]),
        rtol=0.0, atol=1.0e-10)
    assert not np.isclose(
        float(fused_a["qg"][0]),
        float(graupel_then_snow["qg"][0]),
        rtol=0.0, atol=1.0e-10)


def test_complete_cold_rain_source_network_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cold_rain_source_network,
        launch_graupel_sedimentation,
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
    )

    scenario = "rain-ice-graupel-overlap"
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_],
                          dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # This focused column has no snow.  A canonical-shaped zero table can be
    # shared across all twelve validated rain/snow arguments.
    rain_snow_zero = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float64, order="F")
    rain_snow_tables = (rain_snow_zero,) * 12

    rain_graupel_entries = (
        ((19, 19, 5, 20, 20), (
            2.046240565630309e-08, 2.6926901952091645e-09,
            2.8760191662703972e-08, 0.03017018819507457,
            7.249724881318835)),
        ((19, 20, 5, 21, 21), (
            4.5376215440873715e-08, 5.735751677028559e-09,
            8.861937053824464e-08, 0.05299800193669387,
            18.442432023026445)),
        ((19, 19, 5, 19, 20), (
            2.3041668206226197e-08, 3.101599718152162e-09,
            2.80964426598413e-08, 0.031146049626146786,
            6.431321343628986)),
        ((19, 18, 5, 16, 17), (
            8.881497498128415e-09, 1.209694776802273e-09,
            1.0013155475304397e-08, 0.012228239123788943,
            2.3144612110491893)),
        ((19, 12, 5, 11, 11), (
            1.2960905339528014e-09, 1.9750969716544525e-10,
            7.574105533020767e-10, 0.002393942276342421,
            0.2157154054586159)),
        ((16, 9, 5, 3, 4), (
            1.6121521042326786e-10, 3.1106805402542444e-11,
            3.538820177809372e-11, 0.00029130033065528417,
            0.008824830571789651)),
    )
    rain_graupel_hosts = [
        np.zeros((37, 37, 1, 37, 37), dtype=np.float64, order="F")
        for _ in range(5)
    ]
    for one_based_index, values in rain_graupel_entries:
        g1, graupel, density, rain_intercept, rain = one_based_index
        assert density == 5
        index = (g1 - 1, graupel - 1, 0,
                 rain_intercept - 1 + density - 1, rain - 1)
        for table, value in zip(
                rain_graupel_hosts, values, strict=True):
            table[index] = value
    rain_graupel_tables = tuple(
        cp.asarray(table, order="F") for table in rain_graupel_hosts)

    rain_freezing_entries = (
        ((20, 20, 20, 28), (
            1.2173210518885194e-08, 1.4290755587934971,
            8.860281140058314e-07, 4.609008498583289)),
        ((21, 21, 20, 28), (
            1.82598157783278e-08, 2.1436133381902462,
            1.3290421710087468e-06, 6.9135127478749325)),
        ((20, 19, 20, 28), (
            8.474834058440846e-09, 0.9514909025866307,
            1.4926365265810664e-06, 5.079140447042585)),
        ((17, 16, 20, 28), (
            4.560832086361712e-09, 0.5303779756090053,
            3.9217599919686036e-07, 1.884455770892024)),
        ((11, 11, 20, 28), (
            1.2173210518885193e-09, 0.14290755587934964,
            8.860281140058314e-08, 0.4609008498583289)),
        ((4, 3, 20, 28), (
            2.1089062362886966e-10, 0.024271183296337907,
            2.202925503138714e-08, 0.0964425533740178)),
    )
    rain_freezing_hosts = [
        np.zeros((37, 37, 45, 55), dtype=np.float64, order="F")
        for _ in range(4)
    ]
    for one_based_index, values in rain_freezing_entries:
        index = tuple(value - 1 for value in one_based_index)
        for table, value in zip(
                rain_freezing_hosts, values, strict=True):
            table[index] = value
    rain_freezing_tables = tuple(
        cp.asarray(table, order="F") for table in rain_freezing_hosts)

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)
    reference_density = cp.empty_like(qr)

    launch_cold_rain_source_network(
        qr, nr, qi, ni, qs, qg, temperature, pressure, qv,
        rain_snow_tables, rain_graupel_tables, rain_freezing_tables, 10.0)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 10.0,
        reference_density=reference_density)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, 10.0,
        reference_density=reference_density)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, 10.0,
        reference_density=reference_density, accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0, reference_density=reference_density,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 4.0e-6, 4.0e-5),
            (qv, "qv", 6.0e-6, 4.0e-10),
            (qr, "qr", 1.2e-5, 4.0e-12),
            (nr, "nr_per_kg", 1.2e-5, 6.0),
            (qi, "qi", 1.2e-5, 4.0e-12),
            (ni, "ni_per_kg", 1.2e-5, 6.0),
            (qs, "qs", 0.0, 0.0),
            (qg, "qg", 1.2e-5, 4.0e-10)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)

    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=4.0e-12)


def test_cold_rain_snow_graupel_network_rejects_bad_tables():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_cold_rain_snow_graupel_network,
    )

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(7)]
    rain_snow_table = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float64, order="F")
    rain_graupel_table = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float64, order="F")
    rain_snow_tables = (rain_snow_table,) * 12
    rain_graupel_tables = (rain_graupel_table,) * 5
    with pytest.raises(ValueError, match="rain_snow_tables.*12"):
        launch_cold_rain_snow_graupel_network(
            *fields, rain_snow_tables[:-1], rain_graupel_tables, 10.0)
    with pytest.raises(ValueError, match="rain_graupel_tables.*5"):
        launch_cold_rain_snow_graupel_network(
            *fields, rain_snow_tables, rain_graupel_tables[:-1], 10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        launch_cold_rain_snow_graupel_network(
            *fields, rain_snow_tables, rain_graupel_tables, 0.0)


def test_complete_cold_rain_source_network_rejects_bad_tables():
    import cupy as cp

    from gpuwm.core.thompson import launch_cold_rain_source_network

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(9)]
    rain_snow = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float64, order="F")
    rain_graupel = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float64, order="F")
    rain_freezing = cp.zeros(
        (37, 37, 45, 55), dtype=cp.float64, order="F")
    rain_snow_tables = (rain_snow,) * 12
    rain_graupel_tables = (rain_graupel,) * 5
    rain_freezing_tables = (rain_freezing,) * 4
    with pytest.raises(ValueError, match="rain_snow_tables.*12"):
        launch_cold_rain_source_network(
            *fields, rain_snow_tables[:-1], rain_graupel_tables,
            rain_freezing_tables, 10.0)
    with pytest.raises(ValueError, match="rain_graupel_tables.*5"):
        launch_cold_rain_source_network(
            *fields, rain_snow_tables, rain_graupel_tables[:-1],
            rain_freezing_tables, 10.0)
    with pytest.raises(ValueError, match="rain_freezing_tables.*4"):
        launch_cold_rain_source_network(
            *fields, rain_snow_tables, rain_graupel_tables,
            rain_freezing_tables[:-1], 10.0)
    bad_freezing = list(rain_freezing_tables)
    bad_freezing[0] = cp.zeros(
        (37, 37, 45, 55), dtype=cp.float32, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_cold_rain_source_network(
            *fields, rain_snow_tables, rain_graupel_tables,
            bad_freezing, 10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        launch_cold_rain_source_network(
            *fields, rain_snow_tables, rain_graupel_tables,
            rain_freezing_tables, float("nan"))


def test_full_frozen_vapor_network_rejects_partial_cold_rain_groups():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_frozen_vapor_network,
        launch_frozen_vapor_network_from_owner,
    )

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(9)]
    ice_table = cp.zeros((64, 55), dtype=cp.float64, order="F")
    rain_snow = cp.zeros(
        (37, 9, 37, 37), dtype=cp.float64, order="F")
    rain_graupel = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float64, order="F")
    rain_freezing = cp.zeros(
        (37, 37, 45, 55), dtype=cp.float64, order="F")
    groups = {
        "rain_snow_tables": (rain_snow,) * 12,
        "rain_graupel_tables": (rain_graupel,) * 5,
        "rain_freezing_tables": (rain_freezing,) * 4,
    }
    for name, value in groups.items():
        with pytest.raises(ValueError, match="must be supplied together"):
            launch_frozen_vapor_network(
                *fields, ice_table, ice_table, ice_table, 10.0,
                **{name: value})
    qc = cp.zeros_like(fields[0])
    rain_cloud_efficiency = cp.zeros(
        (100, 100), dtype=cp.float64, order="F")
    cloud_freezing = cp.zeros(
        (37, 100, 45, 55), dtype=cp.float64, order="F")
    cloud_groups = {
        "qc": qc,
        "rain_cloud_efficiency": rain_cloud_efficiency,
        "cloud_freezing_tables": (cloud_freezing,) * 2,
    }
    for name, value in cloud_groups.items():
        with pytest.raises(ValueError, match="must be supplied together"):
            launch_frozen_vapor_network(
                *fields, ice_table, ice_table, ice_table, 10.0,
                **{name: value})
    with pytest.raises(ValueError, match="contain 2 arrays"):
        launch_frozen_vapor_network(
            *fields, ice_table, ice_table, ice_table, 10.0,
            qc=qc, rain_cloud_efficiency=rain_cloud_efficiency,
            cloud_freezing_tables=(cloud_freezing,))
    with pytest.raises(TypeError, match="float64"):
        launch_frozen_vapor_network(
            *fields, ice_table, ice_table, ice_table, 10.0,
            qc=qc, rain_cloud_efficiency=cp.zeros(
                (100, 100), dtype=cp.float32, order="F"),
            cloud_freezing_tables=(cloud_freezing,) * 2)
    with pytest.raises(TypeError, match="verified DeviceClassicTableSet"):
        launch_frozen_vapor_network_from_owner(*fields, object(), 10.0)


def test_ice_autoconversion_plus_fallout_matches_wrf_column():
    import cupy as cp

    from gpuwm.core.thompson import (
        launch_frozen_vapor_network,
        launch_ice_autoconversion,
        launch_ice_sedimentation,
        launch_snow_sedimentation,
    )

    with (_ORACLE / "ice-auto-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = rows[:24]
    after = rows[24:]

    def host(rows_, name):
        return np.asarray([float(row[name]) for row in rows_], dtype=np.float32)

    def volume(name):
        return cp.asarray(host(before, name)[:, None, None])

    # The one active bin from the hash-pinned canonical FP64 tps_iaus and
    # tni_iaus assets.  Production accepts both complete validated tables.
    mass_host = np.zeros((64, 55), dtype=np.float64, order="F")
    number_host = np.zeros_like(mass_host, order="F")
    mass_host[46, 41] = 2.7966921131601065e-06
    number_host[46, 41] = 133.0085783505213
    ice_to_snow_mass = cp.asarray(mass_host, order="F")
    ice_to_snow_number = cp.asarray(number_host, order="F")

    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    dz = volume("dz_m")
    surfaces = [cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]

    launch_ice_autoconversion(
        qi, ni, qs, temperature, pressure, qv,
        ice_to_snow_mass, ice_to_snow_number, 10.0)
    launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz, *surfaces, 10.0)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz, *surfaces, 10.0,
        accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cp.asnumpy(qi[:, 0, 0]), host(after, "qi"),
        rtol=8.0e-6, atol=2.0e-12)
    np.testing.assert_allclose(
        cp.asnumpy(ni[:, 0, 0]), host(after, "ni_per_kg"),
        rtol=8.0e-6, atol=1.0e-2)
    np.testing.assert_allclose(
        cp.asnumpy(qs[:, 0, 0]), host(after, "qs"),
        rtol=8.0e-6, atol=2.0e-12)

    # The fused cold-ice network must retain autoconversion at exact ice
    # saturation, where all vapor and nucleation rates are zero. This also
    # prevents an optimization for zero supersaturation from skipping the
    # non-vapor members of the source group.
    fused_temperature = volume("temp_k")
    fused_pressure = volume("p_pa")
    fused_qv = volume("qv")
    fused_qi = volume("qi")
    fused_ni = volume("ni_per_kg")
    fused_qs = volume("qs")
    fused_qg = cp.zeros_like(fused_qs)
    fused_qr = cp.zeros_like(fused_qs)
    fused_nr = cp.zeros_like(fused_qs)
    fused_surfaces = [
        cp.zeros((1, 1), dtype=cp.float32) for _ in range(4)]
    partition = cp.ones((64, 55), dtype=cp.float64, order="F")
    launch_frozen_vapor_network(
        fused_qi, fused_ni, fused_qs, fused_qg, fused_qr, fused_nr,
        fused_temperature, fused_pressure, fused_qv,
        partition, ice_to_snow_mass, ice_to_snow_number, 10.0)
    launch_ice_sedimentation(
        fused_qi, fused_ni, fused_temperature, fused_pressure, fused_qv, dz,
        *fused_surfaces, 10.0)
    launch_snow_sedimentation(
        fused_qs, fused_temperature, fused_pressure, fused_qv, dz,
        *fused_surfaces, 10.0, accumulate_surface=True)
    cp.cuda.Stream.null.synchronize()
    for actual, name, rtol, atol in (
            (fused_qi, "qi", 8.0e-6, 2.0e-12),
            (fused_ni, "ni_per_kg", 8.0e-6, 1.0e-2),
            (fused_qs, "qs", 8.0e-6, 2.0e-12)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), host(after, name),
            rtol=rtol, atol=atol)
    with (_ORACLE / "ice-auto-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    for field, name in zip(
            surfaces, ("rainnc_mm", "rainncv_mm",
                       "snownc_mm", "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=8.0e-6, atol=2.0e-15)
    for field, name in zip(
            fused_surfaces, ("rainnc_mm", "rainncv_mm",
                             "snownc_mm", "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(field), np.float32(float(surface[name])),
            rtol=8.0e-6, atol=2.0e-15)


def test_ice_autoconversion_rejects_noncanonical_table_layout():
    import cupy as cp

    from gpuwm.core.thompson import launch_ice_autoconversion

    fields = [cp.zeros((2, 2, 2), dtype=cp.float32) for _ in range(6)]
    canonical = cp.zeros((64, 55), dtype=cp.float64, order="F")
    with pytest.raises(TypeError, match="float64"):
        launch_ice_autoconversion(
            *fields, cp.zeros((64, 55), dtype=cp.float32, order="F"),
            canonical, 10.0)
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        launch_ice_autoconversion(
            *fields, canonical,
            cp.zeros((64, 55), dtype=cp.float64), 10.0)
