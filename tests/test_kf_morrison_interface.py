"""CPU authority for WRF's KF-to-Morrison mixed-phase interface.

WRF v4.6.1 keeps KF cloud water, cloud ice, rain, and snow tendencies
independent (``module_cu_kfeta.F:2311-2382,2599-2634``).  Morrison consumes
the raw rain/snow/ice rates to seed the matching number moments before its
entry cleanup and PSD reconstruction
(``module_mp_morr_two_moment.F:1327-1343``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_REAL74_COLUMNS = Path(__file__).parent / "data" / "kf_real74_12z_columns.npz"


class _NumpyCupyShim:
    """CPU-only subset used to exercise the driver seam without a GPU."""

    ndarray = np.ndarray
    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    isfinite = staticmethod(np.isfinite)
    zeros = staticmethod(np.zeros)
    maximum = staticmethod(np.maximum)
    floor = staticmethod(np.floor)
    where = staticmethod(np.where)


def _real74_sounding(label: str = "unstable") -> dict[str, np.ndarray | float]:
    with np.load(_REAL74_COLUMNS, allow_pickle=False) as data:
        fields = {
            name: data[f"{label}_{name}"].astype(np.float64)
            for name in ("u", "v", "temperature", "qv", "qc", "pressure",
                         "exner", "dz", "w")
        }
    return {**fields, "dx": 12000.0, "dt": 60.0, "cudt": 300.0}


def _mixed_phase_shallow_sounding() -> dict[str, np.ndarray | float]:
    nz = 49
    z_ifc = 16000.0 * np.linspace(0.0, 1.0, nz + 1) ** 1.18
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    pressure = 98000.0 * np.exp(-z / 8100.0)
    temperature = np.maximum(288.0 - 0.004 * z, 205.0)
    qv = 1.25 * (
        0.014 * np.exp(-z / 3500.0)
        / (1.0 + np.exp((z - 1600.0) / 150.0))
        + 0.001 * np.exp(-z / 5000.0))
    return {
        "u": 8.0 + 0.002 * z, "v": np.zeros(nz),
        "temperature": temperature, "qv": qv, "qc": np.zeros(nz),
        "pressure": pressure,
        "exner": (pressure / 100000.0) ** (287.0 / 1004.5), "dz": dz,
        "w": np.where(z < 2500.0, 0.01, 0.0),
        "dx": 12000.0, "dt": 60.0, "cudt": 300.0,
    }


def test_kf_mirror_preserves_all_four_mixed_phase_tendencies():
    from gpuwm.verify.npref import np_kf_column

    # The real74 fixture specifically pins the independently nonzero liquid
    # and ice tendencies reported by the audit.
    real74 = np_kf_column(**_real74_sounding())
    assert np.any(real74["rqccuten"] != 0.0)
    assert np.any(real74["rqicuten"] != 0.0)

    # KF's shallow fallback returns both liquid and frozen fallout to the
    # resolved rain/snow fields, exercising all four outputs simultaneously.
    result = np_kf_column(**_mixed_phase_shallow_sounding())

    assert result["triggered"] and result["shallow"]
    for name in ("rqccuten", "rqicuten", "rqrcuten", "rqscuten"):
        assert name in result
        assert np.isfinite(result[name]).all()
    assert np.any(result["rqccuten"] != 0.0)
    assert np.any(result["rqicuten"] != 0.0)
    assert np.any(result["rqrcuten"] != 0.0)
    assert np.any(result["rqscuten"] != 0.0)

    active = slice(0, result["cloud_top"] + 1)
    np.testing.assert_allclose(
        result["rqccuten"][active],
        result["closure_liquid"][active] / result["timec"],
        rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result["rqicuten"][active],
        result["closure_ice"][active] / result["timec"],
        rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result["rqrcuten"][active],
        result["closure_rain"][active] / result["timec"],
        rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result["rqscuten"][active],
        result["closure_snow"][active] / result["timec"],
        rtol=0.0, atol=0.0)


def test_kf_mirror_wrf_phase_modes_conserve_water_and_apply_latent_fusion():
    """Pin all WRF 2599-2640 output branches from closure-local fields."""
    from gpuwm.core.kf import KFPhaseMode
    from gpuwm.verify.npref import np_kf_column

    sounding = _mixed_phase_shallow_sounding()
    mixed = np_kf_column(
        **sounding, phase_mode=KFPhaseMode.SEPARATE_ICE_SNOW)
    assert mixed["triggered"] and mixed["shallow"]
    active = slice(0, mixed["cloud_top"] + 1)
    timec = mixed["timec"]
    ql = mixed["closure_liquid"][active]
    qi = mixed["closure_ice"][active]
    qr = mixed["closure_rain"][active]
    qs = mixed["closure_snow"][active]
    source_total_rate = (ql + qi + qr + qs) / timec
    base_tg = mixed["closure_temperature"][active]
    qg = mixed["closure_qv"][active]
    exner = np.asarray(sounding["exner"])[active]
    temperature = np.asarray(sounding["temperature"])[active]
    cpm = 1004.5 * (1.0 + 0.887 * qg)
    rlf = 3.339e5

    warm = np_kf_column(**sounding, phase_mode=KFPhaseMode.WARM_RAIN)
    np.testing.assert_allclose(
        warm["rqccuten"][active], (ql + qi) / timec,
        rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        warm["rqrcuten"][active], (qr + qs) / timec,
        rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(warm["rqicuten"], 0.0)
    np.testing.assert_array_equal(warm["rqscuten"], 0.0)
    np.testing.assert_allclose(
        warm["rqccuten"][active] + warm["rqrcuten"][active],
        source_total_rate, rtol=5.0e-16, atol=0.0)
    expected_tg = base_tg - (qi + qs) * rlf / cpm
    np.testing.assert_allclose(
        warm["rthcuten"][active],
        (expected_tg - temperature) / (exner * timec),
        rtol=2.0e-15, atol=0.0)

    no_snow = np_kf_column(
        **sounding, phase_mode=KFPhaseMode.NO_SEPARATE_SNOW)
    np.testing.assert_allclose(
        no_snow["rqccuten"][active] + no_snow["rqrcuten"][active],
        source_total_rate, rtol=5.0e-16, atol=0.0)
    np.testing.assert_array_equal(no_snow["rqicuten"], 0.0)
    np.testing.assert_array_equal(no_snow["rqscuten"], 0.0)
    warm_levels = np.flatnonzero(
        np.asarray(sounding["temperature"])[active] > 273.16)
    melting_level = int(warm_levels[-1]) if warm_levels.size else -1
    below = np.arange(mixed["cloud_top"] + 1) <= melting_level
    expected_tg = base_tg.copy()
    expected_tg[below] -= (qi[below] + qs[below]) * rlf / cpm[below]
    expected_tg[~below] += (ql[~below] + qr[~below]) * rlf / cpm[~below]
    np.testing.assert_allclose(
        no_snow["rthcuten"][active],
        (expected_tg - temperature) / (exner * timec),
        rtol=2.0e-15, atol=0.0)

    snow_only = np_kf_column(
        **sounding, phase_mode=KFPhaseMode.SEPARATE_SNOW)
    np.testing.assert_allclose(
        snow_only["rqccuten"][active], ql / timec,
        rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        snow_only["rqrcuten"][active], qr / timec,
        rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        snow_only["rqscuten"][active], (qs + qi) / timec,
        rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(snow_only["rqicuten"], 0.0)
    np.testing.assert_array_equal(
        snow_only["rthcuten"], mixed["rthcuten"])


def test_kf_phase_mode_mapping_keeps_kessler_and_no_mp_distinct():
    from gpuwm.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics

    assert kf_phase_mode_for_microphysics(0) == \
        KFPhaseMode.NO_SEPARATE_SNOW
    assert kf_phase_mode_for_microphysics(1) == KFPhaseMode.WARM_RAIN
    assert kf_phase_mode_for_microphysics(10) == \
        KFPhaseMode.SEPARATE_ICE_SNOW
    with pytest.raises(ValueError, match="no verified phase-output contract"):
        kf_phase_mode_for_microphysics(99)


def test_morrison_cumulus_number_seed_matches_wrf_formulas_and_threshold():
    from gpuwm.verify.npref import _np_morrison_seed_cumulus_numbers

    rho = np.array([1.0, 0.8], dtype=np.float64)
    qrcu = np.array([1.0e-4, 0.999e-10], dtype=np.float64)
    qscu = np.array([2.0e-4, -1.0e-5], dtype=np.float64)
    qicu = np.array([3.0e-8, 0.0], dtype=np.float64)
    nr0 = np.array([11.0, 12.0], dtype=np.float64)
    ns0 = np.array([21.0, 22.0], dtype=np.float64)
    ni0 = np.array([31.0, 32.0], dtype=np.float64)
    dt = 60.0

    nr, ns, ni = _np_morrison_seed_cumulus_numbers(
        nr0, ns0, ni0, qrcu, qscu, qicu, rho, dt)

    expected_nr = nr0.copy()
    expected_ns = ns0.copy()
    expected_ni = ni0.copy()
    expected_nr[0] += 1.8e5 * (
        qrcu[0] * dt / (np.pi * 997.0 * rho[0] ** 3)) ** 0.25
    expected_ns[0] += 3.0e5 * (
        qscu[0] * dt / (100.0 * np.pi * rho[0] ** 3)) ** 0.25
    expected_ni[0] += qicu[0] * dt / (
        (500.0 * np.pi / 6.0) * (80.0e-6) ** 3)

    np.testing.assert_allclose(nr, expected_nr, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(ns, expected_ns, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(ni, expected_ni, rtol=2.0e-15, atol=0.0)
    # The comparison is >= 1e-10 in WRF.  Sub-threshold and negative rates
    # leave their corresponding moments exactly unchanged.
    assert nr[1] == nr0[1]
    assert ns[1] == ns0[1]
    assert ni[1] == ni0[1]


def test_custom_warm_rain_attachment_requires_energy_defined_phase_closure(
        monkeypatch):
    """The driver cannot infer latent heating from already diagnosed rates."""
    from gpuwm.core import physics
    from gpuwm.core.physics import CumulusResult, PhysicsDriver

    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    shape = (2, 2, 3)
    surface = shape[1:]
    state = SimpleNamespace(
        p=np.zeros(shape, dtype=np.float32),
        c1h=np.ones(shape[0], dtype=np.float32),
        c2h=np.zeros(shape[0], dtype=np.float32),
        has_msf=False,
        total_mu=lambda: np.full(surface, 2.0, dtype=np.float32),
    )
    # Deliberately do not define state.qi: that is the real mp=0/1 contract.
    rates = {
        "rthcuten": np.full(shape, 1.0, dtype=np.float32),
        "rqvcuten": np.full(shape, 2.0, dtype=np.float32),
        "rqccuten": np.full(shape, 3.0, dtype=np.float32),
        "rqicuten": np.full(shape, 5.0, dtype=np.float32),
        "rqrcuten": np.full(shape, 7.0, dtype=np.float32),
        "rqscuten": np.full(shape, 11.0, dtype=np.float32),
    }
    driver = object.__new__(PhysicsDriver)
    # This focused clock fixture bypasses __init__; optional LSM parameter
    # bundles were added later and are absent for this Morrison-only path.
    driver.ruc_params = None
    driver.noahmp_params = None
    driver.fields = {}
    driver.cu_rates = {name: np.zeros(shape, dtype=np.float32)
                       for name in rates}
    driver.rainc = np.zeros(surface, dtype=np.float32)
    driver._pending_rainbl = np.zeros(surface, dtype=np.float32)
    # SEAM A widened the coupler ring-mask predicate to
    # `specified or nested`; this stub models the same non-LBC,
    # non-nested domain it always did, now with the attribute the
    # production RunConfig has always carried.
    cfg = SimpleNamespace(
        specified=False, nested=False, cu_physics=1, mp_physics=1)

    driver.cumulus_callable = lambda **_kwargs: CumulusResult(**rates)
    with pytest.raises(ValueError, match="latent-energy-consistent"):
        driver._run_cumulus({}, state, cfg)

    # A custom scheme may instead return its own energy-consistent, already
    # folded warm-rain contract and omit the unsupported frozen categories.
    folded = {**rates, "rqccuten": rates["rqccuten"] + rates["rqicuten"],
              "rqrcuten": rates["rqrcuten"] + rates["rqscuten"]}
    folded.pop("rqicuten")
    folded.pop("rqscuten")
    driver.cumulus_callable = lambda **_kwargs: CumulusResult(**folded)
    driver._run_cumulus({}, state, cfg)

    np.testing.assert_array_equal(driver.cu_rates["rqccuten"], 8.0)
    np.testing.assert_array_equal(driver.cu_rates["rqrcuten"], 18.0)
    np.testing.assert_array_equal(driver.cu_rates["rqicuten"], 0.0)
    np.testing.assert_array_equal(driver.cu_rates["rqscuten"], 0.0)
    # The held scalar stack receives dry-mass coupling after phase folding.
    np.testing.assert_array_equal(driver.cumulus_tendencies.rqc, 16.0)
    np.testing.assert_array_equal(driver.cumulus_tendencies.rqr, 36.0)
    assert driver.cumulus_tendencies.rqi is None
    assert driver.cumulus_tendencies.rqs is None


def test_nca_expiry_preserves_current_rk_copy_but_zeros_morrison_rates(
        monkeypatch):
    """advance_ppt precedes Morrison while this step's RK copy survives."""
    from gpuwm.core import physics
    from gpuwm.core.physics import PhysicsDriver

    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    shape = (2, 1, 1)
    surface = (1, 1)
    driver = object.__new__(PhysicsDriver)
    driver.ruc_params = None
    driver.noahmp_params = None
    driver.cu_pratec = np.zeros(surface, dtype=np.float32)
    driver.rainc = np.zeros(surface, dtype=np.float32)
    driver._pending_rainbl = np.zeros(surface, dtype=np.float32)
    driver.cu_nca = np.full(surface, 60.0, dtype=np.float32)
    driver.cu_expiring = np.zeros(surface, dtype=np.float32)
    names = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
             "rqrcuten", "rqscuten")
    driver.cu_rates = {
        name: np.full(shape, index + 1.0, dtype=np.float32)
        for index, name in enumerate(names)
    }
    scalar_names = ("rtheta", "rqv", "rqc", "rqi", "rqr", "rqs")
    driver.cumulus_tendencies = SimpleNamespace(**{
        name: np.full(shape, index + 11.0, dtype=np.float32)
        for index, name in enumerate(scalar_names)
    })
    # _compose_tendencies has already copied these values into the target
    # consumed by this RK step; it is deliberately distinct from the held
    # component that advance_ppt clears for the next step.
    current_rk = SimpleNamespace(**{
        name: getattr(driver.cumulus_tendencies, name).copy()
        for name in scalar_names
    })
    driver.tendencies = current_rk

    driver._advance_cumulus_clock(
        SimpleNamespace(elapsed_seconds=0.0),
        SimpleNamespace(dt=60.0, clock_dt=0.0))

    for name in scalar_names:
        np.testing.assert_array_equal(
            getattr(driver.tendencies, name),
            np.full(shape, scalar_names.index(name) + 11.0,
                    dtype=np.float32))
        np.testing.assert_array_equal(
            getattr(driver.cumulus_tendencies, name), 0.0)
    # These are exactly the three arrays apply_morrison forwards to the
    # number-moment seed interface later in the step.
    for name in ("rqrcuten", "rqicuten", "rqscuten"):
        np.testing.assert_array_equal(driver.cu_rates[name], 0.0)
    np.testing.assert_array_equal(driver.cu_nca, 0.0)
    np.testing.assert_array_equal(driver.cu_expiring, 0.0)
    assert driver._cu_expiry_pending is False


def test_cuda_sources_keep_four_rates_and_wrf_number_seed_statement_order():
    root = Path(__file__).parents[1]
    kf = (root / "gpuwm" / "core" / "kernels" / "kf.cu").read_text(
        encoding="utf-8")
    morrison = (
        root / "gpuwm" / "core" / "kernels" / "morrison.cu").read_text(
            encoding="utf-8")

    assert "int phase_mode" in kf
    assert "KF_PHASE_WARM_RAIN" in kf
    assert "KF_PHASE_NO_SEPARATE_SNOW" in kf
    assert "KF_PHASE_SEPARATE_SNOW" in kf
    assert "(parcel_q[nk]+thetaeu[nk])*KF_RLF/cpm" in kf
    assert "rqccuten[index] = parcel_t[nk]/timec" in kf
    assert "rqicuten[index] = parcel_q[nk]/timec" in kf
    assert "rqrcuten[index] = resolved_precip[nk]/timec" in kf
    assert "rqscuten[index] = thetaeu[nk]/timec" in kf
    # WRF's warm-rain branch folds frozen mass only after applying the
    # corresponding latent-fusion adjustment to TG.
    assert "rqccuten[index] = (parcel_t[nk]+parcel_q[nk])/timec" in kf
    assert "rqrcuten[index] = (resolved_precip[nk]+thetaeu[nk])/timec" in kf
    latent = kf.index("tg[nk] -= (parcel_q[nk]+thetaeu[nk])*KF_RLF/cpm")
    theta_rate = kf.rindex("rthcuten[index] =")
    assert latent < theta_rate

    rain = morrison.index("if (qrcu >= 1.0e-10f)")
    snow = morrison.index("if (qscu >= 1.0e-10f)")
    ice = morrison.index("if (qicu >= 1.0e-10f)")
    process = morrison.index("morr_process_level(&qvk")
    assert rain < snow < ice < process
    assert "MPI * MRHOW * rhoa * rhoa * rhoa" in morrison
    assert "100.0f * MPI * rhoa * rhoa * rhoa" in morrison
    assert "MCI * 80.0e-6f * 80.0e-6f * 80.0e-6f" in morrison
