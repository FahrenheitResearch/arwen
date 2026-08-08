"""Morrison two-moment microphysics (WRF ``mp_physics=10``).

Most tests here are implementation and self-consistency checks.  In particular,
``np_morrison_column`` is a float64 transcription mirror of gpuwm's arithmetic,
not an oracle.  The byte-unmodified WRF v4.6.1 authority is exercised separately
by ``test_morrison_wrf461_parity.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core import constants as c


MASS_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER_NAMES = ("nc", "nr", "ni", "ns", "ng")

# Measured only at the mixed-phase hail Courant discontinuity (k=7) after
# enabling WRF's QSCUTEN number seed.  See the CPU evidence gate below and
# the branch-local explanation in the CUDA-vs-mirror test.
_SEEDED_MIXED_HAIL_NS_MAX_DIFF = 1541.133829009
_SEEDED_MIXED_HAIL_NS_SCALE_128ULP = 1.02e8
_SEEDED_MIXED_HAIL_EFFS_MAX_REL = 0.4004771322
_SEEDED_MIXED_HAIL_EFFS_RTOL = 0.402


def _polysvp_water(t):
    """WRF Morrison Flatau liquid polynomial, source lines 4090-4097."""
    x = np.asarray(t, np.float64) - 273.15
    a = (6.11239921, 0.443987641, 0.142986287e-1,
         0.264847430e-3, 0.302950461e-5, 0.206739458e-7,
         0.640689451e-10, -0.952447341e-13, -0.976195544e-15)
    out = np.full_like(x, a[-1])
    for coeff in a[-2::-1]:
        out = coeff + x * out
    return out * 100.0


def _regime_column(regime: str, nz: int = 28):
    """FP32 WK82-like column spanning one requested phase regime."""
    z = np.linspace(150.0, 13850.0, nz, dtype=np.float64)
    p = 98000.0 * np.exp(-z / 8000.0)
    if regime == "warm":
        temp = 298.0 - 0.0045 * z
        temp = np.maximum(temp, 276.0)
    elif regime == "mixed":
        temp = 289.0 - 0.0065 * z
    elif regime in ("glaciated", "flagged"):
        temp = 263.0 - 0.0030 * z
    else:
        raise ValueError(regime)
    pii = (p / c.P0) ** c.RCP
    theta = temp / pii
    rho = p / (c.RD * temp)
    dz = np.full(nz, z[1] - z[0], dtype=np.float64)
    qvs = c.EP2 * np.minimum(0.99 * p, _polysvp_water(temp)) \
        / (p - np.minimum(0.99 * p, _polysvp_water(temp)))
    # Keep the FP32/FP64 comparison away from the discontinuous corner where
    # a one-ULP cold-process temperature difference decides whether the last
    # ~5e-6 kg/kg of cloud water evaporates or freezes.  The colder levels
    # remain strongly ice-supersaturated and exercise homogeneous freezing.
    cold_rh = np.where(temp <= 233.15, 1.10, 1.04)
    qv = qvs * np.where(temp < 273.15, cold_rh, 1.015)

    cloud = np.exp(-((z - 3500.0) / 2600.0) ** 2)
    precip = np.exp(-((z - 2600.0) / 3000.0) ** 2)
    qc = 1.6e-3 * cloud
    qr = 7.0e-4 * precip
    qi = np.zeros(nz)
    qs = np.zeros(nz)
    qg = np.zeros(nz)
    if regime != "warm":
        cold = temp < 273.15
        qi[cold] = 3.0e-4 * cloud[cold]
        qs[cold] = 8.0e-4 * precip[cold]
        qg[cold] = 3.0e-4 * precip[cold]
    if regime in ("glaciated", "flagged"):
        qc *= 0.15
        qr *= 0.10
        qi += 5.0e-4 * cloud
        qs += 1.2e-3 * precip
        qg += 4.0e-4 * precip

    # WRF concentrations are per kg dry air (wrapper lines 585-588).
    nc = 250.0e6 / rho
    nr = np.where(qr > 0.0, 1.5e6 / rho, 0.0)
    ni = np.where(qi > 0.0, 8.0e4 / rho, 0.0)
    if regime == "flagged":
        # Large cloud ice: ni/qi=1e7 kg-1/(kg kg-1) puts 1/lambda_i
        # beyond WRF's 250 um ice-to-snow recategorization threshold.
        ni = np.where(qi > 0.0, qi * 1.0e7, 0.0)
    ns = np.where(qs > 0.0, 2.0e5 / rho, 0.0)
    ng = np.where(qg > 0.0, 8.0e4 / rho, 0.0)
    def f32(value):
        return np.asarray(value, np.float32)
    return {name: f32(value) for name, value in {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr, "qi": qi,
        "qs": qs, "qg": qg, "nc": nc, "nr": nr, "ni": ni,
        "ns": ns, "ng": ng, "rho": rho, "pii": pii,
        "pressure": p, "dz": dz,
    }.items()}


def _water_mass(column, rho):
    return float(np.sum(sum(np.asarray(column[n], np.float64)
                            for n in MASS_NAMES)
                        * np.asarray(rho, np.float64)
                        * np.asarray(column["dz"], np.float64)))


def test_morrison_config_selector_accepts_10_and_rejects_unknown(tmp_path):
    """The config-file surface exposes WRF's ``mp_physics=10`` selector."""
    from gpuwm.config import load_config

    path = tmp_path / "morrison.toml"
    template = ("[grid]\nnx=3\nny=1\nnz=4\ndx=1000.0\ndy=1000.0\n"
                "ztop=4000.0\n[dynamics]\ndt=6.0\nmoist=true\n"
                "mp_physics={mp}\n"
                "[run]\nrun_seconds=0.0\n")
    path.write_text(template.format(mp=10))
    assert load_config(path).mp_physics == 10
    path.write_text(template.format(mp=11))
    with pytest.raises(ValueError, match="mp_physics"):
        load_config(path)


def test_morrison_sedimentation_uses_one_wrf_column_substep_count():
    """Every category uses WRF's fastest-species column-wide ``NSTEP``."""
    from gpuwm.verify.npref import _np_morrison_advect_sedimentation

    # Bottom-to-top storage.  WRF fills a zero fall speed below existing
    # precipitation from the level above (source lines 3505-3537).  Rain's
    # Courant number therefore selects two substeps for snow as well, even
    # though snow alone would select one.
    qdens = {
        "r": np.array([0.0, 1.0]),
        "s": np.array([0.0, 1.0]),
    }
    ndens = {name: value.copy() for name, value in qdens.items()}
    velocities = {
        "r": (np.array([0.0, 1.6]), np.array([0.0, 1.6])),
        "s": (np.array([0.0, 0.6]), np.array([0.0, 0.6])),
    }
    qout, nout, exported, nstep = _np_morrison_advect_sedimentation(
        qdens, ndens, velocities, np.ones(2), dt=1.0)

    assert nstep == 2
    np.testing.assert_allclose(qout["s"], (0.42, 0.49), atol=1e-15)
    np.testing.assert_allclose(nout["s"], qout["s"], atol=1e-15)
    assert exported["s"] == pytest.approx(0.09, abs=1e-15)
    for name in qout:
        assert np.sum(qout[name]) + exported[name] == pytest.approx(
            np.sum(qdens[name]), abs=2e-15)


def test_seeded_mixed_hail_ns_bound_is_measured_and_branch_local():
    """Pin the measured QSCUTEN/Courant A/B, not a dimensional blanket.

    At the one discontinuous mixed-hail cell, the unseeded CUDA/float64
    difference was 1429.43171 m-3.  WRF's QSCUTEN seed changes the CUDA
    result by 112.1748 m-3 but the continuous float64 result by only
    0.4726817 m-3, accounting for the observed seeded 1541.133829 m-3
    difference.  The 128-ULP scale below leaves about one percent headroom
    and remains over 39 times smaller than the removed 4e9 blanket.
    """
    unseeded = 1429.43171
    discontinuous_seed_shift = 112.1748 - 0.4726817
    assert np.isclose(
        unseeded + discontinuous_seed_shift,
        _SEEDED_MIXED_HAIL_NS_MAX_DIFF, rtol=0.0, atol=2.0e-4)

    bound = (128.0 * np.finfo(np.float32).eps
             * _SEEDED_MIXED_HAIL_NS_SCALE_128ULP)
    assert _SEEDED_MIXED_HAIL_NS_MAX_DIFF < bound
    assert bound < 1.011 * _SEEDED_MIXED_HAIL_NS_MAX_DIFF
    assert _SEEDED_MIXED_HAIL_NS_SCALE_128ULP < 4.0e9 / 39.0

    # The same k=7 cell's bounded snow radius moves from 1833.484131 um
    # unseeded to 1798.568604 um seeded while the float64 PSD remains at its
    # 3000 um cap.  Its qs mass still agrees to 1.1e-9 relative, isolating
    # this to the already-masked PSD/Courant diagnostic branch rather than a
    # hydrometeor-state mismatch.  Keep less than 0.4% relative headroom.
    unseeded_effs_rel = abs(1833.484131 - 3000.0) / 3000.0
    seeded_effs_rel = abs(1798.568604 - 3000.0) / 3000.0
    assert np.isclose(unseeded_effs_rel, 0.3888386230, atol=3.0e-10)
    assert np.isclose(
        seeded_effs_rel, _SEEDED_MIXED_HAIL_EFFS_MAX_REL, atol=3.0e-10)
    assert (_SEEDED_MIXED_HAIL_EFFS_MAX_REL
            < _SEEDED_MIXED_HAIL_EFFS_RTOL
            < 1.004 * _SEEDED_MIXED_HAIL_EFFS_MAX_REL)


def test_morrison_final_phase_cleanup_is_mass_and_number_conserving():
    """WRF's post-sedimentation instantaneous phase changes conserve water."""
    from gpuwm.verify.npref import _np_morrison_final_phase_cleanup

    q = {
        "qv": np.zeros(2), "qc": np.array([0.0, 2.0e-4]),
        "qr": np.array([0.0, 3.0e-4]), "qi": np.array([4.0e-4, 0.0]),
        "qs": np.zeros(2), "qg": np.zeros(2),
    }
    n = {
        "nc": np.array([0.0, 2.0e8]), "nr": np.array([0.0, 3.0e5]),
        "ni": np.array([4.0e4, 0.0]), "ns": np.zeros(2),
        "ng": np.zeros(2),
    }
    before = sum(value.copy() for value in q.values())
    temperature = np.array([278.0, 230.0])
    q, n, temperature = _np_morrison_final_phase_cleanup(
        q, n, temperature, np.ones(2), np.array([False, False]))

    np.testing.assert_array_equal(q["qi"], (0.0, 2.0e-4))
    np.testing.assert_array_equal(q["qg"], (0.0, 3.0e-4))
    np.testing.assert_array_equal(q["qc"], (0.0, 0.0))
    np.testing.assert_array_equal(q["qr"], (4.0e-4, 0.0))
    np.testing.assert_array_equal(n["nr"], (4.0e4, 0.0))
    np.testing.assert_array_equal(n["ni"], (0.0, 2.0e8))
    np.testing.assert_array_equal(n["ng"], (0.0, 3.0e5))
    np.testing.assert_allclose(sum(q.values()), before, atol=1e-18)
    assert temperature[0] < 278.0
    assert temperature[1] > 230.0


def test_morrison_final_low_rh_cleanup_returns_tiny_categories_to_vapor():
    """WRF 3729-3761 repeats tiny-category cleanup after fallout."""
    from gpuwm.verify.npref import _np_morrison_final_phase_cleanup

    tiny = 5.0e-9
    q = {"qv": np.array([1.0e-5]), "qc": np.array([tiny]),
         "qr": np.array([tiny]), "qi": np.array([tiny]),
         "qs": np.array([tiny]), "qg": np.array([tiny])}
    n = {name: np.ones(1) for name in NUMBER_NAMES}
    before = sum(value.copy() for value in q.values())
    q, n, temperature = _np_morrison_final_phase_cleanup(
        q, n, np.array([260.0]), np.array([1.0]), np.array([False]),
        pressure=np.array([70000.0]))

    np.testing.assert_allclose(sum(q.values()), before, atol=1e-20)
    assert all(q[name][0] == 0.0 for name in MASS_NAMES if name != "qv")
    assert q["qv"][0] == pytest.approx(before[0], abs=1e-20)
    assert temperature[0] < 260.0


def test_morrison_cloud_sedimentation_uses_tendency_free_nc():
    """INUM=1 DUMFNC is bounded NC3D, not NC3D+NC3DTEN (F:3367-3374)."""
    from gpuwm.verify.npref import (
        _np_morrison_apply_level, _np_morrison_polysvp)

    q = {"qv": 0.010, "qc": 1.0e-3, "qr": 2.0e-4,
         "qi": 0.0, "qs": 0.0, "qg": 0.0}
    n = {"nc": 2.0e8, "nr": 2.0e5, "ni": 0.0, "ns": 0.0, "ng": 0.0}
    temperature, pressure, dt = 285.0, 90000.0, 45.0
    rhoa = pressure / (c.RD * temperature)
    ew = min(0.99 * pressure,
             float(_np_morrison_polysvp(temperature, False)))
    ei = min(ew, 0.99 * pressure,
             float(_np_morrison_polysvp(temperature, True)))
    qvs = c.EP2 * ew / (pressure - ew)
    qvi = c.EP2 * ei / (pressure - ei)
    xlv = 3.1484e6 - 2370.0 * temperature
    xls = 3.15e6 - 2370.0 * temperature + 0.3337e6
    cpm = c.CP * (1.0 + 0.887 * q["qv"])
    _, updated, _, _, sediment_nc = _np_morrison_apply_level(
        q, n, temperature, pressure, rhoa, dt, qvs, qvi,
        xlv, xls, cpm, True)

    assert sediment_nc == pytest.approx(250.0e6 / rhoa)
    assert updated["nc"] != pytest.approx(sediment_nc)


def test_morrison_final_cleanup_uses_latched_conversion_and_stale_cpm():
    """Pin F:3683 and 3729-3849 statement-order carry-ins."""
    from gpuwm.verify.npref import _np_morrison_final_phase_cleanup

    amount = 4.0e-4
    q = {"qv": np.full(2, 0.010), "qc": np.zeros(2),
         "qr": np.zeros(2), "qi": np.full(2, amount),
         "qs": np.zeros(2), "qg": np.zeros(2)}
    n = {"nc": np.zeros(2), "nr": np.zeros(2),
         "ni": np.full(2, 4.0e4), "ns": np.zeros(2),
         "ng": np.zeros(2)}
    stale_cpm = np.array([2000.0, 2000.0])
    q, n, temperature = _np_morrison_final_phase_cleanup(
        q, n, np.array([275.0, 275.0]), np.ones(2),
        np.array([True, False]), pressure=np.full(2, 90000.0),
        xlv_stale=np.full(2, 2.4e6),
        cpm_stale=stale_cpm)

    # The ice-to-snow mask was latched while cold; a later temperature above
    # freezing does not cancel its application.
    assert q["qi"][0] == 0.0
    assert q["qs"][0] == pytest.approx(amount)
    assert n["ns"][0] == pytest.approx(4.0e4)
    assert temperature[0] == 275.0
    # The unlatched second level melts, using the supplied stale CPM rather
    # than a recomputation from the updated state.
    assert q["qi"][1] == 0.0
    assert q["qr"][1] == pytest.approx(amount)
    assert temperature[1] == pytest.approx(
        275.0 - amount * 0.3353e6 / stale_cpm[1])


def test_morrison_final_effc_rebounds_from_transient_nc_before_restore():
    """F:3918-4053 computes EFFC before the fixed-Nc restore."""
    from gpuwm.verify.npref import _np_morrison_slopes

    q = {"c": np.array([2.0e-4]), "r": np.zeros(1),
         "i": np.zeros(1), "s": np.zeros(1), "g": np.zeros(1)}
    transient = {"c": np.array([2.0e7]), "r": np.zeros(1),
                 "i": np.zeros(1), "s": np.zeros(1), "g": np.zeros(1)}
    rho = np.array([1.0])
    temperature = np.array([280.0])
    pressure = np.array([90000.0])
    lam_transient, pgam_transient, _ = _np_morrison_slopes(
        q, transient, rho, temperature, pressure=pressure,
        reset_cloud_number=False)
    lam_fixed, pgam_fixed, _ = _np_morrison_slopes(
        q, transient, rho, temperature, pressure=pressure,
        reset_cloud_number=True)
    eff_transient = ((pgam_transient + 3.0)
                     / (2.0 * lam_transient["c"]) * 1.0e6)
    eff_fixed = (pgam_fixed + 3.0) / (2.0 * lam_fixed["c"]) * 1.0e6

    assert eff_transient[0] != pytest.approx(eff_fixed[0])


@pytest.mark.parametrize("regime", ("warm", "mixed", "glaciated", "flagged"))
@pytest.mark.parametrize("morr_rimed_ice", (0, 1))
def test_morrison_float64_mirror_regimes_conserve_and_bound(
        regime, morr_rimed_ice):
    """Mirror-first gate over warm, mixed, and glaciated WK82 columns."""
    from gpuwm.verify.npref import np_morrison_column

    src = _regime_column(regime)
    rho = (src["pressure"].astype(np.float64)
           / (c.RD * src["theta"].astype(np.float64)
              * src["pii"].astype(np.float64)))
    initial = _water_mass(src, rho)
    out = np_morrison_column(**{k: v.astype(np.float64)
                                for k, v in src.items()}, dt=45.0,
                             morr_rimed_ice=morr_rimed_ice)
    final = _water_mass(out, rho)
    exported = out["precip_step"]
    residual = abs(final + exported - initial)
    assert residual <= 256.0 * np.finfo(np.float64).eps * max(initial, 1.0)

    for name in MASS_NAMES + NUMBER_NAMES:
        assert np.isfinite(out[name]).all(), name
        assert out[name].min() >= 0.0, name
    # Fixed 250 cm-3 droplets and WRF's 0.3e6 m-3 ice-number cap
    # (source lines 1493-1496 and 4044-4053).
    np.testing.assert_allclose(out["nc"], 250.0e6 / rho, rtol=2e-13)
    assert np.all(out["ni"] <= 0.3e6 / rho * (1.0 + 2e-13))
    # Exponential PSD moments remain inside WRF's slope bounds
    # (source lines 3862-4000).  The fixed cloud-droplet number is governed
    # separately by INUM=1 above.
    specs = {
        "r": ("qr", "nr", np.pi * 997.0,
              1.0 / 2800.0e-6, 1.0 / 20.0e-6),
        "i": ("qi", "ni", np.pi * 500.0,
              1.0 / 350.0e-6, 1.0 / 1.0e-6),
        "s": ("qs", "ns", np.pi * 100.0,
              1.0 / 2000.0e-6, 1.0 / 10.0e-6),
        "g": ("qg", "ng", np.pi * (400.0 if morr_rimed_ice == 0 else 900.0),
              1.0 / 2000.0e-6, 1.0 / 20.0e-6),
    }
    for mass, moment, six_c, lam_lo, lam_hi in specs.values():
        active = out[mass] >= 1.0e-14
        lower = out[mass][active] * lam_lo ** 3 / six_c
        upper = out[mass][active] * lam_hi ** 3 / six_c
        assert np.all(out[moment][active] >= lower * (1.0 - 2e-13))
        assert np.all(out[moment][active] <= upper * (1.0 + 2e-13))
    assert out["precip_step"] > 0.0
    if regime == "warm":
        assert out["qr"].max() > src["qr"].max()
    else:
        assert out["qs"].max() > 0.0 and out["qg"].max() > 0.0


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("regime", ("warm", "mixed", "glaciated", "flagged"))
@pytest.mark.parametrize("morr_rimed_ice", (0, 1))
def test_morrison_kernel_matches_float64_mirror_at_fp32_floor(
        regime, morr_rimed_ice):
    """One CUDA thread per column matches the float64 mirror at FP32 floors."""
    import cupy as cp

    from gpuwm.core.morrison import launch_morrison
    from gpuwm.verify.npref import np_morrison_column

    col = _regime_column(regime)
    # Exercise a small true 3-D batch with distinct column amplitudes.
    host = {}
    for name, value in col.items():
        factors = ((1.0,) * 4 if regime == "flagged" else
                   (1.0, 0.93, 1.07, 0.88)
                   if name in MASS_NAMES and name != "qv" else (1.0,) * 4)
        host[name] = np.stack([value * f for f in factors], axis=1)[:, None, :]
        host[name] = np.ascontiguousarray(host[name], dtype=np.float32)
    dev = {name: cp.asarray(value) for name, value in host.items()}
    rho_before = host["rho"].copy()
    precip = {name: cp.zeros((1, 4), cp.float32) for name in
              ("rainnc", "rainncv", "snownc", "snowncv",
               "graupelnc", "graupelncv", "sr")}
    effective = {name: cp.empty_like(dev["theta"])
                 for name in ("effc", "effr", "effi", "effs")}
    # Exercise WRF's KF-to-Morrison interface on every regime: raw held
    # rain/snow/ice mass rates seed Nr/Ns/Ni before process calculations.
    cu_host = {
        "qrcuten": np.full_like(host["theta"], 2.0e-8),
        "qscuten": np.full_like(host["theta"], 3.0e-8),
        "qicuten": np.full_like(host["theta"], 4.0e-8),
    }
    cu_dev = {name: cp.asarray(value) for name, value in cu_host.items()}
    launch_morrison(**dev, **precip, **effective, dt=45.0,
                    **cu_dev, morr_rimed_ice=morr_rimed_ice)
    np.testing.assert_array_equal(cp.asnumpy(dev["rho"]), rho_before)

    ref = {name: np.empty_like(value, dtype=np.float64)
           for name, value in host.items() if name not in ("rho", "pressure", "dz")}
    ref_step = np.zeros((1, 4))
    ref_snow = np.zeros((1, 4))
    ref_graupel = np.zeros((1, 4))
    ref_effective = {name: np.empty_like(host["theta"], dtype=np.float64)
                     for name in effective}
    for i in range(4):
        kwargs = {name: value[:, 0, i].astype(np.float64)
                  for name, value in host.items()}
        cu_kwargs = {name: value[:, 0, i].astype(np.float64)
                     for name, value in cu_host.items()}
        ans = np_morrison_column(**kwargs, dt=45.0,
                                 **cu_kwargs,
                                 morr_rimed_ice=morr_rimed_ice)
        for name in ref:
            ref[name][:, 0, i] = ans[name]
        for name in ref_effective:
            ref_effective[name][:, 0, i] = ans[name]
        ref_step[0, i] = ans["precip_step"]
        ref_snow[0, i] = ans["snow_step"]
        ref_graupel[0, i] = ans["graupel_step"]

    eps = np.finfo(np.float32).eps
    scales = {"theta": 300.0, **{n: 0.02 for n in MASS_NAMES},
              "pii": 1.0}
    # Replace the former unexplained blanket 4e9 number-moment scale with the
    # measured per-field magnitude of this fixture's float64 transcription
    # mirror.  The tolerance is therefore a stated count of FP32 ULPs at the
    # value actually exercised,
    # rather than an unrelated dimensional constant that can hide PSD errors.
    # A mixed-phase FP32 Courant-boundary fixture takes one extra shared
    # sedimentation substep.  Direct measurement of that documented branch is
    # 8.213e3 m-3 for Nr and 6.401e2 m-3 for Ns.  At 128 ULPs these imply scales
    # 5.383e8 and 4.195e7 respectively--both far below the removed 4e9 blanket.
    # WRF-default hail changes the dense-ice sedimentation path enough that
    # the same mixed-phase FP32 discontinuity measured 1.429e3 m-3 for Ns
    # without cumulus seeding.  With the exact WRF QSCUTEN=3e-8 s-1 seed,
    # the continuous float64 result changes only 0.4726817 m-3 while FP32's
    # Courant branch changes 112.1748 m-3, moving the measured maximum to
    # 1541.133829 m-3 at k=7.  A 1.02e8 scale gives a 1556.40 m-3 128-ULP
    # bound (about 1% headroom and >39x below the removed 4e9 blanket).
    # Keep the measured branch explicit instead of restoring a scheme-wide
    # dimensional tolerance; the CPU A/B evidence gate pins these numbers.
    measured_branch_scale = {
        "nr": 5.4e8,
        "ns": (_SEEDED_MIXED_HAIL_NS_SCALE_128ULP
               if morr_rimed_ice == 1 else 4.2e7),
    }
    scales.update({name: max(float(np.max(np.abs(ref[name]))),
                             measured_branch_scale.get(name, 1.0))
                   for name in ("nc", "nr", "ni", "ns", "ng")})
    # Cold donor exhaustion is an FP32-discontinuous Courant mask: a trace
    # positive device remainder can select another shared sedimentation
    # substep.  The mirror explicitly matches that mask; 128 state ULPs
    # cover the resulting multi-substep accumulation without relative slack.
    ulps = 16.0 if regime == "warm" else 128.0
    for name, expected in ref.items():
        if name not in scales:
            continue
        np.testing.assert_allclose(cp.asnumpy(dev[name]), expected,
                                   rtol=0.0,
                                   atol=ulps * eps * scales[name],
                                   err_msg=f"{regime}: {name}")
    effective_mass = {"effc": "qc", "effr": "qr",
                      "effi": "qi", "effs": "qs"}
    # Radius sensitivity inherits the documented cold donor/Courant branch
    # divergence in the bounded moments.  Measured non-boundary maxima over
    # these fixtures are 6.36e-5 (cloud), 1.31e-3 (rain), 5.46e-4 (ice), and
    # 6.74e-5 (graupel-mode snow).  Hail's changed dense-ice rates move the
    # non-discontinuous snow maximum to 3.71e-4.  The FP32 donor/PSD branch
    # exceptions are confined to mixed-phase level 9 for graupel and levels
    # 6-7 for hail.  QSCUTEN seeding moves the same k=7 hail cell from the
    # measured unseeded 0.388839 relative radius difference to 0.400477 while
    # qs remains matched at 1.1e-9 relative.  Do not let those isolated
    # branches weaken other snow-radius points.
    # The FP32 POLYSVP correction made the kernel match WRF's declared REAL
    # precision, while this non-authoritative mirror deliberately stays FP64.
    # Its ice-radius diagnostic now differs by 0.1802%; keep this mirror check
    # descriptive and let test_morrison_wrf461_parity.py own the WRF contract.
    effective_rtol = {"effc": 8.0e-5, "effr": 1.5e-3,
                      "effi": 2.0e-3,
                      "effs": (4.0e-4 if morr_rimed_ice == 1 else 8.0e-5)}
    for name, expected in ref_effective.items():
        actual = cp.asnumpy(effective[name])
        # An O(QSMALL) FP32 donor remainder may retain a diagnostic radius
        # where the float64 mirror has exactly cleared the category.  Its
        # optical path is negligible; compare radii only where both paths
        # carry radiatively material mass, and separately require finiteness.
        mass = effective_mass[name]
        mask = ((cp.asnumpy(dev[mass]) > 1.0e-10)
                & (ref[mass] > 1.0e-10))
        assert np.isfinite(actual).all()
        atol = ulps * eps * max(float(np.max(np.abs(expected))), 25.0)
        courant_boundary = np.zeros_like(mask)
        if name == "effs" and regime == "mixed":
            if morr_rimed_ice == 1:
                courant_boundary[6:8, :, :] = True
            else:
                courant_boundary[9, :, :] = True
        tight = mask & ~courant_boundary
        np.testing.assert_allclose(
            actual[tight], expected[tight], rtol=effective_rtol[name],
            atol=atol, err_msg=f"{regime}: {name} non-boundary")
        branch = mask & courant_boundary
        np.testing.assert_allclose(
            actual[branch], expected[branch],
            rtol=(_SEEDED_MIXED_HAIL_EFFS_RTOL
                  if morr_rimed_ice == 1 else 3.2e-1),
            atol=atol, err_msg=f"{regime}: {name} Courant boundary")
    np.testing.assert_allclose(cp.asnumpy(precip["rainncv"]), ref_step,
                               rtol=0.0,
                               atol=ulps * eps * max(ref_step.max(), 1e-3))
    np.testing.assert_allclose(cp.asnumpy(precip["snowncv"]), ref_snow,
                               rtol=0.0,
                               atol=16.0 * eps * max(ref_snow.max(), 1e-3))
    np.testing.assert_allclose(cp.asnumpy(precip["graupelncv"]), ref_graupel,
                               rtol=0.0,
                               atol=16.0 * eps * max(ref_graupel.max(), 1e-3))

    # Apply the scheme's moment gates to the device result itself rather
    # than relying only on its agreement with the bounded mirror.  The
    # additive slack is a fixed FP32 floor for concentration state.
    got = {name: cp.asnumpy(dev[name]).astype(np.float64)
           for name in MASS_NAMES + NUMBER_NAMES}
    for name in MASS_NAMES + NUMBER_NAMES:
        assert np.isfinite(got[name]).all(), f"{regime}: {name}"
        assert got[name].min() >= 0.0, f"{regime}: {name}"
    rhoa = (host["pressure"].astype(np.float64)
            / (c.RD * host["theta"].astype(np.float64)
               * host["pii"].astype(np.float64)))
    concentration_floor = 16.0 * eps * 4.0e8
    np.testing.assert_allclose(got["nc"], 250.0e6 / rhoa, rtol=0.0,
                               atol=concentration_floor)
    assert np.all(got["ni"] <= 0.3e6 / rhoa + concentration_floor)
    specs = {
        "r": ("qr", "nr", np.pi * 997.0,
              1.0 / 2800.0e-6, 1.0 / 20.0e-6),
        "i": ("qi", "ni", np.pi * 500.0,
              1.0 / 350.0e-6, 1.0 / 1.0e-6),
        "s": ("qs", "ns", np.pi * 100.0,
              1.0 / 2000.0e-6, 1.0 / 10.0e-6),
        "g": ("qg", "ng", np.pi * (400.0 if morr_rimed_ice == 0 else 900.0),
              1.0 / 2000.0e-6, 1.0 / 20.0e-6),
    }
    for mass, moment, six_c, lam_lo, lam_hi in specs.values():
        active = got[mass] >= 1.0e-14
        lower = got[mass][active] * lam_lo ** 3 / six_c
        upper = got[mass][active] * lam_hi ** 3 / six_c
        assert np.all(got[moment][active] >= lower - concentration_floor)
        assert np.all(got[moment][active] <= upper + concentration_floor)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("regime", ("warm", "mixed", "glaciated", "flagged"))
def test_morrison_kernel_water_conservation_is_sedimentation_flux_aware(regime):
    """FP32 column water plus exported bottom flux closes at its floor."""
    import cupy as cp

    from gpuwm.core.morrison import launch_morrison

    col = _regime_column(regime)
    host = {name: np.ascontiguousarray(value[:, None, None])
            for name, value in col.items()}
    dev = {name: cp.asarray(value) for name, value in host.items()}
    p0 = host["pressure"][:, 0, 0].astype(np.float64)
    t0 = (host["theta"][:, 0, 0].astype(np.float64)
          * host["pii"][:, 0, 0].astype(np.float64))
    rho = p0 / (c.RD * t0)
    before = _water_mass({n: host[n][:, 0, 0] for n in MASS_NAMES}
                         | {"dz": host["dz"][:, 0, 0]}, rho)
    precip = {name: cp.zeros((1, 1), cp.float32) for name in
              ("rainnc", "rainncv", "snownc", "snowncv",
               "graupelnc", "graupelncv", "sr")}
    launch_morrison(**dev, **precip, dt=90.0)
    after_host = {n: cp.asnumpy(dev[n])[:, 0, 0] for n in MASS_NAMES}
    after = _water_mass(after_host | {"dz": host["dz"][:, 0, 0]}, rho)
    exported = float(precip["rainncv"][0, 0])
    rel = abs(after + exported - before) / before
    assert rel <= np.finfo(np.float32).eps
    assert min(a.min() for a in after_host.values()) >= 0.0


def _fall_only_columns(nz=12):
    """Two sedimentation columns: one surface-only, one ordinary.

    Column 0 carries rain and graupel in the lowest model level and nothing
    above it; column 1 is a deep precipitating layer, so one warp holds both
    vertical spans.  Levels are 100 m apart, which puts a 90 s call several
    substeps deep.
    """
    z = 50.0 + 100.0 * np.arange(nz, dtype=np.float64)
    p = 100000.0 * np.exp(-z / 8000.0)
    temp = 290.0 - 0.0065 * z
    pii = (p / c.P0) ** c.RCP
    rho = (p / (c.RD * temp))[:, None]
    active = np.zeros((nz, 2), dtype=bool)
    active[0, 0] = True
    active[:nz - 2, 1] = True
    column = {
        "theta": np.repeat((temp / pii)[:, None], 2, axis=1),
        "pii": np.repeat(pii[:, None], 2, axis=1),
        "pressure": np.repeat(p[:, None], 2, axis=1),
        "rho": np.repeat(rho, 2, axis=1),
        "dz": np.full((nz, 2), 100.0),
        "nc": np.repeat(250.0e6 / rho, 2, axis=1),
        "qr": np.where(active, 8.0e-4, 0.0),
        "nr": np.where(active, 1.5e6 / rho, 0.0),
        "qg": np.where(active, 4.0e-4, 0.0),
        "ng": np.where(active, 8.0e4 / rho, 0.0),
    }
    for name in ("qv", "qc", "qi", "qs", "ni", "ns"):
        column[name] = np.zeros((nz, 2))
    return {name: np.ascontiguousarray(value[:, None, :], dtype=np.float32)
            for name, value in column.items()}


def _launch_morrison_sedimentation(host, dt):
    """Run the sedimentation stage alone over a prepared FP32 column batch."""
    import cupy as cp

    from gpuwm.core import morrison as morr
    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.morrison_constants import rimed_ice_constants

    nz, ny, nx = host["theta"].shape
    dev = {name: cp.asarray(value) for name, value in host.items()}
    precip = {name: cp.zeros((ny, nx), cp.float32) for name in
              ("rainnc", "rainncv", "snownc", "snowncv",
               "graupelnc", "graupelncv", "sr")}
    rimed = rimed_ice_constants(1)
    kernel = ("morrison_sediment_64" if nz <= morr._SHALLOW_KMAX
              else "morrison_sediment_256")
    blocks = (ny * nx + morr._COLUMN_TPB - 1) // morr._COLUMN_TPB
    get_kernel("morrison", kernel)(
        (blocks,), (morr._COLUMN_TPB,),
        (dev["qc"], dev["qr"], dev["qi"], dev["qs"], dev["qg"],
         dev["nc"], dev["nr"], dev["ni"], dev["ns"], dev["ng"],
         dev["nc"], dev["theta"], dev["pii"], dev["pressure"],
         dev["rho"], dev["dz"],
         precip["rainnc"], precip["rainncv"],
         precip["snownc"], precip["snowncv"],
         precip["graupelnc"], precip["graupelncv"], precip["sr"],
         np.float32(dt), np.float32(rimed.ag), np.float32(rimed.bg),
         np.float32(rimed.rhog), np.int32(nz), np.int32(ny), np.int32(nx)))
    return ({name: cp.asnumpy(value) for name, value in dev.items()},
            {name: cp.asnumpy(value) for name, value in precip.items()})


@pytest.mark.gpu
@requires_gpu
def test_morrison_sedimentation_exports_a_category_held_in_the_lowest_level():
    """Rain and graupel confined to k=0 still reach the ground.

    The substep sweep runs only over the levels that carry a fall speed, so
    for this column the top of the span and the level that owns the surface
    export are the same level.  Treating those as separate cases drops the
    export -- rainnc/rainncv/graupelnc/graupelncv/sr all go to zero -- while
    the column mass still falls out of k=0.  Categories that fall nowhere,
    and the levels above each span, must come back untouched.
    """
    nz = 12
    host = _fall_only_columns(nz)
    after, precip = _launch_morrison_sedimentation(host, dt=90.0)

    for name in ("qc", "qi", "qs", "ni", "ns", "nc"):
        np.testing.assert_array_equal(after[name], host[name],
                                      err_msg=f"{name} does not sediment")
    for name in ("qr", "qg", "nr", "ng"):
        np.testing.assert_array_equal(after[name][1:, 0, 0],
                                      np.zeros(nz - 1, np.float32),
                                      err_msg=f"{name} above the span")
        np.testing.assert_array_equal(after[name][nz - 2:, 0, 1],
                                      np.zeros(2, np.float32),
                                      err_msg=f"{name} above the span")

    rho = host["rho"][:, 0, :].astype(np.float64)
    dz = host["dz"][:, 0, :].astype(np.float64)
    fell = {name: np.sum((host[name][:, 0, :].astype(np.float64)
                          - after[name][:, 0, :].astype(np.float64))
                         * rho * dz, axis=0)
            for name in ("qr", "qg")}
    total = precip["rainncv"][0].astype(np.float64)
    graupel = precip["graupelncv"][0].astype(np.float64)
    assert np.all(total > 0.0) and np.all(graupel > 0.0)
    # The surface-only column drained, and the deep column's span really
    # begins at its own top level rather than one below it.
    assert after["qr"][0, 0, 0] < host["qr"][0, 0, 0]
    assert after["qr"][nz - 3, 0, 1] < host["qr"][nz - 3, 0, 1]
    # rainncv is the all-category total and graupelncv the graupel part of
    # it; no other category carries mass here, so the two surface fluxes are
    # exactly the mass that left each column, to the FP32 substep floor.
    np.testing.assert_allclose(graupel, fell["qg"], rtol=1.0e-5)
    np.testing.assert_allclose(total - graupel, fell["qr"], rtol=1.0e-5)
    np.testing.assert_array_equal(precip["snowncv"],
                                  np.zeros((1, 2), np.float32))
    np.testing.assert_allclose(precip["sr"][0], graupel / (total + 1.0e-12),
                               rtol=1.0e-6)
    np.testing.assert_array_equal(precip["rainnc"], precip["rainncv"])


@pytest.mark.gpu
@requires_gpu
def test_morrison_public_dispatch_allocates_two_moment_state():
    """mp=10 uses persistent state fields; mp=1 retains its original path."""
    import cupy as cp

    from gpuwm.core import microphysics
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    cfg = RunConfig(nx=3, ny=1, nz=24, dx=1000.0, dy=1000.0,
                    ztop=12000.0, dt=30.0, run_seconds=0.0,
                    moist=True, mp_physics=10)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, vc, base,
                               lambda z: 0.012 * np.exp(-np.asarray(z) / 3000.0))
    state.qc[...] = 1.0e-3
    update_diagnostics(state)
    microphysics.apply(state, cfg, cfg.dt)
    for name in ("qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"):
        value = getattr(state, name)
        assert value is not None and value.shape == state.p.shape
        assert bool(cp.isfinite(value).all()) and float(value.min()) >= 0.0
    assert float(state.scratch((cfg.ny, cfg.nx), "mp_rainnc").min()) >= 0.0


def test_wk82_morrison_gate_admits_complete_no_pbl_operator():
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases import wk82

    cfg = replace(wk82.default_config(), mp_physics=10,
                  run_seconds=3600.0)
    admitted = validate_run_config(cfg)
    assert admitted.km_opt == 4 and admitted.bl_pbl_physics == 0
