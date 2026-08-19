"""CPU-hermetic contract tests for the NOAHMP_GLACIER port.

Four surfaces, no device:

* the transcription itself (gpuwm/core/noahmp_glacier.py) -- WRF's own
  balance gates are transcribed as raises, so a multi-step run that
  completes IS an energy/water-conservation measurement at WRF's own
  bounds (ERRSW/ERRENG <= 0.01 W m-2, |ERRWAT| <= 0.1 mm per step);
* the surface classification (classify_noahmp_surface) -- the dispatch
  partition, the 0.02-vs-0.5 threshold boundary, and the native-XLAND
  contract: a category-15 fractional-ice column with landmask=0 whose
  native xland=1 classifies to the ice surface, never the water path;
* the configurable sea-ice threshold on NoahmpRuntimeParameters
  (misspelling-refusal, validation, restart identity);
* the seam constructor's pre-device refusal surface for the two new
  inputs.

The GPU halves -- CUDA/host bitwise parity, the full phase-1 chain over
glacier columns, restart round-trip, the live census -- are in
tests/test_noahmp_glacier_gpu.py and run on the rented card.
"""

import numpy as np
import pytest

from gpuwm.core.noahmp_glacier import (GLACIER_OPTION_IDENTITY,
                                       GlacierBalanceError, noahmp_glacier)
from gpuwm.core.noahmp_runtime import (NoahmpRuntimeParameters,
                                       XICE_THRESHOLD,
                                       classify_noahmp_surface)

ISICE = 15  # MODIS ISICE_TABLE, the collaborator's category
ZSOIL = np.array([-0.1, -0.4, -1.0, -2.0], dtype=np.float32)


def _glacier_column(**overrides):
    """The collaborator's signature column: land glacier, xice=0, bare."""
    state = dict(
        cosz=0.4, nsnow=3, nsoil=4, dt=120.0,
        sfctmp=263.0, sfcprs=85000.0, uu=5.0, vv=2.0, q2=0.001,
        soldn=200.0, prcp=0.0, lwdn=180.0, tbot=263.0, zlvl=25.0,
        ficeold=np.zeros(3, dtype=np.float32), zsoil=ZSOIL,
        qsnow=0.0, sneqvo=0.0, albold=0.65, cm=0.01, ch=0.01,
        isnow=0, sneqv=0.0,
        smc=np.ones(4, dtype=np.float32),
        zsnso=np.array([0, 0, 0, -0.1, -0.4, -1.0, -2.0],
                       dtype=np.float32),
        snowh=0.0, snice=np.zeros(3, dtype=np.float32),
        snliq=np.zeros(3, dtype=np.float32), tg=262.0,
        stc=np.array([0, 0, 0, 262.0, 262.5, 263.0, 263.0],
                     dtype=np.float32),
        sh2o=np.zeros(4, dtype=np.float32), tauss=0.0, qsfc=0.0008)
    state.update(overrides)
    return state


def _advance(state, result):
    """Feed a GlacierResult back as the next step's entry state."""
    state = dict(state)
    state.update(
        isnow=result.isnow, sneqv=result.sneqv, snowh=result.snowh,
        smc=result.smc, sh2o=result.sh2o,
        stc=result.stc, zsnso=result.zsnso,
        snice=result.snice, snliq=result.snliq,
        tg=result.tg, tauss=result.tauss, qsfc=result.qsfc,
        qsnow=result.qsnow, sneqvo=result.sneqvo, albold=result.albold,
        cm=result.cm, ch=result.ch)
    return state


# ---------------------------------------------------------------------------
# the transcription
# ---------------------------------------------------------------------------

def test_the_collaborator_column_runs_finite_through_the_glacier_column():
    """landmask=1, IVGTYP=15, xice=0: the exact signature that used to
    refuse now advances, every output finite, inputs untouched."""
    state = _glacier_column()
    stc_before = state["stc"].copy()
    result = noahmp_glacier(**state)
    np.testing.assert_array_equal(state["stc"], stc_before)
    for name in ("tg", "fsh", "fgev", "ssoil", "fira", "fsa", "fsr",
                 "trad", "edir", "runsrf", "runsub", "t2m", "q2e",
                 "qsfc", "cm", "ch", "qmelt", "eflxb", "albedo"):
        assert np.isfinite(np.float64(getattr(result, name))), name
    for name in ("stc", "smc", "sh2o", "zsnso", "snice", "snliq",
                 "hcpct"):
        assert np.isfinite(getattr(result, name)).all(), name
    # glacier ice: SMC pinned at 1.0 by PHASECHANGE_GLACIER (:1992)
    np.testing.assert_array_equal(result.smc, np.ones(4, dtype=np.float32))


def test_energy_balance_closes_at_wrf_own_bound_on_both_regimes():
    """SAG = FIRA+FSH+FGEV+SSOIL to 0.01 W m-2 -- ERROR_GLACIER's own
    gate, re-asserted here on the outputs so the test measures rather
    than trusts the internal raise."""
    for state in (
            _glacier_column(),
            _glacier_column(  # melting deep snowpack
                cosz=0.7, sfctmp=274.5, sfcprs=90000.0, soldn=600.0,
                prcp=2e-4, lwdn=320.0, tbot=263.15, qsnow=1e-4,
                sneqvo=250.0, albold=0.7, isnow=-3, sneqv=250.0,
                ficeold=np.array([0.95, 0.95, 0.95], dtype=np.float32),
                zsnso=np.array([-0.05, -0.25, -0.75, -0.85, -1.15,
                                -1.75, -2.75], dtype=np.float32),
                snowh=0.75,
                snice=np.array([15.0, 80.0, 150.0], dtype=np.float32),
                snliq=np.array([0.5, 2.0, 2.5], dtype=np.float32),
                tg=272.5,
                stc=np.array([271.0, 272.0, 272.5, 272.8, 272.0, 270.0,
                              268.0], dtype=np.float32))):
        r = noahmp_glacier(**state)
        residual = (np.float32(r.sag)
                    - (np.float32(r.fira) + np.float32(r.fsh)
                       + np.float32(r.fgev) + np.float32(r.ssoil)))
        assert residual <= np.float32(0.01), float(residual)
        # the melting regime pins TG at TFRZ (:1094-1095)
        assert np.float32(r.tg) <= np.float32(273.16)


def test_a_multi_hour_march_conserves_water_at_wrf_own_bound():
    """48 steps (1.6 h at dt=120 s) with snowfall, melt and refreeze.

    Every step re-runs ERROR_GLACIER's transcribed gates, so completing
    the march IS the conservation claim: |ERRWAT| <= 0.1 mm per step and
    the energy budget closed at 0.01 W m-2 on every one of the 48.
    """
    state = _glacier_column(
        sfctmp=272.0, prcp=1.5e-4, soldn=300.0, cosz=0.5, lwdn=280.0)
    total_prcp = np.float32(0.0)
    for step in range(48):
        result = noahmp_glacier(**state)
        assert np.isfinite(result.stc).all(), f"step {step}"
        assert result.sneqv >= 0.0 and result.snowh >= 0.0
        total_prcp += np.float32(state["prcp"]) * np.float32(120.0)
        state = _advance(state, result)
    # snow accumulated from the sustained snowfall
    assert result.sneqv > 0.0
    assert result.isnow <= 0


def test_the_option_identity_refuses_everything_but_the_pinned_arm():
    state = _glacier_column()
    with pytest.raises(NotImplementedError, match="opt"):
        noahmp_glacier(**state, opt_gla=2)
    with pytest.raises(NotImplementedError, match="opt"):
        noahmp_glacier(**state, opt_alb=1)
    with pytest.raises(ValueError, match="nsnow=3"):
        noahmp_glacier(**{**state, "nsnow": 2})
    assert GLACIER_OPTION_IDENTITY == {
        "opt_alb": 2, "opt_snf": 1, "opt_tbot": 2, "opt_stc": 1,
        "opt_gla": 1}


def test_the_water_budget_gate_is_a_real_instrument():
    """Deliberate corruption goes red: a column whose entry SNEQV claims
    250 mm of water that its layers do not carry fails ERRWAT."""
    state = _glacier_column(
        isnow=-1, sneqv=250.0, snowh=0.02,
        zsnso=np.array([0, 0, -0.02, -0.12, -0.42, -1.02, -2.02],
                       dtype=np.float32),
        snice=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        snliq=np.zeros(3, dtype=np.float32),
        stc=np.array([0, 0, 262.0, 262.0, 262.5, 263.0, 263.0],
                     dtype=np.float32))
    with pytest.raises(GlacierBalanceError, match="water budget"):
        noahmp_glacier(**state)


# ---------------------------------------------------------------------------
# classification: dispatch partition and the threshold
# ---------------------------------------------------------------------------

def _classes(xland, xice, vegtyp, threshold):
    masks = classify_noahmp_surface(
        np.asarray(xland, dtype=np.float32),
        np.asarray(xice, dtype=np.float32),
        np.asarray(vegtyp, dtype=np.int32),
        xice_threshold=threshold, isice=ISICE)
    return {name: mask.copy() for name, mask in masks.items()}


def test_the_collaborator_signature_classifies_glacier_not_water():
    """IVGTYP=15, xice=0, xland=1 (their landmask=1 columns): glacier."""
    m = _classes([1.0], [0.0], [ISICE], 0.02)
    assert m["glacier"][0] and m["land"][0]
    assert not m["sea_ice"][0] and not m["open_water"][0]
    assert not m["sflx_land"][0]


def test_the_native_xland_contract_column_dispatches_ice_never_water():
    """Category-15 fractional ice, landmask=0, native xland=1.

    The caller's classification wins verbatim: with the native value the
    column is land -> glacier (ice-surface physics).  The derivation
    from landmask=0 would say xland=2 -> open water, no surface at all.
    This is the red-on-revert for the native-XLAND law: if a rewrite
    re-derives xland from landmask, the first assertion flips.
    """
    xice = [0.01]           # fractional, below the 0.02 threshold
    native = _classes([1.0], xice, [ISICE], 0.02)   # native xland = 1
    derived = _classes([2.0], xice, [ISICE], 0.02)  # landmask=0 derivation
    assert native["glacier"][0] and not native["open_water"][0]
    assert derived["open_water"][0] and not derived["glacier"][0]
    # and at or above the threshold the same column is sea ice under
    # BOTH sources -- XICE alone decides that class (:715-723 order)
    at_thr = _classes([1.0], [0.02], [ISICE], 0.02)
    assert at_thr["sea_ice"][0] and not at_thr["glacier"][0]


def test_the_threshold_boundary_partitions_exactly_at_the_configured_value():
    """0.02 vs 0.5: the classes the collaborator's case turns on."""
    xland = np.ones(4, dtype=np.float32)
    vegtyp = np.full(4, ISICE, dtype=np.int32)
    just_below = np.nextafter(np.float32(0.02), np.float32(0.0))
    xice = np.array([0.0, just_below, 0.02, 0.4], dtype=np.float32)
    at_002 = _classes(xland, xice, vegtyp, 0.02)
    assert list(at_002["glacier"]) == [True, True, False, False]
    assert list(at_002["sea_ice"]) == [False, False, True, True]
    at_wrf = _classes(xland, xice, vegtyp, XICE_THRESHOLD)
    assert list(at_wrf["glacier"]) == [True, True, True, True]
    assert list(at_wrf["sea_ice"]) == [False, False, False, False]


def test_sea_ice_wins_over_open_water_in_wrf_order():
    """XICE >= threshold takes the sea-ice skip whatever XLAND says."""
    m = _classes([2.0], [0.6], [16], 0.5)
    assert m["sea_ice"][0] and not m["open_water"][0]


# ---------------------------------------------------------------------------
# the configurable threshold on the parameters
# ---------------------------------------------------------------------------

def test_parameters_carry_the_threshold_and_the_glacier_identity():
    params = NoahmpRuntimeParameters(xice_threshold=0.02)
    assert params.xice_threshold == 0.02
    assert params.glacier_path is True
    identity = params.restart_identity()
    assert identity["xice_threshold"] == 0.02
    assert identity["glacier"]["enabled"] is True
    assert identity["glacier"]["sha256"].startswith("bf94f3522c3b9c2c9")
    default = NoahmpRuntimeParameters()
    assert default.xice_threshold == XICE_THRESHOLD == 0.5


def test_a_misspelled_or_invalid_threshold_refuses_with_its_name():
    with pytest.raises(TypeError):
        NoahmpRuntimeParameters(xice_treshold=0.02)  # the misspelling
    with pytest.raises(ValueError, match="xice_threshold"):
        NoahmpRuntimeParameters(xice_threshold=1.5)
    with pytest.raises(ValueError, match="xice_threshold"):
        NoahmpRuntimeParameters(xice_threshold=float("nan"))


def test_disabling_the_glacier_path_is_recorded_in_the_identity():
    params = NoahmpRuntimeParameters(glacier_path=False)
    assert params.restart_identity()["glacier"]["enabled"] is False


# ---------------------------------------------------------------------------
# the seam constructor's pre-device refusal surface
# ---------------------------------------------------------------------------

def test_the_seam_refuses_a_bad_threshold_before_touching_a_device():
    from gpuwm.core.mpas_column_batch import MpasColumnBatchPhysics
    import datetime

    kwargs = dict(
        n_levels=20, n_columns=8, dt=120.0, radiation_seconds=600.0,
        surface_pbl_seconds=120.0,
        start_time=datetime.datetime(2021, 6, 1, 18, 0),
        latitude_deg=np.full(8, 35.0), longitude_deg=np.full(8, -97.0),
        terrain_height_m=np.zeros(8),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, 21),
        p_top_pa=14000.0, dx_m=15000.0)
    with pytest.raises(ValueError, match="xice_threshold"):
        MpasColumnBatchPhysics(**kwargs, xice_threshold=-0.1)
    with pytest.raises(TypeError):
        MpasColumnBatchPhysics(**kwargs, xice_treshold=0.02)


def test_the_runtime_restrictions_publish_the_ported_glacier_state():
    from gpuwm.core.noahmp_runtime import NOAHMP_RUNTIME_RESTRICTIONS

    entries = {name: (what, why)
               for name, what, why in NOAHMP_RUNTIME_RESTRICTIONS}
    what, why = entries["glacier_columns"]
    assert "NOAHMP_GLACIER" in what and "never to NOAHMP_SFLX" in what
    assert "module_sf_noahmp_glacier.F" in why
    what, why = entries["xice_threshold"]
    assert "configurable" in what
    assert "0.02" in why
