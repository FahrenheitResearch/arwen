"""EXPERIMENTAL (ArWen v1.2): rung-1 reflectivity nudging.

CPU-only by construction (no CuPy import), so the whole file runs
without a device.

The load-bearing properties of a nudging scheme are not its typical
output but its edges: that it does nothing when the forecast is already
right, that its bounds hold against inputs chosen to break them, and
that the branch most likely to hurt is off unless someone asks for it.
Those are what this file spends its assertions on.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import constants as c
from gpuwm.da import hotstart
from gpuwm.da.hotstart import HotStartConfig


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _state(nz=6, ny=2, nx=2, *, pressure=8.0e4, temperature=285.0,
           top_m=12000.0, qv=6.0e-3):
    """Duck-typed state carrying only what the nudge reads."""
    s = SimpleNamespace()
    s.php = np.zeros((nz + 1, ny, nx), np.float32)
    s.phb = np.asarray(
        np.float32(c.G) * np.linspace(0.0, top_m, nz + 1), np.float32)
    s.p = np.full((nz, ny, nx), pressure, np.float32)
    s.thp = np.zeros((nz, ny, nx), np.float32)
    theta = temperature * (np.float32(c.P0) / pressure) ** np.float32(c.RCP)
    s.thb = np.full((nz,), theta, np.float32)
    s.qv = np.full((nz, ny, nx), qv, np.float32)
    for name in ("qr", "qs", "qg"):
        setattr(s, name, np.zeros((nz, ny, nx), np.float32))
    return s


def _grids(state, *, z_obs, sim, mask=True):
    shape = state.p.shape
    return (np.full(shape, z_obs, np.float32),
            np.full(shape, bool(mask)),
            np.full(shape, sim, np.float32))


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def test_the_clear_air_branch_is_off_by_default():
    """The half of the scheme most likely to hurt must be opt-in."""
    assert HotStartConfig().clear_air_enabled is False


def test_defaults_are_conservative():
    cfg = HotStartConfig()
    assert cfg.max_theta_increment_k <= 3.0
    assert cfg.max_qv_increment_kg_kg <= 3.0e-3
    assert cfg.deficit_threshold_db > 0.0, (
        "a zero deficit threshold would nudge a perfect forecast")
    assert cfg.warm_base_height_m > 0.0, (
        "the warm layer must not reach the surface")


def test_a_bad_ramp_shape_is_rejected():
    with pytest.raises(ValueError, match="ramp_shape"):
        HotStartConfig(ramp_shape="gaussian")


def test_an_inverted_warm_layer_is_rejected():
    with pytest.raises(ValueError, match="must exceed"):
        HotStartConfig(warm_base_height_m=8000.0, warm_top_height_m=1000.0)


def test_negative_bounds_are_rejected():
    with pytest.raises(ValueError, match="max_theta_increment_k"):
        HotStartConfig(max_theta_increment_k=-1.0)


def test_an_rh_target_given_as_a_percentage_is_rejected():
    """95 is not 0.95, and silently treating it as one would uncap the
    moisture target."""
    with pytest.raises(ValueError, match="fraction"):
        HotStartConfig(rh_target=95.0)


def test_the_config_must_actually_be_a_config():
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=40.0, sim=0.0)
    with pytest.raises(TypeError, match="HotStartConfig"):
        hotstart.hotstart_increments(state, z_obs, mask,
                                     SimpleNamespace(echo_threshold_dbz=15.0),
                                     simulated_dbz=sim)


# --------------------------------------------------------------------------
# vertical ramp
# --------------------------------------------------------------------------

def test_the_ramp_vanishes_outside_the_warm_layer():
    cfg = HotStartConfig(warm_base_height_m=1000.0, warm_top_height_m=8000.0)
    heights = np.array([0.0, 500.0, 999.0, 1000.0, 4500.0, 8000.0, 8001.0,
                        15000.0])
    weight = hotstart.vertical_ramp(heights, cfg)
    assert weight[0] == 0.0 and weight[1] == 0.0 and weight[2] == 0.0
    assert weight[6] == 0.0 and weight[7] == 0.0
    # sin2 is zero at both ends of its support and one in the middle.
    assert weight[3] == pytest.approx(0.0, abs=1e-15)
    assert weight[5] == pytest.approx(0.0, abs=1e-15)
    assert weight[4] == pytest.approx(1.0, rel=1e-12)


def test_every_ramp_shape_stays_within_zero_and_one():
    heights = np.linspace(-2000.0, 20000.0, 401)
    for shape in hotstart.RAMP_SHAPES:
        weight = hotstart.vertical_ramp(heights, HotStartConfig(
            ramp_shape=shape))
        assert weight.min() >= 0.0
        assert weight.max() <= 1.0 + 1e-12


def test_the_uniform_ramp_is_a_top_hat():
    cfg = HotStartConfig(ramp_shape="uniform", warm_base_height_m=1000.0,
                         warm_top_height_m=8000.0)
    heights = np.array([999.0, 1000.0, 4000.0, 8000.0, 8001.0])
    weight = hotstart.vertical_ramp(heights, cfg)
    np.testing.assert_array_equal(weight, [0.0, 1.0, 1.0, 1.0, 0.0])


def test_the_tent_ramp_peaks_mid_layer():
    cfg = HotStartConfig(ramp_shape="tent", warm_base_height_m=0.0,
                         warm_top_height_m=1000.0)
    weight = hotstart.vertical_ramp(
        np.array([0.0, 250.0, 500.0, 750.0, 1000.0]), cfg)
    np.testing.assert_allclose(weight, [0.0, 0.5, 1.0, 0.5, 0.0])


# --------------------------------------------------------------------------
# the no-op guarantee
# --------------------------------------------------------------------------

def test_a_forecast_that_already_matches_gets_bitwise_zero_increments():
    """The whole scheme must be a no-op on a good forecast."""
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=45.0, sim=45.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    for name, increment in result.increments.items():
        np.testing.assert_array_equal(
            increment, np.zeros_like(increment),
            err_msg=f"{name} moved on an already-correct forecast")
    assert result.provenance["warm_increment_cells"] == 0
    assert result.provenance["theta_increment_sum_k"] == 0.0


def test_a_deficit_below_the_threshold_is_still_a_no_op():
    state = _state()
    cfg = HotStartConfig(deficit_threshold_db=5.0)
    z_obs, mask, sim = _grids(state, z_obs=45.0, sim=41.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))


def test_an_echo_below_the_detection_threshold_is_a_no_op():
    """A 10 dBZ observation is not an echo worth building a storm for,
    even though the model has nothing there at all."""
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=10.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(echo_threshold_dbz=15.0),
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))


def test_a_fully_masked_volume_produces_nothing():
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=60.0, sim=-35.0, mask=False)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))
    np.testing.assert_array_equal(result.increments["qv"],
                                  np.zeros_like(result.increments["qv"]))
    assert result.provenance["valid_obs_cells"] == 0


def test_the_mask_is_honoured_cell_by_cell():
    state = _state(nz=4, ny=1, nx=2)
    shape = state.p.shape
    z_obs = np.full(shape, 55.0, np.float32)
    sim = np.full(shape, -35.0, np.float32)
    mask = np.zeros(shape, bool)
    mask[:, :, 0] = True                       # only the first column
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    d_theta = result.increments["thp"]
    assert np.any(d_theta[:, :, 0] > 0.0)
    np.testing.assert_array_equal(d_theta[:, :, 1],
                                  np.zeros_like(d_theta[:, :, 1]))


# --------------------------------------------------------------------------
# the warm branch does something, in the right place
# --------------------------------------------------------------------------

def test_an_observed_echo_the_model_lacks_gets_warmed_and_moistened():
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=50.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    d_theta = result.increments["thp"]
    d_qv = result.increments["qv"]
    assert np.max(d_theta) > 0.0
    assert np.max(d_qv) > 0.0
    assert np.min(d_theta) >= 0.0, "the warm branch must never cool"
    assert np.min(d_qv) >= 0.0, "the warm branch must never dry"
    assert result.provenance["warm_increment_cells"] > 0


def test_the_increment_lives_only_inside_the_warm_layer():
    """Nothing reaches the surface layer or the upper troposphere."""
    state = _state(nz=12, ny=1, nx=1, top_m=18000.0)
    cfg = HotStartConfig(warm_base_height_m=2000.0, warm_top_height_m=8000.0)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    d_theta = result.increments["thp"][:, 0, 0]

    z_full = np.asarray(state.phb) / np.float32(c.G)
    z_mass = 0.5 * (z_full[:-1] + z_full[1:])
    outside = (z_mass < cfg.warm_base_height_m) | (
        z_mass > cfg.warm_top_height_m)
    np.testing.assert_array_equal(d_theta[outside],
                                  np.zeros(int(outside.sum()), np.float32))
    assert np.any(d_theta[~outside] > 0.0)


def test_a_bigger_deficit_earns_a_bigger_increment():
    state = _state(nz=6, ny=1, nx=3)
    shape = state.p.shape
    z_obs = np.empty(shape, np.float32)
    z_obs[:, :, 0] = 25.0
    z_obs[:, :, 1] = 40.0
    z_obs[:, :, 2] = 55.0
    sim = np.full(shape, 0.0, np.float32)
    mask = np.ones(shape, bool)
    # Lift the ceiling so the ordering is not hidden by the clamp.
    cfg = HotStartConfig(max_theta_increment_k=10.0)
    d_theta = hotstart.hotstart_increments(
        state, z_obs, mask, cfg, simulated_dbz=sim).increments["thp"]
    peak = d_theta.max(axis=0)[0]
    assert peak[0] < peak[1] < peak[2]


def test_the_theta_increment_equals_theta_per_db_times_deficit_at_the_peak():
    """At the ramp maximum the shaping factor is exactly one, so the
    increment is the bare configured sensitivity."""
    state = _state(nz=3, ny=1, nx=1, top_m=9000.0)
    # Mass levels at 1500, 4500, 7500 m; centre the layer on 4500.
    cfg = HotStartConfig(warm_base_height_m=1500.0,
                         warm_top_height_m=7500.0,
                         theta_per_db=0.02, max_theta_increment_k=10.0)
    z_obs, mask, sim = _grids(state, z_obs=50.0, sim=10.0)
    d_theta = hotstart.hotstart_increments(
        state, z_obs, mask, cfg, simulated_dbz=sim).increments["thp"]
    assert float(d_theta[1, 0, 0]) == pytest.approx(0.02 * 40.0, rel=1e-5)


def test_moisture_moves_toward_the_configured_rh_target():
    state = _state(qv=1.0e-3)
    cfg = HotStartConfig(rh_target=0.90, max_qv_increment_kg_kg=1.0e-1)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    d_qv = result.increments["qv"]

    temperature = hotstart._diagnose_temperature(state)
    qsat = hotstart.saturation_mixing_ratio(temperature, state.p)
    target = cfg.rh_target * qsat
    ramp = hotstart.vertical_ramp(hotstart._height_agl(state), cfg)

    # Never overshoots the target ...
    assert np.all(state.qv + d_qv <= target + 1e-9)
    # ... and closes exactly the ramp-weighted fraction of the shortfall,
    # so the moisture adjustment inherits the same vertical shape as the
    # warming rather than stepping at the layer edges.
    np.testing.assert_allclose(d_qv, ramp * (target - state.qv), rtol=1e-5)
    # The premise: some model level sits near the ramp maximum.
    assert float(np.max(ramp)) > 0.9
    assert float(np.max(d_qv)) > 0.0


def _es_ice_murphy_koop_pa(t_kelvin):
    """Murphy and Koop (2005) ice saturation vapour pressure, Pa.

    Written here, not imported: the test is a check on the module, and the
    module's ice branch has to agree with an authority the test states for
    itself.  Same transcription the repo's WPS RH conversion uses
    (gpuwm/ingest/horiz.py), in Pa rather than hPa.
    """
    t = np.asarray(t_kelvin, dtype=np.float64)
    return np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t)
                  - 0.00728332 * t)


def test_saturation_phase_defaults_to_mixed_and_is_validated():
    """The phase is an explicit, stated choice, not a silent liquid path."""
    assert HotStartConfig().saturation_phase == "mixed"
    with pytest.raises(ValueError, match="saturation_phase"):
        HotStartConfig(saturation_phase="plasma")


def test_liquid_phase_is_the_old_behaviour_exactly():
    """phase='liquid' must reproduce the bare Tetens-over-water formula.

    So the change is additive: a caller who deliberately wants liquid-water
    saturation, supersaturation aloft and all, still has it and has to name
    it.
    """
    t = np.full((1, 1, 1), 253.15, np.float32)
    p = np.full((1, 1, 1), 6.0e4, np.float32)
    es = (1000.0 * c.SVP1
          * np.exp(c.SVP2 * (t - c.SVPT0) / (t - c.SVP3)))
    expected = c.EP2 * es / np.maximum(p - es, 1.0)
    got = hotstart.saturation_mixing_ratio(t, p, phase="liquid")
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_ice_phase_uses_murphy_koop_below_freezing():
    """Below freezing, ice saturation is well under liquid saturation.

    This is the whole point: qsat_ice(-20 C) is about 14% below
    qsat_liquid, so targeting the same RH fraction against it stops the
    insertion from driving the vapour into ice supersaturation.
    """
    t = np.full((1, 1, 1), 253.15, np.float32)
    p = np.full((1, 1, 1), 6.0e4, np.float32)
    es_ice = _es_ice_murphy_koop_pa(253.15)
    expected = c.EP2 * es_ice / (6.0e4 - es_ice)
    got = hotstart.saturation_mixing_ratio(t, p, phase="ice")
    np.testing.assert_allclose(got, expected, rtol=2e-4)
    liquid = hotstart.saturation_mixing_ratio(t, p, phase="liquid")
    assert float(got.reshape(-1)[0]) < 0.90 * float(liquid.reshape(-1)[0])


def test_mixed_phase_blends_and_brackets_the_two_pure_phases():
    """Liquid above 273.15 K, ice below 253.15 K, blended between.

    The same 273.15->253.15 linear blend the repo's WPS RH conversion
    uses.  At the two anchors the mixed value equals the pure phase; in the
    mixing band it sits strictly between them.
    """
    p = np.full((1, 1, 1), 6.0e4, np.float32)
    warm = np.full((1, 1, 1), 280.0, np.float32)
    cold = np.full((1, 1, 1), 250.0, np.float32)
    band = np.full((1, 1, 1), 263.15, np.float32)

    np.testing.assert_allclose(
        hotstart.saturation_mixing_ratio(warm, p, phase="mixed"),
        hotstart.saturation_mixing_ratio(warm, p, phase="liquid"), rtol=1e-6)
    np.testing.assert_allclose(
        hotstart.saturation_mixing_ratio(cold, p, phase="mixed"),
        hotstart.saturation_mixing_ratio(cold, p, phase="ice"), rtol=1e-6)
    mixed = float(hotstart.saturation_mixing_ratio(
        band, p, phase="mixed").reshape(-1)[0])
    liquid = float(hotstart.saturation_mixing_ratio(
        band, p, phase="liquid").reshape(-1)[0])
    ice = float(hotstart.saturation_mixing_ratio(
        band, p, phase="ice").reshape(-1)[0])
    assert ice < mixed < liquid


def test_default_mixed_phase_does_not_create_ice_supersaturation():
    """The audit's finding, closed: a 0.95 RH target aloft stays sub-ice.

    With the old liquid-only saturation, rh_target = 0.95 at -20 C put the
    vapour at ~116% RH over ice.  Under the mixed-phase default the target
    is 0.95 of ice saturation there, so the inserted vapour is at 95% over
    ice -- unsaturated, no manufactured deposition on the next physics call.
    """
    state = _state(temperature=253.15, pressure=6.0e4, qv=1.0e-4)
    cfg = HotStartConfig(rh_target=0.95, max_qv_increment_kg_kg=1.0e-1)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    qv_post = state.qv + result.increments["qv"]

    temperature = hotstart._diagnose_temperature(state)
    es_ice = _es_ice_murphy_koop_pa(np.asarray(temperature, np.float64))
    qsat_ice = c.EP2 * es_ice / (np.asarray(state.p, np.float64) - es_ice)
    rh_ice = qv_post / qsat_ice
    # Where the insertion actually raised the moisture, it stayed under ice
    # saturation rather than shooting past it.
    raised = result.increments["qv"] > 0.0
    assert np.any(raised), "the test column was already saturated"
    assert np.all(rh_ice[raised] <= 1.0 + 1e-6)
    assert float(np.max(rh_ice[raised])) > 0.9      # it did fill toward it


def test_provenance_records_the_saturation_phase():
    state = _state(temperature=253.15, qv=1.0e-4)
    cfg = HotStartConfig(saturation_phase="ice")
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    assert result.provenance["config"]["saturation_phase"] == "ice"


def test_moisture_adjustment_can_be_switched_off():
    state = _state(qv=1.0e-3)
    cfg = HotStartConfig(adjust_moisture=False)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["qv"],
                                  np.zeros_like(result.increments["qv"]))
    assert np.max(result.increments["thp"]) > 0.0


def test_an_already_moist_column_gets_no_moisture_increment():
    """The RH target is a target, not a floor to be added to."""
    state = _state(qv=1.0e-3)
    temperature = hotstart._diagnose_temperature(state)
    qsat = hotstart.saturation_mixing_ratio(temperature, state.p)
    state.qv = (0.99 * qsat).astype(np.float32)
    cfg = HotStartConfig(rh_target=0.80)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["qv"],
                                  np.zeros_like(result.increments["qv"]))


# --------------------------------------------------------------------------
# bounds under adversarial input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("z_obs_value", [
    60.0, 1.0e3, 1.0e6, 1.0e30, float("inf"),
])
def test_increment_bounds_hold_against_absurd_observations(z_obs_value):
    state = _state()
    cfg = HotStartConfig()
    z_obs, mask, sim = _grids(state, z_obs=z_obs_value, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    d_theta = result.increments["thp"]
    d_qv = result.increments["qv"]
    assert np.all(np.isfinite(d_theta)), "a bad ob must not poison the state"
    assert np.all(np.isfinite(d_qv))
    assert np.max(d_theta) <= cfg.max_theta_increment_k
    assert np.min(d_theta) >= -cfg.max_theta_decrement_k
    assert np.max(d_qv) <= cfg.max_qv_increment_kg_kg
    assert np.min(d_qv) >= -cfg.max_qv_decrement_kg_kg


def test_a_nan_observation_produces_no_increment():
    """Every gate is a comparison, and NaN compares False."""
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=float("nan"), sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))
    np.testing.assert_array_equal(result.increments["qv"],
                                  np.zeros_like(result.increments["qv"]))


def test_a_nan_simulated_reflectivity_produces_no_increment():
    """Morrison signals an invalid number moment with a non-finite dBZ;
    that must suppress the nudge, not trigger it."""
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=float("nan"))
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))


def test_the_moisture_increment_can_never_drive_vapour_negative():
    state = _state(qv=1.0e-8)
    cfg = HotStartConfig(clear_air_enabled=True,
                         max_qv_decrement_kg_kg=1.0)
    z_obs, mask, sim = _grids(state, z_obs=-30.0, sim=60.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    assert np.all(state.qv + result.increments["qv"] >= 0.0)


def test_an_enormous_configured_sensitivity_is_still_clamped():
    state = _state()
    cfg = HotStartConfig(theta_per_db=1.0e6, max_theta_increment_k=2.0,
                         max_qv_increment_kg_kg=1.0e-3)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    assert np.max(result.increments["thp"]) == pytest.approx(2.0, rel=1e-6)
    assert np.max(result.increments["qv"]) <= 1.0e-3


# --------------------------------------------------------------------------
# cumulative insertion: caps are per analysis time, not per call
# --------------------------------------------------------------------------

def _apply(state, increments):
    """Add hot-start increments to a duck state, returning a fresh copy."""
    updated = _state(nz=state.p.shape[0], ny=state.p.shape[1],
                     nx=state.p.shape[2])
    for name in ("p", "php", "phb", "thb", "thp", "qv", "qr", "qs", "qg"):
        setattr(updated, name, np.array(getattr(state, name), copy=True))
    updated.thp = (updated.thp + increments["thp"]).astype(np.float32)
    updated.qv = (updated.qv + increments["qv"]).astype(np.float32)
    return updated


def test_two_maximum_warm_applications_do_not_add_four_kelvin():
    """The audit's compounding case: a persistent deficit, applied twice.

    This rung changes thp and qv but not the hydrometeors that dominate
    simulated Z, so recomputing on the updated state does NOT shut the warm
    branch off -- the deficit is still there.  Without a cumulative bound,
    two maximum applications add 4 K under a nominal 2 K cap.  Threading the
    applied-increment ledger makes the cap a TOTAL analysis-time bound: the
    second call gets only the headroom the first left.
    """
    state = _state()
    cfg = HotStartConfig(theta_per_db=100.0)        # saturates the cap
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)

    first = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                         simulated_dbz=sim)
    assert float(np.max(first.increments["thp"])) == pytest.approx(2.0,
                                                                   rel=1e-6)

    state2 = _apply(state, first.increments)
    second = hotstart.hotstart_increments(state2, z_obs, mask, cfg,
                                          simulated_dbz=sim,
                                          applied=first.increments)
    cumulative = first.increments["thp"] + second.increments["thp"]
    assert float(np.max(cumulative)) <= cfg.max_theta_increment_k + 1e-6
    # The second call added essentially nothing, because the first used the
    # whole cap.
    assert float(np.max(second.increments["thp"])) < 1e-6


def test_the_ledger_leaves_headroom_when_the_first_call_did_not_fill_it():
    """A partial first insertion still lets the second finish to the cap."""
    state = _state()
    # theta_per_db * deficit * ramp small enough that one call is well
    # under the cap: deficit is 55 - (-35) = 90 dB, so 0.005 K/dB -> <=0.45 K.
    cfg = HotStartConfig(theta_per_db=0.005)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)

    first = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                         simulated_dbz=sim)
    peak_one = float(np.max(first.increments["thp"]))
    assert 0.0 < peak_one < cfg.max_theta_increment_k

    state2 = _apply(state, first.increments)
    second = hotstart.hotstart_increments(state2, z_obs, mask, cfg,
                                          simulated_dbz=sim,
                                          applied=first.increments)
    # The second call added something (there was headroom) ...
    assert float(np.max(second.increments["thp"])) > 0.0
    # ... but the cumulative never exceeds the cap.
    cumulative = first.increments["thp"] + second.increments["thp"]
    assert float(np.max(cumulative)) <= cfg.max_theta_increment_k + 1e-6


def test_the_cumulative_bound_also_holds_for_moisture():
    state = _state(qv=1.0e-4)
    cfg = HotStartConfig(rh_target=1.5, max_qv_increment_kg_kg=2.0e-4,
                         saturation_phase="liquid")
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    first = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                         simulated_dbz=sim)
    state2 = _apply(state, first.increments)
    second = hotstart.hotstart_increments(state2, z_obs, mask, cfg,
                                          simulated_dbz=sim,
                                          applied=first.increments)
    cumulative = first.increments["qv"] + second.increments["qv"]
    assert float(np.max(cumulative)) <= cfg.max_qv_increment_kg_kg + 1e-12


def test_without_a_ledger_the_caps_are_per_call_and_provenance_says_so():
    """The chosen semantics, stated: no ledger means per-invocation caps.

    A caller that does not thread the ledger gets the documented
    per-invocation behaviour, and the provenance records which of the two
    regimes produced the increment so the compounding is never silent.
    """
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    per_call = hotstart.hotstart_increments(state, z_obs, mask,
                                            HotStartConfig(theta_per_db=100.0),
                                            simulated_dbz=sim)
    assert per_call.provenance["cumulative"] is False
    assert "applied_theta_increment_max_k" not in per_call.provenance

    ledgered = hotstart.hotstart_increments(
        state, z_obs, mask, HotStartConfig(theta_per_db=100.0),
        simulated_dbz=sim,
        applied={"thp": np.zeros_like(state.thp),
                 "qv": np.zeros_like(state.qv)})
    assert ledgered.provenance["cumulative"] is True
    # With a zero ledger the increment matches the per-call one.
    np.testing.assert_array_equal(ledgered.increments["thp"],
                                  per_call.increments["thp"])


def test_a_mismatched_ledger_is_rejected():
    state = _state(nz=4, ny=2, nx=2)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    with pytest.raises(ValueError, match="applied"):
        hotstart.hotstart_increments(
            state, z_obs, mask, HotStartConfig(), simulated_dbz=sim,
            applied={"thp": np.zeros((4, 2, 3), np.float32),
                     "qv": np.zeros((4, 2, 2), np.float32)})


# --------------------------------------------------------------------------
# F7 -- the insertion cap is an invariant of this function, not a
#       convention the caller is trusted with
# --------------------------------------------------------------------------

def test_the_insertion_cap_is_a_config_field_with_a_default():
    """It was advertised at 55 dBZ and implemented nowhere.

    No constant, no clamp, no caller, no CLI wiring and no provenance
    field existed for it, so the stated field-run cap could not be
    reconstructed from the repository at all.  A claim without a mechanism
    is not a cap.
    """
    assert HotStartConfig().max_insertion_dbz == 55.0


@pytest.mark.parametrize("z_obs_value", [55.0, 56.0, 70.0, 95.0])
def test_the_deficit_is_formed_from_the_capped_observation(z_obs_value):
    """At the cap and above it: the increment stops responding.

    Keyed on the observed value across the boundary, because a cap that
    only bound at one value would be indistinguishable from the theta
    ceiling doing the work.  The ceiling is lifted here precisely so the
    cap is the thing under test.
    """
    state = _state(nz=3, ny=1, nx=1, top_m=9000.0)
    cfg = HotStartConfig(warm_base_height_m=1500.0, warm_top_height_m=7500.0,
                         theta_per_db=0.02, max_theta_increment_k=100.0,
                         max_insertion_dbz=55.0)
    z_obs, mask, sim = _grids(state, z_obs=z_obs_value, sim=10.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    # Ramp peak: shaping factor exactly one, so the increment is
    # theta_per_db * (capped deficit) and nothing else.
    assert float(result.increments["thp"][1, 0, 0]) == pytest.approx(
        0.02 * (55.0 - 10.0), rel=1e-5)
    assert result.provenance["insertion_cap_dbz"] == 55.0
    assert result.provenance["capped_obs_cells"] == (
        0 if z_obs_value <= 55.0 else 3)


def test_below_the_cap_the_observation_is_untouched():
    state = _state(nz=3, ny=1, nx=1, top_m=9000.0)
    cfg = HotStartConfig(warm_base_height_m=1500.0, warm_top_height_m=7500.0,
                         theta_per_db=0.02, max_theta_increment_k=100.0)
    z_obs, mask, sim = _grids(state, z_obs=50.0, sim=10.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    assert float(result.increments["thp"][1, 0, 0]) == pytest.approx(
        0.02 * 40.0, rel=1e-5)
    assert result.provenance["capped_obs_cells"] == 0


def test_the_cap_is_configurable_and_the_value_used_is_recorded():
    """A different experiment may want a different ceiling -- by saying so."""
    state = _state(nz=3, ny=1, nx=1, top_m=9000.0)
    cfg = HotStartConfig(warm_base_height_m=1500.0, warm_top_height_m=7500.0,
                         theta_per_db=0.02, max_theta_increment_k=100.0,
                         max_insertion_dbz=45.0)
    z_obs, mask, sim = _grids(state, z_obs=70.0, sim=10.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    assert float(result.increments["thp"][1, 0, 0]) == pytest.approx(
        0.02 * (45.0 - 10.0), rel=1e-5)
    assert result.provenance["insertion_cap_dbz"] == 45.0
    assert result.provenance["config"]["max_insertion_dbz"] == 45.0


def test_an_infinite_observation_is_bounded_by_the_cap_not_by_the_ceiling():
    """The audit's own case: z_obs = +inf reached the clamp, not the cap.

    With the cap implemented the infinite observation becomes a 55 dBZ
    one before the deficit exists, so the increment is a finite, ordinary
    one rather than the theta ceiling saturating.
    """
    state = _state(nz=3, ny=1, nx=1, top_m=9000.0)
    cfg = HotStartConfig(warm_base_height_m=1500.0, warm_top_height_m=7500.0,
                         theta_per_db=0.02, max_theta_increment_k=100.0)
    z_obs, mask, sim = _grids(state, z_obs=float("inf"), sim=10.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    d_theta = result.increments["thp"]
    assert np.all(np.isfinite(d_theta))
    assert float(d_theta[1, 0, 0]) == pytest.approx(0.02 * 45.0, rel=1e-5)
    assert float(np.max(d_theta)) < cfg.max_theta_increment_k


def test_the_cap_does_not_turn_a_missing_observation_into_an_echo():
    """``minimum``, not ``fmin``.

    ``np.fmin(nan, 55)`` is 55: capping with it would manufacture a
    55 dBZ observation everywhere the radar had none, which is the exact
    inverse of the NaN contract.
    """
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=float("nan"), sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))
    assert result.provenance["capped_obs_cells"] == 0


def test_a_cap_below_the_echo_threshold_is_refused():
    """It would silently make the warm branch unreachable."""
    with pytest.raises(ValueError, match="max_insertion_dbz"):
        HotStartConfig(echo_threshold_dbz=15.0, max_insertion_dbz=10.0)


def test_a_nonfinite_cap_is_refused():
    with pytest.raises(ValueError, match="max_insertion_dbz"):
        HotStartConfig(max_insertion_dbz=float("inf"))


# --------------------------------------------------------------------------
# clear-air branch
# --------------------------------------------------------------------------

def test_spurious_model_echo_is_left_alone_by_default():
    """Model says 55 dBZ, radar says clear -- and the default does
    nothing about it."""
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=-30.0, sim=55.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))
    np.testing.assert_array_equal(result.increments["qv"],
                                  np.zeros_like(result.increments["qv"]))
    assert result.provenance["clear_air_enabled"] is False
    assert result.provenance["clear_air_increment_cells"] == 0


def test_the_clear_air_branch_cools_and_dries_when_switched_on():
    state = _state(qv=1.0e-2)
    cfg = HotStartConfig(clear_air_enabled=True)
    z_obs, mask, sim = _grids(state, z_obs=-30.0, sim=55.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    d_theta = result.increments["thp"]
    d_qv = result.increments["qv"]
    assert np.min(d_theta) < 0.0
    assert np.max(d_theta) <= 0.0, "the clear-air branch must never warm"
    assert np.min(d_qv) < 0.0
    assert np.min(d_theta) >= -cfg.max_theta_decrement_k
    assert np.min(d_qv) >= -cfg.max_qv_decrement_kg_kg
    assert result.provenance["clear_air_increment_cells"] > 0


def test_enabling_clear_air_does_not_disturb_the_warm_branch():
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    off = hotstart.hotstart_increments(
        state, z_obs, mask, HotStartConfig(clear_air_enabled=False),
        simulated_dbz=sim).increments
    on = hotstart.hotstart_increments(
        state, z_obs, mask, HotStartConfig(clear_air_enabled=True),
        simulated_dbz=sim).increments
    np.testing.assert_array_equal(off["thp"], on["thp"])
    np.testing.assert_array_equal(off["qv"], on["qv"])


# --------------------------------------------------------------------------
# provenance and determinism
# --------------------------------------------------------------------------

def test_the_provenance_records_the_thresholds_counts_and_totals():
    state = _state(nz=6, ny=2, nx=2)
    cfg = HotStartConfig(echo_threshold_dbz=20.0, deficit_threshold_db=7.0)
    z_obs, mask, sim = _grids(state, z_obs=50.0, sim=0.0)
    result = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    p = result.provenance

    assert p["schema"] == hotstart.PROVENANCE_SCHEMA
    assert p["experimental"] is True
    assert p["obs_schema"] == "gpuwm-obs.radar-grid.v1"
    assert p["config"]["echo_threshold_dbz"] == 20.0
    assert p["config"]["deficit_threshold_db"] == 7.0
    assert p["grid_shape"] == [6, 2, 2]
    assert p["total_cells"] == 24
    assert p["valid_obs_cells"] == 24
    assert p["observed_echo_cells"] == 24
    assert 0 < p["warm_increment_cells"] <= 24
    assert p["theta_increment_sum_k"] > 0.0
    assert p["deterministic"] is True
    assert p["seed_free"] is True


def test_the_provenance_is_json_serialisable():
    import json

    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=50.0, sim=0.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    round_tripped = json.loads(json.dumps(result.provenance))
    assert round_tripped["schema"] == hotstart.PROVENANCE_SCHEMA


def test_the_provenance_flags_a_bound_clamp():
    state = _state()
    cfg = HotStartConfig(theta_per_db=100.0)
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    bound = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                         simulated_dbz=sim).provenance
    assert bound["theta_clamp_bound"] is True

    quiet = hotstart.hotstart_increments(
        state, z_obs, mask, HotStartConfig(theta_per_db=1.0e-6),
        simulated_dbz=sim).provenance
    assert quiet["theta_clamp_bound"] is False


def test_repeated_calls_are_bitwise_identical():
    """No RNG, no clock, no ordering dependence."""
    state = _state(nz=5, ny=3, nx=3)
    rng_free = np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3)
    z_obs = (10.0 + rng_free).astype(np.float32)
    sim = (rng_free * 0.5).astype(np.float32)
    mask = (rng_free % 3 != 0)
    cfg = HotStartConfig()
    first = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                         simulated_dbz=sim)
    second = hotstart.hotstart_increments(state, z_obs, mask, cfg,
                                          simulated_dbz=sim)
    for name in first.increments:
        np.testing.assert_array_equal(first.increments[name],
                                      second.increments[name])
    assert first.provenance == second.provenance


def test_the_operator_never_mutates_the_state():
    state = _state()
    before = {name: np.array(getattr(state, name), copy=True)
              for name in ("p", "thp", "qv", "php")}
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    hotstart.hotstart_increments(state, z_obs, mask,
                                 HotStartConfig(clear_air_enabled=True),
                                 simulated_dbz=sim)
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(state, name), original)


def test_the_increment_keys_name_state_fields():
    state = _state()
    z_obs, mask, sim = _grids(state, z_obs=55.0, sim=-35.0)
    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(),
                                          simulated_dbz=sim)
    assert set(result.increments) == {"thp", "qv"}
    for name, increment in result.increments.items():
        assert increment.shape == getattr(state, name).shape
        # Applying the increment is a plain add.
        updated = getattr(state, name) + increment
        assert updated.shape == increment.shape


# --------------------------------------------------------------------------
# fail-closed contract
# --------------------------------------------------------------------------

def test_a_missing_simulated_field_refuses_to_assume_clear_air():
    state = _state()
    z_obs, mask, _ = _grids(state, z_obs=55.0, sim=0.0)
    with pytest.raises(ValueError, match="simulated_dbz"):
        hotstart.hotstart_increments(state, z_obs, mask, HotStartConfig())


def test_mismatched_observation_shapes_are_rejected():
    state = _state(nz=4, ny=2, nx=2)
    wrong = np.zeros((4, 2, 3), np.float32)
    mask = np.ones((4, 2, 2), bool)
    sim = np.zeros((4, 2, 2), np.float32)
    with pytest.raises(ValueError, match="z_obs has shape"):
        hotstart.hotstart_increments(state, wrong, mask, HotStartConfig(),
                                     simulated_dbz=sim)


def test_a_mismatched_mask_is_rejected():
    state = _state(nz=4, ny=2, nx=2)
    z_obs = np.zeros((4, 2, 2), np.float32)
    with pytest.raises(ValueError, match="z_mask has shape"):
        hotstart.hotstart_increments(state, z_obs, np.ones((4, 2, 3), bool),
                                     HotStartConfig(),
                                     simulated_dbz=z_obs)


# --------------------------------------------------------------------------
# end to end with the bound Z operator
# --------------------------------------------------------------------------

def test_the_nudge_composes_with_the_bound_reflectivity_operator():
    """run_cfg= routes through gpuwm.da.obsop rather than needing a
    hand-supplied simulated field."""
    state = _state(nz=6, ny=2, nx=2)
    state.qv[...] = 6.0e-3
    run_cfg = SimpleNamespace(mp_physics=1)
    shape = state.p.shape
    z_obs = np.full(shape, 50.0, np.float32)
    mask = np.ones(shape, bool)

    result = hotstart.hotstart_increments(state, z_obs, mask,
                                          HotStartConfig(), run_cfg=run_cfg)
    # The model is dry, so every in-layer cell is deficient.
    assert result.provenance["warm_increment_cells"] > 0
    assert np.max(result.increments["thp"]) > 0.0

    # Supplying the same Z by hand must give the identical answer.
    from gpuwm.da import obsop
    sim = obsop.simulated_reflectivity(state, run_cfg)
    explicit = hotstart.hotstart_increments(state, z_obs, mask,
                                            HotStartConfig(),
                                            simulated_dbz=sim)
    np.testing.assert_array_equal(result.increments["thp"],
                                  explicit.increments["thp"])


def test_a_model_with_the_observed_rain_is_not_nudged():
    """The end-to-end no-op: give the model enough rain to match the
    observation and the scheme must stand down."""
    state = _state(nz=6, ny=1, nx=1)
    state.qr[...] = 1.0e-3
    run_cfg = SimpleNamespace(mp_physics=1)
    from gpuwm.da import obsop
    sim = obsop.simulated_reflectivity(state, run_cfg)

    shape = state.p.shape
    mask = np.ones(shape, bool)
    result = hotstart.hotstart_increments(state, np.array(sim), mask,
                                          HotStartConfig(), run_cfg=run_cfg)
    np.testing.assert_array_equal(result.increments["thp"],
                                  np.zeros_like(result.increments["thp"]))
    np.testing.assert_array_equal(result.increments["qv"],
                                  np.zeros_like(result.increments["qv"]))
