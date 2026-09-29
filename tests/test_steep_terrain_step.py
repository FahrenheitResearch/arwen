"""Steep and high terrain through the production step().

Two derivations read a domain's ground before the run and change how it is
integrated: the acoustic substep count (:mod:`gpuwm.acoustic_adaptation`)
and the hybrid coordinate's etac (:mod:`gpuwm.vertical_adaptation`).  Each
test here builds a bell ridge with the generated dynamics, ladder and clock,
shows the configured value running away, and the derived value holding on
the same ground.  They live apart from the derivations' own tests because a
module that touches the device is skipped whole on CPU-only runs.
"""
from dataclasses import replace

import numpy as np
import pytest
from conftest import requires_gpu

import gpuwm.core.constants as c
from gpuwm.acoustic_adaptation import derive_acoustics, steepest_slope
from gpuwm.core.grid import make_vertical_coord
from gpuwm.domain_wizard import _ETA_LEVELS
from gpuwm.vertical_adaptation import (MIN_LAYER_FRACTION, TerrainField,
                                       survey_vertical_coordinate)

LADDER = np.asarray(_ETA_LEVELS, dtype=np.float64)

#: The highest 1 km cell under the generated central-Andes domain.
BARELY_ORDERED_PEAK_M = 6456.0


def _ridge_case(dx, slope, time_step_sound, *, h0=3000.0, u=20.0):
    """A bell ridge with the generated dynamics, in a uniform cross wind."""
    import math

    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core import constants as c
    from gpuwm.core.dycore import set_w_surface
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.terrain import bell_hill
    from gpuwm.domain_wizard import _ETA_LEVELS

    eta = tuple(_ETA_LEVELS)
    cfg = RunConfig(
        nx=96, ny=8, nz=len(eta) - 1, dx=dx, dy=dx, ztop=20000.0,
        dt=5.0 * dx / 1000.0, run_seconds=120.0,
        time_step_sound=time_step_sound, epssm=0.5, smdiv=0.1, emdiv=0.01,
        damp_opt=3, zdamp=5000.0, dampcoef=0.2, w_damping=1,
        terrain_opt=1, hill_height=h0,
        hill_halfwidth=(3.0 * math.sqrt(3.0) / 8.0) * h0 / slope,
        hybrid_opt=2, etac=0.2, top_lid=False, diff_6th_opt=2,
        diff_6th_factor=0.12, diff_6th_slopeopt=1, h_sca_adv_order=5,
        eta_levels=eta)
    coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                                eta_levels=np.asarray(eta))
    terrain = bell_hill(cfg)
    base = make_base_state(
        coord,
        lambda z: 290.0 * np.exp(1.0e-4 * np.asarray(z, dtype=np.float64)
                                 / c.G),
        p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
    state.u[...] = cp.float32(u)
    set_w_surface(state, cfg)
    state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
    return cfg, state, terrain


def _peak_w(cfg, state, seconds):
    import cupy as cp
    from gpuwm.core.dycore import step

    peak = 0.0
    for _ in range(int(round(seconds / cfg.dt))):
        step(state, cfg)
        value = float(cp.abs(state.w).max())
        if not np.isfinite(value):
            return float("inf")
        peak = max(peak, value)
    return peak


@requires_gpu
@pytest.mark.gpu
def test_steep_ridge_holds_with_the_derived_count():
    """Through the production step(): the Andes 500 m slope blows up with
    the configured four substeps and holds with the count derived for it."""
    cfg, state, terrain = _ridge_case(500.0, 0.87, 4)
    reading = steepest_slope(terrain, cfg.dx, cfg.dy, label="d01")
    adaptation = derive_acoustics(1, cfg, reading)
    assert adaptation.time_step_sound == 6
    assert _peak_w(cfg, state, 60.0) > 150.0
    derived = replace(cfg, time_step_sound=adaptation.time_step_sound)
    _cfg, state, _terrain = _ridge_case(500.0, 0.87, 4)
    assert _peak_w(derived, state, 120.0) < 60.0


@requires_gpu
@pytest.mark.gpu
def test_moderate_ridge_is_left_on_four_and_holds():
    cfg, state, terrain = _ridge_case(500.0, 0.6, 4)
    reading = steepest_slope(terrain, cfg.dx, cfg.dy, label="d01")
    assert derive_acoustics(1, cfg, reading).time_step_sound == 4
    assert _peak_w(cfg, state, 120.0) < 60.0


def _crest_case(etac, *, h0=BARELY_ORDERED_PEAK_M, slope=0.6, u=40.0,
                dx=1000.0):
    """A bell ridge as tall as the Andes crest, in a uniform cross wind,
    on the generated dynamics, ladder and clock, at one etac."""
    import math

    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import set_w_surface
    from gpuwm.core.grid import make_base_state
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.terrain import bell_hill

    eta = tuple(_ETA_LEVELS)
    cfg = RunConfig(
        nx=96, ny=8, nz=len(eta) - 1, dx=dx, dy=dx, ztop=20000.0,
        dt=5.0 * dx / 1000.0, run_seconds=900.0, time_step_sound=4,
        epssm=0.5, smdiv=0.1, emdiv=0.01, damp_opt=3, zdamp=5000.0,
        dampcoef=0.2, w_damping=1, terrain_opt=1, hill_height=h0,
        hill_halfwidth=(3.0 * math.sqrt(3.0) / 8.0) * h0 / slope,
        hybrid_opt=2, etac=etac, top_lid=False, diff_6th_opt=2,
        diff_6th_factor=0.12, diff_6th_slopeopt=1, h_sca_adv_order=5,
        eta_levels=eta)
    terrain = bell_hill(cfg)
    # The substeps the terrain rule gives this ground, so only the
    # coordinate differs between the two runs.
    count = derive_acoustics(1, cfg, steepest_slope(
        terrain, cfg.dx, cfg.dy, label="d01")).time_step_sound
    cfg = replace(cfg, time_step_sound=count)
    coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=etac,
                                eta_levels=np.asarray(eta))
    base = make_base_state(
        coord,
        lambda z: 290.0 * np.exp(1.0e-4 * np.asarray(z, dtype=np.float64)
                                 / c.G),
        p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
    state.u[...] = cp.float32(u)
    set_w_surface(state, cfg)
    state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
    return cfg, state, terrain, float(coord.p_top)


def _holds_for(cfg, state, seconds):
    import cupy as cp
    from gpuwm.core.dycore import step

    for _ in range(int(round(seconds / cfg.dt))):
        step(state, cfg)
        peak = float(cp.abs(state.w).max())
        if not np.isfinite(peak) or peak > 500.0:
            return False
    return True


@requires_gpu
@pytest.mark.gpu
def test_a_barely_ordered_crest_holds_on_the_derived_coordinate():
    """Through the production step(): etac 0.2 orders a 6456 m crest with
    its layer 20 a sliver deep, and a 40 m/s cross wind runs that layer
    away within 300 s; the etac the survey derives for the same ground
    holds three times as long."""
    cfg, state, terrain, p_top = _crest_case(0.2)
    assert cfg.time_step_sound == 4
    assert not _holds_for(cfg, state, 300.0)
    adaptation = survey_vertical_coordinate(
        LADDER, 2, 0.2, p_top, [TerrainField("d01 static terrain", terrain)])
    assert adaptation.configured_ordered and adaptation.adapted
    assert adaptation.layer_fraction >= MIN_LAYER_FRACTION
    derived, state, _terrain, _p_top = _crest_case(adaptation.etac)
    assert _holds_for(derived, state, 900.0)


@requires_gpu
@pytest.mark.gpu
def test_a_jet_over_a_3km_ridge_holds_on_the_derived_step():
    """Through the production step(): a 4.5 km crest of slope 0.4 at 3 km
    under a 70 m/s crest-level wind stops within minutes on the generated
    15 s step with six substeps; the step the terrain clock derives for it
    from the same slope, crest and wind holds half an hour."""
    from fractions import Fraction

    from gpuwm.terrain_clock import CrestWind, derive_clock

    cfg, state, terrain = _ridge_case(3000.0, 0.4, 6, h0=4500.0, u=70.0)
    assert _peak_w(cfg, state, 300.0) > 150.0
    reading = steepest_slope(terrain, cfg.dx, cfg.dy, label="d01")
    wind = CrestWind(label="d01", crest_height_m=float(terrain.max()),
                     wind_m_s=70.0, when="start", source="d01",
                     height_m=float(terrain.max()))
    adaptation = derive_clock(1, replace(cfg, time_step_sound=4),
                              Fraction(15), reading.slope, wind)
    assert adaptation.division >= 2
    derived = replace(cfg, dt=float(adaptation.dt),
                      time_step_sound=adaptation.time_step_sound)
    _cfg, state, _terrain = _ridge_case(3000.0, 0.4, 6, h0=4500.0, u=70.0)
    assert _peak_w(derived, state, 1800.0) < 4.0 * 70.0 * 0.4 + 20.0


@requires_gpu
@pytest.mark.gpu
def test_steep_ridge_holds_on_the_adaptive_clock_with_the_floor():
    """Through the production step(): on the adaptive clock the substep
    count comes from the step, and at the generated 500 m step that is
    four, which the Andes 500 m slope blows up on.  The six the substep
    rule derives reach the step as the floor that clock keeps."""
    from fractions import Fraction

    from gpuwm.acoustic_adaptation import adapted_run
    from gpuwm.core.adaptive_clock import adaptive_sound_steps

    cfg, state, terrain = _ridge_case(500.0, 0.87, 4)
    adaptive = replace(cfg, use_adaptive_time_step=True)
    reading = steepest_slope(terrain, cfg.dx, cfg.dy, label="d01")
    floored = adapted_run(
        adaptive, derive_acoustics(1, adaptive, reading).time_step_sound)
    assert floored.min_time_step_sound == 6
    dt = Fraction(cfg.dt).limit_denominator(100)
    bare = replace(adaptive, time_step_sound=adaptive_sound_steps(dt, adaptive))
    live = replace(floored, time_step_sound=adaptive_sound_steps(dt, floored))
    assert (bare.time_step_sound, live.time_step_sound) == (4, 6)
    assert _peak_w(bare, state, 60.0) > 150.0
    _cfg, state, _terrain = _ridge_case(500.0, 0.87, 4)
    assert _peak_w(live, state, 120.0) < 60.0
