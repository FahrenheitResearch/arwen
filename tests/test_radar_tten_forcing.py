"""Radar latent heating, the model side, on the card.

WRF (NOAA-EMC/HRRR v4.1.21, ``dyn_em/module_big_step_utilities_em.F``):
the slot is chosen at ``:5913-5938``; at ``:5991-6005`` a point whose slot
value lies in [-1, 1] K/s below the top level takes ``tendency * dt`` in
place of the microphysics increment, every other point takes the
microphysics increment, and ``h_diabatic`` is the microphysics rate either
way (``:6014``).  All of it sits inside ``no_mp_heating == 0`` (``:5949``)
and on the tile clipped by the specified zone (solve_em.F:3618-3622).
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig

pytestmark = [pytest.mark.gpu, requires_gpu]


def _moist_state(nx=10, ny=8, nz=30, mp=1, dt=30.0, specified=False):
    """A balanced state, saturated and cloudy below 3 km, condensing."""
    import cupy as cp
    from gpuwm.core import constants as c
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.state import DTYPE

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0,
                    ztop=10000.0, dt=dt, run_seconds=0.0, moist=True,
                    mp_physics=mp, specified=specified, spec_zone=1)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_moist_balanced(cfg, vc, base, lambda z: 0.0 * np.asarray(z, float))
    zc = s.height_half()[:, None, None]
    rng = np.random.default_rng(3)
    qc = np.where((zc > 500.0) & (zc < 3000.0), 2.0e-3, 0.0) \
        * rng.uniform(0.5, 1.5, size=s.p.shape)
    s.qc[...] = cp.asarray(qc, dtype=DTYPE)

    def qvs():
        th = base.thb[:, None, None] + cp.asnumpy(s.thp).astype(np.float64)
        pii = (cp.asnumpy(s.p).astype(np.float64) / c.P0) ** c.RCP
        t = th * pii
        es = 1000.0 * c.SVP1 * np.exp(c.SVP2 * (t - c.SVPT0) / (t - c.SVP3))
        return c.EP2 * es / (cp.asnumpy(s.p).astype(np.float64) - es)

    for _ in range(6):
        qv = np.where(zc < 3000.0, 1.02 * qvs(), 0.3 * qvs())
        s.qv[...] = cp.asarray(qv, dtype=DTYPE)
        update_diagnostics(s)
    return s, cfg


def _with_lim(cfg, lim):
    return RunConfig(**{**cfg.__dict__, "mp_tend_lim": float(lim)})


def _mixed_slot(shape, seed=5):
    """Covered values (both signs, the window's edges), zeros, -20, values
    just outside the window, and a flag on top."""
    rng = np.random.default_rng(seed)
    slot = rng.uniform(-0.004, 0.01, size=shape)
    pick = rng.integers(0, 6, size=shape)
    slot = np.where(pick == 0, 0.0, slot)
    slot = np.where(pick == 1, -20.0, slot)
    slot = np.where(pick == 2, rng.choice([-1.0, 1.0, -1.0001, 1.0001],
                                          size=shape), slot)
    slot = slot.astype(np.float32)
    slot[-1] = rng.choice([-10.0, 0.0, 1.0], size=shape[1:])
    return slot


def test_one_finish_call_takes_the_tendency_only_where_radar_covers():
    import cupy as cp
    from gpuwm.core import microphysics
    from gpuwm.da import radar_tten

    plain, cfg = _moist_state(specified=True)
    forced, _ = _moist_state(specified=True)
    thp0 = cp.asnumpy(forced.thp).copy()
    np.testing.assert_array_equal(cp.asnumpy(plain.thp), thp0)
    slot = _mixed_slot(thp0.shape)
    forcing = radar_tten.RadarTtenForcing([cp.asarray(slot)], [15.0])
    assert forcing.mp_tend_lim == radar_tten.HRRR_MP_TEND_LIM == 0.07

    # The forced call runs its scheme under HRRR's companion clamp, so the
    # unforced reference runs under the same clamp.
    microphysics.apply(plain, _with_lim(cfg, 0.07), cfg.dt)
    radar_tten.attach(forced, forcing, cfg)
    microphysics.apply(forced, cfg, cfg.dt)
    assert radar_tten.detach(forced) is forcing
    assert getattr(forced, radar_tten.STATE_ATTRIBUTE, None) is None

    thp_plain = cp.asnumpy(plain.thp)
    thp_forced = cp.asnumpy(forced.thp)
    mp_increment = thp_plain - thp0
    assert np.abs(mp_increment).max() > 1e-3          # the scheme heats

    covered = (slot >= -1.0) & (slot <= 1.0)
    covered[-1] = False                                # the flag level
    ring = np.zeros(thp0.shape[1:], bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    covered[:, ring] = False
    want = np.where(covered,
                    thp0 + slot * np.float32(cfg.dt), thp_plain)
    np.testing.assert_array_equal(thp_forced.view(np.int32),
                                  want.astype(np.float32).view(np.int32))
    # h_diabatic is the microphysics rate everywhere (:6014)
    np.testing.assert_array_equal(cp.asnumpy(forced.h_diabatic),
                                  cp.asnumpy(plain.h_diabatic))
    # the specified-zone ring is not touched by either
    np.testing.assert_array_equal(thp_forced[:, ring], thp0[:, ring])
    # every class was exercised
    interior = ~ring[None].repeat(thp0.shape[0], 0)
    interior[-1] = False
    assert np.count_nonzero(covered & (slot == 0.0)) > 0
    assert np.count_nonzero(covered & (slot != 0.0)) > 0
    assert np.count_nonzero(interior & (slot == -20.0)) > 0
    assert np.count_nonzero(interior & (np.abs(slot) == np.float32(1.0001))) > 0
    assert forcing.calls_by_slot == [1]
    assert forcing.elapsed_seconds == cfg.dt


@pytest.mark.parametrize("mp", [1, 8])
def test_the_forced_scheme_runs_under_the_companion_clamp(mp):
    """HRRR pairs mp_tend_radar = 1 with mp_tend_lim = 0.07
    (parm/conus/hrrr_wrfpre.nl:108-109), and WRF clamps the microphysics
    increment before the radar select (:5968-5969): uncovered points and
    h_diabatic carry the clamped increment, covered points the tendency.

    A clamp small enough to bind everywhere the scheme heats makes the
    rule visible: the forced call (case clamp 10 K/s, forcing clamp L)
    equals, off the covered points, an unforced call whose case clamp is
    L, bit for bit, and differs from one whose case clamp is 10.  mp 8 is
    classic Thompson, which finishes in its own fused kernel.
    """
    import cupy as cp
    from gpuwm.core import microphysics
    from gpuwm.da import radar_tten

    lim = 1.0e-5
    plain_10, cfg = _moist_state(mp=mp, specified=True)
    plain_lim, _ = _moist_state(mp=mp, specified=True)
    forced, _ = _moist_state(mp=mp, specified=True)
    thp0 = cp.asnumpy(forced.thp).copy()
    slot = _mixed_slot(thp0.shape)
    forcing = radar_tten.RadarTtenForcing([cp.asarray(slot)], [15.0],
                                          mp_tend_lim=lim)

    microphysics.apply(plain_10, cfg, cfg.dt)
    microphysics.apply(plain_lim, _with_lim(cfg, lim), cfg.dt)
    radar_tten.attach(forced, forcing, cfg)
    microphysics.apply(forced, cfg, cfg.dt)
    radar_tten.detach(forced)

    covered = (slot >= -1.0) & (slot <= 1.0)
    covered[-1] = False
    ring = np.zeros(thp0.shape[1:], bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    covered[:, ring] = False
    thp_lim = cp.asnumpy(plain_lim.thp)
    want = np.where(covered, thp0 + slot * np.float32(cfg.dt), thp_lim)
    np.testing.assert_array_equal(
        cp.asnumpy(forced.thp).view(np.int32),
        want.astype(np.float32).view(np.int32))
    np.testing.assert_array_equal(cp.asnumpy(forced.h_diabatic),
                                  cp.asnumpy(plain_lim.h_diabatic))
    # the clamp binds: without it the uncovered points heat more
    uncovered = ~covered
    uncovered[:, ring] = False
    assert np.abs(cp.asnumpy(plain_10.thp) - thp_lim)[uncovered].max() \
        > 10 * lim * cfg.dt
    # |h_diabatic| = |clamped increment| / dt, one rounding above lim
    h_max = np.abs(cp.asnumpy(forced.h_diabatic)).max()
    assert h_max <= np.float32(lim) * np.float32(1.000001)
    record = forcing.receipt()
    assert record["mp_tend_lim_k_per_s"] == lim
    assert record["case_mp_tend_lim_k_per_s"] == cfg.mp_tend_lim == 10.0
    assert record["mp_tend_lim_source"] == "the forcing's, set by its caller"


def test_a_forcing_may_keep_the_case_clamp_and_refuses_a_bad_one():
    import cupy as cp
    from gpuwm.core import microphysics
    from gpuwm.da import radar_tten

    plain, cfg = _moist_state(specified=True)
    forced, _ = _moist_state(specified=True)
    shape = forced.thp.shape
    forcing = radar_tten.RadarTtenForcing(
        [cp.full(shape, -20.0, dtype=cp.float32)], [15.0], mp_tend_lim=None)
    assert forcing.scheme_config(cfg) is cfg
    microphysics.apply(plain, cfg, cfg.dt)
    radar_tten.attach(forced, forcing, cfg)
    microphysics.apply(forced, cfg, cfg.dt)
    radar_tten.detach(forced)
    np.testing.assert_array_equal(cp.asnumpy(forced.thp),
                                  cp.asnumpy(plain.thp))
    record = forcing.receipt()
    assert record["mp_tend_lim_k_per_s"] == cfg.mp_tend_lim
    assert record["mp_tend_lim_source"] == "the case's"

    hrrr = radar_tten.RadarTtenForcing(
        [cp.full(shape, -20.0, dtype=cp.float32)], [15.0])
    first = hrrr.scheme_config(cfg)
    assert first.mp_tend_lim == 0.07 and first is hrrr.scheme_config(cfg)
    assert hrrr.scheme_config(first) is first
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(radar_tten.RadarTtenError, match="mp_tend_lim"):
            radar_tten.RadarTtenForcing(
                [cp.zeros(shape, dtype=cp.float32)], [15.0],
                mp_tend_lim=bad)


@pytest.mark.parametrize("seconds, slot", [
    (0.0, 1), (900.0, 1), (900.6, 2), (1800.0, 2), (2700.6, 4),
    (3660.0, 4)])
def test_the_slot_read_on_the_card_follows_the_clock(seconds, slot):
    """0, 15.0, 15.01, 30, 45.01 and 61 minutes read slots 1, 1, 2, 2, 4
    and 4."""
    import cupy as cp
    from gpuwm.core import microphysics
    from gpuwm.da import radar_tten

    state, cfg = _moist_state()
    shape = state.thp.shape
    values = (0.001, 0.002, 0.003, 0.004)
    slots = [cp.full(shape, v, dtype=cp.float32) for v in values]
    forcing = radar_tten.RadarTtenForcing(slots, (15.0, 30.0, 45.0, 60.0))
    forcing.elapsed_seconds = seconds
    before = cp.asnumpy(state.thp).copy()
    radar_tten.attach(state, forcing, cfg)
    microphysics.apply(state, cfg, cfg.dt)
    radar_tten.detach(state)
    change = (cp.asnumpy(state.thp) - before)[1:-1, 1:-1, 1:-1]
    read = [k + 1 for k, v in enumerate(values)
            if np.allclose(change, np.float32(v) * np.float32(cfg.dt),
                           rtol=0, atol=2e-5)]
    assert read == [slot]
    assert forcing.calls_by_slot[slot - 1] == 1
    assert forcing.elapsed_seconds == seconds + cfg.dt


def test_no_mp_heating_skips_the_radar_lines_as_wrf_does():
    import cupy as cp
    from gpuwm.core import microphysics
    from gpuwm.da import radar_tten

    state, cfg = _moist_state()
    cfg1 = RunConfig(**{**cfg.__dict__, "no_mp_heating": 1})
    forcing = radar_tten.RadarTtenForcing(
        [cp.full(state.thp.shape, 0.005, dtype=cp.float32)], [15.0])
    before = cp.asnumpy(state.thp).copy()
    radar_tten.attach(state, forcing, cfg1)
    microphysics.apply(state, cfg1, cfg1.dt)
    radar_tten.detach(state)
    np.testing.assert_array_equal(cp.asnumpy(state.thp), before)
    assert forcing.calls_skipped_no_mp_heating == 1
    assert forcing.calls_by_slot == [0]


def test_the_forcing_is_read_inside_a_real_model_step():
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.da import radar_tten

    state, cfg = _moist_state()
    forcing = radar_tten.RadarTtenForcing(
        [cp.zeros(state.thp.shape, dtype=cp.float32)], [15.0])
    radar_tten.attach(state, forcing, cfg)
    step(state, cfg)
    step(state, cfg)
    radar_tten.detach(state)
    assert forcing.calls_by_slot == [2]
    assert forcing.elapsed_seconds == 2 * cfg.dt


def test_states_the_forcing_cannot_be_read_on_are_refused():
    import cupy as cp
    from gpuwm.core import microphysics
    from gpuwm.da import radar_tten

    state, cfg = _moist_state()
    shape = state.thp.shape
    forcing = radar_tten.RadarTtenForcing(
        [cp.zeros(shape, dtype=cp.float32)], [15.0])

    state._streamed_domain = object()
    with pytest.raises(radar_tten.RadarTtenError, match="tile by tile"):
        radar_tten.attach(state, forcing, cfg)
    del state._streamed_domain

    batched = radar_tten.RadarTtenForcing(
        [cp.zeros((shape[0], 2 * shape[1], shape[2]), dtype=cp.float32)],
        [15.0])
    with pytest.raises(radar_tten.RadarTtenError, match="batched-ensemble"):
        radar_tten.attach(state, batched, cfg)

    class _Batched:
        """Stands in for gpuwm.ensemble.batch_state.BatchedDomainState."""
        thp = state.thp

        def member_view(self, name, member):
            raise AssertionError("never read")

    with pytest.raises(radar_tten.RadarTtenError, match="batched-ensemble"):
        radar_tten.attach(_Batched(), forcing, cfg)

    cfg0 = RunConfig(**{**cfg.__dict__, "mp_physics": 0})
    with pytest.raises(radar_tten.RadarTtenError, match="mp_physics = 0"):
        radar_tten.attach(state, forcing, cfg0)

    radar_tten.attach(state, forcing, cfg)
    with pytest.raises(radar_tten.RadarTtenError, match="already"):
        radar_tten.attach(state, forcing, cfg)
    # a state that starts streaming after the forcing was attached
    state._streamed_domain = object()
    with pytest.raises(radar_tten.RadarTtenError, match="tile by tile"):
        microphysics.apply(state, cfg, cfg.dt)
    del state._streamed_domain
    radar_tten.detach(state)

    with pytest.raises(radar_tten.RadarTtenError, match="before"):
        forcing.after_microphysics(state, cfg, cfg.dt)
    with pytest.raises(radar_tten.RadarTtenError, match="increase"):
        radar_tten.RadarTtenForcing(
            [cp.zeros(shape, dtype=cp.float32)] * 2, [30.0, 15.0])


def test_noaa_file_fields_attach_unchanged():
    """RAD_TTEN_DFI_1..4 and TTEN_TIMES of a start file, as host arrays."""
    import cupy as cp
    from gpuwm.da import radar_tten

    state, cfg = _moist_state()
    shape = state.thp.shape
    fields = [np.full(shape, -20.0, np.float32) for _ in range(4)]
    for field in fields:
        field[-1] = -10.0
    forcing = radar_tten.RadarTtenForcing.from_fields(
        fields, np.asarray([15.0, 30.0, 45.0, 60.0], np.float32))
    for slot, field in zip(forcing.slots, fields):
        np.testing.assert_array_equal(cp.asnumpy(slot), field)
    receipt = radar_tten.tendency_receipt(forcing.slots[0])
    assert receipt["points_no_coverage"] == receipt["points_below_top"]
    assert receipt["flag_columns"]["no_information"] == shape[1] * shape[2]
