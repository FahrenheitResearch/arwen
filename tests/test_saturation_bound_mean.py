"""The analysis holds each member's vapour under saturation and keeps the ensemble mean.

CPU only.  THE BREAKAGE THIS PREVENTS: the applier's per-member saturation
cap alone cuts the upper tail of the members' analysed vapour wherever the
mean sits near saturation and keeps the lower tail, so every analysis
removes water from every member and none from the never-analysed control:
30 to 44 g m-2 per member per analysis on a storm-scale 1 km child, 1.1 to
1.5 kg m-2 over thirty analyses, and 0-2 km air 0.6 to 1.2 g/kg drier than
the control's.
The ensemble bound (``mean_preserving_saturation_bound``) runs at the
analysis, where the whole ensemble is present: members over their limit end
at it, the vapour goes to members with headroom at that cell, and the
filter's ensemble-mean vapour is kept unless the mean is itself over the
limit.  The applier's cap then has nothing left to remove.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from gpuwm.core import constants as c
from gpuwm.da.hotstart import saturation_mixing_ratio
from gpuwm.ensemble.increments import (SATURATION_BOUND_SCHEMA,
                                       apply_increments,
                                       mean_preserving_saturation_bound,
                                       saturation_limit)

MEMBERS = 4
SHAPE = (3, 2, 2)
PRESSURE = np.array([95000.0, 85000.0, 70000.0])
TEMPERATURE = np.array([298.0, 290.0, 280.0])


def _saturation(t, p):
    return saturation_mixing_ratio(t, p, phase="liquid")


def _ensemble(relative_humidity):
    """R members of one column set at the given liquid relative humidity,
    in the state's own equation of state."""
    p = np.broadcast_to(PRESSURE[:, None, None], SHAPE).astype(np.float64)
    t = np.broadcast_to(TEMPERATURE[:, None, None], SHAPE).astype(np.float64)
    qv = relative_humidity * _saturation(t, p)
    alt = c.RD * t * (1.0 + c.RVOVRD * qv) / p
    stack = lambda a: np.stack([a] * MEMBERS)
    return stack(p), stack(alt), stack(qv), _saturation(t, p)


def _near_saturation_increment(saturation):
    """Members spread about a mean that ends at 99 percent: two members
    over saturation, two under, mean kept below the limit."""
    offsets = np.array([0.10, 0.04, -0.06, -0.12])        # of saturation
    return np.stack([(0.09 + k) * saturation for k in offsets])


def test_members_over_their_limit_end_at_it_and_the_mean_is_kept():
    p, alt, q0, saturation = _ensemble(0.90)
    dq = _near_saturation_increment(saturation)
    bounded, receipt = mean_preserving_saturation_bound(q0, dq, p, alt)
    analysed = q0 + bounded
    limit = saturation_limit(np, p[0], alt[0], q0[0], None)
    assert np.all(analysed <= limit[None] * (1.0 + 1e-12))
    # the filter's ensemble-mean vapour, kept
    assert np.allclose(analysed.mean(axis=0), (q0 + dq).mean(axis=0),
                       rtol=1e-12)
    # the members that were over end exactly at the limit
    assert np.allclose(analysed[0], limit, rtol=1e-12)
    assert np.allclose(analysed[1], limit, rtol=1e-12)
    assert receipt["schema"] == SATURATION_BOUND_SCHEMA
    assert receipt["vapour_removed_kg_kg_sum"] == 0.0
    assert receipt["vapour_returned_kg_kg_sum"] == pytest.approx(
        receipt["per_member_cap_would_remove_kg_kg_sum"], rel=1e-12)
    assert receipt["cells_mean_over_limit"] == 0


def test_the_per_member_cap_alone_dries_the_mean_and_after_the_bound_it_does_not():
    # The defect, measured: each member through the applier's cap on its
    # own loses the tail above saturation and gains nothing below it.
    p, alt, q0, saturation = _ensemble(0.90)
    dq = _near_saturation_increment(saturation)

    def applied(increment):
        out = []
        for slot in range(MEMBERS):
            state = types.SimpleNamespace(p=p[slot].copy(), alt=alt[slot].copy(),
                                          qv=q0[slot].copy())
            apply_increments(state, {"qv": increment[slot].copy()})
            out.append(state.qv)
        return np.stack(out)

    raw_mean = (q0 + dq).mean(axis=0)
    capped_alone = applied(dq)
    assert np.all(capped_alone.mean(axis=0) < raw_mean * (1.0 - 1e-3))
    bounded, _ = mean_preserving_saturation_bound(q0, dq, p, alt)
    capped_after = applied(bounded)
    assert np.allclose(capped_after.mean(axis=0), raw_mean, rtol=1e-12)


def test_a_mean_over_the_limit_takes_every_member_to_its_limit():
    # The supersaturation the cap was installed for: the filter's own mean
    # is above saturation.
    p, alt, q0, saturation = _ensemble(0.90)
    dq = np.stack([(0.25 + k) * saturation for k in (0.05, 0.0, -0.02, -0.03)])
    bounded, receipt = mean_preserving_saturation_bound(q0, dq, p, alt)
    limit = saturation_limit(np, p[0], alt[0], q0[0], None)
    assert np.allclose(q0 + bounded, limit[None], rtol=1e-12)
    removed = float(((q0 + dq) - limit[None]).sum())
    assert receipt["vapour_removed_kg_kg_sum"] == pytest.approx(removed,
                                                                rel=1e-9)
    assert receipt["cells_mean_over_limit"] == int(np.prod(SHAPE))


def test_a_cooling_increment_on_saturated_air_is_bounded_the_same_way():
    # Theta alone: a saturated cell cooled by some members and warmed by
    # others, no vapour increment. The cap removed the cooled members'
    # vapour and gave the warmed ones nothing.
    p, alt, q0, saturation = _ensemble(1.0)
    dth = np.stack([np.full(SHAPE, k) for k in (-0.6, -0.2, 0.3, 0.5)])
    dq = np.zeros_like(q0)
    bounded, receipt = mean_preserving_saturation_bound(q0, dq, p, alt, dth)
    analysed = q0 + bounded
    for slot in range(MEMBERS):
        limit = saturation_limit(np, p[slot], alt[slot], q0[slot], dth[slot])
        assert np.all(analysed[slot] <= limit * (1.0 + 1e-12))
    assert np.allclose(analysed.mean(axis=0), q0.mean(axis=0), rtol=1e-12)
    assert receipt["vapour_removed_kg_kg_sum"] == 0.0


def test_nothing_over_the_limit_comes_back_bit_for_bit():
    p, alt, q0, saturation = _ensemble(0.5)
    dq = np.stack([k * saturation for k in (0.1, -0.1, 0.05, 0.0)]).astype(
        np.float32)
    bounded, receipt = mean_preserving_saturation_bound(q0, dq, p, alt)
    assert bounded is dq or np.array_equal(bounded, dq)
    assert bounded.dtype == np.float32
    assert receipt["member_cells_over_limit"] == 0


def test_one_member_is_refused():
    p, alt, q0, saturation = _ensemble(0.9)
    with pytest.raises(ValueError, match="leading member axis"):
        mean_preserving_saturation_bound(q0[:1], np.zeros_like(q0[:1]),
                                         p[:1], alt[:1])


@pytest.mark.parametrize("policy", ["clip", "none"])
def test_the_analysis_bounds_vapour_and_keeps_the_filter_mean(tmp_path, monkeypatch, policy):
    """End to end through assimilate_radar_grid: a moistening point batch on
    members near saturation. Whatever the positivity policy, the posterior
    mean vapour is the filter's mean and no member is over its limit; the
    receipt says what the per-member cap would have removed."""
    import gpuwm.da.radar_assimilation as radar_module
    from gpuwm.da.letkf import GriddedObs, Localization
    from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                             assimilate_radar_grid)
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    nz, ny, nx = 3, 6, 6
    grid = TargetGrid.from_projection(
        LambertGrid(ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
                    stand_lon=-97.0, dx=3000.0, dy=3000.0, e_we=nx + 1,
                    e_sn=ny + 1),
        z_w=np.linspace(0.0, 3000.0, nz + 1), name="saturation-bound-test")
    members = 6
    rng = np.random.default_rng(20260925)
    p = np.broadcast_to(np.array([95000.0, 90000.0, 85000.0])[:, None, None],
                        (nz, ny, nx))
    t = np.broadcast_to(np.array([297.0, 294.0, 291.0])[:, None, None],
                        (nz, ny, nx))
    qs = _saturation(t, p)
    paths = {}
    for index in range(members):
        rh = 0.97 + 0.02 * rng.standard_normal((nz, ny, nx))
        qv = np.clip(rh, 0.9, 0.999) * qs
        alt = c.RD * t * (1.0 + c.RVOVRD * qv) / p
        member_dir = tmp_path / f"member_{index:03d}"
        member_dir.mkdir()
        path = member_dir / "gpuwmrst_d01_000600.npz"
        np.savez(path, **{f"state/{k}": np.asarray(v, np.float32) for k, v in
                          {"qv": qv, "p": p, "alt": alt,
                           "thp": np.zeros((nz, ny, nx))}.items()})
        paths[index] = path
    # The member vapour, read back as the analysis will read it.
    prior = np.stack([np.load(paths[i])["state/qv"].astype(np.float64)
                      for i in range(members)])
    mask = np.zeros((nz, ny, nx), bool)
    mask[0, ny // 2, nx // 2] = True
    simulated = np.zeros((members, nz, ny, nx))
    simulated[:, 0, ny // 2, nx // 2] = prior[:, 0, ny // 2, nx // 2]
    batch = GriddedObs(
        name="point:vapour", values=np.full((nz, ny, nx), 1.2 * qs[0, 0, 0]),
        errors=np.full((nz, ny, nx), 2.0e-4), simulated=simulated, mask=mask,
        localization=Localization(horizontal_m=9000.0, vertical_m=3000.0))
    raw = {}
    real_bound = radar_module._saturation_bound

    def recording_bound(prior_view, increments, states, indices):
        raw["qv"] = np.array(increments["qv"], np.float64)
        return real_bound(prior_view, increments, states, indices)

    monkeypatch.setattr(radar_module, "_saturation_bound", recording_bound)

    cfg = RadarAssimilationConfig(
        solve_device="host", velocity=False, reflectivity=False,
        localization=Localization(horizontal_m=9000.0, vertical_m=3000.0),
        rtps_alpha=0.0, analysis_fields=("qv",),
        positivity_policy=policy)
    increments, provenance = assimilate_radar_grid(
        paths, None, grid, cfg, extra_obs=[batch],
        extra_obs_provenance=[{"source": "fixture"}])
    receipt = provenance["saturation_bound"]
    assert receipt["evaluated"] is True
    assert receipt["member_cells_over_limit"] > 0
    analysed = np.stack([prior[i] + np.asarray(increments[i]["qv"], np.float64)
                         for i in range(members)])
    alt0 = np.stack([np.load(paths[i])["state/alt"].astype(np.float64)
                     for i in range(members)])
    p0 = np.stack([np.load(paths[i])["state/p"].astype(np.float64)
                   for i in range(members)])
    for i in range(members):
        limit = saturation_limit(np, p0[i], alt0[i], prior[i], None)
        assert np.all(analysed[i] <= limit * (1.0 + 1e-6))
    over_mean = receipt["cells_mean_over_limit"]
    kept = np.isclose(analysed.mean(axis=0), (prior + raw["qv"]).mean(axis=0),
                      rtol=1e-6, atol=0.0)
    assert int(np.count_nonzero(~kept)) <= over_mean


def test_the_applier_finds_nothing_left_to_cap_after_the_bound():
    """The two bounds read one limit: a member the analysis left at its limit
    is a member the applier's cap leaves alone, so the cap is the last guard
    on an analysed ensemble, not a sink."""
    p, alt, q0, saturation = _ensemble(0.90)
    dq = _near_saturation_increment(saturation)
    bounded, _ = mean_preserving_saturation_bound(q0, dq, p, alt)
    for slot in range(MEMBERS):
        state = types.SimpleNamespace(p=p[slot].copy(), alt=alt[slot].copy(),
                                      qv=q0[slot].copy())
        receipt = apply_increments(state, {"qv": bounded[slot].copy()})
        # at most rounding at the cells that end exactly at the limit
        assert receipt["saturation"]["vapour_removed_kg_kg_sum"] <= 1e-15


def test_the_block_by_block_bound_is_the_whole_grid_rule(monkeypatch):
    """Worked level block by level block on the moved cells, the bound
    equals the rule evaluated on the whole grid at once, and a cell no
    increment moved (a supersaturated one included) is never written."""
    import gpuwm.ensemble.increments as increments

    rng = np.random.default_rng(12)
    members, shape = 5, (7, 6, 5)
    p = np.broadcast_to(np.linspace(97000.0, 60000.0, shape[0])[:, None, None],
                        shape)
    t = np.broadcast_to(np.linspace(300.0, 270.0, shape[0])[:, None, None],
                        shape)
    rh = rng.uniform(0.7, 1.08, (members,) + shape)
    q0 = rh * _saturation(t, p)
    alt = c.RD * t * (1.0 + c.RVOVRD * q0) / p
    p = np.broadcast_to(p, q0.shape)
    dq = rng.normal(0.0, 0.08, q0.shape) * _saturation(t, p)
    dth = rng.normal(0.0, 0.5, q0.shape)
    still = rng.random(shape) < 0.3
    dq[:, still] = 0.0
    dth[:, still] = 0.0

    limit = saturation_limit(np, p, alt, q0, dth)
    room = limit - (q0 + dq)
    hit = (room < 0).any(axis=0) & ~still
    mean = room[:, hit].mean(axis=0)
    held = np.maximum(room[:, hit], 0.0)
    held_mean = held.mean(axis=0)
    scale = np.minimum(np.where(mean > 0, mean / np.where(
        held_mean > 0, held_mean, 1.0), 0.0), 1.0)
    expected = dq.copy()
    expected[:, hit] = limit[:, hit] - scale * held - q0[:, hit]

    # whole levels together, one level, two rows, one row
    for block in (1 << 18, 150, 50, 25, 1):
        monkeypatch.setattr(increments, "SATURATION_CAP_BLOCK_CELLS", block)
        bounded, receipt = mean_preserving_saturation_bound(q0, dq, p, alt,
                                                            dth)
        np.testing.assert_allclose(bounded, expected, rtol=0, atol=1e-15)
        assert np.all(bounded[:, still] == 0.0)
        assert receipt["cells_touched"] == int(hit.sum())
        assert receipt["cells_touched"] > 0
        # the counts are of moved cells only, and the receipt says so
        assert receipt["member_cells_over_limit"] == int(
            np.count_nonzero((room < 0)[:, ~still]))
        assert "never written or counted" in receipt["rule"]
