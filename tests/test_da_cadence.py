"""Cadence selection, the tuning it drags along, and the overrun rules.

The measurements these are written against are real: KDMX ran 16 volumes
between 04:02 and 05:30 UTC on 2026-08-05 with gaps of 295-418 s, and the
lane assimilated six of them on a 900 s clock.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpuwm.da.cadence import (CadenceError, EXTRAPOLATION_WARN_RATIO,
                              TUNED_BASELINE_INTERVAL_S, check_overrun,
                              plan_fixed, plan_per_volume, quantize_seconds,
                              scaled_settings)

ANCHOR = datetime(2026, 8, 5, 4, 0, 0, tzinfo=timezone.utc)

#: KDMX volume start times, 2026-08-05, as `rw_nexrad list` reported them.
KDMX_VOLUMES = tuple(
    datetime.fromisoformat(f"2026-08-05T{clock}+00:00") for clock in (
        "04:02:43", "04:07:50", "04:12:57", "04:17:53", "04:24:01",
        "04:28:56", "04:34:03", "04:39:09", "04:44:17", "04:49:38",
        "04:55:41", "05:02:10", "05:08:53", "05:15:36", "05:22:34",
        "05:29:32"))


# --------------------------------------------------------------------------
# the fixed cadence, which has to keep behaving exactly as it does today
# --------------------------------------------------------------------------
def test_fixed_cadence_reproduces_the_shipped_six_cycle_shape():
    plan = plan_fixed(KDMX_VOLUMES, anchor=ANCHOR, interval_seconds=900.0,
                      cycles=6, dt_seconds=15.0)
    assert plan.mode == "fixed"
    assert [cycle.elapsed_seconds for cycle in plan.cycles] == [
        900.0, 1800.0, 2700.0, 3600.0, 4500.0, 5400.0]
    # Every leg the same length is the property the whole arm rests on.
    assert set(plan.intervals) == {900.0}


def test_fixed_cadence_counts_the_volumes_it_throws_away():
    """The measurement that motivates the other mode, not a nice-to-have."""

    plan = plan_fixed(KDMX_VOLUMES, anchor=ANCHOR, interval_seconds=900.0,
                      cycles=6, dt_seconds=15.0)
    # Six analyses inside a window holding sixteen volumes: the ten that
    # were never assimilated are named, so "we discard most of the feed"
    # is a number in a receipt rather than an impression.
    assert len(plan.unused_volumes) == 10
    assert all("reason" in entry for entry in plan.unused_volumes)


def test_fixed_cadence_refuses_a_mark_with_no_volume_near_it():
    sparse = (KDMX_VOLUMES[0], KDMX_VOLUMES[-1])
    with pytest.raises(CadenceError, match="no volume within"):
        plan_fixed(sparse, anchor=ANCHOR, interval_seconds=900.0, cycles=4,
                   dt_seconds=15.0, max_offset_seconds=480.0)


# --------------------------------------------------------------------------
# the per-volume cadence
# --------------------------------------------------------------------------
def test_per_volume_cadence_follows_the_antenna_not_a_constant():
    plan = plan_per_volume(
        KDMX_VOLUMES, anchor=ANCHOR, dt_seconds=15.0,
        window_start=datetime(2026, 8, 5, 4, 15, tzinfo=timezone.utc),
        window_end=datetime(2026, 8, 5, 5, 30, tzinfo=timezone.utc))
    assert plan.mode == "per-volume"
    # Thirteen volumes in the window the fixed arm spends six analyses on.
    assert len(plan.cycles) == 13
    legs = plan.intervals
    # The VCP changed underneath this hour; a single interval cannot be
    # right for both halves and the plan must not pretend otherwise.
    assert min(legs) != max(legs)
    assert min(legs) >= 300.0 and max(legs) <= 1080.0


def test_per_volume_analysis_times_land_on_the_timestep_lattice():
    """A boundary off the lattice is rounded inside the clock, silently."""

    plan = plan_per_volume(
        KDMX_VOLUMES, anchor=ANCHOR, dt_seconds=15.0,
        window_start=datetime(2026, 8, 5, 4, 15, tzinfo=timezone.utc))
    for cycle in plan.cycles:
        assert cycle.elapsed_seconds % 15.0 == 0.0
        # Half a timestep is the most quantization can ever move an
        # analysis, and it is recorded rather than absorbed.
        assert abs(cycle.quantization_shift_seconds) <= 7.5
    assert any(cycle.quantization_shift_seconds != 0.0
               for cycle in plan.cycles)


def test_per_volume_records_every_volume_it_declines():
    plan = plan_per_volume(
        KDMX_VOLUMES, anchor=ANCHOR, dt_seconds=15.0,
        min_interval_seconds=600.0)
    dropped = {entry["volume_time"] for entry in plan.unused_volumes}
    assert dropped, "a 600 s floor over a ~340 s feed must drop volumes"
    # Nothing vanishes: every declined volume carries its own reason.
    assert all(entry["reason"] for entry in plan.unused_volumes)
    kept = len(plan.cycles)
    assert kept + len(dropped) == len(KDMX_VOLUMES)


def test_per_volume_refuses_an_empty_window_rather_than_inventing_a_vcp():
    with pytest.raises(CadenceError, match="nothing to cycle"):
        plan_per_volume(
            KDMX_VOLUMES, anchor=ANCHOR, dt_seconds=15.0,
            window_start=datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc))


def test_no_volume_times_is_a_refusal_not_a_nominal_period():
    with pytest.raises(CadenceError, match="invent data"):
        plan_per_volume([], anchor=ANCHOR, dt_seconds=15.0)


def test_quantize_refuses_a_nonsense_timestep():
    # 04:17:53 is 1073 s past a 04:00 anchor: 71.53 steps of 15 s, so the
    # analysis lands at step 72 and the plan carries the +7 s shift.
    assert quantize_seconds(1073.0, 15.0) == 1080.0
    with pytest.raises(CadenceError):
        quantize_seconds(1073.0, 0.0)


# --------------------------------------------------------------------------
# carrying a 15-minute tuning to another cadence
# --------------------------------------------------------------------------
def test_inflation_scales_so_relaxation_per_unit_time_is_invariant():
    block = scaled_settings(cycle_interval_s=300.0, rtps_alpha=0.9,
                            error_inflation=1.0, horizontal_loc_m=12000.0,
                            vertical_loc_m=3000.0)
    # alpha' = 1 - (1 - alpha) * dt'/dt = 1 - 0.1 * (300/900)
    assert block["applied"]["rtps_alpha"] == pytest.approx(0.9666667, abs=1e-6)
    # A third of the interval means three times as many analyses per unit
    # time, so each one must relax three times less.  The applied value is
    # rounded to six decimals on the way into the receipt -- it becomes a
    # CLI flag and a human reads it -- so the invariant is checked to that
    # precision and not past it.
    assert (1.0 - block["applied"]["rtps_alpha"]) == pytest.approx(
        (1.0 - 0.9) / 3.0, abs=1e-6)


def test_observation_error_inflates_so_information_per_unit_time_holds():
    block = scaled_settings(cycle_interval_s=225.0, rtps_alpha=0.9,
                            error_inflation=1.0, horizontal_loc_m=12000.0,
                            vertical_loc_m=3000.0)
    # Four times the cycles per hour, sigma_o up by sqrt(4) = 2.
    assert block["applied"]["error_inflation"] == pytest.approx(2.0, rel=1e-9)


def test_localization_is_left_alone_and_says_so():
    block = scaled_settings(cycle_interval_s=340.0, rtps_alpha=0.9,
                            error_inflation=1.0, horizontal_loc_m=12000.0,
                            vertical_loc_m=3000.0)
    assert block["applied"]["horizontal_loc_m"] == 12000.0
    assert block["applied"]["vertical_loc_m"] == 3000.0
    # Not scaled *and* flagged: an unscaled knob that nobody is told about
    # is the thing this module exists to prevent.
    assert "horizontal_loc_m" in block["needs_retuning"]
    assert "NOT SCALED" in block["reasoning"]["localization"]


def test_scaling_none_keeps_the_old_constants_and_names_the_consequence():
    block = scaled_settings(cycle_interval_s=340.0, rtps_alpha=0.9,
                            error_inflation=1.0, horizontal_loc_m=12000.0,
                            vertical_loc_m=3000.0, scaling="none")
    assert block["applied"] == block["baseline"]
    # Keeping them is allowed; keeping them quietly is not.
    assert "under-disperse" in block["reasoning"]["rtps_alpha"]
    assert set(block["needs_retuning"]) >= {"rtps_alpha", "error_inflation"}


def test_a_baseline_cadence_scales_to_itself():
    block = scaled_settings(cycle_interval_s=TUNED_BASELINE_INTERVAL_S,
                            rtps_alpha=0.9, error_inflation=1.0,
                            horizontal_loc_m=12000.0, vertical_loc_m=3000.0)
    assert block["applied"]["rtps_alpha"] == pytest.approx(0.9, abs=1e-9)
    assert block["applied"]["error_inflation"] == pytest.approx(1.0, rel=1e-9)


def test_a_far_extrapolation_is_labelled_as_one():
    block = scaled_settings(
        cycle_interval_s=TUNED_BASELINE_INTERVAL_S
        / (EXTRAPOLATION_WARN_RATIO + 1.0),
        rtps_alpha=0.9, error_inflation=1.0, horizontal_loc_m=12000.0,
        vertical_loc_m=3000.0)
    assert "extrapolation_warning" in block


def test_a_cadence_the_argument_cannot_reach_is_refused_not_clamped():
    # Scaling 0.9 from 900 s out to 3 hours drives alpha below zero.
    with pytest.raises(CadenceError, match="outside"):
        scaled_settings(cycle_interval_s=10800.0, rtps_alpha=0.9,
                        error_inflation=1.0, horizontal_loc_m=12000.0,
                        vertical_loc_m=3000.0)


def test_unknown_scaling_mode_is_a_refusal():
    with pytest.raises(CadenceError, match="unknown scaling"):
        scaled_settings(cycle_interval_s=300.0, rtps_alpha=0.9,
                        error_inflation=1.0, horizontal_loc_m=12000.0,
                        vertical_loc_m=3000.0, scaling="whatever")


# --------------------------------------------------------------------------
# observations arriving faster than they can be used
# --------------------------------------------------------------------------
def _fast_plan():
    """A feed at 120 s, faster than the ~45 s cycle only in places."""

    stamps = [ANCHOR + timedelta(seconds=60 * (n + 1)) for n in range(8)]
    return plan_per_volume(stamps, anchor=ANCHOR, dt_seconds=15.0)


def test_a_cadence_the_hardware_can_keep_up_with_is_clear():
    plan = plan_per_volume(
        KDMX_VOLUMES, anchor=ANCHOR, dt_seconds=15.0,
        window_start=datetime(2026, 8, 5, 4, 15, tzinfo=timezone.utc))
    _, record = check_overrun(plan, cycle_cost_seconds=45.0)
    assert record["outcome"] == "clear"
    # 45 s of work per ~410 s cycle is the duty cycle the case is sized on.
    assert record["duty_cycle_at_mean_interval"] < 0.2


def test_refuse_is_the_default_and_it_fails_closed():
    plan = _fast_plan()
    with pytest.raises(CadenceError, match="shorter than the measured"):
        check_overrun(plan, cycle_cost_seconds=90.0)


def test_skip_drops_volumes_and_names_every_one_of_them():
    plan = _fast_plan()
    thinned, record = check_overrun(plan, cycle_cost_seconds=90.0,
                                    policy="skip")
    assert record["outcome"] == "skipped"
    assert record["cycles_dropped"] > 0
    assert len(thinned.cycles) < len(plan.cycles)
    # Holding real time costs volumes; the cost is itemised, never implied.
    skipped = [entry for entry in thinned.unused_volumes
               if entry["reason"].startswith("skipped:")]
    assert len(skipped) == record["cycles_dropped"]
    # A kept cycle absorbs the time of the ones skipped before it, so no
    # forecast leg is silently shortened.
    assert min(cycle.leg_seconds for cycle in thinned.cycles) >= 90.0


def test_queue_keeps_every_volume_and_quantifies_the_lag():
    plan = _fast_plan()
    queued, record = check_overrun(plan, cycle_cost_seconds=90.0,
                                   policy="queue")
    assert record["outcome"] == "queued"
    assert len(queued.cycles) == len(plan.cycles)
    assert record["projected_backlog_seconds"] > 0.0
    assert "without bound" in record["detail"]


def test_an_unknown_overrun_policy_is_a_refusal():
    plan = _fast_plan()
    with pytest.raises(CadenceError, match="unknown overrun policy"):
        check_overrun(plan, cycle_cost_seconds=45.0, policy="drop")


def test_overrun_needs_a_measured_cost_not_a_placeholder():
    plan = _fast_plan()
    with pytest.raises(CadenceError, match="positive one"):
        check_overrun(plan, cycle_cost_seconds=0.0)
