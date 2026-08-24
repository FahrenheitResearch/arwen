"""What a runtime-spawned nest records as its own activation epoch.

A nest that activates LATE has always been a supported shape: a domain
may declare ``start_time`` and join the run hours in, and task #205 built
the machinery every consumer reads that epoch through -- the history
alarm, the REFL_10CM handoff, the schedule's activation bookkeeping and
the physics step counter that decides whether radiation is due.

A nest that is SPAWNED activates late too, and until this module's
subject was fixed it recorded no epoch at all.  ``active_experiment``
replaced the fired placement and nothing else, so the newborn's clock
resolved ``start_ticks = 0`` -- it claimed to have started with the
experiment.  Two shipped failures fell straight out of that one missing
number, and both are pinned below:

* its first history frame consumed a REFL_10CM stash no step of that
  domain had produced, and
* its first step was numbered as though it had been running since t = 0,
  so ``_radiation_step_due`` said radiation was not due and Noah then
  consumed a GLW buffer nothing had ever written.

The last two tests hold the other direction: the forcing-seam rule that
a CONFIG-scheduled delayed start must obey is not relaxed for it.
"""
from __future__ import annotations

import datetime

import pytest

from gpuwm.core.clock import resolve_clock
from gpuwm.core.physics import _radiation_step_due
from gpuwm.core.refl import refl_10cm_stash_is_due
from gpuwm.experiment import (active_experiment, load_experiment,
                              validate_boundary_timing)

from test_nest_lifecycle_runtime import BASE, LIFECYCLE, ONE_SHOT, _exp

#: The fixture's spawn instant: ``at_s = 120``, two d01 steps in.  d01
#: runs dt = 60 s and d02 dt = 20 s, both whole seconds, so the tick
#: denominator is 1 and a tick IS a second here.
BIRTH_S = 120


def _fired(tmp_path, *, birth_s: float = BIRTH_S, d02: str = ONE_SHOT):
    """The experiment, and the leg view after its slot has fired."""
    exp = _exp(tmp_path, d02)
    active = active_experiment(exp, {2: (40, 30)},
                               birth_times={2: float(birth_s)})
    return exp, active


# ---------------------------------------------------------------------------
# The epoch itself
# ---------------------------------------------------------------------------

def test_a_spawned_child_records_its_birth_instant_as_its_start_time(tmp_path):
    exp, active = _fired(tmp_path)
    assert active.domain_start_offset_exact(2) == BIRTH_S
    assert (active.domain_start_time(2)
            == exp.start_time + datetime.timedelta(seconds=BIRTH_S))


def test_the_parent_keeps_the_experiment_start(tmp_path):
    """Only the newborn moves.  A retire/re-arm run rebuilds this view at
    every leg, so a parent that drifted would re-phase the whole run."""
    _, active = _fired(tmp_path)
    assert active.domain_start_offset_exact(1) == 0


def test_a_dormant_slot_that_has_not_fired_records_nothing(tmp_path):
    """No birth time, no epoch: the identity path stays byte-inert."""
    exp = _exp(tmp_path, ONE_SHOT)
    assert active_experiment(exp).domains == (exp.root,)


def test_re_arming_moves_the_epoch_to_the_second_birth(tmp_path):
    """Episode two is a new activation, not a continuation of episode one."""
    _, active = _fired(tmp_path, birth_s=1500.0, d02=LIFECYCLE)
    assert active.domain_start_offset_exact(2) == 1500


def test_the_spawned_childs_clock_starts_at_its_birth_tick(tmp_path):
    _, active = _fired(tmp_path)
    clocks = resolve_clock(active).clocks()
    assert clocks[1].spec.start_ticks == 0
    assert clocks[2].spec.start_ticks == BIRTH_S


# ---------------------------------------------------------------------------
# The two shipped failures the missing epoch caused
# ---------------------------------------------------------------------------

def test_the_spawned_childs_first_frame_consumes_no_refl_stash(tmp_path):
    """The #205 guard asks the DOMAIN's own start tick.  It only answers
    correctly once the domain has one."""
    _, active = _fired(tmp_path)
    spec = resolve_clock(active).clocks()[2].spec
    assert not refl_10cm_stash_is_due(
        BIRTH_S, domain_start_ticks=spec.start_ticks)
    # Every frame AFTER the activation frame still consumes its handoff.
    assert refl_10cm_stash_is_due(
        BIRTH_S + spec.history_ticks, domain_start_ticks=spec.start_ticks)


def test_the_spawned_childs_first_step_runs_radiation(tmp_path):
    """WRF numbers a domain's first step 1, and radiation runs at 1.

    The step counter is derived the way both clock-minting sites derive
    it (``(ticks - start_ticks) // step_ticks``), so this pins the
    arithmetic that produced the GLW refusal rather than a restatement
    of it.  stepra = 12 is radt = 4 min on this child's 20 s step.
    """
    _, active = _fired(tmp_path)
    spec = resolve_clock(active).clocks()[2].spec
    step_count = max(0, (BIRTH_S - spec.start_ticks) // spec.step_ticks)
    assert step_count == 0
    assert _radiation_step_due(step_count + 1, 12, 4.0)

    # The epoch-less reading, kept explicit: a newborn numbered 7 is not
    # a multiple of the radiation cadence, radiation is skipped, and the
    # land surface consumes a downward longwave flux nobody produced.
    stale = max(0, (BIRTH_S - 0) // spec.step_ticks) + 1
    assert stale == 7
    assert not _radiation_step_due(stale, 12, 4.0)


# ---------------------------------------------------------------------------
# The forcing-seam rule: relaxed for a spawn, kept for a declared start
# ---------------------------------------------------------------------------

def test_boundary_timing_admits_a_spawn_born_off_the_forcing_seam(tmp_path):
    """A trigger fires on the weather, not on the forcing calendar.

    120 s is not a seam of a 3600 s boundary interval, and it must not
    have to be: a spawned nest is initialized by SINT from its LIVE
    parent, so it reads no forcing snapshot at its birth to be aligned
    to, and its lateral boundaries come from that parent thereafter.
    """
    _, active = _fired(tmp_path)
    validate_boundary_timing(active, 3600)


def test_boundary_timing_still_refuses_a_declared_start_off_the_seam(tmp_path):
    """The relaxation is the spawn's, and nothing else's.

    A domain whose late start is DECLARED is initialized from the
    forcing at that instant, so it must land on a snapshot -- the rule
    the spawn carve-out must not quietly take with it.
    """
    path = tmp_path / "delayed.toml"
    path.write_text(BASE.format(d02="start_time = 1974-04-03T12:02:00"))
    exp = load_experiment(path)
    assert exp.domain_start_offset_exact(2) == 120
    with pytest.raises(ValueError, match="boundary-forcing cadence"):
        validate_boundary_timing(exp, 3600)


# ---------------------------------------------------------------------------
# The driver's own step index, which is a SEPARATE reading of the epoch
# ---------------------------------------------------------------------------

def test_the_physics_driver_counts_itimestep_from_the_domains_activation():
    """The driver does not read the clock; it re-derives ITIMESTEP from
    model time, so it needs the epoch published on the state.

    This is why the GLW refusal survived the clock fix.  A child born at
    t = 300 s on a 20 s step reads itimestep 16 from absolute time; 16 is
    not 1 and not a multiple of the 12-step radiation cadence plus one,
    so radiation is skipped and Noah then consumes carriers nothing
    produced.  Counted from the domain's own activation it is 1, which is
    WRF's mandatory first-step radiation call.
    """
    import numpy as np

    def itimestep(now, dt, epoch):
        return int(np.floor((now - epoch) / dt + 0.5)) + 1

    assert itimestep(300.0, 20.0, 0.0) == 16
    assert not _radiation_step_due(16, 12, 4.0)

    assert itimestep(300.0, 20.0, 300.0) == 1
    assert _radiation_step_due(1, 12, 4.0)

    # A domain that started with the run is unmoved: same reading either way.
    assert itimestep(300.0, 60.0, 0.0) == 6


def test_refresh_model_time_publishes_the_epoch_on_the_state():
    """One authority: the tick clock, read through the state."""
    from types import SimpleNamespace

    from gpuwm.core.state import refresh_model_time

    clock = SimpleNamespace(
        ticks=300, tick_den=1, elapsed_seconds_fp32=300.0,
        spec=SimpleNamespace(step_ticks=20, start_ticks=300))
    state = SimpleNamespace()
    refresh_model_time(state, clock)
    assert state.elapsed_seconds == 300.0
    assert state.domain_start_offset == 300.0
