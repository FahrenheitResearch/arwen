"""Resuming a forecast whose nest moved.

Three things have to be true for this to work at all, and each is cheap
to state and was broken or absent before:

1. the checkpoint has to record WHERE each domain was, because a nest
   that moved is not where its config puts it and every setup array it
   owns belongs to the placement it was at;
2. the checkpoint has to say it crossed a relocation -- that block was
   computed and then dropped on the floor, so every checkpoint written
   after a move claimed ``relocation: null`` and the restore-side
   warning that reads it could never fire;
3. the move chain has to be replayable, so a resume RECONSTRUCTS the
   identity the checkpoint was written under instead of bypassing the
   gate that binds it.

The end-to-end run lives in RESTART-ACROSS-MOVE.md; these are the units
that make it possible.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwm.core.nest_relocation import (RESTART_ACROSS_MOVE_POSTURE,
                                        mark_fingerprint_across_move)
from gpuwm.io.restart import _node_placement


def _node(parent, **cfg):
    return SimpleNamespace(parent=parent, cfg=SimpleNamespace(**cfg))


# ---------------------------------------------------------------------------
# 1. the placement a checkpoint records
# ---------------------------------------------------------------------------

def test_a_nest_records_where_it_actually_was():
    node = _node(object(), i_parent_start=47, j_parent_start=46,
                 parent_grid_ratio=7)
    assert _node_placement(node) == {
        "i_parent_start": 47, "j_parent_start": 46, "parent_grid_ratio": 7}


def test_the_root_has_no_placement_of_its_own():
    assert _node_placement(_node(None, i_parent_start=1)) is None


def test_a_hand_assembled_node_without_placement_reads_as_none():
    """A header field must never be the reason a checkpoint cannot be
    written.  The idealized cases and a good many tests build nodes out
    of a bare SimpleNamespace; recording the placement unconditionally
    broke four of them on AttributeError."""
    assert _node_placement(_node(object())) is None
    assert _node_placement(_node(object(), i_parent_start=3)) is None


@pytest.mark.parametrize("bad", ["x", None, object()])
def test_an_unusable_placement_value_reads_as_none_rather_than_raising(bad):
    node = _node(object(), i_parent_start=bad, j_parent_start=1,
                 parent_grid_ratio=3)
    assert _node_placement(node) is None


# ---------------------------------------------------------------------------
# 2. replaying the chain reconstructs the identity
# ---------------------------------------------------------------------------

def test_replaying_the_move_chain_reproduces_the_checkpoint_identity():
    """The property the resume rests on.

    Every executed move chains its record into the live fingerprint,
    one-way, so no fresh build ever computes a moved tree's value -- that
    is what makes a checkpoint refuse to resume BY CONSTRUCTION.  Replaying
    the same records from the same base reproduces it exactly.
    """
    base = "0" * 64
    records = [f"{n:064x}" for n in range(1, 6)]
    live = base
    for record in records:                      # the run, moving
        live = mark_fingerprint_across_move(live, record)
    replayed = base
    for record in records:                      # the resume, replaying
        replayed = mark_fingerprint_across_move(replayed, record)
    assert replayed == live


def test_a_different_base_does_not_reconstruct_it():
    """The gate keeps its meaning: replaying is not a bypass.  A resume
    of a DIFFERENT configuration starts the chain somewhere else and
    still mismatches, which is the whole point of binding it."""
    records = [f"{n:064x}" for n in range(1, 4)]
    def chain(base):
        for record in records:
            base = mark_fingerprint_across_move(base, record)
        return base
    assert chain("0" * 64) != chain("1" + "0" * 63)


def test_the_order_of_the_chain_binds():
    a, b = f"{1:064x}", f"{2:064x}"
    forward = mark_fingerprint_across_move(
        mark_fingerprint_across_move("0" * 64, a), b)
    reverse = mark_fingerprint_across_move(
        mark_fingerprint_across_move("0" * 64, b), a)
    assert forward != reverse


# ---------------------------------------------------------------------------
# 3. the posture is a fixed string, quoted the same way everywhere
# ---------------------------------------------------------------------------

def test_the_posture_states_what_the_checkpoint_is():
    """It used to say the resume promised NOTHING, and the resume was
    gated behind --allow-restart-across-move on the strength of it.
    Measured against a no-relocation control on the same tree, a resume
    across four moves differs from the unbroken run in exactly what a
    static resume differs in and nothing else -- so the posture states
    what the checkpoint IS, and the flag is gone.
    """
    assert "resumes only into the run that wrote it" in \
        RESTART_ACROSS_MOVE_POSTURE
    assert "bit for bit" in RESTART_ACROSS_MOVE_POSTURE
    # The claim it must NOT make any more.
    assert "promises nothing" not in RESTART_ACROSS_MOVE_POSTURE
    assert "TOLERATED_EXPERIMENT" not in RESTART_ACROSS_MOVE_POSTURE


def test_the_resume_is_not_behind_a_flag():
    """The whole point: a moved checkpoint resumes.  The chain replay is
    a RECONSTRUCTION of the identity, not a bypass of it (the test above
    this section pins that a different base still mismatches), so there
    is nothing left for an opt-in to protect."""
    import inspect

    from gpuwm.prepared_domain_tree_forecast import (build_parser,
                                                     run_prepared_tree)

    assert "allow_restart_across_move" not in \
        inspect.signature(run_prepared_tree).parameters
    flags = {a for action in build_parser()._actions
             for a in action.option_strings}
    assert "--allow-restart-across-move" not in flags


# ---------------------------------------------------------------------------
# 4. continuity: the hysteresis and the counters a resume has to land on
# ---------------------------------------------------------------------------
# Placement gets the tree's GEOMETRY back.  These are the rest of it, and
# the cooldown half is not bookkeeping: a resumed run whose anchor starts
# cold is free to move on a beat the unbroken run was still suppressing,
# which is a different forecast produced by a scalar.

class _Runner:
    """The two methods the checkpoint and the restore actually call."""

    continuity_state = None          # bound below from the real class
    restore_continuity = None

    def __init__(self, provider, moves_executed=0):
        self.provider = provider
        self.moves_executed = moves_executed


def _runner(provider, moves=0):
    from gpuwm.core.relocation_runner import RelocationRunner
    runner = _Runner(provider, moves)
    runner.continuity_state = RelocationRunner.continuity_state.__get__(runner)
    runner.restore_continuity = (
        RelocationRunner.restore_continuity.__get__(runner))
    return runner


def test_continuity_captures_the_cooldown_anchor_and_the_count():
    runner = _runner(SimpleNamespace(_last_move_t=1440.0), moves=4)
    assert runner.continuity_state() == {
        "moves_executed": 4, "last_move_t": 1440.0}


def test_a_provider_that_has_never_moved_reports_no_anchor():
    runner = _runner(SimpleNamespace(_last_move_t=None), moves=0)
    assert runner.continuity_state()["last_move_t"] is None


def test_restoring_continuity_lands_the_anchor_and_the_count():
    provider = SimpleNamespace(_last_move_t=None)
    runner = _runner(provider)
    applied = runner.restore_continuity(
        {"moves_executed": 4, "last_move_t": 1440.0})
    assert provider._last_move_t == 1440.0
    assert runner.moves_executed == 4
    assert applied == {"moves_executed": 4, "last_move_t": 1440.0}


def test_a_provider_without_a_cooldown_says_so_rather_than_failing():
    """A scripted [[relocation.move]] itinerary has no hysteresis of its
    own.  Restoring one must be a recorded no-op, not an AttributeError
    in the middle of a resume."""
    runner = _runner(SimpleNamespace())
    applied = runner.restore_continuity(
        {"moves_executed": 2, "last_move_t": 900.0})
    assert applied["moves_executed"] == 2
    assert "last_move_t_skipped" in applied


def test_an_empty_continuity_block_sets_nothing():
    runner = _runner(SimpleNamespace(_last_move_t=None), moves=7)
    assert runner.restore_continuity({}) == {}
    assert runner.moves_executed == 7


# ---------------------------------------------------------------------------
# The acoustic Omega carrier: carried by a checkpoint, absent from the
# state identity.  Both halves matter and they pull in opposite
# directions, which is why they are pinned together.
# ---------------------------------------------------------------------------

def test_ww_pp_is_carried_by_a_checkpoint_but_not_by_the_state_identity():
    """``ww_pp`` has to survive a restart AND stay out of the identity.

    It is the one acoustic perturbation field ``small_step_init`` does
    not seed, so it carries into every acoustic loop and reaches the
    scalars through WRF ``sumflux`` at the specified lateral boundary --
    a resume that re-zeroed it integrated a measurably different
    forecast.  So a checkpoint must hold it.

    It is also a VIEW INTO THE SHARED DYCORE WORKSPACE, which a child
    rebuild legitimately reuses.  ``STATE_SERIALIZED_ATTRS`` is what
    ``live_state_sha256`` hashes, and therefore what ``relocate_child``
    compares to assert a parent is never written across a move -- so
    putting it there made that assertion fire on workspace churn.  The
    first attempt at this fix did exactly that and broke the moving nest.
    """
    from gpuwm.io.restart import classify_state_attr
    from gpuwm.state_serialization_contract import (CHECKPOINT_ONLY_STATE,
                                                    STATE_SERIALIZED_ATTRS)

    assert "ww_pp" in CHECKPOINT_ONLY_STATE
    assert classify_state_attr("ww_pp") == "checkpoint_only"
    # The half that keeps the relocation parent-invariance check honest.
    assert "ww_pp" not in STATE_SERIALIZED_ATTRS


def test_the_state_identity_never_sees_a_checkpoint_only_carrier():
    """Whatever joins CHECKPOINT_ONLY_STATE stays out of the identity."""
    from gpuwm.ensemble.state_sha import serialized_state_attrs
    from gpuwm.state_serialization_contract import CHECKPOINT_ONLY_STATE

    assert not (set(CHECKPOINT_ONLY_STATE) & set(serialized_state_attrs()))
