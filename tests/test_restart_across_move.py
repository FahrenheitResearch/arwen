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
# 2b. replaying the chain from the checkpoint's own base is not a bypass
# ---------------------------------------------------------------------------
#
# The widening it exists for: the chain is anchored to the base the WRITING
# run computed, so a build that widened an exemption computes a different
# base and every relocation checkpoint on disk becomes unresumable, blaming
# the move history rather than the base.  Replaying from the header's own
# stored components recovers it.
#
# The hole that was in the widening: the replayed value IS the stored
# fingerprint, so adopting it on the strength of the replay alone handed the
# restart gate the header to compare against itself -- and that gate is the
# only place a tree's preparation receipt, cache content, execution plan and
# runtime source identity are ever compared.  A checkpoint from a DIFFERENT
# prepared tree would have resumed silently, its chain being self-consistent
# too.  These pin both directions.

def _chain_case(stored_components, live_components, records=("a" * 64,)):
    """A header/model pair around one stored base and one live one."""
    import hashlib
    from gpuwm.prepared_domain_tree_forecast import (
        _canonical, _strict_json, fingerprint_across_stored_chain)

    def digest(components):
        return hashlib.sha256(
            _canonical(_strict_json(components)).encode("utf-8")).hexdigest()

    stored_fp = digest(stored_components)
    for record in records:
        stored_fp = mark_fingerprint_across_move(stored_fp, record)
    header = {
        "experiment_fingerprint": stored_fp,
        "experiment_fingerprint_components": {
            **stored_components, "relocation": {"records": list(records)}},
        "relocation": {"moves": len(records),
                       "record_sha256": list(records)},
    }

    class _Model:
        pass

    model = _Model()
    model._experiment_fingerprint_components = live_components
    resolved = fingerprint_across_stored_chain(
        header, digest(live_components), model)
    return header, resolved


def test_a_widened_exemption_still_resumes_its_own_checkpoints():
    """The case the replay exists for: same run, moved base."""
    stored = {"experiment_identity": {"domains": [
        {"grid_id": 1, "run": {"use_adaptive_time_step": True,
                               "nz": 49, "target_cfl": 1.2}}]},
        "preparation_receipt_sha256": "r1"}
    # the live build no longer binds target_cfl under an adaptive clock
    live = {"experiment_identity": {"domains": [
        {"grid_id": 1, "run": {"use_adaptive_time_step": True,
                               "nz": 49}}]},
        "preparation_receipt_sha256": "r1"}
    header, resolved = _chain_case(stored, live)
    assert resolved == header["experiment_fingerprint"]


def test_a_different_prepared_tree_is_not_adopted():
    """Self-consistency is not identity.

    The foreign checkpoint's chain replays perfectly from its own stored
    base -- that is what made the hole invisible.  What it cannot do is
    agree with the live components once both are normalised.
    """
    stored = {"experiment_identity": {"domains": [
        {"grid_id": 1, "run": {"use_adaptive_time_step": True, "nz": 49}}]},
        "preparation_receipt_sha256": "SOMEONE ELSE'S TREE"}
    live = {"experiment_identity": {"domains": [
        {"grid_id": 1, "run": {"use_adaptive_time_step": True, "nz": 49}}]},
        "preparation_receipt_sha256": "THIS TREE"}
    header, resolved = _chain_case(stored, live)
    assert resolved != header["experiment_fingerprint"], (
        "a checkpoint from a different prepared tree was adopted")


def test_a_truncated_chain_is_not_adopted():
    """The records fold one-way, so a short chain reaches a different value."""
    import hashlib
    from gpuwm.prepared_domain_tree_forecast import (
        _canonical, _strict_json, fingerprint_across_stored_chain)

    components = {"experiment_identity": {"domains": []},
                  "preparation_receipt_sha256": "r1"}
    base = hashlib.sha256(
        _canonical(_strict_json(components)).encode("utf-8")).hexdigest()
    records = [f"{n:064x}" for n in range(1, 4)]
    full = base
    for record in records:
        full = mark_fingerprint_across_move(full, record)
    header = {
        "experiment_fingerprint": full,
        "experiment_fingerprint_components": {
            **components, "relocation": {"records": records}},
        # the chain the reader is handed drops the last record
        "relocation": {"moves": 2, "record_sha256": records[:2]},
    }

    class _Model:
        pass

    model = _Model()
    model._experiment_fingerprint_components = components
    resolved = fingerprint_across_stored_chain(header, base, model)
    assert resolved != full, "a truncated chain reconstructed the value"


def test_a_checkpoint_with_no_moves_is_untouched():
    """The ordinary case stays the as-built value, not a replayed one."""
    from gpuwm.prepared_domain_tree_forecast import (
        fingerprint_across_stored_chain)

    class _Model:
        pass

    assert fingerprint_across_stored_chain(
        {"experiment_fingerprint": "x" * 64}, "0" * 64, _Model()) == "0" * 64


def test_the_normalised_comparison_is_only_ever_a_widening():
    """The instrument the adoption leans on, in both directions."""
    from gpuwm.io.restart import _identity_matches_under_current_rules

    class _Model:
        pass

    def check(stored, live):
        model = _Model()
        model._experiment_fingerprint_components = live
        return _identity_matches_under_current_rules(
            {"experiment_fingerprint_components": stored}, model)

    adaptive = {"use_adaptive_time_step": True, "nz": 49, "target_cfl": 1.2}
    live_run = {"use_adaptive_time_step": True, "nz": 49}
    ident = lambda run: {"experiment_identity": {                # noqa: E731
        "domains": [{"grid_id": 1, "run": run}]}}
    assert check(ident(adaptive), ident(live_run)), "policy must normalise out"
    assert not check(ident({**adaptive, "nz": 61}), ident(live_run)), (
        "a field that DOES describe the tree must still refuse")

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

    The acoustic namespace remains outside the physical-state digest for
    checkpoint and relocation identity compatibility. The field now owns
    per-domain storage: sharing its retained boundary column was unsafe,
    independently of its serialization namespace.
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


@pytest.mark.parametrize("historical_base", [False, True])
def test_prepared_restore_keeps_the_complete_named_move_history(historical_base):
    import hashlib
    from gpuwm.prepared_domain_tree_forecast import (
        _canonical, _strict_json, fingerprint_across_stored_chain)
    stored = {"experiment_identity": {"domains": [
        {"grid_id": 1, "run": {"use_adaptive_time_step": True, "nz": 49,
                                "target_cfl": 1.2}}]}, "preparation_receipt_sha256": "same-tree"}
    live = {"experiment_identity": {"domains": [
        {"grid_id": 1, "run": {"use_adaptive_time_step": True, "nz": 49,
                                **({} if historical_base else {"target_cfl": 1.2})}}]},
        "preparation_receipt_sha256": "same-tree"}
    def digest(value):
        return hashlib.sha256(_canonical(_strict_json(value)).encode()).hexdigest()
    records = [f"{n:064x}" for n in range(1, 6)]
    # Repeated restoration replaces, rather than appends to, the carried chain.
    for end in (2, 5):
        expected = digest(stored)
        for record in records[:end]:
            expected = mark_fingerprint_across_move(expected, record)
        header = {"experiment_fingerprint": expected,
                  "experiment_fingerprint_components": {**stored, "relocation": {"records": records[:end]}},
                  "relocation": {"record_sha256": records[:end]}}
        model = SimpleNamespace(_experiment_fingerprint_components={
            **live, **({} if historical_base else
                       {"relocation": {"records": ["placement-rebuild-only"]}})})
        result = fingerprint_across_stored_chain(header, digest(live), model)
        assert result == expected
        assert model._experiment_fingerprint_components == {
            **live, "relocation": {"records": records[:end]}}
        assert header["relocation"]["record_sha256"] == records[:end]
        assert live.get("relocation") is None


@pytest.mark.parametrize("damage", ["foreign", "truncated", "reordered"])
def test_failed_prepared_reconstruction_cannot_replace_named_move_history(damage):
    import hashlib
    from gpuwm.prepared_domain_tree_forecast import (
        _canonical, _strict_json, fingerprint_across_stored_chain)
    original = {"preparation_receipt_sha256": "owned-tree"}
    foreign = {"preparation_receipt_sha256": "foreign-tree"} if damage == "foreign" else original
    digest = lambda value: hashlib.sha256(_canonical(_strict_json(value)).encode()).hexdigest()
    records = ["a" * 64, "b" * 64]
    expected = digest(foreign)
    for record in records:
        expected = mark_fingerprint_across_move(expected, record)
    supplied = records[:1] if damage == "truncated" else list(reversed(records)) if damage == "reordered" else records
    header = {"experiment_fingerprint": expected,
              "experiment_fingerprint_components": {**foreign, "relocation": {"records": records}},
              "relocation": {"record_sha256": supplied}}
    model = SimpleNamespace(_experiment_fingerprint_components=original)
    assert fingerprint_across_stored_chain(header, digest(original), model) != expected
    assert model._experiment_fingerprint_components is original
    assert original == {"preparation_receipt_sha256": "owned-tree"}
