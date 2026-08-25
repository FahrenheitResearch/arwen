"""Runtime contracts for the nest lifecycle: output episodes, retirement,
independent followers, and the restart refusal.

The policy half of the lifecycle is covered by ``test_nest_lifecycle.py``.
This file drives the RUNTIME objects the lifecycle series added -- the
writer set's pathname selection, ``RelocationRunnerCollection``,
``SpawnRunner``'s retirement branch and ``_detach_retired_children`` --
because a policy that decides correctly and a runtime that wires it
wrongly are indistinguishable from the policy tests alone.

Two defects this file exists to keep closed, both DEFAULT-PATH
regressions rather than lifecycle bugs:

* a plain one-shot ``spawn`` config (no retire/rearm) must keep writing
  the FLAT ``wrfout_dNN_*`` pathname it has always written.  Episode
  numbering engages on a declared lifecycle table, not on the mere fact
  that a nest was spawned;
* the duplicate-valid-time guard must refuse a duplicate produced WITHIN
  one run and must NOT refuse a re-run into a previous run's output
  directory, which has always overwritten through the atomic replace.

Stand-ins here replace COLLABORATORS (the async writer thread, the
route's child preparer), never the code under test.
"""

from __future__ import annotations

import datetime
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import uh_diag
from gpuwm.core.nest_lifecycle import (admit_restart_with_lifecycle,
                                       declares_lifecycle, output_episode)
from gpuwm.core.relocation_runner import (ManualMoveProvider,
                                          RelocationRunnerCollection,
                                          RelocationRunner)
from gpuwm.experiment import (RelocationConfig, ScheduledRelocationMove,
                              load_experiment)
from gpuwm.io.wrfout import PerDomainWrfoutWriters
from gpuwm.runtime import _detach_retired_children

from test_nest_relocation_staging import (_cpu_tree, _footprint_statics,
                                          _initializer)


# ---------------------------------------------------------------------------
# Config fixtures: one-shot spawn vs a DECLARED lifecycle
# ---------------------------------------------------------------------------

BASE = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 3600.0
restart_interval_s = 0.0

[shared]
nz = 8
ztop = 12000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 100
ny = 80
time_step = 60
dx = 12000.0
history_interval_s = 3600.0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 40
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
e_we = 61
e_sn = 61
history_interval_s = 900.0
{d02}
"""

ONE_SHOT = 'spawn = { trigger = "time", at_s = 120.0 }'
LIFECYCLE = (
    'spawn = { trigger = "time", at_s = 120.0 }\n'
    'retire = { trigger = "uh", threshold = 60.0, sustained_s = 900.0, '
    'min_lifetime_s = 1800.0 }\n'
    'rearm = { max_firings = 3, cooldown_s = 600.0 }')


def _exp(tmp_path, d02):
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(BASE.format(d02=d02)))
    return load_experiment(path)


# ---------------------------------------------------------------------------
# Output episodes gate on a DECLARED lifecycle table
# ---------------------------------------------------------------------------

def test_one_shot_spawn_declares_no_lifecycle(tmp_path):
    exp = _exp(tmp_path, ONE_SHOT)
    dc = exp.domain(2)
    assert dc.spawn is not None
    assert not declares_lifecycle(dc)


def test_retire_or_rearm_declares_a_lifecycle(tmp_path):
    exp = _exp(tmp_path, LIFECYCLE)
    dc = exp.domain(2)
    assert declares_lifecycle(dc)


def test_one_shot_spawn_reports_no_output_episode(tmp_path):
    """The counter still counts; the WRITERS are told zero.

    SpawnRunner.episodes serves the re-arm bound and must keep counting
    from 1 on the first fire.  What must not happen is that count reaching
    the writers for a domain that declared no lifecycle, because the
    writer set turns any episode > 0 into a dNN/episode-NNN/ subdirectory.
    """
    dc = _exp(tmp_path, ONE_SHOT).domain(2)
    assert output_episode(dc, 1) == 0
    assert output_episode(dc, 4) == 0


def test_declared_lifecycle_numbers_episodes_from_one(tmp_path):
    dc = _exp(tmp_path, LIFECYCLE).domain(2)
    assert output_episode(dc, 1) == 1
    assert output_episode(dc, 2) == 2


def test_follow_alone_is_placement_policy_not_episode_policy(tmp_path):
    """[follow] moves a nest; it never creates a second episode."""
    d02 = (ONE_SHOT + '\nfollow = { field = "uh", threshold = 100.0, '
           'fallback_threshold = 35.0, search_margin_cells = 12, '
           'min_shift_cells = 2, max_shift_cells = 10, '
           'cooldown_seconds = 600.0, cadence_seconds = 900.0, '
           'max_move_parent_cells = 8, min_overlap_fraction = 0.70 }')
    dc = _exp(tmp_path, d02).domain(2)
    assert dc.follow is not None
    assert not declares_lifecycle(dc)
    assert output_episode(dc, 1) == 0


def test_shipped_spawn_config_keeps_flat_history_paths():
    """The one shipped config using spawn must not move its output.

    Read through ``build_experiment`` with the companion tables split off,
    the way the loader's own refusal directs a raw-dict caller, so the
    assertion needs no case data on disk.
    """
    import tomllib
    from pathlib import Path

    from gpuwm.experiment import build_experiment

    path = (Path(__file__).resolve().parents[1] / "configs"
            / "moving_nest_20110427_spawn_2km.toml")
    raw = tomllib.loads(path.read_text())
    for companion in ("fetch", "case_data", "static"):
        raw.pop(companion, None)
    exp = build_experiment(raw, str(path))

    spawned = [dc for dc in exp.domains
               if getattr(dc, "spawn", None) is not None]
    assert spawned, "fixture drifted: this config declares a spawn slot"
    for dc in spawned:
        assert not declares_lifecycle(dc)
        assert output_episode(dc, 1) == 0


# ---------------------------------------------------------------------------
# Restart identity: absent lifecycle tables are byte-inert, declared ones bind
# ---------------------------------------------------------------------------

def test_absent_lifecycle_tables_are_outside_the_restart_identity(tmp_path):
    """An undeclared retire/rearm/follow must not reach the payload.

    The same absent-stays-absent convention ``spawn`` and ``perturbation``
    already hold to (gpuwm/core/model.py).  These three fields landed on
    DomainConfig defaulting to ``None``; because the payload is built by
    ``dataclasses.asdict``, a ``None`` that is not popped serializes as
    ``retire: null`` on EVERY domain of EVERY experiment -- including every
    experiment written before the fields existed.  That moves each of their
    fingerprints, and a moved fingerprint is a checkpoint that refuses to
    restore.  Asserted on a plain domain AND on a one-shot spawn, because
    the pop must not be conditional on the domain being lifecycle-free.
    """
    from gpuwm.core.model import restart_identity_payload

    for d02 in ("", ONE_SHOT):
        payload = restart_identity_payload(_exp(tmp_path, d02))
        for domain in payload["domains"]:
            for field in ("retire", "rearm", "follow"):
                assert field not in domain, (
                    f"{field!r} reached the identity of a domain that never "
                    f"declared it (d02={d02!r})")


def test_a_declared_retire_binds_the_identity_and_fingerprint(tmp_path):
    """The control in the other direction: declaring one MUST move it.

    Popping ``None`` would be indistinguishable from dropping the field
    outright without this half.  A retirement policy decides when a child
    stops integrating, so a checkpoint written under one must refuse to
    resume under another.
    """
    from gpuwm.core.model import (experiment_fingerprint,
                                  restart_identity_payload)

    one_shot = _exp(tmp_path, ONE_SHOT)
    lifecycle = _exp(tmp_path, LIFECYCLE)
    assert (restart_identity_payload(one_shot)
            != restart_identity_payload(lifecycle))
    catalog = SimpleNamespace(run_provenance={})
    assert (experiment_fingerprint(one_shot, catalog)
            != experiment_fingerprint(lifecycle, catalog))
    bound = [domain for domain in restart_identity_payload(lifecycle)["domains"]
             if "retire" in domain]
    assert len(bound) == 1 and bound[0]["rearm"] is not None


def test_a_declared_follow_binds_the_identity(tmp_path):
    """``follow`` is a placement policy, and placement is trajectory."""
    from gpuwm.core.model import restart_identity_payload

    d02 = (ONE_SHOT + '\nfollow = { field = "uh", threshold = 100.0, '
           'fallback_threshold = 35.0, search_margin_cells = 12, '
           'min_shift_cells = 2, max_shift_cells = 10, '
           'cooldown_seconds = 600.0, cadence_seconds = 900.0, '
           'max_move_parent_cells = 8, min_overlap_fraction = 0.70 }')
    following = restart_identity_payload(_exp(tmp_path, d02))
    assert (following != restart_identity_payload(_exp(tmp_path, ONE_SHOT)))
    assert [domain for domain in following["domains"]
            if domain.get("follow") is not None]


# ---------------------------------------------------------------------------
# The writer set's pathname selection, driven through the real submit()
# ---------------------------------------------------------------------------

class RecordingWriter:
    """Stand-in for AsyncDomainWrfoutWriter: records, never writes.

    A collaborator, not the code under test.  ``submit`` here is the real
    PerDomainWrfoutWriters.submit; only the async NetCDF thread below it
    is replaced, because the assertion is about the PATHNAME that method
    chooses and the guard it applies before handing the frame over.
    """

    def __init__(self):
        self.paths = []
        self.global_attrs = {}

    def submit(self, path, valid_time, state, **kwargs):
        self.paths.append(path)


def _writer_set(tmp_path, *, episodes):
    """A real PerDomainWrfoutWriters with its collaborators stood in.

    __init__ needs a built model, prepared cases and the Rust cdylib, none
    of which a pathname contract depends on, so the instance is assembled
    directly and the REAL submit() is what the tests call.
    """
    w = object.__new__(PerDomainWrfoutWriters)
    w.output_dir = tmp_path
    w.start_time = datetime.datetime(2011, 4, 27, 18, 0, 0)
    w._metadata_by_grid_id = {}
    w._archived_paths = []
    w._episode_by_grid_id = dict(episodes)
    w._writers = {}
    w._published_paths = set()
    for gid in episodes:
        w._writers[gid] = RecordingWriter()
        w._metadata_by_grid_id[gid] = {}
    return w


def _node(grid_id=2):
    return SimpleNamespace(
        cfg=SimpleNamespace(grid_id=grid_id),
        clock=SimpleNamespace(tick_den=1),
        state=SimpleNamespace())


def test_episode_zero_writes_the_flat_historical_pathname(tmp_path):
    w = _writer_set(tmp_path, episodes={2: 0})
    w.submit(_node(2), 0)
    path = w._writers[2].paths[0]
    assert path.parent == tmp_path
    assert path.name.startswith("wrfout_d02_")


def test_a_declared_episode_writes_under_its_own_subdirectory(tmp_path):
    w = _writer_set(tmp_path, episodes={2: 2})
    w.submit(_node(2), 0)
    path = w._writers[2].paths[0]
    assert path.parent == tmp_path / "d02" / "episode-002"


# ---------------------------------------------------------------------------
# The duplicate-valid-time guard, both directions
# ---------------------------------------------------------------------------

def test_same_run_duplicate_valid_time_refuses(tmp_path):
    """The breakage the guard names: one run, one path, two frames."""
    w = _writer_set(tmp_path, episodes={2: 0})
    w.submit(_node(2), 0)
    written = w._writers[2].paths[0]
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_bytes(b"durable frame from earlier in this run")
    with pytest.raises(RuntimeError, match="THIS run"):
        w.submit(_node(2), 0)


def test_a_previous_runs_frame_is_overwritten_as_before(tmp_path):
    """A re-run into yesterday's output directory must still replace.

    Cross-run overwrite through the atomic replace is long-standing
    behaviour and is NOT what the guard exists to prevent; refusing it
    would strand every repeat run into an existing directory.
    """
    stale = tmp_path / "wrfout_d02_2011-04-27_18_00_00"
    stale.write_bytes(b"a frame from a previous run")
    w = _writer_set(tmp_path, episodes={2: 0})
    w.submit(_node(2), 0)
    assert w._writers[2].paths == [stale]


def test_the_guard_is_per_path_not_per_domain(tmp_path):
    """Two valid times from one domain in one run are not duplicates."""
    w = _writer_set(tmp_path, episodes={2: 0})
    w.submit(_node(2), 0)
    w.submit(_node(2), 3600)
    assert len({p.name for p in w._writers[2].paths}) == 2


# ---------------------------------------------------------------------------
# Independent per-domain followers
# ---------------------------------------------------------------------------

def _second_child(parent, child):
    """A sibling of the CPU tree's child, placed clear of it."""
    from dataclasses import replace as _replace

    dc = _replace(child.cfg, grid_id=3)
    state = SimpleNamespace(**{
        name: np.array(value, copy=True)
        for name, value in vars(child.state).items()})
    return SimpleNamespace(
        cfg=dc, state=state, grid="old-grid-d03", parent=parent,
        coupler=SimpleNamespace(relocate=lambda: {"rolling_tables": "INVALID"}),
        clock=SimpleNamespace(ticks=0), children=[])


def _slotted_provider(moves, slot):
    """A REAL ManualMoveProvider carrying its own follow-window slot.

    ``uh_slot`` is the attribute RelocationRunner reads to decide which
    accumulator to reset; setting it on the real provider is configuration,
    not substitution.  The manual itinerary keeps the MOVE deterministic so
    the assertion is about slot ownership rather than tracker behaviour.
    """
    provider = ManualMoveProvider(tuple(moves))
    provider.uh_slot = slot
    return provider


def _clock(ticks):
    """A stub domain clock, carrying the spec a real one always has.

    ``spec.start_ticks`` is the domain's activation epoch, which
    ``refresh_model_time`` publishes onto the state beside the model
    time; 0 is "this domain started with the run", which is what these
    followers are.
    """
    return SimpleNamespace(ticks=int(ticks), tick_den=1,
                           elapsed_seconds=float(ticks),
                           spec=SimpleNamespace(start_ticks=0,
                                                step_ticks=60))


def _collection_model(parent, children):
    nodes = {int(parent.cfg.grid_id): parent}
    for child in children:
        nodes[int(child.cfg.grid_id)] = child
    model = SimpleNamespace(
        root=parent,
        nodes_by_grid_id=nodes,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)),
        experiment_fingerprint="f" * 64)
    model.node = lambda gid: nodes[gid]
    return model


def _follow_runner(parent_plane, grid_id, moves, slot, tmp_path):
    config = RelocationConfig(
        enabled=True, grid_id=grid_id, max_move_parent_cells=4,
        min_overlap_fraction=0.25, cadence_seconds=None,
        moves=tuple(moves))
    return RelocationRunner(
        config=config,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)),
        provider=_slotted_provider(moves, slot),
        staging="host",
        initializer=_initializer(parent_plane),
        static_provenance="footprint-parametric synthetic statics (test)",
        on_child_built=lambda *args: None,
        receipts_path=tmp_path / f"relocation.d{grid_id:02d}.json")


def test_two_followers_move_independently_in_one_batch(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    sibling = _second_child(parent, child)
    parent.children = [child, sibling]
    model = _collection_model(parent, [child, sibling])

    slot_a = uh_diag.follow_window_slot(2)
    slot_b = uh_diag.follow_window_slot(3)
    assert slot_a != slot_b

    collection = RelocationRunnerCollection([
        _follow_runner(parent_plane, 2,
                       [ScheduledRelocationMove(60.0, 1, 0)], slot_a, tmp_path),
        _follow_runner(parent_plane, 3,
                       [ScheduledRelocationMove(60.0, 0, 1)], slot_b, tmp_path),
    ])
    assert collection.target_grid_ids == (2, 3)

    parent.clock = _clock(60)
    child.clock = _clock(60)
    sibling.clock = _clock(60)

    outcome = collection.on_period_begin(model, {1: parent.clock}, period=1)
    assert outcome["event"] == "batch"
    moved = {int(row["grid_id"]): row for row in outcome["outcomes"]
             if row.get("event") == "relocated"}
    assert set(moved) == {2, 3}
    # Each followed its OWN itinerary: d02 shifts in i, d03 in j.
    assert moved[2]["executed_shift_parent_cells"] == [1, 0]
    assert moved[3]["executed_shift_parent_cells"] == [0, 1]
    # ...and each wrote its own receipts file, not a shared one.
    assert (tmp_path / "relocation.d02.json").exists()
    assert (tmp_path / "relocation.d03.json").exists()


def test_one_followers_window_reset_leaves_the_others_intact(tmp_path):
    """The per-follower accumulator, which a shared slot would clobber."""
    parent_plane, parent, child = _cpu_tree()
    sibling = _second_child(parent, child)
    parent.children = [child, sibling]
    model = _collection_model(parent, [child, sibling])

    slot_a = uh_diag.follow_window_slot(2)
    slot_b = uh_diag.follow_window_slot(3)
    scratch = {slot_a: np.full((4, 4), 7.0, dtype=np.float32),
               slot_b: np.full((4, 4), 9.0, dtype=np.float32)}
    parent.state.existing_scratch = lambda slot: scratch.get(slot)

    runner = _follow_runner(
        parent_plane, 2, [ScheduledRelocationMove(60.0, 1, 0)], slot_a,
        tmp_path)
    parent.clock = _clock(60)
    child.clock = _clock(60)
    runner.on_period_begin(model, {1: parent.clock}, period=1)

    assert not scratch[slot_a].any(), "d02's own window must be reset"
    assert (scratch[slot_b] == 9.0).all(), "d03's window must be untouched"


# ---------------------------------------------------------------------------
# Retirement detaches a subtree at a completed leg boundary
# ---------------------------------------------------------------------------

def _tree_node(grid_id, parent=None):
    node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=grid_id), parent=parent, children=[],
        state=SimpleNamespace(), grid=f"grid-d{grid_id:02d}", _started=True)
    if parent is not None:
        parent.children.append(node)
    return node


class RecordingWriterSet:
    def __init__(self):
        self.removed = []

    def remove_domain(self, grid_id):
        self.removed.append(int(grid_id))


def _detach_model(nodes):
    return SimpleNamespace(
        nodes_by_grid_id=dict(nodes),
        _prepared_by_grid_id={gid: object() for gid in nodes})


def test_detaching_a_retired_parent_takes_its_whole_subtree():
    root = _tree_node(1)
    d02 = _tree_node(2, root)
    d03 = _tree_node(3, d02)
    model = _detach_model({1: root, 2: d02, 3: d03})
    writers = RecordingWriterSet()
    steppers = {2: SimpleNamespace(close=lambda: None),
                3: SimpleNamespace(close=lambda: None)}

    detached = _detach_retired_children(model, [2], writers, steppers)

    # Deepest first, so a child is never orphaned mid-detach.
    assert detached == [3, 2]
    assert set(model.nodes_by_grid_id) == {1}
    assert root.children == []
    assert sorted(writers.removed) == [2, 3]
    assert steppers == {}


def test_detaching_never_removes_the_root():
    root = _tree_node(1)
    model = _detach_model({1: root})
    assert _detach_retired_children(model, [1], None, None) == []
    assert set(model.nodes_by_grid_id) == {1}


def test_detaching_nothing_is_a_no_op():
    root = _tree_node(1)
    d02 = _tree_node(2, root)
    model = _detach_model({1: root, 2: d02})
    assert _detach_retired_children(model, [], None, None) == []
    assert set(model.nodes_by_grid_id) == {1, 2}


# ---------------------------------------------------------------------------
# SpawnRunner retires a live episode at a leg boundary
# ---------------------------------------------------------------------------

class QuietParentState:
    """A parent whose spawn window carries no signal worth keeping a nest for."""

    def __init__(self, shape=(40, 40)):
        self.planes = {"uh_spawn_window": np.zeros(shape, dtype=np.float32)}

    def existing_scratch(self, slot):
        return self.planes.get(slot)


def _spawned_runner(tmp_path, d02=LIFECYCLE):
    """A SpawnRunner whose d02 episode is already live, ready to retire."""
    from gpuwm.core.spawn_runner import SpawnRunner

    exp = _exp(tmp_path, d02)
    runner = SpawnRunner(experiment=exp, on_child_built=lambda *a, **k: None,
                         receipts_path=tmp_path / "spawn.json")
    # The slot fired on an earlier leg: episode 1 is live and integrating.
    dc = exp.domain(2)
    runner.spawned[2] = (int(dc.i_parent_start), int(dc.j_parent_start))
    runner.episodes[2] = 1
    runner.birth_times[2] = 0.0
    watch = runner.controller.watches[2]
    watch.fired = True
    watch.closed = True

    parent = SimpleNamespace(
        cfg=exp.domain(1), state=QuietParentState(), parent=None,
        children=[], _started=True)
    child = SimpleNamespace(
        cfg=dc, state=SimpleNamespace(), parent=parent, children=[],
        _started=True)
    parent.children.append(child)
    nodes = {1: parent, 2: child}
    model = SimpleNamespace(
        nodes_by_grid_id=nodes,
        root=SimpleNamespace(clock=SimpleNamespace(elapsed_seconds=0.0)))
    model.node = lambda gid: nodes[gid]
    model.walk_parent_first = lambda: [parent, child]
    return exp, runner, model


def test_a_quiet_episode_retires_at_a_leg_boundary(tmp_path):
    """min_lifetime_s then sustained_s of quiet, evaluated at boundaries."""
    exp, runner, model = _spawned_runner(tmp_path)

    # Inside the minimum lifetime: nothing retires however quiet it is.
    assert runner.on_leg_boundary(model, t=600.0) is None
    assert 2 in runner.spawned

    # Past the lifetime, the quiet window still has to accumulate.
    assert runner.on_leg_boundary(model, t=1800.0) is None
    assert 2 in runner.spawned

    record = runner.on_leg_boundary(model, t=2700.0)
    assert record is not None
    assert record["retired_grid_ids"] == [2]
    assert 2 not in runner.spawned
    assert 2 in runner.retired


def _retire_d02(runner, model):
    """Walk the boundary ladder a retirement actually requires.

    Quiet has to be OBSERVED across boundaries: the watch opens its window
    at the first boundary past ``min_lifetime_s`` and only retires once
    ``sustained_s`` has elapsed since then, so a single late boundary
    retires nothing however quiet the parent is.
    """
    record = None
    for t in (600.0, 1800.0, 2700.0):
        record = runner.on_leg_boundary(model, t=t)
    return record


def test_a_single_late_boundary_retires_nothing(tmp_path):
    """The quiet window is measured, not inferred from the clock."""
    exp, runner, model = _spawned_runner(tmp_path)
    assert runner.on_leg_boundary(model, t=99999.0) is None
    assert 2 in runner.spawned


def test_the_next_leg_is_built_without_the_retired_domain(tmp_path):
    """Retirement changes the NEXT schedule's domain set, not this one."""
    exp, runner, model = _spawned_runner(tmp_path)
    record = _retire_d02(runner, model)

    next_leg = record["experiment"]
    assert [int(dc.grid_id) for dc in next_leg.domains] == [1]
    # The DECLARED experiment is untouched: retirement is a live view.
    assert [int(dc.grid_id) for dc in exp.domains] == [1, 2]


def test_a_retired_slot_still_counts_its_episode_for_the_rearm_bound(tmp_path):
    exp, runner, model = _spawned_runner(tmp_path)
    _retire_d02(runner, model)
    assert runner.episodes[2] == 1
    assert runner.retired_times[2] == 2700.0


def test_a_one_shot_slot_has_no_retirement_watch(tmp_path):
    """No retire table means no retirement evaluation at any boundary."""
    exp, runner, model = _spawned_runner(tmp_path, d02=ONE_SHOT)
    assert runner._retirement == {}
    assert runner.on_leg_boundary(model, t=2700.0) is None
    assert 2 in runner.spawned


# ---------------------------------------------------------------------------
# The episode on the per-step stream
# ---------------------------------------------------------------------------
#
# An episode beginning and ending was invisible until the run was over:
# it lived in spawn_receipts.json, which is a post-mortem artifact.  A
# live view has to be told at the boundary, which is what these carry.


def _logged(model, tmp_path):
    from datetime import datetime

    from gpuwm.progress_log import StepLog, publish_step_log

    log = StepLog(start_time=datetime(2026, 8, 15, 0, 0, 0),
                  run_seconds=7200.0,
                  jsonl_path=tmp_path / "progress.jsonl")
    publish_step_log(model, log)
    return log


def _nest_records(tmp_path, log):
    from gpuwm.progress_log import NEST_EVENTS, read_step_log

    log.close(status="SUCCESS")
    return [r for r in read_step_log(tmp_path / "progress.jsonl")
            if r["event"] in NEST_EVENTS]


def test_a_retirement_reaches_the_per_step_stream(tmp_path):
    exp, runner, model = _spawned_runner(tmp_path)
    log = _logged(model, tmp_path)
    _retire_d02(runner, model)

    record, = _nest_records(tmp_path, log)
    assert record["event"] == "nest_retired"
    assert record["domain"] == 2
    assert record["episode"] == 1
    assert record["model_seconds"] == 2700.0
    assert record["valid_time"] == "2026-08-15_00:45:00"
    # The decision the watch actually made, not a paraphrase of it.
    assert record["reason"] == "retire"


def test_the_boundaries_before_a_retirement_emit_nothing(tmp_path):
    """A watch that held is not an episode event.

    The retirement ladder walks three boundaries and only the last one
    ends an episode; emitting on the holds would put a record on the
    stream at every leg boundary of every run that declares a retire
    table, with nothing for a consumer to redraw.
    """
    exp, runner, model = _spawned_runner(tmp_path)
    log = _logged(model, tmp_path)
    for t in (600.0, 1800.0):
        runner.on_leg_boundary(model, t=t)
    assert _nest_records(tmp_path, log) == []


#: The same lifecycle, on a FIELD trigger the quiet parent never fires.
#: A ``time`` trigger re-fires the instant its slot re-arms, which is
#: correct behaviour and makes the re-arm impossible to observe on its
#: own; this slot re-arms and then waits for signal that never comes.
LIFECYCLE_QUIET = (
    'spawn = { trigger = "uh", threshold = 500.0, earliest_s = 0.0, '
    'latest_s = 86400.0 }\n'
    'retire = { trigger = "uh", threshold = 60.0, sustained_s = 900.0, '
    'min_lifetime_s = 1800.0 }\n'
    'rearm = { max_firings = 3, cooldown_s = 600.0 }')


def test_a_rearm_reaches_the_stream_naming_the_episode_it_arms_for(tmp_path):
    exp, runner, model = _spawned_runner(tmp_path, d02=LIFECYCLE_QUIET)
    log = _logged(model, tmp_path)
    _retire_d02(runner, model)
    # Past the declared cooldown, the slot re-opens at the next boundary.
    cooldown = float(exp.domain(2).rearm.cooldown_s)
    runner.on_leg_boundary(model, t=2700.0 + cooldown + 600.0)

    events = _nest_records(tmp_path, log)
    assert [r["event"] for r in events] == ["nest_retired", "nest_rearmed"]
    rearmed = events[-1]
    assert rearmed["domain"] == 2
    # The episode it is armed FOR: max_firings is stated in firings, so
    # a consumer counting episodes must see the one about to start.
    assert rearmed["episode"] == 2
    assert rearmed["cooldown_seconds"] == cooldown
    # Nothing has been built yet, so there is nothing to place on a map.
    assert rearmed["lat"] is None and rearmed["lon"] is None


def test_publishing_a_log_changes_no_lifecycle_decision(tmp_path):
    """Telemetry never steers: the same boundaries, the same tree."""
    def run(with_log):
        exp, runner, model = _spawned_runner(tmp_path)
        if with_log:
            _logged(model, tmp_path)
        record = _retire_d02(runner, model)
        return (sorted(runner.retired), sorted(runner.spawned),
                dict(runner.episodes), dict(runner.retired_times),
                record["retired_grid_ids"])

    assert run(False) == run(True)


# ---------------------------------------------------------------------------
# The restart ADMISSION, all four directions
# ---------------------------------------------------------------------------
#
# This seam used to refuse.  It named the breakage it prevented -- a slot
# re-firing from policy state the resume did not have -- and that breakage
# is now built out, so the seam reports instead: does this resume carry
# lifecycle policy across the split?  The refusals that survive belong to
# the checkpoint READER (tests/test_restart.py, the
# read_tree_lifecycle_header family), because they are the ones a
# checkpoint can still be wrong about.

def test_restart_admits_a_declared_lifecycle(tmp_path):
    exp = _exp(tmp_path, LIFECYCLE)
    assert admit_restart_with_lifecycle(exp, restart="ckpt.nc") is True


def test_restart_admits_a_follow_only_lifecycle(tmp_path):
    """A follower carries hysteresis even with no retire/rearm table."""
    d02 = (ONE_SHOT + '\nfollow = { field = "uh", threshold = 100.0, '
           'fallback_threshold = 35.0, search_margin_cells = 12, '
           'min_shift_cells = 2, max_shift_cells = 10, '
           'cooldown_seconds = 600.0, cadence_seconds = 900.0, '
           'max_move_parent_cells = 8, min_overlap_fraction = 0.70 }')
    exp = _exp(tmp_path, d02)
    assert admit_restart_with_lifecycle(exp, restart="ckpt.nc") is True


def test_a_lifecycle_free_restart_carries_no_policy_state(tmp_path):
    """A one-shot spawn has no table that can differ across the split."""
    exp = _exp(tmp_path, ONE_SHOT)
    assert admit_restart_with_lifecycle(exp, restart="ckpt.nc") is False


def test_a_lifecycle_run_from_t0_carries_nothing_across(tmp_path):
    exp = _exp(tmp_path, LIFECYCLE)
    assert admit_restart_with_lifecycle(exp, restart=None) is False


def test_the_route_still_reaches_the_admission_seam():
    """The gate is named at the call site, not lifted by deletion.

    A refusal removed and its call site removed with it is the same
    source diff as a refusal that was never enforced; this pins that the
    tree route still consults the seam by name.
    """
    import inspect

    from gpuwm import runtime

    source = inspect.getsource(runtime.run_experiment)
    assert "admit_restart_with_lifecycle(exp, restart)" in source


# ---------------------------------------------------------------------------
# The resume rebuild: the tree a lifecycle checkpoint describes
# ---------------------------------------------------------------------------
#
# Driven on the REAL objects -- a real SpawnRunner, the real materializer,
# the real attach seam, a real idealized tree -- because the whole claim of
# this leg is that the resume rebuilds through the run's own seams rather
# than assembling a lookalike tree beside them.

def _restore_case():
    """A dormant-d02 experiment and a root-only tree to resume into."""
    from dataclasses import replace as _replace

    from gpuwm.core.nest_spawn import SpawnConfig
    from gpuwm.core.spawn_runner import SpawnRunner
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree
    from test_nest_spawn_init import _experiment, _live_parent
    from test_spawn_runner import (_CountingCoupler, _CountingPreparer,
                                   _Prepared)

    class _RelocatableCoupler(_CountingCoupler):
        """The executor's coupler surface plus the relocation seam.

        A collaborator: ``relocate_child`` rebuilds the SINT donor tables
        through it, and a CPU tier has no donor tables to rebuild.
        """

        def relocate(self):
            self.valid = False
            return {"rebuilt": True}

    exp = _experiment()
    dormant = _replace(exp.domains[1],
                       spawn=SpawnConfig(trigger="time", at_s=120.0))
    exp = _replace(exp, domains=(exp.domains[0], dormant))
    parent, grids = _live_parent(exp)
    preparer = _CountingPreparer()
    runner = SpawnRunner.from_experiment(
        exp, on_child_built=preparer, array_module=np)
    model = assemble_idealized_tree(
        runner.active, parent.state, grids=(grids[0],),
        coupler_factory=_RelocatableCoupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}
    return SimpleNamespace(exp=exp, model=model, runner=runner,
                           preparer=preparer,
                           coupler_factory=_RelocatableCoupler)


def _spawn_block(*, fired=(20, 18), current=None, episode=1):
    """The ``spawn`` half of a lifecycle block, as the writer emits it."""
    current = fired if current is None else current
    return {
        "watches": {"2": {"fired": True, "closed": False}},
        "spawned": {"2": {"fired": list(fired), "current": list(current),
                          "episode": episode, "born_t": 120.0}},
        "retired": {},
        "episodes": {"2": episode},
        "quiet_since": {},
        "spawns_executed": 1,
    }


def _peek(spawn, *, followers=None, leg_seconds=120.0, relocation=None):
    from gpuwm.io.restart import NEST_LIFECYCLE_CONTRACT, TreeLifecycleHeader

    return TreeLifecycleHeader(
        root_path=None, domain_ids=(1, 2),
        block={"contract": NEST_LIFECYCLE_CONTRACT,
               "leg_seconds": leg_seconds, "spawn": spawn,
               "followers": dict(followers or {}), "window_slots": {}},
        relocation=relocation)


def test_a_restored_episode_rejoins_the_tree_at_its_fired_placement():
    """The tree a checkpoint's member set describes, rebuilt before a
    single array lands: without this the restore refuses on a partial
    domain set, and with a GUESSED tree it restores d02's arrays into a
    nest that is not where d02 was."""
    from gpuwm.runtime import restore_nest_lifecycle

    case = _restore_case()
    assert sorted(case.model.nodes_by_grid_id) == [1]

    episodes = restore_nest_lifecycle(
        case.model, case.exp, _peek(_spawn_block()),
        spawn_runner=case.runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)

    assert sorted(case.model.nodes_by_grid_id) == [1, 2]
    child = case.model.node(2)
    assert child.parent is case.model.root
    assert child in case.model.root.children
    assert child.coupler is not None and child._started
    assert (int(child.cfg.i_parent_start),
            int(child.cfg.j_parent_start)) == (20, 18)
    # The route preparer ran for the newborn: a nest rebuilt without its
    # physics driver integrates nothing.
    assert case.preparer.calls == [(2, 1)]
    assert episodes == {2: 0}, "a one-shot spawn keeps the flat pathname"


def test_the_restored_runner_holds_the_child_result_the_next_leg_adopts():
    """``refresh_from_model`` carries live spawned children across legs by
    walking ``_child_results``; a resume that leaves it empty makes the
    restored episode invisible to its own retire watch."""
    from gpuwm.runtime import restore_nest_lifecycle

    case = _restore_case()
    restore_nest_lifecycle(
        case.model, case.exp, _peek(_spawn_block()),
        spawn_runner=case.runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)

    assert sorted(case.runner._child_results) == [2]
    case.runner.refresh_from_model(case.model)
    assert case.runner._child_results[2].state is case.model.node(2).state


def test_a_slot_that_never_fired_is_not_materialized():
    """The block is what says which slots fired.  A watch that reports
    un-fired must leave the tree root-only, or the resume invents an
    episode the run never had."""
    from gpuwm.runtime import restore_nest_lifecycle

    case = _restore_case()
    block = {"watches": {"2": {"fired": False, "closed": False}},
             "spawned": {}, "retired": {}, "episodes": {"2": 0},
             "quiet_since": {}, "spawns_executed": 0}

    episodes = restore_nest_lifecycle(
        case.model, case.exp, _peek(block), spawn_runner=case.runner,
        lbc_interval_s=None, coupler_factory=case.coupler_factory)

    assert sorted(case.model.nodes_by_grid_id) == [1]
    assert episodes == {}
    assert case.runner.pending == (2,), "the slot is still watched"


def test_the_restored_tree_re_aims_its_schedule_over_the_restored_set():
    """``restore_tree_restart`` reads the tick denominator and the period
    off ``model.schedule``, and both are derived from the LIVE domain set.
    A root-only schedule under a two-domain member set makes a legitimate
    resume look like a mismatched tick pair."""
    from gpuwm.core.clock import resolve_clock
    from gpuwm.runtime import restore_nest_lifecycle

    case = _restore_case()
    assert sorted(case.model.schedule.clock.clocks()) == [1]
    restore_nest_lifecycle(
        case.model, case.exp, _peek(_spawn_block()),
        spawn_runner=case.runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)

    straight = resolve_clock(case.runner.active, lbc_interval_s=None)
    assert sorted(case.model.schedule.clock.clocks()) == [1, 2]
    assert int(case.model.schedule.clock.tick_den) == int(straight.tick_den)


def test_a_declared_lifecycle_resumes_into_its_own_episode_directory(
        tmp_path):
    """``output_episode`` is the one rule: a declared retire/rearm table
    numbers episodes, a one-shot spawn keeps the flat name.  The resume
    seed and the live spawn boundary read it the same way."""
    from dataclasses import replace as _replace

    from gpuwm.core.nest_lifecycle import RetireConfig
    from gpuwm.core.spawn_runner import SpawnRunner
    from gpuwm.runtime import restore_nest_lifecycle

    case = _restore_case()
    child = _replace(case.exp.domains[1], retire=RetireConfig(
        trigger="uh", threshold=25.0, sustained_s=600.0,
        min_lifetime_s=900.0))
    exp = _replace(case.exp, domains=(case.exp.domains[0], child))
    runner = SpawnRunner.from_experiment(
        exp, on_child_built=case.preparer, array_module=np)

    episodes = restore_nest_lifecycle(
        case.model, exp, _peek(_spawn_block(episode=2)),
        spawn_runner=runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)

    assert episodes == {2: 2}
    writers = _writer_set(tmp_path, episodes=episodes)
    writers.submit(_node(2), 0)
    assert writers._writers[2].paths[0].parent == \
        tmp_path / "d02" / "episode-002"


def test_the_writer_episode_seed_defaults_to_the_flat_pathname(tmp_path):
    """Byte-inert by default: a run that is not a lifecycle resume passes
    nothing and every domain is episode 0, which is the pathname every
    existing run already writes."""
    writers = _writer_set(tmp_path, episodes={2: 0})
    writers.submit(_node(2), 0)
    assert writers._writers[2].paths[0].parent == tmp_path


def test_a_follower_history_with_no_follower_to_seed_refuses():
    """The entry carries a segment chain, a move count and two cooldown
    anchors.  Dropping them leaves the nest at a placement the config
    cannot explain, free to move again at the first cadence boundary."""
    from gpuwm.runtime import restore_nest_followers

    case = _restore_case()
    case.model._relocation_runner = None
    peek = _peek(_spawn_block(), followers={"2": {
        "kind": "per-domain", "current_placement": [20, 18],
        "declared_placement": [20, 18], "segment": None,
        "moves_executed": 0, "last_proposal_t": None,
        "last_move_t": None}})

    with pytest.raises(RuntimeError, match="builds no follower"):
        restore_nest_followers(case.model, peek)


def test_a_follower_the_block_says_nothing_about_is_built_fresh():
    """A follow target still DORMANT when the checkpoint was taken has no
    entry: nothing was consulted and nothing moved, which is exactly what
    a never-consulted runner already holds."""
    from gpuwm.runtime import restore_nest_followers

    case = _restore_case()
    case.model._relocation_runner = None
    assert restore_nest_followers(case.model, _peek(_spawn_block())) == []


# ---------------------------------------------------------------------------
# The resume rebuild: moved placements and the fingerprint re-mark
# ---------------------------------------------------------------------------

def _follower_entry(*, current, declared=(20, 18), segment=None,
                    moves=0, last_proposal_t=None, last_move_t=None):
    return {"kind": "per-domain",
            "current_placement": list(current),
            "declared_placement": list(declared),
            "segment": segment, "moves_executed": moves,
            "last_proposal_t": last_proposal_t, "last_move_t": last_move_t}


def _numpy_relocation_initializer(new_dc, parent_node, **kwargs):
    """The REAL cold-start rebuild, on the CPU array module.

    ``relocate_child`` forwards no array module (production initializers
    close over their own), so a CPU-tier test binds one here rather than
    standing in for the rebuild it is measuring.
    """
    from gpuwm.ingest.nest_init import parent_only_init

    return parent_only_init(new_dc, parent_node, array_module=np, **kwargs)


def _restored_follower(case, *, cadence=120.0):
    """A real RelocationRunner over the restored tree's d02."""
    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.core.storm_tracking import FollowConfig, StormTracker

    tracker = FollowConfig(
        field="uh", threshold=60.0, search_margin_cells=4,
        min_shift_cells=1, max_shift_cells=4, cooldown_seconds=600.0,
        fallback_threshold=35.0)
    # The fixture trap: RelocationConfig refuses a cadence with no follow
    # source, even when the provider is handed in explicitly.
    config = RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=8,
        min_overlap_fraction=0.10, cadence_seconds=cadence, follow=tracker)
    runner = RelocationRunner(
        config=config, provider=StormTracker(tracker),
        schedule=SimpleNamespace(period_ticks=60,
                                 clock=SimpleNamespace(tick_den=1)),
        on_child_built=case.preparer, staging="device",
        initializer=_numpy_relocation_initializer,
        static_provenance="parent-interpolated (CPU tier)")
    case.model._relocation_runner = runner
    return runner


def test_a_moved_episode_is_brought_to_its_current_placement_in_one_hop():
    """The nest is materialized where it FIRED, because that is what the
    spawn rule adjudicated; every move after that is re-admitted under the
    RELOCATION rule, in a single hop rather than a replay."""
    from gpuwm.runtime import restore_nest_followers, restore_nest_lifecycle

    case = _restore_case()
    restore_nest_lifecycle(
        case.model, case.exp,
        _peek(_spawn_block(fired=(20, 18), current=(22, 20))),
        spawn_runner=case.runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)
    child = case.model.node(2)
    assert (int(child.cfg.i_parent_start),
            int(child.cfg.j_parent_start)) == (20, 18)

    runner = _restored_follower(case)
    peek = _peek(_spawn_block(fired=(20, 18), current=(22, 20)),
                 followers={"2": _follower_entry(current=(22, 20))})
    assert restore_nest_followers(
        case.model, peek, spawn_runner=case.runner) == [2]

    child = case.model.node(2)
    assert (int(child.cfg.i_parent_start),
            int(child.cfg.j_parent_start)) == (22, 20)
    # The persisted history lands OVER the hop's own, or the next move
    # chains onto the base preparation instead of its real predecessor.
    assert runner.moves_executed == 0
    assert runner._segment is None


def _three_move_segment_json():
    """A REAL segment chain of generation 3, harvested from real hops.

    Hand-writing one is not an option: ``RelocationSegment.from_json`` is
    total at the top and digest-preserving underneath, so a chain whose
    counter and records disagree is refused -- which is the invariant this
    test needs intact in order to mean anything.
    """
    from gpuwm.core.nest_relocation import base_segment, relocate_child
    from gpuwm.runtime import restore_nest_lifecycle

    case = _restore_case()
    restore_nest_lifecycle(
        case.model, case.exp, _peek(_spawn_block()),
        spawn_runner=case.runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)
    runner = _restored_follower(case)
    node = case.model.node(2)
    segment = base_segment(node.cfg)
    for step in range(3):
        receipt = relocate_child(
            node, i_parent_start=int(node.cfg.i_parent_start) + 1,
            j_parent_start=int(node.cfg.j_parent_start) + 1,
            segment=segment, bounds=runner.config,
            initializer=runner.initializer,
            static_provenance=runner.static_provenance,
            on_child_built=runner.on_child_built, staging="device")
        segment = receipt["segment_state"]
        del step
    return segment.to_json()


def test_the_restored_segment_is_the_checkpoints_not_the_hops():
    """Generation continues; it does not restart at the hop.

    A follower that resumes at the hop's own generation chains its next
    move's record onto the wrong predecessor, so the digest the
    checkpoint's fingerprint was folded from can never be reproduced
    again."""
    from gpuwm.runtime import restore_nest_followers, restore_nest_lifecycle

    stored = _three_move_segment_json()
    assert int(stored["generation"]) == 3

    case = _restore_case()
    restore_nest_lifecycle(
        case.model, case.exp,
        _peek(_spawn_block(fired=(20, 18), current=(22, 20))),
        spawn_runner=case.runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)
    runner = _restored_follower(case)

    peek = _peek(_spawn_block(fired=(20, 18), current=(22, 20)),
                 followers={"2": _follower_entry(
                     current=(22, 20), segment=stored, moves=3,
                     last_proposal_t=240.0, last_move_t=240.0)})
    restore_nest_followers(case.model, peek, spawn_runner=case.runner)

    assert runner.moves_executed == 3
    assert int(runner._segment.generation) == 3
    assert runner._segment.to_json() == stored
    assert runner.provider._last_move_t == 240.0


def test_a_move_with_no_follower_entry_refuses():
    """Only a follower moves a spawned nest, so a current placement away
    from the fired one with no follower history means the block's two
    halves disagree and no band exists to re-admit the placement under."""
    from gpuwm.runtime import restore_nest_followers, restore_nest_lifecycle

    case = _restore_case()
    restore_nest_lifecycle(
        case.model, case.exp,
        _peek(_spawn_block(fired=(20, 18), current=(22, 20))),
        spawn_runner=case.runner, lbc_interval_s=None,
        coupler_factory=case.coupler_factory)
    case.model._relocation_runner = None

    with pytest.raises(RuntimeError, match="no follower history"):
        restore_nest_followers(
            case.model, _peek(_spawn_block(fired=(20, 18),
                                           current=(22, 20))),
            spawn_runner=case.runner)


def test_the_fingerprint_re_mark_reproduces_the_moved_runs_own():
    """A moved run's checkpoints are keyed to its move history, so a fresh
    build refuses them by construction.  A legitimate resume gets in by
    doing the same arithmetic over the header's own record chain."""
    from gpuwm.core.nest_relocation import mark_fingerprint_across_move
    from gpuwm.runtime import remark_relocation_fingerprint

    case = _restore_case()
    fresh = "fingerprint-of-the-fresh-build"
    case.model.experiment_fingerprint = fresh
    case.model._experiment_fingerprint_components = {"experiment": {"n": 2}}

    records = ["a" * 64, "b" * 64]
    live = fresh
    for record in records:
        live = mark_fingerprint_across_move(live, record)

    peek = _peek(_spawn_block(), relocation={
        "moves": 2, "record_sha256": records, "segment_id": "seg-1",
        "grid_id": 2, "posture": "..."})
    assert remark_relocation_fingerprint(case.model, peek) == live
    assert case.model.experiment_fingerprint == live
    assert case.model._experiment_fingerprint_components["relocation"] == {
        "records": records}


def test_a_foreign_record_chain_does_not_reproduce_the_fingerprint():
    """The mark is one-way and deterministic in the digests, so folding
    someone else's moves cannot land on this run's value."""
    from gpuwm.core.nest_relocation import mark_fingerprint_across_move
    from gpuwm.runtime import remark_relocation_fingerprint

    case = _restore_case()
    case.model.experiment_fingerprint = "base"
    case.model._experiment_fingerprint_components = None
    live = mark_fingerprint_across_move("base", "a" * 64)

    foreign = _peek(_spawn_block(), relocation={
        "moves": 1, "record_sha256": ["c" * 64], "segment_id": "seg-9",
        "grid_id": 2, "posture": "..."})
    assert remark_relocation_fingerprint(case.model, foreign) != live


def test_a_wrong_reconstructed_placement_is_refused_by_the_setup_gate(
        tmp_path):
    """The loud gate behind the whole rebuild.

    Every serialized array is overwritten by the restore, so what makes a
    misplaced reconstruction detectable is the SETUP state -- the grid,
    map factors and geometry the placement determines.  ``_validate_restart``
    compares exactly that, per member, before a single live array is
    touched."""
    from gpuwm.io import restart as restart_io
    from gpuwm.runtime import restore_nest_lifecycle

    right = _restore_case()
    restore_nest_lifecycle(
        right.model, right.exp, _peek(_spawn_block(fired=(20, 18))),
        spawn_runner=right.runner, lbc_interval_s=None,
        coupler_factory=right.coupler_factory)
    wrong = _restore_case()
    restore_nest_lifecycle(
        wrong.model, wrong.exp, _peek(_spawn_block(fired=(24, 22))),
        spawn_runner=wrong.runner, lbc_interval_s=None,
        coupler_factory=wrong.coupler_factory)

    right_state = right.model.node(2).state
    wrong_state = wrong.model.node(2).state
    assert restart_io.setup_fingerprint(right_state) != \
        restart_io.setup_fingerprint(wrong_state), \
        "a placement the restore cannot tell apart is a placement the " \
        "restore cannot refuse"

    path = tmp_path / "child.npz"
    restart_io.write_restart(path, right_state, right.model.node(2).cfg.run)
    with pytest.raises(restart_io.RestartMismatchError,
                       match="different model setup"):
        restart_io._validate_restart(
            path, wrong_state, wrong.model.node(2).cfg.run)


def test_a_resumed_writer_set_does_not_trip_the_same_run_duplicate_guard(
        tmp_path):
    """Both halves of the guard, at a lifecycle resume.

    The boundary instant belongs to both segments: the run that wrote the
    checkpoint published that frame, and the resumed run's suppression of
    it lives in ``_resume_committed_history_grid_ids``, not here.  What
    must NOT happen is the guard treating the previous SEGMENT's frame as
    this run's own -- it is scoped to paths THIS writer set published, and
    a resumed process starts with none.  A genuine same-run duplicate
    still refuses."""
    first = _writer_set(tmp_path, episodes={2: 1})
    first.submit(_node(2), 0)

    resumed = _writer_set(tmp_path, episodes={2: 1})
    resumed.submit(_node(2), 0)
    assert resumed._writers[2].paths == first._writers[2].paths

    with pytest.raises(RuntimeError, match="already published"):
        resumed.submit(_node(2), 0)


# ---------------------------------------------------------------------------
# The resume boundary pass
# ---------------------------------------------------------------------------

def _leg_walk_model(monkeypatch, stepped):
    """The real leg-walk case: d02 dormant on a 120 s trigger, 240 s run."""
    from gpuwm.core.spawn_runner import SpawnRunner
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree
    from test_nest_spawn_init import _live_parent
    from test_spawn_runner import (_CountingCoupler, _CountingPreparer,
                                   _Prepared, _leg_walk_case)

    dexp = _leg_walk_case()
    parent, grids = _live_parent(dexp)
    monkeypatch.setattr("gpuwm.core.dycore.step",
                        lambda state, cfg, **_k: stepped.append(cfg.grid_id))
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = assemble_idealized_tree(
        runner.active, parent.state, grids=(grids[0],),
        coupler_factory=_CountingCoupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}
    # The spawn consumer's own window, folded to a value the boundary must
    # spend: "max since I last looked" is reset by the evaluation itself.
    root = model.root
    window = root.state.scratch(
        (int(root.cfg.run.ny), int(root.cfg.run.nx)),
        uh_diag.UH_SPAWN_WINDOW_SLOT)
    window[...] = np.float32(7.5)
    return dexp, model, runner, _CountingCoupler


def _carry_to(model, seconds):
    """Put the tree's clocks where a restore would leave them."""
    for node in model.walk_parent_first():
        clock = node.clock
        clock.ticks = int(round(float(seconds) * clock.tick_den))
        clock.step_count = max(
            0, (clock.ticks - clock.spec.start_ticks) // clock.spec.step_ticks)
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(
        gid for gid, node in model.nodes_by_grid_id.items()
        if node.clock.history_due())


def test_a_resume_on_the_leg_lattice_births_the_same_nest_as_straight_through(
        monkeypatch):
    """The checkpoint is written at PERIOD_BEGIN, always BEFORE the
    boundary is evaluated.  A resume that skips straight into the next leg
    skips that evaluation forever: the nest never born, the episode never
    retired, nothing anywhere saying so."""
    from gpuwm.runtime import walk_spawn_legs

    straight_steps = []
    dexp, straight, straight_runner, coupler = _leg_walk_model(
        monkeypatch, straight_steps)
    walk_spawn_legs(straight, dexp, None, spawn_runner=straight_runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=coupler, validate_state=False)

    resumed_steps = []
    dexp, resumed, resumed_runner, coupler = _leg_walk_model(
        monkeypatch, resumed_steps)
    _carry_to(resumed, 120.0)
    walk_spawn_legs(resumed, dexp, None, spawn_runner=resumed_runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=coupler, validate_state=False)

    assert sorted(resumed.nodes_by_grid_id) == sorted(
        straight.nodes_by_grid_id) == [1, 2]
    assert resumed_runner.spawns_executed == straight_runner.spawns_executed
    assert resumed_runner.spawned == straight_runner.spawned
    # Only the SECOND leg is replayed, so the resumed run steps exactly the
    # straight run's post-boundary work and none of what came before.
    assert resumed_steps.count(2) == straight_steps.count(2)
    assert resumed_steps.count(1) == straight_steps.count(1) // 2
    assert float(resumed.root.clock.elapsed_seconds) == 240.0
    # The window zeroing is the boundary's too: the watch looked, so its
    # "max since I last looked" window is spent in both arms.
    for arm in (straight, resumed):
        assert not arm.root.state.existing_scratch(
            uh_diag.UH_SPAWN_WINDOW_SLOT).any()


def test_a_resume_that_does_not_replay_the_boundary_never_births_the_nest(
        monkeypatch):
    """Non-vacuity, stated as the defect: without the replayed pass the
    slot's one opportunity is gone and the run finishes root-only."""
    from gpuwm.runtime import walk_spawn_legs

    steps = []
    dexp, model, runner, coupler = _leg_walk_model(monkeypatch, steps)
    _carry_to(model, 120.0)
    model._resumed = False          # the predicate's resume condition

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=coupler, validate_state=False)

    assert sorted(model.nodes_by_grid_id) == [1]
    assert runner.spawns_executed == 0
    # Nothing looked, so nothing was spent: the boundary really did not run.
    assert model.root.state.existing_scratch(
        uh_diag.UH_SPAWN_WINDOW_SLOT).any()


def test_the_resume_boundary_predicate_names_its_four_conditions(monkeypatch):
    """Each condition, dropped, makes the replay wrong in its own way."""
    from gpuwm.runtime import resume_boundary_due

    _dexp, model, runner, _coupler = _leg_walk_model(monkeypatch, [])
    _carry_to(model, 120.0)
    assert resume_boundary_due(model, runner, leg=120.0, total=240.0)

    model._resumed = False
    assert not resume_boundary_due(model, runner, leg=120.0, total=240.0)
    model._resumed = True

    # Off the lattice: the boundary this run stops on is not here.
    _carry_to(model, 60.0)
    assert not resume_boundary_due(model, runner, leg=120.0, total=240.0)

    # t = 0 is on the lattice, and the straight run does not evaluate it.
    _carry_to(model, 0.0)
    assert not resume_boundary_due(model, runner, leg=120.0, total=240.0)

    # The run's last boundary was taken by the run that finished.
    _carry_to(model, 240.0)
    assert not resume_boundary_due(model, runner, leg=120.0, total=240.0)

    # A runner with nothing left to decide asks nothing.
    _carry_to(model, 120.0)
    runner.controller.watches[2].fired = True
    assert not runner.needs_boundaries
    assert not resume_boundary_due(model, runner, leg=120.0, total=240.0)


def test_a_lifecycle_free_resume_leaves_the_fingerprint_alone():
    """No moves, no mark: the byte-inert half of the re-mark."""
    from gpuwm.runtime import remark_relocation_fingerprint

    case = _restore_case()
    case.model.experiment_fingerprint = "base"
    assert remark_relocation_fingerprint(
        case.model, _peek(_spawn_block())) == "base"
    assert case.model.experiment_fingerprint == "base"
