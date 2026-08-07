"""Seam A: SpawnRunner activating a dormant nest inside a REAL runner walk.

The lane left the pieces proven in isolation -- the controller's trigger
arithmetic (tests/test_nest_spawn.py), the materializer's terrain
calibration (tests/test_nest_spawn_init.py) -- plus one hand-written
end-to-end that performed the leg surgery inline.  This file covers the
seam that replaces that inline sequence: the runner that a route mounts
once and consults at every leg boundary.

Everything below walks REAL clocks, a REAL schedule and the REAL
executor; only ``gpuwm.core.dycore.step`` is mocked, because NumPy
states are not valid CUDA integrator inputs -- the same instrument rule
the spawn lane's own end-to-end states.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpuwm.core.nest_spawn import SpawnConfig
from gpuwm.core.spawn_runner import (SPAWN_RUNNER_CONTRACT, SpawnRunner,
                                     SpawnRunnerRefusal)
from gpuwm.static.lambert import grids_from_projection_config
from test_nest_spawn_init import _experiment, _host, _live_parent

LEG = 120.0          # 2 parent steps at dt = 60 s; 6 child steps at ratio 3


@pytest.fixture()
def case():
    """A fresh two-domain micro case whose d02 is DORMANT on a time trigger."""
    exp = _experiment()
    parent, grids = _live_parent(exp)
    dormant = replace(exp.domains[1],
                      spawn=SpawnConfig(trigger="time", at_s=LEG))
    return {"exp": replace(exp, domains=(exp.domains[0], dormant)),
            "parent": parent, "grids": grids}


class _CountingCoupler:
    """The one-way coupler surface the executor drives, counted."""

    def __init__(self, child, feedback=0):
        self.child = child
        self.valid = False
        self.forces = 0

    def force(self, node):
        assert node is self.child
        self.valid = True
        self.forces += 1

    def feedback_prepare(self, node, out):
        out.payload = None

    def feedback_commit(self, node):
        pass

    def feedback_finalize(self, node):
        pass


def _leg(exp, seconds=LEG):
    """Trim an experiment view to one leg's length."""
    return replace(exp, run_seconds=seconds, domains=tuple(
        replace(dc, history_interval_s=seconds,
                run=replace(dc.run, run_seconds=seconds,
                            output_interval_s=seconds))
        for dc in exp.domains))


def _walk(exp, root_state, *, grids=None, child_initializer=None,
          coupler_factory=_CountingCoupler):
    """Assemble and integrate one leg; return the model and its report."""
    from gpuwm.core.model import execute_experiment
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree

    kwargs = {}
    if child_initializer is not None:
        kwargs["child_initializer"] = child_initializer
    model = assemble_idealized_tree(
        exp, root_state,
        grids=(grids_from_projection_config(exp) if grids is None else grids),
        coupler_factory=coupler_factory,
        domain_preparer=lambda *_a, **_k: None, **kwargs)
    return model, execute_experiment(model, validate_state=False)


# ---------------------------------------------------------------------------
# The headline: a dormant nest is born mid-walk and integrates in leg B
# ---------------------------------------------------------------------------

def test_spawn_fires_inside_a_real_two_leg_walk(case, monkeypatch):
    """Leg A integrates the root alone; the trigger fires at the boundary;
    leg B integrates the newborn, adopted through the runner's own
    initializer, with its node, clock and coupler attached by the
    ordinary leg-boundary rebuild."""
    stepped: list[int] = []
    monkeypatch.setattr("gpuwm.core.dycore.step",
                        lambda state, cfg, **_k: stepped.append(cfg.grid_id))

    built: list[tuple] = []

    def on_child_built(initialized, child_dc, parent_node):
        built.append((int(child_dc.grid_id), initialized, parent_node))

    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=on_child_built, array_module=np)
    assert runner is not None and runner.pending == (2,)

    # ---- leg A: the dormant nest costs zero compute ---------------------
    pre = runner.active
    assert [dc.grid_id for dc in pre.domains] == [1]
    model_a, report_a = _walk(
        _leg(pre), case["parent"].state, grids=(case["grids"][0],))
    assert report_a.steps == 2 and stepped == [1, 1]

    # ---- the leg boundary: one call, the whole activation ---------------
    record = runner.on_leg_boundary(model_a)
    assert record is not None
    assert record["contract"] == SPAWN_RUNNER_CONTRACT
    assert record["event"] == "spawned"
    assert record["grid_ids"] == [2]
    assert record["active_grid_ids"] == [1, 2]
    assert record["pending_after"] == []
    assert runner.spawns_executed == 1
    assert runner.spawned == {2: (20, 18)}

    # The physics/land seam fired, with the newborn's own config.
    assert [gid for gid, _i, _p in built] == [2]
    assert built[0][2] is model_a.root

    # The birth certificate rode along, and the parent was not written.
    born = record["born"][0]
    assert born["placement"] == [20, 18]
    assert born["spawn_receipt"]["parent_bitwise_unchanged"] is True
    assert born["spawn_receipt"]["trigger"]["decision"] == "fired"
    assert born["spawn_receipt"]["atmosphere_source"]["kind"] == "parent-sint"
    assert "child_result" not in born["spawn_receipt"]

    # ---- leg B: the activated tree --------------------------------------
    act = record["experiment"]
    assert [dc.grid_id for dc in act.domains] == [1, 2]
    stepped.clear()
    model_b, report_b = _walk(
        _leg(act), model_a.root.state,
        child_initializer=record["child_initializer"])

    child = model_b.node(2)
    assert child.parent is model_b.root
    assert child.state is built[0][1].state      # the newborn, not a rebuild
    assert child.clock is not None               # clock attached by rebuild
    assert child.coupler is not None             # coupler attached too
    assert report_b.steps == 2 + 6
    assert stepped.count(1) == 2 and stepped.count(2) == 6
    assert child.coupler.forces == 2             # one FORCE per parent step
    for name in ("thp", "mup", "u", "v", "w"):
        assert np.isfinite(_host(getattr(child.state, name))).all()


def test_a_second_boundary_after_the_fire_is_a_no_op(case, monkeypatch):
    """Once every watch has fired the runner stops doing work, and the
    active view keeps the newborn at its fired placement."""
    monkeypatch.setattr("gpuwm.core.dycore.step",
                        lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model_a, _ = _walk(_leg(runner.active), case["parent"].state,
                       grids=(case["grids"][0],))
    assert runner.on_leg_boundary(model_a) is not None

    act = runner.active
    model_b, _ = _walk(_leg(act), model_a.root.state,
                       child_initializer=runner.child_initializer())
    assert runner.pending == ()
    assert runner.on_leg_boundary(model_b) is None
    assert runner.spawns_executed == 1
    assert [dc.grid_id for dc in runner.active.domains] == [1, 2]


# ---------------------------------------------------------------------------
# Holding, and the closing account
# ---------------------------------------------------------------------------

def test_a_boundary_before_the_window_holds_and_says_so(case, monkeypatch):
    monkeypatch.setattr("gpuwm.core.dycore.step",
                        lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    # Half a leg: the manual trigger's instant has not arrived.
    model, _ = _walk(_leg(runner.active, 60.0), case["parent"].state,
                     grids=(case["grids"][0],))
    assert runner.on_leg_boundary(model) is None
    assert runner.spawns_executed == 0
    held = runner.receipts[-1]
    assert held["event"] == "held" and held["pending"] == [2]
    assert held["elapsed_seconds"] == 60.0
    # Nothing joined the tree.
    assert [dc.grid_id for dc in runner.active.domains] == [1]


def test_close_receipt_names_the_watch_that_never_fired(case):
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    entry = runner.close_receipt()
    assert entry["event"] == "closed"
    assert entry["spawns_executed"] == 0
    assert [row["grid_id"] for row in entry["never_fired"]] == [2]
    assert "reservation was held" in entry["never_fired"][0]["note"] or \
        "still open" in entry["never_fired"][0]["note"]


def test_receipts_are_written_when_a_path_is_given(case, tmp_path,
                                                   monkeypatch):
    import json

    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *_a, **_k: None)
    path = tmp_path / "spawn-receipts.json"
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np,
        receipts_path=path)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    runner.on_leg_boundary(model)
    runner.close_receipt()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["contract"] == SPAWN_RUNNER_CONTRACT
    events = [row["event"] for row in payload["receipts"]]
    assert events == ["spawned", "closed"]


# ---------------------------------------------------------------------------
# The wiring itself: what the runner hands the controller
# ---------------------------------------------------------------------------

class _Node:
    def __init__(self, cfg, state, parent=None):
        self.cfg = cfg
        self.state = state
        self.grid = None
        self.parent = parent
        self._started = True


class _Model:
    """The three surfaces the runner reads off a model."""

    def __init__(self, nodes):
        self._nodes = {int(n.cfg.grid_id): n for n in nodes}
        self.root = nodes[0]

    def walk_parent_first(self):
        return [self._nodes[gid] for gid in sorted(self._nodes)]

    def node(self, gid):
        return self._nodes[int(gid)]


def test_live_child_footprints_are_handed_to_the_controller(case):
    """The exclusion rule only works if the runner actually reports the
    LIVE children; this pins the argument, which is the seam's job (the
    exclusion arithmetic itself is the controller's, proven upstream)."""
    exp = case["exp"]
    live_child = replace(exp.domains[1], spawn=None, grid_id=3,
                         run=replace(exp.domains[1].run, grid_id=3))
    with_live = replace(exp, domains=(*exp.domains, live_child))

    runner = SpawnRunner.from_experiment(
        with_live, on_child_built=lambda *_a: None, array_module=np)
    seen = {}

    def spy(parent_states, t, *, active_footprints=()):
        seen["parents"] = dict(parent_states)
        seen["footprints"] = tuple(active_footprints)
        seen["t"] = t
        return ()

    runner.controller.evaluate_all = spy
    root_node = _Node(exp.domains[0], case["parent"].state)
    child_node = _Node(live_child, object(), parent=root_node)
    assert runner.on_leg_boundary(_Model([root_node, child_node]),
                                  t=300.0) is None

    assert seen["t"] == 300.0
    assert sorted(seen["parents"]) == [1, 3]
    assert [fp.grid_id for fp in seen["footprints"]] == [3]
    fp = seen["footprints"][0]
    assert (fp.i_parent_start, fp.j_parent_start) == (
        live_child.i_parent_start, live_child.j_parent_start)


def test_the_materialized_config_is_the_adjudicated_one(case, monkeypatch):
    """A field trigger's placement is nothing like the declared
    placeholder, so the runner must materialize the config
    ``active_experiment`` produced -- not ``exp.domain(gid)``."""
    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *_a, **_k: None)
    seen = {}

    real = None

    def spy(child_dc, parent_node, **kwargs):
        seen["placement"] = (int(child_dc.i_parent_start),
                             int(child_dc.j_parent_start))
        return real(child_dc, parent_node, **kwargs)

    import gpuwm.ingest.nest_spawn_init as init_mod
    real = init_mod.spawn_child_from_parent
    monkeypatch.setattr(init_mod, "spawn_child_from_parent", spy)

    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    record = runner.on_leg_boundary(model)

    fired = tuple(runner.spawned[2])
    assert seen["placement"] == fired
    activated = record["experiment"].domain(2)
    assert (activated.i_parent_start, activated.j_parent_start) == fired


# ---------------------------------------------------------------------------
# Refusals: the two things the mechanism must not invent
# ---------------------------------------------------------------------------

def test_no_dormant_nest_means_no_runner():
    exp = _experiment()
    assert SpawnRunner.from_experiment(
        exp, on_child_built=lambda *_a: None) is None
    with pytest.raises(SpawnRunnerRefusal, match="declares no dormant"):
        SpawnRunner(experiment=exp, on_child_built=lambda *_a: None)


def test_a_runner_without_the_physics_preparer_refuses(case):
    with pytest.raises(SpawnRunnerRefusal, match="on_child_built"):
        SpawnRunner(experiment=case["exp"], on_child_built=None)
    with pytest.raises(SpawnRunnerRefusal, match="on_child_built"):
        SpawnRunner.from_experiment(case["exp"], on_child_built=None)


# ---------------------------------------------------------------------------
# The run route's leg walk: the production consumer of on_leg_boundary
# ---------------------------------------------------------------------------

class _Prepared:
    """The prepared-case surface the attach step reads."""

    def __init__(self):
        self.static_fields = None
        self.geog_selection = None
        self.initial_result = None


class _CountingPreparer:
    """A route preparer: attaches nothing, records everything."""

    def __init__(self):
        self.prepared_by_grid_id = {}
        self.calls = []

    def __call__(self, initialized, child_dc, parent_node):
        gid = int(child_dc.grid_id)
        self.calls.append((gid, int(parent_node.cfg.grid_id)))
        self.prepared_by_grid_id[gid] = _Prepared()


def _leg_walk_case():
    """A micro case whose leg cadence (120 s) straddles the trigger."""
    exp = _experiment()
    root = replace(exp.domains[0], history_interval_s=120.0,
                   run=replace(exp.domains[0].run, run_seconds=240.0,
                               output_interval_s=120.0))
    child = replace(exp.domains[1], history_interval_s=120.0,
                    spawn=SpawnConfig(trigger="time", at_s=120.0),
                    run=replace(exp.domains[1].run, run_seconds=240.0,
                                output_interval_s=120.0))
    return replace(exp, run_seconds=240.0, domains=(root, child))


def test_walk_spawn_legs_births_the_nest_inside_one_run(monkeypatch):
    """The whole product story on the real executor: a dormant nest costs
    zero compute across the first leg, is born at the 120 s boundary from
    the LIVE parent, and integrates its own substeps for the rest of the
    run -- one `walk_spawn_legs` call, no caller-side leg bookkeeping."""
    from gpuwm.runtime import walk_spawn_legs
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree

    dexp = _leg_walk_case()
    parent, grids = _live_parent(dexp)

    stepped: list[int] = []
    monkeypatch.setattr("gpuwm.core.dycore.step",
                        lambda state, cfg, **_k: stepped.append(cfg.grid_id))

    preparer = _CountingPreparer()
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=preparer, array_module=np)
    assert runner.pending == (2,)

    # The tree the run starts with: the dormant nest is absent.
    pre = runner.active
    assert [dc.grid_id for dc in pre.domains] == [1]
    model = assemble_idealized_tree(
        pre, parent.state, grids=(grids[0],),
        coupler_factory=_CountingCoupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    validate_state=False)

    # The nest was born, once, from the live parent.
    assert runner.spawns_executed == 1
    assert preparer.calls == [(2, 1)]
    assert sorted(model.nodes_by_grid_id) == [1, 2]

    child = model.node(2)
    assert child.parent is model.root
    assert child in model.root.children
    assert child.coupler is not None and child._started

    # Zero compute while dormant, full cadence afterwards: the parent ran
    # 4 steps (2 per leg); the child ran 6 -- 3 per parent step, but only
    # across the SECOND leg.
    assert stepped.count(1) == 4
    assert stepped.count(2) == 6
    assert child.coupler.forces == 2

    # Both clocks land on the run end together.
    assert float(model.root.clock.elapsed_seconds) == 240.0
    assert float(child.clock.elapsed_seconds) == 240.0
    for name in ("thp", "mup", "u", "v", "w"):
        assert np.isfinite(_host(getattr(child.state, name))).all()

    # The closing account is on the model.
    assert model._spawn_receipts[-1]["event"] == "closed"
    assert model._spawn_receipts[-1]["spawns_executed"] == 1


def test_walk_spawn_legs_runs_straight_through_when_nothing_fires(
        monkeypatch):
    """A window that never opens costs its reservation and nothing else:
    the walk still reaches the end, the tree stays root-only, and the
    closing receipt names the watch that never fired."""
    from gpuwm.runtime import walk_spawn_legs
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree

    exp = _leg_walk_case()
    # A trigger instant beyond the run: the watch can never fire.
    dexp = replace(exp, domains=(
        exp.domains[0],
        replace(exp.domains[1], spawn=SpawnConfig(trigger="time",
                                                  at_s=100000.0))))
    parent, grids = _live_parent(dexp)
    stepped: list[int] = []
    monkeypatch.setattr("gpuwm.core.dycore.step",
                        lambda state, cfg, **_k: stepped.append(cfg.grid_id))

    preparer = _CountingPreparer()
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=preparer, array_module=np)
    model = assemble_idealized_tree(
        runner.active, parent.state, grids=(grids[0],),
        coupler_factory=_CountingCoupler,
        domain_preparer=lambda *_a, **_k: None)
    model._prepared_by_grid_id = {1: _Prepared()}

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    validate_state=False)

    assert runner.spawns_executed == 0
    assert preparer.calls == []
    assert sorted(model.nodes_by_grid_id) == [1]
    assert stepped.count(1) == 4 and stepped.count(2) == 0
    assert float(model.root.clock.elapsed_seconds) == 240.0
    closed = model._spawn_receipts[-1]
    assert closed["event"] == "closed"
    assert [row["grid_id"] for row in closed["never_fired"]] == [2]


def test_the_recorded_receipts_stay_json_serialisable(case, monkeypatch,
                                                      tmp_path):
    """self.receipts is durable evidence; the live objects the driver
    needs must ride on the RETURNED record only.  Caught by the
    idealized demo, whose receipt writer has no default=str."""
    import json

    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    record = runner.on_leg_boundary(model)
    assert record is not None
    # The driver's half: live objects present.
    assert "experiment" in record and "child_results" in record
    assert callable(record["child_initializer"])
    # The durable half: no live objects, and strictly serialisable.
    for row in runner.receipts:
        assert "experiment" not in row
        assert "child_initializer" not in row
        assert "child_results" not in row
    json.dumps(runner.receipts)      # raises if anything live leaked in
