"""A dormant nest's watch and its re-arm belong to the grid it is nested on.

Two defects shared one cause: spawn state was kept at a coarser key than
the thing it belongs to.

- Footprints.  A live nest's footprint is cell numbers on its PARENT's
  grid.  The controller handed every footprint to every watch, so d02's
  placement on d01 (cells 30..50 of d01) masked cells 30..50 of d02, and a
  storm there never fired the dormant d03 nested on d02: it stayed dormant
  with a ``no-signal`` receipt.  The same held for a nest born on d01 at
  the same boundary.
- Re-arming.  A retired slot re-armed only when its parent was the root
  or a SPAWNED episode, so a nest below a permanent d02 retired once and
  never came back, whatever its cooldown and firing budget said.

The boundary decisions here run through the real ``SpawnRunner`` and
``SpawnController`` on configs the real loader accepts; only the newborn's
materialization is stubbed (``_stub_materializer``), as in
tests/test_spawn_runner.py.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpuwm.core.nest_spawn import SpawnConfig, SpawnController, SpawnRefusal
from gpuwm.core.spawn_runner import SpawnRunner
from gpuwm.core.state import DomainState
from gpuwm.core.storm_tracking import NestFootprint
from gpuwm.experiment import load_experiment
from test_spawn_runner import _Model, _Node, _stub_materializer

HEAD = """[experiment]
name = "parent-grids"
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
nx = 120
ny = 100
time_step = 60
dx = 12000.0
history_interval_s = 3600.0
"""


def _domain(grid_id, parent_id, i, j, n, extra=""):
    return f"""[[domain]]
grid_id = {grid_id}
parent_id = {parent_id}
i_parent_start = {i}
j_parent_start = {j}
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = {n}
ny = {n}
history_interval_s = 3600.0
{extra}"""


REFL = 'spawn = { trigger = "reflectivity", threshold = 40.0, earliest_s = 0.0, latest_s = 1800.0 }\n'


def _load(tmp_path, *domains):
    path = tmp_path / "exp.toml"
    path.write_text(HEAD + "".join(domains), encoding="utf-8")
    return load_experiment(path)


def _state(exp, grid_id, storm_at=None):
    """A live state whose reflectivity plane holds one 60 dBZ cell."""
    run = exp.domain(grid_id).run
    state = DomainState(run, array_module=np)
    plane = np.zeros((int(run.ny), int(run.nx)), dtype=np.float32)
    if storm_at is not None:
        plane[storm_at] = 60.0
    state.scratch(plane.shape, "refl_10cm")[...] = plane
    return state


def _runner(exp):
    return SpawnRunner.from_experiment(
        exp, on_child_built=lambda *_a: None, array_module=np)


# ---------------------------------------------------------------------------
# A footprint masks only the grid it is counted on
# ---------------------------------------------------------------------------

def test_a_permanent_d02_does_not_hide_the_storm_its_own_dormant_d03_watches(
        tmp_path, monkeypatch):
    """The review's three-domain case: 60 dBZ at cell (35, 35) of d02.

    d02 sits on d01 at (30, 30), so its footprint covers cells 29..50 of
    d01.  Applied to d02's own plane it erased the storm and d03 held
    with ``no-signal``; counted on the grid it belongs to, d03 fires on
    the storm at (31, 31).
    """
    exp = _load(tmp_path, _domain(2, 1, 30, 30, 60),
                _domain(3, 2, 21, 21, 30, REFL))
    root = _Node(exp.domain(1), _state(exp, 1))
    d02 = _Node(exp.domain(2), _state(exp, 2, storm_at=(35, 35)), parent=root)
    runner = _runner(exp)
    _stub_materializer(monkeypatch)

    record = runner.on_leg_boundary(_Model([root, d02]), t=120.0)

    assert record is not None and record["grid_ids"] == [3]
    assert record["born"][0]["placement"] == [31, 31]
    assert runner.spawned == {3: (31, 31)}
    fired = [row for row in record["watch_receipts"]
             if row.get("decision") == "fired"]
    assert fired and fired[0]["excluded_active_footprints"] == []


def test_a_nest_on_the_same_grid_still_owns_its_storm(tmp_path):
    """The control: a live nest ON d02 over the storm keeps d03 off it."""
    exp = _load(tmp_path, _domain(2, 1, 30, 30, 60),
                _domain(3, 2, 21, 21, 30, REFL))
    d02 = _state(exp, 2, storm_at=(35, 35))
    sibling = NestFootprint(grid_id=4, i_parent_start=30, j_parent_start=30,
                            child_nx=31, child_ny=31, parent_grid_ratio=3,
                            parent_id=2)
    elsewhere = NestFootprint.coerce(exp.domain(2))      # counted on d01
    assert elsewhere.parent_id == 1

    held = SpawnController.from_experiment(exp)
    assert held.evaluate_all({1: _state(exp, 1), 2: d02}, 120.0,
                             active_footprints=(elsewhere, sibling)) == ()
    receipt = held.watches[3].receipts[-1]
    assert receipt["decision"] == "no-signal"
    assert [row["grid_id"] for row in receipt["excluded_active_footprints"]] == [4]

    fired = SpawnController.from_experiment(exp)
    events = fired.evaluate_all({1: _state(exp, 1), 2: d02}, 120.0,
                                active_footprints=(elsewhere,))
    assert [(e.grid_id, e.position) for e in events] == [(3, (31, 31))]


def test_a_nest_born_on_d01_at_this_boundary_does_not_mask_a_watch_on_d03(
        tmp_path, monkeypatch):
    """Same-boundary births are counted on their own parent grid too.

    d02 is dormant on d01 and d04 is dormant on the permanent d03; both
    grids hold a storm at cell (35, 35).  d02 fires first (grid_id order)
    and its new footprint covers cells ~25..46 of d01.  Handed to d04 it
    erased d04's storm on d03; counted on d01 it leaves d03 alone.
    """
    exp = _load(tmp_path, _domain(2, 1, 30, 30, 60, REFL),
                _domain(3, 1, 80, 40, 60),
                _domain(4, 3, 21, 21, 30, REFL))
    root = _Node(exp.domain(1), _state(exp, 1, storm_at=(35, 35)))
    d03 = _Node(exp.domain(3), _state(exp, 3, storm_at=(35, 35)), parent=root)
    runner = _runner(exp)
    _stub_materializer(monkeypatch)

    record = runner.on_leg_boundary(_Model([root, d03]), t=120.0)

    assert record is not None and record["grid_ids"] == [2, 4]
    placements = {row["grid_id"]: row["placement"] for row in record["born"]}
    assert placements[4] == [31, 31]
    # d02's birth still joins the exclusion set of later watches on d01.
    d02_footprint = NestFootprint(
        grid_id=2, i_parent_start=placements[2][0],
        j_parent_start=placements[2][1], child_nx=60, child_ny=60,
        parent_grid_ratio=3, parent_id=1)
    assert d02_footprint.i_parent_start < 36 < (
        d02_footprint.i_parent_start + d02_footprint.span_parent_i)


def test_a_footprint_that_names_no_grid_is_refused_when_watches_span_grids(
        tmp_path):
    """Unlabelled cells cannot be placed on one of two grids; guessing all
    of them is exactly how the storm on d02 was hidden."""
    exp = _load(tmp_path, _domain(2, 1, 30, 30, 60, REFL),
                _domain(3, 1, 80, 40, 60),
                _domain(4, 3, 21, 21, 30, REFL))
    bare = NestFootprint(grid_id=3, i_parent_start=80, j_parent_start=40,
                         child_nx=60, child_ny=60, parent_grid_ratio=3)
    controller = SpawnController.from_experiment(exp)
    with pytest.raises(SpawnRefusal, match="which grid its cells"):
        controller.evaluate_all(
            {1: _state(exp, 1), 3: _state(exp, 3)}, 120.0,
            active_footprints=(bare,))


# ---------------------------------------------------------------------------
# A nest below a permanent intermediate domain re-arms
# ---------------------------------------------------------------------------

TIME_EPISODES = ('spawn = { trigger = "time", at_s = 120.0 }\n'
                 'retire = { trigger = "time", at_s = 120.0 }\n'
                 'rearm = { max_firings = 2, cooldown_s = 120.0 }\n')


def _boundary(runner, model, t, parent_of, exp):
    """One leg boundary, then the tree the next leg integrates."""
    record = runner.on_leg_boundary(model, t=t)
    for gid in sorted(runner.spawned):
        if gid not in model.nodes_by_grid_id:
            parent = model.node(parent_of[gid])
            model.nodes_by_grid_id[gid] = _Node(
                runner.active.domain(gid),
                DomainState(exp.domain(gid).run, array_module=np),
                parent=parent)
    for gid in list(model.nodes_by_grid_id):
        if gid in runner.retired:
            del model.nodes_by_grid_id[gid]
    return record


def test_a_nest_below_a_permanent_d02_gets_its_second_episode(
        tmp_path, monkeypatch):
    exp = _load(tmp_path, _domain(2, 1, 30, 30, 60),
                _domain(3, 2, 21, 21, 30, TIME_EPISODES))
    root = _Node(exp.domain(1), _state(exp, 1))
    d02 = _Node(exp.domain(2), _state(exp, 2), parent=root)
    model = _Model([root, d02])
    runner = _runner(exp)
    _stub_materializer(monkeypatch)
    parent_of = {3: 2}

    assert _boundary(runner, model, 120.0, parent_of, exp)["grid_ids"] == [3]
    assert _boundary(runner, model, 240.0, parent_of, exp)[
        "retired_grid_ids"] == [3]
    # The cooldown is over at 360: re-armed and fired as episode 2.
    record = _boundary(runner, model, 360.0, parent_of, exp)
    assert record["rearmed_grid_ids"] == [3]
    assert record["grid_ids"] == [3]
    assert runner.episodes[3] == 2
    assert _boundary(runner, model, 480.0, parent_of, exp)[
        "retired_grid_ids"] == [3]
    # max_firings = 2: spent, whatever the clock says.
    for t in (600.0, 720.0, 1800.0):
        assert _boundary(runner, model, t, parent_of, exp) is None
    assert runner.episodes[3] == 2


def test_a_nest_below_a_spawned_parent_still_waits_for_its_parent_episode(
        tmp_path, monkeypatch):
    """The guard the old check stood for is kept: a slot re-arms only
    while its parent is live, and a retired parent slot is not.

    The loader refuses a nest under a dormant one, so this ladder is
    assembled the way the runtime receives it: d02's lifecycle, as the
    loader parsed it for a leaf d02, placed on the d02 of the three-level
    tree.
    """
    parent_episodes = ('spawn = { trigger = "time", at_s = 120.0 }\n'
                       'retire = { trigger = "time", at_s = 240.0 }\n'
                       'rearm = { max_firings = 2, cooldown_s = 120.0 }\n')
    (tmp_path / "leaf").mkdir()
    leaf = _load(tmp_path / "leaf", _domain(2, 1, 30, 30, 60, parent_episodes))
    exp = _load(tmp_path, _domain(2, 1, 30, 30, 60),
                _domain(3, 2, 21, 21, 30, TIME_EPISODES))
    lifecycle = leaf.domain(2)
    exp = replace(exp, domains=(
        exp.domain(1),
        replace(exp.domain(2), spawn=lifecycle.spawn,
                retire=lifecycle.retire, rearm=lifecycle.rearm),
        exp.domain(3)))
    root = _Node(exp.domain(1), _state(exp, 1))
    model = _Model([root])
    runner = _runner(exp)
    _stub_materializer(monkeypatch)
    parent_of = {2: 1, 3: 2}

    assert _boundary(runner, model, 120.0, parent_of, exp)["grid_ids"] == [2]
    assert _boundary(runner, model, 240.0, parent_of, exp)["grid_ids"] == [3]
    # d02 (born 120) and d03 (born 240) both retire at 360.
    assert _boundary(runner, model, 360.0, parent_of, exp)[
        "retired_grid_ids"] == [2, 3]
    # 480: both cooldowns are over, but only d02's parent is live.
    record = _boundary(runner, model, 480.0, parent_of, exp)
    assert record["rearmed_grid_ids"] == [2] and record["grid_ids"] == [2]
    assert 3 in runner.retired
    # 600: d02 is live again, so d03 re-arms and fires its second episode.
    record = _boundary(runner, model, 600.0, parent_of, exp)
    assert record["rearmed_grid_ids"] == [3] and record["grid_ids"] == [3]
    assert runner.episodes == {2: 2, 3: 2}
