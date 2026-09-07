"""CPU controls for live tile operands and the acoustic planning envelope."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.adaptive_clock import (
    AdaptiveClockDriver, acoustic_step_ceiling, maximum_map_factor,
)
from gpuwm.core.physics_step_control import PhysicsStepControl
from gpuwm.core import streaming


def config(**changes):
    return RunConfig(nx=192, ny=160, nz=12, dx=1000., dy=1000.,
                     ztop=8000., dt=1., run_seconds=30., **changes)


@pytest.mark.parametrize("due", [True, False, None])
def test_control_replaces_each_tile_cadence_including_cleared_override(due):
    source = NS(stepra=7, stepcu=5, stepbl=3, radt_seconds=15.,
                cudt_seconds=12., bldt_seconds=9.,
                radiation_due_override=due, cumulus_due_override=due)
    control = PhysicsStepControl.from_driver(source)
    for old in (True, False, None):
        tile = NS(physics=NS(radiation_due_override=old, cumulus_due_override=old))
        control.apply(tile)
        assert vars(tile.physics) == vars(source)
    source.radt_seconds = 1000.
    assert dict(control.values)["radt_seconds"] == 15.


def test_dry_control_does_not_create_a_physics_driver():
    assert PhysicsStepControl.from_driver(None) is None


def test_live_carrier_records_override_stale_resident_producer_stamp():
    driver = object.__new__(AdaptiveClockDriver)
    driver._radiation_seen, driver._radiation_actual = {}, {}
    records = {"swdown": {"last_update_model_time": 10.}}
    driver.carrier_source = lambda gid: {"records": records}
    node = NS(state=NS(physics=NS(carriers=NS(records={
        "swdown": NS(last_update_model_time=0.)}))))
    assert driver._observe_radiation(1, node) is None
    records["swdown"]["last_update_model_time"] = 22.
    assert driver._observe_radiation(1, node) == 12.
    assert driver._radiation_seen[1] == 22.


def test_full_store_geometry_is_used_instead_of_template_row():
    slab = NS(msfu=np.ones((1, 4)), msfv=np.ones((2, 3)))
    geography = {"setup/msfu": np.array([[1., 1.6]]),
                 "setup/msfv": np.array([[1.2]])}
    assert maximum_map_factor(slab, geography) == 1.6


def test_acoustic_envelope_preserves_config_and_explicit_fractional_maximum():
    cfg = config(use_adaptive_time_step=True, max_time_step=15,
                 max_time_step_den=2)
    # dt=7.5, map factor=1.6: 300*7.5*1.6/1000=3.6 -> 8 sound steps.
    assert acoustic_step_ceiling(cfg, 1.6) == 8
    assert acoustic_step_ceiling(cfg, 1.) == 6
    assert cfg.time_step_sound == 4 and cfg.dt == 1.
    assert acoustic_step_ceiling(replace(cfg, use_adaptive_time_step=False), 9.) == 4


def test_pinned_decision_reserves_actual_static_acoustic_reach_without_gpu(monkeypatch):
    cfg = config(use_adaptive_time_step=True, max_time_step=15,
                 max_time_step_den=2)
    options = streaming.StreamingOptions(mode="on", tile_nx=24, tile_ny=20)
    options = streaming._options_with_map_factor(
        options, NS(msfu=np.array([1.6]), msfv=np.ones(2)), cfg)
    decision = streaming.decide(cfg, options)
    assert decision.halo == 22
    assert decision.tile_nx == 24 and decision.tile_ny == 20


def test_planner_prices_the_same_enlarged_window_as_the_decision(monkeypatch):
    from tilestream import autoplan
    cfg = config(use_adaptive_time_step=True, max_time_step=15,
                 max_time_step_den=2)
    options = streaming.StreamingOptions(mode="on", acoustic_map_factor=1.6)
    machine = autoplan.Machine(vram_bytes=8 * 1024**3, host_bytes=32 * 1024**3)
    planned = []
    original = autoplan.plan
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        planned.append(result)
        return result
    monkeypatch.setattr(autoplan, "plan", record)
    decision = streaming.decide(cfg, options, machine=machine)
    assert decision.halo == planned[0].halo == 22
    assert decision.resident_bytes == int(planned[0].vram_bytes)


def test_changed_live_config_invalidates_graphs_before_the_next_capture():
    from tilestream.graphcap import GraphStepper
    cfg = config()
    graph = GraphStepper(cfg)
    graph.graphs[("old",)] = object()
    graph._verified.add(("old",))
    graph._uncapturable.add(("old",))
    graph.reason = "previous topology"
    graph.set_config(replace(cfg, dt=2., time_step_sound=6))
    assert not graph.graphs and not graph._verified and not graph._uncapturable
    assert graph.cfg.dt == 2. and graph.reason is None


@pytest.mark.parametrize("adaptive", [False, True])
def test_prepared_tree_first_plan_uses_complete_static_map_factors(adaptive, monkeypatch):
    from gpuwm.prepared_domain_tree_forecast import _prepared_planning_nodes
    from tilestream import autoplan
    from tilestream.harness import halo_radius

    cfg = RunConfig(nx=192, ny=160, nz=12, dx=3000., dy=3000.,
        ztop=8000., dt=18., run_seconds=36., use_adaptive_time_step=adaptive,
        starting_time_step=18, min_time_step=9, max_time_step=18)
    options = streaming.StreamingOptions(mode="on", tile_nx=48, tile_ny=40)
    root = NS(grid_id=1, parent_id=0, run=cfg, tiles=None)
    child = NS(grid_id=2, parent_id=1, parent_grid_ratio=3,
        run=replace(cfg, grid_id=2), tiles=None)
    u = np.ones((cfg.ny, cfg.nx + 1), dtype=np.float64)
    v = np.ones((cfg.ny + 1, cfg.nx), dtype=np.float64)
    # A maximum outside the first restore slab must price the full store.
    v[-2, -2] = 1.6
    bundles = tuple(NS(grid_id=gid, static_fields={"MAPFAC_U": u, "MAPFAC_V": v})
                    for gid in (2, 1))  # Match by grid id, not incidental order.
    inputs = NS(experiment=NS(domains=(root, child), tiles=options), domains=bundles)
    monkeypatch.setattr(autoplan.Machine, "detect", lambda **kw: pytest.fail("pinned planning probed GPU"))
    nodes = _prepared_planning_nodes(inputs)
    decisions = {}
    streaming.decide_tree(nodes, options, decisions=decisions)
    # Match the float32 static fields that DomainState.set_map_coriolis installs.
    live_factor = maximum_map_factor(NS(msfu=u.astype(np.float32), msfv=v.astype(np.float32)))
    required = halo_radius(replace(cfg, time_step_sound=acoustic_step_ceiling(cfg, live_factor)))
    assert required == (19 if adaptive else 16)
    assert [node.cfg.grid_id for node in nodes] == [1, 2]
    assert nodes[1].parent is nodes[0]
    for decision in decisions.values():
        assert decision.halo == required
        assert (decision.tile_nx, decision.tile_ny) == (48, 40)
    assert cfg.time_step_sound == 4 and cfg.dt == 18.


def test_graph_sweep_expiry_keeps_workspace_reuse_bounded():
    from tilestream.graphcap import GraphStepper
    graph = GraphStepper(config())
    graph._verified.add(("stable",))
    graph.graphs[("old",)] = object()
    graph.set_sweep(1)
    assert not graph.graphs and graph._verified == {("stable",)}
    graph.graphs[("current",)] = object()
    graph.set_sweep(1)
    assert len(graph.graphs) == 1
    graph.set_sweep(2)
    assert not graph.graphs
