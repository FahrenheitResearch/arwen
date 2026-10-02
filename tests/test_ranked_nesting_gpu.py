"""Two-way nest transactions through resident ranked slabs."""
from __future__ import annotations

import gc
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu


def _trajectory(monkeypatch, *, mode, control=None, steps=6):
    import cupy as cp
    from gpuwm.core import nest_stream, streaming
    from gpuwm.core.devices import DeviceOptions
    from gpuwm.core.model import execute_experiment
    from gpuwm.core.nest import NestCoupler
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
    from gpuwm.state_digest import canonical_state_digest
    from tilestream import driver, gather, physics_inventory
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor

    monkeypatch.setattr(executor, "I_PARENT_START", 41)
    monkeypatch.setattr(executor, "J_PARENT_START", 41)
    monkeypatch.setattr(executor, "PARENT_DT", 6.)
    cfg = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6.,
                            run_seconds=steps*6., radt_minutes=.5,
                            cudt_minutes=.5)
    child_cfg = moving.child_cfg(cfg, nx=48, ny=48)
    boundaries = moving.domain_boundaries(cfg, seconds=21600.)
    state, geography = moving.build_parent(cfg, boundaries=boundaries)
    experiment = executor.experiment(cfg, child_cfg, steps=steps)
    model = executor.assemble(experiment, state, geography.grid, feedback=1)
    # The product root binds its external Davies recurrence to the domain
    # clock. The idealized assembler deliberately leaves that choice open.
    bind_lateral_boundary_clock(model.root.state, model.root.clock)
    executor.initial_feedback(model)
    steppers = {}
    cards = tuple(range(min(2, cp.cuda.runtime.getDeviceCount())))
    options = DeviceOptions(count=2, grid=(1, 2),
                            ids=(cards[0], cards[-1]))
    for node in model.walk_parent_first():
        # Both storage roads own the same primed frame stash, as the
        # prepared-domain product does before its first output frame.
        streaming.prime_lazy_carriers(node.state, node.cfg.run)
        gid = int(node.cfg.grid_id)
        if mode == "resident" or (mode == "nest" and gid == 1):
            continue
        take = streaming.streamed_store_inventory()
        store = {}
        for name, value in take(node.state).items():
            host = cp.asnumpy(value)
            store[name] = gather.pinned_empty(host.shape, host.dtype)
            store[name][...] = host
        bundle = SimpleNamespace(
            store=store, scalars=physics_inventory.carrier_scalars(node.state),
            geography=driver.geography_store(node.state, host=True),
            boundaries=boundaries if gid == 1 else None, template=node.state)
        decision = streaming.ranked_decision(node.cfg.run, options)
        owner = streaming.ranked_domain_builder(
            bundle, clock=node.clock, options=options, node=node,
            check_geography=False)(None, node.cfg.run, decision)
        owner._state = node.state
        node.state._streamed_domain = owner
        steppers[gid] = owner

    if control == "stale_force":
        original = nest_stream._copy_owned_sides
        def stale_tables(specs, source):
            if source.rolling_generation > 1:
                return 0
            return original(specs, source)
        monkeypatch.setattr(nest_stream, "_copy_owned_sides", stale_tables)
    elif control == "no_feedback":
        monkeypatch.setattr(NestCoupler, "_restrict_windowed", lambda *a: None)
    elif control is not None:
        raise ValueError(control)
    try:
        result = execute_experiment(model, steppers=steppers, validate_state=True,
                                    pool_trim_per_period=False)
        domains = {}
        for node in model.walk_parent_first():
            owner = steppers.get(int(node.cfg.grid_id))
            source = node.state if owner is None else owner.store
            digest, fields = moving.carrier_digest(source)
            canonical = (canonical_state_digest(node.state, node.clock)
                         if owner is None else owner.canonical_digest(node.clock))
            domains[int(node.cfg.grid_id)] = dict(
                digest=digest, fields=fields, canonical=canonical)
        output = dict(domains=domains, cadence=moving.cadence_census(model, steppers),
                      steps=result.steps, forces=result.forces,
                      feedback=model.node(2).coupler.feedback_count)
        print("RANKS NEST ARM " + json.dumps(dict(
            mode=mode, control=control, cards=list(options.device_ids()),
            steps=output["steps"], forces=output["forces"],
            feedback=output["feedback"], domains={
                gid: dict(sha256=row["digest"], arrays=len(row["fields"]),
                          canonical_sha256=row["canonical"]["sha256"],
                          canonical_arrays=row["canonical"]["array_count"])
                for gid, row in domains.items()})), flush=True)
        return output
    finally:
        for owner in steppers.values():
            owner.tiled_run.close()
        del model, state, steppers
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


def test_ranked_tree_matches_resident_and_detects_missing_transactions(monkeypatch):
    with monkeypatch.context() as arm:
        resident = _trajectory(arm, mode="resident")
    for mode in ("both", "nest"):
        with monkeypatch.context() as arm:
            actual = _trajectory(arm, mode=mode)
        for gid in (1, 2):
            fields = actual["domains"][gid]["fields"]
            reference = resident["domains"][gid]["fields"]
            differing = sorted(key for key in fields.keys() | reference.keys()
                               if fields.get(key) != reference.get(key))
            assert not differing, (mode, gid, differing)
            names = actual["domains"][gid]["canonical"]["field_order"]
            reference_names = resident["domains"][gid]["canonical"]["field_order"]
            assert names == reference_names, (mode, gid, sorted(set(names)^set(reference_names)))
        assert actual == resident
    assert resident["forces"] == 6 and resident["feedback"] == 7
    for control, affected in (("stale_force", 2), ("no_feedback", 1)):
        with monkeypatch.context() as arm:
            faulty = _trajectory(arm, mode="both", control=control)
        assert faulty["domains"][affected]["digest"] != resident["domains"][affected]["digest"], control
        assert faulty["forces"] == resident["forces"]
        assert faulty["feedback"] == resident["feedback"]


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        test_ranked_tree_matches_resident_and_detects_missing_transactions(patch)
    print("RANKS NEST PASS")
