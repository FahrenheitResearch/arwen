"""Hybrid edge dispatch retains stock member coupling and integer schedules.

The carried STEP is a GPU word-addition fixture. This qualifies executor and
nest-edge composition, not a production atmospheric STEP or full forecast.
"""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _member_models(members, smooth):
    import cupy as cp
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import DomainNode
    from gpuwm.core.nest import NestCoupler
    from test_clock import _chain_experiment
    from test_model import _model
    from test_ensemble_batch_nesting_gpu import _fixture
    # The existing physical fixture distinguishes every parent donor and
    # exercises actual resident coupling, restriction and diagnostic code.
    _, _, pairs, _ = _fixture(members, ratio=3, mapped=True, smooth=smooth)
    models = []
    add = cp.ElementwiseKernel("float32 delta", "float32 u",
        "u = __fadd_rn(u, delta);", "test_member_nested_carried_step")
    for member, (parent, child, _) in enumerate(pairs):
        exp = _chain_experiment((1, 3), run_seconds=54., history_s=54.)
        domains = tuple(replace(domain, run=replace(source.cfg.run, run_seconds=54., output_interval_s=54.),
            history_interval_s=54., i_parent_start=1 if domain.parent_id == 0 else 5,
            j_parent_start=1 if domain.parent_id == 0 else 5,
            time_step=9 if domain.parent_id == 0 else None)
            for domain, source in zip(exp.domains, (parent, child), strict=True))
        exp = replace(exp, domains=domains, feedback=1, smooth_option=smooth)
        clock = resolve_clock(exp, lbc_interval_s=54.)
        clocks = clock.clocks()
        _, model = _model()
        pnode = DomainNode(domains[0], model.root.grid, parent.state, clocks[1], None, [], None)
        cnode = DomainNode(domains[1], model.node(2).grid, child.state, clocks[2], pnode, [], None)
        cnode.coupler = NestCoupler(cnode, feedback=1, smooth_option=smooth)
        pnode.children.append(cnode)
        model.root, model.nodes_by_grid_id = pnode, {1: pnode, 2: cnode}
        model.schedule = build_schedule(exp, clock)
        def step(state, cfg, *, member=member, **unused):
            add(np.float32((member + 1) * .03125), state.u)
        options = {"validate_state": False, "pool_trim_per_period": False,
                   "steppers": {1: step, 2: step}, "experiment": exp}
        models.append((model, options))
    return models


def _words(cp, model):
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.state import refresh_model_time
    result = {}
    for node in model.walk_parent_first():
        gid, clock = int(node.cfg.grid_id), node.clock
        result[gid] = {"fields": {name: cp.asnumpy(getattr(node.state, name)).tobytes()
                                    for name in state_array_shapes(node.cfg.run)},
            "clock": (clock.ticks, clock.step_ticks, clock.tick_den, clock.step_count,
                clock.dt_fp32.tobytes(), clock.dtbc_fp32.tobytes(), clock.adaptive_state),
            "elapsed": node.state.elapsed_seconds}
        if node.coupler is not None:
            result[gid]["edge"] = {name: getattr(node.coupler, name) for name in (
                "force_count", "feedback_count", "first_parent_ticks", "last_parent_ticks",
                "first_parent_step", "last_parent_step", "last_feedback_ticks", "valid")}
            result[gid]["feedback_pending"] = node.coupler._prepared_feedback
    return result


@pytest.mark.parametrize("members", [2, 4])
@pytest.mark.parametrize("smooth", [0, 2])
def test_hybrid_original_model_callbacks_and_packed_edges_equal_standalone_words(members, smooth):
    import cupy as cp
    from gpuwm.core.model import execute_experiment
    from gpuwm.ensemble.prepared_nested_batch import (
        capture_original_member_schedule, PreparedHybridNestedBatch, ResidentNestedEdgeFactory,
    )
    standalone = _member_models(members, smooth)
    for model, options in standalone:
        execute_experiment(model, **options)
    cp.cuda.get_current_stream().synchronize()
    expected = [_words(cp, model) for model, _ in standalone]
    together = _member_models(members, smooth)
    bindings = [capture_original_member_schedule(member, model,
        operation_authority=lambda **unused: {"layout": "resident-carried-step-edge-fixture"},
        execution_options=options) for member, (model, options) in enumerate(together)]
    batch = PreparedHybridNestedBatch(bindings, ordinary_forecast_bytes={member: 1 << 26 for member in range(members)},
        available_bytes=2 << 30, ordinary_memory_evidence="bounded physical fixture full ordinary reserve",
        pack_factory=ResidentNestedEdgeFactory(qualification_receipt={
            "scope": "resident nest component integration gate", "production_step": False}),
        completion_wait=cp.cuda.get_current_stream().synchronize, array_module=cp)
    receipt = batch.execute()
    assert [_words(cp, model) for model, _ in together] == expected
    assert any(row["packed"] and row["kind"] == "force" for row in receipt["operations"])
    assert any(row["packed"] and row["kind"] == "feedback_commit" for row in receipt["operations"])
    assert all(not row["packed"] for row in receipt["operations"] if row["kind"] == "step")
    assert receipt["copies"]["to_originals_bytes"] > 0
    assert receipt["memory"]["ordinary_forecasts_retained_bytes"] == members * (1 << 26)
    assert receipt["complete_native_forecast"] is False


@pytest.mark.parametrize("members", [2, 4, 8])
@pytest.mark.parametrize("smooth", [0, 2])
def test_edge_dependency_subset_preserves_all_original_state_and_edge_words(members, smooth):
    import cupy as cp
    from gpuwm.core.model import execute_experiment
    from gpuwm.ensemble.prepared_nested_batch import (
        capture_original_member_schedule, PreparedHybridNestedBatch, ResidentNestedEdgeFactory)
    from gpuwm.ensemble.batch_state import state_array_specs
    standalone = _member_models(members, smooth)
    for model, options in standalone:
        execute_experiment(model, **options)
    cp.cuda.get_current_stream().synchronize()
    expected = [_words(cp, model) for model, _ in standalone]
    together = _member_models(members, smooth)
    bindings = [capture_original_member_schedule(member, model,
        operation_authority=lambda **unused: {"layout": "edge-dependency-original-step"},
        execution_options=options) for member, (model, options) in enumerate(together)]
    batch = PreparedHybridNestedBatch(bindings, ordinary_forecast_bytes={member: 1 << 26 for member in range(members)},
        available_bytes=2 << 30, ordinary_memory_evidence="unchanged complete physical fixture forecast envelope",
        pack_factory=ResidentNestedEdgeFactory(qualification_receipt={"scope": "minimal original edge dependencies"},
            edge_state_only=True), completion_wait=cp.cuda.get_current_stream().synchronize, array_module=cp)
    try:
        receipt = batch.execute()
        assert [_words(cp, model) for model, _ in together] == expected
        for gid, bank in batch.banks.items():
            assert len(bank.plan.arrays) < len(state_array_specs(together[0][0].node(gid).cfg.run))
        assert receipt["memory"]["ordinary_forecasts_retained_bytes"] == members * (1 << 26)
        assert any(row["packed"] and row["kind"] == "force" for row in receipt["operations"])
        assert any(row["packed"] and row["kind"] == "feedback_finalize" for row in receipt["operations"])
    finally:
        batch.close()
