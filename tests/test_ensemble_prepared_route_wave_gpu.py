"""Opt-in route-worker GPU proofs; root launches these under OWNER.

These use ordinary executor closures around initialized physical fixtures.
They do not qualify the full prepared CLI, products or production admission.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _reservation(plans, models):
    import cupy as cp
    from gpuwm.core.preflight import estimate_experiment, local_memory_profile_from_device
    from gpuwm.ensemble.admission import ordinary_memory_model_from_estimate
    from gpuwm.ensemble.prepared_route_wave import RouteWaveReservation
    cp.cuda.get_current_stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    free, total = cp.cuda.runtime.memGetInfo()
    profile = local_memory_profile_from_device(cp, device_id=0)
    ordinary = {member: ordinary_memory_model_from_estimate(estimate_experiment(options["experiment"],
        forcing_interval_seconds=options["experiment"].run_seconds, profile=profile,
        vram_gib=int(total) / (1024 ** 3)), inventory_id="complete-original-fixture-forecast").required_bytes(1)
        for member, (_, options) in enumerate(models)}
    return RouteWaveReservation(ordinary, plans,
        {"collector": 0, "stochastic": 0, "cuda_owners": 1 << 24, "allocator_margin": 1 << 29},
        {"ordinary": "actual estimate_experiment complete forecast envelope on live owner GPU profile for every fixture",
         "native": "every actual fixture domain-bank and edge/physics allocation plan before route workers",
         "collector": "metadata-only fixture callback; no diagnostic GPU arrays",
         "stochastic": "inactive fixture", "cuda_owners": "16 MiB extra fixture stream/event reserve",
         "allocator_margin": "512 MiB explicit extra allocator/transient fixture reserve"}, int(free))


def _run(models, *, pack_factory=None, plans=None):
    import cupy as cp
    from gpuwm.core.model import execute_experiment
    from gpuwm.ensemble.member_stream import member_cuda_scope
    from gpuwm.ensemble.prepared_route_wave import RouteMember, PreparedRouteWave
    from gpuwm.ensemble.runtime_context import MemberOutputCapture
    def runner(inputs, *, output_directory, schedule_dispatch):
        model, options = inputs
        result = execute_experiment(model, schedule_dispatch=schedule_dispatch, **options)
        return {"status": "PASS", "schedule": result}
    members = tuple(RouteMember(member, pair, f"unused-{member}",
        MemberOutputCapture(lambda **unused: None, member),
        lambda **unused: {"scope": "tiny-original-route-owner-fixture"}) for member, pair in enumerate(models))
    return PreparedRouteWave(members, reservation=_reservation(plans or {}, models), runner=runner,
        member_scope=member_cuda_scope, array_module=cp, pack_factory=pack_factory).run()


def test_route_worker_native_edges_match_independent_stock_member_words_and_clocks():
    import cupy as cp
    from gpuwm.core.model import execute_experiment
    from gpuwm.ensemble.prepared_nested_batch import ResidentNestedEdgeFactory, _domain_bank_plan
    from test_ensemble_prepared_nested_batch_gpu import _member_models, _words
    standalone = _member_models(2, 0)
    for model, options in standalone:
        execute_experiment(model, **options)
    cp.cuda.get_current_stream().synchronize()
    expected = [_words(cp, model) for model, _ in standalone]
    together = _member_models(2, 0)
    plans = {f"bank:{gid}": _domain_bank_plan(tuple(model.node(gid) for model, _ in together),
        shared_fields=(), array_module=cp) for gid in together[0][0].nodes_by_grid_id}
    descriptors = {gid: SimpleNamespace(cfg=together[0][0].node(gid).cfg.run,
        storage=SimpleNamespace(specs={spec.name: spec for spec in plans[f"bank:{gid}"].arrays}))
        for gid in together[0][0].nodes_by_grid_id}
    factory = ResidentNestedEdgeFactory(qualification_receipt={"scope": "tiny-route-worker-native-edge-test"})
    bindings = tuple(SimpleNamespace(member_id=member, model=model) for member, (model, _) in enumerate(together))
    plans.update(factory.memory_plans(bindings, descriptors))
    # Plans deliberately own every field. Shared-field admission is covered
    # by the native state bank tests rather than inferred from this fixture.
    from gpuwm.ensemble.prepared_route_wave import RouteMember, PreparedRouteWave
    from gpuwm.ensemble.runtime_context import MemberOutputCapture
    from gpuwm.ensemble.member_stream import member_cuda_scope
    def runner(inputs, *, output_directory, schedule_dispatch):
        model, options = inputs
        return {"status": "PASS", "schedule": execute_experiment(model, schedule_dispatch=schedule_dispatch, **options)}
    members = tuple(RouteMember(member, pair, f"unused-{member}", MemberOutputCapture(lambda **unused: None, member),
        lambda **unused: {"scope": "same-layout-physical-nest-fixture"}) for member, pair in enumerate(together))
    result = PreparedRouteWave(members, reservation=_reservation(plans, together), runner=runner,
        member_scope=member_cuda_scope, array_module=cp, pack_factory=factory, shared_fields=()).run()
    assert [_words(cp, model) for model, _ in together] == expected
    assert any(row["packed"] and row["kind"] == "force" for row in result["execution"]["operations"])
    assert any(row["packed"] and row["kind"] == "feedback_commit" for row in result["execution"]["operations"])
    assert any(row.get("original_dispatch") == "group_callback" for row in result["execution"]["operations"])
    assert result["execution"]["complete_native_forecast"] is False


def _atmospheric_models():
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import DomainNode
    from test_clock import _chain_experiment
    from test_model import _model
    from test_ensemble_packed_production_physics_gpu import _driver
    result = []
    for member in range(2):
        driver, cfg = _driver(member, specified=False)
        cfg = replace(cfg, use_adaptive_time_step=False, dt=12.0, run_seconds=12.0, output_interval_s=12.0)
        exp = _chain_experiment((1,), run_seconds=12.0, history_s=12.0)
        domain = replace(exp.domains[0], run=cfg, time_step=12, history_interval_s=12.0)
        exp = replace(exp, domains=(domain,), feedback=0)
        clock = resolve_clock(exp, lbc_interval_s=12.0)
        _, model = _model()
        root = DomainNode(domain, model.root.grid, driver.state, clock.clocks()[1], None, [], None)
        model.root, model.nodes_by_grid_id = root, {1: root}
        model.schedule = build_schedule(exp, clock)
        result.append((model, {"experiment": exp, "validate_state": False, "pool_trim_per_period": False,
                               "skip_feedback_path": True}))
    return result


def test_route_worker_concurrent_stock_dycore_mynn_ruc_rte_matches_standalone_members():
    import cupy as cp
    from gpuwm.core.model import execute_experiment
    from gpuwm.ensemble.member_stream import member_cuda_scope
    from gpuwm.io.restart import state_manifest, _scratch_manifest, _driver_manifest
    from test_ensemble_packed_production_physics_gpu import _assert_driver_words
    standalone, together = _atmospheric_models(), _atmospheric_models()
    cp.cuda.get_current_stream().synchronize()
    for member, (model, options) in enumerate(standalone):
        with member_cuda_scope(member_id=member, device_id=0, array_module=cp):
            execute_experiment(model, **options)
    result = _run(together)
    cp.cuda.get_current_stream().synchronize()
    for member, ((reference, _), (candidate, _)) in enumerate(zip(standalone, together, strict=True)):
        def words(model):
            node = model.root
            arrays = state_manifest(node.state)
            arrays.update(_scratch_manifest(node.state))
            arrays.update(_driver_manifest(node.state.physics))
            return {name: value.get().tobytes() for name, value in arrays.items()}
        assert words(candidate) == words(reference)
        _assert_driver_words(candidate.root.state.physics, reference.root.state.physics, member, 1)
        assert candidate.root.clock.ticks == reference.root.clock.ticks
        assert candidate.root.clock.step_count == reference.root.clock.step_count
    assert result["execution"]["packed_components"] is False
    assert result["default_door_enabled"] is False
