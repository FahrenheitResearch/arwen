"""Auto selection must use the same configured admission as its caller."""
from dataclasses import asdict, replace
import json

import pytest

from gpuwm.core import preflight as pf, streaming as st
from gpuwm.experiment import DomainConfig, experiment_config_document
from tilestream import autoplan as ap
from tilestream.test_ledger_gate import _exp

GIB = 1024 ** 3


def _windows_estimate(exp, **kwargs):
    # Exercise the measured WDDM arithmetic on either test host, without a GPU.
    return replace(pf.estimate_experiment(exp, **kwargs), envelope_family="windows")


def _tree(root_n=704, child_n=64):
    exp = _exp(root_n, "auto")
    root = exp.root
    run = replace(root.run, grid_id=2, nx=child_n, ny=child_n,
                  dx=root.run.dx / 3, dy=root.run.dy / 3, dt=root.run.dt / 3,
                  nested=True, specified=False)
    child = DomainConfig(grid_id=2, parent_id=1, i_parent_start=20,
                         j_parent_start=20, parent_grid_ratio=3,
                         parent_time_step_ratio=3,
                         history_interval_s=root.history_interval_s, run=run)
    return replace(exp, domains=(root, child))


@pytest.mark.parametrize("n,free_gib", [(704, 23.5), (960, 31.4)])
def test_old_empirical_fit_cannot_strand_an_auto_forecast(n, free_gib):
    exp = _exp(n, "auto")
    machine = ap.Machine(int(free_gib * GIB), 256 * GIB)
    estimate = _windows_estimate(exp)
    # Discriminating control: the old decision really did select resident.
    old = st.decide(exp.root.run, replace(exp.tiles, resident_context=None),
                    machine=machine)
    assert not old.stream
    assert estimate.peak_envelope_bytes > machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    decision = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=estimate)
    assert decision.stream
    envelope = st.streamed_envelope(exp.root.run, exp.tiles,
                                    machine=machine, decision=decision)
    assert envelope.peak_vram_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert decision.detail["resident_admission"]["envelope_bytes"] == estimate.peak_envelope_bytes
    assert decision.detail["resident_admission"]["budget_bytes"] == machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES


@pytest.mark.parametrize("n,free_gib,streams", [(512, 23.5, False),
                                                (816, 23.5, True),
                                                (976, 31.4, True)])
def test_safe_resident_and_adjacent_streaming_roads_remain(n, free_gib, streams):
    exp = _exp(n, "auto")
    machine = ap.Machine(int(free_gib * GIB), 256 * GIB)
    decision = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=_windows_estimate(exp))
    assert decision.stream is streams


def test_an_admitted_small_resident_grid_does_not_pay_the_tile_reserve_twice():
    exp = _exp(200, "auto")
    machine = ap.Machine(int(7.25 * GIB), 128 * GIB)
    estimate = _windows_estimate(exp)
    assert estimate.peak_envelope_bytes < machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    # The coarse, context-free tile footprint still has its own use, but it
    # must not overturn the complete configuration's admitted resident road.
    coarse = st.decide(exp.root.run, replace(exp.tiles, resident_context=None), machine=machine)
    assert coarse.stream
    selected = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=estimate)
    assert not selected.stream
    assert selected.resident_bytes == estimate.peak_envelope_bytes
    assert selected.budget_bytes == machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert selected.detail["host_claim_bytes"] == 0


def test_exact_phase_estimate_is_forwarded_once_with_actual_operands(monkeypatch):
    exp = replace(_exp(704, "auto"), column_chunk=157,
                  vertical=replace(_exp(704).vertical, p_top=10000.0))
    machine = ap.Machine(int(23.5 * GIB), 256 * GIB)
    original = pf.estimate_experiment
    calls = []
    estimates = []
    def counted(value, **kwargs):
        calls.append((value, kwargs))
        result = replace(original(value, **kwargs), envelope_family="windows")
        estimates.append(result)
        return result
    monkeypatch.setattr(pf, "estimate_experiment", counted)
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: pytest.fail("second device observation"))
    phases = pf.estimate_phases(exp, source=None, machine=machine,
        forcing_intervals=47, profile=pf.MEASURED_LOCAL_MEMORY_PROFILE, column_chunk=91)
    assert len(calls) == 1
    assert calls[0][0] is exp
    assert calls[0][1]["forcing_intervals"] == 47
    assert calls[0][1]["profile"] is pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert calls[0][1]["column_chunk"] == 91
    assert phases.forecast is estimates[0]
    assert phases.forecast.column_chunk == 91
    # The configured snapshot has the non-default scientific controls too.
    context = exp.tiles.resident_context.experiment
    assert context.column_chunk == 157 and context.vertical.p_top == 10000.0


def test_context_is_finite_and_not_traversed_by_public_identity():
    from gpuwm.core.model import restart_identity_payload
    exp = _tree()
    assert exp.tiles.resident_context.experiment.tiles.resident_context is None
    assert all(dc.tiles is None for dc in exp.tiles.resident_context.experiment.domains)
    assert "resident_context" in asdict(exp)["tiles"]  # finite, even for raw callers
    original_document = experiment_config_document(exp)
    original_identity = restart_identity_payload(exp)
    class DoNotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("derived planner state entered public serialization")
    object.__setattr__(exp, "tiles", replace(exp.tiles,
        resident_context=st.ResidentAdmissionContext(DoNotCopy())))
    assert experiment_config_document(exp) == original_document
    assert restart_identity_payload(exp) == original_identity
    text = json.dumps(original_document, default=str)
    assert "resident_context" not in text and "radiation_context" not in text
    assert original_document["tiles"] == exp.tiles.to_mapping()


def test_tree_just_over_budget_keeps_the_earlier_resident_preference():
    exp = _tree()
    estimate = _windows_estimate(exp)
    machine = ap.Machine(estimate.peak_envelope_bytes + pf.EXTERNAL_MARGIN_BYTES - 1,
                         256 * GIB)
    rows = {}
    outcome = st.decide_tree(st._config_tree_nodes(exp.domains), exp.tiles,
                            machine=machine, decisions=rows, resident_estimate=estimate)
    assert not rows[1].stream and rows[2].stream
    assert outcome.resident_subset_envelope_bytes < estimate.peak_envelope_bytes
    assert outcome.resident_subset_envelope_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert rows[2].detail["resident_admission"]["preference_changed"]


def test_streaming_a_tiny_child_does_not_hide_an_oversized_resident_parent():
    exp = _tree()
    estimate = _windows_estimate(exp)
    nodes = st._config_tree_nodes(exp.domains)
    root_bound = st._resident_subset_envelope(estimate, nodes, {1})
    machine = ap.Machine(root_bound + pf.EXTERNAL_MARGIN_BYTES - 1, 256 * GIB)
    rows = {}
    result = st.decide_tree(nodes, exp.tiles, machine=machine, decisions=rows,
                           resident_estimate=estimate)
    assert rows[1].stream
    assert not rows[2].stream  # mixed road fits; do not stream everything
    assert result.resident_subset_envelope_bytes < root_bound


def test_declared_budget_is_not_charged_external_margin_twice():
    exp = _exp(704, "auto", vram_budget_bytes=24 * GIB)
    estimate = _windows_estimate(exp)
    machine = ap.Machine(64 * GIB, 256 * GIB)
    decision = st.decide(exp.root.run, exp.tiles, machine=machine, resident_estimate=estimate)
    assert decision.detail["resident_admission"]["budget_bytes"] == 24 * GIB
    assert not decision.stream


def test_explicit_roads_do_not_price_or_probe_resident_admission(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit road consulted configured auto admission")
    monkeypatch.setattr(pf, "estimate_experiment", forbidden)
    monkeypatch.setattr(ap.Machine, "detect", forbidden)
    cfg = _exp(704).root.run
    assert not st.decide(cfg, st.OFF).stream
    pinned = st.StreamingOptions(mode="on", tile_nx=176, tile_ny=176, nbuffers=2)
    decision = st.decide(cfg, pinned)
    assert decision.stream and decision.tile_nx == 176 and decision.nbuffers == 2
    assert st.cold_planning_machine(replace(_exp(704), tiles=pinned)) is None


def test_cold_machine_is_captured_once_and_reused_during_geometry_refinement(monkeypatch):
    from types import SimpleNamespace
    from gpuwm import runtime
    exp = _exp(704, "auto")
    machine = ap.Machine(24 * GIB, 256 * GIB)
    calls = []
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: calls.append(kw) or machine)
    captured = st.cold_planning_machine(exp)
    assert captured is machine and len(calls) == 1
    cfg = replace(exp.root.run, use_adaptive_time_step=True)
    prepared = SimpleNamespace(cfg=cfg, grid=SimpleNamespace(
        mapfac_u=lambda: [[1.0]], mapfac_v=lambda: [[1.0]]))
    estimate = _windows_estimate(exp)
    observed = []
    monkeypatch.setattr(st, "decide", lambda cfg, options, **kw: observed.append(kw) or "decision")
    _, result = runtime._refine_single_streaming_plan(prepared, exp.tiles, object(), captured, estimate)
    assert result == "decision" and observed == [{"machine": machine, "resident_estimate": estimate}]
    assert len(calls) == 1


def test_mixed_fit_pays_for_streamed_marginal_and_corridor_beside_resident():
    exp = _tree(child_n=256)
    estimate = _windows_estimate(exp)
    nodes = st._config_tree_nodes(exp.domains)
    root_bound = st._resident_subset_envelope(estimate, nodes, {1})
    budget = root_bound + 1024 ** 2
    rows = {}
    result = st.decide_tree(nodes, exp.tiles,
        machine=ap.Machine(budget + pf.EXTERNAL_MARGIN_BYTES, 256 * GIB),
        decisions=rows, resident_estimate=estimate)
    # The old max-only admission selected a resident root plus >2GiB of
    # child tile/corridor allocations with merely1MiB beside the root bound.
    assert rows[1].stream
    streamed_claims = sum(int(d.detail.get(key, 0))
        for d in rows.values() if d.stream
        for key in ("claim_bytes", "corridor_claim_bytes"))
    if any(not d.stream for d in rows.values()):
        assert result.configured_mixed_envelope_bytes == (
            result.resident_subset_envelope_bytes + streamed_claims)
    assert result.configured_mixed_envelope_bytes <= budget


def test_immutable_process_floor_prunes_many_auto_preferences(monkeypatch):
    exp = _tree(64, 32)
    children = tuple(replace(exp.domains[1], grid_id=gid,
                            run=replace(exp.domains[1].run, grid_id=gid))
                     for gid in range(2, 11))
    exp = replace(exp, domains=(exp.root, *children))
    original = st._decide_tree
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(st, "_decide_tree", counted)
    with pytest.raises(st.StreamingRefused, match="shared process/radiation floor"):
        st.decide_tree(st._config_tree_nodes(exp.domains), exp.tiles,
                      machine=ap.Machine(GIB, 256 * GIB))
    assert len(calls) == 1  # previously1025 identical impossible walks


def test_auto_ordinary_dispatch_prices_actual_intervals_before_initialization(tmp_path, monkeypatch):
    from datetime import timedelta
    from types import SimpleNamespace
    from gpuwm import runtime
    from gpuwm.ingest import case_store, preflight as ingest_preflight
    from test_runtime import _fixture_pair
    exp, data = _fixture_pair(tmp_path)
    exp = replace(exp, tiles=st.StreamingOptions(mode="auto"))
    events = []
    machine = ap.Machine(24 * GIB, 256 * GIB)
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: events.append("cold") or machine)
    catalog = SimpleNamespace(valid_times=tuple(exp.start_time + timedelta(hours=i)
                                               for i in range(48)), excluded_valid_times=())
    monkeypatch.setattr(ingest_preflight, "build_input_catalog",
                        lambda data: events.append("catalog") or catalog)
    monkeypatch.setattr(runtime, "forcing_snapshots", lambda *a: events.append("decode") or {})
    monkeypatch.setattr(runtime, "forcing_schedule", lambda *a: catalog.valid_times)
    marker = object()
    def price(value, **kwargs):
        assert value is exp and kwargs == {"forcing_intervals": 47}
        events.append("price")
        return marker
    monkeypatch.setattr(pf, "estimate_experiment", price)
    def choose(cfg, options, **kwargs):
        assert kwargs == {"machine": machine, "resident_estimate": marker}
        events.append("decide")
        return st.StreamingDecision(True, "test", 8, 8, 2, 16)
    monkeypatch.setattr(st, "decide", choose)
    monkeypatch.setattr(case_store, "initialization_resources", lambda options: {})
    class ReachedInitializer(Exception):
        pass
    def prepare(*args, **kwargs):
        events.append("prepare")
        assert kwargs["store_request"].backend == "cuda"
        raise ReachedInitializer
    monkeypatch.setattr(runtime, "prepare_experiment_case", prepare)
    with pytest.raises(ReachedInitializer):
        runtime.run_experiment(exp, data, tmp_path / "out")
    assert events == ["cold", "catalog", "decode", "price", "decide", "prepare"]


def test_all_streamed_tree_still_prices_retained_global_workspaces():
    exp = _tree(child_n=256)
    estimate = _windows_estimate(exp)
    nodes = st._config_tree_nodes(exp.domains)
    empty = st._resident_subset_envelope(estimate, nodes, set())
    assert empty > estimate.scratch_arena_bytes + estimate.dycore_state_workspace_bytes
    assert empty == replace(estimate, domains=(), legacy_call_peak_by_domain=()).peak_envelope_bytes
    rows = {}
    result = st.decide_tree(nodes, exp.tiles,
        machine=ap.Machine(int(13.5 * GIB), 256 * GIB), decisions=rows,
        resident_estimate=estimate)
    assert all(row.stream for row in rows.values())
    assert result.configured_mixed_envelope_bytes >= empty
    assert result.configured_mixed_envelope_bytes <= 13 * GIB


def test_only_auto_treats_redundancy_as_advice_and_real_shortages_stay_errors():
    # A small open grid is legally tiled but inevitably exceeds the low-level
    # default4x throughput preference. Pinned and hard API limits remain intact.
    cfg = _exp(48).root.run
    machine = ap.Machine(32 * GIB, 256 * GIB)
    with pytest.raises(ap.CannotPlan) as hard:
        ap.plan(cfg, machine, prefer_resident=False, max_redundancy=4.0)
    assert "redundancy" in hard.value.detail
    legal = st._plan_with_efficiency_advice(
        cfg, machine, mode="auto", prefer_resident=False)
    assert legal.redundancy > 4 and legal.vram_bytes <= legal.vram_budget_bytes
    assert any("slower legal tiling" in item for item in legal.warnings)
    with pytest.raises(ap.CannotPlan):
        st._plan_with_efficiency_advice(cfg, machine, mode="on", prefer_resident=False)
    with pytest.raises(ap.CannotPlan) as shortage:
        st._plan_with_efficiency_advice(cfg, replace(machine, vram_bytes=1),
                                       mode="auto", prefer_resident=False)
    assert "redundancy" not in shortage.value.detail


def test_tree_public_dispatch_preserves_cold_machine_and_domain_options(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gpuwm import runtime
    from gpuwm.core import model as model_module

    exp = _tree(64, 32)
    child = replace(exp.domains[1], tiles=st.StreamingOptions(mode="off"))
    exp = replace(exp, domains=(exp.root, child))
    machine = ap.Machine(24 * GIB, 256 * GIB)
    model = SimpleNamespace(_input_catalog=object())
    data, result = object(), object()
    events = []
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: events.append("cold") or machine)
    def build(value, companion):
        assert value is exp and companion is data
        assert value.domains[1].tiles.mode == "off"
        events.append("build")
        return model
    monkeypatch.setattr(model_module, "build_experiment", build)
    monkeypatch.setattr(runtime, "resolved_tree_config_report", lambda *args: "test report")
    def execute(value, companion, outdir, built, **kwargs):
        assert value is exp and companion is data and built is model
        assert kwargs["planning_machine"] is machine
        assert value.tiles.mode == "auto" and value.domains[1].tiles.mode == "off"
        events.append("execute")
        return result
    monkeypatch.setattr(runtime, "_run_built_experiment", execute)
    assert runtime.run_experiment(exp, data, tmp_path / "tree") is result
    assert events == ["cold", "build", "execute"]
