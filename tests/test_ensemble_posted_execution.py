"""Posted execution lifecycle controls; native forecast qualification is separate."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from gpuwm.ensemble import posted_execution as execution
from gpuwm.ensemble.source_preparation import PostedPreparationFactory
from gpuwm.ingest import boundary_stream


class _Factory(PostedPreparationFactory):
    def __init__(self, root, kind):
        trajectory = SimpleNamespace(identity="base", source="gfs", member=None,
                                     cycle=SimpleNamespace(isoformat=lambda: "2024-05-25T18:00:00+00:00"))
        self.recipe = SimpleNamespace(kind=kind, base=trajectory, sha256="r" * 64,
            members=tuple(SimpleNamespace(index=index, seed=2**63 + index, trajectory=trajectory)
                          for index in (17, 23)))
        self.provider = SimpleNamespace(root=root / "provider", recipe=self.recipe,
                                        plan={"kind": kind, "members": [17, 23]})
        self.provider.root.mkdir()
        self.source_root = (root / "ordinary").resolve()
        self.sources = {"base": SimpleNamespace(prepared_root=self.source_root, trajectory=trajectory)}
        self.sources["base"].verify = lambda: self.sources["base"]
        (self.provider.root / "provider-head.json").write_text(json.dumps({
            "plan": self.provider.plan, "sources": {"base": {
                "prepared_head_sha256": "a" * 64}}}))
        self.calls = []
        self.heads = {self.source_root: {"head_sha256": "a" * 64, "basis": {}}}
        self.future_ready = False

    def prepare_member(self, member_index, *, output_root, forecast=None, observer=None):
        self.calls.append(member_index)
        output_root = Path(output_root).resolve()
        self.heads[output_root] = {"head_sha256": "b" * 64, "basis": {"ensemble_physical": {
            "provider_plan": self.provider.plan, "member_index": member_index}}}
        result = forecast("b" * 64)
        self.future_ready = True
        return {"member_index": member_index, "head_sha256": "b" * 64}, result


def _owner(tmp_path, monkeypatch, kind="control", indices=None):
    factory = _Factory(tmp_path, kind)
    source = SimpleNamespace(prepared_root=factory.source_root, source="gfs",
        stream_head=factory.heads[factory.source_root], experiment=object())
    def read(root, expected_sha256=None):
        head = factory.heads[Path(root).resolve()]
        if expected_sha256 is not None and expected_sha256 != head["head_sha256"]:
            raise ValueError("immutable source head changed")
        return head
    monkeypatch.setattr(boundary_stream, "read_head", read)
    monkeypatch.setattr(boundary_stream, "bind_head", lambda root, sha: read(root, sha))
    monkeypatch.setattr(boundary_stream, "verify_seal", lambda root, **kw: {"status": "verified"})
    def preflight(original, *, prepared_root, head_sha256):
        assert original is source and head_sha256 == "b" * 64
        return SimpleNamespace(prepared_root=prepared_root, stream_head=read(prepared_root))
    monkeypatch.setattr(execution, "preflight_member_inputs", preflight)
    owner = execution.PostedRecipeExecution(factory, {"base": source},
        member_root=tmp_path / "members", member_indices=indices)
    return owner, factory, source


def test_admission_reuses_actual_source_inputs_without_member_preparation(tmp_path, monkeypatch):
    owner, factory, source = _owner(tmp_path, monkeypatch, kind="recentered")
    assert owner.planning_inputs(17) is source
    assert owner.planning_inputs(23) is source
    assert not factory.calls and not factory.future_ready
    assert owner.receipt()["pending_members"] == [17, 23]


def test_unchanged_sparse_members_never_repeat_native_source_preparation(tmp_path, monkeypatch):
    owner, factory, source = _owner(tmp_path, monkeypatch)
    seen = []
    for member in owner.member_order:
        assert owner.run_member(member, forecast=lambda inputs: seen.append(inputs) or member) == member
    assert seen == [source, source]
    assert factory.calls == []
    receipt = owner.require_complete()
    assert receipt["member_order"] == [17, 23]
    assert receipt["members"][17]["seed"] == 2**63 + 17
    assert receipt["members"][17]["native_head_sha256"] == "a" * 64


def test_changed_member_forecasts_at_head_before_future_preparation(tmp_path, monkeypatch):
    owner, factory, source = _owner(tmp_path, monkeypatch, kind="recentered", indices=(23,))
    def forecast(inputs):
        assert not factory.future_ready
        assert inputs is not source
        assert inputs.prepared_root == tmp_path / "members/member-0023"
        assert owner.receipt()["members"][23]["status"] == "forecasting"
        return {"actual_member": 23}
    result = owner.run_member(23, forecast=forecast)
    assert result == {"actual_member": 23}
    assert factory.future_ready and factory.calls == [23]
    assert owner.require_complete()["members"][23]["native_preparation_receipt"]["member_index"] == 23
    # Selecting one replay member does not rewrite the provider population.
    assert factory.provider.plan["members"] == [17, 23]


def test_failed_forecast_never_reports_a_completed_member(tmp_path, monkeypatch):
    owner, _, _ = _owner(tmp_path, monkeypatch)
    def fail(inputs):
        raise RuntimeError("ordinary source failed")
    with pytest.raises(RuntimeError, match="ordinary source failed"):
        owner.run_member(17, forecast=fail)
    assert owner.receipt()["members"][17]["status"] == "failed"
    with pytest.raises(ValueError, match="unfinished"):
        owner.require_complete()


def test_mutated_provider_or_ordinary_head_is_detected_before_execution(tmp_path, monkeypatch):
    owner, factory, _ = _owner(tmp_path, monkeypatch)
    factory.heads[factory.source_root]["head_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="head changed"):
        owner.planning_inputs(17)
    factory.heads[factory.source_root]["head_sha256"] = "a" * 64
    (factory.provider.root / "provider-head.json").write_text("changed")
    with pytest.raises(ValueError, match="descriptor changed"):
        owner.planning_inputs(17)


@pytest.mark.parametrize("indices", [(0,), (17, 17), (True,), ()])
def test_selection_preserves_original_unique_member_ids(tmp_path, monkeypatch, indices):
    with pytest.raises(ValueError, match="unique original"):
        _owner(tmp_path, monkeypatch, indices=indices)


def test_ordinary_preflight_relocates_only_published_member_authorities(tmp_path, monkeypatch):
    from gpuwm import stage_cli, prepared_single_domain_forecast
    original = tmp_path / "source"
    member = tmp_path / "member"
    external = tmp_path / "caller-wps.nml"
    source = SimpleNamespace(prepared_root=original, preflight_arguments={
        "experiment_config": original / "experiment.toml", "wps_namelist": external,
        "source": "gfs", "run_seconds": 48., "history_interval_seconds": 12.,
        "source_manifest_sha256": None, "tiles": object()})
    monkeypatch.setattr(stage_cli, "resolve_head_bundle", lambda root, sha: {
        "source": "gfs", "source_manifest_sha256": None})
    monkeypatch.setattr(prepared_single_domain_forecast, "preflight_prepared_forecast", lambda **kw: kw)
    result = execution.preflight_member_inputs(source, prepared_root=member, head_sha256="b" * 64)
    assert result["experiment_config"] == member / "experiment.toml"
    assert result["wps_namelist"] == external
    assert result["prepared_head_sha256"] == "b" * 64
    assert result["tiles"] is source.preflight_arguments["tiles"]
    assert "proof_sha256" not in result and result["run_seconds"] == 48.


def test_session_consumes_posted_owner_with_sparse_ids_and_shared_source(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from gpuwm.ensemble.packing import CardBudget
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_context import current_capture
    from test_ensemble_production_execution import inputs, Collector

    owner, factory, source = _owner(tmp_path, monkeypatch)
    original = inputs()
    source.experiment = original.experiment
    source.boundary_interval_seconds = 3600
    source.shared_static = original.shared_static
    source.posted_source = original.posted_source
    observed = []
    def runner(actual, *, output_directory, **options):
        member = current_capture().member_id
        assert actual is source and actual.shared_static is original.shared_static
        assert not factory.calls
        observed.append(member)
        return {"status": "PASS", "member_id": member}
    run = PreparedEnsembleSession({"members": 2, "sources": ["first", "second"]},
        output_directory=tmp_path / "run", collector=Collector(), source_execution=owner,
        cards=(CardBudget(0, 1000),), device_scope=lambda _: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)))
    result = run.run_prepared(runner, source)
    assert observed == [17, 23] and result["member_order"] == [17, 23]
    assert result["posted_source_execution"]["status"] == "complete"
    assert json.loads((tmp_path / "run/ensemble-run.json").read_text())["status"] == "PASS"


def test_posted_stochastic_callback_retains_seed_recipe_and_actual_native_head(tmp_path, monkeypatch):
    from dataclasses import FrozenInstanceError
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.stochastic_execution import StochasticPhysicsBinding
    owner, _, _ = _owner(tmp_path, monkeypatch, kind="recentered", indices=(23,))
    run = PreparedEnsembleSession(1, output_directory=tmp_path / "run", source_execution=owner)
    seen = []
    run.stochastic_provider = SimpleNamespace(bind_state=lambda **kw: seen.append(kw))
    callback = run._initialization_callback(23)
    def forecast(inputs):
        callback(state=object(), cfg=object())
        binding = seen[-1]["prepared_member"]
        assert binding.member_id == 23 and binding.seed == 2**63 + 23
        assert binding.recipe_sha256 == owner.recipe.sha256
        assert binding.native_head_sha256 == "b" * 64
        assert seen[-1]["seed"] == binding.seed
        with pytest.raises(FrozenInstanceError):
            binding.recipe_sha256 = "changed"
        hook = SimpleNamespace(snapshot=lambda: {}, restore=lambda value: None)
        original = StochasticPhysicsBinding(hook, member_id=23, recipe_sha256=binding.recipe_sha256)
        changed = StochasticPhysicsBinding(hook, member_id=23, recipe_sha256="different recipe")
        with pytest.raises(ValueError, match="source recipe"):
            changed.restore(original.snapshot())
        return {"status": "PASS"}
    owner.run_member(23, forecast=forecast)
