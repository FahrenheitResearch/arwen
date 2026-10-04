"""Original headed source ownership, independent of numerical GPU qualification.

Acquisition grammar/configuration and immutable descriptor bytes are real.
The original native reader and seal APIs are controlled orchestration seams.
No numerical source initializer or forecast is replaced inside engine code.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from gpuwm.config import RunConfig
from gpuwm.experiment import experiment_from_run_config
from gpuwm.ensemble import ordinary_execution as execution
from gpuwm.ensemble.recipes import SourceRecipe, SourceTrajectory, RecipeMember
from gpuwm.ensemble.source_preparation import PostedSourcePreparation
from gpuwm.prepared_domain_tree_forecast import PreparedTreeInputs, PreparedDomainBundle
from gpuwm.ingest import boundary_stream
from test_posted_prep_handoff import _handoff


START = datetime(2024, 5, 21, 12, tzinfo=timezone.utc)
ORIGINAL_PREFLIGHT = execution._ordinary_preflight


def _source(root, member, *, pbl):
    acquisition = root / "fetch"
    acquisition.mkdir(parents=True)
    document, ready, future = _handoff(acquisition)
    document["member"] = document["member_verification"]["member"] = member
    (acquisition / "prep-arguments.json").write_text(json.dumps(document))
    schedule_path = acquisition / "posting" / "schedule.json"
    schedule = json.loads(schedule_path.read_text())
    schedule["member"] = member
    schedule_path.write_text(json.dumps(schedule))
    configuration = root / "configuration.toml"
    configuration.write_text(f"# original source controls\nphysics = {pbl}\n")
    trajectory = SourceTrajectory("gefs", START, member)
    prepared_root = root / "ordinary"
    prepared_root.mkdir()
    specification = PostedSourcePreparation.from_acquisition(trajectory,
        acquisition_root=acquisition, prepared_root=prepared_root,
        physical_root=root / "unused-physical", native_arguments=("--experiment-config", str(configuration)))
    config_digest = hashlib.sha256(configuration.read_bytes()).hexdigest()
    plan = {"schema": boundary_stream.INPUT_PLAN_SCHEMA, "route_table_sha256": "a" * 64,
        "manifest": {"schema": "gpuwm-fetch-route-manifest-v1", "request": {
            "source": trajectory.source, "cycle": trajectory.cycle.isoformat(), "member": member},
            "files": {"future": {"name": str(future), "sha256": None}}}}
    head_body = {"basis": {"tree": {"domain_ids": [1, 2]}, "cache": {
        "identity": {"namelist_sha256": config_digest}}, "as_posted": {
        "input_plan": plan, "input_plan_sha256": boundary_stream.input_plan_sha256(plan)}}}
    head_path = prepared_root / "head-fixture.json"
    head_path.write_text(json.dumps(head_body, sort_keys=True) + "\n")
    head = dict(head_body, head_sha256=hashlib.sha256(head_path.read_bytes()).hexdigest())
    cfg = RunConfig(nx=24, ny=20, nz=12, dx=3000., dy=3000., ztop=20000., dt=12.,
        run_seconds=10800., mp_physics=8, bl_pbl_physics=pbl, sf_sfclay_physics=pbl,
        sf_surface_physics=2, cu_physics=0)
    exp = experiment_from_run_config(cfg, START)
    child = replace(exp.root, grid_id=2, parent_id=1, parent_grid_ratio=3, parent_time_step_ratio=3,
        i_parent_start=5, j_parent_start=5, run=replace(cfg, grid_id=2, nx=12, ny=12,
            dx=1000., dy=1000., dt=4., nested=True, specified=False), time_step=4)
    exp = replace(exp, domains=(exp.root, child))
    domains = tuple(PreparedDomainBundle(grid_id, 0 if grid_id == 1 else 1,
        prepared_root / f"d{grid_id:02d}", prepared_root / f"cache{grid_id}",
        prepared_root / f"static{grid_id}", prepared_root / f"geometry{grid_id}",
        prepared_root / f"domain{grid_id}", SimpleNamespace(header={"original": member}),
        MappingProxyType({"source_member": member, "grid_id": grid_id}), MappingProxyType({}),
        MappingProxyType({"original_static": "c" * 64})) for grid_id in (1, 2))
    inputs = PreparedTreeInputs(prepared_root, prepared_root, prepared_root / "preparation",
        prepared_root / "artifact", prepared_root / "manifest", configuration, exp,
        (object(), object()), domains, (0, 3), 10800,
        MappingProxyType({"original_source_member": member}), MappingProxyType({"original": True}),
        MappingProxyType({"experiment_config": config_digest}), "gefs",
        stream_head=MappingProxyType(head), prepared_head_sha256=head["head_sha256"])
    return specification, inputs, head_path, future


def _fixture(tmp_path, monkeypatch, *, selection=None):
    sources = [_source(tmp_path / "p01", "p01", pbl=1), _source(tmp_path / "p02", "p02", pbl=5)]
    recipe = SourceRecipe("input-ensemble", sources[0][0].trajectory, START, START+timedelta(hours=3),
        (RecipeMember(19, 2**63+19, sources[0][0].trajectory), RecipeMember(3, 2**63+3, sources[1][0].trajectory)))
    by_root = {item[0].prepared_root: item for item in sources}
    preflights, seals = [], []
    def read(root, expected_sha256=None):
        path = by_root[Path(root).resolve()][2]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("original immutable source head changed")
        return dict(json.loads(path.read_text()), head_sha256=digest)
    monkeypatch.setattr(boundary_stream, "read_head", read)
    monkeypatch.setattr(boundary_stream, "bind_head", lambda root, sha: read(root, sha))
    def preflight(inputs, *, head_sha256):
        assert head_sha256 == read(inputs.prepared_root)["head_sha256"]
        preflights.append(inputs)
        return inputs
    monkeypatch.setattr(execution, "_ordinary_preflight", preflight)
    monkeypatch.setattr(boundary_stream, "verify_seal", lambda root, **kwargs:
        seals.append(Path(root)) or {"status": "verified-original-seal"})
    specifications = {row[0].trajectory.identity: row[0] for row in sources}
    inputs = {row[0].trajectory.identity: row[1] for row in sources}
    owner = execution.OrdinaryRecipeExecution(recipe, specifications, inputs,
        root=tmp_path / "descriptor", member_indices=selection)
    return owner, sources, preflights, seals


def test_distinct_nested_sources_physics_global_ids_and_seeds_are_forwarded_unchanged(tmp_path, monkeypatch):
    owner, sources, preflights, seals = _fixture(tmp_path, monkeypatch)
    assert owner.member_order == (19, 3) and len(preflights) == 2 and seals == []
    seen = []
    for index, item in zip(owner.member_order, sources):
        assert owner.planning_inputs(index) is item[1]
        binding = owner.stochastic_member_binding(index)
        assert binding.member_id == index and binding.seed == 2**63+index
        assert binding.recipe_sha256 == owner.recipe.sha256 and binding.trajectory == item[0].trajectory
        def forecast(actual):
            assert actual is item[1] and actual.domains is item[1].domains
            assert actual.experiment.root.run.bl_pbl_physics == (1 if index == 19 else 5)
            assert actual.experiment.domains[1].run.dt == 4.
            assert binding.native_head_sha256 == actual.stream_head["head_sha256"]
            seen.append((index, actual))
            return {"status": "PASS", "member_id": index}
        assert owner.run_member(index, forecast=forecast)["member_id"] == index
    assert [index for index, _ in seen] == [19, 3]
    assert seals == [row[0].prepared_root for row in sources]
    assert owner.require_complete()["member_order"] == [19, 3]
    assert all(not row[3].exists() and not row[0].physical_root.exists() for row in sources)


def test_sparse_replay_keeps_full_recipe_and_never_requires_unused_source_head(tmp_path, monkeypatch):
    owner, sources, preflights, _ = _fixture(tmp_path, monkeypatch, selection=(3,))
    assert owner.member_order == (3,) and preflights == [sources[1][1]]
    descriptor = json.loads((owner.root / "ordinary-roster.json").read_text())
    assert descriptor["recipe"] == owner.recipe.describe()
    assert descriptor["recipe_sha256"] == owner.recipe.sha256
    assert owner.stochastic_member_binding(3).seed == 2**63+3
    with pytest.raises(ValueError, match="selected member"):
        owner.planning_inputs(19)


@pytest.mark.parametrize("mutation", ["config", "handoff", "head", "descriptor", "reader", "physics"])
def test_changed_authority_rejected_before_original_forecast(tmp_path, monkeypatch, mutation):
    owner, sources, _, seals = _fixture(tmp_path, monkeypatch)
    specification, inputs, head_path, _ = sources[0]
    if mutation == "config":
        inputs.experiment_config.write_text("changed source configuration")
    elif mutation == "handoff":
        path = specification.acquisition_root / "prep-arguments.json"
        doc = json.loads(path.read_text())
        doc["member"] = "p02"
        path.write_text(json.dumps(doc))
    elif mutation == "head":
        head_path.write_text("{}")
    elif mutation == "descriptor":
        (owner.root / "ordinary-roster.json").write_text("{}")
    elif mutation == "reader":
        object.__setattr__(inputs, "authority_sha256", {"experiment_config": "f" * 64})
    else:
        wrong = replace(inputs.experiment, domains=(replace(inputs.experiment.root,
            run=replace(inputs.experiment.root.run, mp_physics=10)), inputs.experiment.domains[1]))
        monkeypatch.setattr(execution, "_ordinary_preflight", lambda value, **kwargs: replace(value, experiment=wrong))
    with pytest.raises((ValueError, RuntimeError), match="changed|differs|differ|another"):
        owner.run_member(19, forecast=lambda *args: pytest.fail("forecast must not begin after source authority changed"))
    assert seals == []


def test_forecast_or_seal_failure_never_completes_member(tmp_path, monkeypatch):
    owner, _, _, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="original runner failed"):
        owner.run_member(19, forecast=lambda *_: (_ for _ in ()).throw(RuntimeError("original runner failed")))
    assert owner.receipt()["members"][19]["status"] == "failed"
    monkeypatch.setattr(boundary_stream, "verify_seal", lambda *args, **kwargs:
                        (_ for _ in ()).throw(RuntimeError("original source seal failed")))
    with pytest.raises(RuntimeError, match="original source seal failed"):
        owner.run_member(3, forecast=lambda *_: {"status": "PASS"})
    with pytest.raises(ValueError, match="unfinished"):
        owner.require_complete()


def test_returned_failure_receipt_never_verifies_seal_or_completes_member(tmp_path, monkeypatch):
    owner, _, _, seals = _fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="failing forecast receipt"):
        owner.run_member(19, forecast=lambda *_: {"status": "FAIL"})
    assert owner.receipt()["members"][19]["status"] == "failed"
    assert seals == []


def test_changed_recentered_recipe_routes_to_its_existing_physical_owner(tmp_path, monkeypatch):
    owner, _, _, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="physical provider"):
        execution.OrdinaryRecipeExecution(replace(owner.recipe, kind="recentered"), {}, {}, root=tmp_path / "wrong")


def test_original_tree_reader_is_used_without_physical_initialization(tmp_path, monkeypatch):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    from gpuwm import prepared_domain_tree_forecast
    original = sources[0][1]
    calls = []
    monkeypatch.setattr(prepared_domain_tree_forecast, "preflight_prepared_tree", lambda **kwargs:
        calls.append(kwargs) or original)
    assert ORIGINAL_PREFLIGHT(original, head_sha256=original.stream_head["head_sha256"]) is original
    assert calls == [dict(prepared_root=original.prepared_root,
        prepared_head_sha256=original.stream_head["head_sha256"], experiment_config=original.experiment_config,
        experiment_config_sha256=hashlib.sha256(original.experiment_config.read_bytes()).hexdigest(),
        physics_profile=None,
        devices_options=original.experiment.devices)]


def test_reader_physics_and_common_window_are_required_before_descriptor_publication(tmp_path, monkeypatch):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    inputs = sources[0][1]
    original = inputs.experiment
    object.__setattr__(inputs, "experiment", replace(original, run_seconds=3600.))
    with pytest.raises(ValueError, match="common recipe forecast window"):
        execution.OrdinaryRecipeExecution(owner.recipe, owner.specifications, owner.sources,
            root=tmp_path / "window-refused")
    assert not (tmp_path / "window-refused" / "ordinary-roster.json").exists()


def test_duplicate_member_run_cannot_reuse_mutable_forecast_state(tmp_path, monkeypatch):
    owner, _, _, _ = _fixture(tmp_path, monkeypatch)
    owner.run_member(19, forecast=lambda *_: {"status": "PASS"})
    with pytest.raises(ValueError, match="cannot run twice"):
        owner.run_member(19, forecast=lambda *_: pytest.fail("duplicate forecast"))


@pytest.mark.parametrize("captured", [False, True])
def test_static_override_needs_its_actual_ordinary_role_digest(tmp_path, monkeypatch, captured):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    specification, inputs, head_path, _ = sources[0]
    static = tmp_path / "authored-static.bin"
    static.write_bytes(b"original source static metadata fixture")
    replacement = PostedSourcePreparation.from_acquisition(specification.trajectory,
        acquisition_root=specification.acquisition_root, prepared_root=specification.prepared_root,
        physical_root=specification.physical_root,
        native_arguments=specification.native_arguments + ("--static-input", str(static)))
    specs = dict(owner.specifications, **{replacement.trajectory.identity: replacement})
    if captured:
        doc = json.loads(head_path.read_text())
        posted = doc["basis"]["as_posted"]
        posted["input_plan"]["manifest"]["files"]["native_static_input"] = {
            "name": str(static), "sha256": hashlib.sha256(static.read_bytes()).hexdigest()}
        posted["input_plan_sha256"] = boundary_stream.input_plan_sha256(posted["input_plan"])
        head_path.write_text(json.dumps(doc, sort_keys=True) + "\n")
        sha = hashlib.sha256(head_path.read_bytes()).hexdigest()
        object.__setattr__(inputs, "stream_head", MappingProxyType(dict(doc, head_sha256=sha)))
        object.__setattr__(inputs, "prepared_head_sha256", sha)
        changed = execution.OrdinaryRecipeExecution(owner.recipe, specs, owner.sources, root=tmp_path / "captured")
        bindings = json.loads((changed.root / "ordinary-roster.json").read_text())["sources"][
            specification.trajectory.identity]["configured_file_bindings"]
        assert any("native_static_input" in role for role in bindings["--static-input"]["ordinary_roles"])
    else:
        with pytest.raises(ValueError, match="--static-input=.*no captured ordinary role"):
            execution.OrdinaryRecipeExecution(owner.recipe, specs, owner.sources, root=tmp_path / "unbound")
        assert not (tmp_path / "unbound" / "ordinary-roster.json").exists()


@pytest.mark.parametrize("arguments, accepted", [
    (("--run-seconds", "10800", "--history-interval-seconds", "1800"), True),
    (("--preprocess-workers", "2", "--no-stock-wrf-export"), True),
    (("--run-seconds", "3600"), False),
    (("--unrecorded-initialization-knob", "7"), False),
])
def test_native_argument_values_are_bound_or_named_before_admission(tmp_path, monkeypatch, arguments, accepted):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    specification, inputs, _, _ = sources[0]
    # The exact original fixture defaults determine its history interval.
    if arguments[0] == "--run-seconds" and accepted:
        arguments = arguments[:3] + (str(inputs.experiment.root.history_interval_s),)
    replacement = PostedSourcePreparation.from_acquisition(specification.trajectory,
        acquisition_root=specification.acquisition_root, prepared_root=specification.prepared_root,
        physical_root=specification.physical_root,
        native_arguments=specification.native_arguments + arguments)
    specs = dict(owner.specifications, **{replacement.trajectory.identity: replacement})
    if accepted:
        checked = execution.OrdinaryRecipeExecution(owner.recipe, specs, owner.sources, root=tmp_path / "checked")
        bindings = json.loads((checked.root / "ordinary-roster.json").read_text())["sources"][
            specification.trajectory.identity]["native_control_bindings"]
        assert arguments[0] in bindings
    else:
        with pytest.raises(ValueError, match="no matching captured initialization value"):
            execution.OrdinaryRecipeExecution(owner.recipe, specs, owner.sources, root=tmp_path / "refused")
        assert not (tmp_path / "refused" / "ordinary-roster.json").exists()


def test_scalar_reader_dispatch_keeps_original_preflight_authorities(tmp_path, monkeypatch):
    from gpuwm.ensemble import posted_execution
    from gpuwm.prepared_single_domain_forecast import PreparedForecastInputs
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    original = sources[0][1]
    # Only the dispatch is under test; the scalar preflight retains its own
    # exact source/static/configuration checks in its existing reader gates.
    scalar = object.__new__(PreparedForecastInputs)
    object.__setattr__(scalar, "prepared_root", original.prepared_root)
    calls = []
    monkeypatch.setattr(posted_execution, "preflight_member_inputs", lambda value, **kwargs:
        calls.append((value, kwargs)) or scalar)
    assert ORIGINAL_PREFLIGHT(scalar, head_sha256="b" * 64) is scalar
    assert calls == [(scalar, {"prepared_root": scalar.prepared_root, "head_sha256": "b" * 64})]


@pytest.mark.parametrize("backend, accepted", [("cpu", True), ("gpu", False)])
@pytest.mark.parametrize("location", ["metadata", "metadata_user", "source_identity"])
def test_backend_control_matches_all_original_domain_readers(tmp_path, monkeypatch, backend, accepted, location):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    specification, inputs, _, _ = sources[0]
    headers = {"metadata": {"metadata": {"preprocess_backend": "cpu"}},
        "metadata_user": {"metadata": {"user": {"preprocessing": {"backend": "cpu"}}}},
        "source_identity": {"identity": {"source_identity": {"preprocessing": {"backend": "cpu"}}}}}
    domains = tuple(replace(domain, cache_reader=SimpleNamespace(header=headers[location]))
                    for domain in inputs.domains)
    object.__setattr__(inputs, "domains", domains)
    replacement = PostedSourcePreparation.from_acquisition(specification.trajectory,
        acquisition_root=specification.acquisition_root, prepared_root=specification.prepared_root,
        physical_root=specification.physical_root,
        native_arguments=specification.native_arguments + ("--preprocess-backend", backend))
    specs = dict(owner.specifications, **{replacement.trajectory.identity: replacement})
    if accepted:
        checked = execution.OrdinaryRecipeExecution(owner.recipe, specs, owner.sources,
            root=tmp_path / "backend-checked")
        row = json.loads((checked.root / "ordinary-roster.json").read_text())["sources"][
            specification.trajectory.identity]
        assert row["native_control_bindings"]["--preprocess-backend"]["value"] == "cpu"
    else:
        with pytest.raises(ValueError, match="--preprocess-backend=gpu.*no matching captured"):
            execution.OrdinaryRecipeExecution(owner.recipe, specs, owner.sources,
                root=tmp_path / "backend-refused")


def test_tree_preflight_preserves_original_profile_selection(tmp_path, monkeypatch):
    from gpuwm import prepared_domain_tree_forecast
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    inputs = replace(sources[0][1], physics_profile_assertion=MappingProxyType({"profile": "legacy"}))
    calls = []
    monkeypatch.setattr(prepared_domain_tree_forecast, "preflight_prepared_tree", lambda **kwargs:
        calls.append(kwargs) or inputs)
    assert ORIGINAL_PREFLIGHT(inputs, head_sha256=inputs.prepared_head_sha256) is inputs
    assert calls[0]["physics_profile"] == "legacy"
    assert calls[0]["devices_options"] is inputs.experiment.devices


def test_session_consumes_ordinary_owner_without_preparation_or_member_renumbering(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from gpuwm.ensemble.packing import CardBudget
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_context import current_capture
    from test_ensemble_production_execution import Collector
    owner, sources, _, seals = _fixture(tmp_path, monkeypatch)
    by_member = {index: item[1] for index, item in zip(owner.member_order, sources)}
    observed = []
    def runner(actual, *, output_directory, **options):
        member = current_capture().member_id
        assert actual is by_member[member]
        assert actual.domains is by_member[member].domains
        assert len(actual.experiment.domains) == 2
        assert actual.experiment.root.run.bl_pbl_physics == (1 if member == 19 else 5)
        assert Path(output_directory).is_dir()
        observed.append(member)
        return {"status": "PASS", "member_id": member}
    collector = Collector()
    session = PreparedEnsembleSession({"members": 2, "sources": ["first", "second"]},
        output_directory=tmp_path / "run", source_execution=owner, collector=collector,
        cards=(CardBudget(0, 1000),), device_scope=lambda _: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)))
    result = session.run_prepared(runner, sources[0][1])
    assert observed == [19, 3] and result["member_order"] == [19, 3]
    assert result["posted_source_execution"]["status"] == "complete"
    assert result["posted_source_execution"]["members"][19]["seed"] == 2**63+19
    assert seals == [item[0].prepared_root for item in sources]
    assert collector.finished and current_capture() is None
    assert all(not item[0].physical_root.exists() and not item[3].exists() for item in sources)


def _replace_plan(inputs, head_path, manifest):
    doc = json.loads(head_path.read_text())
    posted = doc["basis"]["as_posted"]
    posted["input_plan"]["manifest"] = manifest
    posted["input_plan_sha256"] = boundary_stream.input_plan_sha256(posted["input_plan"])
    head_path.write_text(json.dumps(doc, sort_keys=True) + "\n")
    sha = hashlib.sha256(head_path.read_bytes()).hexdigest()
    object.__setattr__(inputs, "stream_head", MappingProxyType(dict(doc, head_sha256=sha)))
    object.__setattr__(inputs, "prepared_head_sha256", sha)


@pytest.mark.parametrize("different", [False, True])
def test_original_nested_source_record_uses_registered_adapter_and_exact_cycle(tmp_path, monkeypatch, different):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    specification, inputs, head_path, _ = sources[0]
    _replace_plan(inputs, head_path, {"schema": "original-native-direct-manifest",
        "source": {"model": "GEFS", "cycle": START.isoformat(), "member": "p02" if different else "p01"}})
    if different:
        with pytest.raises(ValueError, match="another source, cycle or member"):
            execution.OrdinaryRecipeExecution(owner.recipe, owner.specifications, owner.sources,
                root=tmp_path / "source-refused")
    else:
        checked = execution.OrdinaryRecipeExecution(owner.recipe, owner.specifications, owner.sources,
            root=tmp_path / "source-checked")
        assert checked.planning_inputs(19) is inputs


def test_declarative_manifest_binds_original_acquisition_paths_without_future_reads(tmp_path, monkeypatch):
    from gpuwm.mapped_authoring import INPUT_MANIFEST_SCHEMA
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    specification, inputs, head_path, future = sources[0]
    original_paths = (specification.acquisition_root / "input-list.txt").read_text().splitlines()
    _replace_plan(inputs, head_path, {"schema": INPUT_MANIFEST_SCHEMA,
        "mapping_sha256": "c" * 64, "composition_sha256": "d" * 64,
        "primary_files": [{"path": Path(path).name, "sha256": None, "bytes": None} for path in original_paths],
        "supplements": {}, "provenance": {}, "decoders": {}})
    checked = execution.OrdinaryRecipeExecution(owner.recipe, owner.specifications, owner.sources,
        root=tmp_path / "mapped-checked")
    assert checked.planning_inputs(19) is inputs and not future.exists()
    row = json.loads((checked.root / "ordinary-roster.json").read_text())["sources"][
        specification.trajectory.identity]["source_binding"]
    assert row["original_primary_paths"] == original_paths
    assert row["input_list_sha256"] == hashlib.sha256(
        (specification.acquisition_root / "input-list.txt").read_bytes()).hexdigest()
    # Same basenames in another acquisition still denote different original
    # fields, even though no future payload has been opened or materialized.
    replacement = [str(tmp_path / "another-acquisition" / Path(path).name) for path in original_paths]
    (specification.acquisition_root / "input-list.txt").write_text("\n".join(replacement) + "\n")
    with pytest.raises(ValueError, match="primary-file inventory differs"):
        checked.planning_inputs(19)


def test_original_fractional_stop_uses_reader_clock_without_acquisition_alarm(tmp_path, monkeypatch):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    recipe = replace(owner.recipe, end=START+timedelta(seconds=15))
    for _, inputs, _, _ in sources:
        object.__setattr__(inputs, "experiment", replace(inputs.experiment, run_seconds=15.))
    checked = execution.OrdinaryRecipeExecution(recipe, owner.specifications, owner.sources,
        root=tmp_path / "fractional-stop")
    for member in checked.member_order:
        assert checked.planning_inputs(member).experiment.run_seconds == 15.
    assert checked.recipe.end == recipe.end


def _same_trajectory_variants(tmp_path, monkeypatch, *, selection=None):
    original, sources, _, seals = _fixture(tmp_path, monkeypatch)
    variant = _source(tmp_path / "p01-physics-variant", "p01", pbl=5)
    old_read = boundary_stream.read_head
    def read(root, expected_sha256=None):
        if Path(root).resolve() != variant[0].prepared_root:
            return old_read(root, expected_sha256)
        digest = hashlib.sha256(variant[2].read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("original immutable source head changed")
        return dict(json.loads(variant[2].read_text()), head_sha256=digest)
    monkeypatch.setattr(boundary_stream, "read_head", read)
    monkeypatch.setattr(boundary_stream, "bind_head", lambda root, sha: read(root, sha))
    preflights = []
    def preflight(inputs, *, head_sha256):
        assert head_sha256 == read(inputs.prepared_root)["head_sha256"]
        preflights.append(inputs)
        return inputs
    monkeypatch.setattr(execution, "_ordinary_preflight", preflight)
    base = sources[0]
    recipe = replace(original.recipe, kind="control", members=tuple(
        replace(member, trajectory=base[0].trajectory) for member in original.recipe.members))
    member_sources = {3: execution.OrdinaryMemberSource(variant[0], variant[1])}
    owner = execution.OrdinaryRecipeExecution(recipe, {base[0].trajectory.identity: base[0]},
        {base[0].trajectory.identity: base[1]}, root=tmp_path / "multi-physics",
        member_sources=member_sources, member_indices=selection)
    return owner, base, variant, preflights, seals


def test_same_trajectory_members_keep_independently_bound_physics_heads(tmp_path, monkeypatch):
    owner, base, variant, preflights, seals = _same_trajectory_variants(tmp_path, monkeypatch)
    assert base[0].trajectory == variant[0].trajectory
    assert preflights == [base[1], variant[1]]
    assert owner.planning_inputs(19) is base[1]
    assert owner.planning_inputs(3) is variant[1]
    assert base[1].experiment.root.run.bl_pbl_physics == 1
    assert variant[1].experiment.root.run.bl_pbl_physics == 5
    bindings = [owner.stochastic_member_binding(index) for index in owner.member_order]
    assert [binding.member_id for binding in bindings] == [19, 3]
    assert [binding.seed for binding in bindings] == [2**63+19, 2**63+3]
    assert bindings[0].trajectory == bindings[1].trajectory == owner.recipe.base
    assert bindings[0].native_head_sha256 != bindings[1].native_head_sha256
    for index, expected in ((19, base[1]), (3, variant[1])):
        assert owner.run_member(index, forecast=lambda actual: {"status": "PASS", "original": actual is expected})["original"]
    assert seals == [base[0].prepared_root, variant[0].prepared_root]
    descriptor = json.loads((owner.root / "ordinary-roster.json").read_text())
    assert descriptor["member_source_keys"] == {"19": base[0].trajectory.identity, "3": "member:3"}
    assert descriptor["recipe"] == owner.recipe.describe()
    assert descriptor["sources"]["member:3"]["trajectory_sha256"] == owner.recipe.base.identity


def test_sparse_replay_does_not_preflight_an_unselected_physics_variant(tmp_path, monkeypatch):
    owner, base, variant, preflights, _ = _same_trajectory_variants(tmp_path, monkeypatch, selection=(19,))
    assert preflights == [base[1]]
    variant[2].unlink()
    assert owner.planning_inputs(19) is base[1]
    assert owner.recipe.members[1].index == 3
    assert owner.stochastic_member_binding(19).recipe_sha256 == owner.recipe.sha256


def test_member_specific_configuration_mutation_cannot_affect_another_member(tmp_path, monkeypatch):
    owner, base, variant, _, _ = _same_trajectory_variants(tmp_path, monkeypatch)
    variant[1].experiment_config.write_text("different physics selection")
    assert owner.planning_inputs(19) is base[1]
    with pytest.raises(ValueError, match="configuration or static authority changed"):
        owner.planning_inputs(3)


@pytest.mark.parametrize("mutation", ["unknown_member", "wrong_trajectory", "untyped", "wrong_window"])
def test_member_source_selection_needs_its_exact_preflight_and_original_id(tmp_path, monkeypatch, mutation):
    owner, base, variant, _, _ = _same_trajectory_variants(tmp_path, monkeypatch)
    overrides = dict(owner.member_sources)
    if mutation == "unknown_member":
        overrides[0] = overrides.pop(3)
    elif mutation == "wrong_trajectory":
        wrong = _source(tmp_path / "p02-wrong-trajectory", "p02", pbl=5)
        overrides[3] = execution.OrdinaryMemberSource(wrong[0], wrong[1])
    elif mutation == "untyped":
        overrides[3] = (variant[0], variant[1])
    else:
        object.__setattr__(variant[1], "experiment", replace(variant[1].experiment, run_seconds=3600.))
    with pytest.raises((ValueError, TypeError), match="indices|trajectory|typed|forecast window"):
        execution.OrdinaryRecipeExecution(owner.recipe, owner.specifications, owner.sources,
            root=tmp_path / "variant-refused", member_sources=overrides)
    assert not (tmp_path / "variant-refused" / "ordinary-roster.json").exists()


def test_session_same_source_variants_use_each_original_physics_configuration(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from gpuwm.ensemble.packing import CardBudget
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_context import current_capture
    from test_ensemble_production_execution import Collector
    owner, base, variant, _, _ = _same_trajectory_variants(tmp_path, monkeypatch)
    seen = []
    def runner(actual, **options):
        member = current_capture().member_id
        assert actual is (base[1] if member == 19 else variant[1])
        seen.append((member, actual.experiment.root.run.bl_pbl_physics))
        return {"status": "PASS"}
    session = PreparedEnsembleSession({"members": 2}, output_directory=tmp_path / "variant-run",
        collector=Collector(), source_execution=owner, cards=(CardBudget(0, 1000),),
        device_scope=lambda _: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)))
    result = session.run_prepared(runner, base[1])
    assert seen == [(19, 1), (3, 5)]
    assert result["posted_source_execution"]["status"] == "complete"
    assert result["member_order"] == [19, 3]


def test_absent_member_override_preserves_descriptor_and_default_head_words(tmp_path, monkeypatch):
    owner, sources, _, _ = _fixture(tmp_path, monkeypatch)
    explicit_empty = execution.OrdinaryRecipeExecution(owner.recipe, owner.specifications, owner.sources,
        root=tmp_path / "explicit-empty", member_sources={})
    assert (explicit_empty.root / "ordinary-roster.json").read_bytes() == (owner.root / "ordinary-roster.json").read_bytes()
    assert explicit_empty.stochastic_member_binding(19) == owner.stochastic_member_binding(19)
