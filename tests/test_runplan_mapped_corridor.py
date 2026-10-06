"""TC authoring, staged preparation requirements, reuse and tree dispatch.

The native preparation and numerical runner are observed at their call
boundaries. Canonical tiny caches exercise the real preparation-reuse owner;
the proof fixture tests digest relay, not meteorological preparation validity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import cyclone_setup, cyclone_sources, fetch_routes, runplan, stage_cli, stage_reuse
from gpuwm.hrrr_prepared_bundle import render_wps_namelist
from gpuwm.prepared_source_schemas import source_schemas
from gpuwm.source_adapters import packaged_profile_sources
from test_stage_reuse_hierarchy import ENGINE, _tree
from test_stage_seams import _mapped_evidence, _tree_bundle

def _mapped_source():
    # The tiny reuse proof needs a shipped schema, packaged profile and a
    # route row that its synthetic-source graft can copy. Archive fetchers
    # outside that route table need their own fixture and supported epoch.
    schemas = source_schemas()
    profiles = packaged_profile_sources()
    routes = fetch_routes.route_ids()
    return next(row["source"] for row in cyclone_sources.source_options()
                if row["source"] in schemas and row["source"] in profiles
                and row["source"] in routes
                and row["coverage_envelope"] is None
                and runplan.source_follow_statics(row["source"])["chain"]
                == "prepared:staged"
                and not runplan.drivability_for(row["source"]).get("requires_source_root"))


def _authored(tmp_path, source=None):
    source = source or _mapped_source()
    # A nest this fixture can actually follow.  16x16 at ratio 4 is four
    # parent cells wide, and a 0.7 overlap floor admits no move at all on
    # it, so the door refuses to author a follow table for it by name.  28
    # parent cells of parent around a seven-parent-cell nest is the same
    # cheap two-domain tree and admits a move of one.
    text, exp = cyclone_setup.configuration_text(
        cycle="2026090900", point=(35., -97.), forcing_source=source,
        tiles="off", dimensions=((40, 40), (28, 28)))
    config = tmp_path / "storm.toml"
    config.write_text(text, encoding="utf-8")
    config.with_suffix(".namelist.wps").write_text(
        render_wps_namelist(exp, interval_seconds=cyclone_setup._forcing_interval(source)),
        encoding="utf-8")
    document = {"schema": runplan.PLAN_SCHEMA, "name": "storm",
                "route": "prepared", "config": {"path": str(config)},
                "output_root": str(tmp_path / "out"),
                "run_options": {"geog_root": str(tmp_path / "geography")}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(document), encoding="utf-8")
    return runplan.load_plan(plan_path), exp, config, source


def test_authored_mapped_tc_resolves_to_corridor_preparation(tmp_path):
    plan, _exp, _config, source = _authored(tmp_path)
    resolution, exp, _data = runplan.resolve_plan(plan, require_inputs=False)
    assert exp.domains[1].follow is not None
    assert runplan.source_follow_statics(source)["chain"] == "prepared:staged"
    assert resolution["moving_nest"]["statics_corridor"] is True
    assert resolution["moving_nest"]["delivery"] == "statics_corridor"


def test_each_preparer_consumes_its_declared_corridor_option(tmp_path):
    from gpuwm import source_cli

    for row in source_cli.preparation_runners().values():
        args = source_cli._parser().parse_args(["--statics-corridor"])
        # A split root/hierarchy preparer receives this option at its child
        # stage. This is the real dispatcher command callback in every case.
        args.root_preparation = tmp_path / "root"
        args.input_list = tmp_path / "inputs.txt"
        args.supplement = []
        args.provenance = []
        command = row.command(args)
        assert command.count(row.moving_statics()["option"]) == 1
        assert command[2] in row.corridor_stage
        assert stage_cli._schema_index()[row.hierarchy_schema]["layout"] == "tree"


def test_an_unrelated_single_domain_preparer_cannot_disable_mapped_tc(tmp_path, monkeypatch):
    from dataclasses import replace
    from gpuwm import fetch_routes, source_adapters as registry, source_cli

    source = _mapped_source()
    donor = registry.get_source_adapter(source)
    runner_id = "synthetic_single_domain"
    added = replace(donor, source_id="synthetic-single-domain", aliases=(), runner=runner_id)
    monkeypatch.setattr(registry, "_ADAPTERS", (*registry.source_adapters(), added))
    monkeypatch.setattr(registry, "_ALIASES", {**registry._ALIASES, added.source_id: added})
    monkeypatch.setattr(fetch_routes, "_ROUTES", {
        **fetch_routes._ROUTES, added.source_id: fetch_routes._ROUTES[source]})
    runners = source_cli.preparation_runners()
    runners[runner_id] = replace(runners[donor.runner], hierarchy_schema=None, corridor_stage=None)
    monkeypatch.setattr(source_cli, "preparation_runners", lambda: runners)

    assert cyclone_sources.follow_statics(source)["integrates_moving_nest"] is True
    plan, _exp, _config, _source = _authored(tmp_path, source)
    assert runplan.resolve_plan(plan, require_inputs=False)[0]["moving_nest"]["statics_corridor"] is True
    missing = cyclone_sources.follow_statics(added.source_id)
    assert missing["integrates_moving_nest"] is False
    assert "implementation produces no domain hierarchy" in missing["reason"]
    other = tmp_path / "single"
    other.mkdir()
    plan, _exp, _config, _source = _authored(other, added.source_id)
    with pytest.raises(runplan.PlanError, match="implementation produces no domain hierarchy"):
        runplan.resolve_plan(plan, require_inputs=False)


@pytest.mark.parametrize("old_bundle,new_source", [(False, False), (True, False), (True, True)])
def test_staged_tc_binds_corridor_before_reuse_and_dispatches_tree(
        tmp_path, monkeypatch, old_bundle, new_source):
    source = _mapped_source()
    if new_source:
        from dataclasses import replace
        from gpuwm import source_adapters as registry, fetch_routes, source_cli

        donor = registry.get_source_adapter(source)
        grafted = replace(donor, source_id="synthetic-compatible-source", aliases=())
        monkeypatch.setattr(registry, "_ADAPTERS", (*registry.source_adapters(), grafted))
        monkeypatch.setattr(registry, "_ALIASES", {**registry._ALIASES, grafted.source_id: grafted})
        monkeypatch.setattr(fetch_routes, "_ROUTES", {
            **fetch_routes._ROUTES, grafted.source_id: fetch_routes._ROUTES[source]})
        assert source_cli.source_preparation_outputs(grafted.source_id) == source_cli.source_preparation_outputs(source)
        assert cyclone_sources.follow_statics(grafted.source_id)["integrates_moving_nest"] is True
        runners = source_cli.preparation_runners()
        runners["synthetic_single_domain"] = replace(
            runners[donor.runner], hierarchy_schema=None, corridor_stage=None)
        monkeypatch.setattr(source_cli, "preparation_runners", lambda: runners)
        source = grafted.source_id
    plan, exp, config, source = _authored(tmp_path, source)
    seen = {"prepare": [], "decisions": [], "forecast": []}
    monkeypatch.setattr(stage_reuse, "engine_source_identity", lambda: dict(ENGINE))
    from gpuwm import fetch_routes, prepared_domain_tree_forecast as runner

    def fetch(arguments, *_args, **_kwargs):
        root = Path(arguments[arguments.index("--out") + 1])
        root.mkdir(parents=True, exist_ok=True)
        (root / fetch_routes.PREP_ARGUMENTS_NAME).write_text(json.dumps({
            "schema": fetch_routes.PREP_ARGUMENTS_SCHEMA,
            "source": source, "prep_source": source,
            "argv": ["--source", source], "unbound_supplement_roles": [],
            "member": None, "member_set": None,
        }), encoding="utf-8")
        return {}

    def publish(root, *, corridor):
        _tree(root)
        _tree_bundle(root, source="mapped", domains=2)
        _mapped_evidence(root, schema=source_schemas()[source],
                         profile=packaged_profile_sources()[source])
        proof_path = root / "proof.json"
        proof = json.loads(proof_path.read_text())
        if corridor:
            proof["statics_corridor"] = {"domains": {"d02": {
                "cache": {"path": "statics-corridor-d02.npz",
                          "sha256": hashlib.sha256(
                              (root / "statics-corridor-d02.npz").read_bytes()).hexdigest()}}}}
        else:
            (root / "statics-corridor-d02.npz").unlink()
        proof_path.write_text(json.dumps(proof), encoding="utf-8")

    def prep(arguments):
        seen["prepare"].append(list(arguments))
        assert arguments.count("--statics-corridor") == 1
        publish(Path(arguments[arguments.index("--output-root") + 1]), corridor=True)

    actual_prepare = runplan._prepare_stage

    def prepare_stage(root, *, arguments, stated, run, built=None):
        if old_bundle and not root.exists():
            publish(root, corridor=False)
            stage_reuse.write_binding(
                root, arguments=[a for a in arguments if a != "--statics-corridor"],
                stated=stated)
            seen["old_proof"] = (root / "proof.json").read_bytes()
        result = actual_prepare(root, arguments=arguments, stated=stated,
                                run=run, built=built)
        seen["decisions"].append(result)
        assert stage_reuse.decide(root, arguments=arguments, stated=stated)["decision"] == stage_reuse.REUSE
        return result

    def forecast(argv, *, observer):
        seen["forecast"].append(list(argv))
        root = Path(argv[argv.index("--prepared-root") + 1])
        proof = root / "proof.json"
        assert argv[argv.index("--preparation-receipt-sha256") + 1] == hashlib.sha256(proof.read_bytes()).hexdigest()
        assert argv[argv.index("--experiment-config") + 1] == str(config)
        assert "d02" in json.loads(proof.read_text())["statics_corridor"]["domains"]
        # A new catalog name reusing the same scientific mapping resolves to
        # the original canonical profile by its bytes, never a forged identity.
        assert stage_cli.resolve_bundle(root)["source"] == stage_cli.packaged_source_of(root)
        return 0

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(runplan, "_run_prep", prep)
    monkeypatch.setattr(runplan, "_prepare_stage", prepare_stage)
    monkeypatch.setattr(runner, "main", forecast)
    monkeypatch.setattr(runplan, "_chain_render", lambda *_args, **_kwargs: {})
    observer = SimpleNamespace(enter_stage=lambda *_a, **_k: None,
                               finish_stage=lambda **_k: None,
                               arm_first_products=lambda *_a, **_k: None)
    # Invoke the real staged chain directly so admission cannot mask a missing
    # preparation flag. The preceding control tests admission independently.
    for _ in range(2):
        runplan._staged_chain(plan, exp=exp, config_path=config,
                             run_dir=plan.run_dir, observer=observer)
    assert len(seen["prepare"]) == 1
    assert len(seen["forecast"]) == 2
    assert [d["decision"] for d in seen["decisions"]] == [
        stage_reuse.REBUILD if old_bundle else stage_reuse.BUILD, stage_reuse.REUSE]
    if old_bundle:
        first = seen["decisions"][0]
        assert any(d["field"] == "arguments --statics-corridor" for d in first["differences"])
        retained = Path(first["superseded"]["path"])
        assert (retained / "proof.json").read_bytes() == seen["old_proof"]
