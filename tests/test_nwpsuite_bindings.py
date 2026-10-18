"""Planner-to-controller integration, including the ordinary run-plan loader."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import pytest

from gpuwm.nwpsuite.bindings import bind_plan, is_due, materialize, row_at, SourcePending
from gpuwm.nwpsuite.registry import ROOT
from gpuwm.nwpsuite.run import SuiteRunner, apply_plan, load_plan

CYCLE = "2026-10-03T12:00:00Z"


def reservations(hours=(12, 18), members=1):
    settings = dict(priority=1, cycle=dict(every="6h", lead="3h"))
    plan = dict(schema="gpuwm.suite.plan.v1", admitted=True, execution_ready=False,
        suite_sha256="resolved", rows=[dict(id="forecast", box="local", settings=settings)],
        boxes=[dict(id="local", cards=3)], defaults=dict(retention={}), jobs=[])
    for hour in hours:
        for member in range(members):
            offset = hour % 6 * 3600
            plan["jobs"].append(dict(id=f"forecast:{hour:02d}:forecast:{member}", row="forecast", phase="forecast",
                cycle_hour=hour, block_hour=hour//6*6, member=member, ready_offset_s=offset,
                start_s=offset+member*10, end_s=offset+member*10+5, deadline_s=offset+1000,
                box="local", card_ids=[], env={}, settings=settings, bindings=[], dependencies=[],
                argv=[sys.executable, "-c", "pass"]))
    return plan


def deployment(**row):
    return dict(schema="gpuwm.suite.bindings.v1", suite_sha256="resolved", rows=dict(forecast=row))


def test_member_hour_and_placement_are_distinct():
    planned = reservations((0, 1), 2)
    planned["boxes"][0]["launcher_prefix"] = ["test-queue", "{cards}"]
    for job in planned["jobs"]:
        job["settings"] = dict(job["settings"], cycle=dict(every="1h"))
        job["card_ids"] = [(job["cycle_hour"] + job["member"]) % 3]
        job["env"] = {"CUDA_VISIBLE_DEVICES": str(job["card_ids"][0])}
    bound = bind_plan(planned)
    assert [r["id"] for r in bound["rows"]] == ["forecast", "forecast.member-1"]
    row = bound["rows"][0]
    assert row_at(row, "2026-10-03T01:00:00Z")["card_ids"] == [1]
    assert row_at(row, "2026-10-03T01:00:00Z")["deadline"] == 1000
    assert is_due(row, "2026-10-03T01:00:00Z")
    assert not is_due(row, "2026-10-03T02:00:00Z")


def test_binding_document_cannot_change_admission():
    with pytest.raises(ValueError, match="different resolved suite"):
        bind_plan(reservations(), dict(deployment(), suite_sha256="other"))
    with pytest.raises(ValueError, match="forecast changes"):
        bind_plan(reservations(), deployment(run_options={"devices": 8}))
    with pytest.raises(ValueError, match="supplement inventory"):
        bind_plan(reservations(), dict(deployment(), boxes=dict(local=dict(cards=8))))


def test_job_dependencies_expand_all_parent_members():
    planned = reservations(members=2)
    parent = planned["jobs"][:2]
    child = dict(parent[0], id="child:12:forecast:0", row="child", dependencies=[j["id"] for j in parent])
    planned["rows"].append(dict(id="child", box="local", settings=parent[0]["settings"]))
    planned["jobs"].append(child)
    bound = bind_plan(planned)
    assert next(r for r in bound["rows"] if r["id"] == "child")["dependencies"] == ["forecast", "forecast.member-1"]


def prepared(tmp_path):
    planned = reservations((12,))
    job = planned["jobs"][0]
    prefix = "artifacts.forecast.12.0"
    job.update(argv=["gpuwm", "run-plan", "{" + prefix + ".run_plan}"],
        bindings=[prefix + ".run_plan", "selected.fetch.source"],
        config_edits={"experiment.run_seconds": 10800, "experiment.start_time": "{start_time}",
                      "fetch.cycle": "{cycle}", "fetch.hours": 3, "fetch.source": "{selected.fetch.source}",
                      "devices.count": 1},
        run_plan_document=dict(schema="gpuwm.run-plan.v1", name="prepared-test", route="prepared",
            config=dict(path="{derived_config}"), output_root="{prepared.outdir}",
            run_options=dict(devices=1, cycle="{cycle}")),
        source_selection=dict(ic=["public"], requires_prepared_source_receipt=True))
    base = tmp_path / "authority.toml"
    base.write_bytes((ROOT / "configs/recipes/hrrr_v4_gsd41.toml").read_bytes())
    receipt = tmp_path / "prepared.json"
    receipt.write_text(json.dumps(dict(schema="gpuwm.suite.prepared.v1", suite_sha256="resolved",
        row="forecast", cycle=CYCLE, source="public", files=[dict(path=str(base), sha256=hashlib.sha256(base.read_bytes()).hexdigest())])))
    document = deployment(base_config=str(base), source_receipt=str(receipt),
        run_options=dict(prepared_root=str(tmp_path / "native-bundle")),
        source_bindings={"public": {"selected.fetch.source": "rap-native"}})
    bound = bind_plan(planned, document)
    bound["suite_hash"] = "suite"
    return bound, base, receipt


def test_materialized_config_and_runplan_pass_existing_loaders(tmp_path):
    bound, base, _ = prepared(tmp_path)
    runner = SuiteRunner(bound, tmp_path / "state")
    row = row_at(bound["rows"][0], CYCLE)
    context = materialize(row, runner.context(row, CYCLE, 0), bound)
    path = Path(context["artifacts.forecast.12.0.run_plan"])
    generated = json.loads(path.read_text())
    config = tomllib.loads(Path(generated["config"]["path"]).read_text())
    assert config["fetch"]["source"] == "rap-native"
    assert config["experiment"]["run_seconds"] == 10800
    assert config["domain"] == tomllib.loads(base.read_text())["domain"]
    from gpuwm.runplan import load_plan as door_load
    assert door_load(path).route == "prepared"
    assert generated["run_options"]["prepared_root"] == str(tmp_path / "native-bundle")
    assert "cycle" not in generated["run_options"]


def test_unbound_prepared_route_prepares_on_a_zero_card_stage_first(tmp_path):
    """No prepared_root: a CPU prepare stage runs ahead, never the forecast's cards."""
    bound, _, _ = prepared(tmp_path)
    job = deepcopy(bound["rows"][0]["materialization"])
    job["recipe_reference"] = "configs/recipes/hrrr_v4_gsd41.toml"
    planned = reservations((12,))
    planned["jobs"] = [job]
    planned["boxes"][0]["launcher_prefix"] = ["test-queue"]
    bound = bind_plan(planned)
    rows = {r["id"]: r for r in bound["rows"]}
    assert set(rows) == {"forecast", "forecast.prepare"}
    stage, forecast = row_at(rows["forecast.prepare"], CYCLE), row_at(rows["forecast"], CYCLE)
    assert stage["cards"] == 0 and stage["card_ids"] == [] and stage["env"] == {}
    assert stage["argv"][1:3] == ["-m", "gpuwm.nwpsuite.prepare"]
    assert forecast["dependencies"] == ["forecast.prepare"] and forecast["auto_prepared"]
    assert forecast["source_receipt"].endswith("forecast/{cycle_compact}/suite-prepared.json")
    assert bound["execution_ready"]


def test_prepare_stage_derives_cycle_config_and_forecast_binds_its_publication(tmp_path):
    from gpuwm.nwpsuite.prepare import publish
    bound, base, _ = prepared(tmp_path)
    job = deepcopy(bound["rows"][0]["materialization"])
    job["recipe_reference"] = "configs/recipes/hrrr_v4_gsd41.toml"
    planned = reservations((12,))
    planned["jobs"] = [job]
    bound = bind_plan(planned, deployment(base_config=str(base),
                                          source_bindings={"public": {"selected.fetch.source": "hrrr"}}))
    bound["suite_hash"] = "suite"
    rows = {r["id"]: r for r in bound["rows"]}
    runner = SuiteRunner(bound, tmp_path / "state")
    stage = row_at(rows["forecast.prepare"], CYCLE)
    context = materialize(stage, runner.context(stage, CYCLE, 0), bound)
    config = tomllib.loads(Path(context["derived_config"]).read_text())
    assert config["fetch"]["cycle"] == "2026-10-03T12" and config["fetch"]["source"] == "hrrr"
    assert config["experiment"]["run_seconds"] == 10800
    from gpuwm.nwpsuite.run import expand
    argv = expand(stage["argv"], context)
    receipt = Path(argv[argv.index("--receipt") + 1])
    forecast = row_at(rows["forecast"], CYCLE)
    with pytest.raises(SourcePending):
        materialize(forecast, runner.context(forecast, CYCLE, 0), bound)
    bundle = tmp_path / "bundle"; bundle.mkdir()
    published = bundle / "experiment.toml"; published.write_bytes(Path(context["derived_config"]).read_bytes())
    namelist = bundle / "namelist.wps"; namelist.write_text("&share\n/\n")
    publish(dict(status="PREPARED", prepared_root=str(bundle), experiment_config=str(published),
                 wps_namelist=str(namelist)), receipt=receipt, suite_sha256="resolved", row="forecast",
            cycle=CYCLE, source="public", config=context["derived_config"])
    bound_context = materialize(forecast, runner.context(forecast, CYCLE, 0), bound)
    generated = json.loads(Path(bound_context["artifacts.forecast.12.0.run_plan"]).read_text())
    assert generated["config"]["path"] == str(published)
    assert generated["run_options"]["prepared_root"] == str(bundle.resolve())
    assert generated["run_options"]["wps_namelist"] == str(namelist.resolve())
    assert "cycle" not in generated["run_options"]
    namelist.write_text("&share\n changed\n/\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        materialize(forecast, runner.context(forecast, CYCLE, 0), bound)


def test_go_door_forecast_binds_the_prepared_bundle_not_a_cycle(tmp_path):
    bound, base, _ = prepared(tmp_path)
    job = deepcopy(bound["rows"][0]["materialization"])
    job.update(recipe_reference="configs/recipes/hrrr_v4_gsd41.toml",
               argv=["gpuwm", "go", "{derived_config}", "--cycle", "{cycle}", "--devices", "1"],
               bindings=["selected.fetch.source"])
    planned = reservations((12,))
    planned["jobs"] = [job]
    bound = bind_plan(planned)
    forecast = row_at(next(r for r in bound["rows"] if r["id"] == "forecast"), CYCLE)
    assert forecast["argv"] == ["gpuwm", "go", "{derived_config}", "--prepared-root", "{prepared.root}",
                                "--wps-namelist", "{prepared.wps_namelist}", "--devices", "1",
                                "--outdir", "{out}"]
    stage = row_at(next(r for r in bound["rows"] if r["id"] == "forecast.prepare"), CYCLE)
    assert stage["argv"][-1] == "--go-forecast"


def test_boundary_ladder_the_go_door_cannot_honour_is_named_before_preparing(tmp_path):
    bound, _, _ = prepared(tmp_path)
    job = deepcopy(bound["rows"][0]["materialization"])
    job["recipe_reference"] = "configs/recipes/hrrr_v4_gsd41.toml"
    job["source_selection"] = dict(job["source_selection"], lbc=["rap"])
    planned = reservations((12,))
    planned["jobs"] = [job]
    with pytest.raises(ValueError, match=r"suite_prepared_lbc: forecast: .*\(public\).*\['rap'\]"):
        bind_plan(planned)


def test_unbound_suite_names_every_row_once(tmp_path):
    planned = reservations((0, 6))
    other = deepcopy(planned["jobs"][0]); other.update(id="hex:00:forecast:0", row="hex", argv=["door", "{prepared.mesh}"])
    planned["jobs"].append(other)
    planned["rows"].append(dict(id="hex", box="local", settings=other["settings"]))
    planned["jobs"][0]["argv"] = ["door", "{prepared.init}"]
    planned["jobs"][1]["argv"] = ["door", "{prepared.init}"]
    with pytest.raises(ValueError) as refused:
        bind_plan(planned)
    said = str(refused.value).split(" | ")
    assert said == ["suite_execution_unbound: forecast: ['prepared.init']",
                    "suite_execution_unbound: hex: ['prepared.mesh']"]


def test_prepare_door_publishes_only_a_prepared_report(tmp_path):
    from gpuwm.nwpsuite.prepare import prepared_line, publish
    assert prepared_line("go: fetch\n{\"status\": \"FAILED\"}\n") is None
    line = prepared_line('go: prepare\n{"status": "PREPARED", "prepared_root": "/b"}\n')
    assert line["prepared_root"] == "/b"
    config = tmp_path / "c.toml"; config.write_text('[fetch]\nsource = "hrrr"\n')
    with pytest.raises(ValueError, match="no WPS namelist"):
        publish(dict(status="PREPARED", prepared_root=str(tmp_path), experiment_config=str(config)),
                receipt=tmp_path / "r.json", suite_sha256="s", row="r", cycle=CYCLE, source="hrrr",
                config=config, go_forecast=True)
    publish(dict(status="PREPARED", prepared_root=str(tmp_path), experiment_config=str(config)),
            receipt=tmp_path / "r.json", suite_sha256="s", row="r", cycle=CYCLE, source="hrrr", config=config)
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["lbc_source"] == "hrrr" and receipt["files"][0]["path"] == str(config.resolve())


@pytest.mark.parametrize("change,error", [("hash", "changed after preparation"), ("cycle", "receipt differs")])
def test_prepared_authority_is_checked(tmp_path, change, error):
    bound, base, receipt = prepared(tmp_path)
    if change == "hash":
        base.write_text(base.read_text() + "\n# changed\n")
    else:
        doc = json.loads(receipt.read_text()); doc["cycle"] = "2026-10-03T18:00:00Z"
        receipt.write_text(json.dumps(doc))
    runner = SuiteRunner(bound, tmp_path / "state")
    row = row_at(bound["rows"][0], CYCLE)
    with pytest.raises(ValueError, match=error):
        materialize(row, runner.context(row, CYCLE, 0), bound)


def test_unpublished_preparation_uses_source_ladder(tmp_path):
    bound, _, receipt = prepared(tmp_path)
    receipt.unlink()  # test owns this file
    runner = SuiteRunner(bound, tmp_path / "state")
    runner.run(cycles=[CYCLE], once=True)
    entry = runner.ledger.state()["jobs"][("forecast", CYCLE)]
    assert entry["phase"] == "skipped" and entry["reason"] == "source_behind"
    assert not (tmp_path / "state/runs/forecast/2026-10-03T120000Z/0/intent.json").exists()


def test_bound_plan_runs_cpu_stages_once_and_pins_both_inputs(tmp_path):
    planned = reservations(members=2)
    command = [sys.executable, "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')", "{out}/score.json"]
    binding = deployment(cpu_stages=[dict(kind="products", argv=[sys.executable, "-c", "pass"]),
                                    dict(kind="verify", argv=command, score_files=["{out}/score.json"])])
    file = tmp_path / "plan.json"; file.write_text(json.dumps(planned))
    bindings = tmp_path / "bindings.json"; bindings.write_text(json.dumps(binding))
    bound = load_plan(file, bindings)
    runner = SuiteRunner(bound, tmp_path / "state", rehearsal=True)
    cycles = [CYCLE, "2026-10-03T18:00:00Z"]
    runner.run(cycles=cycles, poll=.02)
    assert len(runner.ledger.state()["jobs"]) == 8
    assert all(x["phase"] == "completed" for x in runner.ledger.state()["jobs"].values())
    from gpuwm.nwpsuite.status import status_document
    status = status_document(bound, tmp_path / "state")
    assert [r["id"] for r in status["rows"]] == ["forecast", "forecast.products", "forecast.verify"]
    assert len(status["rows"][0]["members"]) == 2
    count = sum(e.get("phase") == "launch_intent" for e in runner.ledger.settle())
    runner.run(cycles=cycles, poll=.02)
    assert sum(e.get("phase") == "launch_intent" for e in runner.ledger.settle()) == count
    assert all(e["suite_hash"] == hashlib.sha256(file.read_bytes()).hexdigest() and
               e["bindings_hash"] == bound["bindings_hash"] for e in runner.ledger.settle() if e.get("phase"))
    original = bound["plan_hash"]
    binding["on_state_change"] = ["true"]; bindings.write_text(json.dumps(binding))
    changed = load_plan(file, bindings)
    assert changed["suite_hash"] == bound["suite_hash"] and changed["plan_hash"] != original
    restored = SuiteRunner(changed, tmp_path / "state")
    assert restored.plan["plan_hash"] == original
    apply_plan(tmp_path / "state", changed)
    restored.tick(cycles)
    assert restored.plan["plan_hash"] == changed["plan_hash"]


def test_da_without_previous_background_skips_without_new_run(tmp_path):
    planned = reservations((12,))
    shadow = deepcopy(planned["jobs"][0])
    shadow.update(id="forecast:12:da-shadow:0", phase="da-shadow", argv=[],
                  bindings=["da.shadow.command"], background_dependency=dict(row="forecast", cycle="previous"))
    planned["jobs"].append(shadow)
    bound = bind_plan(planned, deployment(da_argv=[sys.executable, "-c", "pass"], bindings={"da.shadow.command": "bound"}))
    bound["suite_hash"] = "suite"
    runner = SuiteRunner(bound, tmp_path, rehearsal=True)
    runner.run(cycles=[CYCLE], poll=.02)
    jobs = runner.ledger.state()["jobs"]
    assert jobs[("forecast.da", CYCLE)]["reason"] == "da_failed"
    assert jobs[("forecast", CYCLE)]["phase"] == "completed"


def test_calendar_uses_each_hours_cadence():
    from gpuwm.nwpsuite.bindings import cadence
    planned = reservations((0, 1))
    planned["jobs"][1]["settings"] = dict(planned["jobs"][1]["settings"], cycle=dict(every="1h"))
    bound = bind_plan(planned)
    assert cadence(bound["rows"][0]) == 3600
    assert is_due(bound["rows"][0], "2026-10-03T01:00:00Z")


def test_remote_inventory_cannot_launch_local_workers():
    planned = reservations()
    planned["boxes"][0]["host"] = "ssh://other-box"
    with pytest.raises(ValueError, match="suite_transport_unbound"):
        bind_plan(planned)


def test_lagged_members_reuse_receipts_without_rerunning_parent(tmp_path):
    planned = reservations((12,))
    child = deepcopy(planned["jobs"][0])
    settings = dict(child["settings"], members=dict(perturbed=1, lagged=2), **{"from": "forecast"})
    child.update(row="ensemble", id="ensemble:12:forecast:0", settings=settings,
                 dependencies=[planned["jobs"][0]["id"]])
    planned["jobs"].append(child)
    planned["rows"].append(dict(id="ensemble", box="local", settings=settings))
    command = [sys.executable, "-c", "import json,sys; assert len(json.load(open(sys.argv[1]))['members']) == 2", "{lagged_manifest}"]
    document = dict(deployment(), rows=dict(ensemble=dict(cpu_stages=[dict(kind="products", argv=command)])))
    bound = bind_plan(planned, document); bound["suite_hash"] = "suite"
    runner = SuiteRunner(bound, tmp_path)
    for cycle in ("2026-10-03T00:00:00Z", "2026-10-03T06:00:00Z"):
        receipt = tmp_path / (cycle.replace(":", "") + ".json"); receipt.write_text('{}')
        runner.ledger.append("phase", row="forecast", cycle=cycle, suite_hash="suite", phase="completed",
            result=dict(receipts=[dict(path=str(receipt), sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())]))
    runner.rehearsal = True
    runner.run(cycles=[CYCLE], poll=.02)
    assert runner.ledger.state()["jobs"][("ensemble.products", CYCLE)]["phase"] == "completed"
    launches = [e for e in runner.ledger.settle() if e.get("phase") == "launch_intent"]
    assert all(e["cycle"] == CYCLE for e in launches)
    reuse = next(e for e in runner.ledger.settle() if e["event"] == "reuse")
    assert [m["cycle"] for m in reuse["members"]] == ["2026-10-03T06:00:00Z", "2026-10-03T00:00:00Z"]


def test_actual_toml_planner_connects_to_runtime_without_gpu_launch(tmp_path):
    from gpuwm.nwpsuite.cli import write_suite
    from gpuwm.nwpsuite.planner import plan_suite
    from gpuwm.nwpsuite.validator import load_suite
    suite = load_suite(ROOT / "configs/suites/us-1box.toml")
    suite["system"] = [next(r for r in suite["system"] if r["id"] == "cam-det")]
    path = tmp_path / "country.suite.toml"
    write_suite(path, suite)
    planned = plan_suite(load_suite(path), base=tmp_path)
    assert planned["admitted"]
    doc = dict(schema="gpuwm.suite.bindings.v1", suite_sha256=planned["suite_sha256"],
        boxes={b["id"]: dict(launcher_prefix=["test-queue-must-not-run"]) for b in planned["boxes"]},
        rows={"cam-det": dict(base_config=str(ROOT / "configs/recipes/hrrr_v4_gsd41.toml"),
            source_receipt=str(tmp_path / "unpublished-source.json"),
            run_options=dict(prepared_root=str(tmp_path / "native-bundle")),
            source_bindings={"hrrr": {"selected.fetch.source": "hrrr"}},
            cpu_stages=[dict(kind="products", argv=[sys.executable, "-c", "pass"]),
                        dict(kind="verify", argv=[sys.executable, "-c", "pass"])])})
    bindings = tmp_path / "deployment.json"; bindings.write_text(json.dumps(doc))
    bound = load_plan(path, bindings)
    assert bound["execution_ready"] and len(bound["rows"][0]["variants"]) == 24
    import subprocess
    state = tmp_path / "state"
    result = subprocess.run([sys.executable, "-m", "gpuwm.cli", "suite", "run", str(path),
        "--bindings", str(bindings), "--state-dir", str(state), "--once"], capture_output=True, text=True, timeout=30)
    assert result.returncode in (1, 75), result.stderr
    from gpuwm.nwpsuite.ledger import SuiteLedger
    assert not any(e.get("phase") == "launch_intent" for e in SuiteLedger(state).settle())


def test_hyphenated_row_artifacts_expand_in_the_launched_command():
    # Every shipped row id has a hyphen; the run-plan argv token stayed literal.
    from gpuwm.nwpsuite.run import expand
    context = {"artifacts.cam-det.12.0.run_plan": "/state/run-plan.json"}
    assert expand(["gpuwm", "run-plan", "{artifacts.cam-det.12.0.run_plan}"], context)[2] == "/state/run-plan.json"
    with pytest.raises(ValueError, match="suite_binding_missing: artifacts.cam-det.12.0.run_plan"):
        expand(["{artifacts.cam-det.12.0.run_plan}"], {})
    assert expand(['{"a": 1}', "f{hour:02d}"], {}) == ['{"a": 1}', "f{hour:02d}"]
