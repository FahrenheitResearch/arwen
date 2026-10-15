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


def test_unbound_prepared_route_cannot_hold_cards_during_preparation(tmp_path):
    bound, _, _ = prepared(tmp_path)
    job = deepcopy(bound["jobs"][0])
    planned = reservations((12,))
    planned["jobs"] = [job]
    with pytest.raises(ValueError, match="suite_prepared_root_missing.*holding queued forecast cards"):
        bind_plan(planned, deployment(base_config="base.toml", source_receipt="prepared.json",
            source_bindings={"public": {"selected.fetch.source": "rap-native"}}))


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
