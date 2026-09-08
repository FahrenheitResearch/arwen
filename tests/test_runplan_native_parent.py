"""Real outer run-plan -> go -> native run-plan telemetry, with GPU seam replaced.

These are orchestration regressions, not scientific integration tests. The
actual native planners, manifests, streams, observer and supervisor heartbeat
execute; only device/resource gates and runtime integration use CPU fixtures.
"""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gpuwm import capabilities, go_cli, runplan, runtime
from test_case_data import make_case_toml


def _run(tmp_path, monkeypatch, *, fail=False, async_outputs=False):
    config = make_case_toml(tmp_path)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(go_cli, "geography_refusal", lambda *a: None)
    monkeypatch.setattr(go_cli, "memory_gate", lambda *a, **k: {
        "verdict": "CPU test fixture", "refuse": False, "warn": False})
    monkeypatch.setattr(runplan, "_run_fetch", lambda *a, **k: pytest.fail("No test fetch is authorized"))
    samples = []

    def integrate(exp, data, outdir, *, progress_callback, **kwargs):
        # The callback is the actual native observer; no fabricated event
        # records or manually authored completion reports enter this path.
        progress_callback.preparing("prepare-case")
        progress_callback.preparing("cold-start-wrfout")
        paths = []

        def commit(domain, step):
            path = Path(outdir) / f"wrfout_fixture_d{domain:02d}_{step}"
            path.write_bytes(b"CPU orchestration fixture; not a scientific field")
            paths.append(path)
            progress_callback.output_committed(domain=domain,
                valid_time=exp.start_time + timedelta(seconds=60 * step), path=path)

        for step in range(1, 4):
            if async_outputs:
                workers = [threading.Thread(target=commit, args=(domain, step)) for domain in (1, 2, 3)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()
            else:
                commit(1, step)
            progress_callback(model_elapsed_seconds=step * 60., outer_step=step,
                last_durable_wrfout=paths[-1], last_checkpoint=None,
                phase="post-d01-sync", step_wall_seconds=.004)
            samples.append(json.loads((tmp_path / "outer" / "run-progress.json").read_text()))
            if fail:
                raise RuntimeError("native fixture refuses after its first durable step")
        return SimpleNamespace(wrfout_paths=tuple(paths), completed_seconds=180., nan_free=True)

    monkeypatch.setattr(runtime, "run_experiment", integrate)
    plan = runplan.build_plan({"schema": runplan.PLAN_SCHEMA, "name": "outer-fixture",
        "route": "prepared", "config": {"path": str(config)},
        "output_root": str(tmp_path / "outer"), "run_options": {"render_products": "none"}},
        source="native parent regression", base_dir=tmp_path, sha256="d" * 64)
    with runplan.EventStream(plan.run_dir / runplan.EVENTS_FILENAME, mirror=None) as events:
        code = runplan.execute_plan(plan, events=events)
    return code, plan, runplan.read_events(plan.run_dir / runplan.EVENTS_FILENAME), samples


@pytest.mark.parametrize("async_outputs", [False, True])
def test_native_front_door_relays_every_commit_and_progress_before_completion(tmp_path, monkeypatch, async_outputs):
    code, plan, outer, samples = _run(tmp_path, monkeypatch, async_outputs=async_outputs)
    assert code == 0, outer[-1]
    manifest = json.loads((plan.run_dir / runplan.MANIFEST_FILENAME).read_text())
    binding = manifest["native_run"]
    native_path = Path(binding["events_path"])
    native = runplan.read_events(native_path)
    lines = native_path.read_bytes().splitlines(keepends=True)
    assert binding["pid"] == manifest["pid"]
    assert binding["run_id"] != manifest["run_id"]
    assert Path(binding["run_dir"]).parent == plan.run_dir / "chain"
    assert hashlib.sha256(Path(binding["manifest_path"]).read_bytes()).hexdigest() == binding["manifest_sha256"]
    forwarded = [row for row in outer if row.get("native_source")]
    assert [row["native_source"]["sequence"] for row in forwarded] == list(range(1, len(native) + 1))
    for row in forwarded:
        source = row["native_source"]
        original = native[source["sequence"] - 1]
        assert source["event_sha256"] == hashlib.sha256(lines[source["sequence"] - 1]).hexdigest()
        assert source["emitted_unix_ms"] == original["emitted_unix_ms"]
        if original["event"] in {"output_committed", "model_progress", "stage_started", "stage_finished"}:
            for key, value in original.items():
                if key not in runplan._ENVELOPE_KEYS:
                    assert row[key] == value
    count = 9 if async_outputs else 3
    assert sum(row["event"] == "output_committed" for row in outer) == count
    assert outer[-1]["event"] == native[-1]["event"] == "completed"
    assert outer[-1]["summary"] == native[-1]["summary"]
    assert outer[-1]["outputs_committed"] == count
    assert outer[-1]["summary"] == {"wrfout_count": count, "completed_seconds": 180., "nan_free": True,
                                    "restarted": False}
    assert [row["model_elapsed_seconds"] for row in samples] == [60., 120., 180.]
    assert [row["outer_step"] for row in samples] == [1, 2, 3]
    beat = json.loads((plan.run_dir / "run-progress.json").read_text())
    assert (beat["status"], beat["model_elapsed_seconds"], beat["outer_step"]) == ("complete", 180., 3)
    assert Path(beat["last_durable_wrfout"]).is_file()
    assert runplan._PREPARED_PARENT.get() is None


def test_native_failure_preserves_observed_progress_and_reason_without_completion(tmp_path, monkeypatch):
    code, plan, events, samples = _run(tmp_path, monkeypatch, fail=True)
    assert code == 1
    assert events[-1]["event"] == "failed"
    assert "native fixture refuses" in events[-1]["message"]
    assert not any(row["event"] == "completed" for row in events)
    assert any(row.get("code") == "native_producer_failed" for row in events)
    beat = json.loads((plan.run_dir / "run-progress.json").read_text())
    assert (beat["status"], beat["model_elapsed_seconds"], beat["outer_step"]) == ("failed", 60., 1)
    assert runplan._PREPARED_PARENT.get() is None


@pytest.mark.parametrize("mismatch", ["config", "directory", "source"])
def test_native_producer_must_match_the_owned_chain_and_exact_config(tmp_path, mismatch):
    config = make_case_toml(tmp_path)
    child_config = config
    if mismatch == "config":
        child_config = tmp_path / "different.toml"
        child_config.write_text(config.read_text().replace("3600.0", "7200.0"))
    chain = tmp_path / "outer" / "chain"
    directory = chain / "native" if mismatch != "directory" else tmp_path / "unrelated"
    plan = runplan.build_plan({"schema": runplan.PLAN_SCHEMA, "name": "child-fixture",
        "route": "experiment", "config": {"path": str(child_config)}, "output_root": str(directory)},
        source=f"gpuwm go {config}" if mismatch != "source" else "unrelated invocation",
        base_dir=tmp_path, sha256="a" * 64)
    with runplan.EventStream(tmp_path / "outer" / runplan.EVENTS_FILENAME, mirror=None) as outer:
        relay = runplan._PreparedRunRelay(runplan.RunObserver(outer), chain, config)
        with runplan.EventStream(directory / runplan.EVENTS_FILENAME, mirror=None) as child:
            with pytest.raises(runplan.PlanError, match="owned chain and exact configuration"):
                relay.attach(plan, child, directory / runplan.MANIFEST_FILENAME)
        assert outer.sequence == 0
