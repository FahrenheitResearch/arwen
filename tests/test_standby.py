"""The standby contract against an object store that publishes after warming."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest

from gpuwm.standby import HTTPObjectStore, PostedObject, run_standby


@contextmanager
def appearing_store(payload, *, delay=0.15, incomplete=False):
    state = {"appears": float("inf"), "gets": [], "ready": None, "delay": delay}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            if time.monotonic() < state["appears"]:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", '"test-version"')
            self.send_header("Last-Modified", "Wed, 21 Oct 2026 06:52:00 GMT")
            self.end_headers()

        def do_GET(self):
            assert self.headers["If-Match"] == '"test-version"'
            lo, hi = map(int, self.headers["Range"][6:].split("-"))
            state["gets"].append((lo, hi))
            body = payload[lo:hi + 1]
            if incomplete and hi == len(payload) - 1:
                body = body[:-4] + b"nope"
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(payload)}")
            self.send_header("ETag", '"test-version"')
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/arbitrary-analysis", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class Backend:
    def __init__(self, state, *, fail=False, no_step=False):
        self.state, self.fail, self.no_step = state, fail, no_step
        self.calls = Counter()
        self.resident = object()

    def prestage(self, receipt):
        for name in ("doctor", "kernels", "static", "boundaries", "construct"):
            with receipt.stage(name, parallelism="stub worker"):
                self.calls[name] += 1
        self.state["ready"] = time.monotonic()
        self.state["appears"] = time.monotonic() + self.state["delay"]
        self.saved = self.resident

    def launch(self, path, receipt):
        self.state["launch"] = time.monotonic()
        self.calls["launch"] += 1
        assert self.resident is self.saved
        assert Path(path).read_bytes().startswith(b"GRIB")
        if self.fail:
            raise RuntimeError("forecast construct failed")
        with receipt.stage("decode", parallelism="stub worker"):
            pass
        with receipt.stage("start_state", parallelism="stub GPU"):
            pass
        if not self.no_step:
            receipt.first_step(outer_step=1, model_elapsed_seconds=20)
        return "forecast/report.json"

    def close(self):
        self.calls["close"] += 1


def test_prestage_before_object_launch_within_poll_no_repeated_stages(tmp_path):
    payload = b"GRIB" + b"x" * 100 + b"7777"
    with appearing_store(payload, delay=1.0) as (url, state):
        backend = Backend(state)
        receipt = run_standby(backend, urls=[url], outdir=tmp_path,
                              poll_s=3.0, timeout_s=10)
    assert state["ready"] < state["appears"] <= state["launch"]
    assert state["launch"] - state["appears"] < 3.0
    assert all(n == 1 for n in backend.calls.values())
    assert receipt["status"] == "DONE"
    assert receipt["object_seen_to_first_step_seconds"] < 3.0
    assert receipt["object"]["last_modified"] == "Wed, 21 Oct 2026 06:52:00 GMT"
    assert all(s["start_wall"] <= s["end_wall"] and s["parallelism"] for s in receipt["stages"])
    assert json.loads((tmp_path / "standby-receipt.json").read_text()) == receipt


@pytest.mark.parametrize("failure", ["exception", "no-step", "incomplete"])
def test_warmed_failure_is_durable_and_loud(tmp_path, failure):
    with appearing_store(b"GRIB" + b"x" * 100 + b"7777",
                         incomplete=failure == "incomplete") as (url, state):
        backend = Backend(state, fail=failure == "exception", no_step=failure == "no-step")
        with pytest.raises((RuntimeError, TimeoutError)):
            run_standby(backend, urls=[url], outdir=tmp_path, poll_s=0.05, timeout_s=0.3)
    receipt = json.loads((tmp_path / "standby-receipt.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["prestage_complete_wall"]
    assert backend.calls["close"] == 1
    if failure == "incomplete":
        assert backend.calls["launch"] == 0


def test_parallel_ranges_reassemble_exactly(tmp_path):
    payload = b"GRIB" + bytes(range(256)) * (70 * 1024) + b"7777"
    with appearing_store(payload) as (url, state):
        state["appears"] = 0
        store = HTTPObjectStore(workers=3)
        obj = store.head(url)
        target = tmp_path / "data.grib2"
        result = store.fetch(obj, target)
    assert result["workers"] == 3
    assert target.read_bytes() == payload
    ranges = sorted(state["gets"])
    assert ranges[0][0] == 0 and ranges[-1][1] == len(payload) - 1
    assert all(a[1] + 1 == b[0] for a, b in zip(ranges, ranges[1:]))


def test_in_process_adapter_keeps_context_and_does_not_spawn(monkeypatch):
    from types import SimpleNamespace
    from contextvars import ContextVar
    from gpuwm import source_cli
    import importlib
    value = ContextVar("resident-test", default=None)
    token = value.set(object())
    resident = value.get()
    seen = []
    def main(argv):
        seen.append((argv, value.get()))
        return 0
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(main=main))
    monkeypatch.setattr(source_cli.command_output, "run_adapter_command",
                        lambda *a, **k: pytest.fail("spawned an adapter"))
    try:
        with source_cli.in_process_adapters():
            result = source_cli._run_adapter_command(["python", "-m", "gpuwm.arbitrary", "--arg"])
        assert result.returncode == 0
        assert seen == [(["--arg"], resident)]
        assert source_cli._IN_PROCESS_ADAPTERS.get() is False
    finally:
        value.reset(token)


def test_arbitrary_recipe_cli_uses_table_source_and_nominal_clock(tmp_path, monkeypatch):
    """A newly named recipe/source reaches the door without model branches."""
    import argparse
    import tomllib
    from types import SimpleNamespace
    from gpuwm import standby_cli, standby_runtime, fetch_routes
    from gpuwm.standby_cli import register_cli
    recipe = tmp_path / "arbitrary.toml"
    recipe.write_text('[experiment]\nname="arbitrary"\nrun_seconds=3600\n'
                      'start_time=2026-10-20T00:00:00\n'
                      '[fetch]\nsource="boundaries"\ninitial_source="arbitrary-native"\n')
    monkeypatch.setattr(fetch_routes, "route_for",
                        lambda source: SimpleNamespace(cycle_hours=(0, 6, 12, 18)))
    with appearing_store(b"GRIB" + b"x" * 100 + b"7777") as (url, state):
        obj = SimpleNamespace(relpath="analysis", urls=lambda ladder: (url,))
        plan = SimpleNamespace(primary_files=[Path("analysis")], objects=[obj], ladder=())
        def resolve(source, *, cycle, hours):
            assert source == "arbitrary-native"
            assert cycle == datetime(2026, 10, 21, 6)
            assert hours == 0
            return plan
        monkeypatch.setattr(fetch_routes, "resolve_request", resolve)
        def backend(args, planned, watched):
            assert planned is plan and watched is obj
            config = tomllib.loads(args.config.read_text())
            assert config["experiment"]["start_time"] == datetime(2026, 10, 21, 6)
            assert config["experiment"]["run_seconds"] == 3600
            value = Backend(state)
            value.store = HTTPObjectStore(timeout=1)
            return value
        monkeypatch.setattr(standby_runtime, "PreparedStandby", backend)
        parser = argparse.ArgumentParser()
        register_cli(parser.add_subparsers())
        args = parser.parse_args(["standby", "--recipe", str(recipe), "--cycle", "2026-10-21T06",
                                  "--hydro", "off", "--poll-s", "0.1", "--outdir", str(tmp_path / "run")])
        assert args.func(args) == 0
    receipt = json.loads((tmp_path / "run" / "standby-receipt.json").read_text())
    assert receipt["recipe"] == "arbitrary"
    assert receipt["hydro"] is not None


def test_next_cycle_uses_table_cadence(monkeypatch):
    from types import SimpleNamespace
    from gpuwm import fetch_routes
    from gpuwm.standby_cli import resolve_cycle
    monkeypatch.setattr(fetch_routes, "route_for",
                        lambda source: SimpleNamespace(cycle_hours=(0, 12)))
    assert resolve_cycle("next", source="arbitrary",
        now=datetime(2026, 10, 21, 12, tzinfo=timezone.utc)) == datetime(2026, 10, 22)
    with pytest.raises(ValueError, match="no 06Z cycle"):
        resolve_cycle("2026-10-21T06", source="arbitrary")


def test_native_stage_receipt_separates_decode_and_state(tmp_path):
    from gpuwm.standby import StandbyReceipt
    from gpuwm.progress import event_sink, prep_stage
    receipt = StandbyReceipt(tmp_path / "receipt.json")
    with event_sink(receipt.preparation_event):
        for name in ("initial_decode", "start_state"):
            with prep_stage(name, backend="parallel Rust", stderr=False):
                pass
    rows = receipt.document["stages"]
    assert [row["name"] for row in rows] == ["initial_decode", "start_state"]
    assert all(row["status"] == "DONE" and row["start_wall"] <= row["end_wall"]
               and row["parallelism"] == "parallel Rust" for row in rows)


def test_prepared_backend_keeps_deferred_prep_in_process(tmp_path, monkeypatch):
    """Exercise the real orchestration with CPU seams at the GPU boundary."""
    import sys
    from types import SimpleNamespace
    from contextlib import nullcontext
    from gpuwm import capabilities, doctor, kernel_prewarm, mapped_engine_bridge, runplan, source_cli
    from gpuwm.core import preflight, clock
    monkeypatch.setattr(clock, "resolve_clock", lambda *a, **k: object())
    monkeypatch.setattr(clock, "build_schedule", lambda *a, **k: object())
    from gpuwm.standby import StandbyReceipt
    from gpuwm.standby_runtime import PreparedStandby
    from gpuwm.region_recipe import RECIPE_TABLE
    config = RECIPE_TABLE.parent / "hrrr_v4_gsd41.toml"
    calls = Counter()
    monkeypatch.setattr(doctor, "collect_checks", lambda sources: [])
    monkeypatch.setattr(capabilities, "require_for_command", lambda name: None)
    monkeypatch.setattr(mapped_engine_bridge, "require_engine", lambda: "native-engine")
    monkeypatch.setattr(preflight, "check_main", lambda args: 0)
    def warm(*args, **kwargs):
        calls["warm"] += 1
        return {"errors": []}
    monkeypatch.setattr(kernel_prewarm, "warm_static", warm)
    monkeypatch.setattr(kernel_prewarm, "first_step_ready", lambda *a, **k: True)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda device: nullcontext(), runtime=SimpleNamespace(
            getDeviceCount=lambda: 1, free=lambda value: None))))
    def chain(plan, **kwargs):
        calls["prepare"] += 1
        assert plan.run_options["as_posted"] is False
        assert kwargs["prepare_only"] is True and kwargs["defer_initial"] is True
        assert source_cli._IN_PROCESS_ADAPTERS.get() is True
        return object()
    monkeypatch.setattr(runplan, "_staged_chain", chain)
    class Handle:
        receipt = {"status": "READY"}
        def wait_ready(self, timeout):
            self.prepared = self.prepare()
            return True
        def close(self):
            calls["close"] += 1
    def prestage(prepare, *, devices):
        assert devices == [0]
        handle = Handle()
        handle.prepare = prepare
        return handle
    monkeypatch.setitem(sys.modules, "gpuwm.boundary_prestage",
                        SimpleNamespace(prestage_boundaries=prestage))
    args = SimpleNamespace(outdir=tmp_path, config=config, fetch_workers=3,
        poll_s=3, devices=None, products="none", geog_root=None,
        data_dir=None, wait_timeout_s=10)
    backend = PreparedStandby(args, SimpleNamespace(source_id="arbitrary"), None)
    receipt = StandbyReceipt(tmp_path / "receipt.json")
    try:
        backend.prestage(receipt)
        assert backend.handle.prepared is not None
        assert calls["prepare"] == calls["warm"] == 1
    finally:
        backend.close()
    assert calls["close"] == 1


def test_execution_plan_rejects_changed_config_or_boundary_cadence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gpuwm.standby_runtime import PreparedExecution
    from gpuwm.file_digest import sha256_file
    from gpuwm import kernel_prewarm
    config = tmp_path / "experiment.toml"
    config.write_text("original configuration")
    monkeypatch.setattr(kernel_prewarm, "first_step_ready", lambda *a, **k: True)
    plan = PreparedExecution(sha256_file(config), 3600, object(), object(), ({},))
    inputs = SimpleNamespace(experiment_config=config, boundary_interval_seconds=3600)
    plan.verify(inputs)
    inputs.boundary_interval_seconds = 7200
    with pytest.raises(ValueError, match="cadence"):
        plan.verify(inputs)
    inputs.boundary_interval_seconds = 3600
    config.write_text("changed configuration")
    with pytest.raises(ValueError, match="configuration"):
        plan.verify(inputs)
