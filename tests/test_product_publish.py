"""Publication only advertises closed, immutable product generations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from gpuwm.product_publish import LocalPublisher, PRIORITY_PRODUCTS, SCHEMA

CYCLE = datetime(2026, 10, 21, tzinfo=timezone.utc)


def pictures(root, slugs=PRIORITY_PRODUCTS):
    paths = []
    for slug in slugs:
        path = root / "d01-3km" / slug / "20261021" / "frame.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(slug.encode())
        paths.append(path)
    return paths


def publish(publisher, source, paths, *, hour=3, complete=False, domain=1):
    return publisher.publish(source, paths, domain=domain,
        valid_time=CYCLE + timedelta(hours=hour),
        hour_closed_at=CYCLE.isoformat(), render_start=CYCLE.isoformat(),
        render_end=(CYCLE + timedelta(seconds=1)).isoformat(), complete=complete)


def test_manifest_never_names_a_missing_file_and_priority_bytes_are_immutable(tmp_path, monkeypatch):
    from gpuwm import product_publish
    source, root = tmp_path / "render", tmp_path / "www"
    paths = pictures(source)
    publisher = LocalPublisher(root, cycle=CYCLE, hydro="off")
    atomic_text = product_publish._atomic_text
    publications = []

    def inspect(target, text):
        if target.name == "manifest.json":
            manifest = json.loads(text)
            for hour in manifest["hours"]:
                for product in hour["products"]:
                    assert (root / product["url"]).is_file()
                assert (root / hour["index"]).is_file()
            publications.append(manifest)
        atomic_text(target, text)

    monkeypatch.setattr(product_publish, "_atomic_text", inspect)
    row = publish(publisher, source, paths)
    original = (root / row["products"][0]["url"]).read_bytes()
    paths[0].write_bytes(b"full render overwrites scratch")
    assert (root / row["products"][0]["url"]).read_bytes() == original
    full = publish(publisher, source, paths, complete=True)
    assert full["online_at"] == row["online_at"]
    assert full["all_products_online_at"] >= row["online_at"]
    assert full["online_lag_seconds"] >= 0
    assert full["hydro"] == "off"
    assert full["priority_present"] == list(PRIORITY_PRODUCTS)
    assert full["products_count"] == 5
    assert publications[-1]["schema"] == SCHEMA
    assert (root / "index.html").is_file()


def test_incomplete_priority_set_never_claims_online(tmp_path):
    publisher = LocalPublisher(tmp_path / "www", cycle=CYCLE)
    source = tmp_path / "render"
    row = publish(publisher, source, pictures(source, PRIORITY_PRODUCTS[:-1]), complete=True)
    assert row["online_at"] is None
    assert row["online_lag_seconds"] is None


def test_analysis_has_no_previous_hour_precipitation(tmp_path):
    publisher = LocalPublisher(tmp_path / "www", cycle=CYCLE)
    source = tmp_path / "render"
    paths = pictures(source, [s for s in PRIORITY_PRODUCTS if s != "qpf_1h"])
    assert publish(publisher, source, paths, hour=0)["online_at"] is not None
    assert publish(publisher, source, paths, hour=1)["online_at"] is None


def test_independent_publishers_keep_every_hour_and_cycle(tmp_path):
    root, source = tmp_path / "www", tmp_path / "render"
    paths = pictures(source)

    def one(hour):
        return publish(LocalPublisher(root, cycle=CYCLE), source, paths, hour=hour)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, range(4)))
    publish(LocalPublisher(root, cycle=CYCLE - timedelta(days=1)), source, paths)
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["hours"]) == 5


def test_copy_failure_leaves_previous_manifest_intact(tmp_path, monkeypatch):
    from gpuwm import product_publish
    root, source = tmp_path / "www", tmp_path / "render"
    paths = pictures(source)
    publisher = LocalPublisher(root, cycle=CYCLE)
    publish(publisher, source, paths)
    before = (root / "manifest.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected full disk")

    monkeypatch.setattr(product_publish.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="full disk"):
        publish(publisher, source, paths, hour=4)
    assert (root / "manifest.json").read_bytes() == before


def test_hook_is_off_by_default_and_explicit_command_gets_committed_row(tmp_path, monkeypatch):
    root, source = tmp_path / "www", tmp_path / "render"
    paths = pictures(source)
    calls = []

    def command(argv, **kwargs):
        row = json.loads(kwargs["input"])
        manifest = json.loads((root / "manifest.json").read_text())
        assert row in manifest["hours"]
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", command)
    publish(LocalPublisher(root, cycle=CYCLE), source, paths)
    assert calls == []
    publish(LocalPublisher(root, cycle=CYCLE, command='local-hook "a path"'), source, paths)
    assert calls[0][0] == ["local-hook", "a path"]
    assert "shell" not in calls[0][1]
    assert calls[0][1]["env"]["GPUWM_PUBLISH_ROOT"] == str(root)


def test_live_worker_publishes_priority_before_launching_full_set(tmp_path):
    from gpuwm.live_products import LiveProducts
    frame = tmp_path / "wrfout_d01_2026-10-21_03_00_00"
    frame.write_bytes(b"test frame identity")
    root, render = tmp_path / "www", tmp_path / "png"
    requests, reports, warnings = [], [], []

    def runner(argv):
        request = argv[argv.index("--products") + 1]
        requests.append(request)
        scratch = Path(argv[argv.index("--out") + 1])
        if request == "all":
            row = json.loads((root / "manifest.json").read_text())["hours"][0]
            assert row["online_at"] and not row["complete"]
            assert row["hour_closed_at"]
        pictures(scratch)
        return subprocess.CompletedProcess(argv, 0, "", "")

    live = LiveProducts(dict(render=render, render_products="all", cycle=CYCLE,
                             publish_root=root), runner=runner,
        report=reports.append, warn=lambda *args, **kwargs: warnings.append(args))
    valid = CYCLE + timedelta(hours=3)
    live.frame_submitted(domain=1, valid_time=valid)
    live.frame_committed(domain=1, valid_time=valid, path=frame)
    live.stop(timeout=10)
    assert not warnings
    assert requests == [",".join(PRIORITY_PRODUCTS), "all"]
    assert len(reports) == 1
    row = json.loads((root / "manifest.json").read_text())["hours"][0]
    assert row["forecast_hour"] == 3
    assert row["hour_closed_at"] <= row["render_start"] <= row["render_end"]
    assert row["complete"] and row["online_at"] <= row["all_products_online_at"]


def test_final_render_takes_all_available_cores_after_stepping(tmp_path, monkeypatch):
    from gpuwm import first_products, live_products
    calls = []
    monkeypatch.setattr(live_products, "_available_cpus", lambda: 16)
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.setattr(first_products, "_run_render", lambda command, **kw: calls.append(kw))
    live = live_products.LiveProducts(dict(render=tmp_path / "png"),
        report=lambda row: None, warn=lambda *a, **k: None)
    live._closed, live._active = True, 1
    live._run_render(["renderer"])
    assert calls[0]["env_overrides"] == {"RUSTWX_LIVE_RENDER_SLOTS": "1", "RAYON_NUM_THREADS": "16"}
    assert calls[0]["nice"] == 0


def test_submission_fanout_does_not_call_step_log_twice():
    from gpuwm.prepared_single_domain_forecast import _LandingObservers

    class Observer:
        def __init__(self):
            self.calls = []
        def frame_submitted(self, **event):
            self.calls.append(event)
        def output_committed(self, **event):
            pass

    step_log, publisher = Observer(), Observer()
    fan = _LandingObservers(step_log.output_committed, publisher.output_committed,
                            on_submit=step_log.frame_submitted)
    event = dict(domain=1, valid_time=CYCLE)
    fan.frame_submitted(**event)
    assert step_log.calls == publisher.calls == [event]


def test_publish_command_and_root_reach_runner_without_shell_expansion(tmp_path):
    from gpuwm import go_cli, prepared_single_domain_forecast, prepared_domain_tree_forecast
    command = 'local-hook "path with spaces"'
    root = tmp_path / "web root"
    flags = go_cli._early_render_flags(dict(render=tmp_path / "png", publish_root=root, publish_cmd=command), "all")
    for module in (prepared_single_domain_forecast, prepared_domain_tree_forecast):
        args, _ = module.build_parser().parse_known_args([
            "--source", sorted(prepared_single_domain_forecast.SUPPORTED_SOURCES)[0],
            "--prepared-root", str(tmp_path), "--experiment-config", str(tmp_path / "config"),
            "--experiment-config-sha256", "0" * 64, "--wps-namelist", str(tmp_path / "wps"),
            "--io-mode", "history", "--outdir", str(tmp_path / "run"), *flags])
        assert args.publish_root == root
        assert args.publish_cmd == command
