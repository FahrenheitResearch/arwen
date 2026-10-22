"""Race clocks, metadata contracts, and concurrent object observation."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest

from gpuwm import race
from gpuwm.race_cli import register_cli


def arguments(*tokens):
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers())
    return parser.parse_args(["race", "hrrr", "--cycle", "2026-10-21T00", *tokens])


def test_dry_study_and_receipt_numbers(tmp_path):
    row = race.recipe_row("hrrr")
    assert row == race.recipe_row("hrrr_v4") == race.recipe_row("hrrr_v4_gsd41")
    cycle = race.cycle_time("2026-10-21T00")
    _, analysis, hours = race.object_names(row, cycle, 48)
    study = tmp_path / "posting-times.json"
    study.write_text(json.dumps({"rows": [
        {"cycle": "2026-10-21T00:00:00Z", "key": analysis, "s3_last_modified": "2026-10-21T00:52:00Z"},
        {"cycle": "2026-10-21T00:00:00Z", "key": hours[48], "s3_last_modified": "2026-10-21T01:46:00Z"},
        {"cycle": "2026-10-20T00:00:00Z", "file": analysis.rsplit("/", 1)[-1], "s3_last_modified": "2026-10-20T00:53:00Z"},
    ]}))
    inputs = tmp_path / "measured.json"
    inputs.write_text(json.dumps({"seconds_per_hour": 56.4, "launch_to_first_step": 60, "publish_lag": 60}))
    args = arguments("--dry", "--posting-times", str(study), "--receipt", str(inputs))
    receipt = dict(mode="dry", recipe=row["id"], cycle=race.stamp(cycle), hours=48, notes=[])
    race.dry_race(args, row, cycle, tmp_path, receipt)
    assert receipt["margin_seconds"] == pytest.approx(412.8)
    assert receipt["noaa_f00_posted_at"] == "2026-10-21T00:52:00Z"
    assert receipt["hourly"][1]["noaa_posted_at"] is None
    assert receipt["required_seconds_per_hour"] == 65
    assert "Prediction only" in (tmp_path / "RACE.md").read_text()
    assert json.loads((tmp_path / "race.json").read_text())["mode"] == "dry"


def test_missing_or_nonfinite_cost_does_not_invent_prediction():
    with pytest.raises(ValueError, match="seconds-per-hour"):
        race.dry_inputs(arguments("--dry"))
    with pytest.raises(ValueError, match="finite"):
        race.dry_inputs(arguments("--dry", "--seconds-per-hour", "nan", "--launch-to-first-step", "60", "--publish-lag", "60"))


def test_watch_observes_delayed_objects_without_blocking_forecast():
    visible = threading.Event()
    requests = []
    class Store(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            requests.append(self.path)
            entries = '<Contents><Key>cycle/f00</Key><LastModified>2026-10-21T00:52:00Z</LastModified><Size>123</Size></Contents>'
            if visible.is_set():
                entries += '<Contents><Key>cycle/f01</Key><LastModified>2026-10-21T00:53:00Z</LastModified><Size>456</Size></Contents>'
            body = ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">' + entries + '</ListBucketResult>').encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    watch = race.ObjectWatch(f"http://127.0.0.1:{server.server_port}", "cycle/", ["cycle/f00", "cycle/f01"], .05)
    watch.thread.start()
    try:
        # The forecast-side main thread continues while the poller is running.
        visible.set()
        deadline = time.monotonic() + 3
        work = 0
        while "cycle/f01" not in watch.snapshot()["objects"] and time.monotonic() < deadline:
            work += 1
            time.sleep(.01)
        snapshot = watch.snapshot()
        assert snapshot["objects"]["cycle/f01"]["last_modified"] == "2026-10-21T00:53:00Z"
        assert snapshot["objects"]["cycle/f01"]["first_seen_at"] != "2026-10-21T00:53:00Z"
        assert all("prefix=cycle%2F" in path for path in requests)
        assert work > 0
    finally:
        watch.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_publish_receipt_filters_cycle_domain_and_uncommitted_hour(tmp_path):
    web = tmp_path / "www"
    web.mkdir()
    cycle = race.cycle_time("2026-10-21T00")
    common = dict(cycle="20261021T000000Z", domain=1, forecast_hour=1, online_at="2026-10-21T00:54:00Z", priority_products=["2m_temperature"])
    (web / "manifest.json").write_text(json.dumps({"schema": "gpuwm.publish.v1", "hours": [common,
        {**common, "forecast_hour": 2, "online_at": None},
        {**common, "cycle": "20261020T000000Z", "online_at": "2026-10-20T00:54:00Z"},
        {**common, "domain": 2, "online_at": "2026-10-21T00:55:00Z"}]}))
    got = race.collect_receipts(tmp_path, cycle=cycle)
    assert got["online"] == {1: "2026-10-21T00:54:00Z"}
    hourly = race.compare_hours({1: "one", 2: "two"}, {"one": {"last_modified": "2026-10-21T00:53:00Z"}}, got["online"])
    assert hourly[0]["margin_seconds"] == -60
    assert hourly[1]["margin_seconds"] is None


def test_recipe_materialization_preserves_physics_and_hourly_output(tmp_path):
    import tomllib
    row = race.recipe_row("hrrr")
    source = tomllib.loads((race.REGISTRY.parent / row["config"]).read_text())
    written = race.materialize_config(row, race.cycle_time("2026-10-21T06"), 48, tmp_path)
    config = tomllib.loads(written.read_text())
    assert config["shared"] == source["shared"]
    assert config["domain"][0]["nx"] == source["domain"][0]["nx"] == 1799
    assert config["domain"][0]["ny"] == source["domain"][0]["ny"] == 1059
    assert config["domain"][0]["history_interval_s"] == 3600
    assert "history_interval2_s" not in config["domain"][0]
    assert config["experiment"]["run_seconds"] == 172800
    assert config["fetch"]["hours"] == 48
    assert config["fetch"]["cadence"] == 1


def test_live_failure_still_writes_race(tmp_path, monkeypatch):
    import sys
    args = arguments("--hours", "1", "--watch-timeout", "3", "--object-endpoint", "http://unused")
    monkeypatch.setattr(race, "command_for", lambda *a: ([sys.executable, "-c", "raise SystemExit(7)"], True, False))
    monkeypatch.setattr(race, "list_objects", lambda *a: {})
    receipt = dict(mode="live", recipe="test", cycle="2026-10-21T00:00:00Z", hours=1, notes=[])
    assert race.live_race(args, race.recipe_row("hrrr"), race.cycle_time(args.cycle), tmp_path, receipt) == 7
    assert receipt["status"] == "failed"
    assert receipt["hourly"][1]["margin_seconds"] is None
    assert (tmp_path / "RACE.md").is_file()


def test_completed_race_can_price_next_box(tmp_path):
    receipt = dict(mode="live", launch_at="2026-10-21T00:52:05Z", noaa_f00_posted_at="2026-10-21T00:52:00Z",
        first_step_at="2026-10-21T00:53:05Z", publish_receipt={"hours": [
            dict(domain=1, forecast_hour=1, hour_closed_at="2026-10-21T00:54:05Z", online_at="2026-10-21T00:54:15Z"),
            dict(domain=1, forecast_hour=2, hour_closed_at="2026-10-21T00:55:05Z", online_at="2026-10-21T00:55:25Z")]})
    path = tmp_path / "measured.json"
    path.write_text(json.dumps(receipt))
    inputs = race.dry_inputs(arguments("--dry", "--receipt", str(path)))
    assert inputs["seconds_per_hour"] == 60
    assert inputs["launch_to_first_step"] == 60
    assert inputs["publish_lag"] == 20
    assert inputs["launch_delay"] == 5


def test_first_step_uses_completion_and_retains_stage_wall(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    timeline = [dict(pid=123, stage="forecast.first_step", phase="begin", t_utc="2026-10-21T00:53:00Z", t_mono=10),
                dict(pid=123, stage="forecast.first_step", phase="end", t_utc="2026-10-21T00:53:03Z", t_mono=13)]
    (forecast / "timeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in timeline))
    result = race.collect_receipts(tmp_path)
    assert result["first_step_at"] == "2026-10-21T00:53:03Z"
    assert result["stages"][0]["wall_seconds"] == 3


def test_real_door_command_uses_available_flags(tmp_path):
    # Exercise the actual fallback registrar rather than inventing its argv.
    import gpuwm.go_cli
    args = arguments("--hours", "1", "--devices", "1")
    command, standby, publish = race.command_for(args, race.recipe_row("hrrr"), race.cycle_time(args.cycle), tmp_path)
    if standby:
        import gpuwm.standby_cli as door
    else:
        door = gpuwm.go_cli
    parser = argparse.ArgumentParser()
    door.register_cli(parser.add_subparsers())
    parsed = parser.parse_args(command[3:])
    assert parsed.devices == 1
    assert parsed.tiles == "auto"
    assert parsed.hydro == ("auto" if standby else None)


@pytest.mark.parametrize("requested,expected", [("auto", None), ("on", ""),
                                                ("off", "off"), ("water.toml", "water.toml")])
def test_fallback_hydro_reaches_runtime_without_fake_settings_path(tmp_path, monkeypatch, requested, expected):
    import gpuwm.go_cli
    from gpuwm.hydro import runtime
    monkeypatch.setattr(race.importlib.util, "find_spec", lambda name: None)
    args = arguments("--hours", "1", "--hydro", requested)
    command, standby, _ = race.command_for(args, race.recipe_row("hrrr"), race.cycle_time(args.cycle), tmp_path)
    assert not standby
    parser = argparse.ArgumentParser()
    gpuwm.go_cli.register_cli(parser.add_subparsers())
    parsed = parser.parse_args(command[3:])
    assert parsed.hydro == expected
    # Use the real runtime's filename validation. Only the fabric resolver is
    # isolated: this test must never fetch a fabric or start a forecast.
    if requested == "water.toml":
        settings = tmp_path / requested
        settings.write_text("[hydro]\nenabled = true\n")
        parsed.hydro = str(settings)
    decisions = []
    def resolve(settings, *, requested, config_path, table=None, plan_only=False,
                prepared_launch=False):
        decisions.append(requested)
        return {"enabled": False}, None, {"reason": "test resolver"}
    monkeypatch.setattr(runtime._door, "resolve_launch", resolve)
    runtime._launch_settings(parsed.hydro, parsed.config, plan_only=True)
    assert decisions == [parsed.hydro]


def test_parser_exposes_race_without_loading_full_numerical_parser():
    from gpuwm.cli import build_parser
    args = build_parser(race_only=True).parse_args(["race", "hrrr", "--cycle", "00Z", "--dry"])
    assert args.dry
    assert args.hours == 48


def test_earliest_mirror_and_final_product_define_margin():
    row = race.recipe_row("hrrr")
    cycle = race.cycle_time("2026-10-21T00")
    _, _, keys = race.object_names(row, cycle, 48)
    references = race.comparison_keys(row, cycle, 48, keys)
    alternative = references[48][1]
    objects = {keys[48]: {"last_modified": "2026-10-21T01:46:00Z", "nomads_mtime": "2026-10-21T01:45:00Z"},
               alternative: {"last_modified": "2026-10-21T01:47:00Z", "nomads_mtime": "2026-10-21T01:44:00Z"}}
    got = race.compare_hours(keys, objects, {48: "2026-10-21T01:44:30Z"}, references)[-1]
    assert got["margin_seconds"] == -30
    assert got["noaa_reference_key"] == alternative
    assert got["noaa_s3_posted_at"] == "2026-10-21T01:46:00Z"


def test_standby_nested_forecast_receipt_and_hydro(tmp_path):
    base = tmp_path / "forecast"
    (base / "forecast").mkdir(parents=True)
    (base / "standby-receipt.json").write_text(json.dumps({
        "launch_wall": "2026-10-21T00:52:00Z", "first_step_wall": "2026-10-21T00:53:00Z",
        "hydro": {"mode": "off", "reason": "explicit opt-out"},
        "stages": [{"name": "warm", "wall_seconds": 12, "parallelism": "GPU"}]}))
    (base / "forecast" / "report.json").write_text(json.dumps({
        "timing_seconds": {"construct": 5}, "hydro_activation": {"mode": "off", "reason": "explicit opt-out"}}))
    result = race.collect_receipts(tmp_path)
    assert result["launch_at"] == "2026-10-21T00:52:00Z"
    assert result["first_step_at"] == "2026-10-21T00:53:00Z"
    assert result["forecast_receipt"]["timing_seconds"]["construct"] == 5
    assert result["hydro"]["mode"] == "off"
    assert [stage["name"] for stage in result["stages"]] == ["warm", "construct"]


@pytest.mark.parametrize("devices,requested,expected", [(8, "auto", "off"), (1, "auto", "auto"),
                                                        (8, "on", "on"), (None, "auto", "auto")])
def test_multi_card_race_resolves_tiles_auto_to_off(tmp_path, devices, requested, expected):
    # box E, 2026-10-21: `gpuwm race hrrr --devices 8` passed the default
    # --tiles auto to standby and stopped at once on "[devices] count > 1
    # beside enabled [tiles] is refused".
    extra = ("--devices", str(devices)) if devices else ()
    args = arguments("--hours", "1", "--tiles", requested, *extra)
    command, _, _ = race.command_for(args, race.recipe_row("hrrr"), race.cycle_time(args.cycle), tmp_path)
    assert command[command.index("--tiles") + 1] == expected
