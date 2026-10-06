"""Startup accounting keeps measured wall intervals and worker work separate."""

import json
import gzip

from tools.startup_timeline import Timeline, main, timestamp, union_seconds


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_native_output_uses_valid_time_but_only_wall_timestamp_for_latency(tmp_path):
    start = timestamp("2026-01-01T00:00:00Z")
    path = write_lines(tmp_path / "progress.jsonl", [
        {"event": "run_start", "start_time": "2026-01-01_12:00:00", "emitted_unix_ms": start * 1000},
        {"event": "output_written", "domain": 1, "valid_time": "2026-01-01_12:00:00", "emitted_unix_ms": (start + 4) * 1000},
        {"event": "step", "domain": 1, "step": 1, "model_seconds": 20, "step_wall_seconds": 3, "emitted_unix_ms": (start + 7) * 1000},
        {"event": "output_written", "domain": 2, "valid_time": "2026-01-01_13:00:00", "emitted_unix_ms": (start + 9) * 1000},
        {"event": "output_written", "domain": 1, "valid_time": "2026-01-01_13:00:00", "emitted_unix_ms": (start + 20) * 1000},
    ])
    timeline = Timeline(origin=start - 5)
    timeline.read(path)
    result = timeline.report()
    assert result["total_seconds"] == 25
    assert result["latencies_seconds"]["forecast_start_to_first_hour_written"] == 20
    assert result["milestones"]["first_step"]["unix_s"] == start + 7
    assert result["first_forecast_hour_written_utc"] == "2026-01-01T00:00:20.000Z"


def test_nested_parallel_and_repeated_intervals_do_not_double_count():
    assert union_seconds([(0, 10), (0, 10), (4, 8), (8, 15)]) == 15
    timeline = Timeline(origin=0)
    timeline.interval("model_build", "model_build", 0, 10, "a:1")
    timeline.interval("model_build", "model_build", 0, 10, "b:1")
    timeline.interval("kernel_compile", "compile", 3, 8, "a:2")
    timeline.milestone("first_forecast_hour_written", 15, "a:3")
    report = timeline.report()
    assert report["located_union_seconds"] == 10
    assert report["unattributed_seconds"] == 5
    assert next(row for row in report["stage_summary"] if row["stage"] == "model_build")["located_union_seconds"] == 10
    assert report["intervals"][0]["evidence"] == ["a:1", "b:1"]


def test_receipt_durations_and_total_do_not_invent_elapsed_time(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({
        "timing_seconds": {"static": 3, "decode_and_compose": 5, "initialize_all_times": 20, "total": 23},
        "forcing_stage_timings": [{"horizontal_seconds": 4, "column_stages_seconds": {
            "thermodynamic_vertical_interpolation": 6, "number_moment_vertical_interpolation": 2}}],
    }), encoding="utf-8")
    timeline = Timeline()
    timeline.read(path)
    result = timeline.report()
    assert result["total_seconds"] is None
    assert result["located_union_seconds"] is None
    assert not result["intervals"]
    summary = {row["stage"]: row for row in result["stage_summary"]}
    assert summary["vertical_interpolation"]["unplaced_recorded_work_seconds"] == 8
    assert summary["decode"]["status"] == "unknown"
    assert sum(row["aggregate"] for row in result["duration_only"]) == 1


def test_compile_estimate_emitted_at_run_end_is_not_a_compile_wall_interval(tmp_path):
    path = write_lines(tmp_path / "progress.jsonl", [
        {"event": "phase", "name": "kernel_compile", "wall_seconds": 100,
         "measured_as": "first step excess", "emitted_unix_ms": 2000000},
    ])
    timeline = Timeline()
    timeline.read(path)
    assert not timeline.intervals
    assert timeline.durations[0]["duration_seconds"] == 100


def test_clock_logs_roll_over_midnight_and_need_explicit_date(tmp_path):
    path = tmp_path / "bootstrap.log"
    path.write_text("[23:59:58Z] start rust\n[00:00:02Z] done rust (4 s)\n", encoding="utf-8")
    timeline = Timeline()
    timeline.read(path, date="2026-01-01")
    assert timeline.intervals[0]["duration_seconds"] == 4
    unknown = Timeline()
    unknown.read(path)
    assert not unknown.intervals
    assert unknown.durations[0]["duration_seconds"] == 4
    assert any("--date" in warning for warning in unknown.warnings)


def test_cli_emits_json_csv_and_markdown_without_importing_gpu_code(tmp_path):
    path = write_lines(tmp_path / "stages.jsonl", [
        {"event": "stage_start", "stage": "downloads", "utc": "2026-01-01T00:00:00Z"},
        {"event": "stage_end", "stage": "downloads", "utc": "2026-01-01T00:00:10Z"},
        {"event": "first_forecast_hour_written", "utc": "2026-01-01T00:00:15Z"},
    ])
    output, csv, markdown = (tmp_path / name for name in ("out.json", "out.csv", "out.md"))
    assert main(["--log", str(path), "--origin", "2026-01-01T00:00:00Z",
                 "--output", str(output), "--csv", str(csv), "--markdown", str(markdown)]) == 0
    assert json.loads(output.read_text())["total_seconds"] == 15
    assert "downloads,10.0" in csv.read_text()
    assert "Stage rows can overlap" in markdown.read_text()


def test_driver_api_host_calls_do_not_claim_synchronized_device_transfer(tmp_path):
    path = write_lines(tmp_path / "startup.jsonl", [
        {"event": "run_claimed", "utc_unix_s": 100},
        {"event": "startup_stage", "stage": "device_upload", "call": "cupy.asarray",
         "start_unix_s": 101, "end_unix_s": 104, "host_call_only": True, "bytes": 1024},
        {"event": "startup_stage", "stage": "model_build", "start_unix_s": 100, "end_unix_s": 110},
        {"event": "first_forecast_hour_written", "utc_unix_s": 115},
    ])
    timeline = Timeline()
    timeline.read(path)
    report = timeline.report()
    rows = {row["stage"]: row for row in report["stage_summary"]}
    assert rows["device_upload"]["located_union_seconds"] is None
    assert rows["device_upload"]["host_call_union_seconds"] == 3
    assert rows["device_upload"]["status"] == "host_calls_only"
    assert report["located_union_seconds"] == 10
    assert report["total_seconds"] == 15
    upload = next(row for row in report["intervals"] if row["stage"] == "device_upload")
    assert upload["bytes"] == 1024
    assert upload["timing_scope"] == "host_call"


def test_recurring_callbacks_and_terminal_checks_do_not_inflate_startup(tmp_path):
    path = write_lines(tmp_path / "startup.jsonl", [
        {"event": "run_claimed", "utc_unix_s": 100},
        {"event": "startup_stage", "stage": "model_build", "call": "initialize_cached_physics",
         "start_unix_s": 100, "end_unix_s": 110},
        {"event": "startup_stage", "stage": "device_upload", "start_unix_s": 102,
         "end_unix_s": 106, "host_call_only": True},
        {"event": "first_step", "utc_unix_s": 110},
        {"event": "startup_stage", "stage": "model_build", "call": "initialize_cached_physics",
         "start_unix_s": 112, "end_unix_s": 120},
        {"event": "startup_stage", "stage": "device_upload", "start_unix_s": 115,
         "end_unix_s": 125, "host_call_only": True},
        {"event": "first_forecast_hour_written", "utc_unix_s": 130},
        {"event": "startup_stage", "stage": "preflight", "start_unix_s": 132, "end_unix_s": 140},
    ])
    timeline = Timeline()
    timeline.read(path)
    report = timeline.report()
    startup = report["startup_until_first_step"]
    first_hour = report["first_hour_window"]
    recurring = report["after_first_step_until_first_hour"]
    rows = lambda window: {row["stage"]: row for row in window["stage_summary"]}
    assert startup["total_seconds"] == 10
    assert first_hour["total_seconds"] == 30
    assert recurring["total_seconds"] == 20
    assert rows(startup)["model_build"]["located_union_seconds"] == 10
    assert rows(recurring)["model_build"]["located_union_seconds"] == 8
    assert rows(first_hour)["model_build"]["located_union_seconds"] == 18
    assert rows(startup)["device_upload"]["host_call_union_seconds"] == 4
    assert rows(recurring)["device_upload"]["host_call_union_seconds"] == 10
    assert rows(first_hour)["bundle_verify"]["located_union_seconds"] is None
    assert any(row["stage"] == "bundle_verify" for row in report["intervals"])


def test_window_crossing_callback_is_split_without_changing_stage():
    timeline = Timeline(origin=100)
    timeline.interval("model_build", "physical_callback", 105, 115, "callback:1")
    timeline.milestone("first_step", 110, "step:1")
    timeline.milestone("first_forecast_hour_written", 120, "output:1")
    timeline.duration("kernel_compile", "terminal_estimate", 7, "receipt:1")
    report = timeline.report()
    startup = report["startup_until_first_step"]
    recurring = report["after_first_step_until_first_hour"]
    assert startup["intervals"][0]["duration_seconds"] == 5
    assert startup["intervals"][0]["name"] == "physical_callback"
    assert any(row["name"] == "physical_callback" and row["duration_seconds"] == 5
               for row in recurring["intervals"])
    assert report["intervals"][0]["duration_seconds"] == 10
    assert all(row["unplaced_recorded_work_seconds"] is None for row in startup["stage_summary"])


def test_missing_first_step_cannot_assign_later_calls_to_startup():
    timeline = Timeline(origin=100)
    timeline.interval("model_build", "callback", 105, 115, "callback:1")
    timeline.milestone("first_forecast_hour_written", 120, "output:1")
    startup = timeline.report()["startup_until_first_step"]
    assert startup["boundary_status"] == "unknown"
    assert not startup["intervals"]
    assert startup["total_seconds"] is None


def test_truncated_native_stream_cannot_substitute_later_step_or_hour(tmp_path):
    path = write_lines(tmp_path / "truncated.jsonl", [
        {"event": "run_start", "start_time": "2026-01-01_00:00:00", "utc_unix_s": 100},
        {"event": "step", "domain": 1, "step": 60, "model_seconds": 1200, "utc_unix_s": 110},
        {"event": "output_written", "domain": 1, "valid_time": "2026-01-01_02:00:00", "utc_unix_s": 130},
    ])
    timeline = Timeline(origin=100)
    timeline.read(path)
    result = timeline.report()
    assert "first_step" not in result["milestones"]
    assert "first_forecast_hour_written" not in result["milestones"]
    assert result["total_seconds"] is None


def test_native_forcing_work_retains_every_leaf_and_keeps_totals_separate(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"forcing_stage_timings": [
        {"forcing_index": 0, "valid_time": "2026-01-01 00:00:00", "horizontal_seconds": 4,
         "total_seconds": 20, "column_stages_seconds": {"total_seconds": 16,
             "base_state_and_moist_rebalance": 8, "base_state_upload": 2,
             "hydrometeor_vertical_interpolation": 6}},
        {"forcing_index": 0, "valid_time": "2026-01-01 00:00:00", "horizontal_seconds": 3,
         "total_seconds": 10, "column_stages_seconds": {"total_seconds": 7,
             "base_state_and_moist_rebalance": 5, "base_state_upload": 2}},
    ]}), encoding="utf-8")
    timeline = Timeline()
    timeline.read(path)
    report = timeline.report()
    detail = report["forcing_work_breakdown"]
    work = {row["name"]: row for row in detail["summary"]}
    assert len(detail["records"]) == 2
    assert work["base_state_and_moist_rebalance"]["recorded_work_seconds"] == 13
    assert work["horizontal_interpolation"]["recorded_work_seconds"] == 7
    assert "total_seconds" not in work
    assert detail["records"][0]["column_stages_seconds"]["total_seconds"] == 16
    assert next(row for row in report["stage_summary"] if row["stage"] == "device_upload")["status"] == "unknown"


def test_store_health_text_duration_has_no_invented_wall_placement(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("prepared forecast: store-direct initialization -- the initialized.d01 full-state health gate IS armed, over the store: 221 fields, 21.96 GiB, in 503.4s\n")
    timeline = Timeline(origin=100)
    timeline.read(path)
    report = timeline.report()
    health = next(row for row in report["stage_summary"] if row["stage"] == "health_scan")
    assert health["located_union_seconds"] is None
    assert health["unplaced_recorded_work_seconds"] == 503.4
    assert not report["intervals"]


def test_health_wrapper_interval_keeps_nested_model_construction_separate(tmp_path):
    path = write_lines(tmp_path / "startup.jsonl", [
        {"event": "run_claimed", "utc_unix_s": 100},
        {"event": "startup_stage", "stage": "model_build", "start_unix_s": 101, "end_unix_s": 120},
        {"event": "startup_stage", "stage": "_store_full_state_health", "start_unix_s": 103, "end_unix_s": 113},
        {"event": "first_step", "utc_unix_s": 125},
        {"event": "first_forecast_hour_written", "utc_unix_s": 130},
    ])
    timeline = Timeline()
    timeline.read(path)
    startup = timeline.report()["startup_until_first_step"]
    rows = {row["stage"]: row for row in startup["stage_summary"]}
    assert rows["health_scan"]["located_union_seconds"] == 10
    assert rows["model_build"]["located_union_seconds"] == 19
    assert startup["located_union_seconds"] == 19


def test_gzip_trace_and_compact_summary_keep_unions_counts_bytes_and_lines(tmp_path):
    path = write_lines(tmp_path / "startup.jsonl", [
        {"event": "run_claimed", "utc_unix_s": 100},
        {"event": "startup_stage", "stage": "device_upload", "start_unix_s": 101,
         "end_unix_s": 105, "host_call_only": True, "bytes": 1024},
        {"event": "startup_stage", "stage": "device_upload", "start_unix_s": 103,
         "end_unix_s": 106, "host_call_only": True, "bytes": 2048},
        {"event": "first_step", "utc_unix_s": 110},
        {"event": "first_forecast_hour_written", "utc_unix_s": 115},
    ])
    compressed = tmp_path / "startup.jsonl.gz"
    with gzip.open(compressed, "wt") as stream:
        stream.write(path.read_text())
    original, zipped = Timeline(), Timeline()
    original.read(path)
    zipped.read(compressed)
    full, compact = original.report(), zipped.report(compact=True)
    assert full["total_seconds"] == compact["total_seconds"] == 15
    assert full["located_union_seconds"] == compact["located_union_seconds"]
    assert "intervals" not in compact
    assert "intervals" not in compact["startup_until_first_step"]
    row = next(row for row in compact["startup_until_first_step"]["stage_summary"] if row["stage"] == "device_upload")
    assert row["host_call_union_seconds"] == 5
    assert row["observed_interval_count"] == 2
    assert row["recorded_input_bytes"] == 3072
    assert row["evidence_ranges"] == [{"source": str(compressed), "first_line": 2,
                                       "last_line": 3, "reference_count": 2}]
    out = tmp_path / "timeline.json.gz"
    assert main(["--log", str(compressed), "--summary", "--output", str(out)]) == 0
    with gzip.open(out, "rt") as stream:
        assert json.load(stream)["compact"] is True
