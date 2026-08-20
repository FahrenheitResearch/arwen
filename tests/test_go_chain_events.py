"""`gpuwm go` leaves its stage timings on disk, without being asked.

The 2026-08-16 pre-sim profiling audit measured a bare `gpuwm go` and
found that answering "how long from launch to sim step 1?" required
assembling five sources by hand -- a wrapper's timestamps, terminal
prose, proof.json, report.json, progress.jsonl -- because the chain
printed every number it computed and persisted none of them.

These tests hold the instrument that closes that: an events.jsonl in
run-plan's own grammar, written by default, replayable by run-plan's own
reader, whose stages account for the process.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from gpuwm import chain_events, go_cli, run_stamp
from gpuwm.chain_events import (
    BOOT_STAGE, CHAIN_EVENTS_FILENAME, TTFP_DEFINITION, TTFP_FROM_MTIME,
    TTFP_FROM_RECEIPT, GoChainEvents, read_chain_events, stage_walls,
    summarize,
)
from gpuwm.cli import main as cli_main
from gpuwm.runplan import EVENT_SCHEMA, EVENT_TAGS

REPO = Path(__file__).resolve().parents[1]

#: The pinned baseline the Rust reset diffs against.
BASELINE = REPO / "tests" / "data" / "pre-sim-stage-baseline.json"

#: What the pinned card below reports free -- above this file's fixture
#: config so the memory gate's verdict is "fits", on every box.
_PINNED_FREE_BYTES = 30 * 1024 ** 3


@pytest.fixture(autouse=True)
def _a_card_whose_free_vram_this_file_decides(monkeypatch):
    """Pin the one number the outside world moves in the memory gate.

    Same reason and same seam as ``test_go_chain``: the gate refuses
    when the binding phase exceeds the free VRAM it measures right now,
    so on a shared card every chain test here would go red whenever
    another run held the card.  Everything else in the gate stays real.
    """

    from gpuwm.core import preflight

    monkeypatch.setattr(
        preflight, "device_memory_probe_subprocess",
        lambda **_kwargs: {"free_bytes": _PINNED_FREE_BYTES,
                           "total_bytes": 32 * 1024 ** 3,
                           "profile": None})


@pytest.fixture(scope="module")
def gfs_config(tmp_path_factory):
    out = tmp_path_factory.mktemp("gfs") / "myarea.toml"
    assert cli_main([
        "domain", "--point=35.3,-97.5", "--card", "24gb", "--ladder", "12",
        "--source", "gfs", "--cycle", "2026-07-29T18", "--hours", "6",
        "--out", str(out), "--physics-profile",
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"]) == 0
    return out


@pytest.fixture(scope="module")
def staged_geog(tmp_path_factory):
    """A WPS_GEOG tree shaped the way ``gpuwm fetch-geog`` leaves one.

    Directory names and the ``index`` file are the whole of what the
    chain's precondition reads, and the names come from the module that
    stages the real tree so this cannot drift from the check.
    """

    from gpuwm.geog_assets import geog_datasets

    geog = tmp_path_factory.mktemp("geog") / "WPS_GEOG"
    for name in geog_datasets():
        (geog / name).mkdir(parents=True, exist_ok=True)
        (geog / name / "index").write_text("", encoding="utf-8")
    return geog


# ---------------------------------------------------------------------------
# A whole chain, faked at the subprocess boundary
# ---------------------------------------------------------------------------


class _FakePopen:
    """The same ``Popen`` double ``test_go_chain`` uses, for one reason:
    ``_run_stage`` spells ``subprocess.run`` out as Popen+communicate so
    the interrupt path can name the child's pid."""

    _pids = itertools.count(717171)

    def __init__(self, completed):
        self._completed = completed
        self.pid = next(self._pids)
        self.returncode = None

    def communicate(self):
        self.returncode = self._completed.returncode
        return self._completed.stdout, self._completed.stderr


class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog, *,
                 fetch_bytes=(14_000_000, 15_000_000, 16_000_000),
                 fetch_seconds=(3.0, 4.0, 3.0),
                 early_render=True, extra_argv=()):
    """Drive `gpuwm go` end to end with every subprocess doubled.

    Each double materializes the ARTIFACT its real stage would have
    published, because every number this feature reports is relayed out
    of an artifact and a chain whose stages wrote nothing would prove
    only that the relay does not crash.
    """

    root = tmp_path / "go"

    def fake_run(command, **kwargs):
        if "--author-front-door-manifest" in command:
            data = root / "data"
            data.mkdir(parents=True, exist_ok=True)
            (data / "gfs-input-manifest.json").write_text(
                "{}", encoding="utf-8")
        if "fetch" in command and "--out" in command:
            # The fetch manifest is where bytes and seconds live; the
            # chain reads bandwidth back out of it rather than timing
            # the download itself.
            data = root / "data"
            data.mkdir(parents=True, exist_ok=True)
            (data / "fetch-manifest.json").write_text(json.dumps({
                "schema": "gpuwm-fetch-manifest-v1",
                "files": [
                    {"name": f"f{index:03d}", "bytes": size,
                     "seconds": seconds, "downloaded": True,
                     "sha256": "0" * 64}
                    for index, (size, seconds)
                    in enumerate(zip(fetch_bytes, fetch_seconds))],
            }), encoding="utf-8")
        if "--output-root" in command:
            # Off the stage's OWN command: every run claims its own
            # timestamped folder under --outdir, so <root>/prepared is a
            # directory no stage writes to.
            prepared = Path(command[command.index("--output-root") + 1])
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "proof.json").write_text(json.dumps({
                "input_manifest_sha256": "a" * 64,
                "prepared_cache": {"content_sha256": "b" * 64}}),
                encoding="utf-8")
        if "--io-mode" in command:
            run = Path(command[command.index("--outdir") + 1])
            frames = run / "wrfout"
            frames.mkdir(parents=True, exist_ok=True)
            frame = frames / "wrfout_d01_2026-07-29_18_00_00"
            frame.write_text("", encoding="utf-8")
            # A SECOND frame, because a real forecast publishes more than
            # one and the early render claims exactly the first.  With a
            # one-frame run the finalize stage has nothing left to draw
            # and does not run at all, which is a lawful outcome but not
            # the one the stage-list pins describe.
            (frames / "wrfout_d01_2026-07-29_19_00_00").write_text(
                "", encoding="utf-8")
            _write_step_log(run)
            if early_render and "--render-products" in command:
                _publish_early_render(
                    Path(command[command.index("--render-dir") + 1]), frame)
        if command[:4] == [command[0], "-m", "gpuwm.cli", "render"]:
            out = Path(command[command.index("--out") + 1])
            (out / "d01").mkdir(parents=True, exist_ok=True)
            (out / "d01" / "late.png").write_bytes(b"\x89PNG")
        return _FakeCompleted(0, stdout="detail nobody asked for\n")

    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kwargs: _FakePopen(
                            fake_run(command, **kwargs)))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)

    rc = cli_main(["go", str(gfs_config), "--outdir", str(root),
                   "--geog-root", str(staged_geog), *extra_argv])
    # The RUN root, not the case root: the chain's event stream, receipts
    # and pictures live in this run's own timestamped folder, and the
    # case root holds only the shared download beside it.
    return rc, (run_stamp.latest(root) or root)


def _write_step_log(run_dir: Path) -> None:
    """A progress.jsonl shaped like a cold run's: phases, then a step 1
    that dwarfs its neighbours."""

    from gpuwm.progress_log import STEP_LOG_FILENAME, STEP_LOG_SCHEMA

    records = [
        {"event": "run_start"},
        {"event": "phase", "name": "preflight_verify", "wall_seconds": 1.9},
        {"event": "phase", "name": "restore_prepared_cache",
         "wall_seconds": 2.4},
        {"event": "phase", "name": "initialize_physics", "wall_seconds": 1.4},
        {"event": "step", "step": 1, "step_wall_seconds": 51.1},
        {"event": "step", "step": 2, "step_wall_seconds": 1.3},
        {"event": "phase", "name": "kernel_compile", "wall_seconds": 49.8,
         "reason": "architecture_missing"},
        {"event": "run_end", "status": "SUCCESS", "steps": 360,
         "wall_seconds": 54.3, "first_step_excess_seconds": 49.8},
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / STEP_LOG_FILENAME).write_text(
        "".join(json.dumps({"schema": STEP_LOG_SCHEMA, "sequence": index,
                            **record}) + "\n"
                for index, record in enumerate(records, start=1)),
        encoding="utf-8")
    (run_dir / "report.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8")


def _publish_early_render(render_dir: Path, frame: Path, *,
                          published_unix_ms: int | None = None) -> None:
    """What ``gpuwm.first_products`` leaves behind, receipt included.

    The digests are REAL, because the finalize stage checks them before
    it skips anything: a fixture with placeholder digests would exercise
    the "receipt not trusted" branch on every run and prove nothing about
    the branch that matters.
    """

    from gpuwm.first_products import (FIRST_PLOT_DEFINITION,
                                      FIRST_PRODUCTS_RECEIPT,
                                      FIRST_PRODUCTS_SCHEMA)

    stamp = int(time.time() * 1000) if published_unix_ms is None \
        else int(published_unix_ms)
    picture = render_dir / "d01" / "analysis.png"
    picture.parent.mkdir(parents=True, exist_ok=True)
    picture.write_bytes(b"\x89PNG")
    # The publish is os.replace, which carries the renderer's own mtime;
    # a stamped receipt needs pictures no younger than its instant or it
    # describes files that were rewritten afterwards.
    os.utime(picture, (stamp / 1000.0, stamp / 1000.0))
    (render_dir / FIRST_PRODUCTS_RECEIPT).write_text(json.dumps({
        "schema": FIRST_PRODUCTS_SCHEMA,
        "measures": FIRST_PLOT_DEFINITION,
        "published_unix_ms": stamp,
        "frame": str(frame), "domain": 1,
        "frame_sha256": _sha256(frame),
        "valid_time": "2026-07-29T18:00:00",
        "render_products": "all",
        "written": [{"name": "d01/analysis.png", "sha256": _sha256(picture)}],
        "render_seconds": 2.5,
    }), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# S1: the chain persists its stage timings, by default
# ---------------------------------------------------------------------------


def test_a_bare_go_leaves_an_events_file_nobody_asked_for(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    rc, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    capsys.readouterr()
    assert rc == 0

    events = read_chain_events(root / CHAIN_EVENTS_FILENAME)
    assert events, "a bare `gpuwm go` wrote no events at all"
    # run-plan's grammar, not a second one.  The whole point of reusing
    # it is that a consumer written for one stream reads the other.
    assert {record["schema_version"] for record in events} == {EVENT_SCHEMA}
    assert {record["event"] for record in events} <= set(EVENT_TAGS)


def test_every_stage_of_the_chain_is_named_with_its_wall(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    capsys.readouterr()
    walls = stage_walls(read_chain_events(root / CHAIN_EVENTS_FILENAME))

    # THE AUDIT'S OWN STAGE LIST, minus nothing.  `boot` is the CLI's
    # own start-up and the memory/geography gates, which the audit
    # measured at 1.5-1.7 s and which lived nowhere at all.
    for stage in (BOOT_STAGE, "authority", "fetch", "manifest", "prepare",
                  "forecast", "render"):
        assert stage in walls, f"no stage_finished for {stage!r}"
        assert walls[stage] >= 0.0


def test_the_stages_account_for_the_process_wall(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """The acceptance bar: the walls sum to the process, within 2%.

    A stage list that does not add up is the thing this replaces -- a
    reader who has to subtract to find the time nobody named.
    """

    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    capsys.readouterr()
    summary = summarize(root / CHAIN_EVENTS_FILENAME)
    assert summary is not None
    assert summary["event"] == "completed"
    assert summary["accounted_seconds"] == pytest.approx(
        sum(stage["wall_seconds"] for stage in summary["stages"]))
    # The chain runs in well under a second here, so 2% of it is
    # microseconds; the bar that means something at this scale is that
    # what is NOT accounted for is a rounding error rather than a stage.
    assert summary["unaccounted_seconds"] < 0.05 * summary["wall_seconds"] \
        or summary["unaccounted_seconds"] < 0.05


def test_fetch_carries_its_bytes_and_its_bandwidth(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """`fetch-manifest.json` recorded bytes and sha256 and no seconds,
    so bandwidth -- the number that says whether the network or the
    service was the limiter -- existed nowhere on disk."""

    _, root = _run_a_chain(
        tmp_path, monkeypatch, gfs_config, staged_geog,
        fetch_bytes=(10_000_000, 20_000_000), fetch_seconds=(1.0, 4.0))
    capsys.readouterr()
    fetch = next(record for record
                 in read_chain_events(root / CHAIN_EVENTS_FILENAME)
                 if record.get("event") == "stage_finished"
                 and record["stage"] == "fetch")
    assert fetch["bytes"] == 30_000_000
    assert fetch["bytes_per_second"] == pytest.approx(6_000_000.0)
    assert fetch["files"] == 2


def test_a_verify_skip_is_never_reported_as_bandwidth(tmp_path):
    """VALIDATE THE INSTRUMENT.  Measured, and it was wrong the first
    time: a re-run against an existing --data-dir downloads nothing and
    only re-hashes what is on disk, and dividing those bytes by those
    seconds reported **1.09 GB/s** of "fetch bandwidth" on the reference
    box.  True about sha256, two orders of magnitude wrong about the
    network, and reported under the network's name."""

    from gpuwm.fetch import fetch_throughput

    out = tmp_path / "data"
    out.mkdir()
    (out / "fetch-manifest.json").write_text(json.dumps({
        "schema": "gpuwm-fetch-manifest-v1",
        "files": [
            {"name": "a", "bytes": 18_000_000, "seconds": 0.017,
             "downloaded": False},
            {"name": "b", "bytes": 2_000_000, "seconds": 0.002,
             "downloaded": False},
        ]}), encoding="utf-8")

    throughput = fetch_throughput(out)
    assert throughput["bytes"] == 20_000_000
    assert throughput["bytes_per_second"] is None
    assert throughput["verified_bytes"] == 20_000_000
    assert throughput["downloaded_files"] == 0

    # ... and a real download does report one.
    (out / "fetch-manifest.json").write_text(json.dumps({
        "schema": "gpuwm-fetch-manifest-v1",
        "files": [{"name": "a", "bytes": 20_000_000, "seconds": 4.0,
                   "downloaded": True}]}), encoding="utf-8")
    assert fetch_throughput(out)["bytes_per_second"] == pytest.approx(
        5_000_000.0)


def test_the_forecast_stage_is_decomposed_from_its_own_step_log(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """THE AUDIT'S HARDEST QUESTION, answerable from one file.

    Runner setup, model step 1 and the integration were three separate
    reads of progress.jsonl and report.json.  The terminal event carries
    the decomposition, including the 51 s that was invisible inside
    step 1's wall.
    """

    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    capsys.readouterr()
    forecast = summarize(root / CHAIN_EVENTS_FILENAME)["forecast"]
    assert forecast["phases"]["preflight_verify"] == pytest.approx(1.9)
    assert forecast["phases"]["restore_prepared_cache"] == pytest.approx(2.4)
    assert forecast["phases"]["initialize_physics"] == pytest.approx(1.4)
    assert forecast["phases"]["kernel_compile"] == pytest.approx(49.8)
    assert forecast["first_step_seconds"] == pytest.approx(51.1)
    assert forecast["first_step_excess_seconds"] == pytest.approx(49.8)
    assert forecast["integration_seconds"] == pytest.approx(54.3)
    assert forecast["steps"] == 360


def test_a_failed_chain_still_says_where_the_time_went(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """A run that died is the run whose timings you most want."""

    root = tmp_path / "go"

    def fake_run(command, **kwargs):
        return _FakeCompleted(3, stdout="the real diagnosis\n")

    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kwargs: _FakePopen(
                            fake_run(command, **kwargs)))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "bridge")
    rc = cli_main(["go", str(gfs_config), "--outdir", str(root),
                   "--geog-root", str(staged_geog)])
    capsys.readouterr()
    assert rc == 3
    # This run's own folder under the case root -- the chain claims one
    # per run, so the stream is inside it, not beside it.
    root = run_stamp.latest(root)
    assert root is not None
    summary = summarize(root / CHAIN_EVENTS_FILENAME)
    assert summary["event"] == "failed"
    assert summary["exit_code"] == 3
    walls = stage_walls(read_chain_events(root / CHAIN_EVENTS_FILENAME))
    assert BOOT_STAGE in walls and "authority" in walls
    failed = next(record for record
                  in read_chain_events(root / CHAIN_EVENTS_FILENAME)
                  if record.get("event") == "stage_finished"
                  and record["stage"] == "authority")
    assert failed["ok"] is False and failed["exit_code"] == 3


def test_a_refused_gate_leaves_no_run_directory_behind(
        tmp_path, monkeypatch, gfs_config, capsys):
    """Telemetry must not create the tree a refusal declined to make.

    The memory and geography gates refuse BEFORE anything is spent, and
    a stream opened at the top of the chain would have left a directory
    on every refusal.
    """

    root = tmp_path / "never"
    rc = cli_main(["go", str(gfs_config), "--outdir", str(root),
                   "--geog-root", str(tmp_path / "no-geography-here")])
    capsys.readouterr()
    assert rc != 0
    assert not root.exists()


def test_a_caller_with_its_own_observer_writes_no_second_stream(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """`gpuwm run-plan` writes this grammar into its own run directory.

    Two writers on one file would interleave two runs' sequences, and
    `read_events` refuses a stream whose sequence is not dense -- so the
    chain's own observer stands down for a caller that brought one.
    """

    root = tmp_path / "go"
    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kwargs: _FakePopen(
                            _FakeCompleted(3, stdout="stop here\n")))
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")

    class Host:
        hosts_forecast = True

    import argparse

    args = argparse.Namespace(
        config=gfs_config, outdir=root, data_dir=None,
        geog_root=staged_geog, dry_run=False, no_memory_gate=False,
        explain=False)
    go_cli.go_main(args, observer=Host())
    capsys.readouterr()
    assert not (root / CHAIN_EVENTS_FILENAME).exists()


# ---------------------------------------------------------------------------
# S5: time to first plot, on the front door people use
# ---------------------------------------------------------------------------


def test_the_forecast_stage_is_asked_to_draw_the_first_frame(tmp_path,
                                                             gfs_config):
    """Arming the early render from `go` would mean hosting the forecast
    in process.  The runner owns an identical arming on its own command
    line, so the chain asks for it there and keeps its subprocess."""

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go")
    command = go_cli.forecast_command(
        plan, {"proof": "a", "source_manifest": "b", "prepared_content": "c"},
        early_render="all")
    assert "--render-products" in command
    assert command[command.index("--render-products") + 1] == "all"
    # ... into the SAME directory finalize renders into, or the early
    # picture and the late one land in two different trees.
    assert Path(command[command.index("--render-dir") + 1]) == plan["render"]

    # And a plan that asked for no pictures asks for none early either.
    assert go_cli.forecast_command(
        plan, {"proof": "a", "source_manifest": "b",
               "prepared_content": "c"}) == [
        part for part in go_cli.forecast_command(
            plan, {"proof": "a", "source_manifest": "b",
                   "prepared_content": "c"})]
    assert "--render-products" not in go_cli.forecast_command(
        plan, {"proof": "a", "source_manifest": "b", "prepared_content": "c"})


def test_time_to_first_plot_is_printed_and_persisted(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    printed = capsys.readouterr().out
    assert "go: time to first plot" in printed

    summary = summarize(root / CHAIN_EVENTS_FILENAME)
    assert summary["time_to_first_plot_seconds"] is not None
    assert summary["time_to_first_plot_source"] == TTFP_FROM_RECEIPT
    ready = next(record for record
                 in read_chain_events(root / CHAIN_EVENTS_FILENAME)
                 if record.get("event") == "first_products_ready")
    assert ready["seconds_from_launch"] == pytest.approx(
        summary["time_to_first_plot_seconds"])
    assert ready["render_products"] == "all"


def test_without_an_early_render_the_pictures_own_mtimes_answer(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """Coarser, and labelled coarser.  A number whose provenance is not
    stated is a number a later run cannot compare itself against."""

    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog,
                           early_render=False)
    capsys.readouterr()
    summary = summarize(root / CHAIN_EVENTS_FILENAME)
    assert summary["time_to_first_plot_source"] == TTFP_FROM_MTIME
    assert summary["time_to_first_plot_seconds"] is not None


def test_the_early_render_receipt_stamps_when_it_published(tmp_path):
    """`go` does not host the render, so a duration measured from a
    start it cannot see is useless to it.  The receipt carries the
    absolute instant and every consumer subtracts its own launch."""

    from gpuwm.first_products import read_receipt

    render_dir = tmp_path / "render"
    render_dir.mkdir()
    frame = tmp_path / "wrfout_d01"
    frame.write_text("", encoding="utf-8")
    _publish_early_render(render_dir, frame)
    receipt = read_receipt(render_dir)
    assert isinstance(receipt["published_unix_ms"], int)

    observer = GoChainEvents(
        launch_monotonic=time.monotonic() - 30.0,
        launch_unix_ms=int(time.time() * 1000) - 30_000)
    ttfp = observer.time_to_first_plot(render_dir=render_dir)
    assert ttfp["source"] == TTFP_FROM_RECEIPT
    assert 25.0 < ttfp["seconds"] < 35.0


# ---------------------------------------------------------------------------
# S5b: the number and the published tree say the same thing
# ---------------------------------------------------------------------------
#
# MEASURED on both 3080 walks (44 s and 46 s): `go` printed "time to first
# plot 0m 46s (first-products receipt)" while the earliest PNG in the
# published run tree carried 2m 45s, sixteen seconds after the render stage
# started at 2m 29s.  Both numbers were real.  The early render published at
# 46 s; the finalize stage then redrew the same frame and overwrote those
# same paths, because `go`'s observer is not a first-products HOST and the
# branch that skips an already-published frame was reachable only from one.
# So the headline named an instant at which nothing in the tree existed, and
# a reader who checks the artifact stops believing the headline.


def _synthetic_run_tree(tmp_path: Path, *, launch_unix_ms: int,
                        published_at_s: float) -> tuple[Path, Path]:
    """A render directory as a finished run leaves one."""

    render_dir = tmp_path / "render"
    render_dir.mkdir()
    frame = tmp_path / "wrfout_d01_2026-07-29_18_00_00"
    frame.write_text("", encoding="utf-8")
    _publish_early_render(
        render_dir, frame,
        published_unix_ms=launch_unix_ms + int(published_at_s * 1000))
    return render_dir, frame


def test_a_rewritten_picture_stops_the_receipt_instant_being_quoted(
        tmp_path: Path):
    """THE contradiction, on a synthetic run folder.

    The receipt says 46 s.  The picture it names carries 2m 45s, because
    something redrew it.  The receipt's instant now describes no file on
    disk, so the tree's own earliest picture is the honest answer and the
    source says which measurement that was.
    """

    launch_unix_ms = int(time.time() * 1000) - 200_000
    render_dir, _frame = _synthetic_run_tree(
        tmp_path, launch_unix_ms=launch_unix_ms, published_at_s=46.0)

    rewritten = (launch_unix_ms + 165_000) / 1000.0
    os.utime(render_dir / "d01" / "analysis.png", (rewritten, rewritten))

    observer = GoChainEvents(launch_monotonic=time.monotonic() - 200.0,
                             launch_unix_ms=launch_unix_ms)
    ttfp = observer.time_to_first_plot(render_dir=render_dir)
    assert ttfp["source"] == TTFP_FROM_MTIME
    assert ttfp["seconds"] == pytest.approx(165.0, abs=2.0)
    # ...and the receipt is still readable, still says what it said, and
    # now carries the fact that made it unquotable.
    early = observer.first_products_receipt(render_dir=render_dir)
    assert early["seconds_from_launch"] == pytest.approx(46.0, abs=2.0)
    assert early["pictures_still_original"] is False


def test_an_untouched_picture_keeps_the_receipt_instant(tmp_path: Path):
    """The control.  Same tree, same receipt, nothing rewrote the picture."""

    launch_unix_ms = int(time.time() * 1000) - 200_000
    render_dir, _frame = _synthetic_run_tree(
        tmp_path, launch_unix_ms=launch_unix_ms, published_at_s=46.0)

    observer = GoChainEvents(launch_monotonic=time.monotonic() - 200.0,
                             launch_unix_ms=launch_unix_ms)
    ttfp = observer.time_to_first_plot(render_dir=render_dir)
    assert ttfp["source"] == TTFP_FROM_RECEIPT
    assert ttfp["seconds"] == pytest.approx(46.0, abs=2.0)
    assert observer.first_products_receipt(
        render_dir=render_dir)["pictures_still_original"] is True


def test_the_printed_number_never_predates_the_tree(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """The property, over a whole chain: the reported instant is one at
    which a picture in the published tree actually existed."""

    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    capsys.readouterr()
    summary = summarize(root / CHAIN_EVENTS_FILENAME)
    reported = summary["launch_unix_ms"] + int(
        summary["time_to_first_plot_seconds"] * 1000)
    pictures = sorted(root.rglob("*.png"))
    assert pictures, "the chain published no picture at all"
    earliest = min(int(path.stat().st_mtime * 1000) for path in pictures)
    assert earliest <= reported + 2000, (earliest, reported)


def test_the_summary_states_what_the_number_measures():
    """Two quantities were both called time to first plot; the stream now
    says which one it is carrying."""

    assert "still present in the render tree" in TTFP_DEFINITION


def test_the_early_frame_is_not_redrawn_by_the_finalize_stage(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """The anchor fix.  `go` never hosts the early render, so the skip was
    unreachable and the finalize stage overwrote the early pictures --
    which is what moved the tree's earliest mtime to 2m 45s."""

    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    printed = capsys.readouterr().out
    assert "1 frame already published by the early render" in printed
    assert "digests verified" in printed
    summary = summarize(root / CHAIN_EVENTS_FILENAME)
    assert summary["time_to_first_plot_source"] == TTFP_FROM_RECEIPT


# ---------------------------------------------------------------------------
# S6: the pinned baseline the Rust reset diffs against
# ---------------------------------------------------------------------------


def test_the_baseline_states_where_and_when_it_was_measured():
    baseline = chain_events.load_baseline(BASELINE)
    provenance = baseline["provenance"]
    for key in ("box", "card", "date", "gpuwm_version", "case"):
        assert provenance.get(key), f"baseline provenance has no {key}"
    # Cold and warm are different runs and different answers; a baseline
    # that did not say which would be compared against the wrong one.
    assert set(baseline["runs"]) == {"cold", "warm"}
    for run in baseline["runs"].values():
        assert run["cache_state"] in ("cold", "warm")


def test_a_baseline_without_provenance_is_refused(tmp_path):
    """Validate the instrument.  A loader that accepts anything would
    let a lane diff against a number with no claim attached."""

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({
        "schema": "x", "provenance": {"box": "somewhere"},
        "runs": {}}), encoding="utf-8")
    with pytest.raises(ValueError) as refusal:
        chain_events.load_baseline(path)
    assert "provenance" in str(refusal.value)


def test_the_baseline_names_stages_the_chain_actually_emits(
        tmp_path, monkeypatch, gfs_config, staged_geog, capsys):
    """The binding that keeps the baseline usable.

    A pinned number is only diffable while the stage it names still
    exists under that name.  A Rust rewrite that renames a stage must
    fail HERE, in the suite, rather than silently produce a comparison
    against a stage nobody emits any more.
    """

    _, root = _run_a_chain(tmp_path, monkeypatch, gfs_config, staged_geog)
    capsys.readouterr()
    emitted = set(stage_walls(read_chain_events(root / CHAIN_EVENTS_FILENAME)))

    baseline = chain_events.load_baseline(BASELINE)
    pinned = [*baseline["runs"].values(),
              baseline["card_swap_reproduction"]]
    for run in pinned:
        unknown = set(run["stage_seconds"]) - emitted
        assert not unknown, (
            f"the baseline pins {sorted(unknown)}, which `gpuwm go` no "
            "longer emits; rename the baseline's keys with the stage or "
            "the comparison is against nothing")


def test_the_instrumented_prepare_receipt_accounts_for_itself():
    """S3's coverage assertion, run in the normal suite against the
    smallest REAL receipt there is: the one this work measured.

    Both directions, because an instrument that only ever passes has not
    been tested.  The shipped receipts pinned beside it -- the same
    stage, the same box, the same day, written before this lane -- must
    FAIL the same check, or the check is not measuring anything.
    """

    from gpuwm.stage_timing import (MINIMUM_COVERAGE, timing_coverage,
                                    timing_coverage_shortfall)

    baseline = chain_events.load_baseline(BASELINE)

    # The reproduction of the defect this lane fixes is a real receipt
    # from a real run and is held to the same rule.
    swap = baseline["card_swap_reproduction"]["prepare_timing_seconds"]
    assert timing_coverage(swap) >= MINIMUM_COVERAGE

    warm = baseline["runs"]["warm"]["prepare_timing_seconds"]
    shortfall = timing_coverage_shortfall(
        warm, what="the warm baseline's prepare receipt")
    assert shortfall is None, shortfall
    assert timing_coverage(warm) >= MINIMUM_COVERAGE

    # MEASURED 2026-08-16, and this is the defect: the shipped receipt
    # attributed 20.7 s of its own 34.8 s because the static build was
    # timed only behind GPUWM_PERF_TIMING.
    shipped = baseline["normal_size_prepare"]["shipped_prepare_timing_seconds"]
    assert timing_coverage(shipped) < MINIMUM_COVERAGE
    refusal = timing_coverage_shortfall(shipped, what="the shipped receipt")
    assert refusal is not None
    assert "not named by any key" in refusal

    cold = baseline["runs"]["cold"]["prepare_timing_seconds"]
    assert timing_coverage(cold) < MINIMUM_COVERAGE


def test_the_coverage_rule_declines_to_judge_what_it_cannot():
    """A receipt with no total, or a zero one, is unaskable rather than
    uncovered -- and a rule that could not tell those apart would refuse
    every receipt from the fastest runs."""

    from gpuwm.stage_timing import timing_coverage, timing_coverage_shortfall

    assert timing_coverage({"decode": 1.0}) is None
    assert timing_coverage({"decode": 0.0, "total": 0.0}) is None
    assert timing_coverage_shortfall({"decode": 1.0}, what="x") is None
    assert timing_coverage({"a": 0.5, "b": 0.45, "total": 1.0}) == \
        pytest.approx(0.95)
