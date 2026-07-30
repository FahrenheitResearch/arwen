import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import tools.hrrr_pipeline as pipeline_module
from tools.hrrr_pipeline import (
    HrrrPipelineProducer,
    _cleanup_stale_producer_trees,
    _parse_series,
)


def _series(path: Path, hours: tuple[int, ...]) -> Path:
    path.write_text("".join(
        f"{hour}\tatmos-f{hour:02d}\tsoil-f{hour:02d}\n"
        for hour in hours
    ), encoding="ascii")
    return path


def test_pipeline_series_accepts_minimum_and_absolute_windows(tmp_path):
    assert [row[0] for row in _parse_series(
        _series(tmp_path / "short.tsv", (0, 1)))] == [0, 1]
    assert [row[0] for row in _parse_series(
        _series(tmp_path / "offset.tsv", tuple(range(12, 19))))] \
        == list(range(12, 19))
    assert [row[0] for row in _parse_series(
        _series(tmp_path / "full.tsv", tuple(range(49))))] == list(range(49))


@pytest.mark.parametrize("hours", ((0,), (12, 14), (47, 48, 49)))
def test_pipeline_series_rejects_short_gapped_or_overlong_horizons(
        tmp_path, hours):
    with pytest.raises(ValueError, match="pipeline series"):
        _parse_series(_series(tmp_path / "bad.tsv", hours))


def test_wait_hour_binds_stable_staging_across_canonical_publication(tmp_path):
    series = _series(tmp_path / "series.tsv", (0, 1))
    decoder = tmp_path / "decoder"
    decoder.write_bytes(b"decoder")
    decoder.chmod(0o755)
    output = tmp_path / "native-bridge"
    signals = tmp_path / "signals"
    signals.mkdir()
    staging = tmp_path / f".native-bridge.partial-{os.getpid()}"
    atmosphere = staging / "atmosphere-f00"
    soil = staging / "soil-f00"
    atmosphere.mkdir(parents=True)
    soil.mkdir()
    (atmosphere / "TT.f32le").write_bytes(b"stable-staging")
    (soil / "SOILT.f32le").write_bytes(b"soil")
    # Canonical publication may complete between wait_hour and the caller's
    # first field open.  It is a distinct tree; staging must remain selected.
    (output / "atmosphere-f00").mkdir(parents=True)
    (output / "soil-f00").mkdir()
    (output / "atmosphere-f00" / "TT.f32le").write_bytes(b"canonical")
    (signals / "f00.ready").write_text(
        "status\tPASS\nforecast_hour\t0\npayload_files\t24\n"
        "producer_elapsed_seconds\t1.25\n",
        encoding="ascii",
    )
    producer = HrrrPipelineProducer(
        decoder=decoder, series=series, output=output, signals=signals,
        cycle="2026-07-20 00:00:00", window=(0, 1, 0, 1), workers="1",
        log=tmp_path / "decoder.log",
    )
    producer.started = 0.0
    producer.process = SimpleNamespace(poll=lambda: None, pid=os.getpid())
    producer.preflight = {
        "staging_root": str(staging),
        "staging_retention": "until_consumer_finish",
    }

    selected = producer.wait_hour(0)
    assert selected == staging
    assert (selected / "atmosphere-f00" / "TT.f32le").read_bytes() \
        == b"stable-staging"


def test_finish_removes_only_retained_staging_after_canonical_publish(tmp_path):
    series = _series(tmp_path / "series.tsv", (0, 1))
    decoder = tmp_path / "decoder"
    decoder.write_bytes(b"decoder")
    output = tmp_path / "native-bridge"
    staging = tmp_path / f".native-bridge.partial-{os.getpid()}"
    signals = tmp_path / "signals"
    (staging / "atmosphere-f00").mkdir(parents=True)
    (output / "atmosphere-f00").mkdir(parents=True)
    signals.mkdir()
    (staging / "atmosphere-f00" / "TT.f32le").write_bytes(b"staging")
    (output / "atmosphere-f00" / "TT.f32le").write_bytes(b"canonical")
    (signals / "complete.ready").write_text(
        "status\tPASS\nseries_count\t2\nstaging_retained\ttrue\n"
        f"canonical_output\t{output}\nproducer_elapsed_seconds\t1.5\n",
        encoding="ascii",
    )
    producer = HrrrPipelineProducer(
        decoder=decoder, series=series, output=output, signals=signals,
        cycle="2026-07-20 00:00:00", window=(0, 1, 0, 1), workers="1",
        log=tmp_path / "decoder.log",
    )
    producer.started = 0.0
    producer.process = SimpleNamespace(
        poll=lambda: 0, pid=os.getpid(), wait=lambda timeout: 0)
    producer.preflight = {
        "staging_root": str(staging),
        "staging_retention": "until_consumer_finish",
    }

    receipt = producer.finish()

    assert receipt["staging_cleanup"] == "removed"
    assert not staging.exists()
    assert (output / "atmosphere-f00" / "TT.f32le").read_bytes() \
        == b"canonical"


def test_restart_removes_dead_pid_staging_and_publish_trees(tmp_path):
    output = tmp_path / "native-bridge"
    stale_pid = 99999999
    partial = tmp_path / f".native-bridge.partial-{stale_pid}"
    publish = tmp_path / f".native-bridge.publish-{stale_pid}"
    partial.mkdir()
    publish.mkdir()
    (partial / "unfinished").write_bytes(b"partial")
    (publish / "unfinished").write_bytes(b"publish")

    removed = _cleanup_stale_producer_trees(output)

    assert set(removed) == {str(partial.resolve()), str(publish.resolve())}
    assert not partial.exists()
    assert not publish.exists()


def test_killed_producer_trees_are_refused_while_live_then_restart_cleaned(
        tmp_path, monkeypatch):
    output = tmp_path / "native-bridge"
    process = subprocess.Popen([
        sys.executable, "-c",
        "from threading import Event; Event().wait(60)",
    ])
    partial = tmp_path / f".native-bridge.partial-{process.pid}"
    publish = tmp_path / f".native-bridge.publish-{process.pid}"
    partial.mkdir()
    publish.mkdir()
    try:
        with pytest.raises(RuntimeError, match="live HRRR producer PID"):
            _cleanup_stale_producer_trees(output)
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10.0)

    launched = []

    def fake_popen(command, **options):
        launched.append((command, options))
        return SimpleNamespace(pid=os.getpid(), poll=lambda: 0)

    monkeypatch.setattr("tools.hrrr_pipeline.subprocess.Popen", fake_popen)
    producer = HrrrPipelineProducer(
        decoder=Path(sys.executable),
        series=_series(tmp_path / "restart-series.tsv", (0, 1)),
        output=output,
        signals=tmp_path / "restart-signals",
        cycle="2026-07-20 00:00:00",
        window=(0, 1, 0, 1), workers="1",
        log=tmp_path / "restart-decoder.log",
    )
    producer.start()

    assert set(producer.stale_cleanup) == {
        str(partial.resolve()), str(publish.resolve())}
    assert not partial.exists()
    assert not publish.exists()
    assert launched[0][0][0] == sys.executable


def test_cancel_waits_for_termination_then_removes_only_its_pid_trees(
        tmp_path, monkeypatch):
    series = _series(tmp_path / "series.tsv", (0, 1))
    output = tmp_path / "native-bridge"
    pid = 87654321
    partial = tmp_path / f".native-bridge.partial-{pid}"
    publish = tmp_path / f".native-bridge.publish-{pid}"
    partial.mkdir()
    publish.mkdir()
    foreign = tmp_path / ".native-bridge.partial-87654322"
    foreign.mkdir()
    events = []

    class Process:
        returncode = None

        def __init__(self):
            self.pid = pid

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            assert events == ["terminate"]
            events.append("wait")
            self.returncode = -15
            return self.returncode

    original_remove = pipeline_module._remove_owned_producer_tree

    def observed_remove(requested_output, role, requested_pid):
        assert "wait" in events
        events.append(f"remove-{role}")
        return original_remove(requested_output, role, requested_pid)

    monkeypatch.setattr(
        pipeline_module, "_remove_owned_producer_tree", observed_remove)
    producer = HrrrPipelineProducer(
        decoder=Path(sys.executable), series=series, output=output,
        signals=tmp_path / "signals", cycle="2026-07-20 00:00:00",
        window=(0, 1, 0, 1), workers="1", log=tmp_path / "decoder.log")
    producer.process = Process()

    producer.cancel()

    assert events == [
        "terminate", "wait", "remove-partial", "remove-publish"]
    assert producer.cancel_cleanup == {
        "partial": str(partial.resolve()),
        "publish": str(publish.resolve()),
    }
    assert not partial.exists()
    assert not publish.exists()
    assert foreign.is_dir()


def test_cancel_cleans_retained_staging_after_failed_producer_already_exited(
        tmp_path):
    series = _series(tmp_path / "series.tsv", (0, 1))
    output = tmp_path / "native-bridge"
    pid = 87654323
    partial = tmp_path / f".native-bridge.partial-{pid}"
    partial.mkdir()
    producer = HrrrPipelineProducer(
        decoder=Path(sys.executable), series=series, output=output,
        signals=tmp_path / "signals", cycle="2026-07-20 00:00:00",
        window=(0, 1, 0, 1), workers="1", log=tmp_path / "decoder.log")
    producer.process = SimpleNamespace(pid=pid, poll=lambda: 9)

    producer.cancel()

    assert not partial.exists()
    assert producer.cancel_cleanup["partial"] == str(partial.resolve())
    assert producer.cancel_cleanup["publish"] is None


def test_cancel_reaps_exit_between_poll_and_terminate_before_cleanup(tmp_path):
    series = _series(tmp_path / "series.tsv", (0, 1))
    output = tmp_path / "native-bridge"
    pid = 87654324
    partial = tmp_path / f".native-bridge.partial-{pid}"
    partial.mkdir()

    class RacedExit:
        def __init__(self):
            self.pid = pid
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            raise ProcessLookupError(pid)

        def wait(self, timeout):
            self.returncode = 1
            return self.returncode

    producer = HrrrPipelineProducer(
        decoder=Path(sys.executable), series=series, output=output,
        signals=tmp_path / "signals", cycle="2026-07-20 00:00:00",
        window=(0, 1, 0, 1), workers="1", log=tmp_path / "decoder.log")
    producer.process = RacedExit()

    producer.cancel()

    assert not partial.exists()
    assert producer.cancel_cleanup["partial"] == str(partial.resolve())


def test_cancel_cleanup_permission_error_preserves_primary_failure_and_owner_scope(
        tmp_path, monkeypatch):
    series = _series(tmp_path / "series.tsv", (0, 1))
    output = tmp_path / "native-bridge"
    pid = 87654325
    partial = tmp_path / f".native-bridge.partial-{pid}"
    publish = tmp_path / f".native-bridge.publish-{pid}"
    foreign = tmp_path / ".native-bridge.partial-87654326"
    partial.mkdir()
    publish.mkdir()
    foreign.mkdir()
    producer = HrrrPipelineProducer(
        decoder=Path(sys.executable), series=series, output=output,
        signals=tmp_path / "signals", cycle="2026-07-20 00:00:00",
        window=(0, 1, 0, 1), workers="1", log=tmp_path / "decoder.log")
    producer.process = SimpleNamespace(pid=pid, poll=lambda: 9)
    original_remove = pipeline_module._remove_owned_producer_tree

    def windows_locked_remove(requested_output, role, requested_pid):
        assert requested_output == output
        assert requested_pid == pid
        if role == "partial":
            raise PermissionError("consumer still maps a staging field")
        return original_remove(requested_output, role, requested_pid)

    monkeypatch.setattr(
        pipeline_module, "_remove_owned_producer_tree", windows_locked_remove)

    def fail_after_consumer_setup():
        try:
            raise RuntimeError("primary NVRTC compilation failed")
        except BaseException:
            producer.cancel()
            raise

    with pytest.raises(
            RuntimeError, match="primary NVRTC compilation failed") as caught:
        fail_after_consumer_setup()

    assert type(caught.value) is RuntimeError
    assert producer.cancel_cleanup == {
        "partial": (
            "retained:PermissionError:consumer still maps a staging field"),
        "publish": str(publish.resolve()),
    }
    assert partial.is_dir()
    assert not publish.exists()
    assert foreign.is_dir()


@pytest.mark.parametrize(
    ("file_mode", "file_attributes", "reparse_tag"),
    (
        (stat.S_IFLNK, 0, 0),
        (stat.S_IFDIR, 0x00000400, 0),
        (stat.S_IFDIR, 0, 0xA0000003),
    ),
)
def test_stale_cleanup_rejects_posix_link_and_python311_windows_reparse_metadata(
        tmp_path, monkeypatch, file_mode, file_attributes, reparse_tag):
    output = tmp_path / "native-bridge"
    candidate = tmp_path / ".native-bridge.partial-99999999"
    candidate.mkdir()
    real_lstat = os.lstat

    def windows_311_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if Path(path) == candidate:
            return SimpleNamespace(
                st_mode=file_mode,
                st_file_attributes=file_attributes,
                st_reparse_tag=reparse_tag,
            )
        return result

    monkeypatch.setattr(pipeline_module.os, "lstat", windows_311_lstat)

    with pytest.raises(RuntimeError, match="unsafe stale HRRR partial"):
        _cleanup_stale_producer_trees(output)
    assert candidate.is_dir()
