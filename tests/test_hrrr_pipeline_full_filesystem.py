"""The decoder failure report, taken with the output filesystem failing.

Every other test of this producer replaces ``subprocess.Popen`` with a
stand-in, which is exactly the gap that let a release gate lose a route:
when the disk filled during an HRRR decode the operator was handed

    RuntimeError: pipeline producer exited 1:

-- no message, no log path, no argv.  Both diagnostic channels wrote to
the filesystem that had just failed (``failure.ready`` through a
write-then-rename, stderr into a log inside the output tree), both
ignored their write errors, and the consumer's unwind then removed the
staging tree that held the last clue.

So these tests run a REAL child process and make REAL writes fail:

* ``/dev/full`` is a kernel device whose every write returns ENOSPC.
  Pointing the decoder log at it reproduces a mid-write full disk
  exactly, at the syscall boundary, without root or a loopback mount.
* A log path that is a directory makes the log unopenable, which is what
  a filesystem with no free inodes/blocks does to ``open(..., "wb")``.
  That branch runs everywhere, including Windows.
* A read-only signals directory keeps the producer from publishing
  ``failure.ready`` at all, which is the state the field failure was
  found in (``preflight.ready`` present, no ``failure.ready``, no
  ``failure.ready.tmp``).

The bar in all three: exit code, argv and the producer's own stderr
reach the caller anyway.
"""

from pathlib import Path
import os
import stat
import sys

import pytest

from tools.hrrr_pipeline import HrrrPipelineProducer, HrrrProducerFailure


#: What the stand-in decoder prints before dying.  A real HRRR decoder
#: fails here with an io error from `write_atmosphere`; what matters to
#: this test is that whatever it says reaches the caller.
DECODER_MESSAGE = "decode f00: No space left on device (os error 28)"

_DECODER_SOURCE = f"""\
import os, sys
from pathlib import Path
try:
    signals = Path(sys.argv[5])
    signals.mkdir(parents=True, exist_ok=True)
    (signals / "preflight.ready").write_text("status\\tPASS\\n")
except OSError:
    # A signals directory this process cannot create or write is exactly
    # what a full filesystem gives it.  Keep going: the point is that the
    # message below still reaches the parent.
    pass
# Fail the way the sealed decoder fails on a full filesystem: say so on
# stderr, and do NOT manage to publish failure.ready.
sys.stderr.write({DECODER_MESSAGE!r} + chr(10))
sys.stderr.flush()
sys.exit(1)
"""


def _decoder(tmp_path: Path) -> Path:
    """A real executable stand-in for the sealed Rust decoder."""

    script = tmp_path / "stub-decoder.py"
    script.write_text(_DECODER_SOURCE, encoding="ascii")
    if os.name == "nt":
        # Windows Popen needs a real image; wrap the script in a .bat so
        # the process that runs is still a genuine child with a genuine
        # argv and a genuine pipe.
        launcher = tmp_path / "stub-decoder.bat"
        launcher.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="ascii")
        return launcher
    launcher = tmp_path / "stub-decoder"
    launcher.write_text(
        f"#!{sys.executable}\n" + _DECODER_SOURCE, encoding="ascii")
    launcher.chmod(0o755)
    return launcher


def _series(path: Path) -> Path:
    path.write_text("0\tatmos-f00\tsoil-f00\n1\tatmos-f01\tsoil-f01\n",
                    encoding="ascii")
    return path


def _producer(tmp_path: Path, *, log: Path,
              signals: Path | None = None) -> HrrrPipelineProducer:
    return HrrrPipelineProducer(
        decoder=_decoder(tmp_path),
        series=_series(tmp_path / "series.tsv"),
        output=tmp_path / "native-bridge",
        signals=tmp_path / "signals" if signals is None else signals,
        cycle="2026-07-20 00:00:00",
        window=(0, 1, 0, 1), workers="1", log=log)


def _run_to_failure(producer: HrrrPipelineProducer) -> HrrrProducerFailure:
    producer.start()
    with pytest.raises(HrrrProducerFailure) as caught:
        producer.finish(timeout=60.0)
    return caught.value


def _assert_diagnosis_is_complete(error: HrrrProducerFailure) -> None:
    """Exit code, argv and the child's own stderr, all present."""

    text = str(error)
    assert "exited 1" in text
    assert error.diagnostics["returncode"] == 1
    # argv: the decoder path and every positional the producer chose.
    assert "--series-workers-ready" in text
    assert "native-bridge" in text
    assert list(error.diagnostics["argv"])[1] == "--series-workers-ready"
    # stderr: the thing that used to be lost.
    assert DECODER_MESSAGE in text
    assert DECODER_MESSAGE in error.diagnostics["output_tail"]
    # The absent failure receipt is reported as absent rather than as
    # nothing at all.
    assert error.diagnostics["signals"]["failure_ready"] is None
    assert "failure.ready: ABSENT" in text
    # And free space, which is the number that explains the whole thing.
    assert error.diagnostics["output_filesystem"]["free_bytes"] is not None


@pytest.mark.skipif(
    not Path("/dev/full").exists(),
    reason="/dev/full (writes fail ENOSPC for real) is Linux-only")
def test_diagnosis_survives_a_log_whose_every_write_returns_enospc(tmp_path):
    """The real-full-disk case: the log opens, then every write fails."""

    producer = _producer(tmp_path, log=Path("/dev/full"))
    error = _run_to_failure(producer)

    _assert_diagnosis_is_complete(error)
    # The failure names the write error rather than silently swallowing it.
    status = error.diagnostics["log"]["status"]
    assert "ENOSPC" in status, status
    assert "ENOSPC" in str(error)


def test_diagnosis_survives_a_log_that_cannot_be_opened_at_all(tmp_path):
    """No free blocks to create the log with: the run still explains itself."""

    unopenable = tmp_path / "decoder.log"
    unopenable.mkdir()
    producer = _producer(tmp_path, log=unopenable)
    error = _run_to_failure(producer)

    _assert_diagnosis_is_complete(error)
    assert error.diagnostics["log"]["status"].startswith("unavailable:")


@pytest.mark.skipif(
    os.name == "nt",
    reason="a read-only directory does not stop writes on Windows")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses the permission bits this test freezes the tree with")
def test_diagnosis_survives_a_signals_directory_that_cannot_be_created(
        tmp_path):
    """``failure.ready`` never appears; the caller is told so, not left blank."""

    signals_parent = tmp_path / "ro"
    signals_parent.mkdir()
    producer = _producer(
        tmp_path, log=tmp_path / "decoder.log",
        signals=signals_parent / "signals")
    # Frozen before the child starts, so neither preflight.ready nor
    # failure.ready can ever be published -- the state the field failure
    # was found in.
    signals_parent.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        error = _run_to_failure(producer)
    finally:
        signals_parent.chmod(0o700)

    _assert_diagnosis_is_complete(error)
    assert not (signals_parent / "signals").exists()


def test_the_unwind_reports_the_staging_tree_before_it_removes_it(tmp_path):
    """Evidence first, cleanup second -- the ordering the field case lost."""

    producer = _producer(tmp_path, log=tmp_path / "decoder.log")
    producer.start()
    staging = (producer.output.parent
               / f".{producer.output.name}.partial-{producer.process.pid}")
    staging.mkdir(parents=True)
    (staging / ".inventory-f00.tsv").write_bytes(b"the last clue")

    producer.process.wait(timeout=60.0)
    with pytest.raises(HrrrProducerFailure) as caught:
        producer.finish(timeout=60.0)
    # The consumer's unwind: this is what used to destroy the tree before
    # anybody had read it.
    producer.cancel()

    assert not staging.exists()
    described = caught.value.diagnostics["staging"]
    assert str(staging) in described
    assert any(".inventory-f00.tsv" in entry
               for entry in described[str(staging)])
    assert ".inventory-f00.tsv" in str(caught.value)
    assert caught.value.diagnostics["removed_after_report"] == [
        str(staging.resolve())]
