"""A scratch disk that cannot hold the frame stream is a disk refusal.

The decode engine stages every decoded valid time on disk before the
preparation reads it back; a global source over two days is tens of GB.
When that disk fills, the refusal has to say so: the folder, the bytes
and ``GPUWM_COMPOSE_SCRATCH``, raised as its own class, never as a
``FileNotFoundError`` telling the user to supply an input file they
already supplied.
"""
from __future__ import annotations

import errno
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gpuwm import mapped_engine_bridge as bridge
from gpuwm.ingest.source_coverage import (
    PreparationRefusal, ScratchDiskRefusal, report_preparation_refusal)

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tools" / "rw_wps" / "crates" / "mapped-engine" / "tests" / "goldens"
DEV_FULL = Path("/dev/full")


def _golden():
    """The owned NetCDF golden: its mapping and its one input file."""

    document = json.loads(
        (GOLDENS / "netcdf-pressure-level.json").read_text(encoding="utf-8"))
    return (ROOT / document["mapping"],
            [GOLDENS / name for name in document["input_names"]])


def _engine() -> Path:
    try:
        return bridge.require_engine()
    except FileNotFoundError as error:
        pytest.skip(f"engine not built in this checkout: {error}")


def test_a_disk_full_refusal_is_its_own_preparation_refusal():
    error = bridge.refusal_error(
        {"class": "disk_full",
         "message": "the frame stream in /scratch/composed needs 88 bytes",
         "remedy": "free space"},
        ["gpuwm_mapped_engine", "compose"])
    assert type(error) is ScratchDiskRefusal
    assert isinstance(error, PreparationRefusal)
    assert not isinstance(error, FileNotFoundError)
    assert "/scratch/composed" in str(error)
    assert "GPUWM_COMPOSE_SCRATCH" in error.remedy

    # The door prints it as two sentences, the remedy naming the variable.
    stream = io.StringIO()
    report_preparation_refusal(error, stream=stream)
    first, second = stream.getvalue().splitlines()
    assert first.startswith("prep: REFUSED: the frame stream in /scratch/composed")
    assert second.startswith("remedy: set GPUWM_COMPOSE_SCRATCH")


def test_any_other_write_failure_is_an_oserror_that_says_what_failed():
    error = bridge.refusal_error(
        {"class": "write_failed",
         "message": "cannot create the output directory /x: Permission denied",
         "remedy": "make it writable"}, [])
    assert type(error) is OSError
    assert "cannot create the output directory /x" in str(error)


@pytest.mark.parametrize("code, raised", [
    (errno.ENOSPC, ScratchDiskRefusal),
    (getattr(errno, "EDQUOT", errno.ENOSPC), ScratchDiskRefusal),
    (errno.EACCES, PermissionError),
])
def test_the_python_frameset_writer_classifies_its_own_write_failures(
        tmp_path, monkeypatch, code, raised):
    def fail(frames, sink):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(bridge, "_emit_frameset", fail)
    with pytest.raises(OSError if raised is PermissionError else raised) as caught:
        bridge.write_frameset(tmp_path / "frames", [])
    assert type(caught.value) is raised
    if raised is ScratchDiskRefusal:
        assert str(tmp_path / "frames" / "frames.f64") in str(caught.value)


@pytest.mark.skipif(not DEV_FULL.exists(), reason="needs /dev/full")
def test_the_engine_input_list_on_a_full_disk_is_a_disk_refusal(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "inputs.txt").symlink_to(DEV_FULL)
    with pytest.raises(ScratchDiskRefusal) as caught:
        bridge.run_engine("decode", mapping=tmp_path / "m.json",
                          files=[tmp_path / "a.nc"], output=output,
                          engine=tmp_path / "not-launched")
    assert str(output / "inputs.txt") in str(caught.value)


@pytest.mark.skipif(not DEV_FULL.exists(), reason="needs /dev/full")
def test_the_engine_meeting_a_full_disk_mid_stream_refuses_as_one(tmp_path):
    """Run the exe: a stream that fills its disk is a ScratchDiskRefusal.

    ``/dev/full`` answers every write with ENOSPC, so a frame stream
    linked to it is a disk that fills at the first byte.
    """

    engine = _engine()
    mapping, inputs = _golden()
    output = tmp_path / "composed"
    output.mkdir()
    (output / "frames.f64").symlink_to(DEV_FULL)
    with pytest.raises(Exception) as caught:      # noqa: PT011 - typed below
        bridge.run_engine("decode", mapping=mapping, files=inputs,
                          output=output, engine=engine)
    assert type(caught.value) is ScratchDiskRefusal, (
        f"{type(caught.value).__name__}: {caught.value}")
    text = str(caught.value)
    assert f"cannot write the frame stream {output / 'frames.f64'}" in text
    assert "os error 28" in text
    assert "the stream needs " in text
    assert "input list" not in text


def _small_disk_runner() -> list[str] | None:
    """A command prefix under which the child may mount a small tmpfs.

    An unprivileged user namespace can mount a tmpfs of any size; boxes
    whose kernel or security policy refuses one skip the test.
    """

    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    probe = subprocess.run(
        [unshare, "--user", "--map-root-user", "--mount", "true"],
        capture_output=True, check=False)
    return None if probe.returncode else [unshare, "--user", "--map-root-user", "--mount"]


def test_the_engine_refuses_a_stream_bigger_than_a_small_scratch_disk(tmp_path):
    """Run the exe on a 64 KiB disk: refused before the first byte.

    The golden's stream is about 130 kB per valid time, so its two do not
    fit on a 64 KiB tmpfs.  The refusal names the folder and both
    numbers, and the stream file stays empty.
    """

    runner = _small_disk_runner()
    if runner is None:
        pytest.skip("no unprivileged mount namespace for a small tmpfs")
    engine = _engine()
    mapping, inputs = _golden()
    disk = tmp_path / "disk"
    disk.mkdir()
    listing = tmp_path / "inputs.txt"
    listing.write_text("".join(f"{path}\n" for path in inputs), encoding="utf-8")
    script = (
        f"mount -t tmpfs -o size=64k tmpfs {disk} && "
        f"{engine} decode --mapping {mapping} --input-list {listing} "
        f"--output {disk}/composed; "
        f"echo stream-bytes=$(stat -c %s {disk}/composed/frames.f64 2>/dev/null)")
    completed = subprocess.run([*runner, "sh", "-c", script],
                               capture_output=True, text=True, check=False)
    refusal = bridge.parse_refusal(completed.stderr)
    assert refusal is not None, completed.stderr
    error = bridge.refusal_error(refusal, [])
    assert type(error) is ScratchDiskRefusal, str(error)
    text = str(error)
    assert text.startswith("the frame stream needs ")
    assert f" in {disk}/composed and the disk that holds that folder has " in text
    assert "2 valid times" in text
    assert "refused before its first byte" in text
    assert "stream-bytes=0" in completed.stdout
