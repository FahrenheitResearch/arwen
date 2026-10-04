"""Built Rust payload writer: exact files, bounded batches and failure cleanup."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest import prepared_cache, prepared_writer


@pytest.fixture
def native_writer():
    try:
        entry = prepared_writer.native_writer()
    except (OSError, RuntimeError) as error:
        pytest.skip(str(error))
    if entry is None:
        pytest.skip("built CPU bridge lacks the native prepared-array writer")
    return entry


def _writer(directory):
    directory.mkdir()
    return prepared_cache._BundleWriter(directory)


def _arrays():
    return [
        np.arange(192, dtype=np.float32).reshape(3, 8, 8),
        np.asarray([-0.0, np.nan, np.inf, -np.inf], dtype=np.float64),
        np.asarray(17, dtype=np.int32),
        np.empty((0, 3), dtype=np.float32),
        np.arange(96, dtype=np.float32).reshape(8, 12)[:, ::2],
        np.arange(32, dtype=">i4"),
        np.zeros(5, dtype=[("label", "U2"), ("value", ">f4")]),
    ]


@pytest.mark.parametrize("width", [1, 8, 32])
def test_complete_npy_files_and_content_hashes_match_serial(native_writer, tmp_path, monkeypatch, width):
    arrays = _arrays()
    serial = _writer(tmp_path / "serial")
    for index, array in enumerate(arrays):
        serial.add(f"field/{index}", array)
    parallel = _writer(tmp_path / "parallel")
    monkeypatch.setattr(prepared_writer, "batch_budget", lambda: (width, 1024**3))
    with parallel.immutable_batch():
        for index, array in enumerate(arrays):
            parallel.add(f"field/{index}", array)
    assert parallel.manifest == serial.manifest
    assert parallel.payload_bytes == serial.payload_bytes
    for item in serial.manifest.values():
        name = item["file"]
        assert (tmp_path / "parallel" / name).read_bytes() == (tmp_path / "serial" / name).read_bytes()


def test_add_finishes_a_snapshot_outside_an_immutable_scope(tmp_path):
    writer = _writer(tmp_path / "snapshot")
    array = np.arange(16, dtype=np.float32)
    writer.add("first", array)
    array[:] = -1
    got = np.load(tmp_path / "snapshot" / "a00000.npy", allow_pickle=False)
    np.testing.assert_array_equal(got, np.arange(16, dtype=np.float32))


def test_explicit_single_worker_reaches_the_native_writer(native_writer, tmp_path, monkeypatch, capsys):
    from gpuwm.ingest.preparation_workers import PREPARATION_THREADS_ENV
    monkeypatch.setenv(PREPARATION_THREADS_ENV, "1")
    assert prepared_writer.batch_budget()[0] == 1
    writer = _writer(tmp_path / "one-worker")
    with writer.immutable_batch():
        writer.add("first", np.arange(16, dtype=np.float32))
        writer.add("second", np.arange(32, dtype=np.float32))
    import json
    rows = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().err.splitlines()
            if line.startswith("GPUWM_PREP_WRITE ")]
    assert rows and all(row["requested_workers"] == row["effective_workers"] == 1 for row in rows)


def test_batches_flush_before_exceeding_host_transfer_budget(native_writer, tmp_path, monkeypatch):
    original = prepared_writer.write_arrays
    observed = []
    def measured(entry, arrays, *, workers):
        observed.append(sum(array.nbytes for _path, array in arrays))
        return original(entry, arrays, workers=workers)
    monkeypatch.setattr(prepared_writer, "write_arrays", measured)
    monkeypatch.setattr(prepared_writer, "batch_budget", lambda: (8, 2 * 4096))
    writer = _writer(tmp_path / "bounded")
    class DeviceArray:
        nbytes = 4096
        def get(self):
            return np.ones(1024, dtype=np.float32)
    with writer.immutable_batch():
        for index in range(7):
            writer.add(str(index), DeviceArray())
    assert observed == [8192, 8192, 8192, 4096]
    assert len(writer.manifest) == 7


def test_native_failure_keeps_existing_private_file_and_publishes_no_arrays(native_writer, tmp_path, monkeypatch):
    writer = _writer(tmp_path / "failed")
    sentinel = writer.temporary / "a00001.npy.tmp"
    sentinel.write_bytes(b"existing-file")
    monkeypatch.setattr(prepared_writer, "batch_budget", lambda: (8, 1024**3))
    with pytest.raises(OSError):
        with writer.immutable_batch():
            writer.add("first", np.arange(16, dtype=np.float32))
            writer.add("second", np.arange(32, dtype=np.float32))
    assert sentinel.read_bytes() == b"existing-file"
    assert list(writer.temporary.iterdir()) == [sentinel]
    assert not writer.manifest


def test_interrupted_atomic_rename_leaves_no_unpublished_owned_temporaries(native_writer, tmp_path, monkeypatch):
    writer = _writer(tmp_path / "rename-failed")
    original = prepared_cache._replace_file
    def interrupted(source, target):
        if Path(target).name == "a00001.npy":
            raise OSError("injected second rename failure")
        return original(source, target)
    monkeypatch.setattr(prepared_cache, "_replace_file", interrupted)
    monkeypatch.setattr(prepared_writer, "batch_budget", lambda: (8, 1024**3))
    with pytest.raises(OSError, match="injected"):
        with writer.immutable_batch():
            for index in range(3):
                writer.add(str(index), np.arange(16, dtype=np.float32))
    assert sorted(path.name for path in writer.temporary.iterdir()) == ["a00000.npy"]
    assert list(writer.manifest) == ["0"]
