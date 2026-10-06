"""Built Rust mapping verification keeps prepared bytes and corruption checks."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest import prepared_cache, prepared_store, prepared_writer


@pytest.fixture
def native_hasher():
    try:
        entry = prepared_writer.native_hasher()
    except (OSError, RuntimeError) as error:
        pytest.skip(str(error))
    if entry is None:
        pytest.skip("built CPU bridge lacks native prepared-array hashing")
    return entry


def _reader(directory, arrays):
    directory.mkdir()
    rows = {}
    for index, array in enumerate(arrays):
        key = f"field/{index}"
        filename = f"a{index:05d}.npy"
        np.save(directory / filename, array, allow_pickle=False)
        rows[key] = {"file": filename, "shape": list(array.shape),
                     "dtype": str(array.dtype), "nbytes": array.nbytes,
                     "sha256": prepared_cache._array_sha256(array)}
    reader = SimpleNamespace(path=directory, arrays=rows,
                             payload_bytes=sum(a.nbytes for a in arrays))
    reader.read_array = lambda key: prepared_cache.read_manifest_array(
        directory, key, rows[key])
    return reader


def _arrays():
    return [np.arange(192, dtype=np.float32).reshape(3, 8, 8),
            np.asarray([-0.0, np.nan, np.inf, -np.inf], dtype=np.float64),
            np.asarray(17, dtype=np.int32), np.empty((0, 3), dtype=np.float32),
            np.arange(32, dtype=">i4"),
            np.zeros(5, dtype=[("label", "U2"), ("value", ">f4")])]


@pytest.mark.parametrize("workers", [1, 8, 32])
def test_built_native_hash_matches_manifest_bytes(native_hasher, tmp_path, workers):
    arrays = _arrays()
    reader = _reader(tmp_path / "arrays", arrays)
    maps = [np.load(reader.path / spec["file"], mmap_mode="r", allow_pickle=False)
            for spec in reader.arrays.values()]
    hashes = prepared_writer.hash_arrays(native_hasher, maps, workers=workers)
    assert hashes == [spec["sha256"] for spec in reader.arrays.values()]
    assert all(not item.flags.writeable for item in maps)


def test_native_verification_reuses_mapping_without_full_array_read(
        native_hasher, tmp_path, monkeypatch):
    arrays = _arrays()
    reader = _reader(tmp_path / "cache", arrays)
    reader.read_array = lambda _: pytest.fail("native verification materialized a full array")
    monkeypatch.setattr(prepared_writer, "batch_budget", lambda: (2, 512))
    payload = prepared_store._CachePayload(reader, log=lambda _: None)
    for key, expected in zip(reader.arrays, arrays):
        first = payload.verified_array(key)
        assert first is payload[key]
        assert isinstance(first, np.ndarray)
        assert not first.flags.writeable
        with pytest.raises(ValueError):
            first.setflags(write=True)
        assert first.tobytes() == expected.tobytes()


def test_older_bridge_keeps_whole_array_integrity_checks(tmp_path, monkeypatch):
    reader = _reader(tmp_path / "legacy", _arrays())
    monkeypatch.setattr(prepared_writer, "native_hasher", lambda: None)
    original = reader.read_array
    checked = []
    def read(key):
        checked.append(key)
        return original(key)
    reader.read_array = read
    payload = prepared_store._CachePayload(reader, verify=False)
    payload.verified_array("field/0")
    payload.verify_all(log=lambda _: None)
    payload.verified_array("field/0")
    assert sorted(checked) == sorted(reader.arrays)


def test_fortran_order_legacy_payload_preserves_digest_interpretation(
        native_hasher, tmp_path):
    array = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(4, 6))
    reader = _reader(tmp_path / "fortran", [array])
    payload = prepared_store._CachePayload(reader, log=lambda _: None)
    assert payload["field/0"].tobytes() == array.tobytes()


def test_native_verification_refuses_corrupt_payload(native_hasher, tmp_path):
    reader = _reader(tmp_path / "corrupt", [np.arange(64, dtype=np.float32)])
    path = reader.path / "a00000.npy"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="fails its manifest"):
        prepared_store._CachePayload(reader, log=lambda _: None)


@pytest.mark.parametrize("field,value", [
    ("shape", [4, 16]), ("dtype", "float64"), ("nbytes", 255)])
def test_mapped_metadata_must_match_manifest(native_hasher, tmp_path, field, value):
    reader = _reader(tmp_path / "metadata", [np.arange(64, dtype=np.float32)])
    reader.arrays["field/0"][field] = value
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="against a manifest"):
        prepared_store._CachePayload(reader, log=lambda _: None)


def test_missing_payload_is_refused(native_hasher, tmp_path):
    reader = _reader(tmp_path / "missing", [np.arange(64, dtype=np.float32)])
    (reader.path / "a00000.npy").unlink()
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="unreadable"):
        prepared_store._CachePayload(reader, log=lambda _: None)


def test_replaced_verified_file_is_refused(native_hasher, tmp_path):
    reader = _reader(tmp_path / "replace", [np.arange(64, dtype=np.float32)])
    payload = prepared_store._CachePayload(reader, log=lambda _: None)
    replacement = reader.path / "replacement.npy"
    np.save(replacement, np.zeros(64, dtype=np.float32), allow_pickle=False)
    replacement.replace(reader.path / "a00000.npy")
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="changed during mapped use"):
        payload["field/0"]


def test_change_during_verification_is_refused(native_hasher, tmp_path, monkeypatch):
    reader = _reader(tmp_path / "changed", [np.arange(64, dtype=np.float32)])
    original = prepared_writer.hash_arrays
    def changed(entry, arrays, *, workers):
        digests = original(entry, arrays, workers=workers)
        with (reader.path / "a00000.npy").open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\xff")
        return digests
    monkeypatch.setattr(prepared_writer, "hash_arrays", changed)
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="changed during mapped use"):
        prepared_store._CachePayload(reader, log=lambda _: None)


def test_native_hash_abi_rejects_zero_workers(native_hasher):
    array = np.arange(8, dtype=np.float32)
    prefix = ctypes.create_string_buffer(b"<f4;[8];")
    jobs = (prepared_writer._ArrayHash * 1)(prepared_writer._ArrayHash(
        array.ctypes.data, array.nbytes, ctypes.addressof(prefix), 8))
    digests = (ctypes.c_ubyte * 32)()
    assert native_hasher(jobs, 1, 0, digests) == 2
