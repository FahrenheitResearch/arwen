"""Descriptor-free NPY mappings preserve bytes and escaped-view lifetimes."""
from __future__ import annotations

import gc
import mmap
import os
import struct
import weakref

import numpy as np
import pytest

from gpuwm.ingest import prepared_mmap


@pytest.fixture(params=["default", "native"])
def mapping_mode(request, monkeypatch):
    if request.param == "native":
        if os.name != "posix":
            pytest.skip("native fallback uses the Unix mmap ABI")
        monkeypatch.setattr(prepared_mmap, "_readonly_file_buffer",
                            prepared_mmap._native_unix_buffer)
    return request.param


def _arrays():
    return [
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        np.asfortranarray(np.arange(24, dtype=np.float64).reshape(4, 6)),
        np.asarray(-0.0, dtype=np.float64),
        np.asarray([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32),
        np.arange(16, dtype=">i4"),
        np.zeros(3, dtype=[("label", "U2"), ("value", ">f4")]),
        np.empty((0, 3), dtype=np.float32),
        np.empty(3, dtype="V0"),
        np.asarray(b"", dtype="V0"),
    ]


def test_array_metadata_layout_and_bytes(mapping_mode, tmp_path):
    for index, expected in enumerate(_arrays()):
        path = tmp_path / f"array-{index}.npy"
        np.save(path, expected, allow_pickle=False)
        observed = prepared_mmap.map_npy_readonly(path)
        assert observed.shape == expected.shape
        assert observed.dtype == expected.dtype
        assert observed.tobytes() == expected.tobytes()
        assert observed.tobytes(order="F") == expected.tobytes(order="F")
        assert observed.flags.c_contiguous == expected.flags.c_contiguous
        assert observed.flags.f_contiguous == expected.flags.f_contiguous
        assert not observed.flags.writeable
        with pytest.raises(ValueError):
            observed.setflags(write=True)
        if observed.size and observed.dtype.itemsize:
            with pytest.raises(ValueError, match="read-only"):
                observed.flat[0] = observed.flat[0]


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_supported_npy_versions(mapping_mode, tmp_path, version):
    path = tmp_path / "array.npy"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, expected, version=version,
                                  allow_pickle=False)
    observed = prepared_mmap.map_npy_readonly(path)
    assert observed.shape == expected.shape
    assert observed.dtype == expected.dtype
    assert observed.tobytes() == expected.tobytes()


def test_format_three_unicode_field_names(mapping_mode, tmp_path):
    expected = np.zeros(4, dtype=[("\u03b8", ">f4"), ("\u6c34", "U2")])
    path = tmp_path / "unicode.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, expected, version=(3, 0),
                                  allow_pickle=False)
    observed = prepared_mmap.map_npy_readonly(path)
    assert observed.dtype == expected.dtype
    assert observed.tobytes() == expected.tobytes()


def test_native_mapping_owner_lives_until_last_array_view(tmp_path):
    if os.name != "posix":
        pytest.skip("native fallback uses the Unix mmap ABI")
    path = tmp_path / "array.npy"
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.save(path, expected, allow_pickle=False)
    with path.open("rb") as stream:
        buffer = prepared_mmap._native_unix_buffer(stream.fileno(),
                                                  path.stat().st_size)
        owner = buffer.obj._mapping_owner
        address = owner._finalizer.peek()[2][0]
        length = owner._finalizer.peek()[2][1]
        releases = []
        _create, release = prepared_mmap._unix_mapping_functions()
        owner._finalizer.detach()
        owner._finalizer = weakref.finalize(
            owner, lambda: (releases.append((address, length)),
                            release(address, length)))
    owner_ref = weakref.ref(owner)
    array = np.frombuffer(buffer, dtype=np.uint8)
    escaped = array[128:136]
    expected_bytes = escaped.tobytes()
    del owner, buffer, array
    gc.collect()
    assert owner_ref() is not None
    assert not releases
    assert escaped.tobytes() == expected_bytes
    with pytest.raises(ValueError):
        escaped.setflags(write=True)
    del escaped
    gc.collect()
    assert owner_ref() is None
    assert releases == [(address, length)]


def test_default_mapping_keeps_escaped_views(mapping_mode, tmp_path):
    path = tmp_path / "array.npy"
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.save(path, expected, allow_pickle=False)
    array = prepared_mmap.map_npy_readonly(path)
    escaped = array[1:, 1::2]
    del array
    gc.collect()
    assert escaped.tobytes() == expected[1:, 1::2].tobytes()
    assert not escaped.flags.writeable
    with pytest.raises(ValueError):
        escaped.setflags(write=True)


def test_file_pages_keep_their_memory_admission_price(mapping_mode, tmp_path):
    from gpuwm.core.devices_memory import priced_host_live_bytes
    path = tmp_path / "array.npy"
    np.save(path, np.arange(32, dtype=np.float32).reshape(4, 8), allow_pickle=False)
    mapped = prepared_mmap.map_npy_readonly(path)
    view = mapped[1:]
    assert prepared_mmap.is_file_backed_array(mapped)
    assert prepared_mmap.is_file_backed_array(view)
    assert priced_host_live_bytes({"mapped": mapped, "view": view}, {}) == {
        "host_store_bytes": 0, "host_boundary_bytes": 0}
    for materialized in (mapped.copy(), np.array(mapped, copy=True), mapped + 1):
        assert not prepared_mmap.is_file_backed_array(materialized)
        assert priced_host_live_bytes({"copy": materialized}, {}) == {
            "host_store_bytes": mapped.nbytes, "host_boundary_bytes": 0}


def test_anonymous_mapping_earns_ordinary_allocation_credit():
    from gpuwm.core.devices_memory import priced_host_live_bytes
    buffer = mmap.mmap(-1, 128)
    array = np.frombuffer(buffer, dtype=np.uint8)
    assert not prepared_mmap.is_file_backed_array(array)
    assert priced_host_live_bytes({"array": array}, {}) == {
        "host_store_bytes": 128, "host_boundary_bytes": 0}


def test_unix_maps_do_not_retain_descriptors(mapping_mode, tmp_path):
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("descriptor inventory requires procfs")
    path = tmp_path / "array.npy"
    expected = np.arange(16, dtype=np.float32)
    np.save(path, expected, allow_pickle=False)
    prepared_mmap._unix_mapping_functions()
    before = len(os.listdir("/proc/self/fd"))
    arrays = [prepared_mmap.map_npy_readonly(path) for _ in range(128)]
    assert len(os.listdir("/proc/self/fd")) == before
    assert all(array.tobytes() == expected.tobytes() for array in arrays)
    del arrays
    gc.collect()
    assert len(os.listdir("/proc/self/fd")) == before


def test_empty_array_needs_no_mapping(tmp_path, monkeypatch):
    path = tmp_path / "empty.npy"
    np.save(path, np.empty((4, 0, 7), dtype=np.float64), allow_pickle=False)
    monkeypatch.setattr(prepared_mmap, "_readonly_file_buffer",
                        lambda *_: pytest.fail("mapped a zero-byte payload"))
    result = prepared_mmap.map_npy_readonly(path)
    assert result.shape == (4, 0, 7)
    assert not result.flags.writeable


def test_object_array_is_not_loaded_or_mapped(tmp_path, monkeypatch):
    path = tmp_path / "object.npy"
    np.save(path, np.asarray(["value"], dtype=object), allow_pickle=True)
    monkeypatch.setattr(prepared_mmap, "_readonly_file_buffer",
                        lambda *_: pytest.fail("mapped object payload"))
    with pytest.raises(ValueError, match="object arrays"):
        prepared_mmap.map_npy_readonly(path)


def test_truncated_payload_refused_before_mapping(tmp_path, monkeypatch):
    path = tmp_path / "truncated.npy"
    np.save(path, np.arange(16, dtype=np.float32), allow_pickle=False)
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    monkeypatch.setattr(prepared_mmap, "_readonly_file_buffer",
                        lambda *_: pytest.fail("mapped truncated payload"))
    with pytest.raises(ValueError, match="shorter than its declared shape"):
        prepared_mmap.map_npy_readonly(path)


def test_declared_header_length_is_bounded_before_read(tmp_path):
    path = tmp_path / "large-header.npy"
    path.write_bytes(np.lib.format.magic(2, 0) + struct.pack("<I", 2**30))
    with pytest.raises(ValueError, match="safe size limit"):
        prepared_mmap.map_npy_readonly(path)


@pytest.mark.parametrize("shape", [(-1, -1), (True,)])
def test_invalid_shape_is_refused_before_mapping(tmp_path, monkeypatch, shape):
    path = tmp_path / "invalid-shape.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "<f4", "fortran_order": False, "shape": shape})
    monkeypatch.setattr(prepared_mmap, "_readonly_file_buffer",
                        lambda *_: pytest.fail("mapped an invalid array shape"))
    with pytest.raises(ValueError, match="nonnegative integer dimensions"):
        prepared_mmap.map_npy_readonly(path)


@pytest.mark.parametrize("contents", [b"", b"not-npy", np.lib.format.magic(9, 0)])
def test_bad_magic_and_versions_are_refused(tmp_path, contents):
    path = tmp_path / "bad.npy"
    path.write_bytes(contents)
    with pytest.raises((ValueError, EOFError)):
        prepared_mmap.map_npy_readonly(path)


def test_mapping_os_error_is_not_reclassified(tmp_path, monkeypatch):
    path = tmp_path / "array.npy"
    np.save(path, np.arange(16, dtype=np.float32), allow_pickle=False)
    error = OSError(24, "descriptor limit")
    monkeypatch.setattr(prepared_mmap, "_readonly_file_buffer",
                        lambda *_: (_ for _ in ()).throw(error))
    with pytest.raises(OSError) as captured:
        prepared_mmap.map_npy_readonly(path)
    assert captured.value is error
