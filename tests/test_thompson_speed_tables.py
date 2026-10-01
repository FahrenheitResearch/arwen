"""The single-read classic table load returns exactly the records it replaced.

``load_validated_classic_tables`` reads each asset once, checks it and parses
those bytes in place; ``load_classic_device_tables`` hashes in place and
round-trips through one page-locked buffer.  The references are the
streaming readers (``validate_table_assets`` plus
``read_classic_table_directory``) and the former copy-then-hash digest.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from gpuwm.core import thompson_contract as contract
from gpuwm.core.thompson_contract import TableRecord


def _write_records(path: Path, records, rng, *, marker=4) -> bytes:
    fmt = "<i" if marker == 4 else "<q"
    blob = bytearray()
    for record in records:
        payload = rng.standard_normal(record.values).astype("<f8").tobytes()
        blob += struct.pack(fmt, len(payload)) + payload
        blob += struct.pack(fmt, len(payload))
    path.write_bytes(bytes(blob))
    return bytes(blob)


RECORDS = (TableRecord("a", (3, 4)), TableRecord("b", (5,)),
           TableRecord("c", (2, 3, 2)))


def test_in_memory_records_equal_the_streamed_ones(tmp_path):
    path = tmp_path / "records.dat"
    data = _write_records(path, RECORDS, np.random.default_rng(3))
    streamed = contract.read_sequential_records(path, RECORDS)
    in_place = contract.read_sequential_records(path, RECORDS, data=data)
    assert list(in_place) == list(streamed)
    for name in streamed:
        a, b = streamed[name], in_place[name]
        assert a.shape == b.shape and b.flags.f_contiguous
        assert a.dtype == b.dtype == np.float64
        assert a.tobytes(order="F") == b.tobytes(order="F")
        # A view of immutable bytes cannot be written through.
        assert not b.flags.writeable


def test_in_memory_records_keep_every_refusal(tmp_path):
    path = tmp_path / "records.dat"
    data = _write_records(path, RECORDS, np.random.default_rng(4))
    with pytest.raises(TypeError, match="immutable bytes"):
        contract.read_sequential_records(path, RECORDS, data=bytearray(data))
    with pytest.raises(ValueError, match="truncated|missing|mismatched"):
        contract.read_sequential_records(path, RECORDS, data=data[:-9])
    with pytest.raises(ValueError, match="unexpected bytes"):
        contract.read_sequential_records(path, RECORDS, data=data + b"\0")
    bad = bytearray(data)
    bad[0:4] = struct.pack("<i", 8)
    with pytest.raises(ValueError, match="marker declares"):
        contract.read_sequential_records(path, RECORDS, data=bytes(bad))


def _staged_root() -> Path | None:
    try:
        from gpuwm.physics_compat import thompson_table_root
        root = Path(thompson_table_root())
    except Exception:
        return None
    ok = all((root / asset.filename).is_file()
             for asset in contract.CLASSIC_TABLE_ASSETS)
    return root if ok else None


def test_single_read_load_equals_the_streaming_readers():
    root = _staged_root()
    if root is None:
        pytest.skip("canonical Thompson tables are not staged")
    table_set = contract.load_validated_classic_tables(root)
    contract.validate_table_assets(root)
    reference = contract.read_classic_table_directory(root)
    assert list(table_set.arrays) == list(reference)
    for name, expected in reference.items():
        actual = table_set.arrays[name]
        assert actual.shape == expected.shape and actual.flags.f_contiguous
        assert not actual.flags.writeable
        assert actual.tobytes(order="F") == expected.tobytes(order="F"), name


def test_single_read_load_refuses_a_substituted_byte(tmp_path):
    root = _staged_root()
    if root is None:
        pytest.skip("canonical Thompson tables are not staged")
    for asset in contract.CLASSIC_TABLE_ASSETS:
        source = root / asset.filename
        if asset.filename == contract.AUXILIARY_TABLE_FILE:
            blob = bytearray(source.read_bytes())
            blob[4096] ^= 0x01
            (tmp_path / asset.filename).write_bytes(bytes(blob))
        else:
            try:
                os.symlink(source, tmp_path / asset.filename)
            except OSError:
                pytest.skip("no symlinks on this filesystem")
    with pytest.raises(ValueError, match="SHA-256"):
        contract.load_validated_classic_tables(tmp_path)


@pytest.mark.gpu
def test_device_tables_hash_and_round_trip_as_before():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    root = _staged_root()
    if root is None:
        pytest.skip("canonical Thompson tables are not staged")
    from gpuwm.core.thompson_runtime import load_classic_device_tables
    owner = load_classic_device_tables(root, cache=False)
    assert owner.roundtrip_verified
    reference = contract.read_classic_table_directory(root)
    for name, host in reference.items():
        digest = hashlib.sha256(host.tobytes(order="F")).hexdigest()
        assert owner.array_sha256[name] == digest, name
        device = owner.arrays[name]
        assert device.flags.f_contiguous and device.shape == host.shape
        assert cp.asnumpy(device, order="F").tobytes(order="F") == (
            host.tobytes(order="F")), name
