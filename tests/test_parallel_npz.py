"""gpuwm.io.parallel_npz writes the archive np.savez writes, byte for byte, from many threads."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

from gpuwm.io import parallel_npz


def _payload():
    rng = np.random.default_rng(7)
    big = rng.standard_normal((5, 1100, 1700), dtype=np.float32)       # 37 MB: crosses a piece
    return {
        "__header__": np.frombuffer(b'{"format_version": 9}', dtype=np.uint8),
        "acoustic/ww_pp": big,
        "driver/scalar": np.float64(3.25),
        "state/empty": np.zeros((0, 4), np.float32),
        "state/fortran": np.asfortranarray(rng.standard_normal((40, 30))),
        "state/strided": rng.standard_normal((50, 60))[::2, ::3],
        "state/ints": np.arange(1000, dtype=np.int64).reshape(10, 100),
        "state/bool": rng.standard_normal(999) > 0,
        "state/nan": np.array([np.nan, -0.0, np.inf, 1e-45], np.float32),
    }


@pytest.mark.parametrize("workers", [1, 6])
def test_archive_is_np_savez_byte_for_byte(tmp_path, monkeypatch, workers):
    payload = _payload()
    reference = tmp_path / "savez.npz"
    with reference.open("wb") as stream:
        np.savez(stream, **payload)
    written = tmp_path / "parallel.npz"
    monkeypatch.setattr(parallel_npz, "PIECE_BYTES", 1 << 20)          # many pieces per member
    size = parallel_npz.write_npz(written, payload, workers=workers)
    assert size == written.stat().st_size
    assert written.read_bytes() == reference.read_bytes()
    with zipfile.ZipFile(written) as archive:
        assert archive.testzip() is None
    with np.load(written, allow_pickle=False) as data:
        assert data.files == list(payload)
        for key, value in payload.items():
            stored = data[key]
            assert stored.dtype == np.asarray(value).dtype and stored.shape == np.shape(value)
            assert stored.tobytes() == np.ascontiguousarray(value).tobytes()


def test_crc32_combine():
    import os
    import zlib
    a, b = os.urandom(12345), os.urandom(3 << 20)
    assert parallel_npz.crc32_combine(zlib.crc32(a), zlib.crc32(b), len(b)) == zlib.crc32(a + b)
    assert parallel_npz.crc32_combine(zlib.crc32(a), zlib.crc32(b""), 0) == zlib.crc32(a)


def test_zip64_end_records_match_np_savez(tmp_path, monkeypatch):
    """Archives past 2 GiB (a 3 km member restart is ~3.5 GB) take zipfile's ZIP64 branches:
    offsets in the central directory's extra field, the ZIP64 end record and its locator.
    Lowering the limit in both writers exercises those branches on a small archive."""
    import zipfile as zipfile_module
    limit = 1 << 20
    monkeypatch.setattr(zipfile_module, "ZIP64_LIMIT", limit)
    monkeypatch.setattr(parallel_npz, "_ZIP64_LIMIT", limit)
    monkeypatch.setattr(parallel_npz, "PIECE_BYTES", 1 << 18)
    rng = np.random.default_rng(11)
    payload = {f"state/f{i}": rng.standard_normal((300, 400)).astype(np.float32) for i in range(12)}
    payload["state/big"] = rng.standard_normal((3, 700, 700))          # one member past the limit
    reference = tmp_path / "savez.npz"
    np.savez(reference, **payload)
    written = tmp_path / "parallel.npz"
    parallel_npz.write_npz(written, payload, workers=4)
    assert b"PK\x06\x06" in reference.read_bytes()[-200:]               # the branch really ran
    assert written.read_bytes() == reference.read_bytes()
