"""A preparation reads the inputs its fetch already hashed once, beside the work, not on the critical path.

Named breakage (WOOF fast start, 2026-10-23): the input manifest's authoring and the composed
decode re-hashed the ~22 GB of fetched HRRR GRIB four times on one core (38 s in the
authoring, ~20 s at each end of the decode).
"""

import hashlib
import json
import os

import pytest

from gpuwm import authority_digests
from gpuwm.mapped_source import _require_authority_snapshot, _snapshot_authority


def _fetched(tmp_path, payload=b"GRIB" + b"x" * 4096):
    data = tmp_path / "hrrr.t12z.wrfprsf00.grib2"
    data.write_bytes(payload)
    manifest = tmp_path / "fetch-manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "relpath": data.name, "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest()}]}))
    return data, hashlib.sha256(payload).hexdigest()


def test_a_fetched_file_takes_its_recorded_digest_and_is_reread_beside_the_work(tmp_path, monkeypatch):
    data, digest = _fetched(tmp_path)
    reads = []
    real_hash = authority_digests.hash_file
    monkeypatch.setattr(authority_digests, "hash_file", lambda path: reads.append(path) or real_hash(path))
    with authority_digests.deferred_checks():
        snapshot = _snapshot_authority(data)
        assert snapshot.sha256 == digest
        _require_authority_snapshot(snapshot)
        authority_digests.join()
        assert reads, "the full re-read still runs, beside the work"
    assert snapshot.sha256 == _snapshot_authority(data).sha256  # outside the scope: read on the caller


def test_bytes_that_differ_from_the_fetch_record_fail_the_preparation(tmp_path):
    data, _ = _fetched(tmp_path)
    manifest = tmp_path / "fetch-manifest.json"
    document = json.loads(manifest.read_text())
    document["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="differ from the digest"):
        with authority_digests.deferred_checks():
            assert _snapshot_authority(data).sha256 == "0" * 64


def test_a_file_written_after_its_fetch_record_is_hashed_on_the_caller(tmp_path):
    data, _ = _fetched(tmp_path)
    data.write_bytes(b"GRIB" + b"y" * 4096)
    later = os.stat(tmp_path / "fetch-manifest.json").st_mtime_ns + 10**9
    os.utime(data, ns=(later, later))
    with authority_digests.deferred_checks():
        assert _snapshot_authority(data).sha256 == hashlib.sha256(data.read_bytes()).hexdigest()


def test_a_change_after_validation_is_refused_on_the_caller(tmp_path):
    data, _ = _fetched(tmp_path)
    with pytest.raises(ValueError, match="changed after validation"):
        with authority_digests.deferred_checks():
            snapshot = _snapshot_authority(data)
            data.write_bytes(b"GRIB" + b"z" * 8192)
            _require_authority_snapshot(snapshot)
