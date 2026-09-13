"""A fresh writer proof avoids rereads without accepting another revision."""
import hashlib
import inspect
import os
from pathlib import Path

import pytest

from gpuwm import output_identity


def _completed(path):
    capture = getattr(output_identity, "completed_file_record", None)
    return output_identity.file_record(path) if capture is None else capture(path)


def _record(path, proof):
    if "completed" in inspect.signature(output_identity.file_record).parameters:
        return output_identity.file_record(path, completed=proof)
    return output_identity.file_record(path)


def test_completed_record_reuse_reads_payload_when_change_tokens_can_collide(tmp_path, monkeypatch):
    path = tmp_path / "frame"
    payload = bytes(range(256)) * 4096
    path.write_bytes(payload)
    proof = _completed(path)
    read_bytes = 0
    opening = Path.open

    class Reading:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def fileno(self):
            return self.stream.fileno()
        def read(self, size=-1):
            nonlocal read_bytes
            value = self.stream.read(size)
            read_bytes += len(value)
            return value

    def open_file(candidate, mode="r", *args, **kwargs):
        stream = opening(candidate, mode, *args, **kwargs)
        return Reading(stream) if candidate == path and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", open_file)
    assert _record(path, proof) == {
        "path": str(path.resolve()), "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest()}
    assert read_bytes == (len(payload) if os.name == "nt" else 0)


@pytest.mark.skipif(os.name != "nt", reason="Windows ChangeTime collision control")
def test_windows_change_time_collision_cannot_reuse_a_completed_digest(tmp_path, monkeypatch):
    from dataclasses import replace

    path = tmp_path / "frame"
    path.write_bytes(b"original")
    proof = _completed(path)
    before = path.stat()
    revision = output_identity._revision

    # Reproduce the observed equal ChangeTime after a same-size edit with
    # restored LastWriteTime, independent of scheduling within a clock tick.
    def coarse_revision(handle):
        current = revision(handle)
        return replace(current, changed_ns=proof.revision.changed_ns,
                       stat_ctime_ns=proof.revision.stat_ctime_ns)

    monkeypatch.setattr(output_identity, "_revision", coarse_revision)
    with path.open("r+b") as stream:
        stream.write(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with path.open("rb") as stream:
        assert output_identity._revision(stream) == proof.revision
    with pytest.raises(RuntimeError, match="changed"):
        _record(path, proof)


def test_same_size_replacement_cannot_inherit_writer_proof(tmp_path):
    path = tmp_path / "frame"
    path.write_bytes(b"original")
    proof = _completed(path)
    before = path.stat()
    replacement = tmp_path / "next"
    replacement.write_bytes(b"replaced")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="changed"):
        _record(path, proof)


def test_in_place_change_with_restored_mtime_is_detected(tmp_path):
    path = tmp_path / "frame"
    path.write_bytes(b"original")
    proof = _completed(path)
    before = path.stat()
    with path.open("r+b") as stream:
        stream.write(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match="changed"):
        _record(path, proof)


def test_attribute_change_rechecks_the_original_digest(tmp_path):
    path = tmp_path / "frame"
    path.write_bytes(b"unchanged")
    proof = _completed(path)
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10000000))
    assert _record(path, proof)["sha256"] == hashlib.sha256(b"unchanged").hexdigest()


def test_legacy_record_still_hashes_the_actual_file(tmp_path):
    path = tmp_path / "frame"
    path.write_bytes(b"legacy")
    assert output_identity.file_record(path) == {
        "path": str(path.resolve()), "bytes": 6,
        "sha256": hashlib.sha256(b"legacy").hexdigest()}


def test_writer_publication_revision_cannot_be_replaced_before_hash(tmp_path):
    path = tmp_path / "frame"
    path.write_bytes(b"original")
    published = output_identity.publication_revision(path)
    replacement = tmp_path / "next"
    replacement.write_bytes(b"replaced")
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="changed"):
        output_identity.completed_file_record(path, published=published)


def test_repointed_directory_link_cannot_turn_a_fresh_record_into_fallback(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    (first / "frame").write_bytes(b"original")
    (second / "frame").write_bytes(b"replaced")
    address = tmp_path / "address"
    try:
        address.symlink_to(first, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    proof = output_identity.completed_file_record(address / "frame")
    address.unlink()
    address.symlink_to(second, target_is_directory=True)
    with pytest.raises(RuntimeError, match="changed"):
        output_identity.file_records([address / "frame"], completed=[proof])


def test_unavailable_reuse_token_falls_back_to_exact_original_digest(tmp_path, monkeypatch):
    from dataclasses import replace

    path = tmp_path / "frame"
    path.write_bytes(b"original")
    proof = output_identity.completed_file_record(path)
    revision = output_identity._revision
    monkeypatch.setattr(output_identity, "_revision",
                        lambda handle: replace(revision(handle), reusable=False))
    hashed = []
    hasher = output_identity._hash_open_file

    def observe(*args, **kwargs):
        result = hasher(*args, **kwargs)
        hashed.append(result.size)
        return result

    monkeypatch.setattr(output_identity, "_hash_open_file", observe)
    assert output_identity.file_record(path, completed=proof) == proof.record()
    assert hashed == [8]
