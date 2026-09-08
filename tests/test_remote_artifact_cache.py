"""Protocol-only raw byte fixtures; real OS locks, no weather decoding."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import pytest

from gpuwm.remote_artifact_cache import Cache, Lease


def frame(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def obtain(root, raw, *, target=12, maximum=8):
    with Cache(root, target_bytes=target, max_frame_bytes=maximum) as cache:
        return cache.obtain(frame(raw), lambda path: path.write_bytes(raw))


@contextmanager
def reader(root, raw):
    path = root / "leases" / (frame(raw)["sha256"] + ".lock")
    with path.open("r+b", buffering=0) as file:
        if os.name == "nt":
            import msvcrt
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBRLCK, 1)
        else:
            import fcntl
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
        try:
            yield file
        finally:
            if os.name == "nt":
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def test_long_run_rotates_unleased_old_objects_without_a_cache_limit_stop(tmp_path):
    root = tmp_path / "cache"
    previous = []
    for index in range(40):
        raw = bytes([index]) * 8
        path, copied, status = obtain(root, raw)
        previous.append(path)
        assert copied == 8 and path.read_bytes() == raw
        assert status["used_bytes"] <= status["target_bytes"] == 12
    assert sum(path.exists() for path in previous) == 1
    assert len(list((root / "leases").iterdir())) == 40  # Permanent lock inodes.


def test_active_reader_survives_replacement_and_release_allows_automatic_eviction(tmp_path):
    root = tmp_path / "cache"
    old = b"oldbytes"
    old_path, _, _ = obtain(root, old)
    lock_path = root / "leases" / (frame(old)["sha256"] + ".lock")
    lock_inode = lock_path.stat().st_ino
    with reader(root, old):
        new_path, _, status = obtain(root, b"newbytes")
        assert old_path.read_bytes() == old and new_path.read_bytes() == b"newbytes"
        assert status["used_bytes"] == 16
        assert status["used_bytes"] <= status["target_bytes"] + status["replacement_headroom_bytes"]
    new_path, copied, status = obtain(root, b"newbytes")
    assert copied == 0 and not old_path.exists() and new_path.is_file()
    assert status["used_bytes"] == 8 and status["evicted_bytes"] == 8
    assert lock_path.stat().st_ino == lock_inode


def test_cache_hit_does_not_change_raw_object_metadata_bound_by_native_reader(tmp_path):
    root = tmp_path / "cache"
    path, _, _ = obtain(root, b"same")
    before = path.stat()
    with reader(root, b"same"):
        again, copied, _ = obtain(root, b"same")
    after = again.stat()
    assert copied == 0 and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)


def test_corrupt_unleased_object_is_downloaded_again_under_its_exact_sha(tmp_path):
    root = tmp_path / "cache"
    path, _, _ = obtain(root, b"original")
    path.write_bytes(b"badbytes")
    repaired, copied, _ = obtain(root, b"original")
    assert repaired == path and copied == 8 and path.read_bytes() == b"original"


def test_corrupt_leased_object_is_never_replaced_under_a_native_reader(tmp_path):
    root = tmp_path / "cache"
    path, _, _ = obtain(root, b"original")
    path.write_bytes(b"badbytes")
    # A reader may create the permanent lock inode while reattaching an old
    # receipt. Initializing its marker must not write under a shared lock.
    (root / "leases" / (frame(b"original")["sha256"] + ".lock")).write_bytes(b"")
    with reader(root, b"original"):
        with pytest.raises(ValueError, match="failed integrity while retained by a reader"):
            obtain(root, b"original")
        assert path.read_bytes() == b"badbytes"
    assert obtain(root, b"original")[1] == 8
    assert path.read_bytes() == b"original"


def test_only_recognized_abandoned_transfer_parts_are_removed(tmp_path):
    root = tmp_path / "cache"
    path, _, _ = obtain(root, b"valid")
    temporary = path.with_name("." + path.name + "." + "a" * 16 + ".part")
    temporary.write_bytes(b"incomplete")
    unknown = path.parent / "user-note.txt"
    unknown.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="unexpected"):
        obtain(root, b"valid")
    assert not temporary.exists() and unknown.read_bytes() == b"preserve"
    with Lease(root / ".writer.lock") as lock:
        assert lock.file is not None


def test_failed_download_never_publishes_wrong_bytes(tmp_path):
    root = tmp_path / "cache"
    with Cache(root) as cache:
        with pytest.raises(ValueError, match="immutable cache identity"):
            cache.obtain(frame(b"expected"), lambda path: path.write_bytes(b"wrong"))
    assert not list((root / "objects").iterdir())


def test_replacement_headroom_is_bounded_even_when_all_existing_readers_are_active(tmp_path):
    root = tmp_path / "cache"
    first, _, _ = obtain(root, b"first123")
    with reader(root, b"first123"):
        second, _, _ = obtain(root, b"second12")
        with reader(root, b"second12"):
            with pytest.raises(ValueError, match="bounded cache replacement allowance"):
                obtain(root, b"third123")
            assert first.is_file() and second.is_file()
    assert obtain(root, b"third123")[2]["used_bytes"] <= 12


def test_old_client_without_reader_lease_opt_in_never_prunes_or_repairs_active_objects(tmp_path):
    root = tmp_path / "cache"
    old, _, _ = obtain(root, b"original")
    with Cache(root, target_bytes=12, max_frame_bytes=8, reader_leases=False) as cache:
        with pytest.raises(ValueError, match="Update the visual workspace"):
            cache.obtain(frame(b"newbytes"), lambda path: path.write_bytes(b"newbytes"))
    assert old.read_bytes() == b"original"
    old.write_bytes(b"badbytes")
    with Cache(root, reader_leases=False) as cache:
        with pytest.raises(ValueError, match="safe repair"):
            cache.obtain(frame(b"original"), lambda path: path.write_bytes(b"original"))
    assert old.read_bytes() == b"badbytes"
