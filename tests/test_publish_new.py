"""Create-only publication on a drive without hard links (exFAT, FAT32).

A same-directory hard link publishes a complete temporary without ever
replacing a file, and every create-only publisher used one. exFAT and FAT32
have no hard links: Windows answers "Incorrect function" (winerror 1) on
exFAT and Linux EPERM on both, so on such a drive preparation failed at its
first manifest ("mapped contract authoring failed: [WinError 1] Incorrect
function"), and multi-run summaries, ERA5 retrievals, analysis records and
research bundles failed the same way. These tests make os.link answer as such a volume does;
tests/test_exfat_volume.py runs the same publication on a real exFAT volume.
"""
from __future__ import annotations

from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from gpuwm import filesystem_paths
from gpuwm.filesystem_paths import publish_new


def _volume_without_hard_links(monkeypatch):
    calls = []

    def link(source, destination, *args, **kwargs):
        calls.append((Path(source), Path(destination)))
        if os.name == "nt":
            raise OSError(errno.EINVAL, "Incorrect function", str(source), 1,
                          str(destination))
        raise OSError(errno.EPERM, "Operation not permitted", str(source),
                      None, str(destination))

    monkeypatch.setattr(os, "link", link)
    return calls


def test_the_refusal_is_recognised_as_a_volume_without_hard_links(
        tmp_path, monkeypatch):
    _volume_without_hard_links(monkeypatch)
    with pytest.raises(OSError) as refused:
        os.link(tmp_path / "a", tmp_path / "b")
    assert filesystem_paths._no_hard_links(refused.value)


def test_a_complete_file_publishes_where_the_volume_has_no_hard_links(
        tmp_path, monkeypatch):
    calls = _volume_without_hard_links(monkeypatch)
    source, destination = tmp_path / ".manifest.tmp-1", tmp_path / "inputs.json"
    source.write_bytes(b"complete bytes")
    publish_new(source, destination)
    assert calls == [(source, destination)]
    assert destination.read_bytes() == b"complete bytes"
    assert not source.exists()


def test_it_still_never_replaces_an_existing_file(tmp_path, monkeypatch):
    _volume_without_hard_links(monkeypatch)
    source, destination = tmp_path / ".manifest.tmp-1", tmp_path / "inputs.json"
    source.write_bytes(b"ours")
    destination.write_bytes(b"another publisher")
    with pytest.raises(FileExistsError):
        publish_new(source, destination)
    assert destination.read_bytes() == b"another publisher"
    assert source.read_bytes() == b"ours"


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EXDEV, errno.ENOENT])
def test_other_link_failures_are_raised_unchanged(tmp_path, monkeypatch, code):
    failure = OSError(code, "the link itself failed")

    def link(*args, **kwargs):
        raise failure

    monkeypatch.setattr(os, "link", link)
    source, destination = tmp_path / ".manifest.tmp-1", tmp_path / "inputs.json"
    source.write_bytes(b"ours")
    with pytest.raises(OSError) as caught:
        publish_new(source, destination)
    assert caught.value is failure
    assert source.read_bytes() == b"ours" and not destination.exists()


def test_the_probe_finds_hard_links_here_and_leaves_nothing(tmp_path):
    assert filesystem_paths.has_hard_links(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_the_probe_reports_a_drive_without_hard_links(tmp_path, monkeypatch):
    calls = _volume_without_hard_links(monkeypatch)
    assert not filesystem_paths.has_hard_links(tmp_path)
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_the_probe_raises_any_other_link_failure(tmp_path, monkeypatch):
    failure = OSError(errno.ENOSPC, "the link itself failed")

    def link(*args, **kwargs):
        raise failure

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(OSError) as caught:
        filesystem_paths.has_hard_links(tmp_path)
    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []


def test_prep_publishes_its_manifest_without_hard_links(tmp_path, monkeypatch):
    from gpuwm import mapped_authoring

    _volume_without_hard_links(monkeypatch)
    path = tmp_path / "prep" / "inputs.json"
    mapped_authoring._write_new(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        mapped_authoring._write_new(path, b"[]\n")
    assert path.read_bytes() == b"{}\n"
    assert [p.name for p in path.parent.iterdir()] == ["inputs.json"]


def test_a_multi_run_summary_folder_without_hard_links_is_accepted(
        tmp_path, monkeypatch):
    from gpuwm import multi_run

    _volume_without_hard_links(monkeypatch)
    summary = tmp_path / "summary" / "production-summary.json"
    multi_run._prepare_summary_destination(summary)
    assert list(summary.parent.iterdir()) == []
    multi_run._write_summary(summary, {"status": "ours"})
    assert json.loads(summary.read_text(encoding="utf-8")) == {"status": "ours"}
    assert [p.name for p in summary.parent.iterdir()] == [summary.name]


def test_records_and_the_authority_store_publish_without_hard_links(
        tmp_path, monkeypatch):
    from gpuwm import supervisor
    from gpuwm.ensemble import analysis_commit

    _volume_without_hard_links(monkeypatch)
    record = tmp_path / "records" / "intent.json"
    body = analysis_commit.write_record(record, {"kind": "intent"})
    assert analysis_commit.read_record(record) == body
    assert [p.name for p in record.parent.iterdir()] == ["intent.json"]

    source = tmp_path / "forcing.grib2"
    source.write_bytes(b"one exact forcing payload")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store = tmp_path / "cas"
    store.mkdir()
    authority = supervisor._content_authority_path(source, store, digest)
    assert authority == store / digest
    assert authority.read_bytes() == b"one exact forcing payload"
    assert not list(store.glob(".*.tmp"))


def _arco(tmp_path, monkeypatch, *, racing_receipt=False):
    from gpuwm import era5_arco, zarr_bridge

    out = tmp_path / "era5"

    def extract(request, *, request_path, output, progress):
        Path(output).write_bytes(b"validated forcing")
        if racing_receipt:
            (out / era5_arco._RECEIPT).write_text("another retrieval\n",
                                                  encoding="utf-8")
        return {"chunks": 1}

    monkeypatch.setattr(zarr_bridge, "extract_regular_zarr", extract)
    monkeypatch.setattr(era5_arco, "_validate",
                        lambda *args, **kwargs: {"checks": []})
    return era5_arco, out, lambda: era5_arco.retrieve_era5_arco(
        cycle=datetime(2026, 7, 29), hours=24, cadence=12,
        area="30,-100,40,-90", out=out, progress=lambda _: None)


def test_an_era5_retrieval_publishes_without_hard_links(tmp_path, monkeypatch):
    _volume_without_hard_links(monkeypatch)
    era5_arco, out, retrieve = _arco(tmp_path, monkeypatch)
    target = retrieve()
    assert target.read_bytes() == b"validated forcing"
    receipt = json.loads((out / era5_arco._RECEIPT).read_text(encoding="utf-8"))
    assert receipt["artifact"]["bytes"] == len(b"validated forcing")
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [target.name, era5_arco._RECEIPT])


def test_an_era5_retrieval_that_loses_its_receipt_race_withdraws_its_file(
        tmp_path, monkeypatch):
    """The rollback recognised the published file through the stage's copy,
    which a no-replace rename moves; it now uses the published file's stat."""
    _volume_without_hard_links(monkeypatch)
    era5_arco, out, retrieve = _arco(tmp_path, monkeypatch, racing_receipt=True)
    with pytest.raises(FileExistsError):
        retrieve()
    assert (out / era5_arco._RECEIPT).read_text(encoding="utf-8") \
        == "another retrieval\n"
    assert [p.name for p in out.iterdir()] == [era5_arco._RECEIPT]


def test_the_sealed_hrrr_extension_copies_a_reused_file_on_exfat(
        tmp_path, monkeypatch):
    """On exFAT the reused file is copied and proven identical, not linked.
    This refused, citing only that the copy rewrites the preparation's
    bytes every hour."""
    from tools import prepare_hrrr_wrf

    calls = _volume_without_hard_links(monkeypatch)
    source = tmp_path / "prior" / "met_em.d01.nc"
    source.parent.mkdir()
    source.write_bytes(b"prepared")
    destination = tmp_path / "extended" / "met_em.d01.nc"
    prepare_hrrr_wrf._link_file_create(source, destination)
    assert calls == [(source, destination)]
    assert destination.read_bytes() == b"prepared"
    assert not destination.samefile(source)


# A copy standing in for a hard link (link_or_copy_verified): the prepared
# cache's hourly extension and the HRRR preparation's extension reuse the
# previous hour's files this way, and used to refuse such a drive outright,
# citing only the copy's disk cost.

def _file(path: Path, size: int) -> Path:
    path.write_bytes(os.urandom(size))
    return path


def test_a_link_stand_in_links_where_the_drive_can(tmp_path):
    source = _file(tmp_path / "prior.npy", 4096)
    destination = tmp_path / "next.npy"
    assert filesystem_paths.link_or_copy_verified(source, destination) == "link"
    assert destination.samefile(source)


def test_a_link_stand_in_copies_proves_the_bytes_and_says_so_once(
        tmp_path, monkeypatch, capsys):
    from gpuwm import explain

    monkeypatch.setattr(explain, "_PRINTED_ONCE", set())
    calls = _volume_without_hard_links(monkeypatch)
    source = _file(tmp_path / "prior.npy", (3 << 20) + 17)
    first, second = tmp_path / "a.npy", tmp_path / "b.npy"

    assert filesystem_paths.link_or_copy_verified(source, first) == "copy"
    assert filesystem_paths.link_or_copy_verified(source, second) == "copy"

    assert len(calls) == 2
    assert first.read_bytes() == source.read_bytes() == second.read_bytes()
    assert not first.samefile(source) and not second.samefile(source)
    note = capsys.readouterr().err
    assert note.count("copied rather than linked") == 1
    assert "costs its own size in disk space" in note


def test_a_copy_that_cannot_fit_is_refused_before_its_first_byte(
        tmp_path, monkeypatch):
    _volume_without_hard_links(monkeypatch)
    monkeypatch.setattr(filesystem_paths, "_free_bytes", lambda folder: 10)
    source = _file(tmp_path / "prior.npy", 100)
    destination = tmp_path / "next.npy"

    with pytest.raises(filesystem_paths.CopyWouldNotFitError) as refused:
        filesystem_paths.link_or_copy_verified(source, destination)

    assert refused.value.errno == errno.ENOSPC
    message = str(refused.value)
    assert "needs 100 bytes" in message and "10 bytes free" in message
    assert not destination.exists()


def test_a_copy_never_replaces_a_file(tmp_path, monkeypatch):
    _volume_without_hard_links(monkeypatch)
    source = _file(tmp_path / "prior.npy", 64)
    destination = tmp_path / "next.npy"
    destination.write_bytes(b"another writer's file")
    with pytest.raises(FileExistsError):
        filesystem_paths.link_or_copy_verified(source, destination)
    assert destination.read_bytes() == b"another writer's file"


def test_a_copy_that_does_not_match_its_source_is_removed(
        tmp_path, monkeypatch):
    _volume_without_hard_links(monkeypatch)
    source = _file(tmp_path / "prior.npy", 4096)
    destination = tmp_path / "next.npy"
    real_fsync = os.fsync

    def torn(descriptor):
        # The disk kept only part of what was written.
        os.ftruncate(descriptor, 1000)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", torn)
    with pytest.raises(OSError) as refused:
        filesystem_paths.link_or_copy_verified(source, destination)
    assert refused.value.errno == errno.EIO
    assert "does not match" in str(refused.value)
    assert not destination.exists()


def test_a_link_failure_a_copy_would_meet_too_is_not_turned_into_one(
        tmp_path, monkeypatch):
    def denied(source, destination, *args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", str(source),
                              None, str(destination))

    monkeypatch.setattr(os, "link", denied)
    source = _file(tmp_path / "prior.npy", 64)
    destination = tmp_path / "next.npy"
    with pytest.raises(PermissionError):
        filesystem_paths.link_or_copy_verified(source, destination)
    assert not destination.exists()


def test_two_drives_are_a_link_failure_a_copy_answers():
    assert filesystem_paths.links_unavailable(
        OSError(errno.EXDEV, "Invalid cross-device link"))
    assert not filesystem_paths.links_unavailable(
        OSError(errno.ENOSPC, "No space left on device"))
