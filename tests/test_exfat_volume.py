"""Forecasts publish their files on an exFAT drive.

Two things exFAT does differently from NTFS and ext4 each failed every
run whose folder was on such a drive:

- It derives a file's id from its directory entry, and the rename that
  publishes a history file moves that entry, so the id the writer validated
  was not the id the published file reported. Every forecast failed at its
  first wrfout (GS-09).
- It has no hard links, which every create-only publisher used, so
  preparation failed at its first manifest with "[WinError 1] Incorrect
  function".

Only a real exFAT volume on Windows reproduces either: name a folder on one
with GPUWM_EXFAT_TEST_DIR. A FAT32 folder is accepted too; FAT32 has no
hard links either, but its file id across a rename was never measured.
"""
import os
from pathlib import Path
import tempfile
from threading import Event

import pytest

_FOLDER_ENV = "GPUWM_EXFAT_TEST_DIR"


def _exfat_folder():
    folder = os.environ.get(_FOLDER_ENV)
    if os.name != "nt" or not folder or not Path(folder).is_dir():
        return None
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    root = ctypes.create_unicode_buffer(260)
    name = ctypes.create_unicode_buffer(64)
    if not kernel.GetVolumePathNameW(str(Path(folder).resolve()), root, 260):
        return None
    if not kernel.GetVolumeInformationW(root, None, 0, None, None, None,
                                        name, 64):
        return None
    return Path(folder) if name.value in ("exFAT", "FAT32", "FAT") else None


pytestmark = pytest.mark.skipif(
    _exfat_folder() is None,
    reason=f"needs Windows and {_FOLDER_ENV} naming a folder on an exFAT "
           "or FAT32 volume")


def test_a_history_file_publishes_on_exfat(monkeypatch):
    from gpuwm import output_identity, runtime
    from test_wrfout import _manual_async_writer, _queue_cpu_ticket

    observed = []
    published = output_identity.PublicationFile.published

    def observe(self, path):
        revision = published(self, path)
        observed.append((self.written.file_id, revision.file_id))
        return revision

    monkeypatch.setattr(output_identity.PublicationFile, "published", observe)
    with tempfile.TemporaryDirectory(dir=_exfat_folder()) as folder:
        path = Path(folder) / "wrfout_d01_2026-09-26_00_00_00"
        writer = _manual_async_writer(Event())
        try:
            _queue_cpu_ticket(writer, path)
        finally:
            writer.close()
        # The condition that failed every exFAT forecast is present...
        assert len(observed) == 1
        assert observed[0][0] != observed[0][1], (
            "this volume kept the file id across the rename, so it does not "
            "reproduce the exFAT condition")
        # ...and the frame lands with a receipt bound to its bytes.
        assert writer.paths == [path]
        assert len(writer.completed_records) == 1
        records = runtime._frame_records(
            writer.paths, completed_records=writer.completed_records)
        assert records == [output_identity.file_record(path)]


def test_a_manifest_publishes_create_only_on_exfat():
    from gpuwm import filesystem_paths, mapped_authoring

    with tempfile.TemporaryDirectory(dir=_exfat_folder()) as folder:
        folder = Path(folder)
        probe = folder / "probe"
        probe.write_bytes(b"x")
        # The condition that failed preparation on this volume is present...
        with pytest.raises(OSError) as refused:
            os.link(probe, folder / "linked")
        assert filesystem_paths._no_hard_links(refused.value), refused.value
        # ...and the manifest publishes, still refusing to replace a file.
        manifest = folder / "inputs.json"
        mapped_authoring._write_new(manifest, b"{}\n")
        assert manifest.read_bytes() == b"{}\n"
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            mapped_authoring._write_new(manifest, b"[]\n")
        assert manifest.read_bytes() == b"{}\n"
        assert sorted(p.name for p in folder.iterdir()) == ["inputs.json", "probe"]
