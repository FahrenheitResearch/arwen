"""A prepared tree crossing platforms is the same tree.

Origin: a tree prepared on Windows could not be run on Linux.  The
preparation records the paths it actually opened, and on Windows those
strings carry backslashes.  Every receipt check that read one back through
``pathlib.Path`` got the RUNNING platform's separator rules: on POSIX a
Windows spelling has no separators at all, so ``.name`` returns the whole
string and every basename comparison fails.  The refusal that came out
said the receipt "differs from supplied file", which blames the bytes --
and the bytes were identical, digest and all.

These tests are platform-independent on purpose: each spelling is asserted
against both readings, so a green run on Windows is a green run on Linux.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gpuwm.receipt_paths import receipt_basename
import tools.prepared_single_domain_forecast as runner


#: The user segment every recorded path below is built from.  A recorded
#: receipt path is a per-user absolute path by construction -- that is
#: this module's whole subject -- so the fixtures are ASSEMBLED from this
#: fragment rather than written out, and the assembled strings are the
#: real spellings the functions under test receive.  The reason is the
#: release snapshot scan: it reads this file, a literal home directory
#: here ships one developer's machine path to every user, and
#: tests/test_release_snapshot_machine_paths.py fails the build days
#: before the cut.  ``tests/test_report_bundle.py`` builds its redaction
#: fixture the same way, and the scan's own tests pin that a substituted
#: user segment is the REMEDY rather than the defect.  This one names
#: nobody.
_USER = "somebody"
_WINDOWS_HOME = rf"C:\Users\{_USER}"
_POSIX_HOME = f"/home/{_USER}"

WINDOWS_SPELLING = rf"{_WINDOWS_HOME}\prep\case\experiment.toml"
POSIX_SPELLING = f"{_POSIX_HOME}/prep/case/experiment.toml"


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _receipt(path_value: str, target: Path) -> dict[str, object]:
    return {
        "path": path_value,
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


class TestReceiptBasename:
    """Both spellings read the same on both platforms."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (WINDOWS_SPELLING, "experiment.toml"),
            (POSIX_SPELLING, "experiment.toml"),
            (r"prep\case\namelist.wps", "namelist.wps"),
            ("prep/case/namelist.wps", "namelist.wps"),
            # Mixed spellings happen when a POSIX-rooted string is joined
            # with a Windows-recorded tail, or the reverse.
            (r"C:\prep/case\namelist.wps", "namelist.wps"),
            ("/prep/case\\namelist.wps", "namelist.wps"),
            ("experiment.toml", "experiment.toml"),
            # Drive-relative, which Windows itself resolves to a bare name.
            ("C:experiment.toml", "experiment.toml"),
            # A directory spelling keeps the directory's own name.
            (rf"{_WINDOWS_HOME}\prep\prepared-cache", "prepared-cache"),
            (f"{_POSIX_HOME}/prep/prepared-cache/", "prepared-cache"),
            (rf"{_WINDOWS_HOME}\prep\prepared-cache" + "\\",
             "prepared-cache"),
            # Roots have no name, exactly as pathlib reports.
            ("/", ""),
            ("\\", ""),
        ],
    )
    def test_reads_both_separator_spellings(self, value, expected):
        assert receipt_basename(value) == expected

    def test_a_windows_recorded_path_is_not_its_own_basename(self):
        """The check a manifest uses to refuse a directory-bearing name.

        On POSIX ``Path(r"sub\\file.grb").name`` is the whole string, so a
        name check written that way accepts a name carrying a directory
        component -- the traversal it was written to refuse.
        """
        assert receipt_basename(r"sub\file.grb") != r"sub\file.grb"
        assert receipt_basename("sub/file.grb") != "sub/file.grb"


class TestExecutionFileReceiptCrossPlatform:
    """Prepare on the desktop, run on the node."""

    def test_windows_recorded_path_validates_against_identical_bytes(
            self, tmp_path):
        """The exact blocked workflow: prep on Windows, run on Linux."""
        supplied = _write(tmp_path / "experiment.toml", b"[case]\nnx = 64\n")
        runner._validate_execution_file_receipt(
            _receipt(WINDOWS_SPELLING, supplied),
            supplied,
            "experiment config",
        )

    def test_posix_recorded_path_validates_against_identical_bytes(
            self, tmp_path):
        """And the reverse: prep on Linux, run on Windows."""
        supplied = _write(tmp_path / "experiment.toml", b"[case]\nnx = 64\n")
        runner._validate_execution_file_receipt(
            _receipt(POSIX_SPELLING, supplied),
            supplied,
            "experiment config",
        )

    def test_a_renamed_copy_of_the_same_bytes_validates(self, tmp_path):
        """Identity is the digest; the recorded path is provenance.

        This is the general form of the cross-platform case.  On POSIX the
        Windows spelling read back as a basename IS a different name, so
        the fix is not a separator patch on the comparison -- it is that a
        name difference over identical bytes was never a content
        difference.
        """
        supplied = _write(tmp_path / "run.toml", b"[case]\nnx = 64\n")
        runner._validate_execution_file_receipt(
            _receipt(WINDOWS_SPELLING, supplied),
            supplied,
            "experiment config",
        )

    def test_different_bytes_are_still_refused_and_the_refusal_names_them(
            self, tmp_path):
        supplied = _write(tmp_path / "experiment.toml", b"[case]\nnx = 64\n")
        receipt = _receipt(POSIX_SPELLING, supplied)
        supplied.write_bytes(b"[case]\nnx = 128\n")
        with pytest.raises(ValueError) as excinfo:
            runner._validate_execution_file_receipt(
                receipt, supplied, "experiment config")
        message = str(excinfo.value)
        assert "sha256" in message
        assert hashlib.sha256(supplied.read_bytes()).hexdigest() in message
        assert receipt["sha256"] in message
        # The remedy, named.
        assert "re-prepare" in message
        # And it must not blame the path spelling, which is identical here.
        assert "differs from supplied file" not in message

    def test_a_truncated_file_names_the_size_difference(self, tmp_path):
        supplied = _write(tmp_path / "namelist.wps", b"&share\n/\n")
        receipt = _receipt(POSIX_SPELLING, supplied)
        supplied.write_bytes(b"&share\n")
        with pytest.raises(ValueError) as excinfo:
            runner._validate_execution_file_receipt(
                receipt, supplied, "WPS namelist")
        message = str(excinfo.value)
        assert "WPS namelist" in message
        assert str(receipt["bytes"]) in message
        assert str(supplied.stat().st_size) in message

    def test_a_receipt_without_a_path_is_refused_for_the_missing_provenance(
            self, tmp_path):
        supplied = _write(tmp_path / "experiment.toml", b"[case]\n")
        receipt = _receipt("", supplied)
        with pytest.raises(ValueError) as excinfo:
            runner._validate_execution_file_receipt(
                receipt, supplied, "experiment config")
        assert "path" in str(excinfo.value)

    def test_a_malformed_receipt_is_still_refused(self, tmp_path):
        supplied = _write(tmp_path / "experiment.toml", b"[case]\n")
        with pytest.raises(ValueError, match="malformed"):
            runner._validate_execution_file_receipt(
                {"path": POSIX_SPELLING}, supplied, "experiment config")


_DIGEST = "0" * 64


class TestSiblingReceiptPathReads:
    """The same recorded-path read, at the other places a tree carries one.

    Each of these read a recorded path through ``pathlib.Path`` too, so
    each refused (or, once, wrongly ACCEPTED) a manifest for the platform
    that wrote it rather than for anything about its bytes.
    """

    @pytest.mark.parametrize(
        "recorded",
        [
            rf"{_WINDOWS_HOME}\stage\icon-eu\pressure-level.grib2",
            f"{_POSIX_HOME}/stage/icon-eu/pressure-level.grib2",
        ],
    )
    def test_composition_manifest_names_the_file_not_the_whole_path(
            self, recorded):
        manifest = {
            "schema": runner._SOURCE_SCHEMA["icon-eu"],
            "mapping_sha256": _DIGEST,
            "composition_sha256": _DIGEST,
            "primary_files": [
                {"path": recorded, "bytes": 4096, "sha256": _DIGEST}],
            "supplements": {
                "terrain": [
                    {"path": recorded, "bytes": 4096, "sha256": _DIGEST}]},
            "provenance": {
                "terrain": {
                    "path": recorded, "bytes": 4096, "sha256": _DIGEST}},
            "decoders": {},
        }
        specs = runner._mapped_composition_manifest_file_specs(
            "icon-eu", manifest)
        assert specs["primary[0]"]["name"] == "pressure-level.grib2"
        # The recorded spelling survives untouched: it is provenance.
        assert specs["primary[0]"]["path"] == recorded

    @pytest.mark.parametrize(
        "unsafe",
        [
            r"sub\experiment.toml",
            "sub/experiment.toml",
            rf"{_WINDOWS_HOME}\prep\experiment.toml",
            f"{_POSIX_HOME}/prep/experiment.toml",
        ],
    )
    def test_a_portable_manifest_role_name_carrying_a_directory_is_refused(
            self, unsafe):
        """A portable manifest stores BARE names, so the tree relocates.

        On POSIX the backslash spellings used to pass this check: no
        separator, so the name looked bare, and a name carrying a
        directory component reached the join that opens it.
        """
        manifest = {
            "schema": runner._SOURCE_SCHEMA["gfs"],
            "source": "gfs",
            "files": {"experiment_config": {
                "name": unsafe, "sha256": _DIGEST}},
        }
        with pytest.raises(ValueError, match="unsafe name"):
            runner._manifest_file_specs("gfs", manifest, None, {})

    @pytest.mark.parametrize(
        "recorded",
        [
            rf"{_WINDOWS_HOME}\runs\case\prepared-cache",
            f"{_POSIX_HOME}/runs/case/prepared-cache",
            rf"{_WINDOWS_HOME}\runs\case\prepared-cache" + "\\",
            f"{_POSIX_HOME}/runs/case/prepared-cache/",
        ],
    )
    def test_the_relocatable_cache_receipt_reads_on_either_platform(
            self, recorded):
        """The tree runner's relocatability check, both dialects.

        A hierarchy prepared on Windows records the cache directory with
        backslashes; on POSIX this read the whole string as the name and
        refused with "cache receipt path is not relocatable" -- about a
        tree that was perfectly relocatable.
        """
        assert receipt_basename(recorded) == "prepared-cache"

    @pytest.mark.parametrize(
        "recorded,filename,accepted",
        [
            (rf"{_WINDOWS_HOME}\stage\20cr\air.2m.1974.nc",
             "air.2m.1974.nc", True),
            (f"{_POSIX_HOME}/stage/20cr/air.2m.1974.nc",
             "air.2m.1974.nc", True),
            (rf"{_WINDOWS_HOME}\stage\20cr\air.2m.1974.nc",
             "other.nc", False),
            (f"{_POSIX_HOME}/stage/20cr/air.2m.1974.nc",
             "other.nc", False),
        ],
    )
    def test_a_member_manifest_row_pairs_path_and_filename_either_way(
            self, recorded, filename, accepted):
        assert (receipt_basename(recorded) == filename) is accepted
