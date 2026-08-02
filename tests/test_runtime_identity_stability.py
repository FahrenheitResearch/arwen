"""The run identity must name the install, not the process's progress.

``prepared_single_domain_forecast`` snapshots ``_runtime_source_identity()``
when a forecast starts and compares it again at the very END, after the
physics, after the frames are written.  Any part of that identity which
is not a property of the *installation* turns a finished run into

    RuntimeError: forecast runtime implementation changed during run

``wheel_record_identity`` carried one such part: ``record_unhashed_count``
counted every RECORD entry with no digest, and pip's RECORD lists
``__pycache__/*.pyc`` with no digest.  Bytecode materializes lazily, so
the count grows as the process imports modules.

Measured on the shipped 1.4.0 wheel on a fresh `pip install gpuwm[all]`,
one process importing the package tree:

    imported 276 modules
    DIFFERS: installed_wheel
        record_unhashed_count 109 -> 358

and a 474x378x49 forecast died on it at the end of a 5-minute run, twice,
whenever bytecode was not already warm -- which is the new-user path.

Bytecode is DERIVED from the source these digests already bind, so it is
excluded outright rather than counted.
"""

from __future__ import annotations

import pytest

from gpuwm import runtime_manifest


class _Hash:
    def __init__(self, value: str, mode: str = "sha256"):
        self.mode = mode
        self.value = value


class _Item(str):
    """A ``PackagePath``-shaped RECORD entry."""

    def __new__(cls, name, digest=None):
        self = super().__new__(cls, name)
        self.hash = _Hash(digest) if digest else None
        return self


class _Dist:
    metadata = {"Name": "gpuwm"}
    version = "1.4.0"

    def __init__(self, files):
        self.files = files


_SOURCE = [
    _Item("gpuwm/__init__.py", "AAAA"),
    _Item("gpuwm/config.py", "BBBB"),
    _Item("gpuwm-1.4.0.dist-info/RECORD"),          # genuinely unhashed
]
_BYTECODE = [
    _Item("gpuwm/__pycache__/__init__.cpython-312.pyc"),
    _Item("gpuwm/__pycache__/config.cpython-312.pyc"),
    _Item("gpuwm/core/__pycache__/dycore.cpython-312.pyc"),
]


@pytest.mark.parametrize("name,is_bytecode", [
    ("gpuwm/__pycache__/config.cpython-312.pyc", True),
    ("gpuwm/config.pyc", True),
    ("gpuwm/config.pyo", True),
    ("gpuwm\\__pycache__\\config.cpython-312.pyc", True),
    ("gpuwm/config.py", False),
    ("gpuwm/physics_registry_v2.json", False),
    ("gpuwm-1.4.0.dist-info/RECORD", False),
])
def test_bytecode_is_recognized_on_both_path_separators(name, is_bytecode):
    assert runtime_manifest._is_compiled_bytecode(_Item(name)) is is_bytecode


def test_the_identity_does_not_move_when_bytecode_appears(monkeypatch):
    """The regression bar: cold cache and warm cache must agree."""

    monkeypatch.setattr(runtime_manifest, "installed_distribution",
                        lambda package="gpuwm": _Dist(list(_SOURCE)))
    cold = runtime_manifest.wheel_record_identity()

    monkeypatch.setattr(runtime_manifest, "installed_distribution",
                        lambda package="gpuwm": _Dist(_SOURCE + _BYTECODE))
    warm = runtime_manifest.wheel_record_identity()

    assert cold == warm, (
        "an identity that moves as .pyc files are written fails a run at "
        "the end for having run")
    assert cold["record_unhashed_count"] == 1     # the RECORD file only
    assert cold["record_file_count"] == 2         # the two hashed sources


def test_a_changed_source_still_moves_the_identity(monkeypatch):
    """The guard must keep catching what it exists to catch."""

    monkeypatch.setattr(runtime_manifest, "installed_distribution",
                        lambda package="gpuwm": _Dist(list(_SOURCE)))
    before = runtime_manifest.wheel_record_identity()

    swapped = [_Item("gpuwm/__init__.py", "AAAA"),
               _Item("gpuwm/config.py", "CCCC"),      # different digest
               _Item("gpuwm-1.4.0.dist-info/RECORD")]
    monkeypatch.setattr(runtime_manifest, "installed_distribution",
                        lambda package="gpuwm": _Dist(swapped))
    assert runtime_manifest.wheel_record_identity() != before


def test_an_install_of_nothing_but_bytecode_still_refuses(monkeypatch):
    """Excluding bytecode must not turn "no identity" into a silent one."""

    monkeypatch.setattr(runtime_manifest, "installed_distribution",
                        lambda package="gpuwm": _Dist(list(_BYTECODE)))
    with pytest.raises(runtime_manifest.IdentityError,
                       match=r"no hashed\s+RECORD entries"):
        runtime_manifest.wheel_record_identity()


def test_verification_never_re_hashes_bytecode(monkeypatch):
    """``verify=True`` re-reads payload bytes; a .pyc that has not been
    written yet is not a missing install file."""

    monkeypatch.setattr(runtime_manifest, "installed_distribution",
                        lambda package="gpuwm": _Dist(_SOURCE + _BYTECODE))
    located: list[str] = []

    class _D(_Dist):
        def locate_file(self, item):
            located.append(str(item))
            raise AssertionError("verification reached a payload read")

    monkeypatch.setattr(runtime_manifest, "installed_distribution",
                        lambda package="gpuwm": _D(list(_BYTECODE)))
    with pytest.raises(runtime_manifest.IdentityError):
        runtime_manifest.wheel_record_identity(verify=True)
    assert located == []
