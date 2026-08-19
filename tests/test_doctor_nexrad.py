"""``gpuwm doctor``'s verdict on the radar front door (``rw_nexrad``).

This line exists because the report used to pass without it.  ``rw_nexrad``
sat outside both audited sets -- it is not a GRIB bridge and not a render
engine -- so ``gpuwm doctor`` printed a clean estate on installs where every
radar route was dead, and the first news a user got came from the launcher
refusing to print a plan.  A doctor that passes on a box that cannot do the
thing is worse than no doctor, which is what these tests pin.

Two failures are distinguished on purpose, because their remedies differ:

* **absent** -- nothing to point at; get one (bundle or build);
* **stale** -- a binary that exists, launches, and reports the wrong
  ``--abi`` contract.  Both the current and the superseded binary answer
  ``--version`` with the same ``rw_nexrad 0.1.0``, so version is not a
  discriminator and "point it at the other copy you have" is the one piece
  of advice guaranteed to waste a user's time.  The remedy must say
  *rebuild*.

Every branch is forced here rather than left to whatever this box has, and
none of them touches the GPU: this file must stay cheap on a contended card,
which is why it is separate from ``tests/test_doctor.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm import bridge_assets, doctor
from gpuwm.obs import nexrad


def _check(monkeypatch, *, found, probe=None):
    """``_nexrad_front_door_check`` with the resolution ladder forced."""

    if isinstance(found, BaseException):
        def _find():
            raise found
    else:
        def _find():
            return found
    monkeypatch.setattr(nexrad, "find_nexrad_bin", _find)
    if probe is not None:
        monkeypatch.setattr(nexrad, "probe_nexrad_bin", lambda path: probe)
    return doctor._nexrad_front_door_check()


# ---------------------------------------------------------------------------
# The gap this check closes
# ---------------------------------------------------------------------------

def test_the_estate_actually_contains_a_radar_front_door_check():
    """The regression in one line: doctor must ask about ``rw_nexrad``.

    Read off the assembling function rather than by running it.  A real
    ``collect_checks()`` spawns the CUDA eigensolver probe and a subprocess
    per Python package -- minutes on a contended card, for a fact that is
    static.
    """

    assert "_nexrad_front_door_check" in doctor.collect_checks.__code__.co_names, (
        "gpuwm doctor does not check for the radar front door, so it "
        "reports a healthy estate on a box that cannot ingest a single "
        "radar observation")


def test_an_absent_front_door_is_missing_and_blocks(monkeypatch):
    """Absent is not `info`: there is no second way to read a volume."""

    check = _check(monkeypatch, found=None)
    assert check.status == "missing"
    assert check.blocking, (
        "the fetch backbone falls back to Python and the renderer falls "
        "back to matplotlib; radar ingest falls back to nothing, so this "
        "one has to affect the exit code")


def test_a_present_current_front_door_is_verified(monkeypatch):
    check = _check(monkeypatch, found=Path("rw_nexrad.exe"),
                   probe=(True, "rw_nexrad 0.1.0 -- --abi matches"))
    assert check.status == "verified"
    assert check.remedy is None and check.action is None


# ---------------------------------------------------------------------------
# It says what is missing, what that blocks, and how to fix it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("found,probe", [
    (None, None),
    (Path("rw_nexrad.exe"), (False, "--abi does not match")),
    (FileNotFoundError("GPUWM_RW_NEXRAD names a missing file: nope"), None),
])
def test_every_failure_names_what_it_blocks(monkeypatch, found, probe):
    """A user must not have to infer that this is the whole nowcast."""

    check = _check(monkeypatch, found=found, probe=probe)
    assert check.status == "missing"
    assert "radar observation ingest" in check.detail.lower()


@pytest.mark.parametrize("found,probe", [
    (None, None),
    (Path("rw_nexrad.exe"), (False, "--abi does not match")),
    (FileNotFoundError("GPUWM_RW_NEXRAD names a missing file: nope"), None),
])
def test_every_failure_carries_a_remedy_and_one_next_command(
        monkeypatch, found, probe):
    check = _check(monkeypatch, found=found, probe=probe)
    assert check.remedy, "a blocking gap with no remedy is a dead end"
    assert check.action, "the terse report needs THE single next command"


# ---------------------------------------------------------------------------
# Stale: rebuild, do not re-point
# ---------------------------------------------------------------------------

def test_a_stale_front_door_is_told_to_rebuild_not_to_be_re_pointed(
        monkeypatch):
    """The advice that is right for staleness and wrong for absence."""

    check = _check(monkeypatch, found=Path("rw_nexrad.exe"),
                   probe=(False, "--abi does not match the record contract"))
    assert check.status == "missing"
    remedy = check.remedy.lower()
    assert "rebuild" in remedy
    assert "do not re-point" in remedy, (
        "both binaries answer --version identically, so re-pointing at "
        "another copy is the one remedy that cannot work")


def test_the_abi_marker_carries_the_live_half_that_stale_binaries_lack():
    """What "stale" concretely means, pinned to the contract line itself.

    The superseded binary published the archive half alone; a wrapper that
    drives the real-time route needs the live half, and pinning it here is
    what turns "fails at the first live fetch" into "fails the probe".
    """

    assert "nexrad-live-fetch" in nexrad.NEXRAD_ABI_MARKER
    assert nexrad.LIVE_FETCH_SCHEMA in nexrad.NEXRAD_ABI_MARKER


# ---------------------------------------------------------------------------
# The real fix: it ships in the bundle, so no user needs a Rust toolchain
# ---------------------------------------------------------------------------

def test_the_front_door_is_a_bundled_artifact():
    """Otherwise the remedy above is "install Rust", for every user."""

    names = {artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert nexrad.NEXRAD_NAME in names, (
        "rw_nexrad is not in the prebuilt bundle, so gpuwm fetch-bridges "
        "cannot deliver radar ingest and a toolchain-less install is stuck")


def test_the_bundled_front_door_uses_the_resolution_ladders_own_env_var():
    """A staged bundle and a hand-built tree must be found by one code path."""

    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == nexrad.NEXRAD_NAME]
    assert artifact.env_var == nexrad.NEXRAD_ENV
    assert artifact.kind == "executable"


def test_the_bundle_prose_counts_the_artifacts_it_actually_carries():
    """The docstring is the contract a release engineer reads; keep it true."""

    assert len(bridge_assets.BUNDLED_ARTIFACTS) == 21
    assert "twenty-one artifacts" in bridge_assets.__doc__
    for stale in ("eight artifacts", "nine artifacts", "nine files",
                  "ten artifacts", "ten files", "eleven artifacts",
                  "eleven files", "fourteen artifacts", "fourteen files",
                  "sixteen artifacts", "sixteen files",
                  "seventeen artifacts", "seventeen files",
                  "eighteen artifacts", "eighteen files",
                  "nineteen artifacts", "nineteen files",
                  "twenty artifacts", "twenty files"):
        assert stale not in bridge_assets.__doc__


def test_doctor_imports_without_the_scientific_stack():
    """``import gpuwm.doctor`` must not require numpy.

    doctor is the tool for diagnosing a broken or partial install, so
    needing the full scientific stack merely to IMPORT it defeats the
    purpose: the environments it exists for are exactly the ones that
    cannot satisfy that.

    This is a regression pin, not a hypothetical.  1.6 added
    ``from gpuwm.obs import nexrad`` at doctor's module scope.
    ``gpuwm.obs.nexrad`` itself needs only the standard library, but
    reaching it runs ``gpuwm/obs/__init__.py``, which imports the
    gridding stack and therefore numpy.  The release pipeline's bridges
    job imports this module for ``_exec_probe`` alone and installs no
    dependencies, so it died on ``import numpy`` three jobs into a cut,
    after the tag existed.

    Run in a subprocess with numpy made unimportable, because the point
    is what happens at import time in a process that never had it.
    """

    import subprocess
    import sys

    program = (
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'numpy' or name.startswith('numpy.'):\n"
        "            raise ImportError('numpy blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import gpuwm.doctor\n"
        "print('OK')\n"
    )
    done = subprocess.run([sys.executable, "-c", program],
                          capture_output=True, text=True)
    assert done.returncode == 0, (
        "gpuwm.doctor no longer imports without numpy; a module-scope "
        f"import pulled the scientific stack in:\n{done.stderr}")
    assert "OK" in done.stdout


def test_the_radar_check_reports_an_unimportable_obs_stack():
    """The lazy import must not become a quiet skip.

    Making doctor importable without numpy is only half the fix.  This
    check exists because the report used to pass without it, so a
    version that silently drops the radar line when the import fails
    would reintroduce the same silent-green hole through a new door.
    An obs stack that will not import is a broken install and says so.
    """

    import builtins

    from gpuwm import doctor

    real_import = builtins.__import__

    def _refuse(name, *args, **kwargs):
        if name == "gpuwm.obs" or name.startswith("gpuwm.obs."):
            raise ImportError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _refuse
    try:
        check = doctor._nexrad_front_door_check()
    finally:
        builtins.__import__ = real_import

    assert check.status == "missing"
    assert "not importable" in check.detail
    assert "radar" in check.name
