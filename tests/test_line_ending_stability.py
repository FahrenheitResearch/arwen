"""A checkout's bytes are the committed bytes, for everything hashed.

git-for-Windows defaults to ``core.autocrlf=true``.  Without a
``.gitattributes`` saying otherwise, a Windows clone rewrites LF to CRLF
on checkout, so a file in a clone and the same file in a wheel are
different bytes.  That is invisible until something hashes them -- and
this product hashes a lot of them:

* ``_runtime_source_identity`` in both prepared-forecast runners binds
  the SHA-256 of the model sources it is about to run;
* the Thompson physics receipt binds its implementation files;
* the preparation proof binds an export-source set, and the forecast
  stage re-derives it and compares.

A mixed estate -- ``rw-wps`` from a pip venv, the runner from a fresh
clone -- therefore failed with ``proof export source hashes differ from
preparation``.  Nothing was corrupt; the two halves had different line
endings.

The remedy is ``* -text`` in ``.gitattributes``.  The gate is this file:
it asks git for the committed blob and compares it with what is on
disk.  On a Linux runner it is nearly tautological; on a Windows runner
it is the actual check, and Windows is where the failure lives.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)


def _is_checkout() -> bool:
    try:
        _git("rev-parse", "--show-toplevel")
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _hashed_paths() -> list[str]:
    """Every repository file the product hashes and then compares.

    Read from the runners themselves rather than re-listed here: a file
    added to one of those tuples has to be covered by this gate without
    anyone remembering to add it twice.
    """

    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm import prepared_single_domain_forecast as single

    paths = {
        "gpuwm/core/model.py",
        "gpuwm/ingest/hrrr_physics.py",
        "gpuwm/ingest/hrrr_surface.py",
        "gpuwm/ingest/lateral_bc.py",
        "gpuwm/ingest/prepared_cache.py",
        "gpuwm/ingest/real.py",
        "gpuwm/io/wrfout.py",
        "gpuwm/native_wrf_contract.py",
        "gpuwm/source_authorities.py",
        "gpuwm/prepared_single_domain_forecast.py",
        "gpuwm/prepared_domain_tree_forecast.py",
        "gpuwm/core/nest.py",
        "gpuwm/core/microphysics_transition.py",
        "gpuwm/core/kernels/nest_microphysics.cu",
        "gpuwm/physics_registry_v2.json",
        "gpuwm/wrf_direct_v461_contract.json",
    }
    paths.update(single._THOMPSON_IMPLEMENTATION_FILES)
    # Named so a reader can see the two runners are the authority here.
    assert tree.RUNNER == "tools.prepared_domain_tree_forecast"
    return sorted(paths)


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_every_hashed_file_matches_its_committed_blob():
    """Disk bytes == object-database bytes, file by file."""

    drifted = []
    for relative in _hashed_paths():
        path = ROOT / relative
        if not path.is_file():
            continue  # an externalized asset; its own gate covers it
        try:
            committed = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{relative}"],
                cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue  # not committed yet on this branch
        if committed != path.read_bytes():
            drifted.append(relative)
    assert not drifted, (
        "these files differ from their committed blobs, which on Windows "
        "means core.autocrlf rewrote them on checkout and every hash the "
        "product takes of them now disagrees with the wheel's: "
        f"{drifted}")


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_the_repository_declares_no_conversion_at_all():
    """``* -text`` is present, so a new file is covered by default.

    A per-path allowlist is the arrangement that produced the bug: it
    covered the JSON and the shell scripts and not ``gpuwm/*.py``, which
    is what the runners hash.  The catch-all is the fix; this holds it.
    """

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    lines = [line.strip() for line in attributes.splitlines()]
    assert "* -text" in lines

    # And git agrees, for a file nothing names individually.
    reported = _git("check-attr", "text", "--", "gpuwm/core/model.py")
    assert reported.strip().endswith(": text: unset"), reported
