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
def test_every_hashed_blob_is_lf_only():
    """The committed blobs themselves carry no CR, absolutely.

    The disk-vs-blob comparison above is necessary but blind to one
    arrangement: a file committed WITH CRLF checks out as CRLF and the
    two sides agree byte for byte.  That is exactly what a Windows
    worktree with ``core.autocrlf=true`` produces when a lane edits a
    hashed file -- the flip enters the object database and every hash
    the product takes of it disagrees with a Linux clone's, while the
    gate above stays green on the machine that caused it.  So the blob
    is asked directly: zero CR bytes, no allowlist, no tolerance.
    """

    crlf = []
    for relative in _hashed_paths():
        try:
            committed = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{relative}"],
                cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue  # not committed yet on this branch
        if b"\r" in committed:
            crlf.append(relative)
    assert not crlf, (
        "these committed blobs contain CR bytes; a CRLF blob hashes "
        "differently on every platform that checks out LF, so the "
        "runtime source identity splits by clone: "
        f"{crlf}")


# ---------------------------------------------------------------------------
# The hashed set is not the whole promise
# ---------------------------------------------------------------------------
#
# The two gates above cover `_hashed_paths()` -- about twenty files -- and
# `* -text` promises the whole repository.  The gap between those two is
# not hypothetical: between v2.3.3 and v2.4.1 the object database gained
# CRLF in thirty-four files, sixteen of them existing files flipped whole
# (`gpuwm/cli.py`, `gpuwm/doctor.py`, `gpuwm/bridge_assets.py`,
# `pyproject.toml`, `CHANGELOG.md`, eight test modules and more) and
# eighteen of them newly authored that way.  Every one of the three tests
# above stayed green, because not one of those files is hashed.
#
# The damage was not a hash split, it was the merge: folding v2.4.1 onto
# the 2.5.0 line produced line-for-line conflicts across whole files --
# `gpuwm/doctor.py` alone is 3,859 lines -- with the real semantic
# conflicts invisible underneath them.  So the gate widens to the surface
# this repository AUTHORS, and the recorded debt below is what it found
# there when it was written.

#: Trees whose bytes are deliberately preserved as they were received, so
#: their line endings are not this repository's to normalize.
#: `gpuwm/data/**`, `tools/*/vendor/**` and `patches/**` are the same
#: three `.gitattributes` names them individually for.
_BYTE_PRESERVED_PREFIXES = (
    "gpuwm/data/",
    "tools/rustwx/vendor/",
    "tools/grib1_bridge/vendor/",
    "patches/",
)

#: The rustwx crates this repository authors, identified the same way
#: `tests/test_bridge_source_rev_stamp.py` identifies them: a crate this
#: repository stamps with its own source revision is a crate it wrote.
#: Everything else under `tools/rustwx/crates/` is the verbatim subset of
#: the upstream rusty-weather workspace that `tools/rustwx/Cargo.toml`
#: documents, vendored at an exact working-tree state.
_AUTHORED_RUSTWX_CRATES = (
    "rw-fetch", "rw-wrfbatch", "rw-nexrad", "rw-odim", "rw-obs",
    "rw-goes", "rw-netcdf", "rw-mpas",
)

#: The 62 files that already carried CR when this gate was widened,
#: recorded so the gate can be a ratchet instead of a 62-file cleanup
#: nobody asked for in a merge, across an authored surface of 1,827
#: files.  This list may only shrink.  Nothing here is blessed:
#: it is debt with a name, and `test_the_crlf_debt_does_not_rot` fails
#: the moment an entry is fixed or deleted without being struck off.
_CRLF_DEBT = frozenset({
    # the repository root -- CRLF since before v2.3.3, and the only root
    # file that is: CHANGELOG.md and pyproject.toml were flipped by the
    # 2.4 line and are LF again.
    ".gitignore",
    # gpuwm/ -- the wheel's own sources
    "gpuwm/core/kernels/shinhong.cu",
    "gpuwm/core/landuse.py",
    "gpuwm/core/rrtm_taumol.py",
    "gpuwm/da/letkf.py",
    "gpuwm/da/obs_goes.py",
    "gpuwm/da/obs_radar.py",
    "gpuwm/ingest/nest_init.py",
    "gpuwm/obs/goes_cwp.py",
    "gpuwm/obs/goes_grid.py",
    "gpuwm/obs/goes_pack.py",
    "gpuwm/state_digest.py",
    "gpuwm/verify/cases/nest_spawn_ideal.py",
    "gpuwm/verify/feedback_reference.py",
    "gpuwm/verify/obs/battery.py",
    "gpuwm/verify/obs/fss.py",
    "gpuwm/verify/obs/promotion.py",
    "gpuwm/verify/obs/registration.py",
    "gpuwm/verify/obs/stubs.py",
    # tests/
    "tests/data/wrfout_global_attribute_set_v1.json",
    "tests/goes_pack_fixtures.py",
    "tests/test_certification_capsule.py",
    "tests/test_da_nowcast.py",
    "tests/test_goes_cwp.py",
    "tests/test_goes_pack_crosslane.py",
    "tests/test_grayzone_nest_config.py",
    "tests/test_inflow_fetch_meter.py",
    "tests/test_les_tornado_attempt1_config.py",
    "tests/test_nest_init.py",
    "tests/test_obs_battery.py",
    "tests/test_obs_battery_ingest.py",
    "tests/test_obs_controls.py",
    "tests/test_obs_model_source.py",
    "tests/test_obs_promotion.py",
    "tests/test_render_basemap_delivery.py",
    "tests/test_shipped_configs_mixing_stability.py",
    "tests/test_spawn_runner.py",
    "tests/test_tke_km2.py",
    # tools/
    "tools/da_nowcast.py",
    "tools/emit_les_route_inputs.py",
    "tools/inflow_fetch_meter.py",
    "tools/obs_battery_score.py",
    "tools/obs_fetch_asos.py",
    "tools/obs_fetch_mrms.py",
    "tools/obs_fetch_stage4.py",
    "tools/obs_goes_grid_build.py",
    "tools/obs_radar_grid_build.py",
    "tools/obs_radar_grid_from_pack.py",
    "tools/rustwx/crates/rw-fetch/Cargo.toml",
    "tools/rustwx/crates/rw-fetch/src/main.rs",
    "tools/rustwx/crates/rw-wrfbatch/src/grib_import.rs",
    "tools/rustwx/crates/rw-wrfbatch/src/wrf_volumes.rs",
    # docs/ -- receipts are transcripts and sit outside the surface
    "docs/les/ATTEMPT1-EXPECTATIONS.md",
    "docs/les/attempt1/eta49-shipped-control.json",
    "docs/les/attempt1/eta72-ladder.json",
    "docs/les/attempt2/eta72-raisedlid.json",
    "docs/obs-goes-cwp-assimilation.md",
    "docs/obs-goes-cwp-bridge-design.md",
    "docs/public/LES.md",
    # configs/
    "configs/frozen/les_nest_250m_km3.toml",
    "configs/les_nest_250m_km3.toml",
    "configs/moving_nest_20110427_move_proof.toml",
})


def _is_authored(relative: str) -> bool:
    """Is this path one this repository wrote, rather than received?"""

    if any(relative.startswith(p) for p in _BYTE_PRESERVED_PREFIXES):
        return False
    if relative.startswith("docs/"):
        # Receipts are captured transcripts of other tools' output.
        return "/receipts/" not in relative
    if relative.startswith("tools/rustwx/crates/"):
        crate = relative.split("/")[3]
        return crate in _AUTHORED_RUSTWX_CRATES
    if relative.startswith("tools/"):
        return relative.endswith(".py")
    if relative.startswith(("gpuwm/", "tests/", "configs/")):
        return True
    return "/" not in relative  # the release mechanics at the repo root


#: Handed to git as pathspecs so it never opens the vendored trees at
#: all.  `_is_authored` is still the authority -- this only keeps the
#: scan off the ~2.5 GB of vendored crates.
_SCAN_EXCLUDES = tuple(f":!{prefix}" for prefix in _BYTE_PRESERVED_PREFIXES)


def _authored_paths() -> set[str]:
    """Every authored path committed at HEAD."""

    listing = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "HEAD"],
        cwd=ROOT, capture_output=True, check=True).stdout
    found = set()
    for record in listing.split(b"\x00"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        _mode, kind, _sha = meta.split(b" ")
        if kind != b"blob":
            continue
        relative = raw_path.decode("utf-8", "surrogateescape")
        if _is_authored(relative):
            found.add(relative)
    return found


def _cr_bearing_authored() -> set[str]:
    """Authored paths whose COMMITTED blob contains a CR byte.

    ``git grep`` rather than reading every blob here: it is one process
    against the object database instead of a thousand, it already skips
    what it considers binary (``-I``), and the alternative -- feeding
    ``git cat-file --batch`` a thousand object ids down a pipe while
    nothing drains its output -- deadlocks on a 64 KB pipe buffer, which
    is a slow way to learn that a gate is a test too.
    """

    found = subprocess.run(
        ["git", "grep", "-l", "-z", "-I", "-F", "-e", "\r",
         "HEAD", "--", *_SCAN_EXCLUDES],
        cwd=ROOT, capture_output=True)
    if found.returncode not in (0, 1):  # 1 is git grep for "no matches"
        raise AssertionError(
            f"git grep failed: {found.stderr.decode('utf-8', 'replace')}")
    reported = found.stdout.decode("utf-8", "surrogateescape")
    # Each record is `HEAD:<path>`; the revision is fixed, so split once.
    paths = {record.split(":", 1)[1]
             for record in reported.split("\x00") if record}
    return {relative for relative in paths if _is_authored(relative)}


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_no_authored_file_gains_a_carriage_return():
    """The `* -text` promise, held across everything this repo writes.

    A per-path allowlist covering only the hashed files is what let
    thirty-four files through between two releases.  This asks the whole
    authored surface and names the difference against the recorded debt,
    so the next flip is one file in a failure message rather than twenty
    thousand lines in a merge.
    """

    fresh = sorted(_cr_bearing_authored() - _CRLF_DEBT)
    assert not fresh, (
        "these committed blobs gained CR bytes and are not recorded debt. "
        "A Windows worktree with core.autocrlf=true does this on its own, "
        "and the flip is invisible until a merge or a hash disagrees. "
        "Normalize them to LF -- the change must be CR-only -- rather "
        f"than adding them below: {fresh}")


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_the_crlf_debt_does_not_rot():
    """The debt list may only shrink, and only honestly.

    An entry that has been fixed or deleted has to be struck off in the
    same commit, otherwise the list slowly becomes a description of a
    repository that no longer exists and stops being evidence of
    anything.
    """

    present = _authored_paths()
    carrying = _cr_bearing_authored()
    settled = sorted(entry for entry in _CRLF_DEBT
                     if entry in present and entry not in carrying)
    departed = sorted(entry for entry in _CRLF_DEBT if entry not in present)
    assert not settled, (
        "these files are LF now, so strike them off _CRLF_DEBT in the "
        f"commit that fixed them: {settled}")
    assert not departed, (
        "these _CRLF_DEBT entries are no longer authored files in this "
        f"tree; remove them: {departed}")


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
