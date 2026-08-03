"""Bridge binaries staged for release carry the revision they were built from.

The 1.4.1 cut nearly shipped prepared bridge bundles that predated the
source tip -- the two platform zips were not even the same source
revision.  Every existing release check hashed the bytes it was handed,
so a stale binary pinned and verified perfectly; nothing anywhere asked
"were these bytes built from the commit being released?".

The closure has two halves, and these tests bind both:

* every bundled artifact's build embeds a plain-text stamp,
  ``GPUWM_BRIDGE_SOURCE_REV=<40-hex git commit>``, in the binary
  (the same convention as the GPUWM_* attribute literals the renderer
  already carries, which is how the staleness was noticed at all); and
* the release tool's ``pin`` step -- the cut-path choke point every
  bundle passes through before a wheel is built -- extracts that stamp
  from every binary member and refuses a bundle whose stamp is absent,
  unparseable, ambiguous, or names any commit other than the one being
  released.

The check reads bytes, never executes the artifact: it must hold on a
GPU-less runner and against a binary for the other platform.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from gpuwm import bridge_assets, bridges

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = REPO_ROOT / "tools" / "build_bridge_bundle.py"

MARKER = bridge_assets.SOURCE_REV_MARKER

#: The revision "being released" in these tests, and a stale one.  Both
#: are well-formed 40-hex commits; neither is a hash of anything.
RELEASED = "ab12" * 10
STALE = "cd34" * 10

PLATFORM = "linux-x86_64"
RELEASE = "v0.0.0-stamp"


def _stamp(revision: str) -> bytes:
    return MARKER + revision.encode("ascii")


def _bundle(tmp_path: Path, payloads: dict[str, bytes]) -> Path:
    """A synthetic bundle carrying the eight artifacts and one map asset.

    Only the stamp contents vary per test; everything else satisfies the
    pin step's existing membership and asset requirements so a refusal
    can only come from the staleness check under test.
    """

    archive = tmp_path / f"gpuwm-bridges-{RELEASE}-{PLATFORM}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            name = bridge_assets.artifact_filename(artifact, PLATFORM)
            zf.writestr(name, payloads[artifact.name])
        zf.writestr("assets/basemap/coast.shp", b"not really a shapefile")
    return archive


def _stamped_payloads(revision: str) -> dict[str, bytes]:
    return {artifact.name: b"\x7fELF junk " + _stamp(revision) + b" tail"
            for artifact in bridge_assets.BUNDLED_ARTIFACTS}


def _pin(archive: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", RELEASE,
         "--bundle", str(archive), "--out", str(out), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# The extractor and the verifier, over bytes
# ---------------------------------------------------------------------------

def test_extraction_finds_the_stamp_between_arbitrary_bytes():
    payload = b"\x00\x01" + _stamp(RELEASED) + b"\xff" + _stamp(RELEASED)
    assert bridge_assets.embedded_source_revisions(payload) == (RELEASED,)


def test_extraction_reports_distinct_revisions_separately():
    payload = _stamp(RELEASED) + b"::" + _stamp(STALE)
    assert bridge_assets.embedded_source_revisions(payload) == (
        RELEASED, STALE)


def test_extraction_ignores_a_stamp_that_is_not_forty_hex():
    # "unknown" is what a build outside a git checkout embeds.
    payload = MARKER + b"unknown" + b" " + MARKER + b"ABC123"
    assert bridge_assets.embedded_source_revisions(payload) == ()


def test_verification_accepts_the_released_revision():
    bridge_assets.verify_source_revision(
        b"prefix" + _stamp(RELEASED), expected=RELEASED, label="x")


def test_verification_refuses_an_absent_stamp():
    """The exact shape of the near-miss: a binary with no stamp at all."""

    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            b"no stamp anywhere", expected=RELEASED, label="rw_wrfbatch.exe")
    message = str(refusal.value)
    assert "rw_wrfbatch.exe" in message
    assert "carries no GPUWM_BRIDGE_SOURCE_REV" in message


def test_verification_refuses_an_unparseable_stamp():
    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            MARKER + b"unknown", expected=RELEASED, label="lib.so")
    assert "outside a git checkout" in str(refusal.value)


def test_verification_refuses_a_stale_revision_naming_both():
    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            _stamp(STALE), expected=RELEASED, label="grib1_bridge")
    message = str(refusal.value)
    assert STALE in message and RELEASED in message


def test_verification_refuses_two_distinct_revisions():
    with pytest.raises(bridge_assets.BridgeAssetError) as refusal:
        bridge_assets.verify_source_revision(
            _stamp(RELEASED) + _stamp(STALE), expected=RELEASED, label="x")
    assert "distinct" in str(refusal.value)


def test_verification_refuses_a_malformed_expected_revision():
    """The gate cannot be aimed at a non-commit and quietly pass."""

    with pytest.raises(bridge_assets.BridgeAssetError):
        bridge_assets.verify_source_revision(
            _stamp(RELEASED), expected="HEAD", label="x")


# ---------------------------------------------------------------------------
# The cut path: pin refuses what it cannot prove
# ---------------------------------------------------------------------------

def test_pin_refuses_a_bundle_with_an_unstamped_binary(tmp_path):
    """Point the check at a binary with no stamp and watch it fire.

    This is the 1.4.1 near-miss re-staged: seven current binaries and
    one stale one that predates the stamped builds.
    """

    payloads = _stamped_payloads(RELEASED)
    payloads["rw_wrfbatch"] = b"stale build, no stamp"
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, payloads), out,
                  "--source-rev", RELEASED)
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "rw_wrfbatch" in message
    assert "carries no GPUWM_BRIDGE_SOURCE_REV" in message
    assert not out.exists()


def test_pin_refuses_a_bundle_stamped_with_another_revision(tmp_path):
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, _stamped_payloads(STALE)), out,
                  "--source-rev", RELEASED)
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert STALE in message and RELEASED in message
    assert not out.exists()


def test_pin_requires_the_source_revision(tmp_path):
    """The gate is not optional: a cut cannot skip it by omission."""

    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, _stamped_payloads(RELEASED)), out)
    assert result.returncode != 0
    assert "--source-rev" in (result.stdout + result.stderr)
    assert not out.exists()


def test_pin_accepts_a_bundle_stamped_with_the_released_revision(tmp_path):
    out = tmp_path / "pins.json"
    result = _pin(_bundle(tmp_path, _stamped_payloads(RELEASED)), out,
                  "--source-rev", RELEASED)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(out.read_text(encoding="utf-8"))
    pins = bridge_assets.parse_pins(document)
    assert pins.release == RELEASE
    assert PLATFORM in pins.platforms


# ---------------------------------------------------------------------------
# The build half: every bundled artifact's source embeds the stamp
# ---------------------------------------------------------------------------

#: Where each artifact's entry point lives, relative to its workspace.
#: An artifact added to BUNDLED_ARTIFACTS without an entry here -- or an
#: entry whose file stops referencing the stamp -- fails below, so a new
#: bridge cannot ship un-provable.
_STAMP_SOURCES = {
    "grib1_bridge": "src/main.rs",
    "gfs_grib2_bridge": "src/bin/gfs_grib2_bridge.rs",
    "hrrr_grib2_bridge": "src/bin/hrrr_grib2_bridge.rs",
    "grib2_inventory": "src/bin/grib2_inventory.rs",
    "grib2_dump": "src/bin/grib2_dump.rs",
    "gpuwm_preprocess_cpu": "src/lib.rs",
    "rw_fetch": "crates/rw-fetch/src/main.rs",
    "rw_wrfbatch": "crates/rw-wrfbatch/src/main.rs",
}

#: The build script that injects the revision for each artifact.
_STAMP_BUILDS = {
    bridges.CRATE_RELATIVE: ("build.rs",),
    bridges.RUSTWX_CRATE_RELATIVE: ("crates/rw-fetch/build.rs",
                                    "crates/rw-wrfbatch/build.rs"),
}


def _require_bridge_sources() -> None:
    if not (REPO_ROOT / bridges.CRATE_RELATIVE / "src").is_dir():
        pytest.skip("the stamp contract needs the source tree, not an "
                    "install")


def test_every_bundled_artifact_entry_point_references_the_stamp():
    _require_bridge_sources()
    unmapped = [artifact.name
                for artifact in bridge_assets.BUNDLED_ARTIFACTS
                if artifact.name not in _STAMP_SOURCES]
    assert not unmapped, (
        f"BUNDLED_ARTIFACTS gained {unmapped} with no stamp source mapped "
        "here; a bridge the cut cannot prove must not join a bundle")
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        source = (REPO_ROOT / artifact.crate
                  / _STAMP_SOURCES[artifact.name])
        text = source.read_text(encoding="utf-8")
        assert "SOURCE_REV_STAMP" in text, (
            f"{source} does not reference the source-revision stamp, so "
            f"{artifact.name} would build without one and the cut would "
            "refuse it")


def test_every_bridge_workspace_build_script_injects_the_revision():
    _require_bridge_sources()
    for workspace, scripts in _STAMP_BUILDS.items():
        for relative in scripts:
            script = REPO_ROOT / workspace / relative
            assert script.is_file(), f"{script} is missing"
            text = script.read_text(encoding="utf-8")
            assert "rustc-env=GPUWM_BRIDGE_SOURCE_REV" in text, (
                f"{script} no longer injects GPUWM_BRIDGE_SOURCE_REV")
