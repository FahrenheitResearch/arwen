"""Drive the release-artifact verifier outside a live cut.

``tools/verify_release_artifacts.py`` is the last thing between a built
wheel and PyPI, and until now the only way to run it was to be in the
middle of a release.  That is how a defect in it -- a bundle membership
test that compared the archive against the binary pins alone, and so
refused every bundle the current packer produces -- reached a live 1.4.1
round instead of a CI run.  ``--dry-run`` runs the same assertions, and
this module drives them.

The fixtures are deliberately not this module's own idea of a bundle:
the bundles, pins, and manifest come from ``tools/build_bridge_bundle.py``
-- the real writer the cut runs -- over fixture binaries carrying real
``GPUWM_BRIDGE_SOURCE_REV`` stamps.  A verifier tested against a fixture
its own author shaped proves only that two wrong ideas agree.
"""

from __future__ import annotations

from base64 import urlsafe_b64encode
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from gpuwm import bridge_assets
from tools import build_bridge_bundle, verify_release_artifacts

RELEASE = "v9.8.7"
VERSION = "9.8.7"
SOURCE_REV = "b" * 40
PINS_MEMBER = "gpuwm/data/bridges/bridge-pins.json"


def _stamped(name: str, source_rev: str) -> bytes:
    """Fixture bytes carrying exactly one well-formed revision stamp."""

    return b"".join(
        (
            b"fixture native artifact ",
            name.encode("utf-8"),
            b"\n",
            bridge_assets.SOURCE_REV_MARKER,
            source_rev.encode("ascii"),
            b"\n",
        )
    )


def _pack_and_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_rev: str = SOURCE_REV,
) -> tuple[Path, Path, Path]:
    """A cut's bundle payload, produced by the packer the workflow runs."""

    asset_source = tmp_path / "asset-source"
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        directory = asset_source / subdir
        directory.mkdir(parents=True)
        # Two files, so a test can drop one pin and still leave a bundle
        # that carries map assets at all -- the refusal for carrying none
        # is a different check, and it would mask this one.
        for index in (1, 2):
            (directory / f"fixture-geometry-{index}.bin").write_bytes(
                f"fixture {subdir} geometry {index}\n".encode("utf-8")
            )
    # The one seam the packer names for the asset tree.  Pointing it at a
    # fixture keeps the real walk, the real member naming, and the real
    # refusals, without carrying 36 MiB of basemaps through a unit test.
    monkeypatch.setattr(
        build_bridge_bundle, "source_asset_dir", lambda: asset_source
    )

    bundles = tmp_path / "dist-bridges"
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        search = tmp_path / f"target-{platform}"
        search.mkdir()
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            filename = bridge_assets.artifact_filename(artifact, platform)
            (search / filename).write_bytes(_stamped(filename, source_rev))
        assert (
            build_bridge_bundle.main(
                [
                    "pack",
                    "--release",
                    RELEASE,
                    "--platform",
                    platform,
                    "--search",
                    str(search),
                    "--out",
                    str(bundles),
                ]
            )
            == 0
        )

    pins = tmp_path / "bridge-pins.json"
    manifest = bundles / build_bridge_bundle.MANIFEST_ASSET
    bundle_args: list[str] = []
    for archive in sorted(bundles.glob("*.zip")):
        bundle_args += ["--bundle", str(archive)]
    assert (
        build_bridge_bundle.main(
            [
                "pin",
                "--release",
                RELEASE,
                "--source-rev",
                source_rev,
                *bundle_args,
                "--out",
                str(pins),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    return bundles, pins, manifest


def _build_dists(tmp_path: Path, pins: Path) -> tuple[Path, Path]:
    """A wheel and sdist carrying the generated pins the way a cut's do."""

    payload = pins.read_bytes()
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)

    wheel = dist / f"gpuwm-{VERSION}-py3-none-any.whl"
    digest = (
        urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    record = (
        f"{PINS_MEMBER},sha256={digest},{len(payload)}\n"
        f"gpuwm-{VERSION}.dist-info/RECORD,,\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(PINS_MEMBER, payload)
        archive.writestr(f"gpuwm-{VERSION}.dist-info/RECORD", record)

    sdist = dist / f"gpuwm-{VERSION}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo(f"gpuwm-{VERSION}/{PINS_MEMBER}")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return wheel, sdist


def _dry_run(
    tmp_path: Path,
    bundles: Path,
    pins: Path,
    manifest: Path,
    wheel: Path,
    sdist: Path,
    *,
    release: str = RELEASE,
    source_rev: str = SOURCE_REV,
) -> Path:
    receipt = tmp_path / "release-artifact-proof.json"
    assert (
        verify_release_artifacts.main(
            [
                "--dry-run",
                "--wheel",
                str(wheel),
                "--sdist",
                str(sdist),
                "--pins",
                str(pins),
                "--manifest",
                str(manifest),
                "--bundles",
                str(bundles),
                "--release",
                release,
                "--source-rev",
                source_rev,
                "--receipt",
                str(receipt),
            ]
        )
        == 0
    )
    return receipt


def test_dry_run_passes_the_packers_own_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt_path = _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["release"] == RELEASE
    assert receipt["source_rev"] == SOURCE_REV
    assert set(receipt["bundles"]) == set(bridge_assets.SUPPORTED_PLATFORMS)
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        assert receipt["bundles"][platform]["binaries"] == len(
            bridge_assets.BUNDLED_ARTIFACTS)
        assert receipt["bundles"][platform]["assets"] >= 1


def test_a_dry_run_receipt_never_claims_what_it_did_not_prove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode is on the receipt, and so is everything it skipped."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    receipt = json.loads(
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist).read_text(
            encoding="utf-8"
        )
    )

    assert receipt["mode"] == "dry-run"
    assert receipt["not_proven"] == list(verify_release_artifacts.DRY_RUN_SKIPS)
    assert receipt["installed_version"] is None
    assert receipt["host_probes"] == []
    assert receipt["prebuilt_offer_live_lines"] == []


def test_a_live_cut_still_requires_the_installation_it_proves() -> None:
    """Without ``--dry-run`` the cut-only inputs stay mandatory, so the
    mode can never be entered by forgetting an argument."""

    with pytest.raises(SystemExit) as refusal:
        verify_release_artifacts._parse_args(
            [
                "--wheel", "w", "--sdist", "s", "--pins", "p",
                "--manifest", "m", "--bundles", "b", "--release", RELEASE,
                "--source-rev", SOURCE_REV, "--receipt", "r",
            ]
        )
    assert refusal.value.code == 2


def _rewrite_platforms(pins: Path, manifest: Path, mutate) -> None:
    """Apply the same edit to both documents the cut publishes.

    They are cross-checked for equality before membership is looked at,
    so a one-sided edit would only ever prove that check.
    """

    for path, key in ((pins, "platforms"), (manifest, "platforms")):
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document[key])
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


@pytest.mark.parametrize("direction", ("under-declared", "over-declared"))
def test_dry_run_refuses_pins_that_do_not_name_the_bundle_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    """The 1.4.1 defect class, now reachable without a live cut.

    A bundle must contain exactly what the release pinned, so the check
    is an equality and both directions have to bite: a pinned member the
    archive does not carry is a download that 404s member-by-member, and
    an unpinned member the archive does carry is bytes nobody verified.
    """

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)

    def mutate(platforms: dict) -> None:
        record = platforms[bridge_assets.SUPPORTED_PLATFORMS[0]]
        if direction == "under-declared":
            record["assets"].pop()
        else:
            record["assets"].append(
                {
                    "path": "assets/basemap/never-packed.bin",
                    "bytes": 3,
                    "sha256": "0" * 64,
                }
            )

    _rewrite_platforms(pins, manifest, mutate)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(AssertionError) as refusal:
        _dry_run(tmp_path, bundles, pins, manifest, wheel, sdist)

    expected = (
        "unexpected: ['assets/basemap/fixture-geometry-2.bin']"
        if direction == "under-declared"
        else "missing: ['assets/basemap/never-packed.bin']"
    )
    assert expected in str(refusal.value)


def test_dry_run_refuses_binaries_stamped_with_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hash proves the bytes; only the stamp proves their source."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(bridge_assets.BridgeAssetError, match="was built from"):
        _dry_run(
            tmp_path, bundles, pins, manifest, wheel, sdist,
            source_rev="c" * 40,
        )


def test_dry_run_refuses_a_wheel_whose_record_misstates_the_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RECORD binds the pins' hash and size, not merely their name."""

    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)
    payload = pins.read_bytes()
    rebuilt = wheel.with_name("rebuilt.whl")
    with zipfile.ZipFile(rebuilt, "w") as archive:
        archive.writestr(PINS_MEMBER, payload)
        archive.writestr(
            f"gpuwm-{VERSION}.dist-info/RECORD",
            f"{PINS_MEMBER},sha256={'A' * 43},{len(payload)}\n",
        )

    with pytest.raises(AssertionError):
        _dry_run(tmp_path, bundles, pins, manifest, rebuilt, sdist)


def test_dry_run_refuses_pins_generated_for_another_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, pins, manifest = _pack_and_pin(tmp_path, monkeypatch)
    wheel, sdist = _build_dists(tmp_path, pins)

    with pytest.raises(AssertionError):
        _dry_run(
            tmp_path, bundles, pins, manifest, wheel, sdist, release="v0.0.1"
        )
