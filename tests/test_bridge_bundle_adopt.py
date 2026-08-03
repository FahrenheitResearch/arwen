"""The re-run adoption contract for already-uploaded release assets.

Rust bridge builds are not byte-reproducible, so a re-dispatched release
run rebuilds different bundle zips than the ones a previous run already
uploaded to the draft.  ``build_bridge_bundle.py adopt`` is the recovery
path: when the draft already carries uploaded assets, they are accepted
by CONTENT CONTRACT -- the exact member list (binaries by filename, map
assets by path) with the map-asset bytes hash-checked against the source
tree at the tag commit, and the uploaded manifest, when present, pinning
the uploaded zips member by member -- and then adopted in place of the
freshly built zips, so the pins, the wheel, and every downstream
byte-identity gate derive from the bytes the release actually holds.

These tests drive the real packer against a small monkeypatched asset
tree (one integration test uses the repository's real tree), because the
first run's ``pack`` is the only honest writer of the archives ``adopt``
must judge.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from gpuwm import bridge_assets, bridges

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = REPO_ROOT / "tools" / "build_bridge_bundle.py"
RELEASE = "v9.9.9"


def _module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("tools.build_bridge_bundle")


#: The 40-hex source revision every adopt-rig stub embeds and pin()
#: verifies against (bridge_assets.SOURCE_REV_MARKER contract).
SOURCE_REV = "ab" * 20


def _stub_binaries(directory: Path, flavor: bytes) -> Path:
    """Every platform's artifact files, embedding ``flavor`` in the bytes.

    Two different flavors model the non-reproducible rebuild: the same
    filenames, different bytes -- exactly what a workflow re-run builds.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            name = bridge_assets.artifact_filename(artifact, platform)
            marker = bridges.BRIDGE_ABI_MARKERS.get(artifact.name, b"")
            (directory / name).write_bytes(
                b"stub::" + artifact.name.encode("ascii") + b"::"
                + flavor + b"::" + marker
                # The staleness gate (test/bridge-staleness) makes pin()
                # verify every binary carries the release commit's stamp;
                # these stubs carry it so the adopt rig pins like a real
                # release build does.
                + b"::GPUWM_BRIDGE_SOURCE_REV=" + SOURCE_REV.encode("ascii"))
    return directory


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hold(held: Path, files: dict[str, Path],
          inventory_extra: tuple[dict, ...] = ()) -> None:
    """Lay out a draft-asset capture: inventory.json plus assets/<name>."""

    assets = held / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name, source in files.items():
        target = assets / name
        shutil.copyfile(source, target)
        payload = target.read_bytes()
        inventory.append({
            "name": name, "state": "uploaded", "size": len(payload),
            "digest": f"sha256:{_sha256(payload)}",
        })
    inventory.extend(inventory_extra)
    (held / "inventory.json").write_text(json.dumps(inventory),
                                         encoding="utf-8")


def _rewrite(archive: Path, *, replace: dict[str, bytes] = {},
             add: dict[str, bytes] = {}, drop: frozenset = frozenset()
             ) -> None:
    """Rebuild ``archive`` with members replaced, added, or dropped."""

    with zipfile.ZipFile(archive) as zf:
        entries = [(info, zf.read(info.filename)) for info in zf.infolist()]
    with zipfile.ZipFile(archive, "w",
                         compression=zipfile.ZIP_DEFLATED) as zf:
        for info, payload in entries:
            if info.filename in drop:
                continue
            clone = zipfile.ZipInfo(info.filename,
                                    date_time=info.date_time)
            clone.compress_type = info.compress_type
            clone.external_attr = info.external_attr
            zf.writestr(clone, replace.get(info.filename, payload))
        for name, payload in add.items():
            clone = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            clone.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(clone, payload)


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch):
    """Two builds of the same release: the uploaded one and this run's."""

    module = _module()
    source = tmp_path / "source-assets"
    (source / "basemap" / "nested").mkdir(parents=True)
    (source / "basemap" / "coast.bin").write_bytes(b"coastline bytes")
    (source / "basemap" / "nested" / "lakes.bin").write_bytes(b"lake bytes")
    monkeypatch.setattr(module, "source_asset_dir", lambda: source)

    uploaded = tmp_path / "uploaded-build"
    fresh = tmp_path / "dist-bridges"
    for flavor, out in ((b"first-run", uploaded), (b"re-run", fresh)):
        stub = _stub_binaries(tmp_path / f"stub-{flavor.decode()}", flavor)
        for platform in bridge_assets.SUPPORTED_PLATFORMS:
            module.pack(RELEASE, platform, [stub], out)
    pins = tmp_path / "uploaded-pins.json"
    manifest = tmp_path / "uploaded" / module.MANIFEST_ASSET
    module.pin(RELEASE, sorted(uploaded.glob("*.zip")), pins, manifest,
               SOURCE_REV)

    names = {platform: module.bundle_filename(RELEASE, platform)
             for platform in bridge_assets.SUPPORTED_PLATFORMS}
    return SimpleNamespace(module=module, source=source, uploaded=uploaded,
                           fresh=fresh, pins=pins, manifest=manifest,
                           names=names, held=tmp_path / "draft-held",
                           tmp=tmp_path)


def _adopt(rig) -> int:
    return rig.module.main([
        "adopt", "--release", RELEASE,
        "--held", str(rig.held), "--bundles", str(rig.fresh),
        "--source-rev", SOURCE_REV])


def _refusal(rig) -> str:
    with pytest.raises(SystemExit) as outcome:
        _adopt(rig)
    message = str(outcome.value)
    assert message and message != "0", "adopt refused without a message"
    return message


# ---------------------------------------------------------------------------
# The first run is untouched
# ---------------------------------------------------------------------------

def test_no_uploaded_assets_is_a_noop(rig, capsys):
    """First-run behaviour: nothing uploaded yet, nothing adopted."""

    rig.held.mkdir()
    (rig.held / "inventory.json").write_text("[]", encoding="utf-8")
    before = {path.name: path.read_bytes()
              for path in sorted(rig.fresh.iterdir())}

    assert _adopt(rig) == 0

    after = {path.name: path.read_bytes()
             for path in sorted(rig.fresh.iterdir())}
    assert after == before
    assert "no uploaded assets" in capsys.readouterr().out


def test_a_missing_capture_inventory_is_refused(rig):
    """No inventory means the capture never ran; adopting blind is out."""

    rig.held.mkdir()
    message = _refusal(rig)
    assert "inventory.json" in message


# ---------------------------------------------------------------------------
# The re-run adopts what the draft holds
# ---------------------------------------------------------------------------

def test_rerun_adopts_uploaded_bundles_and_repins_them_byte_exactly(rig):
    """The whole point: pins regenerated from adopted zips byte-match the
    uploaded manifest, so every downstream byte-identity gate passes."""

    _hold(rig.held, {name: rig.uploaded / name
                     for name in rig.names.values()}
          | {rig.module.MANIFEST_ASSET: rig.manifest})
    for name in rig.names.values():
        assert (rig.fresh / name).read_bytes() \
            != (rig.uploaded / name).read_bytes(), \
            "fixture must model a non-reproducible rebuild"

    assert _adopt(rig) == 0

    for name in rig.names.values():
        assert (rig.fresh / name).read_bytes() \
            == (rig.uploaded / name).read_bytes()
    repins = rig.tmp / "repins.json"
    remanifest = rig.tmp / "re" / rig.module.MANIFEST_ASSET
    rig.module.pin(RELEASE, sorted(rig.fresh.glob("*.zip")), repins,
                   remanifest, SOURCE_REV)
    assert remanifest.read_bytes() == rig.manifest.read_bytes()
    assert repins.read_bytes() == rig.pins.read_bytes()


def test_a_partial_upload_without_manifest_adopts_only_what_uploaded(rig):
    """Bundles upload before the manifest; a crash between them leaves
    zips the re-run adopts and a manifest slot this run fills freshly."""

    linux = rig.names["linux-x86_64"]
    windows = rig.names["win-x86_64"]
    fresh_windows = (rig.fresh / windows).read_bytes()
    _hold(rig.held, {linux: rig.uploaded / linux})

    assert _adopt(rig) == 0

    assert (rig.fresh / linux).read_bytes() \
        == (rig.uploaded / linux).read_bytes()
    assert (rig.fresh / windows).read_bytes() == fresh_windows


def test_starter_inventory_entries_are_not_adopted(rig):
    """An interrupted upload's starter record is the assets job's to
    delete; adoption considers only assets in the uploaded state."""

    linux = rig.names["linux-x86_64"]
    _hold(rig.held, {linux: rig.uploaded / linux},
          inventory_extra=({"name": rig.names["win-x86_64"],
                            "state": "starter", "size": 0,
                            "digest": None},))

    assert _adopt(rig) == 0
    assert (rig.fresh / linux).read_bytes() \
        == (rig.uploaded / linux).read_bytes()


# ---------------------------------------------------------------------------
# The acceptance path is not vacuous: corruption is named and refused
# ---------------------------------------------------------------------------

def test_a_corrupted_map_asset_member_is_refused_naming_it(rig):
    linux = rig.names["linux-x86_64"]
    _hold(rig.held, {linux: rig.uploaded / linux})
    member = "assets/basemap/coast.bin"
    _rewrite(rig.held / "assets" / linux,
             replace={member: b"corrupted coastline"})
    _reinventory(rig.held)

    message = _refusal(rig)
    assert member in message
    assert linux in message


def test_a_corrupted_binary_member_is_refused_when_the_manifest_pins_it(rig):
    linux = rig.names["linux-x86_64"]
    _hold(rig.held, {linux: rig.uploaded / linux,
                     rig.names["win-x86_64"]:
                         rig.uploaded / rig.names["win-x86_64"],
                     rig.module.MANIFEST_ASSET: rig.manifest})
    member = bridge_assets.artifact_filename(
        bridge_assets.BUNDLED_ARTIFACTS[0], "linux-x86_64")
    _rewrite(rig.held / "assets" / linux,
             replace={member: b"stub::swapped-binary-bytes"})
    _reinventory(rig.held)

    message = _refusal(rig)
    assert member in message


def test_an_extra_member_is_refused_naming_it(rig):
    linux = rig.names["linux-x86_64"]
    _hold(rig.held, {linux: rig.uploaded / linux})
    _rewrite(rig.held / "assets" / linux,
             add={"assets/basemap/smuggled.bin": b"not in the tree"})
    _reinventory(rig.held)

    message = _refusal(rig)
    assert "assets/basemap/smuggled.bin" in message


def test_a_missing_member_is_refused_naming_it(rig):
    linux = rig.names["linux-x86_64"]
    _hold(rig.held, {linux: rig.uploaded / linux})
    _rewrite(rig.held / "assets" / linux,
             drop=frozenset({"assets/basemap/nested/lakes.bin"}))
    _reinventory(rig.held)

    message = _refusal(rig)
    assert "assets/basemap/nested/lakes.bin" in message


def test_an_unexpected_uploaded_asset_name_is_refused(rig):
    linux = rig.names["linux-x86_64"]
    stray = rig.tmp / "stray.bin"
    stray.write_bytes(b"who put this on the draft")
    _hold(rig.held, {linux: rig.uploaded / linux, "stray.bin": stray})

    message = _refusal(rig)
    assert "stray.bin" in message


def test_an_orphan_manifest_without_its_bundles_is_refused(rig):
    """A manifest pinning bundle bytes the draft does not hold can never
    be reconciled: the pinned bytes are gone forever.  Say which."""

    _hold(rig.held, {rig.module.MANIFEST_ASSET: rig.manifest})

    message = _refusal(rig)
    assert rig.names["linux-x86_64"] in message
    assert rig.names["win-x86_64"] in message


def test_a_manifest_disagreeing_with_an_uploaded_zip_is_refused(rig):
    """Same names, different bytes: the manifest pins the true first-run
    zips while the draft's zip was swapped for this run's rebuild."""

    linux = rig.names["linux-x86_64"]
    windows = rig.names["win-x86_64"]
    _hold(rig.held, {linux: rig.fresh / linux,   # NOT the pinned bytes
                     windows: rig.uploaded / windows,
                     rig.module.MANIFEST_ASSET: rig.manifest})

    message = _refusal(rig)
    assert linux in message


def test_a_reformatted_manifest_is_refused(rig):
    """Semantically equal but not the byte-exact document ``pin`` writes:
    the assets job's byte-identity backstop would refuse it later, so
    refuse it here, where the message can say why."""

    reformatted = rig.tmp / "reformatted.json"
    reformatted.write_text(
        json.dumps(json.loads(rig.manifest.read_text(encoding="utf-8")),
                   indent=4, sort_keys=True) + "\n",
        encoding="utf-8")
    _hold(rig.held, {name: rig.uploaded / name
                     for name in rig.names.values()}
          | {rig.module.MANIFEST_ASSET: reformatted})

    message = _refusal(rig)
    assert rig.module.MANIFEST_ASSET in message


def test_a_held_file_disagreeing_with_its_inventory_record_is_refused(rig):
    linux = rig.names["linux-x86_64"]
    _hold(rig.held, {linux: rig.uploaded / linux})
    with (rig.held / "assets" / linux).open("ab") as stream:
        stream.write(b"trailing garbage")

    message = _refusal(rig)
    assert linux in message


def _reinventory(held: Path) -> None:
    """Recompute inventory records after a held zip was rewritten, so a
    corruption test exercises the content contract, not the transport
    check."""

    inventory = json.loads((held / "inventory.json").read_text("utf-8"))
    for entry in inventory:
        path = held / "assets" / entry["name"]
        if entry.get("state") == "uploaded" and path.is_file():
            payload = path.read_bytes()
            entry["size"] = len(payload)
            entry["digest"] = f"sha256:{_sha256(payload)}"
    (held / "inventory.json").write_text(json.dumps(inventory),
                                         encoding="utf-8")


# ---------------------------------------------------------------------------
# Against the real tree, through the real CLI
# ---------------------------------------------------------------------------

def test_adoption_against_the_repository_asset_tree(tmp_path: Path):
    """One un-monkeypatched pass: pack with the repository's real map
    assets, rebuild with different binary bytes, adopt via the CLI."""

    if not (REPO_ROOT / "tools" / "rustwx" / "assets").is_dir():
        pytest.skip("the adoption contract needs the source tree")
    module = _module()
    platform = "linux-x86_64"
    name = module.bundle_filename(RELEASE, platform)
    uploaded = tmp_path / "uploaded-build"
    fresh = tmp_path / "dist-bridges"
    for flavor, out in ((b"first-run", uploaded), (b"re-run", fresh)):
        stub = _stub_binaries(tmp_path / f"stub-{flavor.decode()}", flavor)
        result = subprocess.run(
            [sys.executable, str(BUNDLE_TOOL), "pack", "--release", RELEASE,
             "--platform", platform, "--search", str(stub),
             "--out", str(out)],
            capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, result.stdout + result.stderr
    held = tmp_path / "draft-held"
    _hold(held, {name: uploaded / name})

    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "adopt", "--release", RELEASE,
         "--held", str(held), "--bundles", str(fresh),
         "--source-rev", SOURCE_REV],
        capture_output=True, text=True, cwd=REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (fresh / name).read_bytes() == (uploaded / name).read_bytes()
