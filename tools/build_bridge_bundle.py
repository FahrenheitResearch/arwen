"""Build the prebuilt-bridge release assets and the pins the wheel carries.

Two steps, run in this order by the release workflow (and reproducible
by hand from a checkout):

``pack``
    On each target platform, after ``cargo build --release --locked`` in
    the decoder, renderer and mapped-engine workspaces and the vendored
    dealiasing crate, collect every artifact of
    ``gpuwm.bridge_assets.BUNDLED_ARTIFACTS``, plus the renderer's map
    assets, into one zip named for the release and the platform.  The
    archive is deterministic: binaries first in the declared artifact
    order, then the asset tree in sorted path order, fixed timestamps,
    no directory entries, so two packs of the same bytes produce the
    same archive.

    The asset file list is *walked from the tree*, never enumerated
    here.  An enumeration is what let the packaged basemaps fall out of
    the distribution in the first place: a list nobody regenerates goes
    quietly stale, and the plot that results has a storm on it and no
    coastline.  ``REQUIRED_ASSET_SUBDIRS`` declares the directories, the
    walk finds the files, and ``pin`` hashes whatever the walk found.

``pin``
    On one machine, from the bundles ``pack`` produced, compute the
    size + SHA-256 pins of every bundle and of every artifact inside it,
    and write them into the packaged pins document
    (``gpuwm/data/bridges/bridge-pins.json``) *before* the wheel is
    built.  Optionally also write the release-asset manifest, the
    human-readable index of what a release published.

There is a third step the release workflow runs between them, only ever
relevant to a RE-RUN:

``adopt``
    Rust builds are not byte-reproducible, so a re-dispatched release
    run packs different zips than the ones a previous run already
    uploaded to the draft -- and every byte-identity gate downstream
    would strand the release on that difference.  ``adopt`` accepts the
    uploaded assets by content contract instead: the member list must be
    exactly what this commit packs (the binaries by filename, the map
    assets by path), every map-asset member must hash-match the source
    tree, and the uploaded manifest, when present, must pin the uploaded
    zips member by member and byte-match the manifest ``pin`` would
    regenerate from them.  What passes replaces the freshly built zips,
    so the pins, the wheel, and the byte-identity gates all describe the
    bytes the draft actually holds.  A member that diverges is named
    individually and the whole adoption is refused.  A draft with no
    uploaded assets adopts nothing: the first run is untouched.

Nothing here invents a hash: every number written comes from hashing
bytes that exist on this disk at the moment it runs.  ``pin`` refuses a
bundle whose contents do not match the artifact set gpuwm expects for
that platform, so a half-built bundle cannot be pinned into a wheel.

``pin`` also refuses a STALE bundle.  Hashing binds a wheel to exact
bytes, but says nothing about which source produced them: the 1.4.1
cut nearly shipped prepared bundles predating the source tip, the two
platform zips not even from the same revision, and every check passed
because every check hashed what it was handed.  Each bundled artifact
now embeds ``GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`` at build time
(the workspace build scripts inject the checkout's HEAD), and ``pin
--source-rev COMMIT`` -- a required argument, so no cut can skip it --
extracts the stamp from every binary member and refuses a bundle whose
stamp is absent, unparseable, ambiguous, or names any other commit.
String extraction, never execution: it works on the other platform's
binaries and on a GPU-less runner.

The one exception is a VENDORED artifact, whose source is a verbatim
upstream tree frozen at a recorded commit and therefore does not move
with this checkout at all.  A HEAD stamp would say nothing about it and
injecting one would edit a tree whose whole claim is that it is
unmodified, so ``pin`` asks it for its declared contract marker instead
-- equally a property of the bytes, equally read without executing
anything, and equally fatal when a build predating this release's ABI
turns up in a bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm import bridge_assets, bridges  # noqa: E402

#: Fixed member timestamp so two packs of identical bytes produce
#: identical archives (zip stores mtime per member).
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

#: The release-asset name of the manifest published beside the bundles.
#: ``adopt`` recognises it among a draft's uploaded assets, and the
#: workflow uploads it under exactly this name.
MANIFEST_ASSET = "bridge-bundle-manifest.json"


def bundle_filename(release: str, platform: str) -> str:
    return f"gpuwm-bridges-{release}-{platform}.zip"


def _locate(name: str, search: list[Path]) -> Path:
    for directory in search:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"build_bridge_bundle: {name} is in none of the search "
        f"directories: {', '.join(str(d) for d in search)}")


def source_asset_dir() -> Path:
    """The renderer's asset tree in a checkout: ``tools/rustwx/assets``."""

    return REPO_ROOT / "tools" / "rustwx" / "assets"


def collect_assets(asset_dir: Path | None = None
                   ) -> list[tuple[str, Path]]:
    """Every map asset file, as ``(archive member name, source path)``.

    Walked, not enumerated: each required subdirectory of
    ``bridge_assets.REQUIRED_ASSET_SUBDIRS`` is descended in full and
    every regular file in it is carried.  A required subdirectory that
    does not exist, or that holds no files, is a hard refusal -- packing
    a bundle whose renderer would draw blank geography is the failure
    this whole path exists to make impossible.

    Member names are ``assets/<subdir>/...`` with forward slashes on
    every platform, which is both the zip convention and the exact
    relative path staging writes to.
    """

    root = source_asset_dir() if asset_dir is None else Path(asset_dir)
    collected: list[tuple[str, Path]] = []
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        source = root / subdir
        if not source.is_dir():
            raise SystemExit(
                f"build_bridge_bundle: required asset directory {source} "
                "does not exist; refusing to pack a bundle whose renderer "
                "would draw plots with no coastlines or borders")
        found = sorted(p for p in source.rglob("*") if p.is_file())
        if not found:
            raise SystemExit(
                f"build_bridge_bundle: required asset directory {source} "
                "holds no files; refusing to pack a bundle whose renderer "
                "would draw plots with no coastlines or borders")
        for path in found:
            member = "/".join(
                (bridge_assets.ASSET_ROOT, subdir,
                 path.relative_to(source).as_posix()))
            collected.append((member, path))
    return collected


def pack(release: str, platform: str, search: list[Path],
         out_dir: Path) -> Path:
    if platform not in bridge_assets.SUPPORTED_PLATFORMS:
        raise SystemExit(
            f"build_bridge_bundle: unknown platform {platform!r}; known: "
            f"{', '.join(bridge_assets.SUPPORTED_PLATFORMS)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / bundle_filename(release, platform)
    sources = [(bridge_assets.artifact_filename(artifact, platform),
                _locate(bridge_assets.artifact_filename(artifact, platform),
                        search))
               for artifact in bridge_assets.BUNDLED_ARTIFACTS]
    assets = collect_assets()
    with zipfile.ZipFile(archive, "w",
                         compression=zipfile.ZIP_DEFLATED) as zf:
        for name, source in sources:
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            # 0o755 in the high half of external_attr: the unix mode a
            # zip can carry.  Staging chmods anyway (a zip's mode is not
            # something to depend on), but an operator who unzips by
            # hand should get executables.
            info.external_attr = (0o100755 << 16)
            zf.writestr(info, source.read_bytes())
        for name, source in assets:
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            # Data, not programs: 0o644.
            info.external_attr = (0o100644 << 16)
            zf.writestr(info, source.read_bytes())
    asset_bytes = sum(source.stat().st_size for _, source in assets)
    print(f"build_bridge_bundle: packed {archive} "
          f"({archive.stat().st_size:,} B) from "
          f"{len(sources)} artifacts and {len(assets)} map asset files "
          f"({asset_bytes:,} B unpacked)")
    for name, source in sources:
        print(f"  {name} <- {source}")
    print(f"  {bridge_assets.ASSET_ROOT}/ <- {source_asset_dir()} "
          f"({len(assets)} files under "
          f"{', '.join(bridge_assets.REQUIRED_ASSET_SUBDIRS)})")
    return archive


def _platform_of(archive: Path, release: str) -> str:
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        if archive.name == bundle_filename(release, platform):
            return platform
    raise SystemExit(
        f"build_bridge_bundle: {archive.name} is not a bundle name for "
        f"release {release}; expected one of "
        + ", ".join(bundle_filename(release, p)
                    for p in bridge_assets.SUPPORTED_PLATFORMS))


def _pin_bundle(archive: Path, release: str,
                source_rev: str) -> tuple[str, dict]:
    platform = _platform_of(archive, release)
    expected = [bridge_assets.artifact_filename(artifact, platform)
                for artifact in bridge_assets.BUNDLED_ARTIFACTS]
    with zipfile.ZipFile(archive) as zf:
        held = set(zf.namelist())
        missing = [name for name in expected if name not in held]
        if missing:
            raise SystemExit(
                f"build_bridge_bundle: {archive.name} is missing "
                f"{', '.join(missing)}; refusing to pin a partial bundle")
        binaries = []
        for artifact, name in zip(bridge_assets.BUNDLED_ARTIFACTS, expected):
            payload = zf.read(name)
            # A hash binds the wheel to these bytes; the stamp proves the
            # bytes came from the commit being released.  Refusing here,
            # before anything is written, is what keeps a stale binary
            # from becoming a perfectly-pinned release asset.
            #
            # A VENDORED artifact is a verbatim upstream crate frozen at a
            # recorded commit: it does not move with this checkout, so the
            # release commit says nothing about it, and stamping it would
            # mean editing a tree whose whole claim is that no file in it
            # differs from upstream by a byte.  Its declared contract
            # marker is asked for instead -- a property of these exact
            # bytes, naming the ABI this release speaks -- so the
            # equivalent staleness (a library built from an older vendored
            # crate) is still refused here rather than shipped.
            try:
                if artifact.vendored:
                    marker = bridges.BRIDGE_ABI_MARKERS.get(artifact.name)
                    if marker is None:
                        raise bridge_assets.BridgeAssetError(
                            f"{archive.name}: {name} is a vendored artifact "
                            "with no declared contract marker, so nothing "
                            "proves which build of it this is; add one to "
                            "gpuwm.bridges.BRIDGE_ABI_MARKERS")
                    if marker not in payload:
                        raise bridge_assets.BridgeAssetError(
                            f"{archive.name}: {name} does not carry the "
                            f"{artifact.name} contract marker "
                            f"{marker.decode('ascii', 'replace')!r}, so it "
                            "was built from a vendored crate older than the "
                            "one this release speaks to; rebuild it from "
                            f"{artifact.crate} in the release checkout")
                else:
                    bridge_assets.verify_source_revision(
                        payload, expected=source_rev,
                        label=f"{archive.name}: {name}")
            except bridge_assets.BridgeAssetError as error:
                raise SystemExit(f"build_bridge_bundle: {error}; refusing "
                                 "to pin a bundle that is not provably "
                                 "built from the release source revision")
            binaries.append({
                "artifact": artifact.name,
                "filename": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        # Whatever asset members the archive actually holds, hashed from
        # its own bytes.  Nothing is listed here that pack did not put
        # there, and nothing pack put there can be left out.
        prefix = f"{bridge_assets.ASSET_ROOT}/"
        asset_names = sorted(name for name in held
                             if name.startswith(prefix))
        assets = []
        for name in asset_names:
            payload = zf.read(name)
            assets.append({
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    if not assets:
        raise SystemExit(
            f"build_bridge_bundle: {archive.name} carries no "
            f"{bridge_assets.ASSET_ROOT}/ members; refusing to pin a "
            "bundle whose renderer would draw plots with no coastlines "
            "or borders.  Re-pack it with a `pack` that includes the "
            "map assets")
    covered = {name.split("/")[1] for name in asset_names if "/" in name}
    absent = [subdir for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS
              if subdir not in covered]
    if absent:
        raise SystemExit(
            f"build_bridge_bundle: {archive.name} carries no asset files "
            f"for required subdirectory/-ies {', '.join(absent)}; refusing "
            "to pin a partial asset set")
    record = {
        "bundle": {
            "filename": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": bridge_assets.sha256_file(archive),
        },
        "binaries": binaries,
        "assets": assets,
    }
    return platform, record


def pin(release: str, archives: list[Path], out: Path,
        manifest: Path | None, source_rev: str) -> Path:
    platforms: dict[str, dict] = {}
    for archive in archives:
        platform, record = _pin_bundle(archive, release, source_rev)
        if platform in platforms:
            raise SystemExit(
                f"build_bridge_bundle: two bundles for {platform}")
        platforms[platform] = record
    document = {
        "schema": bridge_assets.PINS_SCHEMA,
        "release": release,
        "platforms": platforms,
        "note": (
            "Size + SHA-256 pins for the prebuilt Rust bundles "
            "`gpuwm fetch-bridges` stages.  Generated by "
            "`python tools/build_bridge_bundle.py pin` from the exact "
            "bytes this release published."),
    }
    # Parse what we are about to write with the consumer's own validator:
    # a pins document the runtime would refuse must never reach a wheel.
    bridge_assets.parse_pins(document, origin=str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"build_bridge_bundle: wrote {out} pinning "
          f"{len(platforms)} platform(s) for {release}")
    for platform, record in sorted(platforms.items()):
        print(f"  {platform}: {record['bundle']['filename']} "
              f"{record['bundle']['bytes']:,} B "
              f"sha256 {record['bundle']['sha256']}")
        print(f"    {len(record['binaries'])} binaries, "
              f"{len(record['assets'])} map asset files")
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(_manifest_bytes(release, platforms))
        print(f"build_bridge_bundle: wrote release manifest {manifest}")
    return out


def _manifest_bytes(release: str, platforms: dict[str, dict]) -> bytes:
    """The exact bytes ``pin`` publishes as the release-asset manifest.

    One serialisation, shared by the writer and by ``adopt``'s
    byte-equality check, so the two can never drift apart.
    """

    payload = {
        "schema": bridge_assets.BUNDLE_MANIFEST_SCHEMA,
        "release": release,
        "platforms": platforms,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


# ---------------------------------------------------------------------------
# Adoption of a draft's already-uploaded assets (workflow re-runs)
# ---------------------------------------------------------------------------

def _held_uploaded_assets(release: str, held: Path
                          ) -> tuple[dict[str, str], Path | None]:
    """What the draft-asset capture delivered, verified against its
    inventory: ``{bundle filename: platform}`` plus the manifest path.

    The capture step recorded each uploaded asset's size and digest from
    the GitHub API and verified the download; re-verifying here means a
    truncated artifact transfer between jobs is a refusal, not an
    adoption.
    """

    inventory_path = held / "inventory.json"
    if not inventory_path.is_file():
        raise SystemExit(
            f"build_bridge_bundle: {inventory_path} does not exist; the "
            "draft-asset capture did not deliver its inventory, and "
            "adopting assets nobody inventoried is out of the question")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise SystemExit(
            f"build_bridge_bundle: {inventory_path} is not JSON: {error}")
    if not isinstance(inventory, list) \
            or not all(isinstance(entry, dict) for entry in inventory):
        raise SystemExit(
            f"build_bridge_bundle: {inventory_path} is not a list of "
            "asset records")
    zip_platform = {bundle_filename(release, platform): platform
                    for platform in bridge_assets.SUPPORTED_PLATFORMS}
    bundles: dict[str, str] = {}
    manifest: Path | None = None
    for entry in inventory:
        if entry.get("state") != "uploaded":
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(
                "build_bridge_bundle: an uploaded inventory entry has "
                "no name")
        if name == MANIFEST_ASSET:
            manifest = held / "assets" / name
        elif name in zip_platform:
            bundles[name] = zip_platform[name]
        else:
            raise SystemExit(
                "build_bridge_bundle: the draft carries an uploaded "
                f"asset this release does not produce: {name}; refusing "
                "to adopt around it")
        path = held / "assets" / name
        if not path.is_file():
            raise SystemExit(
                f"build_bridge_bundle: the inventory lists {name} as "
                "uploaded but the capture did not deliver the file")
        size = entry.get("size")
        if isinstance(size, int) and path.stat().st_size != size:
            raise SystemExit(
                f"build_bridge_bundle: {name} is "
                f"{path.stat().st_size:,} B on disk but the draft "
                f"recorded {size:,} B; refusing a capture that does not "
                "match its own inventory")
        digest = entry.get("digest")
        if isinstance(digest, str) and digest:
            observed = f"sha256:{bridge_assets.sha256_file(path)}"
            if observed != digest:
                raise SystemExit(
                    f"build_bridge_bundle: {name} hashes to {observed} "
                    f"but the draft recorded {digest}; refusing a "
                    "capture that does not match its own inventory")
    return bundles, manifest


def _contract_problems(name: str, archive: Path, platform: str,
                       asset_sources: dict[str, Path]) -> list[str]:
    """Every way ``archive`` diverges from this commit's content contract.

    The contract is exactly what ``pack`` produces from this checkout:
    the ten binaries by filename (their bytes are the previous run's
    non-reproducible build, proven on its native runner before upload,
    and pinned by the uploaded manifest when there is one), and the
    walked map-asset tree by path AND bytes, because for a data file the
    source tree at this tag commit is the ground truth.
    """

    problems: list[str] = []
    binary_names = [bridge_assets.artifact_filename(artifact, platform)
                    for artifact in bridge_assets.BUNDLED_ARTIFACTS]
    expected = set(binary_names) | set(asset_sources)
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as error:
        return [f"{name}: not a readable zip archive: {error}"]
    with zf:
        members = set(zf.namelist())
        for member in sorted(members - expected):
            problems.append(
                f"{name}: member {member} is not part of this release's "
                "content contract")
        for member in sorted(expected - members):
            problems.append(f"{name}: member {member} is missing")
        for member, source in sorted(asset_sources.items()):
            if member not in members:
                continue
            payload = zf.read(member)
            want = source.read_bytes()
            if payload != want:
                problems.append(
                    f"{name}: member {member} diverges from the source "
                    f"tree at this commit: sha256 "
                    f"{hashlib.sha256(payload).hexdigest()} != "
                    f"{hashlib.sha256(want).hexdigest()}")
        for member in binary_names:
            if member in members and zf.getinfo(member).file_size == 0:
                problems.append(f"{name}: member {member} is empty")
    return problems


def _manifest_problems(release: str, uploaded_manifest: bytes,
                       records: dict[str, dict]) -> list[str]:
    """Every way the uploaded manifest disagrees with the uploaded zips.

    ``records`` is what ``pin`` computes from the uploaded zips
    themselves, so a disagreement here names the exact member whose
    bytes the manifest pinned differently -- or a bundle the manifest
    pins that the draft no longer holds, which is unrecoverable: those
    bytes are a previous run's build and cannot be rebuilt.
    """

    problems: list[str] = []
    try:
        document = json.loads(uploaded_manifest.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        return [f"{MANIFEST_ASSET}: not JSON: {error}"]
    if not isinstance(document, dict):
        return [f"{MANIFEST_ASSET}: not a JSON object"]
    if document.get("schema") != bridge_assets.BUNDLE_MANIFEST_SCHEMA:
        problems.append(
            f"{MANIFEST_ASSET}: schema is {document.get('schema')!r}, "
            f"expected {bridge_assets.BUNDLE_MANIFEST_SCHEMA!r}")
    if document.get("release") != release:
        problems.append(
            f"{MANIFEST_ASSET}: release is {document.get('release')!r}, "
            f"expected {release!r}")
    recorded = document.get("platforms")
    if not isinstance(recorded, dict):
        problems.append(f"{MANIFEST_ASSET}: platforms is not an object")
        return problems
    for platform in sorted(set(recorded) - set(records)):
        record = recorded[platform]
        pinned = record.get("bundle", {}).get("filename") \
            if isinstance(record, dict) else None
        problems.append(
            f"{MANIFEST_ASSET}: pins {pinned or platform}, which the "
            "draft does not hold as an uploaded asset; those exact "
            "bytes were a previous run's build and cannot be rebuilt")
    for platform in sorted(set(records) - set(recorded)):
        problems.append(
            f"{MANIFEST_ASSET}: does not pin the uploaded bundle for "
            f"{platform}")
    for platform in sorted(set(records) & set(recorded)):
        want = records[platform]
        got = recorded[platform]
        if not isinstance(got, dict):
            problems.append(
                f"{MANIFEST_ASSET}: {platform}: record is not an object")
            continue
        if got.get("bundle") != want["bundle"]:
            problems.append(
                f"{MANIFEST_ASSET}: {platform}: pins "
                f"{want['bundle']['filename']} as {got.get('bundle')}, "
                f"but the uploaded zip is {want['bundle']}")
        for kind, key in (("binaries", "filename"), ("assets", "path")):
            raw = got.get(kind, [])
            named = {entry.get(key): entry for entry in raw
                     if isinstance(entry, dict)} \
                if isinstance(raw, list) else {}
            actual = {entry[key]: entry for entry in want[kind]}
            for member in sorted(set(named) | set(actual)):
                if named.get(member) != actual.get(member):
                    problems.append(
                        f"{MANIFEST_ASSET}: {platform}: member {member} "
                        "diverges between the manifest and the uploaded "
                        "zip")
    return problems


def adopt(release: str, held: Path, bundles_dir: Path,
          source_rev: str) -> None:
    """Adopt a draft's already-uploaded assets, or refuse loudly.

    Rust builds are not byte-reproducible, so this is the only honest
    way a re-dispatched run can coexist with assets a previous run
    already uploaded: prove the uploaded zips satisfy this commit's
    content contract, then make them THIS run's bundles, so the pins,
    the wheel, and every byte-identity gate downstream describe the
    bytes the draft actually holds.  With nothing uploaded this is a
    no-op and the first run proceeds byte-for-byte as it always did.
    """

    bundles, manifest = _held_uploaded_assets(release, held)
    if not bundles and manifest is None:
        print("build_bridge_bundle: the draft carries no uploaded assets; "
              "nothing to adopt, this run's freshly built bundles stand")
        return
    asset_sources = dict(collect_assets())
    problems: list[str] = []
    for name, platform in sorted(bundles.items()):
        problems.extend(_contract_problems(
            name, held / "assets" / name, platform, asset_sources))
    if manifest is not None and not problems:
        # Only structurally whole zips can be re-pinned for the
        # comparison; structural refusals above already name the members.
        records = {platform: _pin_bundle(held / "assets" / name, release,
                                         source_rev)[1]
                   for name, platform in bundles.items()}
        uploaded_manifest = manifest.read_bytes()
        if uploaded_manifest != _manifest_bytes(release, records):
            divergences = _manifest_problems(
                release, uploaded_manifest, records)
            problems.extend(divergences or [
                f"{MANIFEST_ASSET}: pins the same contents but is not "
                "the byte-exact document `pin` regenerates, so the "
                "byte-identity reconcile would refuse it; a hand-edited "
                "manifest cannot be adopted"])
    if problems:
        raise SystemExit(
            "build_bridge_bundle: refusing to adopt the draft's uploaded "
            "assets:\n  " + "\n  ".join(problems))
    bundles_dir.mkdir(parents=True, exist_ok=True)
    for name, platform in sorted(bundles.items()):
        shutil.copyfile(held / "assets" / name, bundles_dir / name)
        print(f"build_bridge_bundle: adopted {name} for {platform} "
              "(uploaded by a previous run; member list and map-asset "
              "bytes verified against this commit)")
    for platform in bridge_assets.SUPPORTED_PLATFORMS:
        if bundle_filename(release, platform) not in bundles:
            print(f"build_bridge_bundle: {platform}: no uploaded bundle "
                  "on the draft; this run's freshly built bundle stands")
    if manifest is not None:
        print(f"build_bridge_bundle: the uploaded {MANIFEST_ASSET} pins "
              "the adopted bundles exactly; `pin` will regenerate it "
              "byte-for-byte")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_bridge_bundle",
        description="Build the prebuilt-bridge release assets and the "
                    "pins the wheel carries.")
    sub = parser.add_subparsers(dest="command", required=True)

    packer = sub.add_parser(
        "pack", help="collect the built artifacts into one release bundle")
    packer.add_argument("--release", required=True, metavar="TAG",
                        help="release tag the bundle is named for")
    packer.add_argument("--platform", required=True,
                        choices=bridge_assets.SUPPORTED_PLATFORMS)
    packer.add_argument("--search", action="append", required=True,
                        type=Path, metavar="DIR",
                        help="directory to look for built artifacts in "
                             "(repeat: one per cargo target directory)")
    packer.add_argument("--out", type=Path, required=True, metavar="DIR",
                        help="directory to write the bundle into")

    pinner = sub.add_parser(
        "pin", help="write the packaged pins document from built bundles")
    pinner.add_argument("--release", required=True, metavar="TAG")
    pinner.add_argument("--bundle", action="append", required=True,
                        type=Path, metavar="ZIP",
                        help="a bundle produced by `pack` (repeat per "
                             "platform)")
    pinner.add_argument(
        "--source-rev", required=True, metavar="COMMIT",
        help="the full 40-hex git commit being released; every binary in "
             "every bundle must carry a GPUWM_BRIDGE_SOURCE_REV stamp "
             "naming exactly this commit, or the pin is refused.  "
             "Required, so a cut cannot skip the staleness check the "
             "way the 1.4.1 preparation nearly did")
    pinner.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "gpuwm" / Path(bridge_assets.PINS_RESOURCE),
        metavar="JSON", help="packaged pins document to write")
    pinner.add_argument("--manifest", type=Path, default=None, metavar="JSON",
                        help="also write the release-asset manifest here")

    adopter = sub.add_parser(
        "adopt", help="adopt the assets a draft release already carries, "
                      "verified by content contract (workflow re-runs)")
    adopter.add_argument("--release", required=True, metavar="TAG")
    adopter.add_argument("--held", required=True, type=Path, metavar="DIR",
                         help="the draft-asset capture: inventory.json "
                              "plus assets/<name>")
    adopter.add_argument("--bundles", required=True, type=Path,
                         metavar="DIR",
                         help="directory whose freshly built zips the "
                              "adopted ones replace (dist-bridges)")
    adopter.add_argument("--source-rev", required=True, metavar="COMMIT",
                         help="the release commit; adopted binaries must "
                              "carry its GPUWM_BRIDGE_SOURCE_REV stamp, "
                              "the same proof `pin` demands of fresh ones")

    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(args.release, args.platform, args.search, args.out)
    elif args.command == "adopt":
        adopt(args.release, args.held, args.bundles, args.source_rev)
    else:
        pin(args.release, args.bundle, args.out, args.manifest,
            args.source_rev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
