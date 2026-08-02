"""Build the prebuilt-bridge release assets and the pins the wheel carries.

Two steps, run in this order by the release workflow (and reproducible
by hand from a checkout):

``pack``
    On each target platform, after ``cargo build --release --locked`` in
    both vendored workspaces, collect the eight artifacts of
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

Nothing here invents a hash: every number written comes from hashing
bytes that exist on this disk at the moment it runs.  ``pin`` refuses a
bundle whose contents do not match the artifact set gpuwm expects for
that platform, so a half-built bundle cannot be pinned into a wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm import bridge_assets  # noqa: E402

#: Fixed member timestamp so two packs of identical bytes produce
#: identical archives (zip stores mtime per member).
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


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


def _pin_bundle(archive: Path, release: str) -> tuple[str, dict]:
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
        manifest: Path | None) -> Path:
    platforms: dict[str, dict] = {}
    for archive in archives:
        platform, record = _pin_bundle(archive, release)
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
        payload = {
            "schema": bridge_assets.BUNDLE_MANIFEST_SCHEMA,
            "release": release,
            "platforms": platforms,
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        print(f"build_bridge_bundle: wrote release manifest {manifest}")
    return out


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
        "--out", type=Path,
        default=REPO_ROOT / "gpuwm" / Path(bridge_assets.PINS_RESOURCE),
        metavar="JSON", help="packaged pins document to write")
    pinner.add_argument("--manifest", type=Path, default=None, metavar="JSON",
                        help="also write the release-asset manifest here")

    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(args.release, args.platform, args.search, args.out)
    else:
        pin(args.release, args.bundle, args.out, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
