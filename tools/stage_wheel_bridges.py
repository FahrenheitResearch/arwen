"""Stage the built Rust artifacts INTO the package, for a platform wheel.

``pip install gpuwm`` used to land a Python half with no compiled Rust at
all: every decoder, the fetch backbone, the renderer and the radar front
door were downloaded afterwards by ``gpuwm setup`` / ``gpuwm
fetch-bridges``.  That made the post-install step a failure surface on
the one command every user runs first, and it made "the decode runs in
Rust" unreachable on a bare install -- which is the whole point of doing
the decode in Rust.

This tool closes that by copying the artifacts
:data:`gpuwm.bridge_assets.BUNDLED_ARTIFACTS` declares out of the two
Cargo workspaces and into :func:`gpuwm.bridges.packaged_bridge_dir`
(``gpuwm/libexec/bridges``), which
``[tool.setuptools.package-data]`` sweeps into the wheel.  Run it
between ``cargo build --release`` and ``python -m build``; the resulting
wheel is platform-tagged by ``setup.py`` because it now carries native
code.

It is deliberately NOT a build step: it copies bytes that already exist
and refuses when they do not, so a wheel can never be built with a
silently missing artifact and ship a door that dies at first use.

    python tools/stage_wheel_bridges.py --platform win-x86_64
    python tools/stage_wheel_bridges.py --clean

``--dest DIR`` points the same table-driven copy at a directory other
than the package, which is what completes a HOME from a build tree::

    python tools/stage_wheel_bridges.py --dest ~/.gpuwm/bridges

That rung -- ``gpuwm.bridges.default_bridge_dir`` -- is the last one
every resolver searches, and it is the rung a wheel-less checkout user
fills.  Filling it by hand is how a home came to carry sixteen of the
declared artifacts, missing the NetCDF decoder and the NetCDF writer
behind the default wrfout engine; a hand copy has no declaration to
check itself against and this tool does.  Only the in-package
destination is emptied first (a wheel carrying two platforms is both
wrong and over PyPI's cap); any other destination keeps everything this
tool did not stage.

What is NOT staged, and why: the renderer's basemap shapefiles.  They
are 21.1 MB compressed, and there is no room for them.  They are also
DATA (Natural Earth coastlines and county polygons), not a processing
path: nothing about them is a Python reimplementation of something Rust
does, so leaving them on the ``gpuwm fetch-bridges`` route costs the
mandate nothing.  A render without them draws the fields and omits the
cartographic overlay.

Sizes, measured 2026-08-17 at the 2.5.0 integration tip, against
PyPI's 100 MB per-file cap read strictly as 100,000,000 B::

    pure wheel      gpuwm-2.5.0-py3-none-any.whl          91,926,618 B
    pure sdist      gpuwm-2.5.0.tar.gz                    95,245,042 B
    win-x86_64      gpuwm-2.5.0-py3-none-win_amd64.whl   108,695,109 B
    linux-x86_64    ...-manylinux_2_28_x86_64.whl        111,697,820 B

The pure pair is what ``.github/workflows/publish.yml`` ships (it runs
``--clean`` first) and it fits, with 8.07 MB of headroom on the wheel
and 4.75 MB on the sdist -- the sdist, not the wheel, is the binding
constraint of the published pair.  The win and linux numbers are this
Windows host and weather-node-1 respectively; the linux one is from the
2.5.0 Linux shakeout, everything else was measured together on one box.

BOTH PLATFORM WHEELS ARE OVER THE CAP AT THIS TIP -- by 8.70 MB and
11.70 MB.  The bridge payload has roughly doubled since these numbers
were last written (compressed, it is now 16.77 MB on Windows and 19.77
MB on Linux over the pure base), which is ``rw_netcdf``,
``netcdf_writer`` and ``region_global_dealias`` joining
:data:`gpuwm.bridge_assets.BUNDLED_ARTIFACTS`.  Whatever this docstring
claimed before, do not read a headroom number off it again without
re-measuring: the figures it carried into this tip (96.66 MB win, 98.35
MB linux, "1.65 MB of headroom on Linux") were 12-13 MB low and had the
sign inverted -- the wheels were over the cap while the doc said they
were under it, and the whole documented rationale for excluding the
basemaps rested on that dead number.

Nothing published is broken by this: the per-platform wheel is not a
shape PyPI receives today.  The remedy for making it one is the
companion split -- the payload moves into a companion distribution
rather than a decoder being dropped -- and until that lands, a platform
wheel built here is a local artifact, not an uploadable one.
``tools/verify_release_artifacts.py`` measures every distribution file
in the dist directory against the cap and refuses the ones a cut is
about to publish, so this cannot be discovered at upload time with the
tag already public.

67 MB of the pure base is physics tables (Thompson and RRTMGP).  If the
PURE pair ever runs out of room, the remedy is externalising another
table through ``gpuwm fetch-tables`` -- the route ``freezeH2O.dat`` and
``qr_acr_qg_V4.dat`` already take -- rather than dropping a decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpuwm import bridge_assets, bridges  # noqa: E402

#: Name of the record this tool writes beside the staged artifacts.  A
#: wheel that carries binaries must be able to SAY which platform they
#: are for and which commit built them, or a mis-tagged wheel is only
#: discoverable by running it.
MANIFEST_NAME = "BUNDLE.json"

#: Schema of that record.
MANIFEST_SCHEMA = "gpuwm-wheel-bridge-bundle-v1"

#: Where each workspace puts its release build, by crate-relative path.
_WORKSPACE_TARGET = {
    bridges.CRATE_RELATIVE: "target/release",
    bridges.RUSTWX_CRATE_RELATIVE: "target/release",
    "tools/region_global_dealias": "target/release",
    "tools/rw_wps": "target/release",
}


def purge_build_staging(root: Path | None = None) -> list[Path]:
    """Remove staged artifacts from setuptools' ``build/`` copy tree.

    setuptools copies package data into ``build/lib.<plat>-<pyver>/`` and
    never deletes files that have gone away, so building a second
    platform wheel in the same checkout silently ships the FIRST
    platform's binaries alongside the second's.  Measured, not theorised:
    the linux wheel came out at 105.99 MB carrying 22 artifacts -- every
    Linux binary plus every Windows ``.exe`` and ``.dll`` -- which is
    both wrong and over PyPI's 100 MB cap.

    So every staging operation purges that tree first.  Putting it here
    rather than in a workflow step is deliberate: a release runs this
    tool once per platform, and a cleanup that lives beside the thing it
    cleans cannot be left out of a new caller.
    """

    root = root or _REPO_ROOT
    removed: list[Path] = []
    build = root / "build"
    if not build.is_dir():
        return removed
    for staged in build.glob("lib*/gpuwm/libexec"):
        shutil.rmtree(staged, ignore_errors=True)
        removed.append(staged)
    return removed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(artifact: bridge_assets.BundledArtifact, platform: str) -> Path:
    """Where ``artifact``'s built bytes live in this checkout."""

    target = _WORKSPACE_TARGET.get(artifact.crate)
    if target is None:
        raise SystemExit(
            f"{artifact.name}: crate {artifact.crate!r} has no declared "
            f"release-build directory in stage_wheel_bridges._WORKSPACE_TARGET; "
            f"add it there beside the other workspaces")
    filename = bridge_assets.artifact_filename(artifact, platform)
    return _REPO_ROOT / artifact.crate / target / filename


def _is_packaged_destination(destination: Path) -> bool:
    """Is ``destination`` the in-package directory a wheel carries?

    The two destinations this tool writes have opposite emptying rules,
    and the difference is not cosmetic.  The PACKAGE directory must be
    emptied first: setuptools never removes package data that has gone
    away, so a second platform's staging would ship both platforms'
    binaries in one wheel -- measured at 105.99 MB, over PyPI's cap.
    Any OTHER destination is somebody's directory: a
    ``~/.gpuwm/bridges`` holds the renderer's ``assets/`` basemap tree
    and whatever kept-stale copies its owner parked there, none of which
    this tool staged and none of which it may delete.
    """

    try:
        return destination.resolve() == bridges.packaged_bridge_dir().resolve()
    except OSError:                              # pragma: no cover - exotic FS
        return False


def stage(platform: str, *, destination: Path | None = None) -> Path:
    """Copy every declared artifact into ``destination``, or refuse.

    Fail closed and fail COMPLETE: every missing artifact is collected
    and reported at once, because a build operator who has to re-run
    cargo wants the whole list, not the first name alphabetically.

    The declaration is the whole point.  A directory staged BY HAND has
    nothing to check itself against, which is how a
    ``~/.gpuwm/bridges`` came to hold sixteen of the declared
    artifacts -- missing ``rw_netcdf`` and ``netcdf_writer``, so every
    NetCDF source refused to decode and every history write would have
    refused, on a home that looked staged.  Driven from
    :data:`gpuwm.bridge_assets.BUNDLED_ARTIFACTS`, the artifact that
    joins the declaration next travels to every destination on the same
    commit.
    """

    destination = destination or bridges.packaged_bridge_dir()
    missing: list[str] = []
    plan: list[tuple[bridge_assets.BundledArtifact, Path, Path]] = []
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        origin = source_path(artifact, platform)
        if not origin.is_file():
            missing.append(f"  {artifact.name:24s} expected at {origin}")
            continue
        plan.append((artifact, origin,
                     destination / bridge_assets.artifact_filename(artifact, platform)))
    if missing:
        raise SystemExit(
            f"stage_wheel_bridges: {len(missing)} of "
            f"{len(bridge_assets.BUNDLED_ARTIFACTS)} declared artifacts are "
            f"not built for {platform}:\n" + "\n".join(missing) +
            "\n\nBuild them first:\n"
            "  cargo build --release --locked --offline "
            "--manifest-path tools/grib1_bridge/Cargo.toml\n"
            "  cargo build --release --locked --offline "
            "--manifest-path tools/rustwx/Cargo.toml\n"
            "  cargo build --release --locked --offline "
            "--manifest-path tools/region_global_dealias/Cargo.toml\n"
            "  cargo build --release --locked --offline "
            "--manifest-path tools/rw_wps/Cargo.toml")

    purge_build_staging()
    if _is_packaged_destination(destination) and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for artifact, origin, target in plan:
        shutil.copy2(origin, target)
        if artifact.kind == "executable" and os.name != "nt":
            target.chmod(target.stat().st_mode | 0o111)
        records.append({
            "artifact": artifact.name,
            "filename": target.name,
            "kind": artifact.kind,
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        })

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "platform": platform,
        "artifacts": records,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def stage_from_bundle(archive: Path, platform: str,
                      *, destination: Path | None = None) -> Path:
    """Stage from a packed release bundle instead of a build tree.

    The release builds each platform's binaries on its own target-native
    runner, then one job assembles the distributions.  That job cannot
    have a Windows cargo target directory and a Linux one, so the
    platform wheels are staged from the BUNDLES those runners uploaded --
    the identical bytes the release publishes and pins, which also means
    the wheel and the downloadable bundle can never disagree.

    Only the binaries are taken.  ``assets/`` members are the renderer's
    basemap shapefiles, deliberately not shipped in the wheel (see the
    module docstring).
    """

    destination = destination or bridges.packaged_bridge_dir()
    wanted = {
        bridge_assets.artifact_filename(artifact, platform): artifact
        for artifact in bridge_assets.BUNDLED_ARTIFACTS
    }
    purge_build_staging()
    if _is_packaged_destination(destination) and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        members = set(bundle.namelist())
        missing = sorted(name for name in wanted if name not in members)
        if missing:
            raise SystemExit(
                f"stage_wheel_bridges: {archive} does not carry "
                f"{len(missing)} declared artifact(s) for {platform}: "
                + ", ".join(missing))
        for filename, artifact in sorted(wanted.items()):
            payload = bundle.read(filename)
            target = destination / filename
            target.write_bytes(payload)
            if artifact.kind == "executable" and os.name != "nt":
                target.chmod(target.stat().st_mode | 0o111)
            records.append({
                "artifact": artifact.name,
                "filename": filename,
                "kind": artifact.kind,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "platform": platform,
        "artifacts": records,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def clean(destination: Path | None = None) -> None:
    purge_build_staging()
    destination = destination or bridges.packaged_bridge_dir()
    if destination.exists():
        shutil.rmtree(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage built Rust artifacts into the package for a "
                    "platform wheel.")
    parser.add_argument(
        "--platform", choices=bridge_assets.SUPPORTED_PLATFORMS,
        default=bridge_assets.host_platform(),
        help="platform key the artifacts were built for (default: this host)")
    parser.add_argument(
        "--from-bundle", type=Path, default=None, metavar="ARCHIVE",
        help="stage from a packed release bundle rather than a build tree "
             "(requires --platform)")
    parser.add_argument(
        "--dest", type=Path, default=None, metavar="DIR",
        help="stage into DIR instead of the in-package directory.  "
             f"{bridges.default_bridge_dir()} is the one gpuwm searches "
             "last, so staging there completes a checkout's own home "
             "from a build tree -- every declared artifact, driven from "
             "the same table the wheel and the release bundle use.  Only "
             "the in-package destination is emptied first; anywhere else "
             "keeps what this tool did not stage")
    parser.add_argument(
        "--clean", action="store_true",
        help="remove the staged directory and exit (restores a pure wheel)")
    args = parser.parse_args(argv)

    if args.clean:
        if args.dest is not None:
            raise SystemExit(
                "--clean removes its destination entirely, so it is "
                "restricted to the in-package directory this tool owns.  "
                f"{args.dest} may hold the renderer's assets/ tree and "
                "copies nobody here staged; delete those deliberately, "
                "not as a flag combination")
        clean()
        print(f"removed {bridges.packaged_bridge_dir()}")
        return 0
    if args.from_bundle is not None:
        if args.platform is None:
            raise SystemExit(
                "--from-bundle needs --platform: a bundle's platform is a "
                "property of the release that built it, not of this host")
        destination = stage_from_bundle(args.from_bundle, args.platform,
                                        destination=args.dest)
        total = sum(path.stat().st_size for path in destination.iterdir()
                    if path.is_file())
        print(f"staged {len(bridge_assets.BUNDLED_ARTIFACTS)} artifacts for "
              f"{args.platform} from {args.from_bundle} into {destination} "
              f"({total / 1e6:.2f} MB)")
        return 0
    if args.platform is None:
        raise SystemExit(
            "no bundle is published for this host "
            f"({bridge_assets.host_platform_description()}); pass --platform "
            f"explicitly to cross-stage, or build from source.  Known: "
            f"{', '.join(bridge_assets.SUPPORTED_PLATFORMS)}")
    destination = stage(args.platform, destination=args.dest)
    total = sum(path.stat().st_size for path in destination.iterdir()
                if path.is_file())
    print(f"staged {len(bridge_assets.BUNDLED_ARTIFACTS)} artifacts for "
          f"{args.platform} into {destination} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
