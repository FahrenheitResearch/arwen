#!/usr/bin/env python3
"""Load every bundled library artifact on this host and resolve its ABI.

Why this exists
---------------
``tools/verify_release_artifacts.py --dry-run`` proves every document,
wheel, sdist and bundle assertion before a tag exists, and skips exactly
four things that need a live cut -- among them the host-staging leg that
loads each ``kind == "library"`` artifact.  That skip is where the 2.1.0
cut lost a number: the verifier's staging leg asked every library for
``gpuwm_preprocess_cpu_abi_version``, the vendored dealiasing cdylib
exports ``bw_abi_version``, and nothing local could run the leg, so the
refusal arrived in the prepare job with the tag already public.

A full local non-dry-run leg is not available before a tag: the verifier's
cut path needs the release wheel installed outside the checkout AND a
pinned bundle for *every* supported platform, and a bundle for the other
platform cannot be built on this host.  So this is the narrowest leg that
is nonetheless real -- the actual libraries this checkout builds, loaded
by the actual loader, answering the actual declared symbol -- and it is a
mandatory pre-tag step in ``RELEASE_CHECKLIST.md``.  It exercises the same
:func:`tools.verify_release_artifacts.probe_library_abi` and the same
:data:`gpuwm.bridge_assets.LIBRARY_ABI` the cut and the workflow use, so a
library added without a declared handshake, or declared with the wrong
symbol, is refused here rather than between the tag and PyPI.

Usage
-----
Build the libraries first (both crates build offline)::

    cargo build --release --locked --manifest-path tools/grib1_bridge/Cargo.toml
    cargo build --release --locked \
        --manifest-path tools/region_global_dealias/Cargo.toml
    python tools/probe_library_abi.py --receipt library-abi-probe.json

``--search DIR`` may be repeated to point at other build directories; the
three crate ``target/release`` directories are searched by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpuwm import bridge_assets  # noqa: E402
from tools.verify_release_artifacts import probe_library_abi  # noqa: E402

DEFAULT_SEARCH = (
    "tools/grib1_bridge/target/release",
    "tools/rustwx/target/release",
    "tools/region_global_dealias/target/release",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--search", type=Path, action="append", default=None,
        help="directory holding built artifacts; repeatable")
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    search = args.search or [args.repo_root / part for part in DEFAULT_SEARCH]

    platform = bridge_assets.host_platform()
    if platform is None:
        raise SystemExit(
            "no bundle is published for "
            f"{bridge_assets.host_platform_description()}, so there are "
            "no library artifacts to load here")

    libraries = [artifact for artifact in bridge_assets.BUNDLED_ARTIFACTS
                 if artifact.kind == "library"]
    if not libraries:
        raise SystemExit(
            "BUNDLED_ARTIFACTS declares no library artifact, which is not "
            "a state this probe should be reached in")

    probes: list[dict] = []
    for artifact in libraries:
        filename = bridge_assets.artifact_filename(artifact, platform)
        matches = [directory / filename for directory in search
                   if (directory / filename).is_file()]
        if len(matches) != 1:
            raise SystemExit(
                f"{artifact.name}: expected exactly one {filename} across "
                f"{[str(d) for d in search]}, found "
                f"{[str(m) for m in matches]}; build the crate first")
        probe = probe_library_abi(matches[0], artifact)
        probe["path"] = str(matches[0])
        probes.append(probe)
        print(f"{platform} {artifact.name} {probe['symbol']}() = "
              f"{probe['abi']} <- {matches[0]}")

    receipt = {
        "schema": "gpuwm-library-abi-probe-v1",
        "host_platform": platform,
        "libraries_probed": len(probes),
        "probes": probes,
    }
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
