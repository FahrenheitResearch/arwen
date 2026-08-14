"""Stage and publish the WPS_GEOG mirror to Hugging Face.

The mirror (:data:`gpuwm.geog_assets.HF_MIRROR_REPO`) republishes the
nine NCAR per-dataset WPS_GEOG tarballs **byte-for-byte** so `gpuwm
fetch-geog` gets CDN bandwidth while one pin table covers both hosts.
This script is the only publication path:

    # 1. verify the staged tarballs against the packaged pins and
    #    generate MANIFEST.sha256 + README.md beside them (offline)
    python tools/publish_geog_mirror.py prepare --archives DIR

    # 2. one command, run by the repo owner with their own HF token
    #    (huggingface-cli login, or HF_TOKEN in the environment)
    python tools/publish_geog_mirror.py upload --archives DIR

``prepare`` refuses any tarball whose bytes do not match the pins in
:mod:`gpuwm.geog_assets` -- the mirror must never diverge from what
fetch-geog verifies.  ``upload`` needs ``pip install 'gpuwm[publish]'``
and write access to the repo; uploads are chunked LFS commits and a
re-run resumes/no-ops files already present with the same content.
``[publish]`` is maintainer-only and deliberately outside ``[all]``:
nobody without write credentials on the dataset repository can use it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gpuwm.geog_assets import (  # noqa: E402
    GEOG_ARCHIVES,
    HF_MIRROR_REPO,
    NCAR_BASE_URL,
    sha256_file,
)

MANIFEST_NAME = "MANIFEST.sha256"
README_NAME = "README.md"

_PROVENANCE = {
    "topo_gmted2010_30s": (
        "30-arc-second terrain elevation",
        "USGS/NGA GMTED2010 (Danielson & Gesch 2011, USGS OFR "
        "2011-1073); U.S. public domain"),
    "modis_landuse_20class_30s_with_lakes": (
        "Noah-modified 20-category IGBP land use with inland lakes",
        "NASA MODIS (MCD12Q1-derived); NASA data are free and open"),
    "soiltype_top_30s": (
        "16-category top-layer soil texture",
        "hybrid STATSGO (USDA, public domain) + FAO Digital Soil Map "
        "of the World"),
    "soiltype_bot_30s": (
        "16-category bottom-layer soil texture",
        "hybrid STATSGO (USDA, public domain) + FAO Digital Soil Map "
        "of the World"),
    "greenfrac_fpar_modis": (
        "monthly green-vegetation-fraction climatology",
        "NASA MODIS FPAR; NASA data are free and open"),
    "lai_modis_10m": (
        "monthly leaf-area-index climatology (10 arc-min)",
        "NASA MODIS; NASA data are free and open"),
    "albedo_modis": (
        "monthly surface albedo climatology",
        "NASA MODIS; NASA data are free and open"),
    "maxsnowalb_modis": (
        "maximum snow albedo",
        "MODIS-derived (Barlage et al. 2005); NASA data are free and "
        "open"),
    "soiltemp_1deg": (
        "1-degree annual-mean deep-soil temperature climatology",
        "distributed with WPS by NCAR; ultimate source not stated on "
        "NCAR's pages"),
}


def _readme_text() -> str:
    rows = "\n".join(
        f"| `{a.filename}` | {a.archive_bytes:,} | `{a.archive_sha256}` |"
        for a in GEOG_ARCHIVES)
    prov = "\n".join(
        f"| `{a.dataset}` | {_PROVENANCE[a.dataset][0]} | "
        f"{_PROVENANCE[a.dataset][1]} |"
        for a in GEOG_ARCHIVES)
    return f"""\
---
pretty_name: WPS_GEOG static geography mirror (ArWen)
license: other
license_name: mixed-public-domain-and-open-data
license_link: >-
  https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html
tags:
  - weather
  - WRF
  - WPS
  - static-geography
---

# WPS_GEOG static geography mirror (ArWen)

A **byte-for-byte mirror** of the nine standard NCAR WPS v4
geographical (static) dataset tarballs that
[ArWen](https://github.com/FahrenheitResearch/arwen)'s static builder
reads -- republished here for CDN bandwidth.  The authoritative source
is NCAR/UCAR's WPS geographical data distribution:
<{NCAR_BASE_URL.rsplit('/', 2)[0]}/users/download/get_sources_wps_geog.html>
(tarballs under `{NCAR_BASE_URL}/`).  Nothing is modified: each file
here is the NCAR tarball as downloaded on 2026-07-29, and the SHA-256
pins below are enforced by `gpuwm fetch-geog` for downloads from this
mirror and from NCAR alike.

## Fetch

```bash
pip install gpuwm
gpuwm fetch-geog          # this mirror, verified against the pins
gpuwm fetch-geog --source ncar   # upstream instead
```

Or plain HTTPS: `.../resolve/main/<filename>` and untar the nine
tarballs into one directory.  `MANIFEST.sha256` is `sha256sum -c`
compatible.

## Pinned contents

| file | bytes | sha256 |
|---|---|---|
{rows}

## Provenance and attribution

These datasets are assembled and distributed publicly by NCAR/UCAR for
the WRF ecosystem (the NCAR download page attaches no license or
citation text).  Underlying sources:

| dataset | contents | ultimate source |
|---|---|---|
{prov}

If you publish work built on these fields, credit NCAR/UCAR's WPS
geographical data distribution and the underlying providers (USGS for
GMTED2010; NASA for the MODIS-derived fields).

## Update policy

This mirror tracks the pins packaged in `gpuwm.geog_assets` -- it is
re-snapshotted only together with a gpuwm release that updates those
pins, never silently.
"""


def prepare(archives: Path) -> int:
    failures = []
    lines = []
    for archive in GEOG_ARCHIVES:
        path = archives / archive.filename
        if not path.is_file():
            failures.append(f"{archive.filename}: missing from {archives}")
            continue
        size = path.stat().st_size
        if size != archive.archive_bytes:
            failures.append(
                f"{archive.filename}: {size:,} B, pin says "
                f"{archive.archive_bytes:,} B")
            continue
        digest = sha256_file(path)
        if digest != archive.archive_sha256:
            failures.append(
                f"{archive.filename}: sha256 {digest} does not match the "
                f"pin {archive.archive_sha256}")
            continue
        lines.append(f"{digest}  {archive.filename}")
        print(f"prepare: {archive.filename} verified against the pin")
    if failures:
        print("prepare: REFUSED -- the mirror must match the packaged "
              "pins exactly:")
        for line in failures:
            print(f"  {line}")
        return 2
    (archives / MANIFEST_NAME).write_text("\n".join(lines) + "\n",
                                          encoding="utf-8", newline="\n")
    (archives / README_NAME).write_text(_readme_text(),
                                        encoding="utf-8", newline="\n")
    print(f"prepare: wrote {archives / MANIFEST_NAME}")
    print(f"prepare: wrote {archives / README_NAME}")
    print("prepare: staging complete; publish with:\n"
          f"  python tools/publish_geog_mirror.py upload "
          f"--archives {archives}")
    return 0


def upload(archives: Path, repo: str, private: bool) -> int:
    for name in ([a.filename for a in GEOG_ARCHIVES]
                 + [MANIFEST_NAME, README_NAME]):
        if not (archives / name).is_file():
            print(f"upload: {name} missing from {archives}; run "
                  "`prepare` first")
            return 2
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("upload: pip install 'gpuwm[publish]'  (and authenticate "
              "with `huggingface-cli login` or HF_TOKEN)")
        return 2
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=private,
                    exist_ok=True)
    api.upload_folder(
        repo_id=repo, repo_type="dataset", folder_path=str(archives),
        allow_patterns=[a.filename for a in GEOG_ARCHIVES]
        + [MANIFEST_NAME, README_NAME],
        commit_message="WPS_GEOG mirror snapshot (NCAR bytes of "
                       "2026-07-29, pinned in gpuwm.geog_assets)")
    print(f"upload: published to https://huggingface.co/datasets/{repo}")
    print("upload: smoke-check the resolve route with:\n"
          "  GPUWM_NETWORK_TESTS=1 python -m pytest "
          "tests/test_fetch_geog.py -m network -q")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--archives", type=Path, required=True,
                       metavar="DIR",
                       help="directory holding the nine pinned tarballs")
        if name == "upload":
            p.add_argument("--repo", default=HF_MIRROR_REPO,
                           help=f"dataset repo id (default {HF_MIRROR_REPO})")
            p.add_argument("--private", action="store_true",
                           help="create the repo private (flip public "
                                "after inspection)")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.archives)
    return upload(args.archives, args.repo, args.private)


if __name__ == "__main__":
    sys.exit(main())
