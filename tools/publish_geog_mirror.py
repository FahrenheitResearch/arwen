"""Stage and publish the WPS_GEOG mirror to Hugging Face.

The mirror (:data:`gpuwm.geog_assets.HF_MIRROR_REPO`) republishes every
archive in :data:`gpuwm.geog_assets.GEOG_ARCHIVES` **byte-for-byte** --
the nine standard WPS_GEOG tarballs and the Noah-MP ``soilgrids``
archive the mesh door requires -- so `gpuwm fetch-geog` gets CDN
bandwidth while one pin table covers both hosts.  A pinned archive
with no row here is exactly how the mirror lagged the pin table by a
full archive for nine days in 2026-08; ``prepare`` refuses the staging
directory until every pinned archive is present and verified.
This script is the only publication path:

    # 1. verify the staged tarballs against the packaged pins and
    #    generate MANIFEST.sha256 + README.md beside them (offline)
    python tools/publish_geog_mirror.py prepare --archives DIR

    # 2. one command, run by the repo owner with their own HF token
    #    (huggingface-cli login, or HF_TOKEN in the environment)
    python tools/publish_geog_mirror.py upload --archives DIR

    # 3. remove the held-out archives (licence_hold rows) an earlier
    #    snapshot carried, and rewrite README.md + MANIFEST.sha256 to
    #    what stays; dry run unless --apply
    python tools/publish_geog_mirror.py prune [--apply]

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
    GEOG_ARCHIVES as _ALL_ARCHIVES,
    HF_MIRROR_REPO,
    NCAR_BASE_URL,
    sha256_file,
)

#: What the mirror carries: the downloaded tarballs.  A
#: produced dataset has no tarball, and a pin with a licence hold (the
#: FAO-UNESCO soil tiles, whose terms forbid reproducing the data for third
#: parties; ISLSCP-I soiltemp_1deg, licence unclear) is fetched from NCAR
#: only and never republished here.
GEOG_ARCHIVES = tuple(a for a in _ALL_ARCHIVES
                      if a.producer is None and not a.licence_hold)

MANIFEST_NAME = "MANIFEST.sha256"
README_NAME = "README.md"

#: Soil provenance shared by the table rows and the notices (gpuwm NOTICE,
#: docs/public/DATA.md).  SoilGrids is CC BY 4.0 and needs its credit on
#: every page that redistributes the archive; FAO's terms for the Soil Map
#: of the World are stated as found, not assumed.
_FAO = ("FAO-UNESCO Digital Soil Map of the World, \"(C) FAO/UNESCO, "
        "1995 All rights reserved worldwide\" (DSMW notes v3.6); not "
        "permissive and not carried by this mirror")
_SOILGRIDS = ("ISRIC -- World Soil Information, SoilGrids250m (Hengl "
              "et al. 2017, PLOS ONE 12(2): e0169748), from whose 2017 "
              "texture classes NCAR generated the tiles; CC BY 4.0 (ISRIC's "
              "terms for SoilGrids products since 2019), attribution "
              "required")
_DUST_WPS = ("NCAR WPS geography distribution; exact pins captured "
             "2026-09-30; underlying source/licence not declared in "
             "this engine pin receipt")

_NATIVE_MPAS_WPS = ("NCAR WPS geography distribution, read by the native MPAS "
                    "static path; pins captured 2026-10-14; the download index "
                    "does not state a separate redistribution licence")

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
        "STATSGO inside CONUS (USDA NRCS, public domain) + " + _FAO),
    "soiltype_bot_30s": (
        "16-category bottom-layer soil texture",
        "STATSGO inside CONUS (USDA NRCS, public domain) + " + _FAO),
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
        "ISLSCP-I (ECMWF 1987-88 analyses); Permitted Use \"Unstated\" at "
        "CEDA; not carried by this mirror"),
    "soiltemp_era5_1deg": (
        "1-degree annual-mean deep-soil temperature, built by fetch-geog "
        "from ERA5 (not carried by this mirror)",
        "ERA5 1990-2019 climatology (WeatherBench 2); Licence to Use "
        "Copernicus Products"),
    "soilgrids": (
        "SoilGrids250m soil categories (the default top and bottom soil "
        "texture), Noah-MP soil composition and four-layer texture (seven "
        "WPS datasets under one parent)",
        _SOILGRIDS),
    "modis_landuse_20class_30s": (
        "Noah-modified 20-category IGBP land use without a lake class "
        "(lakes are water); the `gpuwm mesh` door's native MPAS static",
        "NASA MODIS (MCD12Q1-derived); NASA data are free and open"),
    "landuse_30s": (
        "USGS 24-category land use; the water mask of the `gpuwm mesh` "
        "door's native gravity-wave-drag statistics",
        "USGS global land cover characteristics, distributed for WPS by "
        "NCAR; U.S. public domain"),
    "lake_depth": (
        "30-arc-second lake depth for optional CLM lake physics",
        "distributed for WPS by NCAR; the download index does not state "
        "an ultimate source or a separate redistribution licence"),
    "orogwd3_10m": (
        "10-arc-minute GSL terrain-drag statistics in twenty datasets",
        "NCAR WPS geography distribution for gwd_opt=3; the download "
        "index does not state a separate redistribution licence"),
    "bnu_soiltype_top": (
        "alternative 30-arc-second, 16-category top-layer soil texture",
        "NCAR WPS geography distribution; the download index does not "
        "state a separate redistribution licence"),
    "bnu_soiltype_bot": (
        "alternative 30-arc-second, 16-category bottom-layer soil texture",
        "NCAR WPS geography distribution; the download index does not "
        "state a separate redistribution licence"),
    "topo_30s": (
        "30-arc-second terrain elevation (native MPAS GTOPO30 option)",
        _NATIVE_MPAS_WPS),
    "modis_landuse_20class_15s": (
        "15-arc-second Noah-modified IGBP land use (native MPAS "
        "MODIFIED_IGBP_MODIS_NOAH_15s option; WPS geog_data_res "
        "modis_15s)",
        _NATIVE_MPAS_WPS),
    "lai_modis_30s": (
        "30-arc-second monthly MODIS leaf area index (WPS geog_data_res "
        "modis_lai)",
        "NASA MODIS, distributed for WPS by NCAR; NASA data are free and "
        "open"),
    "greenfrac": (
        "monthly green-vegetation fraction (native MPAS NCEP option)",
        _NATIVE_MPAS_WPS),
    "albedo_ncep": (
        "monthly surface albedo (native MPAS NCEP option)",
        _NATIVE_MPAS_WPS),
    "maxsnowalb": (
        "maximum snow albedo (native MPAS NCEP option)",
        _NATIVE_MPAS_WPS),
    "erod": (
        "erodible surface fractions for sand, silt and clay dust classes",
        _DUST_WPS),
    "clayfrac_5m": (
        "5-arc-minute soil clay fraction for optional dust emissions",
        _DUST_WPS),
    "sandfrac_5m": (
        "5-arc-minute soil sand fraction for optional dust emissions",
        _DUST_WPS),
    "orogwd_10m": (
        "10-arc-minute subgrid orography statistics (VAR, CON, OA1-4, OL1-4) "
        "for orographic gravity-wave drag",
        "NCAR WPS geography distribution; the download index does not "
        "state a separate redistribution licence"),
    "varsso_10m": (
        "10-arc-minute small-scale orography variance (VAR_SSO)",
        "NCAR WPS geography distribution; the download index does not "
        "state a separate redistribution licence"),
}


#: Pins whose licence keeps them off the mirror (FAO-UNESCO soil tiles,
#: ISLSCP-I soiltemp_1deg).  Snapshots before 2026-10-21 carried them;
#: ``prune`` deletes exactly these filenames and nothing else.
HELD_ARCHIVES = tuple(a for a in _ALL_ARCHIVES
                      if a.licence_hold and a.filename)


def _readme_text(archives=None) -> str:
    archives = GEOG_ARCHIVES if archives is None else tuple(archives)
    rows = "\n".join(
        f"| `{a.filename}` | {a.archive_bytes:,} | `{a.archive_sha256}` |"
        for a in archives)
    prov = "\n".join(
        f"| `{a.dataset}` | {_PROVENANCE[a.dataset][0]} | "
        f"{_PROVENANCE[a.dataset][1]} |"
        for a in archives)
    held = ", ".join(f"`{a.dataset}`" for a in HELD_ARCHIVES)
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

A **byte-for-byte mirror** of the NCAR WPS v4 geographical (static)
dataset tarballs that [ArWen](https://github.com/FahrenheitResearch/arwen)'s
static builders read by default, including the SoilGrids-derived
`soilgrids` archive -- republished here for CDN bandwidth.  The authoritative source
is NCAR/UCAR's WPS geographical data distribution:
<{NCAR_BASE_URL.rsplit('/', 2)[0]}/users/download/get_sources_wps_geog.html>
(tarballs under `{NCAR_BASE_URL}/`).  Nothing is modified: each file
here is the NCAR tarball as downloaded on 2026-07-29 (`soilgrids.tar.bz2`
on 2026-09-01, matching the pin gpuwm computed on 2026-08-23), and the SHA-256
pins below are enforced by `gpuwm fetch-geog` for downloads from this
mirror and from NCAR alike.

## Fetch

```bash
pip install gpuwm
gpuwm fetch-geog          # this mirror, verified against the pins
gpuwm fetch-geog --source ncar   # upstream instead
```

Or plain HTTPS: `.../resolve/main/<filename>` and untar the
tarballs into one directory (`soilgrids.tar.bz2` carries its seven
datasets under one parent; `gpuwm fetch-geog` lays them out).  `MANIFEST.sha256` is `sha256sum -c`
compatible.

## Not on this mirror

{held} are not republished here: the FAO-UNESCO soil tiles are
"All rights reserved" and ISLSCP-I states no terms for
`soiltemp_1deg`.  `gpuwm fetch-geog` fetches them from NCAR when they
are named.  If your gpuwm reports HTTP 404 from this mirror for one of
them, run `gpuwm fetch-geog --source ncar` or upgrade gpuwm.

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
GMTED2010; NASA for the MODIS-derived fields; ISRIC for SoilGrids250m).

`soilgrids.tar.bz2` is derived from ISRIC SoilGrids250m and is
redistributed here under CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>), which requires this
credit: *Soil data: SoilGrids250m, ISRIC -- World Soil Information
(Hengl T. et al. 2017, "SoilGrids250m: Global gridded soil information
based on machine learning", PLOS ONE 12(2): e0169748,
doi:10.1371/journal.pone.0169748); WPS tiles generated by NCAR; not
modified.*

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
        commit_message="WPS_GEOG mirror snapshot: every archive pinned in "
                       "gpuwm.geog_assets, NCAR bytes verified by prepare")
    print(f"upload: published to https://huggingface.co/datasets/{repo}")
    print("upload: smoke-check the resolve route with:\n"
          "  GPUWM_NETWORK_TESTS=1 python -m pytest "
          "tests/test_fetch_geog.py -m network -q")
    return 0


def _lfs_sha256(entry) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def prune(repo: str, apply: bool) -> int:
    """Delete the held-out archives from the live mirror, in one commit.

    Touches only filenames in :data:`HELD_ARCHIVES` whose live bytes
    match their pin, plus README.md and MANIFEST.sha256, which are
    rewritten to list exactly the pinned archives that stay.  Any other
    file is left alone and reported.  Refuses (changes nothing) when a
    held file's live sha256 is not its pin, or a staying archive's live
    bytes do not match its pin.
    """
    try:
        from huggingface_hub import (CommitOperationAdd,
                                     CommitOperationDelete, HfApi)
    except ImportError:
        print("prune: pip install 'gpuwm[publish]'  (and authenticate "
              "with `huggingface-cli login` or HF_TOKEN)")
        return 2
    api = HfApi()
    head = api.dataset_info(repo).sha
    live = {e.path: e for e in api.list_repo_tree(
        repo, repo_type="dataset", recursive=True, revision=head)
        if hasattr(e, "size")}
    held_by_name = {a.filename: a for a in HELD_ARCHIVES}
    pinned_by_name = {a.filename: a for a in GEOG_ARCHIVES}
    service = {MANIFEST_NAME, README_NAME, ".gitattributes"}
    failures, doomed, kept, other = [], [], [], []
    for name, entry in sorted(live.items()):
        sha = _lfs_sha256(entry)
        if name in held_by_name:
            pin = held_by_name[name]
            if sha != pin.archive_sha256 or entry.size != pin.archive_bytes:
                failures.append(f"{name}: live {entry.size:,} B sha256 "
                                f"{sha} is not the pin; not deleting it")
            else:
                doomed.append((name, entry.size, sha))
        elif name in pinned_by_name:
            pin = pinned_by_name[name]
            if sha != pin.archive_sha256 or entry.size != pin.archive_bytes:
                failures.append(f"{name}: live bytes differ from the pin")
            else:
                kept.append(pin)
        elif name not in service:
            other.append(name)
    print(f"prune: {repo} at {head}")
    for name, size, sha in doomed:
        print(f"  delete  {name}  {size:,} B  sha256 {sha}")
    for pin in kept:
        print(f"  keep    {pin.filename}  {pin.archive_bytes:,} B")
    for name in other:
        print(f"  ignore  {name}  (not in the pin table; left alone)")
    absent = [a.filename for a in GEOG_ARCHIVES if a.filename not in live]
    if absent:
        print(f"  (pins not on the mirror, unchanged: {', '.join(absent)})")
    if failures:
        print("prune: REFUSED, nothing changed:")
        for line in failures:
            print(f"  {line}")
        return 2
    manifest = "".join(f"{a.archive_sha256}  {a.filename}\n" for a in kept)
    readme = _readme_text(kept)
    live_readme = Path(api.hf_hub_download(
        repo, README_NAME, repo_type="dataset", revision=head)).read_text(
            encoding="utf-8")
    if not doomed and live_readme == readme:
        print("prune: nothing to do (no held archive on the mirror; "
              "README current)")
        return 0
    if not apply:
        print(f"prune: dry run; {len(doomed)} file(s) would be deleted "
              f"and {README_NAME} + {MANIFEST_NAME} rewritten for "
              f"{len(kept)} archive(s).  Re-run with --apply.")
        return 0
    ops = [CommitOperationDelete(path_in_repo=name) for name, _, _ in doomed]
    ops += [CommitOperationAdd(path_in_repo=MANIFEST_NAME,
                               path_or_fileobj=manifest.encode("utf-8")),
            CommitOperationAdd(path_in_repo=README_NAME,
                               path_or_fileobj=readme.encode("utf-8"))]
    info = api.create_commit(
        repo, repo_type="dataset", operations=ops, parent_commit=head,
        commit_message="Remove the licence-held WPS_GEOG archives "
                       "(FAO-UNESCO soil tiles, ISLSCP-I soiltemp_1deg); "
                       "fetch-geog reads them from NCAR when named")
    after = {e.path for e in api.list_repo_tree(
        repo, repo_type="dataset", recursive=True, revision=info.oid)}
    still = [n for n, _, _ in doomed if n in after]
    lost = [a.filename for a in kept if a.filename not in after]
    print(f"prune: committed {info.oid} ({info.commit_url})")
    if still or lost:
        print(f"prune: VERIFY FAILED: still present {still}, lost {lost}")
        return 2
    print(f"prune: verified: {len(doomed)} deleted, {len(kept)} kept")
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
    p = sub.add_parser("prune")
    p.add_argument("--repo", default=HF_MIRROR_REPO,
                   help=f"dataset repo id (default {HF_MIRROR_REPO})")
    p.add_argument("--apply", action="store_true",
                   help="commit the deletion (default: dry run)")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.archives)
    if args.command == "prune":
        return prune(args.repo, args.apply)
    return upload(args.archives, args.repo, args.private)


if __name__ == "__main__":
    sys.exit(main())
