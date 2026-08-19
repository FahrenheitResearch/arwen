"""Download a small real 20CRv3 window for ``gpuwm prep --source 20crv3-cf``.

NOAA PSL publishes the 20th Century Reanalysis version 3 as one NetCDF
file per variable per year, on a global 1-degree grid at three-hourly
analysis times -- about 4.6 GB per variable-year (MEASURED 2026-08-16 on
``prsSI/air.1974.nc``).  Nobody wants 60 GB to run one 3-hour case, and
the THREDDS NetCDF Subset Service will cut a spatial and temporal window
out of each file server-side and hand back real NetCDF.  That is what
this does, once per variable the packaged profile needs.

**The distribution is the ENSEMBLE MEAN analysis.**  Every variable in it
carries ``statistic = "Ensemble Mean"``; the 80 members live in the
every-member GRIB2 archive, which is the ``--source 20crv3`` route.  The
packaged NetCDF mapping binds that attribute, so a member file cannot be
read through this profile by accident.

**PSL publishes no orography and no land mask for 20CRv3** (MEASURED
2026-08-16: no ``hgt.sfc``, ``land`` or ``lsmask`` file exists anywhere
under ``Datasets/20thC_ReanV3/``).  Both are recovered from 20CRv3's own
published fields by ``tools/build_pressure_level_invariant_supplement.py``,
which this script runs for you and which writes a receipt naming the
method and the divergence.

The variable table below is the only 20CRv3-specific thing here.  It is a
table: a different reanalysis on the same server is a different table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


#: THREDDS NetCDF Subset Service, PSL's 20CRv3 collection.
NCSS_ROOT = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/20thC_ReanV3"

#: ``(directory, file stem, variable, local name)`` for every input the
#: packaged ``20crv3-netcdf-v1`` profile reads, plus the two the
#: invariant-supplement builder needs.
#:
#: One row per file because PSL publishes one variable per file, and the
#: mapping resolves each by NAME PLUS the ``level_desc`` attribute -- so
#: ``air`` on pressure levels and ``air`` at 2 m are two rows here and two
#: selectors there, never a guess about which file is which.
SOURCE_FILES = (
    ("prsSI", "air", "air", "air.nc"),
    ("prsSI", "hgt", "hgt", "hgt.nc"),
    ("prsSI", "shum", "shum", "shum.nc"),
    ("prsSI", "uwnd", "uwnd", "uwnd.nc"),
    ("prsSI", "vwnd", "vwnd", "vwnd.nc"),
    ("sfcSI", "pres.sfc", "pres", "pres.sfc.nc"),
    ("sfcSI", "skt", "skt", "skt.nc"),
    ("2mSI", "air.2m", "air", "air.2m.nc"),
    ("2mSI", "shum.2m", "shum", "shum.2m.nc"),
    ("10mSI", "uwnd.10m", "uwnd", "uwnd.10m.nc"),
    ("10mSI", "vwnd.10m", "vwnd", "vwnd.10m.nc"),
    ("subsfcSI", "tsoil", "tsoil", "tsoil.nc"),
    ("subsfcSI", "soilw", "soilw", "soilw.nc"),
    # Not read by the mapping: the land-only field the supplement builder
    # takes its land mask from.
    ("sfc_paramsSI", "wilt", "wilt", "wilt.nc"),
)

#: Which of the above the mapping actually decodes.  ``wilt`` is an input
#: to the supplement, not to the mapping, so it is not passed to `prep`.
SUPPLEMENT_ONLY = frozenset({"wilt.nc"})

#: 20CRv3's analysis cadence.  Not a flag: it is what the archive is.
CADENCE = timedelta(hours=3)

#: PSL's THREDDS opens a multi-gigabyte file per request and answers 502
#: while it is busy.  Retry rather than fail a fourteen-file fetch on one.
_ATTEMPTS = 5
_BACKOFF_S = 8.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_url(
    directory: str, stem: str, variable: str, year: int, *,
    north: float, south: float, west: float, east: float,
    start: datetime, end: datetime,
) -> str:
    query = urlencode({
        "var": variable,
        "north": north, "south": south, "west": west, "east": east,
        "horizStride": 1,
        "time_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accept": "netcdf",
    })
    return f"{NCSS_ROOT}/{directory}/{stem}.{year}.nc?{query}"


def _download(url: str, destination: Path) -> int:
    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            with urlopen(url, timeout=900) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last = error
            time.sleep(_BACKOFF_S * (attempt + 1))
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return len(payload)
    raise SystemExit(f"{url}\n  failed after {_ATTEMPTS} attempts: {last}")


def fetch(arguments: argparse.Namespace) -> dict[str, object]:
    start = datetime.strptime(arguments.start, "%Y-%m-%dT%H:%M:%S")
    if start.hour % 3 or start.minute or start.second:
        raise SystemExit(
            f"{arguments.start} is not a 20CRv3 analysis time; the archive "
            "publishes 00/03/06/09/12/15/18/21Z")
    if arguments.frames < 2:
        raise SystemExit(
            "--frames must be at least 2: a limited-area run needs lateral "
            "boundaries, so it needs a second valid time")
    end = start + CADENCE * (arguments.frames - 1)
    if end.year != start.year:
        raise SystemExit(
            "the window crosses a year boundary and PSL files are per-year; "
            "shorten --frames or start later")

    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory, stem, variable, name in SOURCE_FILES:
        destination = arguments.output / name
        if destination.is_file() and not arguments.refetch:
            size = destination.stat().st_size
        else:
            size = _download(
                subset_url(
                    directory, stem, variable, start.year,
                    north=arguments.north, south=arguments.south,
                    west=arguments.west, east=arguments.east,
                    start=start, end=end,
                ),
                destination,
            )
        rows.append({
            "name": name, "directory": directory, "variable": variable,
            "bytes": size, "sha256": _sha256(destination),
            "read_by_the_mapping": name not in SUPPLEMENT_ONLY,
        })
        print(f"  {name:<16} {size:>10,} bytes", file=sys.stderr)

    supplement = arguments.output / "invariant.nc"
    receipt = arguments.output / "invariant.provenance.json"
    if supplement.is_file() and arguments.refetch:
        supplement.unlink()
    if not supplement.is_file():
        builder = Path(__file__).with_name(
            "build_pressure_level_invariant_supplement.py")
        completed = subprocess.run(
            [sys.executable, str(builder),
             "--height", str(arguments.output / "hgt.nc"),
             "--height-variable", "hgt",
             "--surface-pressure", str(arguments.output / "pres.sfc.nc"),
             "--surface-pressure-variable", "pres",
             "--land-source", str(arguments.output / "wilt.nc"),
             "--land-source-variable", "wilt",
             "--output", str(supplement),
             "--receipt", str(receipt)],
            check=False,
        )
        if completed.returncode:
            raise SystemExit(
                "the invariant supplement could not be built; 20CRv3 "
                "publishes no orography or land mask, so preparation cannot "
                "proceed without it")

    inputs = [
        arguments.output / row["name"]
        for row in rows if row["read_by_the_mapping"]
    ] + [supplement]
    command = ["gpuwm", "prep", "--source", "20crv3-cf"]
    for path in inputs:
        command.extend(["--input", str(path)])
    command.extend([
        "--supplement", str(supplement),
        "--author-input-manifest", str(arguments.output / "inputs.json"),
        "--wps-namelist", "<your namelist.wps>",
        "--geog-root", "<your WPS_GEOG>",
        "--experiment-config", "<your experiment.toml>",
        "--output-root", "<a new directory>",
    ])
    print("\nnext: prepare, with the window already bound:\n", file=sys.stderr)
    print("  " + " \\\n    ".join(command) + "\n", file=sys.stderr)
    return {
        "schema": "gpuwm-20crv3-netcdf-subset-v1",
        "source": "NOAA-CIRES-DOE 20CRv3, NOAA PSL sub-daily NetCDF (SI)",
        "statistic": "Ensemble Mean",
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "frames": arguments.frames,
            "cadence_seconds": int(CADENCE.total_seconds()),
            "north": arguments.north, "south": arguments.south,
            "west": arguments.west, "east": arguments.east,
        },
        "files": rows,
        "supplement": {
            "path": str(supplement),
            "sha256": _sha256(supplement),
            "provenance": str(receipt),
        },
        "total_bytes": sum(int(row["bytes"]) for row in rows)
        + supplement.stat().st_size,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.download_20crv3_native_subset",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--start", required=True,
                        help="first analysis time, YYYY-MM-DDTHH:MM:SS "
                             "(20CRv3 publishes 00/03/.../21Z)")
    parser.add_argument("--frames", type=int, default=2,
                        help="how many three-hourly analyses (>= 2)")
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--west", type=float, required=True)
    parser.add_argument("--east", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="directory to write the subset into")
    parser.add_argument("--refetch", action="store_true",
                        help="re-download files that are already present")
    return parser


def main(argv: list[str] | None = None) -> int:
    receipt = fetch(_parser().parse_args(argv))
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
