"""Download a manifest-bound GFS subset for ``rw-wps --source gfs``.

The NOMADS selector is deliberately explicit.  In particular, GFS soil
temperature is published as ``TSOIL`` rather than the generic ``TMP`` name
used for atmospheric and surface temperature.  Omitting it creates a valid
GRIB2 file that the fail-closed native bridge correctly rejects.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PRESSURE_LEVELS_HPA = (
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
    650, 700, 750, 800, 850, 900, 925, 950, 975, 1000,
)
NOMADS_LEVELS = (
    "lev_surface",
    "lev_2_m_above_ground",
    "lev_10_m_above_ground",
    "lev_0-0.1_m_below_ground",
    "lev_0.1-0.4_m_below_ground",
    "lev_0.4-1_m_below_ground",
    "lev_1-2_m_below_ground",
)
NOMADS_VARIABLES = (
    "var_HGT", "var_TMP", "var_RH", "var_UGRD", "var_VGRD",
    "var_PRES", "var_WEASD", "var_SNOD", "var_LAND", "var_ICEC",
    "var_TSOIL", "var_SOILW",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cycle(raw: str) -> datetime:
    value = datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S")
    if value.hour not in {0, 6, 12, 18} or value.minute or value.second:
        raise ValueError("GFS cycle must be exactly 00/06/12/18 UTC")
    return value


def _forecast_hours(raw: str) -> tuple[int, ...]:
    try:
        hours = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("forecast hours must be comma-separated integers") from error
    if len(hours) < 2 or hours[0] != 0 or any(
            later <= earlier for earlier, later in zip(hours, hours[1:])):
        raise ValueError("forecast hours must contain increasing f000 plus a later time")
    steps = {later - earlier for earlier, later in zip(hours, hours[1:])}
    if len(steps) != 1 or steps.pop() not in {1, 3}:
        raise ValueError("forecast-hour cadence must be uniformly 1 or 3 hours")
    if hours[-1] > 384:
        raise ValueError("forecast hours exceed the supported GFS f384 horizon")
    return hours


#: The NCEP models this selector serves, keyed by gpuwm source name.
#:
#: GDAS is the GFS assimilation cycle's own output: the *same*
#: ``pgrb2.0p25`` container -- 0.25-degree regular lat/lon, the same
#: variable and level codes, the same 124-record census under this
#: selector, the same originating centre and table versions -- published
#: under a different directory and grib-filter script.  Verified against
#: a live cycle: 124 records, scan 0x40, DRT 5.0, shape 6, centre 7,
#: master table 2, local table 1, PDT 4.0, generating process 81.  Only
#: the naming and the forecast horizon differ, so the certified GFS
#: mapping, bridge and front door serve it unchanged.
NOMADS_MODELS = {
    "gfs": {"script": "filter_gfs_0p25.pl", "prefix": "gfs"},
    "gdas": {"script": "filter_gdas_0p25.pl", "prefix": "gdas"},
}


def nomads_query(
        cycle: datetime, forecast_hour: int, *, left_lon: float,
        right_lon: float, bottom_lat: float, top_lat: float,
        model: str = "gfs",
) -> str:
    if not 0.0 <= left_lon < right_lon <= 360.0:
        raise ValueError("GFS subset longitudes must increase within [0, 360]")
    if not -90.0 <= bottom_lat < top_lat <= 90.0:
        raise ValueError("GFS subset latitudes must increase within [-90, 90]")
    try:
        spec = NOMADS_MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown NOMADS model {model!r}; expected one of "
            f"{sorted(NOMADS_MODELS)}") from None
    prefix = spec["prefix"]
    hour = cycle.strftime("%H")
    parameters: list[tuple[str, str]] = [
        ("file", f"{prefix}.t{hour}z.pgrb2.0p25.f{forecast_hour:03d}"),
        ("subregion", ""),
        ("leftlon", format(left_lon, "g")),
        ("rightlon", format(right_lon, "g")),
        ("toplat", format(top_lat, "g")),
        ("bottomlat", format(bottom_lat, "g")),
    ]
    parameters.extend(
        (f"lev_{level}_mb", "on") for level in PRESSURE_LEVELS_HPA)
    parameters.extend((name, "on") for name in NOMADS_LEVELS)
    parameters.extend((name, "on") for name in NOMADS_VARIABLES)
    parameters.append((
        "dir", f"/{prefix}.{cycle:%Y%m%d}/{hour}/atmos",
    ))
    return ("https://nomads.ncep.noaa.gov/cgi-bin/" + spec["script"] + "?"
            + urlencode(parameters))


def _download(url: str, destination: Path, *, retries: int = 5) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "rw-wps-gfs-fetch/1"})
            with urlopen(request, timeout=300) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            with partial.open("rb") as stream:
                signature = stream.read(4)
                stream.seek(-4, os.SEEK_END)
                terminator = stream.read(4)
            if (signature != b"GRIB" or terminator != b"7777"
                    or partial.stat().st_size <= 1024):
                raise RuntimeError("NOMADS response is not a complete GRIB2 stream")
            os.replace(partial, destination)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def _write_manifest(
        output: Path, *, cycle: datetime, records: tuple[tuple[int, Path], ...],
        series: Path, bridge: Path, wps_namelist: Path,
        experiment_config: Path,
) -> tuple[Path, str]:
    roles = {
        "series": series,
        "bridge": bridge,
        "wps_namelist": wps_namelist,
        "experiment_config": experiment_config,
        **{f"grib-f{hour:03d}": path for hour, path in records},
    }
    for path in roles.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "schema": "gpuwm-gfs-direct-input-manifest-v1",
        "source": {
            "model": "GFS",
            "product": "pgrb2.0p25",
            "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "files": {
            role: {"name": path.name, "sha256": _sha256(path)}
            for role, path in roles.items()
        },
    }
    path = output / "input-manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = _sha256(path)
    (output / "input-manifest.sha256").write_text(digest + "\n", encoding="ascii")
    return path, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--forecast-hours", default="0,3")
    parser.add_argument("--left-lon", type=float, required=True)
    parser.add_argument("--right-lon", type=float, required=True)
    parser.add_argument("--bottom-lat", type=float, required=True)
    parser.add_argument("--top-lat", type=float, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cycle = _cycle(args.cycle)
    hours = _forecast_hours(args.forecast_hours)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "input-manifest.json").exists():
        raise FileExistsError("refusing to replace an existing GFS input manifest")
    records = []
    for forecast_hour in hours:
        path = output / f"gfs.t{cycle:%H}z.pgrb2.0p25.f{forecast_hour:03d}.subset.grib2"
        _download(nomads_query(
            cycle, forecast_hour, left_lon=args.left_lon,
            right_lon=args.right_lon, bottom_lat=args.bottom_lat,
            top_lat=args.top_lat,
        ), path)
        records.append((forecast_hour, path))
    records_tuple = tuple(records)
    series = output / "gfs-series.tsv"
    series.write_text(
        "".join(f"{hour}\t{path}\n" for hour, path in records_tuple),
        encoding="utf-8",
    )
    manifest, digest = _write_manifest(
        output, cycle=cycle, records=records_tuple, series=series,
        bridge=args.bridge.resolve(), wps_namelist=args.wps_namelist.resolve(),
        experiment_config=args.experiment_config.resolve(),
    )
    print(json.dumps({
        "status": "READY",
        "series": str(series),
        "input_manifest": str(manifest),
        "input_manifest_sha256": digest,
        "records": [
            {"forecast_hour": hour, "path": str(path), "bytes": path.stat().st_size,
             "sha256": _sha256(path)}
            for hour, path in records_tuple
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
