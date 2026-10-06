"""Table-driven annual CF archive acquisition, with Rust-owned data binding."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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

from gpuwm import fetch_guard

TABLE = Path(__file__).with_name("authorities") / "native-cf-fetch.v1.json"
MANIFEST_SCHEMA = "gpuwm-native-cf-fetch-manifest-v1"
REQUEST_NAME = "native-cf-request.json"


def sources() -> tuple[str, ...]:
    return tuple(json.loads(TABLE.read_text(encoding="utf-8"))["sources"])


def contract(source: str) -> str | None:
    return row(source).get("contract") if source in sources() else None


def row(source: str) -> dict:
    document = json.loads(TABLE.read_text(encoding="utf-8"))
    if document["schema"] != "gpuwm-native-cf-fetch-v1":
        raise ValueError("Unsupported native CF fetch table schema")
    return document["sources"][source]


def validate_window(source: str, cycle: datetime, hours: int,
                    cadence: int | None = None) -> tuple[datetime, ...]:
    metadata = row(source)
    spacing = metadata["cadence_hours"] if cadence is None else cadence
    native = metadata["cadence_hours"]
    if cycle.hour % native or cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError(f"{source} publishes {native}-hourly analyses; choose 00/03/06/09/12/15/18/21 UTC")
    if type(spacing) is not int or spacing < native or spacing % native:
        raise ValueError(f"{source} cadence must be a positive multiple of its {native} h analysis clock")
    if type(hours) is not int or hours < 0 or hours % spacing:
        raise ValueError(f"--hours must be a nonnegative multiple of the {spacing} h cadence")
    times = tuple(cycle + timedelta(hours=i) for i in range(0, hours + 1, spacing))
    first = datetime.fromisoformat(metadata["coverage_start"])
    last = datetime.fromisoformat(metadata["coverage_end"])
    if times[0] < first or times[-1] > last:
        raise ValueError(f"{source} coverage is {first:%Y-%m-%d %H} UTC to {last:%Y-%m-%d %H} UTC; the requested boundary window leaves it")
    return times


def plans(source: str, times: tuple[datetime, ...], area) -> list[dict]:
    metadata = row(source)
    result = []
    for year in sorted({value.year for value in times}):
        suffix = next(era["suffix"] for era in metadata["eras"] if year <= era["through_year"])
        selected = [value for value in times if value.year == year]
        for field in metadata["files"]:
            directory = field["directory"] + suffix
            query = urlencode({
                "var": field["variable"], "north": area.lat_north,
                "south": area.lat_south, "west": area.lon_west, "east": area.lon_east,
                "horizStride": 1, "time_start": selected[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "time_end": selected[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timeStride": (int((times[1]-times[0]).total_seconds()/3600) // metadata["cadence_hours"]
                               if len(times) > 1 else 1), "accept": "netcdf"})
            name = f"{field['stem']}.{year}.nc"
            result.append(dict(field, name=name, year=year, primary=True,
                valid_times=[value.isoformat() for value in selected],
                url=f"{metadata['ncss_root']}/{directory}/{name}?{query}"))
    for role, field in metadata["invariants"].items():
        result.append(dict(field, role=role, name=f"published-{role}.nc", primary=False,
                           url=f"{metadata['file_root']}/{field['path']}"))
    return result


def latest_cycle(source: str, hours: int, cadence: int | None = None) -> datetime:
    cycle = datetime.fromisoformat(row(source)["coverage_end"]) - timedelta(hours=hours)
    validate_window(source,cycle,hours,cadence)
    return cycle


def request_identity(source: str, times: tuple[datetime, ...], area) -> dict:
    return {"schema": "gpuwm-native-cf-request-v1", "source": source,
            "valid_times": [value.isoformat() for value in times], "area": area.as_manifest(),
            "table_sha256": sha256(TABLE), "urls": [item["url"] for item in plans(source,times,area)]}


def check_prior_request(out: Path, *, source: str, cycle: datetime,
                        hours: int, cadence: int | None, area) -> None:
    expected = request_identity(source,validate_window(source,cycle,hours,cadence),area)
    if _cached_document(Path(out) / REQUEST_NAME) != expected:
        raise ValueError("The native CF cache belongs to another acquisition request")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, document) -> None:
    fetch_guard.atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _cached_document(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    last = None
    for attempt in range(5):
        try:
            with urlopen(url, timeout=180) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            temporary.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            last = error
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                break
            if attempt < 4:
                time.sleep(min(8 * (attempt + 1), 30))
    if temporary.exists():
        fetch_guard.quarantine(temporary, tag="incomplete")
    raise RuntimeError(f"CF archive acquisition failed after bounded retries: {url}: {last}")


def _validate(path: Path, item: dict, metadata: dict) -> None:
    from gpuwm.netcdf_bridge import open_dataset
    with open_dataset(path) as dataset:
        if item["variable"] not in dataset.variables:
            raise ValueError(f"{path.name}: expected variable {item['variable']!r} is absent")
        variable = dataset.variables[item["variable"]]
        if getattr(variable, "statistic", None) != metadata["statistic"]:
            raise ValueError(f"{path.name}: expected {metadata['statistic']}; a member or another statistic cannot initialize this route")
        if item["primary"]:
            actual = [value.isoformat() for value in dataset.variables["time"].times()]
            if actual != item["valid_times"]:
                raise ValueError(f"{path.name}: the provider returned times {actual}, expected {item['valid_times']}")


def acquire(*, source: str, cycle: datetime, hours: int, cadence: int | None,
            area, out: Path, workers: int = 2, force: bool = False) -> dict:
    from gpuwm import fetch_routes
    from gpuwm.netcdf_bridge import resolve_netcdf_bin, _run
    metadata = row(source)
    times = validate_window(source, cycle, hours, cadence)
    if area is None:
        raise ValueError("A native CF subset needs --area or --point with --radius-km so it does not download variable-year global archives")
    if type(workers) is not int or workers <= 0:
        raise ValueError("--fetch-workers must be a positive whole number")
    binary = resolve_netcdf_bin()
    abi = _run([str(binary), "--abi"], what="CF invariant binding capability").stdout
    if "bind_published_invariants_v1" not in abi:
        raise RuntimeError("The installed rw_netcdf lacks published invariant binding; install the matching engine bridges before fetching this source")
    items = plans(source, times, area)
    identity = request_identity(source,times,area)
    out = Path(out).resolve()
    with fetch_guard.hold("native-cf", out):
        out.mkdir(parents=True, exist_ok=True)
        request_path = out / REQUEST_NAME
        old = _cached_document(request_path)
        # Preserve completed or partial earlier requests, then recover without a filesystem decision.
        if force or (old is not None and old != identity):
            owned = [REQUEST_NAME, "fetch-manifest.json", "SHA256SUMS", "inputs.txt",
                     "prep-command.txt", "prep-arguments.json", "invariant.nc", "invariant.provenance.json"]
            if old is not None:
                old_manifest = out / "fetch-manifest.json"
                if old_manifest.is_file():
                    owned += [entry["path"] for entry in json.loads(old_manifest.read_text())["files"]]
                owned += [path.name for path in out.glob("*.fetch.json")]
                for url in old["urls"]:
                    # Primary filenames are the annual key's last component, before its query.
                    owned.append(url.split("?", 1)[0].rsplit("/", 1)[-1])
                owned += ["published-height.nc", "published-land.nc"]
            for name in dict.fromkeys(owned):
                path = out / name
                if path.is_file():
                    fetch_guard.quarantine(path, tag="previous-request")
        _json(request_path, identity)

        def transfer(item):
            destination = out / item["name"]
            stamp = destination.with_suffix(".nc.fetch.json")
            cached = _cached_document(stamp)
            reusable = (destination.is_file() and cached is not None
                        and cached.get("url") == item["url"] and cached.get("sha256") == sha256(destination))
            if not reusable:
                if destination.exists():
                    fetch_guard.quarantine(destination, tag="unverified")
                _download(item["url"], destination)
            try:
                _validate(destination, item, metadata)
            except ValueError:
                fetch_guard.quarantine(destination, tag="invalid-cf")
                raise
            entry = {"path": item["name"], "bytes": destination.stat().st_size,
                     "sha256": sha256(destination), "url": item["url"], "variable": item["variable"],
                     "primary": item["primary"], "reused": reusable}
            _json(stamp, entry)
            print(f"fetch {source}: {item['name']} ({entry['bytes']} bytes)", file=sys.stderr)
            return entry

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            files = list(pool.map(transfer, items))
        supplement = out / "invariant.nc"
        receipt_path = out / "invariant.provenance.json"
        time_files = [out / item["name"] for item in items if item["primary"] and item["stem"] == "hgt"]
        receipt = _cached_document(receipt_path)
        reusable = (supplement.is_file() and receipt is not None
                    and receipt.get("supplement_sha256") == sha256(supplement))
        if not reusable:
            if supplement.exists():
                fetch_guard.quarantine(supplement, tag="unverified")
            if receipt_path.exists():
                fetch_guard.quarantine(receipt_path, tag="unverified")
            completed = _run([str(binary), "bind-invariants", str(out / "published-height.nc"),
                metadata["invariants"]["height"]["variable"], str(out / "published-land.nc"),
                metadata["invariants"]["land"]["variable"], str(supplement), *map(str, time_files)],
                what="published invariant binding")
            receipt = json.loads(completed.stdout)
            receipt["sources"] = [entry for entry in files if not entry["primary"]]
            receipt["supplement_sha256"] = sha256(supplement)
            _json(receipt_path, receipt)
        files.append({"path": supplement.name, "bytes": supplement.stat().st_size,
                      "sha256": sha256(supplement), "primary": True, "role": "published_invariant_binding"})
        inputs = out / fetch_routes.INPUT_LIST_NAME
        fetch_guard.atomic_write_text(inputs, "".join(str(out / entry["path"]) + "\n" for entry in files if entry["primary"]))
        fetch_guard.atomic_write_text(out / fetch_routes.SHA256SUMS_NAME,
            "".join(entry["sha256"] + "  " + entry["path"] + "\n" for entry in files))
        tokens = ["--source", source, "--input-list", str(inputs), "--supplement", str(supplement),
                  "--author-input-manifest", str(out / "inputs.json")]
        fetch_routes.write_prep_arguments(out, source=source, prep_source=source, cycle=cycle, tokens=tokens)
        fetch_guard.atomic_write_text(out / fetch_routes.PREP_COMMAND_NAME, fetch_routes.render_prep_command(tokens) + "\n")
        manifest = {"schema": MANIFEST_SCHEMA, "source": source, "contract": metadata["contract"],
                    "statistic": metadata["statistic"], "resolution_degrees": metadata["resolution_degrees"],
                    "request": identity, "files": files, "valid_times": identity["valid_times"],
                    "surface_policy": metadata["surface_policy"], "invariant_binding": receipt,
                    "total_bytes": sum(entry["bytes"] for entry in files),
                    "wall_seconds": time.monotonic() - started, "workers": workers}
        _json(out / fetch_routes.MANIFEST_NAME, manifest)
        return manifest


def cli(args) -> int:
    from gpuwm.fetch import _resolve_area, parse_cycle
    source = args.source
    unsupported = ["member", "validate", "transport", "engine", "mode", "cache_dir",
                   "p_top_pa", "forecast_start_hour", "bridge", "wps_namelist", "experiment_config",
                   "static_input", "static_receipt", "manifest_out"]
    supplied = ["--" + key.replace("_", "-") for key in unsupported if getattr(args, key, None) is not None]
    supplied += ["--" + key.replace("_", "-") for key in ("retrieve", "all_levels", "author_front_door_manifest", "accept_inventory_change") if getattr(args, key, False)]
    if supplied:
        raise ValueError(f"{', '.join(supplied)} does not describe annual CF subset acquisition; use --cycle, --hours, --cadence and --area")
    if args.cycle is None or args.hours is None or args.out is None:
        raise ValueError("fetch requires --cycle, --hours and --out")
    if args.cycle == "latest":
        cycle = latest_cycle(source,args.hours,args.cadence)
    else:
        cycle = parse_cycle(args.cycle, source)
    manifest = acquire(source=source, cycle=cycle, hours=args.hours, cadence=args.cadence,
                       area=_resolve_area(args), out=args.out,
                       workers=args.fetch_workers or 2, force=args.force_refetch)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0
