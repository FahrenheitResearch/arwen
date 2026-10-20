"""Initial water state for gpuwm hydro: orchestration only.

Rust does every data step (rw_zarr reads and crops the source, rw_hydro_init
builds area weights, maps the state onto catchments and runs the spin-up).
This module picks the table row, splits the fetch into time slices that run
as parallel rw_zarr processes, and chains the Rust stages. Sources are rows
in gpuwm/data/hydro/init-state-sources.v1.json; adding one is table work.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time

TABLE = Path(__file__).resolve().parent.parent / "data" / "hydro" / "init-state-sources.v1.json"
INIT_ENV = "GPUWM_RW_HYDRO_INIT"
ISO = "%Y-%m-%dT%H:%M:%SZ"


def load_table(path: Path | None = None) -> dict:
    return json.loads(Path(path or TABLE).read_text(encoding="utf-8"))


def admit(table: dict, name: str) -> dict:
    """The row, or a refusal naming why it cannot be used.

    Prevents: building a public product from a source whose licence was not
    read at the source, that needs an account Drew has not ruled on, or that
    has no reader yet (the row says which).
    """
    rows = table["rows"]
    if name not in rows:
        raise KeyError(f"no initial-state source row {name!r}; rows: {sorted(rows)}")
    row = rows[name]
    if row.get("status") != "admitted":
        raise PermissionError(f"initial-state source {name} is {row.get('status')}: {row.get('reason')}")
    if row.get("licence", {}).get("status") != "verified":
        raise PermissionError(f"initial-state source {name} has no licence read at the source")
    return row


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, ISO).replace(tzinfo=timezone.utc)


def hours(start: datetime, count: int, step_s: int = 3600) -> list[str]:
    return [(start + timedelta(seconds=step_s * i)).strftime(ISO) for i in range(count)]


def _rw_zarr() -> Path:
    from gpuwm.zarr_bridge import resolve_zarr_bin
    return resolve_zarr_bin()


def _rw_hydro_init() -> Path:
    override = os.environ.get(INIT_ENV)
    if override:
        return Path(override)
    root = Path(__file__).resolve().parent.parent.parent
    for candidate in (root / "tools/hydro_world_init/target/release/rw_hydro_init",
                      root / "libexec/bridges/rw_hydro_init"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"rw_hydro_init is not built; set {INIT_ENV} or build tools/hydro_world_init")


def _state_variables(row: dict) -> list[tuple[str, str]]:
    out = []
    state = row["state"]
    for group in ("soil_water", "soil_temperature"):
        for layer in state[group]["layers"]:
            out.append((layer["variable"], state[group]["units"]))
    for key in ("snow_water_equivalent", "snow_density", "snow_temperature", "land_fraction", "soil_class"):
        if key in state and "variable" in state[key]:
            out.append((state[key]["variable"], state[key]["units"]))
    return out


def _forcing_variables(row: dict) -> list[tuple[str, str]]:
    seen = {}
    for binding in row["forcing"].values():
        for term in binding["terms"]:
            seen[term["variable"]] = binding["units"]
    return list(seen.items())


def _zarr_request(row: dict, variables, times, area) -> dict:
    access = row["access"]
    return {"store": access["store"], "times": times, "area": list(area),
            "expected_levels_hpa": access["expected_levels_hpa"],
            "fields": [{"source": v, "output": v, "units": u, "pressure": False} for v, u in variables]}


def _fetch_jobs(row, area, state_times, forcing_times, slice_hours=24):
    jobs = []
    if state_times:
        jobs.append(("state", _zarr_request(row, _state_variables(row), state_times, area)))
    forcing_vars = _forcing_variables(row)
    for i in range(0, len(forcing_times), slice_hours):
        part = forcing_times[i:i + slice_hours]
        jobs.append((f"forcing-{i // slice_hours:04d}", _zarr_request(row, forcing_vars, part, area)))
    return jobs


def _payload_identity(path):
    """Hash opaque source bytes; scientific decoding remains in Rust."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _payload_matches(item):
    path, part = item
    if "size_bytes" not in part and "sha256" not in part:
        return True  # Legacy receipts carry no payload integrity evidence.
    size, digest = part.get("size_bytes"), part.get("sha256")
    if (type(size) is not int or size < 0 or not isinstance(digest, str)
            or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or path.stat().st_size != size):
        return False
    return _payload_identity(path) == {"size_bytes": size, "sha256": digest}


def _matching_fetch(directory, jobs):
    """Match the actual extraction requests, including crop, variables and units.

    New receipts also seal the opaque payload bytes. Legacy receipts remain
    usable, without claiming integrity evidence they did not record.
    """
    try:
        receipt = json.loads((directory / "fetch-receipt.json").read_text())
        parts = {part["name"]: part for part in receipt["slices"]}
        if len(parts) != len(receipt["slices"]) or set(parts) != {name for name, _ in jobs}:
            return None
        payloads = []
        for name, request in jobs:
            if (not (directory / f"{name}.nc").is_file()
                    or json.loads((directory / f"{name}.request.json").read_text()) != request):
                return None
            payloads.append((directory / f"{name}.nc", parts[name]))
        if payloads:
            with ThreadPoolExecutor(max_workers=min(len(payloads), os.cpu_count() or 1)) as pool:
                if not all(pool.map(_payload_matches, payloads)):
                    return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _fetch_location(directory, jobs):
    """Keep existing inputs and select a reusable directory for this request."""
    signature = hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    candidate, attempt = directory, 0
    while candidate.exists():
        receipt = _matching_fetch(candidate, jobs)
        if receipt is not None:
            return candidate, receipt
        # A different crop or incomplete attempt stays untouched. Later calls
        # find the same completed request instead of downloading it again.
        suffix = signature if attempt == 0 else f"{signature}-{attempt}"
        candidate = directory.with_name(f"{directory.name}-{suffix}")
        attempt += 1
    return candidate, None


def _shared_fetch(directory, row, *, jobs, area, state_times, forcing_times, progress):
    """One live writer per shared cache address; other preparations reuse it.

    Selection and extraction share the lease: observing another writer's
    partial directory must not start a second download of the same request.
    The OS releases ownership on interruption; incomplete inputs stay intact
    and the next preparation selects a fresh attempt.
    """
    from gpuwm.remote_artifact_cache import Lease

    directory = Path(directory).resolve()
    selected, fetched = _fetch_location(directory, jobs)
    if fetched is not None:
        # Completed cache entries are immutable. Reuse remains read-only,
        # including a source extraction supplied by another preparation.
        progress(f"hydro init: reusing fetched source in {selected}")
        return selected, fetched
    locks = directory.parent / ".hydro-init-fetch-locks"
    locks.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(os.path.normcase(str(directory)).encode()).hexdigest()
    lock_path = locks / (key + ".lock")
    lease = Lease(lock_path)
    if lease.file is None:
        progress(f"hydro init: waiting for shared source in {directory}")
        lease = Lease(lock_path, timeout=float("inf"))
    with lease:
        selected, fetched = _fetch_location(directory, jobs)
        if fetched is None:
            fetched = fetch(row, area=area, state_times=state_times,
                            forcing_times=forcing_times, out=selected, progress=progress)
        else:
            progress(f"hydro init: reusing fetched source in {selected}")
        return selected, fetched


def fetch(row: dict, *, area, state_times, forcing_times, out: Path,
          workers: int | None = None, slice_hours: int = 24, progress=print) -> dict:
    """Fetch the state snapshots and the forcing series as parallel rw_zarr slices.

    Each rw_zarr process reads its slice's chunks with its own worker pool, so
    the fetch runs as many processes as there are slices (bounded by workers).
    """
    if row["access"]["protocol"] != "zarr-http" or row["access"]["reader"] != "rw_zarr":
        raise NotImplementedError(f"no fetch path for access {row['access']}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    exe = _rw_zarr()
    jobs = _fetch_jobs(row, area, state_times, forcing_times, slice_hours)
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 8)))
    started = time.perf_counter()

    def run(job):
        name, request = job
        req = out / f"{name}.request.json"
        req.write_text(json.dumps(request, indent=1), encoding="utf-8")
        t0 = time.perf_counter()
        proc = subprocess.run([os.fspath(exe), "extract", os.fspath(req), os.fspath(out / f"{name}.nc")],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"rw_zarr {name} failed: {proc.stderr[-2000:]}")
        identity = _payload_identity(out / f"{name}.nc")
        return {"name": name, "seconds": time.perf_counter() - t0, "times": len(request["times"]),
                "fields": len(request["fields"]), "file": f"{name}.nc", **identity}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    receipt = {"schema": "arwen.hydro-world.init-fetch.v1", "store": row["access"]["store"],
               "area_south_west_north_east": list(area), "state_times": state_times,
               "forcing_first": forcing_times[0] if forcing_times else None,
               "forcing_last": forcing_times[-1] if forcing_times else None,
               "forcing_records": len(forcing_times), "slices": results,
               "wall_seconds": time.perf_counter() - started,
               "parallelism": f"{workers} rw_zarr processes, each reading its fields concurrently",
               "chunks_read": sum(r["times"] * r["fields"] for r in results)}
    (out / "fetch-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    progress(f"hydro init: fetched {receipt['chunks_read']} source chunks in {receipt['wall_seconds']:.1f} s")
    return receipt


def run_stage(command: str, request: dict, workdir: Path) -> dict:
    """One rw_hydro_init stage from a request file; returns its JSON receipt."""
    workdir.mkdir(parents=True, exist_ok=True)
    req = workdir / f"{command}.request.json"
    req.write_text(json.dumps(request, indent=1), encoding="utf-8")
    proc = subprocess.run([os.fspath(_rw_hydro_init()), command, os.fspath(req)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rw_hydro_init {command} failed: {proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


def prepare(*, fabric: Path, date: str, out: Path, source: str = "era5-arco",
            spinup_days: int = 0, area=None, params: Path | None = None,
            fetch_dir: Path | None = None, trace=(), progress=print) -> dict:
    """Prepare fresh results, preserving previous runs and reusing source data."""
    from gpuwm.output_claim import acquire_output
    from .output import _select_output_directory

    # Invalid source/date requests must fail before creating output or leases.
    admit(load_table(), source)
    parse_time(date)
    requested = Path(out).absolute()
    selected = _select_output_directory(requested)

    def reserve():
        selected.mkdir(parents=True, exist_ok=True)
        if selected.is_symlink() or next(selected.iterdir(), None) is not None:
            raise FileExistsError(f"initial-state output acquired other files: {selected}")
        return selected

    while True:
        try:
            claim = acquire_output(selected, prepare=reserve)
            break
        except FileExistsError:
            # Another preparation can reserve an empty directory between
            # selection and acquisition. A new sibling keeps both runs safe.
            selected = Path(tempfile.mkdtemp(prefix=requested.name + "-",
                                             dir=requested.absolute().parent))
    with claim:
        progress(f"Hydro initial state: {claim.path} (previous files preserved)")
        return _prepare(fabric=fabric, date=date, out=claim.path, source=source,
                        spinup_days=spinup_days, area=area, params=params,
                        fetch_dir=fetch_dir, trace=trace, progress=progress,
                        requested_out=requested)


def _prepare(*, fabric, date, out, source, spinup_days, area, params,
             fetch_dir, trace, progress, requested_out):
    """Initial state for every catchment of FABRIC at DATE, optionally spun up.

    With spinup_days N the source state at DATE - N days is mapped onto the
    catchments and CFE runs N days of the source's own forcing to DATE.
    """
    table = load_table()
    row = admit(table, source)
    t = parse_time(date)
    start = t - timedelta(days=spinup_days)
    out = Path(out)
    if area is None:
        area = run_stage("bbox", {"fabric": os.fspath(fabric), "pad_degrees": 0.5}, out / "bbox")["area_south_west_north_east"]
    state_times = sorted({start.strftime(ISO), t.strftime(ISO)})
    forcing_times = hours(start + timedelta(hours=1), spinup_days * 24) if spinup_days else []
    jobs = _fetch_jobs(row, area, state_times, forcing_times)
    if fetch_dir is None:
        # Source cache belongs to the requested address, independent of fresh
        # result directories. Existing pre-cache-layout extractions still work.
        legacy = requested_out / "source"
        if not requested_out.is_symlink() and _matching_fetch(legacy, jobs) is not None:
            fetch_dir = legacy
        else:
            fetch_dir = requested_out.with_name(requested_out.name + "-source")
    fetch_dir, fetched = _shared_fetch(fetch_dir, row, jobs=jobs, area=area,
                                      state_times=state_times, forcing_times=forcing_times,
                                      progress=progress)
    state_nc = os.fspath(fetch_dir / "state.nc")
    weights = run_stage("weights", {"fabric": os.fspath(fabric), "grid_file": state_nc,
                                    "row": source, "table": os.fspath(TABLE),
                                    "output": os.fspath(out / "weights")}, out / "weights-stage")
    state_at = start.strftime(ISO)
    state = run_stage("state", {"table": os.fspath(TABLE), "row": source, "fabric": os.fspath(fabric),
                                "weights": os.fspath(out / "weights"), "source_file": state_nc,
                                "valid_time": state_at, "params": os.fspath(params) if params else None,
                                "output": os.fspath(out / "state")}, out / "state-stage")
    result = {"schema": "arwen.hydro-world.init-run.v1", "fabric": os.fspath(fabric), "date": date,
              "requested_output_directory": os.fspath(requested_out), "output_directory": os.fspath(out),
              "source": source, "spinup_days": spinup_days, "fetch": fetched, "weights": weights, "state": state}
    if spinup_days:
        # Reanalysis state at the forecast start itself (Noah-OWP soil, snow),
        # beside the spun-up CFE storages.
        result["state_at_date"] = run_stage("state", {"table": os.fspath(TABLE), "row": source, "fabric": os.fspath(fabric),
                                                      "weights": os.fspath(out / "weights"), "source_file": state_nc,
                                                      "valid_time": t.strftime(ISO), "params": os.fspath(params) if params else None,
                                                      "output": os.fspath(out / "state-at-date")}, out / "state-at-date-stage")
        files = sorted(os.fspath(p) for p in fetch_dir.glob("forcing-*.nc"))
        result["spinup"] = run_stage("spinup", {"table": os.fspath(TABLE), "row": source, "fabric": os.fspath(fabric),
                                                 "weights": os.fspath(out / "weights"),
                                                 "initial_state": os.fspath(out / "state"),
                                                 "land_file": state_nc,
                                                 "forcing_files": files, "start": state_at, "hours": spinup_days * 24,
                                                 "params": os.fspath(params) if params else None,
                                                 "trace": list(trace),
                                                 "retain_runoff_replay": True,
                                                 "output": os.fspath(out / "spinup")}, out / "spinup-stage")
    (out / "init-run.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def register_cli(commands):
    p = commands.add_parser("init-state", help="initial soil, snow, groundwater and channel state from a reanalysis row, with optional CFE spin-up")
    p.add_argument("--fabric", type=Path, required=True, help="HYDRO-WORLD contract GeoPackage")
    p.add_argument("--date", required=True, help="forecast start, e.g. 2023-01-01T00:00:00Z")
    p.add_argument("--source", default="era5-arco", help="row in init-state-sources.v1.json")
    p.add_argument("--spinup-days", type=int, default=0)
    p.add_argument("--params", type=Path, help="CFE parameter row file when the fabric has no model_attributes")
    p.add_argument("--outdir", type=Path, required=True)
    p.set_defaults(func=lambda a: (prepare(fabric=a.fabric, date=a.date, out=a.outdir, source=a.source,
                                           spinup_days=a.spinup_days, params=a.params), 0)[1])
