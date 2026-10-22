"""Published composition files: Rust decoding, local import and IC/BC receipts.

Source rows select coordinates, fields, units and pressure construction. This
module manages files and identities only. No credentials are inspected and no
download occurs while initializing a forecast.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

from gpuwm.chem_boundary_remap import BoundaryRemapError, SourceFrames
from gpuwm.grib2_stack import StackedField


def _fingerprint(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@lru_cache(maxsize=2048)
def _hash_version(path, fingerprint):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if _fingerprint(path) != fingerprint:
        raise BoundaryRemapError(f"composition file changed while hashing: {path}")
    return h.hexdigest()


def _hash(path):
    # A changed inode, size, mtime or ctime forces rehashing. Forecast forcing
    # frames reuse the admitted immutable bytes rather than hashing GB again.
    path = str(Path(path).resolve(strict=True))
    return _hash_version(path, _fingerprint(path))


def _canonical(value):
    from gpuwm.chem_table import _thaw
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decoder():
    from gpuwm.netcdf_bridge import find_netcdf_bin
    binary = find_netcdf_bin()
    if binary is None:
        raise BoundaryRemapError("composition input needs rw_netcdf; build tools/rustwx with cargo build --release -p rw-netcdf")
    answer = subprocess.run([str(binary), "--abi"], capture_output=True, text=True, timeout=30, check=True)
    if "composition_stack_v1" not in answer.stdout:
        raise BoundaryRemapError("rw_netcdf lacks composition_stack_v1; rebuild cargo build --release -p rw-netcdf")
    return binary


def _manifest(row, paths):
    grid = row.grid["decode"]
    variables = [{"key": key, "name": spec["netcdf_name"], "units": spec.get("netcdf_units", spec["units"]), "group": spec["group"]}
                 for key, spec in row.variables.items() if "netcdf_name" in spec]
    return dict(schema="gpuwm-composition-stack-v1", files=[dict(path=str(p), group=g) for g, p in paths],
                latitude=grid["latitude"], longitude=grid["longitude"],
                time=grid["time"], level=grid["level"], variables=variables,
                pressure=row.vertical.get("pressure_from_thickness"),
                pressure_levels=row.vertical.get("pressure_from_levels"))


def _identity(row, paths, binary):
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        hashes = list(pool.map(_hash, [p for g, p in paths]))
    return dict(schema="gpuwm-composition-input-v1", source=row.name,
                row=json.loads(_canonical(dict(grid=row.grid, vertical=row.vertical,
                                               variables=row.variables, time=row.time,
                                               acquisition=row.acquisition, licence=row.licence,
                                               attribution=row.attribution))),
                inputs=[dict(path=str(p), group=g, sha256=h,
                             bytes=p.stat().st_size) for (g, p), h in zip(paths, hashes)],
                decoder=dict(sha256=_hash(binary), bytes=binary.stat().st_size))


def _input_remedy(row):
    acquisition = row.acquisition or {}
    credential = acquisition.get("required_credential")
    access = (f"Downloading this source requires a {credential}. " if credential else "")
    opt_in = acquisition.get("research_only_licence")
    return (access + f"Import matched files: python -m gpuwm.composition_source --source {row.name} "
            "--files GROUP=FILE... --cache CACHE" +
            (f" --licence {opt_in}" if opt_in else "") +
            "; initialization never downloads or reads credentials")


def import_files(row, paths, cache, *, licence=None, workers=8):
    """Register existing public files, with a decoder and raw-content identity."""
    if row.format != "netcdf" or not row.acquisition or row.acquisition.get("kind") != "published_files":
        raise ValueError("source does not declare published NetCDF composition files")
    required = row.acquisition.get("research_only_licence")
    if required and licence != required:
        raise ValueError(f"{row.name} requires explicit research opt-in: --licence {required}")
    try:
        paths = [(str(g), Path(p).resolve(strict=True)) for g, p in paths]
    except OSError as error:
        raise BoundaryRemapError(
            f"{row.name}: missing or unreadable composition input: {error}. "
            "Matched source files are needed to populate chemistry IC/BC; clean air cannot substitute. "
            + _input_remedy(row)) from error
    allowed = {s["group"] for s in row.variables.values() if "netcdf_name" in s}
    if {g for g, p in paths} != allowed:
        raise ValueError(f"source files must cover exactly these groups: {sorted(allowed)}")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("supply a nonempty list of distinct source files")
    binary = _decoder()
    identity = _identity(row, paths, binary)
    digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
    root = Path(cache) / "composition" / row.name
    root.mkdir(parents=True, exist_ok=True)
    decoded = root / digest
    if not decoded.exists():
        with tempfile.TemporaryDirectory(prefix=".decode-", dir=root) as scratch:
            manifest = Path(scratch) / "manifest.json"
            manifest.write_text(_canonical(_manifest(row, paths)), encoding="utf-8")
            staged = Path(scratch) / "decoded"
            subprocess.run([str(binary), "composition-stack", str(manifest), str(staged), str(workers)],
                           check=True, timeout=3600)
            if identity != _identity(row, paths, binary):
                raise BoundaryRemapError("composition input or decoder changed while decoding")
            payloads = {p.name: dict(bytes=p.stat().st_size, sha256=_hash(p))
                        for p in staged.iterdir() if p.is_file()}
            (staged / "authority.json").write_text(_canonical(dict(identity=identity, payloads=payloads)))
            try:
                os.replace(staged, decoded)
            except OSError:
                if not decoded.exists():
                    raise
    # Validate before publishing the registration; a failed import leaves the
    # preceding registration intact.
    frames = _read(row, decoded, identity)
    receipt = dict(schema="gpuwm-composition-registration-v1", identity=identity,
                   directory=decoded.name, licence=row.licence,
                   research_only=bool(required), attribution=row.attribution,
                   times=[t.isoformat() for t in frames.times])
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".receipt-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical(receipt) + "\n")
    os.replace(temporary, root / "import.json")
    return receipt


def _read(row, directory, identity):
    authority = json.loads((directory / "authority.json").read_text())
    if authority["identity"] != identity:
        raise BoundaryRemapError("composition decode authority differs from current files or source table")
    def validate_payload(item):
        name, info = item
        path = directory / name
        if path.parent != directory or Path(name).name != name:
            raise BoundaryRemapError("composition payload escapes decode directory")
        if path.stat().st_size != info["bytes"] or _hash(path) != info["sha256"]:
            raise BoundaryRemapError(f"composition decode payload changed: {path}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(validate_payload, authority["payloads"].items()))
    stack = json.loads((directory / "stack.json").read_text())
    if stack["schema"] != "gpuwm-composition-stack-v1":
        raise BoundaryRemapError("unknown composition stack schema")
    lat = np.asarray(stack["latitude"], dtype=np.float64)
    lon = np.asarray(stack["longitude"], dtype=np.float64)
    frames = {}
    for output in stack["outputs"]:
        name = output["file"]
        if Path(name).name != name or name not in authority["payloads"]:
            raise BoundaryRemapError("unregistered composition field payload")
        levels = tuple(output["levels"])
        if (directory / name).stat().st_size != len(levels) * lat.size * lon.size * 4:
            raise BoundaryRemapError("composition payload shape differs from its bytes")
        time = datetime.fromisoformat(output["valid_time"].replace("Z", "+00:00")).astimezone(timezone.utc)
        slot = frames.setdefault(time, {})
        if output["key"] in slot:
            raise BoundaryRemapError("duplicate composition variable at one valid time")
        coefficients = tuple(row.vertical.get("half_level_a_pa", ())) + tuple(row.vertical.get("half_level_b", ()))
        slot[output["key"]] = StackedField(output["key"], time, levels, lat, lon, coefficients,
                                          directory / name, 0, 0)
    return SourceFrames(row.name, frames, tuple(i["path"] for i in identity["inputs"]))


def registered_frames(row, cache):
    root = Path(cache) / "composition" / row.name
    try:
        receipt = json.loads((root / "import.json").read_text())
        directory = root / receipt["directory"]
        if directory.parent != root or directory.name != receipt["directory"]:
            raise BoundaryRemapError("composition registration escapes cache")
        paths = [(i["group"], Path(i["path"])) for i in receipt["identity"]["inputs"]]
        identity = _identity(row, paths, _decoder())
        return _read(row, directory, identity)
    except (OSError, KeyError, ValueError) as error:
        raise BoundaryRemapError(
            f"{row.name}: no valid registered composition files: {error}. "
            "Matched source files are needed to populate chemistry IC/BC; clean air cannot substitute. "
            + _input_remedy(row)) from error


def covering_files(row, variables, valid_time, cache):
    from gpuwm.chem_source_init import _needed_times, _utc
    frames = registered_frames(row, cache)
    for time in _needed_times(row, _utc(valid_time)):
        if time not in frames.frames or not set(variables) <= set(frames.frames[time]):
            raise BoundaryRemapError(f"{row.name}: registered files do not cover {sorted(variables)} at {time.isoformat()}")
    return {str(i): Path(path) for i, path in enumerate(frames.files)}


def inventory_rows():
    """Composition registry facts, distinct from meteorological forcing."""
    from gpuwm.chem_table import catalog, _thaw
    return [dict(source_id=row.name, display_name=row.name, title=row.name,
                 source_kind="composition", file_family=row.format,
                 description=row.description, credential=row.credential,
                 acquisition=_thaw(row.acquisition),
                 variables=_thaw(row.variables), vertical=_thaw(row.vertical),
                 licence=row.licence, attribution=row.attribution,
                 maturity=dict(status="implemented", runnable=True)
                     if row.acquisition.get("kind") == "published_files"
                     else dict(status="needs-event-proof", runnable=False),
                 fetch=dict(kind=row.acquisition.get("kind")),
                 run_plan=dict(intent_supported=False, intent_chain=None,
                               intent_refusal="Composition supplements chemistry IC/BC; it does not supply meteorological forcing"),
                 use=dict(config_key="chem_sources", value=row.name,
                          import_command=(f"python -m gpuwm.composition_source --source {row.name} --files GROUP=FILE... --cache CACHE"
                                          if row.acquisition.get("kind") == "published_files" else None)))
            for row in catalog().sources.values()
            if row.kind == "boundary" and row.acquisition]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--files", nargs="+", required=True, metavar="GROUP=FILE")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--licence")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    from gpuwm.chem_table import catalog
    from gpuwm.chem_source_init import data_store_cache_root
    if not 1 <= args.workers <= 64:
        parser.error("workers must be 1..64")
    try:
        files = [item.split("=", 1) for item in args.files]
        if any(len(item) != 2 for item in files):
            raise ValueError
    except ValueError:
        parser.error("each file must be GROUP=FILE")
    receipt = import_files(catalog().sources[args.source], files,
                           args.cache or data_store_cache_root(),
                           licence=args.licence, workers=args.workers)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
