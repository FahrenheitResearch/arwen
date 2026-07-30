"""GFS GRIB2 -> native GPU initialization -> stock-WRF direct export."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Mapping

import numpy as np

from gpuwm import __version__
from gpuwm.era5_direct import (
    _canonical_surface,
    _load_static,
    _static_from_geog,
    _write_geometry_receipt,
    _write_static_cache,
)
from gpuwm.experiment import load_experiment
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import (
    _regular_coordinates,
    interpolate_era5_to_lambert,
    interpolate_lake_skin_temperature,
)
from gpuwm.ingest.lateral_bc import (
    attach_lateral_boundaries,
    build_state_lateral_boundaries,
)
from gpuwm.ingest.prepared_cache import (
    prepared_cache_identity,
    write_prepared_cache,
)
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
from gpuwm.ingest.real import initialize_real
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.native_domain_artifacts import _atomic_staging_sibling
from gpuwm.native_wrf_contract import (
    native_geometry_contract,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    verify_native_static_receipt,
)
from gpuwm.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
)
from gpuwm.physics_compat import (
    WSM6_PROFILE_ID,
    land_surface_component_for_selector,
    land_surface_route_blocker,
    multi_domain_physics_selection,
    validate_single_domain_physics_profile,
)
from gpuwm.wrf_direct import export_prepared_wrf


INPUT_MANIFEST_SCHEMA = "gpuwm-gfs-direct-input-manifest-v1"
BRIDGE_SCHEMA = "gpuwm-gfs-grib2-bridge-v1"
PROOF_SCHEMA = "gpuwm-gfs-direct-wrf-proof-v3"
LEGACY_PROOF_SCHEMA = "gpuwm-gfs-direct-wrf-proof-v2"
#: The multi-domain product.  v2 adds the front-door physics receipt the
#: v3 single-domain proof already carried; v1 documents written by 1.1.0
#: and earlier stay independently verifiable and are never promoted to
#: v2 by inference, exactly as the direct proof's v2 is not.
HIERARCHY_PROOF_SCHEMA = "gpuwm-gfs-native-hierarchy-proof-v2"
LEGACY_HIERARCHY_PROOF_SCHEMA = "gpuwm-gfs-native-hierarchy-proof-v1"
_PRESSURE_LEVELS_HPA = np.asarray(
    [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
     650, 700, 750, 800, 850, 900, 925, 950, 975, 1000],
    dtype=np.float64,
)
_THREE_D = ("GHT", "T", "RH", "U", "V")
_TWO_D = (
    "PSFC", "SOURCE_OROGRAPHY", "SKINTEMP", "SNOW", "SNOWH",
    "T2", "RH2", "U10", "V10", "LANDSEA", "XICE",
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200",
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200",
)
_IMPLEMENTATION_PATHS = (
    "gpuwm/gfs_direct.py",
    "gpuwm/era5_direct.py",
    "gpuwm/native_wrf_contract.py",
    "gpuwm/source_hierarchy.py",
    "gpuwm/native_hierarchy.py",
    "gpuwm/native_domain_artifacts.py",
    "gpuwm/source_cli.py",
    "gpuwm/source_adapters.py",
    "gpuwm/experiment.py",
    "gpuwm/core/grid.py",
    "gpuwm/core/noah.py",
    "gpuwm/ingest/grib.py",
    "gpuwm/ingest/horiz.py",
    "gpuwm/ingest/cpu_backend.py",
    "gpuwm/ingest/preprocess_backend.py",
    "gpuwm/ingest/soil.py",
    "gpuwm/ingest/real.py",
    "gpuwm/ingest/lateral_bc.py",
    "gpuwm/ingest/prepared_cache.py",
    "gpuwm/ingest/nest_init.py",
    "gpuwm/static/lambert.py",
    "gpuwm/static/build.py",
    "gpuwm/wrf_direct.py",
    "gpuwm/data/noah_tables/GENPARM.TBL",
    "gpuwm/data/noah_tables/SOILPARM.TBL",
    "gpuwm/data/noah_tables/VEGPARM.TBL",
    "tools/prepare_gfs_cpu_wrf.py",
    "tools/grib1_bridge/Cargo.lock",
    "tools/grib1_bridge/Cargo.toml",
    "tools/grib1_bridge/src/bin/gfs_grib2_bridge.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/mod.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/parser.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/search.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/unpack.rs",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    result = {
        name: _sha256(root / name)
        for name in _IMPLEMENTATION_PATHS
        if (root / name).is_file()
    }
    missing = [name for name in _IMPLEMENTATION_PATHS
               if not (root / name).is_file()]
    if missing:
        # Installed wheels do not carry the repo-only paths (tools/,
        # Rust workspace sources).  A sealed runtime binds its own
        # distribution manifest here; a plain pip install records the
        # absent inventory honestly instead of demanding one.
        if os.environ.get("GPUWM_NATIVE_DISTRIBUTION_MANIFEST"):
            manifest_path, _ = _distribution_manifest()
            result["distribution/manifest.json"] = _sha256(manifest_path)
        result["distribution/repo_only_paths.json"] = hashlib.sha256(
            json.dumps(missing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return result


def _distribution_manifest() -> tuple[Path, dict[str, object]]:
    raw_path = os.environ.get("GPUWM_NATIVE_DISTRIBUTION_MANIFEST")
    if not raw_path:
        raise RuntimeError(
            "installed GFS runtime requires GPUWM_NATIVE_DISTRIBUTION_MANIFEST")
    path = Path(raw_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = payload.get("artifact")
    source = payload.get("source")
    if (payload.get("schema") != "gpuwm-native-wrf-runtime-v1"
            or payload.get("status") != "READY"
            or not isinstance(artifact, dict)
            or artifact.get("gpuwm_version") != __version__
            or not isinstance(source, dict)
            or not source.get("worktree_clean")):
        raise RuntimeError("unrecognized or incompatible native distribution manifest")
    return path, payload


def _git_source_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent

    if os.environ.get("GPUWM_NATIVE_DISTRIBUTION_MANIFEST"):
        manifest_path, manifest = _distribution_manifest()
        source = manifest["source"]
        return {
            "available": True,
            "commit": source["commit"],
            "tree": source["tree"],
            "tracked_worktree_clean": True,
            "identity_source": "gpuwm-native-distribution-manifest",
            "distribution_manifest_sha256": _sha256(manifest_path),
        }

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], check=True,
            text=True, capture_output=True)
        return completed.stdout.strip()

    # An installed wheel is neither a git checkout nor a sealed runtime:
    # record the identity as honestly unavailable instead of refusing to
    # run (the input manifest, decoder digest, and implementation hashes
    # above still bind the run's provenance).
    unavailable = {
        "available": False,
        "commit": None,
        "tree": None,
        "tracked_worktree_clean": None,
        "identity_source": "unavailable-installed-runtime",
    }
    try:
        top_level = Path(run("rev-parse", "--show-toplevel")).resolve()
    except (OSError, subprocess.CalledProcessError):
        return unavailable
    if top_level != root.resolve():
        # site-packages inside some unrelated repository: that
        # repository's history is NOT this code's provenance.
        return unavailable
    return {
        "available": True,
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": not bool(
            run("status", "--porcelain", "--untracked-files=no")),
        "identity_source": "git",
    }


def _geometry_contract(grid, cfg) -> dict[str, object]:
    return native_geometry_contract(grid, cfg)


def _validate_grid_and_vertical_contract(exp, wps_namelist: Path):
    return validate_native_lambert_contract(
        exp, wps_namelist, source_name="GFS",
        source_top_pressure_pa=float(_PRESSURE_LEVELS_HPA.min() * 100.0),
    )


def _verify_static_receipt(
    receipt_path: Path,
    static_input: Path,
    grid,
    cfg,
) -> dict[str, object]:
    return verify_native_static_receipt(
        receipt_path, static_input, grid, cfg)


def _source_coverage_receipt(snapshot: Era5Snapshot, grid, lake_mask) -> dict[str, object]:
    ny = snapshot.latitude.size
    nx = snapshot.longitude.size
    result: dict[str, object] = {
        "source_shape": [int(ny), int(nx)],
        "parabolic_required_offsets": [-1, 2],
        "masked_deterministic_donor_offsets": [-1, 2],
        "masked_search_reach":
            "full-cropped-source (WPS FIFO BFS, depth 1200)",
        "staggerings": {},
    }
    coordinates = {
        "mass": grid.latlon_mass(),
        "u": grid.latlon_u(),
        "v": grid.latlon_v(),
    }
    mapped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (latitude, longitude) in coordinates.items():
        y, x = _regular_coordinates(
            snapshot.latitude, snapshot.longitude, latitude, longitude)
        mapped[name] = (y, x)
        floor_y = np.floor(y).astype(np.int64)
        floor_x = np.floor(x).astype(np.int64)
        donor_bounds = {
            "x": [int(floor_x.min()) - 1, int(floor_x.max()) + 2],
            "y": [int(floor_y.min()) - 1, int(floor_y.max()) + 2],
        }
        if (donor_bounds["x"][0] < 0 or donor_bounds["x"][1] >= nx
                or donor_bounds["y"][0] < 0 or donor_bounds["y"][1] >= ny):
            raise ValueError(
                f"GFS crop lacks full parabolic donor halo for {name}")
        result["staggerings"][name] = {
            "source_x_range": [float(x.min()), float(x.max())],
            "source_y_range": [float(y.min()), float(y.max())],
            "parabolic_donor_x": donor_bounds["x"],
            "parabolic_donor_y": donor_bounds["y"],
        }

    # Masked-chain coverage contract (v2): deterministic operators reach the
    # floor-based [-1, +2] stencil; the WPS FIFO search fallback reaches the
    # whole cropped source, so the crop must retain usable donors of both
    # surface classes instead of certifying a fixed radius.
    mass_y, mass_x = mapped["mass"]
    floor_my = np.floor(mass_y).astype(np.int64)
    floor_mx = np.floor(mass_x).astype(np.int64)
    masked_bounds = {
        "x": [int(floor_mx.min()) - 1, int(floor_mx.max()) + 2],
        "y": [int(floor_my.min()) - 1, int(floor_my.max()) + 2],
    }
    if (masked_bounds["x"][0] < 0 or masked_bounds["x"][1] >= nx
            or masked_bounds["y"][0] < 0 or masked_bounds["y"][1] >= ny):
        raise ValueError(
            "GFS crop lacks the complete masked-surface deterministic "
            "donor halo")
    result["masked_surface_donor_x"] = masked_bounds["x"]
    result["masked_surface_donor_y"] = masked_bounds["y"]
    # Class support is reported, not enforced: single-class crops are
    # legitimate and an absent class also fills on the full WPS grid.
    source_landsea = np.asarray(
        snapshot.fields["LANDSEA"], dtype=np.float64)
    result["masked_class_support"] = {
        "land": int(np.sum(source_landsea >= 0.5)),
        "water": int(np.sum(source_landsea < 0.5)),
    }

    lakes = np.asarray(lake_mask, dtype=np.bool_)
    source_land = np.asarray(snapshot.fields["LANDSEA"], dtype=np.float64)
    skin = np.asarray(snapshot.fields["SKINTEMP"], dtype=np.float64)
    water = np.isfinite(source_land) & (source_land < 0.5) & np.isfinite(skin)
    max_nearest_distance = 0.0
    max_search_radius = 0.0
    for j, i in np.argwhere(lakes):
        y = float(mass_y[j, i])
        x = float(mass_x[j, i])
        # Initial window only; the loop doubles it until the nearest water
        # donor is provably inside the searched rectangle.
        search_radius = 8.0
        while True:
            j0 = max(0, int(np.ceil(y - search_radius)))
            j1 = min(ny - 1, int(np.floor(y + search_radius)))
            i0 = max(0, int(np.ceil(x - search_radius)))
            i1 = min(nx - 1, int(np.floor(x + search_radius)))
            rows, columns = np.nonzero(water[j0:j1 + 1, i0:i1 + 1])
            if rows.size:
                rows = rows + j0
                columns = columns + i0
                best_squared = float(
                    np.min((rows - y) ** 2 + (columns - x) ** 2))
                unseen_distance = []
                if j0 > 0:
                    unseen_distance.append(y - (j0 - 1))
                if j1 < ny - 1:
                    unseen_distance.append((j1 + 1) - y)
                if i0 > 0:
                    unseen_distance.append(x - (i0 - 1))
                if i1 < nx - 1:
                    unseen_distance.append((i1 + 1) - x)
                if (not unseen_distance
                        or best_squared < min(unseen_distance) ** 2):
                    break
            if j0 == 0 and j1 == ny - 1 and i0 == 0 and i1 == nx - 1:
                raise ValueError("GFS crop contains no finite source-water lake donor")
            search_radius *= 2.0
        outside_squared = min(x + 1.0, nx - x, y + 1.0, ny - y) ** 2
        if not best_squared < outside_squared:
            raise ValueError(
                "a source-water donor outside the GFS crop could be nearer to a model lake")
        max_nearest_distance = max(max_nearest_distance, best_squared ** 0.5)
        max_search_radius = max(max_search_radius, search_radius)
    result["lake_cells"] = int(lakes.sum())
    result["lake_source_search"] = "global expanding nearest-water"
    result["max_lake_source_search_radius_cells"] = max_search_radius
    result["max_lake_source_water_distance_cells"] = max_nearest_distance
    return result


def _read_series(path: Path) -> tuple[tuple[int, Path], ...]:
    parent = path.parent
    records: list[tuple[int, Path]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        columns = raw.split("\t")
        if not 2 <= len(columns) <= 3:
            raise ValueError(
                "GFS series line "
                f"{line_number} must be "
                "HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]")
        hour = int(columns[0])
        expected_process_id = int(columns[2]) if len(columns) == 3 else 81
        if expected_process_id not in {81, 96}:
            raise ValueError(
                f"GFS series line {line_number} declares uncertified "
                f"forecast process ID {expected_process_id}")
        if hour == 0 and expected_process_id != 81:
            raise ValueError(
                f"GFS series line {line_number} must declare analysis "
                "process ID 81 for f000")
        source = Path(columns[1])
        if not source.is_absolute():
            source = parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"missing GFS series file: {source}")
        records.append((hour, source))
    hours = [hour for hour, _ in records]
    if len(hours) < 2 or hours[0] != 0:
        raise ValueError("GFS series must have at least two times and begin at f000")
    if hours[-1] > 384:
        raise ValueError("GFS series exceeds the certified f384 product horizon")
    deltas = [later - earlier for earlier, later in zip(hours, hours[1:])]
    if not deltas or any(delta <= 0 or delta != deltas[0] for delta in deltas):
        raise ValueError("GFS series cadence must be positive and uniform")
    if deltas[0] not in {1, 3}:
        raise ValueError("GFS series cadence must be exactly 1 or 3 hours")
    return tuple(records)


def _verify_input_manifest(
    path: Path,
    expected_sha256: str,
    roles: Mapping[str, Path],
) -> dict[str, object]:
    observed_manifest_sha256 = _sha256(path)
    if observed_manifest_sha256 != expected_sha256.lower():
        raise ValueError(
            "GFS input manifest SHA mismatch: expected "
            f"{expected_sha256}, got {observed_manifest_sha256}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported GFS direct-input manifest schema")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or set(declared) != set(roles):
        raise ValueError("GFS input manifest role inventory differs from request")
    for role, source in roles.items():
        spec = declared[role]
        if not isinstance(spec, dict) or spec.get("name") != source.name:
            raise ValueError(f"GFS input manifest path mismatch for {role}")
        observed = _sha256(source)
        if spec.get("sha256") != observed:
            raise ValueError(
                f"GFS input digest mismatch for {role}: "
                f"expected {spec.get('sha256')}, got {observed}")
    return manifest


def _parse_gate(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        columns = raw.split("\t", 1)
        if len(columns) != 2 or columns[0] in result:
            raise ValueError("malformed GFS bridge gate")
        result[columns[0]] = columns[1]
    if result.get("status") != "PASS" or result.get("schema") != BRIDGE_SCHEMA:
        raise ValueError("GFS bridge did not publish a recognized PASS gate")
    return result


def _read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape))
    value = np.fromfile(path, dtype="<f4")
    if value.size != expected:
        raise ValueError(
            f"{path.name} contains {value.size} FP32 values; expected {expected}")
    return value.reshape(shape).astype(np.float64)


def _load_bridge_snapshots(
    root: Path,
    cycle: datetime,
    records: tuple[tuple[int, Path], ...],
    expected_source_sha256: Mapping[int, str] | None = None,
) -> tuple[Era5Snapshot, ...]:
    gate = _parse_gate(root / "gate.tsv")
    hours = tuple(int(value) for value in gate["forecast_hours"].split(","))
    if hours != tuple(hour for hour, _ in records):
        raise ValueError("GFS bridge forecast-hour inventory differs from series")
    if gate["cycle"] != cycle.strftime("%Y-%m-%d %H:%M:%S"):
        raise ValueError("GFS bridge cycle differs from requested cycle")
    if tuple(float(value) / 100.0 for value in gate["pressure_levels_pa"].split(",")) != tuple(_PRESSURE_LEVELS_HPA):
        raise ValueError("GFS bridge pressure-level inventory differs from contract")
    expected_identity = {
        "originating_center": "7",
        "master_table_version": "2",
        "local_table_version": "1",
        "forecast_generating_process_ids": ",".join(
            f"{hour}:{81 if hour == 0 else 96}" for hour, _ in records),
    }
    if any(gate.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("GFS bridge product identity differs from contract")
    if gate.get("land_mask_parameter") not in {"0", "218"}:
        raise ValueError("GFS bridge did not record a recognized LAND mask choice")
    if (gate.get("invariant_fields") != "SOURCE_OROGRAPHY,LANDSEA"
            or not gate.get("invariant_fingerprint_fnv1a64")):
        raise ValueError("GFS bridge did not prove invariant terrain/LAND fields")
    try:
        source_sha256 = {
            int(item.split(":", 1)[0]): item.split(":", 1)[1]
            for item in gate["source_sha256"].split(",")
        }
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("GFS bridge source SHA inventory is malformed") from error
    if expected_source_sha256 is None:
        expected_source_sha256 = {
            hour: _sha256(path) for hour, path in records}
    if source_sha256 != dict(expected_source_sha256):
        raise ValueError("GFS bridge decoded source bytes differ from input manifest")
    inventory_path = root / "inventory.tsv"
    if _sha256(inventory_path) != gate.get("inventory_sha256"):
        raise ValueError("GFS bridge inventory SHA mismatch")
    decoded_manifest_path = root / "decoded-sha256.tsv"
    if _sha256(decoded_manifest_path) != gate.get("decoded_manifest_sha256"):
        raise ValueError("GFS decoded-output manifest SHA mismatch")
    nx = int(gate["nx"])
    ny = int(gate["ny"])
    if int(gate["num_data_points"]) != nx * ny:
        raise ValueError("GFS bridge grid point count differs from nx*ny")
    dx = float(gate["dx"])
    dy = float(gate["dy"])
    if dx != 0.25 or dy != 0.25:
        raise ValueError("GFS bridge grid is not pgrb2.0p25")
    lat1 = float(gate["lat1"])
    lon1 = float(gate["lon1"])
    if (float(gate["lat2"]) != lat1 + (ny - 1) * dy
            or float(gate["lon2"]) != lon1 + (nx - 1) * dx):
        raise ValueError("GFS bridge grid endpoints differ from declared increments")
    latitude = lat1 + np.arange(ny, dtype=np.float64) * dy
    # Continuous ascending longitude axis anchored at the crop's western
    # edge wrapped into [-180, 180); a crop crossing the antimeridian
    # simply continues past 180 (e.g. 177.0 .. 181.0).  The horizontal
    # interpolator unwraps every target longitude onto the branch
    # nearest the axis midpoint, so no seam rotation is required -- the
    # previous rotate-and-argsort produced a non-uniform axis for
    # antimeridian crops and failed downstream.
    lon_start = (lon1 + 180.0) % 360.0 - 180.0
    longitude = lon_start + np.arange(nx, dtype=np.float64) * dx
    if not np.all(np.diff(latitude) > 0.0) or not np.all(np.diff(longitude) > 0.0):
        raise ValueError("GFS regular-grid axes are not strictly increasing")

    decoded_rows: dict[tuple[int, str], tuple[int, str, str]] = {}
    lines = decoded_manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "hour\tvariable\tbytes\tsha256\tfilename":
        raise ValueError("GFS decoded-output manifest header is malformed")
    for raw in lines[1:]:
        columns = raw.split("\t")
        if len(columns) != 5:
            raise ValueError("GFS decoded-output manifest row is malformed")
        hour = int(columns[0])
        name = columns[1]
        key = (hour, name)
        if key in decoded_rows:
            raise ValueError("GFS decoded-output manifest contains a duplicate")
        decoded_rows[key] = (int(columns[2]), columns[3], columns[4])
    expected_decoded = {
        (hour, name) for hour, _ in records for name in (*_THREE_D, *_TWO_D)}
    if set(decoded_rows) != expected_decoded:
        raise ValueError("GFS decoded-output manifest inventory differs from contract")
    for (hour, name), (declared_bytes, declared_sha256, declared_name) in decoded_rows.items():
        relative = f"f{hour:03d}/{name}.f32le"
        path = root / relative
        if declared_name != relative or not path.is_file():
            raise ValueError("GFS decoded-output manifest path differs from contract")
        if path.stat().st_size != declared_bytes or _sha256(path) != declared_sha256:
            raise ValueError(f"GFS decoded array identity mismatch for {relative}")

    snapshots = []
    for hour, _ in records:
        time_root = root / f"f{hour:03d}"
        fields: dict[str, np.ndarray] = {}
        for name in _THREE_D:
            field = _read_f32(
                time_root / f"{name}.f32le",
                (_PRESSURE_LEVELS_HPA.size, ny, nx),
            )
            fields[name] = field
        for name in _TWO_D:
            field = _read_f32(time_root / f"{name}.f32le", (ny, nx))
            fields[name] = field
        snapshots.append(Era5Snapshot(
            valid_time=cycle + timedelta(hours=hour),
            levels_hpa=_PRESSURE_LEVELS_HPA,
            latitude=latitude,
            longitude=longitude,
            fields=fields,
        ))
    return tuple(snapshots)


def _prepare_output_root_parent(output_root: Path) -> Path:
    """Make sure the directory that will hold --output-root exists.

    The bridge's scratch space is a TemporaryDirectory in the PARENT of
    ``--output-root`` -- so it can be renamed into place on the same
    filesystem -- and an absent parent surfaced as a bare
    ``FileNotFoundError`` traceback about a random ``gpuwm-gfs-bridge-*``
    name, roughly 40 s into the work, after --dry-run had passed
    cleanly.  Creating the parent is what every user did by hand
    (``mkdir -p``); doing it here keeps the create-only guarantee on
    ``--output-root`` itself, which is the directory that matters.
    """

    parent = output_root.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise NotADirectoryError(
            f"cannot create the parent directory of --output-root "
            f"{output_root}: {error}.  The GFS bridge stages its scratch "
            f"space in {parent} so it can be renamed into place on one "
            "filesystem; choose an --output-root whose parent is writable"
        ) from error
    if not parent.is_dir():
        raise NotADirectoryError(
            f"the parent of --output-root {output_root} is not a "
            f"directory: {parent}")
    return parent


def front_door_physics_selection(
        exp,
        *,
        physics_profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
) -> dict[str, object]:
    """The front door's physics gate, scoped by DOMAIN COUNT.

    One door, two routes, and the routes have never had the same
    physics contract.  The prepared SINGLE-domain forecast runner
    accepts a fixed list of named profiles and compares an experiment's
    switches to one of them for exact equality; the multi-domain
    (domain-tree) runner has no such list and runs the selected suite as
    written.  ``gpuwm domain`` prints that boundary in the note beside
    every config it emits.

    v1.1.0 lost the boundary: the any-combo work made this door call
    :func:`validate_single_domain_physics_profile` for every
    configuration, so the whitelist bound at the front door instead of
    at the runner that owns it.  Three things followed, all of them
    field-observed -- a two-domain config that prepared cleanly on 1.0.1
    was refused; the product default suite is deliberately not in the
    whitelist, so the config the wizard emits BY DEFAULT could not pass
    at all; and the refusal arrived as a stack trace.

    Nothing is widened, and nothing new is refused either.  A single
    domain meets the whitelist exactly as it did in 1.1.0, defaulting to
    WSM6 when the caller named no profile.  A domain tree keeps the
    authority it already had -- the configuration loader's WRF selector
    schema and readiness rails, which every domain passed before an
    experiment existed -- and now records what it selected, per domain,
    in a receipt the proof carries.  An explicitly named profile is
    still enforced on either route, because a gate the caller ASKED for
    is not a gate to drop.
    """

    if len(exp.domains) == 1:
        selection = validate_single_domain_physics_profile(
            WSM6_PROFILE_ID if physics_profile is None else physics_profile,
            config=exp.root.run,
            expert_acknowledgements=expert_acknowledgements)
        _refuse_unoffered_land_surface(
            {1: selection.get("selectors")})
        return selection
    selection = multi_domain_physics_selection(
        {int(domain.grid_id): domain.run for domain in exp.domains},
        profile=physics_profile,
        expert_acknowledgements=expert_acknowledgements)
    _refuse_unoffered_land_surface({
        int(grid_id): domain.get("selectors")
        for grid_id, domain in selection["domains"].items()})
    return selection


def _refuse_unoffered_land_surface(selectors_by_domain) -> None:
    """Refuse a land-surface component this route does not offer.

    Preparation time, before any GRIB is decoded: the registry has long
    declared which templates this route offers this source, and nothing
    consulted that declaration before a run.  RUC was therefore
    selectable here, prepared in full -- proof PASS, nine soil layers,
    339 MB of state -- and then died on its first surface-temperature
    call having advanced no model time.  A refusal that arrives after
    the preparation is a refusal that costs the preparation.

    Both routes of this front door are checked, because both build
    their surface state from the same GFS initialisation and both reach
    the same cold start.  The refusal names only this source; see
    :func:`land_surface_route_blocker` for why it speaks for no other.
    """

    for grid_id in sorted(selectors_by_domain):
        selectors = selectors_by_domain[grid_id]
        if not isinstance(selectors, Mapping):
            continue
        component = land_surface_component_for_selector(
            selectors.get("sf_surface_physics"))
        if component is None:
            # A selector the registry has no option for is refused by
            # the configuration schema long before this door, so there
            # is nothing here for this gate to add.
            continue
        blocker = land_surface_route_blocker(component, source="gfs")
        if blocker is not None:
            raise ValueError(f"d{grid_id:02d}: {blocker}")


def prepare_gfs_wrf(
    *,
    series: Path,
    cycle: str,
    bridge: Path,
    wps_namelist: Path,
    static_input: Path | None,
    static_receipt: Path | None,
    experiment_config: Path,
    input_manifest: Path,
    input_manifest_sha256: str,
    output_root: Path,
    preprocess_backend: str = "cuda",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: Path | None = None,
    geog_root: Path | None = None,
    hierarchy_workers: int | None = None,
    physics_profile: str | None = None,
    expert_acknowledgements: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build native GFS initial/boundary files and return the proof receipt.

    ``geog_root`` activates the internal multi-domain route when the
    experiment declares children.  The existing public single-domain CLI is
    unchanged; source-neutral public hierarchy configuration is owned by the
    higher-level integration layer.
    """

    cycle_time = datetime.strptime(cycle, "%Y-%m-%d_%H:%M:%S")
    if (cycle_time.hour not in {0, 6, 12, 18}
            or cycle_time.minute != 0 or cycle_time.second != 0):
        raise ValueError("GFS cycle must be an exact 00/06/12/18 UTC cycle")
    records = _read_series(Path(series))
    base_roles = {
        "series": Path(series),
        "bridge": Path(bridge),
        "wps_namelist": Path(wps_namelist),
        "experiment_config": Path(experiment_config),
    }
    static_pair = static_input is not None or static_receipt is not None
    if static_pair and (static_input is None or static_receipt is None):
        raise ValueError(
            "GFS static-input and static-receipt must be supplied together")
    if static_input is not None:
        base_roles["static_input"] = Path(static_input)
        base_roles["static_receipt"] = Path(static_receipt)
    elif geog_root is None:
        raise ValueError(
            "GFS requires either static-input/static-receipt or geog-root")
    grib_roles = {f"grib-f{hour:03d}": path for hour, path in records}
    roles = {**base_roles, **grib_roles}
    for role, path in roles.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing GFS adapter input {role}: {path}")
    if not os.access(Path(bridge), os.X_OK):
        raise PermissionError("GFS Rust bridge is not executable")
    if Path(output_root).exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    _prepare_output_root_parent(Path(output_root))
    manifest = _verify_input_manifest(
        Path(input_manifest), input_manifest_sha256, roles)
    manifest_digest = input_manifest_sha256.lower()
    source_identity = manifest.get("source")
    expected_source_identity = {
        "model": "GFS",
        "product": "pgrb2.0p25",
        "cycle": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if (not isinstance(source_identity, dict)
            or any(source_identity.get(key) != value
                   for key, value in expected_source_identity.items())):
        raise ValueError(
            "GFS input manifest source identity differs from model/product/cycle")
    implementation_sha256 = _implementation_sha256()
    git_source_identity = _git_source_identity()
    decoder_digest = manifest["files"]["bridge"]["sha256"]
    experiment_config_digest = manifest["files"]["experiment_config"]["sha256"]
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_preprocess_bridge)
    preprocess_receipt = preprocess.receipt()

    total_started = time.perf_counter()
    exp = load_experiment(Path(experiment_config))
    physics_selection = front_door_physics_selection(
        exp, physics_profile=physics_profile,
        expert_acknowledgements=expert_acknowledgements)
    if exp.start_time != cycle_time:
        raise ValueError("GFS cycle must equal experiment start_time")
    hours = [hour for hour, _ in records]
    if hours[-1] * 3600 < exp.run_seconds:
        raise ValueError("GFS series does not cover the configured run")
    boundary_interval_seconds = (hours[1] - hours[0]) * 3600
    cfg = exp.root.run
    if len(exp.domains) == 1:
        grid = _validate_grid_and_vertical_contract(exp, Path(wps_namelist))
        grids = (grid,)
    else:
        if geog_root is None:
            raise ValueError(
                "nested GFS preparation requires an explicit geog_root")
        grids = validate_native_lambert_contracts(
            exp, Path(wps_namelist), source_name="GFS",
            source_top_pressure_pa=float(
                _PRESSURE_LEVELS_HPA.min() * 100.0),
        )
        grid = grids[0]
    if static_input is None:
        static, root_static_receipt = _static_from_geog(
            Path(wps_namelist), Path(geog_root), grid, cfg)
        root_static_provider = "native-wps-geog"
    else:
        root_static_receipt = _verify_static_receipt(
            Path(static_receipt), Path(static_input), grid, cfg)
        static = _load_static(Path(static_input), grid, cfg.ny, cfg.nx)
        root_static_provider = "prebuilt-hash-bound-cache"

    decode_started = time.perf_counter()
    with tempfile.TemporaryDirectory(
            prefix="gpuwm-gfs-bridge-", dir=Path(output_root).parent) as temporary:
        decoded = Path(temporary) / "decoded"
        completed = subprocess.run(
            [str(Path(bridge)), "--series", str(Path(series)), str(decoded),
             cycle_time.strftime("%Y-%m-%d %H:%M:%S")],
            check=False, text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "GFS Rust bridge failed: " + completed.stderr.strip())
        expected_source_sha256 = {
            hour: manifest["files"][f"grib-f{hour:03d}"]["sha256"]
            for hour, _ in records
        }
        snapshots = _load_bridge_snapshots(
            decoded, cycle_time, records, expected_source_sha256)
        decode_seconds = time.perf_counter() - decode_started
        lake_mask = static["LU_INDEX"] == 21
        coverage_receipt = {
            f"f{hour:03d}": _source_coverage_receipt(snapshot, grid, lake_mask)
            for (hour, _), snapshot in zip(records, snapshots)
        }

        from gpuwm.core.grid import make_vertical_coord

        initialize_started = time.perf_counter()
        results = []
        horizontal = []
        interpolation_landmask = np.asarray(
            static["LANDMASK"] >= 0.5, dtype=np.bool_).copy()
        # GEOG lakes can be much smaller than a 0.25-degree GFS cell.  Select
        # their atmospheric/soil source as land here, then apply the existing
        # explicit nearest-source-water lake initialization below.
        interpolation_landmask[lake_mask] = True
        for source in snapshots:
            met = interpolate_era5_to_lambert(
                source, grid,
                target_landmask=interpolation_landmask,
                relative_humidity_convention="water",
                backend=preprocess,
            )
            coord = make_vertical_coord(
                cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
                eta_levels=exp.vertical.eta_levels)
            initialized = initialize_real(
                met, cfg, coord, static["HGT_M"],
                p_top=exp.vertical.p_top, sfcp_to_sfcp=True,
                preprocess_backend=preprocess,
                state_backend="preprocess")
            initialized.state.set_map_coriolis(
                static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
                static["F"], static["E"], sina=static["SINALPHA"],
                cosa=static["COSALPHA"])
            horizontal.append(met)
            results.append(initialized)
        times = tuple(snapshot.valid_time for snapshot in snapshots)
        boundaries = build_state_lateral_boundaries(
            [value.state for value in results], times,
            spec_bdy_width=cfg.spec_bdy_width,
            spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
        attach_lateral_boundaries(results[0].state, boundaries)
        lake_skin = interpolate_lake_skin_temperature(
            snapshots[0], grid, lake_mask)
        soil = preprocess_land_surface_soil(
            horizontal[0].fields,
            sf_surface_physics=int(cfg.sf_surface_physics),
            num_soil_layers=int(cfg.num_soil_layers),
            soil_type=static["SCT_DOM"],
            deep_soil_temperature=static["TMN"], lake_mask=lake_mask,
            lake_skin_temperature=lake_skin)
        initialize_seconds = time.perf_counter() - initialize_started

        native_source_identity = {
            "adapter": "gfs-pgrb2-0p25-direct-v1",
            "input_manifest_schema": manifest["schema"],
            "input_manifest_sha256": manifest_digest,
            "decoder": {
                "name": Path(bridge).name,
                "sha256": decoder_digest,
                "implementation": "gpuwm-all-rust-gfs-grib2-bridge",
            },
            "relative_humidity_convention": "GFS water",
            "soil_mapping": "exact GFS Noah 4-layer copy",
            "initial_hydrometeors": (
                "explicit zero (WRF Vtable.GFS parity)"),
            "implementation_sha256": implementation_sha256,
            "git_source_identity": git_source_identity,
            "preprocessing": preprocess_receipt,
        }

        staging = _atomic_staging_sibling(Path(output_root))
        if staging.exists():
            raise FileExistsError(f"stale GFS staging directory exists: {staging}")
        staging.mkdir(parents=True)
        try:
            shutil.copy2(decoded / "gate.tsv", staging / "decoder-gate.tsv")
            shutil.copy2(
                decoded / "inventory.tsv", staging / "decoder-inventory.tsv")
            shutil.copy2(
                decoded / "decoded-sha256.tsv",
                staging / "decoder-sha256.tsv")
            static_cache = staging / "native-static.npz"
            geometry_receipt = staging / "geometry-receipt.json"
            portable_source_manifest = staging / "source-input-manifest.json"
            prepared_cache = staging / "prepared-cache"
            wrf_output = staging / "wrf-native-input"
            shutil.copy2(Path(input_manifest), portable_source_manifest)
            _write_static_cache(static_cache, static)
            _write_geometry_receipt(geometry_receipt, grid, cfg, static_cache)
            if len(exp.domains) > 1:
                selected_workers = hierarchy_workers
                if selected_workers is None:
                    selected_workers = (
                        8 if preprocess_receipt["backend"] == "cpu" else 1)
                hierarchy_started = time.perf_counter()
                hierarchy = initialize_and_export_regular_source_hierarchy(
                    exp=exp, grids=grids, snapshots=snapshots,
                    forcing_hours=hours,
                    wps_namelist=Path(wps_namelist),
                    geog_root=Path(geog_root), source_name="GFS",
                    artifact_output=staging / "hierarchy-artifacts",
                    wrf_output=staging / "wrf-native-input",
                    root_initial_result=results[0], root_met=horizontal[0],
                    root_soil=soil, root_static_fields=static,
                    root_boundaries=boundaries,
                    bridge_manifest_sha256=manifest_digest,
                    source_manifest_sha256=manifest_digest,
                    namelist_sha256=experiment_config_digest,
                    source_identity=native_source_identity,
                    source_inventory=tuple(snapshots[0].fields),
                    workers=selected_workers,
                    preprocess_backend=preprocess_receipt["backend"],
                    cpu_bridge=cpu_preprocess_bridge,
                    input_provenance={
                        "input_manifest_sha256": manifest_digest,
                        "decoder_sha256": decoder_digest,
                        "preprocessing": preprocess_receipt,
                    },
                    artifact_manifest_reference=(
                        "../hierarchy-artifacts/domain-artifacts.json"),
                )
                hierarchy_seconds = time.perf_counter() - hierarchy_started
                final_manifest = _verify_input_manifest(
                    Path(input_manifest), manifest_digest, roles)
                if final_manifest != manifest:
                    raise ValueError(
                        "GFS input manifest content changed during preparation")
                if _implementation_sha256() != implementation_sha256:
                    raise ValueError(
                        "GFS adapter implementation changed during preparation")
                proof = {
                    "schema": HIERARCHY_PROOF_SCHEMA,
                    "status": "READY_NOT_YET_STOCK_WRF_GATED",
                    "domain_count": len(exp.domains),
                    "physics": physics_selection,
                    "forcing_times": [value.isoformat() for value in times],
                    "forcing_hours": hours,
                    "boundary_interval_seconds": boundary_interval_seconds,
                    "input_manifest_sha256": manifest_digest,
                    "decoder_sha256": decoder_digest,
                    "implementation_sha256": implementation_sha256,
                    "git_source_identity": git_source_identity,
                    "preprocessing": preprocess_receipt,
                    "root_static_provider": root_static_provider,
                    "root_static_receipt": root_static_receipt,
                    "hierarchy_workers": selected_workers,
                    "source_coverage": coverage_receipt,
                    "decoder_stdout": completed.stdout.strip(),
                    "static_catalog": dict(
                        hierarchy.static_catalog_receipt),
                    "hierarchy_source_coverage": dict(
                        hierarchy.source_coverage_receipt),
                    "hierarchy_topology": dict(
                        hierarchy.topology_receipt),
                    "artifact_receipt": dict(
                        hierarchy.hierarchy.artifacts.receipt),
                    "wrf_manifest": dict(
                        hierarchy.hierarchy.wrf_manifest),
                    "timing_seconds": {
                        "decode": decode_seconds,
                        "initialize_all_root_times": initialize_seconds,
                        **dict(hierarchy.hierarchy.timings_seconds),
                        "hierarchy_call_wall": hierarchy_seconds,
                        "total": time.perf_counter() - total_started,
                    },
                }
                (staging / "proof.json").write_text(
                    json.dumps(proof, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
                os.replace(staging, Path(output_root))
                return proof
            identity = prepared_cache_identity(
                bridge_manifest_sha256=manifest_digest,
                source_manifest_sha256=manifest_digest,
                static_cache_sha256=_sha256(static_cache),
                namelist_sha256=experiment_config_digest,
                domain_config=exp.root,
                forcing_hours=hours,
                source_identity=native_source_identity,
            )
            cache_started = time.perf_counter()
            cache_receipt = write_prepared_cache(
                prepared_cache, identity=identity,
                initial_result=results[0], met=horizontal[0],
                boundaries=boundaries, surface=_canonical_surface(soil),
                metadata={
                    "source_adapter": "gfs",
                    "initial_valid_time": times[0].isoformat(),
                    "last_valid_time": times[-1].isoformat(),
                    "forcing_hours": hours,
                    "boundary_interval_seconds": boundary_interval_seconds,
                    "preprocessing": preprocess_receipt,
                },
            )
            cache_seconds = time.perf_counter() - cache_started
            portable_cache_receipt = dict(cache_receipt)
            portable_cache_receipt["path"] = prepared_cache.name
            export_started = time.perf_counter()
            export_receipt = export_prepared_wrf(
                prepared_cache, static_cache, geometry_receipt, wrf_output,
                valid_time=times[0],
                boundary_interval_seconds=boundary_interval_seconds,
                # The RESOLVED profile, not the caller's argument: the
                # single-domain route defaults to WSM6 when nobody named
                # one, and the exporter must be handed the same profile
                # the gate just checked or its receipt cannot match.
                physics_profile=physics_selection["profile"],
                expert_acknowledgements=expert_acknowledgements)
            if (export_receipt.get("schema")
                    != "gpuwm-native-direct-wrf-export-v3"
                    or export_receipt.get("physics") != physics_selection):
                raise RuntimeError(
                    "direct-WRF export physics provenance differs from the "
                    "selected preparation profile")
            export_seconds = time.perf_counter() - export_started
            final_manifest = _verify_input_manifest(
                Path(input_manifest), manifest_digest, roles)
            if final_manifest != manifest:
                raise ValueError(
                    "GFS input manifest content changed during preparation")
            if _implementation_sha256() != implementation_sha256:
                raise ValueError(
                    "GFS adapter implementation changed during preparation")
            preprocessing_receipt_sha256 = hashlib.sha256(json.dumps(
                preprocess_receipt, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            initialization_artifacts = {
                "source_manifest": {
                    "path": portable_source_manifest.name,
                    "bytes": portable_source_manifest.stat().st_size,
                    "sha256": _sha256(portable_source_manifest),
                },
                "static_cache": {
                    "path": static_cache.name,
                    "bytes": static_cache.stat().st_size,
                    "sha256": _sha256(static_cache),
                },
                "geometry_receipt": {
                    "path": geometry_receipt.name,
                    "bytes": geometry_receipt.stat().st_size,
                    "sha256": _sha256(geometry_receipt),
                },
                "prepared_cache": {
                    "path": prepared_cache.name,
                    "content_sha256": portable_cache_receipt["content_sha256"],
                    "payload_bytes": portable_cache_receipt["payload_bytes"],
                },
                "wrf_files": {
                    name: {
                        "path": f"{wrf_output.name}/{name}",
                        **details,
                    }
                    for name, details in export_receipt["files"].items()
                },
            }
            proof = {
                "schema": PROOF_SCHEMA,
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                "forcing_times": [value.isoformat() for value in times],
                "forcing_hours": hours,
                "boundary_interval_seconds": boundary_interval_seconds,
                "input_manifest_sha256": manifest_digest,
                "decoder_sha256": decoder_digest,
                "implementation_sha256": implementation_sha256,
                "git_source_identity": git_source_identity,
                "preprocessing": preprocess_receipt,
                "preprocessing_receipt_sha256": preprocessing_receipt_sha256,
                "source_inputs": {
                    "manifest_schema": manifest["schema"],
                    "manifest_sha256": manifest_digest,
                    "files": manifest["files"],
                },
                "initialization_artifacts": initialization_artifacts,
                "source_coverage": coverage_receipt,
                "decoder_stdout": completed.stdout.strip(),
                "prepared_cache": portable_cache_receipt,
                "physics": physics_selection,
                "export": export_receipt,
                "timing_seconds": {
                    "decode": decode_seconds,
                    "initialize_all_times": initialize_seconds,
                    "write_prepared_cache": cache_seconds,
                    "direct_wrf_export": export_seconds,
                    "total": time.perf_counter() - total_started,
                },
            }
            (staging / "proof.json").write_text(
                json.dumps(proof, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(staging, Path(output_root))
            return proof
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def prepared_forecast_next_command(
        proof: Mapping[str, object], *, output_root, experiment_config,
        wps_namelist, source: str = "gfs") -> list[str]:
    """The ready-to-run forecast command, hashes filled in.

    ``gpuwm fetch --author-front-door-manifest`` already prints the
    complete ``rw-wps`` line with its digest filled in, and it is the
    single most-praised thing in the pilots.  Nothing did the equivalent
    here: the front door finished by dumping 42 KB of proof.json to
    stdout and printing no next step at all, so every user hand-extracted
    three SHA-256 values -- one of them only findable by grepping the
    JSON for ``content_sha256`` -- before they could start a forecast.

    Everything below is already known to this process.  Emitted on
    stderr so the proof document on stdout stays machine-readable.

    **Every line printed as a command must run when pasted, exactly as
    printed.**  v1.0.0 printed ``--physics-profile <the profile this
    config was materialized for>`` and ``--outdir OUTPUT_DIR``: two
    placeholders in a command whose entire value was that a user did not
    have to reconstruct it, and the second one is a shell metacharacter
    error rather than an honest gap.  Both are resolved here -- the
    profile by asking the same table the runner's guard asks, the outdir
    by naming a real directory beside the preparation.  Where a value
    genuinely cannot be resolved, this prints prose saying so instead of
    a command that cannot be run.

    When the proof carries a ``gpuwm-front-door-physics-selection-v1``
    receipt (proof v3 and later), that receipt IS what was materialized,
    so it names the profile.  Re-deriving it from the config would be a
    second opinion about a decision already recorded.  The config-table
    lookup stays as the fallback for the historical v2 proof, which
    carries no physics receipt.

    ``source`` names the prepared-forecast runner's own ``--source``
    value.  Every native front door that reaches a prepared bundle can
    print this, and one of them printing a complete hash-bound command
    while another printed nothing at all was a node-8 finding in its own
    right: the 20CRv3 route ran end to end and then said nothing about
    how to run the forecast.
    """

    from gpuwm.experiment import load_experiment
    from gpuwm.physics_compat import identify_single_domain_profile

    output_root = Path(output_root)
    proof_path = output_root / "proof.json"
    lines = [
        "",
        "rw-wps: preparation complete.  Run the forecast with:",
    ]
    try:
        proof_digest = _sha256(proof_path)
    except OSError:
        return lines + [
            f"  (cannot read {proof_path}; no next command to print)"]
    # A real, pasteable directory: a sibling of the preparation, so the
    # command works from anywhere and does not overwrite the inputs.
    #
    # It has to be a genuine SIBLING, not a child.  Both runners declare
    # --prepared-root a protected input and refuse any --outdir that
    # overlaps it in either direction, so the <root>/forecast this used
    # to print was a command the runner rejected -- which is how a node-8
    # pilot got a traceback out of the front door's own suggestion.
    outdir = output_root.parent / f"{output_root.name}-forecast"
    cache = proof.get("prepared_cache")
    content = (cache.get("content_sha256")
               if isinstance(cache, Mapping) else None)
    manifest_digest = proof.get("input_manifest_sha256")
    if not isinstance(content, str) or not isinstance(manifest_digest, str):
        # The multi-domain hierarchy product.  Its runner binds the
        # experiment config by digest too, so the full command is
        # printable here rather than a fragment the user completes.
        config_path = Path(experiment_config)
        try:
            config_digest = _sha256(config_path)
        except OSError:
            return lines + [
                f"  (cannot read {config_path}; no next command to print)"]
        return lines + [
            "  python tools/prepared_domain_tree_forecast.py \\",
            f"      --prepared-root {output_root} \\",
            f"      --preparation-receipt-sha256 {proof_digest} \\",
            f"      --experiment-config {config_path} \\",
            f"      --experiment-config-sha256 {config_digest} \\",
            f"      --io-mode history --outdir {outdir}",
            "  (this proof carries no single prepared-cache identity "
            "because it is a multi-domain hierarchy product.)",
        ]
    physics = proof.get("physics")
    profile = physics.get("profile") if isinstance(physics, Mapping) else None
    if not isinstance(profile, str):
        try:
            profile = identify_single_domain_profile(
                load_experiment(Path(experiment_config)).root.run)
        except Exception:  # a config this process cannot load is not a command
            profile = None
    if profile is None:
        return lines + [
            f"  (the physics in {Path(experiment_config)} matches none of "
            "the profiles the prepared single-domain runner accepts, so "
            "no runnable command can be printed for it.  Re-emit the "
            "config with `gpuwm domain --physics-profile <id>` -- "
            "`gpuwm domain --help` lists the ids -- and the runner will "
            "then accept it.)"]
    lines += [
        "  python tools/prepared_single_domain_forecast.py \\",
        f"      --source {source} \\",
        f"      --prepared-root {output_root} \\",
        f"      --proof-sha256 {proof_digest} \\",
        f"      --source-manifest-sha256 {manifest_digest} \\",
        f"      --prepared-content-sha256 {content} \\",
        f"      --experiment-config {Path(experiment_config)} \\",
        f"      --wps-namelist {Path(wps_namelist)} \\",
        f"      --physics-profile {profile} \\",
        f"      --io-mode history --outdir {outdir}",
        "  (--run-seconds and --history-interval-seconds default to the "
        "hash-bound experiment; the runner refuses any other value.)",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--static-input", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="cuda")
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--geog-root", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    parser.add_argument(
        "--physics-profile", default=None,
        help="registered fixed physics template materialized in the "
             "experiment; the single-domain route defaults to "
             f"{WSM6_PROFILE_ID} and a domain tree, which has no profile "
             "whitelist, defaults to the suite the config selects")
    parser.add_argument(
        "--expert-acknowledgement", action="append", default=[],
        help="registry-owned expert acknowledgement id; repeat as needed")
    args = parser.parse_args(argv)
    try:
        proof = prepare_gfs_wrf(
            series=args.series, cycle=args.cycle, bridge=args.bridge,
            wps_namelist=args.wps_namelist, static_input=args.static_input,
            static_receipt=args.static_receipt,
            experiment_config=args.experiment_config,
            input_manifest=args.input_manifest,
            input_manifest_sha256=args.input_manifest_sha256,
            output_root=args.output_root,
            preprocess_backend=args.preprocess_backend,
            preprocess_workers=args.preprocess_workers,
            cpu_preprocess_bridge=args.cpu_preprocess_bridge,
            geog_root=args.geog_root,
            hierarchy_workers=args.hierarchy_workers,
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.expert_acknowledgement),
        )
    except (ValueError, OSError) as error:
        # Every gate this door applies is a refusal, and a refusal is a
        # sentence.  A pilot met three of these as stack traces -- a
        # missing input, a manifest bound to a different namelist, and
        # the mis-scoped physics whitelist this release fixes -- and a
        # traceback tells nobody which of their inputs to change.  The
        # gates themselves are untouched: only how they reach the reader
        # is.  RC 2 is what the 20CRv3 door already costs for the same
        # class of refusal, and `rw-wps` passes it straight through.
        print(f"rw-wps --source gfs: {error}.", file=sys.stderr)
        return 2
    print(json.dumps(proof, indent=2, sort_keys=True))
    for line in prepared_forecast_next_command(
            proof, output_root=args.output_root,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist):
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
