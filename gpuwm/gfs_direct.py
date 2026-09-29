"""GFS GRIB2 -> native GPU initialization -> stock-WRF direct export."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np

from gpuwm import __version__
from gpuwm.bridges import decode_failure_message
from gpuwm.explain import warn
from gpuwm.vertical_adaptation import (
    adapt_experiment_for_statics,
    vertical_coordinate_receipt as _vertical_coordinate_receipt,
)
from gpuwm.era5_direct import (
    _canonical_surface,
    _load_static,
    _static_from_geog,
    _write_geometry_receipt,
    _write_static_cache,
)
from gpuwm.experiment import load_experiment
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.source_coverage import (ForcingSeriesRefusal,
                                          PreparationRefusal,
                                          owns_source_coverage_refusal)
from gpuwm.ingest.horiz import (
    _regular_coordinates,
    interpolate_era5_to_lambert,
    interpolate_lake_skin_temperature,
    orient_global_source_longitudes,
    unrolled_source_ring,
)
from gpuwm.ingest.lateral_bc import (
    StateBoundaryFrames,
    attach_lateral_boundaries,
    start_last_forcing_order,
)
from gpuwm.ingest.soil_downscale import (
    declared_soil_texture_downscale, soil_mesh_plan_from_case)
from gpuwm.ingest.boundary_stream import (
    PreparedTreeWriter,
    producer_device_bytes,
    remove_unfinished_tree,
)
from gpuwm.ingest.prepared_cache import (
    prepared_cache_identity,
)
from gpuwm.ingest.cpu_backend import host_step_workers
from gpuwm.ingest.memory_refusal import InitializationMemoryRefused
from gpuwm.ingest.preparation_price import (
    price_forcing_preparation, price_preparation_floor)
from gpuwm.ingest.preprocess_backend import (
    admit_preparation,
    release_backend_memory,
    resolve_preprocess_backend,
)
from gpuwm.ingest.real import initialize_real
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.ingest.soil import (door_reconciled_soil_category,
                               soil_source_orography)
from gpuwm.ingest.water_temperature import (
    MODIS_LAKE_CATEGORY, WaterTemperatureStatics,
    announce_water_temperature, assemble_for_route)
from gpuwm.moisture_floor_receipt import moisture_floor_proof_entry
from gpuwm.native_domain_artifacts import (
    _atomic_staging_sibling, published_path_refusal)
from gpuwm.native_wrf_contract import (
    native_geometry_contract,
    require_land_terrain,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    verify_native_static_receipt,
)
from gpuwm.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
)
from gpuwm.open_files import (
    ensure_descriptor_headroom,
    required_descriptors,
)
from gpuwm.physics_compat import (
    acknowledgement_delivery,
    multi_domain_physics_selection,
    single_domain_physics_selection,
)
from gpuwm.wrf_direct import (
    StockWrfExportUnsupported,
    export_prepared_wrf,
    stock_wrf_export_not_requested,
    stock_wrf_export_refused,
)


INPUT_MANIFEST_SCHEMA = "gpuwm-gfs-direct-input-manifest-v1"
BRIDGE_SCHEMA = "gpuwm-gfs-grib2-bridge-v1"

#: The live progress document this stage publishes while it runs.
#: Shaped like the forecast runner's ``progress.json`` -- ``schema`` and
#: ``status`` -- because ``gpuwm go``'s heartbeat reads ``status`` off
#: whichever file the running stage names, and a second shape would mean
#: a second reader.
PREPARE_PROGRESS_SCHEMA = "gpuwm.prepare-progress/v1"

#: The phases this stage announces, in the order it passes through them.
#: Named as one tuple so the publisher, the acceptance test and any
#: reader agree about what "how far in is it" can say.
PREPARE_PHASES = (
    "verify_inputs", "static_build", "decode", "initialize_all_times",
    "write_prepared_cache", "direct_wrf_export", "publish",
)
PROOF_SCHEMA = "gpuwm-gfs-direct-wrf-proof-v3"
LEGACY_PROOF_SCHEMA = "gpuwm-gfs-direct-wrf-proof-v2"
#: The multi-domain product.  v2 adds the front-door physics receipt the
#: v3 single-domain proof already carried; v1 documents written by 1.1.0
#: and earlier stay independently verifiable and are never promoted to
#: v2 by inference, exactly as the direct proof's v2 is not.
HIERARCHY_PROOF_SCHEMA = "gpuwm-gfs-native-hierarchy-proof-v2"
LEGACY_HIERARCHY_PROOF_SCHEMA = "gpuwm-gfs-native-hierarchy-proof-v1"
#: The isobaric ladder every GFS subset carries, and the floor this
#: front door requires -- not the ceiling.  A case whose model top sits
#: above 100 hPa is fetched with the ladder EXTENDED UPWARD
#: (``gpuwm fetch --source gfs --p-top-pa 5000`` adds the 70 and 50 hPa
#: levels), and the bridge reports the ladder it actually decoded in its
#: gate.  This constant is what that ladder must still contain, and what
#: stands in for a directory fetched before the ladder was recorded.
_CERTIFIED_PRESSURE_LEVELS_HPA = np.asarray(
    [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
     650, 700, 750, 800, 850, 900, 925, 950, 975, 1000],
    dtype=np.float64,
)
#: Kept as the historical spelling: several tests and downstream readers
#: import it by this name, and it means exactly what it always did --
#: the certified 21-level ladder.
_PRESSURE_LEVELS_HPA = _CERTIFIED_PRESSURE_LEVELS_HPA
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
    "gpuwm/vertical_adaptation.py",
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
    "gpuwm/static/highres_production.py",
    "gpuwm/static/highres.py",
    "gpuwm/static/highres_fetch.py",
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


#: The GFS adapter's name in every water-temperature refusal and
#: receipt.
_WATER_ROUTE = "the GFS direct route"


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
        # absent inventory accurately instead of demanding one.
        if os.environ.get("GPUWM_NATIVE_DISTRIBUTION_MANIFEST"):
            manifest_path, _ = _distribution_manifest()
            result["distribution/manifest.json"] = _sha256(manifest_path)
        result["distribution/repo_only_paths.json"] = hashlib.sha256(
            json.dumps(missing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return result


class GfsRouteError(ValueError):
    """A refusal by this door, with a remedy the reader can act on.

    Subclasses ValueError like every other refusal class in the tree
    (``GoRefusal``, ``HrrrRouteInputError``, ``ManifestError``, ...) so
    ``main``'s handler renders it as one sentence at rc 2.  The point of
    the distinct type is what it is NOT used for: a violated internal
    invariant stays a bare ``RuntimeError`` and keeps its traceback,
    because that is a defect here rather than a problem with the inputs,
    and flattening the two together would report our bug as the reader's
    mistake.
    """


def _distribution_manifest() -> tuple[Path, dict[str, object]]:
    """The sealed manifest, validated against its WHOLE schema."""

    from gpuwm.runtime_manifest import manifest_from_environment

    bound = manifest_from_environment()
    if bound is None:
        raise GfsRouteError(
            "installed GFS runtime requires GPUWM_NATIVE_DISTRIBUTION_MANIFEST")
    return bound


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
    # record the identity as accurately unavailable instead of refusing to
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


#: The source top a fetch directory implies when its receipt predates the
#: recorded ladder: the certified ladder is the only thing it can have
#: been, because nothing before this release could fetch another.
_CERTIFIED_SOURCE_TOP_PA = float(_CERTIFIED_PRESSURE_LEVELS_HPA.min() * 100.0)


def _validate_ladder(levels: np.ndarray, context: str) -> np.ndarray:
    """The certified ladder, optionally extended upward.  Or a refusal.

    Mirrors ``gfs_grib2_bridge::observed_pressure_levels`` on the Python
    side of the same handoff, so the two ends of the decode cannot
    disagree about what a legal ladder is.  Two clauses say it all: the
    ladder is strictly increasing, and it ENDS with the certified 21
    exactly.  Together those force every extra level strictly above the
    certified top -- an increasing list whose last 21 begin at 100 hPa
    cannot carry anything at or below 100 hPa ahead of them -- so an
    interior insertion, a dropped certified level, and a reordering are
    all refused without a third clause that could never fire.
    """

    if levels.ndim != 1 or levels.size < _CERTIFIED_PRESSURE_LEVELS_HPA.size:
        raise ValueError(
            f"{context}: {levels.size} isobaric level(s), fewer than the "
            f"{_CERTIFIED_PRESSURE_LEVELS_HPA.size} every GFS subset carries")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError(
            f"{context}: isobaric ladder is not strictly increasing")
    tail = levels[-_CERTIFIED_PRESSURE_LEVELS_HPA.size:]
    if not np.allclose(tail, _CERTIFIED_PRESSURE_LEVELS_HPA, rtol=0.0,
                       atol=1e-6):
        raise ValueError(
            f"{context}: isobaric ladder does not end with the certified "
            f"{_CERTIFIED_PRESSURE_LEVELS_HPA.size}-level ladder")
    return levels


def _manifest_source_top_pa(manifest: Mapping[str, object]) -> float:
    """The source top the fetch receipt declares, in Pa.

    The vertical contract is checked BEFORE the bridge runs, so the model
    top a case may ask for has to come from the receipt rather than from
    a constant -- a constant is exactly what capped every GFS run at
    10000 Pa regardless of what was actually fetched.
    """

    identity = manifest.get("source")
    if not isinstance(identity, dict):
        return _CERTIFIED_SOURCE_TOP_PA
    levels = identity.get("pressure_levels_hpa")
    declared = identity.get("top_pressure_pa")
    if not isinstance(levels, list) or not levels:
        return _CERTIFIED_SOURCE_TOP_PA
    try:
        ladder = np.asarray([float(level) for level in levels],
                            dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "GFS input manifest declares an unreadable pressure ladder"
        ) from error
    _validate_ladder(ladder, "GFS input manifest")
    top = float(ladder.min() * 100.0)
    if isinstance(declared, (int, float)) and not math.isclose(
            float(declared), top, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"GFS input manifest declares a source top of {float(declared):g} "
            f"Pa but its pressure ladder tops out at {top:g} Pa")
    return top


def _refuse_a_fetch_below_the_model_top(exp, source_top_pressure_pa: float):
    """Refuse a download that stops under the config's model top, with the way out.

    Every level above the source top would be extrapolated past the top
    of the analysis, which is the vertical contract's own refusal.  Said
    here first because on this route the way out is a fetch flag, not the
    eta ladder: ``gpuwm go`` and ``gpuwm run-plan`` ask for the top on
    their own, so only a folder fetched by hand gets here.

    It is the vertical-ladder member of the preparation refusal family,
    so the door prints it as the refusal and its own remedy at the
    family's exit status, the way it printed the vertical contract's
    refusal for the same breakage, rather than as one flat error line.
    """

    from gpuwm.ingest.source_coverage import VerticalLadderRefusal
    from gpuwm.source_adapters import fetch_model_top_pa

    p_top = float(exp.vertical.p_top)
    source_top = float(source_top_pressure_pa)
    if not source_top > p_top:
        return
    asked = fetch_model_top_pa("gfs", p_top)
    if asked is None:
        remedy = (f"remedy: raise [shared].p_top in the experiment config "
                  f"to {source_top:g} Pa or above.")
    else:
        remedy = (f"remedy: gpuwm fetch --source gfs --p-top-pa {asked:g} "
                  "into a new folder fetches the levels this top needs "
                  "(gpuwm go and gpuwm run-plan ask for them on their own), "
                  f"or raise [shared].p_top to {source_top:g} Pa or above.")
    raise VerticalLadderRefusal(
        f"GFS source atmosphere stops at {source_top:g} Pa but this config's "
        f"model top is {p_top:g} Pa, so every level above {source_top:g} Pa "
        "would be extrapolated past the top of the analysis; this folder "
        "was fetched without the levels above it",
        remedy=remedy)


def _validate_grid_and_vertical_contract(exp, wps_namelist: Path, *,
                                         source_top_pressure_pa: float
                                         = _CERTIFIED_SOURCE_TOP_PA):
    return validate_native_lambert_contract(
        exp, wps_namelist, source_name="GFS",
        source_top_pressure_pa=float(source_top_pressure_pa),
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

    # Lake coverage is reported, not enforced, for the same reason.  The
    # nearest GFS water to a lake is the crop's nearest (WPS searches the
    # source it is given), and a lake whose nearer donor could lie past
    # the crop's edge is counted rather than refused: refusing it stopped
    # a preparation over the extent of its download.  A crop with no
    # water at all gives its lakes the skin temperature GFS has there
    # (:func:`lake_skin_with_source_skin_fallback`), counted the same way.
    # A whole-globe file is a ring with no edge in longitude, searched as
    # one exactly as the lake search itself does.
    lakes = np.asarray(lake_mask, dtype=np.bool_)
    source_land = np.asarray(snapshot.fields["LANDSEA"], dtype=np.float64)
    skin = np.asarray(snapshot.fields["SKINTEMP"], dtype=np.float64)
    water = np.isfinite(source_land) & (source_land < 0.5) & np.isfinite(skin)
    (water,), lake_x, period = unrolled_source_ring(
        (water,), snapshot.longitude, mass_x)
    lake_nx = water.shape[1]
    max_nearest_distance = 0.0
    max_search_radius = 0.0
    unproven = 0
    for j, i in (np.argwhere(lakes) if np.any(water) else ()):
        y = float(mass_y[j, i])
        x = float(lake_x[j, i])
        # Initial window only; the loop doubles it until the nearest water
        # donor is provably inside the searched rectangle.
        search_radius = 8.0
        while True:
            j0 = max(0, int(np.ceil(y - search_radius)))
            j1 = min(ny - 1, int(np.floor(y + search_radius)))
            i0 = max(0, int(np.ceil(x - search_radius)))
            i1 = min(lake_nx - 1, int(np.floor(x + search_radius)))
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
                if i1 < lake_nx - 1:
                    unseen_distance.append((i1 + 1) - x)
                if (not unseen_distance
                        or best_squared < min(unseen_distance) ** 2):
                    break
            search_radius *= 2.0
        outside_squared = min(
            (y + 1.0, ny - y) if period is not None
            else (x + 1.0, lake_nx - x, y + 1.0, ny - y)) ** 2
        if not best_squared < outside_squared:
            unproven += 1
        max_nearest_distance = max(max_nearest_distance, best_squared ** 0.5)
        max_search_radius = max(max_search_radius, search_radius)
    result["lake_cells"] = int(lakes.sum())
    result["lake_source_search"] = "global expanding nearest-water"
    result["max_lake_source_search_radius_cells"] = max_search_radius
    result["max_lake_source_water_distance_cells"] = max_nearest_distance
    # Present only when they happened, so every other receipt keeps the
    # bytes it had.
    if np.any(lakes) and not np.any(water):
        result["lake_cells_without_source_water"] = int(lakes.sum())
    if unproven:
        result["lake_cells_nearest_water_past_crop"] = unproven
    return result


def lake_skin_with_source_skin_fallback(lake_skin, lake_mask, mapped_skin):
    """Each lake's nearest GFS water, or the skin GFS has at the lake.

    ``lake_skin`` is :func:`interpolate_lake_skin_temperature` on the
    crop, NaN at every lake when the crop holds no water to search.  Such
    a lake takes the mapped skin temperature at its own cells.  This route
    maps its lakes as land, so that is the skin of the land GFS has where
    the lake lies: the source model's own surface state there, and never
    another basin's water.  Returns ``(lake_skin, cells)``, ``cells``
    being how many lake cells took the mapped skin.
    """
    if hasattr(mapped_skin, "get"):
        mapped_skin = mapped_skin.get()
    lakes = np.asarray(lake_mask, dtype=bool)
    lake_skin = np.array(lake_skin, dtype=np.float64, copy=True)
    missing = lakes & ~np.isfinite(lake_skin)
    if np.any(missing):
        lake_skin[missing] = np.asarray(
            mapped_skin, dtype=np.float64)[missing]
    return lake_skin, int(np.count_nonzero(missing))


def _announce_lake_source_water(on_source_skin, past_crop, *, lake_cells):
    """One line when a lake's GFS water could not come from its nearest donor."""
    parts = []
    if on_source_skin:
        parts.append(
            f"{on_source_skin} took the skin temperature GFS has at the "
            "lake, because the fetched area holds no GFS water at all")
    if past_crop:
        parts.append(
            f"{past_crop} took the nearest GFS water inside the fetched "
            "area, though nearer GFS water could lie past its edge")
    if parts:
        print(
            f"lake water temperature: of {lake_cells} lake cell(s), "
            + "; ".join(parts) + " (counted under source_coverage in the "
            "preparation proof); a wider fetch area gives every lake its "
            "nearest GFS water", file=sys.stderr)


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
    # The series names the SOURCE leads it carries, and its first entry is
    # not required to be f000.  Initializing a model from a forecast lead
    # is routine practice -- WPS/real does it from any met_em set -- and
    # the f000 anchor here was the reason a user who wanted the f174..f240
    # window had to integrate 240 hours to reach it.  What a series must
    # still be is at least two times, nonnegative, strictly increasing on
    # one positive uniform cadence, inside the published lead ladder: the
    # first is the initial condition and the rest are its boundaries.
    if len(hours) < 2:
        raise ForcingSeriesRefusal(
            "GFS series must have at least two times")
    if hours[0] < 0:
        raise ValueError("GFS series forecast hours must be nonnegative")
    deltas = [later - earlier for earlier, later in zip(hours, hours[1:])]
    if not deltas or any(delta <= 0 or delta != deltas[0] for delta in deltas):
        raise ValueError("GFS series cadence must be positive and uniform")
    from gpuwm.fetch import gfs_forecast_hours
    gfs_forecast_hours(hours[-1] - hours[0], deltas[0], hours[0])
    return tuple(records)


def resolve_initial_forecast_lead(
        *, start_time: datetime, cycle_time: datetime,
        source_hours: Sequence[int]) -> int:
    """The source lead the model starts from, or a precise refusal.

    ``start_time = cycle + K hours`` for any K the series actually
    carries.  K = 0 is the analysis and is what every GFS run did
    before; a positive K initializes from a forecast lead, which is
    ordinary practice (WPS/real does it) and is what the front door used
    to refuse outright with "GFS cycle must equal experiment start_time".
    A user who wanted the f174..f240 window therefore had to integrate
    240 hours to reach it.

    Three things are refused here and nothing else: a start BEFORE the
    cycle (no such product exists), a start that is not a whole number of
    hours after it (the series is hourly), and a lead the fetched set does
    not contain.  The last one names the set, because "which leads do I
    have" is the question a caller who hits it is actually asking.
    """

    offset = start_time - cycle_time
    seconds = offset.total_seconds()
    if seconds < 0:
        raise ValueError(
            f"experiment start_time {start_time:%Y-%m-%d %H:%M:%S} is "
            f"BEFORE GFS cycle {cycle_time:%Y-%m-%d %H:%M:%S}; a model "
            "cannot be initialized from a product its source has not "
            "produced yet")
    if seconds % 3600:
        raise ValueError(
            f"experiment start_time {start_time:%Y-%m-%d %H:%M:%S} is "
            f"{seconds / 3600.0:g} h after GFS cycle "
            f"{cycle_time:%Y-%m-%d %H:%M:%S}; a GFS forecast lead is a "
            "whole number of hours")
    lead = int(seconds // 3600)
    hours = [int(hour) for hour in source_hours]
    if lead not in hours:
        raise ValueError(
            f"experiment start_time {start_time:%Y-%m-%d %H:%M:%S} asks "
            f"GFS cycle {cycle_time:%Y-%m-%d %H:%M:%S} for forecast lead "
            f"f{lead:03d}, which this series does not carry.  The fetched "
            "forecast hours are "
            + ", ".join(f"f{hour:03d}" for hour in hours)
            + ".  Fetch that lead (`gpuwm fetch --source gfs "
            f"--forecast-start-hour {lead}`) or set start_time to a lead "
            "the series has")
    return lead


def initial_condition_provenance(
        *, cycle_time: datetime, lead_hours: int) -> dict[str, object]:
    """What the initial condition IS, said in one sentence and in fields.

    A forecast lead is never relabelled as an analysis: at K = 0 this
    says analysis and names process 81; at K > 0 it says forecast, names
    process 96, and says out loud that the initial condition is itself a
    K-hour forecast.  Every receipt this door writes carries this block,
    so a reader can never recover the cycle without also recovering the
    lead.
    """

    analysis = lead_hours == 0
    start_time = cycle_time + timedelta(hours=lead_hours)
    return {
        "schema": "gpuwm-gfs-initial-condition-provenance-v1",
        "cycle": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "initial_forecast_lead_hours": int(lead_hours),
        "model_start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "initial_condition_kind": "analysis" if analysis else "forecast",
        "forecast_generating_process_id": 81 if analysis else 96,
        "statement": (
            f"initialized from GFS cycle "
            f"{cycle_time:%Y-%m-%dT%H:%M:%SZ} analysis (f000)"
            if analysis else
            f"initialized from GFS cycle "
            f"{cycle_time:%Y-%m-%dT%H:%M:%SZ} at lead f{lead_hours:03d}: "
            f"the initial condition is itself a {lead_hours} h forecast"),
    }


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
    expected_levels_hpa: tuple[float, ...] | None = None,
) -> tuple[Era5Snapshot, ...]:
    gate = _parse_gate(root / "gate.tsv")
    hours = tuple(int(value) for value in gate["forecast_hours"].split(","))
    if hours != tuple(hour for hour, _ in records):
        raise ValueError("GFS bridge forecast-hour inventory differs from series")
    if gate["cycle"] != cycle.strftime("%Y-%m-%d %H:%M:%S"):
        raise ValueError("GFS bridge cycle differs from requested cycle")
    # The ladder is the bridge's report of what it decoded, not a constant
    # to match: a case whose model top sits above 100 hPa is fetched with
    # extra levels on top, and asserting equality against the certified 21
    # is what made 10000 Pa a silent ceiling on every GFS run.  It still
    # has to BE the certified ladder extended upward -- same rule the
    # bridge applies to the source -- and when the caller knows which
    # ladder was fetched, the two must agree exactly.
    levels_hpa = _validate_ladder(
        np.asarray(
            [float(value) / 100.0
             for value in gate["pressure_levels_pa"].split(",")],
            dtype=np.float64),
        "GFS bridge gate")
    if expected_levels_hpa is not None and not np.array_equal(
            levels_hpa, np.asarray(expected_levels_hpa, dtype=np.float64)):
        raise ValueError(
            "GFS bridge decoded a pressure ladder the input manifest does "
            "not declare")
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
                (levels_hpa.size, ny, nx),
            )
            fields[name] = field
        for name in _TWO_D:
            field = _read_f32(time_root / f"{name}.f32le", (ny, nx))
            fields[name] = field
        snapshots.append(Era5Snapshot(
            valid_time=cycle + timedelta(hours=hour),
            levels_hpa=levels_hpa,
            latitude=latitude,
            longitude=longitude,
            fields=fields,
        ))
    return tuple(snapshots)


def prepare_progress_path(output_root) -> Path:
    """Where this stage publishes what it is doing, while it does it.

    A SIBLING of ``--output-root``, not a file inside it, and that is
    forced rather than chosen: the stage builds its whole product in a
    staging directory and publishes it with one ``os.replace``, so
    ``--output-root`` does not exist until the work is finished.  A
    progress file inside it could only appear after there was nothing
    left to report.

    One function, called by the publisher and by ``gpuwm go``'s
    heartbeat, so the path cannot be spelled two ways.  It survives the
    run on purpose: the finished file is the stage's phase timeline, and
    the same thing that answered "what is it doing" answers "what did it
    spend its time on".
    """

    root = Path(output_root)
    return root.parent / f"{root.name}.progress.json"


class _PreparePhases:
    """Publish which phase this stage is in, at each boundary it times.

    THE DEFECT: prepare is the longest quiet stage in the chain (15 s
    warm, 36 s at the documented normal size) and it published nothing
    at all, so ``gpuwm go``'s heartbeat -- which polls exactly this path
    and names whatever the running stage says about itself -- showed a
    bare stopwatch for the whole of it.  Every other long stage in this
    tree answers that question; this one is the reason the heartbeat
    looked like a hang.

    Every write is best-effort.  A progress file that cannot be written
    is not a reason to fail a preparation, and the caller must never
    have to guard a telemetry call.
    """

    def __init__(self, path: Path, *, phases=PREPARE_PHASES):
        self._path = Path(path)
        self._phases = tuple(phases)
        self._started = time.perf_counter()
        self._index = 0

    def enter(self, phase: str, **fields) -> None:
        try:
            self._index = self._phases.index(phase) + 1
        except ValueError:
            self._index = 0
        self._publish(phase, **fields)

    def _publish(self, phase: str, **fields) -> None:
        payload = {
            "schema": PREPARE_PROGRESS_SCHEMA,
            # `status` is what go's heartbeat prints, and it prints it
            # verbatim beside the runner's own SHOUTING_SNAKE statuses,
            # so it is spelled the way they are.
            "status": f"PREPARING_{phase.upper()}",
            "phase": phase,
            "phase_index": self._index,
            "phases_total": len(self._phases),
            "elapsed_seconds": round(time.perf_counter() - self._started, 6),
            **fields,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(self._path.name + ".partial")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8", newline="\n")
            os.replace(temporary, self._path)
        except OSError:
            pass


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
    """The front door's physics selection: record, govern, never whitelist.

    One door, one contract now (owner ruling 2026-07-31: the physics
    suite is the user's choice).  The single-domain route used to
    substitute WSM6 when no profile was named and compare the config's
    switches against it for exact equality, so the product default suite
    -- and every hand-authored suite -- was refused at preparation time.
    That whitelist is gone: an UNNAMED configuration, one domain or a
    tree, is governed the way the domain-tree route has always been
    governed -- engine-valid selectors, the registry's source-neutral
    tuple governance with its expert acknowledgements, and a recorded
    blocker (never a refusal) where the registry has no spelling for an
    engine-valid tuple.  Verification status is receipt metadata.

    v1.1.0's regression (this whitelist binding at the front door for
    every configuration, refusing the wizard's default emission as a
    stack trace) and v1.1.1's scoping fix are recorded in this module's
    history; the 2026-07-31 ruling removed the single-domain half too.

    An explicitly named profile is still enforced on either route,
    because a gate the caller asked for remains binding: naming
    ``--physics-profile`` asserts the config IS that shipped suite.
    Land-surface initialization follows the selected scheme and soil
    geometry. Registry source/template membership describes evidence;
    the shared field and category checks decide whether input can run.
    """

    acknowledgements, provenance = acknowledgement_delivery(
        flag=expert_acknowledgements,
        toml=getattr(exp, "acknowledgements", ()),
    )
    if len(exp.domains) == 1:
        # THE single-domain selection spelling, shared with the
        # prepared-forecast runner's governance and recompute so the
        # two sides cannot drift (physics_compat owns it).
        selection = single_domain_physics_selection(
            exp.root.run,
            profile=physics_profile,
            expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=provenance)
    else:
        selection = multi_domain_physics_selection(
            {int(domain.grid_id): domain.run for domain in exp.domains},
            profile=physics_profile,
            expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=provenance)
    return selection


def _prepared_initial_perturbation(exp) -> dict[str, object] | None:
    """Declare the tree runner's deferred mutation without changing prep arrays.

    GFS prepares the source analysis/forecast, including boundary times, as
    before. Only the domain-tree forecast runner applies the configured
    bubble to restored initial states; the single-domain runner cannot.
    """
    from gpuwm.experiment import refuse_unrouted_perturbation

    if len(exp.domains) < 2:
        refuse_unrouted_perturbation(exp, "GFS-direct prepared-cache")
    if exp.perturbation is None:
        return None
    return {
        "schema": "gpuwm-gfs-initial-perturbation-deferred-v1",
        "status": "DEFERRED_TO_FORECAST_INITIALIZATION",
        "prepared_arrays": "unperturbed source initial and boundary states",
        "application_route": "gpuwm.prepared_domain_tree_forecast",
        "application_point": "restored states at experiment start time",
        "applied_on_restart": False,
        "applied_to_delayed_domains": False,
        "config": exp.perturbation.receipt(),
    }


def _announce_adaptation(sentence: str) -> None:
    """Say, once, that the run is not on the configured vertical coordinate."""

    warn(sentence,
         "WRF v4.6.1 dyn_em/nest_init_utils.F:1158-1182 calls a column "
         "the coordinate cannot order fatal and names reducing etac as "
         "the remedy, and a column it only just orders keeps one layer "
         "too thin to integrate, which a lower etac thickens; the etac is "
         "derived here from the terrain this run can actually touch and "
         "applied, so the prepared inputs, their receipt and the forecast "
         "all carry the same coordinate.  p_top is untouched.")


def _survey_static_catalog(exp, wps_namelist, geog_root):
    """The WPS_GEOG catalog the terrain survey needs, or None.

    A single-domain run surveys only the root terrain the caller already
    holds, and a prebuilt static cache with no geography is exactly that
    case; a tree needs the catalog and already refuses without a geog
    root one screen above.
    """

    if geog_root is None or len(exp.domains) < 2:
        return None
    from gpuwm.hrrr_native_static import verified_static_catalog

    catalog, _ = verified_static_catalog(
        Path(wps_namelist), Path(geog_root),
        [domain.grid_id for domain in exp.domains])
    return catalog


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
    preprocess_backend: str = "auto",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: Path | None = None,
    geog_root: Path | None = None,
    hierarchy_workers: int | None = None,
    physics_profile: str | None = None,
    expert_acknowledgements: tuple[str, ...] = (),
    stock_wrf_export: bool = True,
    statics_corridor=None,
    preprocess_backend_reason: str | None = None,
) -> dict[str, object]:
    """Build native GFS initial/boundary files and return the proof receipt.

    ``geog_root`` activates the internal multi-domain route when the
    experiment declares children.  The existing public single-domain CLI is
    unchanged; source-neutral public hierarchy configuration is owned by the
    higher-level integration layer.

    ``stock_wrf_export`` asks for the bonus unchanged-WRF file set beside
    the forecast this prepares, on the single-domain route and the
    domain-tree route alike.  It defaults to True, so a domain the
    exporter can represent publishes exactly what it always published.
    One it cannot represent (Kessler, Milbrandt-Yau and WDM6 have no
    stock package contract) does not fail: the export is refused, by
    name, in the proof's export slot, and the forecast preparation
    completes, because the forecast restores the prepared cache, not the
    exported files.  Which physics a forecast may run belongs to the
    registry and the acknowledgement gate, not to a downstream
    file-format contract.  Pass False to skip the export outright; the
    slot then records it as not requested.  On the single-domain route an
    export that is written must carry exactly the physics this
    preparation selected, or the preparation stops.

    ``statics_corridor`` opts the DOMAIN-TREE route into emitting sealed
    child-resolution statics corridors (``"all"`` or a sequence of child
    grid ids; :mod:`gpuwm.static.corridor`), which is what lets the
    prepared tree runner honor ``[relocation]`` follow sources.  ``None``
    (the default) leaves every published byte exactly as before.
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
    # A head published early whose producer failed, was stopped or went
    # silent is this tool's own unfinished product, not a finished run.
    remove_unfinished_tree(Path(output_root))
    if Path(output_root).exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    # A domain tree is published through staging deeper than its
    # output root; refused here, before any GRIB is hashed or decoded.
    # The experiment is read only to count its domains, and only when
    # the limit binds: a single-domain bundle publishes no tree and is
    # not measured against it.  The load the preparation uses follows
    # the manifest verification below.
    refusal = published_path_refusal(
        Path(output_root), wrf_export=stock_wrf_export)
    if (refusal is not None
            and len(load_experiment(Path(experiment_config)).domains) > 1):
        raise ValueError(refusal)
    _prepare_output_root_parent(Path(output_root))
    # THE STAGE CLOCK STARTS HERE, not after the verification below.
    # It used to start after it, so `total` in the receipt excluded the
    # sha256 of every GRIB this stage is bound to -- work that is
    # linear in the download and is nobody's idea of free.  A receipt
    # whose `total` is not the stage's wall clock cannot be summed
    # against the chain's own stage timing, which is the whole point of
    # both numbers existing.
    total_started = time.perf_counter()
    progress = _PreparePhases(prepare_progress_path(output_root))
    progress.enter("verify_inputs", files=len(roles))
    verify_started = time.perf_counter()
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
        cpu_bridge=cpu_preprocess_bridge, reason=preprocess_backend_reason)
    preprocess_receipt = preprocess.receipt()
    # Everything above verified or identified an input: the manifest's
    # digest over every GRIB, this build's implementation hash, the
    # git identity, the preprocess backend's own receipt.  Named as one
    # key because it is one answer to one question -- "how long did
    # binding this stage to its inputs take" -- and because a reader
    # chasing a slow prepare needs to know whether the wall went to
    # hashing the download or to the work after it.
    verify_inputs_seconds = time.perf_counter() - verify_started

    exp = load_experiment(Path(experiment_config))
    # THE FLOOR, BEFORE THE DECODE (A98).  The domains alone set a lower
    # bound on the card price: an explicit cuda they cannot fit is refused
    # here, in seconds, instead of after the host decode and the statics,
    # and auto moves to the CPU here.  The decoded price below stays the
    # binding check.
    preprocess = admit_preparation(
        preprocess, lambda: price_preparation_floor("gfs", exp),
        workers=preprocess_workers)
    preprocess_receipt = preprocess.receipt()
    from gpuwm.static.highres_production import (
        load_static_highres, apply_prepared_highres, static_highres_identity)
    static_highres = load_static_highres(experiment_config)
    from gpuwm.case_data import optional_case_data_from_config, preparation_case_policy
    case_data = optional_case_data_from_config(experiment_config)
    case_policy = preparation_case_policy(case_data)
    from gpuwm.ingest.water_overlay import (
        load_bound_water_overlay, overlay_snapshot_sequence, verify_overlay_sequence)
    water_overlay, water_overlay_binding = load_bound_water_overlay(
        None if case_data is None else case_data.water_temperature_overlay)
    from gpuwm.experiment import refuse_unrouted_spawn
    refuse_unrouted_spawn(exp, "GFS-direct prepared-cache")
    initial_perturbation = _prepared_initial_perturbation(exp)
    if initial_perturbation is not None:
        warn(
            "initial perturbation is deferred to prepared-tree forecast initialization",
            "Prepared initial and boundary arrays remain unperturbed. The "
            "original configuration stays bound to the bundle; its bubbles "
            "are applied once to restored states at experiment start, never "
            "on restart or to a nest born later. Companion stock-WRF export "
            "cannot represent this deferred mutation.")
    physics_selection = front_door_physics_selection(
        exp, physics_profile=physics_profile,
        expert_acknowledgements=expert_acknowledgements)
    # The model's time zero is the experiment's, and it may sit at any
    # source lead the series carries -- not only at the cycle.  Two hour
    # vocabularies live from here down and they are never interchanged:
    # SOURCE leads (f000, f018, f174 ...) name NOAA products, and MODEL
    # forcing offsets (0, 3, 6 ...) count from start_time.  Everything
    # downstream of this door -- the prepared cache, the WRF export, the
    # forecast runner -- speaks model offsets, which is why a run at lead
    # 0 is bit-for-bit the run it always was.
    series_hours = [hour for hour, _ in records]
    lead_hours = resolve_initial_forecast_lead(
        start_time=exp.start_time, cycle_time=cycle_time,
        source_hours=series_hours)
    provenance = initial_condition_provenance(
        cycle_time=cycle_time, lead_hours=lead_hours)
    initial_index = series_hours.index(lead_hours)
    source_hours = series_hours[initial_index:]
    if lead_hours:
        # Said BEFORE the window checks below, so a refusal about the
        # window is read in the light of the lead that shaped it.  Warn,
        # do not block: this is a legitimate initialization whose one
        # consequence the reader has to be told once.
        warn(provenance["statement"],
             "Every field this door consumes is an instantaneous GRIB2 "
             "product definition template 4.0 record, refused by "
             "gfs_grib2_bridge if it is anything else, so no consumed "
             "field carries an accumulation or averaging window whose "
             "meaning would depend on the lead.  What does depend on the "
             "lead is forecast skill: the state you are starting from "
             f"is GFS's own {lead_hours} h forecast, not an analysis.")
        if initial_index:
            warn(
                f"the GFS series carries {initial_index} forecast hour(s) "
                f"before f{lead_hours:03d} "
                + "(" + ", ".join(f"f{hour:03d}"
                                  for hour in series_hours[:initial_index])
                + "); they are decoded and bound by the input manifest but "
                "are not used by this run",
                "The series and its manifest are one hash-bound document: "
                "the door verifies every file the series names rather than "
                "silently ignoring some.  Author a manifest over the tail "
                "you want (`gpuwm fetch --author-front-door-manifest "
                f"--forecast-start-hour {lead_hours}`) to decode only it.")
    if len(source_hours) < 2:
        raise ValueError(
            f"GFS series ends at forecast lead f{lead_hours:03d}, the lead "
            "this experiment starts from, so it carries no lateral boundary "
            "time at all; fetch at least one more forecast hour")
    hours = [hour - lead_hours for hour in source_hours]
    if hours[-1] * 3600 < exp.run_seconds:
        raise ValueError(
            f"GFS series does not cover the configured run: it reaches "
            f"f{source_hours[-1]:03d}, which is {hours[-1]} h after the "
            f"f{lead_hours:03d} this experiment starts from, and the run "
            f"is {exp.run_seconds / 3600.0:g} h long")
    boundary_interval_seconds = (hours[1] - hours[0]) * 3600
    cfg = exp.root.run
    # What the fetch actually reached, not what the default ladder would
    # have reached: the case's p_top is checked against this.
    source_top_pressure_pa = _manifest_source_top_pa(manifest)
    _refuse_a_fetch_below_the_model_top(exp, source_top_pressure_pa)
    manifest_levels = manifest.get("source", {}).get("pressure_levels_hpa") \
        if isinstance(manifest.get("source"), dict) else None
    expected_levels_hpa = (tuple(float(level) for level in manifest_levels)
                           if isinstance(manifest_levels, list)
                           and manifest_levels else None)
    if len(exp.domains) == 1:
        grid = _validate_grid_and_vertical_contract(
            exp, Path(wps_namelist),
            source_top_pressure_pa=source_top_pressure_pa)
        grids = (grid,)
    else:
        if geog_root is None:
            raise ValueError(
                "nested GFS preparation requires an explicit geog_root")
        grids = validate_native_lambert_contracts(
            exp, Path(wps_namelist), source_name="GFS",
            source_top_pressure_pa=source_top_pressure_pa,
        )
        grid = grids[0]
    landuse_attrs = None
    # THE 14.1 SECONDS.  MEASURED, 2026-08-16: at the documented normal
    # size this block was 40% of the prepare stage and the largest
    # single item in it, and proof.json named it nowhere -- the only
    # timing that existed was `static.build_static` under the opt-in
    # GPUWM_PERF_TIMING env var, which under the fixed-means-default law
    # is a deep-dive diagnostic and not instrumentation.  The number is
    # taken here, unconditionally, with a plain perf_counter pair
    # exactly like the four keys the receipt already had.  The env var
    # is untouched and still does what it always did.
    static_started = time.perf_counter()
    progress.enter("static_build",
                   cells=int(cfg.ny) * int(cfg.nx),
                   source=("geog" if static_input is None else "prebuilt"))
    if static_input is None:
        static, root_static_receipt, landuse_attrs = _static_from_geog(
            Path(wps_namelist), Path(geog_root), grid, cfg)
        root_static_provider = "native-wps-geog"
    else:
        root_static_receipt = _verify_static_receipt(
            Path(static_receipt), Path(static_input), grid, cfg)
        static = _load_static(Path(static_input), grid, cfg.ny, cfg.nx)
        root_static_provider = "prebuilt-hash-bound-cache"
        if geog_root is not None:
            # The prebuilt NPZ is numeric fields only, so the land-use
            # table has to come from the GEOG index beside it.
            from gpuwm.hrrr_native_static import verified_static_catalog
            from gpuwm.static.build import geog_selection_from_catalog

            catalog, _ = verified_static_catalog(
                Path(wps_namelist), Path(geog_root), (1,))
            landuse_attrs = geog_selection_from_catalog(
                catalog, 1).landuse_global_attrs()
    static, root_static_receipt = apply_prepared_highres(
        static, grid, config=static_highres, domain_id=1,
        case_date=exp.start_time.date(), landuse_attrs=landuse_attrs,
        baseline_receipt=root_static_receipt)
    # The land-height check reads the terrain the run will integrate: a
    # declared high-resolution terrain gives the islands the baseline
    # dataset holds at 0 m their height.
    require_land_terrain(static["HGT_M"], static["LANDMASK"])
    # Both roads, one key: "build it from the geography tree" and "load
    # and verify the prebuilt cache" are two answers to one question,
    # and a receipt that timed only the first would go quiet on exactly
    # the runs that chose the second.
    static_build_seconds = time.perf_counter() - static_started

    # THE COORDINATE, BEFORE ANYTHING IS BUILT ON IT.  The root terrain
    # exists now and every other terrain this run can touch is derivable
    # from the same geography, so this is the last moment at which the
    # vertical coordinate is still a decision rather than an assumption.
    # Past here the root coordinate, the children's shared one, the
    # exported WRF input and the prepared caches all carry whatever was
    # chosen here.
    exp, vertical_adaptation = adapt_experiment_for_statics(
        exp, grids, root_terrain=static["HGT_M"],
        static_catalog=_survey_static_catalog(exp, wps_namelist, geog_root),
        static_highres=static_highres, announce=_announce_adaptation)
    cfg = exp.root.run

    # One descriptor per decoded array is held for the whole bundle
    # write, and the array count is forcing-times x fields-per-time --
    # linear in forecast length.  Ask for the budget now, while the
    # answer can still be a sentence: past here the shortfall arrives as
    # an [Errno 24] naming whichever file was next, which is how a hard
    # ceiling on forecast length looked like a random IO fault.
    refusal = ensure_descriptor_headroom(
        required_descriptors(len(records), len(_THREE_D) + len(_TWO_D)),
        what=f"preparing {len(records)} GFS forcing time(s)")
    if refusal is not None:
        raise ValueError(refusal)

    decode_started = time.perf_counter()
    progress.enter("decode", forcing_times=len(records))
    # ignore_cleanup_errors: when the body fails for a reason that also
    # stops the scratch tree being removed -- descriptor exhaustion is
    # exactly that, since rmtree needs one per directory it walks -- the
    # cleanup's exception REPLACES the body's, and the run reports the
    # rmtree's victim instead of its own fault.  Both measured EMFILE
    # failures named `decoded/f000`, the first directory the cleanup
    # reached and a path each run had already read successfully.  The
    # tree is still removed; only its failure stops overwriting the real
    # one.
    with tempfile.TemporaryDirectory(
            prefix="gpuwm-gfs-bridge-", dir=Path(output_root).parent,
            ignore_cleanup_errors=True) as temporary:
        decoded = Path(temporary) / "decoded"
        bridge_command = [
            str(Path(bridge)), "--series", str(Path(series)), str(decoded),
            cycle_time.strftime("%Y-%m-%d %H:%M:%S")]
        if expected_levels_hpa is not None:
            # Declare the request's ladder rather than letting the
            # bridge derive one from the file.  A NOMADS subset carries
            # exactly the requested levels, so this changes nothing
            # there; a --mode full-file object carries every published
            # level up to 1 Pa, and deriving the ladder from IT would
            # decode the mesosphere nobody asked for.  The gate-versus-
            # manifest ladder comparison below still runs on the
            # bridge's own report of what it decoded.
            bridge_command += ["--pressure-levels-pa", ",".join(
                format(level * 100.0, "g") for level in expected_levels_hpa)]
        completed = subprocess.run(
            bridge_command,
            check=False, text=True, capture_output=True,
        )
        if completed.returncode != 0:
            # The bridge already fails closed and says why, including for
            # a GRIB whose valid time is not the one requested.  Raising
            # a bare RuntimeError meant that careful, content-based
            # diagnosis was delivered as a traceback at rc 1, because
            # main() catches only (ValueError, OSError) -- the detection
            # was right and only the delivery was wrong.
            raise GfsRouteError(decode_failure_message(
                "GFS Rust bridge", completed.stderr))
        expected_source_sha256 = {
            hour: manifest["files"][f"grib-f{hour:03d}"]["sha256"]
            for hour, _ in records
        }
        decoded_snapshots = _load_bridge_snapshots(
            decoded, cycle_time, records, expected_source_sha256,
            expected_levels_hpa=expected_levels_hpa)
        # A NOMADS box that crosses the source's own 0/360 origin cannot be
        # asked for as one subregion, so `Area.as_nomads` widens it to the
        # whole band (fetch.py:348-362) and the decode returns a ring stored
        # with an arbitrary cut at longitude 0.  Re-cut it opposite the
        # target before anything indexes it; a regional crop is untouched.
        target_longitudes = tuple(
            pair[1] for pair in (grid.latlon_mass(), grid.latlon_u(),
                                 grid.latlon_v()))
        decoded_snapshots = tuple(
            orient_global_source_longitudes(snapshot, *target_longitudes)
            for snapshot in decoded_snapshots)
        # From here on the run sees only the window that starts at the
        # experiment's lead.  Each snapshot still carries the SOURCE
        # valid time the bridge decoded (cycle + its own lead), which is
        # exactly start_time + the model offset.
        snapshots = overlay_snapshot_sequence(decoded_snapshots[initial_index:], water_overlay,
                                              binding=water_overlay_binding,
                                              workers=host_step_workers(preprocess))
        decode_seconds = time.perf_counter() - decode_started
        # The land-use table's own ISLAKE, never a hard-coded 21: the
        # category number is a property of the selected table, and a table
        # without an inland-water class is a state the assembly declares
        # rather than a silently empty mask.  lake_override=True because
        # the soil call below hands the router a lake_mask/lake_skin pair,
        # which makes every one of these cells water inside the router.
        if (landuse_attrs is None and case_data is not None
                and (case_data.water_temperature_policy is not None or water_overlay is not None)
                and case_policy["water_temperature_policy"] != "wrf_compat"):
            raise ValueError("declared water-temperature policy needs the selected land-use "
                             "table's ISLAKE metadata; provide --geog-root with this static cache")
        water_statics = (
            None if landuse_attrs is None
            else WaterTemperatureStatics.for_route(
                route=_WATER_ROUTE, policy=case_policy["water_temperature_policy"],
                landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
                landuse_attrs=landuse_attrs, lake_override=True))
        if water_statics is None:
            # LOUD, and no coherent receipt is printed below: with a
            # prebuilt static cache and no geog root there is no
            # land-use table to read ISLAKE from, so the shipped MODIS
            # category is ASSUMED rather than resolved.  Say which
            # number is being assumed and how to stop assuming it.
            print(
                "lake mask: the prebuilt static cache carries no "
                "land-use metadata, so this run assumes the MODIS "
                f"inland-water category {MODIS_LAKE_CATEGORY} instead "
                "of reading ISLAKE, and keeps the historical per-cell "
                "water temperature.  Pass --geog-root, which is read "
                "only for the land-use index, to resolve it.",
                file=sys.stderr)
            lake_mask = static["LU_INDEX"] == MODIS_LAKE_CATEGORY
        else:
            lake_mask = water_statics.lake
        coverage_receipt = {
            f"f{hour:03d}": _source_coverage_receipt(snapshot, grid, lake_mask)
            for hour, snapshot in zip(source_hours, snapshots)
        }

        from gpuwm.core.grid import make_vertical_coord

        # THE FIT, BEFORE THE FIRST DEVICE ALLOCATION.  The decode above
        # ran on the host; everything below builds on the chosen backend.
        # auto moves a preparation the card cannot hold to the CPU, and an
        # explicit cuda that cannot fit is refused here, by name, instead
        # of stopping in DomainState.__init__ minutes later (A65).  A
        # backend already on the CPU is not priced.
        preprocess = admit_preparation(
            preprocess,
            lambda: price_forcing_preparation("gfs", exp, snapshots),
            workers=preprocess_workers)
        preprocess_receipt = preprocess.receipt()
        initialize_started = time.perf_counter()
        progress.enter("initialize_all_times",
                       forcing_times=len(records))
        # ONE forcing time is ever resident.  A single domain builds the
        # start time FIRST, publishes it in the prepared head and releases
        # it, then writes each boundary interval as soon as its two times
        # exist (gpuwm.ingest.boundary_stream).  A hierarchy builds the
        # start time LAST (start_last_forcing_order) and retains only that
        # met/state; every other time contributes its perimeter frames
        # against its own position and is released before the next one is
        # interpolated.  Walking the times in order instead meant holding
        # the start time -- which nothing reads until the boundaries are
        # complete -- while each later time was built underneath it.  At
        # 800x800x49 with mp=10 and three GFS times that second resident
        # time is 14.67 GiB of device residency against 7.66, a priced
        # peak envelope of 23.92 GiB against 15.86: the difference
        # between preparing the domain on a 16 GiB card and OOMing after
        # the whole forcing chain had already been fetched.
        initial_result = None
        initial_met = None
        forcing = StateBoundaryFrames(
            spec_bdy_width=cfg.spec_bdy_width,
            spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
        interpolation_landmask = np.asarray(
            static["LANDMASK"] >= 0.5, dtype=np.bool_).copy()
        # GEOG lakes can be much smaller than a 0.25-degree GFS cell.  Select
        # their atmospheric/soil source as land here, then apply the existing
        # explicit nearest-source-water lake initialization below.
        interpolation_landmask[lake_mask] = True
        def build_forcing_time(index):
            # One forcing time's build, unchanged.  A single domain calls
            # it start first (gpuwm.ingest.boundary_stream); a hierarchy
            # keeps the start time last until its children are chained.
            source = snapshots[index]
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
                met, cfg, coord, static["HGT_M"], grid=grid,
                landmask=static["LANDMASK"],
                p_top=exp.vertical.p_top, sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                preprocess_backend=preprocess,
                state_backend="preprocess", boundary_only=index != 0)
            initialized.state.set_map_coriolis(
                static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
                static["F"], static["E"], sina=static["SINALPHA"],
                cosa=static["COSALPHA"])
            return met, initialized

        times = tuple(snapshot.valid_time for snapshot in snapshots)
        if len(exp.domains) > 1:
            for index in start_last_forcing_order(len(snapshots)):
                met, initialized = build_forcing_time(index)
                forcing.add_state(initialized.state, index=index)
                if index == 0:
                    initial_met = met
                    initial_result = initialized
                else:
                    del met, initialized
                    release_backend_memory(preprocess)
            boundaries = forcing.build(times)
            attach_lateral_boundaries(initial_result.state, boundaries)
        else:
            # START FIRST: the start time makes the head, the later times
            # are built after it is published, one resident at a time.
            initial_met, initial_result = build_forcing_time(0)
            forcing.add_state(initial_result.state, index=0)
            boundaries = None
        lake_skin, lakes_on_source_skin = lake_skin_with_source_skin_fallback(
            interpolate_lake_skin_temperature(
                snapshots[0], grid, lake_mask,
                workers=host_step_workers(preprocess)),
            lake_mask, initial_met.fields["SKINTEMP"])
        _announce_lake_source_water(
            lakes_on_source_skin,
            max((receipt.get("lake_cells_nearest_water_past_crop", 0)
                 for receipt in coverage_receipt.values()), default=0),
            lake_cells=int(np.count_nonzero(lake_mask)))
        # The assembly runs HERE and not inside the mapping, because this
        # route's water surface is only settled now: the interpolation
        # landmask deliberately calls GEOG lakes land (they are smaller
        # than a 0.25 degree GFS cell), and their skin temperature is the
        # separately selected nearest source water above.  Feeding the
        # mapping's landmask to the assembly would have classed every lake
        # as land and handed the soil router a land value on exactly the
        # cells the lake override makes water.
        water_temperature = None
        if water_statics is not None:
            from gpuwm.ingest.horiz import _as_host_float64

            assembly_skin = _as_host_float64(
                initial_met.fields["SKINTEMP"]).copy()
            # Every lake value is finite wherever the source carries a
            # skin temperature at all (see
            # lake_skin_with_source_skin_fallback), and only finite ones
            # are substituted.
            usable_lake = lake_mask & np.isfinite(lake_skin)
            assembly_skin[usable_lake] = lake_skin[usable_lake]
            assembly = assemble_for_route(
                water_statics, mapped_sst=None, mapped_skin=assembly_skin,
                workers=host_step_workers(preprocess))
            water_temperature = assembly.values
            # The receipt is printed only now that the field it describes
            # is the one soil consumes, two statements below.
            announce_water_temperature(assembly.receipt)
        # WRF's `adjust_soil_temp_new` lapse, which this route was not
        # arming.  The elevation pair is all-or-none in
        # `preprocess_noah_soil`, and passing NEITHER satisfies that guard
        # silently -- so TSK and every GFS_ST* level came through with no
        # elevation correction at all and nothing said so.  Measured, the
        # regression of TSK on terrain was -0.0003 K/m against WRF's
        # -0.0065.  `SOURCE_OROGRAPHY` was decoded all along: the bridge
        # gate above refuses any decode whose invariant_fields is not
        # "SOURCE_OROGRAPHY,LANDSEA".  Resolved through the one shared
        # resolver, with no declared artifact on this route.
        soil_orography = soil_source_orography(None, initial_met.fields)
        # THE RECONCILED CATEGORY, never the raw geogrid SCT_DOM: the one
        # rulebook the ERA5 door and the nested child already follow
        # (gpuwm/ingest/soil.py: door_reconciled_soil_category).  Raw, a
        # shoreline land column with the water soil category killed a RUC
        # run on its first surface call (`mavail must be finite`) after a
        # full preparation -- the death the retired GFS+RUC route refusal
        # used to pre-empt (ENG-009).
        soil = preprocess_land_surface_soil(
            initial_met.fields,
            sf_surface_physics=int(cfg.sf_surface_physics),
            num_soil_layers=int(cfg.num_soil_layers),
            soil_type=door_reconciled_soil_category(
                static, initial_met.fields, landuse_attrs, route="GFS"),
            deep_soil_temperature=static["TMN"], lake_mask=lake_mask,
            lake_skin_temperature=lake_skin,
            landmask=static["LANDMASK"],
            # Land the source holds no land for takes the column the
            # router builds (gpuwm/ingest/soil.py: island_soil_columns).
            soil_no_source_land=getattr(
                initial_met, "soil_no_source_land", None),
            terrain=static["HGT_M"] if soil_orography is not None else None,
            source_orography=soil_orography,
            water_temperature=water_temperature,
            water_temperature_policy=case_policy["water_temperature_policy"],
            # GFS's 0.25 degree soil state is 5.5 by 9.1 cells wide on a
            # 3 km European domain, which is exactly the regime the
            # sub-source-cell reconstitution exists for.
            soil_mesh=soil_mesh_plan_from_case(
                snapshots[0], grid, experiment_config),
            route=_WATER_ROUTE)
        if len(exp.domains) > 1:
            # Every forcing time has been consumed; a single domain proves
            # this after its remaining times, below.
            verify_overlay_sequence(snapshots)
        initialize_seconds = time.perf_counter() - initialize_started

        native_source_identity = {
            "adapter": "gfs-pgrb2-0p25-direct-v1",
            **({"static_highres": static_highres_identity(static_highres)}
               if static_highres is not None else {}),
            "preparation_case_policy": case_policy,
            "water_temperature_overlay": water_overlay_binding,
            "input_manifest_schema": manifest["schema"],
            "input_manifest_sha256": manifest_digest,
            # Cycle AND lead, never one standing in for the other.  A
            # cache restored from this identity can say what its initial
            # condition was without consulting anything else.
            "initial_condition": provenance,
            "source_forecast_hours": list(source_hours),
            "decoded_forecast_hours": list(series_hours),
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
            **({"initial_perturbation": initial_perturbation}
               if initial_perturbation is not None else {}),
        }

        staging = _atomic_staging_sibling(Path(output_root))
        if staging.exists():
            raise FileExistsError(f"stale GFS staging directory exists: {staging}")
        staging.mkdir(parents=True)
        writer = None
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
            if statics_corridor is not None and len(exp.domains) < 2:
                raise ValueError(
                    "--statics-corridor prepares child-resolution statics "
                    "over the ground a child domain can reach, and this "
                    "experiment has no "
                    "child domain; remove the flag or prepare a domain "
                    "tree")
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
                    root_initial_result=initial_result, root_met=initial_met,
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
                    stock_wrf_export=(
                        "optional" if stock_wrf_export else "off"),
                    statics_corridor=statics_corridor,
                    static_highres=static_highres,
                    sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                    water_temperature_policy=case_policy["water_temperature_policy"],
                    # A child sees the catalog, not the config.  Without
                    # this a WRF-comparison reproduction would disable the
                    # soil reconstitution on the parent and silently keep
                    # it on every nest.
                    soil_texture_downscale=declared_soil_texture_downscale(
                        experiment_config),
                )
                hierarchy_seconds = time.perf_counter() - hierarchy_started
                verify_overlay_sequence(snapshots)
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
                    "initial_condition": provenance,
                    # The coordinate this bundle was built on and why, on
                    # every run: the forecast adopts it from these
                    # artifacts, so a reader must be able to see it here
                    # without the configuration in hand.
                    "vertical_coordinate": _vertical_coordinate_receipt(
                        exp, vertical_adaptation),
                    "source_forecast_hours": list(source_hours),
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
                    # The soil-state SOURCE resolution, on every run: a reader of
                    # this forecast must be able to answer "how coarse was the
                    # soil I started from" without the config, and whether the
                    # sub-source-cell reconstitution ran on it.
                    "soil_texture_downscale": dict(
                        getattr(soil, "soil_texture_downscale", {}) or {}),
                    "decoder_stdout": completed.stdout.strip(),
                    "static_catalog": dict(
                        hierarchy.static_catalog_receipt),
                    "hierarchy_source_coverage": dict(
                        hierarchy.source_coverage_receipt),
                    "hierarchy_topology": dict(
                        hierarchy.topology_receipt),
                    "artifact_receipt": dict(
                        hierarchy.hierarchy.artifacts.receipt),
                    # WHETHER EACH DOMAIN'S INITIALIZATION MODIFIED VAPOUR
                    # ON THE WAY IN, root and children alike.
                    # Unconditional, and stated even when no floor fired:
                    # an absent key would read as "prepared before the
                    # receipt existed", a different claim and one no reader
                    # of the bundle could check.
                    **dict(hierarchy.hierarchy.moisture_floor_receipts),
                    **({"initial_perturbation": initial_perturbation}
                       if initial_perturbation is not None else {}),
                    "wrf_manifest": dict(
                        hierarchy.hierarchy.wrf_manifest),
                    # Present only when the preparation opted in: the
                    # sealed statics-corridor set, digest-bound here the
                    # way every other sealed artifact is.  Absent, the
                    # bundle is byte-for-byte what it always was.
                    **(
                        {"statics_corridor": dict(
                            hierarchy.statics_corridor_receipt)}
                        if hierarchy.statics_corridor_receipt is not None
                        else {}),
                    "timing_seconds": {
                        "verify_inputs": verify_inputs_seconds,
                        "static_build": static_build_seconds,
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
            writer = PreparedTreeWriter(
                staging=staging, output_root=Path(output_root),
                identity=identity)
            preprocessing_receipt_sha256 = hashlib.sha256(json.dumps(
                preprocess_receipt, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            # Everything the proof says that the start time already knows.
            # The seal adds the artifact records, the cache receipt, the
            # export and the wall times (SEAL_ONLY_PROOF_KEYS) and refuses
            # a proof that differs from this head anywhere else.
            proof_head = {
                "schema": PROOF_SCHEMA,
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                # Cycle, lead and start, all three, in the document a
                # forecast run is pinned to.  ``forcing_hours`` below
                # are MODEL offsets from start_time;
                # ``source_forecast_hours`` are the NOAA leads they came
                # from, and the two are equal exactly when the lead is 0.
                "initial_condition": provenance,
                # The coordinate this bundle was built on and why, on
                # every run: the forecast adopts it from these artifacts,
                # so a reader must be able to see it here without the
                # configuration in hand.
                "vertical_coordinate": _vertical_coordinate_receipt(
                    exp, vertical_adaptation),
                "source_forecast_hours": list(source_hours),
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
                # WHETHER THIS INITIALIZATION MODIFIED VAPOUR ON THE WAY
                # IN.  Unconditional, and stated even when no floor fired.
                **moisture_floor_proof_entry(
                    initial_result,
                    when_unrecorded=(
                        "this preparation's initialization result carries "
                        "no moisture-floor field, so it came from an "
                        "ingest predating the receipt; re-prepare the "
                        "case to record whether its vapour was floored "
                        "on the way in")),
                "source_coverage": coverage_receipt,
                # The soil-state SOURCE resolution, on every run: a reader of
                # this forecast must be able to answer "how coarse was the
                # soil I started from" without the config, and whether the
                # sub-source-cell reconstitution ran on it.
                "soil_texture_downscale": dict(
                    getattr(soil, "soil_texture_downscale", {}) or {}),
                "decoder_stdout": completed.stdout.strip(),
                "physics": physics_selection,
                # Whether the unchanged-WRF files were asked for: the
                # forecast reads this beside the export slot, which says
                # whether they were written (READY), declined
                # (NOT_REQUESTED) or refused by name (REFUSED).
                "stock_wrf_export": ("optional" if stock_wrf_export
                                     else "off"),
            }
            # One machine, one card: the forecast may start beside this
            # producer only when both fit (boundary_stream.chained_admission).
            writer.admit(
                experiment=exp, backend=str(preprocess_receipt["backend"]),
                device_bytes=producer_device_bytes(str(preprocess_receipt["backend"])))
            progress.enter("write_prepared_cache")
            cache_started = time.perf_counter()
            writer.write_head(
                initial_result=initial_result, met=initial_met,
                surface=_canonical_surface(soil),
                metadata={
                    "source_adapter": "gfs",
                    "initial_valid_time": times[0].isoformat(),
                    "last_valid_time": times[-1].isoformat(),
                    "forcing_hours": hours,
                    "boundary_interval_seconds": boundary_interval_seconds,
                    "preprocessing": preprocess_receipt,
                },
                lbc={
                    "spec_bdy_width": cfg.spec_bdy_width,
                    "spec_zone": cfg.spec_zone,
                    "relax_zone": cfg.relax_zone,
                    "schedule": [
                        [float(earlier * 3600), float(later * 3600)]
                        for earlier, later in zip(hours, hours[1:])],
                    "fields": forcing.inventory,
                },
                proof_head=proof_head,
                input_manifest_sha256=manifest_digest,
                forcing=forcing,
            )
            cache_seconds = time.perf_counter() - cache_started
            # The start state has done its work: it is in the head.
            del initial_result, initial_met
            release_backend_memory(preprocess)
            progress.enter("initialize_all_times",
                           forcing_times=len(records))
            boundaries_started = time.perf_counter()
            writer.stream_forcing_times(
                count=len(snapshots), build_forcing_time=build_forcing_time,
                forcing=forcing, times=times,
                release=lambda: release_backend_memory(preprocess))
            verify_overlay_sequence(snapshots)
            initialize_seconds += time.perf_counter() - boundaries_started
            progress.enter("write_prepared_cache")
            cache_started = time.perf_counter()
            cache_receipt = writer.seal_cache()
            cache_seconds += time.perf_counter() - cache_started
            portable_cache_receipt = dict(cache_receipt)
            portable_cache_receipt["path"] = prepared_cache.name
            # Where the tree lives now: published at the head when
            # chained, still staged otherwise.
            static_cache = writer.root / static_cache.name
            geometry_receipt = writer.root / geometry_receipt.name
            portable_source_manifest = (
                writer.root / portable_source_manifest.name)
            prepared_cache = writer.cache_path
            wrf_output = writer.root / wrf_output.name
            progress.enter("direct_wrf_export")
            export_started = time.perf_counter()
            export_receipt, _refusal = _single_domain_stock_export(
                prepared_cache, static_cache, geometry_receipt, wrf_output,
                valid_time=times[0],
                boundary_interval_seconds=boundary_interval_seconds,
                physics_selection=physics_selection,
                stock_wrf_export=stock_wrf_export)
            export_seconds = time.perf_counter() - export_started
            verify_overlay_sequence(snapshots)
            final_manifest = _verify_input_manifest(
                Path(input_manifest), manifest_digest, roles)
            if final_manifest != manifest:
                raise ValueError(
                    "GFS input manifest content changed during preparation")
            if _implementation_sha256() != implementation_sha256:
                raise ValueError(
                    "GFS adapter implementation changed during preparation")
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
                    for name, details in (
                        export_receipt.get("files") or {}).items()
                },
            }
            proof = {
                **proof_head,
                "initialization_artifacts": initialization_artifacts,
                "prepared_cache": portable_cache_receipt,
                "export": export_receipt,
                "boundary_stream": writer.boundary_stream_proof(),
                # Every phase this stage passes through, and a `total`
                # its children account for.  `gpuwm.stage_timing` states
                # the rule and the acceptance test holds this document
                # to it: an unnamed phase is time no reader can find.
                "timing_seconds": {
                    "verify_inputs": verify_inputs_seconds,
                    "static_build": static_build_seconds,
                    "decode": decode_seconds,
                    "initialize_all_times": initialize_seconds,
                    "write_prepared_cache": cache_seconds,
                    "direct_wrf_export": export_seconds,
                    "total": time.perf_counter() - total_started,
                },
            }
            progress.enter("publish")
            writer.publish(proof)
            return proof
        except BaseException as error:
            shutil.rmtree(staging, ignore_errors=True)
            if writer is not None:
                # A head already published stays, marked failed, so a
                # waiting forecast ends with this reason and the next
                # preparation of this output root rebuilds it.
                writer.fail(error)
            raise


#: The export receipt schema of the single-domain route.
SINGLE_DOMAIN_EXPORT_SCHEMA = "gpuwm-native-direct-wrf-export-v3"


def _single_domain_stock_export(prepared_cache, static_cache,
                                geometry_receipt, wrf_output, *, valid_time,
                                boundary_interval_seconds: int,
                                physics_selection: Mapping[str, object],
                                stock_wrf_export: bool = True,
                                ) -> tuple[dict[str, object],
                                           StockWrfExportUnsupported | None]:
    """The single-domain route's stock-WRF export slot and any refusal.

    Not requested (``stock_wrf_export`` False), nothing is attempted and
    the slot says NOT_REQUESTED.  Requested, the export runs, but a
    wrfinput for an unchanged WRF cannot represent every preparation
    (WDM6, Kessler and Milbrandt-Yau have no stock package contract),
    which says nothing about running it here: the forecast restores the
    prepared cache, not these files.  Such a slot records the refusal by
    name, as the mapped and HRRR routes write it for the same case, and
    the proof's ``stock_wrf_export = "optional"`` beside it is what the
    forecast reader admits.  An export that runs keeps its READY receipt
    byte for byte, and must carry exactly the selected physics.
    """

    if not stock_wrf_export:
        return (stock_wrf_export_not_requested(
            schema=SINGLE_DOMAIN_EXPORT_SCHEMA), None)
    try:
        receipt = export_prepared_wrf(
            prepared_cache, static_cache, geometry_receipt, wrf_output,
            valid_time=valid_time,
            boundary_interval_seconds=boundary_interval_seconds,
            # The RESOLVED selection, not the caller's argument.  Named,
            # the exporter re-validates the same profile the front door
            # checked.  Unnamed (owner ruling 2026-07-31), the exporter
            # recomputes the experiment-config suite selection from its
            # OWN cache-bound config -- the profileless contract -- and
            # the equality check below proves both spellings agree byte
            # for byte.
            physics_profile=physics_selection["profile"],
            experiment_config_suite=physics_selection["profile"] is None,
            expert_acknowledgements=tuple(
                physics_selection["acknowledgements"]),
            acknowledgement_provenance=physics_selection[
                "acknowledgement_provenance"])
    except StockWrfExportUnsupported as error:
        shutil.rmtree(wrf_output, ignore_errors=True)
        return (stock_wrf_export_refused(
            error, schema=SINGLE_DOMAIN_EXPORT_SCHEMA), error)
    if (receipt.get("schema") != SINGLE_DOMAIN_EXPORT_SCHEMA
            or receipt.get("physics") != physics_selection):
        raise RuntimeError(
            "direct-WRF export physics provenance differs from the "
            "selected preparation profile")
    return receipt, None


def _arg(value) -> str:
    """One printed argument, quoted if a shell would split it.

    A perfectly valid ``--outdir`` or config path with a space in it was
    printed bare, so the command whose entire purpose is to be pasted
    broke the moment it was.  POSIX display form, exactly as
    ``source_cli._quote_command`` renders ``--dry-run`` argv, because
    the certified runtime is Linux/CUDA and forward slashes are accepted
    by every path API on Windows too.  Ordinary paths come back
    unquoted, so this is invisible except where it matters.
    """

    return shlex.quote(str(value).replace("\\", "/"))


def stock_wrf_export_notice(proof: Mapping[str, object]) -> list[str]:
    """Say plainly when the bonus stock-WRF export did not happen.

    The forecast preparation succeeded either way -- that is the point of
    the layering -- but a user who expected ``wrf-native-input/`` has to
    be told it is not there and why, in the same sentence form every
    other refusal on this door uses.  A READY export prints nothing.
    """

    # A domain tree's slot is its hierarchy manifest; a single domain's
    # is the export receipt.  Both carry the same three statuses.
    export = proof.get("wrf_manifest")
    if not isinstance(export, Mapping):
        export = proof.get("export")
    if not isinstance(export, Mapping):
        return []
    status = export.get("status")
    if status in (None, "READY"):
        return []
    if status == "NOT_REQUESTED":
        return ["", "rw-wps --source gfs: no stock-WRF export was requested; "
                    "the prepared forecast is complete."]
    return [
        "",
        "rw-wps --source gfs: the prepared forecast is complete, but the "
        f"bonus stock-WRF export was refused: {export.get('reason')}.",
        "  The prepared forecast itself is unaffected -- run the "
        "forecast command below.",
    ]


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
    error rather than an accurate gap.  Both are resolved here -- the
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
    if not isinstance(manifest_digest, str):
        # The mapped single-domain proof records the published manifest's
        # digest inside its composition receipt instead of at the top
        # level, and it is the SAME digest: the copy this command points
        # the runner at, `source-evidence/input-manifest.json`, is
        # written from those bytes and verified against this value at
        # publication.  Without the fallback a perfectly ordinary
        # single-domain mapped bundle -- every packaged source runs on
        # one -- looked manifest-less here and got the multi-domain
        # hierarchy command printed for it, which is the wrong runner.
        composition = proof.get("source_composition")
        record = (composition.get("input_manifest")
                  if isinstance(composition, Mapping) else None)
        if isinstance(record, Mapping) and isinstance(
                record.get("sha256"), str):
            manifest_digest = record["sha256"]
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
            "  python -m gpuwm.prepared_domain_tree_forecast \\",
            f"      --prepared-root {_arg(output_root)} \\",
            f"      --preparation-receipt-sha256 {proof_digest} \\",
            f"      --experiment-config {_arg(config_path)} \\",
            f"      --experiment-config-sha256 {config_digest} \\",
            f"      --io-mode history --outdir {_arg(outdir)}",
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
    profile_line = (
        [] if profile is None else [f"      --physics-profile {profile} \\"])
    lines += [
        "  python -m gpuwm.prepared_single_domain_forecast \\",
        f"      --source {source} \\",
        f"      --prepared-root {_arg(output_root)} \\",
        f"      --proof-sha256 {proof_digest} \\",
        f"      --source-manifest-sha256 {manifest_digest} \\",
        f"      --prepared-content-sha256 {content} \\",
        f"      --experiment-config {_arg(experiment_config)} \\",
        f"      --wps-namelist {_arg(wps_namelist)} \\",
        *profile_line,
        f"      --io-mode history --outdir {_arg(outdir)}",
        "  (--run-seconds and --history-interval-seconds default to the "
        "hash-bound experiment; the runner refuses any other value.)",
    ]
    if profile is None:
        # One sentence, and the command above runs (owner posture
        # 2026-07-31: verification status is stated, never gating).
        lines.append(
            "  (physics: this config's suite is supported, not yet "
            "WRF-verified; the runner executes it as written.)")
    return lines


def _parser() -> argparse.ArgumentParser:
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
        default="auto",
        help="where the source-grid/WRF-real setup transforms run "
             "(default auto: CUDA when the certified runtime is usable, "
             "otherwise the deterministic parallel CPU backend, "
             "announced in one line)")
    # Why a caller's configuration policy named the backend; recorded in
    # the receipt's selection block in place of "named by the caller".
    parser.add_argument("--preprocess-backend-reason",
                        help=argparse.SUPPRESS)
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--geog-root", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    parser.add_argument(
        "--physics-profile", default=None,
        help="optional assertion that the experiment IS this registered "
             "fixed physics template, refused on any switch drift; "
             "omitted, the suite the config selects runs as written on "
             "both routes and its WRF-verification status is reported, "
             "never gating")
    parser.add_argument(
        "--ack", action="append", default=[],
        help="registry-owned expert acknowledgement id; repeat as needed")
    parser.add_argument(
        "--no-stock-wrf-export", dest="stock_wrf_export",
        action="store_false", default=True,
        help="prepare the forecast only, and do not attempt the bonus "
             "unchanged-WRF wrfinput/wrfbdy export")
    parser.add_argument(
        "--statics-corridor", nargs="?", const="all", default=None,
        metavar="GRID_IDS",
        help="also seal child-resolution statics over the ground each "
             "child can reach (the moving-nest corridor); bare flag covers "
             "every child domain, or pass comma-separated child grid ids "
             "(e.g. 2,3).  Required before the prepared tree runner will "
             "honor a [relocation] follow source; omitted, the bundle is "
             "byte-for-byte unchanged")
    return parser


@owns_source_coverage_refusal
def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    statics_corridor = args.statics_corridor
    if statics_corridor is not None and statics_corridor != "all":
        try:
            statics_corridor = tuple(
                int(part) for part in statics_corridor.split(",") if part)
        except ValueError:
            print("rw-wps --source gfs: --statics-corridor accepts 'all' "
                  f"or comma-separated child grid ids, got "
                  f"{args.statics_corridor!r}.", file=sys.stderr)
            return 2
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
            preprocess_backend_reason=args.preprocess_backend_reason,
            preprocess_workers=args.preprocess_workers,
            cpu_preprocess_bridge=args.cpu_preprocess_bridge,
            geog_root=args.geog_root,
            hierarchy_workers=args.hierarchy_workers,
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.ack),
            stock_wrf_export=args.stock_wrf_export,
            statics_corridor=statics_corridor,
        )
    except PreparationRefusal:
        # The decorator on this main owns the whole refusal family: two
        # lines, the message and ITS remedy.  Catching it here as a
        # plain ValueError flattened the remedy away -- the eta-ladder
        # refusal reached the walk as one sentence with no door named
        # (UX finding R2) -- so it is re-raised to the owner.
        raise
    except InitializationMemoryRefused as error:
        # A preparation the card cannot hold under an explicit
        # --preprocess-backend cuda (PreparationDeviceRefused), or any
        # other measured initialization budget that refuses the case, is
        # a refusal with its own remedy line: one message at exit 2, as
        # `gpuwm` itself answers it, not a traceback that buries the
        # remedy at the bottom of a stack.
        print(f"rw-wps --source gfs: {error}", file=sys.stderr)
        return 2
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
    corridor = proof.get("statics_corridor")
    if isinstance(corridor, dict):
        # Size accuracy at the door: the corridor's cost, and the ground
        # it covers, are stated where they are paid.
        from gpuwm.static.corridor import corridor_summary_line
        for label, entry in sorted(corridor.get("domains", {}).items()):
            print(corridor_summary_line(label, entry), file=sys.stderr)
    for line in stock_wrf_export_notice(proof):
        print(line, file=sys.stderr)
    for line in prepared_forecast_next_command(
            proof, output_root=args.output_root,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist):
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
