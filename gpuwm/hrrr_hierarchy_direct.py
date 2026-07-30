"""Sealed HRRR root preparation -> parallel nested stock-WRF inputs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import tomllib
from types import SimpleNamespace

from gpuwm import __version__
from gpuwm.experiment import build_experiment
from gpuwm.hrrr_native_static import (
    sha256_file,
    verified_static_catalog,
    verify_hrrr_native_static,
)
from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
from gpuwm.ingest.hrrr import load_hrrr_native_series
from gpuwm.ingest.hrrr_target import load_hrrr_target_domain
from gpuwm.ingest.nest_init import NestedInputCatalog, ParentInitView
from gpuwm.ingest.prepared_cache import (
    PreparedCacheReader,
    compare_prepared_domain_config,
    prepared_domain_config_identity,
    prepared_cache_identity,
    restore_prepared_cache,
)
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.namelist_import import import_namelists, parse_namelist
from gpuwm.native_domain_artifacts import _atomic_staging_sibling
from gpuwm.native_hierarchy import initialize_and_export_native_hierarchy
from gpuwm.native_wrf_contract import validate_native_lambert_contracts
from gpuwm.static.lambert import grids_from_projection_config
from gpuwm.core.microphysics_transition import resolve_microphysics_transition


SCHEMA = "gpuwm-native-hrrr-hierarchy-direct-v1"
_SUPPORTED_PHYSICS = {
    "bl_pbl_physics": 1,
    "sf_sfclay_physics": 91,
    "sf_surface_physics": 2,
    "ra_physics": 0,
    "ra_sw_physics": 1,
    "cu_physics": 0,
}
_SUPPORTED_MICROPHYSICS = frozenset({6, 8, 10, 18})
_DOMAIN_PREPARATION_OVERRIDES = frozenset({
    "cu_physics", "cudt_minutes", "radt", "radt_minutes", "bldt",
    "diff_6th_factor", "epssm", "spec_exp", "mp_physics", "moist",
    "moist_cq", "nest_microphysics_transition",
})
_MAX_PUBLIC_DOMAINS = 21


def _json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _same_typed_values(observed, expected) -> bool:
    return (isinstance(observed, list)
            and len(observed) == len(expected)
            and all(type(left) is type(right) and left == right
                    for left, right in zip(observed, expected)))


def _native_experiment(wps_namelist: Path, namelist_input: Path):
    resolved, report = import_namelists(
        wps_namelist, namelist_input, name="native_hrrr_hierarchy")
    exp = build_experiment(
        tomllib.loads(resolved),
        source=f"native HRRR hierarchy {wps_namelist} + {namelist_input}",
    )
    validate_native_lambert_contracts(
        exp, wps_namelist, source_name="native HRRR hierarchy")
    return exp, resolved, report


def _require_raw_stock_delta(
        native_namelist: Path,
        stock_namelist: Path,
) -> dict[str, object]:
    """Require only the explicit native-to-stock runtime deltas.

    ``ghg_input=0`` is stock-only because this gate substitutes WRF's legacy
    RRTM longwave for native preparation's disabled longwave.  Pinning it is
    nevertheless mandatory: WRF's default ``1`` reads the time-varying CAM
    gas table and is not a valid implicit substitute for the fixed-gas RRTM
    configuration used by the acceptance gate.
    """

    native = parse_namelist(native_namelist)
    stock = parse_namelist(stock_namelist)
    try:
        native_max_dom = native["domains"]["max_dom"]
        stock_max_dom = stock["domains"]["max_dom"]
        native_lw = native["physics"]["ra_lw_physics"]
        stock_lw = stock["physics"]["ra_lw_physics"]
        native_theta = native["dynamics"]["use_theta_m"]
        stock_theta = stock["dynamics"]["use_theta_m"]
        stock_ghg = stock["physics"]["ghg_input"]
    except KeyError as exc:
        raise ValueError(
            "both native and stock namelists must explicitly declare "
            "&domains/max_dom, "
            "&physics/ra_lw_physics and &dynamics/use_theta_m; the stock "
            "namelist must also declare &physics/ghg_input") from exc
    if (len(native_max_dom) != 1 or isinstance(native_max_dom[0], bool)
            or not isinstance(native_max_dom[0], int)
            or not 1 <= native_max_dom[0] <= _MAX_PUBLIC_DOMAINS
            or not _same_typed_values(stock_max_dom, native_max_dom)):
        raise ValueError(
            "native and stock namelists must explicitly declare the same "
            f"integer max_dom in [1, {_MAX_PUBLIC_DOMAINS}]")
    max_dom = native_max_dom[0]
    certified_pins = {
        ("time_control", "interval_seconds"): [3600],
        ("time_control", "frames_per_outfile"): [1] * max_dom,
        ("time_control", "restart"): [False],
        ("time_control", "io_form_history"): [2],
        ("time_control", "io_form_restart"): [2],
        ("time_control", "io_form_input"): [2],
        ("time_control", "io_form_boundary"): [2],
        ("domains", "num_metgrid_levels"): [51],
        ("domains", "num_metgrid_soil_levels"): [9],
        ("domains", "sfcp_to_sfcp"): [True],
        ("physics", "isfflx"): [1],
        ("physics", "ifsnow"): [1],
        ("physics", "icloud"): [1],
        ("physics", "surface_input_source"): [1],
        ("physics", "num_soil_layers"): [4],
        ("physics", "sf_urban_physics"): [0] * max_dom,
        ("physics", "sst_update"): [0],
    }
    observed_pins = {}
    for (section, key), expected in certified_pins.items():
        observed = native.get(section, {}).get(key)
        observed_pins[f"{section}.{key}"] = observed
        if not _same_typed_values(observed, expected):
            raise ValueError(
                "native hierarchy namelist is outside the certified raw "
                f"runtime contract: &{section}/{key} expected {expected}, "
                f"got {observed}")
    for section, key in (
            ("time_control", "nwp_diagnostics"),
            ("physics", "do_radar_ref")):
        if key in native.get(section, {}):
            raise ValueError(
                "native hierarchy namelist is outside the certified raw "
                f"runtime contract: &{section}/{key} must be omitted")

    native_lw_expected = [0] * max_dom
    stock_lw_expected = [1] * max_dom
    if (not _same_typed_values(native_lw, native_lw_expected)
            or not _same_typed_values(stock_lw, stock_lw_expected)
            or not _same_typed_values(native_theta, [0])
            or not _same_typed_values(stock_theta, [1])
            or "ghg_input" in native["physics"]
            or not _same_typed_values(stock_ghg, [0])):
        raise ValueError(
            "the evidenced raw namelist deltas require one explicit "
            "ra_lw_physics=0 -> 1 entry per domain, use_theta_m=0 -> 1, "
            "and stock-only ghg_input=0")
    normalized_stock = deepcopy(stock)
    normalized_stock["physics"]["ra_lw_physics"] = list(native_lw)
    normalized_stock["physics"].pop("ghg_input")
    normalized_stock["dynamics"]["use_theta_m"] = list(native_theta)
    if native != normalized_stock:
        differences = []
        for section in sorted(set(native) | set(normalized_stock)):
            left = native.get(section, {})
            right = normalized_stock.get(section, {})
            for key in sorted(set(left) | set(right)):
                if left.get(key) != right.get(key):
                    differences.append(f"&{section}/{key}")
        raise ValueError(
            "stock-WRF namelist raw settings differ beyond "
            "ra_lw_physics 0 -> 1, use_theta_m 0 -> 1, and stock-only "
            "ghg_input=0: "
            + ", ".join(differences))
    return {
        "schema": "gpuwm-native-to-stock-namelist-delta-v3",
        "status": "PASS",
        "max_dom": max_dom,
        "certified_native_runtime": observed_pins,
        "allowed_deltas": {
            "physics.ra_lw_physics": {
                "native": native_lw_expected, "stock": stock_lw_expected},
            "dynamics.use_theta_m": {"native": [0], "stock": [1]},
            "physics.ghg_input": {"native": None, "stock": [0]},
        },
    }


def _require_raw_wps_contract(
        wps_namelist: Path, max_dom: int) -> dict[str, object]:
    wps = parse_namelist(wps_namelist)
    required = {
        ("share", "max_dom"): [max_dom],
        ("share", "interval_seconds"): [3600],
    }
    observed = {}
    for (section, key), expected in required.items():
        value = wps.get(section, {}).get(key)
        observed[f"{section}.{key}"] = value
        if not _same_typed_values(value, expected):
            raise ValueError(
                "WPS namelist is outside the certified raw hierarchy "
                f"contract: &{section}/{key} expected {expected}, got "
                f"{value}")
    return {
        "schema": "gpuwm-native-wps-hierarchy-contract-v1",
        "status": "PASS",
        "max_dom": max_dom,
        "certified": observed,
    }


def _domain_layout(domain) -> dict[str, object]:
    return {
        "grid_id": domain.grid_id,
        "parent_id": domain.parent_id,
        "i_parent_start": domain.i_parent_start,
        "j_parent_start": domain.j_parent_start,
        "parent_grid_ratio": domain.parent_grid_ratio,
        "parent_time_step_ratio": domain.parent_time_step_ratio,
        "nx": domain.run.nx, "ny": domain.run.ny,
        "nz": domain.run.nz, "dx": domain.run.dx,
        "dy": domain.run.dy, "dt": domain.run.dt,
        "specified": domain.run.specified,
        "nested": domain.run.nested,
        "history_interval_s": domain.history_interval_s,
    }


def _supported_hierarchy_slice(exp, root_target) -> None:
    """Bind the public adapter to its sealed root and generic valid nests.

    ``build_experiment`` is the geometry authority: it rejects cycles,
    orphans, invalid ratios, misaligned extents, and insufficient parent-row
    clearance before this gate runs.  The target-domain document and sealed
    root preparation are the d01 authority; this function verifies that the
    requested namelist describes that root without pinning one location,
    duration, or vertical grid.  The fixed native HRRR physics product remains
    explicit and children may vary only in their geometry and nest identity.
    """

    if not 1 <= len(exp.domains) <= _MAX_PUBLIC_DOMAINS:
        raise ValueError(
            "the public HRRR hierarchy gate requires between 1 and "
            f"{_MAX_PUBLIC_DOMAINS} domains")
    domain_ids = tuple(domain.grid_id for domain in exp.domains)
    expected_ids = tuple(range(1, len(exp.domains) + 1))
    if domain_ids != expected_ids:
        raise ValueError(
            "the public HRRR hierarchy requires contiguous, parent-before-"
            f"child grid ids {expected_ids}, got {domain_ids}")
    if exp.feedback != 0 or exp.smooth_option != 0:
        raise ValueError("the public HRRR hierarchy gate is one-way only")
    if not 0.0 < exp.run_seconds <= 43_200.0:
        raise ValueError(
            "the public HRRR hierarchy duration must be positive and no "
            "longer than the native f00..f12 forcing horizon")
    expected_projection = {
        "map_proj": root_target.map_proj,
        "ref_lat": root_target.ref_lat,
        "ref_lon": root_target.ref_lon,
        "truelat1": root_target.truelat1,
        "truelat2": root_target.truelat2,
        "stand_lon": root_target.stand_lon,
    }
    if exp.projection is None or asdict(exp.projection) != expected_projection:
        raise ValueError(
            "the public HRRR hierarchy projection differs from the sealed "
            "root target-domain document")
    observed_root = _domain_layout(exp.domains[0])
    expected_root = {
        "grid_id": 1,
        "parent_id": 0,
        "i_parent_start": 1,
        "j_parent_start": 1,
        "parent_grid_ratio": 1,
        "parent_time_step_ratio": 1,
        "nx": root_target.nx,
        "ny": root_target.ny,
        "nz": root_target.nz,
        "dx": root_target.dx_m,
        "dy": root_target.dy_m,
        "dt": float(root_target.time_step_seconds),
        "specified": True,
        "nested": False,
        "history_interval_s": observed_root["history_interval_s"],
    }
    if observed_root != expected_root:
        raise ValueError(
            "d01 differs from the sealed HRRR root target-domain document: "
            f"{observed_root}")
    root_run = exp.domains[0].run
    if (exp.spec_bdy_width != root_target.spec_bdy_width
            or root_run.spec_zone != root_target.spec_zone
            or root_run.relax_zone != root_target.relax_zone):
        raise ValueError(
            "d01 boundary-zone controls differ from the sealed HRRR root "
            "target-domain document")

    root_domain = exp.domains[0]
    root_run = asdict(root_run)
    seen = {1}
    by_id = {domain.grid_id: domain for domain in exp.domains}
    for domain in exp.domains:
        if domain.grid_id > 1:
            if domain.parent_id not in seen:
                raise ValueError(
                    f"d{domain.grid_id:02d} names unavailable parent "
                    f"d{domain.parent_id:02d}; domains must be stored "
                    "parent-before-child")
            if domain.parent_id == domain.grid_id:
                raise ValueError(
                    f"d{domain.grid_id:02d} cannot be its own parent")
        seen.add(domain.grid_id)
        if domain.run.nz != root_target.nz:
            raise ValueError(
                f"d{domain.grid_id:02d} must use the sealed root's "
                f"{root_target.nz}-level vertical grid")
        if domain.run.mp_physics not in _SUPPORTED_MICROPHYSICS:
            raise ValueError(
                f"d{domain.grid_id:02d} requests MP"
                f"{domain.run.mp_physics}; HRRR hierarchy preparation "
                f"supports {sorted(_SUPPORTED_MICROPHYSICS)}")
        mismatch = {
            name: (getattr(domain.run, name), expected)
            for name, expected in _SUPPORTED_PHYSICS.items()
            if getattr(domain.run, name) != expected
        }
        if mismatch:
            raise ValueError(
                f"d{domain.grid_id:02d} is outside the certified native "
                f"HRRR physics slice: {mismatch}")
        if domain.run.ra_lw_physics != 0:
            raise ValueError(
                "native hierarchy preparation requires ra_lw_physics=0; "
                "the separately supplied stock-WRF namelist must select 1")
        expected_run = dict(root_run)
        expected_run.update({
            "grid_id": domain.grid_id,
            "nx": domain.run.nx, "ny": domain.run.ny,
            "dx": domain.run.dx, "dy": domain.run.dy,
            "dt": domain.run.dt,
            "specified": domain.grid_id == 1,
            "nested": domain.grid_id != 1,
        })
        observed_run = asdict(domain.run)
        drift = {
            name: (observed_run[name], expected_run[name])
            for name in observed_run
            if observed_run[name] != expected_run[name]
            and name not in _DOMAIN_PREPARATION_OVERRIDES
        }
        if drift:
            raise ValueError(
                f"d{domain.grid_id:02d} trajectory controls differ from "
                f"the HRRR root outside supported per-domain overrides: "
                f"{drift}")
        if domain is not root_domain:
            # This is the executable authority, including the exact MP8->MP18
            # diagnosis contract.  Unsupported mixed pairs and missing CQ
            # requirements remain genuine compatibility errors; an admitted
            # but not acceptance-gated edge is recorded as unverified by the
            # forecast runner rather than blocked for consent.
            resolve_microphysics_transition(
                by_id[domain.parent_id].run, domain.run)


def _compare_stock_experiment(native_exp, stock_exp) -> None:
    native = asdict(native_exp)
    stock = asdict(stock_exp)
    for document in (native, stock):
        document["name"] = "normalized"
        for domain in document["domains"]:
            domain["run"]["ra_lw_physics"] = 0
    if native != stock:
        raise ValueError(
            "stock-WRF namelist differs from the native hierarchy setup "
            "beyond the allowed ra_lw_physics 0 -> 1 runtime change")
    if any(domain.run.ra_lw_physics != 1 for domain in stock_exp.domains):
        raise ValueError("stock-WRF namelist must select ra_lw_physics=1")


def _effective_domain_config(domain) -> dict[str, object]:
    """Canonicalize cadence fields that are inactive under selected schemes."""

    document = prepared_domain_config_identity(domain)
    run = document["run"]
    if run["radt"] > 0.0:
        run["radt_minutes"] = run["radt"]
    if run["cu_physics"] == 0:
        run["cudt_minutes"] = 0.0
    return document


def _validated_root_preparation_binding(
        identity: dict[str, object], root_domain,
) -> tuple[str, dict[str, object]]:
    """Authorize cache reuse by exact effective-d01 trajectory equality.

    A prepared root is intentionally independent of later child topology.
    Its original full-namelist digest remains part of the immutable cache
    identity, while the requested hierarchy namelist is bound separately in
    the hierarchy receipt.  Reuse is allowed only when the cache's resolved
    d01 configuration is exactly the requested effective d01 configuration.
    """

    prepared_domain = identity.get("domain_config")
    if not isinstance(prepared_domain, dict):
        raise ValueError("root preparation identity lacks domain_config")
    effective_prepared_domain = deepcopy(prepared_domain)
    prepared_run = effective_prepared_domain.get("run", {})
    if not isinstance(prepared_run, dict):
        raise ValueError("root preparation domain_config lacks run controls")
    if prepared_run.get("radt", 0.0) > 0.0:
        prepared_run["radt_minutes"] = prepared_run["radt"]
    if prepared_run.get("cu_physics") == 0:
        prepared_run["cudt_minutes"] = 0.0
    # Same rule as every other prepared-identity comparison: a field the
    # header predates, holding its not-in-use default, is schema growth
    # and not a trajectory difference.  Everything else still refuses.
    live_root = _effective_domain_config(root_domain)
    # d01 is never a delayed domain -- the delayed-start rules apply to
    # exp.domains[1:] -- so the root's own start IS the experiment's, and
    # that is precisely the not-in-use value a header written before the
    # field existed describes.
    _, differing = compare_prepared_domain_config(
        effective_prepared_domain, live_root,
        not_in_use={"start_time": live_root.get("start_time")})
    if differing:
        raise ValueError(
            "native namelist d01 trajectory controls differ from the sealed "
            f"root preparation: {', '.join(sorted(differing))}")

    root_namelist_sha256 = identity.get("namelist_sha256")
    if (not isinstance(root_namelist_sha256, str)
            or len(root_namelist_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in root_namelist_sha256)):
        raise ValueError(
            "root preparation identity has an invalid namelist_sha256")
    return root_namelist_sha256, prepared_domain


def _source_identity(cpu_bridge: Path) -> dict[str, object]:
    package = Path(__file__).resolve().parent
    root = package.parent
    distribution = os.environ.get("GPUWM_NATIVE_DISTRIBUTION_MANIFEST")
    if distribution:
        manifest_path = Path(distribution).resolve()
        manifest = _json(manifest_path)
        source = manifest.get("source")
        artifact = manifest.get("artifact")
        if (manifest.get("schema") != "gpuwm-native-wrf-runtime-v1"
                or manifest.get("status") != "READY"
                or not isinstance(source, dict)
                or not source.get("worktree_clean")
                or not isinstance(artifact, dict)
                or artifact.get("gpuwm_version") != __version__):
            raise RuntimeError(
                "unrecognized or incompatible native distribution manifest")
        return {
            "identity_source": "gpuwm-native-distribution-manifest",
            "git_commit": source.get("commit"),
            "git_tree": source.get("tree"),
            "distribution_manifest_sha256": sha256_file(manifest_path),
            "cpu_preprocess_bridge": {
                "path": str(cpu_bridge),
                "bytes": cpu_bridge.stat().st_size,
                "sha256": sha256_file(cpu_bridge),
            },
        }
    identity: dict[str, object] = {
        "identity_source": "git",
        "cpu_preprocess_bridge": {
            "path": str(cpu_bridge),
            "bytes": cpu_bridge.stat().st_size,
            "sha256": sha256_file(cpu_bridge),
        },
    }
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True,
            stderr=subprocess.DEVNULL).strip()

    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top != root.resolve():
        raise RuntimeError(f"git provenance escaped the gpuwm tree: {top}")
    status = git("status", "--porcelain").splitlines()
    if status:
        raise RuntimeError(
            "public HRRR hierarchy requires a completely clean source tree")
    identity.update({
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "git_status_short": [],
    })
    return identity


def _surface_state(restored, static_fields, *, sf_surface_physics):
    if restored.surface is None:
        return preprocess_land_surface_soil(
            restored.met.fields,
            sf_surface_physics=int(sf_surface_physics),
            soil_type=static_fields["SCT_DOM"],
            deep_soil_temperature=static_fields["TMN"],
        )
    fields = restored.surface.fields
    return SimpleNamespace(
        tsk=fields["TSK"],
        soil_temperature=fields["TSLB"],
        soil_moisture=fields["SMOIS"],
        liquid_moisture=fields["SH2O"],
        deep_soil_temperature=fields["TMN"],
        xice=fields["SEAICE"],
        xland=fields["XLAND"],
        landmask=fields["LANDMASK"],
        snow_water=fields["SNOW"],
        snow_depth=fields["SNOWH"],
    )


def _root_paths(root: Path) -> dict[str, Path]:
    return {
        "static_cache": root / "native-static.npz",
        "static_receipt": root / "native-static-receipt.json",
        "prepared_cache": root / "native" / "prepared-cache",
        "bridge": root / "native" / "native-bridge",
        "bridge_manifest": root / "native" / "native-bridge" / "SHA256SUMS",
        "preparation_report": (
            root / "native" / "preparation-report" / "report.json"),
    }


def prepare_hrrr_hierarchy(
        *, root_preparation: Path, root_domain_spec: Path,
        wps_namelist: Path, namelist_input: Path,
        stock_wrf_namelist_input: Path, geog_root: Path,
        source_manifest: Path, source_manifest_sha256: str,
        valid_time: datetime, output_root: Path, workers: int = 8,
        cpu_bridge: Path | None = None,
) -> dict[str, object]:
    """Verify one root preparation and publish generic stock-WRF inputs."""

    if isinstance(workers, bool) or workers not in range(1, 33):
        raise ValueError("workers must be an integer from 1 through 32")
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing existing output root {output_root}")
    cpu_bridge = resolve_cpu_bridge(cpu_bridge)
    paths = _root_paths(Path(root_preparation))
    required = (
        root_domain_spec, wps_namelist, namelist_input,
        stock_wrf_namelist_input, geog_root, source_manifest, cpu_bridge,
        *paths.values(),
    )
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    stock_runtime_delta = _require_raw_stock_delta(
        Path(namelist_input), Path(stock_wrf_namelist_input))
    native_exp, native_resolved, native_report = _native_experiment(
        Path(wps_namelist), Path(namelist_input))
    target = load_hrrr_target_domain(root_domain_spec)
    _supported_hierarchy_slice(native_exp, target)
    wps_runtime_contract = _require_raw_wps_contract(
        Path(wps_namelist), len(native_exp.domains))
    if native_exp.start_time != valid_time:
        raise ValueError("valid time differs from the native namelist start")

    static_fields, static_receipt = verify_hrrr_native_static(
        paths["static_cache"], paths["static_receipt"], target)
    cache_header = _json(paths["prepared_cache"] / "header.json")
    if (cache_header.get("schema") != "gpuwm-prepared-real-cache-v1"
            or cache_header.get("status") != "READY"):
        raise ValueError("root preparation cache is not READY")
    identity = cache_header.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("root preparation cache lacks an identity")
    root_namelist_sha, prepared_domain = _validated_root_preparation_binding(
        identity, native_exp.root)

    observed_source_sha = sha256_file(Path(source_manifest))
    if (observed_source_sha != source_manifest_sha256
            or identity.get("source_manifest_sha256") != observed_source_sha):
        raise ValueError("source manifest differs from the root preparation")
    bridge_sha = sha256_file(paths["bridge_manifest"])
    if identity.get("bridge_manifest_sha256") != bridge_sha:
        raise ValueError("HRRR bridge manifest differs from root preparation")
    static_sha = sha256_file(paths["static_cache"])
    if identity.get("static_cache_sha256") != static_sha:
        raise ValueError("static cache differs from root preparation identity")
    namelist_sha = sha256_file(Path(namelist_input))
    forcing_hours = tuple(identity.get("forcing_hours", ()))
    expected_forcing_hours = tuple(range(
        0, int(native_exp.run_seconds // 3600) + 1))
    if forcing_hours != expected_forcing_hours:
        raise ValueError(
            "root preparation must contain consecutive hourly HRRR forcing "
            f"through the forecast endpoint: expected {expected_forcing_hours}, "
            f"got {forcing_hours}")

    preparation_report = _json(paths["preparation_report"])
    if preparation_report.get("status") != "PASS":
        raise ValueError("root preparation report is not PASS")
    if preparation_report.get("source_identity") != identity.get("source_identity"):
        raise ValueError("root preparation source identity differs from cache")
    prepared_report = preparation_report.get("prepared_cache", {})
    if prepared_report.get("content_sha256") != cache_header.get(
            "content_sha256"):
        raise ValueError("root preparation report does not bind its cache")
    user_metadata = cache_header.get("metadata", {}).get("user", {})
    if datetime.fromisoformat(user_metadata.get("initial_valid_time", "")) \
            != valid_time:
        raise ValueError("root preparation valid time differs from request")

    expected_identity = prepared_cache_identity(
        bridge_manifest_sha256=bridge_sha,
        source_manifest_sha256=observed_source_sha,
        static_cache_sha256=static_sha,
        namelist_sha256=root_namelist_sha,
        domain_config=native_exp.root,
        forcing_hours=forcing_hours,
        source_identity=identity["source_identity"],
    )
    # The sealed single-domain preparer stores legacy cadence shadow fields.
    # They are trajectory-inactive when positive ``radt`` overrides
    # ``radt_minutes`` and cumulus is disabled.  Their effective equality was
    # checked above; preserve the exact sealed document for cache verification.
    expected_identity["domain_config"] = prepared_domain
    reader = PreparedCacheReader(
        paths["prepared_cache"], expected_identity=expected_identity)
    reader.verify_all()
    preflight_seconds = time.perf_counter() - started

    restore_started = time.perf_counter()
    restored = restore_prepared_cache(
        paths["prepared_cache"], expected_identity=expected_identity,
        cfg=native_exp.root.run, static=static_fields)
    root_soil = _surface_state(
        restored, static_fields,
        sf_surface_physics=native_exp.root.run.sf_surface_physics)
    restore_seconds = time.perf_counter() - restore_started

    snapshots_started = time.perf_counter()
    snapshots = load_hrrr_native_series(
        paths["bridge"], forcing_hours[:1],
        expected_manifest_sha256=bridge_sha)
    static_catalog, catalog_receipt = verified_static_catalog(
        Path(wps_namelist), Path(geog_root),
        [domain.grid_id for domain in native_exp.domains])
    if catalog_receipt["selections"]["d01"] != static_receipt.get(
            "geog_selection"):
        raise ValueError(
            "WPS d01 GEOG selection differs from the sealed root static "
            "selection")
    catalog = NestedInputCatalog(
        snapshots=tuple(snapshots), static_catalog=static_catalog,
        files=tuple(static_catalog.files),
        provenance={
            "adapter": "native-HRRR-hierarchy-direct-v1",
            "bridge_manifest_sha256": bridge_sha,
            "static_catalog_receipt": catalog_receipt,
            "surface_fallback_radius_cells": (
                target.surface_fallback_radius_cells),
        })
    snapshot_seconds = time.perf_counter() - snapshots_started

    grids = tuple(grids_from_projection_config(native_exp))
    root = ParentInitView(
        cfg=native_exp.root, grid=grids[0], state=restored.initial_result.state)
    hierarchy_identity = _source_identity(cpu_bridge)
    provenance = {
        "source_manifest_sha256": observed_source_sha,
        "bridge_manifest_sha256": bridge_sha,
        "root_static_receipt_sha256": sha256_file(paths["static_receipt"]),
        "root_prepared_content_sha256": restored.receipt["content_sha256"],
        "root_preparation_namelist_sha256": root_namelist_sha,
        "wps_namelist_sha256": sha256_file(Path(wps_namelist)),
        "native_namelist_input_sha256": namelist_sha,
        "stock_wrf_namelist_input_sha256": sha256_file(
            Path(stock_wrf_namelist_input)),
        "native_resolved_experiment_sha256": hashlib.sha256(
            native_resolved.encode("utf-8")).hexdigest(),
        "native_translation_report": asdict(native_report),
        "stock_runtime_delta": stock_runtime_delta,
        "wps_runtime_contract": wps_runtime_contract,
        "static_catalog": catalog_receipt,
        "surface_fallback_radius_cells": (
            target.surface_fallback_radius_cells),
        "hierarchy_source_identity": hierarchy_identity,
    }

    hierarchy_source_identity = {
        "root_preparation": identity["source_identity"],
        "hierarchy": hierarchy_identity,
        "static_catalog": catalog_receipt,
        "wps_namelist_sha256": provenance["wps_namelist_sha256"],
        "native_namelist_input_sha256": namelist_sha,
        "stock_wrf_namelist_input_sha256": provenance[
            "stock_wrf_namelist_input_sha256"],
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = _atomic_staging_sibling(output_root)
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        hierarchy_started = time.perf_counter()
        result = initialize_and_export_native_hierarchy(
            exp=native_exp, root_node=root, catalog=catalog,
            artifact_output=staging / "hierarchy-artifacts",
            wrf_output=staging / "wrf-native-input",
            root_initial_result=restored.initial_result,
            root_met=restored.met, root_soil=root_soil,
            root_static_fields=static_fields,
            root_boundaries=restored.boundaries,
            bridge_manifest_sha256=bridge_sha,
            source_manifest_sha256=observed_source_sha,
            namelist_sha256=namelist_sha,
            forcing_hours=forcing_hours,
            source_identity=hierarchy_source_identity,
            workers=workers, preprocess_backend="cpu", cpu_bridge=cpu_bridge,
            boundary_interval_seconds=3600,
            root_metadata={
                "source_prepared_content_sha256": restored.receipt[
                    "content_sha256"],
            },
            input_provenance=provenance,
            artifact_manifest_reference=(
                "../hierarchy-artifacts/domain-artifacts.json"),
        )
        hierarchy_seconds = time.perf_counter() - hierarchy_started
        stock_copy = staging / "namelist.input"
        shutil.copyfile(stock_wrf_namelist_input, stock_copy)
        payload = {
            "schema": SCHEMA,
            "status": "PASS",
            "valid_time": valid_time.isoformat(),
            "workers": workers,
            "preprocess_backend": "cpu",
            "domain_count": len(native_exp.domains),
            "forcing_hours": list(forcing_hours),
            "provenance": provenance,
            "stock_wrf_namelist": {
                "path": stock_copy.name,
                "sha256": sha256_file(stock_copy),
            },
            "timing_seconds": {
                "verified_preflight": preflight_seconds,
                "restore_root_prepared_cache": restore_seconds,
                "load_initial_snapshot_and_static_catalog": snapshot_seconds,
                **dict(result.timings_seconds),
                "hierarchy_call_wall": hierarchy_seconds,
                "total": time.perf_counter() - started,
            },
            "artifact_receipt": dict(result.artifacts.receipt),
            "wrf_manifest": dict(result.wrf_manifest),
        }
        receipt = staging / "receipt.json"
        temporary = receipt.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(temporary, receipt)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-preparation", type=Path, required=True)
    parser.add_argument("--root-domain-spec", type=Path, required=True)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument(
        "--stock-wrf-namelist-input", type=Path, required=True)
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--valid-time", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        valid_time = datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            "valid-time must use YYYY-MM-DD_HH:MM:SS") from exc
    payload = prepare_hrrr_hierarchy(
        root_preparation=args.root_preparation,
        root_domain_spec=args.root_domain_spec,
        wps_namelist=args.wps_namelist,
        namelist_input=args.namelist_input,
        stock_wrf_namelist_input=args.stock_wrf_namelist_input,
        geog_root=args.geog_root,
        source_manifest=args.source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        valid_time=valid_time, output_root=args.output_root,
        workers=args.workers, cpu_bridge=args.cpu_preprocess_bridge,
    )
    print(json.dumps({
        "status": payload["status"],
        "output_root": str(args.output_root.resolve()),
        "workers": payload["workers"],
        "timing_seconds": payload["timing_seconds"],
        "wrf_files": payload["wrf_manifest"]["files"],
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
