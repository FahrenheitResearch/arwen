"""Mapped-source initialization and atomic stock-WRF hierarchy export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Mapping, Sequence
import uuid

import numpy as np

from gpuwm.core.grid import make_vertical_coord
from gpuwm.experiment import load_experiment, validate_boundary_timing
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.soil_downscale import (
    declared_soil_texture_downscale, soil_mesh_plan_from_case)
from gpuwm.ingest.lateral_bc import (
    StateBoundaryFrames,
    attach_lateral_boundaries,
    start_last_forcing_order,
)
from gpuwm.ingest.prepared_cache import (
    prepared_cache_identity,
    write_prepared_cache,
)
from gpuwm.ingest.preprocess_backend import (
    release_backend_memory,
    resolve_preprocess_backend,
)
from gpuwm.ingest.real import initialize_real
from gpuwm.ingest.source_coverage import (
    PreparationRefusal,
    RunInputRefusal,
    VerticalLadderRefusal,
    owns_source_coverage_refusal,
)
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.ingest.water_temperature import WaterTemperatureStatics
from gpuwm.mapped_composition import (
    MappedSourceBundle,
    _decoder_inventory,
    _path_inventory,
    decode_composed_source,
    mapped_composition_receipt,
)
from gpuwm.mapped_engine_bridge import (
    ENGINE_ENV as _ENGINE_ENV,
    ENGINE_RUST as _ENGINE_RUST,
    ENGINES as _ENGINES,
    MAPPED_ROUTE_SUBCOMMAND as _ROUTE_SUBCOMMAND,
)
from gpuwm.mapped_source import (
    _load_json_bytes,
    _load_json_document,
    _mapped_engine_choice,
    _require_authority_snapshot,
    _sha256,
    _snapshot_authority,
    load_mapping,
    read_input_list,
)
from gpuwm.native_wrf_contract import (
    canonical_noah_surface,
    load_native_static_cache,
    native_static_export_fields,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    verify_native_static_receipt,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from gpuwm.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
)
from gpuwm.static.build import GeogSelection, build_static
from gpuwm.vertical_contract import validate_explicit_eta_grid
from gpuwm.wrf_direct import export_prepared_wrf


PROOF_SCHEMA = "gpuwm-mapped-direct-wrf-proof-v1"
HIERARCHY_PROOF_SCHEMA = "gpuwm-mapped-native-hierarchy-proof-v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


#: The mapped/rw-wps route's name in every water-temperature refusal
#: and receipt.  One name for every composition that reaches this
#: adapter, 20CRv3 included.
_WATER_ROUTE = "the mapped rw-wps route"


def _forcing_series(bundle):
    """The bundle's snapshots in valid-time order.

    A composed bundle answers with a sequence that packs ONE valid time
    when that time is asked for and drops it when the next is, which is
    what keeps a seven-time preparation from holding seven valid times
    (the measured 15.9 GiB per forcing time on a 3 km CONUS source).  A
    caller that hands over a plain tuple -- an adapter fixture, a route
    that already has its snapshots -- is sorted the way it always was.
    """

    snapshots = bundle.regular_snapshots()
    ordered = getattr(snapshots, "sorted_by_valid_time", None)
    if ordered is not None:
        return ordered()
    return tuple(sorted(
        snapshots, key=lambda snapshot: snapshot.valid_time))


def _forcing_valid_times(snapshots) -> tuple:
    """Every forcing valid time, without packing a snapshot to read one."""

    declared = getattr(snapshots, "valid_times", None)
    if declared is not None:
        return tuple(declared)
    return tuple(snapshot.valid_time for snapshot in snapshots)


def _source_top_pressure_pa(snapshots) -> float:
    """The highest source level the whole forcing series reaches, in Pa.

    Read from the pressure field alone where the series can do that:
    packing a whole valid time to answer one number would read every
    array a frame carries, for every forcing time, before the first one
    is interpolated.
    """

    levels = getattr(snapshots, "source_pressure_hpa", None)
    if levels is None:
        return max(
            float(np.min(snapshot.levels_hpa) * 100.0)
            for snapshot in snapshots)
    return max(
        float(np.min(levels(index)) * 100.0)
        for index in range(len(snapshots)))


def _file_receipt(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _copy_bound_authority(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    """Copy one small evidence authority and prove the copied bytes."""

    shutil.copy2(source, destination)
    if _sha256(destination) != expected_sha256:
        raise ValueError(
            f"mapped evidence changed before publication: {source}"
        )


def _provenance_evidence_name(role: str, suffix: str) -> str:
    """Return a short deterministic filename for one provenance role.

    Composition role names remain recorded in the bound composition receipt;
    repeating an arbitrary role verbatim in the filesystem only burns scarce
    Windows path budget.  Sixteen SHA-256 hex digits keep names stable and
    collision-resistant, while the create-only destination check below still
    fails closed if two roles ever collide.
    """

    role_digest = hashlib.sha256(role.encode("utf-8")).hexdigest()[:16]
    return f"provenance-{role_digest}{suffix}"


def _bound_decoder_receipts(
    paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Recheck small decoder executables before emitting proof evidence."""

    if set(paths) != set(expected_sha256):
        raise ValueError("decoded decoder inventory is internally inconsistent")
    receipts = {}
    for role, path in paths.items():
        receipt = _file_receipt(path)
        if receipt["sha256"] != expected_sha256[role]:
            raise ValueError(f"mapped decoder changed after decode: {role}")
        receipts[role] = receipt
    return receipts


def _validate_target_contract(
    mapping: Mapping[str, object],
    exp,
    boundary_interval_seconds: int | None,
    *,
    hierarchy: bool,
    experiment_config: Path | None = None,
) -> dict[str, object]:
    """Bind mapped target limits to one resolved experiment and cadence.

    The vertical dimension is the EXPERIMENT'S, not the mapping's: the
    whole route interpolates onto the experiment's explicit eta ladder
    (``make_vertical_coord``/``validate_explicit_eta_grid`` read only
    ``exp``), so a config that carries a valid ladder is adopted at its
    own level count and the mapping's ``target_vertical_levels`` is
    recorded as the reference it is.  Refusing a valid ladder over the
    count named no breakage, and paired with the later shape-(0,) error
    it formed the circle of UX finding N6: 44 levels refused for 49,
    then 49 refused for an explicit ladder nothing the user ran had
    ever written.  What IS refused, in one message with both counts and
    both reconciling doors, is a config with no ladder at all -- a
    count alone does not define the interpolation target and WRF's
    automatic level generator is not implemented.
    """

    target = mapping["target"]
    if not isinstance(target, dict):
        raise TypeError("mapping target must be an object")
    domain_count = len(exp.domains)
    max_dom = target["max_dom"]
    if domain_count > max_dom:
        raise ValueError(
            f"mapped target allows max_dom={max_dom}, experiment requests "
            f"{domain_count} domains"
        )
    target_vertical_levels = target["target_vertical_levels"]
    nz = int(exp.root.run.nz)
    eta_levels = tuple(getattr(exp.vertical, "eta_levels", ()) or ())
    config_name = (
        "the experiment config" if experiment_config is None
        else str(experiment_config))
    if not eta_levels:
        counts = (
            f"declares nz={nz} mass levels (WRF e_vert={nz + 1}) and no "
            "explicit eta_levels ladder")
        against = (
            f"matching this mapping's reference count"
            if nz == target_vertical_levels else
            f"and this mapping's reference target is "
            f"{target_vertical_levels} levels "
            f"(WRF e_vert={target_vertical_levels + 1})")
        raise VerticalLadderRefusal(
            f"mapped vertical ladder is missing: {config_name} {counts}, "
            f"{against}.  The mapped route interpolates every forcing "
            "time onto an explicit full-level eta ladder; a level count "
            "alone does not define one, and WRF's automatic level "
            "generator (real.exe) is not implemented",
            remedy=(
                f"remedy: two doors reconcile this.  Keep your {nz} "
                f"levels: add an explicit eta_levels ladder of {nz + 1} "
                "interfaces -- `eta_levels = [1.0, ..., 0.0]`, strictly "
                f"decreasing -- to the [shared] block of {config_name}; "
                "prep adopts your ladder at your level count.  Or use "
                "the packaged reference ladder: `gpuwm domain` authors "
                "a config whose [shared] block carries the certified "
                f"{target_vertical_levels}-level ladder; copy its "
                "nz/p_top/eta_levels lines into your imported config."))
    try:
        validate_explicit_eta_grid(
            eta_levels, nz=nz, p_top=exp.vertical.p_top,
            context="mapped experiment vertical ladder",
        )
    except ValueError as error:
        raise VerticalLadderRefusal(
            str(error),
            remedy=(
                f"remedy: fix the [shared] eta_levels ladder in "
                f"{config_name}: nz + 1 entries (WRF e_vert), running "
                "1.0 (surface) to 0.0 (top), strictly decreasing, with "
                "p_top in pascals inside the source atmosphere."),
        ) from error
    if target["require_lateral_boundaries"] is not True:
        raise ValueError(
            "mapped direct export requires a target contract with lateral "
            "boundaries"
        )
    if (isinstance(boundary_interval_seconds, bool)
            or not isinstance(boundary_interval_seconds, int)
            or boundary_interval_seconds <= 0):
        raise ValueError("mapped boundary interval must be a positive integer")
    declared_interval = target["boundary_interval_seconds"]
    if boundary_interval_seconds != declared_interval:
        raise ValueError(
            f"mapped boundary interval {boundary_interval_seconds} differs "
            f"from target contract {declared_interval}"
        )
    validate_boundary_timing(
        exp, boundary_interval_seconds,
        source=(
            "mapped hierarchy target"
            if hierarchy else "mapped target"))
    return {
        "status": "PASS",
        "domain_count": domain_count,
        "domain_ids": [domain.grid_id for domain in exp.domains],
        "domain_start_times": {
            f"d{domain.grid_id:02d}": (
                exp.domain_start_time(domain.grid_id).isoformat()
                if hasattr(exp, "domain_start_time")
                else (
                    exp.start_time
                    if getattr(domain, "start_time", None) is None
                    else domain.start_time
                ).isoformat())
            for domain in exp.domains
        },
        "mapping_max_dom": max_dom,
        # The ADOPTED count -- the experiment's, which is what every
        # array downstream is allocated at; the mapping's declared
        # count is recorded beside it as the reference it is.
        "target_vertical_levels": nz,
        "mapping_reference_vertical_levels": target_vertical_levels,
        "vertical_levels_adopted_from": (
            "mapping-reference-count" if nz == target_vertical_levels
            else "experiment-eta-ladder"),
        "require_lateral_boundaries": True,
        "boundary_interval_seconds": boundary_interval_seconds,
        "hierarchy": hierarchy,
    }


#: Which subprocess tools each mapped format's Python engine launches.
#: The same table ``mapped_composition._decoder_inventory`` enforces;
#: named here only to say what a MISSING role means to a reader.
_FORMAT_TOOL_ROLES = {
    "grib1": ("grib1_bridge",),
    "grib2": ("grib2_inventory", "grib2_dump"),
    "netcdf": (),
}


#: The tool-role flag each role is spelled with on the command line.
_TOOL_ROLE_FLAGS = {
    "grib1_bridge": "--grib1-bridge",
    "grib2_inventory": "--grib2-inventory",
    "grib2_dump": "--grib2-dump",
}


def _decoder_inventory_refusal(message, formats, *, missing=(), extra=()):
    """The decoder-contract fault, as a refusal this door can deliver.

    Named breakage: a bare ``gpuwm prep`` of any composed GRIB2 source
    reached ``ValueError: grib2 decoder inventory differs from the
    contract; missing=['grib2_dump', 'grib2_inventory']`` as an
    unhandled traceback, so a reader met this package's line numbers
    before meeting a remedy.  The contract is unchanged and still
    refuses; this gives the same sentence a delivery.

    The remedy is composed for THIS install, and the two directions get
    opposite ones because they are opposite faults with the same
    message shape: a MISSING tool gets the estate answer ``gpuwm
    doctor`` would print (staged bundle, or a clone and a build), and a
    tool supplied to a route that decodes in process gets told to drop
    the pin -- there is nothing here that would ever launch it.
    """

    from gpuwm import bridges
    from gpuwm.ingest.source_coverage import DecoderInventoryRefusal
    from gpuwm.mapped_engine_bridge import ENGINE_ENV, ENGINE_PYTHON

    text = str(message)
    missing = tuple(sorted(missing))
    extra = tuple(sorted(extra))
    label = "+".join(sorted({str(name) for name in formats}))
    if missing:
        remedy = (
            f"remedy: this route composes {label} on the Python engine, "
            f"which reads it through {', '.join(missing)}.\n"
            + bridges.bridge_remedy(missing[0])
            + "\n  # `gpuwm prep` resolves these through the same ladder "
            "with no flags once they are staged; "
            + " / ".join(_TOOL_ROLE_FLAGS[role] for role in missing)
            + " override it.")
    elif extra:
        remedy = (
            "remedy: drop "
            + " / ".join(_TOOL_ROLE_FLAGS[role] for role in extra)
            + f" -- this route reads {label} with no such tool and has "
            "nothing to launch it for.  To decode with those exact "
            "executables instead, ask for the engine that runs them: "
            f"{ENGINE_ENV}={ENGINE_PYTHON} (equivalently --mapped-engine "
            f"{ENGINE_PYTHON}).")
    else:
        remedy = DecoderInventoryRefusal.remedy
    return DecoderInventoryRefusal(text, remedy=remedy)


def prepare_mapped_wrf(
    *,
    composition: str | Path,
    mapping: str | Path,
    primary_files: Sequence[str | Path],
    supplement_files: Mapping[
        str, str | Path | Sequence[str | Path]
    ],
    provenance_files: Mapping[str, str | Path],
    input_manifest: str | Path,
    input_manifest_sha256: str,
    contributing_mappings: Mapping[str, str | Path] | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
    wps_namelist: str | Path,
    geog_root: str | Path,
    static_input: str | Path | None = None,
    static_receipt: str | Path | None = None,
    experiment_config: str | Path,
    output_root: str | Path,
    preprocess_backend: str = "cuda",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: str | Path | None = None,
    hierarchy_workers: int | None = None,
    _source_manifest: str | Path | None = None,
    _source_manifest_sha256: str | None = None,
    _source_adapter: str = "rw-wps-mapped-composition-v2",
) -> dict[str, object]:
    """Build native WRF inputs from one complete mapped-source composition.

    All mixed-product semantics are resolved before target interpolation.
    The root owns the complete external-LBC sequence; mapped child inputs are
    initialized independently and finalized parent-before-child.  Products
    are published atomically only after source and run-control authorities are
    reverified.

    ``static_input``/``static_receipt`` supply a previously published
    native-static NPZ instead of rebuilding the root domain's geography from
    WPS_GEOG, exactly as the ERA5/GFS/HRRR adapters already accept.  The
    receipt binds the target geometry contract and the NPZ SHA-256, and the
    loader re-derives MAPFAC/F/E/SINALPHA/COSALPHA from the grid, so a cache
    that does not belong to this domain is refused rather than trusted.
    ``geog_root`` remains required either way: the proof's geog_datasets
    binding and any child-domain statics are still resolved from the tree.

    ``_source_manifest``/``_source_manifest_sha256`` are the seam for a
    NAMED-SOURCE route whose user-facing sealed authority is its own
    manifest schema (the 20CRv3 member manifest, with its filename-bound
    member identity): the route verifies its own document, authors the
    generic composition-inputs manifest FROM it, and hands both here.
    The composition decodes against the bridged manifest -- which the
    composition receipt seals -- while the prepared tree's evidence copy,
    identity chain and proof stay bound to the route's own document, the
    one its forecast leg re-verifies with the route's own reader.  Absent,
    the two manifests are the same document and nothing changes.
    """

    started = time.perf_counter()
    composition = Path(composition).resolve()
    mapping = Path(mapping).resolve()
    primary = tuple(Path(path).resolve() for path in primary_files)
    supplements = {
        str(role): _path_inventory(path, f"supplement {role!r}")
        for role, path in supplement_files.items()
    }
    provenance = {
        str(role): Path(path).resolve() for role, path in provenance_files.items()
    }
    contributing = {
        str(role): Path(path).resolve()
        for role, path in (contributing_mappings or {}).items()
    }
    input_manifest = Path(input_manifest).resolve()
    if (_source_manifest is None) != (_source_manifest_sha256 is None):
        raise ValueError(
            "source manifest and source manifest SHA are an atomic pair"
        )
    if _source_manifest is None:
        source_manifest_path: Path | None = None
        source_manifest_sha256: str | None = None
    else:
        source_manifest_path = Path(_source_manifest).resolve()
        source_manifest_sha256 = str(_source_manifest_sha256).lower()
        if not source_manifest_path.is_file():
            raise RunInputRefusal(
                f"mapped run input is missing: --source-manifest names "
                f"{source_manifest_path}, which does not exist")
        if _sha256(source_manifest_path) != source_manifest_sha256:
            raise ValueError(
                "source manifest bytes differ from the declared SHA-256, so "
                "the identity chain would be sealed to a document nobody "
                "verified"
            )
    mapping_snapshot = _snapshot_authority(mapping, retain_bytes=True)
    mapping_contract = load_mapping(
        mapping,
        _raw=_load_json_bytes(mapping_snapshot.data, "mapping", mapping),
    )
    # The decoder role set is the union across the primary's format and
    # every contributing source's format.  Only the format key is probed
    # here; ``decode_composed_source`` fully validates each contributing
    # mapping against the composition's pinned hash before decoding.
    decode_formats = {str(mapping_contract["format"])}
    for role, contributing_path in contributing.items():
        raw = _load_json_document(
            contributing_path, f"contributing mapping {role!r}",
        )
        observed_format = raw.get("format") if isinstance(raw, dict) else None
        if observed_format in ("grib1", "grib2", "netcdf"):
            decode_formats.add(str(observed_format))
    engine_binary = None
    # This route ALWAYS composes, and every contributing format has to be
    # one the engine can read: a union with one unported format in it is
    # a Python-engine job whole, because half a composition decoded by
    # each engine would be one frameset with two provenances.
    if all(
        _mapped_engine_choice(
            grib1_bridge=grib1_bridge,
            grib2_inventory=grib2_inventory,
            grib2_dump=grib2_dump,
            subcommand=_ROUTE_SUBCOMMAND,
            source_format=source_format,
        ) == _ENGINE_RUST
        for source_format in sorted(decode_formats)
    ):
        # Resolved HERE, before any output directory exists, so an
        # unstaged engine costs a refusal at the door instead of a
        # half-built preparation: this is the same pre-flight the
        # subprocess decoders get below.
        from gpuwm.mapped_engine_bridge import require_engine

        engine_binary = require_engine()
    try:
        decoders = _decoder_inventory(
            sorted(decode_formats),
            grib1_bridge=grib1_bridge,
            grib2_inventory=grib2_inventory,
            grib2_dump=grib2_dump,
            engine=engine_binary,
        )
    except ValueError as error:
        # Delivered, not demoted: the contract still refuses and still
        # names the same breakage.  What changes is that a user reading
        # it meets a remedy instead of nineteen frames of this package's
        # internals, which is what they met on a bare default prep of
        # every composed source until 2.5.0.
        present = {
            role for role, path in (
                ("grib1_bridge", grib1_bridge),
                ("grib2_inventory", grib2_inventory),
                ("grib2_dump", grib2_dump),
            ) if path is not None
        }
        required = set() if engine_binary is not None else {
            role for name in decode_formats
            for role in _FORMAT_TOOL_ROLES[str(name)]
        }
        raise _decoder_inventory_refusal(
            str(error), decode_formats,
            missing=required - present, extra=present - required,
        ) from error
    wps_namelist = Path(wps_namelist).resolve()
    geog_root = Path(geog_root).resolve()
    experiment_config = Path(experiment_config).resolve()
    output_root = Path(output_root).resolve()
    cpu_bridge = (
        None if cpu_preprocess_bridge is None
        else Path(cpu_preprocess_bridge).resolve()
    )
    if (static_input is None) != (static_receipt is None):
        raise ValueError(
            "mapped static-input and static-receipt must be supplied together"
        )
    prebuilt_static = (
        None if static_input is None else Path(static_input).resolve())
    prebuilt_receipt = (
        None if static_receipt is None else Path(static_receipt).resolve())
    # Every file the caller named, checked under the FLAG that named
    # it and reported together: the old anonymous loop relayed the
    # first miss as a bare ``FileNotFoundError`` holding only a path,
    # which is how a pasted prep command with one wrong working
    # directory answered a user with a traceback (UX finding N6).
    labeled = [
        ("--composition", composition),
        ("--mapping", mapping),
        ("--input-manifest", input_manifest),
        ("--wps-namelist", wps_namelist),
        ("--experiment-config", experiment_config),
        *((f"decoder tool {role}", path) for role, path in decoders.items()),
        *(("--input/--input-list", path) for path in primary),
        *((f"--supplement {role}", path)
          for role, paths in supplements.items() for path in paths),
        *((f"--provenance {role}", path)
          for role, path in provenance.items()),
        *((f"--contributing-mapping {role}", path)
          for role, path in contributing.items()),
        *(() if prebuilt_static is None
          else (("--static-input", prebuilt_static),
                ("--static-receipt", prebuilt_receipt))),
        *(() if cpu_bridge is None
          else (("--cpu-preprocess-bridge", cpu_bridge),)),
    ]
    missing = [
        (flag, path) for flag, path in labeled if not path.is_file()
    ]
    if missing:
        listing = "; ".join(
            f"{flag} names {path}, which does not exist"
            for flag, path in missing)
        remedy = RunInputRefusal.remedy
        if any(flag == "--experiment-config" for flag, _path in missing):
            remedy += (
                "  For --experiment-config: `gpuwm import-namelist WPS "
                "INPUT --output FILE.toml` translates an existing WRF "
                "namelist pair into one, and `gpuwm domain` authors one "
                "from scratch.")
        raise RunInputRefusal(
            f"mapped run inputs are missing: {listing}", remedy=remedy)
    if not geog_root.is_dir():
        raise RunInputRefusal(
            f"--geog-root names {geog_root}, which is not a directory",
            remedy=(
                "remedy: point --geog-root at a staged WPS_GEOG tree.  "
                "`gpuwm fetch-geog` stages one (~16 GB) and prints the "
                "root to pass."))
    if output_root.exists():
        raise PreparationRefusal(
            f"refusing to overwrite mapped output {output_root}: a "
            "prepared tree is published atomically, and a folder that "
            "already exists may be a finished run something else reads",
            remedy=(
                "remedy: pass a fresh --output-root, or move or remove "
                "the old folder yourself first; prep never deletes an "
                "output tree."))
    run_control_before = {
        "wps_namelist": _file_receipt(wps_namelist),
        "experiment_config": _file_receipt(experiment_config),
    }
    if cpu_bridge is not None:
        run_control_before["cpu_preprocess_bridge"] = _file_receipt(cpu_bridge)

    exp = load_experiment(experiment_config)
    from gpuwm.experiment import (
        refuse_unrouted_perturbation, refuse_unrouted_spawn,
    )
    refuse_unrouted_perturbation(exp, "mapped-adapter prepared-cache")
    refuse_unrouted_spawn(exp, "mapped-adapter prepared-cache")
    from gpuwm.static.highres_production import refuse_inert_highres
    refuse_inert_highres(experiment_config, lane="mapped adapter")
    hierarchy = len(exp.domains) > 1
    target_contract = _validate_target_contract(
        mapping_contract,
        exp,
        mapping_contract["target"].get("boundary_interval_seconds"),
        hierarchy=hierarchy,
        experiment_config=experiment_config,
    )
    cfg = exp.root.run
    if hierarchy:
        grids = validate_native_lambert_contracts(
            exp, wps_namelist, source_name="mapped source",
        )
        grid = grids[0]
    else:
        grid = validate_native_lambert_contract(
            exp, wps_namelist, source_name="mapped source",
        )
        grids = (grid,)
    selection = GeogSelection.from_case_data(
        SimpleNamespace(wps_namelist=wps_namelist, geog_root=geog_root), 1,
    )

    decode_started = time.perf_counter()
    bundle = decode_composed_source(
        composition, mapping, primary, supplements, provenance,
        input_manifest=input_manifest,
        input_manifest_sha256=input_manifest_sha256,
        contributing_mappings=contributing,
        grib1_bridge=decoders.get("grib1_bridge"),
        grib2_inventory=decoders.get("grib2_inventory"),
        grib2_dump=decoders.get("grib2_dump"),
        # The engine's compose scratch holds the whole composed frame
        # stream; naming this run's destination keeps that stream on the
        # output's disk-backed filesystem instead of a tmpfs system temp
        # with a quota smaller than the biggest registered sources.
        scratch_destination=output_root,
    )
    decode_seconds = time.perf_counter() - decode_started
    if bundle.mapping_sha256 != mapping_snapshot.sha256:
        raise ValueError(
            "mapped contract changed between target validation and decode"
        )
    _require_authority_snapshot(mapping_snapshot)
    if dict(bundle.decoder_paths) != decoders:
        raise ValueError("decoded decoder paths differ from requested decoders")
    composition_receipt = mapped_composition_receipt(bundle)
    snapshots = _forcing_series(bundle)
    times = _forcing_valid_times(snapshots)
    if not times or times[0] != exp.start_time:
        raise ValueError(
            f"mapped forcing must begin at {exp.start_time}, got {times[:1]}"
        )
    forcing_offsets = tuple((value - times[0]).total_seconds() for value in times)
    if any(not value.is_integer() for value in forcing_offsets):
        raise ValueError("mapped forcing times must use whole-second offsets")
    forcing_seconds = tuple(int(value) for value in forcing_offsets)
    if any(value < 0 for value in forcing_seconds) \
            or forcing_seconds[-1] < exp.run_seconds:
        raise ValueError("mapped forcing does not cover the configured run")
    deltas = {
        later - earlier
        for earlier, later in zip(forcing_seconds, forcing_seconds[1:])
    }
    if len(deltas) != 1 or next(iter(deltas), 0) <= 0:
        raise ValueError("mapped forcing cadence must be positive and uniform")
    boundary_interval_seconds = deltas.pop()
    target_contract = _validate_target_contract(
        mapping_contract,
        exp,
        boundary_interval_seconds,
        hierarchy=hierarchy,
        experiment_config=experiment_config,
    )
    source_top_pressure_pa = _source_top_pressure_pa(snapshots)
    if hierarchy:
        grids = validate_native_lambert_contracts(
            exp,
            wps_namelist,
            source_name="mapped source",
            source_top_pressure_pa=source_top_pressure_pa,
        )
        grid = grids[0]
    else:
        try:
            validate_explicit_eta_grid(
                exp.vertical.eta_levels,
                nz=cfg.nz,
                p_top=exp.vertical.p_top,
                source_top_pressure_pa=source_top_pressure_pa,
                context="mapped direct adapter",
            )
        except ValueError as error:
            # The ladder's shape and values were validated at the door;
            # what this re-check adds is the SOURCE TOP, so the only
            # user-reachable raise left here is a model top the staged
            # atmosphere does not reach.
            raise VerticalLadderRefusal(
                str(error),
                remedy=(
                    "remedy: lower the model top -- raise p_top in the "
                    f"experiment config {experiment_config} to a "
                    "pressure the staged source atmosphere reaches -- "
                    "or prepare from a source whose levels reach this "
                    "top (`gpuwm prep --show-source NAME` prints each "
                    "source's vertical coverage)."),
            ) from error

    static_started = time.perf_counter()
    if prebuilt_static is None:
        static = build_static(grid, geog_root, selection=selection)
        root_static_provider = "native-wps-geog"
        root_static_receipt = None
    else:
        # The receipt binds native_geometry_contract(grid, cfg) AND the NPZ
        # SHA-256; the loader then re-derives the geometry fields from the
        # grid and refuses a stored copy that disagrees.  A cache built for
        # another domain cannot survive either check.
        root_static_receipt = verify_native_static_receipt(
            prebuilt_receipt, prebuilt_static, grid, cfg)
        static = load_native_static_cache(
            prebuilt_static, grid, cfg.ny, cfg.nx)
        root_static_provider = "prebuilt-hash-bound-cache"
    static_seconds = time.perf_counter() - static_started

    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_bridge,
    )
    preprocess_receipt = preprocess.receipt()
    # The surface the water-temperature assembly decides on.  This route
    # carries no [case_data], so the policy is the default one silence
    # selects; the mapped compositions declare a skin temperature and no
    # SST, which is the case where the class-coherent assembly and the
    # historical selector agree cell for cell, so no policy is reachable
    # that would change a number here.  What the statics DO change is the
    # guarantee: with ISLAKE named, a lake and the ocean cannot share a
    # provider on a target where a coarse coastline joins them.
    water_statics = WaterTemperatureStatics.for_route(
        route=_WATER_ROUTE, policy=None,
        landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
        landuse_attrs=selection.landuse_global_attrs())
    initialize_started = time.perf_counter()
    # ONE forcing time is ever resident.  The start time is built LAST
    # (start_last_forcing_order) and is the only met/state this loop
    # retains; every other time contributes its perimeter frames against
    # its own position and is released before the next one is
    # interpolated.  Walking the times in order instead meant holding the
    # start time -- which nothing reads until the boundaries are complete
    # -- while each later time was built underneath it.  At 800x800x49
    # with mp=10 and three GFS times that second resident time is 14.67
    # GiB of device residency against 7.66, a priced peak envelope of
    # 23.92 GiB against 15.86: the difference between preparing the
    # domain on a 16 GiB card and OOMing after the whole forcing chain
    # had already been fetched.
    initial_result = None
    initial_met = None
    forcing = StateBoundaryFrames(
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    coord = make_vertical_coord(
        cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
        eta_levels=exp.vertical.eta_levels,
    )
    mapfac_m, mapfac_u, mapfac_v = (
        grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(),
    )
    coriolis_f, coriolis_e = grid.coriolis_m()
    rotation_sin, rotation_cos = grid.rotation_m()
    for index in start_last_forcing_order(len(snapshots)):
        source = snapshots[index]
        # Metgrid classifies masked-field TARGET cells by the model
        # (geogrid) landmask; the mapped lane declares it like the ERA5
        # lanes so soil, skin, snow, and physics share one surface.
        met = interpolate_era5_to_lambert(
            source, grid, backend=preprocess,
            target_landmask=np.asarray(static["LANDMASK"]) >= 0.5,
            water_temperature_statics=water_statics)
        initialized = initialize_real(
            met, cfg, coord, static["HGT_M"],
            p_top=exp.vertical.p_top, sfcp_to_sfcp=True,
            preprocess_backend=preprocess,
            state_backend="preprocess",
        )
        initialized.state.set_map_coriolis(
            mapfac_m, mapfac_u, mapfac_v, coriolis_f, coriolis_e,
            sina=rotation_sin, cosa=rotation_cos,
        )
        forcing.add_state(initialized.state, index=index)
        if index == 0:
            initial_met = met
            initial_result = initialized
        else:
            del met, initialized
            release_backend_memory(preprocess)
    boundaries = forcing.build(times)
    attach_lateral_boundaries(initial_result.state, boundaries)
    # No lake skin override: the masked=both SKINTEMP chain with
    # static-landmask targets already yields water-source skin at lakes,
    # matching real.exe's no-TAVGSFC behavior.  The static landmask drives
    # WRF's process_soil_real land/water branches, and terrain plus the
    # composition's canonical source orography enable adjust_soil_temp_new's
    # elevation lapse on skin and soil temperature inputs.  The router
    # forwards this exact argument list to preprocess_noah_soil for
    # Noah-geometry schemes.
    soil = preprocess_land_surface_soil(
        initial_met.fields,
        sf_surface_physics=int(cfg.sf_surface_physics),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["TMN"],
        soil_layer_contract=bundle.soil_layer_contract,
        landmask=static["LANDMASK"],
        terrain=static["HGT_M"],
        source_orography=initial_met.fields["SOURCE_OROGRAPHY"],
        water_temperature=getattr(initial_met, "water_temperature", None),
        water_temperature_policy=water_statics.policy,
        # Same seam as the elevation lapse above, for moisture and for the
        # deep temperature: a mapped source's mesh can be coarser than this
        # grid, and where it is, the soil state gets the target grid's own
        # soil texture instead of the source cell's average.
        soil_mesh=soil_mesh_plan_from_case(
            snapshots[0], grid, experiment_config),
        route=_WATER_ROUTE,
    )
    initialize_seconds = time.perf_counter() - initialize_started

    # Keep atomic siblings short.  Repeating a user-supplied output name at
    # every nested transaction exceeded legacy Windows MAX_PATH before the
    # first hierarchy artifact could be written.
    staging = output_root.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    if staging.exists():
        raise FileExistsError(f"mapped staging output already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        static_path = staging / "native-static.npz"
        geometry_path = staging / "geometry-receipt.json"
        prepared_path = staging / "prepared-cache"
        wrf_path = staging / "wrf-native-input"
        evidence_path = staging / "source-evidence"
        evidence_path.mkdir()
        # The USER-FACING manifest: the route's own sealed document when
        # the route bridged one in, otherwise the composition manifest
        # itself.  The evidence copy, the identity chain and the proof
        # bind this one -- it is what the forecast leg re-verifies with
        # the route's own reader -- while the bridged twin stays sealed
        # inside the composition receipt.
        published_manifest = (
            input_manifest if source_manifest_path is None
            else source_manifest_path)
        published_manifest_sha256 = (
            bundle.input_manifest_sha256 if source_manifest_sha256 is None
            else source_manifest_sha256)
        for source, name, digest in (
            (mapping, "mapping.json", bundle.mapping_sha256),
            (
                composition,
                "composition.json",
                bundle.composition_sha256,
            ),
            (
                published_manifest,
                "input-manifest.json",
                published_manifest_sha256,
            ),
            # The bridged composition-inputs manifest travels TOO on a
            # named-source route: the composition receipt seals its
            # digest, and the forecast leg re-hashes this copy against
            # that seal -- without the bytes, the receipt's manifest
            # record would be a digest nothing can re-check once the raw
            # inputs move.
            *(
                ()
                if source_manifest_path is None
                else ((
                    input_manifest,
                    "composition-inputs.json",
                    bundle.input_manifest_sha256,
                ),)
            ),
        ):
            _copy_bound_authority(source, evidence_path / name, digest)
        for role, source in sorted(provenance.items()):
            destination = evidence_path / _provenance_evidence_name(
                role, source.suffix,
            )
            if destination.exists():
                raise ValueError(
                    f"provenance roles collide after filename encoding: {role!r}"
                )
            if source != bundle.terrain_provenance_path:
                raise ValueError(
                    f"decoded provenance path differs for role {role!r}"
                )
            _copy_bound_authority(
                source,
                destination,
                bundle.terrain_provenance_sha256,
            )
        static_output_receipt = write_native_static_cache(
            static_path, native_static_export_fields(static, grid),
        )
        geometry_receipt = write_native_geometry_receipt(
            geometry_path, grid, cfg, static_path,
        )
        source_identity = {
            "adapter": _source_adapter,
            "mapping_sha256": bundle.mapping_sha256,
            "composition_sha256": bundle.composition_sha256,
            "input_manifest_sha256": published_manifest_sha256,
            "composition_receipt_sha256": composition_receipt[
                "receipt_content_sha256"
            ],
            "preprocessing": preprocess_receipt,
        }
        forcing_identity = (
            {"forcing_hours": tuple(
                value // 3600 for value in forcing_seconds)}
            if all(value % 3600 == 0 for value in forcing_seconds)
            else {"forcing_offsets_seconds": forcing_seconds})
        forcing_key, forcing_axis = next(iter(forcing_identity.items()))
        if hierarchy:
            selected_workers = hierarchy_workers
            if selected_workers is None:
                selected_workers = (
                    8 if preprocess_receipt["backend"] == "cpu" else 1
                )
            hierarchy_started = time.perf_counter()
            hierarchy_result = initialize_and_export_regular_source_hierarchy(
                exp=exp,
                grids=grids,
                snapshots=snapshots,
                # A child sees the catalog, not the config.
                soil_texture_downscale=declared_soil_texture_downscale(
                    experiment_config),
                **forcing_identity,
                wps_namelist=wps_namelist,
                geog_root=geog_root,
                source_name="RW-WPS-MAPPED",
                artifact_output=staging / "hierarchy-artifacts",
                wrf_output=wrf_path,
                root_initial_result=initial_result,
                root_met=initial_met,
                root_soil=soil,
                root_static_fields=static,
                root_boundaries=boundaries,
                bridge_manifest_sha256=published_manifest_sha256,
                source_manifest_sha256=published_manifest_sha256,
                namelist_sha256=_sha256(experiment_config),
                source_identity={
                    **source_identity,
                    "target_contract": target_contract,
                },
                source_inventory=tuple(snapshots[0].fields),
                workers=selected_workers,
                preprocess_backend=preprocess_receipt["backend"],
                cpu_bridge=cpu_bridge,
                soil_layer_contract=bundle.soil_layer_contract,
                root_metadata={
                    "composition_receipt_sha256": composition_receipt[
                        "receipt_content_sha256"
                    ],
                    "mapped_target_contract": target_contract,
                },
                input_provenance={
                    "mapping_sha256": bundle.mapping_sha256,
                    "composition_sha256": bundle.composition_sha256,
                    "input_manifest_sha256": published_manifest_sha256,
                    "decoder_sha256": dict(bundle.decoder_sha256),
                    "preprocessing": preprocess_receipt,
                    "mapped_target_contract": target_contract,
                },
                artifact_manifest_reference=(
                    "../hierarchy-artifacts/domain-artifacts.json"
                ),
            )
            hierarchy_seconds = time.perf_counter() - hierarchy_started
            run_control_after = {
                "wps_namelist": _file_receipt(wps_namelist),
                "experiment_config": _file_receipt(experiment_config),
            }
            if cpu_bridge is not None:
                run_control_after["cpu_preprocess_bridge"] = _file_receipt(
                    cpu_bridge
                )
            if run_control_after != run_control_before:
                raise ValueError(
                    "mapped run-control bytes changed during preparation"
                )
            proof = {
                "schema": HIERARCHY_PROOF_SCHEMA,
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                "domain_count": len(exp.domains),
                "forcing_times": [value.isoformat() for value in times],
                # The soil-state SOURCE resolution and whether the
                # sub-source-cell reconstitution ran on it.
                "soil_texture_downscale": dict(
                    getattr(soil, "soil_texture_downscale", {}) or {}),
                forcing_key: list(forcing_axis),
                "boundary_interval_seconds": boundary_interval_seconds,
                "target_contract": target_contract,
                "execution_inputs": {
                    "decoders": _bound_decoder_receipts(
                        bundle.decoder_paths,
                        bundle.decoder_sha256,
                    ),
                    **run_control_before,
                    "geog_root": str(geog_root),
                    "geog_source_binding": (
                        "per_domain_resolved_dataset_paths_plus_native_"
                        "static_output_sha256"
                    ),
                    "root_static_provider": root_static_provider,
                    "root_static_receipt": root_static_receipt,
                },
                "source_composition": composition_receipt,
                "preprocessing": preprocess_receipt,
                "hierarchy_workers": selected_workers,
                "root_static": static_output_receipt,
                "root_geometry": geometry_receipt,
                "static_catalog": dict(
                    hierarchy_result.static_catalog_receipt
                ),
                "source_coverage": dict(
                    hierarchy_result.source_coverage_receipt
                ),
                "artifact_receipt": dict(
                    hierarchy_result.hierarchy.artifacts.receipt
                ),
                "wrf_manifest": dict(
                    hierarchy_result.hierarchy.wrf_manifest
                ),
                "timing_seconds": {
                    "static_root": static_seconds,
                    "decode_and_compose": decode_seconds,
                    "initialize_all_root_times": initialize_seconds,
                    **dict(hierarchy_result.hierarchy.timings_seconds),
                    "hierarchy_call_wall": hierarchy_seconds,
                    "total": time.perf_counter() - started,
                },
            }
            proof["proof_content_sha256"] = hashlib.sha256(
                _canonical(proof).encode("utf-8")
            ).hexdigest()
            (staging / "proof.json").write_text(
                json.dumps(
                    proof, indent=2, sort_keys=True, allow_nan=False
                ) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, output_root)
            # The composed frame stream is spent: every valid time has
            # been interpolated and the tree is published, so the engine
            # scratch it streamed from goes now rather than at collection.
            bundle.close()
            return proof
        identity = prepared_cache_identity(
            bridge_manifest_sha256=published_manifest_sha256,
            source_manifest_sha256=published_manifest_sha256,
            static_cache_sha256=static_output_receipt["sha256"],
            namelist_sha256=_sha256(experiment_config),
            domain_config=exp.root,
            **forcing_identity,
            source_identity=source_identity,
        )
        cache_started = time.perf_counter()
        cache_receipt = write_prepared_cache(
            prepared_path, identity=identity,
            initial_result=initial_result, met=initial_met,
            boundaries=boundaries, surface=canonical_noah_surface(soil),
            metadata={
                "source_adapter": "mapped",
                "initial_valid_time": times[0].isoformat(),
                "last_valid_time": times[-1].isoformat(),
                forcing_key: list(forcing_axis),
                "boundary_interval_seconds": boundary_interval_seconds,
                "composition_receipt_sha256": composition_receipt[
                    "receipt_content_sha256"
                ],
            },
        )
        cache_receipt = dict(cache_receipt)
        cache_receipt["path"] = "prepared-cache"
        cache_seconds = time.perf_counter() - cache_started
        export_started = time.perf_counter()
        export_receipt = export_prepared_wrf(
            prepared_path, static_path, geometry_path, wrf_path,
            valid_time=times[0],
            boundary_interval_seconds=boundary_interval_seconds,
        )
        export_seconds = time.perf_counter() - export_started
        run_control_after = {
            "wps_namelist": _file_receipt(wps_namelist),
            "experiment_config": _file_receipt(experiment_config),
        }
        if cpu_bridge is not None:
            run_control_after["cpu_preprocess_bridge"] = _file_receipt(cpu_bridge)
        if run_control_after != run_control_before:
            raise ValueError("mapped run-control bytes changed during preparation")
        proof = {
            "schema": PROOF_SCHEMA,
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "forcing_times": [value.isoformat() for value in times],
            # The soil-state SOURCE resolution and whether the
            # sub-source-cell reconstitution ran on it.
            "soil_texture_downscale": dict(
                getattr(soil, "soil_texture_downscale", {}) or {}),
            forcing_key: list(forcing_axis),
            "boundary_interval_seconds": boundary_interval_seconds,
            "execution_inputs": {
                "decoders": _bound_decoder_receipts(
                    bundle.decoder_paths,
                    bundle.decoder_sha256,
                ),
                **run_control_before,
                "geog_root": str(geog_root),
                "geog_source_binding": (
                    "resolved_dataset_paths_plus_native_static_output_sha256"
                ),
                "root_static_provider": root_static_provider,
                "root_static_receipt": root_static_receipt,
                "geog_resolution_tokens": list(selection.resolution_tokens),
                "geog_datasets": {
                    field: str(selection.path(field))
                    for field in (
                        "terrain", "landuse", "soil_top", "soil_bottom",
                        "greenfrac", "lai", "albedo", "snow_albedo",
                        "soil_temperature",
                    )
                },
            },
            "source_composition": composition_receipt,
            "preprocessing": preprocess.receipt(),
            "static": static_output_receipt,
            "geometry": geometry_receipt,
            "prepared_cache": cache_receipt,
            "export": export_receipt,
            "timing_seconds": {
                "static": static_seconds,
                "decode_and_compose": decode_seconds,
                "initialize_all_times": initialize_seconds,
                "write_prepared_cache": cache_seconds,
                "direct_wrf_export": export_seconds,
                "total": time.perf_counter() - started,
            },
        }
        proof["proof_content_sha256"] = hashlib.sha256(
            _canonical(proof).encode("utf-8")
        ).hexdigest()
        (staging / "proof.json").write_text(
            json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        # The composed frame stream is spent: every valid time has been
        # interpolated and the tree is published, so the engine scratch
        # it streamed from goes now rather than at collection.
        bundle.close()
        return proof
    except BaseException:
        bundle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


_ROLE_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _role_bindings(
    values: Sequence[str], *, multiple: bool,
) -> dict[str, Path | tuple[Path, ...]]:
    """Parse repeatable ROLE=PATH bindings without silent replacement."""

    grouped: dict[str, list[Path]] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not _ROLE_PATTERN.fullmatch(role) or not raw_path:
            raise ValueError(
                "role bindings must use non-empty ROLE=PATH with a portable "
                "role name"
            )
        if not multiple and role in grouped:
            raise ValueError(f"duplicate binding for role {role!r}")
        grouped.setdefault(role, []).append(Path(raw_path))
    return {
        role: tuple(paths) if multiple else paths[0]
        for role, paths in grouped.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rw-wps mapped",
        description=(
            "Prepare atomic stock-WRF initial and boundary files from a "
            "strict mapped GRIB/NetCDF composition without WPS or real.exe."
        ),
    )
    parser.add_argument(
        "--source-format", choices=("grib1", "grib2", "netcdf"),
        required=True,
    )
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    # One set of primary files, two transports.  The repeated flag IS
    # the grammar; the list file carries the same ordered paths for the
    # command lines Windows cannot launch (CreateProcess caps the whole
    # line at 32 KB, and a field-per-file source needs hundreds of
    # inputs per state).
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input", dest="primary_files", action="append", type=Path,
    )
    inputs.add_argument(
        "--input-list", dest="input_list", type=Path,
        help="file naming the --input files, one path per line, in the "
             "same deterministic time/file order",
    )
    parser.add_argument(
        "--supplement", action="append", default=[], metavar="ROLE=PATH",
        help="repeat a role for a deterministic multi-file supplement",
    )
    parser.add_argument(
        "--provenance", action="append", default=[], metavar="ROLE=PATH",
    )
    parser.add_argument(
        "--contributing-mapping", action="append", default=[],
        metavar="ROLE=PATH",
        help=(
            "a contributing source's own mapping document for a "
            "cross-source composition, under the mapping_role its "
            "field_sources binding declares; the bytes must hash to the "
            "SHA-256 the composition pins"
        ),
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--grib1-bridge", type=Path)
    parser.add_argument("--grib2-inventory", type=Path)
    parser.add_argument("--grib2-dump", type=Path)
    parser.add_argument(
        "--mapped-engine", choices=_ENGINES, default=None,
        help=(
            "which engine decodes the source bytes; omitted, the default "
            "engine runs. `python` is a WORKAROUND, not a mode: it runs "
            "the slower Python decode path and is here so a decode the "
            "Rust engine gets wrong has a way around it while the defect "
            "is fixed"),
    )
    parser.add_argument("--wps-namelist", type=Path, required=True)
    # --geog-root stays required even with a prebuilt static cache: the proof
    # binds the resolved geog dataset paths, and child domains in a hierarchy
    # still build their own statics from the tree.
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument(
        "--static-input", type=Path,
        help=(
            "previously published native-static.npz to load instead of "
            "rebuilding root geography; requires --static-receipt"),
    )
    parser.add_argument(
        "--static-receipt", type=Path,
        help=(
            "geometry-receipt.json binding --static-input by geometry "
            "contract and SHA-256"),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="auto",
        help="where the source-grid/WRF-real setup transforms run "
             "(default auto: CUDA when the certified runtime is usable, "
             "otherwise the deterministic parallel CPU backend, "
             "announced in one line)",
    )
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    parser.add_argument(
        "--prepared-forecast-source", default=None, metavar="ID",
        help="the source id the finished bundle is run under, so this "
             "door can print the complete hash-bound forecast command "
             "when it is done; `gpuwm prep`/`rw-wps` forward the "
             "--source you gave them, and a hand-written call that "
             "omits it gets prose instead of a command",
    )
    return parser


def _next_command_lines(args, proof: Mapping[str, object]) -> list[str]:
    """What this door says after the proof document, on stderr.

    The GFS, ERA5 and 20CRv3 front doors all finish by printing the
    prepared-forecast command with its three digests filled in, and the
    mapped route -- the one every packaged source runs on -- printed
    nothing at all.  A user who had just prepared a cycle was left to
    hand-extract three SHA-256 values from a 42 KB JSON document, and
    the first value the document offers under a hash-shaped name,
    ``proof_content_sha256``, is the one that can never be right.

    Where a complete command cannot be printed, this prints prose saying
    so.  It never prints a partial one: the whole value of the line is
    that it runs when pasted, and the two ways it could not are a
    preparation made from a user's own mapping (which the prepared
    runners do not accept by name) and a hand-written call to this
    module that never said which source id it is preparing.
    """

    # `packaged_profile_sources()`, NOT the forecast runner's own
    # SUPPORTED_SOURCES, and the reason is a packaging boundary rather
    # than a preference: the RW-WPS standalone distribution stages this
    # preprocessing module and deliberately EXCLUDES the forecast
    # executor, so importing it here breaks that wheel
    # (tests/test_native_wrf_distribution.py catches it).  The two sets
    # cannot drift: the runner builds `_MAPPED_PACKAGED_PROFILE` from
    # this same table, and every id in it is one it accepts -- held by
    # test_every_packaged_source_is_a_source_the_runner_accepts.
    from gpuwm.gfs_direct import prepared_forecast_next_command
    from gpuwm.source_adapters import packaged_profile_sources

    source = args.prepared_forecast_source
    if source is None:
        return [
            "",
            "rw-wps: preparation complete.  No forecast command is "
            "printed because this call named no prepared-forecast source "
            "id; pass --prepared-forecast-source ID (the same id `gpuwm "
            "prep --source` takes) to get the complete hash-bound "
            "command here.",
        ]
    if source not in packaged_profile_sources():
        return [
            "",
            f"rw-wps: preparation complete, and there is no forecast "
            f"command to print for --source {source}: the "
            f"prepared-forecast runners bind a bundle to a PACKAGED "
            f"source certificate, and this preparation was made from a "
            f"mapping of your own.  The bundle under "
            f"{args.output_root} is complete; a run of it goes through "
            f"the same runner with a packaged source id.",
        ]
    return prepared_forecast_next_command(
        proof, output_root=args.output_root,
        experiment_config=args.experiment_config,
        wps_namelist=args.wps_namelist, source=source)


@owns_source_coverage_refusal
def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mapped_engine is not None:
        # Published into the environment rather than threaded through
        # `prepare_mapped_wrf`: the design freezes that signature, and
        # the environment variable is the SAME switch a library caller
        # uses, so the flag and the documented workaround cannot drift
        # into meaning different things.
        os.environ[_ENGINE_ENV] = args.mapped_engine
    if args.input_list is not None:
        # Resolved before any other input is opened, so a bad list file
        # is refused at the door rather than after the mapping loads.
        try:
            args.primary_files = read_input_list(args.input_list)
        except ValueError as error:
            parser.error(str(error))
    mapping = load_mapping(args.mapping)
    if mapping["format"] != args.source_format:
        parser.error(
            f"--source-format {args.source_format} differs from mapping "
            f"format {mapping['format']}"
        )
    decoder_values = {
        "grib1_bridge": args.grib1_bridge,
        "grib2_inventory": args.grib2_inventory,
        "grib2_dump": args.grib2_dump,
    }
    present_decoders = {
        name for name, value in decoder_values.items() if value is not None
    }
    # On the Rust engine no subprocess decoder runs, so demanding paths
    # to one is the staged-tool papercut again: a refusal asking for
    # executables nothing is going to launch.  The flags stay ACCEPTED
    # (supplying one is how a caller pins a tool, which routes the call
    # to the Python engine) but they stop being required.
    # ``compose`` unconditionally, because `prepare_mapped_wrf` composes
    # unconditionally -- contributing mappings only widen the format
    # union, they are not what makes this route a composition.  Asking
    # ``decode`` for the single-source spelling is what let this
    # validator wave a bare GRIB2 call through on the Rust engine that
    # the route then handed to the Python engine with no tools, so the
    # refusal arrived nineteen frames deep in the decoder contract
    # instead of here.
    engine_decodes_in_process = (
        not present_decoders
        and _mapped_engine_choice(
            grib1_bridge=args.grib1_bridge,
            grib2_inventory=args.grib2_inventory,
            grib2_dump=args.grib2_dump,
            subcommand=_ROUTE_SUBCOMMAND,
            source_format=args.source_format,
        ) == _ENGINE_RUST
    )
    required_decoders = set() if engine_decodes_in_process else {
        "grib1": {"grib1_bridge"},
        "grib2": {"grib2_inventory", "grib2_dump"},
        "netcdf": set(),
    }[args.source_format]
    # Stated as the decoder-contract refusal rather than as an argparse
    # usage error, and in the same sentence the route's own contract
    # uses.  A bare `usage: ... error: grib2 requires decoder flags` is
    # the staged-tool papercut in its original form: it demands a flag
    # pointing at a file the reader may not have, and says nothing about
    # getting one.  `gpuwm prep` resolves these through the shared
    # ladder and forwards them, so this door is reached bare only by a
    # hand-written call -- which deserves the estate answer, not a
    # vocabulary complaint.
    if not args.contributing_mapping:
        unsatisfied = present_decoders != required_decoders
    else:
        # A contributing source may decode a different format, so extra
        # decoder flags are arbitrated by the exact format union inside
        # prepare_mapped_wrf; the primary's own decoders stay mandatory.
        unsatisfied = not required_decoders <= present_decoders
    if unsatisfied:
        raise _decoder_inventory_refusal(
            f"{args.source_format} decoder inventory differs from the "
            f"contract; missing={sorted(required_decoders - present_decoders)}"
            f", extra={sorted(present_decoders - required_decoders)}",
            (args.source_format,),
            missing=required_decoders - present_decoders,
            extra=present_decoders - required_decoders,
        )
    if args.preprocess_workers is not None and args.preprocess_workers <= 0:
        parser.error("--preprocess-workers must be positive")
    if args.hierarchy_workers is not None \
            and args.hierarchy_workers not in range(1, 33):
        parser.error("--hierarchy-workers must be between 1 and 32")
    if args.preprocess_backend != "cpu" \
            and args.cpu_preprocess_bridge is not None:
        parser.error(
            "--cpu-preprocess-bridge requires --preprocess-backend cpu"
        )
    if (args.static_input is None) != (args.static_receipt is None):
        parser.error("--static-input and --static-receipt must be given together")
    try:
        supplements = _role_bindings(args.supplement, multiple=True)
        provenance = _role_bindings(args.provenance, multiple=False)
        contributing = _role_bindings(
            args.contributing_mapping, multiple=False,
        )
    except ValueError as error:
        parser.error(str(error))
    proof = prepare_mapped_wrf(
        composition=args.composition,
        mapping=args.mapping,
        primary_files=args.primary_files,
        supplement_files=supplements,
        provenance_files=provenance,
        contributing_mappings=contributing,
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
        grib1_bridge=args.grib1_bridge,
        grib2_inventory=args.grib2_inventory,
        grib2_dump=args.grib2_dump,
        wps_namelist=args.wps_namelist,
        geog_root=args.geog_root,
        static_input=args.static_input,
        static_receipt=args.static_receipt,
        experiment_config=args.experiment_config,
        output_root=args.output_root,
        preprocess_backend=args.preprocess_backend,
        preprocess_workers=args.preprocess_workers,
        cpu_preprocess_bridge=args.cpu_preprocess_bridge,
        hierarchy_workers=args.hierarchy_workers,
    )
    print(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False))
    for line in _next_command_lines(args, proof):
        print(line, file=sys.stderr)
    return 0


__all__ = [
    "HIERARCHY_PROOF_SCHEMA",
    "PROOF_SCHEMA",
    "main",
    "prepare_mapped_wrf",
]


if __name__ == "__main__":
    raise SystemExit(main())
