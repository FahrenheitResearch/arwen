#!/usr/bin/env python3
"""Run one GFS/ERA5/20CRv3 RW-WPS prepared cache through GPUWM.

The runner accepts a portable direct single-domain output or the external-LBC
d01 bundle in a portable hierarchy published by the public GFS and ERA5
adapters.  It reconstructs the cache identity from the caller-pinned proof,
source manifest, static geometry, experiment, and WPS namelist before importing
CuPy.  WRF-shaped history files are published at the exact cadence bound into
the experiment through gpuwm's existing atomic writer and carry
``GPUWM_WRITE_COMPLETE=1``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib
import traceback
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm import __version__  # noqa: E402
from gpuwm.config import radiation_scheme_ids  # noqa: E402
from gpuwm.core.nssl2_contract import (  # noqa: E402
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    DEFAULT_MODE as NSSL2_DEFAULT_MODE,
    MP_PHYSICS as NSSL2_MP_PHYSICS,
    WRF_NAMELIST_DEFAULTS as NSSL2_WRF_NAMELIST_DEFAULTS,
    WRF_REFERENCE_COMMIT as NSSL2_WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION as NSSL2_WRF_REFERENCE_VERSION,
)
from gpuwm.core.thompson_contract import (  # noqa: E402
    AUXILIARY_TABLE_RECORDS as THOMPSON_AUXILIARY_TABLE_RECORDS,
    CLASSIC_TABLE_ASSETS as THOMPSON_CLASSIC_TABLE_ASSETS,
    GENERATED_TABLE_FILES as THOMPSON_GENERATED_TABLE_FILES,
    MP_PHYSICS as THOMPSON_MP_PHYSICS,
    TABLE_SET_ID as THOMPSON_TABLE_SET_ID,
    TRANSPORTED_SPECIES as THOMPSON_TRANSPORTED_SPECIES,
    WRF_REFERENCE_COMMIT as THOMPSON_WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION as THOMPSON_WRF_REFERENCE_VERSION,
    validate_table_assets as validate_thompson_table_assets,
)
from gpuwm.experiment import build_experiment, load_experiment  # noqa: E402
from gpuwm.explain import warn  # noqa: E402
from gpuwm.kernel_compile_notice import (  # noqa: E402
    COMPILING_STATUS, kernel_compile_notice,
)
from gpuwm.ingest.prepared_cache import (  # noqa: E402
    PREPARED_CACHE_SCHEMA,
    PreparedCacheReader,
    prepared_cache_identity,
)
from gpuwm.native_wrf_contract import (  # noqa: E402
    NATIVE_LANDUSE_IDENTITY,
    load_native_static_cache,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    verify_native_static_receipt,
)
from gpuwm.supervisor import atomic_write_json  # noqa: E402
from gpuwm.physics_compat import (  # noqa: E402
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    MULTI_DOMAIN_SELECTION_SCHEMA,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    PhysicsCapabilityError,
    RUC_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_TABLE_ROOT_ENV,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_SUBSTITUTION_TOKENS,
    WSM6_PROFILE_ID,
    acknowledgement_delivery,
    identify_single_domain_profile,
    land_surface_component_for_selector,
    land_surface_route_blocker,
    single_domain_physics_selection,
    single_domain_runtime_switches,
    single_domain_verification_status,
    thompson_runtime_requirements,
    thompson_table_root,
    validate_single_domain_physics_profile,
)
from gpuwm.certify.capsule import emit_run_capsule  # noqa: E402
from gpuwm.source_authorities import twentycrv3_authority_sha256  # noqa: E402
from gpuwm.table_assets import require_thompson_tables  # noqa: E402
from gpuwm.wrf_physics_inventory import (  # noqa: E402
    stock_wrf_physics_inventory,
)


REPORT_SCHEMA = "gpuwm-prepared-single-domain-forecast-v1"
PROGRESS_SCHEMA = "gpuwm-prepared-single-domain-progress-v1"
RUNNER_CAPABILITIES_SCHEMA = "gpuwm-runner-capabilities-v1"
AUTHORITY_MATERIALIZATION_SCHEMA = (
    "gpuwm-named-source-physics-authorities-v1")
PHYSICS_PROFILE = WSM6_PROFILE_ID
KESSLER_PHYSICS_PROFILE = KESSLER_PROFILE_ID
TWENTYCRV3_WSM6_PHYSICS_PROFILE = (
    "20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1")
THOMPSON_PHYSICS_PROFILE = THOMPSON_PROFILE_ID
MORRISON_PHYSICS_PROFILE = MORRISON_PROFILE_ID
NSSL2_PHYSICS_PROFILE = NSSL2_PROFILE_ID
NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE = NSSL2_LEGACY_RRTMG_PROFILE_ID
RUC_PHYSICS_PROFILE = RUC_PROFILE_ID
MYNN_RUC_PHYSICS_PROFILE = MYNN_RUC_PROFILE_ID
MYNN_PHYSICS_PROFILE = MYNN_PROFILE_ID
NOAHMP_PHYSICS_PROFILE = NOAHMP_PROFILE_ID
MYNN_NOAHMP_PHYSICS_PROFILE = MYNN_NOAHMP_PROFILE_ID
PHYSICS_PROFILES = (
    PHYSICS_PROFILE,
    KESSLER_PHYSICS_PROFILE,
    TWENTYCRV3_WSM6_PHYSICS_PROFILE,
    THOMPSON_PHYSICS_PROFILE,
    MORRISON_PHYSICS_PROFILE,
    NSSL2_PHYSICS_PROFILE,
    NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
    MYNN_PHYSICS_PROFILE,
    RUC_PHYSICS_PROFILE,
    MYNN_RUC_PHYSICS_PROFILE,
    NOAHMP_PHYSICS_PROFILE,
    MYNN_NOAHMP_PHYSICS_PROFILE,
)
SUPPORTED_SOURCES = frozenset({"gfs", "era5", "20crv3", "hrrr"})

#: The HRRR bundle this runner reads is the one
#: ``tools/prepare_hrrr_wrf.py`` publishes, and it is NOT the portable
#: single-domain layout the other sources share.  Its artifacts keep the
#: names the native HRRR route has always written -- the preparation is
#: the certified one and renaming its outputs to suit a reader would put
#: the certification at risk for nothing -- so the reader learns the
#: layout instead.  Everything hash-bound about it is identical: the
#: same prepared-cache format, the same identity recomputation, the same
#: fail-closed comparison against the caller's pins.
HRRR_DIRECT_LAYOUT = "hrrr-native-direct-v1"

#: Where an HRRR bundle keeps each artifact, relative to --prepared-root.
HRRR_BUNDLE_PATHS = MappingProxyType({
    "static": "native-static.npz",
    "geometry_receipt": "native-geometry-receipt.json",
    "prepared_cache": "native/prepared-cache",
    "bridge_manifest": "native/native-bridge/SHA256SUMS",
})

#: ``mp_physics`` values whose microphysics call stages a scheme-native
#: REFL_10CM field, i.e. exactly ``gpuwm.runtime.REFL_10CM_MICROPHYSICS``.
#: Duplicated rather than imported: ``gpuwm.runtime`` pulls in the GRIB,
#: static-geography and ERA5 ingest stack, and this runner deliberately
#: imports none of it.  The duplication is held equal by
#: ``tests/test_mp28_runtime_reachability.py``, which imports both and
#: asserts they are the same set -- and which fails on ANY surviving
#: ``mp_physics in (1, 6, 8, 10, 18)`` literal anywhere under ``gpuwm/``,
#: so a fifth copy of this gate cannot be added silently.
REFL_10CM_MICROPHYSICS = (1, 6, 8, 10, 18, 28)
_SOURCE_PHYSICS_PROFILES = MappingProxyType({
    # REPORTED METADATA, NOT A GATE (owner ruling 2026-07-31): these
    # per-source lists name the shipped profiles whose verification
    # evidence this runner can vouch for on each source.  They feed the
    # capabilities receipt and the registry drift check; nothing refuses
    # a suite for being absent from them.  The one real per-source
    # blocker they used to encode -- RUC absent from GFS because a
    # GFS-initialised RUC forecast prepares in full and then dies on its
    # first surface-temperature call with `mavail must be finite`
    # (v1.1.1 field finding; completing that initialisation is a v1.2
    # item) -- is enforced on the resolved sf_surface_physics selector
    # by the registry's land-surface route declaration instead, in
    # ``_validate_physics`` below, for named and unnamed suites alike.
    "gfs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE,
        NOAHMP_PHYSICS_PROFILE, MYNN_NOAHMP_PHYSICS_PROFILE),
    "era5": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE,
        RUC_PHYSICS_PROFILE, MYNN_RUC_PHYSICS_PROFILE),
    "20crv3": (
        TWENTYCRV3_WSM6_PHYSICS_PROFILE, PHYSICS_PROFILE,
        THOMPSON_PHYSICS_PROFILE, MORRISON_PHYSICS_PROFILE,
        NSSL2_PHYSICS_PROFILE, NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE),
    # HRRR's own stock-WRF gate is the WSM6/YSU/MM5-91/Noah slice, and
    # gpuwm/hrrr_route_inputs.py's SUPPORTED_MICROPHYSICS admits the
    # same scheme set the other sources report here.
    "hrrr": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
})
_TWENTYCRV3_WSM6_RUNTIME_SWITCHES = MappingProxyType({
    "moist": True, "moist_cq": False, "mp_physics": 6,
    "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
    "wsm6_hail_opt": 0, "ra_physics": 4,
    "ra_lw_physics": -1, "ra_sw_physics": -1, "radt": 12.0,
    "wrf_rrtmg_compatibility": "none",
    "sf_sfclay_physics": 91, "sf_surface_physics": 2,
    "bl_pbl_physics": 1, "cu_physics": 1, "cudt_minutes": 5.0,
    "num_soil_layers": 4, "terrain_opt": 1,
    "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.12,
    "diff_6th_slopeopt": 1,
})
#: The switch names an experiment-config-selected (custom) suite's
#: receipt records: the union of what the shipped profile rows pin,
#: in one canonical order, so two receipts for one config are
#: byte-identical.
_SUITE_RECEIPT_SWITCH_KEYS = (
    "moist", "moist_cq", "mp_physics", "top_lid", "epssm",
    "morr_rimed_ice", "wsm6_hail_opt", "ra_physics",
    "ra_lw_physics", "ra_sw_physics", "radt",
    "wrf_rrtmg_compatibility", "ra_rrtmg_variant",
    "sf_sfclay_physics", "sf_surface_physics",
    "bl_pbl_physics", "cu_physics", "cudt_minutes",
    "num_soil_layers", "terrain_opt", "km_opt", "diff_6th_opt",
    "diff_6th_factor", "diff_6th_slopeopt",
)
_MATERIALIZED_PHYSICS_KEYS = frozenset({
    "moist", "moist_cq", "mp_physics", "top_lid", "epssm",
    "morr_rimed_ice", "wsm6_hail_opt", "ra_physics",
    "ra_lw_physics", "ra_sw_physics", "radt", "radt_minutes",
    "wrf_rrtmg_compatibility", "ra_rrtmg_variant", "sf_sfclay_physics",
    "sf_surface_physics", "bl_pbl_physics", "cu_physics",
    "cudt_minutes", "num_soil_layers", "terrain_opt", "km_opt",
    "diff_6th_opt", "diff_6th_factor", "diff_6th_slopeopt",
    "nest_microphysics_transition",
})
_SOURCE_SCHEMA = {
    "gfs": "gpuwm-gfs-direct-input-manifest-v1",
    "era5": "gpuwm-era5-direct-input-manifest-v1",
    "20crv3": "gpuwm-20crv3-grib2-inputs-v1",
    "hrrr": "gpuwm-hrrr-native-input-manifest-v1",
}
_PROOF_SCHEMA = {
    "gfs": "gpuwm-gfs-direct-wrf-proof-v3",
    "era5": "gpuwm-era5-direct-wrf-proof-v2",
    "20crv3": "gpuwm-mapped-direct-wrf-proof-v1",
    "hrrr": "gpuwm-hrrr-native-direct-wrf-proof-v1",
}
_LEGACY_PROOF_SCHEMAS = {
    # v2 remains independently verifiable.  It predates the explicit
    # front-door physics selection receipt and therefore cannot be promoted
    # to v3 by inference.
    "gfs": frozenset({"gpuwm-gfs-direct-wrf-proof-v2"}),
    "era5": frozenset(),
    "20crv3": frozenset(),
    # New in this runner as of the DA background lane: there is no
    # earlier HRRR bundle for it to have to accept.
    "hrrr": frozenset(),
}
_HIERARCHY_PROOF_SCHEMA = {
    "gfs": "gpuwm-gfs-native-hierarchy-proof-v2",
    "era5": "gpuwm-era5-native-hierarchy-proof-v1",
    "20crv3": "gpuwm-mapped-native-hierarchy-proof-v1",
    # HRRR's multi-domain route is gpuwm.hrrr_hierarchy_direct feeding
    # gpuwm.prepared_domain_tree_forecast, a designed division of labour
    # this lane does not widen.  Naming a schema no HRRR preparation
    # writes keeps _resolve_prepared_layout's generic path from matching
    # by accident; the explicit refusal below is what a caller sees.
    "hrrr": "gpuwm-hrrr-native-hierarchy-proof-unreachable-here",
}
_LEGACY_HIERARCHY_PROOF_SCHEMAS = {
    # v1 predates the front-door physics receipt the v2 hierarchy proof
    # carries and cannot be promoted to it by inference, the same rule
    # the direct proof's v2 lives under.
    "gfs": frozenset({"gpuwm-gfs-native-hierarchy-proof-v1"}),
    "era5": frozenset(),
    "20crv3": frozenset(),
    "hrrr": frozenset(),
}
_SOURCE_ADAPTER = {
    "gfs": "gfs-pgrb2-0p25-direct-v1",
    "era5": "era5-grib1-direct-v1",
    "20crv3": "rw-wps-20crv3-member-grib2-v1",
    # The adapter identity gpuwm/source_adapters.py declares for HRRR.
    "hrrr": "hrrr-native-state-v1",
}
_DECODER_IMPLEMENTATION = {
    "gfs": "gpuwm-all-rust-gfs-grib2-bridge",
    "era5": "gpuwm-all-rust-grib1-bridge",
}
_LANDUSE_IDENTITY = MappingProxyType(dict(NATIVE_LANDUSE_IDENTITY))
_CANONICAL_SURFACE_FIELDS = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})
_REQUIRED_MET_FIELDS = frozenset({
    "LANDSEA", "SKINTEMP", "T2", "U10", "V10",
})
_LBC_FIELDS = ("mu", "phi", "qv", "theta", "u", "v")
_HEX = frozenset("0123456789abcdef")
_TWENTYCRV3_SOURCE = "NOAA-CIRES-DOE 20CRv3 every-member GRIB2"
_TWENTYCRV3_MEMBER_IDENTITY = "filename_memNNN_not_grib2_pdt"
_TWENTYCRV3_FILENAME = re.compile(
    r"^mem(?P<member>[0-9]{3})_(?P<time>[0-9]{10})_"
    r"(?P<role>pl|sfc)\.grb2$")
_TWENTYCRV3_DECODER_ROLES = frozenset({"grib2_inventory", "grib2_dump"})
_DIRECT_LAYOUTS = frozenset({
    "portable-single-domain-v2", "mapped-direct-d01-v1",
    HRRR_DIRECT_LAYOUT})
_HIERARCHY_LAYOUTS = frozenset({
    "hierarchy-d01-v1", "mapped-hierarchy-d01-v1"})
_THOMPSON_IMPLEMENTATION_FILES = (
    "gpuwm/core/microphysics.py",
    "gpuwm/core/refl.py",
    "gpuwm/core/thompson.py",
    "gpuwm/core/thompson_contract.py",
    "gpuwm/core/thompson_runtime.py",
    "gpuwm/core/kernels/refl.cu",
    "gpuwm/core/kernels/thompson.cu",
)
_FORECAST_EXECUTOR_MODULES = (
    "gpuwm.core.clock",
    "gpuwm.core.dycore",
    "gpuwm.core.health",
    "gpuwm.core.model",
    "gpuwm.core.refl",
    "gpuwm.io.wrfout",
    "gpuwm.state_digest",
)


def _missing_forecast_executor_modules() -> list[str]:
    missing = []
    for module in _FORECAST_EXECUTOR_MODULES:
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(module)
    return missing


def runner_capabilities() -> dict[str, object]:
    """Return the side-effect-free contract consumed by Studio doctor."""

    missing_executor_modules = _missing_forecast_executor_modules()
    forecast_available = not missing_executor_modules
    thompson = thompson_runtime_requirements()
    # The key name physics_profile_ids predates the 2026-07-31 owner
    # ruling, when these per-source lists were the ADMISSION lists.  The
    # key survives for existing consumers, but its meaning is now the
    # verification evidence this runner can vouch for on each source,
    # and each entry says so beside the list rather than relying on a
    # consumer to find the top-level physics_admission block.
    source_profile_ids_semantics = (
        "per-source WRF-verification evidence this runner vouches for; "
        "NOT an admission list -- any engine-valid suite runs from the "
        "hash-bound experiment config on every supported source (see "
        "physics_admission)")
    sources = {
        "gfs": {
            "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
            "prepared_layouts": [
                "portable-single-domain-v2", "hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(_SOURCE_PHYSICS_PROFILES["gfs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
        },
        "era5": {
            "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
            "prepared_layouts": [
                "portable-single-domain-v2", "hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(_SOURCE_PHYSICS_PROFILES["era5"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
        },
        "20crv3": {
            "readiness": (
                "NATIVE_EXACT_MEMBER_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "member_identity": _TWENTYCRV3_MEMBER_IDENTITY,
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["20crv3"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "prepared source is hash-bound to one filename member",
                "GPU forecast execution has no public 20CRv3 acceptance gate",
                "hierarchy input may be consumed only as executable d01",
            ],
        },
    }
    physics_profiles = {
        PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "SUPPORTED_RUNNER_PROFILE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
        },
        KESSLER_PHYSICS_PROFILE: {
            "selector": 1,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "Kessler mp1 has no gpuwm/WRF forecast trajectory "
                "comparison on this runner's prepared sources"),
        },
        TWENTYCRV3_WSM6_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "source_scope": ["20crv3"],
            "warning": (
                "GPU execution is implemented but has no public 20CRv3 "
                "forecast acceptance gate"),
        },
        THOMPSON_PHYSICS_PROFILE: {
            "selector": THOMPSON_MP_PHYSICS,
            **thompson,
        },
        MORRISON_PHYSICS_PROFILE: {
            "selector": 10,
            "readiness": "WRF_MATCHED_RUN_RUNTIME_PROFILE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
        },
        NSSL2_PHYSICS_PROFILE: {
            "selector": NSSL2_MP_PHYSICS,
            "readiness": "WRF_MATCHED_RUN_CANDIDATE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
            "external_table_assets": [],
            "contract_id": NSSL2_CONTRACT_ID,
            "resolved_fixed_preset": True,
        },
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE: {
            "selector": NSSL2_MP_PHYSICS,
            "readiness": "WRF_MATCHED_RUN_CANDIDATE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
            "external_table_assets": [],
            "contract_id": NSSL2_CONTRACT_ID,
            "radiation_solver": "legacy RRTMG",
            "resolved_fixed_preset": True,
        },
        MYNN_PHYSICS_PROFILE: {
            # Same microphysics selector as PHYSICS_PROFILE: what this profile
            # selects is the coupled MYNN 5/5 surface-layer/PBL pair, and
            # gpuwm/config.py validate_run_config refuses half of it.
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "MYNN 5/5 runs a 300-step coupled forecast with a bitwise "
                "restart but has no gpuwm/WRF trajectory comparison; the CUDA "
                "leaves also differ from the CPU references by up to 137 ULP "
                "in the diffusivities away from their oracle fixtures"),
        },
        RUC_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "RUC LSM runs with its nine-layer soil state but has no "
                "gpuwm/WRF forecast trajectory comparison"),
        },
        MYNN_RUC_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "MYNN/MYNN/RUC follows WRF's ownership sequence but has no "
                "gpuwm/WRF forecast trajectory comparison"),
        },
        NOAHMP_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED_EXPERT",
            "explicit_expert_consent_required": True,
            "explicit_profile_selection_required": True,
            "runtime_guards": [
                "measured 360,000-column ceiling or explicit expert budget",
                "glacier columns refused",
            ],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "Noah-MP runs on the device but has no gpuwm/WRF forecast "
                "trajectory comparison"),
        },
        MYNN_NOAHMP_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED_EXPERT",
            "explicit_expert_consent_required": True,
            "explicit_profile_selection_required": True,
            "runtime_guards": [
                "measured 360,000-column ceiling or explicit expert budget",
                "glacier columns refused",
            ],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "MYNN/MYNN/Noah-MP follows WRF's ownership sequence but has "
                "no gpuwm/WRF forecast trajectory comparison"),
        },
    }
    return {
        "schema": RUNNER_CAPABILITIES_SCHEMA,
        "runner": "tools.prepared_single_domain_forecast",
        "supported_sources": (
            sorted(SUPPORTED_SOURCES) if forecast_available else []),
        "physics_profile_ids": (
            list(PHYSICS_PROFILES) if forecast_available else []),
        # Owner ruling 2026-07-31: the ids above are the shipped suites
        # whose WRF-verification evidence this runner reports; they are
        # not an admission list.  Any suite the engine implements runs
        # from the hash-bound experiment config, --physics-profile is an
        # optional exactness assertion, and verification status is
        # receipt metadata, never a gate.
        "physics_admission": {
            "mode": "any-engine-valid-suite-from-hash-bound-experiment",
            "physics_profile_flag": "optional-exactness-assertion",
            "verification_status": "reported-never-gating",
        },
        "report_schema": REPORT_SCHEMA,
        "progress_schema": PROGRESS_SCHEMA,
        "readiness": (
            "FORECAST_IMPLEMENTATION_PRESENT_RUNTIME_PREFLIGHT_REQUIRED"
            if forecast_available
            else "FORECAST_EXECUTOR_OMITTED"
        ),
        "modes": {
            "forecast": {
                "available": forecast_available,
                "availability_scope": "executor-module-presence-only",
                "launch_ready": None,
                "launch_readiness_check": (
                    "validate CuPy, CUDA, GPU allocation, prepared authorities, "
                    "and profile guards before launch"
                ),
                "requires_cupy": True,
                "requires_compatible_cuda_gpu": True,
                "missing_executor_modules": missing_executor_modules,
                "included_in_standalone_rw_wps_wheel": False,
                "unavailable_reason": (
                    None
                    if forecast_available
                    else "GPUWM forecast executor modules are absent"
                ),
            },
        },
        "standalone_rw_wps_wheel": {
            "runner_included": False,
            "forecast_executor_included": False,
            "reason": "the standalone wheel is preprocessing/export only",
        },
        "source_profiles": sources if forecast_available else {},
        "physics_profiles": physics_profiles if forecast_available else {},
        "guards": {
            "claim_output_directory_create_only": True,
            "single_specified_non_nested_d01": True,
            "feedback_required": 0,
            "restart_interval_seconds_required": 0,
            "hash_bound_inputs": [
                "proof",
                "source-manifest",
                "mapped-source-evidence",
                "prepared-cache-content",
                "experiment-config",
                "wps-namelist",
            ],
        },
        "window": {
            "kind": "model-relative-prepared-cache-forcing-hours",
            "minimum_frame_count": 2,
            "maximum_source_forecast_hour": None,
            "maximum_run_seconds": None,
            "limit_policy": "prepared-cache-forcing-coverage",
            "model_forcing_must_start_at_zero": True,
            "run_seconds": {
                "finite_positive_required": True,
                "whole_hour_required": True,
                "must_equal_hash_bound_experiment": True,
            },
            "source_forcing_cadence_hours": {
                "gfs": [1, 3],
                "era5": "uniform-positive-whole-hour",
                "20crv3": "manifest-bound-uniform-positive-whole-hour",
            },
            "forcing_must_cover_run": True,
        },
        "output": {
            "io_modes": ["history"],
            "history_interval_seconds": {
                "explicit_cli_value_required": True,
                "finite_positive_required": True,
                "must_equal_hash_bound_experiment": True,
                "must_be_whole_model_steps": True,
                "must_evenly_divide_run": False,
                "schedule_policy": (
                    "initial-and-floor-multiples-at-or-before-run-end"),
                "initial_frame_required": True,
                "frame_at_run_end_required": False,
                "last_scheduled_frame_may_precede_run_end": True,
            },
            "configurable_cadence": True,
            "restart_output": False,
        },
        "capability_query": {
            "flag": "--show-capabilities",
            "side_effect_free": True,
            "requires_cupy": False,
            "validates_gpu_or_runtime_assets": False,
        },
        "authority_materialization": {
            "available": True,
            "mode_flag": "--materialize-authorities",
            "requires_cupy": False,
            "create_only_output_directory": True,
            "atomic_file_publication": True,
            "inputs": [
                "source", "base-experiment-config", "base-wps-namelist",
                "physics-profile", "output-directory",
            ],
            "outputs": {
                "experiment_config": "experiment.toml",
                "wps_namelist": "namelist.wps",
                "receipt": "authority-receipt.json",
            },
            "receipt_schema": AUTHORITY_MATERIALIZATION_SCHEMA,
            "preserved_controls": [
                "domain topology and horizontal geometry",
                "experiment start and duration",
                "history and restart cadence",
                "explicit eta levels, p_top, hybrid_opt, and etac",
                "WPS namelist bytes",
            ],
            "source_physics_profile_ids": {
                source: list(profiles)
                for source, profiles in _SOURCE_PHYSICS_PROFILES.items()
            },
            "source_physics_profile_ids_semantics": (
                source_profile_ids_semantics),
        },
    }


@dataclass(frozen=True)
class PreparedForecastInputs:
    source: str
    layout: str
    prepared_root: Path
    domain_bundle_path: Path
    proof_path: Path
    source_manifest_path: Path
    static_path: Path
    geometry_receipt_path: Path
    prepared_cache_path: Path
    experiment_config: Path
    wps_namelist: Path
    proof: Mapping[str, object]
    source_manifest: Mapping[str, object]
    geometry_receipt: Mapping[str, object]
    cache_identity: Mapping[str, object]
    cache_identity_compatibility: Mapping[str, object]
    cache_reader: PreparedCacheReader
    experiment: object
    grid: object
    static: Mapping[str, np.ndarray]
    landuse_identity: Mapping[str, object]
    forcing_hours: tuple[int, ...]
    boundary_interval_seconds: int
    physics_receipt: Mapping[str, object]
    export_source_receipt: Mapping[str, object]
    #: The bound source-manifest identity plus the fetch level provenance
    #: it carries, for sources that publish one (GFS today).  Recorded so
    #: an acceptance harness can read back WHICH pressure ladder the run's
    #: inputs came off, rather than inferring it from a constant.
    source_manifest_receipt: Mapping[str, object] | None
    file_sha256: Mapping[str, str]
    authority_paths: Mapping[str, Path]
    source_domain_count: int
    source_member: str | None


@dataclass(frozen=True)
class _PreparedLayout:
    kind: str
    domain_bundle: Path
    static_path: Path
    geometry_receipt_path: Path
    prepared_cache_path: Path
    authority_paths: Mapping[str, Path]
    domain_receipt: Mapping[str, object] | None = None
    hierarchy_receipt: Mapping[str, object] | None = None
    artifact_manifest: Mapping[str, object] | None = None
    wrf_manifest: Mapping[str, object] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False)


def _history_period_count(run_seconds: float, cadence_seconds: float) -> int:
    """Return complete history periods due at or before model stop."""

    run_seconds = float(run_seconds)
    cadence_seconds = float(cadence_seconds)
    if not math.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError("run-seconds must be finite and positive")
    if not math.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        raise ValueError(
            "history-interval-seconds must be finite and positive")
    return Fraction(run_seconds) // Fraction(cadence_seconds)


def _history_output_schedule(
        *, start_time: datetime, run_seconds: float, cadence_seconds: float,
        domain_id: int = 1,
) -> tuple[tuple[float, datetime, str], ...]:
    """Resolve exact model-relative offsets, valid times, and WRF filenames."""

    periods = _history_period_count(run_seconds, cadence_seconds)
    cadence = Fraction(float(cadence_seconds))
    records = []
    for index in range(periods + 1):
        offset_seconds = float(index * cadence)
        valid_time = start_time + timedelta(seconds=offset_seconds)
        if valid_time.microsecond != 0:
            raise ValueError(
                "history cadence produces sub-second valid times that cannot "
                "be represented by second-complete WRF history filenames")
        records.append((
            offset_seconds,
            valid_time,
            valid_time.strftime(
                f"wrfout_d{int(domain_id):02d}_%Y-%m-%d_%H_%M_%S"),
        ))
    names = [record[2] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("history cadence produces duplicate WRF filenames")
    return tuple(records)


def _validate_hash_bound_history_cadence(
        exp, history_interval_seconds: float,
) -> dict[str, object]:
    """Cross-check CLI cadence against the immutable experiment authority."""

    requested = float(history_interval_seconds)
    domain = exp.root
    experiment_cadence = float(domain.history_interval_s)
    run_copy_cadence = float(domain.run.output_interval_s)
    if requested != experiment_cadence or requested != run_copy_cadence:
        # The hash-bound experiment value is what the run uses either
        # way; a stale flag is named and overridden, never a refusal.
        warn(f"--history-interval-seconds {requested:g} differs from "
             f"the hash-bound experiment history_interval_s "
             f"({experiment_cadence:g} s); the experiment value is "
             "authoritative and is used")
    exact_steps = Fraction(experiment_cadence) / exp.dt_exact(domain.grid_id)
    if exact_steps.denominator != 1 or exact_steps < 1:
        raise ValueError(
            "hash-bound history cadence is not a positive whole number of "
            "exact model time steps")
    schedule = _history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=experiment_cadence)
    periods = len(schedule) - 1
    first = schedule[0][1]
    last_offset, last_scheduled, _name = schedule[-1]
    run_end = first + timedelta(seconds=float(exp.run_seconds))
    last_equals_run_end = (
        Fraction(periods) * Fraction(experiment_cadence)
        == Fraction(float(exp.run_seconds)))
    return {
        "schema": "gpuwm-hash-bound-history-cadence-v1",
        "requested_seconds": requested,
        "experiment_seconds": experiment_cadence,
        "exact_model_steps_per_interval": int(exact_steps),
        "complete_intervals": periods,
        "expected_frame_count": periods + 1,
        "initial_valid_time": first.isoformat(),
        "last_scheduled_offset_seconds": last_offset,
        "last_scheduled_valid_time": last_scheduled.isoformat(),
        "run_end_offset_seconds": float(exp.run_seconds),
        "run_end_valid_time": run_end.isoformat(),
        "last_scheduled_equals_run_end": last_equals_run_end,
        "initial_frame_required": True,
        "run_end_frame_scheduled": last_equals_run_end,
    }


def _profile_runtime_switches(source: str, profile: str) -> dict[str, object]:
    """Resolve one complete named physics product.

    No per-source membership refusal any more (owner ruling 2026-07-31:
    the suite choice is the user's; ``_SOURCE_PHYSICS_PROFILES`` is
    reported verification metadata, not a gate).  What remains
    fail-closed here is a NAME the shipped switch tables genuinely do
    not define, and the one real per-source blocker -- the registry's
    land-surface route declaration -- is enforced on the resolved
    selector in :func:`_validate_physics`, for named and unnamed suites
    alike.
    """

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported prepared forecast source {source!r}")
    if profile == TWENTYCRV3_WSM6_PHYSICS_PROFILE:
        return dict(_TWENTYCRV3_WSM6_RUNTIME_SWITCHES)
    try:
        return single_domain_runtime_switches(profile)
    except ValueError:
        raise ValueError(
            f"physics profile {profile!r} is not a shipped runner "
            f"profile; omit --physics-profile to run the hash-bound "
            f"experiment's own suite as written, or pick one of "
            f"{list(PHYSICS_PROFILES)!r}") from None


def _profile_readiness(source: str, profile: str) -> tuple[str, str | None]:
    if source == "20crv3":
        suffix = {
            THOMPSON_PHYSICS_PROFILE: (
                "; Thompson MP8 also remains an experimental table-bound "
                "runtime"),
            NSSL2_PHYSICS_PROFILE: (
                "; NSSL-2 MP18 also remains a matched-run candidate"),
            NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE: (
                "; NSSL-2 MP18 plus legacy RRTMG remains a matched-run "
                "candidate"),
        }.get(profile, "")
        return "IMPLEMENTED_UNVERIFIED", (
            "this 20CRv3 GPU forecast profile has no public acceptance gate"
            f"{suffix}")
    if profile == TWENTYCRV3_WSM6_PHYSICS_PROFILE:
        # Selectable on any prepared source now; its verification
        # standing does not improve with the source it runs on.
        return "IMPLEMENTED_UNVERIFIED", (
            "this profile has no public forecast acceptance gate")
    if profile == THOMPSON_PHYSICS_PROFILE:
        return "WRF_MATCHED_RUN_EXPERIMENTAL_RUNTIME", (
            "Thompson MP8 remains an experimental table-bound runtime")
    if profile == MORRISON_PHYSICS_PROFILE:
        return "WRF_MATCHED_RUN_RUNTIME_PROFILE", None
    if profile in (
            NSSL2_PHYSICS_PROFILE,
            NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
    ):
        return "WRF_MATCHED_RUN_CANDIDATE", (
            "NSSL-2 MP18 is implemented but remains a matched-run candidate")
    if profile == MYNN_PHYSICS_PROFILE:
        return "IMPLEMENTED_UNVERIFIED", (
            "MYNN 5/5 has no gpuwm/WRF forecast trajectory comparison")
    if profile == RUC_PHYSICS_PROFILE:
        return "IMPLEMENTED_UNVERIFIED", (
            "RUC LSM has no gpuwm/WRF forecast trajectory comparison")
    if profile == MYNN_RUC_PHYSICS_PROFILE:
        return "IMPLEMENTED_UNVERIFIED", (
            "MYNN/MYNN/RUC has no gpuwm/WRF forecast trajectory comparison")
    if profile in (NOAHMP_PHYSICS_PROFILE, MYNN_NOAHMP_PHYSICS_PROFILE):
        return "IMPLEMENTED_UNVERIFIED_EXPERT", (
            "Noah-MP has no gpuwm/WRF forecast trajectory comparison and "
            "retains its registry-owned expert acknowledgement")
    # Every other shipped template answers with its REGISTRY maturity
    # rather than a flat supported default: the fallback used to be
    # reachable only by WSM6 (maturity 'supported', which it matched),
    # and now that any registered template runs by name, a Kessler-class
    # implemented-unverified template must not read as more mature than
    # the verification block beside it in the same receipt.
    from gpuwm.physics_registry import physics_registry

    template = physics_registry()["templates"].get(profile)
    maturity = (
        template.get("maturity") if isinstance(template, Mapping) else None)
    if maturity == "supported":
        return "SUPPORTED_RUNNER_PROFILE", None
    if maturity == "wrf-matched-run":
        return "WRF_MATCHED_RUN_RUNTIME_PROFILE", None
    return "IMPLEMENTED_UNVERIFIED", (
        f"registry maturity {maturity!r}: this profile has no "
        "gpuwm/WRF forecast trajectory comparison on this runner")


def _toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("physics profile contains a non-finite float")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    raise TypeError(f"unsupported TOML physics value {value!r}")


def _without_materialized_physics(raw: Mapping[str, object]) -> object:
    """Return a comparison copy containing every non-profile control."""

    normalized = copy.deepcopy(dict(raw))
    shared = normalized.get("shared")
    if isinstance(shared, dict):
        for key in _MATERIALIZED_PHYSICS_KEYS:
            shared.pop(key, None)
    domains = normalized.get("domain")
    if isinstance(domains, list):
        for domain in domains:
            if isinstance(domain, dict):
                for key in _MATERIALIZED_PHYSICS_KEYS:
                    domain.pop(key, None)
    return normalized


def _json_safe_toml(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_toml(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_safe_toml(item) for item in value]
    if isinstance(value, datetime):
        return {"toml_datetime": value.isoformat()}
    return value


def _non_physics_descriptor_sha256(raw: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(_json_safe_toml(
        _without_materialized_physics(raw))).encode("ascii")).hexdigest()


def _experiment_tables(raw: Mapping[str, object]) -> dict[str, object]:
    """The [experiment]/[[domain]] view ``build_experiment`` accepts.

    ``[fetch]`` is an advisory acquisition-hints table that ``gpuwm
    domain`` writes into every config it emits, and that ``gpuwm check``
    and ``rw-wps`` both accept -- but the materializer used to hand the
    raw dict straight to ``build_experiment``, which refused it with
    ``unknown table(s) ['fetch']``.  The wizard's own output was
    therefore rejected by the one step that has to run BEFORE the front
    door.  Split it off exactly as the CLI loaders do, validating it on
    the way past rather than ignoring it.
    """

    tables = dict(raw)
    fetch_table = tables.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source="materialized experiment")
    tables.pop("case_data", None)
    return tables


def _render_materialized_experiment(
        base_text: str, *, source: str, profile: str | None,
) -> tuple[str, object, dict[str, object]]:
    """Patch only profile-owned TOML keys and preserve all other controls.

    ``profile=None`` (owner ruling 2026-07-31) means the base config's
    own physics IS the product: the authority pair is still published --
    later stages bind these exact bytes -- with no switch rewritten, and
    the receipt reports the suite's verification status instead of a
    profile.
    """

    base_raw = tomllib.loads(base_text)
    base_exp = build_experiment(_experiment_tables(base_raw),
                                source="base named-source experiment")
    from gpuwm.experiment import (
        refuse_unrouted_perturbation, refuse_unrouted_spawn,
    )
    refuse_unrouted_perturbation(
        base_exp, "prepared single-domain forecast")
    refuse_unrouted_spawn(base_exp, "prepared single-domain forecast")
    if profile is None:
        for domain in base_exp.domains:
            component = land_surface_component_for_selector(
                getattr(domain.run, "sf_surface_physics", None))
            if component is not None:
                blocker = land_surface_route_blocker(
                    component, source=source)
                if blocker is not None:
                    raise ValueError(
                        f"d{int(domain.grid_id):02d}: {blocker}")
        base_non_physics = _non_physics_descriptor_sha256(base_raw)
        return base_text, base_exp, {
            "base_non_physics_descriptor_sha256": base_non_physics,
            "generated_non_physics_descriptor_sha256": base_non_physics,
            "profile_validation": {
                "schema": "gpuwm-prepared-physics-suite-v1",
                "source": source,
                "profile": None,
                "profile_binding": "experiment-config",
                "verification": single_domain_verification_status(
                    base_exp.root.run),
            },
        }
    switches = _profile_runtime_switches(source, profile)
    component = land_surface_component_for_selector(
        switches.get("sf_surface_physics"))
    if component is not None:
        blocker = land_surface_route_blocker(component, source=source)
        if blocker is not None:
            raise ValueError(blocker)
    header = re.compile(
        r"^\s*(\[\[|\[)([A-Za-z0-9_.-]+)(\]\]|\])\s*(?:#.*)?$")
    assignment = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*=")
    has_shared = any(
        (match := header.match(line)) is not None
        and match.group(1) == "[" and match.group(2) == "shared"
        for line in base_text.splitlines()
    )
    profile_lines = [
        "# RW-WPS generated exact physics authority; do not hand-edit.",
        f"# source={source} profile={profile}",
        *(
            f"{key} = {_toml_literal(value)}"
            for key, value in switches.items()
        ),
        'nest_microphysics_transition = "same-scheme-only"',
    ]
    output: list[str] = []
    section: str | None = None
    shared_emitted = False

    def finish_section() -> None:
        nonlocal shared_emitted
        if section == "shared" and not shared_emitted:
            if output and output[-1] != "":
                output.append("")
            output.extend(profile_lines)
            shared_emitted = True

    for line in base_text.splitlines():
        match = header.match(line)
        if match is not None:
            finish_section()
            is_array = match.group(1) == "[["
            next_section = match.group(2)
            if is_array and next_section == "domain" and not has_shared:
                output.extend(["[shared]", *profile_lines, ""])
                shared_emitted = True
                has_shared = True
            section = next_section
            output.append(line)
            continue
        key_match = assignment.match(line)
        if (section in {"shared", "domain"} and key_match is not None
                and key_match.group(1) in _MATERIALIZED_PHYSICS_KEYS):
            continue
        output.append(line)
    finish_section()
    if not shared_emitted:
        if output and output[-1] != "":
            output.append("")
        output.extend(["[shared]", *profile_lines])
    rendered = "\n".join(output) + "\n"
    rendered_raw = tomllib.loads(rendered)
    rendered_exp = build_experiment(
        _experiment_tables(rendered_raw),
        source="materialized named-source experiment")
    base_non_physics = _non_physics_descriptor_sha256(base_raw)
    generated_non_physics = _non_physics_descriptor_sha256(rendered_raw)
    if generated_non_physics != base_non_physics:
        raise RuntimeError(
            "materialized experiment changed a non-physics descriptor control")
    validation = _validate_profile_switches(
        rendered_exp, source=source, profile=profile, all_domains=True)
    return rendered, rendered_exp, {
        "base_non_physics_descriptor_sha256": base_non_physics,
        "generated_non_physics_descriptor_sha256": generated_non_physics,
        "profile_validation": validation,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def materialize_named_source_authorities(
        *, source: str, base_experiment_config: Path,
        base_wps_namelist: Path, physics_profile: str | None,
        output_directory: Path,
) -> dict[str, object]:
    """Publish create-only per-case experiment/WPS physics authorities.

    ``physics_profile=None`` publishes the base pair with the config's
    own physics unchanged; the receipt reports its verification status.
    """

    base_experiment_config = _require_file(
        base_experiment_config, "base experiment config")
    base_wps_namelist = _require_file(
        base_wps_namelist, "base WPS namelist")
    base_text = base_experiment_config.read_text(encoding="utf-8")
    rendered, exp, validation = _render_materialized_experiment(
        base_text, source=source, profile=physics_profile)
    # THE earliest point on the documented route at which the physics
    # this run will execute is known: step 2 of 6, before the fetch and
    # before preprocessing.  An mp8 suite whose lookup tables were never
    # staged is refused here, in one sentence naming the table and
    # `gpuwm fetch-tables`, rather than eight minutes and two paid
    # stages later at the top of the GPU run.
    if any(int(domain.run.mp_physics) == THOMPSON_MP_PHYSICS
           for domain in exp.domains):
        require_thompson_tables(assets=THOMPSON_CLASSIC_TABLE_ASSETS)
    output_directory = claim_output_directory(
        output_directory, flag="--output-directory")
    experiment = output_directory / "experiment.toml"
    wps = output_directory / "namelist.wps"
    receipt_path = output_directory / "authority-receipt.json"
    _atomic_write_bytes(experiment, rendered.encode("utf-8"))
    _atomic_write_bytes(wps, base_wps_namelist.read_bytes())
    if _sha256(wps) != _sha256(base_wps_namelist):
        raise RuntimeError("materialized WPS namelist is not a byte-exact copy")
    if physics_profile is None:
        verification = validation["profile_validation"]["verification"]
        readiness = "EXPERIMENT_CONFIG_SUITE"
        warning = (
            None if verification["status"] == "wrf-verified"
            else verification["sentence"])
    else:
        readiness, warning = _profile_readiness(source, physics_profile)
    receipt = {
        "schema": AUTHORITY_MATERIALIZATION_SCHEMA,
        "status": "PASS",
        "source": source,
        "physics_profile": physics_profile,
        "readiness": readiness,
        "warning": warning,
        "warning_only": warning is not None,
        "base": {
            "experiment_config": {
                "path": str(base_experiment_config.resolve()),
                "bytes": base_experiment_config.stat().st_size,
                "sha256": _sha256(base_experiment_config),
            },
            "wps_namelist": {
                "path": str(base_wps_namelist.resolve()),
                "bytes": base_wps_namelist.stat().st_size,
                "sha256": _sha256(base_wps_namelist),
            },
        },
        "generated": {
            "experiment_config": {
                "path": str(experiment.resolve()),
                "bytes": experiment.stat().st_size,
                "sha256": _sha256(experiment),
            },
            "wps_namelist": {
                "path": str(wps.resolve()),
                "bytes": wps.stat().st_size,
                "sha256": _sha256(wps),
            },
        },
        "normalized_selected_physics": validation["profile_validation"],
        "non_physics_descriptor": {
            "base_sha256": validation[
                "base_non_physics_descriptor_sha256"],
            "generated_sha256": validation[
                "generated_non_physics_descriptor_sha256"],
            "status": "EXACT_UNCHANGED",
        },
        "preserved": {
            "domain_count": len(exp.domains),
            "run_seconds": float(exp.run_seconds),
            "start_time": exp.start_time.isoformat(),
            "mass_levels": int(exp.root.run.nz),
            "eta_levels": list(exp.vertical.eta_levels),
            "p_top": float(exp.vertical.p_top),
            "hybrid_opt": int(exp.vertical.hybrid_opt),
            "etac": float(exp.vertical.etac),
        },
    }
    _atomic_json(receipt_path, receipt)
    receipt["receipt"] = {
        "path": str(receipt_path.resolve()),
        "bytes": receipt_path.stat().st_size,
        "sha256": _sha256(receipt_path),
    }
    return receipt


def _resolved_wrf_direct_contract_sha256(mp_physics: int) -> str:
    """Reproduce the exact resolved contract digest emitted by the exporter."""

    from gpuwm.wrf_direct import (
        _contract_payload_sha256,
        _load_contract,
        _physics_contract_bundle,
    )

    return _contract_payload_sha256(
        _physics_contract_bundle(_load_contract(), int(mp_physics)))


def _strict_json(value):
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _stability_diagnosis(sample: Mapping[str, object], state, run) -> str:
    """Say which Courant term tripped, and that dt is the remedy.

    The vertical term is the maximum co-located ``dt*|w_upper|/dz_cell``;
    it never combines a global updraft with an unrelated thin layer.
    """

    def number(key, default):
        try:
            return float(sample.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    dt = float(run.dt)
    u_max = number("u_max", float("nan"))
    w_max = number("w_max", float("nan"))
    horizontal = number(
        "horizontal_cfl", dt * u_max / float(run.dx))
    vertical = number("vertical_cfl", float("nan"))
    if vertical >= horizontal:
        term = (
            f"VERTICAL Courant {vertical:.2f} "
            "(maximum co-located |w|/layer-thickness)")
    else:
        term = (f"HORIZONTAL Courant {horizontal:.2f} (u_max {u_max:.2f} "
                f"m/s over dx {float(run.dx) / 1000:.3f} km)")
    suggested = max(1.0, dt / 4.0)
    return (
        f"{term} dominates at time_step {dt:g} s "
        f"(horizontal {horizontal:.2f}, vertical {vertical:.2f}); "
        f"REMEDY: retry with a lower dt -- set time_step to about "
        f"{suggested:g} s in the experiment TOML and re-run the front "
        "door (the config is hash-bound to it).  Shortening the step is "
        "cheaper than the step count suggests: radiation and cumulus "
        "are called on wall-clock intervals, so a 4x shorter step "
        "measured +22% wall time, not 4x"
    )


def _atomic_json(path: Path, payload) -> None:
    atomic_write_json(Path(path), _strict_json(payload))


def _duplicate_checked_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    path = Path(path)

    def reject_constant(value):
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require_digest(value, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _HEX for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path.resolve()


def _require_directory(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path.resolve()


def _validated_thompson_runtime_contract() -> dict[str, object]:
    """Bind the guarded MP8 runtime to exact WRF tables and source bytes."""

    # The two process-environment gates that used to stand here predate
    # the packaging promotion: the classic tables ship as package data,
    # `gpuwm fetch-tables` stages the externalized one, and `gpuwm
    # doctor` byte-validates all four.  Requiring the env vars on top
    # made mp8 fail twice at runtime on a machine doctor had just called
    # clean, with neither variable documented.  thompson_table_root()
    # honours GPUWM_THOMPSON_TABLE_ROOT as an override and falls back to
    # the packaged directory; either way the assets are re-validated
    # byte for byte below, which is the check that ever mattered.
    #
    # Absence is answered BEFORE the byte validation, and in a sentence
    # that names the command which fixes it.  This function runs while
    # the physics receipt is built -- at --materialize-authorities,
    # which is step 2 of 6 on the documented GFS route -- so an install
    # that never staged the externalized tables learns it here, before
    # the fetch and before preprocessing, instead of from a
    # FileNotFoundError traceback at the top of the GPU run.
    root = _require_directory(
        Path(require_thompson_tables(assets=THOMPSON_CLASSIC_TABLE_ASSETS)),
        "Thompson table root")
    assets = validate_thompson_table_assets(root)
    if tuple(assets) != tuple(THOMPSON_CLASSIC_TABLE_ASSETS):
        raise ValueError("validated Thompson assets differ from the classic contract")

    table_identity = {
        "schema": 1,
        "table_set": THOMPSON_TABLE_SET_ID,
        "wrf_version": THOMPSON_WRF_REFERENCE_VERSION,
        "wrf_commit": THOMPSON_WRF_REFERENCE_COMMIT,
        "assets": [
            {
                "filename": asset.filename,
                "bytes": int(asset.bytes),
                "sha256": asset.sha256,
            }
            for asset in assets
        ],
    }
    implementation_sha256 = {
        relative: _sha256(_require_file(
            REPO / relative, f"Thompson implementation {relative}"))
        for relative in _THOMPSON_IMPLEMENTATION_FILES
    }
    payload_bytes = sum(
        record.payload_bytes
        for records in THOMPSON_GENERATED_TABLE_FILES.values()
        for record in records
    ) + sum(
        record.payload_bytes for record in THOMPSON_AUXILIARY_TABLE_RECORDS)
    return {
        "selector": THOMPSON_MP_PHYSICS,
        "wrf_reference_version": THOMPSON_WRF_REFERENCE_VERSION,
        "wrf_reference_commit": THOMPSON_WRF_REFERENCE_COMMIT,
        "transported_fields": list(THOMPSON_TRANSPORTED_SPECIES),
        "guard": {
            # Retired: the enable gate.  What remains is the resolution
            # rule and the byte validation that always did the work.
            "table_root_environment": THOMPSON_TABLE_ROOT_ENV,
            "table_root_source": (
                "environment override" if os.environ.get(
                    THOMPSON_TABLE_ROOT_ENV) else "packaged default"),
        },
        "table_authority": {
            **table_identity,
            "root": str(root),
            "payload_bytes": int(payload_bytes),
            "identity_sha256": hashlib.sha256(
                _canonical(table_identity).encode("ascii")).hexdigest(),
        },
        "implementation_sha256": implementation_sha256,
        "native_reflectivity": {
            "field": "REFL_10CM",
            "producer": "classic-Thompson same-call graupel-number shadow",
            "consumer": "gpuwm.core.refl.consume_refl_10cm",
            "fallback": None,
        },
    }


def _receipt_mp_physics(physics_receipt: Mapping[str, object]) -> int:
    """The resolved microphysics selector a physics receipt records."""

    resolved = physics_receipt.get("resolved")
    if not isinstance(resolved, Mapping):
        return -1
    try:
        return int(resolved.get("mp_physics"))
    except (TypeError, ValueError):
        return -1


def _thompson_authority_paths(
        physics_receipt: Mapping[str, object],
) -> dict[str, Path]:
    # Keyed on the resolved selector, not the profile id: every admitted
    # mp8 suite binds the table authority (matching the domain-tree
    # runner, which has always keyed this on mp_physics == 8).
    if _receipt_mp_physics(physics_receipt) != THOMPSON_MP_PHYSICS:
        return {}
    contract = physics_receipt.get("thompson_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("Thompson physics receipt lacks its contract")
    table = contract.get("table_authority")
    if not isinstance(table, Mapping):
        raise RuntimeError("Thompson physics receipt lacks table authority")
    root = Path(str(table.get("root"))).resolve()
    paths = {
        f"thompson_table_{asset.filename}": _require_file(
            root / asset.filename, f"Thompson table {asset.filename}")
        for asset in THOMPSON_CLASSIC_TABLE_ASSETS
    }
    paths.update({
        f"thompson_implementation_{index:02d}": _require_file(
            REPO / relative, f"Thompson implementation {relative}")
        for index, relative in enumerate(_THOMPSON_IMPLEMENTATION_FILES)
    })
    return paths


def _verify_thompson_runtime_environment(
        physics_receipt: Mapping[str, object],
) -> None:
    if _receipt_mp_physics(physics_receipt) != THOMPSON_MP_PHYSICS:
        return
    contract = physics_receipt.get("thompson_contract")
    table = contract.get("table_authority") if isinstance(contract, Mapping) else None
    expected_root = table.get("root") if isinstance(table, Mapping) else None
    actual_root = str(Path(thompson_table_root()).resolve())
    if actual_root != expected_root:
        raise RuntimeError(
            "Thompson table root changed between preflight and run: "
            f"preflight resolved {expected_root}, now {actual_root}")


def _sibling_outdir(protected: Path) -> Path:
    """A concrete --outdir the guard below will accept, beside ``protected``.

    Named in the refusal so it reads as an instruction rather than a rule,
    and matching what the GFS front door suggests, so the two surfaces
    send the user to the same directory.
    """

    protected = Path(protected)
    return protected.parent / f"{protected.name}-forecast"


def claim_output_directory(
        path: Path, *, protected_roots=(), flag: str = "--outdir",
) -> Path:
    """Create one owned output directory without merging prior state.

    Refusing to reuse a directory is deliberate -- a run that merged
    into a previous run's output would publish a receipt describing two
    runs -- but ``mkdir(exist_ok=False)``'s own ``FileExistsError``
    reaches the reader as a traceback whose last line is a Windows
    error number.  A person who has just re-run a command with the same
    ``--output-directory`` gets one sentence naming the directory and
    the two ways out instead; ``FLAG`` is the flag they actually typed,
    because "pass --outdir" is unhelpful advice to someone who typed
    ``--output-directory``.
    """

    path = Path(path)
    # resolve() walks symlinks and raises OSError on a loop (Errno 40 /
    # "Symlink loop from ...").  This function's whole subject is turning
    # an OS-level exception into a sentence, and this was the one it let
    # through: a looping --outdir reached the reader as a traceback at
    # rc 1, from the line before the first refusal.
    resolved = _resolve_or_refuse(path, flag)
    for protected in protected_roots:
        protected = _resolve_or_refuse(Path(protected), flag)
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                f"forecast output must not be inside protected input tree "
                f"{protected}; the forecast may not write into its own "
                f"inputs.  Pass an {flag} beside them instead, for "
                f"example {_sibling_outdir(protected)}")
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"{resolved} already exists, and this runner never merges "
            f"into an earlier run's directory -- the receipt it writes "
            f"has to describe one run.  Pass a new {flag}, or remove "
            f"the old directory first.") from error
    return _resolve_or_refuse(path, flag)


def _resolve_or_refuse(path: Path, flag: str) -> Path:
    """``path.resolve()``, with its failures delivered as sentences.

    RuntimeError is caught alongside OSError, and deliberately so: on a
    symlink loop pathlib raises ``RuntimeError("Symlink loop from ...")``
    rather than ``OSError(ELOOP)``, which an OSError-only guard misses
    entirely.  Found by running the fix on Linux against the installed
    wheel -- the Windows box this was written on cannot create a looping
    symlink without privilege, so the test skipped and the guard looked
    correct.  The catch is narrow enough to stay honest: it wraps one
    call whose only failure mode is "this path does not resolve".
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError) as error:
        detail = getattr(error, "strerror", None) or str(error)
        errno = getattr(error, "errno", None)
        where = f" (errno {errno})" if errno is not None else ""
        raise ValueError(
            f"{flag} {path} cannot be resolved to a real location: "
            f"{detail}{where}.  A symlink that points into its own chain "
            f"has no destination to create, so there is nothing here to "
            f"refuse or to reuse -- pass an {flag} that names a real "
            f"path.") from error


def _validate_restored_source_adapter(metadata, source: str) -> None:
    """Reject a conflicting optional cache hint after identity verification.

    Direct bundles historically persist ``source_adapter`` in user metadata;
    hierarchy domain caches do not.  The mandatory source identity is already
    reconstructed and checked during preflight, so absence is valid but a
    contradictory redundant value is not.
    """

    adapter = dict(metadata).get("source_adapter")
    expected = "mapped" if source == "20crv3" else source
    if adapter is not None and adapter != expected:
        raise ValueError("restored cache source adapter differs from request")


def _physics_update_count(component) -> int:
    """Return zero for an explicitly disabled optional physics component."""

    if component is None:
        return 0
    value = int(component.update_count)
    if value < 0:
        raise ValueError("physics update count cannot be negative")
    return value


def _validate_restored_cache_receipt(
        receipt: Mapping[str, object], expected_content_sha256: str,
) -> None:
    """Recheck the caller's content pin immediately after cache restore."""

    if (not isinstance(receipt, Mapping)
            or receipt.get("schema") != PREPARED_CACHE_SCHEMA
            or receipt.get("status") != "RESTORED"
            or receipt.get("content_sha256") != expected_content_sha256):
        raise ValueError(
            "restored prepared cache differs from the caller-pinned content")


def _runtime_source_identity() -> dict[str, object]:
    """Bind the exact forecast implementation, not only preparation code."""

    paths = (
        REPO / "gpuwm/core/model.py",
        REPO / "gpuwm/ingest/hrrr_physics.py",
        REPO / "gpuwm/ingest/hrrr_surface.py",
        REPO / "gpuwm/ingest/lateral_bc.py",
        REPO / "gpuwm/ingest/prepared_cache.py",
        REPO / "gpuwm/ingest/real.py",
        REPO / "gpuwm/io/wrfout.py",
        REPO / "gpuwm/native_wrf_contract.py",
        REPO / "gpuwm/source_authorities.py",
        REPO / "gpuwm/prepared_single_domain_forecast.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"prepared forecast runtime sources are missing: {missing}")
    source_sha256 = {
        str(path.relative_to(REPO)).replace("\\", "/"): _sha256(path)
        for path in paths
    }
    # One resolver for all three installs (gpuwm.runtime_manifest).  The
    # branch this replaced degraded a wheel install to
    # `runtime-module-sha256-only` -- honest, but weaker than the truth:
    # pip knows exactly which artifact it wrote, and RECORD says so.  It
    # also raised, uncaught, when site-packages happened to sit INSIDE
    # some unrelated repository, binding a stranger's commit or dying;
    # that case is now simply "not a checkout of this tree".
    from gpuwm.runtime_manifest import IdentityError, provenance

    try:
        identity = provenance(REPO)
    except IdentityError:
        # Neither a manifest, nor a checkout, nor an installed
        # distribution: a source tree someone copied into place.  The
        # per-module digests above still bind what ran.
        identity = {
            "git_commit": None, "git_tree": None, "git_status_short": None,
            "identity_source": "runtime-module-sha256-only",
            "distribution_manifest_sha256": None, "installed_wheel": None,
        }
    return {**identity, "source_sha256": source_sha256}


def _durable_wrfout_inventory(output_directory: Path) -> list[dict[str, object]]:
    """Inventory atomic-complete files retained by a failed or passing run."""

    inventory = []
    for path in sorted((Path(output_directory) / "wrfout").glob("wrfout_d??_*")):
        if not path.is_file():
            continue
        inventory.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return inventory


def _hierarchy_rows(exp) -> list[dict[str, object]]:
    return [{
        "grid_id": int(domain.grid_id),
        "parent_id": int(domain.parent_id),
        "i_parent_start": int(domain.i_parent_start),
        "j_parent_start": int(domain.j_parent_start),
        "parent_grid_ratio": int(domain.parent_grid_ratio),
        "parent_time_step_ratio": int(domain.parent_time_step_ratio),
        "nx": int(domain.run.nx),
        "ny": int(domain.run.ny),
        "nz": int(domain.run.nz),
        "dx_m": float(domain.run.dx),
        "dy_m": float(domain.run.dy),
        "dt_s": float(domain.run.dt),
        "mp_physics": int(domain.run.mp_physics),
    } for domain in exp.domains]


def _resolve_prepared_layout(
        *, source: str, prepared_root: Path, proof: Mapping[str, object],
        source_exp, domain_bundle: Path | None,
) -> _PreparedLayout:
    """Resolve either a direct bundle or hash-bound hierarchy d01 bundle."""

    schema = proof.get("schema")
    if proof.get("status") != "READY_NOT_YET_STOCK_WRF_GATED":
        raise ValueError(
            f"{source.upper()} preparation proof is not READY for forecast")
    if source == "hrrr":
        if schema != _PROOF_SCHEMA[source]:
            raise ValueError(
                f"unsupported HRRR preparation proof schema {schema!r}; "
                "this runner reads the single-domain bundle "
                "tools/prepare_hrrr_wrf.py publishes.  A prepared HRRR "
                "domain TREE is the other route: run it with "
                "gpuwm-prepared-tree-forecast (module form: python -m "
                "gpuwm.prepared_domain_tree_forecast)")
        if len(source_exp.domains) != 1:
            raise ValueError(
                "the HRRR single-domain bundle requires a one-domain "
                "experiment; a domain tree runs on the tree runner")
        expected = prepared_root.resolve()
        if domain_bundle is not None \
                and Path(domain_bundle).resolve() != expected:
            raise ValueError(
                "HRRR preparation --domain-bundle must equal --prepared-root")
        return _PreparedLayout(
            kind=HRRR_DIRECT_LAYOUT,
            domain_bundle=expected,
            static_path=_require_file(
                expected / HRRR_BUNDLE_PATHS["static"],
                "native static cache"),
            geometry_receipt_path=_require_file(
                expected / HRRR_BUNDLE_PATHS["geometry_receipt"],
                "geometry receipt"),
            prepared_cache_path=_require_directory(
                expected / HRRR_BUNDLE_PATHS["prepared_cache"],
                "prepared cache"),
            authority_paths=MappingProxyType({
                "bridge_manifest":
                    expected / HRRR_BUNDLE_PATHS["bridge_manifest"],
            }),
        )
    if (schema == _PROOF_SCHEMA[source]
            or schema in _LEGACY_PROOF_SCHEMAS[source]):
        if len(source_exp.domains) != 1:
            raise ValueError(
                "single-domain preparation proof requires a one-domain "
                "experiment")
        expected = prepared_root.resolve()
        if domain_bundle is not None and Path(domain_bundle).resolve() != expected:
            raise ValueError(
                "direct preparation --domain-bundle must equal --prepared-root")
        return _PreparedLayout(
            kind=(
                "mapped-direct-d01-v1"
                if source == "20crv3" else "portable-single-domain-v2"),
            domain_bundle=expected,
            static_path=_require_file(
                expected / "native-static.npz", "native static cache"),
            geometry_receipt_path=_require_file(
                expected / "geometry-receipt.json", "geometry receipt"),
            prepared_cache_path=_require_directory(
                expected / "prepared-cache", "prepared cache"),
            authority_paths=MappingProxyType({}),
        )
    if (schema != _HIERARCHY_PROOF_SCHEMA[source]
            and schema not in _LEGACY_HIERARCHY_PROOF_SCHEMAS[source]):
        raise ValueError(
            f"unsupported {source.upper()} preparation proof schema {schema!r}")
    if len(source_exp.domains) < 2:
        raise ValueError(
            "hierarchy preparation proof requires a multi-domain experiment")
    expected_ids = list(range(1, len(source_exp.domains) + 1))
    if (proof.get("domain_count") != len(source_exp.domains)
            or [int(domain.grid_id) for domain in source_exp.domains]
            != expected_ids):
        raise ValueError(
            "hierarchy proof/domain configuration is not contiguous parent-first")

    hierarchy_root = _require_directory(
        prepared_root / "hierarchy-artifacts", "hierarchy artifact root")
    artifact_manifest_path = _require_file(
        hierarchy_root / "domain-artifacts.json",
        "hierarchy artifact manifest")
    hierarchy_receipt_path = _require_file(
        hierarchy_root / "receipt.json", "hierarchy artifact receipt")
    # This route runs d01 of a prepared hierarchy THROUGH the unchanged-WRF
    # file set, so the export is a genuine input here and its absence is a
    # refusal.  Say which absence it is: a preparation whose export was
    # refused on representability, or never requested, is a complete
    # forecast that this particular runner cannot serve -- and the domain
    # tree runner can.  A bare "file not found" sent users looking for a
    # broken preparation that is not broken.
    export_slot = proof.get("wrf_manifest")
    if isinstance(export_slot, Mapping) \
            and export_slot.get("status") not in (None, "READY"):
        raise ValueError(
            "this hierarchy preparation published no stock-WRF file set "
            f"({export_slot.get('status')}: {export_slot.get('reason')}); "
            "the prepared domain tree is complete -- run it with "
            "gpuwm-prepared-tree-forecast (module form: python -m "
            "gpuwm.prepared_domain_tree_forecast)")
    wrf_manifest_path = _require_file(
        prepared_root / "wrf-native-input" / "manifest.json",
        "hierarchy direct-WRF manifest")
    artifact_manifest = _load_json_object(
        artifact_manifest_path, "hierarchy artifact manifest")
    hierarchy_receipt = _load_json_object(
        hierarchy_receipt_path, "hierarchy artifact receipt")
    wrf_manifest = _load_json_object(
        wrf_manifest_path, "hierarchy direct-WRF manifest")
    expected_manifest = {
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": grid_id,
            "prepared_cache": f"domains/d{grid_id:02d}/prepared-cache",
            "static_cache": f"domains/d{grid_id:02d}/native-static.npz",
            "geometry_receipt": (
                f"domains/d{grid_id:02d}/geometry-receipt.json"),
        } for grid_id in expected_ids],
    }
    if artifact_manifest != expected_manifest:
        raise ValueError(
            "hierarchy artifact manifest is not the canonical domain layout")
    expected_receipt_identity = {
        "schema": "gpuwm-native-hierarchy-artifact-build-v1",
        "status": "READY",
        "domain_count": len(expected_ids),
        "grid_ids": expected_ids,
        "manifest": {
            "path": "domain-artifacts.json",
            "sha256": _sha256(artifact_manifest_path),
        },
        "boundary_inventory": {
            "external": [1],
            "nested_parent_forced": expected_ids[1:],
        },
    }
    if any(hierarchy_receipt.get(key) != value
           for key, value in expected_receipt_identity.items()):
        raise ValueError("hierarchy artifact receipt identity is inconsistent")
    domain_receipts = hierarchy_receipt.get("domains")
    if (not isinstance(domain_receipts, list)
            or len(domain_receipts) != len(expected_ids)):
        raise ValueError("hierarchy domain receipt inventory is incomplete")
    if proof.get("artifact_receipt") != hierarchy_receipt:
        raise ValueError(
            "hierarchy proof artifact receipt differs from the published tree")
    if proof.get("wrf_manifest") != wrf_manifest:
        raise ValueError(
            "hierarchy proof WRF manifest differs from the published tree")

    expected_bundle = _require_directory(
        hierarchy_root / "domains" / "d01", "hierarchy d01 bundle")
    if (domain_bundle is not None
            and Path(domain_bundle).resolve() != expected_bundle):
        raise ValueError(
            "--domain-bundle differs from the manifest-derived hierarchy d01")
    domain_receipt_path = _require_file(
        expected_bundle / "receipt.json", "hierarchy d01 receipt")
    domain_receipt = _load_json_object(
        domain_receipt_path, "hierarchy d01 receipt")
    if domain_receipt != domain_receipts[0]:
        raise ValueError(
            "hierarchy d01 receipt differs from the hierarchy authority")
    expected_domain_identity = {
        "schema": "gpuwm-native-domain-artifact-build-v1",
        "status": "READY",
        "grid_id": 1,
        "parent_id": 0,
        "boundary_mode": "external-specified",
        "valid_time": source_exp.start_time.isoformat(),
        "forcing_hours": proof.get("forcing_hours"),
    }
    if any(domain_receipt.get(key) != value
           for key, value in expected_domain_identity.items()):
        raise ValueError("hierarchy d01 receipt identity is inconsistent")
    hierarchy_authority_paths = {
        "hierarchy_artifact_manifest": artifact_manifest_path,
        "hierarchy_receipt": hierarchy_receipt_path,
        "domain_receipt": domain_receipt_path,
        "wrf_manifest": wrf_manifest_path,
    }
    if source == "20crv3":
        hierarchy_authority_paths.update({
            "mapped_root_static": _require_file(
                prepared_root / "native-static.npz",
                "mapped hierarchy root static cache"),
            "mapped_root_geometry": _require_file(
                prepared_root / "geometry-receipt.json",
                "mapped hierarchy root geometry receipt"),
        })
    return _PreparedLayout(
        kind=(
            "mapped-hierarchy-d01-v1"
            if source == "20crv3" else "hierarchy-d01-v1"),
        domain_bundle=expected_bundle,
        static_path=_require_file(
            expected_bundle / "native-static.npz", "hierarchy d01 static cache"),
        geometry_receipt_path=_require_file(
            expected_bundle / "geometry-receipt.json",
            "hierarchy d01 geometry receipt"),
        prepared_cache_path=_require_directory(
            expected_bundle / "prepared-cache", "hierarchy d01 prepared cache"),
        authority_paths=MappingProxyType(hierarchy_authority_paths),
        domain_receipt=MappingProxyType(domain_receipt),
        hierarchy_receipt=MappingProxyType(hierarchy_receipt),
        artifact_manifest=MappingProxyType(artifact_manifest),
        wrf_manifest=MappingProxyType(wrf_manifest),
    )


def _twentycrv3_manifest_file_specs(
        manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Validate the portable copy of one exact-member 20CRv3 manifest.

    The raw GRIB paths intentionally remain historical evidence: a prepared
    tree must remain runnable after those large inputs are moved or deleted.
    Their declared byte/hash identities are nevertheless bound into the
    copied manifest, the mapped composition receipt, and the cache identity.
    """

    expected_keys = {
        "schema", "source", "member_identity", "member", "valid_times",
        "cadence_seconds", "file_count", "files", "content_sha256",
    }
    if (set(manifest) != expected_keys
            or manifest.get("schema") != _SOURCE_SCHEMA["20crv3"]
            or manifest.get("source") != _TWENTYCRV3_SOURCE
            or manifest.get("member_identity") != _TWENTYCRV3_MEMBER_IDENTITY):
        raise ValueError(
            "20CRv3 portable source manifest has an unsupported identity or "
            "top-level inventory")
    content = dict(manifest)
    declared_content_sha256 = content.pop("content_sha256")
    if declared_content_sha256 != hashlib.sha256(
            _canonical(content).encode("utf-8")).hexdigest():
        raise ValueError("20CRv3 portable source manifest content hash is stale")
    member = manifest.get("member")
    if not isinstance(member, str) or re.fullmatch(r"[0-9]{3}", member) is None:
        raise ValueError("20CRv3 manifest member must be a three-digit label")
    rows = manifest.get("files")
    if (not isinstance(rows, list)
            or isinstance(manifest.get("file_count"), bool)
            or manifest.get("file_count") != len(rows)):
        raise ValueError("20CRv3 portable source manifest file count differs")

    normalized: dict[str, dict[str, object]] = {}
    observed: dict[datetime, set[str]] = {}
    observed_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "path", "filename", "member", "valid_time", "role",
                "bytes", "sha256"}:
            raise ValueError(f"20CRv3 manifest file row {index} is malformed")
        path_value = row.get("path")
        filename = row.get("filename")
        if (not isinstance(path_value, str) or not path_value
                or not isinstance(filename, str)
                or Path(path_value).name != filename):
            raise ValueError(
                f"20CRv3 manifest file row {index} has an unsafe path/name")
        match = _TWENTYCRV3_FILENAME.fullmatch(filename)
        if match is None:
            raise ValueError(
                f"20CRv3 manifest file row {index} has an invalid filename")
        try:
            valid_time = datetime.strptime(
                match.group("time"), "%Y%m%d%H")
        except ValueError as exc:
            raise ValueError(
                f"20CRv3 manifest file row {index} has an invalid time") from exc
        if (match.group("member") != member or row.get("member") != member
                or row.get("role") != match.group("role")
                or row.get("valid_time") != valid_time.isoformat()):
            raise ValueError(
                f"20CRv3 manifest file row {index} disagrees with its filename")
        byte_count = row.get("bytes")
        if (isinstance(byte_count, bool) or not isinstance(byte_count, int)
                or byte_count <= 0):
            raise ValueError(
                f"20CRv3 manifest file row {index} has invalid byte count")
        digest = _require_digest(
            row.get("sha256"), f"20CRv3 manifest row {index} sha256")
        if path_value in observed_paths or filename in normalized:
            raise ValueError("20CRv3 manifest contains duplicate inputs")
        observed_paths.add(path_value)
        observed.setdefault(valid_time, set()).add(match.group("role"))
        normalized[filename] = {
            "name": filename,
            "path": path_value,
            "member": member,
            "valid_time": valid_time.isoformat(),
            "role": match.group("role"),
            "bytes": byte_count,
            "sha256": digest,
        }
    if len(observed) < 2 or any(
            roles != {"pl", "sfc"} for roles in observed.values()):
        raise ValueError(
            "20CRv3 manifest requires at least two complete pressure/surface "
            "pairs")
    times = tuple(sorted(observed))
    valid_times = [value.isoformat() for value in times]
    if manifest.get("valid_times") != valid_times:
        raise ValueError("20CRv3 manifest valid-time inventory differs")
    deltas = {
        int((later - earlier).total_seconds())
        for earlier, later in zip(times, times[1:])
    }
    cadence_seconds = manifest.get("cadence_seconds")
    if (len(deltas) != 1 or isinstance(cadence_seconds, bool)
            or not isinstance(cadence_seconds, int)
            or cadence_seconds != next(iter(deltas))
            or cadence_seconds <= 0 or cadence_seconds % 3600 != 0):
        raise ValueError(
            "20CRv3 manifest cadence must be uniform, positive, and whole-hour")
    return normalized


#: The three fields that ARE the GFS cycle identity.  Bound strictly: a
#: manifest authored for another cycle is refused before any GPU setup.
_GFS_MANIFEST_IDENTITY_KEYS = ("model", "product", "cycle")

#: The fetch provenance a GFS front-door manifest's ``source`` object may
#: also carry: the pressure ladder the fetch actually took and the source
#: top it implies.  `gpuwm fetch` began writing these in `1f0fc039` so the
#: preparation lane's vertical contract could stop assuming a 100 hPa
#: constant, and `gpuwm/gfs_direct.py` reads them BEFORE the bridge runs.
#: They are real provenance, not identity -- a deeper fetch changes them
#: and changes nothing about which cycle the document names -- so folding
#: them into the identity comparison is what refused every manifest this
#: release's own fetch authors.  Enumerated rather than ignored: a field
#: nobody here knows about is still a refusal.
_GFS_MANIFEST_LEVEL_KEYS = ("pressure_levels_hpa", "top_pressure_pa")


def proof_initial_forecast_lead(proof: Mapping[str, object]) -> int:
    """The source lead the prepared run starts from, per its own proof.

    A proof written before initialization from a forecast lead existed
    carries no such block, and there is exactly one lead it can have
    meant: 0, the cycle's analysis.  Reading it as 0 is a statement of
    what those runs WERE, not a default applied to a missing field.
    """

    block = proof.get("initial_condition")
    if block is None:
        return 0
    if not isinstance(block, Mapping):
        raise ValueError(
            "preparation proof initial-condition provenance is malformed")
    lead = block.get("initial_forecast_lead_hours")
    if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
        raise ValueError(
            "preparation proof declares an unreadable initial forecast lead")
    kind = block.get("initial_condition_kind")
    process = block.get("forecast_generating_process_id")
    # A lead is never readable as an analysis, in either direction.
    if (kind != ("analysis" if lead == 0 else "forecast")
            or process != (81 if lead == 0 else 96)):
        raise ValueError(
            f"preparation proof labels forecast lead f{lead:03d} as "
            f"{kind!r} (process {process!r}); a forecast lead is not an "
            "analysis")
    return lead


def _gfs_manifest_source_receipt(
        manifest: Mapping[str, object], exp,
        proof: Mapping[str, object]) -> dict[str, object]:
    """Bind the GFS cycle identity; validate and record the level ladder.

    The identity comparison binds ``model``/``product``/``cycle`` and
    compares those three exactly.  The level fields are routed to the
    contract they serve instead: ``gfs_direct._manifest_source_top_pa`` is
    the producing lane's own validator -- it refuses an unreadable or
    non-certified ladder and cross-checks the declared top against it --
    and the resulting source top is held to the same rule the vertical
    contract applies, that the source atmosphere must reach at least as
    high as the requested model top.  Using the producer's validator is
    deliberate: two readers of one document that disagree about it is the
    defect this function exists to close.
    """

    from gpuwm.gfs_direct import _manifest_source_top_pa

    identity = manifest.get("source")
    if not isinstance(identity, dict):
        raise ValueError("GFS source manifest carries no source identity")
    known = set(_GFS_MANIFEST_IDENTITY_KEYS) | set(_GFS_MANIFEST_LEVEL_KEYS)
    unknown = sorted(set(identity) - known)
    if unknown:
        # Forward-compat: a NEWER fetch may stamp fields this runner
        # does not know.  Every field this runner reads is still bound
        # and checked below; extras are named and ignored.  (The old
        # closed-set refusal rejected every manifest this release's
        # own fetch authored.)
        warn(f"GFS source manifest identity carries field(s) {unknown} "
             "this runner does not read; ignoring them")
    bound = {key: identity.get(key) for key in _GFS_MANIFEST_IDENTITY_KEYS}
    # start_time is the model's time zero and the cycle is the source's;
    # they coincide only at lead 0.  Deriving the cycle by subtracting
    # the proof's declared lead keeps the binding exact at every lead
    # instead of demanding that a run start when its source did.
    lead_hours = proof_initial_forecast_lead(proof)
    cycle_time = exp.start_time - timedelta(hours=lead_hours)
    expected = {
        "model": "GFS",
        "product": "pgrb2.0p25",
        "cycle": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if bound != expected:
        raise ValueError(
            "GFS source manifest identity differs from the cycle this "
            "experiment starts from (start_time "
            f"{exp.start_time:%Y-%m-%d %H:%M:%S} at forecast lead "
            f"f{lead_hours:03d})")
    provenance = proof.get("initial_condition")
    if isinstance(provenance, Mapping) and (
            provenance.get("cycle") != expected["cycle"]
            or provenance.get("model_start_time")
            != exp.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")):
        raise ValueError(
            "preparation proof initial-condition provenance disagrees with "
            "the manifest cycle or the experiment start_time")
    # Absent levels mean a directory fetched by an older ArWen, whose only
    # possible ladder is the certified 100 hPa one; the validator returns
    # that constant, which is exactly what the preparation assumed.
    source_top_pa = _manifest_source_top_pa(manifest)
    p_top_pa = float(exp.vertical.p_top)
    if source_top_pa > p_top_pa:
        raise ValueError(
            f"GFS source manifest records a fetch reaching {source_top_pa:g} "
            f"Pa but the experiment requests p_top {p_top_pa:g} Pa")
    levels = identity.get("pressure_levels_hpa")
    return {
        "schema": "gpuwm-gfs-source-manifest-identity-v1",
        "identity": bound,
        "pressure_levels_hpa": (
            [float(level) for level in levels]
            if isinstance(levels, list) else None),
        "source_top_pressure_pa": source_top_pa,
        "experiment_p_top_pa": p_top_pa,
    }


def _manifest_file_specs(
        source: str, manifest: Mapping[str, object], exp,
        proof: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], Mapping[str, object] | None]:
    """Normalized role specs, plus the GFS source receipt when there is one."""

    if source == "20crv3":
        return _twentycrv3_manifest_file_specs(manifest), None
    expected_schema = _SOURCE_SCHEMA[source]
    expected_keys = {"schema", "files", "source"} \
        if source in {"gfs", "hrrr"} else {"schema", "files"}
    if set(manifest) != expected_keys or manifest.get("schema") != expected_schema:
        raise ValueError(
            f"{source.upper()} portable source manifest has an unsupported "
            "schema or top-level inventory")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("portable source manifest files must be an object")
    normalized = {}
    for role, spec in files.items():
        if (not isinstance(role, str) or not role
                or not isinstance(spec, dict)
                or set(spec) != {"name", "sha256"}):
            raise ValueError(
                f"portable source manifest role {role!r} is malformed")
        name = spec.get("name")
        if (not isinstance(name, str) or not name
                or Path(name).name != name or Path(name).is_absolute()):
            raise ValueError(
                f"portable source manifest role {role!r} has an unsafe name")
        normalized[role] = {
            "name": name,
            "sha256": _require_digest(
                spec.get("sha256"), f"source manifest {role} sha256"),
        }
    common = {"bridge", "experiment_config", "wps_namelist"}
    if source == "hrrr":
        # HRRR's portable manifest binds one more authority than the
        # others: the WRF namelist.input.  It is not decoration -- the
        # prepared-cache identity's ``namelist_sha256`` IS that file's
        # digest on this route (the native preparer reads the explicit
        # eta ladder and p_top out of it), so a bundle that did not name
        # it could not have its identity recomputed here at all.
        required = common | {"namelist_input", "source_manifest"}
        optional = {"static_input", "static_receipt", "domain_spec"}
        if not required <= set(normalized) \
                or set(normalized) - required - optional:
            raise ValueError(
                "HRRR source manifest role inventory is unsupported: "
                f"{sorted(normalized)}")
        static_pair = {"static_input", "static_receipt"} & set(normalized)
        if static_pair not in (set(), {"static_input", "static_receipt"}):
            raise ValueError("HRRR source manifest static roles are incomplete")
        return normalized, None
    if source == "gfs":
        allowed = common | {"series", "static_input", "static_receipt"}
        dynamic = {role for role in normalized if role.startswith("grib-f")}
        unknown = set(normalized) - allowed - dynamic
        if unknown or not common | {"series"} <= set(normalized):
            raise ValueError(
                f"GFS source manifest role inventory is unsupported: "
                f"{sorted(normalized)}")
        static_pair = {"static_input", "static_receipt"} & set(normalized)
        if static_pair not in (set(), {"static_input", "static_receipt"}):
            raise ValueError("GFS source manifest static roles are incomplete")
        parsed_hours = []
        for role in dynamic:
            suffix = role.removeprefix("grib-f")
            if len(suffix) != 3 or not suffix.isdigit():
                raise ValueError(f"invalid GFS GRIB role {role!r}")
            parsed_hours.append(int(suffix))
        if len(parsed_hours) < 2 or len(set(parsed_hours)) != len(parsed_hours):
            raise ValueError("GFS manifest requires at least two unique GRIB hours")
        return normalized, _gfs_manifest_source_receipt(
            manifest, exp, proof)
    else:
        required = common | {"grib", "vtable"}
        optional = {"static_input", "static_receipt", "source_orography"}
        if not required <= set(normalized) or set(normalized) - required - optional:
            raise ValueError(
                f"ERA5 source manifest role inventory is unsupported: "
                f"{sorted(normalized)}")
        static_pair = {"static_input", "static_receipt"} & set(normalized)
        if static_pair not in (set(), {"static_input", "static_receipt"}):
            raise ValueError("ERA5 source manifest static roles are incomplete")
    return normalized, None


def _validate_profile_switches(
        exp, *, source: str, profile: str, all_domains: bool = False,
) -> dict[str, object]:
    """Validate descriptor physics before cache identity or GPU setup."""

    expected = _profile_runtime_switches(source, profile)
    domains = exp.domains if all_domains else (exp.root,)
    observed_domains = []
    expected_radiation = (
        (int(expected["ra_physics"]), int(expected["ra_physics"]))
        if (int(expected["ra_lw_physics"]),
            int(expected["ra_sw_physics"])) == (-1, -1)
        else (int(expected["ra_lw_physics"]),
              int(expected["ra_sw_physics"]))
    )
    labels = {
        PHYSICS_PROFILE: "supported prepared-cache profile",
        TWENTYCRV3_WSM6_PHYSICS_PROFILE: (
            "implemented-unverified 20CRv3 WSM6+KF+RTE profile"),
        THOMPSON_PHYSICS_PROFILE: "guarded Thompson MP8 profile",
        MORRISON_PHYSICS_PROFILE: "Morrison MP10 runtime profile",
        NSSL2_PHYSICS_PROFILE: "NSSL-2 wrf-matched-run-candidate profile",
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE: (
            "NSSL-2 + legacy RRTMG wrf-matched-run-candidate profile"),
        MYNN_PHYSICS_PROFILE: "MYNN 5/5 implemented-unverified profile",
        RUC_PHYSICS_PROFILE: "RUC LSM implemented-unverified profile",
        MYNN_RUC_PHYSICS_PROFILE: (
            "MYNN 5/5 + RUC implemented-unverified profile"),
        NOAHMP_PHYSICS_PROFILE: "Noah-MP expert profile",
        MYNN_NOAHMP_PHYSICS_PROFILE: "MYNN 5/5 + Noah-MP expert profile",
    }
    for domain in domains:
        cfg = domain.run
        observed = {name: getattr(cfg, name) for name in expected}
        observed_radiation = radiation_scheme_ids(cfg)
        if observed != expected or observed_radiation != expected_radiation:
            label = labels.get(profile, f"named {profile} profile")
            raise ValueError(
                f"experiment physics differs from the {label} on "
                f"d{int(domain.grid_id):02d}: expected={expected}, "
                f"radiation={expected_radiation}, observed={observed}, "
                f"resolved_radiation={observed_radiation}.  A named "
                f"--physics-profile asserts the experiment IS that suite; "
                f"omit the flag to run the experiment's own physics as "
                f"written")
        observed_domains.append({
            "grid_id": int(domain.grid_id),
            **observed,
            "radiation_scheme_ids": list(observed_radiation),
        })
    readiness, warning = _profile_readiness(source, profile)
    return {
        "schema": "gpuwm-prepared-physics-profile-v1",
        "source": source,
        "profile": profile,
        "readiness": readiness,
        "warning": warning,
        "warning_only": warning is not None,
        "resolved": {
            **expected,
            "radiation_scheme_ids": list(expected_radiation),
        },
        "validated_domains": observed_domains,
    }


def _validate_physics(
        exp, profile: str | None, run_seconds: float,
        history_interval_seconds: float, *, source: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
) -> dict[str, object]:
    """Admit the experiment's physics and build its receipt.

    ``profile`` is an OPTIONAL assertion (owner ruling 2026-07-31): when
    named, the experiment must BE that shipped suite, switch for switch,
    exactly as before.  When omitted, the hash-bound experiment config is
    the authority and any suite the engine implements runs; its
    verification status -- "WRF-verified" against a shipped profile, or
    "supported, not yet WRF-verified" -- is recorded in the receipt and
    never gates.  Fail-closed refusals remain only where the code
    genuinely does not implement the request: engine selector validation
    (``gpuwm.config.validate_run_config``, which names the exact switch)
    and the registry's land-surface route declaration.

    The registry's source-neutral tuple governance is APPLIED here too,
    on every prepared source, for named, matched and unnamed suites
    alike -- but as a WARNING, not a refusal (warn-not-block ruling):
    an expert-template tuple is implemented and individually verified,
    so an unacknowledged one runs and prints one line naming both
    published delivery spellings, ``--ack <id>`` and
    ``acknowledgements = ["<id>"]`` in the hash-bound experiment, and
    the receipt carries ``acknowledged=false`` either way.  Applying
    the governance is what keeps the removal of the profile whitelist
    from widening consent SILENTLY: pre-ruling, expert tuples could not
    reach the non-GFS lanes at all, and every run record now states
    exactly which unblessed tuple executed.  A caller who NAMES an
    expert ``--physics-profile`` has asked for that gate and still
    meets it at the front door
    (:func:`gpuwm.physics_compat.validate_single_domain_physics_profile`).
    """

    if len(exp.domains) != 1:
        raise ValueError(
            "prepared single-domain forecast runner requires exactly one domain")
    dc = exp.root
    cfg = dc.run
    if not math.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError("run-seconds must be a positive duration")
    if run_seconds % 3600.0 != 0.0:
        warn(f"run-seconds {run_seconds:g} is not a whole number of "
             "hours; continuing (forcing coverage is checked "
             "separately)")
    if float(exp.run_seconds) != float(run_seconds) \
            or float(cfg.run_seconds) != float(run_seconds):
        warn(f"--run-seconds {run_seconds:g} differs from the "
             f"hash-bound experiment run_seconds "
             f"({float(exp.run_seconds):g} s); the experiment value is "
             "authoritative and is used")
    cadence_receipt = _validate_hash_bound_history_cadence(
        exp, history_interval_seconds)
    if (dc.grid_id != 1 or dc.parent_id != 0 or cfg.grid_id != 1
            or cfg.specified is not True or cfg.nested is not False):
        raise ValueError(
            "prepared forecast requires one specified, non-nested d01")
    if (float(exp.restart_interval_s) != 0.0
            or int(exp.feedback) != 0):
        warn("restart_interval_s/feedback are inert on the prepared "
             "single-domain runner (it writes no checkpoints, and one "
             "domain has nothing to feed back); continuing with them "
             "ignored")
    # The one genuine per-source blocker, keyed on the resolved selector
    # rather than a profile name, so a hand-authored suite meets exactly
    # the gate a named one does.  The registry declaration has no
    # opinion for sources it does not cover.
    if source is not None:
        component = land_surface_component_for_selector(
            getattr(cfg, "sf_surface_physics", None))
        if component is not None:
            blocker = land_surface_route_blocker(component, source=source)
            if blocker is not None:
                raise ValueError(f"d01: {blocker}")

    matched = identify_single_domain_profile(cfg)
    effective = profile if profile is not None else matched
    # The receipt's ``source`` is the run's actual provenance.  (Until
    # this fix the 20CRv3-vouched profile rewrote it to the vouching
    # source on cross-source runs -- a wrong label on exactly the
    # cross-source admission the ruling opened.)
    validation_source = source or "gfs"
    verification = single_domain_verification_status(cfg)
    # The registry's source-neutral tuple governance, the same call the
    # front doors and the domain-tree route make: an unacknowledged
    # expert tuple WARNS here naming the acknowledgement and runs; a
    # tuple the registry has no spelling for records its blocker and
    # runs.  The acknowledgement channels are the runner's own --ack
    # flag and the hash-bound experiment's ``acknowledgements`` array,
    # and either one silences the line.
    acknowledgements, ack_provenance = acknowledgement_delivery(
        flag=tuple(expert_acknowledgements),
        toml=tuple(getattr(exp, "acknowledgements", ()) or ()),
    )
    if effective is not None:
        receipt = _validate_profile_switches(
            exp, source=validation_source, profile=effective)
        receipt["profile_binding"] = (
            "named" if profile is not None else "matched")
        # Switch drift refuses first (the named gate's own words); only
        # a config that IS the suite reaches the consent question.
        governance_selection = single_domain_physics_selection(
            cfg, expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=ack_provenance)
        domain_governance = governance_selection["domains"]["1"]
    else:
        observed = {
            name: getattr(cfg, name)
            for name in _SUITE_RECEIPT_SWITCH_KEYS
            if hasattr(cfg, name)
        }
        observed_radiation = radiation_scheme_ids(cfg)
        # Recording is not permission withheld: the registry may simply
        # have no spelling for an engine-valid tuple (the shipped
        # hierarchy descriptor's aggregate radiation-off is the
        # exhibit), and the domain-tree route has always recorded that
        # blocker and run.  Same mechanism here -- and an expert tuple
        # the registry DOES spell warns above without its
        # acknowledgement, exactly as it does on the domain-tree route.
        governance_selection = single_domain_physics_selection(
            cfg, expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=ack_provenance)
        domain_governance = governance_selection["domains"]["1"]
        receipt = {
            "schema": "gpuwm-prepared-physics-suite-v1",
            "source": validation_source,
            "profile": None,
            "profile_binding": "experiment-config",
            "readiness": "IMPLEMENTED_SUITE_NOT_WRF_VERIFIED",
            "warning": verification["sentence"],
            "warning_only": True,
            "registry_components": domain_governance["components"],
            "registry_blocker": domain_governance["registry_blocker"],
            "resolved": {
                **observed,
                "radiation_scheme_ids": list(observed_radiation),
            },
            "validated_domains": [{
                "grid_id": int(dc.grid_id),
                **observed,
                "radiation_scheme_ids": list(observed_radiation),
            }],
        }
    receipt["registry_governance"] = {
        **domain_governance["governance"],
        "registry_sha256": governance_selection["registry_sha256"],
        "acknowledgements": list(governance_selection["acknowledgements"]),
        "acknowledgement_provenance": dict(
            governance_selection["acknowledgement_provenance"]),
    }
    receipt["verification"] = verification
    receipt["landuse_identity"] = dict(_LANDUSE_IDENTITY)
    # Scheme contracts key on the RESOLVED SELECTORS, never on the
    # profile name: an mp8 suite admitted without the Thompson profile
    # id must still bind the table authority and the preflight/run
    # table-root stability check (the domain-tree runner has always
    # keyed this on mp_physics == 8).
    mp_selector = int(cfg.mp_physics)
    if mp_selector == THOMPSON_MP_PHYSICS:
        receipt["thompson_contract"] = _validated_thompson_runtime_contract()
    elif mp_selector == 10:
        rimed_ice = int(cfg.morr_rimed_ice)
        receipt["morrison_contract"] = {
            "selector": 10,
            "morr_rimed_ice": rimed_ice,
            "rimed_ice_category": "hail" if rimed_ice == 1 else "graupel",
        }
    elif mp_selector == NSSL2_MP_PHYSICS:
        receipt["nssl2_contract"] = {
            "selector": NSSL2_MP_PHYSICS,
            "contract_id": NSSL2_CONTRACT_ID,
            "wrf_reference_version": NSSL2_WRF_REFERENCE_VERSION,
            "wrf_reference_commit": NSSL2_WRF_REFERENCE_COMMIT,
            "resolved_default_mode": asdict(NSSL2_DEFAULT_MODE),
            "transported_fields": list(NSSL2_DEFAULT_MODE.transported_fields),
            "wrf_namelist_defaults": dict(NSSL2_WRF_NAMELIST_DEFAULTS),
        }
    compatibility = str(getattr(cfg, "wrf_rrtmg_compatibility", "none"))
    if compatibility in WRF_RRTMG_SUBSTITUTION_TOKENS:
        scheme_ids = list(radiation_scheme_ids(cfg))
        receipt["radiation_substitution"] = {
            "contract": compatibility,
            "requested_wrf_scheme_ids": scheme_ids,
            "resolved_gpuwm_scheme_ids": scheme_ids,
            "resolved_gpuwm_solver": "RTE+RRTMGP",
        }
    elif compatibility == WRF_RRTMG_LEGACY:
        scheme_ids = list(radiation_scheme_ids(cfg))
        receipt["radiation_identity"] = {
            "contract": WRF_RRTMG_LEGACY,
            "requested_wrf_scheme_ids": scheme_ids,
            "resolved_gpuwm_scheme_ids": scheme_ids,
            "resolved_gpuwm_solver": "legacy RRTMG",
        }
    receipt["output_cadence"] = cadence_receipt
    return receipt


def _validate_front_door_physics_proof(
        proof: Mapping[str, object], *, source: str, profile: str | None, cfg,
) -> dict[str, object] | None:
    """Verify v3's explicit physics receipt without rewriting v2 history.

    THE profile-agnostic invariant of this route: the physics executed
    is the physics prepared.  The preparation proof records the front
    door's selection receipt, and this recomputes the same receipt from
    the same hash-bound experiment config and requires byte equality.
    The proof's own receipt says which of the two selection spellings
    the preparation used -- a named-profile receipt or the per-domain
    config-derived one -- and the recomputation follows it; both are
    pure functions of (config, registry, acknowledgements), so equality
    still proves the prepared selection came from THIS config under
    THIS registry.  The runner's own ``--physics-profile``, when named,
    is separately enforced against the config by ``_validate_physics``.
    """

    # HRRR joins this gate at its FIRST schema rather than a later one:
    # its bundle is new here, so there is no history of proofs written
    # before the receipt existed, and a new source has no reason to be
    # admitted on weaker evidence than the one beside it.
    if source not in {"gfs", "hrrr"} \
            or proof.get("schema") in _LEGACY_PROOF_SCHEMAS[source]:
        return None
    if proof.get("schema") != _PROOF_SCHEMA[source]:
        return None
    label = source.upper()
    selected = proof.get("physics")
    if not isinstance(selected, dict):
        raise ValueError(
            f"{label} v3 preparation proof physics receipt is missing")
    acknowledgements = selected.get("acknowledgements")
    acknowledgement_provenance = selected.get(
        "acknowledgement_provenance")
    if (not isinstance(acknowledgements, list)
            or any(not isinstance(value, str) for value in acknowledgements)):
        raise ValueError(
            f"{label} v3 preparation proof acknowledgements are malformed")
    if not isinstance(acknowledgement_provenance, dict):
        raise ValueError(
            f"{label} v3 preparation proof acknowledgement provenance is "
            "malformed")
    if selected.get("schema") == MULTI_DOMAIN_SELECTION_SCHEMA:
        expected = single_domain_physics_selection(
            cfg,
            expert_acknowledgements=tuple(acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    else:
        proof_profile = selected.get("profile")
        if not isinstance(proof_profile, str):
            raise ValueError(
                f"{label} v3 preparation proof physics receipt names no "
                "profile and is not a per-domain selection receipt")
        expected = validate_single_domain_physics_profile(
            proof_profile, config=cfg,
            expert_acknowledgements=tuple(acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    if selected != expected:
        raise ValueError(
            f"{label} v3 preparation proof physics selection differs from "
            "the hash-bound experiment/profile")
    return expected


def _validate_proof_content_sha256(proof: Mapping[str, object]) -> str:
    content = dict(proof)
    declared = content.pop("proof_content_sha256", None)
    expected = hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()
    if declared != expected:
        raise ValueError("mapped preparation proof content hash is stale")
    return expected


def _execution_plan_receipt(
        *, source_exp, executed_exp, profile: str,
        history_interval_seconds: float,
) -> dict[str, object]:
    def domain_receipt(domain) -> dict[str, object]:
        cfg = domain.run
        return {
            "grid_id": int(domain.grid_id),
            "mp_physics": int(cfg.mp_physics),
            "cu_physics": int(cfg.cu_physics),
            "bl_pbl_physics": int(cfg.bl_pbl_physics),
            "sf_sfclay_physics": int(cfg.sf_sfclay_physics),
            "sf_surface_physics": int(cfg.sf_surface_physics),
            "radiation_scheme_ids": list(radiation_scheme_ids(cfg)),
            "history_interval_seconds": float(domain.history_interval_s),
        }

    source_domains = [domain_receipt(domain) for domain in source_exp.domains]
    executed = domain_receipt(executed_exp.root)
    executed["history_interval_seconds"] = float(history_interval_seconds)
    overrides: list[dict[str, object]] = []
    if len(source_domains) != 1:
        overrides.append({
            "kind": "domain-selection",
            "source_grid_ids": [row["grid_id"] for row in source_domains],
            "executed_grid_ids": [1],
            "reason": "prepared forecast runner executes hash-bound d01 only",
        })
    source_history = float(source_exp.root.history_interval_s)
    if source_history != float(history_interval_seconds):
        overrides.append({
            "kind": "output-cadence",
            "source_experiment_seconds": source_history,
            "executed_seconds": float(history_interval_seconds),
            "mechanism": "explicit-hash-bound-history-output-schedule-v1",
            "model_state_or_physics_changed": False,
        })
    return {
        "schema": "gpuwm-prepared-d01-execution-plan-v1",
        "profile": profile,
        "source_experiment": {
            "domain_count": len(source_domains),
            "run_seconds": float(source_exp.run_seconds),
            "domains": source_domains,
        },
        "executed_d01": {
            "run_seconds": float(executed_exp.run_seconds),
            "domain": executed,
        },
        "physics_overrides": [],
        "execution_overrides": overrides,
    }


def _validate_execution_file_receipt(
        receipt, actual: Path, label: str,
) -> None:
    if not isinstance(receipt, dict) or set(receipt) != {
            "path", "bytes", "sha256"}:
        raise ValueError(f"mapped {label} execution receipt is malformed")
    path_value = receipt.get("path")
    if (not isinstance(path_value, str)
            or Path(path_value).name != actual.name
            or receipt.get("bytes") != actual.stat().st_size
            or receipt.get("sha256") != _sha256(actual)):
        raise ValueError(
            f"mapped {label} execution receipt differs from supplied file")


def _validate_twentycrv3_mapped_evidence(
        *, prepared_root: Path, proof: Mapping[str, object],
        manifest: Mapping[str, object], manifest_sha256: str,
        experiment_config: Path, wps_namelist: Path,
) -> tuple[Mapping[str, Path], Mapping[str, object], str]:
    """Bind a mapped proof to the exact packaged 20CRv3 source profile."""

    direct_proof_keys = {
        "schema", "status", "forcing_times", "forcing_hours",
        "boundary_interval_seconds", "execution_inputs",
        "source_composition", "preprocessing", "static", "geometry",
        "prepared_cache", "export", "timing_seconds", "proof_content_sha256",
    }
    hierarchy_proof_keys = {
        "schema", "status", "domain_count", "forcing_times", "forcing_hours",
        "boundary_interval_seconds", "target_contract", "execution_inputs",
        "source_composition", "preprocessing", "hierarchy_workers",
        "root_static", "root_geometry", "static_catalog", "source_coverage",
        "artifact_receipt", "wrf_manifest", "timing_seconds",
        "proof_content_sha256",
    }
    schema = proof.get("schema")
    expected_proof_keys = (
        direct_proof_keys if schema == _PROOF_SCHEMA["20crv3"]
        else hierarchy_proof_keys)
    if (schema not in {
            _PROOF_SCHEMA["20crv3"], _HIERARCHY_PROOF_SCHEMA["20crv3"]}
            or set(proof) != expected_proof_keys):
        raise ValueError("mapped 20CRv3 proof top-level inventory differs")
    if schema == _HIERARCHY_PROOF_SCHEMA["20crv3"] \
            and not isinstance(proof.get("target_contract"), dict):
        raise ValueError("mapped 20CRv3 hierarchy target contract is missing")
    _validate_proof_content_sha256(proof)
    evidence_root = _require_directory(
        prepared_root / "source-evidence", "mapped source evidence")
    mapping_path = _require_file(
        evidence_root / "mapping.json", "mapped source mapping evidence")
    composition_path = _require_file(
        evidence_root / "composition.json",
        "mapped source composition evidence")
    copied_manifest = _require_file(
        evidence_root / "input-manifest.json",
        "mapped source manifest evidence")
    if copied_manifest != (prepared_root / "source-evidence" /
                           "input-manifest.json").resolve():
        raise RuntimeError("mapped source manifest resolved unexpectedly")
    expected_authority_sha256 = dict(twentycrv3_authority_sha256())
    if (_sha256(mapping_path) != expected_authority_sha256["mapping"]
            or _sha256(composition_path)
            != expected_authority_sha256["composition"]):
        raise ValueError(
            "mapped preparation does not use the packaged 20CRv3 authorities")
    if _sha256(copied_manifest) != manifest_sha256:
        raise ValueError("mapped source manifest evidence differs from caller pin")

    known_names = {"mapping.json", "composition.json", "input-manifest.json"}
    provenance_candidates = [
        path.resolve() for path in evidence_root.iterdir()
        if path.is_file() and path.name not in known_names
    ]
    if len(provenance_candidates) != 1:
        raise ValueError(
            "mapped 20CRv3 source evidence must contain one provenance file")
    provenance_path = provenance_candidates[0]
    if (_sha256(provenance_path)
            != expected_authority_sha256["provenance"]):
        raise ValueError("mapped 20CRv3 provenance authority differs")

    receipt = proof.get("source_composition")
    expected_receipt_keys = {
        "schema", "status", "mapping", "composition", "input_manifest",
        "decoders", "terrain_products", "terrain_provenance", "alignment",
        "soil_layers", "frame_count", "valid_times", "frames",
        "receipt_content_sha256",
    }
    if (not isinstance(receipt, dict) or set(receipt) != expected_receipt_keys
            or receipt.get("schema") != "gpuwm-mapped-composition-receipt-v1"
            or receipt.get("status")
            != "CANONICAL_FRAMES_COMPLETE_NOT_STOCK_WRF_CERTIFIED"):
        raise ValueError("mapped 20CRv3 composition receipt is malformed")
    receipt_content = dict(receipt)
    declared_receipt_sha256 = receipt_content.pop(
        "receipt_content_sha256", None)
    expected_receipt_sha256 = hashlib.sha256(
        _canonical(receipt_content).encode("utf-8")).hexdigest()
    if declared_receipt_sha256 != expected_receipt_sha256:
        raise ValueError("mapped 20CRv3 composition receipt hash is stale")

    def evidence_record(record, expected_sha256: str, label: str) -> None:
        if (not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or record.get("sha256") != expected_sha256
                or not isinstance(record.get("path"), str)):
            raise ValueError(f"mapped 20CRv3 {label} receipt differs")

    evidence_record(
        receipt.get("mapping"), expected_authority_sha256["mapping"],
        "mapping")
    evidence_record(
        receipt.get("composition"), expected_authority_sha256["composition"],
        "composition")
    evidence_record(receipt.get("input_manifest"), manifest_sha256, "manifest")
    provenance = receipt.get("terrain_provenance")
    if (not isinstance(provenance, dict)
            or set(provenance) != {"provenance_path", "provenance_sha256"}
            or provenance.get("provenance_sha256")
            != expected_authority_sha256["provenance"]
            or not isinstance(provenance.get("provenance_path"), str)):
        raise ValueError("mapped 20CRv3 terrain provenance receipt differs")

    execution = proof.get("execution_inputs")
    if not isinstance(execution, dict):
        raise ValueError("mapped 20CRv3 execution input receipt is missing")
    _validate_execution_file_receipt(
        execution.get("experiment_config"), experiment_config,
        "experiment config")
    _validate_execution_file_receipt(
        execution.get("wps_namelist"), wps_namelist, "WPS namelist")
    composition_decoders = receipt.get("decoders")
    execution_decoders = execution.get("decoders")
    if (not isinstance(composition_decoders, dict)
            or not isinstance(execution_decoders, dict)
            or set(composition_decoders) != _TWENTYCRV3_DECODER_ROLES
            or set(execution_decoders) != _TWENTYCRV3_DECODER_ROLES):
        raise ValueError("mapped 20CRv3 decoder inventory differs")
    decoder_sha256: dict[str, str] = {}
    for role in sorted(_TWENTYCRV3_DECODER_ROLES):
        composed = composition_decoders[role]
        executed = execution_decoders[role]
        if (not isinstance(composed, dict)
                or set(composed) != {"path", "sha256"}
                or not isinstance(executed, dict)
                or set(executed) != {"path", "bytes", "sha256"}
                or composed.get("path") != executed.get("path")
                or composed.get("sha256") != executed.get("sha256")
                or isinstance(executed.get("bytes"), bool)
                or not isinstance(executed.get("bytes"), int)
                or executed.get("bytes") <= 0):
            raise ValueError(f"mapped 20CRv3 decoder receipt differs: {role}")
        decoder_sha256[role] = _require_digest(
            composed.get("sha256"), f"mapped 20CRv3 decoder {role} sha256")

    valid_times = manifest.get("valid_times")
    rows = manifest.get("files")
    surface_rows = [row for row in rows if row["role"] == "sfc"]
    expected_terrain = [{
        "path": row["path"],
        "sha256": row["sha256"],
    } for row in surface_rows]
    if receipt.get("terrain_products") != expected_terrain:
        raise ValueError("mapped 20CRv3 terrain product receipt differs")
    alignment = receipt.get("alignment")
    expected_alignment_keys = {
        "strategy", "terrain_external_supplement", "member",
        "member_identity", "surface_file_count", "valid_time_count",
        "canonical_receipt_content_sha256", "coordinate_match",
        "terrain_invariant_across_all_times",
    }
    expected_alignment = {
        "strategy": "20crv3_in_band_surface_same_grid",
        "terrain_external_supplement": False,
        "member": manifest["member"],
        "member_identity": _TWENTYCRV3_MEMBER_IDENTITY,
        "surface_file_count": len(surface_rows),
        "valid_time_count": len(valid_times),
        "coordinate_match": "same_decoded_grib2_grid_fingerprint",
        "terrain_invariant_across_all_times": True,
    }
    if (not isinstance(alignment, dict) or set(alignment) != expected_alignment_keys
            or any(alignment.get(key) != value
                   for key, value in expected_alignment.items())):
        raise ValueError("mapped 20CRv3 member/alignment receipt differs")
    _require_digest(
        alignment.get("canonical_receipt_content_sha256"),
        "20CRv3 canonical frame receipt sha256")
    composition_document = _load_json_object(
        composition_path, "mapped 20CRv3 composition authority")
    if receipt.get("soil_layers") != composition_document.get("soil_layers"):
        raise ValueError("mapped 20CRv3 soil-layer receipt differs")
    frames = receipt.get("frames")
    if (receipt.get("frame_count") != len(valid_times)
            or receipt.get("valid_times") != valid_times
            or not isinstance(frames, list) or len(frames) != len(valid_times)):
        raise ValueError("mapped 20CRv3 canonical frame inventory differs")
    for index, frame in enumerate(frames):
        if (not isinstance(frame, dict)
                or set(frame) != {
                    "header_sha256", "terrain_sha256", "field_count"}
                or isinstance(frame.get("field_count"), bool)
                or not isinstance(frame.get("field_count"), int)
                or frame.get("field_count") <= 0):
            raise ValueError(
                f"mapped 20CRv3 canonical frame {index} is malformed")
        _require_digest(
            frame.get("header_sha256"),
            f"mapped 20CRv3 frame {index} header sha256")
        _require_digest(
            frame.get("terrain_sha256"),
            f"mapped 20CRv3 frame {index} terrain sha256")
    return MappingProxyType({
        "mapped_mapping": mapping_path,
        "mapped_composition": composition_path,
        "mapped_provenance": provenance_path,
    }), MappingProxyType({
        "receipt_content_sha256": expected_receipt_sha256,
        "mapping_sha256": expected_authority_sha256["mapping"],
        "composition_sha256": expected_authority_sha256["composition"],
        "decoder_sha256": decoder_sha256,
        "preprocessing": proof.get("preprocessing"),
        "target_contract": proof.get("target_contract"),
    }), str(manifest["member"])


#: What the native HRRR preparation writes into every prepared cache's
#: ``source_identity``, beyond the install provenance block whose keys
#: differ between a git checkout, a wheel and a sealed distribution.
#: These five ARE checked, because each one changes what the arrays mean.
_HRRR_IDENTITY_REQUIRED = ("source_sha256", "source_cycle",
                           "model_start_time", "source_forecast_hours",
                           "model_forcing_hours")

#: The gpuwm modules whose bytes decide how HRRR GRIB2 becomes model
#: state.  The preparation hashes exactly these into the cache identity;
#: this reader requires them to be present and to agree with the proof,
#: so a cache decoded by a different ingest cannot restore under a proof
#: that names this one.
_HRRR_DECODE_SOURCES = (
    "gpuwm/hrrr_forecast.py",
    "gpuwm/ingest/hrrr.py",
    "gpuwm/ingest/hrrr_physics.py",
    "gpuwm/ingest/hrrr_surface.py",
    "gpuwm/ingest/real.py",
    "gpuwm/ingest/soil.py",
    "gpuwm/ingest/lateral_bc.py",
    "gpuwm/ingest/prepared_cache.py",
    "gpuwm/state_serialization_contract.py",
    "tools/hrrr_single_domain_benchmark.py",
)


def _validate_hrrr_source_identity(
        identity: Mapping[str, object], proof: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind an HRRR prepared cache to the proof published beside it.

    HRRR's source identity is a different SHAPE from the portable one --
    it carries per-file digests of the ingest that decoded the GRIB2 and
    the cycle/lead vocabulary that names which HRRR run it came from,
    where the portable sources carry an adapter name and a single bridge
    digest.  Both answer the same question; only one of them can be
    checked with the portable code, so this checks the other.

    The check is a comparison against the PROOF, which the caller pinned
    by digest before this ran.  A cache whose identity says something
    the proof does not is refused -- that is the whole point of
    re-deriving the front door in every worker rather than passing a
    verdict across a process boundary.
    """

    missing = [key for key in _HRRR_IDENTITY_REQUIRED if key not in identity]
    if missing:
        raise ValueError(
            f"HRRR prepared cache source identity is incomplete: {missing}")
    digests = identity.get("source_sha256")
    if not isinstance(digests, dict):
        raise ValueError(
            "HRRR prepared cache source identity carries no decode digests")
    absent = [name for name in _HRRR_DECODE_SOURCES if name not in digests]
    if absent:
        raise ValueError(
            f"HRRR prepared cache decode identity omits {absent}")
    if any(not isinstance(value, str) or len(value) != 64
           for value in digests.values()):
        raise ValueError(
            "HRRR prepared cache decode digests are malformed")
    if identity.get("source_sha256") != proof.get("source_sha256"):
        raise ValueError(
            "HRRR prepared cache decode identity differs from its proof: "
            "the cache was built by a different ingest than the proof names")
    for key in ("source_cycle", "model_start_time", "source_forecast_hours",
                "model_forcing_hours"):
        if identity.get(key) != proof.get(key):
            raise ValueError(
                f"HRRR prepared cache source identity {key} differs from "
                "the preparation proof")
    return identity


def _validate_source_identity(
        source: str, identity: object, manifest_sha256: str,
        manifest_files: Mapping[str, Mapping[str, object]], proof,
        *, layout: str, mapped_authority: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    if not isinstance(identity, dict):
        raise ValueError("prepared cache source identity must be an object")
    if source == "20crv3":
        if mapped_authority is None:
            raise ValueError("20CRv3 mapped source authority is missing")
        expected = {
            "adapter": _SOURCE_ADAPTER[source],
            "mapping_sha256": mapped_authority["mapping_sha256"],
            "composition_sha256": mapped_authority["composition_sha256"],
            "input_manifest_sha256": manifest_sha256,
            "composition_receipt_sha256": mapped_authority[
                "receipt_content_sha256"],
            "preprocessing": mapped_authority["preprocessing"],
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "prepared cache source identity differs from the exact-member "
                "20CRv3 mapped authorities")
        allowed = set(expected)
        if layout == "mapped-hierarchy-d01-v1":
            allowed.update({
                "target_contract", "nested_source_orography",
                "hierarchy_implementation_sha256", "grid_id"})
            if (identity.get("target_contract")
                    != mapped_authority.get("target_contract")
                    or identity.get("grid_id") != 1
                    or not isinstance(identity.get("nested_source_orography"), dict)
                    or not isinstance(
                        identity.get("hierarchy_implementation_sha256"), dict)):
                raise ValueError(
                    "20CRv3 hierarchy cache source identity is incomplete")
        if set(identity) != allowed:
            raise ValueError(
                "20CRv3 prepared cache source identity has an unsupported shape")
        return identity
    if source == "hrrr":
        return _validate_hrrr_source_identity(identity, proof)
    required = {"adapter", "input_manifest_schema", "input_manifest_sha256",
                "decoder", "preprocessing"}
    if not required <= set(identity):
        raise ValueError("prepared cache source identity is incomplete")
    if (identity.get("adapter") != _SOURCE_ADAPTER[source]
            or identity.get("input_manifest_schema") != _SOURCE_SCHEMA[source]
            or identity.get("input_manifest_sha256") != manifest_sha256):
        raise ValueError(
            "prepared cache source identity differs from the selected adapter "
            "or portable manifest")
    decoder = identity.get("decoder")
    expected_decoder = {
        "name": manifest_files["bridge"]["name"],
        "sha256": manifest_files["bridge"]["sha256"],
        "implementation": _DECODER_IMPLEMENTATION[source],
    }
    if decoder != expected_decoder or proof.get("decoder_sha256") \
            != expected_decoder["sha256"]:
        raise ValueError(
            "prepared cache decoder identity differs from proof/manifest")
    if identity.get("preprocessing") != proof.get("preprocessing"):
        raise ValueError(
            "prepared cache preprocessing identity differs from the proof")
    if source == "gfs":
        if (identity.get("implementation_sha256")
                != proof.get("implementation_sha256")
                or identity.get("git_source_identity")
                != proof.get("git_source_identity")):
            raise ValueError(
                "GFS implementation/source identity differs from its proof")
    return identity


def _validate_artifact_record(
        record, *, expected_path: str, actual: Path, label: str,
) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} artifact record is malformed")
    expected = {
        "path": expected_path,
        "bytes": actual.stat().st_size,
        "sha256": _sha256(actual),
    }
    if record != expected:
        raise ValueError(f"{label} artifact record differs from the supplied file")


def _validate_cache_metadata(
        reader: PreparedCacheReader, *, source: str, exp, forcing_hours,
        boundary_interval_seconds: int, proof, layout: str,
) -> None:
    metadata = reader.header.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("prepared cache metadata must be an object")
    user = metadata.get("user")
    expected_user = {
        "initial_valid_time": exp.start_time.isoformat(),
        "last_valid_time": (
            exp.start_time + timedelta(hours=forcing_hours[-1])).isoformat(),
        "forcing_hours": list(forcing_hours),
    }
    if layout == HRRR_DIRECT_LAYOUT:
        # The native preparation's own user metadata, which names the
        # HRRR cycle and both lead vocabularies rather than an adapter
        # string and a preprocessing receipt.  Compared exactly, like
        # every other layout's -- the shape differs, the strictness does
        # not.  ``mapping_reports`` is a per-field decode report whose
        # contents are the preparer's, so its PRESENCE is required and
        # its value is not re-derived here.
        expected_user.update({
            "source_cycle": proof.get("source_cycle"),
            "source_forecast_hours": proof.get("source_forecast_hours"),
            "model_forcing_hours": list(forcing_hours),
        })
        reports = user.get("mapping_reports") if isinstance(user, dict) else None
        if not isinstance(reports, dict):
            raise ValueError(
                "HRRR prepared cache carries no field mapping reports")
        expected_user["mapping_reports"] = reports
    elif layout == "portable-single-domain-v2":
        expected_user.update({
            "source_adapter": source,
            "boundary_interval_seconds": boundary_interval_seconds,
            "preprocessing": proof.get("preprocessing"),
        })
    elif layout == "mapped-direct-d01-v1":
        expected_user.update({
            "source_adapter": "mapped",
            "boundary_interval_seconds": boundary_interval_seconds,
            "composition_receipt_sha256": proof.get(
                "source_composition", {}).get("receipt_content_sha256"),
        })
    elif layout == "mapped-hierarchy-d01-v1":
        expected_user.update({
            "composition_receipt_sha256": proof.get(
                "source_composition", {}).get("receipt_content_sha256"),
            "mapped_target_contract": proof.get("target_contract"),
        })
    if user != expected_user:
        raise ValueError(
            "prepared cache user metadata differs from source/proof/experiment")
    if set(metadata.get("surface_fields", ())) != _CANONICAL_SURFACE_FIELDS:
        raise ValueError(
            "prepared cache lacks the exact source-neutral Noah surface inventory")
    if not _REQUIRED_MET_FIELDS <= set(metadata.get("met_fields", ())):
        raise ValueError("prepared cache lacks required near-surface physics fields")
    lbc = metadata.get("lbc")
    if not isinstance(lbc, dict):
        raise ValueError("prepared cache has no standalone external LBC inventory")
    cfg = exp.root.run
    if ({key: lbc.get(key) for key in (
            "spec_bdy_width", "spec_zone", "relax_zone")} != {
                "spec_bdy_width": int(cfg.spec_bdy_width),
                "spec_zone": int(cfg.spec_zone),
                "relax_zone": int(cfg.relax_zone),
            }):
        raise ValueError("prepared cache LBC width differs from experiment")
    intervals = lbc.get("intervals")
    if not isinstance(intervals, list) or len(intervals) != len(forcing_hours) - 1:
        raise ValueError("prepared cache LBC interval count differs from forcing")
    for index, interval in enumerate(intervals):
        expected = {
            "start_seconds": float(forcing_hours[index] * 3600),
            "end_seconds": float(forcing_hours[index + 1] * 3600),
            "fields": list(_LBC_FIELDS),
        }
        if interval != expected:
            raise ValueError(
                f"prepared cache LBC interval {index} differs from forcing")


def _validate_hierarchy_d01_artifacts(
        layout: _PreparedLayout, *, static: Mapping[str, object],
        geometry_receipt: Mapping[str, object], reader: PreparedCacheReader,
        static_sha256: str, geometry_sha256: str,
) -> None:
    receipt = layout.domain_receipt
    if receipt is None:
        raise ValueError("hierarchy d01 receipt is missing")
    expected_artifacts = {
        "prepared_cache": {
            "path": "prepared-cache",
            "content_sha256": reader.content_sha256,
            "payload_bytes": reader.payload_bytes,
            "array_count": len(reader.arrays),
        },
        "static_cache": {
            "path": "native-static.npz",
            "bytes": layout.static_path.stat().st_size,
            "sha256": static_sha256,
            "fields": sorted(static),
        },
        "geometry_receipt": {
            "path": "geometry-receipt.json",
            "sha256": geometry_sha256,
            "geometry": geometry_receipt.get("geometry"),
        },
    }
    expected_verification = {
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "PASS",
        "path": "prepared-cache",
        "content_sha256": reader.content_sha256,
        "array_count": len(reader.arrays),
        "payload_bytes": reader.payload_bytes,
    }
    if (receipt.get("artifacts") != expected_artifacts
            or receipt.get("verification") != expected_verification):
        raise ValueError(
            "hierarchy d01 artifact receipt differs from its verified files")


def _validate_twentycrv3_static_proof(
        proof: Mapping[str, object], layout: _PreparedLayout,
        *, static: Mapping[str, object],
        geometry_receipt: Mapping[str, object], static_sha256: str,
) -> None:
    if layout.kind == "mapped-direct-d01-v1":
        expected_static = {
            "path": "native-static.npz",
            "bytes": layout.static_path.stat().st_size,
            "sha256": static_sha256,
            "fields": sorted(static),
        }
        if (proof.get("static") != expected_static
                or proof.get("geometry") != geometry_receipt):
            raise ValueError(
                "mapped 20CRv3 direct static/geometry proof differs")
        return
    root_static = layout.authority_paths.get("mapped_root_static")
    root_geometry_path = layout.authority_paths.get("mapped_root_geometry")
    if root_static is None or root_geometry_path is None:
        raise ValueError("mapped 20CRv3 hierarchy root authorities are missing")
    try:
        with np.load(root_static, allow_pickle=False) as archive:
            root_fields = sorted(archive.files)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "mapped 20CRv3 hierarchy root static cache is unreadable") from exc
    expected_root_static = {
        "path": "native-static.npz",
        "bytes": root_static.stat().st_size,
        "sha256": _sha256(root_static),
        "fields": root_fields,
    }
    root_geometry = _load_json_object(
        root_geometry_path, "mapped hierarchy root geometry receipt")
    if (proof.get("root_static") != expected_root_static
            or proof.get("root_geometry") != root_geometry):
        raise ValueError(
            "mapped 20CRv3 hierarchy root static/geometry proof differs")


def _validate_hierarchy_wrf_authority(
        layout: _PreparedLayout, *, source: str, source_exp,
        forcing_hours: tuple[int, ...], boundary_interval_seconds: int,
        source_manifest_sha256: str, decoder_sha256: object,
        preprocessing, contract_sha256: str, prepared_content_sha256: str,
        prepared_header_sha256: str, static_sha256: str,
        geometry_sha256: str,
        mapped_authority: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    wrf = layout.wrf_manifest
    if wrf is None or layout.artifact_manifest is None:
        raise ValueError("hierarchy WRF/artifact authority is missing")
    if (wrf.get("schema") != "gpuwm-native-direct-wrf-hierarchy-export-v1"
            or wrf.get("status") != "READY"
            or wrf.get("valid_time")
            != source_exp.start_time.strftime("%Y-%m-%d_%H:%M:%S")
            or wrf.get("forcing_hours") != list(forcing_hours)
            or wrf.get("boundary_interval_seconds")
            != boundary_interval_seconds
            or wrf.get("hierarchy") != _hierarchy_rows(source_exp)):
        raise ValueError(
            "hierarchy direct-WRF manifest differs from source experiment")
    source_receipt = wrf.get("source")
    if not isinstance(source_receipt, dict) \
            or source_receipt.get("contract_sha256") != contract_sha256:
        raise ValueError("hierarchy direct-WRF contract identity differs")
    domain_sources = source_receipt.get("domains")
    expected_domain_keys = {
        f"d{int(domain.grid_id):02d}" for domain in source_exp.domains}
    if not isinstance(domain_sources, dict) \
            or set(domain_sources) != expected_domain_keys:
        raise ValueError("hierarchy direct-WRF domain source inventory differs")
    expected_d01 = {
        "prepared_header_sha256": prepared_header_sha256,
        "prepared_content_sha256": prepared_content_sha256,
        "static_cache_sha256": static_sha256,
        "geometry_receipt_sha256": geometry_sha256,
        "mp_physics": int(source_exp.root.run.mp_physics),
        "microphysics": stock_wrf_physics_inventory(
            int(source_exp.root.run.mp_physics)).scheme,
        "resolved_physics_contract_sha256": (
            _resolved_wrf_direct_contract_sha256(
                source_exp.root.run.mp_physics)),
    }
    if domain_sources.get("d01") != expected_d01:
        raise ValueError(
            "hierarchy direct-WRF d01 hashes differ from preparation")
    provenance = source_receipt.get("input_provenance")
    expected_provenance: dict[str, object] = {
        "input_manifest_sha256": source_manifest_sha256,
        "decoder_sha256": decoder_sha256,
        "preprocessing": preprocessing,
        "regular_source_adapter": (
            "rw-wps-mapped" if source == "20crv3" else source),
        "native_artifact_manifest": (
            "../hierarchy-artifacts/domain-artifacts.json"),
        "native_artifact_manifest_sha256": _sha256(
            layout.authority_paths["hierarchy_artifact_manifest"]),
    }
    if source == "20crv3":
        if mapped_authority is None:
            raise ValueError("20CRv3 hierarchy mapped authority is missing")
        expected_provenance.update({
            "mapping_sha256": mapped_authority["mapping_sha256"],
            "composition_sha256": mapped_authority["composition_sha256"],
            "mapped_target_contract": mapped_authority["target_contract"],
        })
    if not isinstance(provenance, dict) or any(
            provenance.get(key) != value
            for key, value in expected_provenance.items()):
        raise ValueError(
            "hierarchy direct-WRF input provenance differs from authorities")
    return MappingProxyType(expected_d01)


def _resolve_cache_identity_compatibility(
        *, source: str, observed: Mapping[str, object],
        expected: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if observed == expected:
        return expected, MappingProxyType({
            "schema": "gpuwm-prepared-cache-identity-compatibility-v1",
            "status": "EXACT",
            "compatibility_overrides": [],
        })
    compatible = json.loads(_canonical(expected))
    run = compatible.get("domain_config", {}).get("run", {})
    removed = run.pop("nest_microphysics_transition", None)
    if (source == "20crv3" and removed == "same-scheme-only"
            and observed == compatible):
        return observed, MappingProxyType({
            "schema": "gpuwm-prepared-cache-identity-compatibility-v1",
            "status": "COMPATIBLE_LEGACY_DEFAULT",
            "compatibility_overrides": [{
                "field": "domain_config.run.nest_microphysics_transition",
                "prepared_identity": "field-absent",
                "current_loader_default": "same-scheme-only",
                "model_state_or_physics_changed": False,
                "reason": (
                    "the hash-bound legacy config predates this explicit "
                    "same-scheme default"),
            }],
        })
    raise ValueError(
        "prepared cache identity differs from the requested source, static "
        "data, configuration, namelist, or allowed legacy default")


def preflight_prepared_forecast(
        *, source: str, prepared_root: Path, proof_sha256: str,
        source_manifest_sha256: str, prepared_content_sha256: str,
        experiment_config: Path, wps_namelist: Path,
        physics_profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        run_seconds: float, history_interval_seconds: float,
        domain_bundle: Path | None = None,
) -> PreparedForecastInputs:
    """Validate every portable preparation authority without importing CuPy."""

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported prepared forecast source {source!r}")
    proof_sha256 = _require_digest(proof_sha256, "proof-sha256")
    source_manifest_sha256 = _require_digest(
        source_manifest_sha256, "source-manifest-sha256")
    prepared_content_sha256 = _require_digest(
        prepared_content_sha256, "prepared-content-sha256")
    prepared_root = _require_directory(prepared_root, "prepared root")
    proof_path = _require_file(prepared_root / "proof.json", "preparation proof")
    source_manifest_path = _require_file(
        prepared_root / (
            "source-evidence/input-manifest.json"
            if source == "20crv3" else "source-input-manifest.json"),
        "portable source manifest")
    experiment_config = _require_file(experiment_config, "experiment config")
    wps_namelist = _require_file(wps_namelist, "WPS namelist")

    if _sha256(proof_path) != proof_sha256:
        raise ValueError("preparation proof SHA differs from --proof-sha256")
    if _sha256(source_manifest_path) != source_manifest_sha256:
        raise ValueError(
            "portable source manifest SHA differs from "
            "--source-manifest-sha256")
    proof = _load_json_object(proof_path, "preparation proof")
    manifest = _load_json_object(source_manifest_path, "portable source manifest")
    source_exp = load_experiment(experiment_config)
    from gpuwm.experiment import (
        refuse_unrouted_perturbation, refuse_unrouted_spawn,
    )
    refuse_unrouted_perturbation(
        source_exp, "prepared single-domain forecast")
    refuse_unrouted_spawn(source_exp, "prepared single-domain forecast")
    layout = _resolve_prepared_layout(
        source=source, prepared_root=prepared_root, proof=proof,
        source_exp=source_exp, domain_bundle=domain_bundle)
    static_path = layout.static_path
    geometry_receipt_path = layout.geometry_receipt_path
    prepared_cache_path = layout.prepared_cache_path
    cache_header_path = _require_file(
        prepared_cache_path / "header.json", "prepared cache header")
    geometry_receipt = _load_json_object(
        geometry_receipt_path, "geometry receipt")
    header = _load_json_object(cache_header_path, "prepared cache header")

    if (header.get("schema") != PREPARED_CACHE_SCHEMA
            or header.get("status") != "READY"
            or header.get("content_sha256") != prepared_content_sha256):
        raise ValueError("prepared cache header identity/status/content differs")

    exp = source_exp if len(source_exp.domains) == 1 else replace(
        source_exp, domains=(source_exp.root,))
    physics_receipt = _validate_physics(
        exp, physics_profile, run_seconds, history_interval_seconds,
        source=source, expert_acknowledgements=expert_acknowledgements)
    front_door_physics = _validate_front_door_physics_proof(
        proof, source=source, profile=physics_profile, cfg=exp.root.run)
    physics_receipt["execution_plan"] = _execution_plan_receipt(
        source_exp=source_exp, executed_exp=exp, profile=physics_profile,
        history_interval_seconds=history_interval_seconds)
    manifest_files, source_manifest_receipt = _manifest_file_specs(
        source, manifest, source_exp, proof)
    mapped_paths: Mapping[str, Path] = MappingProxyType({})
    mapped_authority: Mapping[str, object] | None = None
    source_member: str | None = None
    if source == "20crv3":
        mapped_paths, mapped_authority, source_member = (
            _validate_twentycrv3_mapped_evidence(
                prepared_root=prepared_root, proof=proof, manifest=manifest,
                manifest_sha256=source_manifest_sha256,
                experiment_config=experiment_config,
                wps_namelist=wps_namelist))
    else:
        for role, actual in (
                ("experiment_config", experiment_config),
                ("wps_namelist", wps_namelist)):
            spec = manifest_files[role]
            if spec["name"] != actual.name or spec["sha256"] != _sha256(actual):
                raise ValueError(
                    f"supplied {role} differs from the portable source manifest")

    forcing_hours = proof.get("forcing_hours")
    if (not isinstance(forcing_hours, list) or len(forcing_hours) < 2
            or any(isinstance(hour, bool) or not isinstance(hour, int)
                   for hour in forcing_hours)
            or forcing_hours[0] != 0):
        raise ValueError("preparation proof forcing hours are malformed")
    forcing_hours = tuple(forcing_hours)
    deltas = {
        later - earlier
        for earlier, later in zip(forcing_hours, forcing_hours[1:])
    }
    if len(deltas) != 1 or next(iter(deltas)) <= 0:
        raise ValueError("preparation proof forcing cadence is not uniform")
    cadence_hours = next(iter(deltas))
    if source == "hrrr" and cadence_hours != 1:
        # Not a warning like the others: HRRR publishes hourly and the
        # native preparer's whole forcing contract
        # (gpuwm.hrrr_forecast) is built on contiguous hourly leads, so
        # a non-hourly HRRR cadence is a malformed preparation rather
        # than a coarser one.
        raise ValueError(
            f"HRRR forcing cadence of {cadence_hours} h is not hourly; the "
            "native HRRR route prepares contiguous hourly leads")
    if (source == "gfs" and cadence_hours not in {1, 3}) \
            or (source in {"era5", "20crv3"} and cadence_hours < 1):
        # The cadence is uniform (checked above) and the coverage is
        # checked below; a cadence outside the blessed set is coarser
        # boundary forcing, not a broken preparation.
        warn(f"{source.upper()} forcing cadence of {cadence_hours} h is "
             "outside the set this runner has demonstrated "
             "(1 h/3 h for GFS); continuing with it as prepared")
    boundary_interval_seconds = cadence_hours * 3600
    if (proof.get("boundary_interval_seconds") != boundary_interval_seconds
            or forcing_hours[-1] * 3600 < run_seconds):
        raise ValueError("prepared forcing cadence/coverage differs from the run")
    expected_times = [
        (exp.start_time + timedelta(hours=hour)).isoformat()
        for hour in forcing_hours
    ]
    if proof.get("forcing_times") != expected_times:
        raise ValueError(
            "prepared forcing valid times differ from the experiment start")
    if source == "gfs":
        manifest_hours = sorted(
            int(role.removeprefix("grib-f"))
            for role in manifest_files if role.startswith("grib-f"))
        # Two vocabularies, bound to each other rather than assumed
        # equal: the manifest names absolute NOAA leads, the proof's
        # forcing hours are model offsets from start_time, and the
        # declared lead is what maps one onto the other.  At lead 0 this
        # is the identity comparison it has always been.
        lead_hours = proof_initial_forecast_lead(proof)
        expected_source_hours = [lead_hours + hour for hour in forcing_hours]
        declared_source_hours = proof.get("source_forecast_hours")
        if declared_source_hours is None:
            declared_source_hours = expected_source_hours
        if declared_source_hours != expected_source_hours:
            raise ValueError(
                "preparation proof source forecast hours are not its "
                "forcing offsets shifted by the declared initial lead")
        if not set(expected_source_hours) <= set(manifest_hours):
            raise ValueError(
                "GFS manifest GRIB hours do not carry the proof's forcing "
                f"window f{expected_source_hours[0]:03d}.."
                f"f{expected_source_hours[-1]:03d}")
        unused = sorted(set(manifest_hours) - set(expected_source_hours))
        if unused:
            # Bound, decoded, and honestly not used: a manifest authored
            # over a whole fetch while the run starts partway into it.
            warn("the GFS source manifest binds forecast hour(s) "
                 + ", ".join(f"f{hour:03d}" for hour in unused)
                 + " that this run's forcing window does not use")
    elif source == "hrrr":
        # Two vocabularies again, and on this route they are ALWAYS
        # different: NOAA's absolute leads (f004..f010 of one cycle) and
        # the model-relative forcing offsets (0..6) the cache and the
        # exporter use.  gpuwm.hrrr_forecast owns the window contract;
        # this binds the proof's two lists to each other so a bundle
        # cannot claim one window and carry another.
        from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours

        declared = proof.get("source_forecast_hours")
        cycle_text = proof.get("source_cycle")
        if not isinstance(declared, list) or not isinstance(cycle_text, str):
            raise ValueError(
                "HRRR preparation proof names no source cycle/lead window")
        try:
            cycle = datetime.fromisoformat(cycle_text)
            source_hours = validate_hrrr_source_forecast_hours(
                declared, cycle=cycle)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"HRRR preparation proof source window is invalid: {error}"
            ) from None
        if len(source_hours) != len(forcing_hours) \
                or list(range(len(source_hours))) != list(forcing_hours):
            raise ValueError(
                "HRRR preparation proof forcing offsets are not 0..N-1 of "
                "its own source lead window")
        if (cycle + timedelta(hours=source_hours[0])) != exp.start_time:
            raise ValueError(
                f"HRRR preparation proof cycle {cycle_text} at "
                f"f{source_hours[0]:03d} is not the experiment start "
                f"{exp.start_time.isoformat()}")
        if proof.get("model_start_time") != exp.start_time.isoformat() \
                or proof.get("model_forcing_hours") != list(forcing_hours):
            raise ValueError(
                "HRRR preparation proof model start/forcing differ from "
                "the hash-bound experiment")
    elif source == "20crv3":
        if (manifest.get("valid_times") != expected_times
                or manifest.get("cadence_seconds")
                != boundary_interval_seconds):
            raise ValueError(
                "20CRv3 exact-member manifest cadence/coverage differs from "
                "proof forcing")

    if source != "20crv3" \
            and proof.get("input_manifest_sha256") != source_manifest_sha256:
        raise ValueError("preparation proof input manifest hash differs")
    if layout.kind in {"portable-single-domain-v2", HRRR_DIRECT_LAYOUT}:
        expected_source_inputs = {
            "manifest_schema": _SOURCE_SCHEMA[source],
            "manifest_sha256": source_manifest_sha256,
            "files": manifest.get("files"),
        }
        if proof.get("source_inputs") != expected_source_inputs:
            raise ValueError(
                "preparation proof source inputs differ from manifest")
        grid = validate_native_lambert_contract(
            exp, wps_namelist, source_name=source.upper())
    elif layout.kind in _HIERARCHY_LAYOUTS:
        grid = validate_native_lambert_contracts(
            source_exp, wps_namelist, source_name=source.upper())[0]
    else:
        grid = validate_native_lambert_contract(
            exp, wps_namelist, source_name=source.upper())
    verify_native_static_receipt(
        geometry_receipt_path, static_path, grid, exp.root.run)
    static = load_native_static_cache(
        static_path, grid, exp.root.run.ny, exp.root.run.nx)
    static_sha256 = _sha256(static_path)
    geometry_sha256 = _sha256(geometry_receipt_path)
    header_sha256 = _sha256(cache_header_path)

    header_identity = header.get("identity")
    if not isinstance(header_identity, dict):
        raise ValueError("prepared cache identity must be an object")
    source_identity = _validate_source_identity(
        source, header_identity.get("source_identity"),
        source_manifest_sha256, manifest_files, proof, layout=layout.kind,
        mapped_authority=mapped_authority)
    if layout.kind in _HIERARCHY_LAYOUTS \
            and source_identity.get("grid_id") != 1:
        raise ValueError("hierarchy prepared-cache source identity is not d01")
    if layout.kind == HRRR_DIRECT_LAYOUT:
        # Three of this identity's inputs are genuinely different values
        # on the HRRR route, where for the portable sources two of them
        # happen to coincide.  The bridge manifest is the DECODED
        # intermediate's SHA256SUMS, the source manifest is the raw
        # GRIB2 set's, and the namelist is WRF's namelist.input (the
        # native preparer reads the explicit eta ladder out of it), not
        # the experiment TOML.  Each is taken from the portable manifest
        # -- pinned by --source-manifest-sha256 -- and each is verified
        # against the file in the bundle before it is used.
        bridge_path = layout.authority_paths["bridge_manifest"]
        bridge_digest = _sha256(
            _require_file(bridge_path, "HRRR native bridge manifest"))
        if bridge_digest != manifest_files["bridge"]["sha256"]:
            raise ValueError(
                "HRRR native bridge manifest differs from the portable "
                "source manifest")
        namelist_path = _require_file(
            prepared_root / manifest_files["namelist_input"]["name"],
            "HRRR WRF namelist.input")
        if _sha256(namelist_path) != manifest_files["namelist_input"]["sha256"]:
            raise ValueError(
                "bundled HRRR namelist.input differs from the portable "
                "source manifest")
        expected_identity = prepared_cache_identity(
            bridge_manifest_sha256=bridge_digest,
            source_manifest_sha256=manifest_files["source_manifest"]["sha256"],
            static_cache_sha256=static_sha256,
            namelist_sha256=manifest_files["namelist_input"]["sha256"],
            domain_config=exp.root,
            forcing_hours=forcing_hours,
            source_identity=source_identity,
            namelist_extension_invariant=proof.get(
                "namelist_extension_invariant"),
        )
    else:
        expected_identity = prepared_cache_identity(
            bridge_manifest_sha256=source_manifest_sha256,
            source_manifest_sha256=source_manifest_sha256,
            static_cache_sha256=static_sha256,
            namelist_sha256=_sha256(experiment_config),
            domain_config=exp.root,
            forcing_hours=forcing_hours,
            source_identity=source_identity,
        )
    cache_identity, cache_identity_compatibility = (
        _resolve_cache_identity_compatibility(
            source=source, observed=header_identity,
            expected=expected_identity))
    reader = PreparedCacheReader(
        prepared_cache_path, expected_identity=cache_identity)
    verified_cache = reader.verify_all()
    if verified_cache["content_sha256"] != prepared_content_sha256:
        raise ValueError("verified prepared cache differs from the caller pin")
    _validate_cache_metadata(
        reader, source=source, exp=exp, forcing_hours=forcing_hours,
        boundary_interval_seconds=boundary_interval_seconds, proof=proof,
        layout=layout.kind)
    if source == "20crv3":
        _validate_twentycrv3_static_proof(
            proof, layout, static=static,
            geometry_receipt=geometry_receipt,
            static_sha256=static_sha256)

    contract_path = _require_file(
        REPO / "gpuwm" / "wrf_direct_v461_contract.json",
        "direct-WRF contract")
    if layout.kind in _DIRECT_LAYOUTS:
        # An HRRR bundle keeps its artifacts where the certified native
        # preparation has always written them, so the paths its proof
        # declares are that layout's, not the portable one's.  The
        # DIGESTS compared against them are identical either way -- the
        # relative path is a label, the sha256 is the binding.
        bundle_relative = (
            dict(HRRR_BUNDLE_PATHS) if layout.kind == HRRR_DIRECT_LAYOUT
            else {"static": "native-static.npz",
                  "geometry_receipt": "geometry-receipt.json",
                  "prepared_cache": "prepared-cache"})
        cache_proof = proof.get("prepared_cache")
        expected_cache_proof = {
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "BUILT",
            "path": bundle_relative["prepared_cache"],
            "content_sha256": reader.content_sha256,
            "array_count": len(reader.arrays),
            "payload_bytes": reader.payload_bytes,
        }
        if cache_proof != expected_cache_proof:
            raise ValueError(
                "proof prepared-cache receipt differs from the bundle")
        if source != "20crv3":
            artifacts = proof.get("initialization_artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError(
                    "proof initialization artifact inventory is missing")
            _validate_artifact_record(
                artifacts.get("source_manifest"),
                expected_path="source-input-manifest.json",
                actual=source_manifest_path, label="source manifest")
            _validate_artifact_record(
                artifacts.get("static_cache"),
                expected_path=bundle_relative["static"],
                actual=static_path, label="static cache")
            _validate_artifact_record(
                artifacts.get("geometry_receipt"),
                expected_path=bundle_relative["geometry_receipt"],
                actual=geometry_receipt_path, label="geometry receipt")
            expected_artifact_cache = {
                "path": bundle_relative["prepared_cache"],
                "content_sha256": reader.content_sha256,
                "payload_bytes": reader.payload_bytes,
            }
            if artifacts.get("prepared_cache") != expected_artifact_cache:
                raise ValueError(
                    "proof prepared-cache artifact differs from the bundle")

        export = proof.get("export")
        expected_dimensions = {
            "nx": int(exp.root.run.nx),
            "ny": int(exp.root.run.ny),
            "nz": int(exp.root.run.nz),
        }
        expected_export_schema = (
            "gpuwm-native-direct-wrf-export-v3"
            if source in {"gfs", "hrrr"}
            and proof.get("schema") == _PROOF_SCHEMA[source]
            else "gpuwm-native-direct-wrf-export-v2"
        )
        if (not isinstance(export, dict)
                or export.get("schema") != expected_export_schema
                or export.get("status") != "READY"
                or export.get("forcing_hours") != list(forcing_hours)
                or export.get("boundary_interval_seconds")
                != boundary_interval_seconds
                or export.get("dimensions") != expected_dimensions
                or export.get("valid_time")
                != exp.start_time.strftime("%Y-%m-%d_%H:%M:%S")):
            raise ValueError(
                "proof direct-WRF export identity differs from the run")
        if expected_export_schema.endswith("-v3") \
                and export.get("physics") != front_door_physics:
            raise ValueError(
                f"{source.upper()} v3 export physics receipt differs from "
                "the preparation proof")
        export_source = export.get("source")
        expected_export_source = {
            "contract_sha256": _sha256(contract_path),
            "geometry_receipt_sha256": geometry_sha256,
            "prepared_content_sha256": reader.content_sha256,
            "prepared_header_sha256": header_sha256,
            "resolved_physics_contract_sha256": (
                _resolved_wrf_direct_contract_sha256(
                    exp.root.run.mp_physics)),
            "static_cache_sha256": static_sha256,
        }
        if export_source != expected_export_source:
            raise ValueError(
                "proof export source hashes differ from preparation")
        if source != "20crv3":
            preprocessing_digest = hashlib.sha256(
                _canonical(proof.get("preprocessing")).encode("utf-8")
            ).hexdigest()
            if proof.get("preprocessing_receipt_sha256") \
                    != preprocessing_digest:
                raise ValueError(
                    "preparation proof preprocessing receipt hash differs")
        export_source_receipt = MappingProxyType(expected_export_source)
    else:
        _validate_hierarchy_d01_artifacts(
            layout, static=static, geometry_receipt=geometry_receipt,
            reader=reader, static_sha256=static_sha256,
            geometry_sha256=geometry_sha256)
        export_source_receipt = _validate_hierarchy_wrf_authority(
            layout, source=source, source_exp=source_exp,
            forcing_hours=forcing_hours,
            boundary_interval_seconds=boundary_interval_seconds,
            source_manifest_sha256=source_manifest_sha256,
            decoder_sha256=(
                mapped_authority["decoder_sha256"]
                if source == "20crv3" else
                manifest_files["bridge"]["sha256"]),
            preprocessing=proof.get("preprocessing"),
            contract_sha256=_sha256(contract_path),
            prepared_content_sha256=reader.content_sha256,
            prepared_header_sha256=header_sha256,
            static_sha256=static_sha256,
            geometry_sha256=geometry_sha256,
            mapped_authority=mapped_authority)

    authority_paths = MappingProxyType({
        "proof": proof_path,
        "source_manifest": source_manifest_path,
        "static": static_path,
        "geometry_receipt": geometry_receipt_path,
        "cache_header": cache_header_path,
        "experiment_config": experiment_config,
        "wps_namelist": wps_namelist,
        "wrf_direct_contract": contract_path,
        **dict(layout.authority_paths),
        **dict(mapped_paths),
        **_thompson_authority_paths(physics_receipt),
    })
    file_sha256 = MappingProxyType({
        name: _sha256(path) for name, path in authority_paths.items()
    })
    if (file_sha256["proof"] != proof_sha256
            or file_sha256["source_manifest"] != source_manifest_sha256):
        raise RuntimeError("preparation authorities changed during preflight")
    return PreparedForecastInputs(
        source=source, layout=layout.kind, prepared_root=prepared_root,
        domain_bundle_path=layout.domain_bundle,
        proof_path=proof_path, source_manifest_path=source_manifest_path,
        static_path=static_path,
        geometry_receipt_path=geometry_receipt_path,
        prepared_cache_path=prepared_cache_path,
        experiment_config=experiment_config, wps_namelist=wps_namelist,
        proof=MappingProxyType(proof),
        source_manifest=MappingProxyType(manifest),
        geometry_receipt=MappingProxyType(geometry_receipt),
        cache_identity=cache_identity,
        cache_identity_compatibility=cache_identity_compatibility,
        cache_reader=reader,
        experiment=exp, grid=grid, static=MappingProxyType(static),
        landuse_identity=MappingProxyType(dict(_LANDUSE_IDENTITY)),
        forcing_hours=forcing_hours,
        boundary_interval_seconds=boundary_interval_seconds,
        physics_receipt=MappingProxyType(physics_receipt),
        export_source_receipt=export_source_receipt,
        source_manifest_receipt=(
            None if source_manifest_receipt is None
            else MappingProxyType(dict(source_manifest_receipt))),
        file_sha256=file_sha256, authority_paths=authority_paths,
        source_domain_count=len(source_exp.domains),
        source_member=source_member,
    )


def _verify_inputs_unchanged(inputs: PreparedForecastInputs) -> None:
    _verify_thompson_runtime_environment(inputs.physics_receipt)
    current = {
        name: _sha256(path) for name, path in inputs.authority_paths.items()
    }
    if current != dict(inputs.file_sha256):
        raise RuntimeError("prepared forecast inputs changed during execution")
    verified = PreparedCacheReader(
        inputs.prepared_cache_path,
        expected_identity=inputs.cache_identity).verify_all()
    if verified["content_sha256"] != inputs.cache_reader.content_sha256:
        raise RuntimeError("prepared cache changed during execution")


def _consume_due_native_refl_10cm(state, ticks: int, consumer):
    """Consume the scheme-native field staged by an output-due MP call."""

    if (ticks != 0 and state.qv is not None
            and state.physics.mp_physics in REFL_10CM_MICROPHYSICS):
        return consumer(state)
    return None


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ModuleNotFoundError:
        if os.name != "nt":
            return 0
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.PeakWorkingSetSize)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if sys.platform == "darwin" else 1024
    return int(usage.ru_maxrss * scale)


def run_prepared_forecast(
        inputs: PreparedForecastInputs, *, output_directory: Path,
        observer=None,
) -> dict[str, object]:
    """Restore, integrate, and publish hash-bound source-neutral history.

    ``observer`` is an optional second consumer of this run's progress,
    for a caller driving the runner in-process rather than reading its
    ``progress.json`` from outside (:class:`gpuwm.runplan.RunObserver`
    is one).  It receives every per-step progress event this runner
    already builds, and -- if it carries the ``output_committed`` hook
    -- each wrfout as it becomes durable.  ``None`` leaves every byte of
    this runner's behaviour unchanged.
    """

    _verify_thompson_runtime_environment(inputs.physics_receipt)

    import cupy as cp

    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.dycore import stability_gate_failed, stability_report
    from gpuwm.core.gpu_mem_watch import (
        GpuPeakMemoryWatcher, default_cupy_probes,
    )
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.model import (
        DomainNode, ExperimentState, ModelRuntimeStatus, execute_experiment,
    )
    from gpuwm.core.refl import consume_refl_10cm
    from gpuwm.core.uh_diag import reset_up_heli_max
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    from gpuwm.state_digest import canonical_state_digest

    outdir = Path(output_directory).resolve()
    progress_path = outdir / "progress.json"
    exp = inputs.experiment
    cfg = exp.root.run
    timing = {}
    total_started = time.perf_counter()
    runtime_source_identity = _runtime_source_identity()
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": "RESTORING_PREPARED_CACHE",
        "source": inputs.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": float(exp.run_seconds),
    })

    started = time.perf_counter()
    restored = restore_prepared_cache(
        inputs.prepared_cache_path,
        expected_identity=inputs.cache_identity,
        cfg=cfg, static=inputs.static)
    timing["restore_prepared_cache"] = time.perf_counter() - started
    _validate_restored_cache_receipt(
        restored.receipt, inputs.cache_reader.content_sha256)
    if restored.surface is None:
        raise ValueError(
            "prepared cache has no source-neutral canonical surface state")
    _validate_restored_source_adapter(restored.metadata, inputs.source)

    # Physics initialization is where a first run pays its one-time
    # NVRTC compile (~100 s on a modern card), and it used to pay it
    # under the stale RESTORING status above -- the first field run of
    # the published wheel watched that silence and concluded a hang.
    # One line and one status flip, only when the kernel cache says the
    # compile is actually coming.
    compile_notice = kernel_compile_notice()
    if compile_notice is not None:
        print(compile_notice, flush=True)
        _atomic_json(progress_path, {
            "schema": PROGRESS_SCHEMA,
            "status": COMPILING_STATUS,
            "source": inputs.source,
            "model_elapsed_seconds": 0.0,
            "requested_run_seconds": float(exp.run_seconds),
        })

    started = time.perf_counter()
    driver = initialize_prepared_physics(
        restored.initial_result, cfg, restored.met, restored.surface,
        inputs.static, inputs.landuse_identity, inputs.grid, exp.start_time)
    timing["initialize_physics"] = time.perf_counter() - started

    tick_clock = resolve_clock(
        exp, lbc_interval_s=float(inputs.boundary_interval_seconds))
    schedule = build_schedule(exp, tick_clock)
    clocks = tick_clock.clocks()
    node = DomainNode(
        exp.root, inputs.grid, restored.initial_result.state, clocks[1],
        None, [], None)
    fingerprint = hashlib.sha256(_canonical({
        "schema": REPORT_SCHEMA,
        "source": inputs.source,
        "prepared_content_sha256": inputs.cache_reader.content_sha256,
        "experiment_config_sha256": inputs.file_sha256["experiment_config"],
        "wps_namelist_sha256": inputs.file_sha256["wps_namelist"],
        "runtime_source_identity": runtime_source_identity,
    }).encode("utf-8")).hexdigest()
    model = ExperimentState(
        node, MappingProxyType({1: node}), schedule, None, fingerprint)
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({
        1: SimpleNamespace(
            static_fields=inputs.static, geog_selection=None,
            initial_result=restored.initial_result),
    })

    initial_health = _strict_json(vars(
        StateHealthValidator(node.state).validate(phase="initialized.d01")))
    if not initial_health["ok"]:
        raise FloatingPointError(
            f"prepared forecast initial health failed: {initial_health}")

    history = []
    # Boundary-only sampling under-reported the peak: the executor trims
    # the CuPy pool per STEP and at period commit BEFORE the progress
    # callback fires, so samples taken only in those callbacks missed
    # the intra-step transient working set (19.41 GiB reported against
    # 22.34 GiB true on the four-domain tree shape).  The watcher polls
    # from a daemon thread as well, and the boundary/end-of-run
    # sample() calls below fold into the same maxima.
    memory_watch = GpuPeakMemoryWatcher(default_cupy_probes())

    writers = PerDomainWrfoutWriters(
        model, outdir / "wrfout", start_time=exp.start_time,
        title=(f"gpuwm {inputs.source.upper()} prepared-cache "
               f"{cfg.nx}x{cfg.ny}x{cfg.nz}"),
        # The wrfout outlives the run directory: it is what gets
        # archived, what `gpuwm downscale` reads back, and what the
        # pictures are made from.  `report.json` alone is not where a
        # forecast-lead initialization may live -- separate the artifact
        # from its run directory and that provenance is gone.
        initial_condition=inputs.proof.get("initial_condition"),
        source=inputs.source)
    model._io_manager = writers
    forecast_started = time.perf_counter()

    def history_handler(_model, current, ticks):
        report = stability_report(
            current.state, current.cfg.run,
            boundary_width=current.cfg.run.spec_bdy_width)
        sample = {
            "ticks": int(ticks),
            "elapsed_seconds": float(current.clock.elapsed_seconds),
            **report,
        }
        history.append(_strict_json(sample))
        if stability_gate_failed(
                report, max_cfl=10.0, max_w_ms=150.0):
            raise FloatingPointError(
                "prepared forecast stability threshold failed: "
                + _stability_diagnosis(sample, current.state, current.cfg.run)
                + f"; sample {sample}")
        refl = _consume_due_native_refl_10cm(
            current.state, ticks, consume_refl_10cm)
        writers.submit(current, ticks, refl_field=refl)
        # History-interval reset of the UP_HELI_MAX window
        # (module_diag_nwp.F:246-269; gpuwm's ratified placement is
        # immediately after the frame is durable).  Every other route
        # that publishes history frames does this -- runtime.py's
        # prepared-case integrator, the downscale child, and the tree
        # runner through runtime._submit_tree_history_frame -- and THIS
        # one, the runner `gpuwm go` uses, did not.  The running max
        # therefore never restarted: frame 2 onward reported the maximum
        # since model start rather than since the previous frame, so
        # every value after the first history period was wrong, and
        # wrong in the direction that never comes back down.
        #
        # Safe ordering is submit's, unchanged: its producer-stream
        # wait_event fences the side-stream D2H snapshot ahead of any
        # later default-stream mutation, so zeroing here cannot race the
        # staged copy.
        reset_up_heli_max(current.state)
        memory_watch.sample()

    def progress_callback(**event):
        # The external observer first and unconditionally: this
        # runner's own publication is throttled to every 60th step, and
        # a caller driving the run in-process wants the cadence, not the
        # sample of it.
        if observer is not None:
            observer(**event)
        outer_step = int(event["outer_step"])
        elapsed = float(event["model_elapsed_seconds"])
        if outer_step == 1 or outer_step % 60 == 0 \
                or elapsed == float(exp.run_seconds):
            memory_watch.sample()
            durable_paths = writers.paths
            _atomic_json(progress_path, {
                "schema": PROGRESS_SCHEMA,
                "status": "RUNNING",
                "source": inputs.source,
                "model_elapsed_seconds": elapsed,
                "outer_step": outer_step,
                "requested_run_seconds": float(exp.run_seconds),
                "forecast_wall_seconds": time.perf_counter() - forecast_started,
                "gpu_peak_used_bytes_observed": memory_watch.peak_bytes(
                    "cuda_device_used"),
                "last_durable_wrfout": (
                    None if not durable_paths
                    else str(durable_paths[-1].resolve())),
            })

    # Here and not at the writers' construction above: the closure this
    # binds reads `writers.paths`, so it cannot exist before the writers
    # do.  Still ahead of every submit -- the first one happens inside
    # execute_experiment below -- and attach_progress_callback refuses
    # outright if that ever stops being true.
    if observer is not None:
        writers.attach_progress_callback(observer)

    try:
        memory_watch.start()
        with writers:
            execution = execute_experiment(
                model, history_handler=history_handler,
                progress_callback=progress_callback, validate_state=True,
                skip_feedback_path=True, pool_trim_per_period=True)
            cp.cuda.Stream.null.synchronize()
            timing["forecast_execution_with_async_io"] = (
                time.perf_counter() - forecast_started)
            drain_started = time.perf_counter()
            writers.drain()
            timing["final_writer_drain"] = time.perf_counter() - drain_started
            wrfout_paths = writers.paths
        timing["forecast_and_io_inclusive"] = (
            time.perf_counter() - forecast_started)
    finally:
        memory_watch.stop()
        model._io_manager = None
    memory_watch.sample()

    cadence_seconds = float(exp.root.history_interval_s)
    cadence_receipt = _validate_hash_bound_history_cadence(
        exp, cadence_seconds)
    output_schedule = _history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=cadence_seconds)
    expected_frames = len(output_schedule)
    if len(wrfout_paths) != expected_frames:
        raise RuntimeError(
            f"history writer published {len(wrfout_paths)} frames, "
            f"expected {expected_frames}")
    expected_names = tuple(record[2] for record in output_schedule)
    if tuple(path.name for path in wrfout_paths) != expected_names:
        raise RuntimeError(
            "WRF history output filenames/cadence differ from the "
            "hash-bound request")

    final_health = _strict_json(vars(
        StateHealthValidator(node.state).validate(phase="final.d01")))
    final_stability = _strict_json(stability_report(
        node.state, cfg, boundary_width=cfg.spec_bdy_width))
    if not final_health["ok"]:
        raise FloatingPointError(
            f"prepared forecast final health failed: {final_health}")
    started = time.perf_counter()
    final_digest = canonical_state_digest(
        node.state, node.clock, scope="trajectory")
    timing["canonical_final_state_digest"] = time.perf_counter() - started

    _verify_inputs_unchanged(inputs)
    if _runtime_source_identity() != runtime_source_identity:
        raise RuntimeError("forecast runtime implementation changed during run")
    output_inventory = []
    for path, (offset_seconds, valid_time, _name) in zip(
            wrfout_paths, output_schedule, strict=True):
        output_inventory.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "model_elapsed_seconds": offset_seconds,
            "valid_time": valid_time.isoformat(),
            "atomic_writer_readback_verified": True,
        })
    timing["total"] = time.perf_counter() - total_started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "source": inputs.source,
        "prepared_layout": inputs.layout,
        "scope": (
            f"single {cfg.nx}x{cfg.ny}x{cfg.nz} prepared-cache GPUWM "
            f"forecast using "
            f"{inputs.physics_receipt.get('profile') or 'the hash-bound experiment physics suite'}"),
        "gpuwm_version": __version__,
        "runtime_source_identity": runtime_source_identity,
        "domain": {
            "grid_id": 1,
            "nx": int(cfg.nx),
            "ny": int(cfg.ny),
            "nz": int(cfg.nz),
            "dx_m": float(cfg.dx),
            "dy_m": float(cfg.dy),
            "dt_seconds": float(cfg.dt),
        },
        "run_seconds": float(exp.run_seconds),
        "wall_seconds": timing["total"],
        "io_mode": "history",
        "history_interval_seconds": cadence_seconds,
        "timing_seconds": timing,
        "executor": {
            "steps": int(execution.steps),
            "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "health": {
            "initial": initial_health,
            "final": final_health,
            "final_stability": final_stability,
            "history": history,
            "limits": {"max_cfl": 10.0, "max_w_ms": 150.0},
        },
        "final_state_digest": final_digest,
        "physics": {
            **dict(inputs.physics_receipt),
            "resolved_lw_sw": list(radiation_scheme_ids(cfg)),
            "radiation_update_count": _physics_update_count(
                driver.radiation_callable),
            "microphysics_update_count": int(driver.microphysics_updates),
            "swdown_min_wm2": float(cp.min(driver.fields["swdown"]).get()),
            "swdown_max_wm2": float(cp.max(driver.fields["swdown"]).get()),
            "rainnc_max_mm": float(cp.max(driver.microphysics.rainnc).get()),
        },
        "memory": {
            "gpu_peak_used_bytes_observed": memory_watch.peak_bytes(
                "cuda_device_used"),
            "cupy_pool_peak_total_bytes_observed": memory_watch.peak_bytes(
                "cupy_pool_total"),
            "cupy_pool_peak_used_bytes_observed": memory_watch.peak_bytes(
                "cupy_pool_used"),
            "cpu_peak_rss_bytes": _peak_rss_bytes(),
            # What each number above actually measured, how often it was
            # sampled, and whether observation stayed complete.
            "gpu_peak_sampling": memory_watch.summary(),
        },
        "gridded_output": {
            "cadence_seconds": cadence_seconds,
            "cadence_receipt": cadence_receipt,
            "frames_per_file": 1,
            "expected_frame_count": expected_frames,
            "exact_frame_count_verified": True,
            "initial_frame_verified": True,
            "last_scheduled_frame_verified": True,
            "last_scheduled_offset_seconds": cadence_receipt[
                "last_scheduled_offset_seconds"],
            "last_scheduled_valid_time": cadence_receipt[
                "last_scheduled_valid_time"],
            "last_scheduled_equals_run_end": cadence_receipt[
                "last_scheduled_equals_run_end"],
            "initial_and_final_frames_verified": cadence_receipt[
                "last_scheduled_equals_run_end"],
            "all_frames_readback_verified": all(
                item["atomic_writer_readback_verified"]
                for item in output_inventory),
            "completion_attribute": {
                "name": "GPUWM_WRITE_COMPLETE",
                "value": 1,
            },
            "frame_count": len(output_inventory),
            "total_bytes": sum(item["bytes"] for item in output_inventory),
            "files": output_inventory,
        },
        "input": {
            "prepared_root": str(inputs.prepared_root),
            "domain_bundle": str(inputs.domain_bundle_path),
            "source_domain_count": inputs.source_domain_count,
            "source_member": inputs.source_member,
            "proof_sha256": inputs.file_sha256["proof"],
            "source_manifest_sha256": inputs.file_sha256["source_manifest"],
            "source_manifest_identity": (
                None if inputs.source_manifest_receipt is None
                else dict(inputs.source_manifest_receipt)),
            "prepared_content_sha256": inputs.cache_reader.content_sha256,
            "prepared_header_sha256": inputs.file_sha256["cache_header"],
            "static_cache_sha256": inputs.file_sha256["static"],
            "geometry_receipt_sha256": inputs.file_sha256["geometry_receipt"],
            "experiment_config_sha256": inputs.file_sha256["experiment_config"],
            "wps_namelist_sha256": inputs.file_sha256["wps_namelist"],
            "wrf_direct_contract_sha256": inputs.file_sha256[
                "wrf_direct_contract"],
            "forcing_hours": list(inputs.forcing_hours),
            "boundary_interval_seconds": inputs.boundary_interval_seconds,
            "preprocessing": inputs.proof["preprocessing"],
            "source_identity": inputs.cache_identity["source_identity"],
            "cache_identity_compatibility": dict(
                inputs.cache_identity_compatibility),
            "landuse_identity": dict(inputs.landuse_identity),
            "direct_wrf_export_source": inputs.export_source_receipt,
            "authority_sha256": dict(inputs.file_sha256),
        },
    }
    _atomic_json(outdir / "report.json", report)
    emit_run_capsule(
        outdir, emission_site="prepared_single_domain_forecast",
        run_context={
            "runner_route_and_io_mode": {
                "route": "prepared_single_domain_forecast",
                "io_mode": "history"},
            "output_and_diagnostic_mode": {
                "io_mode": "history",
                "history_interval_seconds": cadence_seconds},
            # DETERMINISM.md lists the configuration bytes as a
            # load-bearing pin, and this route hash-binds the experiment
            # TOML before it runs a step -- the published wheel's capsule
            # still said "the run context did not supply this pin"
            # because the value was never handed over (the 4090 stress
            # run's honesty finding).  Both byte pins bind here now,
            # git or no git.
            "config_bytes": {
                "path": str(inputs.experiment_config),
                "sha256": inputs.file_sha256["experiment_config"]},
            "input_artifact_bytes": dict(inputs.file_sha256),
        },
        input_bytes={"entries": {
            name: {"algorithm": "sha256", "digest": digest,
                   "path": str(inputs.authority_paths[name])}
            for name, digest in inputs.file_sha256.items()}},
        run_shape={
            "route": "prepared_single_domain_forecast",
            "domain_count": 1,
            "run_seconds": float(exp.run_seconds),
            "nx": int(cfg.nx), "ny": int(cfg.ny), "nz": int(cfg.nz),
            "dt_seconds": float(cfg.dt),
        },
        output={"frames": output_inventory,
                "trajectory_digest": {"d01": final_digest}},
        receipts={"report": {"path": str((outdir / "report.json").resolve())}},
    )
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": "PASS",
        "source": inputs.source,
        "model_elapsed_seconds": float(exp.run_seconds),
        "requested_run_seconds": float(exp.run_seconds),
        "report": str((outdir / "report.json").resolve()),
        "frame_count": len(output_inventory),
    })
    return report


def _positive_finite_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a number of seconds") from error
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return seconds


def _clock_defaults(experiment_config: Path) -> tuple[float, float]:
    """The run length and history cadence the experiment already declares.

    ``--run-seconds`` and ``--history-interval-seconds`` are validated to
    equal these exactly -- the experiment TOML is hash-bound, so no other
    value can ever be accepted.  Requiring the user to retype them bought
    nothing and cost a wasted cycle to anyone who tried to shorten a
    retry.  They are optional now and default to what the file says; when
    supplied, the exact-match guard is unchanged.
    """

    raw = tomllib.loads(Path(experiment_config).read_text(encoding="utf-8"))
    experiment = raw.get("experiment")
    domains = raw.get("domain")
    if not isinstance(experiment, Mapping) or not isinstance(domains, list):
        raise ValueError(
            f"{experiment_config} is not an [experiment]/[[domain]] TOML, "
            "so --run-seconds and --history-interval-seconds cannot be "
            "defaulted from it; pass them explicitly")
    roots = [d for d in domains
             if isinstance(d, Mapping) and int(d.get("grid_id", 0)) == 1]
    if len(roots) != 1:
        raise ValueError(
            f"{experiment_config} does not declare exactly one grid_id 1 "
            "domain; pass --run-seconds and --history-interval-seconds "
            "explicitly")
    try:
        return (float(experiment["run_seconds"]),
                float(roots[0]["history_interval_s"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{experiment_config} declares no usable run_seconds / d01 "
            "history_interval_s; pass --run-seconds and "
            "--history-interval-seconds explicitly") from error


def _resolve_clock_arguments(args) -> None:
    """Fill omitted clock flags from the hash-bound experiment TOML."""

    if args.run_seconds is not None \
            and args.history_interval_seconds is not None:
        return
    run_seconds, history_interval_seconds = _clock_defaults(
        args.experiment_config)
    if args.run_seconds is None:
        args.run_seconds = run_seconds
        print(f"prepared forecast: --run-seconds defaulted to "
              f"{run_seconds:g} from {args.experiment_config}",
              file=sys.stderr)
    if args.history_interval_seconds is None:
        args.history_interval_seconds = history_interval_seconds
        print(f"prepared forecast: --history-interval-seconds defaulted to "
              f"{history_interval_seconds:g} from {args.experiment_config}",
              file=sys.stderr)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SUPPORTED_SOURCES), required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument(
        "--domain-bundle", type=Path,
        help=("explicit hierarchy d01 bundle; if omitted it is derived from "
              "the hash-bound domain-artifacts manifest"))
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--prepared-content-sha256", required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument(
        "--physics-profile", default=None,
        help=("optional assertion that the hash-bound experiment IS this "
              "shipped suite, refused on any switch drift; omitted, the "
              "experiment's own physics runs as written and its "
              "WRF-verification status is reported, never gating"))
    parser.add_argument(
        "--ack", action="append", default=[],
        help=("registry-owned expert acknowledgement id; repeat as "
              "needed.  The hash-bound experiment's acknowledgements "
              "array delivers the same consent"))
    parser.add_argument(
        "--run-seconds", type=float, default=None,
        help=("forecast length; must equal the hash-bound experiment's "
              "run_seconds, and defaults to it when omitted"))
    parser.add_argument(
        "--history-interval-seconds", type=_positive_finite_seconds,
        default=None,
        help=("history cadence; must equal the hash-bound experiment's d01 "
              "history_interval_s, and defaults to it when omitted"))
    parser.add_argument("--io-mode", choices=("history",), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args(argv)


def _parse_materialize_args(argv=None):
    parser = argparse.ArgumentParser(
        prog=(
            "python -m gpuwm.prepared_single_domain_forecast "
            "--materialize-authorities"),
        description=(
            "Create one hash-receipted named-source experiment/WPS authority "
            "pair for an exact physics profile."),
    )
    parser.add_argument("--source", choices=sorted(SUPPORTED_SOURCES), required=True)
    parser.add_argument(
        "--base-experiment-config", type=Path, required=True)
    parser.add_argument("--base-wps-namelist", type=Path, required=True)
    parser.add_argument(
        "--physics-profile", default=None,
        help=("shipped suite to materialize into the experiment; omitted, "
              "the base config's own physics is published unchanged and "
              "its WRF-verification status is reported"))
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None, *, observer=None) -> int:
    """The runner's command line.

    ``observer`` is forwarded to :func:`run_prepared_forecast` and is
    otherwise inert: it lets a caller that hosts this runner in its own
    process (``gpuwm run-plan``, through ``gpuwm go``) receive the same
    progress this function publishes to ``progress.json``, plus each
    wrfout as it lands.  ``None`` -- every command-line invocation --
    changes nothing.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--show-capabilities"]:
        print(json.dumps(runner_capabilities(), sort_keys=True))
        return 0
    if argv[:1] == ["--materialize-authorities"]:
        args = _parse_materialize_args(argv[1:])
        # Same standard as --outdir below, and for the same reason: the
        # commonest way to meet this stage twice is to re-run the
        # documented command, which lands on an --output-directory that
        # already exists.  That is a usage answer, not a crash.
        try:
            receipt = materialize_named_source_authorities(
                source=args.source,
                base_experiment_config=args.base_experiment_config,
                base_wps_namelist=args.base_wps_namelist,
                physics_profile=args.physics_profile,
                output_directory=args.output_directory,
            )
        except FileExistsError as error:
            print("prepared_single_domain_forecast: "
                  f"--output-directory refused: {error}", file=sys.stderr)
            return 2
        except (OSError, ValueError) as error:
            # A refusal this stage owns -- a profile its source cannot
            # prepare, a base config that is not there -- is still a
            # sentence rather than a traceback.  It is labelled as what
            # it is, not as an --output-directory problem.
            print(f"prepared_single_domain_forecast: refused: {error}",
                  file=sys.stderr)
            return 2
        print(json.dumps({
            "schema": receipt["schema"],
            "status": receipt["status"],
            "source": receipt["source"],
            "physics_profile": receipt["physics_profile"],
            "experiment_config": receipt["generated"][
                "experiment_config"],
            "wps_namelist": receipt["generated"]["wps_namelist"],
            "receipt": receipt["receipt"],
        }, sort_keys=True))
        return 0
    args = _parse_args(argv)
    _resolve_clock_arguments(args)
    # A rejected --outdir is a usage mistake, not a crash: one sentence
    # naming the problem and a directory that works, never a traceback.
    try:
        outdir = claim_output_directory(
            args.outdir, protected_roots=(args.prepared_root,))
    except (ValueError, FileExistsError) as error:
        print(f"prepared_single_domain_forecast: --outdir refused: {error}",
              file=sys.stderr)
        return 2
    started = time.perf_counter()
    _atomic_json(outdir / "progress.json", {
        "schema": PROGRESS_SCHEMA,
        "status": "VALIDATING_PREPARATION",
        "source": args.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": args.run_seconds,
    })
    try:
        inputs = preflight_prepared_forecast(
            source=args.source, prepared_root=args.prepared_root,
            proof_sha256=args.proof_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            prepared_content_sha256=args.prepared_content_sha256,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist,
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.ack),
            run_seconds=args.run_seconds,
            history_interval_seconds=args.history_interval_seconds,
            domain_bundle=args.domain_bundle)
        verification = dict(inputs.physics_receipt).get("verification")
        if (isinstance(verification, dict)
                and verification.get("status") != "wrf-verified"
                and isinstance(verification.get("sentence"), str)):
            # One sentence, and the run continues (owner posture:
            # warn-not-block; detail lives in the receipt).
            print(f"prepared forecast: {verification['sentence']}",
                  file=sys.stderr)
        report = run_prepared_forecast(inputs, output_directory=outdir,
                                       observer=observer)
    except BaseException as error:
        model_elapsed_seconds = 0.0
        try:
            current_progress = _load_json_object(
                outdir / "progress.json", "forecast progress")
            model_elapsed_seconds = float(
                current_progress.get("model_elapsed_seconds", 0.0))
        except (OSError, TypeError, ValueError):
            pass
        inventory_error = None
        try:
            durable_inventory = _durable_wrfout_inventory(outdir)
        except (OSError, ValueError) as inventory_failure:
            durable_inventory = []
            inventory_error = (
                f"{type(inventory_failure).__name__}: {inventory_failure}")
        failure = {
            "schema": REPORT_SCHEMA,
            "status": "FAIL",
            "source": args.source,
            "run_seconds": args.run_seconds,
            "io_mode": args.io_mode,
            "history_interval_seconds": args.history_interval_seconds,
            "wall_seconds": time.perf_counter() - started,
            "model_elapsed_seconds": model_elapsed_seconds,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "durable_partial_output": {
                "status": "PARTIAL_NOT_RUN_PASS",
                "frame_count": len(durable_inventory),
                "total_bytes": sum(
                    item["bytes"] for item in durable_inventory),
                "files": durable_inventory,
                "inventory_error": inventory_error,
            },
        }
        _atomic_json(outdir / "report.json", failure)
        _atomic_json(outdir / "progress.json", {
            "schema": PROGRESS_SCHEMA,
            "status": "FAIL",
            "source": args.source,
            "model_elapsed_seconds": model_elapsed_seconds,
            "requested_run_seconds": args.run_seconds,
            "report": str((outdir / "report.json").resolve()),
            "error": str(error),
            "last_durable_wrfout": (
                None if not durable_inventory
                else durable_inventory[-1]["path"]),
        })
        raise
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "source": report["source"],
        "run_seconds": report["run_seconds"],
        "history_interval_seconds": report["history_interval_seconds"],
        "frame_count": report["gridded_output"]["frame_count"],
        "prepared_content_sha256": report["input"][
            "prepared_content_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
