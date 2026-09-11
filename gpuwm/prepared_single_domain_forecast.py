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
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
import threading
import time
import tomllib
import traceback
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm.core import streaming  # noqa: E402
from gpuwm import __version__  # noqa: E402
from gpuwm.config import radiation_scheme_ids  # noqa: E402
from gpuwm.core.nssl2_contract import (  # noqa: E402
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    MP_PHYSICS as NSSL2_MP_PHYSICS,
    nssl2_contract_receipt,
    resolve_nssl2_mode_for_config,
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
from gpuwm.aerosol_source_receipt import (  # noqa: E402
    AEROSOL_SOURCE_KEY,
    aerosol_source_report_entry,
)
from gpuwm.experiment import build_experiment, load_experiment  # noqa: E402
from gpuwm.explain import (  # noqa: E402
    add_explain_flag, explain_enabled, layered, render as render_explanation,
    warn,
)
from gpuwm.kernel_compile_notice import (  # noqa: E402
    ARCHITECTURE_MISSING, COLD_CACHE, COMPILING_STATUS,
    current_compute_capability, kernel_cache_state, scan_kernel_cache,
)
from gpuwm.ingest.prepared_cache import (  # noqa: E402
    CONDITIONAL_PREPARATION_RECEIPTS,
    SOIL_PREPARATION_RECEIPTS,
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
from gpuwm.progress_log import (  # noqa: E402
    ProgressOptions, add_progress_arguments)
from gpuwm.receipt_paths import receipt_basename  # noqa: E402
from gpuwm.supervisor import atomic_write_json  # noqa: E402
from gpuwm.physics_compat import (  # noqa: E402
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    MULTI_DOMAIN_SELECTION_SCHEMA,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
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
    single_domain_physics_selection,
    single_domain_runtime_switches,
    single_domain_verification_status,
    thompson_runtime_requirements,
    thompson_table_root,
    validate_single_domain_physics_profile,
)
from gpuwm.certify.capsule import emit_run_capsule  # noqa: E402
from gpuwm.spectral_seam import (  # noqa: E402
    seam_capsule_receipts as _seam_capsule_receipts,
)
from gpuwm.source_adapters import packaged_profile_sources  # noqa: E402
# Re-exported, not called here: `tests/test_prepared_single_domain_forecast`
# reads the GRIB2 profile's pins through this module because this module is
# where the certificate that enforces them lives.
from gpuwm.source_authorities import (packaged_profile,  # noqa: E402,F401
                                      twentycrv3_authority_sha256)
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
MYNN_RTE_RRTMGP_PHYSICS_PROFILE = MYNN_RTE_RRTMGP_PROFILE_ID
MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE = MYNN_RUC_RTE_RRTMGP_PROFILE_ID
MYNN_NOAHMP_RTE_RRTMGP_PHYSICS_PROFILE = MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID
PHYSICS_PROFILES = (
    PHYSICS_PROFILE,
    KESSLER_PHYSICS_PROFILE,
    TWENTYCRV3_WSM6_PHYSICS_PROFILE,
    THOMPSON_PHYSICS_PROFILE,
    MORRISON_PHYSICS_PROFILE,
    NSSL2_PHYSICS_PROFILE,
    NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
    MYNN_PHYSICS_PROFILE,
    MYNN_RTE_RRTMGP_PHYSICS_PROFILE,
    RUC_PHYSICS_PROFILE,
    MYNN_RUC_PHYSICS_PROFILE,
    MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE,
    NOAHMP_PHYSICS_PROFILE,
    MYNN_NOAHMP_PHYSICS_PROFILE,
    MYNN_NOAHMP_RTE_RRTMGP_PHYSICS_PROFILE,
)
#: Sources prepared through the declarative mapped route against a
#: PACKAGED profile -- source id -> the profile id
#: :mod:`gpuwm.source_authorities` pins.
#:
#: These share one certificate: the bundle's copied mapping, composition
#: and provenance must be byte-equal to the three documents this
#: distribution ships for that profile.  What differs between them is the
#: SHAPE OF THEIR INPUT MANIFEST, and only that: `20crv3` carries the
#: every-member GRIB2 manifest with its filename-bound member identity,
#: while any composed mapped preparation carries
#: `gpuwm-mapped-composition-inputs-v1`.  Both shapes are validated below;
#: neither pin is relaxed for the other.
#:
#: A row here is what makes a packaged source runnable end to end, and it
#: is a row -- not an arm.  A profile whose composition role is a PENDING
#: declaration (an atmosphere-only source awaiting its cross-source
#: land-surface donor) can never have prepared a bundle, so it holds no
#: row: including it would promise this stage an arm the profile itself
#: refuses to supply.
#: Sentinel for "this identity document has no such key at all", so a
#: difference report can say `absent` rather than `None`, which is a value
#: a field can legitimately hold.
_MISSING = object()

from gpuwm.prepared_source_schemas import composed_packaged_profiles, source_schemas
_MAPPED_PACKAGED_PROFILE = composed_packaged_profiles()
_MAPPED_SOURCES = frozenset({"mapped", *_MAPPED_PACKAGED_PROFILE})
SUPPORTED_SOURCES = frozenset({"gfs", "era5", "hrrr"} | _MAPPED_SOURCES)

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
REFL_10CM_MICROPHYSICS = (1, 6, 8, 9, 10, 16, 18, 28, 50)
_SOURCE_PHYSICS_PROFILES_BY_SOURCE = {
    "mapped": (),  # No source-specific verification claim; every suite remains selectable.
    # REPORTED METADATA, NOT A GATE (owner ruling 2026-07-31): these
    # per-source lists name the shipped profiles whose verification
    # evidence this runner can vouch for on each source.  They feed the
    # capabilities receipt and the registry drift check; nothing refuses
    # a suite for being absent from them.  The one real per-source
    # blocker they used to encode -- RUC absent from GFS because a
    # GFS-initialised RUC forecast prepared in full and then died on its
    # first surface-temperature call with `mavail must be finite` -- was
    # root-caused, not re-gated: a shoreline land column carrying the
    # water soil category (SOILTYP 14 under a land LU_INDEX) reached RUC's
    # soilvegin, which has no arm for it, and `0./0.` went into MAVAIL.
    # real.exe reconciles that column at initialization
    # (module_initialize_real.F:3608-3650) and so does every ArWen door
    # now, through gpuwm/ingest/soil.py:door_reconciled_soil_category
    # (proven both ways by tests/test_ruc_shoreline_soil_category.py).
    # No route blocker remains and none is enforced below (ENG-009).
    # Each radiation-bearing MYNN twin follows the MYNN row it mirrors,
    # in the same order the registry route declares it: the drift check in
    # tests/test_physics_registry.py compares these lists to the route's
    # own, element for element.
    "gfs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE,
        NOAHMP_PHYSICS_PROFILE, MYNN_NOAHMP_PHYSICS_PROFILE,
        MYNN_NOAHMP_RTE_RRTMGP_PHYSICS_PROFILE),
    "era5": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE,
        RUC_PHYSICS_PROFILE, MYNN_RUC_PHYSICS_PROFILE,
        MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE),
    "20crv3": (
        TWENTYCRV3_WSM6_PHYSICS_PROFILE, PHYSICS_PROFILE,
        THOMPSON_PHYSICS_PROFILE, MORRISON_PHYSICS_PROFILE,
        NSSL2_PHYSICS_PROFILE, NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE),
    # The NetCDF profile decodes the same 20CRv3 model on the same
    # pressure-level/soil contract, so it reports the same suites.  These
    # rows are REPORTED METADATA, not a gate (see the note above), so this
    # is a statement about what has been verified, not a permission.
    "20crv3-cf": (
        TWENTYCRV3_WSM6_PHYSICS_PROFILE, PHYSICS_PROFILE,
        THOMPSON_PHYSICS_PROFILE, MORRISON_PHYSICS_PROFILE,
        NSSL2_PHYSICS_PROFILE, NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE),
    # HRRR's own stock-WRF gate is the WSM6/YSU/MM5-91/Noah slice, and
    # gpuwm/hrrr_route_inputs.py's SUPPORTED_MICROPHYSICS admits the
    # same scheme set the other sources report here.
    "hrrr": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged pressure-level profile decodes the same HRRR model
    # through the generic mapped route, so it reports the same verified
    # suite set the native route does (reported metadata, not a gate).
    "hrrr-prs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # One verified GEFS member through the same generic mapped route
    # (reported metadata, not a gate).
    "gefs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged GDPS profile is the same generic mapped route again
    # (reported metadata, not a gate).
    "gem-gdps": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged AIFS single profile initialises the same WSM6/YSU/
    # MM5-91/Noah slice from ECMWF's AI forecast (reported metadata, not
    # a gate); no suite has AIFS-initialised verification evidence yet.
    "aifs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged AI-ensemble member-hybrid profile is the same generic
    # mapped route again (reported metadata, not a gate); no suite has
    # member-initialised verification evidence yet.
    "aigefs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged GDAS pgrb2 profile is the GFS field catalogue through
    # the same generic mapped route (reported metadata, not a gate).
    "gdas": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged ECMWF open-data profile shares the mapped route's
    # WSM6/YSU/MM5-91/Noah contract (reported metadata, not a gate).
    "ecmwf-open-data": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged ICON-EU regular-lat-lon profile runs the same generic
    # mapped route, so it reports the same verified suite set the other
    # packaged pressure profiles do (reported metadata, not a gate).
    "icon-eu": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # RAP through the same packaged mapped route: the generic suites,
    # with no RAP-verified subset to prefer (reported metadata, not a
    # gate; the route is not stock-WRF certified and says so).
    "rap": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # RRFS through the same packaged mapped route: the generic suites,
    # with no RRFS-verified subset to prefer (reported metadata, not a
    # gate; the route is not stock-WRF certified and says so).
    "rrfs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
}

#: The three Noah-MP suites, which every source above offers and none of
#: them lists by hand.  They were declared for ``gfs`` alone while this
#: runner supports eighteen sources, and a route that declares any expert
#: list is EXHAUSTIVE -- so on an ERA5 single domain the three were
#: undeclared: nothing offered them, and a plan that named one was told
#: the template was off-route instead of being handed the acknowledgement
#: advisory the option exists to raise.  What gates them is the route's
#: expert acknowledgement and the option's own evidence warnings, which
#: are source-independent statements: Noah-MP is implemented-unverified
#: everywhere, not unverified on era5 and verified on gfs.  Appended per
#: source rather than written into each tuple so the offer cannot drift
#: source by source again, and appended in the registry's own order so
#: the drift check in tests/test_physics_registry.py compares equal.
_NOAHMP_EXPERT_PROFILES = (
    NOAHMP_PHYSICS_PROFILE,
    MYNN_NOAHMP_PHYSICS_PROFILE,
    MYNN_NOAHMP_RTE_RRTMGP_PHYSICS_PROFILE,
)
_SOURCE_PHYSICS_PROFILES = MappingProxyType({
    source: (
        profiles if source == "mapped"
        # "mapped" claims nothing on purpose: it names no model, so it
        # reports no per-source list at all and its expert offer would
        # have no source to attach to.
        else profiles + tuple(
            profile for profile in _NOAHMP_EXPERT_PROFILES
            if profile not in profiles))
    for source, profiles in _SOURCE_PHYSICS_PROFILES_BY_SOURCE.items()
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
#: The keys a NAMED profile is allowed to own in the materialized
#: authority -- i.e. every key whose value the generated experiment.toml
#: may state on the profile's behalf.  It is deliberately WIDER than any
#: one profile's switch table (`radt_minutes` is a compatibility
#: spelling of `radt`, `ra_rrtmg_variant` is pinned only by the
#: RRTMG-family profiles), because it doubles as the mask
#: :func:`_without_materialized_physics` applies to BOTH sides of the
#: non-physics descriptor digest: a key that one profile pins and
#: another does not must be outside that digest either way, or the
#: "descriptors unchanged" proof would read a physics choice as a
#: descriptor change.
#:
#: Membership here does NOT license deleting a value the user wrote.
#: What a given materialization may rewrite is the narrower
#: :func:`_profile_pinned_physics` map, and even that only after
#: :func:`_declared_physics_conflicts` has proved the config agrees.
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
#: The nest-transition rule every materialized authority states.  No
#: profile switch table carries it, so it is pinned here, once, and read
#: from here by both the emitter and the agreement check -- a second
#: literal would let the two disagree about what the profile "is".
_NEST_MICROPHYSICS_TRANSITION = "same-scheme-only"
#: The three keys that spell ONE selection.  ``ra_lw_physics =
#: ra_sw_physics = -1`` is documented in gpuwm/config.py as preserving
#: the historical ``ra_physics`` aggregate exactly, and explicit pairs
#: require ``ra_physics = 0``, so a config and a profile can name the
#: same two schemes through different keys.  They are compared as a
#: resolved pair, never key by key; see :func:`_declared_physics_conflicts`.
_RADIATION_SELECTION_KEYS = frozenset({
    "ra_physics", "ra_lw_physics", "ra_sw_physics"})
# The four per-source lookups below, and the two legacy companions
# after them, are DERIVED for the mapped rows and then overridden for
# the routes that genuinely differ.  They used to be literal id lists,
# and every composed packaged profile spelled the identical value in
# all six -- so adding a registry row meant remembering six more edits,
# and forgetting one surfaced as a bare `KeyError` inside the stage
# that runs a prepared bundle rather than as a refusal a user could
# act on.  The arbitrary acceptance test says a new model is table
# work; this is the seam where that used to stop being true.
#
# The overrides are the whole of what is NOT generic: `gfs`, `era5` and
# `hrrr` have their own direct runners and their own document schemas,
# and `20crv3` writes its OWN input-manifest schema because its member
# identity is sealed there -- a difference the stage reads to decide
# whether to demand a member manifest, so it must survive.
_SOURCE_SCHEMA = source_schemas(_MAPPED_SOURCES)
_PROOF_SCHEMA = {
    **{source: "gpuwm-mapped-direct-wrf-proof-v1"
       for source in _MAPPED_SOURCES},
    "gfs": "gpuwm-gfs-direct-wrf-proof-v3",
    "era5": "gpuwm-era5-direct-wrf-proof-v2",
    "hrrr": "gpuwm-hrrr-native-direct-wrf-proof-v1",
}
_LEGACY_PROOF_SCHEMAS = {
    # The mapped proof has never had an earlier revision, so no mapped
    # source accepts a legacy document.
    **{source: frozenset() for source in _MAPPED_SOURCES},
    # v2 remains independently verifiable.  It predates the explicit
    # front-door physics selection receipt and therefore cannot be promoted
    # to v3 by inference.
    "gfs": frozenset({"gpuwm-gfs-direct-wrf-proof-v2"}),
    "era5": frozenset(),
    # New in this runner as of the DA background lane: there is no
    # earlier HRRR bundle for it to have to accept.
    "hrrr": frozenset(),
}
_HIERARCHY_PROOF_SCHEMA = {
    **{source: "gpuwm-mapped-native-hierarchy-proof-v1"
       for source in _MAPPED_SOURCES},
    "gfs": "gpuwm-gfs-native-hierarchy-proof-v2",
    "era5": "gpuwm-era5-native-hierarchy-proof-v1",
    # HRRR's multi-domain route is gpuwm.hrrr_hierarchy_direct feeding
    # gpuwm.prepared_domain_tree_forecast, a designed division of labour
    # this lane does not widen.  Naming a schema no HRRR preparation
    # writes keeps _resolve_prepared_layout's generic path from matching
    # by accident; the explicit refusal below is what a caller sees.
    "hrrr": "gpuwm-hrrr-native-hierarchy-proof-unreachable-here",
}
_LEGACY_HIERARCHY_PROOF_SCHEMAS = {
    **{source: frozenset() for source in _MAPPED_SOURCES},
    # v1 predates the front-door physics receipt the v2 hierarchy proof
    # carries and cannot be promoted to it by inference, the same rule
    # the direct proof's v2 lives under.
    "gfs": frozenset({"gpuwm-gfs-native-hierarchy-proof-v1"}),
    "era5": frozenset(),
    "hrrr": frozenset(),
}
#: The EXACT top-level inventory of the two documents
#: :mod:`gpuwm.mapped_direct` publishes -- the mapped route's direct
#: proof and its hierarchy proof.  ``_validate_packaged_mapped_evidence``
#: compares ``set(proof)`` against these and refuses on any difference,
#: which is right: a proof carrying a key this reader does not know
#: about is a document from a preparation this reader cannot vouch for.
#:
#: They live here, named, for one reason.  When these lists were spelled
#: inline inside the validator, the 2.5.0 soil-mesh work added
#: ``soil_texture_downscale`` to BOTH proofs in ``gpuwm/mapped_direct.py``
#: and nothing here changed -- so on this line the forecast runner
#: rejected every mapped bundle the preparation stage produced, with
#: "mapped 20CRv3 proof top-level inventory differs", and the whole
#: mapped route's last leg was dead.  The test fixture did not catch it
#: because the fixture wrote its own proof rather than the writer's.
#: ``tests/test_prepared_single_domain_forecast.py`` now parses
#: ``gpuwm/mapped_direct.py``'s own proof literals and asserts they equal
#: these sets, so the next key added to the writer fails a test instead
#: of a user's run.
MAPPED_DIRECT_PROOF_KEYS = frozenset({
    "schema", "status", "stock_wrf_export", "forcing_times", "soil_texture_downscale",
    "forcing_hours", "boundary_interval_seconds", "execution_inputs",
    "source_composition", "preprocessing", "static", "geometry",
    "prepared_cache", "export", "timing_seconds", "proof_content_sha256",
})
MAPPED_HIERARCHY_PROOF_KEYS = frozenset({
    "schema", "status", "stock_wrf_export", "domain_count", "forcing_times",
    "soil_texture_downscale", "forcing_hours",
    "boundary_interval_seconds", "target_contract", "execution_inputs",
    "source_composition", "preprocessing", "hierarchy_workers",
    "root_static", "root_geometry", "static_catalog", "source_coverage",
    "artifact_receipt", "wrf_manifest", "timing_seconds",
    "proof_content_sha256",
})
#: Top-level proof keys the writer publishes ONLY when that preparation
#: opted in, so the runner has to take the document with them and
#: without them.  ``gpuwm/mapped_direct.py`` spreads these in
#: conditionally -- ``**({...} if <receipt> is not None else {})`` --
#: which is why they cannot sit in the required sets above: a mapped
#: bundle prepared without the corridor carries byte-for-byte the proof
#: it always did, and every retained hash stays valid.  Requiring them
#: would refuse that bundle; ignoring them would refuse the opted-in one,
#: and the exactness for every OTHER key is what this inventory is for.
MAPPED_OPTIONAL_PROOF_KEYS = frozenset({"statics_corridor"})
_SOURCE_ADAPTER = {
    # The generic mapped adapter, truthfully: a packaged profile is
    # prepared by `gpuwm.mapped_direct` with nothing model-specific in
    # the code path, so the adapter string is the mapped one and WHICH
    # profile it was is bound where it actually lives -- the mapping and
    # composition digests, checked byte for byte above.  Writing a
    # per-model adapter id here would be a per-model label on a route
    # that has no per-model half.
    **{source: "rw-wps-mapped-composition-v2"
       for source in _MAPPED_SOURCES},
    "gfs": "gfs-pgrb2-0p25-direct-v1",
    "era5": "era5-grib1-direct-v1",
    # The one mapped route whose adapter id IS model-specific, because
    # its member identity is sealed by a packaged member manifest.
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
#: The dynamical fields every external LBC interval carries.  The scalar
#: half (``qv``, and for an aerosol-aware ``mp_physics = 28`` the two
#: aerosol tracers the preparation read in) comes from
#: :func:`gpuwm.boundary_fields.external_scalar_fields`, the same function
#: the preparation writes the inventory from; a hand-typed six-field
#: constant here refused every mp=28 cache as "differs from forcing".
_LBC_DYNAMICS_FIELDS = ("mu", "phi", "theta", "u", "v")
#: The inventory every scheme other than an aerosol-reading mp=28 carries,
#: in the cache's sorted spelling; kept as the name test fixtures build
#: their headers from.
_LBC_FIELDS = tuple(sorted((*_LBC_DYNAMICS_FIELDS, "qv")))


def _lbc_field_inventories(cfg) -> tuple[list[str], ...]:
    """The LBC field lists a prepared cache may legitimately carry for ``cfg``.

    One list for most schemes.  For ``mp_physics = 28`` two: the preparation
    resolves whether aerosols were read from the dataset when it runs, and
    the cache records the outcome in its inventory, so both outcomes are
    admitted here and the aerosol receipt beside them says which happened.
    """
    from types import SimpleNamespace

    from gpuwm.boundary_fields import external_scalar_fields

    # The four attributes the inventory reads.  A caller that hands a
    # boundary-width view rather than a RunConfig (the receipt validators
    # do) is priced as the moist, non-aerosol run every earlier cache was,
    # which is the six-field inventory this check used to hard-code.
    scheme = SimpleNamespace(
        moist=bool(getattr(cfg, "moist", True)),
        mp_physics=int(getattr(cfg, "mp_physics", 0)),
        aer_init_opt=int(getattr(cfg, "aer_init_opt", 0)),
        mp28_aerosol_source=getattr(cfg, "mp28_aerosol_source", "auto"))
    inventories = []
    for aerosol_from_input in (False, True):
        fields = sorted(
            (*_LBC_DYNAMICS_FIELDS,
             *external_scalar_fields(scheme, aerosol_from_input=aerosol_from_input)))
        if fields not in inventories:
            inventories.append(fields)
    return tuple(inventories)
_HEX = frozenset("0123456789abcdef")
_TWENTYCRV3_SOURCE = "NOAA-CIRES-DOE 20CRv3 every-member GRIB2"
_TWENTYCRV3_MEMBER_IDENTITY = "filename_memNNN_not_grib2_pdt"
_TWENTYCRV3_FILENAME = re.compile(
    r"^mem(?P<member>[0-9]{3})_(?P<time>[0-9]{10})_"
    r"(?P<role>pl|sfc)\.grb2$")
#: The two decode-route inventories a 20CRv3 member preparation can have
#: sealed, exactly: the in-process engine on the bare default, or the
#: subprocess pair on the documented Python-engine workaround.  The
#: member manifest itself declares no decoder section (it predates the
#: generic manifest and stays the route's user-facing authority), so the
#: pin here is "one of the two shipped decode routes, in full" -- never a
#: partial inventory, never a role from each.
_TWENTYCRV3_DECODER_ROLE_SETS = (
    frozenset({"gpuwm_mapped_engine"}),
    frozenset({"grib2_inventory", "grib2_dump"}),
)
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
        "mapped": {
            "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
            "prepared_layouts": ["mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": [],
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "authority_binding": "caller mapping/composition/input receipts and sealed cache identity",
            "limitations": ["Forecast skill depends on the supplied data and selected physics; no model-specific verification is claimed."],
        },
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
        "20crv3-cf": {
            "readiness": (
                "PACKAGED_NETCDF_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "member_identity": "ensemble_mean_analysis_not_a_member",
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["20crv3-cf"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "NOAA PSL's 20CRv3 NetCDF distribution is the ENSEMBLE MEAN "
                "analysis; it is smoother than any single member and is not "
                "one of the 80 trajectories",
                "orography and land mask are recovered from 20CRv3's own "
                "published fields and supplied as a provenance-bound "
                "supplement, because PSL publishes neither",
                "GPU forecast execution has no public 20CRv3 acceptance gate",
            ],
        },
        "hrrr-prs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["hrrr-prs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from HRRR's pressure-level wrfprs product, "
                "which is smoother near terrain than the native hybrid "
                "levels the certified --source hrrr route consumes",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "icon-eu": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["icon-eu"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from DWD's regular-lat-lon ICON-EU product "
                "set (field-per-file bz2 GRIB2); pressure-level moisture "
                "is derived from relative humidity because DWD publishes "
                "no specific humidity there",
                "FR_LAND and HSURF are once-per-cycle invariants, "
                "broadcast after an invariance proof; sentinel-valued "
                "ocean soil is masked from the mapping's own land "
                "fraction",
                "the native icosahedral ICON global product set (GDT "
                "101) is refused with the grid family named",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "gdas": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["gdas"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from the hourly f000..f009 pgrb2.0p25 set "
                "-- the field-complete analysis-cycle files, which the "
                "bytes stamp forecasts even at hour 0; the one "
                "analysis-stamped product (pgrb2.1p00.anl) publishes no "
                "soil, no land mask and no 2 m/10 m state and is not an "
                "initialization route",
                "publication lags the cycle by about seven hours (the "
                "delayed-cutoff assimilation cycle); a scheduler "
                "assuming GFS timing chases a 404",
                "the run stops at f009 -- GDAS is a background "
                "trajectory for the next assimilation window, not a "
                "public forecast product",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "gefs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["gefs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises ONE verified ensemble member from its "
                "pgrb2a+pgrb2b pair (the two level sets are exactly "
                "disjoint, so both files of the same member are "
                "mandatory per valid time); stage and verify the member "
                "with gpuwm-member-prep first",
                "ensemble mean/spread files sharing the member "
                "directories are refused at the byte level by the "
                "PDT-1 selector pins",
                "snow is published only under a 55-percent ocean "
                "bitmap and stays policy-controlled, so runs "
                "initialise snow-free",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "gem-gdps": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["gem-gdps"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from GDPS's 15 km pressure-level product; "
                "the soil column is the single 0-10 cm ISBA layer the "
                "product publishes, anchored by skin and deep-soil "
                "temperature",
                "orography, land mask and ice analysis are the "
                "once-per-cycle analysis invariants, broadcast after an "
                "invariance proof",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "aifs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["aifs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "the published soil column reaches 0.28 m; deeper Noah "
                "layers are WRF's shallow-column interpolation anchored "
                "by the skin and static deep-soil temperatures",
                "no snow state and no sea-ice fraction are published, so "
                "runs initialise bare-ground and open-water everywhere",
                "an AI emulator's fields carry no hydrometeors and no "
                "dynamical-balance constraint",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "aigefs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["aigefs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "the member product publishes NO land-surface state: "
                "soil, land mask, orography, skin temperature and 2 m "
                "humidity are borrowed from the same cycle's physical "
                "analysis through the cross-source composition and "
                "held at their analysis values across every lead",
                "no snow state and no sea-ice fraction are published "
                "by either contributor's borrowed set, so runs "
                "initialise bare-ground and open-water everywhere",
                "an AI emulator's fields carry no hydrometeors and no "
                "dynamical-balance constraint",
                "member identity is carried by the gpuwm-member-prep "
                "receipt, not by the byte-identical leaf filenames",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "ecmwf-open-data": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["ecmwf-open-data"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from the 0.25-degree open-data distribution "
                "(CC-BY-4.0); ECMWF's native 9 km HRES is "
                "access-restricted, a data-licensing fact rather than a "
                "capability gap",
                "snow and sea-ice fields are policy-controlled: the "
                "open data publishes them only as local-table/"
                "bitmap-masked records whose semantics are not yet bound",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "rap": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["rap"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from RAP's 32 km awip32 pressure-level "
                "product; the 13 km CONUS products are not reachable as "
                "tables (awp130pgrb has no soil state, and the native "
                "wrfprs grid is rotated lat-lon GDT 32769, outside the "
                "declared grid families)",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "rrfs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["rrfs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from the 3 km CONUS prslev+2dfld pair; "
                "prslev alone carries no surface fields and no soil, so "
                "both products must be supplied per valid time",
                "the natlev native-level product, the 3 km North-America "
                "rotated grid and every per-member ensemble file exist "
                "only in the frozen prototype bucket with no live front "
                "door, and are not claimed by this route",
                "not yet accepted by unchanged stock WRF",
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
            "explicit_expert_consent_required": False,
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
            "explicit_expert_consent_required": False,
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
            "restart_interval_seconds_required": None,
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
                "20crv3-cf": "uniform-positive-whole-hour",
                "hrrr-prs": "uniform-positive-whole-hour",
                "gdas": "uniform-positive-whole-hour",
                "gefs": "uniform-positive-whole-hour",
                "gem-gdps": "uniform-positive-whole-hour",
                "aifs": "uniform-positive-whole-hour",
                "aigefs": "uniform-positive-whole-hour",
                "aigfs": "uniform-positive-whole-hour",
                "ecmwf-open-data": "uniform-positive-whole-hour",
                "icon-eu": "uniform-positive-whole-hour",
                "rap": "uniform-positive-whole-hour",
                "rrfs": "uniform-positive-whole-hour",
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
            "restart_output": True,
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
        domain_id: int = 1, after_seconds: float | None = None,
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
    return tuple(record for record in records
                 if after_seconds is None or record[0] > after_seconds)


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
    if source in _MAPPED_SOURCES:
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
            f"this {source} GPU forecast profile has no public acceptance gate"
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
            "Noah-MP has no gpuwm/WRF forecast trajectory comparison; "
            "its registry acknowledgement is advisory")
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


def _profile_pinned_physics(
        switches: Mapping[str, object]) -> dict[str, object]:
    """Every key THIS materialization writes on the profile's behalf.

    The profile's own switch table plus the nest-transition rule the
    emitter appends.  A key outside it -- ``radt_minutes``, which no
    switch table pins -- is one the profile writes nothing for, so the
    materializer leaves the config's own value exactly where the user
    wrote it rather than deleting a setting nothing is replacing.
    (Keeping ``radt_minutes`` is physically inert under every shipped
    profile, all of which pin a positive ``radt`` that every consumption
    site prefers; what it moves is the prepared-cache domain identity
    for a user config declaring a non-default value, disclosed in the
    CHANGELOG and measured in
    ``test_a_key_no_profile_pins_is_kept_rather_than_deleted``.)

    Unpinned is not the same as unGOVERNED: a kept key whose value the
    profile still resolves is checked for agreement -- see
    :func:`_profile_resolved_unpinned_physics`.
    """

    pinned = dict(switches)
    pinned["nest_microphysics_transition"] = _NEST_MICROPHYSICS_TRANSITION
    return pinned


def _profile_resolved_unpinned_physics(
        pinned: Mapping[str, object]) -> dict[str, object]:
    """Keys the profile does not PIN but its authority still RESOLVES.

    ``ra_rrtmg_variant`` selects which radiation IMPLEMENTATION a
    resolved RRTMG (4, 4) pair executes -- gpuwm/core/rrtmg_legacy.py or
    gpuwm/core/rrtmgp.py.  A profile that pins the pair without pinning
    the variant (the 20CRv3 WSM6 suite is the one shipped example)
    resolves the variant through the RunConfig default, so a config
    declaring the OTHER variant would keep it under the kept-unpinned
    rule and run a different radiation implementation under the
    profile's name -- and step 5 (:func:`_validate_profile_switches`)
    could never notice, because the key is outside the switch table.
    Such a key is governed for AGREEMENT exactly like a pinned one,
    while staying outside the strip-and-rewrite set: a declaration that
    matches the resolved value survives where the user wrote it.

    Under a profile whose pair is not (4, 4) the variant is inert and
    ungoverned here: a config coherent enough to build with the legacy
    variant declares a 4/4 pair of its own, which the radiation
    comparison already refuses by name, and a variant row against a
    no-RRTMG profile would be noise beside it.
    """

    if "ra_rrtmg_variant" in pinned \
            or _profile_radiation_pair(pinned) != (4, 4):
        return {}
    from dataclasses import fields as dataclass_fields
    from gpuwm.config import RunConfig
    return {"ra_rrtmg_variant": next(
        field.default for field in dataclass_fields(RunConfig)
        if field.name == "ra_rrtmg_variant")}


def _profile_radiation_pair(pinned: Mapping[str, object]) -> tuple[int, int]:
    """The ``(lw, sw)`` this profile resolves to, aggregate included."""

    lw = int(pinned["ra_lw_physics"])
    sw = int(pinned["ra_sw_physics"])
    if (lw, sw) == (-1, -1):
        aggregate = int(pinned["ra_physics"])
        return aggregate, aggregate
    return lw, sw


def _physics_values_agree(declared: object, pinned: object) -> bool:
    """Whether a config's value IS the profile's value.

    ``true``/``1`` are different statements about a boolean switch and
    never agree; ``12`` and ``12.0`` are the same number written twice
    and always do, because TOML types an unsuffixed integer as ``int``
    while the switch tables carry the float the RunConfig field holds.
    """

    if isinstance(declared, bool) or isinstance(pinned, bool):
        return (isinstance(declared, bool) and isinstance(pinned, bool)
                and declared is pinned)
    if isinstance(declared, (int, float)) \
            and isinstance(pinned, (int, float)):
        return float(declared) == float(pinned)
    return type(declared) is type(pinned) and declared == pinned


def _declared_physics_conflicts(
        base_raw: Mapping[str, object], base_exp, *,
        pinned: Mapping[str, object],
) -> list[dict[str, object]]:
    """Every profile-owned physics value the config states differently.

    Only keys the config actually WRITES are considered: a key it is
    silent about is a key the profile is free to supply, which is the
    whole point of materializing an authority.  Radiation is compared as
    one resolved ``(lw, sw)`` selection rather than key by key, so a
    config that reaches the profile's exact two schemes through the
    aggregate spelling is agreement and not a conflict.  The governed
    set is the pinned keys plus the resolved-but-unpinned ones
    (:func:`_profile_resolved_unpinned_physics`): both are values the
    generated authority states on the profile's behalf, whether by
    writing them or by resolving their defaults.
    """

    governed = {**_profile_resolved_unpinned_physics(pinned), **pinned}
    expected_radiation = _profile_radiation_pair(pinned)

    def radiation_agrees(run) -> bool:
        try:
            return radiation_scheme_ids(run) == expected_radiation
        except ValueError:
            # A config whose own radiation keys are incoherent cannot
            # have reached here (build_experiment resolves them first),
            # but a resolver that raises must not read as agreement.
            return False

    runs = {int(domain.grid_id): domain.run for domain in base_exp.domains}
    scopes: list[tuple[str, Mapping[str, object], bool]] = []
    shared = base_raw.get("shared")
    if isinstance(shared, Mapping):
        # A [shared] radiation key is inherited by every domain, so it
        # agrees only when every domain resolves to the profile's pair.
        scopes.append(("[shared]", shared,
                       all(radiation_agrees(run) for run in runs.values())))
    domains = base_raw.get("domain")
    if isinstance(domains, list):
        for index, domain in enumerate(domains):
            if not isinstance(domain, Mapping):
                continue
            grid_id = int(domain.get("grid_id", index + 1))
            run = runs.get(grid_id)
            scopes.append((f"[[domain]] d{grid_id:02d}", domain,
                           run is not None and radiation_agrees(run)))

    conflicts: list[dict[str, object]] = []
    for label, table, radiation_ok in scopes:
        for key in sorted(table):
            if key not in governed:
                continue
            if key in _RADIATION_SELECTION_KEYS and radiation_ok:
                continue
            if _physics_values_agree(table[key], governed[key]):
                continue
            conflicts.append({
                "scope": label,
                "key": key,
                "config_value": table[key],
                "profile_value": governed[key],
            })
    return conflicts


def _refuse_declared_physics_drift(
        conflicts: list[dict[str, object]], *, profile: str, origin: str,
        base_exp,
) -> None:
    """Name every overwritten key, both values, and both remedies.

    The sibling rail is the physics-fidelity axis
    (:func:`gpuwm.experiment._reject_axis_authored_keys`), and it refuses
    a second author for a governed key even when the two AGREE.  This
    one does not, on purpose: ``physics_mode``'s remedy is "strip the
    key and let the axis write it", while a named ``--physics-profile``
    is an ASSERTION that the config IS that suite -- step 5 of the
    documented chain (:func:`_validate_profile_switches`) refuses the run
    unless the experiment states the profile's values switch for switch.
    A config that agrees is therefore the documented happy path (`gpuwm
    domain --physics-profile P` writes P and step 2 names P), and
    refusing it would refuse the route's own output.  Disagreement is
    the whole of what is wrong here.
    """

    if not conflicts:
        return
    matched = identify_single_domain_profile(base_exp.root.run)
    rows = "\n".join(
        f"    {row['scope']} {row['key']}: config {row['config_value']!r}, "
        f"profile {row['profile_value']!r}"
        for row in conflicts)
    plural = "value" if len(conflicts) == 1 else "values"
    if matched == profile:
        # A remedy that cannot help is not a remedy.  The root already
        # resolves to the named profile, so "name the profile you meant"
        # would hand back the very flag that refused, and "edit those
        # keys to the profile values" would flatten the config's own
        # deliberate departures -- on the shipped LES trees that means
        # switching a PBL parameterization back ON over nests running
        # resolved turbulence.  The flag asserts every domain runs the
        # suite; this config says more than the suite on purpose, and
        # the one honest instruction is to stop asserting.
        remedy = (
            f"  REMEDY: omit --physics-profile.  This config's root "
            f"domain already resolves to {profile}, and the values "
            f"above are its own deliberate departures from that suite "
            f"on other scopes (a nest running its own turbulence or "
            f"cumulus choice, for example).  Without the flag, the "
            f"config's own physics is published unchanged on every "
            f"domain and the receipt records the suite's verification "
            f"status; editing those keys to the profile values would "
            f"change the physics this config was written to run.")
    else:
        own_suite = (
            f"--physics-profile {matched} (the shipped suite this "
            f"config already IS)" if matched is not None else
            "omit --physics-profile, which publishes the config's own "
            "suite unchanged and records its verification status in "
            "the receipt")
        remedy = (
            f"  REMEDY, either: edit those keys in {origin} to the "
            f"profile values printed above, so the config and the "
            f"profile say the same thing; or name the profile you "
            f"meant -- {own_suite}.")
    raise ValueError(layered(
        f"--physics-profile {profile} contradicts the physics {origin} "
        f"declares: {len(conflicts)} {plural} would be overwritten.\n"
        f"{rows}\n"
        f"{remedy}",
        "--materialize-authorities publishes the experiment.toml every "
        "later stage binds by hash: the fetch manifest, the front door, "
        "the prepared-cache identity and the forecast all read THAT "
        "file, not the one you passed in.  Rewriting a physics value you "
        "wrote, into a file you never see, would run a different "
        "forecast than the one your config describes and would still "
        "pass every downstream check, because those checks compare the "
        "generated authority against the same profile.  Step 5 of the "
        "documented chain refuses this exact drift when a run reaches it "
        "(`experiment physics differs from the ... profile`); this "
        "refusal is the same rule applied at the first step, before a "
        "fetch and a preprocessing run have been paid for."))


def named_profile_config_conflicts(
        base_text: str, *, source: str, profile: str,
) -> list[dict[str, object]]:
    """The conflicts ``--materialize-authorities`` would refuse on.

    The seam ``gpuwm go`` (and through it `gpuwm run-plan`'s prepared
    route) derives its forwarded ``--physics-profile`` with.  The
    derivation used to read the ROOT domain alone
    (``identify_single_domain_profile(experiment.root.run)``) while the
    refusal above reads every ``[[domain]]`` table, so on the wizard's
    own ``--ladder`` trees -- root = the profile, nests deliberately
    departing from it -- the chain composed a stage-1 command guaranteed
    to refuse its own config.  Deriving through the SAME predicate the
    refusal runs is what keeps the two doors reading one sentence: an
    empty list here is exactly the promise that stage 1 will not raise
    the drift refusal for this (config, source, profile) triple.

    Returns the conflict rows (empty = the profile may be asserted).
    Raises what :func:`_render_materialized_experiment`'s own base
    parsing would raise for a config that does not load; callers that
    already loaded the experiment will not see that.
    """

    base_raw = tomllib.loads(base_text)
    base_exp = build_experiment(_experiment_tables(base_raw),
                                source="base named-source experiment")
    pinned = _profile_pinned_physics(
        _profile_runtime_switches(source, profile))
    return _declared_physics_conflicts(base_raw, base_exp, pinned=pinned)


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
    static_table = tables.pop("static", None)
    if static_table is not None:
        from gpuwm.static.highres_production import parse_static_table
        # Validate the companion's schema here; its path is resolved by the
        # consuming file loader. The original TOML remains unchanged.
        parse_static_table(static_table, source="materialized experiment", base_dir=Path("."))
    return tables


def _profile_acknowledgements(switches, base_exp) -> tuple[str, ...]:
    """Governance declarations a named profile's own switches require.

    Two, and they are separate claims about the same selectors:

    * :data:`~gpuwm.physics_compat.CONSTANT_DOWNWARD_LONGWAVE_ACK` --
      ``ra_lw_physics = 0`` under a land-surface scheme, so nothing
      computes downward longwave and the surface integrates a declared
      constant.  True at noon as much as at midnight.
    * :data:`~gpuwm.physics_compat.ASYMMETRIC_RADIATION_NOCTURNAL_ACK`
      -- additionally, shortwave is ON and this experiment's window
      contains local night at its reference point.
    """

    from gpuwm.physics_compat import (
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK, CONSTANT_DOWNWARD_LONGWAVE_ACK,
        downward_longwave_disposition, first_local_night_time)

    lw = int(switches.get("ra_lw_physics", switches.get("ra_physics", 0)))
    sw = int(switches.get("ra_sw_physics", switches.get("ra_physics", 0)))
    surface = int(switches.get("sf_surface_physics", 0))
    required: list[str] = []
    if sw > 0 and lw == 0 and base_exp.projection is not None:
        if first_local_night_time(
                base_exp.start_time, float(base_exp.run_seconds),
                ref_lat=base_exp.projection.ref_lat,
                ref_lon=base_exp.projection.ref_lon) is not None:
            required.append(ASYMMETRIC_RADIATION_NOCTURNAL_ACK)
    # The load guard's own classification, so a materialized experiment
    # can never need a token this function did not attach.
    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=lw, ra_sw_physics=sw, sf_surface_physics=surface)
    if kind in ("consumed", "published"):
        required.append(CONSTANT_DOWNWARD_LONGWAVE_ACK)
    return tuple(required)


def _acknowledgement_lines(merged, *, profile, added) -> list[str]:
    """The merged acknowledgements array, with why it grew."""

    lines = []
    for value in added:
        lines.append(f"# JUSTIFY {value}: required by the named physics")
        lines.append(f"# profile {profile}, which this materialized")
        lines.append("# experiment runs.  An explicit --physics-profile")
        lines.append("# selection is the declaration, written here in ink")
        lines.append("# rather than left to the base config's silence.")
        lines.append("# See docs/public/PHYSICS.md, \"Nocturnal validity\".")
    lines.append(
        "acknowledgements = ["
        + ", ".join(f'"{value}"' for value in merged) + "]")
    return lines


def _render_materialized_experiment(
        base_text: str, *, source: str, profile: str | None,
        origin: str = "the base experiment config",
) -> tuple[str, object, dict[str, object]]:
    """Patch only profile-owned TOML keys and preserve all other controls.

    ``profile=None`` (owner ruling 2026-07-31) means the base config's
    own physics IS the product: the authority pair is still published --
    later stages bind these exact bytes -- with no switch rewritten, and
    the receipt reports the suite's verification status instead of a
    profile.

    A NAMED profile supplies every profile-owned key the config is
    SILENT about, and must agree with every one it states.  It may not
    replace a physics value the user wrote: a disagreement is refused
    here by name (:func:`_refuse_declared_physics_drift`), before the
    output directory is claimed, before the fetch, and before
    preprocessing.  Contract change 2026-08-09; until then this function
    deleted all 26 profile-owned keys from the config and rewrote them
    from the profile without saying so.
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
    pinned = _profile_pinned_physics(switches)
    _refuse_declared_physics_drift(
        _declared_physics_conflicts(base_raw, base_exp, pinned=pinned),
        profile=profile, origin=origin, base_exp=base_exp)
    from gpuwm.toml_document import iter_toml_statements
    statements = list(iter_toml_statements(base_text))
    # DECLARATIONS THE NAMED PROFILE ITSELF REQUIRES.
    #
    # Naming a profile on the command line is an explicit selection, and
    # the project's rule is that an explicit selection is declared in
    # INK rather than by silence -- the domain wizard has written the
    # same declarations into its emissions since 1.7.1.  A materialized
    # experiment is not a file anybody hand-wrote; its physics comes
    # from the profile the operator named, so the profile's governance
    # consequences are written beside it here and appear in the
    # published experiment.toml where a reader meets them.
    #
    # Through 1.8.7 this was invisible because the shipped proof configs
    # carried a standing nocturnal acknowledgement that covered every
    # profile materialized onto them.  They no longer do (they run real
    # radiation now), so the declaration is attached to the thing that
    # actually needs it.
    required_acknowledgements = _profile_acknowledgements(
        switches, base_exp)
    declared = tuple(base_raw.get("experiment", {}).get(
        "acknowledgements", ()) or ())
    merged = list(declared) + [
        value for value in required_acknowledgements if value not in declared]
    rewrite_acknowledgements = tuple(merged) != declared
    has_shared = any(kind == "table" and path == ("shared",)
                     for kind, path, _lines in statements)
    profile_lines = [
        "# RW-WPS generated exact physics authority; do not hand-edit.",
        f"# source={source} profile={profile}",
        *(
            f"{key} = {_toml_literal(value)}"
            for key, value in switches.items()
        ),
        f"nest_microphysics_transition = "
        f"{_toml_literal(_NEST_MICROPHYSICS_TRANSITION)}",
    ]
    output: list[str] = []
    section: tuple[str, ...] = ()
    shared_emitted = False

    def finish_section() -> None:
        nonlocal shared_emitted
        if section == ("shared",) and not shared_emitted:
            if output and output[-1] != "":
                output.append("")
            output.extend(profile_lines)
            shared_emitted = True

    for kind, path, lines in statements:
        if kind in {"table", "array"}:
            finish_section()
            if kind == "array" and path == ("domain",) and not has_shared:
                output.extend(["[shared]", *profile_lines, ""])
                shared_emitted = True
                has_shared = True
            section = path
            output.extend(lines)
            if rewrite_acknowledgements and section == ("experiment",):
                output.extend(_acknowledgement_lines(
                    merged, profile=profile,
                    added=required_acknowledgements))
            continue
        # ``pinned``, not ``_MATERIALIZED_PHYSICS_KEYS``: only a key this
        # profile actually states is replaced by the block above.  Every
        # surviving line has been proved to agree with the profile, so
        # dropping it changes no value -- and a key the profile does not
        # pin keeps the value the config gave it instead of vanishing.
        if (section in {("shared",), ("domain",)} and kind == "assignment"
                and len(path) == 1 and path[0] in pinned):
            continue
        if (rewrite_acknowledgements and section == ("experiment",)
                and kind == "assignment" and path == ("acknowledgements",)):
            continue
        output.extend(lines)
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
    # The base digest is taken with the profile's own declarations
    # already merged in.  They are a CONSEQUENCE of the named physics,
    # not a descriptor control the materializer chose to move, so the
    # guard still says exactly what it always said -- nothing else
    # changed -- while letting a no-radiation profile carry the
    # declaration it requires.
    base_for_digest = base_raw
    if rewrite_acknowledgements:
        base_for_digest = dict(base_raw)
        experiment_table = dict(base_for_digest.get("experiment", {}))
        experiment_table["acknowledgements"] = list(merged)
        base_for_digest["experiment"] = experiment_table
    base_non_physics = _non_physics_descriptor_sha256(base_for_digest)
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
        # Never silent: what the named profile forced into the published
        # experiment.toml, so the receipt carries it too.
        "profile_acknowledgements": list(required_acknowledgements),
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
        base_text, source=source, profile=physics_profile,
        origin=str(base_experiment_config))
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


def _atomic_json(path: Path, payload, *, heartbeat: bool = False) -> None:
    """Publish one JSON document atomically, via the supervisor's replace.

    ``heartbeat=True`` marks the progress publications a watcher is TOLD
    to read (``progress.json``): on Windows a reader's plain ``open()``
    denies rename over the file, so a poll racing the republish would
    otherwise raise WinError 5 out of the forecast loop and kill the run
    it reports on (measured on the tree runner's identical seam; `gpuwm
    go`'s stopwatch heartbeat reads this file every 20 s).  Supervisor
    doctrine: bounded 0.50 s retry for everyone, then durable receipts
    fail loudly while a heartbeat quarantines its temporary and the
    worker stays up.
    """
    atomic_write_json(Path(path), _strict_json(payload),
                      _quarantine_on_permission_error=heartbeat)


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


class PreparationProofDigestMismatch(ValueError):
    """``--proof-sha256`` is not this preparation proof's file digest.

    Its own class because the CLI has to tell this apart from every other
    admission failure: it is the one a correct, complete, uncorrupted
    preparation produces when the reader took the digest from the wrong
    place, and the remedy is a different string to paste rather than a
    re-run of anything.
    """


def _proof_digest_refusal(
        proof_path: Path, given: str, actual: str,
) -> PreparationProofDigestMismatch:
    """The named refusal for a ``--proof-sha256`` that does not match.

    ``preparation proof SHA differs from --proof-sha256`` was the whole
    message, it arrived as an uncaught traceback, and it named neither
    which of the two digests in play is wanted nor what was given.  The
    document carries a field spelled ``proof_content_sha256`` -- the
    canonical hash of its own content WITHOUT that field -- and a reader
    hunting a digest inside a 42 KB proof finds that one first.  It can
    never equal the file digest, so the confusion is a dead end that the
    old sentence left the reader to discover.
    """

    breakage = (
        "this digest is what binds the forecast to the exact preparation "
        "that produced it, so a run past a mismatch would integrate a "
        "preparation nobody certified")
    content = None
    try:
        document = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        if isinstance(document, Mapping):
            content = document.get("proof_content_sha256")
    except (OSError, UnicodeDecodeError, ValueError):
        content = None
    if isinstance(content, str) and content == given:
        what = (
            "the value given is that document's own "
            "\"proof_content_sha256\" field, which covers the proof's "
            "content WITHOUT that field and therefore never equals the "
            "file's digest")
    elif isinstance(content, str):
        what = (
            "the value given matches neither the file nor the document's "
            "own \"proof_content_sha256\" field")
    else:
        what = "the value given is not that file's digest"
    return PreparationProofDigestMismatch(
        f"--proof-sha256 refused: it must be the sha256 of the file "
        f"{proof_path}, and {what}.  {breakage[0].upper()}{breakage[1:]}.\n"
        f"  expected (sha256 of the file proof.json): {actual}\n"
        f"  given    (--proof-sha256):                {given}\n"
        f"  Remedy: pass the expected digest above.  `rw-wps` and `gpuwm "
        f"prep` print the complete forecast command, with this digest "
        f"and the other two already filled in, when a preparation "
        f"finishes.")


def _proof_digest_refusal_at_the_door(args) -> str | None:
    """The ``--proof-sha256`` answer a door can give before doing work.

    Returns the refusal text, or ``None`` when there is nothing to say
    here -- either the digest matches, or the prepared root is not there
    at all, which is the preflight's refusal to give and not this one's.
    The check is the same digest comparison the preflight makes, so the
    two cannot disagree about what the flag means.
    """

    try:
        _require_digest(args.proof_sha256, "proof-sha256")
    except ValueError as error:
        return f"--proof-sha256 refused: {error}"
    proof_path = Path(args.prepared_root) / "proof.json"
    if not proof_path.is_file():
        return None
    try:
        actual = _sha256(proof_path)
    except OSError:
        return None
    if actual == args.proof_sha256:
        return None
    return str(_proof_digest_refusal(
        proof_path.resolve(), args.proof_sha256, actual))


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
        # EXISTS is not the breakage; HOLDS AN EARLIER RUN is.  An empty
        # directory carries no frames to merge with and no receipt to
        # overwrite, so refusing it prevents nothing.
        #
        # And it is the normal case, not an edge one: `gpuwm sim`
        # allocates this run's stamped folder before it dispatches here,
        # create-exclusively, because two launches inside one second must
        # not share a folder.  The door therefore hands this function a
        # directory that exists and is empty, every single time.
        # Refusing it killed the whole prepare-then-simulate route --
        # measured on weather-node-1, where the forecast never started.
        try:
            held = sorted(child.name for child in path.iterdir())
        except OSError as probe_error:
            # Reading it failed, so this is not the empty-folder case at
            # all -- and this function's whole subject is turning an
            # OS-level exception into a sentence, so the probe must not
            # become the one that escapes.
            #
            # A looping symlink is the case that gets here, and it is the
            # one E-12 already names: `mkdir` says FileExistsError
            # because the link IS a path entry, and scandir on it says
            # ELOOP.  `resolve()` does NOT raise for it on Linux/3.14 --
            # measured -- so deferring to _resolve_or_refuse would have
            # this function silently ACCEPT an --outdir with no
            # destination, which is worse than the traceback it replaced.
            # The refusal is raised here, in E-12's own words, with the
            # probe's error as the cause.
            detail = (getattr(probe_error, "strerror", None)
                      or str(probe_error))
            errno = getattr(probe_error, "errno", None)
            where = f" (errno {errno})" if errno is not None else ""
            raise ValueError(
                f"{flag} {path} cannot be resolved to a real location: "
                f"{detail}{where}.  A symlink that points into its own "
                f"chain has no destination to create, so there is nothing "
                f"here to refuse or to reuse -- pass an {flag} that names "
                f"a real path.") from probe_error
        if held:
            raise FileExistsError(
                f"{resolved} already holds a run's output "
                f"({', '.join(held)[:120]}), and this runner never merges "
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
    expected = "mapped" if source in _MAPPED_SOURCES else source
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


def _provenance_receipt() -> dict:
    """The running tree, for the report.  Never raises.

    Beside ``runtime_source_identity`` rather than instead of it: that
    hashes the forecast implementation's BYTES, which is the strongest
    binding available and stays the authority.  This says which INSTALL
    those bytes came out of -- wheel, editable or checkout, on which
    branch, dirty or clean -- which is the question a reader holding two
    receipts with different numbers actually has.
    """

    try:
        from gpuwm.provenance_gate import receipt_block

        return receipt_block()
    except Exception as error:                          # noqa: BLE001
        return {"unavailable": f"{type(error).__name__}: {error}"}


#: Attempts per identity resolution.  The failure this exists for is
#: transient and environmental -- a ``git`` that does not spawn -- not a
#: repository answering "no".  Two retries cost milliseconds on a
#: healthy box and are not paid at all outside the one install shape
#: that can degrade this way (see :func:`_runtime_provenance`).
_IDENTITY_ATTEMPTS = 3

#: The ``identity_source`` values a CHECKOUT can only reach by falling
#: THROUGH the git branch of :func:`gpuwm.runtime_manifest.provenance`:
#: that branch catches ``(OSError, SubprocessError)`` around its three
#: ``git`` calls and continues down the ladder rather than raising, so
#: one subprocess that fails to start re-answers the whole question from
#: a lower rung.  The manifest rung is deliberately absent -- it is
#: resolved BEFORE git and so is never a fall-through.
_GIT_FALLTHROUGH_SOURCES = frozenset({
    "installed-editable-source",
    "installed-wheel-record",
    "runtime-module-sha256-only",
})

#: The identity of a tree that is neither a manifest, nor a checkout,
#: nor an installed distribution: a source tree someone copied into
#: place.  The per-module digests in :func:`_runtime_source_identity`
#: still bind what ran.
_UNRESOLVED_PROVENANCE: Mapping[str, object] = MappingProxyType({
    "git_commit": None, "git_tree": None, "git_status_short": None,
    "identity_source": "runtime-module-sha256-only",
    "distribution_manifest_sha256": None, "installed_wheel": None,
    "installed_editable": None,
})


def _runtime_provenance() -> dict[str, object]:
    """Which install is executing, asked so a git hiccup cannot rename it.

    One resolver for all three installs (``gpuwm.runtime_manifest``).
    Its git branch, though, is the one place a healthy checkout can
    answer as something else entirely.  ``provenance`` wraps its three
    ``git`` calls in ``except (OSError, subprocess.SubprocessError):
    pass`` and CONTINUES to the editable/wheel ladder, so a single
    ``git rev-parse`` that fails to spawn moves ``identity_source``,
    ``git_commit``, ``git_tree``, ``git_status_short`` and one of
    ``installed_wheel`` / ``installed_editable`` in one step, with
    nothing on disk changed.  That is the cheap event this run's receipt
    was destroyed by; see :func:`_runtime_source_identity_change`.

    So a checkout that did not answer ``"git"`` is asked again.  The
    precondition is a FILESYSTEM one -- ``.git`` is a name on disk --
    because :func:`gpuwm.runtime_manifest.git_checkout_root` answers the
    same question by spawning git, and during the outage this exists for
    it would answer "not a checkout" every time and skip the retry that
    fixes it.  A wheel install has no ``.git`` and so pays nothing.
    """

    from gpuwm.runtime_manifest import IdentityError, provenance

    identity: dict[str, object] = dict(_UNRESOLVED_PROVENANCE)
    for attempt in range(_IDENTITY_ATTEMPTS):
        try:
            identity = dict(provenance(REPO))
        except IdentityError:
            identity = dict(_UNRESOLVED_PROVENANCE)
        if identity.get("identity_source") not in _GIT_FALLTHROUGH_SOURCES:
            break
        if not (REPO / ".git").exists():
            # Not a checkout at all.  This rung's answer is the truth
            # rather than a degraded one -- pip knows exactly which
            # artifact it wrote, and RECORD says so -- so asking again
            # would only buy the same answer three times.
            break
        if attempt + 1 < _IDENTITY_ATTEMPTS:
            time.sleep(0.1 * (attempt + 1))
    # A checkout that STILL did not answer "git" has now been asked
    # _IDENTITY_ATTEMPTS times, so this is no longer a hiccup: git
    # cannot be launched at all.  The rung that answered instead reports
    # on the tree PIP WAS POINTED AT, which in a linked worktree is a
    # different tree entirely -- measured here: the worktree executing
    # is eada530d and the editable rung answered c1b32d0, the main
    # checkout's HEAD.  Neither this module nor anything in
    # ``source_sha256`` came out of that tree, so the top-level commit
    # is re-read from REPO's own ``.git``.
    if (identity.get("identity_source") in _GIT_FALLTHROUGH_SOURCES
            and (REPO / ".git").exists()):
        identity.update(_repo_head_identity())
    return identity


def _repo_head_identity() -> dict[str, object]:
    """The git half of the identity, for REPO and no other tree.

    ``git_commit`` is read out of ``.git`` bytes in a format git has not
    changed in its lifetime, so a process that cannot SPAWN git can
    still say which commit is executing.  Worktree-aware, which is what
    matters here: ``.git`` is a FILE in a linked worktree and this
    project's work happens almost entirely in them.

    ``git_tree`` and ``git_status_short`` stay ``None``, and say so: the
    tree id lives inside the commit object and the working tree cannot
    be inspected without git at all.  The comparison reads ``None`` as
    "this was not asked" rather than as an answer.
    """

    from gpuwm.provenance import git_dir_identity

    try:
        head = git_dir_identity(REPO)
    except Exception:                                   # noqa: BLE001
        head = None
    return {
        "git_commit": None if head is None else str(head["commit_full"]),
        "git_tree": None,
        "git_status_short": None,
    }


def _tracked_status_lines(value: object) -> list[str] | None:
    """``git status --short`` with its untracked (``??``) rows dropped.

    ``None`` in, ``None`` out, because "nobody asked git" and "git found
    nothing" are different statements and the caller reads the
    difference.

    Untracked files are not the forecast implementation.  A run that
    writes anything inside the checkout creates them -- ``runs/`` is not
    ignored, so a single scratch file lands in ``git status --short`` as
    ``?? runs/...`` -- and comparing the raw list made such a run fail
    its own receipt for having produced output.  Measured in this
    worktree: 0 status lines before ``touch runs/probe``, 1 after.

    What survives the filter is every TRACKED path git reports as
    modified, added, deleted or renamed, which is the signal worth
    keeping: it catches an edit to a tracked file anywhere in the tree,
    including the many that ``source_sha256`` does not hash.
    """

    if value is None:
        return None
    return [str(line) for line in value if not str(line).startswith("??")]


def _untracked_blind(value: object) -> object:
    """One identity component with its untracked-file COUNT removed.

    ``installed_editable`` carries ``git.untracked_files`` -- an integer
    counting exactly the rows :func:`_tracked_status_lines` drops -- so
    on an editable install one scratch file moves the identity twice.
    ``git.dirty`` and ``git.dirty_files`` count TRACKED changes and are
    left alone, on purpose: they are the same signal the filtered status
    lines carry, arriving from the other install shape.
    """

    if not isinstance(value, Mapping):
        return value
    git = value.get("git")
    if not isinstance(git, Mapping) or "untracked_files" not in git:
        return value
    return {**value,
            "git": {key: item for key, item in git.items()
                    if key != "untracked_files"}}


def _runtime_source_identity_change(
        before: Mapping[str, object], after: Mapping[str, object],
) -> str | None:
    """Name the component that moved, or ``None`` if nothing did.

    A plain ``before != after`` conflated unrelated events, and the
    cheap ones destroyed the expensive one.  A finished forecast --
    every step integrated, every ``wrfout`` frame written and verified
    -- loses its receipt to ``forecast runtime implementation changed
    during run`` in two shapes that share no code with an implementation
    change:

    * one ``git rev-parse`` failing to spawn at the end of the run,
      which drops :func:`gpuwm.runtime_manifest.provenance` through its
      git branch and re-answers ``identity_source``, ``git_commit``,
      ``git_tree``, ``git_status_short`` and ``installed_wheel`` /
      ``installed_editable`` from a lower rung, all at once (that one is
      now also retried -- see :func:`_runtime_provenance` -- so reaching
      this comparison degraded is rare);
    * the run creating ONE untracked file inside the checkout, which
      appends a ``??`` row to ``git_status_short`` and increments
      ``installed_editable.git.untracked_files``.  ``runs/`` is not
      ignored, so that is not an unusual run: it is a run that writes
      where it was told to.

    So the components are compared on their own terms:

    * ``source_sha256`` is compared ALWAYS and strictly.  It is read
      from bytes on disk, it hashes the ten modules that ARE this
      forecast implementation, it cannot fail to resolve, and it is what
      makes skipping anything else survivable.
    * ``git_commit`` and ``git_tree`` are compared whenever BOTH ends
      resolved them.  ``None`` means "this was not asked", which is an
      unanswered question rather than a changed answer.
      :func:`_runtime_provenance` guarantees both describe REPO, so they
      stay comparable even when the rung underneath them changed.
    * ``git_status_short`` is compared, both ends resolved, over TRACKED
      paths only.
    * ``distribution_manifest_sha256``, ``installed_wheel`` and
      ``installed_editable`` are compared only when the SAME rung
      answered at both ends.  A sealed manifest, a wheel RECORD and an
      editable source tree answer three different questions, and one
      rung's answer measured against another's is not a comparison.
    * ``identity_source`` is not compared at all.  It names which
      RESOLVER answered, not which code is executing, and the only way
      it moves while every digest above holds is the fall-through this
      exists for.  An install genuinely replaced mid-run rewrites the
      ten hashed modules and is caught by ``source_sha256``.

    That is narrower than the old test in exactly one case: a commit
    that lands mid-run, touches none of the ten hashed files and no
    tracked file at all, AND coincides with a git failure at one end.
    Every other real change still fails: a tracked edit to a hashed file
    moves ``source_sha256``, a tracked edit anywhere else moves the
    filtered ``git_status_short``, and a HEAD that moves while git works
    moves ``git_commit``.
    """

    if before == after:
        return None
    # 1. The bytes on disk.  Always, strictly.
    first = dict(before.get("source_sha256") or {})
    second = dict(after.get("source_sha256") or {})
    for name in sorted(set(first) | set(second)):
        if first.get(name) != second.get(name):
            return (f"source_sha256 {name} {first.get(name)} -> "
                    f"{second.get(name)}")
    # 2. The commit and the tree, whenever both ends resolved them.
    #    _runtime_provenance guarantees both describe REPO and no other
    #    tree, so they stay comparable even across a changed rung.
    for field in ("git_commit", "git_tree"):
        one, other = before.get(field), after.get(field)
        if one is None or other is None:
            continue
        if one != other:
            return f"{field} {one!r} -> {other!r}"
    # 3. The rung's OWN artifact, only when the same rung answered at
    #    both ends.  A wheel RECORD, an editable source tree and a
    #    sealed manifest answer three different questions; comparing one
    #    rung's answer against another's is not a comparison at all.
    if before.get("identity_source") == after.get("identity_source"):
        for field in ("distribution_manifest_sha256", "installed_wheel",
                      "installed_editable"):
            one = _untracked_blind(before.get(field))
            other = _untracked_blind(after.get(field))
            if one is None or other is None:
                continue
            if one != other:
                return f"{field} {one!r} -> {other!r}"
    # 4. The working tree, tracked paths only.
    one = _tracked_status_lines(before.get("git_status_short"))
    other = _tracked_status_lines(after.get("git_status_short"))
    if one is not None and other is not None and one != other:
        return f"git_status_short {one!r} -> {other!r}"
    return None


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
    # ``.as_posix()`` is the one spelling every serialized identity key
    # in this product uses; ``str()`` of a relative Path is the Windows
    # bug it replaced.  Same bytes as the ``.replace`` this had.
    source_sha256 = {
        path.relative_to(REPO).as_posix(): _sha256(path)
        for path in paths
    }
    # One resolver for all three installs (gpuwm.runtime_manifest).  The
    # branch this replaced degraded a wheel install to
    # `runtime-module-sha256-only` -- honest, but weaker than the truth:
    # pip knows exactly which artifact it wrote, and RECORD says so.  It
    # also raised, uncaught, when site-packages happened to sit INSIDE
    # some unrelated repository, binding a stranger's commit or dying;
    # that case is now simply "not a checkout of this tree".
    #
    # The KEYS this publishes, and their meanings, are unchanged.  This
    # mapping is hashed into ``experiment_fingerprint``, which is
    # published in checkpoint headers, so a new field here would move
    # every already-sealed fingerprint and strand the legs carrying
    # them.  What improved is the RELIABILITY of the resolution
    # (:func:`_runtime_provenance`) and the end-of-run COMPARISON, which
    # moved out to :func:`_runtime_source_identity_change`.
    return {**_runtime_provenance(), "source_sha256": source_sha256}


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
                if source in _MAPPED_SOURCES
                else "portable-single-domain-v2"),
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
    # Both runners restore this prepared cache. The WRF file set is an
    # optional companion; when present its redundant lineage still binds.
    export_slot = proof.get("wrf_manifest")
    without_export = _optional_stock_wrf_export(proof, export_slot)
    wrf_manifest_path = None if without_export else _require_file(
        prepared_root / "wrf-native-input" / "manifest.json",
        "hierarchy direct-WRF manifest")
    artifact_manifest = _load_json_object(
        artifact_manifest_path, "hierarchy artifact manifest")
    hierarchy_receipt = _load_json_object(
        hierarchy_receipt_path, "hierarchy artifact receipt")
    wrf_manifest = dict(export_slot) if without_export else _load_json_object(
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
        **({"wrf_manifest": wrf_manifest_path} if wrf_manifest_path else {}),
    }
    if source in _MAPPED_SOURCES:
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
            if source in _MAPPED_SOURCES else "hierarchy-d01-v1"),
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
                or receipt_basename(path_value) != filename):
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


def _mapped_composition_manifest_file_specs(
        source: str, manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Validate the portable copy of one composed mapped input manifest.

    ``gpuwm-mapped-composition-inputs-v1`` is a different document from the
    per-source manifests: it binds the two AUTHORITIES (mapping and
    composition) plus every input file by path and digest, and it carries
    no experiment config or namelist -- those are bound by the proof's own
    ``execution_inputs`` receipt, which
    :func:`_validate_packaged_mapped_evidence` checks against the files
    the caller actually supplied.  So this reader validates the shape and
    the identity chain and returns the file inventory; it does NOT invent
    the roles the named-source manifests carry, because a role this
    document does not declare is a role nothing can bind.
    """

    expected_keys = {
        "schema", "mapping_sha256", "composition_sha256", "primary_files",
        "supplements", "provenance", "decoders",
    }
    member_keys = {"member", "member_identity"} & set(manifest)
    if (member_keys and (member_keys != {"member", "member_identity"}
            or any(not isinstance(manifest[key], str) or not manifest[key]
                   for key in member_keys))):
        raise ValueError("mapped manifest member identity is incomplete")
    if (set(manifest) - member_keys != expected_keys
            or manifest.get("schema") != _SOURCE_SCHEMA[source]):
        raise ValueError(
            f"{source} portable source manifest has an unsupported schema or "
            "top-level inventory")
    for role in ("mapping_sha256", "composition_sha256"):
        _require_digest(manifest.get(role), f"mapped manifest {role}")
    normalized: dict[str, dict[str, object]] = {}

    def bind(label: str, rows: object) -> None:
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"mapped composition manifest {label} inventory is empty")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {
                    "path", "bytes", "sha256"}:
                raise ValueError(
                    f"mapped composition manifest {label} row {index} is "
                    "malformed")
            path_value = row.get("path")
            byte_count = row.get("bytes")
            if (not isinstance(path_value, str) or not path_value
                    or isinstance(byte_count, bool)
                    or not isinstance(byte_count, int) or byte_count <= 0):
                raise ValueError(
                    f"mapped composition manifest {label} row {index} has an "
                    "unsafe path or byte count")
            key = f"{label}[{index}]"
            if key in normalized:
                raise ValueError("mapped composition manifest repeats a role")
            normalized[key] = {
                "name": receipt_basename(path_value),
                "path": path_value,
                "bytes": byte_count,
                "sha256": _require_digest(
                    row.get("sha256"), f"mapped manifest {key} sha256"),
            }

    bind("primary", manifest.get("primary_files"))
    supplements = manifest.get("supplements")
    if not isinstance(supplements, dict) or not supplements:
        raise ValueError(
            "mapped composition manifest binds no supplement; a packaged "
            "profile's terrain has to come from somewhere named")
    for role, rows in sorted(supplements.items()):
        bind(f"supplement:{role}", rows)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("mapped composition manifest binds no provenance")
    for role, row in sorted(provenance.items()):
        bind(f"provenance:{role}", [row])
    decoders = manifest.get("decoders")
    if not isinstance(decoders, dict):
        raise ValueError("mapped composition manifest decoders must be an object")
    for role, row in sorted(decoders.items()):
        bind(f"decoder:{role}", [row])
    return normalized


def _manifest_file_specs(
        source: str, manifest: Mapping[str, object], exp,
        proof: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], Mapping[str, object] | None]:
    """Normalized role specs, plus the GFS source receipt when there is one."""

    if source == "20crv3":
        return _twentycrv3_manifest_file_specs(manifest), None
    if source in _MAPPED_SOURCES:
        return _mapped_composition_manifest_file_specs(source, manifest), None
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
                or receipt_basename(name) != name
                or Path(name).is_absolute()):
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
    if exp.feedback != 0:
        warn("feedback is inactive for a single domain; the declared value is preserved")
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
        # The mode is RESOLVED FROM THIS CONFIG, never assumed to be the
        # shipped default lane: a hail-off or diagnosed-CCN run carries a
        # different mode and a different transported-field list, and a
        # receipt that describes some other configuration is worse than no
        # receipt at all.
        receipt["nssl2_contract"] = nssl2_contract_receipt(
            resolve_nssl2_mode_for_config(cfg))
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
    export = proof.get("export")
    mapped_config_suite = (
        source in _MAPPED_SOURCES
        and isinstance(export, dict)
        and export.get("status") == "READY"
        and export.get("schema") == "gpuwm-native-direct-wrf-export-v3")
    if not mapped_config_suite and (
            source not in {"gfs", "hrrr"}
            or proof.get("schema") in _LEGACY_PROOF_SCHEMAS[source]):
        return None
    if proof.get("schema") != _PROOF_SCHEMA[source]:
        return None
    label = source.upper()
    # Mapped preparation binds the selection inside its export receipt.
    # Older v2 exports stay readable under their original stock contract.
    selected = export.get("physics") if mapped_config_suite else proof.get("physics")
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
        receipt, actual: Path | None, label: str,
) -> None:
    """Bind a run-control file to the bytes the preparation validated.

    IDENTITY IS THE DIGEST.  The recorded ``path`` is provenance -- where
    the preparation found the file on the machine that prepared it -- and
    it is spelled in that machine's dialect.  This used to compare
    ``Path(path_value).name`` against the supplied file's name, which
    reads a recorded string with the RUNNING platform's separator rules:
    a Windows-prepared tree records ``C:\\...\\experiment.toml``, POSIX
    finds no separator in it, ``.name`` is the whole string, and the
    comparison failed on a tree whose bytes and SHA-256 matched exactly.
    A prepared tree therefore could not cross platforms -- prepare on the
    desktop, run on the node -- and the refusal said the receipt
    "differs from supplied file", blaming content that was identical.

    A name difference over identical bytes was never a content
    difference, and it prevents no breakage, so it is not a refusal.  The
    two facts that ARE the binding are checked here, separately, and the
    refusal names whichever one moved.
    """

    if not isinstance(receipt, dict) or set(receipt) != {
            "path", "bytes", "sha256"}:
        raise ValueError(f"mapped {label} execution receipt is malformed")
    path_value = receipt.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(
            f"mapped {label} execution receipt records no source path, so "
            "the file this preparation read cannot be named in any "
            "refusal about it. Re-prepare with a distribution that writes "
            "the provenance path.")
    byte_count = receipt.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise ValueError(f"mapped {label} execution byte count is malformed")
    _require_digest(receipt.get("sha256"), f"mapped {label} execution sha256")
    if actual is None:
        # A prepared hierarchy restores its sealed cache without the raw
        # preparation files. Its caller validates typed domain configuration
        # and compares this original digest to every cache's namelist identity.
        return
    differences: list[str] = []
    actual_bytes = actual.stat().st_size
    if receipt.get("bytes") != actual_bytes:
        differences.append(
            f"the receipt records {receipt.get('bytes')!r} bytes and the "
            f"supplied file is {actual_bytes}")
    actual_sha256 = _sha256(actual)
    if receipt.get("sha256") != actual_sha256:
        differences.append(
            f"the receipt records sha256 {receipt.get('sha256')!r} and the "
            f"supplied file hashes to {actual_sha256}")
    if differences:
        raise ValueError(
            f"mapped {label} execution receipt does not describe {actual}: "
            + "; ".join(differences)
            + ". A prepared tree is bound to the exact run-control bytes "
            "it was prepared from, so running it against different bytes "
            "would integrate a configuration the preparation never "
            f"validated. The preparation read this file as {path_value!r}; "
            "supply the bytes that receipt describes, or re-prepare "
            "against the file you have.")


def _optional_stock_wrf_export(proof, export) -> bool:
    """Recognize the producer's explicit optional/off companion-file receipt."""
    if not isinstance(export, dict) or export.get("status") == "READY":
        return False
    mode = proof.get("stock_wrf_export")
    status = export.get("status")
    valid = ((mode == "off" and status == "NOT_REQUESTED")
             or (mode == "optional" and status == "REFUSED"))
    keys = {"schema", "status", "reason"}
    if status == "REFUSED":
        keys.add("unsupported")
        valid = valid and isinstance(export.get("unsupported"), dict)
    if (not valid or set(export) != keys
            or export.get("schema") not in {
                "gpuwm-native-direct-wrf-export-v2", "gpuwm-native-direct-wrf-export-v3",
                "gpuwm-native-direct-wrf-hierarchy-export-v1"}
            or not isinstance(export.get("reason"), str) or not export["reason"]):
        raise ValueError("stock-WRF export receipt differs from the requested mode")
    return True


def _validate_packaged_mapped_evidence(
        *, prepared_root: Path, proof: Mapping[str, object],
        manifest: Mapping[str, object], manifest_sha256: str,
        experiment_config: Path | None, wps_namelist: Path | None,
        source: str = "20crv3",
) -> tuple[Mapping[str, Path], Mapping[str, object], str | None]:
    """Validate the shared mapped evidence, including explicit model claims.

    Named profiles additionally pin their shipped authorities. ``mapped``
    binds caller-authored authorities to the manifest, composition receipt
    and sealed cache identity. Neither route requires a forecast certificate.
    The historical function name remains for existing callers.
    """

    from gpuwm.source_authorities import packaged_authority_sha256

    profile_id = _MAPPED_PACKAGED_PROFILE.get(source)
    if source not in _MAPPED_SOURCES:
        raise ValueError(f"{source} is not a mapped preparation")
    member_manifest = _SOURCE_SCHEMA[source] == _SOURCE_SCHEMA["20crv3"]

    direct_proof_keys = set(MAPPED_DIRECT_PROOF_KEYS)
    hierarchy_proof_keys = set(MAPPED_HIERARCHY_PROOF_KEYS)
    schema = proof.get("schema")
    expected_proof_keys = (
        direct_proof_keys if schema == _PROOF_SCHEMA[source]
        else hierarchy_proof_keys)
    # Older proofs predate the optional stock-WRF companion export control.
    # Their READY export still goes through the original checks below.
    if "stock_wrf_export" not in proof:
        expected_proof_keys.discard("stock_wrf_export")
    elif (not isinstance(proof["stock_wrf_export"], str)
          or proof["stock_wrf_export"] not in {"off", "optional", "required"}):
        raise ValueError("mapped stock-WRF export mode is invalid")
    # Every required key present, and nothing beyond them but the
    # declared-optional ones: a missing key and an unrecognised key are
    # both still refusals, which is the exactness this inventory exists
    # to hold.
    observed = set(proof)
    if (schema not in {
            _PROOF_SCHEMA[source], _HIERARCHY_PROOF_SCHEMA[source]}
            or not expected_proof_keys <= observed
            or observed - expected_proof_keys - MAPPED_OPTIONAL_PROOF_KEYS):
        raise ValueError(f"mapped {source} proof top-level inventory differs")
    if schema == _HIERARCHY_PROOF_SCHEMA[source] \
            and not isinstance(proof.get("target_contract"), dict):
        raise ValueError(f"mapped {source} hierarchy target contract is missing")
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
    expected_authority_sha256 = (
        dict(packaged_authority_sha256(profile_id)) if profile_id else {
            "mapping": _require_digest(
                manifest.get("mapping_sha256"), "mapped manifest mapping sha256"),
            "composition": _require_digest(
                manifest.get("composition_sha256"),
                "mapped manifest composition sha256"),
        })
    if (_sha256(mapping_path) != expected_authority_sha256["mapping"]
            or _sha256(composition_path)
            != expected_authority_sha256["composition"]):
        if profile_id:
            raise ValueError(
                f"mapped preparation does not use the packaged {source} "
                f"authorities ({profile_id})")
        raise ValueError("mapped authorities differ from the input manifest")
    if _sha256(copied_manifest) != manifest_sha256:
        raise ValueError("mapped source manifest evidence differs from caller pin")
    if not member_manifest and any(
            manifest.get(f"{role}_sha256") != expected_authority_sha256[role]
            for role in ("mapping", "composition")):
        raise ValueError("mapped manifest authority identity differs")

    composition_document = _load_json_object(
        composition_path, "mapped composition authority")
    declared_bindings = composition_document.get("field_sources") or {}
    if not isinstance(declared_bindings, dict):
        raise ValueError("mapped composition field sources must be an object")
    terrain_bindings = [binding for binding in declared_bindings.values()
                        if isinstance(binding, dict)
                        and "terrain_height" in binding.get("fields", ())]
    terrain_spec = composition_document.get("supplements", {}).get(
        "terrain_height")
    if terrain_spec is None and len(terrain_bindings) == 1:
        terrain_spec = terrain_bindings[0]
    if not isinstance(terrain_spec, dict):
        raise ValueError("mapped composition names no unique terrain provider")

    known_names = {"mapping.json", "composition.json", "input-manifest.json",
                   "composition-inputs.json"}
    provenance_candidates = [
        path.resolve() for path in evidence_root.iterdir()
        if path.is_file() and path.name not in known_names
    ]
    if member_manifest:
        if len(provenance_candidates) != 1:
            raise ValueError("mapped source evidence must contain one provenance file")
        provenance_path = provenance_candidates[0]
        if _sha256(provenance_path) != expected_authority_sha256["provenance"]:
            raise ValueError(f"mapped {source} provenance authority differs")
        provenance_paths = {"mapped_provenance": provenance_path}
    else:
        provenance_rows = manifest.get("provenance")
        if (not isinstance(provenance_rows, dict) or not provenance_rows
                or len(provenance_candidates) != len(provenance_rows)):
            raise ValueError("mapped provenance evidence inventory differs")
        provenance_paths = {}
        remaining = {path: _sha256(path) for path in provenance_candidates}
        for role, row in sorted(provenance_rows.items()):
            digest = _require_digest(row.get("sha256"), "mapped provenance sha256")
            candidates = [path for path, value in remaining.items() if value == digest]
            if not candidates:
                raise ValueError(f"mapped provenance authority differs: {role}")
            path = candidates[0]
            if path.stat().st_size != row.get("bytes"):
                raise ValueError(f"mapped provenance byte count differs: {role}")
            del remaining[path]
            provenance_paths[f"mapped_provenance:{role}"] = path
        terrain_role = terrain_spec.get("provenance_role")
        if terrain_role not in provenance_rows:
            raise ValueError("mapped terrain provenance role is missing")
        terrain_digest = provenance_rows[terrain_role]["sha256"]
        if profile_id and terrain_digest != expected_authority_sha256["provenance"]:
            raise ValueError(f"mapped {source} provenance authority differs")
        expected_authority_sha256["provenance"] = terrain_digest
        provenance_path = provenance_paths[f"mapped_provenance:{terrain_role}"]

    receipt = proof.get("source_composition")
    # Composition bytes are already bound above. Their declared donor roles,
    # rather than a profile allowlist or a receipt's claims, govern the shape.
    expected_receipt_keys = {
        "schema", "status", "mapping", "composition", "input_manifest",
        "decoders", "terrain_products", "terrain_provenance", "alignment",
        "soil_layers", "frame_count", "valid_times", "frames",
        "receipt_content_sha256",
    }
    if declared_bindings:
        expected_receipt_keys |= {"contributing_sources"}
    if (not isinstance(receipt, dict) or set(receipt) != expected_receipt_keys
            or receipt.get("schema") != "gpuwm-mapped-composition-receipt-v1"
            or receipt.get("status")
            != "CANONICAL_FRAMES_COMPLETE_NOT_STOCK_WRF_CERTIFIED"):
        if declared_bindings and isinstance(receipt, dict) \
                and "contributing_sources" not in receipt:
            raise ValueError(
                "the composition declares contributing-source "
                "bindings but the receipt names no contributing sources; "
                "a cross-source decode always records them, so this "
                "receipt belongs to some other decode")
        raise ValueError(f"mapped {source} composition receipt is malformed")
    receipt_content = dict(receipt)
    declared_receipt_sha256 = receipt_content.pop(
        "receipt_content_sha256", None)
    expected_receipt_sha256 = hashlib.sha256(
        _canonical(receipt_content).encode("utf-8")).hexdigest()
    if declared_receipt_sha256 != expected_receipt_sha256:
        raise ValueError(f"mapped {source} composition receipt hash is stale")

    def evidence_record(record, expected_sha256: str, label: str) -> None:
        if (not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or record.get("sha256") != expected_sha256
                or not isinstance(record.get("path"), str)):
            raise ValueError(f"mapped {source} {label} receipt differs")

    evidence_record(
        receipt.get("mapping"), expected_authority_sha256["mapping"],
        "mapping")
    evidence_record(
        receipt.get("composition"), expected_authority_sha256["composition"],
        "composition")
    if member_manifest:
        # The member route seals TWO manifests: the caller pinned the
        # member manifest (its evidence copy was hashed above), and the
        # composition receipt seals the BRIDGED composition-inputs
        # document the decode actually verified.  The bridged bytes ride
        # the evidence directory precisely so this record can be
        # re-hashed here rather than taken on faith.
        bridged_manifest = _require_file(
            evidence_root / "composition-inputs.json",
            "mapped member bridged composition-inputs evidence")
        evidence_record(
            receipt.get("input_manifest"), _sha256(bridged_manifest),
            "manifest")
    else:
        evidence_record(
            receipt.get("input_manifest"), manifest_sha256, "manifest")
    provenance = receipt.get("terrain_provenance")
    if (not isinstance(provenance, dict)
            or set(provenance) != {"provenance_path", "provenance_sha256"}
            or provenance.get("provenance_sha256")
            != expected_authority_sha256["provenance"]
            or not isinstance(provenance.get("provenance_path"), str)):
        raise ValueError(f"mapped {source} terrain provenance receipt differs")

    execution = proof.get("execution_inputs")
    if not isinstance(execution, dict):
        raise ValueError(f"mapped {source} execution input receipt is missing")
    _validate_execution_file_receipt(
        execution.get("experiment_config"), experiment_config,
        "experiment config")
    _validate_execution_file_receipt(
        execution.get("wps_namelist"), wps_namelist, "WPS namelist")
    composition_decoders = receipt.get("decoders")
    execution_decoders = execution.get("decoders")
    # The decoder inventory is the manifest's own, not a constant: a GRIB2
    # profile's roles are whatever its sealed manifest declares (the
    # engine on the bare default, the subprocess pair on the workaround,
    # none for NetCDF), and BOTH sides are pinned -- the empty set is
    # checked against the manifest just as any other, so a bundle cannot
    # drop a decoder it actually used.  The MEMBER manifest declares no
    # decoder section, so its pin is the closed pair of shipped decode
    # routes instead: the recorded inventory must be one of the two, in
    # full.
    if member_manifest:
        observed_roles = (
            set(composition_decoders)
            if isinstance(composition_decoders, dict) else set())
        if frozenset(observed_roles) not in _TWENTYCRV3_DECODER_ROLE_SETS:
            raise ValueError(
                f"mapped {source} decoder inventory names neither the "
                "in-process engine nor the subprocess pair this "
                "distribution ships")
        expected_decoder_roles = observed_roles
    else:
        expected_decoder_roles = set(manifest.get("decoders") or {})
    if (not isinstance(composition_decoders, dict)
            or not isinstance(execution_decoders, dict)
            or set(composition_decoders) != expected_decoder_roles
            or set(execution_decoders) != expected_decoder_roles):
        raise ValueError(f"mapped {source} decoder inventory differs")
    decoder_sha256: dict[str, str] = {}
    for role in sorted(expected_decoder_roles):
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
            raise ValueError(f"mapped {source} decoder receipt differs: {role}")
        if not member_manifest:
            declared = manifest["decoders"][role]
            if (declared.get("sha256") != executed.get("sha256")
                    or declared.get("bytes") != executed.get("bytes")
                    or receipt_basename(declared.get("path"))
                    != receipt_basename(executed.get("path"))):
                raise ValueError(f"mapped decoder manifest differs: {role}")
        decoder_sha256[role] = _require_digest(
            composed.get("sha256"), f"mapped {source} decoder {role} sha256")

    composition_document = _load_json_object(
        composition_path, f"mapped {source} composition authority")
    if member_manifest:
        valid_times = manifest.get("valid_times")
        rows = manifest.get("files")
        surface_rows = [row for row in rows if row["role"] == "sfc"]
        expected_terrain = [{
            "path": row["path"],
            "sha256": row["sha256"],
        } for row in surface_rows]
        if receipt.get("terrain_products") != expected_terrain:
            raise ValueError(f"mapped {source} terrain product receipt differs")
        alignment = receipt.get("alignment")
        # The composed exact-subset receipt, exactly as the generic
        # single-source branch validates it, PLUS the ensemble identity
        # the sealed member manifest bound -- the member route's whole
        # reason to exist.  The key set is pinned closed: the packaged
        # composition declares `valid_time_exact`, so a receipt carrying
        # broadcast keys (or any other extra) is a receipt for some other
        # decode.
        expected_alignment_keys = {
            "schema", "status", "field", "primary_shape",
            "supplement_shape", "latitude_index_range",
            "longitude_index_range", "latitude_sha256", "longitude_sha256",
            "terrain_full_sha256", "terrain_subset_sha256",
            "supplement_valid_times", "matched_primary_valid_times",
            "invariant_across_all_supplement_times",
            "latitude_index_direction", "longitude_index_direction",
            "coordinate_match", "longitude_equivalence",
            "member", "member_identity",
        }
        expected_alignment = {
            "schema": "gpuwm-mapped-exact-subset-binding-v1",
            "status": "PASS",
            "field": "terrain_height",
            "coordinate_match": "exact_equivalent_contiguous_subset",
            "invariant_across_all_supplement_times": True,
            "member": manifest["member"],
            "member_identity": _TWENTYCRV3_MEMBER_IDENTITY,
        }
        if (not isinstance(alignment, dict)
                or set(alignment) != expected_alignment_keys
                or any(alignment.get(key) != value
                       for key, value in expected_alignment.items())
                or alignment.get("matched_primary_valid_times")
                != valid_times):
            raise ValueError(f"mapped {source} member/alignment receipt differs")
        _require_digest(
            alignment.get("terrain_subset_sha256"),
            "mapped composition terrain subset sha256")
        source_member: str | None = str(manifest["member"])
    else:
        valid_times = receipt.get("valid_times")
        if not isinstance(valid_times, list) or not valid_times:
            raise ValueError(
                "mapped composition receipt names no canonical valid times")
        # The terrain products a composed preparation used are the
        # supplement files the manifest bound to the profile's declared
        # data role -- not a guess, and not "whatever the receipt says":
        # the manifest is pinned by the caller's --source-manifest-sha256,
        # so this is the caller's own binding checked against the receipt.
        supplements = manifest.get("supplements")
        if not isinstance(supplements, dict):
            raise ValueError("mapped composition supplement inventory is missing")
        field_sources = declared_bindings
        terrain_binding = terrain_binding_name = None
        for name, binding in field_sources.items():
            if "terrain_height" in binding.get("fields", ()):
                terrain_binding, terrain_binding_name = binding, name
        declared_role = terrain_spec.get("data_role")
        expected_roles = {str(binding["data_role"])
                          for binding in declared_bindings.values()}
        expected_roles.add(str(declared_role))
        if set(supplements) != expected_roles:
            raise ValueError("mapped supplement roles differ from the composition")
        supplement_rows = supplements.get(declared_role)
        if not isinstance(supplement_rows, list) or not supplement_rows:
            raise ValueError("mapped composition supplement inventory is empty")
        # NAME plus digest, not the absolute path: the portable manifest
        # stores basenames on purpose so a prepared tree survives its raw
        # inputs being moved, while the composition receipt records the
        # path the decode actually opened.  The digest is the binding; the
        # name is what makes the pairing readable in a refusal.
        expected_terrain = [
            (receipt_basename(row.get("path")), row.get("sha256"))
            for row in supplement_rows
        ]
        recorded = receipt.get("terrain_products")
        if not isinstance(recorded, list) or [
            (receipt_basename(row.get("path")), row.get("sha256"))
            for row in recorded
            if isinstance(row, dict) and set(row) == {"path", "sha256"}
        ] != expected_terrain:
            raise ValueError(f"mapped {source} terrain product receipt differs")
        alignment = receipt.get("alignment")
        # The binding receipt is composed, not predicted: its index ranges
        # and array digests come from the two grids and this runner has no
        # independent source for them.  What IS checked is every claim the
        # receipt makes ABOUT itself -- that it passed, that the
        # coordinate match was exact, and (single-source) that the terrain
        # was invariant across the supplied times -- plus the fact that
        # its bytes are inside the proof, whose content hash the caller
        # pinned.
        if declared_bindings:
            # Cross-source: the top-level alignment is the terrain
            # provider's binding receipt, under the binding's own declared
            # clock; every borrowed field must carry a subset digest, and
            # matched plus carried times must be exactly the canonical
            # valid times so no lead's provenance goes unnamed.
            expected_alignment = {
                "schema": "gpuwm-cross-source-binding-v1",
                "status": "PASS",
                "binding": terrain_binding_name,
                "source_id": terrain_binding.get("source_id"),
                "grid_alignment": terrain_binding.get("grid_alignment"),
                "time_alignment": terrain_binding.get("time_alignment"),
                "coordinate_match": "exact_equivalent_contiguous_subset",
            }
            matched = alignment.get("matched_primary_valid_times") \
                if isinstance(alignment, dict) else None
            carried = alignment.get("broadcast_primary_valid_times", []) \
                if isinstance(alignment, dict) else None
            if (not isinstance(alignment, dict)
                    or any(alignment.get(key) != value
                           for key, value in expected_alignment.items())
                    or not isinstance(matched, list)
                    or not isinstance(carried, list)
                    or sorted(matched + carried) != sorted(valid_times)
                    or sorted(alignment.get("fields") or ())
                    != sorted(terrain_binding.get("fields") or ())):
                raise ValueError(
                    "mapped cross-source alignment receipt differs")
            subset_digests = alignment.get("field_subset_sha256")
            if not isinstance(subset_digests, dict) or set(subset_digests) \
                    != set(terrain_binding.get("fields") or ()):
                raise ValueError(
                    "mapped cross-source subset digests do not cover the "
                    "bound fields")
            for name in sorted(subset_digests):
                _require_digest(
                    subset_digests[name],
                    f"mapped cross-source subset sha256 for {name}")
        else:
            expected_alignment = {
                "schema": "gpuwm-mapped-exact-subset-binding-v1",
                "status": "PASS",
                "field": "terrain_height",
                "coordinate_match": "exact_equivalent_contiguous_subset",
                "invariant_across_all_supplement_times": True,
            }
            if (not isinstance(alignment, dict)
                    or any(alignment.get(key) != value
                           for key, value in expected_alignment.items())
                    or alignment.get("matched_primary_valid_times")
                    != valid_times):
                raise ValueError(
                    "mapped composition alignment receipt differs")
            _require_digest(
                alignment.get("terrain_subset_sha256"),
                "mapped composition terrain subset sha256")
        if declared_bindings:
            from gpuwm.source_authorities import packaged_contributing_sha256

            contributing_sha = (
                dict(packaged_contributing_sha256(profile_id))
                if profile_id else {
                    str(binding["mapping_role"]): _require_digest(
                        binding.get("mapping_sha256"), "contributing mapping sha256")
                    for binding in declared_bindings.values()})
            entries = receipt.get("contributing_sources")
            if (not isinstance(entries, list)
                    or len(entries) != len(declared_bindings)):
                raise ValueError(
                    "contributing source inventory differs from the "
                    "packaged composition's declared bindings")
            seen_bindings = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(
                        "contributing source receipt is malformed")
                binding_name = str(entry.get("binding"))
                binding = declared_bindings.get(binding_name)
                if binding is None or binding_name in seen_bindings:
                    raise ValueError(
                        f"contributing source receipt names binding "
                        f"{binding_name!r}, which the packaged "
                        "composition does not declare")
                seen_bindings.add(binding_name)
                mapping_role = str(binding["mapping_role"])
                mapping_record = entry.get("mapping")
                recorded_sha256 = (
                    mapping_record.get("sha256")
                    if isinstance(mapping_record, dict) else None
                )
                if (recorded_sha256 != binding["mapping_sha256"]
                        or recorded_sha256
                        != contributing_sha.get(mapping_role)):
                    raise ValueError(
                        f"contributing source {binding_name!r} mapping "
                        "hash differs from the packaged pin: the decode "
                        "did not borrow through the mapping this "
                        "distribution ships")
                if entry.get("source_id") != binding["source_id"]:
                    raise ValueError(
                        f"contributing source {binding_name!r} names a "
                        "different source than the packaged composition")
                entry_alignment = entry.get("alignment")
                if (not isinstance(entry_alignment, dict)
                        or entry_alignment.get("status") != "PASS"):
                    raise ValueError(
                        f"contributing source {binding_name!r} alignment "
                        "did not pass")
                data_rows = entry.get("data")
                declared_rows = supplements[str(binding["data_role"])]
                def portable_rows(rows):
                    if not isinstance(rows, list) or not rows:
                        raise ValueError("contributing source data inventory is empty")
                    return [(receipt_basename(row.get("path")), row.get("sha256"))
                            for row in rows if isinstance(row, dict)]
                if (not isinstance(data_rows, list)
                        or len(data_rows) != len(declared_rows)
                        or portable_rows(data_rows) != portable_rows(declared_rows)):
                    raise ValueError(
                        f"contributing source {binding_name!r} data does "
                        "not match the manifest's supplement inventory")
                provenance_record = entry.get("provenance")
                declared_provenance = manifest["provenance"].get(
                    str(binding["provenance_role"]))
                if (not isinstance(provenance_record, dict)
                        or not isinstance(declared_provenance, dict)
                        or provenance_record.get("sha256") != declared_provenance["sha256"]
                        or receipt_basename(provenance_record.get("path"))
                        != receipt_basename(declared_provenance["path"])):
                    raise ValueError("contributing source provenance differs")
        for key in ("member", "member_identity"):
            if alignment.get(key) != manifest.get(key):
                raise ValueError("mapped composition member/alignment receipt differs")
        source_member = manifest.get("member")
    if receipt.get("soil_layers") != composition_document.get("soil_layers"):
        raise ValueError(f"mapped {source} soil-layer receipt differs")
    frames = receipt.get("frames")
    if (receipt.get("frame_count") != len(valid_times)
            or receipt.get("valid_times") != valid_times
            or not isinstance(frames, list) or len(frames) != len(valid_times)):
        raise ValueError(f"mapped {source} canonical frame inventory differs")
    for index, frame in enumerate(frames):
        if (not isinstance(frame, dict)
                or set(frame) != {
                    "header_sha256", "terrain_sha256", "field_count"}
                or isinstance(frame.get("field_count"), bool)
                or not isinstance(frame.get("field_count"), int)
                or frame.get("field_count") <= 0):
            raise ValueError(
                f"mapped {source} canonical frame {index} is malformed")
        _require_digest(
            frame.get("header_sha256"),
            f"mapped {source} frame {index} header sha256")
        _require_digest(
            frame.get("terrain_sha256"),
            f"mapped {source} frame {index} terrain sha256")
    return MappingProxyType({
        "mapped_mapping": mapping_path,
        "mapped_composition": composition_path,
        **provenance_paths,
    }), MappingProxyType({
        "receipt_content_sha256": expected_receipt_sha256,
        "mapping_sha256": expected_authority_sha256["mapping"],
        "composition_sha256": expected_authority_sha256["composition"],
        "decoder_sha256": decoder_sha256,
        "preprocessing": proof.get("preprocessing"),
        "target_contract": proof.get("target_contract"),
    }), source_member


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


def _posix_digest_keys(digests: object) -> dict[str, object] | None:
    """Read a serialized digest dict with machine-independent keys.

    A repository-relative path is one thing; the separator it was
    spelled with is which machine spelled it.  Returns ``None`` for
    anything that is not a dict, so callers keep their own refusal.

    A dict carrying BOTH spellings of one path is refused rather than
    merged: collapsing it would silently drop one of two digests for
    the same file, which is exactly the disagreement this check exists
    to catch.  No producer emits that; only a hand-edited file can.
    """

    if not isinstance(digests, dict):
        return None
    normalized = {str(key).replace("\\", "/"): value
                  for key, value in digests.items()}
    if len(normalized) != len(digests):
        raise ValueError(
            "HRRR prepared cache decode digests name one file under two "
            "path spellings")
    return normalized


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

    Both digest dicts are read through :func:`_posix_digest_keys` first.
    The producers now emit ``as_posix()`` keys everywhere, but caches
    that 1.8.2 sealed on Windows carry ``gpuwm\\hrrr_forecast.py`` --
    ``str()`` of a relative Path -- and every lookup below missed, so
    the Windows prepared route refused its own caches at the handoff.
    The digests bind the same file BYTES either way, and the separator
    is a property of the machine that wrote the JSON rather than of
    what ran, so normalizing on read keeps those sealed caches
    restorable instead of orphaning them.
    """

    missing = [key for key in _HRRR_IDENTITY_REQUIRED if key not in identity]
    if missing:
        raise ValueError(
            f"HRRR prepared cache source identity is incomplete: {missing}")
    digests = _posix_digest_keys(identity.get("source_sha256"))
    if digests is None:
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
    if digests != _posix_digest_keys(proof.get("source_sha256")):
        raise ValueError(
            "HRRR prepared cache decode identity differs from its proof: "
            "the cache was built by a different ingest than the proof names")
    for key in ("source_cycle", "model_start_time", "source_forecast_hours",
                "model_forcing_hours"):
        if identity.get(key) != proof.get(key):
            raise ValueError(
                f"HRRR prepared cache source identity {key} differs from "
                "the preparation proof")
    if "ingest" in identity or "ingest" in proof:
        ingest = identity.get("ingest")
        if (not isinstance(ingest, dict)
                or set(ingest) != {"soil_texture_downscale"}
                or not isinstance(ingest["soil_texture_downscale"], bool)
                or ingest != proof.get("ingest")):
            raise ValueError("prepared cache ingest settings differ from preparation proof")
    if "static_highres" in identity or "static_highres" in proof:
        if identity.get("static_highres") != proof.get("static_highres"):
            raise ValueError("prepared cache high-resolution settings differ from preparation proof")
        from gpuwm.static.highres_production import parse_static_table
        parse_static_table({"highres": identity.get("static_highres")},
                           source="prepared high-resolution identity", base_dir=Path("."))
    if identity.get("trace_gas_overrides") != proof.get("trace_gas_overrides"):
        raise ValueError("prepared cache trace-gas settings differ from preparation proof")
    return identity


def _validate_source_identity(
        source: str, identity: object, manifest_sha256: str,
        manifest_files: Mapping[str, Mapping[str, object]], proof,
        *, layout: str, mapped_authority: Mapping[str, object] | None = None,
        grid_id: int = 1,
        experiment_config: Path | None = None,
        experiment_config_sha256: str | None = None,
) -> Mapping[str, object]:
    if not isinstance(identity, dict):
        raise ValueError("prepared cache source identity must be an object")
    if source in _MAPPED_SOURCES:
        if mapped_authority is None:
            raise ValueError("mapped source authority is missing")
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
                "prepared cache source identity differs from the mapped authorities")
        allowed = set(expected)
        if "static_highres" in identity:
            # Shared preflight below compares this owner-normalized block
            # with the exact experiment authority for every source family.
            allowed.add("static_highres")
        case_fields = {"preparation_case_policy", "water_temperature_overlay"}
        if case_fields & set(identity):
            if not case_fields <= set(identity):
                raise ValueError("mapped preparation case identity is incomplete")
            if experiment_config is None or experiment_config_sha256 is None:
                raise ValueError("mapped preparation case identity needs the pinned experiment authority")
            from gpuwm.case_data import (
                optional_case_data_from_config, preparation_case_policy)
            from gpuwm.ingest.water_overlay import overlay_file_identity
            data = optional_case_data_from_config(
                experiment_config, expected_sha256=experiment_config_sha256)
            policy = preparation_case_policy(data)
            if identity["preparation_case_policy"] != policy:
                raise ValueError("mapped preparation case policy differs from experiment authority")
            overlay = None if data is None else data.water_temperature_overlay
            expected_overlay = None if overlay is None else overlay_file_identity(overlay)
            if identity["water_temperature_overlay"] != expected_overlay:
                raise ValueError("mapped water-temperature overlay identity differs from its declared bytes")
            allowed.update(case_fields)
        if layout == "mapped-hierarchy-d01-v1":
            allowed.update({
                "target_contract", "nested_source_orography",
                "hierarchy_implementation_sha256", "grid_id"})
            if (identity.get("target_contract")
                    != mapped_authority.get("target_contract")
                    or identity.get("grid_id") != grid_id
                    or not isinstance(identity.get("nested_source_orography"), dict)
                    or not isinstance(
                        identity.get("hierarchy_implementation_sha256"), dict)):
                raise ValueError(
                    "mapped hierarchy cache source identity is incomplete")
        if set(identity) != allowed:
            raise ValueError(
                "mapped prepared cache source identity has an unsupported shape")
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
    receipt_keys = CONDITIONAL_PREPARATION_RECEIPTS
    if layout == HRRR_DIRECT_LAYOUT:
        receipt_keys = SOIL_PREPARATION_RECEIPTS
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
    # Preparation receipts the preparer binds only when they fired.  The
    # comparison below is exact, so a receipt the writer recorded and this
    # list omitted made the front door refuse ITS OWN cache -- the SMCDRY
    # floor fires on most clipped ERA5, which is why real domains at and
    # above 448^2 could not be prepared and read back.
    #
    # Bound from the PROOF, which carries the identical receipt, rather
    # than waved through: a cache whose receipt differs from its proof
    # still refuses, and so does one that records a receipt its proof does
    # not (and vice versa) -- each of those is a genuine inconsistency.
    for key in receipt_keys:
        if key in proof or (isinstance(user, dict) and key in user):
            expected_user[key] = proof.get(key)
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
    admitted_fields = _lbc_field_inventories(cfg)
    scheme_id = int(getattr(cfg, "mp_physics", 0))
    for index, interval in enumerate(intervals):
        window = {
            "start_seconds": float(forcing_hours[index] * 3600),
            "end_seconds": float(forcing_hours[index + 1] * 3600),
        }
        recorded = interval if isinstance(interval, dict) else {}
        if {key: recorded.get(key) for key in window} != window:
            raise ValueError(
                f"prepared cache LBC interval {index} spans "
                f"{recorded.get('start_seconds')}-{recorded.get('end_seconds')} s "
                f"where the forcing has {window['start_seconds']:g}-"
                f"{window['end_seconds']:g} s; re-prepare against this forcing")
        fields = recorded.get("fields")
        if not isinstance(fields, list) or fields not in admitted_fields:
            raise ValueError(
                f"prepared cache LBC interval {index} carries fields "
                f"{fields} where mp_physics={scheme_id} forces "
                f"{' or '.join(str(f) for f in admitted_fields)}; re-prepare "
                "with this configuration")


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


def _validate_mapped_static_proof(
        proof: Mapping[str, object], layout: _PreparedLayout,
        *, source: str, static: Mapping[str, object],
        geometry_receipt: Mapping[str, object], static_sha256: str,
) -> None:
    """Static/geometry half of a packaged mapped bundle's proof.

    ``source`` names the profile in every refusal below.  It is a
    parameter for the same reason the function is no longer named for one
    source: this runs for every packaged mapped profile, and a refusal
    that names the wrong one sends the reader down the wrong route.
    """
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
                f"mapped {source} direct static/geometry proof differs")
        return
    root_static = layout.authority_paths.get("mapped_root_static")
    root_geometry_path = layout.authority_paths.get("mapped_root_geometry")
    if root_static is None or root_geometry_path is None:
        raise ValueError(f"mapped {source} hierarchy root authorities are missing")
    try:
        with np.load(root_static, allow_pickle=False) as archive:
            root_fields = sorted(archive.files)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"mapped {source} hierarchy root static cache is unreadable") from exc
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
            f"mapped {source} hierarchy root static/geometry proof differs")


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
            "rw-wps-mapped"
            if source in _MAPPED_SOURCES else source),
        "native_artifact_manifest": (
            "../hierarchy-artifacts/domain-artifacts.json"),
        "native_artifact_manifest_sha256": _sha256(
            layout.authority_paths["hierarchy_artifact_manifest"]),
    }
    if source in _MAPPED_SOURCES:
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


def _tolerated_absent_run_fields(
        observed: Mapping[str, object], run: dict) -> list[dict]:
    """Drop run fields a header written before them could not carry.

    THE V-12 HOLE, on the route the tolerant table does not cover.  A
    RunConfig field that joins the identity document refuses every
    single-domain bundle prepared before it, because the comparison below
    is a bare equality with one hand-written allowance -- and this has now
    happened five times in this package's history.  The tree route solved
    it with ingest.prepared_cache.DEFAULT_TOLERANT_IDENTITY_FIELDS, whose
    entry means "no code read this at preparation time, so a header
    written without it describes the same prepared state".  That argument
    holds identically here, so this reads THAT table rather than a second
    hand-typed list, and a field registered there is covered on both
    routes in the commit that registers it.

    Two conditions, both required: the field must be ABSENT from the
    prepared header (a header that carries a value is compared to it),
    and the current value must be the not-in-use one from
    ``undelayed_identity_defaults`` (a bundle really running the feature
    was prepared for a different clock and is refused by name).
    """
    from gpuwm.ingest.prepared_cache import (
        DEFAULT_TOLERANT_IDENTITY_FIELDS, _run_config_defaults)

    observed_run = {}
    if isinstance(observed, Mapping):
        domain = observed.get("domain_config")
        if isinstance(domain, Mapping) and isinstance(
                domain.get("run"), Mapping):
            observed_run = domain["run"]
    names = sorted(path[len("run."):]
                   for path in DEFAULT_TOLERANT_IDENTITY_FIELDS
                   if path.startswith("run."))
    if not names:
        return []
    not_in_use = _run_config_defaults(*names)
    overrides = []
    for name in names:
        if name in observed_run or name not in run:
            continue
        if run[name] != not_in_use[f"run.{name}"]:
            continue
        overrides.append({
            "field": f"domain_config.run.{name}",
            "prepared_identity": "field-absent",
            "current_loader_default": run.pop(name),
            "model_state_or_physics_changed": False,
            "reason": (
                "the field joined RunConfig after this bundle was "
                "prepared and holds its not-in-use value, which no code "
                "read at preparation time"),
        })
    return overrides


def _dropped_preparation_inert_run_fields(
        observed: Mapping[str, object], run: dict
) -> tuple[Mapping[str, object], list[dict]]:
    """Drop run fields preparation never reads, from BOTH sides.

    The companion to :func:`_tolerated_absent_run_fields`, and the
    stronger of the two rulings.  Tolerance covers a field ABSENT from
    an older header and holding its not-in-use value; this covers a
    field whose value cannot describe a difference in the artifact at
    all, so it is dropped in both directions and at any value -- which
    is the only thing that lets a bundle prepared under one adaptive
    target be run under another.

    Same table as the tree route
    (``ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS``), so a field
    registered there is covered on both routes by that one entry rather
    than a second hand-typed list here.
    """
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS

    names = sorted(path[len("run."):]
                   for path in PREPARATION_INERT_RUN_FIELDS
                   if path.startswith("run."))
    if not names:
        return observed, []
    observed_run = {}
    if isinstance(observed, Mapping):
        domain = observed.get("domain_config")
        if isinstance(domain, Mapping) and isinstance(
                domain.get("run"), Mapping):
            observed_run = domain["run"]
    overrides = []
    for name in names:
        in_header = name in observed_run
        in_live = name in run
        if not in_header and not in_live:
            continue
        prepared = observed_run[name] if in_header else "field-absent"
        current = run.pop(name) if in_live else "field-absent"
        if prepared == current:
            continue
        overrides.append({
            "field": f"domain_config.run.{name}",
            "prepared_identity": prepared,
            "current_loader_default": current,
            "model_state_or_physics_changed": False,
            "reason": (
                "no code on the preparation path reads this field, so the "
                "prepared artifact is the same under either value"),
        })
    # Stripped unconditionally: the expected side has already had these
    # names popped, so leaving them on the observed side would turn a
    # dropped field into a difference -- the exact refusal this removes.
    stripped = json.loads(_canonical(observed))
    stripped_run = stripped.get("domain_config", {}).get("run", {})
    for name in names:
        stripped_run.pop(name, None)
    return stripped, overrides


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
    # Inert first: it drops from both sides, so what follows compares
    # only fields that can actually describe the prepared artifact.
    observed_cmp, inert = _dropped_preparation_inert_run_fields(
        observed, run)
    absent = _tolerated_absent_run_fields(observed_cmp, run)
    if (absent or inert) and observed_cmp == compatible:
        return observed, MappingProxyType({
            "schema": "gpuwm-prepared-cache-identity-compatibility-v1",
            "status": "COMPATIBLE_LEGACY_DEFAULT",
            "compatibility_overrides": inert + absent,
        })
    removed = run.pop("nest_microphysics_transition", None)
    if (source in _MAPPED_SOURCES and removed == "same-scheme-only"
            and observed_cmp == compatible):
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
    # NAME WHAT DIFFERS.  The sentence above stood alone, and a user
    # reading it could not tell an upgrade from a changed configuration --
    # measured on a bundle prepared one commit earlier, where the whole
    # difference was twelve fields that had not existed yet.
    raise ValueError(
        "prepared cache identity differs from the requested source, static "
        "data, configuration, namelist, or allowed legacy default: "
        + _describe_identity_difference(observed_cmp, compatible))


def _describe_identity_difference(observed, expected, limit: int = 8) -> str:
    """The first few differing identity paths, as ``path (a -> b)``."""
    lines: list[str] = []

    def walk(a, b, path: str) -> None:
        if len(lines) >= limit:
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key, _MISSING), b.get(key, _MISSING),
                     f"{path}.{key}" if path else str(key))
            return
        if a != b:
            lines.append(
                f"{path} (prepared "
                f"{'absent' if a is _MISSING else a!r} -> requested "
                f"{'absent' if b is _MISSING else b!r})")

    walk(observed, expected, "")
    if not lines:
        return "no field-level difference (the documents differ in shape)"
    return "; ".join(lines)


def preflight_prepared_forecast(
        *, source: str, prepared_root: Path, proof_sha256: str,
        source_manifest_sha256: str, prepared_content_sha256: str,
        experiment_config: Path, wps_namelist: Path,
        physics_profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        run_seconds: float, history_interval_seconds: float,
        domain_bundle: Path | None = None,
        tiles=None,
) -> PreparedForecastInputs:
    """Validate every portable preparation authority without importing CuPy.

    ``tiles`` is an optional :class:`~gpuwm.core.streaming.StreamingOptions`
    that REPLACES whatever the hash-bound experiment declares, and it is
    the one execution control this stage accepts from outside the bundle.

    It has to be, on one route.  The native HRRR chain hands this runner
    the authority its PREPARER published, which is rendered from tables
    that stage builds in code -- ``{experiment, projection, shared,
    domain}``, with no ``[tiles]`` among them and no way for a user's
    block to get into one.  So a config that asked to stream was read,
    validated and reported by the front door and then integrated
    resident, because the document that reached this function had never
    heard of the table.  The mode arrives as an argument instead.

    Overlaying rather than publishing is deliberate and is the reason
    this is sound.  ``[tiles]`` contributes NOTHING to the restart
    identity on purpose (:func:`gpuwm.core.streaming.identity_payload_entry`
    returns ``{}``): a checkpoint written resident must resume streamed
    and one written streamed must resume resident, since a forecast that
    outgrew its card resuming on the machine it outgrew is the operation
    the mode exists for.  Rendering the table into the hash-bound
    document would bind the execution mode into the prepared bundle's
    identity and refuse exactly that.
    """

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
            if source in _MAPPED_SOURCES
            else "source-input-manifest.json"),
        "portable source manifest")
    experiment_config = _require_file(experiment_config, "experiment config")
    wps_namelist = _require_file(wps_namelist, "WPS namelist")

    actual_proof_sha256 = _sha256(proof_path)
    if actual_proof_sha256 != proof_sha256:
        raise _proof_digest_refusal(
            proof_path, proof_sha256, actual_proof_sha256)
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
    # NO streaming refusal here.  This route wires a streamed-domain builder
    # (streaming.builders_for_tree, below), so mode = 'on' is something it
    # can do rather than something to refuse at admission.  The refusal that
    # used to stand here outlived the wiring by a release: it was written
    # when the route passed no builders, and it went on rejecting the one
    # mode that asks for streaming unconditionally -- with a message saying
    # the route "wires no streamed-domain builder" -- while mode = 'auto'
    # went through the very builder it said did not exist.
    # refuse_unrouted_streaming is still the right answer for a route that
    # genuinely reads nothing -- see its docstring -- but `gpuwm run` is no
    # longer such a route: it wires builders_for_tree on its tree arm and
    # standalone_domain_builder on its single-domain arm, so its own copy
    # of this refusal is gone too.
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
    if source in _MAPPED_SOURCES:
        mapped_paths, mapped_authority, source_member = (
            _validate_packaged_mapped_evidence(
                prepared_root=prepared_root, proof=proof, manifest=manifest,
                manifest_sha256=source_manifest_sha256,
                experiment_config=experiment_config,
                wps_namelist=wps_namelist, source=source))
    else:
        for role, actual in (
                ("experiment_config", experiment_config),
                ("wps_namelist", wps_namelist)):
            spec = manifest_files[role]
            # The manifest retains its validated publication name; an
            # explicit caller path is only a locator for those exact bytes.
            # Keep that actual path below so later mutation checks rehash it.
            if spec["sha256"] != _sha256(actual):
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
            or (source == "era5" and cadence_hours < 1) \
            or (source in _MAPPED_SOURCES
                and cadence_hours < 1):
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

    if source not in _MAPPED_SOURCES \
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
        mapped_authority=mapped_authority,
        experiment_config=experiment_config,
        experiment_config_sha256=_sha256(experiment_config))
    if layout.kind in _HIERARCHY_LAYOUTS \
            and source_identity.get("grid_id") != 1:
        raise ValueError("hierarchy prepared-cache source identity is not d01")
    if "ingest" in source_identity:
        from gpuwm.ingest.soil_downscale import declared_soil_texture_downscale
        expected_ingest = {"soil_texture_downscale":
                           declared_soil_texture_downscale(experiment_config)}
        if source_identity["ingest"] != expected_ingest:
            raise ValueError("prepared cache ingest settings differ from experiment authority")
    from gpuwm.static.highres_production import load_static_highres, static_highres_identity
    requested_highres = load_static_highres(experiment_config)
    if ("static_highres" in source_identity
            or (requested_highres is not None and requested_highres.enabled)):
        if source_identity.get("static_highres") != static_highres_identity(requested_highres):
            raise ValueError("prepared cache high-resolution settings differ from experiment authority")
    if "trace_gas_overrides" in source_identity:
        from gpuwm.case_data import trace_gas_overrides_from_config
        if source_identity["trace_gas_overrides"] != trace_gas_overrides_from_config(
                experiment_config):
            raise ValueError("prepared cache trace-gas settings differ from experiment authority")
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
    if source in _MAPPED_SOURCES:
        _validate_mapped_static_proof(
            proof, layout, source=source, static=static,
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
        if source not in _MAPPED_SOURCES:
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
        without_export = _optional_stock_wrf_export(proof, export)
        if without_export:
            export_source_receipt = MappingProxyType(dict(export))
        else:
            export = proof.get("export")
            expected_dimensions = {
                "nx": int(exp.root.run.nx),
                "ny": int(exp.root.run.ny),
                "nz": int(exp.root.run.nz),
            }
            expected_export_schema = (
                "gpuwm-native-direct-wrf-export-v3"
                if front_door_physics is not None
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
            if source not in _MAPPED_SOURCES:
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
        if _optional_stock_wrf_export(proof, dict(layout.wrf_manifest)):
            export_source_receipt = layout.wrf_manifest
        else:
            export_source_receipt = _validate_hierarchy_wrf_authority(
                layout, source=source, source_exp=source_exp,
                forcing_hours=forcing_hours,
                boundary_interval_seconds=boundary_interval_seconds,
                source_manifest_sha256=source_manifest_sha256,
                decoder_sha256=(
                    mapped_authority["decoder_sha256"]
                    if source in _MAPPED_SOURCES else
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
    if tiles is not None:
        # LAST, after every identity comparison above.  [tiles] is not a
        # domain field and could not move one of them, but an execution
        # control that is applied before the checks it cannot affect is
        # an execution control someone will later assume was checked.
        declared = getattr(exp, "tiles", None)
        if declared is not None and declared.enabled and declared != tiles:
            # Cannot happen on the HRRR route -- its published authority
            # has no [tiles] table to declare.  Said out loud anyway,
            # because a caller supplying the mode from outside while the
            # bundle carries its own is two authorities disagreeing about
            # how this forecast integrates, and picking one silently is
            # the failure this sentence exists to prevent.
            print(f"prepared forecast: --tiles mode = '{tiles.mode}' "
                  "replaces the [tiles] the hash-bound experiment "
                  f"declares (mode = '{declared.mode}'); the flag is the "
                  "later and more specific statement, and [tiles] binds "
                  "no identity either way", file=sys.stderr)
        exp = replace(exp, tiles=tiles)
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


def _consume_due_native_refl_10cm(state, ticks: int, consumer, *,
                                  domain_start_ticks: int = 0):
    """Consume the scheme-native field staged by an output-due MP call.

    ``domain_start_ticks`` is the DOMAIN's own start tick, not the
    experiment's: the frame due there is the analysis frame, which
    precedes every step of that domain and therefore every stash.  It is
    0 on this runner -- a single domain starts with its experiment -- and
    the parameter exists so the shared predicate
    (:func:`gpuwm.core.refl.refl_10cm_stash_is_due`) is asked the same
    question here as at the tree seam, where a nest that activates later
    made the absolute-0 reading kill the run at its first frame (#205).
    """
    from gpuwm.core.refl import refl_10cm_stash_is_due

    if (refl_10cm_stash_is_due(ticks,
                               domain_start_ticks=domain_start_ticks)
            and state.qv is not None
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


class _LandingObservers:
    """Fan the wrfout landing hook out to more than one consumer.

    ``PerDomainWrfoutWriters.attach_progress_callback`` reads ONE
    ``output_committed`` attribute off what it is handed and assigns it
    to every writer's ``landing_observer``, unconditionally -- so a
    second consumer cannot be added by calling attach twice.  The
    second call would overwrite the first, silently, and the run would
    keep publishing frames with the earlier consumer no longer watching.

    That matters here because this runner now has two: the observer of a
    caller hosting it in-process (``gpuwm run-plan``), and its OWN early
    render when the command line asked for one.  Both are given the
    frame; neither can suppress the other; and a consumer that raises
    does not take the writer thread or its sibling down with it, which
    is the same contract the writer keeps for a single one.
    """

    def __init__(self, *sinks):
        self._sinks = tuple(sink for sink in sinks if sink is not None)

    def __bool__(self) -> bool:
        return bool(self._sinks)

    def output_committed(self, **event) -> None:
        for sink in self._sinks:
            try:
                sink(**event)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass


# ---------------------------------------------------------------------------
# which initialization road a STREAMED forecast takes
# ---------------------------------------------------------------------------
#
# A streamed forecast integrates out of a pinned host store and never reads
# the resident ``DomainState`` again: ``gpuwm.core.streaming.attach`` copies
# it carrier by carrier into that store and every sweep afterwards writes only
# the store.  Building the state first is nevertheless what this route has
# always done, and it is the CEILING.  MEASURED on a 16 GB card at nz = 49,
# the bare state costs 11 276.5 B per column and the prepared case -- state
# plus physics plus geography plus boundary tables -- about 15 780, so
# 1024 x 1024 is refused inside ``initialize_physics`` while the streamed
# forecast that state would have fed needs about 6 GiB and runs comfortably.
# The streamed engine on the same card has stepped 2624 x 2624.
#
# ``gpuwm.ingest.prepared_store.store_from_prepared_cache`` fills the same
# store one ROW SLAB at a time -- slab-height state, physics attached to THAT,
# scattered into full-domain pinned host arrays -- so no domain-shaped device
# array is ever allocated and the domain the card can carry stops being a
# property of the card.  ``--stream-init`` chooses between the two roads.
#
# ``auto`` prefers the RESIDENT road wherever it comfortably fits, and that
# preference is deliberate rather than conservative: the resident road is the
# one every existing receipt was written by, and the two roads' agreement is
# proven by running a domain small enough for both and comparing frames.  The
# store-direct road is taken when the priced resident state does not fit,
# which is precisely the case the resident road cannot serve at all.

#: MEASURED bytes per column at nz = 49 on this route
#: (``Downloads/node1-streaming/STREAMED-CEILING.md`` section 2): a bare
#: ``DomainState`` costs 11 276.5 B/column, and the prepared case -- the same
#: state plus everything ``initialize_prepared_physics`` puts on the card
#: beside it (the physics carriers, the geography inventory and the lateral
#: boundary tables) -- about 15 780 B/column.
_MEASURED_BARE_STATE_BYTES_PER_COLUMN = 11276.5
_MEASURED_PREPARED_BYTES_PER_COLUMN = 15780.0

#: The ``nz`` both per-column measurements were taken at.  They are scaled
#: LINEARLY in ``nz`` off this reference, which is the honest reading of one
#: measurement at one height: the prepared case is overwhelmingly 3-D arrays,
#: and the 2-D surface inventory that does not scale is a small enough share
#: that pretending it does costs a slight OVER-price below nz = 49 and a
#: slight under-price above it -- both inside the fit fraction's margin.
_MEASURED_BYTES_PER_COLUMN_NZ = 49

#: What the prepared case costs as a multiple of the state the cache carries
#: (1.40).  DERIVED from the two measurements above rather than written as a
#: bare 1.4, so that a later measurement moves the number and the evidence for
#: it in one edit and neither can drift away from the other.
#:
#: THIS RATIO IS NOT THE PRICE OF THE RESIDENT ROAD, and reading it as one is
#: the defect the pricing below was rebuilt to remove.  It relates the prepared
#: case to a BARE ``DomainState``; the cache's ``state/*`` manifest is not a
#: bare DomainState but the far smaller ``STATE_SERIALIZED_ATTRS`` subset --
#: the dynamics and microphysics prognostics only, with none of the RK and
#: dycore working arrays a ``DomainState`` allocates beside them and none of
#: what ``initialize_prepared_physics`` then puts on the card.  MEASURED on the
#: 1024 x 1024 x 49 gate cache: ``state/*`` is 4 945 485 824 B, which is
#: 4 716.4 B per column against the bare state's 11 276.5 -- the manifest
#: undercounts the state alone by 2.39x, and ``manifest x 1.40`` therefore
#: priced the whole prepared case at 6.45 GiB when the resident road went on to
#: die with 16 276 726 272 B allocated.
_PREPARED_RESIDENT_HEADROOM = (_MEASURED_PREPARED_BYTES_PER_COLUMN
                               / _MEASURED_BARE_STATE_BYTES_PER_COLUMN)

#: How much of the card's FREE memory the streamed run's whole PEAK may claim
#: before ``auto`` stops calling the fit comfortable.  A margin for what is
#: left unpriced -- the NVRTC module images and the allocator's own
#: fragmentation -- and no longer a stand-in for the tile buffers themselves.
#:
#: IT USED TO BE THAT STAND-IN, AND THAT WAS THE DEFECT.  The comment here
#: read "20% of a 16 GB card is 3.2 GiB, which covers the default buffer set
#: with room to spare", and on a 16 GB card it did.  But the tile buffers and
#: the radiation transient are ABSOLUTE costs and a fraction is a RELATIVE
#: one, so the margin shrank exactly where the card had least room to give:
#: 20% of a 10 GiB card is 1.9 GiB against a default buffer set of 5.13 GiB
#: and a 2.74 GiB radiation transient.  A 424 x 424 x 49 forecast therefore
#: priced its 2.64 GiB resident state as a comfortable fit, took the resident
#: road, and peaked at 10.51 GiB on 9.50 GiB of free memory.  Under WDDM that
#: over-commit does not raise -- the driver pages it -- so the forecast did
#: not refuse, it just stopped completing steps.
#:
#: Both of those costs are now priced explicitly (:func:`_streamed_peak_bytes`)
#: and added to the resident state before this fraction is applied, so what is
#: left for the fraction to cover is only the genuinely unpriced remainder.
_AUTO_RESIDENT_FIT_FRACTION = 0.80

#: ``--stream-init``'s values.  ``auto`` is the default and is the only one
#: that decides anything; the other two force a road so that a gate can run
#: both over the same domain and compare.
STREAM_INIT_CHOICES = ("auto", "resident", "store")

#: The schema/status ``store_from_prepared_cache`` stamps on its receipt.
#: Pinned here for the same reason ``_validate_restored_cache_receipt`` pins
#: ``PREPARED_CACHE_SCHEMA``: the receipt is re-checked against the caller's
#: content digest, and a receipt of some other shape must not be able to
#: satisfy that check by carrying the right key by accident.
_PREPARED_STORE_SCHEMA = "gpuwm-prepared-store-v1"

_GIB = float(1 << 30)


def _prepared_state_manifest_bytes(reader) -> int:
    """The EXACT device cost of restoring a prepared cache's prognostics.

    ``restore_prepared_cache`` allocates one ``DomainState`` and then runs
    ``target[...] = cp.asarray(host)`` once per ``state/*`` array, so the
    device bytes that restore takes for the trajectory are the sum of those
    arrays' own ``nbytes`` -- a number the cache HEADER already carries per
    array, and one the reader already checked against that array's ``shape``
    and ``dtype`` when it opened the bundle.

    Read, never probed.  The question is asked before anything has been
    allocated, and an answer obtained by trying the allocation would be the
    allocation it exists to avoid.
    """

    return int(sum(int(spec["nbytes"])
                   for key, spec in reader.arrays.items()
                   if str(key).startswith("state/")))


def _streamed_peak_bytes(cfg, decision) -> int | None:
    """What the STREAMED half of this forecast puts on the card, at its peak.

    The resident road does not end at the resident state.  A forecast that
    takes it and then streams allocates the tile buffers ON TOP of that state
    -- ``attach`` copies out of the state and nothing frees it
    (``gpuwm.core.streaming.refresh_from_store``) -- and meets the radiation
    transient on top of THAT.  The card has to hold all three at once, and
    this is the second and third terms.

    Two numbers, each already measured and already carried by this project;
    neither was being read by the road decision:

    * ``streamed_envelope(...).vram_bytes`` -- ``tilestream.autoplan``'s
      footprint for the tiling this decision actually selected: the CUDA
      context, the per-process fixed cost and ``nbuffers`` compute windows.
    * ``footprint_for(cfg).radiation_transient_bytes`` -- the RRTMGP call's
      per-process working set, MEASURED at +2.74 GiB over steady state and
      zero on the rungs that run no radiation.  It is allocated and freed
      inside one radiation step, so it never appears in a steady-state
      reading; it is nevertheless on the card at the instant it matters, and
      the first radiation call is ``itimestep == 1``.

    ``None``, never zero, when this cannot be priced -- a run that does not
    stream, or a decision this function was handed without a tiling on it.
    Zero is the answer that reinstates the defect, so the caller is told the
    term is missing and decides for itself rather than being handed a
    confident nought.
    """

    if cfg is None or decision is None or not getattr(decision, "stream", False):
        return None
    try:
        from gpuwm.core import streaming

        envelope = streaming.streamed_envelope(cfg, None, decision=decision)
        if envelope is None:
            return None
        # The envelope carries BOTH terms since the estimate surfaces
        # learned the reservation, so this reads its peak rather than
        # re-adding the transient beside it -- one place the sum lives.
        return int(envelope.peak_vram_bytes)
    except Exception:
        # The road decision must still be TAKEN.  An unpriceable streamed term
        # leaves the pre-existing resident-only rule standing, which is the
        # behaviour this route had before the term existed, rather than
        # refusing a forecast because an estimate could not be formed.
        return None


def _stream_init_pricing(state_bytes: int, free_bytes: int, total_bytes: int,
                         *, columns: int | None = None,
                         nz: int | None = None,
                         streamed_bytes: int | None = None,
                         ) -> dict[str, object]:
    """Price the resident road against the card, with no device work at all.

    Separated from the device query so the RULE is testable on a machine with
    no GPU: everything below is arithmetic over a handful of integers, and
    :func:`_free_device_bytes` is the only part that needs a card.

    TWO ESTIMATES, AND THE LARGER WINS, because they bound different things
    and each is blind where the other sees.

    ``manifest`` -- the cache's own ``state/*`` bytes times the physics
    headroom -- is the only term that knows what THIS cache holds, so a
    configuration carrying far more prognostics than the one measured below
    (a bigger microphysics scheme, more tracers) raises the price through it.
    On its own it is not a price for the resident road at all: the manifest is
    the SERIALIZED subset, not a ``DomainState``, and it undercounted the
    1024 x 1024 gate case by 2.38x -- 6.45 GiB priced, 15.16 GiB allocated at
    the refusal.  That is the number ``auto`` used to say ``fits: true`` on and
    then die inside ``initialize_physics``.

    ``measured`` -- ``_MEASURED_PREPARED_BYTES_PER_COLUMN`` over this domain's
    columns, scaled by ``nz`` -- is the only term that knows what the ROUTE
    costs: the state plus everything ``initialize_prepared_physics`` builds
    beside it.  It is a whole-route measurement rather than a model, and it
    predicts the refusal it was checked against to 0.4%: 15.41 GiB priced
    against 15.35 GiB at the out-of-memory throw.

    ``max`` rather than a sum: they overlap almost entirely -- both are
    dominated by the same prognostics -- so adding them would double-count the
    state and refuse domains that fit.  ``columns``/``nz`` absent leaves the
    manifest term alone, which is the pre-existing rule; every caller in this
    module supplies them.
    """

    manifest_priced = int(round(float(state_bytes)
                                * _PREPARED_RESIDENT_HEADROOM))
    measured_priced = None
    if columns and nz:
        measured_priced = int(round(
            _MEASURED_PREPARED_BYTES_PER_COLUMN * float(columns)
            * float(nz) / float(_MEASURED_BYTES_PER_COLUMN_NZ)))
    priced = (manifest_priced if measured_priced is None
              else max(manifest_priced, measured_priced))
    allowance = int(float(free_bytes) * _AUTO_RESIDENT_FIT_FRACTION)
    # THE PEAK, not the state.  ``max`` was right for the two resident terms
    # because they price the same bytes two ways; this is a SUM because the
    # tiles and the radiation transient are different bytes on the same card
    # at the same instant as the state.  ``None`` leaves the pre-existing
    # resident-only rule exactly as it was.
    peak = priced if streamed_bytes is None else priced + int(streamed_bytes)
    return {
        "state_manifest_bytes": int(state_bytes),
        "physics_headroom_factor": round(_PREPARED_RESIDENT_HEADROOM, 4),
        "manifest_priced_bytes": manifest_priced,
        "measured_priced_bytes": measured_priced,
        "priced_from": ("manifest" if measured_priced is None
                        or manifest_priced >= measured_priced
                        else "measured-per-column"),
        "columns": None if not columns else int(columns),
        "nz": None if not nz else int(nz),
        "priced_resident_bytes": priced,
        "streamed_peak_bytes": (None if streamed_bytes is None
                                else int(streamed_bytes)),
        "resident_peak_bytes": int(peak),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
        "fit_fraction": _AUTO_RESIDENT_FIT_FRACTION,
        "fit_allowance_bytes": allowance,
        "fits": bool(peak <= allowance),
    }


def _free_device_bytes() -> tuple[int, int]:
    """``(free, total)`` device memory as the DRIVER reports it right now.

    Not the planner's budget and not a nameplate capacity: the question the
    pricing asks is whether this allocation would succeed on this card in this
    process, and the only honest source for that is what is free at the
    instant the decision is taken.
    """

    import cupy as cp

    free, total = cp.cuda.Device().mem_info
    return int(free), int(total)


def _choose_stream_init_road(mode: str, *, decision, reader, cfg=None,
                             device_memory=None) -> tuple[str, dict | None]:
    """Which road this forecast initializes on, and the numbers behind it.

    Returns ``("resident", None)`` -- today's road, with nothing at all
    changed and no device queried -- for every run that does not STREAM.  The
    flag has meaning only for a streamed forecast: with ``[tiles]`` off the
    resident state is not an intermediary the run could skip, it is the
    domain, and there is no second road to choose.

    ``resident`` and ``store`` force a road.  ``store`` is forced by the
    bit-parity gate, which needs the store-direct road to run on a domain that
    WOULD have fitted so the two can be compared frame for frame; ``resident``
    on a domain that cannot fit fails exactly where it fails today, inside the
    allocation, rather than being silently rescued into a mode the caller
    asked not to be in.
    """

    if mode not in STREAM_INIT_CHOICES:
        raise ValueError(
            f"--stream-init must be one of {', '.join(STREAM_INIT_CHOICES)}, "
            f"got {mode!r}")
    streams = decision is not None and bool(decision.stream)
    if not streams:
        if mode == "store":
            raise ValueError(
                "--stream-init store asks this forecast to build its domain "
                "straight into a pinned host store and integrate out of it, "
                "but this run does not stream"
                + ("" if decision is None else f" ({decision.explain()})")
                + ".  The store is only ever read by the tiling sweep, so a "
                "resident forecast initialized into one would have no domain "
                "to step.  Configure [tiles] (mode = 'on', or 'auto' on a "
                "domain that does not fit) or drop the flag.")
        return "resident", None

    pricing = _stream_init_pricing(
        _prepared_state_manifest_bytes(reader),
        *(device_memory if device_memory is not None
          else _free_device_bytes()),
        columns=(None if cfg is None else int(cfg.nx) * int(cfg.ny)),
        nz=(None if cfg is None else int(cfg.nz)),
        # WHAT THE RESIDENT STATE WILL HAVE TO SHARE THE CARD WITH.  Priced
        # here rather than left to the fit fraction: see
        # ``_AUTO_RESIDENT_FIT_FRACTION`` for the 10 GiB card this omission
        # paged into the ground.
        streamed_bytes=_streamed_peak_bytes(cfg, decision))
    priced = pricing["priced_resident_bytes"] / _GIB
    free = pricing["device_free_bytes"] / _GIB
    total = pricing["device_total_bytes"] / _GIB
    manifest = pricing["state_manifest_bytes"] / _GIB
    allowance = pricing["fit_allowance_bytes"] / _GIB
    # WHICH TERM SET THE PRICE, in the sentence itself.  A receipt that
    # printed only the winning number would leave a reader unable to tell a
    # cache-driven price from a route-driven one, and the two are answers to
    # different questions.
    if pricing["priced_from"] == "measured-per-column":
        priced_as = (
            f"the prepared resident case prices at {priced:.2f} GiB "
            f"({pricing['columns']} columns x {int(pricing['nz'])} levels at "
            f"{_MEASURED_PREPARED_BYTES_PER_COLUMN:.0f} B/column/"
            f"{_MEASURED_BYTES_PER_COLUMN_NZ} levels MEASURED for this route, "
            f"over {manifest:.2f} GiB of state/* in the cache manifest x "
            f"{_PREPARED_RESIDENT_HEADROOM:.2f} = "
            f"{pricing['manifest_priced_bytes'] / _GIB:.2f} GiB) against "
            f"{free:.2f} GiB free of {total:.2f} GiB on the card")
    else:
        priced_as = (
            f"the prepared resident case prices at {priced:.2f} GiB "
            f"({manifest:.2f} GiB of state/* in the cache manifest x "
            f"{_PREPARED_RESIDENT_HEADROOM:.2f} measured physics headroom, "
            f"over this route's measured "
            f"{_MEASURED_PREPARED_BYTES_PER_COLUMN:.0f} B/column) against "
            f"{free:.2f} GiB free of {total:.2f} GiB on the card")
    # THE OTHER TWO TERMS, said out loud whenever they were priced.  A receipt
    # that printed only the resident number would leave a reader unable to see
    # why a state that plainly fits was still sent to the store.
    if pricing["streamed_peak_bytes"] is not None:
        streamed = pricing["streamed_peak_bytes"] / _GIB
        peak = pricing["resident_peak_bytes"] / _GIB
        priced_as += (
            f", and the streamed run puts {streamed:.2f} GiB more beside it "
            f"(the tile buffers this decision selected plus the measured "
            f"radiation transient, both on the card while the resident state "
            f"is), for a {peak:.2f} GiB peak")
    if mode == "resident":
        road = "resident"
        why = (f"--stream-init resident was asked for, so the resident state "
               f"is built whether or not it fits; {priced_as}")
    elif mode == "store":
        road = "store"
        why = (f"--stream-init store was asked for, so the store is filled "
               f"slab by slab and no domain-shaped device array is "
               f"allocated; {priced_as}")
    elif pricing["fits"]:
        road = "resident"
        why = (f"--stream-init auto: {priced_as}, which fits inside the "
               f"{allowance:.2f} GiB this rule allows "
               f"({_AUTO_RESIDENT_FIT_FRACTION:.0%} of free), so the resident "
               f"road runs -- it is the faster one and it is the one the "
               f"parity proof is written against")
    else:
        road = "store"
        why = (f"--stream-init auto: {priced_as}, which does NOT fit inside "
               f"the {allowance:.2f} GiB this rule allows "
               f"({_AUTO_RESIDENT_FIT_FRACTION:.0%} of free), so the store is "
               f"filled slab by slab and no domain-shaped device array is "
               f"allocated")
    receipt = {
        "requested": mode,
        "road": road,
        "why": why,
        "measured_bare_state_bytes_per_column":
            _MEASURED_BARE_STATE_BYTES_PER_COLUMN,
        "measured_prepared_bytes_per_column":
            _MEASURED_PREPARED_BYTES_PER_COLUMN,
        **pricing,
    }
    return road, receipt


def _validate_store_bundle_receipt(
        receipt: Mapping[str, object], expected_content_sha256: str,
) -> None:
    """Recheck the caller's content pin immediately after the store is built.

    :func:`_validate_restored_cache_receipt`'s store-direct twin, at the same
    instant and for the same reason: the loader that just read the bundle
    states which bytes it read, and that statement is compared against the
    digest the caller hash-bound before the run began.  A cache swapped
    between preflight and load is refused here rather than integrated.
    """

    if (not isinstance(receipt, Mapping)
            or receipt.get("schema") != _PREPARED_STORE_SCHEMA
            or receipt.get("status") != "LOADED"
            or receipt.get("content_sha256") != expected_content_sha256):
        raise ValueError(
            "prepared store differs from the caller-pinned cache content")


def _validate_store_direct_vertical_contract(reader, cfg, bundle) -> None:
    """The host-side cache checks the slab loader does not make itself.

    ``restore_prepared_cache`` validates the vertical contract while it has
    the whole domain in hand: the coordinate arrays' shapes against ``nz``,
    and the explicit eta grid against ``p_top``.  Neither needs a device or a
    domain -- both are functions of the cache header and the 1-D coordinate
    the bundle already carries -- so both are made here rather than dropped
    on the way to the store.  What genuinely CANNOT be reproduced is named in
    the receipt instead; see ``_store_direct_gaps``.
    """

    from gpuwm.ingest.prepared_cache import PreparedCacheMismatchError
    from gpuwm.vertical_contract import (
        validate_coordinate_shapes, validate_explicit_eta_grid,
    )

    metadata = reader.header["metadata"]
    coord_shapes = {
        name: reader.arrays[f"coord/{name}"]["shape"]
        for name in metadata["coord_arrays"]
        if f"coord/{name}" in reader.arrays
    }
    try:
        validate_coordinate_shapes(
            coord_shapes, nz=cfg.nz, context="prepared-store load")
        validate_explicit_eta_grid(
            bundle.coord.znw, nz=cfg.nz, p_top=bundle.base.p_top,
            context="prepared-store load")
    except (TypeError, ValueError) as exc:
        raise PreparedCacheMismatchError(str(exc)) from exc


def _store_direct_gaps(reader) -> dict[str, object]:
    """What the store-direct road does NOT check, said out loud in a receipt.

    Named rather than quietly dropped, because a check that stops running
    without saying so is indistinguishable from a check that keeps passing.
    One entry only, and it is a real one: ``restore_prepared_cache`` rebuilds
    the domain's setup arrays and compares ``setup_fingerprint(state)`` with
    the cache's, which needs a domain-shaped state -- exactly the object this
    road exists not to build.

    What still holds in its place is stated beside it rather than implied: the
    reader verified the header's content digest and every array against its
    own SHA-256 as it was read, and the cache identity was compared against
    the caller's before a byte was loaded, so the BYTES and the experiment
    they belong to are both pinned.  What is not re-derived is that the active
    config reconstructs the same map factors, Coriolis and base state from
    them.
    """

    return {
        "setup_fingerprint": {
            "checked": False,
            "why": ("setup_fingerprint is taken over a whole domain's "
                    "reconstructed setup arrays, and this road never builds "
                    "a domain-shaped state"),
            "instead": ("the cache identity, the header content digest and "
                        "every array's own SHA-256 were verified on the way "
                        "into the store"),
            "cache_content_sha256": reader.content_sha256,
        },
    }


def _store_health_auxiliaries(bundle, cfg) -> dict[str, object]:
    from gpuwm.core.health import prepared_store_health_auxiliaries

    return prepared_store_health_auxiliaries(bundle, cfg)


def _store_full_state_health(bundle, cfg, *, phase: str) -> dict[str, object]:
    """The full-state descriptor gate over the store, as a health record.

    ARMED, where this road used to declare the gate unarmed.  The reason it
    was unarmed was true and is not a reason to leave it so: the CUDA gate is
    one block per whole FIELD and needs a domain-shaped device state, which
    this road never builds -- but the domain exists, in the pinned host store,
    and :func:`gpuwm.core.health.validate_store_fields` runs the production
    rule set over it there.  The instrument differs from the resident road's
    (NumPy over host memory rather than one kernel launch over device memory);
    the RULES do not, because both sides read them out of
    :func:`gpuwm.core.health.rule_for_field`.

    Same key shape as the resident road's ``vars(ValidationReport)`` so that
    one reader handles both receipts, with the coverage record added rather
    than substituted: what the gate saw is a count, and what it did not see is
    a list of names.
    """

    from gpuwm.core.health import validate_store_fields
    from gpuwm.core.streaming import streamed_store_inventory

    template = bundle.template
    report, coverage = validate_store_fields(
        template,
        # The RUN's own inventory rule, harvested off the same template the
        # descriptors are collected from, so the two walkers join by object
        # identity.  ``streamed_store_inventory`` and not the plain carrier
        # inventory for the reason ``store_from_prepared_cache`` gives: it is
        # the rule the store itself was built by, and REFL_10CM is the key the
        # two disagreed on.
        streamed_store_inventory()(template, None),
        bundle.store,
        domain_shape=(int(cfg.ny), int(cfg.nx)),
        auxiliaries=_store_health_auxiliaries(bundle, cfg),
        # The DOMAIN's model lid, from the same base object
        # ``DomainState.load_base`` reads it out of on the resident road.
        # The theta ceiling is that lid carried down the dry adiabat, so a
        # template that never loaded a base would otherwise gate a deep-top
        # domain against the 100 hPa reference ceiling and refuse it.
        p_top=getattr(bundle.base, "p_top", None),
        phase=phase)
    record = _strict_json(vars(report))
    record["armed"] = True
    record["instrument"] = (
        "full-state descriptor gate over the pinned host store "
        "(gpuwm.core.health.validate_store_fields)")
    record["covers"] = (
        f"{coverage['fields_checked']} gated fields over the whole domain, "
        "each under the rule gpuwm.core.health.rule_for_field gives it -- the "
        "same per-field bounds the resident road's kernel applies")
    record["coverage"] = {
        key: value for key, value in coverage.items() if key != "covered"}
    return record


#: How long model step 1 may run before the runner says out loud that
#: something is being compiled.
#:
#: MEASURED on the reference box: a healthy first step is 1.2-1.3 s and
#: the steps after it are 0.13 s, while the compile this exists to name
#: is 52 s.  Fifteen seconds is ten times any healthy first step seen
#: and a third of the shortest compile seen, and it matches the cadence
#: of `gpuwm go`'s own heartbeat so the two lines do not race.
FIRST_STEP_STALL_SECONDS = 15.0


class _FirstStepStallWatch:
    """Say the compile is happening WHILE it is happening.

    THE HOLE THE CENSUS ALONE DOES NOT CLOSE, measured 2026-08-16.
    Predicting the compile from the kernel cache works on a genuinely
    cold box and fails in the chain: `gpuwm go`'s preprocessing stage
    uses the GPU too, so by the time the forecast runner starts, the
    cache already holds entries for this card -- and a reading taken
    then says "warm" even though the forecast's own kernels are all
    missing.  Staging 200 sm_120 entries against an sm_86 card and
    running the chain reproduced exactly that: silent notice, 52 s of
    compilation inside model step 1.

    So this does not predict.  It measures: if step 1 has not finished
    after :data:`FIRST_STEP_STALL_SECONDS`, the run says so, names what
    the kernel cache looked like at launch, and flips the published
    status -- and it cannot false-positive on a fast run, because a
    fast run disarms it.

    One timer thread, daemonised, cancelled by the first completed step.
    """

    def __init__(self, *, progress_path: Path, inputs, exp, step_log,
                 census, road: str = "unknown",
                 delay: float = FIRST_STEP_STALL_SECONDS,
                 capability: str | None = None, cache_census_now=None):
        self._progress_path = progress_path
        self._inputs = inputs
        self._exp = exp
        self._step_log = step_log
        self._census = census
        self._road = str(road or "unknown")
        self._delay = float(delay)
        self._capability = capability
        self._cache_census_now = (scan_kernel_cache if cache_census_now is None
                                  else cache_census_now)
        self._timer = None
        self._fired = False

    def arm(self) -> None:
        if self._timer is not None:
            return
        self._timer = threading.Timer(self._delay, self._say)
        self._timer.daemon = True
        self._timer.start()

    def disarm(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def observe(self, **event) -> None:
        """The step observer's first call ends the wait."""

        self.disarm()

    def wrap(self, step_observer):
        """Chain :meth:`observe` in front of the real step observer."""

        if step_observer is None:
            return self.observe

        def observed(**event):
            self.disarm()
            step_observer(**event)

        return observed

    def _grown_entries(self) -> int:
        """Cubins that appeared since launch, or 0 if that is unaskable.

        The corroboration the launch census cannot give.  A cache that
        reads warm at launch can still be missing every kernel the
        FORECAST needs -- that is the measured hole in the class
        docstring -- but a compile that is genuinely running WRITES, and
        entries appearing while step 1 stalls are positive evidence no
        timer can supply.
        """

        try:
            entries, _undecodable, _architectures = self._cache_census_now()
        except Exception:            # noqa: BLE001 - telemetry never fails
            return 0
        launched = 0
        if self._census is not None:
            launched = int(self._census[0])
        return max(0, int(entries) - launched)

    def _say(self) -> None:
        if self._fired:
            return
        self._fired = True
        capability = (current_compute_capability() if self._capability is None
                      else self._capability)
        state = kernel_cache_state(compute_capability=capability,
                                   census=self._census)
        # THE STALL IS THE TRIGGER, NEVER THE EVIDENCE -- ledger #324.
        # Captured live: this printed "the one-time NVRTC compile of this
        # run's GPU kernels" while the very same line reported 388 cached
        # entries, 388 of them for this card, and not one cubin was
        # written while it stalled.  The run was transfer-bound on the
        # streamed road.  A timer knows that a reader is waiting; it
        # knows nothing whatever about what they are waiting for, so the
        # cause is claimed only where something was actually measured.
        detail = (f"the kernel cache held {state.entries} entry(s) at "
                  f"launch, {state.entries_for_capability} of them for "
                  f"sm_{state.compute_capability or '?'}")
        grown = self._grown_entries()
        head = (f"forecast: model step 1 has been running for "
                f"{self._delay:.0f} s -- ")
        if state.reason in (COLD_CACHE, ARCHITECTURE_MISSING) or grown:
            reason = state.reason or "cache_grew_during_first_step"
            if grown:
                detail += f", and {grown} more have been written since"
            print(f"{head}this is the one-time NVRTC compile of this run's "
                  f"GPU kernels for the local card ({detail}); typically "
                  "1-3 minutes, and the result is cached so later runs "
                  "skip it", flush=True)
            status = COMPILING_STATUS
        else:
            # No cause is named that was not measured.  The road IS
            # measured -- the run chose it and wrote the receipt -- so
            # the store road may say what feeds each step.  Anything
            # else says only what is known: nothing has finished yet.
            if self._road == "store":
                print(f"{head}first model step still running; streamed "
                      "transfers feed each step on this card, and no model "
                      f"step has completed ({detail})", flush=True)
            else:
                print(f"{head}first model step still running; no model step "
                      f"has completed ({detail})", flush=True)
            reason = None
            status = "RUNNING"
        try:
            _atomic_json(self._progress_path, {
                "schema": PROGRESS_SCHEMA,
                "status": status,
                "source": self._inputs.source,
                "model_elapsed_seconds": 0.0,
                "requested_run_seconds": float(self._exp.run_seconds),
            }, heartbeat=True)
        except OSError:
            pass
        # A receipt that claimed a compile on a timer was the same defect
        # written down, so it is announced on the same evidence as the
        # line -- and not at all without it.
        if self._step_log is not None and reason is not None:
            self._step_log.announce_kernel_compile(
                reason=reason,
                compute_capability=state.compute_capability,
                cached_entries=state.entries,
                cached_entries_for_this_card=state.entries_for_capability,
                detected_by=f"model step 1 exceeded {self._delay:.0f} s")


def _announce_kernel_compile(progress_path: Path, inputs, exp,
                             step_log=None, census=None) -> None:
    """Flip the published status when the NVRTC compile is actually coming.

    Physics initialization is where a first run pays its one-time NVRTC
    compile (~100 s on a modern card), and it used to pay it under the stale
    RESTORING status -- the first field run of the published wheel watched
    that silence and concluded a hang.  One line and one status flip, only
    when the kernel cache says the compile is really about to happen.

    Shared by both initialization roads: the store-direct road pays the same
    compile inside its first slab, and a road that skipped this would
    reintroduce exactly the silence for the runs most likely to be long.

    The card is asked here, not inside the notice module: this runner has
    long since initialised CUDA, and
    :func:`gpuwm.kernel_compile_notice.kernel_cache_state` stays a pure
    filesystem predicate so a caller that must not touch the device can
    still use it.  Asking is what catches the card-swap case -- the cache
    is keyed by architecture, so 7,164 entries for the previous card are
    7,164 entries this run cannot load.

    ``census`` is the cache reading taken before this run compiled
    anything.  It is not optional in practice and the reason is
    measured: by the time this function is reached, this run's own
    first kernels are already in the cache it would otherwise scan, so
    a cache that was unusable at launch reads as warm.  See
    :func:`gpuwm.kernel_compile_notice.kernel_cache_state`.

    ``step_log`` is told that a compile was announced.  It cannot be told
    what it COST yet -- most of the compile lands inside model step 1 --
    so the log holds the claim and emits one ``phase`` record once it has
    a measurement to attach.
    """

    state = kernel_cache_state(
        compute_capability=current_compute_capability(), census=census)
    if state.notice is None:
        return
    print(state.notice, flush=True)
    if step_log is not None:
        step_log.announce_kernel_compile(
            reason=state.reason,
            compute_capability=state.compute_capability,
            cached_entries=state.entries,
            cached_entries_for_this_card=state.entries_for_capability)
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": COMPILING_STATUS,
        "source": inputs.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": float(exp.run_seconds),
    }, heartbeat=True)


def _store_domain_extreme(store: Mapping[str, object], key: str,
                          reducer) -> float | None:
    """One reduction over a whole-domain carrier, read where the domain is.

    The receipt's ``swdown``/``rainnc`` numbers are reductions over a physics
    field, and on this road there is no domain-shaped state to take them from
    -- the only domain-shaped copy of those fields is the store, which is
    also the one the sweep keeps current.  ``None`` for a key the store does
    not carry, so a missing diagnostic is a hole in the receipt rather than
    an exception at the end of a completed forecast; the caller records WHICH
    key was missing beside it.
    """

    array = store.get(key)
    if array is None:
        return None
    return float(reducer(np.asarray(array)))


def _single_prepared_root(domain_cfg, grid, state, clock, *, store_direct):
    """Give both initialization roads the same scheduled boundary clock."""
    from gpuwm.core.model import DomainNode
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock

    node = DomainNode(domain_cfg, grid, state, clock, None, [], None)
    if not store_direct:
        # Bind before tile factories or restart setup can inherit the mirror.
        # The store road has no resident mirror; its builder receives this
        # same node.clock explicitly below.
        bind_lateral_boundary_clock(node.state, node.clock)
    return node


def _single_checkpoint_identity(inputs, runtime_source_identity):
    """Bind every sealed authority, including the unchanged stop time in TOML."""
    return {
        "schema": "gpuwm.prepared-single-checkpoint.v1",
        "source": inputs.source,
        "prepared_content_sha256": inputs.cache_reader.content_sha256,
        "authority_sha256": dict(inputs.file_sha256),
        "runtime_source_identity": runtime_source_identity,
    }


@contextmanager
def _checkpoint_restore_resources(writers, step_log):
    """Unwind eager output owners if checkpoint or restored health is refused."""
    try:
        yield
    except BaseException as error:
        for close in (
                lambda: writers.__exit__(type(error), error, error.__traceback__),
                lambda: step_log.close(status="FAIL", error=f"{type(error).__name__}: {error}")):
            try:
                close()
            except BaseException as cleanup_error:
                error.add_note(f"restore cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
        raise


def _restore_single_checkpoint(model, restart):
    """Use the canonical tree reader after the canonical store is attached."""
    if restart is None:
        return None
    from gpuwm.io.restart import restore_tree_restart
    from gpuwm.supervisor import validate_manifest_checkpoint

    checkpoint = validate_manifest_checkpoint(Path(restart))
    return restore_tree_restart(checkpoint, model)


def run_prepared_forecast(
        inputs: PreparedForecastInputs, *, output_directory: Path,
        observer=None, first_products=None, stream_init: str = "auto",
        progress_options=None, preflight_seconds: float | None = None,
        kernel_cache_census=None, restart: Path | None = None,
        health_debug: bool = False,
) -> dict[str, object]:
    """Restore, integrate, and publish hash-bound source-neutral history.

    ``observer`` is an optional second consumer of this run's progress,
    for a caller driving the runner in-process rather than reading its
    ``progress.json`` from outside (:class:`gpuwm.runplan.RunObserver`
    is one).  It receives every per-step progress event this runner
    already builds, and -- if it carries the ``output_committed`` hook
    -- each wrfout as it becomes durable.  ``None`` leaves every byte of
    this runner's behaviour unchanged.

    ``first_products`` is a :class:`gpuwm.first_products.FirstProducts`
    this runner OWNS, for the command line rather than for a host.  It
    exists because the feature was reachable only through a host: arming
    lives on ``RunObserver`` and on ``gpuwm go``, so ``python -m
    gpuwm.prepared_single_domain_forecast`` published its analysis frame
    -- fsynced, self-validated and renamed, minutes before the forecast
    ends -- with nothing watching.  MEASURED consequence: reaching a
    89 s launch-to-first-plot on this route took an EXTERNAL process
    polling the frames for ``GPUWM_WRITE_COMPLETE``, which is a watcher
    reimplementing a hook the writer already raises.  ``None`` -- and
    every invocation that names no products -- changes nothing.

    ``stream_init`` is ``--stream-init``: which road a STREAMED forecast
    builds its domain on.  ``"auto"`` (the default) prices the resident
    road against the card's free memory -- from this domain's own columns
    at the route's MEASURED cost per column, and from the cache manifest,
    whichever is larger -- and takes it wherever it comfortably fits,
    otherwise the store; ``"resident"`` and ``"store"`` force one.  A
    forecast that does not stream ignores it entirely -- there is no
    second road when the resident state IS the domain -- so a run without
    ``[tiles]`` executes byte for byte what it executed before this
    parameter existed, through the same calls.

    ``progress_options`` is :class:`gpuwm.progress_log.ProgressOptions`,
    the four ``--progress-*`` / ``--frame-markers`` flags.  ``None`` is
    the DEFAULT SET, not silence: this runner prints WRF's per-step
    ``Timing for main:`` lines, writes ``progress.jsonl`` beside its
    outputs and publishes a frame-ready marker per durable history file
    unless a caller has asked it not to.  The reason it defaults on is
    the report this feature came from -- a run that says nothing about
    which step it is on cannot be driven from a script, and "the user
    did not pass a flag" is not a reason to withhold that.
    """

    _verify_thompson_runtime_environment(inputs.physics_receipt)

    import cupy as cp

    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.dycore import stability_gate_failed
    from gpuwm.core.gpu_mem_watch import (
        GpuPeakMemoryWatcher, default_cupy_probes,
    )
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.model import (
        ExperimentState, ModelRuntimeStatus, execute_experiment,
    )
    from gpuwm.core.refl import consume_refl_10cm, domain_start_ticks_of
    from gpuwm.core.uh_diag import reset_up_heli_max
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.runtime import declared_constant_glw
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    from gpuwm.state_digest import canonical_state_digest

    from gpuwm.progress_log import ProgressOptions

    outdir = Path(output_directory).resolve()
    progress_path = outdir / "progress.json"
    exp = inputs.experiment
    from gpuwm.case_data import trace_gas_overrides_from_config
    trace_gas_overrides = trace_gas_overrides_from_config(
        inputs.experiment_config, expected_sha256=inputs.file_sha256["experiment_config"])

    def initialize_cached_physics(*args, row_start=None, domain_rows=None, **kwargs):
        return initialize_prepared_physics(
            *args, **kwargs, p_top=exp.vertical.p_top, column_chunk=exp.column_chunk,
            trace_gas_overrides=trace_gas_overrides)
    cfg = exp.root.run
    # The per-step log opens HERE, before the restore, so its first line
    # is the run announcing itself rather than the run's third minute.
    step_log = (progress_options or ProgressOptions()).open(
        outdir=outdir, start_time=exp.start_time,
        run_seconds=float(exp.run_seconds),
        # See the tree runner: scope-1 flag, read from the root, and it
        # decides both the `dt` field and the stream's schema string.
        adaptive_dt=bool(exp.root.run.use_adaptive_time_step))
    timing = {}
    # The preflight ran BEFORE this function -- it is what decided this
    # function may run at all -- so its number is handed in rather than
    # measured here.  It was previously recorded nowhere: the status
    # VALIDATING_PREPARATION was published and its duration, which
    # scales with the prepared cache, was dark.
    if preflight_seconds is not None:
        timing["preflight_verify"] = float(preflight_seconds)
    step_log.phase("preflight_verify", preflight_seconds)
    # The kernel cache as it was BEFORE this run compiled anything.
    # Pure filesystem, no device, and taken here rather than at the
    # announcement below because by then this run's own first kernels
    # are in it -- measured: a cache staged with 200 entries for
    # another card read as warm at the announcement, because 160 fresh
    # entries for THIS card had already been written into it.
    if kernel_cache_census is None:
        kernel_cache_census = scan_kernel_cache()
    total_started = time.perf_counter()
    runtime_source_identity = _runtime_source_identity()
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": "RESTORING_PREPARED_CACHE",
        "source": inputs.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": float(exp.run_seconds),
    }, heartbeat=True)

    # WHICH INITIALIZATION ROAD, decided before a byte is loaded.  The
    # decision is only ever made for a forecast that will actually stream:
    # ``exp.tiles`` unconfigured means ``decide`` is never called, no card is
    # queried, ``bundle`` stays None and everything below is the resident
    # route exactly as it was.  The decision is taken here rather than after
    # the restore because the whole point of the store road is that the
    # restore does not happen.
    tiles_options = getattr(exp, "tiles", None)
    planning_machine = streaming.cold_planning_machine(exp)
    resident_estimate = None
    if tiles_options is not None and tiles_options.enabled:
        from gpuwm.core.preflight import estimate_experiment
        boundary_meta = inputs.cache_reader.header.get("metadata", {}).get("lbc")
        resident_estimate = estimate_experiment(
            exp, forcing_intervals=(None if boundary_meta is None
                                    else len(boundary_meta["intervals"])),
            profile=getattr(planning_machine, "device_profile", None))
    stream_decision = (streaming.decide(
        cfg, tiles_options, machine=planning_machine, resident_estimate=resident_estimate)
                       if tiles_options is not None and tiles_options.enabled
                       else None)
    init_road, init_receipt = _choose_stream_init_road(
        stream_init, decision=stream_decision, reader=inputs.cache_reader,
        cfg=cfg)
    if init_receipt is not None:
        print(f"prepared forecast: {init_receipt['why']}", flush=True)

    bundle = None
    if init_road == "store":
        # THE STORE-DIRECT ROAD.  ``store_from_prepared_cache`` reads the
        # cache one row slab at a time, attaches physics to a slab-height
        # state, and scatters the result into full-domain PINNED HOST arrays
        # -- the same store ``attach`` would have copied a resident domain
        # into, arrived at without ever allocating a domain-shaped device
        # array.  The three validations the resident road makes at this
        # instant are made here too, against the same caller-pinned digest:
        # what cannot be reproduced without a domain state is named in the
        # receipt rather than dropped in silence.
        from gpuwm.ingest.prepared_store import store_from_prepared_cache

        _announce_kernel_compile(progress_path, inputs, exp,
                                step_log, kernel_cache_census)
        started = time.perf_counter()
        bundle = store_from_prepared_cache(
            inputs.prepared_cache_path,
            expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static,
            landuse_attrs=inputs.landuse_identity, grid=inputs.grid,
            valid_time=exp.start_time,
            # The user's own pinned-host budget when [tiles] carries one, so
            # the store's pre-allocation guard refuses against the number the
            # operator set rather than against the machine's whole RAM.
            budget_bytes=getattr(tiles_options, "host_budget_bytes", None),
            constant_glw_wm2=declared_constant_glw(exp),
            physics_initializer=initialize_cached_physics,
            log=lambda line: print(line, flush=True))
        # Named for the resident road's key, because it is the same work
        # measured: get the prepared case off disk and onto the machine.
        timing["restore_prepared_cache"] = time.perf_counter() - started
        timing["initialize_physics"] = 0.0
        step_log.phase("restore_prepared_cache",
                       timing["restore_prepared_cache"], road="store")
        _validate_store_bundle_receipt(
            bundle.receipt, inputs.cache_reader.content_sha256)
        _validate_store_direct_vertical_contract(
            inputs.cache_reader, cfg, bundle)
        if not inputs.cache_reader.metadata.get("surface_fields"):
            raise ValueError(
                "prepared cache has no source-neutral canonical surface state")
        if bundle.boundaries is None:
            raise ValueError(
                "prepared cache carries no lateral boundary series; a "
                "specified root cannot be integrated without one")
        # ``restored.metadata`` is the cache's ``metadata['user']`` block, and
        # the reader this route already holds is where it comes from -- so the
        # same optional source-adapter hint is cross-checked on both roads
        # without the store loader having to carry it.
        _validate_restored_source_adapter(
            inputs.cache_reader.metadata.get("user", {}), inputs.source)
        # The SLAB-HEIGHT template's driver.  Its scheme selection, its
        # radiation carrier policy and its per-carrier provenance are
        # properties of the configuration and identical at every height; its
        # ARRAYS are one slab's and are never read for a domain quantity (see
        # the receipt's swdown/rainnc reductions, which come off the store).
        driver = bundle.template.physics
        domain_state = bundle.template
    else:
        started = time.perf_counter()
        restored = restore_prepared_cache(
            inputs.prepared_cache_path,
            expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static)
        timing["restore_prepared_cache"] = time.perf_counter() - started
        step_log.phase("restore_prepared_cache",
                       timing["restore_prepared_cache"], road="resident")
        _validate_restored_cache_receipt(
            restored.receipt, inputs.cache_reader.content_sha256)
        if restored.surface is None:
            raise ValueError(
                "prepared cache has no source-neutral canonical surface state")
        _validate_restored_source_adapter(restored.metadata, inputs.source)

        _announce_kernel_compile(progress_path, inputs, exp,
                                step_log, kernel_cache_census)

        started = time.perf_counter()
        driver = initialize_cached_physics(
            restored.initial_result, cfg, restored.met, restored.surface,
            inputs.static, inputs.landuse_identity, inputs.grid,
            exp.start_time,
            constant_glw_wm2=declared_constant_glw(exp))
        timing["initialize_physics"] = time.perf_counter() - started
        step_log.phase("initialize_physics", timing["initialize_physics"])
        domain_state = restored.initial_result.state

    tick_clock = resolve_clock(
        exp, lbc_interval_s=float(inputs.boundary_interval_seconds))
    schedule = build_schedule(exp, tick_clock)
    clocks = tick_clock.clocks()
    # ON THE STORE ROAD ``node.state`` IS THE SLAB-HEIGHT TEMPLATE, and this
    # is the least-invasive correct answer rather than a convenience.
    # ``execute_experiment`` requires a state object for three things and only
    # three: ``refresh_model_time`` writes ``elapsed_seconds`` onto it,
    # ``impose_clock`` reads that value back, and the stepper is called with
    # it (``StreamedDomain.__call__`` accepts it and, attached without a
    # state, checks nothing against it).  The template satisfies all three,
    # costs one slab, and carries the physics driver the REFL_10CM handoff and
    # the writer's carrier provenance need.  What it must NEVER be used for is
    # a domain-sized READ -- its arrays are 64 rows of the analysis -- and
    # every such reader on this route is routed to the store below, by name.
    node = _single_prepared_root(
        exp.root, inputs.grid, domain_state, clocks[1],
        store_direct=bundle is not None)
    checkpoint_identity = _single_checkpoint_identity(inputs, runtime_source_identity)
    fingerprint = hashlib.sha256(
        _canonical(checkpoint_identity).encode("utf-8")).hexdigest()
    model = ExperimentState(
        node, MappingProxyType({1: node}), schedule, None, fingerprint)
    model._experiment_fingerprint_components = checkpoint_identity
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({
        # The wrfout writers read exactly two things off this: the static
        # fields for the metadata frame and ``initial_result.coord`` for the
        # global attributes.  The store road has no ``CachedInitialResult``
        # -- there is no domain state to put in one -- and both of those are
        # domain-invariant, so the bundle supplies them directly.
        1: SimpleNamespace(
            static_fields=inputs.static, geog_selection=None,
            streamed_store=bundle,
            initial_result=(
                restored.initial_result if bundle is None
                else SimpleNamespace(coord=bundle.coord, base=bundle.base,
                                     state=None))),
    })

    if bundle is None:
        initial_health = _strict_json(vars(
            StateHealthValidator(node.state).validate(
                phase="initialized.d01")))
        if not initial_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast initial health failed: {initial_health}")
    else:
        # ARMED ON THIS ROAD TOO.  The gate needs a DOMAIN, not a device: the
        # CUDA validator is one block per whole field and cannot be pointed at
        # the slab-height template without reporting one slab as the analysis,
        # but the domain is right here in the pinned host store, and the rule
        # set the kernel applies is the same one ``validate_fields_cpu`` reads
        # out of ``rule_for_field``.  So the per-field bounds -- the moisture
        # ranges, the coupled-mass positivity, the geopotential and specific-
        # volume limits -- are checked over the whole 1024 x 1024 analysis
        # rather than declared out of reach.  A failure is terminal here
        # exactly as it is on the resident road.
        initial_health = _store_full_state_health(
            bundle, cfg, phase="initialized.d01")
        if not initial_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast initial health failed: {initial_health}")
        coverage = initial_health["coverage"]
        print("prepared forecast: store-direct initialization -- the "
              "initialized.d01 full-state health gate IS armed, over the "
              f"store: {coverage['fields_checked']} fields, "
              f"{coverage['bytes_checked'] / _GIB:.2f} GiB, in "
              f"{coverage['seconds']:.1f}s"
              + ("" if not coverage["not_in_store"] else
                 "; gated fields the store does not carry: "
                 + ", ".join(coverage["not_in_store"])), flush=True)

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
        source=inputs.source,
        # The tree-wide [output] history selection; a single-domain run's
        # own `output = {...}` on its [[domain]] table overrides it inside
        # the writer set, exactly as it does on a tree.
        history_selection=exp.output)
    model._io_manager = writers
    forecast_started = time.perf_counter()

    def history_handler(_model, current, ticks):
        # ASKED OF THE STEPPER.  ``steppers`` is bound below and read here at
        # CALL time, which is after ``streaming.steppers_for_tree`` has run.
        # It matters because the gate two lines down --
        # ``stability_gate_failed`` at CFL 10 and w 150 m/s -- is this
        # route's safety net, and under ``[tiles] store = "host"`` the
        # domain is in a pinned host store while ``current.state`` is the
        # copy taken at t = 0 that the sweep never writes.  Reducing over it
        # meant the net reported the INITIAL CFL at every history frame,
        # forever, on a run that could already have gone non-finite.
        # ``stability_observer`` returns ``dycore.stability_report`` itself
        # for a resident domain, so nothing changes for a run that
        # configures no [tiles].
        stepper = steppers.get(int(current.cfg.grid_id))
        # THE ANALYSIS FRAME ON THE STORE-DIRECT ROAD.  Before the first
        # sweep there is no fold, and ``StreamedStability`` answers that one
        # instant from the resident state instead -- correct for a domain
        # attached FROM a state, because attach copied it and neither has
        # moved since.  Here the only state is the slab-height template, so
        # that same branch would report the last slab's maxima as the
        # domain's: a plausible number that is not this domain's, in the
        # receipt, on the exact axis the fold exists to stop lying about.
        # Refused by name instead, and the gate is skipped for this one frame
        # rather than run on the wrong memory -- there is nothing to catch
        # yet, because nothing has been integrated.
        if bundle is not None and int(getattr(stepper, "steps", 0)) == 0:
            report = {
                "folded": False,
                "why": ("the store-direct road has no domain-shaped state to "
                        "reduce and the per-tile fold has no record before "
                        "the first sweep, so the analysis frame carries no "
                        "stability sample rather than the slab template's"),
            }
            gate_failed = False
        else:
            report = streaming.stability_observer(stepper)(
                current.state, current.cfg.run,
                boundary_width=current.cfg.run.spec_bdy_width)
            gate_failed = stability_gate_failed(
                report, max_cfl=10.0, max_w_ms=150.0)
        sample = {
            "ticks": int(ticks),
            "elapsed_seconds": float(current.clock.elapsed_seconds),
            **report,
        }
        history.append(_strict_json(sample))
        if gate_failed:
            raise FloatingPointError(
                "prepared forecast stability threshold failed: "
                + _stability_diagnosis(sample, current.state, current.cfg.run)
                + f"; sample {sample}")
        refl = _consume_due_native_refl_10cm(
            current.state, ticks, consume_refl_10cm,
            domain_start_ticks=domain_start_ticks_of(current))
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
                "last_checkpoint": event.get("last_checkpoint"),
                "last_durable_wrfout": (
                    None if not durable_paths
                    else str(durable_paths[-1].resolve())),
            }, heartbeat=True)

    # Here and not at the writers' construction above: the closure this
    # binds reads `writers.paths`, so it cannot exist before the writers
    # do.  Still ahead of every submit -- the first one happens inside
    # execute_experiment below -- and attach_progress_callback refuses
    # outright if that ever stops being true.
    #
    # ONE attach, with both consumers behind it.  attach_progress_callback
    # overwrites `landing_observer` rather than appending to it, so
    # calling it twice would silently unhook the host's observer the
    # moment this runner armed a render of its own.
    #
    # No root-domain guard, unlike RunObserver.output_committed: this
    # runner integrates exactly one domain (preflight replaces a
    # multi-domain experiment with `(source_exp.root,)`), so there is one
    # writer, it is the root's, and a guard here would be a test that
    # cannot fail.  The tree runner is where that guard earns its keep.
    #
    # `getattr` for the observer's hook, exactly as attach_progress_callback
    # itself reads it: this runner's contract has always been that an
    # observer without `output_committed` is fine and simply hears
    # nothing about landings.
    landing = _LandingObservers(
        getattr(observer, "output_committed", None),
        # The per-step log's own frame hook.  It rides HERE rather than
        # anywhere else because this is the one call raised after the
        # frame is fsynced, self-validated and renamed onto its final
        # name -- which is exactly the instant a "this frame is safe to
        # read" marker is allowed to be published.
        step_log.output_committed if step_log.enabled else None,
        None if first_products is None else first_products.frame_committed)
    if landing:
        writers.attach_progress_callback(landing)

    # [tiles]; see the same call in prepared_domain_tree_forecast.  An
    # experiment that does not configure it gets {} and the executor's own
    # dycore.step, unchanged -- builders_for_tree returns {} on the same
    # test and imports nothing either.
    #
    # The builders are the half of this seam that was missing.  Without them
    # steppers_for_tree refused every configuration it was given ("this
    # route wired no streamed-domain builder"), so [tiles] mode = "on"
    # was configurable and unreachable at the same time: no forecast this
    # CLI can launch was capable of streaming.
    #
    # `streaming_decisions` is the OTHER half.  The stepper dict cannot say
    # which way `auto` went: a grid that declined to stream is absent from
    # it, and absent is what an unconfigured run looks like too.  So an
    # `auto` forecast that quietly ran resident would be indistinguishable
    # from one that streamed -- including to a test pointed at it.  The
    # decisions are recorded per grid, reported in the receipt below as
    # report["tiles"], and printed as one line.
    streaming_decisions: dict = {}
    steppers = streaming.steppers_for_tree(
        model, exp.tiles, machine=planning_machine, resident_estimate=resident_estimate,
        # ``store_domain_builder`` is ``prepared_domain_builder``'s
        # counterpart for a domain that was never resident: it reads the
        # geography, the lateral tables and the tile-state template off the
        # BUNDLE instead of off ``node.state``, and calls ``attach`` with the
        # store already built and no state at all.
        #
        # UNPARENTHESISED, deliberately.  The resident half must read as the
        # literal ``builders=streaming.builders_for_tree(``: that string is
        # what tests/test_streaming.py greps both production routes for, and
        # it greps rather than calls because the defect it guards was an
        # ABSENT argument -- a route that stops passing builders refuses
        # every streaming configuration and behaves identically otherwise.
        # Wrapping this expression in one extra paren was enough to make the
        # guard stop seeing the call it is guarding.
        builders=streaming.builders_for_tree(model, exp.tiles)
        if bundle is None
        # ``clock=node.clock`` IS NOT OPTIONAL and the builder refuses
        # without it.  ``attach(None, ...)`` has no DomainState, so the
        # lazy binding every other road relies on derived None here, latched
        # on the first buffer conversion and put the whole store-direct
        # forecast on the retired elapsed-seconds Davies recurrence -- the
        # #219 one-timestep phase error, reopened on the road the LARGEST
        # domains take.  This is the domain's own clock, the same object
        # ``bind_lateral_boundary_clock`` binds to the resident mirror.
        else {int(node.cfg.grid_id): streaming.store_domain_builder(
            bundle, clock=node.clock)},
        decisions=streaming_decisions)
    streaming_report = streaming.streaming_receipt(
        exp.tiles, streaming_decisions)
    if streaming_report:
        print(f"prepared forecast: {streaming_report['summary']}", flush=True)
    if bundle is not None:
        # ``decide`` was consulted twice -- once above to choose the road,
        # once inside ``steppers_for_tree`` -- and under ``auto`` it is a
        # function of the free VRAM at the instant it was taken, so the two
        # can legitimately disagree on a card whose memory moved in between.
        # On this road a disagreement is fatal rather than cosmetic: the
        # domain is in the store and there is no resident state for
        # ``dycore.step`` to integrate, so a second answer of "resident"
        # would step the slab template and publish it as the forecast.
        # Checked, not assumed.
        if int(node.cfg.grid_id) not in steppers:
            raise RuntimeError(
                "the store-direct road built this domain straight into a "
                "pinned host store, and the streaming seam then decided this "
                "grid should run RESIDENT -- there is no resident domain to "
                "run.  [tiles] mode = 'auto' re-decides against the free VRAM "
                "at the instant it is asked; pin the mode ([tiles] mode = "
                "'on') for a run that must stream.")
        # THE MARKER, and it is what routes three whole-domain consumers to
        # the store instead of to the template.  ``attach`` sets it on the
        # state it takes over, and a store-direct attach has no such state --
        # so it is set here, on the object the model actually holds:
        # ``PerDomainWrfoutWriters.submit`` asks the state for it before
        # taking a history frame (without it the frame would be the slab
        # template's), and ``streaming.live_scratch`` follows it to the
        # store's ``scratch/up_heli_max`` so the history-interval UH reset
        # lands on the accumulator the tiles are actually folding into.
        node.state._streamed_domain = steppers[int(node.cfg.grid_id)]

    with _checkpoint_restore_resources(writers, step_log):
        restart_info = _restore_single_checkpoint(model, restart)
        from gpuwm.runtime import _restart_is_complete
        already_complete = _restart_is_complete(restart_info)
        restored_seconds = (None if restart_info is None else
                            float(node.clock.ticks / schedule.clock.tick_den))
        if restart_info is not None:
            from gpuwm.core.health import health_validator_for_domain
            initial_health = _strict_json(vars(health_validator_for_domain(
                model, node).validate(phase="restored.d01")))
            if not initial_health["ok"]:
                raise FloatingPointError(
                    f"prepared forecast restored health failed: {initial_health}")

    checkpoints = []

    def restart_handler(tree, ticks):
        from gpuwm.io.restart import write_tree_restart
        valid = exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
        started = time.perf_counter()
        tree._last_checkpoint = write_tree_restart(outdir, tree, valid)
        checkpoints.append(str(Path(tree._last_checkpoint).resolve()))
        step_log.restart_written(
            domain=exp.root.grid_id, valid_time=valid,
            path=tree._last_checkpoint, wall_seconds=time.perf_counter() - started)

    # Armed immediately before integration and disarmed by the first
    # completed step: the compile this names is paid inside step 1, and
    # a reader watching a silent terminal has no other way to learn
    # that anything is happening at all.
    stall_watch = _FirstStepStallWatch(
        progress_path=progress_path, inputs=inputs, exp=exp,
        step_log=step_log, census=kernel_cache_census, road=init_road)
    try:
        memory_watch.start()
        stall_watch.arm()
        with writers:
            if already_complete:
                from gpuwm.prepared_domain_tree_forecast import _completed_execution_report
                execution = _completed_execution_report(model)
            else:
                execution = execute_experiment(
                    model, history_handler=history_handler,
                    restart_handler=restart_handler,
                    progress_callback=progress_callback,
                    # The common validator follows the attached canonical store.
                    # Preserve the default store cost; explicit diagnostics enable it.
                    validate_state=(bundle is None or health_debug),
                    health_debug=health_debug,
                    skip_feedback_path=True, pool_trim_per_period=True,
                    steppers=steppers,
                    # ONE line per model time step, WRF's own bar.  Handed
                    # the bound method rather than a wrapper so the log is
                    # what the executor calls, and None when the log is off
                    # so a silenced run pays nothing per step.
                    step_observer=stall_watch.wrap(
                        step_log.step_observer if step_log.enabled else None),
                    experiment=exp)
            cp.cuda.Stream.null.synchronize()
            timing["forecast_execution_with_async_io"] = (
                time.perf_counter() - forecast_started)
            drain_started = time.perf_counter()
            writers.drain()
            timing["final_writer_drain"] = time.perf_counter() - drain_started
            wrfout_paths = writers.paths
        timing["forecast_and_io_inclusive"] = (
            time.perf_counter() - forecast_started)
    except BaseException as error:
        # The last line a driving script reads has to say what happened.
        # Closed here, before the exception continues to `main`'s report
        # writer, because that writer can itself fail and the log must
        # still have terminated.
        step_log.close(status="FAIL",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # A forecast that never reached step 1 must not leave a timer
        # thread waiting to announce a compile for a run that is over.
        stall_watch.disarm()
        memory_watch.stop()
        model._io_manager = None
    # After the drain: every frame this run will ever commit is durable
    # and has its marker, so the run_end record is true when it is read.
    step_log.close(status="SUCCESS")
    memory_watch.sample()

    cadence_seconds = float(exp.root.history_interval_s)
    cadence_receipt = _validate_hash_bound_history_cadence(
        exp, cadence_seconds)
    output_schedule = _history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=cadence_seconds, after_seconds=restored_seconds)
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

    # The final health gate validates node.state and CANNOT be folded:
    # StateHealthValidator is one block per whole field with no windowing, so
    # it has no tile-interior form.  Under a host store that state is the
    # snapshot that filled the store, so the gate used to pass on the t = 0
    # ANALYSIS -- and so did canonical_state_digest below it: a receipt
    # reporting nan_free on a run that had integrated for an hour, and a
    # final digest that was the initial condition.  The note was here and
    # the fix was not; this is the fix.  Zero and a getattr on a resident
    # run, so nothing changes for a forecast that configures no [tiles].
    if bundle is None:
        carriers_refreshed = streaming.refresh_streamed_state(
            steppers.get(int(node.cfg.grid_id)), node.state)
        final_health = _strict_json(vars(
            StateHealthValidator(node.state).validate(phase="final.d01")))
        final_stability = _strict_json(streaming.stability_observer(
            None if already_complete else steppers.get(int(node.cfg.grid_id)))(
                node.state, cfg, boundary_width=cfg.spec_bdy_width))
        if not final_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast final health failed: {final_health}")
        started = time.perf_counter()
        final_digest = canonical_state_digest(
            node.state, node.clock, scope="trajectory")
        timing["canonical_final_state_digest"] = time.perf_counter() - started
    else:
        # THE STORE-DIRECT FINAL READS, none of which may touch node.state.
        # There is nothing to refresh: ``refresh_state`` copies the store onto
        # a resident domain, and this road never built one -- calling it would
        # refuse on the first carrier's shape (domain against slab), which is
        # the right refusal and the wrong question.
        stepper = steppers[int(node.cfg.grid_id)]
        carriers_refreshed = 0
        # The last sweep's record, folded per tile out of the STORE and
        # bit-equal to what ``stability_report`` returns for the same domain
        # integrated resident: max is associative and exact, the NaN classes
        # are a bitwise OR, and the argmax carries a DOMAIN flat index.  It
        # RAISES rather than returning a stale record if the sweep produced
        # none, which is the behaviour a final gate must have.
        final_stability = (
            {"available": False, "reason": "checkpoint is already at the stop; no new sweep"}
            if already_complete else _strict_json(stepper.health))
        # TWO INSTRUMENTS, BOTH ARMED, over the same store the sweep has been
        # writing.  The fold answers the question a run loop has to ask every
        # step and answers it cheaply; the descriptor gate answers the one the
        # resident road asks HERE, at the same instant, over the same fields
        # and under the same per-field bounds.  Neither substitutes for the
        # other and neither is inferred from the other's silence.
        final_health = _store_full_state_health(bundle, cfg, phase="final.d01")
        final_health["stability_fold"] = {
            "ok": None if already_complete else not bool(final_stability.get("nan")),
            "available": not already_complete,
            "instrument": "streamed store fold (StreamedDomain.health)",
            "covers": ("u, w and theta-perturbation finiteness and the two "
                       "Courant terms, over the whole domain in the store"),
        }
        if not already_complete and not final_health["stability_fold"]["ok"]:
            raise FloatingPointError(
                f"prepared forecast final stability fold failed: "
                f"{final_health['stability_fold']}; "
                f"stability {final_stability}")
        if not final_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast final health failed: {final_health}; "
                f"stability {final_stability}")
        # THE DIGEST MUST BE TAKEN OVER THE STORE, or it is not this run's.
        # ``canonical_state_digest(node.state, ...)`` would hash the
        # slab-height template -- the right shape of answer over 1/16th of the
        # analysis -- and the receipt would carry a trajectory digest for a
        # trajectory that never happened.  Refused by name rather than
        # substituted, because a wrong digest is indistinguishable from a
        # right one until two runs are compared.
        digest_from_store = getattr(stepper, "canonical_digest", None)
        if digest_from_store is None:
            raise RuntimeError(
                "this build's StreamedDomain has no canonical_digest(clock, "
                "scope=...), so a store-direct run has no way to produce its "
                "final trajectory digest over the memory it integrated.  "
                "gpuwm.state_digest.canonical_state_digest is NOT a fallback "
                "here: the only state this road holds is the slab-height "
                "template, and digesting it would publish a hash of one slab "
                "of the analysis as the forecast's trajectory.")
        started = time.perf_counter()
        final_digest = digest_from_store(node.clock, scope="trajectory")
        timing["canonical_final_state_digest"] = time.perf_counter() - started

    _verify_inputs_unchanged(inputs)
    # The same gate, told what it is looking at.  The bare ``!=`` this
    # replaced could not tell an implementation that changed from a
    # ``git`` that failed to spawn or from a scratch file this very run
    # wrote, and it destroyed the receipt of a finished forecast for
    # either -- see :func:`_runtime_source_identity_change`.
    final_source_identity = _runtime_source_identity()
    moved = _runtime_source_identity_change(
        runtime_source_identity, final_source_identity)
    if moved is not None:
        raise RuntimeError(
            f"forecast runtime implementation changed during run: {moved}")
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
        # ``gpuwm_version`` above is what distribution METADATA claims.
        # This block is which tree actually executed -- package path,
        # install kind, branch/sha/dirt, and whether the two version
        # claims agree.  Both are kept: a reader comparing two receipts
        # needs the number they have always compared, and the reason it
        # can be wrong.
        "provenance": _provenance_receipt(),
        "runtime_source_identity": runtime_source_identity,
        # What the end-of-run recheck could actually SEE.  The identity
        # above is the one taken at launch; the gate that compares it
        # skips a rung neither end resolved, and a skipped comparison
        # that says nothing is how a weakened check goes unnoticed.
        # ``git_compared: false`` on a receipt means the run was bound
        # by ``source_sha256`` alone.
        "runtime_source_identity_recheck": {
            "identity_source_at_launch":
                runtime_source_identity["identity_source"],
            "identity_source_at_end":
                final_source_identity["identity_source"],
            "git_resolved_at_launch":
                runtime_source_identity["git_commit"] is not None,
            "git_resolved_at_end":
                final_source_identity["git_commit"] is not None,
            "git_compared": (
                runtime_source_identity["git_commit"] is not None
                and final_source_identity["git_commit"] is not None),
        },
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
        "restart_contract": {
            "mode": "exact-sealed-setup",
            "restart_input": None if restart is None else str(Path(restart).resolve()),
            "restored_seconds": restored_seconds,
            "already_complete": already_complete,
            "interval_seconds": float(exp.restart_interval_s),
            "checkpoints_written": checkpoints,
        },
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
            # READ WHERE THE DOMAIN IS.  On the resident road these are
            # reductions over the driver's own device fields, unchanged.  On
            # the store-direct road the driver is the SLAB template's, so the
            # same three expressions would report one slab's sky and one
            # slab's rain as the domain's -- three plausible numbers that are
            # not this domain's.  The store holds the full-domain arrays and
            # is what the sweep keeps current, so the reductions are taken
            # there; a key the store does not carry gives None and names
            # itself, rather than raising at the end of a finished forecast.
            **({
                "swdown_min_wm2": float(cp.min(driver.fields["swdown"]).get()),
                "swdown_max_wm2": float(cp.max(driver.fields["swdown"]).get()),
                "rainnc_max_mm": float(
                    cp.max(driver.microphysics.rainnc).get()),
            } if bundle is None else {
                "swdown_min_wm2": _store_domain_extreme(
                    bundle.store, "fields/swdown", np.min),
                "swdown_max_wm2": _store_domain_extreme(
                    bundle.store, "fields/swdown", np.max),
                "rainnc_max_mm": _store_domain_extreme(
                    bundle.store, "scratch/mp_rainnc", np.max),
                "domain_reductions_read_from": "store",
                "domain_reductions_missing_keys": sorted(
                    key for key in ("fields/swdown", "scratch/mp_rainnc")
                    if key not in bundle.store),
            }),
            # PER-CARRIER PROVENANCE (gpuwm/core/radiation_carriers.py):
            # one row per radiative carrier -- its source, the model
            # second its producer last wrote it, and a representative
            # element read from the live buffer at receipt time.  This is
            # the run-receipt surface the contract promises: a reader
            # asking "did this run integrate a sky nobody computed" gets
            # the answer per carrier, at end of run.
            "surface_radiation_policy": str(driver.carriers.policy),
            "surface_radiation_carriers": _strict_json(
                driver.carriers.report(fields=driver.fields)),
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
            # WHICH EXECUTION MODE produced the peaks above; empty, and the
            # receipt therefore unchanged, whenever [tiles] is off.
            "tiles": streaming.receipt_entry(
                exp.tiles, streaming_decisions),
            # How many carriers the final health gate and the canonical
            # digest were brought up to date from the store before they ran.
            # 0 on a resident run; on a streamed one it is the whole
            # manifest, and a reader can tell the two apart from the
            # receipt rather than by trusting that the refresh happened.
            "carriers_refreshed_before_final_reads": carriers_refreshed,
        },
        "gridded_output": {
            "cadence_seconds": cadence_seconds,
            "cadence_receipt": cadence_receipt,
            "frames_per_file": 1,
            "expected_frame_count": expected_frames,
            "exact_frame_count_verified": True,
            "initial_frame_verified": restart_info is None,
            "last_scheduled_frame_verified": bool(output_schedule),
            "last_scheduled_offset_seconds": cadence_receipt[
                "last_scheduled_offset_seconds"],
            "last_scheduled_valid_time": cadence_receipt[
                "last_scheduled_valid_time"],
            "last_scheduled_equals_run_end": cadence_receipt[
                "last_scheduled_equals_run_end"],
            "initial_and_final_frames_verified": (restart_info is None and cadence_receipt[
                "last_scheduled_equals_run_end"]),
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
    # Added rather than inlined above, and only when [tiles] was
    # configured: an unconfigured run must produce the report it produced
    # before this mode existed, key for key, so that every stored receipt
    # and every hash taken over one stays valid.  Same emptiness contract as
    # streaming.identity_payload_entry.
    if streaming_report:
        report["tiles"] = streaming_report
    # WHICH INITIALIZATION ROAD RAN, and the numbers it was chosen on.
    # Present only for a run that streams -- ``init_receipt`` is None for
    # every other -- on the same emptiness contract as report["tiles"]: a
    # forecast that configures no [tiles] must write the receipt it wrote
    # before this flag existed, key for key, so that every stored receipt and
    # every hash taken over one stays valid.
    if init_receipt is not None:
        report["stream_init"] = dict(init_receipt)
    if bundle is not None:
        report["stream_init"]["store"] = dict(bundle.receipt)
        report["stream_init"]["unchecked"] = _store_direct_gaps(
            inputs.cache_reader)
        # WHICH GATES WERE ARMED, said in the receipt and not only on the
        # terminal.  Four of this route's five whole-domain gates are armed;
        # the one that is not says why, with the measured number behind the
        # reason, so a reader holding this receipt can see it without
        # re-deriving it from an absence.
        one_pass = float(
            report["health"]["final"].get("coverage", {}).get("seconds", 0.0))
        report["health"]["gates"] = {
            "initialized_d01_full_state": True,
            "executor_periodic_full_state": bool(health_debug),
            "per_step_stability_fold": not already_complete,
            "final_d01_full_state": True,
            "final_stability_fold": not already_complete,
            "why": (
                "Whole-state gates read the canonical host store. "
                f"A full-store pass took {one_pass:.1f}s; periodic scans are "
                + ("enabled by --health-debug." if health_debug else
                   "off by default; --health-debug enables them.")
                + (" No step or stability fold is claimed for an already-complete restore."
                   if already_complete else " Stability folds cover every new step.")),
        }
    # The early render, collected before this process may exit.  Its
    # worker is a DAEMON thread -- it has to be, so a wedged render can
    # never hold a finished forecast open -- and a daemon thread is
    # killed at interpreter shutdown, so a runner that returned here
    # without joining would publish a receipt for pictures that were
    # never written.  `wait` returns None on every way it declined,
    # which is the same "assume nothing was published" answer the
    # finalize stage acts on.
    #
    # Added rather than inlined above, and only when something was
    # published: a run that named no products writes the report it wrote
    # before this existed, key for key.  Same emptiness contract as
    # report["tiles"].
    if first_products is not None:
        receipt = first_products.wait()
        if receipt is not None:
            report["first_products"] = receipt
    # WHICH AEROSOL INITIAL CONDITION THIS FORECAST STARTED FROM.  For
    # mp_physics=28 that is a choice between WRF's monthly WIF climatology
    # and thompson_init's synthetic CCN/IN profile -- two different
    # forecasts -- and until now the only place it was ever said was the
    # PREPARING process's stderr, which this process does not have and no
    # reader keeps.  The preparation writes it into the cache metadata
    # (gpuwm/ingest/prepared_cache.py) and this reads it back.
    #
    # ``report.update`` of a possibly-empty mapping, not an assignment:
    # gpuwm.aerosol_source_receipt returns NOTHING for a scheme with no
    # aerosol number fields, so a wsm6/Morrison/Kessler forecast writes
    # the report it wrote before this existed, key for key.  The two
    # absences are the whole point and that module states the rule; do not
    # collapse this into an unconditional key.
    report.update(aerosol_source_report_entry(
        inputs.cache_reader.metadata.get(AEROSOL_SOURCE_KEY, {}),
        mp_physics=cfg.mp_physics,
        when_unrecorded=(
            "this forecast reads a prepared cache and does not itself run "
            "the real-data aerosol ingest; the cache at "
            f"{inputs.prepared_root} carries no aerosol-initialization "
            "receipt, so it was written by a preparation that predates the "
            "receipt being stored. Re-prepare to record which source filled "
            "nwfa/nifa -- the fields themselves are in the cache and the "
            "forecast is unaffected, but which dataset produced them is not "
            "recoverable from this bundle")))
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
        # The spectral seam's run receipts merge in; an apply run whose
        # step receipts are incomplete refuses a clean capsule here.
        receipts={"report": {"path": str((outdir / "report.json").resolve())},
                  **_seam_capsule_receipts(model)},
    )
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": "PASS",
        "source": inputs.source,
        "model_elapsed_seconds": float(exp.run_seconds),
        "requested_run_seconds": float(exp.run_seconds),
        "report": str((outdir / "report.json").resolve()),
        "frame_count": len(output_inventory),
        "last_checkpoint": (None if model._last_checkpoint is None else
                            str(Path(model._last_checkpoint).resolve())),
    }, heartbeat=True)
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


def build_parser() -> argparse.ArgumentParser:
    """This runner's forecast parser, built without parsing anything.

    Separated from :func:`_parse_args` so the option surface can be READ
    -- by ``--help``, and by the docs/CLI parity test that holds every
    documented door against the flags it really defines.  A parser that
    only exists inside the function that consumes it cannot be checked
    against a document.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", type=Path, help="Resume a canonical checkpoint with this exact sealed preparation/configuration.")
    parser.add_argument("--health-debug", action="store_true", help="Validate the canonical whole-domain state each step.")
    # The two MODE flags, declared rather than only intercepted.
    #
    # `main` answers both before argparse sees anything, because each is
    # a different program with a different parser and the interception
    # is what makes that possible.  Declaring them here changes no
    # behaviour on the paths that work -- a mode flag in first position
    # never reaches this parser -- and buys the thing their absence cost:
    # `--help` now NAMES them.  FIRST-LIGHT documents
    # `--materialize-authorities` as a step of the documented chain, and
    # a user who lost the doc could not rediscover the step from the
    # tool, because the only parser `--help` rendered was this one and
    # this one had never heard of it.
    modes = parser.add_argument_group(
        "mode flags (each must be the FIRST argument, and selects a "
        "different program with its own --help)")
    modes.add_argument(
        "--materialize-authorities", action="store_true",
        help=("create one hash-receipted named-source experiment/WPS "
              "authority pair for an exact physics profile, then exit.  "
              "Run it first on the line and with --help after it for "
              "that mode's own options"))
    modes.add_argument(
        "--show-capabilities", action="store_true",
        help=("print this runner's capability JSON and exit; it must be "
              "the only argument"))
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
    parser.add_argument(
        "--tiles", default=None, metavar="JSON",
        help=("the [tiles] table this forecast integrates under, as a "
              "JSON object with the keys gpuwm.core.streaming."
              "StreamingOptions takes (mode/tile_nx/tile_ny/nbuffers/"
              "halo/store/write_mode/pipeline/vram_budget_bytes/"
              "host_budget_bytes).  For the caller whose hash-bound "
              "experiment cannot carry one: the native HRRR chain hands "
              "this runner the authority its preparer BUILT, which has "
              "no [tiles] table, so a user's block had nowhere to ride.  "
              "Validated by the same StreamingOptions.from_mapping the "
              "config front door uses, and binds no identity -- omitted, "
              "the hash-bound experiment's own table (usually none) runs"))
    parser.add_argument(
        "--stream-init", choices=STREAM_INIT_CHOICES, default="auto",
        help=("which road a STREAMED forecast builds its domain on.  "
              "`resident` restores the prepared cache into one full-domain "
              "DomainState, attaches physics to it and lets the streaming "
              "seam copy it into the pinned host store -- the road with the "
              "parity proof, and the one that caps the domain at the size of "
              "the CARD rather than of the machine (MEASURED at nz = 49: the "
              "prepared case costs about 15 780 B/column, so 1024x1024 is "
              "refused on a 16 GB card while the streamed forecast it would "
              "have fed needs about 6 GiB).  `store` fills the same store one "
              "ROW SLAB at a time and never allocates a domain-shaped device "
              "array, so the ceiling is the machine's pinned RAM.  `auto`, "
              "the default, prices the resident state from the cache's own "
              "state/* manifest times the measured physics headroom and "
              "takes the resident road wherever it fits inside "
              # NOT a percent sign.  argparse interpolates every help string
              # (`self._get_help_string(action) % params`), so a literal "%"
              # here is read as a conversion -- "80% of" parses as "% o",
              # octal with the space flag, and --help dies with "%o format:
              # an integer is required, not dict" for every option in the
              # parser rather than just this one.  A fraction says the same
              # thing and cannot be misread.
              f"{_AUTO_RESIDENT_FIT_FRACTION:.2f} of the card's free "
              "memory.  "
              "Meaningful only when the run streams: with [tiles] off the "
              "resident state IS the domain and this flag changes nothing"))
    parser.add_argument(
        "--render-products", default=None, metavar="SPEC",
        help=("`gpuwm render --products`' own spec -- a comma-separated "
              "product list, or `all`, or `none` -- for the FIRST frame "
              "this run commits, rendered on a worker thread while the "
              "forecast is still integrating.  Absent is off, and off is "
              "the default: there is deliberately no second switch, so "
              "\"which products\" has one answer that cannot disagree "
              "with itself.  The first frame is the analysis at t = 0, "
              "durable before a single step is integrated"))
    parser.add_argument(
        "--render-dir", type=Path, default=None, metavar="DIR",
        help=("where --render-products publishes; defaults to "
              "OUTDIR/png.  Ignored without --render-products"))
    # The per-step progress surface, registered from one place so this
    # door and the tree runner's cannot drift in spelling or in help.
    add_progress_arguments(parser)
    return parser


#: The mode flags, and the position they are only legal in.
MODE_FLAGS = ("--materialize-authorities", "--show-capabilities")


def _parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_materialize_parser() -> argparse.ArgumentParser:
    """The ``--materialize-authorities`` mode's own parser.

    A separate program with a separate option set, exposed for the same
    reason as :func:`build_parser`: a door the documentation names has
    to be checkable against the flags it really takes.
    """

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
    # This stage now owns a LAYERED refusal (a config whose physics the
    # named profile would have overwritten), and a layered message needs
    # the flag that prints its second half -- otherwise the marker
    # gpuwm/explain.py promises can never reach a terminal does.
    add_explain_flag(parser)
    return parser


def _parse_materialize_args(argv=None):
    return build_materialize_parser().parse_args(argv)


def _streaming_options_argument(text: str | None):
    """``--tiles`` as a validated :class:`StreamingOptions`, or ``None``.

    Through ``StreamingOptions.from_mapping`` and nothing else, so this
    flag and a ``[tiles]`` table in a config are refused by the SAME
    validator: an unknown key, a mode this build does not have, half a
    tiling, a tiling set while the mode is off -- all of them produce
    the sentence the config front door produces, rather than a second
    vocabulary a user has to learn for the flag.
    """

    if text is None:
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(
            f"--tiles must be a JSON object, got {type(payload).__name__}")
    return streaming.StreamingOptions.from_mapping(payload, source="--tiles")


def _route_owned_first_products(args, *, outdir: Path, observer,
                                started: float):
    """This runner's own early render, or ``None``.

    ``None`` for three separate reasons, and the distinction matters:

    * ``--render-products`` was not given, or was given as ``none``.
      Off by ABSENCE, decided by
      :func:`gpuwm.first_products.early_render_requested` rather than
      here, so this runner and every other door agree on what "asked for
      pictures" means without a second switch to disagree with.
    * a HOST already armed one.  ``gpuwm run-plan``'s HRRR chain arms
      the render on its observer before it calls this runner, with the
      dict its own finalize stage will use.  Arming a second trigger on
      the same directory would render the same frame twice, publish two
      receipts over each other, and make the finalize skip depend on
      which worker finished last.  The host's arming wins and this says
      so, because a flag that was passed and quietly did nothing is the
      failure this whole increment is about.

    The report/warn pair is what a hosted run gets from its event
    stream, spelled for a process that has none: one line on stdout at
    the instant the pictures became readable -- carrying the seconds
    from launch, which IS the time-to-first-plot number -- and one line
    on stderr for every way the render declined.  The receipt itself is
    unchanged: :mod:`gpuwm.first_products` writes
    ``gpuwm.first-products.v1`` beside the pictures either way, so a
    finalize stage or a reader cannot tell the two arming routes apart.
    """

    from gpuwm.first_products import FirstProducts, early_render_requested

    if not early_render_requested(args.render_products):
        return None
    if getattr(observer, "first_products", None) is not None:
        print("prepared forecast: --render-products ignored; the caller "
              "hosting this runner already armed an early render, and two "
              "on one directory would draw the first frame twice",
              file=sys.stderr)
        return None
    render_dir = (outdir / "png" if args.render_dir is None
                  else Path(args.render_dir))

    def report(receipt) -> None:
        elapsed = time.perf_counter() - started
        print(f"prepared forecast: first products ready {elapsed:.1f} s "
              "after this runner started "
              f"({len(receipt['paths'])} picture(s), "
              f"--products {receipt['render_products']}): "
              + ", ".join(receipt["paths"]), flush=True)

    def warn(code: str, message: str, **fields) -> None:
        print(f"prepared forecast: {code}: {message}", file=sys.stderr)

    # The plan dict `go_cli.render_command` and `go_cli._render_stage`
    # both take, with `run` naming the directory this runner writes its
    # wrfout subdirectory into -- so a finalize render pointed at the
    # same --outdir composes the same command and lands in the same
    # place, which is what makes the early frame's bytes and the late
    # one's comparable at all.
    return FirstProducts(
        {"run": outdir, "render": render_dir,
         "render_products": args.render_products},
        report=report, warn=warn)


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
    # Which tree is about to integrate this forecast.  Announced before
    # any argument is interpreted, and a refusal when the version this
    # process would stamp into its report disagrees with the code
    # writing it.  ``--show-capabilities`` is answered above, untouched:
    # a front end probing the runner's capabilities gets JSON on stdout
    # and nothing else, exactly as before.
    from gpuwm.provenance_gate import announce_for_main

    refusal = announce_for_main("gpuwm-prepared-forecast")
    if refusal is not None:
        print(f"prepared_single_domain_forecast: {refusal}",
              file=sys.stderr)
        return 2
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
            # prepare, a base config that is not there, a physics value
            # the named profile would have overwritten -- is still a
            # sentence rather than a traceback.  It is labelled as what
            # it is, not as an --output-directory problem.  Rendered,
            # because a layered message printed raw leaks the
            # ``[[explain]]`` marker onto the terminal.
            print("prepared_single_domain_forecast: refused: "
                  + render_explanation(
                      str(error), explain=explain_enabled(args),
                      command=(
                          "python -m gpuwm.prepared_single_domain_forecast "
                          "--materialize-authorities")),
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
    # A mode flag that reached here was not in first position, so no
    # interception above could claim it.  Say that, rather than letting
    # the forecast parser refuse it for the unrelated reason that its own
    # required flags are missing.
    for flag in MODE_FLAGS:
        if flag in argv:
            print(f"prepared_single_domain_forecast: {flag} must be the "
                  "FIRST argument on the command line; it selects a "
                  "different program, with its own --help and its own "
                  "options", file=sys.stderr)
            return 2
    args = _parse_args(argv)
    # THE capability preflight, from the same registry `gpuwm run` and
    # `gpuwm go` refuse with, so this documented `python -m` door cannot
    # drift from the subcommands.  It had no handler at all: a missing
    # GPU runtime surfaced as a raw traceback AFTER this function had
    # claimed the output directory, written progress.json and run the
    # whole preparation preflight -- work a reader then has to clean up
    # for a gap that was knowable before any of it.
    from gpuwm import capabilities
    from gpuwm.explain import split as split_explanation

    try:
        capabilities.require(
            "python -m gpuwm.prepared_single_domain_forecast",
            *capabilities.COMMAND_REQUIREMENTS["run"],
            before=("Refusing here, before the output directory is "
                    "claimed and before the preparation preflight runs."))
    except capabilities.CapabilityMissing as refused:
        # The action half only, with NO ``--explain`` pointer: that flag
        # is registered on the ``--materialize-authorities`` parser, not
        # on this forecast one, so a pointer at it would name a flag this
        # door rejects with a usage dump -- which is precisely the class
        # of defect this whole change exists to remove.  Measured before
        # it shipped: `python -m gpuwm.prepared_single_domain_forecast
        # --explain ...` exits 2 on `unrecognized arguments`.
        print(split_explanation(str(refused))[0], file=sys.stderr)
        return 2
    # Before the clock defaults, before `--tiles`, and above all before
    # the output directory is claimed: `--proof-sha256` is the digest a
    # reader is most likely to take from the wrong place, and answering
    # it from inside the preflight meant a traceback AFTER a run
    # directory had been created for them to clean up.
    proof_refusal = _proof_digest_refusal_at_the_door(args)
    if proof_refusal is not None:
        print(f"prepared_single_domain_forecast: {proof_refusal}",
              file=sys.stderr)
        return 2
    # After the capability gate, because the clock defaults are READ FROM
    # THE CONFIG: a reader with no runtime should hear about the runtime,
    # not about the shape of a file whose contents cannot matter yet.
    _resolve_clock_arguments(args)
    # Before the output directory is claimed: a malformed [tiles] is a
    # usage mistake, and refusing it after creating a run directory
    # leaves the caller a directory to clean up for a typo.
    try:
        tiles = _streaming_options_argument(args.tiles)
    except (ValueError, TypeError) as error:
        print(f"prepared_single_domain_forecast: --tiles refused: {error}",
              file=sys.stderr)
        return 2
    # A rejected --outdir is a usage mistake, not a crash: one sentence
    # naming the problem and a directory that works, never a traceback.
    try:
        outdir = claim_output_directory(
            args.outdir, protected_roots=(
                args.prepared_root, args.experiment_config, args.wps_namelist,
                *((Path(args.restart).parent,) if args.restart is not None else ())))
    except (ValueError, FileExistsError) as error:
        print(f"prepared_single_domain_forecast: --outdir refused: {error}",
              file=sys.stderr)
        return 2
    started = time.perf_counter()
    # Armed here, before the preflight: the frame this renders is the
    # analysis at t = 0, and the history alarm is true at t = 0, so it is
    # durable before a single step is integrated.  Anything armed after
    # the forecast opens is already racing the thing it exists to catch.
    first_products = _route_owned_first_products(
        args, outdir=outdir, observer=observer, started=started)
    _atomic_json(outdir / "progress.json", {
        "schema": PROGRESS_SCHEMA,
        "status": "VALIDATING_PREPARATION",
        "source": args.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": args.run_seconds,
    }, heartbeat=True)
    # Read before ANY of this runner's work, so it describes the cache
    # this run inherited rather than the cache this run has been
    # filling.  Pure filesystem; the device is not touched here.
    kernel_cache_census = scan_kernel_cache()
    try:
        # Timed, because it was dark and it is not free: every digest in
        # the preparation receipt is re-checked against the bytes on
        # disk here, so this scales with the prepared cache and is the
        # first thing a reader waiting on "why has nothing happened yet"
        # is actually waiting for.
        preflight_started = time.perf_counter()
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
            domain_bundle=args.domain_bundle,
            tiles=tiles)
        preflight_seconds = time.perf_counter() - preflight_started
        verification = dict(inputs.physics_receipt).get("verification")
        if (isinstance(verification, dict)
                and verification.get("status") != "wrf-verified"
                and isinstance(verification.get("sentence"), str)):
            # One sentence, and the run continues (owner posture:
            # warn-not-block; detail lives in the receipt).
            print(f"prepared forecast: {verification['sentence']}",
                  file=sys.stderr)
        report = run_prepared_forecast(
            inputs, output_directory=outdir, observer=observer,
            first_products=first_products, stream_init=args.stream_init,
            progress_options=ProgressOptions.from_args(args),
            preflight_seconds=preflight_seconds,
            kernel_cache_census=kernel_cache_census,
            restart=args.restart, health_debug=args.health_debug)
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
        }, heartbeat=True)
        if isinstance(error, PreparationProofDigestMismatch):
            # A run that never started, not one that failed: the door
            # check above catches every command-line spelling of this,
            # and this is the same answer for the paths that reach the
            # preflight some other way.  A traceback where a user can
            # land is the defect; the refusal names both digests.
            print(f"prepared_single_domain_forecast: {error}",
                  file=sys.stderr)
            return 2
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
