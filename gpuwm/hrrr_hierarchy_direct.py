"""Sealed HRRR root preparation -> parallel nested stock-WRF inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tomllib
from types import SimpleNamespace

from gpuwm import runtime_manifest
from gpuwm.aerosol_source_receipt import (
    AEROSOL_SOURCE_KEY,
    aerosol_source_report_entry,
)
from gpuwm.config import radiation_scheme_ids
from gpuwm.experiment import build_experiment
from gpuwm.hrrr_forecast import hrrr_forcing_end_hour
from gpuwm.hrrr_native_static import (
    sha256_file,
    verified_static_catalog,
    verify_hrrr_native_static,
)
from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
from gpuwm.ingest.hrrr import load_hrrr_native_series
from gpuwm.ingest.hrrr_target import load_hrrr_target_domain
from gpuwm.ingest.source_coverage import owns_source_coverage_refusal
from gpuwm.ingest.nest_init import NestedInputCatalog, ParentInitView
from gpuwm.ingest.prepared_cache import (
    PreparedCacheReader,
    compare_prepared_domain_config,
    effective_prepared_domain_config,
    prepared_domain_config_identity,
    prepared_cache_identity,
    restore_prepared_cache,
)
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.namelist_import import import_namelists, parse_namelist
from gpuwm.namelist_seal import validated_namelist_extension_invariant
from gpuwm.native_domain_artifacts import _atomic_staging_sibling
from gpuwm.native_hierarchy import initialize_and_export_native_hierarchy
from gpuwm.native_wrf_contract import validate_native_lambert_contracts
from gpuwm.static.corridor import (
    STATICS_CORRIDOR_DIRNAME,
    emit_statics_corridor_set,
    validated_corridor_selection,
)
from gpuwm.static.lambert import grids_from_projection_config
from gpuwm.core.microphysics_transition import resolve_microphysics_transition


SCHEMA = "gpuwm-native-hrrr-hierarchy-direct-v1"
_SUPPORTED_PHYSICS = {
    "sf_sfclay_physics": 91,
    "sf_surface_physics": 2,
    "cu_physics": 0,
}
#: The PBL closures the certified native HRRR slice admits on the ROOT,
#: the same enumerated set
#: :data:`gpuwm.hrrr_route_inputs.ADMITTED_PBL_PHYSICS` admits at
#: emission, and admitted here for the same reason: each member is the
#: value a REGISTERED HRRR preparation profile pins -- 1 (YSU) for the
#: profiles this route has always carried, 11 (Shin-Hong 2015) for
#: ``thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1``, the composition a
#: physics-fidelity arm selecting divergence-ledger entry L3 resolves to.
#: It was a single pinned 1 in :data:`_SUPPORTED_PHYSICS` until that
#: template was registered.
#:
#: Widening the slice this way costs nothing in PREPARATION and is the
#: same evidence the child exemption below already rests on: preparation
#: writes static fields and an interpolated initial state, and the
#: closure selects tendencies only the forecast computes (the grep
#: recorded at :data:`_DOMAIN_PREPARATION_OVERRIDES` returns a single hit
#: for ``bl_pbl_physics``, and it is a comment).  What the registration
#: adds is the other half: a root PREPARED for this suite, by a shipped
#: profile, rather than a suite nothing can prepare.
_ADMITTED_PBL_PHYSICS = frozenset({1, 11})
#: Radiation is admitted as a (ra_lw_physics, ra_sw_physics) PAIR, the
#: same two pairs :data:`gpuwm.hrrr_route_inputs.ADMITTED_RADIATION_PAIRS`
#: admits at emission (B4 route-qualification motion, items 1-3): the
#: certified (0, 1) validation suite and the resolved RRTMG (4, 4) pair
#: the shipped registry's prepared-tree row already claims for HRRR.
#: The evidence that widening preparation this way is safe is the same
#: evidence this file already published for ``bl_pbl_physics`` at
#: :data:`_CHILD_PHYSICS_SLICE_OVERRIDES`: preparation writes static
#: fields and an interpolated initial state, and a sweep of
#: ``gpuwm/ingest/``, ``gpuwm/native_hierarchy.py``,
#: ``gpuwm/native_domain_artifacts.py`` and ``tools/prepare_hrrr_wrf.py``
#: for ``ra_physics|ra_lw_physics|ra_sw_physics|ra_rrtmg_variant``
#: returns zero hits -- radiation selects tendencies only the forecast
#: computes.  ``ra_physics = 0`` stays pinned beside the pair because
#: both admitted compositions spell radiation explicitly, and
#: ``validate_run_config`` requires exactly that spelling.
_ADMITTED_RADIATION_PAIRS = frozenset({(0, 1), (4, 4)})
# WRF v4.6.1 Registry/Registry.EM_COMMON:3015 declares Kessler's
# qv/qc/qr package.  Native-HRRR initialization retains QC/QR and produces
# an explicit discard receipt for the source-only frozen species before this
# direct hierarchy path sees the state.
#: 28 is aerosol-aware Thompson (Registry/Registry.EM_COMMON:3036).  It is
#: admitted here on the same footing as every other entry -- this set answers
#: "does the direct hierarchy path know this selector", not "is the scheme
#: mature".  Maturity and reachability are the registry's answer, and mp=28
#: is registered as a per-domain component override with no template, so
#: adding it here cannot make it a default anywhere.
_SUPPORTED_MICROPHYSICS = frozenset({1, 6, 8, 10, 16, 18, 28})
_DOMAIN_PREPARATION_OVERRIDES = frozenset({
    "cu_physics", "cudt_minutes", "radt", "radt_minutes", "bldt",
    "diff_6th_factor", "epssm", "spec_exp", "mp_physics", "moist",
    "moist_cq", "nest_microphysics_transition",
    # Per-domain history cadence.  This is the ladder's whole point -- a
    # 3 km parent written hourly beside a 1 km child written every 15
    # minutes -- and the machinery for it already exists end to end: the
    # experiment loader gives every domain its own history_interval_s and
    # divisibility check (gpuwm/experiment.py), the clock registers a
    # per-domain history alarm (gpuwm/core/clock.py:586), and the tree
    # runner writes per-domain wrfouts on it
    # (gpuwm/prepared_domain_tree_forecast.py, io_modes "history cadence:
    # per-domain experiment TOML").  Only this drift check disagreed, and
    # it did so AFTER the expensive preparation: a wizard-emitted
    # 3600/900 ladder passed the wizard, the root preparer and fetch, then
    # died on "output_interval_s (900.0, 3600.0)".  Preparation itself
    # writes no history frame at all.
    "output_interval_s",
    # The per-domain turbulence row (gpuwm/experiment.py
    # _DOMAIN_RUN_OVERRIDES, "a PBL parent may carry a PBL-off
    # Smagorinsky child").  Same shape of finding as output_interval_s
    # above: HRRR hierarchy PREPARATION never reads one of these keys.
    # `grep -rn 'bl_pbl_physics\|km_opt' gpuwm/ingest/ gpuwm/native_hierarchy.py
    # tools/prepare_hrrr_wrf.py` returns a single hit, and it is a comment.
    # Preparation writes static fields and an interpolated initial state;
    # the closure selects tendencies that only the forecast computes.  The
    # one place the turbulence choice reaches preparation is
    # DomainState(cfg) allocating km_opt=2's zero-initialised tke/tke0,
    # which the prepared cache round-trips like any other state array.
    "bl_pbl_physics", "km_opt", "c_s", "c_k", "diff_6th_opt",
    "mix_isotropic", "mix_upper_bound", "isfflx",
    "tke_heat_flux", "tke_drag_coefficient", "tke_upper_bound",
    # The P3 inflow-seeding keys, for the third time the same finding:
    # preparation never reads them, and this drift check refused them
    # AFTER the expensive preparation.  The ruling that they are
    # preparation-inert is not new here and is not being made here -- it
    # is already committed one module over, as
    # gpuwm.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS
    # (prepared_cache.py:241-246), whose published reason applies
    # verbatim: the generator acts at runtime FORCE on the child-owned
    # rolling NEST boundary tables, which preparation never computes, so
    # a tree prepared with any value of these fields holds exactly the
    # prepared state every value of them runs from.  Provenance for the
    # ruling itself: controller decision 2026-08-03 under Drew's standing
    # delegation, docs/superpowers/receipts/les/
    # INFLOW-GENERATOR-ACCEPTANCE-V2.md item 10.  These stay
    # trajectory-RELEVANT inside experiment_fingerprint and the restart
    # identity; only the two preparation-side comparisons drop them, and
    # until now only one of the two did.
    "inflow_perturbation", "inflow_perturbation_seed",
    "inflow_perturbation_amplitude_scale", "inflow_perturbation_faces",
})
#: Switches of the certified HRRR root slice -- :data:`_SUPPORTED_PHYSICS`
#: and the enumerated :data:`_ADMITTED_PBL_PHYSICS` beside it -- that a
#: CHILD may hold away from.  The slice is a statement about what the
#: native-HRRR INITIALIZATION supports, and it is pinned in full on the
#: root, which is the domain that is actually initialized from HRRR and
#: the domain the optional stock-WRF export's own v2-slice branch reads.
#: Nothing in the preparation consumes a child's PBL selection (see the
#: note in _DOMAIN_PREPARATION_OVERRIDES); pinning it here refused the
#: PBL-off LES child that the experiment schema, the forecast runner and
#: the namelist importer all admit, and refused it after the expensive
#: root preparation had already run.  A child therefore stays free of the
#: root's enumerated admission too: an LES child runs PBL off, which no
#: preparation profile pins and none needs to.
_CHILD_PHYSICS_SLICE_OVERRIDES = frozenset({"bl_pbl_physics"})
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


def _sealed_root_rrtmg_variant(identity: dict[str, object]) -> str | None:
    """The RRTMG variant the sealed root preparation pinned, if any.

    A WRF namelist spells radiation as selector integers; which 4/4
    IMPLEMENTATION serves them (RTE+RRTMGP or the exact legacy port) is
    the sealed root's physics profile's decision, recorded in its cache
    identity.  This reads that decision so the hierarchy imports its
    namelists under the same variant.  Absent or malformed shapes return
    None (the importer's default) rather than refusing here: the
    d01-binding comparison downstream still refuses any real mismatch,
    so failing open cannot admit a wrong trajectory -- it only preserves
    the behaviour of headers written before the field existed.
    """
    domain_config = identity.get("domain_config")
    if not isinstance(domain_config, dict):
        return None
    run = domain_config.get("run")
    if not isinstance(run, dict):
        return None
    variant = run.get("ra_rrtmg_variant")
    return variant if isinstance(variant, str) else None


def _native_experiment(wps_namelist: Path, namelist_input: Path,
                       *, rrtmg_variant: str | None = None,
                       acknowledgements: tuple[str, ...] = ()):
    keywords = ({} if rrtmg_variant is None
                else {"rrtmg_variant": rrtmg_variant})
    resolved, report = import_namelists(
        wps_namelist, namelist_input, name="native_hrrr_hierarchy",
        acknowledgements=tuple(acknowledgements), **keywords)
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

    The longwave delta is a PER-DOMAIN substitution, not a constant: the
    stock arm selects RRTM (1) exactly where the native arm runs with
    longwave off (0), and carries any other admitted longwave selection
    unchanged -- so under the resolved RRTMG (4, 4) pair the delta
    collapses and both arms run 4.  ``ghg_input=0`` stays stock-only in
    both cases: on the (0, 1) suite it fixes the substituted RRTM's gas
    table, and on the (4, 4) suite it mirrors what
    ``gpuwm/core/rrtmg_legacy.py`` pins on the native side.  Pinning it
    is mandatory either way: WRF's default ``1`` reads the time-varying
    CAM gas table and is not a valid implicit substitute for the
    fixed-gas configurations used by the acceptance gate.

    ``do_radar_ref=1`` is the second stock-only key and is mandatory for
    the same class of reason: it is a setting the native arm answers in
    CODE rather than from a namelist, so the native file must stay
    silent about it while the stock file has to be told.  gpuwm
    evaluates REFL_10CM at output time unconditionally; WRF allocates
    the array only under ``package radar_refl compute_radar_ref==1``.
    At WRF's default the stock arm's history frames carry no REFL_10CM
    at all, and every reflectivity score on that arm is unanswerable.
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
        stock_radar_ref = stock["physics"]["do_radar_ref"]
    except KeyError as exc:
        raise ValueError(
            "both native and stock namelists must explicitly declare "
            "&domains/max_dom, "
            "&physics/ra_lw_physics and &dynamics/use_theta_m; the stock "
            "namelist must also declare &physics/ghg_input and "
            "&physics/do_radar_ref") from exc
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

    if (not isinstance(native_lw, list) or len(native_lw) != max_dom
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value not in (0, 4) for value in native_lw)):
        raise ValueError(
            "native hierarchy namelist must declare one explicit "
            "ra_lw_physics entry per domain, each 0 (longwave off) or 4 "
            f"(RRTMG); got {native_lw}")
    stock_lw_expected = [1 if value == 0 else value for value in native_lw]
    if (not _same_typed_values(stock_lw, stock_lw_expected)
            or not _same_typed_values(native_theta, [0])
            or not _same_typed_values(stock_theta, [1])
            or "ghg_input" in native["physics"]
            or not _same_typed_values(stock_ghg, [0])
            # do_radar_ref stays FORBIDDEN in the native namelist (the
            # loop above) and is now MANDATORY at 1 in the stock one:
            # gpuwm evaluates REFL_10CM at output time whatever a
            # namelist says, while WRF allocates the array only under
            # `package radar_refl compute_radar_ref==1`
            # (Registry.EM_COMMON:3059, fed by do_radar_ref via
            # module_check_a_mundo.F:3477).  Left at the Registry
            # default the stock arm writes history frames with no
            # REFL_10CM at all and every reflectivity score on that arm
            # is unanswerable -- measured, first WRF-arm scoring
            # attempt.  A scalar: Registry.EM_COMMON:2447, nentries 1.
            or not _same_typed_values(stock_radar_ref, [1])):
        raise ValueError(
            "the evidenced raw namelist deltas require ra_lw_physics "
            "0 -> 1 exactly where the native entry is 0 (a native 4 "
            "carries to the stock arm unchanged), use_theta_m=0 -> 1, "
            "stock-only ghg_input=0, and stock-only do_radar_ref=1")
    normalized_stock = deepcopy(stock)
    normalized_stock["physics"]["ra_lw_physics"] = list(native_lw)
    normalized_stock["physics"].pop("ghg_input")
    normalized_stock["physics"].pop("do_radar_ref")
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
            "ra_lw_physics 0 -> 1, use_theta_m 0 -> 1, stock-only "
            "ghg_input=0, and stock-only do_radar_ref=1: "
            + ", ".join(differences))
    return {
        "schema": "gpuwm-native-to-stock-namelist-delta-v4",
        "status": "PASS",
        "max_dom": max_dom,
        "certified_native_runtime": observed_pins,
        "allowed_deltas": {
            "physics.ra_lw_physics": {
                "native": list(native_lw), "stock": stock_lw_expected},
            "dynamics.use_theta_m": {"native": [0], "stock": [1]},
            "physics.ghg_input": {"native": None, "stock": [0]},
            "physics.do_radar_ref": {"native": None, "stock": [1]},
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


def namelist_start_refusal(*, requested: datetime, observed: datetime,
                           namelist_name: str) -> str:
    """Why the namelist's start time and model time zero must be one instant.

    A pure function so the sentence a mis-copied time meets is testable
    without a sealed root preparation on disk.
    """

    return (
        "model time zero differs from the native namelist start: this "
        f"stage was told {requested:%Y-%m-%d %H:%M:%S} and "
        f"{namelist_name} starts at {observed:%Y-%m-%d %H:%M:%S}.  The two "
        "must be the same instant.  If the root preparation began at a "
        "forecast lead, model time zero is cycle + K, not the cycle: pass "
        "--cycle CYCLE --forecast-start-hour K (the same pair "
        "tools/prepare_hrrr_wrf.py was given), and emit the namelist with "
        "`gpuwm domain --forecast-start-hour K` so its start_* keys carry "
        "the same instant.")


def sealed_start_refusal(*, requested: datetime, sealed_start: datetime,
                         sealed_cycle: object) -> str:
    """Why the sealed cache's model time zero and the request must agree.

    Names the cycle and the lead the root was actually prepared at
    whenever the cache recorded them, because that pair is the answer:
    the same two values passed here reproduce this instant exactly.
    """

    lead = ""
    if isinstance(sealed_cycle, str):
        try:
            offset = sealed_start - datetime.fromisoformat(sealed_cycle)
        except ValueError:
            pass
        else:
            lead = (f"  That root was prepared from cycle {sealed_cycle} at "
                    f"lead f{int(offset.total_seconds() // 3600):02d}.")
    return (
        "root preparation model time zero differs from request: the sealed "
        f"cache starts at {sealed_start:%Y-%m-%d %H:%M:%S}, this stage was "
        f"told {requested:%Y-%m-%d %H:%M:%S}." + lead
        + "  Pass --cycle CYCLE --forecast-start-hour K -- the same pair "
        "the root preparation was given -- so both stages derive the same "
        "model time zero.")


def sealed_source_leads(identity: Mapping[str, object],
                        forcing_hours) -> tuple[int, ...]:
    """The ABSOLUTE HRRR leads the sealed root's decoded bridge tree holds.

    Two hour numberings meet here and are equal only at lead 0.  The
    prepared cache's ``forcing_hours`` are MODEL-relative -- 0, 1, 2 --
    because that is what a forecast clock counts.  The decoder's
    published tree is keyed by NOAA's absolute cycle-relative lead: its
    directories are ``atmosphere-f06``/``soil-f06`` and its gate declares
    ``(6, 7, 8)``.  Handing the model-relative 0 to that tree asked for
    an hour it does not carry, and a root prepared at f06 refused with
    "forecast_hour must be one of (6, 7, 8), got 0" after every check
    above it had passed.

    The sealed cache records both sequences in its own source identity,
    so the absolute leads are read from the bundle rather than
    reconstructed.  A cache sealed before that field existed carries only
    the model-relative inventory, which at the lead 0 those releases
    could express IS the absolute one -- so the fallback is exact, not a
    guess.
    """

    model_relative = tuple(int(hour) for hour in forcing_hours)
    source_identity = identity.get("source_identity")
    leads = (source_identity.get("source_forecast_hours")
             if isinstance(source_identity, Mapping) else None)
    if leads is None:
        return model_relative
    absolute = tuple(int(hour) for hour in leads)
    if len(absolute) != len(model_relative):
        raise ValueError(
            "root preparation source identity declares "
            f"{len(absolute)} absolute forecast lead(s) but its cache "
            f"carries {len(model_relative)} model forcing hour(s)")
    if absolute != tuple(range(absolute[0], absolute[0] + len(absolute))):
        raise ValueError(
            "root preparation absolute forecast leads must be contiguous "
            f"and ordered; got {list(absolute)}")
    return absolute


def sealed_forcing_horizon_seconds(forcing_hours) -> float:
    """The forecast length one sealed root preparation can force.

    The prepared root carries a contiguous hourly forcing inventory
    starting at its own hour zero; a hierarchy may run to its last hour
    and no further.  That is a property of the bundle in front of us, so
    it is read from the bundle.
    """

    hours = [int(hour) for hour in forcing_hours]
    expected = list(range(len(hours)))
    if not hours or hours != expected:
        raise ValueError(
            "root preparation forcing inventory must be contiguous hourly "
            f"leads starting at 0; got {hours}")
    return float(hours[-1]) * 3600.0


def verified_root_forcing_inventory(
        forcing_hours, *, run_seconds: float) -> tuple[int, ...]:
    """Require exactly the inventory the preparer seals for this run.

    ONE endpoint convention on both stages, and it is a ceiling.
    ``tools.prepare_hrrr_wrf`` sizes its fetch/decode window with
    :func:`gpuwm.hrrr_forecast.hrrr_forcing_end_hour`: hourly boundary
    forcing brackets every model instant between two frames, so a 900 s
    run -- whose endpoint at 0.25 h lies BETWEEN f000 and f001 -- is
    prepared with model forcing hours (0, 1).  This check used to
    recompute the endpoint with a floor (``run_seconds // 3600``),
    expected ``(0,)`` for that same run, and refused every sub-hour root
    the preparer can build; ``(0,)`` is also a series the tree runner
    (gpuwm/prepared_domain_tree_forecast.py) rejects outright, because a
    single frame brackets nothing.  The expectation is derived from the
    shared ceiling now, and a genuinely truncated, over-long, or gapped
    inventory still refuses with the same sentence.
    """

    observed = tuple(forcing_hours)
    expected = tuple(range(hrrr_forcing_end_hour(run_seconds) + 1))
    if observed != expected:
        raise ValueError(
            "root preparation must contain consecutive hourly HRRR forcing "
            f"through the forecast endpoint: expected {expected}, "
            f"got {observed}")
    return observed


def _supported_hierarchy_slice(exp, root_target, *, forcing_hours) -> None:
    """Bind the public adapter to its sealed root and generic valid nests.

    ``build_experiment`` is the geometry authority: it rejects cycles,
    orphans, invalid ratios, misaligned extents, and insufficient parent-row
    clearance before this gate runs.  The target-domain document and sealed
    root preparation are the d01 authority; this function verifies that the
    requested namelist describes that root without pinning one location,
    duration, or vertical grid.  The fixed native HRRR physics product remains
    explicit and children may vary only in their geometry and nest identity.

    ``forcing_hours`` is the sealed root preparation's own inventory.  The
    duration ceiling is derived from it rather than from a constant: the
    docstring above already promised not to pin a duration, and every
    other stage of this route -- ``hrrr_source_window`` in the fetch/root
    wrapper, the forcing-hour equality below, the tree runner -- handles
    f00..f24 dynamically.  A hardcoded 12 h refused an f00..f24 root
    preparation that had already been fetched and prepared, with a
    sentence ("the native f00..f12 forcing horizon") describing a
    limitation of neither the data nor the code.
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
    horizon = sealed_forcing_horizon_seconds(forcing_hours)
    if not 0.0 < exp.run_seconds <= horizon:
        raise ValueError(
            "the public HRRR hierarchy duration must be positive and no "
            f"longer than the {horizon / 3600.0:g} h forcing horizon this "
            "root preparation was sealed with (f00.."
            f"f{int(horizon // 3600):02d}); requested "
            f"{exp.run_seconds / 3600.0:g} h.  Re-prepare the root over a "
            "longer window to run longer.")
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
        "dt": float(root_target.time_step_exact),
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
        exempt = (frozenset() if domain is root_domain
                  else _CHILD_PHYSICS_SLICE_OVERRIDES)
        if ("bl_pbl_physics" not in exempt
                and int(domain.run.bl_pbl_physics)
                not in _ADMITTED_PBL_PHYSICS):
            raise ValueError(
                f"d{domain.grid_id:02d} is outside the certified native "
                f"HRRR PBL slice: bl_pbl_physics="
                f"{domain.run.bl_pbl_physics}; the route admits "
                f"{sorted(_ADMITTED_PBL_PHYSICS)}, each pinned by a "
                "registered HRRR preparation profile.")
        mismatch = {
            name: (getattr(domain.run, name), expected)
            for name, expected in _SUPPORTED_PHYSICS.items()
            if getattr(domain.run, name) != expected and name not in exempt
        }
        if mismatch:
            raise ValueError(
                f"d{domain.grid_id:02d} is outside the certified native "
                f"HRRR physics slice: {mismatch}")
        # Compared in the RESOLVED explicit form, through the config's own
        # resolver: the WRF namelist importer emits a coupled 4/4 request
        # as the historical aggregate spelling (ra_physics=4 with the
        # split fields at their -1 defaults), and gpuwm/config.py
        # documents that spelling as preserving the aggregate exactly.
        # Comparing raw fields here made the (4, 4) admission unreachable
        # for every namelist-imported experiment -- the error text
        # advertised a case no import could satisfy -- while the same
        # selection spelled explicitly passed.  radiation_scheme_ids is
        # the production resolver and itself refuses an incoherent
        # spelling (mixed split/aggregate), so nothing is widened: the
        # admitted physics is the same two resolved pairs.
        pair = radiation_scheme_ids(domain.run)
        if pair not in _ADMITTED_RADIATION_PAIRS:
            raise ValueError(
                f"d{domain.grid_id:02d} is outside the certified native "
                f"HRRR radiation slice: resolved (ra_lw_physics, "
                f"ra_sw_physics)={pair}; the route admits "
                f"{sorted(_ADMITTED_RADIATION_PAIRS)}."
                "  Under (0, 1) the separately supplied stock-WRF "
                "namelist selects longwave 1; under (4, 4) both arms "
                "run 4.")
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
    for native_domain, stock_domain in zip(native_exp.domains,
                                           stock_exp.domains):
        expected = (1 if native_domain.run.ra_lw_physics == 0
                    else native_domain.run.ra_lw_physics)
        if stock_domain.run.ra_lw_physics != expected:
            raise ValueError(
                f"stock-WRF namelist d{stock_domain.grid_id:02d} must "
                f"select ra_lw_physics={expected} (RRTM substitutes the "
                "native arm's disabled longwave; any other native "
                "longwave carries unchanged)")
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


def _effective_domain_config(domain) -> dict[str, object]:
    """Canonicalize cadence fields that are inactive under selected schemes.

    One line now, because the normalization moved to
    :func:`gpuwm.ingest.prepared_cache.effective_prepared_domain_config`
    where every consumer of a prepared-domain identity can reach it --
    this gate and the tree forecast's preflight were normalizing
    differently, which is how a wizard-emitted config could pass here and
    be refused there.
    """

    return effective_prepared_domain_config(
        prepared_domain_config_identity(domain))


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
    if not isinstance(prepared_domain.get("run"), dict):
        raise ValueError("root preparation domain_config lacks run controls")
    effective_prepared_domain = effective_prepared_domain_config(
        deepcopy(prepared_domain))
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


def _expected_root_cache_identity(
        identity: dict[str, object], *, root_domain,
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        static_cache_sha256: str, forcing_hours,
) -> dict[str, object]:
    """Reconstruct the exact cache identity consumed by the hierarchy."""

    root_namelist_sha256, prepared_domain = \
        _validated_root_preparation_binding(identity, root_domain)
    invariant = identity.get("namelist_extension_invariant")
    if invariant is not None:
        invariant = validated_namelist_extension_invariant(
            invariant, context="root preparation identity")
    expected = prepared_cache_identity(
        bridge_manifest_sha256=bridge_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        static_cache_sha256=static_cache_sha256,
        namelist_sha256=root_namelist_sha256,
        domain_config=root_domain,
        forcing_hours=forcing_hours,
        source_identity=identity["source_identity"],
        namelist_extension_invariant=invariant,
    )
    # The sealed single-domain preparer stores legacy cadence shadow fields.
    # Their effective equality was checked above; preserve the exact sealed
    # document for cache verification.
    expected["domain_config"] = prepared_domain
    return expected


def _source_identity(cpu_bridge: Path) -> dict[str, object]:
    """What this install is, resolved the one way every consumer does.

    :func:`gpuwm.runtime_manifest.provenance` is the single ladder
    ``gpuwm doctor`` reports: the sealed manifest validated against its
    WHOLE schema, then a genuine checkout of THIS tree, then the
    installed wheel's ``RECORD`` digests.  Both branches this replaced
    were the field report's bugs in one function -- a manifest checked a
    key at a time (a document missing ``contract`` entirely passed here
    and died three consumers later), and an unguarded ``git rev-parse``
    rooted at the package directory, which for the pip install that runs
    this nested HRRR route is ``site-packages``.
    """

    root = Path(__file__).resolve().parent.parent
    identity = dict(runtime_manifest.provenance(root))
    identity["cpu_preprocess_bridge"] = {
        "path": str(cpu_bridge),
        "bytes": cpu_bridge.stat().st_size,
        "sha256": sha256_file(cpu_bridge),
    }
    # The hierarchy's own extra demand, unchanged: a checkout publishing
    # this route may have nothing uncommitted.  It belongs to the git
    # path alone -- a sealed manifest has already asserted
    # `source.worktree_clean`, and a wheel has no worktree to dirty.
    if (identity.get("identity_source") == "git"
            and identity.get("git_status_short")):
        raise RuntimeError(
            "public HRRR hierarchy requires a completely clean source tree")
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
        cpu_bridge: Path | None = None, statics_corridor=None,
        acknowledgements: tuple[str, ...] = (),
) -> dict[str, object]:
    """Verify one root preparation and publish generic stock-WRF inputs.

    ``statics_corridor`` opts this stage into sealing child-resolution
    statics over each child's whole parent extent
    (:mod:`gpuwm.static.corridor`): ``None`` emits nothing and leaves
    the bundle byte-for-byte unchanged, ``"all"`` covers every child
    domain, and a sequence of grid ids covers exactly those children.

    This is the HRRR chain's corridor door, and it is HERE rather than
    in ``tools/prepare_hrrr_wrf`` for one reason: a corridor is
    child-resolution statics, and this is the stage that knows the
    children exist.  The root preparer prepares d01 alone and never
    reads a child geometry; this stage takes ``--geog-root``, builds
    d02..dNN, and already holds the verified GEOG catalog every child's
    statics come from.  The set receipt is bound into ``receipt.json``
    beside ``artifact_receipt``, exactly as the GFS chain binds it into
    ``proof.json``, which is what makes the sealed artifact rather than
    any printed line the contract the tree runner verifies.
    """

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
    # The sealed root preparation is the physics and forcing authority,
    # so read its identity BEFORE importing the namelists: the namelist
    # spells radiation as selector integers, and which 4/4 implementation
    # serves them is the sealed profile's recorded decision
    # (_sealed_root_rrtmg_variant), which the import must inherit for the
    # d01 binding below to compare like with like.  The forcing inventory
    # is read here for the same reason -- the duration ceiling is a
    # property of this bundle rather than a constant.
    cache_header = _json(paths["prepared_cache"] / "header.json")
    if (cache_header.get("schema") != "gpuwm-prepared-real-cache-v1"
            or cache_header.get("status") != "READY"):
        raise ValueError("root preparation cache is not READY")
    identity = cache_header.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("root preparation cache lacks an identity")
    forcing_hours = tuple(identity.get("forcing_hours", ()))
    native_exp, native_resolved, native_report = _native_experiment(
        Path(wps_namelist), Path(namelist_input),
        rrtmg_variant=_sealed_root_rrtmg_variant(identity),
        acknowledgements=tuple(acknowledgements))
    from gpuwm.static.highres_production import refuse_inert_highres
    refuse_inert_highres(root_domain_spec, lane="native-HRRR static path")
    target = load_hrrr_target_domain(root_domain_spec)
    _supported_hierarchy_slice(
        native_exp, target, forcing_hours=forcing_hours)
    # Resolved the moment a domain tree exists, and long before the
    # expensive restore/decode/join below: a corridor asked of a
    # single-domain namelist, or naming a grid id this tree does not
    # have, is a typed flag rather than a failure of the preparation,
    # and refusing it after the join would throw away work that
    # succeeded.  The emission re-resolves the same pure function.
    if statics_corridor is not None and len(native_exp.domains) < 2:
        raise ValueError(
            "--statics-corridor seals child-resolution statics over a "
            "parent extent, and this namelist has no child domain; "
            "remove the flag or prepare a domain tree")
    validated_corridor_selection(native_exp, statics_corridor)
    wps_runtime_contract = _require_raw_wps_contract(
        Path(wps_namelist), len(native_exp.domains))
    if native_exp.start_time != valid_time:
        raise ValueError(namelist_start_refusal(
            requested=valid_time, observed=native_exp.start_time,
            namelist_name=Path(namelist_input).name))

    static_fields, static_receipt = verify_hrrr_native_static(
        paths["static_cache"], paths["static_receipt"], target)
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
    verified_root_forcing_inventory(
        forcing_hours, run_seconds=native_exp.run_seconds)

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
    sealed_start = datetime.fromisoformat(
        user_metadata.get("initial_valid_time", ""))
    if sealed_start != valid_time:
        raise ValueError(sealed_start_refusal(
            requested=valid_time, sealed_start=sealed_start,
            sealed_cycle=user_metadata.get("source_cycle")))

    expected_identity = _expected_root_cache_identity(
        identity, root_domain=native_exp.root,
        bridge_manifest_sha256=bridge_sha,
        source_manifest_sha256=observed_source_sha,
        static_cache_sha256=static_sha,
        forcing_hours=forcing_hours,
    )
    root_namelist_sha = expected_identity["namelist_sha256"]
    namelist_sha = sha256_file(Path(namelist_input))
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
        paths["bridge"], sealed_source_leads(identity, forcing_hours)[:1],
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
            # The GPU runner consumes hierarchy-artifacts/;
            # wrf-native-input/ is the unchanged-WRF oracle set beside
            # it.  gfs_direct already asks for "optional" here, so a
            # state gpuwm can integrate but the WRF file format cannot
            # represent is recorded as a REFUSED export manifest in the
            # receipt instead of destroying the preparation.  This route
            # was the last caller inheriting "required".
            stock_wrf_export="optional",
        )
        hierarchy_seconds = time.perf_counter() - hierarchy_started
        # AFTER the artifact join, and inside the same staging directory
        # the atomic publication renames: the hierarchy tree is already
        # sealed on its own terms, and the corridor set lands beside it
        # under hierarchy-artifacts/, which is where the tree runner
        # looks for it whichever chain wrote the bundle.  The GFS chain
        # emits through this same function.
        corridor_receipt = emit_statics_corridor_set(
            exp=native_exp, grids=grids, static_catalog=static_catalog,
            directory=(staging / "hierarchy-artifacts"
                       / STATICS_CORRIDOR_DIRNAME),
            statics_corridor=statics_corridor)
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
            # Present only when this preparation opted in: the sealed
            # statics-corridor set, digest-bound here the way every
            # other sealed artifact of this bundle is, and read from
            # here by gpuwm-prepared-tree-forecast's preflight.  Absent,
            # the receipt is byte-for-byte what it always was.
            **({"statics_corridor": dict(corridor_receipt)}
               if corridor_receipt is not None else {}),
            # WHICH AEROSOL SOURCE THE ROOT PREPARATION USED.  This stage
            # publishes a hierarchy built ON that root state, so the fact
            # is this bundle's as much as the root's, and every forecast
            # launched from these artifacts inherits it.  Read from the
            # root prepared cache's own metadata rather than re-resolved:
            # re-resolving here would be a second resolution path over the
            # same config field, which is how a run reads one dataset and
            # reports another.  Empty -- and the receipt byte-for-byte
            # unchanged -- for every scheme with no aerosol fields.
            **aerosol_source_report_entry(
                cache_header.get("metadata", {}).get(
                    AEROSOL_SOURCE_KEY, {}),
                mp_physics=native_exp.root.run.mp_physics,
                when_unrecorded=(
                    "the root prepared cache at "
                    f"{paths['prepared_cache']} carries no "
                    "aerosol-initialization receipt, so it was written by "
                    "a preparation predating the receipt being stored; "
                    "re-prepare the root to record which source filled "
                    "its nwfa/nifa")),
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
    parser.add_argument(
        "--cycle",
        help="the HRRR CYCLE the sealed root preparation was built from, "
             "YYYY-MM-DD_HH:MM:SS (UTC).  Model time zero -- the instant "
             "this stage checks the namelist and the sealed cache against "
             "-- is cycle + --forecast-start-hour")
    parser.add_argument(
        "--forecast-start-hour", type=int, default=0,
        help="absolute HRRR cycle-relative lead the root preparation "
             "began at; the same value passed to tools/prepare_hrrr_wrf.py")
    parser.add_argument(
        "--valid-time",
        help="deprecated: on THIS command --valid-time meant model time "
             "zero, not the cycle (the two differ by the lead).  Accepted "
             "unchanged for v1.4.0 scripts; use --cycle with "
             "--forecast-start-hour")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ack", action="append", default=[], metavar="ID",
        help="declared-experiment acknowledgement id, forwarded into the "
             "namelist import (a WRF namelist has no spelling for a gpuwm "
             "governance declaration, so a profile that requires one -- "
             "e.g. the shortwave-only suites' "
             "constant-downward-longwave-v1 -- could never prepare a "
             "nested tree without this flag); repeatable, exactly as "
             "tools/prepare_hrrr_wrf.py spells it")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    # Spelled exactly as the GFS door spells it
    # (gpuwm/gfs_direct.py), because one runner consumes both bundles
    # and its refusal names this flag as the remedy whichever chain
    # prepared the tree.
    parser.add_argument(
        "--statics-corridor", nargs="?", const="all", default=None,
        metavar="GRID_IDS",
        help="also seal child-resolution statics over each child's whole "
             "parent extent (the moving-nest corridor); bare flag covers "
             "every child domain, or pass comma-separated child grid ids "
             "(e.g. 2,3).  Required before the prepared tree runner will "
             "honor a [relocation] follow source; omitted, the bundle is "
             "byte-for-byte unchanged")
    return parser


@owns_source_coverage_refusal
def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from gpuwm import explain
    from gpuwm.hrrr_forecast import resolve_cycle_flags

    if args.forecast_start_hour < 0:
        raise ValueError(
            "--forecast-start-hour must be a nonnegative forecast lead")
    if args.valid_time is not None and args.forecast_start_hour:
        raise ValueError(
            "--forecast-start-hour needs --cycle: on this command "
            "--valid-time is model time zero already, so adding a lead to "
            "it would move the clock twice.  Pass --cycle CYCLE "
            "--forecast-start-hour K instead.")
    # `--valid-time` on this stage always meant model time zero; `--cycle`
    # means the cycle and has the lead added to it.  Both land on the same
    # instant, which is the only thing the checks below compare.
    instant, from_valid_time = resolve_cycle_flags(
        args.cycle, args.valid_time, tool="hrrr_hierarchy_direct",
        legacy_means="model time zero (cycle + lead)", warn=explain.warn)
    valid_time = (
        instant if from_valid_time
        else instant + timedelta(hours=args.forecast_start_hour))
    statics_corridor = args.statics_corridor
    if statics_corridor is not None and statics_corridor != "all":
        try:
            statics_corridor = tuple(
                int(part) for part in statics_corridor.split(",") if part)
        except ValueError:
            raise ValueError(
                "--statics-corridor accepts 'all' or comma-separated "
                f"child grid ids, got {args.statics_corridor!r}") from None
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
        acknowledgements=tuple(args.ack),
        workers=args.workers, cpu_bridge=args.cpu_preprocess_bridge,
        statics_corridor=statics_corridor,
    )
    # The forecast runner takes --preparation-receipt-sha256 over the
    # receipt this stage just wrote, and the emitted chain tells the
    # reader the hierarchy prints it.  It did not: the one placeholder
    # in that chain no stage filled, so the printed route could not be
    # walked without hashing a file by hand.  Print it here, from the
    # published receipt rather than from the payload, so the digest is
    # of the bytes the next stage will read.
    # The export slot is a READY manifest with a `files` inventory OR the
    # NOT_REQUESTED/REFUSED document that says why there is none
    # (gpuwm/native_hierarchy.py: STOCK_WRF_EXPORT_MODES).  Reading
    # ["files"] unconditionally made the refused shape a KeyError here,
    # which threw away the prepared hierarchy that the "optional" mode
    # exists to keep -- the same failure the refusal being uncatchable
    # caused one layer down.  Report the slot as it is.
    wrf_manifest = payload["wrf_manifest"]
    print(json.dumps({
        "status": payload["status"],
        "output_root": str(args.output_root.resolve()),
        "workers": payload["workers"],
        "timing_seconds": payload["timing_seconds"],
        "wrf_export_status": wrf_manifest.get("status"),
        "wrf_files": wrf_manifest.get("files"),
        "wrf_export_refusal": wrf_manifest.get("reason"),
        "preparation_receipt_sha256": sha256_file(
            args.output_root / "receipt.json"),
    }, indent=2, sort_keys=True, allow_nan=False))
    corridor = payload.get("statics_corridor")
    if isinstance(corridor, dict):
        # Size honesty at the door, in the GFS door's own words: the
        # corridor is parent-extent at child resolution, and its cost is
        # stated where it is paid.
        for label, entry in sorted(corridor.get("domains", {}).items()):
            print(
                f"  statics corridor {label}: "
                f"{entry['corridor_nx']}x{entry['corridor_ny']} child "
                f"cells over the whole d{int(entry['parent_id']):02d} "
                f"extent, {entry['cache']['bytes'] / 1.0e6:.1f} MB on "
                f"disk, {entry['host_bytes'] / 1.0e6:.1f} MB host when "
                "loaded by a relocating run (no GPU residency)",
                file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
