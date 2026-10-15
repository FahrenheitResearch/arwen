"""Named member physics selections and fixed surface-state controls.

This roster changes existing single-run selectors and initial surface state.
Preparation remains the ordinary source chain. A prepared authority is
shared only when its trajectory and complete preparation configuration
match; different soil layouts never borrow each other's prepared cache.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from numbers import Real
import re
import struct

from gpuwm.ensemble.surface_controls import DEFAULTS, OPTIONS, validate_surface_recipe

LAND_KEYS = frozenset({"sf_surface_physics", "num_soil_layers"})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def member_physics_keys():
    """Use the ordinary schema, keeping source layout and draws out of arms."""
    from gpuwm.config import RunConfig
    from gpuwm.experiment import _SHARED_FORBIDDEN
    source = SOURCE_RUN_FIELDS | frozenset(_SHARED_FORBIDDEN) | frozenset({
        "eta_levels", "map_proj", "terrain_opt", "hill_height", "hill_halfwidth",
        "open_x", "open_y", "time_step_sound", "min_time_step_sound",
        "use_adaptive_time_step", "step_to_output_time", "adaptation_domain",
        "target_cfl", "target_hcfl", "max_step_increase_pct", "starting_time_step",
        "starting_time_step_den", "max_time_step", "max_time_step_den",
        "min_time_step", "min_time_step_den", "adaptive_nest_lattice",
    })
    # These mechanisms belong to the existing stochastic admission door.
    # A physics table must not provide another route around its refusal.
    return frozenset(item.name for item in fields(RunConfig)
        if item.name not in source and not item.name.startswith(("spp_", "inflow_perturbation")))


def normalize_member_variants(value, count):
    """Validate a complete named roster without importing numerical code."""
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError("member_variants must contain one named record for every requested member")
    result = []
    names = set()
    descriptors = {}
    for number, row in enumerate(value):
        if not isinstance(row, dict) or set(row) - {"name", "physics", "surface"}:
            raise ValueError(f"member variant {number} takes name, physics and surface")
        name = row.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"member variant {number} needs a single identifier of at most 64 characters")
        if name in names:
            raise ValueError(f"member variant names must be unique: {name}")
        names.add(name)
        physics = row.get("physics", {})
        if not isinstance(physics, dict):
            raise ValueError(f"member variant {name} physics must be a table")
        unknown = set(physics) - member_physics_keys()
        if unknown:
            raise ValueError(f"member variant {name} physics has unsupported keys {sorted(unknown)}; "
                             "use existing single-run physics selectors. Geometry and clocks stay "
                             "source-bound, and stochastic controls keep their own admission; "
                             "otherwise the member would borrow the wrong prepared grid or bypass a refusal")
        if set(physics) & LAND_KEYS:
            if not LAND_KEYS <= set(physics) or any(type(physics[key]) is not int for key in LAND_KEYS):
                raise ValueError(f"member variant {name} must name both integer land selectors")
            from types import SimpleNamespace
            from gpuwm.config import soil_layer_count
            try:
                soil_layer_count(SimpleNamespace(**{key: physics[key] for key in LAND_KEYS}))
            except ValueError as exc:
                raise ValueError(f"member variant {name}: {exc}; another pairing would initialize "
                                 "the wrong land column") from exc
            except ModuleNotFoundError as exc:
                # The preparation-only RW-WPS package stages no forecast-side
                # soil contract (gpuwm.config defers scheme 3's read for that
                # reason; tools/build_rw_wps_release.py names why). The pairing
                # is checked again by the forecast door that runs the member,
                # which carries the contract; a preparation install cannot run
                # the column the check protects. Any other missing module is a
                # packaging defect and still raises.
                if not (exc.name or "").startswith("gpuwm.core."):
                    raise
        surface = row.get("surface", {})
        if not isinstance(surface, dict) or set(surface) - set(OPTIONS):
            raise ValueError(f"member variant {name} surface takes soil_moisture_scale and sst_offset_k")
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in surface.values()):
            raise ValueError(f"member variant {name} names fixed scalar surface controls; "
                             "interval draws belong to the surface-state recipe")
        surface = {**DEFAULTS, **surface}
        surface = validate_surface_recipe({"kind": "surface-state", **surface})
        surface.pop("kind")
        descriptor = (_canonical(physics), surface_binding_key(surface))
        if descriptor in descriptors:
            raise ValueError(f"member variants {descriptors[descriptor]} and {name} select identical "
                             "FP32 physics and surface controls; repeating them would fabricate ensemble size")
        descriptors[descriptor] = name
        result.append({"name": name, "physics": dict(physics), "surface": surface})
    return tuple(result)


def member_surface_options(request, member_id):
    """The selected member's fixed GPU descriptor, or the ordinary shared one."""
    variants = tuple(getattr(request, "member_variants", ()) or ())
    if not variants:
        return request.perturbation
    if not 0 <= member_id < len(variants):
        raise ValueError("member surface selection is outside the named roster")
    selected = dict(variants[member_id]["surface"])
    if selected == DEFAULTS:
        return None
    return {"kind": "surface-state", **selected}


def variant_configuration(raw, variant):
    """Override ordinary physics selectors through the existing config tables.

    The ordinary loader validates the complete resolved combination before
    preparation. No selector is translated into a separate scheme adapter.
    """
    changed = deepcopy(raw)
    physics = variant["physics"]
    if physics:
        changed.setdefault("shared", {}).update(physics)
        for domain in changed.get("domain", ()):
            # Explicit child values must not shadow a tree-wide member arm.
            for key, value in physics.items():
                if key in domain:
                    domain[key] = value
    return changed


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=lambda item: item.isoformat()).encode("utf-8")


# RunConfig geometry and clock fields stay in the source half. All other
# registered run fields belong to the member half. Unknown configuration
# fields stay source-bound, so adding a field cannot silently widen sharing.
SOURCE_RUN_FIELDS = frozenset({
    "nx", "ny", "nz", "dx", "dy", "ztop", "dt", "clock_dt", "run_seconds",
    "p_surf", "hybrid_opt", "etac", "specified", "periodic_x", "periodic_y",
})


@dataclass(frozen=True)
class PreparationBinding:
    source_key: str
    physics_tail: str

    @property
    def key(self):
        return hashlib.sha256(_canonical({"schema": "ensemble-preparation-v2",
            "source_key": self.source_key, "physics_tail": self.physics_tail})).hexdigest()


def preparation_binding_parts(trajectory_identity, raw, *, experiment=None):
    """Separate immutable source composition from member initialization.

    The pair still binds every scientific table. Only registered run
    selectors leave the source half; geometry, vertical coordinates, fetch
    inputs and the complete time window remain there. A land arm can reuse
    composed source words, but cannot borrow another arm's prepared arrays.
    """
    authority = deepcopy(raw)
    authority.pop("ensemble", None)
    if experiment is not None:
        from gpuwm.ingest.prepared_cache import prepared_domain_config_identity
        # Resolve inherited and explicit selectors through the same identity
        # reader the actual prepared-domain preflight binds.
        authority.pop("shared", None)
        authority.pop("domain", None)
        authority["resolved_domains"] = [prepared_domain_config_identity(domain)
                                           for domain in experiment.domains]
        authority["resolved_vertical"] = asdict(experiment.vertical)
    from gpuwm.config import RunConfig
    tail_fields = {item.name for item in fields(RunConfig)} - SOURCE_RUN_FIELDS
    tail = {}

    def split(table, label):
        selected = {key: table.pop(key) for key in tuple(table) if key in tail_fields}
        if selected:
            tail[label] = selected

    if experiment is None:
        split(authority.get("shared", {}), "shared")
        for number, domain in enumerate(authority.get("domain", ())):
            split(domain, f"domain-{number}")
    else:
        for number, domain in enumerate(authority["resolved_domains"]):
            split(domain.get("run", {}), f"domain-{number}")
            # These stamps are derived exclusively from member run selectors.
            for name in ("chem_table_identity", "orographic_sampling_contract"):
                if name in domain:
                    tail[f"domain-{number}.{name}"] = domain.pop(name)
    return PreparationBinding(
        hashlib.sha256(_canonical({"schema": "ensemble-source-v1",
            "trajectory": trajectory_identity, "configuration": authority})).hexdigest(),
        hashlib.sha256(_canonical(tail)).hexdigest())


def preparation_binding_key(trajectory_identity, raw, *, experiment=None):
    """Complete source plus physics/land binding for prepared-bank consumers."""
    return preparation_binding_parts(trajectory_identity, raw, experiment=experiment).key


def surface_binding_key(surface):
    """Metadata words actually consumed by fixed GPU surface controls."""
    selected = {**DEFAULTS, **surface}
    # Signed scalar zero is an inactive SST offset on both initialization
    # paths, so it cannot manufacture another identity control.
    return tuple((name, struct.pack("<f", 0.0 if selected[name] == 0.0 else selected[name]).hex())
                 for name in OPTIONS)


def require_distinct_variants(raw, variants, *, experiments):
    """Labels cannot turn identical ordinary initial states into extra members."""
    previous = {}
    if len(experiments) != len(variants):
        raise ValueError("member diversity must be checked against every loaded member experiment")
    for variant, experiment in zip(variants, experiments):
        key = (preparation_binding_key("shared-source", variant_configuration(raw, variant), experiment=experiment),
               surface_binding_key(variant["surface"]))
        if key in previous:
            raise ValueError(f"member variants {previous[key]} and {variant['name']} select identical "
                             "land and surface states; repeating them would fabricate ensemble size")
        previous[key] = variant["name"]


__all__ = ["normalize_member_variants", "member_surface_options", "variant_configuration",
           "PreparationBinding", "preparation_binding_parts", "preparation_binding_key",
           "surface_binding_key", "require_distinct_variants"]
