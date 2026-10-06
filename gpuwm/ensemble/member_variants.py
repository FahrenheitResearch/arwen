"""Named member land-scheme and fixed surface-state controls.

This roster changes existing land selectors and initial surface state.
Preparation remains the ordinary source chain. A prepared authority is
shared only when its trajectory and complete preparation configuration
match; different soil layouts never borrow each other's prepared cache.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from numbers import Real
import re
import struct

from gpuwm.ensemble.surface_controls import DEFAULTS, OPTIONS, validate_surface_recipe

LAND_KEYS = frozenset({"sf_surface_physics", "num_soil_layers"})
LAND_LAYOUTS = frozenset({(3, 6), (3, 9), (2, 4)})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


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
        if not isinstance(physics, dict) or set(physics) - LAND_KEYS:
            raise ValueError(f"member variant {name} physics takes existing land selectors only: "
                             "sf_surface_physics and num_soil_layers")
        if physics:
            if set(physics) != LAND_KEYS or any(type(item) is not int for item in physics.values()):
                raise ValueError(f"member variant {name} must name both integer land selectors")
            layout = (physics["sf_surface_physics"], physics["num_soil_layers"])
            if layout not in LAND_LAYOUTS:
                raise ValueError(f"member variant {name} needs an existing RUC six/nine-layer or Noah four-layer "
                                 "soil layout; another pairing would initialize the wrong land column")
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
                             "FP32 land and surface controls; repeating them would fabricate ensemble size")
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
    """Override existing land selectors on every domain through config tables."""
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


def preparation_binding_key(trajectory_identity, raw, *, experiment=None):
    """A complete ordinary preparation authority, independent of surface arms.

    Prepared domain identities bind land selectors and soil count. Include
    all scientific config tables so a different four-layer/six-layer state,
    grid, clock, static choice or source window cannot be aliased. Fetch
    details are already bound by the source trajectory and common recipe
    window; the ensemble table belongs to execution, not preparation.
    """
    authority = deepcopy(raw)
    authority.pop("ensemble", None)
    authority.pop("fetch", None)
    if experiment is not None:
        from gpuwm.ingest.prepared_cache import prepared_domain_config_identity
        # Resolve inherited and explicit selectors through the same identity
        # reader the actual prepared-domain preflight binds.
        authority.pop("shared", None)
        authority.pop("domain", None)
        authority["resolved_domains"] = [prepared_domain_config_identity(domain)
                                           for domain in experiment.domains]
        authority["resolved_vertical"] = asdict(experiment.vertical)
    return hashlib.sha256(_canonical({"trajectory": trajectory_identity,
                                     "configuration": authority})).hexdigest()


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
           "preparation_binding_key", "surface_binding_key", "require_distinct_variants"]
