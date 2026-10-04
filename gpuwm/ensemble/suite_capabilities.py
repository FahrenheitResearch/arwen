"""Runtime-free ensemble physics routing without changing a selection.

The ordinary registry remains the authority for runnable options. An option
without a qualified member launch uses its original operation per member.
These rows describe execution coverage, rather than new physics admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping

from gpuwm.config import radiation_scheme_ids
from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant

PHYSICS_EXECUTION_CONTRACT = "gpuwm-ensemble-physics-execution-v1"
PHYSICS_COMPONENTS = (
    "microphysics", "radiation", "surface_layer", "land_surface", "pbl", "cumulus",
)

# These are existing, qualified leaves. A leaf row does not assert that its
# surrounding driver, option combinations or dynamically changing clock have
# a qualified packed binding. The caller must supply the admitted binding.
_NATIVE_LEAVES = {
    ("microphysics", "thompson-mp8"): "gpuwm.ensemble.batch_physics.prepare_thompson_column_batch",
    ("surface_layer", "revised-mm5"): "gpuwm.ensemble.batch_physics.prepare_sfclay_column_batch",
    ("surface_layer", "classic-mm5"): "gpuwm.ensemble.batch_physics.prepare_sfclay_column_batch",
    ("land_surface", "noah"): "gpuwm.ensemble.batch_physics.prepare_noah_column_batch",
    ("pbl", "ysu"): "gpuwm.ensemble.batch_physics.prepare_ysu_column_batch",
}


def _value(settings, key, default=None):
    return settings.get(key, default) if isinstance(settings, Mapping) else getattr(settings, key, default)


@dataclass(frozen=True)
class ComponentCapability:
    component: str
    option_id: str | None
    selectors: tuple[tuple[str, object], ...]
    implemented: bool
    native_leaf: str | None = None
    parameters: tuple[tuple[str, object], ...] = ()

    def receipt(self):
        result = asdict(self)
        result["selectors"] = dict(self.selectors)
        result["parameters"] = dict(self.parameters)
        result["fallback"] = "original_member_operation"
        return result


@dataclass(frozen=True)
class SuiteExecutionPlan:
    members: int
    mode: str
    components: tuple[ComponentCapability, ...]
    native_fallback_reasons: tuple[str, ...]

    @property
    def uses_native_batch(self):
        return self.mode == "native_batch"

    def receipt(self):
        return {
            "contract": PHYSICS_EXECUTION_CONTRACT,
            "members": self.members,
            "mode": self.mode,
            "components": [row.receipt() for row in self.components],
            "native_fallback_reasons": list(self.native_fallback_reasons),
            "selection_policy": "unchanged",
            "clock_policy": "independent_original_member_clocks",
        }


def capability_matrix(*, registry=None):
    """Every implemented registry physics option, with original fallback.

    Off rows and newly registered schemes are included automatically. The
    turbulence and urban rows are also included because they affect whether
    the surrounding driver can use the packed native binding.
    """
    if registry is None:
        from gpuwm.physics_registry import physics_registry
        registry = physics_registry()
    rows = []
    for component, declaration in sorted(registry["components"].items()):
        for option_id, option in sorted(declaration["options"].items()):
            if not option.get("implemented", False):
                continue
            rows.append(ComponentCapability(
                component, option_id, tuple(sorted((option.get("selectors") or {}).items())),
                True, _NATIVE_LEAVES.get((component, option_id))))
    return tuple(rows)


def selected_capabilities(cfg, *, registry=None):
    """Resolve metadata for the requested suite, without mutating it."""
    if registry is None:
        from gpuwm.physics_registry import physics_registry
        registry = physics_registry()
    from types import SimpleNamespace
    lw, sw = radiation_scheme_ids(SimpleNamespace(
        ra_physics=_value(cfg, "ra_physics", 0),
        ra_lw_physics=_value(cfg, "ra_lw_physics", -1),
        ra_sw_physics=_value(cfg, "ra_sw_physics", -1)))
    effective = {"ra_lw_physics": lw, "ra_sw_physics": sw}
    rows = capability_matrix(registry=registry)
    result = []
    for component, declaration in sorted(registry["components"].items()):
        values = tuple(sorted((key, effective.get(key, _value(cfg, key, 0)))
                              for key in declaration.get("selector_keys", ())))
        row = next((row for row in rows if row.component == component and row.selectors == values), None)
        # The ordinary config door diagnoses an unknown selector. Ensemble
        # routing must neither reinterpret it nor substitute a known scheme.
        row = row if row is not None else ComponentCapability(component, None, values, False)
        if component == "radiation" and (lw, sw) == (4, 4):
            variant = rrtmg_variant(cfg)
            row = replace(row, parameters=(("ra_rrtmg_variant", variant),),
                          native_leaf="gpuwm.ensemble.batch_physics_init.prepare_legacy_member_radiation"
                          if variant == RRTMG_VARIANT_LEGACY else None)
        result.append(row)
    return tuple(result)


def plan_suite(cfg, *, members=1, registry=None):
    """Pick the qualified packed driver or the unchanged original driver.

    This is a conservative binding decision. Falling back does not refuse an
    ensemble, disable a scheme, pin an adaptive timestep or remove a feature.
    """
    if isinstance(members, bool) or not isinstance(members, int) or members < 1:
        raise ValueError("members must be a positive integer")
    components = selected_capabilities(cfg, registry=registry)
    reasons = []
    for row in components:
        if not row.implemented:
            reasons.append(f"{row.component} selection has no qualified packed registry row")
    selections = {
        "mp_physics": 8, "sf_surface_physics": 2, "bl_pbl_physics": 1,
        "cu_physics": 0, "sf_urban_physics": 0, "sf_surface_mosaic": 0,
        "topo_wind": 0, "gwd_opt": 0, "slope_rad": 0, "topo_shading": 0,
    }
    for key, required in selections.items():
        value = _value(cfg, key, 0)
        if value != required:
            reasons.append(f"{key}={value!r} uses its original member driver")
    if _value(cfg, "sf_sfclay_physics", 0) not in (1, 91):
        reasons.append("the selected surface layer has no qualified packed driver binding")
    lw, sw = (dict(next(row for row in components if row.component == "radiation").selectors)[key]
              for key in ("ra_lw_physics", "ra_sw_physics"))
    if (lw, sw) != (4, 4) or rrtmg_variant(cfg) != RRTMG_VARIANT_LEGACY:
        reasons.append("the selected radiation has no qualified packed initializer binding")
    if not _value(cfg, "moist", False):
        reasons.append("the packed column initializer requires moist state")
    if _value(cfg, "use_adaptive_time_step", False):
        reasons.append("adaptive members retain their independent original clocks")
    if _value(cfg, "bldt", 0.0) != 0.0:
        reasons.append("positive PBL cadence retains its original held composition target")
    mode = "ordinary_single" if members == 1 else "member_local" if reasons else "native_batch"
    return SuiteExecutionPlan(members, mode, components, tuple(reasons))


def preset_capabilities(*, members=2):
    """Route all front-door fixed-template presets from their authority."""
    from gpuwm.physics_compat import SINGLE_DOMAIN_PHYSICS_PROFILES, single_domain_runtime_switches
    return {name: plan_suite(single_domain_runtime_switches(name), members=members)
            for name in SINGLE_DOMAIN_PHYSICS_PROFILES}


__all__ = ["PHYSICS_EXECUTION_CONTRACT", "PHYSICS_COMPONENTS", "ComponentCapability",
           "SuiteExecutionPlan", "capability_matrix", "selected_capabilities", "plan_suite", "preset_capabilities"]
