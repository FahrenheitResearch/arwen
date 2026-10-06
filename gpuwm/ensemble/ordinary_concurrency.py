"""Route known stream-unsafe ordinary bindings to sequential members.

This is execution routing only. Every member retains its selected physics,
clock and input authority, and runs even when same-card concurrency is not
available. A rule names the concrete shared owner that cannot be borrowed
by independent CUDA queues.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from types import SimpleNamespace

from gpuwm.config import radiation_scheme_ids
from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant


CONTRACT = "gpuwm-ensemble-ordinary-concurrency-v1"


@dataclass(frozen=True)
class OrdinaryBindingFallback:
    binding: str
    selectors: tuple[tuple[str, object], ...]
    reason: str


# The shortwave singleton's scratch is stream-owned, but CudaSW uploads its
# constant tables asynchronously and _cuda_sw publishes the engine without
# an upload-ready event. A second queue can borrow it before those uploads
# complete. Legacy longwave already orders its immutable cache on use.
ORDINARY_BINDING_FALLBACKS = (
    OrdinaryBindingFallback(
        "legacy_rrtmg_shortwave_table_upload",
        (("ra_sw_physics", 4), ("ra_rrtmg_variant", RRTMG_VARIANT_LEGACY)),
        "legacy RRTMG shortwave keeps sequential ordinary members because its "
        "shared CudaSW table cache has no upload-ready event for another member stream"),
)


def _value(settings, key, default):
    return settings.get(key, default) if isinstance(settings, Mapping) else getattr(settings, key, default)


def _selectors(cfg):
    lw, sw = radiation_scheme_ids(SimpleNamespace(
        ra_physics=_value(cfg, "ra_physics", 0),
        ra_lw_physics=_value(cfg, "ra_lw_physics", -1),
        ra_sw_physics=_value(cfg, "ra_sw_physics", -1)))
    return {"ra_lw_physics": lw, "ra_sw_physics": sw,
            "ra_rrtmg_variant": rrtmg_variant(cfg)}


@dataclass(frozen=True)
class OrdinaryConcurrencyPlan:
    eligible: bool
    reasons: tuple[str, ...]
    bindings: tuple[dict, ...]

    def receipt(self):
        return {"contract": CONTRACT, "concurrent_ordinary_admitted": self.eligible,
                "fallback": None if self.eligible else "ordinary_members_in_sequence_per_card",
                "fallback_reasons": list(self.reasons), "bindings": list(self.bindings),
                "physics_selection": "unchanged", "clock_selection": "independent_original_member_clocks",
                "cfl_owner_policy": "member_scope_required_in_every_ordinary_wave"}


def plan_ordinary_concurrency(experiment):
    """Check actual selected domain bindings before starting a CUDA wave.

    Absence of a fallback rule is not a new scientific qualification. The
    caller still prices each original model and owns its queues and outputs.
    Independently executing cards require private member CFL scopes even
    when this plan selects sequential members on each card.
    """
    reasons, bindings = [], []
    for domain in getattr(experiment, "domains", ()):
        selected = _selectors(domain.run)
        for rule in ORDINARY_BINDING_FALLBACKS:
            if all(selected[key] == value for key, value in rule.selectors):
                reason = f"domain {domain.grid_id}: {rule.reason}"
                reasons.append(reason)
                bindings.append({"grid_id": int(domain.grid_id), "binding": rule.binding,
                                 "selectors": dict(rule.selectors), "reason": reason})
    return OrdinaryConcurrencyPlan(not reasons, tuple(dict.fromkeys(reasons)), tuple(bindings))


__all__ = ["CONTRACT", "OrdinaryBindingFallback", "ORDINARY_BINDING_FALLBACKS",
           "OrdinaryConcurrencyPlan", "plan_ordinary_concurrency"]
