"""Conservation-closure receipt: schema, three-term budget, atomic writer.

The receipt is the durable form of the mapped column-mass budget the
dycore observers measure.  Three things it deliberately does NOT do:

1. It does not report a single "closure" number.  A forced domain gains
   mass through the lateral-boundary forcing (``rmu_t``) and through the
   specified-zone overwrite as well as through the telescoped lateral
   flux, so the three terms are SEPARATE required keys and the residual
   is what is left after all three.  A flux-only closure on a forced
   domain would be a false receipt, which is why
   :func:`mass_budget_residual` refuses a budget whose forcing terms were
   never supplied rather than defaulting them to zero.
2. It does not decide which residuals are gates.  Every residual is
   carried as a measured value with its own ``tier``; promotion of the
   water and moist-static-energy residuals from observability to gate is
   an owner decision (D-15) and no bound is written here.
3. It does not fabricate the vs-WRF tier.  The nest half of that tier is
   scored from a stock-WRF feedback A/B pair by
   :mod:`gpuwm.verify.feedback_reference`; a receipt built without one
   keeps the schema-reserved shape with null values and a reason rather
   than omitting the tier.

Two absences are recorded as first-class receipt content rather than as
missing keys.  The acoustic and gravity-wave receipts are deferred for
this release and appear as an explicit deferred tier, so a reader sees
the deferral instead of inferring a silent pass.  The guard inventory
enumerates every clamping site the model has, including the ones nothing
counts, each carrying the reason it is uncounted.

Every value in a receipt is measured by the caller.  This module supplies
the shape, the completeness checks, and an atomic ``allow_nan=False``
writer, so a NaN or a missing term fails at write time instead of being
published.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_ID = "gpuwm.conservation-closure/v1"

#: Keys every per-domain entry must carry.  ``boundary_distance_bands``
#: and ``guard_inventory`` are containers; the rest are measured scalars.
REQUIRED_DOMAIN_KEYS: frozenset[str] = frozenset({
    "lateral_flux_integral",
    "lbc_mass_forcing_integral",
    "specified_zone_mass_reset",
    "mass_closure_residual_max",
    "water_closure_residual_max",
    "moist_static_energy_residual",
    "boundary_distance_bands",
    "guard_inventory",
    "observer_overhead",
})

#: The three separately-reported mass sources.  Their names are also the
#: receipt keys, so a budget cannot be reported under a different split
#: than the one the residual was computed from.
MASS_BUDGET_TERMS: tuple[str, ...] = (
    "lateral_flux_integral",
    "lbc_mass_forcing_integral",
    "specified_zone_mass_reset",
)

#: Measured overhead fields; sized before the long run, never bounded here.
REQUIRED_OVERHEAD_KEYS: frozenset[str] = frozenset({
    "wall_seconds_plain",
    "wall_seconds_receipt",
    "wall_seconds_per_simulated_minute_delta",
    "device_bytes_plain",
    "device_bytes_receipt",
})

#: Residual tiers.  ``observability`` records a measurement that no gate
#: reads; ``gate`` records one a gate does.  Which residual sits in which
#: tier is an owner decision, so the tier travels with the value.
RESIDUAL_TIERS: frozenset[str] = frozenset({"observability", "gate"})


class ReceiptError(ValueError):
    """A receipt that would be false, incomplete, or unwritable."""


def mass_budget_residual(measure_start: float, measure_end: float,
                         terms: Mapping[str, float]) -> float:
    """Signed residual of the three-term column-mass budget.

    ``measure_start``/``measure_end`` are the area-weighted domain mass
    measure (``gpuwm.core.dycore.domain_mass_measure``) before and after
    the scored window; ``terms`` must carry every name in
    :data:`MASS_BUDGET_TERMS`.  A missing term raises rather than being
    treated as zero: on an open-boundary domain the forcing terms really
    are zero, but that is a measurement the caller makes, not a default
    this function grants.
    """
    missing = [name for name in MASS_BUDGET_TERMS if name not in terms]
    if missing:
        raise ReceiptError(
            "mass budget is missing separately-measured term(s) "
            f"{missing}: a flux-only closure on a forced domain is a "
            "false receipt, so every term is supplied explicitly")
    extra = sorted(set(terms) - set(MASS_BUDGET_TERMS))
    if extra:
        raise ReceiptError(f"unknown mass budget term(s) {extra}")
    accounted = 0.0
    for name in MASS_BUDGET_TERMS:
        accounted += float(terms[name])
    return (measure_end - measure_start) - accounted


def relative_residual(residual: float, measure_start: float) -> float:
    """``|residual| / measure_start`` with a zero-measure refusal."""
    if not measure_start > 0.0:
        raise ReceiptError(
            f"domain mass measure must be positive, got {measure_start!r}")
    return abs(residual) / measure_start


def guard_entry(site: str, mechanism: str, *, counted: bool,
                count: int | None = None,
                why_not_counted: str | None = None) -> dict[str, Any]:
    """One honest row of the guard inventory.

    An uncounted guard is recorded WITH its reason rather than omitted,
    so the inventory enumerates the model's clamping surface instead of
    only the instrumented part of it.
    """
    if counted:
        if count is None:
            raise ReceiptError(
                f"guard {site!r} is marked counted but carries no count")
        if why_not_counted is not None:
            raise ReceiptError(
                f"guard {site!r} is counted and cannot carry a "
                "why_not_counted reason")
        count_or_null: int | None = int(count)
    else:
        if count is not None:
            raise ReceiptError(
                f"guard {site!r} is not counted but carries a count")
        if not why_not_counted:
            raise ReceiptError(
                f"guard {site!r} is not counted and must say why")
        count_or_null = None
    return {
        "site": site,
        "mechanism": mechanism,
        "counted": bool(counted),
        "count_or_null": count_or_null,
        "why_not_counted": why_not_counted,
    }


#: Every clamping/repair site the receipt enumerates, with the counting
#: posture each one ships with.  ``host_mirror`` names the sanctioned
#: float64 NumPy mirror a site is counted through when the device kernel
#: itself carries no counter; adding a counter to a certified ``.cu``
#: would force a byte-identity re-proof of that kernel, which is why the
#: sites without a mirror are recorded uncounted instead of instrumented.
GUARD_SITES: tuple[dict[str, str | None], ...] = (
    {
        "site": "gpuwm/core/physics.py YSU non-finite guard",
        "mechanism": "host-side counter, persisted across restart",
        "host_mirror": None,
        "why_not_counted": None,
    },
    {
        "site": "gpuwm/core/kernels/openbc.cu w_damp",
        "mechanism": "vertical-velocity limiter, counted through its "
                     "float64 NumPy mirror",
        "host_mirror": "gpuwm.verify.npref.np_w_damp",
        "why_not_counted": None,
    },
    {
        "site": "gpuwm/core/kernels/pd_advection.cu pd_renorm_apply",
        "mechanism": "positive-definite flux renormalization, counted "
                     "through its float64 NumPy mirror",
        "host_mirror": "gpuwm.verify.npref.np_pd_renorm_apply",
        "why_not_counted": None,
    },
    {
        "site": "gpuwm/core/kernels/mynn_pbl.cu moisture-check borrow",
        "mechanism": "borrow-from-below moisture repair, device-side only",
        "host_mirror": None,
        "why_not_counted": (
            "counting it needs a counter inside a certified kernel, which "
            "forces a byte-identity re-proof of that kernel; no NumPy "
            "mirror of the borrow chain exists to count it host-side"),
    },
)


def default_guard_inventory(counts: Mapping[str, int]) -> list[dict[str, Any]]:
    """The enumerated guard inventory, with measured counts supplied.

    ``counts`` carries one entry per countable site.  A countable site
    with no supplied count RAISES rather than being written as zero: an
    uncounted guard reported as ``0`` claims the guard never fired, which
    is a stronger statement than "nobody looked" and the one thing this
    inventory exists to stop.  Sites that carry a ``why_not_counted``
    reason are written uncounted and must NOT appear in ``counts``.
    """
    countable = {entry["site"] for entry in GUARD_SITES
                 if entry["why_not_counted"] is None}
    unknown = sorted(set(counts) - countable)
    if unknown:
        raise ReceiptError(
            f"counts supplied for site(s) {unknown} that the inventory "
            "does not mark countable")
    inventory: list[dict[str, Any]] = []
    for entry in GUARD_SITES:
        site = str(entry["site"])
        mechanism = str(entry["mechanism"])
        reason = entry["why_not_counted"]
        if reason is None:
            if site not in counts:
                raise ReceiptError(
                    f"guard {site!r} is countable and its count was not "
                    "supplied: writing zero would claim it never fired")
            inventory.append(guard_entry(site, mechanism, counted=True,
                                         count=int(counts[site])))
        else:
            inventory.append(guard_entry(site, mechanism, counted=False,
                                         why_not_counted=reason))
    return inventory


def residual_entry(value: float, *, tier: str, definition: str,
                   bound: float | None = None,
                   bound_decision: str | None = None) -> dict[str, Any]:
    """A measured residual with its tier and the bound's provenance.

    ``bound`` stays ``None`` until an owner pins it from a committed
    measurement; a receipt therefore records that a residual is unpinned
    rather than implying a bound nobody set.
    """
    if tier not in RESIDUAL_TIERS:
        raise ReceiptError(f"unknown residual tier {tier!r}")
    if bound is None and tier == "gate":
        raise ReceiptError(
            "a gate-tier residual carries the bound it is gated on")
    if bound is not None and not bound_decision:
        raise ReceiptError(
            "a pinned bound records the decision that pinned it")
    return {
        "value": float(value),
        "tier": tier,
        "definition": definition,
        "bound": None if bound is None else float(bound),
        "bound_decision": bound_decision,
    }


def unavailable_tier(reason: str, decision: str) -> dict[str, Any]:
    """A schema-reserved comparison tier that is not implemented.

    Reserving the shape with an explicit reason keeps the absence
    visible in the receipt; silently omitting the section would let a
    reader infer the comparison was made and passed.
    """
    if not reason or not decision:
        raise ReceiptError("a reserved tier names its reason and decision")
    return {"status": "unavailable", "entries": None,
            "reason": reason, "decision": decision}


def deferred_tier(reason: str, decision: str) -> dict[str, Any]:
    """A receipt tier deliberately not produced this release.

    ``unavailable`` says the evidence could not be obtained; ``deferred``
    says it was not attempted, on purpose, under a decision.  The two are
    different claims and a reader is entitled to know which one applies,
    so they are different statuses rather than one shared "missing".
    """
    if not reason or not decision:
        raise ReceiptError("a deferred tier names its reason and decision")
    return {"status": "deferred", "entries": None,
            "reason": reason, "decision": decision}


#: The acoustic and gravity-wave conservation receipts are deferred for
#: this release; the receipt carries the deferral so their absence is
#: visible in the artifact and not only in the plan.
ACOUSTIC_GRAVITY_WAVE_DEFERRAL_REASON = (
    "acoustic and gravity-wave conservation receipts are deferred for this "
    "release: no receipt is emitted for either, and no statement anywhere "
    "in this release rests on one")


def build_domain_entry(grid_id: int, *, terms: Mapping[str, float],
                       measure_start: float, measure_end: float,
                       mass_residual: Mapping[str, Any],
                       water_residual: Mapping[str, Any],
                       energy_residual: Mapping[str, Any],
                       boundary_distance_bands: Sequence[Mapping[str, Any]],
                       guard_inventory: Iterable[Mapping[str, Any]],
                       observer_overhead: Mapping[str, Any],
                       ) -> dict[str, Any]:
    """One per-domain entry with every required key present."""
    overhead_missing = REQUIRED_OVERHEAD_KEYS - set(observer_overhead)
    if overhead_missing:
        raise ReceiptError(
            f"observer overhead is missing {sorted(overhead_missing)}: the "
            "measurement is taken before the long run is scheduled")
    entry: dict[str, Any] = {
        "grid_id": int(grid_id),
        "domain_mass_measure_start": float(measure_start),
        "domain_mass_measure_end": float(measure_end),
        "mass_closure_residual_max": dict(mass_residual),
        "water_closure_residual_max": dict(water_residual),
        "moist_static_energy_residual": dict(energy_residual),
        "boundary_distance_bands": [dict(b) for b in boundary_distance_bands],
        "guard_inventory": [dict(g) for g in guard_inventory],
        "observer_overhead": dict(observer_overhead),
    }
    for name in MASS_BUDGET_TERMS:
        if name not in terms:
            raise ReceiptError(
                f"domain {grid_id} is missing budget term {name!r}")
        entry[name] = float(terms[name])
    missing = REQUIRED_DOMAIN_KEYS - set(entry)
    if missing:
        raise ReceiptError(
            f"domain {grid_id} entry is missing {sorted(missing)}")
    return entry


def build_conservation_receipt(domains: Sequence[Mapping[str, Any]], *,
                               experiment: str,
                               run_id: str | None = None,
                               config_digest: str | None = None,
                               vs_wrf_tier: Mapping[str, Any] | None = None,
                               self_consistency: Mapping[str, Any] | None = None,
                               acoustic_gravity_wave_tier: Mapping[str, Any] | None = None,
                               ) -> dict[str, Any]:
    """Assemble the receipt payload from measured per-domain entries.

    ``domains`` carries ONE entry per domain present in the run: the
    receipt's domain count is the run's, never a configured expectation.
    """
    if not domains:
        raise ReceiptError(
            "a conservation receipt covers at least one domain present "
            "in the run")
    grid_ids = [int(d["grid_id"]) for d in domains]
    if len(set(grid_ids)) != len(grid_ids):
        raise ReceiptError(f"duplicate grid ids in receipt: {grid_ids}")
    for entry in domains:
        missing = REQUIRED_DOMAIN_KEYS - set(entry)
        if missing:
            raise ReceiptError(
                f"domain {entry.get('grid_id')} entry is missing "
                f"{sorted(missing)}")
    return {
        "schema": SCHEMA_ID,
        "experiment": experiment,
        "run_id": run_id,
        "config_digest": config_digest,
        "domain_count": len(domains),
        "domains": [dict(d) for d in domains],
        "self_consistency": (None if self_consistency is None
                             else dict(self_consistency)),
        "vs_wrf_tier": (dict(vs_wrf_tier) if vs_wrf_tier is not None
                        else unavailable_tier(
                            "no stock-WRF feedback A/B pair was supplied to "
                            "score the nest tier against", "D-8")),
        "acoustic_gravity_wave_tier": (
            dict(acoustic_gravity_wave_tier)
            if acoustic_gravity_wave_tier is not None
            else deferred_tier(ACOUSTIC_GRAVITY_WAVE_DEFERRAL_REASON, "D-31")),
    }


def relaxation_row_entry(row: int, *, cells: int,
                         mean_abs_increment: float,
                         max_abs_increment: float) -> dict[str, Any]:
    """One relaxation-row diagnostic of the LBC self-consistency tier.

    The component tier already proves the relaxation coefficients match
    WRF row by row; what a run-level receipt adds is the increment the
    rows actually applied, which is the quantity a boundary-stratified
    comparison is read against.  Row 0 is the specified row, so its
    increment is a reset rather than a relaxation and is reported under
    its own row index rather than folded into the mean.
    """
    if row < 0:
        raise ReceiptError(f"relaxation row index must be >= 0, got {row}")
    if cells <= 0:
        raise ReceiptError(f"relaxation row {row} covers no cells")
    return {
        "row": int(row),
        "cells": int(cells),
        "mean_abs_increment": float(mean_abs_increment),
        "max_abs_increment": float(max_abs_increment),
    }


def build_self_consistency_section(
        *, lateral_boundary_rows: Sequence[Mapping[str, Any]],
        nest_edges: Sequence[Mapping[str, Any]],
        component_tier_reference: str) -> dict[str, Any]:
    """Run-level nest/LBC self-consistency section.

    ``component_tier_reference`` names the already-existing component-tier
    evidence this section sits on top of, so the receipt states what is
    new (run-level emission and the per-row diagnostic) rather than
    implying the whole tier was built here.
    """
    if not component_tier_reference:
        raise ReceiptError(
            "the self-consistency section names the component-tier "
            "evidence it extends")
    rows = [dict(r) for r in lateral_boundary_rows]
    indices = [int(r["row"]) for r in rows]
    if sorted(indices) != list(range(len(indices))):
        raise ReceiptError(
            "lateral boundary rows must be a contiguous 0..n-1 run, got "
            f"{indices}")
    return {
        "component_tier_reference": component_tier_reference,
        "lateral_boundary_rows": rows,
        "nest_edges": [dict(e) for e in nest_edges],
    }


def encode_receipt(payload: Mapping[str, Any]) -> bytes:
    """Canonical receipt bytes; a NaN or infinity refuses to encode."""
    try:
        text = json.dumps(payload, indent=2, sort_keys=True,
                          allow_nan=False) + "\n"
    except ValueError as exc:
        raise ReceiptError(
            f"conservation receipt carries a non-finite value: {exc}"
        ) from exc
    return text.encode("utf-8")


def write_conservation_receipt(path: Path,
                               payload: Mapping[str, Any]) -> tuple[Path, str]:
    """Atomically publish the receipt; returns ``(path, sha256)``.

    The digest is over the exact bytes on disk, so a later reader can
    bind the receipt without re-serializing it.
    """
    encoded = encode_receipt(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path, hashlib.sha256(encoded).hexdigest()
