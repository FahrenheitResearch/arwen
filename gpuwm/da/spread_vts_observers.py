"""Actual retained-slot observers for valid-time-shift covariance.

VTSM uses the original native states and their original diagnosed H(x).
The radar-grid TargetGrid remains the observation placement and localization
authority. Retained pressure and geopotential belong to each real slot;
this adapter does not claim a new physical vertical reinterpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np

from gpuwm.da.spread_repair import (TimeShiftBuffer,
    central_time_posterior, time_shifted_covariance)

SCHEMA = "gpuwm.da.spread-vts-observers/v1"
OFFSETS = (-900, 0, 900)


class _FieldStack:
    """A real host-array roster, materialized only at a numeric consumer.

    NumPy concatenate dispatch preserves references to the actual samples.
    This avoids constructing another complete 3K ensemble merely to check
    the time-shift covariance contract. No numeric values are synthesized.
    """
    def __init__(self, arrays):
        self.arrays = tuple(arrays)
        first = self.arrays[0]
        self.shape = (len(self.arrays),) + tuple(first.shape)
        self.dtype = np.dtype(first.dtype)
        self.ndim = len(self.shape)

    def copy(self):
        # The native producer supplies readonly sealed RAM views. Mutable
        # inputs are copied once at the pool boundary below. Returning this
        # immutable roster preserves the buffer's ownership contract.
        if any(array.flags.writeable for array in self.arrays):
            raise ValueError("time-shift buffer cannot borrow mutable sample arrays")
        return self

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("a retained member stack needs a copy to become a dense array")
        result = np.stack(self.arrays)
        return result.astype(dtype, copy=False) if dtype is not None else result

    def __array_function__(self, func, types, args, kwargs):
        if func is not np.concatenate:
            return NotImplemented
        values = args[0]
        axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
        if axis != 0 or kwargs.get("out") is not None or any(
                not isinstance(value, _FieldStack) for value in values):
            return NotImplemented
        return _FieldStack(array for value in values for array in value.arrays)


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an exact integer")
    return int(value)


def _seconds(value, label):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite seconds")
    return Fraction(str(value))


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(
            letter not in "0123456789abcdef" for letter in value):
        raise ValueError(f"{label} must be a lower-case SHA256 identity")
    return value


def _owned_immutable(value):
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fi":
        raise ValueError("VTS retained fields must be real numeric native host arrays")
    if value.flags.writeable:
        value = value.copy(order="K")
        value.flags.writeable = False
    return value


@dataclass(frozen=True)
class VtsObserverRoster:
    """Offset-major 3K samples with K original central trajectories."""
    snapshots: Mapping[int, Mapping[str, object]]
    dbz: Mapping[int, object]
    surface: Mapping[str, np.ndarray]
    member_count: int
    pooled_fields: Mapping[str, _FieldStack]
    central_states: tuple[Mapping[str, object], ...]
    receipt: dict

    def reflectivity_provider(self, index, state):
        index = _integer(index, "covariance sample index")
        if index not in self.snapshots or state is not self.snapshots[index]:
            raise ValueError("retained reflectivity belongs to its original captured state")
        return self.dbz[index]

    def reduce_to_central(self, increments):
        """Return K increments after full-posterior mean recentering.

        The driver owns its configured native moment, positivity and
        saturation policy after this reduction. Only analysed fields are
        materialized, one at a time. A broadcastable partial roster is an
        error rather than a substitute for 3K analysed samples.
        """
        if not isinstance(increments, Mapping) or not increments:
            raise ValueError("VTS reduction needs a nonempty analysed field mapping")
        unknown = set(increments) - set(self.pooled_fields)
        if unknown:
            raise ValueError(f"VTS increments contain uncaptured fields: {sorted(unknown)}")
        for field, increment in increments.items():
            if tuple(np.shape(increment)) != self.pooled_fields[field].shape:
                raise ValueError(f"VTS increment {field} must have the exact 3K native field shape")
            if not np.all(np.isfinite(increment)):
                raise ValueError(f"VTS increment {field} contains non-finite analysed values")
        output = {}
        for field, increment in increments.items():
            prior = np.asarray(self.pooled_fields[field], dtype=np.float64)
            posterior = central_time_posterior(
                {field: prior}, {field: np.asarray(increment)}, self.receipt)[field]
            original = np.stack([state[field] for state in self.central_states])
            output[field] = posterior - original
        return output

    def reduce_member_increments(self, increments):
        """Consume assimilate_radar_grid's actual per-sample return contract."""
        expected = set(range(3 * self.member_count))
        if set(increments) != expected or any(
                isinstance(index, bool) or not isinstance(index, Integral)
                for index in increments):
            raise ValueError("VTS member increments must preserve the exact ordered 3K sample roster")
        fields = set(increments[0])
        if not fields or any(set(increments[index]) != fields for index in expected):
            raise ValueError("VTS member increment fields must match every analysed sample")
        output = {member: {} for member in range(self.member_count)}
        for field in sorted(fields):
            stacked = np.stack([increments[index][field] for index in range(3 * self.member_count)])
            reduced = self.reduce_to_central({field: stacked})[field]
            for member in output:
                output[member][field] = reduced[member]
        return output


def pool_retained_slots(slots, *, members, analysis_seconds,
                        grid_identity_sha256, mode="vtsm"):
    """Validate and pool real -900/0/+900 slots without copying their states.

    The producer owns the retained arrays until this roster is consumed.
    Native exact clocks, roster, state geometry, shared immutable setup,
    forecast origin and observation cutoff are checked before H(x) enters
    any enabled observation adapter.
    """
    if mode != "vtsm":
        raise ValueError("VTSP requires a native nonlinear surface rediagnosis observer; only VTSM is supported")
    members = _integer(members, "trajectory member count")
    if members < 2:
        raise ValueError("VTS covariance needs at least two trajectories")
    analysis = _seconds(analysis_seconds, "analysis time")
    if analysis.denominator != 1:
        raise ValueError("the retained time-shift buffer requires an exact whole-second analysis clock")
    grid_id = _digest(grid_identity_sha256, "grid identity")
    slots = tuple(slots)
    if len(slots) != 3 * members:
        raise ValueError("VTS needs every member at all three retained offsets")
    by_key = {}
    for slot in slots:
        member = _integer(slot.member, "retained member")
        offset = _integer(slot.offset_seconds, "retained offset")
        key = (offset, member)
        if offset not in OFFSETS or not 0 <= member < members or key in by_key:
            raise ValueError("VTS retained offsets and member roster must be unique and complete")
        if _seconds(slot.analysis_seconds, "slot analysis time") != analysis:
            raise ValueError("VTS slots belong to a different analysis time")
        ticks = _integer(slot.valid_ticks, "native valid ticks")
        denominator = _integer(slot.tick_den, "native tick denominator")
        if denominator <= 0 or Fraction(ticks, denominator) != analysis + offset:
            raise ValueError("VTS native clock does not equal the retained slot's valid time")
        if _digest(slot.grid_identity_sha256, "slot grid identity") != grid_id:
            raise ValueError("VTS slot grid identity differs from the analysis grid")
        _digest(slot.setup_identity_sha256, "slot immutable observer setup identity")
        if not isinstance(slot.states, Mapping) or not slot.states:
            raise ValueError("VTS retained slot has no captured state fields")
        origin = _seconds(slot.forecast_origin_seconds, "forecast origin")
        cutoff = _seconds(slot.observation_cutoff_seconds, "observation cutoff")
        if origin.denominator != 1 or cutoff.denominator != 1:
            raise ValueError("the retained time-shift buffer requires whole-second origin and cutoff clocks")
        by_key[key] = SimpleNamespace(member=member, offset_seconds=offset,
            valid_ticks=ticks, tick_den=denominator,
            forecast_origin_seconds=int(origin), observation_cutoff_seconds=int(cutoff),
            setup_identity_sha256=slot.setup_identity_sha256,
            states=MappingProxyType({field: _owned_immutable(value)
                for field, value in slot.states.items()}),
            reflectivity=_owned_immutable(slot.reflectivity),
            surface=MappingProxyType({field: _owned_immutable(value)
                for field, value in slot.surface.items()}))
    ordered = [by_key[(offset, member)] for offset in OFFSETS for member in range(members)]
    central = ordered[members:2 * members]
    fields = set(central[0].states)
    surfaces = set(central[0].surface)
    setup_id = central[0].setup_identity_sha256
    mass_shape = tuple(central[0].reflectivity.shape)
    if len(mass_shape) != 3:
        raise ValueError("VTS reflectivity must be the native three-dimensional mass grid")
    reference_shapes = {field: tuple(central[0].states[field].shape) for field in fields}
    reference_dtypes = {field: np.dtype(central[0].states[field].dtype) for field in fields}
    for slot in ordered:
        if set(slot.states) != fields or set(slot.surface) != surfaces:
            raise ValueError("VTS state and surface field rosters must match every retained sample")
        if slot.setup_identity_sha256 != setup_id:
            raise ValueError("VTS samples need the same immutable observer setup")
        if tuple(slot.reflectivity.shape) != mass_shape:
            raise ValueError("VTS reflectivity geometry differs between samples")
        for field, value in slot.states.items():
            if tuple(value.shape) != reference_shapes[field] or np.dtype(value.dtype) != reference_dtypes[field]:
                raise ValueError(f"VTS native state field {field} changed geometry or dtype")
        for field, value in slot.surface.items():
            if tuple(value.shape) != mass_shape[1:]:
                raise ValueError(f"VTS native surface field {field} is not on the retained mass plane")
        for field in ("p", "alt", "thp", "qv"):
            if field not in slot.states or tuple(slot.states[field].shape) != mass_shape:
                raise ValueError(f"VTS requires actual slot {field} on its native mass grid")
        if "php" not in slot.states or tuple(slot.states["php"].shape) != (mass_shape[0] + 1,) + mass_shape[1:]:
            raise ValueError("VTS requires actual slot php on its native vertical faces")
    buffer = TimeShiftBuffer(retention_seconds=1800)
    for offset in OFFSETS:
        slab = [by_key[(offset, member)] for member in range(members)]
        # The real covariance API enforces common origin/cutoff across slabs.
        # Per-member checks prevent one member hiding a mixed-origin sample.
        if len({float(slot.forecast_origin_seconds) for slot in slab}) != 1 or len(
                {float(slot.observation_cutoff_seconds) for slot in slab}) != 1:
            raise ValueError("VTS member samples need a common forecast origin and observation cutoff")
        buffer.retain(int(analysis) + offset,
            {field: _FieldStack(slot.states[field] for slot in slab) for field in fields},
            forecast_origin_seconds=slab[0].forecast_origin_seconds,
            observation_cutoff_seconds=slab[0].observation_cutoff_seconds)
    shifted = buffer.snapshots(int(analysis))
    pooled, receipt = time_shifted_covariance(shifted, analysis_seconds=int(analysis), mode=mode)
    receipt.update({"observer_schema": SCHEMA, "grid_identity_sha256": grid_id,
        "setup_identity_sha256": setup_id,
        "sample_order": "offset-major/member-major",
        "observation_geometry": "fixed radar-grid TargetGrid placement and localization",
        "source_geometry": "each retained native state supplies its own p/alt/php; immutable phb belongs to validated setup",
        "observer_semantics": "original native VTSM H(x); prior inflation uses the existing named linearized observation-anomaly formulation",
        "native_slots": [{"sample": index, "member": slot.member,
            "offset_seconds": slot.offset_seconds, "valid_ticks": slot.valid_ticks,
            "tick_den": slot.tick_den} for index, slot in enumerate(ordered)],
        "state_fields": sorted(fields), "surface_fields": sorted(surfaces),
        "retained_payload_bytes": sum(int(value.nbytes) for slot in ordered
            for value in list(slot.states.values()) + [slot.reflectivity] + list(slot.surface.values()))})
    return VtsObserverRoster(
        snapshots={index: slot.states for index, slot in enumerate(ordered)},
        dbz={index: slot.reflectivity for index, slot in enumerate(ordered)},
        surface={field: np.stack([slot.surface[field] for slot in ordered]) for field in sorted(surfaces)},
        member_count=members, pooled_fields=pooled,
        central_states=tuple(slot.states for slot in central), receipt=receipt)


def build_extra_observations(roster, *, target_grid, analysis_time,
        surface_source=None, surface_config=None, analysis_times=None,
        conventional_rows=None, conventional_table=None, thb=None, rotation=None):
    """Run enabled real surface/conventional adapters on the actual 3K roster."""
    if target_grid.identity_sha256() != roster.receipt["grid_identity_sha256"]:
        raise ValueError("VTS extra observations must use the validated analysis grid")
    batches, provenance = [], {}
    if surface_config is not None:
        from gpuwm.da.obs_surface import surface_to_gridded_obs
        required = set()
        if surface_config.temperature:
            required.add("t2")
        if surface_config.wind_speed:
            required.update(("u10", "v10"))
        if surface_config.dewpoint:
            required.update(("q2", "psfc"))
        if not required <= set(roster.surface):
            raise ValueError("VTS surface observations need actual diagnostics at every retained slot")
        built, provenance["surface"] = surface_to_gridded_obs(
            surface_source, target_grid=target_grid, analysis_time=analysis_time,
            analysis_times=analysis_times, config=surface_config,
            simulated_t2=roster.surface.get("t2"),
            simulated_u10=roster.surface.get("u10"),
            simulated_v10=roster.surface.get("v10"),
            simulated_q2=roster.surface.get("q2"),
            simulated_psfc=roster.surface.get("psfc"))
        batches.extend(built)
    if conventional_table is not None:
        from gpuwm.da.obs_conventional import conventional_batches
        built, provenance["conventional"] = conventional_batches(
            conventional_rows, conventional_table,
            states=[roster.snapshots[index] for index in range(3 * roster.member_count)],
            thb=thb, grid=target_grid, analysis_time=analysis_time,
            surface=roster.surface or None, rotation=rotation)
        batches.extend(built)
    provenance["batches"] = [entry for source in ("surface", "conventional")
        for entry in provenance.get(source, {}).get("batches", ())]
    provenance["valid_time_shifting"] = roster.receipt
    return batches, provenance


def make_cwp_provider(roster, run_cfg, *, setup_arrays,
                      setup_identity_sha256, composition=None):
    """Evaluate the existing GOES operator on each actual retained column."""
    from gpuwm.da.obsop_cwp import checkpoint_cwp_provider
    if setup_identity_sha256 != roster.receipt["setup_identity_sha256"]:
        raise ValueError("VTS CWP setup differs from the captured native column setup")
    provider = checkpoint_cwp_provider(run_cfg, composition=composition,
                                      **setup_arrays)

    def retained_provider(index, state, obs_class=None):
        index = _integer(index, "CWP covariance sample index")
        if index not in roster.snapshots or state is not roster.snapshots[index]:
            raise ValueError("VTS CWP must read the original retained sample")
        return provider(index, state, obs_class)

    return retained_provider
