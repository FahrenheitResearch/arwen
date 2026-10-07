"""Native forecast retention and bounded RAM ownership for VTSM cycles."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from gpuwm.da import spread_vts_ram as ram

OFFSETS = (-900, 0, 900)
SCHEMA = "gpuwm.da.spread-vts-producer/v1"


def drain_native_history(unused_model, node, ticks):
    """Consume the native one-frame echo handoff even with no output writer."""
    from gpuwm.core.refl import consume_refl_10cm, domain_start_ticks_of
    from gpuwm.prepared_single_domain_forecast import _consume_due_native_refl_10cm
    _consume_due_native_refl_10cm(node.state, ticks, consume_refl_10cm,
                                domain_start_ticks=domain_start_ticks_of(node))


@dataclass(frozen=True)
class RetainedSlot:
    member: int
    offset_seconds: int
    analysis_seconds: float
    valid_ticks: int
    tick_den: int
    forecast_origin_seconds: float
    observation_cutoff_seconds: float
    grid_identity_sha256: str
    states: object
    reflectivity: object
    surface: object
    setup_identity_sha256: str


def validate_request(*, analysis_seconds, origin_seconds, history_seconds,
                     observation_heating=False):
    """Refuse missing clocks and current-observation leakage before integration."""
    for value in (analysis_seconds, origin_seconds, history_seconds):
        if not math.isfinite(float(value)) or float(value) != int(value):
            raise ValueError("VTSM requires actual whole-second native retention clocks")
    if analysis_seconds-origin_seconds < 900:
        raise ValueError("VTSM needs a real t-900 forecast after its common forecast origin")
    if history_seconds <= 0 or 900 % int(history_seconds) or (analysis_seconds-900) % int(history_seconds):
        raise ValueError("VTSM native history alarms must land exactly on the retained t-900 clock")
    if observation_heating:
        raise ValueError("VTSM refuses current-volume TTEN: it would leak analysis-time observations into the shifted forecast prior")


def setup_identity(fields):
    digest = sha256()
    for name, value in sorted(fields.items()):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(name.encode()+b"\0"+array.dtype.str.encode()+b"\0")
        digest.update(json.dumps(array.shape).encode()+b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


class ForecastRetention:
    """Capture actual model samples into the cycle's sealed-RAM server."""
    def __init__(self, *, socket_path, member, analysis_seconds,
                 origin_seconds, grid_identity_sha256, fields, surface_fields=()):
        self.socket_path = str(socket_path)
        self.member = int(member)
        self.analysis_seconds = float(analysis_seconds)
        self.origin_seconds = float(origin_seconds)
        self.grid_identity_sha256 = grid_identity_sha256
        self.fields = tuple(fields)
        self.surface_fields = tuple(surface_fields)
        self.records = {}

    def capture(self, node, driver, cfg, *, offset_seconds):
        from gpuwm.da.obsop import simulated_reflectivity
        from tools.da_cycle_prepared import to_host
        if offset_seconds not in OFFSETS or offset_seconds in self.records:
            raise ValueError("a native VTSM slot must be captured exactly once at a declared offset")
        ticks, denominator = int(node.clock.ticks), int(node.clock.tick_den)
        expected = self.analysis_seconds+offset_seconds
        if ticks/denominator != expected:
            raise ValueError("VTSM slot capture is at the wrong native clock; no retimestamped state is allowed")
        needed = {name for name in self.fields if getattr(node.state, name, None) is not None} | {"u", "v", "w", "p", "alt", "php", "qv", "thp", "mup", "thb", "phb"}
        states = {}
        for name in sorted(needed):
            value = getattr(node.state, name, None)
            if value is None:
                raise ValueError(f"VTSM actual observer state lacks {name}; a missing native field cannot be synthesized")
            states[name] = to_host(value)
        setup = {}
        for name in ("c1h", "c2h", "dnw", "mub2d", "phb", "thb"):
            value = getattr(node.state, name, None)
            if value is None:
                raise ValueError(f"VTSM immutable native setup lacks {name}")
            setup[name] = to_host(value)
        fields = {"state/"+name: value for name, value in states.items()}
        fields["reflectivity"] = to_host(simulated_reflectivity(node.state, cfg))
        for name in self.surface_fields:
            value = driver.fields.get(name)
            if value is None:
                raise ValueError(f"VTSM actual shifted surface observer lacks native {name}")
            fields["surface/"+name] = to_host(value)
        metadata = {
            "schema": SCHEMA, "member": self.member, "offset_seconds": offset_seconds,
            "analysis_seconds": self.analysis_seconds, "valid_ticks": ticks,
            "tick_den": denominator, "forecast_origin_seconds": self.origin_seconds,
            "observation_cutoff_seconds": self.origin_seconds,
            "grid_identity_sha256": self.grid_identity_sha256,
            "setup_identity_sha256": setup_identity(setup),
        }
        key = f"analysis-{int(self.analysis_seconds)}-member-{self.member}-offset-{offset_seconds}"
        record = ram.put(self.socket_path, key, fields, metadata)
        self.records[offset_seconds] = {"key": key, "metadata": metadata,
                                       "bytes": record["bytes"]}
        return self.records[offset_seconds]

    def history(self, handler, *, driver, cfg):
        def capture(model, node, ticks):
            result = (handler(model, node, ticks) if handler is not None
                      else drain_native_history(model, node, ticks))
            if int(node.cfg.grid_id) == 1 and ticks/node.clock.tick_den == self.analysis_seconds-900:
                self.capture(node, driver, cfg, offset_seconds=-900)
            return result
        return capture

    def receipt(self):
        if set(self.records) != set(OFFSETS):
            raise ValueError("VTSM lacks an actual past/central/future native capture")
        return {"schema": SCHEMA, "slots": [self.records[offset] for offset in OFFSETS]}


def continue_and_restore(model, *, restart, experiment, boundary_interval_seconds,
                         analysis_seconds, capture, child_domain=None):
    """Continue the existing native tree, then restore its complete central set.

    Only the native restart owner places clocks and restores state. The future
    uses a newly resolved schedule with the same integer lattice and forcing.
    The original physics owners remain attached. No future restart copy is
    written. Restoration is attempted even if continuation/capture fails.
    """
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import execute_experiment
    from gpuwm.da.nested_forecast import nested_experiment
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
    from tools.da_cycle_prepared import restore_leg_restart

    extended = replace(experiment, run_seconds=float(analysis_seconds)+900.)
    children = ()
    if child_domain is not None:
        extended = nested_experiment(extended, child_domain)
        children = (int(child_domain.grid_id),)
    resolution = resolve_clock(extended, lbc_interval_s=float(boundary_interval_seconds),
                               live_born_children=children)
    schedule = build_schedule(extended, resolution)
    clocks = resolution.clocks()
    if set(clocks) != set(model.nodes_by_grid_id):
        raise ValueError("VTSM continuation must carry the exact complete central domain tree")
    denominator = int(model.root.clock.tick_den)
    target = (float(analysis_seconds)+900)*denominator
    if int(resolution.tick_den) != denominator or target != int(target) or int(target) % schedule.period_ticks:
        raise ValueError("VTSM future clock must be an exact native complete-tree period boundary")
    previous_schedule = model.schedule
    previous_clocks = {gid: node.clock for gid, node in model.nodes_by_grid_id.items()}
    previous_history = getattr(model, "_resume_committed_history_grid_ids", frozenset())
    try:
        model.schedule = schedule
        for gid, node in model.nodes_by_grid_id.items():
            node.clock = clocks[gid]
            if getattr(node.cfg.run, "specified", False):
                bind_lateral_boundary_clock(node.state, node.clock)
        restore_leg_restart(model, restart, expected_seconds=analysis_seconds)
        # The continuation integrates ``extended``; handed explicitly so an
        # active [spectral_numerics] reaches the seam on this route too
        # (tests/test_spectral_seam.py, every integrating route).
        execute_experiment(model, history_handler=drain_native_history,
                           progress_callback=None, validate_state=True,
                           skip_feedback_path=True, experiment=extended)
        capture()
    finally:
        model.schedule = previous_schedule
        for gid, node in model.nodes_by_grid_id.items():
            node.clock = previous_clocks[gid]
            if getattr(node.cfg.run, "specified", False):
                bind_lateral_boundary_clock(node.state, node.clock)
        restore_leg_restart(model, restart, expected_seconds=analysis_seconds)
        model._resume_committed_history_grid_ids = previous_history
    return {"future_seconds": float(analysis_seconds)+900,
            "central_restored_seconds": float(model.root.clock.elapsed_seconds),
            "domain_ids": sorted(int(gid) for gid in model.nodes_by_grid_id),
            "restart_owner": "gpuwm.io.restart.restore_tree_restart"}


class VtsCycle:
    """Own retained slots and their borrow leases for one scheduler."""
    def __init__(self, path, *, max_bytes):
        self.server = ram.RetainedRam(path, max_bytes=max_bytes).start()
        self.borrow = ExitStack()
        self.keys = []

    def slots(self, results):
        slots = []
        for member in sorted(key for key in results if key != "control"):
            receipt = results[member].record.get("spread_vts")
            if not receipt or receipt.get("schema") != SCHEMA:
                raise ValueError("VTSM packed member did not publish actual retained forecast slots")
            for reference in receipt["slots"]:
                fields, record = self.borrow.enter_context(ram.get(self.server.path, reference["key"]))
                metadata = record["metadata"]
                if metadata != reference["metadata"] or metadata["member"] != int(member):
                    raise ValueError("VTSM packed slot receipt differs from its sealed native RAM inventory")
                self.keys.append(reference["key"])
                states = MappingProxyType({name[6:]: value for name, value in fields.items() if name.startswith("state/")})
                surface = MappingProxyType({name[8:]: value for name, value in fields.items() if name.startswith("surface/")})
                slots.append(RetainedSlot(states=states, surface=surface,
                    reflectivity=fields["reflectivity"], **{key: metadata[key] for key in
                    ("member", "offset_seconds", "analysis_seconds", "valid_ticks", "tick_den",
                     "forecast_origin_seconds", "observation_cutoff_seconds",
                     "grid_identity_sha256", "setup_identity_sha256")}))
        return slots

    def release(self):
        self.borrow.close()
        self.borrow = ExitStack()
        for key in self.keys:
            ram.drop(self.server.path, key)
        self.keys.clear()

    def clear(self):
        try:
            self.release()
        finally:
            self.server.clear()


def reduce_and_bound(roster, increments, config):
    """Recenter K real central members and rebind their water/moment policy."""
    from gpuwm.da import moments, field_rules
    from gpuwm.da.positivity import apply_positivity, verify_non_negative, BOUNDING_POLICIES
    from gpuwm.da.radar_assimilation import _saturation_bound
    members = tuple(range(roster.member_count))
    states = {index: state for index, state in enumerate(roster.central_states)}
    reduced = roster.reduce_member_increments(increments)
    fields = tuple(reduced[0])
    prior = {name: np.stack([states[i][name] for i in members]) for name in fields}
    stacked = {name: np.stack([reduced[i][name] for i in members]) for name in fields}
    water_fields = {"qv", "qc", "qr", "qi", "qs", "qg", "qh"}
    before = {name: float(np.sum(value, dtype=np.float64))
              for name, value in stacked.items() if name in water_fields}
    positivity = saturation = numbers = None
    if config.positivity_policy is not None:
        stacked, positivity = apply_positivity(prior, stacked,
                                               policy=config.positivity_policy)
        if config.positivity_policy in BOUNDING_POLICIES:
            verify_non_negative(prior, stacked)
    if "qv" in stacked:
        stacked, saturation = _saturation_bound(prior, stacked, states, members)
    if config.field_rules == "design" and config.number_rediagnosis != "off":
        stacked, numbers = field_rules.rediagnose_numbers(
            prior, stacked, states, members, mode=config.number_rediagnosis,
            mp_physics=config.mp_physics)
    repairs = []
    for index in members:
        analysed = dict(states[index])
        for name, value in stacked.items():
            analysed[name] = np.asarray(states[index][name])+value[index]
        if config.moment_policy == "single-moment-with-repair":
            corrected, receipt = moments.repair_moments(analysed, mp_physics=config.mp_physics)
            repairs.append(receipt)
            for name, value in corrected.items():
                if name not in stacked:
                    stacked[name] = np.zeros((len(members),)+np.shape(value), dtype=np.float64)
                stacked[name][index] = np.asarray(value)-states[index][name]
    output = {index: {name: value[index] for name, value in stacked.items()} for index in members}
    after = {name: float(np.sum(value, dtype=np.float64))
             for name, value in stacked.items() if name in water_fields}
    return output, {"covariance_samples": 3*len(members), "posterior_trajectories": len(members),
                    "positivity": positivity, "saturation": saturation,
                    "number_rediagnosis": numbers, "moment_repairs": repairs,
                    "water_increment_sum_kg_kg_before_rebind": before,
                    "water_increment_sum_kg_kg_after_rebind": after,
                    "water_budget_units": "sum of member-cell mixing-ratio increments; no volume integral claimed"}
