"""Per-parent-step nest forcing and experimental two-way restriction.

The coupler is node-facing (``force(node)``), leaves the parent state
read-only, and refreshes child-owned rolling boundary value/tendency tables.
It consumes Task 10's landed full-extent ``bdy_interp1`` unchanged.  F16's
retired perimeter donor strips are replaced by one full-parent coupled field
borrowed from the shared, lifetime-audited arena for each field in turn.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from gpuwm.core.model import FeedbackScratch
from gpuwm.core.microphysics_transition import (
    launch_microphysics_edge_parent_field,
    resolve_microphysics_transition,
    transition_handles_field,
    transition_parent_field_shape,
)
from gpuwm.core.inflow_perturbation import build_inflow_perturbation
from gpuwm.core.nest_interp import bdy_interp1, copy_fcn, register_nest
from gpuwm.core.preflight import (nest_field_kinds, nest_slot_dtypes,
                                  nest_slot_shapes)
from gpuwm.ingest.lateral_bc import (attach_nest_boundaries,
                                     couple_nest_field,
                                     uncouple_feedback_field)


_STAGGER = {"u": "x", "v": "y"}
_APPLICATION_NAME = {"t": "theta", "ph": "phi"}
_SIDES = (("west", "xs"), ("east", "xe"),
          ("south", "ys"), ("north", "ye"))
_GEOMETRY_NAMES = ("ci", "ip", "cj", "jp", "xig", "xjg")
MISMATCHED_MICROPHYSICS_FEEDBACK_BLOCKER = (
    "cross-scheme-feedback-reverse-mapping-unimplemented-v1"
)


def _field_shape(state, kind: str) -> tuple[int, int, int]:
    if kind == "mu":
        return (1, *state.mup.shape)
    name = {"t": "thp", "ph": "php"}.get(kind, kind)
    field = getattr(state, name, None)
    if field is None:
        raise ValueError(f"state has no active nest field {kind!r}")
    return tuple(int(n) for n in field.shape)


class NestCoupler:
    """One parent->child forcing edge.

    Geometry is built on the host at construction and bound to the child's
    F4/F16 manifest slots immediately before the first force-time device use.
    The three registrations correspond to mass, x-staggered, and
    y-staggered horizontal geometry; w/ph use mass horizontal geometry.
    """

    def __init__(self, child_node, *, feedback: int = 0):
        if child_node.parent is None:
            raise ValueError("NestCoupler requires a child DomainNode")
        if child_node.cfg.parent_id != child_node.parent.cfg.grid_id:
            raise ValueError("child/parent configuration link is inconsistent")
        self.child_node = child_node
        if feedback not in (0, 1):
            raise ValueError("NestCoupler feedback must be 0 or 1")
        self.feedback = int(feedback)
        child = child_node.cfg
        parent = child_node.parent.cfg
        self.registrations = self._build_registrations()
        width = child.run.spec_bdy_width
        self.slot_shapes = nest_slot_shapes(child, width, parent)
        self.slot_dtypes = nest_slot_dtypes(child, width, parent)
        self.microphysics_transition = resolve_microphysics_transition(
            parent.run, child.run)
        if self.feedback == 1 and parent.run.nz != child.run.nz:
            raise ValueError(
                "experimental feedback is horizontal-only and requires "
                "identical parent/child vertical level counts")
        if self.feedback == 1 and self.microphysics_transition.mixed:
            raise ValueError(
                f"{MISMATCHED_MICROPHYSICS_FEEDBACK_BLOCKER}: experimental "
                "two-way feedback has no ratified reverse mass/moment "
                f"mapping for MP{parent.run.mp_physics}->"
                f"MP{child.run.mp_physics}")
        if (self.feedback == 1
                and nest_field_kinds(parent.run)
                != nest_field_kinds(child.run)):
            raise ValueError(
                "experimental two-way feedback requires identical active "
                "parent/child prognostic field inventories; the configured "
                f"one-way transition {self.microphysics_transition.policy_id!r} "
                "has no reverse restriction contract")
        # LES-nest inflow seeding (P3): None unless the child config
        # turns it on, and the force path executes nothing of it when
        # None -- the OFF trajectory is gated byte-identical to a build
        # without the mechanism (INFLOW-GENERATOR-ACCEPTANCE-V2 G1).
        self.inflow_perturbation = build_inflow_perturbation(child_node)
        self.force_count = 0
        self.first_parent_ticks = None
        self.last_parent_ticks = None
        self._geometry_bound = False
        self._valid = False
        self._last_tables = None
        self._prepared_feedback = None
        self.feedback_count = 0
        self.last_feedback_ticks = None
        self.placement_generation = 0
        self.relocation_count = 0

    def _build_registrations(self):
        """The three stagger registrations for the child's CURRENT placement.

        Mass, x-staggered and y-staggered horizontal geometry; w/ph use the
        mass registration because z-staggering does not change the
        horizontal stencil.
        """
        child = self.child_node.cfg
        parent = self.child_node.parent.cfg
        ratio = int(child.parent_grid_ratio)
        return MappingProxyType({
            stagger: register_nest(
                nri=ratio, nrj=ratio,
                i_parent_start=child.i_parent_start,
                j_parent_start=child.j_parent_start,
                child_nx=child.run.nx, child_ny=child.run.ny,
                parent_nx=parent.run.nx, parent_ny=parent.run.ny,
                stagger=("" if stagger == "m" else stagger),
                wrapper="bdy")
            for stagger in ("m", "x", "y")
        })

    def relocate(self) -> dict:
        """Rebuild the SINT donor tables for a new placement generation.

        This is the numerics half of a discrete relocation and it is the
        reason the design does not need WRF's per-step moving-nest
        mechanism: the donor index/sub-cell/offset tables are still
        precomputed ONCE, just once *per placement generation* instead of
        once per run.  Between relocations the per-step path is unchanged.

        The caller updates ``child_node.cfg`` first; this reads the new
        placement off it.  Nothing is reallocated -- the F4/F16 manifest
        slot shapes are functions of the child/parent EXTENTS and the
        refinement ratio, none of which a relocation changes -- so the
        rebuilt tables are re-uploaded into the same audited slots.

        The rolling boundary value/tendency tables are classified INVALID
        rather than rebuilt, for the same reason :meth:`invalidate` gives:
        only the next ordinary parent STEP followed by FORCE has the
        correct ``parent(t+dt_p)`` endpoint, and the tables that exist
        describe a footprint the child no longer occupies.
        """
        child = self.child_node.cfg
        previous = tuple(
            (reg.i_parent_start, reg.j_parent_start)
            for reg in self.registrations.values())
        self.registrations = self._build_registrations()
        # Force a fresh bind-before-first-use pass: the new registrations
        # carry no device tables, and _bind_geometry writes them into the
        # same manifest slots the retired ones used.
        self._geometry_bound = False
        self.invalidate()
        self.placement_generation += 1
        self.relocation_count += 1
        return {
            "placement_generation": int(self.placement_generation),
            "relocation_count": int(self.relocation_count),
            "i_parent_start": int(child.i_parent_start),
            "j_parent_start": int(child.j_parent_start),
            "previous_registration_placements": [
                [int(i), int(j)] for i, j in previous],
            "rolling_tables": "INVALID",
        }

    @property
    def valid(self) -> bool:
        return self._valid

    def invalidate(self) -> None:
        """Classify all rolling child boundary tables INVALID on restore.

        Geometry remains deterministic setup state.  Value/tendency tables
        are deliberately not rebuilt here: only the next ordinary parent
        STEP followed by FORCE has the correct ``parent(t+dt_p)`` endpoint.
        """
        self._valid = False
        resident = getattr(
            self.child_node.state, "_lateral_boundary_device", None)
        if resident is not None:
            resident.valid = False

    def _scratch(self, slot: str):
        shape = self.slot_shapes[slot]
        return self.child_node.state.scratch(
            shape, slot, dtype=self.slot_dtypes[slot])

    def _bind_geometry(self) -> None:
        if self._geometry_bound:
            return
        for stagger, reg in self.registrations.items():
            def alloc(name, shape, dtype, *, _stag=stagger):
                if name not in _GEOMETRY_NAMES:
                    raise RuntimeError(f"unknown nest geometry table {name!r}")
                slot = f"nest_sint_{name}_{_stag}"
                expected_shape = self.slot_shapes[slot]
                expected_dtype = self.slot_dtypes[slot]
                if tuple(shape) != expected_shape:
                    raise RuntimeError(
                        f"geometry {slot} shape {tuple(shape)} != F4/F16 "
                        f"manifest {expected_shape}")
                actual_dtype = np.dtype(dtype).name
                if actual_dtype != expected_dtype:
                    raise RuntimeError(
                        f"geometry {slot} dtype {actual_dtype} != manifest "
                        f"{expected_dtype}")
                return self._scratch(slot)

            tables = reg.device_tables(alloc=alloc)
            if set(tables) != set(_GEOMETRY_NAMES):
                raise RuntimeError("NestRegistration device table inventory drift")
        self._geometry_bound = True

    def _coupled_parent_field(self, kind: str):
        """Overwrite and return F16's one full-parent arena prefix."""
        parent_state = self.child_node.parent.state
        if transition_handles_field(self.microphysics_transition, kind):
            shape = transition_parent_field_shape(parent_state, kind)
        else:
            shape = _field_shape(parent_state, kind)
        backing = self._scratch("nest_parent_field")
        count = math.prod(shape)
        if count > backing.size:
            raise RuntimeError("parent field exceeds F16 arena capacity")
        out = backing.reshape(-1)[:count].reshape(shape)
        if transition_handles_field(self.microphysics_transition, kind):
            launch_microphysics_edge_parent_field(
                self.microphysics_transition, parent_state, kind,
                out=out, coupled=True)
        else:
            couple_nest_field(parent_state, kind, out=out)
        return out

    def transition_receipt(self):
        """Return the resolved edge policy plus observed forcing coverage."""

        parent_ticks = int(self.child_node.parent.clock.ticks)
        interval_ticks = int(
            self.child_node.parent.clock.spec.step_ticks)
        process_start_ticks = (
            parent_ticks if self.first_parent_ticks is None
            else int(self.first_parent_ticks) - interval_ticks)
        receipt = dict(self.microphysics_transition.receipt())
        receipt.update({
            "source_domain": int(self.child_node.parent.cfg.grid_id),
            "target_domain": int(self.child_node.cfg.grid_id),
            "requested_policy": str(
                self.child_node.cfg.run.nest_microphysics_transition),
            "effective_policy": self.microphysics_transition.policy_id,
            "observation_scope": "current_process_since_build_or_restore",
            "process_start_parent_ticks": process_start_ticks,
            "process_force_count": int(self.force_count),
            "parent_interval_ticks": interval_ticks,
            "final_parent_ticks": parent_ticks,
            "expected_cumulative_force_count": (
                max(0, (parent_ticks
                        - self.child_node.clock.spec.start_ticks)
                    // interval_ticks)),
            "domain_start_ticks": int(
                self.child_node.clock.spec.start_ticks),
            "current_process_coverage_complete": (
                self.force_count
                == (parent_ticks - process_start_ticks) // interval_ticks),
            "first_parent_ticks": self.first_parent_ticks,
            "last_parent_ticks": self.last_parent_ticks,
        })
        if self.microphysics_transition.mixed:
            init_count = int(getattr(
                self.child_node.state,
                "_microphysics_transition_init_count", 0))
            receipt["initialization_mapping_count"] = init_count
            receipt["per_species_processing_counts"] = [
                {
                    **dict(row),
                    "initialization_count": init_count,
                    "lateral_forcing_count": int(self.force_count),
                    "total_edge_processing_count": (
                        init_count + int(self.force_count)),
                }
                for row in self.microphysics_transition.species_actions()
            ]
        return MappingProxyType(receipt)

    def _coupled_child_field(self, kind: str):
        """Overwrite and return F16's one full-child arena prefix.

        ``bdy_interp1`` validates both full extents.  The child copy therefore
        uses the audited ``nest_child_field`` slot.  Preflight aliases that
        logical slot to an RK backing distinct from ``nest_parent_field``
        whenever two dead backings have sufficient capacity, and otherwise
        retains an explicit correctness-first backing.  The distinction is
        required because parent and child copies are simultaneously live in
        ``bdy_interp1``.  This also supports valid skinny grids where U or V
        is larger than W.
        """
        state = self.child_node.state
        shape = _field_shape(state, kind)
        backing = self._scratch("nest_child_field")
        count = math.prod(shape)
        if count > backing.size:
            raise RuntimeError("child field exceeds F16 arena capacity")
        out = backing.reshape(-1)[:count].reshape(shape)
        couple_nest_field(state, kind, out=out)
        return out

    def _rolling_out(self, kind: str):
        result = {}
        for side, suffix in _SIDES:
            result[side] = (
                self._scratch(f"nest_{kind}_b{suffix}"),
                self._scratch(f"nest_{kind}_bt{suffix}"),
            )
        return result

    def force(self, node) -> None:
        """Refresh this child's rolling tables from ``parent(t+dt_p)``.

        Parent interval ticks and REAL dt come only from ``node.parent.clock``.
        The child clock is reset through :meth:`DomainClock.mark_force`; no
        floating dtbc shadow is maintained by the coupler.  This is the
        non-mutating equivalent of ``mediation_force_domain.F:111-206``;
        the WRF transaction couples parent and child, interpolates, uncouples
        both, then resets ``nested_grid%dtbc`` at line 206.
        """
        if node is not self.child_node:
            raise ValueError("NestCoupler.force called with a different node")
        parent = node.parent
        if parent is None:
            raise ValueError("cannot force a root domain")
        lead = int(parent.clock.ticks) - int(node.clock.ticks)
        parent_interval_ticks = int(parent.clock.spec.step_ticks)
        if lead != parent_interval_ticks:
            raise RuntimeError(
                f"parent must lead child by one parent interval before FORCE; "
                f"lead={lead}, interval={parent_interval_ticks}")

        self._bind_geometry()
        fields = {}
        run = node.cfg.run
        for kind in nest_field_kinds(run):
            parent_field = self._coupled_parent_field(kind)
            child_field = self._coupled_child_field(kind)
            stagger = _STAGGER.get(kind, "m")
            out = self._rolling_out(kind)
            bdy_interp1(
                parent_field, child_field, self.registrations[stagger],
                parent_dt_fp32=parent.clock.spec.dt_fp32,
                parent_interval_ticks=parent_interval_ticks,
                spec_zone=run.spec_zone, relax_zone=run.relax_zone,
                spec_bdy_width=run.spec_bdy_width, out=out)
            fields[_APPLICATION_NAME.get(kind, kind)] = out

        if self.inflow_perturbation is not None:
            # After bdy_interp1 has written every rolling table and
            # before the metadata attach: the theta VALUE tables gain
            # the registered relax-zone increment, in their own coupled
            # units, and nothing else is touched.
            self.inflow_perturbation.apply_at_force(node, fields)

        attach_nest_boundaries(
            node.state, fields, clock=node.clock,
            spec_bdy_width=run.spec_bdy_width,
            spec_zone=run.spec_zone, relax_zone=run.relax_zone)
        node.clock.mark_force()
        parent_ticks = int(parent.clock.ticks)
        if self.first_parent_ticks is None:
            self.first_parent_ticks = parent_ticks
        self.last_parent_ticks = parent_ticks
        self.force_count += 1
        self._last_tables = MappingProxyType(fields)
        self._valid = True

    def feedback_prepare(self, node, out: FeedbackScratch) -> None:
        """Freeze the synchronized feedback field plan.

        The numerical transaction stays inside the two already-audited
        full-field arena slots, so fields are restricted and committed one
        at a time in :meth:`feedback_commit`; no per-field persistent payload
        or unregistered allocation is introduced.
        """
        if node is not self.child_node:
            raise ValueError("feedback node does not match this coupler")
        if self.feedback == 0:
            out.payload = None
            return
        parent = node.parent
        if parent is None:
            raise ValueError("cannot feed back a root domain")
        if int(parent.clock.ticks) != int(node.clock.ticks):
            raise RuntimeError(
                f"feedback requires synchronized clocks, got parent "
                f"{parent.clock.ticks} and child {node.clock.ticks}")
        kinds = nest_field_kinds(parent.cfg.run)
        missing = [
            kind for kind in kinds
            if kind != "mu" and getattr(
                node.state, {"t": "thp", "ph": "php"}.get(kind, kind),
                None) is None
        ]
        if missing:
            raise RuntimeError(
                f"child state lacks parent feedback fields {missing}")
        payload = {
            "kinds": kinds,
            "ticks": int(node.clock.ticks),
        }
        self._prepared_feedback = payload
        out.payload = payload

    def feedback_commit(self, node) -> None:
        if node is not self.child_node:
            raise ValueError("feedback node does not match this coupler")
        if self.feedback == 0:
            return
        payload = self._prepared_feedback
        if payload is None:
            raise RuntimeError("feedback commit has no prepared transaction")
        if int(node.clock.ticks) != int(payload["ticks"]):
            raise RuntimeError("feedback clock changed after prepare")
        parent = node.parent
        run = node.cfg.run
        self._bind_geometry()

        # WRF couples/restricts MU before uncoupling momenta and scalars.
        # MU itself is already in feedback units, so it can be written
        # directly into the exact parent overlap.
        child_mu = self._coupled_child_field("mu")
        copy_fcn(
            parent.state.mup[None], child_mu, self.registrations["m"],
            spec_zone=run.spec_zone)

        for kind in payload["kinds"]:
            if kind == "mu":
                continue
            child_field = self._coupled_child_field(kind)
            shape = _field_shape(parent.state, kind)
            backing = self._scratch("nest_parent_field")
            count = math.prod(shape)
            if count > backing.size:
                raise RuntimeError("parent feedback field exceeds arena capacity")
            restricted = backing.reshape(-1)[:count].reshape(shape)
            stagger = _STAGGER.get(kind, "m")
            reg = self.registrations[stagger]
            copy_fcn(
                restricted, child_field, reg, spec_zone=run.spec_zone)
            uncouple_feedback_field(
                parent.state, kind, restricted, reg,
                spec_zone=run.spec_zone)
        self.feedback_count += 1
        self.last_feedback_ticks = int(node.clock.ticks)

    def feedback_finalize(self, node) -> None:
        if node is not self.child_node:
            raise ValueError("feedback node does not match this coupler")
        if self.feedback == 0:
            return
        if self._prepared_feedback is None:
            raise RuntimeError("feedback finalize has no committed transaction")
        from gpuwm.core.diagnostics import update_diagnostics

        parent = node.parent
        update_diagnostics(parent.state, parent.cfg.run.hypsometric_opt)
        self._prepared_feedback = None


__all__ = [
    "MISMATCHED_MICROPHYSICS_FEEDBACK_BLOCKER", "NestCoupler",
]
