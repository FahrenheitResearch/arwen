"""Per-parent-step one-way nest forcing.

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
    launch_mp8_to_mp18_parent_field,
    resolve_microphysics_transition,
    transition_handles_field,
    transition_parent_field_shape,
)
from gpuwm.core.nest_interp import bdy_interp1, register_nest
from gpuwm.core.preflight import (nest_field_kinds, nest_slot_dtypes,
                                  nest_slot_shapes)
from gpuwm.ingest.lateral_bc import (attach_nest_boundaries,
                                     couple_nest_field)


_STAGGER = {"u": "x", "v": "y"}
_APPLICATION_NAME = {"t": "theta", "ph": "phi"}
_SIDES = (("west", "xs"), ("east", "xe"),
          ("south", "ys"), ("north", "ye"))
_GEOMETRY_NAMES = ("ci", "ip", "cj", "jp", "xig", "xjg")


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

    def __init__(self, child_node):
        if child_node.parent is None:
            raise ValueError("NestCoupler requires a child DomainNode")
        if child_node.cfg.parent_id != child_node.parent.cfg.grid_id:
            raise ValueError("child/parent configuration link is inconsistent")
        self.child_node = child_node
        child = child_node.cfg
        parent = child_node.parent.cfg
        ratio = int(child.parent_grid_ratio)
        self.registrations = MappingProxyType({
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
        width = child.run.spec_bdy_width
        self.slot_shapes = nest_slot_shapes(child, width, parent)
        self.slot_dtypes = nest_slot_dtypes(child, width, parent)
        self.microphysics_transition = resolve_microphysics_transition(
            parent.run, child.run)
        self.force_count = 0
        self.first_parent_ticks = None
        self.last_parent_ticks = None
        self._geometry_bound = False
        self._valid = False
        self._last_tables = None

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
            launch_mp8_to_mp18_parent_field(
                parent_state, kind, out=out, coupled=True)
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
                parent_ticks // interval_ticks),
            "current_process_coverage_complete": (
                self.force_count
                == (parent_ticks - process_start_ticks) // interval_ticks),
            "first_parent_ticks": self.first_parent_ticks,
            "last_parent_ticks": self.last_parent_ticks,
        })
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
        """Dormant Phase-5 feedback prepare (feedback=0 is load-enforced)."""
        if node is not self.child_node:
            raise ValueError("feedback node does not match this coupler")
        out.payload = None

    def feedback_commit(self, node) -> None:
        if node is not self.child_node:
            raise ValueError("feedback node does not match this coupler")

    def feedback_finalize(self, node) -> None:
        if node is not self.child_node:
            raise ValueError("feedback node does not match this coupler")


__all__ = ["NestCoupler"]
