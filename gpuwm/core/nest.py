"""Per-parent-step nest forcing and experimental two-way restriction.

The coupler is node-facing (``force(node)``), leaves the parent state
read-only, and refreshes child-owned rolling boundary value/tendency tables.
It consumes Task 10's landed full-extent ``bdy_interp1`` unchanged.  F16's
retired perimeter donor strips are replaced by one full-parent coupled field
borrowed from the shared, lifetime-audited arena for each field in turn.

WHEN A COUPLED DOMAIN IS STREAMED, ``node.state`` IS NOT THE DOMAIN
------------------------------------------------------------------
``gpuwm.core.streaming.attach`` copies a prepared domain's carriers into a
store and every sweep after that updates THE STORE.  The ``DomainState``
object stays in the tree and its device arrays stop moving.  Every read in
this module is off ``node.state`` / ``parent.state``, so a coupler that did
not know about the store would:

* force the child from the parent's air AT ATTACH TIME, for the whole
  forecast, with no NaN and no warning -- a nest that looks like it is
  running and is being driven by t=0;
* under ``feedback=1``, write the parent's prognostics into arrays the next
  sweep re-reads from the store, so two-way feedback is silently discarded.

Both are repaired by asking :func:`gpuwm.core.streaming.domain_store`
whether a state is a window onto something else -- ``None`` for every
resident domain, which is every domain in a run that configures no
``[tiles]`` -- and moving exactly the fields the transaction reads or
writes.

MEASURED, ``tilestream/test_nest.py``: d01 192x192x49 dx 9 km STREAMED from
a pinned host store at tile 64 (3x3, exact) with halo 16 from
``harness.halo_radius``, d02 96x96x49 nested 3:1 and resident,
full(real74)+KF, 6 parent steps and 18 child steps.  With the store
consulted, both domains are bit-identical to the all-resident control --
0 of 155 carriers differ on each -- at ``feedback=0`` and at ``feedback=1``.
With it not consulted, which is the code path before this paragraph existed,
d01 is still bit-exact (streaming itself was never the problem) and d02
differs in 89 of 155 carriers with a worst absolute difference of 1.04e+08.
The two feedback controls fire too: ``feedback=1`` differs from
``feedback=0`` on the resident control, and disarming ONLY the write-back
leaves the parent's store without the child's mass.

WHAT THIS DOES NOT FIX, AND WHY IT IS SAID HERE
-----------------------------------------------
The refresh writes the domain's OWN resident device arrays, which exist only
because today's streaming route attaches from a prepared resident state and
nothing frees it.  It is a correctness repair, not a capacity one: the
``nest_parent_field`` arena slot is still O(one full parent field), and the
traffic is still O(the fields the transaction touches) per parent step.
``bdy_interp1`` only ever READS the parent inside the child's footprint plus
the +-2 SINT stencil -- a 256-wide 3:1 child covers 86 parent columns, so 90
with the stencil, 3.1% of a 512^2 parent by area and a 32x smaller slot --
so a windowed ``bdy_interp1`` would turn both the slot and the traffic into
O(child footprint).  That is the fix this one stands in for and it is not
implemented.
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


def _state_attr(kind: str) -> str:
    """The ``DomainState`` attribute one nest field kind lives on."""
    return {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)


#: What ``gpuwm.core.diagnostics.update_diagnostics`` reads that is not setup
#: geometry.  ``feedback_finalize`` re-runs it over the WHOLE parent, so a
#: streamed parent needs these four correct everywhere -- not just inside the
#: feedback rectangle -- before the call.
_DIAGNOSTIC_INPUTS = ("mup", "thp", "php", "qv")

#: What that call WRITES.  A streamed parent has to carry them back or its
#: store keeps the pre-feedback diagnostics.
_DIAGNOSTIC_OUTPUTS = ("p", "al", "alt")


def _sync_in(state, attrs) -> int:
    """Pull ``attrs`` from a streamed domain's store; no-op when resident."""
    from gpuwm.core.streaming import refresh_from_store

    return refresh_from_store(state, attrs)


def _sync_out(state, attrs) -> int:
    """Push ``attrs`` into a streamed domain's store; no-op when resident."""
    from gpuwm.core.streaming import commit_to_store

    return commit_to_store(state, attrs)


def _is_streamed(state) -> bool:
    from gpuwm.core.streaming import domain_store

    return domain_store(state) is not None


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
        """Overwrite and return F16's one full-parent arena prefix.

        ``couple_nest_field`` reads exactly two things that MOVE: the field
        itself and ``mup`` (the coupling mass, read at neighbouring points
        for the u/v face averages).  Everything else it touches -- ``mub2d``,
        the hybrid coefficients, the map factors, ``thb`` -- is geography and
        base state, which streaming treats as INPUT and never scatters, so a
        streamed parent's copies of those are still correct.  So the pull
        below is two fields, not a state.
        """
        parent_state = self.child_node.parent.state
        if transition_handles_field(self.microphysics_transition, kind):
            if _is_streamed(parent_state):
                raise RuntimeError(
                    "a cross-scheme microphysics nest edge "
                    f"({self.microphysics_transition.policy_id!r}) off a "
                    "STREAMED parent is unimplemented: "
                    "launch_microphysics_edge_parent_field reads parent "
                    "species this coupler cannot enumerate, so it would "
                    "silently map the parent's air at attach time.  Refused "
                    "rather than run, because the wrong answer here is "
                    "finite and plausible.")
            shape = transition_parent_field_shape(parent_state, kind)
        else:
            _sync_in(parent_state, ("mup", _state_attr(kind)))
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
        _sync_in(state, ("mup", _state_attr(kind)))
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

        if _is_streamed(node.state):
            raise RuntimeError(
                "a STREAMED child cannot be forced: the rolling nest tables "
                "attach to the child's DomainState "
                "(lateral_bc.attach_nest_boundaries installs a device "
                "descriptor on it), while a streamed domain is stepped as "
                "tile buffers whose only boundary hook is "
                "streaming.make_tile_hook -> attach_lateral_boundaries.  "
                "There is no per-tile windowing of nest boundary tables at "
                "all, so a streamed child is not untested, it is "
                "unimplemented.  Stream the PARENT instead, or run the "
                "child resident.")

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

        # A STREAMED parent is mutated here from outside dycore.step, so the
        # transaction has to start from the store and end in it.  Pulled in:
        # every field the restriction READS on the parent side -- ``mup``,
        # because the momentum/scalar inverses divide by the parent's own
        # (just-restricted) column mass and the u/v face averages straddle
        # the feedback rectangle's edge, plus the four inputs
        # ``feedback_finalize``'s whole-parent update_diagnostics consumes.
        # Pushed back out below: everything written.  This is O(the fields
        # the transaction touches) per parent step, which is the honest cost
        # of a whole-domain finalize; see the module docstring.
        streamed_parent = _is_streamed(parent.state)
        written = ["mup"]
        if streamed_parent:
            _sync_in(parent.state,
                     tuple(dict.fromkeys(
                         _DIAGNOSTIC_INPUTS
                         + tuple(_state_attr(k) for k in payload["kinds"]))))

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
            written.append(_state_attr(kind))
        if streamed_parent:
            _sync_out(parent.state, tuple(dict.fromkeys(written)))
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
        # WHOLE-PARENT, and that is the point of the call: WRF re-runs
        # start_domain on the parent after med_nest_feedback
        # (share/mediation_integrate.F:787-838).  ``feedback_commit`` has
        # already made the four inputs correct everywhere on a streamed
        # parent, so this reads the domain rather than the attach-time
        # snapshot; the three outputs go back to the store.
        update_diagnostics(parent.state, parent.cfg.run.hypsometric_opt)
        _sync_out(parent.state, _DIAGNOSTIC_OUTPUTS)
        self._prepared_feedback = None


__all__ = [
    "MISMATCHED_MICROPHYSICS_FEEDBACK_BLOCKER", "NestCoupler",
]
