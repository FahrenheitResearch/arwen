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

BOUNDED OPERANDS FOR A CANONICAL STREAMED CHILD
---------------------------------------------
An actual ``_streamed_domain`` owner now selects bounded operands from
``nest_operands``. FORCE assembles the same rolling tables from child
boundary chunks and exact SINT donor rectangles. Feedback restricts child
chunks directly into the parent owner. A streamed parent runs the same
smoother kernels with one explicitly reported host J-pass scratch rectangle,
then diagnoses changed columns in bounded chunks. These paths need no
resident child field or full-child F16 scratch. The ordinary resident route
and the older published-store correctness seam remain intact.

This closes operand ownership, not admission: allocator pool retention,
the host smoothing scratch, rolling tables and simultaneous tiled buffers
still need inclusion in the route's capacity proof. Mixed microphysics and
optional inflow hooks retain their separate contracts. No admission guard is
relaxed by this implementation.
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
from gpuwm.core.nest_interp import (bdy_interp1, copy_fcn,
                                    feedback_parent_bounds, register_nest,
                                    smoother)
from gpuwm.core.preflight import (nest_field_kinds, nest_slot_dtypes,
                                  nest_slot_shapes)
from gpuwm.ingest.lateral_bc import (attach_nest_boundaries,
                                     couple_nest_field)


_STAGGER = {"u": "x", "v": "y"}
_APPLICATION_NAME = {"t": "theta", "ph": "phi"}

#: The six SIGNED prognostics in the feedback inventory; everything else
#: (qv, qc, qr, ... and every number concentration) is positive-definite
#: and gets the post-smdsm non-negativity clamp in ``feedback_commit``.
_SIGNED_KINDS = frozenset({"u", "v", "w", "t", "ph", "mu"})

#: WRF's ``t0`` (share/module_model_constants.F:37, ``PARAMETER :: t0 =
#: 300.``): the prognostic ``t_2`` is ``theta - t0``, a GLOBAL constant
#: offset, where gpuwm's ``thp`` is ``theta - thb`` against a per-grid base
#: profile.  Feedback is the one place the two spellings meet.
_WRF_T0 = np.float32(300.0)


def _rebase_feedback_theta(state, reg, spec_zone) -> None:
    """Re-express restricted WRF ``t_2`` against the parent's base theta.

    WRF restricts ``t_2 = theta - 300`` and writes it straight into the
    parent's own ``t_2``, because 300 is a global constant and parent and
    child mean the same thing by it.  gpuwm stores ``thp = theta - thb``
    against a base profile that differs between the two grids, so after
    ``copy_fcn`` has written ``mean(theta_child) - 300`` the reference
    frame has to be changed -- ``+ 300 - thb_parent`` -- over exactly the
    cells ``copy_fcn`` just wrote, which are ``feedback_parent_bounds``
    (that IS ``copy_fcn``'s launch rectangle, nest_interp.py:664-668).
    The expression order matches the kernel this replaces
    (kernels/lbc_state.cu:670-672, ``(result + 300) - thb``).
    """
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
        reg, spec_zone=spec_zone)
    if i_hi < i_lo or j_hi < j_lo:
        return
    window = state.thp[:, j_lo:j_hi + 1, i_lo:i_hi + 1]
    window += _WRF_T0
    thb = state.thb
    window -= (thb[:, j_lo:j_hi + 1, i_lo:i_hi + 1] if thb.ndim == 3
               else thb[:, None, None])


def _clip_nonnegative(window) -> None:
    """``max(x, 0)`` in place -- exact in FP32, deterministic, and a
    no-op on any value the convex operators produced."""
    xp = window.__class__.__module__.partition(".")[0]
    if xp == "cupy":
        import cupy as cp

        cp.maximum(window, cp.float32(0.0), out=window)
    else:
        np.maximum(window, np.float32(0.0), out=window)
_SIDES = (("west", "xs"), ("east", "xe"),
          ("south", "ys"), ("north", "ye"))
_GEOMETRY_NAMES = ("ci", "ip", "cj", "jp", "xig", "xjg")
MISMATCHED_MICROPHYSICS_FEEDBACK_BLOCKER = (
    "cross-scheme-feedback-reverse-mapping-unimplemented-v1"
)


def _state_attr(kind: str) -> str:
    """The ``DomainState`` attribute one nest field kind lives on."""
    return {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)


#: How far outside a child's parent-cell footprint the coupler reaches, in
#: PARENT cells.  ``register_nest``'s SINT stencil is +-2 and the
#: specified/relaxation zone the force tables fill is ``spec_bdy_width``
#: CHILD cells; add one for ``couple_nest_field``'s u/v face averages of
#: ``mup`` and 8 bounds all of it at every ratio this tree admits.  A
#: superset is free (the reader ignores what it does not need) while a
#: subset is a stale cell inside the zone the reader does use -- and stale
#: is the failure mode of this whole subsystem: finite, plausible and
#: silent.  Lived in ``gpuwm.core.model`` while the executor projected the
#: window on the coupler's behalf; it moved here when the coupler took
#: ownership of its own reads, so a relocation (which rewrites the child's
#: cfg this is computed from) moves the window with no second copy of the
#: geometry to forget.
NEST_FORCE_HALO_PARENT_CELLS = 8


def parent_footprint_window(dc) -> tuple:
    """The child's padded footprint on its parent, 0-based parent MASS cells.

    ``(j0, j1, i0, i1)``, unclamped: the negative edge of a child that sits
    against the parent's boundary is clamped by
    :func:`gpuwm.core.streaming.window_slices`, the one slice rule every
    consumer of a footprint window shares.
    """
    ratio = int(dc.parent_grid_ratio)
    i0 = int(dc.i_parent_start) - 1
    j0 = int(dc.j_parent_start) - 1
    span_i = -(-int(dc.run.nx) // ratio)
    span_j = -(-int(dc.run.ny) // ratio)
    pad = NEST_FORCE_HALO_PARENT_CELLS
    return (j0 - pad, j0 + span_j + pad, i0 - pad, i0 + span_i + pad)


#: What ``gpuwm.core.diagnostics.update_diagnostics`` reads that is not setup
#: geometry. Feedback refreshes the parent prognostics before restriction
#: and re-diagnoses only the rectangle written by restriction/smoothing.
_DIAGNOSTIC_INPUTS = ("mup", "thp", "php", "qv")

#: What that call WRITES.  A streamed parent has to carry them back or its
#: store keeps the pre-feedback diagnostics.
_DIAGNOSTIC_OUTPUTS = ("p", "al", "alt")


def _sync_in(state, attrs, window=None) -> int:
    """Pull ``attrs`` from a streamed domain's store; no-op when resident.

    ``window`` narrows the pull to a footprint window
    (:func:`parent_footprint_window`); the FORCE path uses it, because its
    reads are bounded and a streamed parent is large by definition.  The
    feedback commit refreshes its full prognostic write-back arrays so
    their untouched columns remain current; diagnostic write-back is
    restricted to the recomputed mass-grid rectangle.
    """
    from gpuwm.core.streaming import refresh_from_store

    return refresh_from_store(state, attrs, window=window)


def _sync_out(state, attrs, window=None) -> int:
    """Push ``attrs`` into a streamed domain's store; no-op when resident."""
    from gpuwm.core.streaming import commit_to_store

    return commit_to_store(state, attrs, window=window)


def _is_streamed(state) -> bool:
    from gpuwm.core.streaming import domain_store

    return domain_store(state) is not None


def _field_shape(state, kind: str) -> tuple[int, int, int]:
    if getattr(state, "_streamed_domain", None) is not None:
        from gpuwm.core.nest_operands import NestWindowSource
        field = NestWindowSource(state).array(_state_attr(kind))
        return ((1, *field.shape) if kind == "mu" else tuple(field.shape))
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

    def __init__(self, child_node, *, feedback: int = 0,
                 smooth_option: int = 0):
        if child_node.parent is None:
            raise ValueError("NestCoupler requires a child DomainNode")
        if child_node.cfg.parent_id != child_node.parent.cfg.grid_id:
            raise ValueError("child/parent configuration link is inconsistent")
        self.child_node = child_node
        if feedback not in (0, 1):
            raise ValueError("NestCoupler feedback must be 0 or 1")
        self.feedback = int(feedback)
        if smooth_option not in (0, 1, 2):
            raise ValueError(
                "NestCoupler smooth_option must be 0 (none), 1 (sm121) or "
                "2 (smdsm)")
        #: WRF's post-feedback parent smoother selection.  Held even at
        #: feedback = 0 -- ``SUBROUTINE smoother`` returns before reading
        #: it when feedback is off (interp_fcn.F:3823-3824), and this
        #: coupler's commit path does the same.
        self.smooth_option = int(smooth_option)
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
                f"identical parent/child vertical level counts (parent "
                f"nz={parent.run.nz}, child nz={child.run.nz}): the reverse "
                "feedback operator averages child cells onto parent cells "
                "with no vertical mapping, so a mismatched pair would feed "
                "the parent values from the wrong levels.  A per-domain "
                "vertical ladder is available on the OFFLINE downscale "
                "route (`gpuwm downscale --child-levels`), which is one-way "
                "by construction and takes no feedback.")
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
        #: H2D bytes the FORCE corridor has pulled through the store seam,
        #: cumulative.  A receipt, not a control: the honest cost of
        #: forcing a nest off a streamed parent is a number the run should
        #: be able to print, and the windowed corridor's whole claim is
        #: that this grows with the CHILD's footprint, not the parent.
        self.force_sync_bytes = 0
        self.feedback_sync_bytes = 0
        self.feedback_host_scratch_bytes = 0
        self.first_parent_ticks = None
        self.last_parent_ticks = None
        #: The parent's STEP COUNT at the first and last force.  Ticks
        #: alone cannot express "one force per parent step" once the step
        #: stops being a constant -- see the coverage receipt below.
        self.first_parent_step = None
        self.last_parent_step = None
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
            # WINDOWED, and this is the FORCE corridor's traffic bound:
            # ``bdy_interp1`` only ever reads the parent inside the child's
            # footprint plus the halo above, ``couple_nest_field`` is
            # pointwise plus one-cell face averages, and everything else it
            # touches is geography the transport never scatters.  So the
            # pull is two windows, not two fields -- O(child footprint) per
            # kind per parent step where it was O(parent).  The coupled
            # values OUTSIDE the window are computed from whatever the
            # state held (attach-time air); they are never read, because
            # the donor maps cannot reach past the stencil the halo
            # bounds.  ``tilestream/test_nest_executor.py`` holds the
            # bitwise proof and the halo-starved negative control.
            self.force_sync_bytes += _sync_in(
                parent_state, ("mup", _state_attr(kind)),
                window=parent_footprint_window(self.child_node.cfg))
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
            self.child_node.parent.clock.step_ticks)   # LIVE, not configured
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
            # THE SAME INVARIANT, STATED IN STEPS.  The tick arithmetic
            # above -- (final - start) % interval, last - first ==
            # (count-1)*interval -- says "one force per parent step" only
            # while every parent step is the same size.  Under an adaptive
            # clock it says nothing: a 30-minute run whose step grew
            # 30 -> 55 s reported 50 expected forces against 36 actually
            # taken, and 36 was CORRECT -- it is exactly the number of
            # steps the parent took.
            #
            # Step counts express the invariant directly and hold under
            # both clocks, so they are published alongside rather than
            # replacing the tick form; the consumer accepts either.
            "first_parent_step": self.first_parent_step,
            "last_parent_step": self.last_parent_step,
            "parent_step_count": int(self.child_node.parent.clock.step_count),
            "force_count_matches_parent_steps": bool(
                (self.force_count == 0
                 and self.first_parent_step is None)
                or (self.force_count > 0
                    and self.last_parent_step is not None
                    and self.first_parent_step is not None
                    and self.force_count
                    == self.last_parent_step - self.first_parent_step + 1)),
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

    def _coupled_child_field(self, kind: str, *, frame: bool = False):
        """Overwrite and return F16's one full-child arena prefix.

        ``bdy_interp1`` validates both full extents.  The child copy therefore
        uses the audited ``nest_child_field`` slot.  Preflight aliases that
        logical slot to an RK backing distinct from ``nest_parent_field``
        whenever two dead backings have sufficient capacity, and otherwise
        retains an explicit correctness-first backing.  The distinction is
        required because parent and child copies are simultaneously live in
        ``bdy_interp1``.  This also supports valid skinny grids where U or V
        is larger than W.

        ``frame=True`` is the FORCE path on a STREAMED child, and it is the
        child-side mirror of the parent corridor's footprint window:
        ``bdy_interp1`` reads the child only inside its boundary zone, so
        the store pull is four frame strips
        (:func:`gpuwm.core.nest_stream.child_frame_windows`), not a domain
        -- O(perimeter) per kind per parent step where a streamed child is
        large by definition.  The coupled values OUTSIDE the strips are
        computed from whatever the state held and are never read.  The
        FEEDBACK path deliberately does not window: the restriction reads
        the whole child interior, so its whole-field pull is the honest
        cost of two-way feedback, not a miss.
        """
        state = self.child_node.state
        if frame and _is_streamed(state):
            from gpuwm.core.nest_stream import child_frame_windows

            for window in child_frame_windows(self.child_node.cfg.run):
                self.force_sync_bytes += _sync_in(
                    state, ("mup", _state_attr(kind)), window=window)
        else:
            _sync_in(state, ("mup", _state_attr(kind)))
        shape = _field_shape(state, kind)
        backing = self._scratch("nest_child_field")
        count = math.prod(shape)
        if count > backing.size:
            raise RuntimeError("child field exceeds F16 arena capacity")
        out = backing.reshape(-1)[:count].reshape(shape)
        couple_nest_field(state, kind, out=out)
        return out

    def _raw_child_field(self, kind: str):
        """The child prognostic WRF's feedback restricts, UNCOUPLED.

        ``med_feedback_domain`` hands ``copy_fcn`` the raw ``ngrid%u_2`` /
        ``t_2`` / ``ph_2`` / ``moist`` arrays
        (inc/nest_feedbackup_interp.inc:23-27 and the blocks after it), and
        the whole transaction -- ``share/mediation_feedback_domain.F``
        read end to end, plus ``feedback_domain_em_part1/part2`` -- contains
        no ``couple_or_uncouple_em``.  Only the FORCE path couples
        (share/mediation_force_domain.F:117/:129) and uncouples again
        (:184/:196).  So there is nothing to build here for any kind
        except theta, whose WRF spelling is ``theta - 300`` where gpuwm
        stores ``theta - thb``; that one is materialized in the audited
        ``nest_child_field`` slot, exactly as ``couple_nest_field``'s
        LBC_THETA branch spells it (kernels/lbc_state.cu:598-601).
        """
        state = self.child_node.state
        _sync_in(state, ("mup", _state_attr(kind)))
        if kind != "t":
            return getattr(state, _state_attr(kind))
        shape = _field_shape(state, kind)
        backing = self._scratch("nest_child_field")
        count = math.prod(shape)
        if count > backing.size:
            raise RuntimeError("child field exceeds F16 arena capacity")
        out = backing.reshape(-1)[:count].reshape(shape)
        thb = state.thb
        out[...] = state.thp
        out += thb if thb.ndim == 3 else thb[:, None, None]
        out -= _WRF_T0
        return out

    def _rolling_out(self, kind: str):
        result = {}
        for side, suffix in _SIDES:
            result[side] = (
                self._scratch(f"nest_{kind}_b{suffix}"),
                self._scratch(f"nest_{kind}_bt{suffix}"),
            )
        return result

    def _force_windowed(self, kind, out, parent_source, child_source):
        """Build rolling strips from bounded canonical operands."""
        from gpuwm.core.nest_interp import bdy_width, window_registration
        from gpuwm.core.nest_operands import boundary_windows, streamed_chunk_shape

        node = self.child_node
        reg = self.registrations[_STAGGER.get(kind, "m")]
        run = node.cfg.run
        width = bdy_width(run.spec_zone, run.relax_zone, run.spec_bdy_width)
        mapped_parent = None
        if transition_handles_field(self.microphysics_transition, kind):
            # The existing mapping inventory guard still refuses a streamed
            # mixed-scheme parent. Resident parents keep their ratified mapper.
            mapped_parent = self._coupled_parent_field(kind)
        for side, window, destination in boundary_windows(
                reg, width, streamed_chunk_shape(node.state)):
            cropped, donor = window_registration(reg, window)
            if mapped_parent is None:
                parent_field = parent_source.coupled(kind, donor)
            else:
                import cupy as cp
                parent_field = cp.ascontiguousarray(mapped_parent[(...,) + donor])
            child_field = child_source.coupled(kind, window)
            tables = bdy_interp1(
                parent_field, child_field, cropped,
                parent_dt_fp32=node.parent.clock.dt_fp32,
                parent_interval_ticks=node.parent.clock.step_ticks,
                spec_zone=run.spec_zone, relax_zone=run.relax_zone,
                spec_bdy_width=run.spec_bdy_width, sides=(side,))
            for target, value in zip(out[side], tables[side]):
                target[destination] = value
            del tables, parent_field, child_field, cropped

    def _restrict_windowed(self, kind, source, parent_source):
        """Restrict canonical child chunks directly into their parent owner."""
        import cupy as cp
        from gpuwm.core.nest_interp import feedback_child_window
        from gpuwm.core.nest_operands import streamed_chunk_shape

        node = self.child_node
        reg = self.registrations[_STAGGER.get(kind, "m")]
        run = node.cfg.run
        attr = _state_attr(kind)
        field = parent_source.array(attr)
        ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=run.spec_zone)
        sy, sx = streamed_chunk_shape(node.state)
        # A parent cell reads at most one ratio-sized child footprint.
        py, px = max(1, sy // reg.nrj), max(1, sx // reg.nri)
        for j in range(jlo, jhi+1, py):
            for i in range(ilo, ihi+1, px):
                window = (slice(j, min(j+py, jhi+1)),
                          slice(i, min(i+px, ihi+1)))
                donor = feedback_child_window(reg, window, spec_zone=run.spec_zone)
                child = source.raw(kind, donor)
                target = field[(...,) + window]
                result = cp.empty(target.shape, dtype=cp.float32)
                copy_fcn(result, child, reg, spec_zone=run.spec_zone,
                         parent_window=window, child_window=donor)
                if kind == "t":
                    result += _WRF_T0
                    thb = parent_source.device("thb", window)
                    result -= thb if thb.ndim == 3 else thb[:, None, None]
                parent_source.write(attr, window, result)

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
        parent_interval_ticks = int(parent.clock.step_ticks)   # LIVE
        if lead != parent_interval_ticks:
            raise RuntimeError(
                f"parent must lead child by one parent interval before FORCE; "
                f"lead={lead}, interval={parent_interval_ticks}")

        # Each endpoint refreshes its own store corridor before the same
        # coupled interpolation. The child's buffers consume packed rolling
        # table windows, regardless of the parent's storage representation.

        self._bind_geometry()
        from gpuwm.core.cam_ozone import transfer_parent_ozone
        # Mass-point interp/bdy registrations have identical donor geometry
        # for every ratio; reuse the existing manifest-backed device tables.
        self.force_sync_bytes += transfer_parent_ozone(node, self.registrations["m"])
        fields = {}
        run = node.cfg.run
        bounded_child = getattr(node.state, "_streamed_domain", None) is not None
        if bounded_child:
            from gpuwm.core.nest_operands import NestWindowSource
            parent_source = NestWindowSource(parent.state)
            child_source = NestWindowSource(node.state)
        for kind in nest_field_kinds(run):
            if bounded_child:
                out = self._rolling_out(kind)
                self._force_windowed(kind, out, parent_source, child_source)
                fields[_APPLICATION_NAME.get(kind, kind)] = out
                continue
            parent_field = self._coupled_parent_field(kind)
            child_field = self._coupled_child_field(kind, frame=True)
            stagger = _STAGGER.get(kind, "m")
            out = self._rolling_out(kind)
            bdy_interp1(
                parent_field, child_field, self.registrations[stagger],
                parent_dt_fp32=parent.clock.dt_fp32,   # LIVE
                parent_interval_ticks=parent_interval_ticks,
                spec_zone=run.spec_zone, relax_zone=run.relax_zone,
                spec_bdy_width=run.spec_bdy_width, out=out)
            fields[_APPLICATION_NAME.get(kind, kind)] = out

        if bounded_child:
            self.force_sync_bytes += (parent_source.host_to_device_bytes
                                      + child_source.host_to_device_bytes)

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
        parent_step = int(parent.clock.step_count)
        if self.first_parent_ticks is None:
            self.first_parent_ticks = parent_ticks
            self.first_parent_step = parent_step
        self.last_parent_ticks = parent_ticks
        self.last_parent_step = parent_step
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
        # every field ``copy_fcn`` WRITES on the parent side -- it fills
        # only the feedback rectangle, so the cells outside it have to be
        # the store's own -- plus the four inputs ``feedback_finalize``'s
        # whole-parent update_diagnostics consumes.  Pushed back out below:
        # everything written.  This is O(the fields the transaction
        # touches) per parent step, which is the honest cost of a
        # whole-domain finalize; see the module docstring.
        streamed_parent = _is_streamed(parent.state)
        bounded_child = getattr(node.state, "_streamed_domain", None) is not None
        canonical_parent = bounded_child and getattr(
            parent.state, "_streamed_domain", None) is not None
        written = ["mup"]
        if streamed_parent and not canonical_parent:
            _sync_in(parent.state,
                     tuple(dict.fromkeys(
                         _DIAGNOSTIC_INPUTS
                         + tuple(_state_attr(k) for k in payload["kinds"]))))

        # WRF hands ``copy_fcn`` the parent's own prognostic as the CD field
        # (inc/nest_feedbackup_interp.inc), so each restriction is written
        # straight into the exact parent overlap and nothing else is
        # touched.  MU is no different from the rest; it leads only because
        # the smoother and finalize below read the updated mass.
        if bounded_child:
            from gpuwm.core.nest_operands import NestWindowSource
            child_source = NestWindowSource(node.state)
            parent_source = NestWindowSource(parent.state)
            self._restrict_windowed("mu", child_source, parent_source)
        else:
            child_mu = self._raw_child_field("mu")
            copy_fcn(
                parent.state.mup[None], child_mu, self.registrations["m"],
                spec_zone=run.spec_zone)

        for kind in payload["kinds"]:
            if kind == "mu":
                continue
            stagger = _STAGGER.get(kind, "m")
            reg = self.registrations[stagger]
            if bounded_child:
                self._restrict_windowed(kind, child_source, parent_source)
            else:
                child_field = self._raw_child_field(kind)
                copy_fcn(
                    getattr(parent.state, _state_attr(kind)), child_field,
                    reg, spec_zone=run.spec_zone)
            if kind == "t" and not bounded_child:
                _rebase_feedback_theta(parent.state, reg, run.spec_zone)
            written.append(_state_attr(kind))

        # The parent smoother, LAST -- feedback_domain_em_part2.F:176-193
        # runs nest_feedbackup_smooth.inc after the unpack, over every
        # fed-back field (Registry flag `s` rides with `u` on all of them:
        # Registry.EM_COMMON:159/172/183/199/211/288/454ff).  The
        # ``nest_parent_field`` slot is unused by the restriction, which
        # writes straight into the parent, so the smoother's scratch is
        # that same audited allocation and this adds no memory.  Runs
        # before the streamed push-back below so a streamed parent's store
        # receives the smoothed field, not the raw restriction.
        if self.smooth_option != 0:
            from gpuwm.core.nest_interp import smoother_parent_window

            for kind in payload["kinds"]:
                reg = self.registrations[_STAGGER.get(kind, "m")]
                if canonical_parent:
                    from gpuwm.core.nest_operands import (
                        smooth_canonical_parent, streamed_chunk_shape)
                    field = parent_source.array(_state_attr(kind))
                    smooth_canonical_parent(
                        parent_source, kind, reg, smooth_option=self.smooth_option,
                        chunk_shape=streamed_chunk_shape(parent.state))
                else:
                    field = getattr(parent.state, _state_attr(kind))
                    smoother(
                        field, reg, smooth_option=self.smooth_option,
                        scratch=self._scratch("nest_parent_field"))
                # smdsm's de-smoothing pass is ANTI-diffusive (xnu =
                # -0.52, interp_fcn.F:3976) and can undershoot a sharp
                # gradient; on a positive-definite species that is a
                # small negative -- MEASURED, first d02->d01 feedback of
                # the Melissa two-way run: qv = -2.2e-07 inside the
                # rectangle, and the health gate rightly refused it.
                # WRF tolerates the transient because its microphysics
                # clamps moisture at CONSUMPTION (e.g. Thompson's
                # MAX(1.e-10, qv)); ArWen's gate asserts the bound at
                # the period boundary, so the transaction re-establishes
                # it here, over exactly the cells the smoother wrote.
                # The restriction itself is a convex average and sm121's
                # weights are convex, so only the smdsm path can trip
                # this; the clamp is a provable no-op everywhere else.
                if kind not in _SIGNED_KINDS:
                    i0, j0, niw, njw = smoother_parent_window(reg)
                    if niw > 0 and njw > 0:
                        window = field[..., j0:j0 + njw, i0:i0 + niw]
                        _clip_nonnegative(window)
        if streamed_parent and not canonical_parent:
            _sync_out(parent.state, tuple(dict.fromkeys(written)))
        if bounded_child:
            self.feedback_sync_bytes += (
                child_source.host_to_device_bytes + parent_source.host_to_device_bytes
                + parent_source.device_to_host_bytes)
            self.feedback_host_scratch_bytes = max(
                self.feedback_host_scratch_bytes, parent_source.max_host_scratch_bytes)
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
        from gpuwm.core.nest_interp import (feedback_parent_bounds,
                                            smoother_parent_window)

        parent = node.parent
        # WINDOWED to the columns the transaction touched.  WRF re-runs
        # start_domain on the parent after med_nest_feedback
        # (share/mediation_integrate.F:787-838), and this call used to
        # mirror that as a whole-parent re-diagnosis -- but calc_p_alpha
        # is column-local and its inputs (thp, php, mup, qv) changed only
        # inside the mass-frame restriction rectangle plus the smoother's
        # window, so the windowed call is BITWISE the whole-parent call:
        # equal on the window by identical arithmetic, equal off it
        # because unchanged inputs reproduce the columns' standing
        # values.  The union below is exact for any spec_zone; on this
        # tree it cuts the re-diagnosis to ~10% of the parent's columns,
        # once per parent step per nest.  ``feedback_commit`` has already
        # made the inputs correct everywhere on a streamed parent, so
        # this reads the domain rather than the attach-time snapshot;
        # the three outputs go back to the store.
        run = node.cfg.run
        reg_m = self.registrations["m"]
        ci_lo, ci_hi, cj_lo, cj_hi = feedback_parent_bounds(
            reg_m, spec_zone=run.spec_zone)
        if self.smooth_option != 0:
            i0, j0, niw, njw = smoother_parent_window(reg_m)
            if niw > 0 and njw > 0:
                ci_lo = min(ci_lo, i0)
                ci_hi = max(ci_hi, i0 + niw - 1)
                cj_lo = min(cj_lo, j0)
                cj_hi = max(cj_hi, j0 + njw - 1)
        window = (cj_lo, ci_lo, cj_hi - cj_lo + 1, ci_hi - ci_lo + 1)
        if (getattr(node.state, "_streamed_domain", None) is not None
                and getattr(parent.state, "_streamed_domain", None) is not None):
            from gpuwm.core.nest_operands import (
                NestWindowSource, diagnose_canonical_parent, streamed_chunk_shape)
            source = NestWindowSource(parent.state)
            diagnose_canonical_parent(
                source, (slice(cj_lo, cj_hi+1), slice(ci_lo, ci_hi+1)),
                hypsometric_opt=parent.cfg.run.hypsometric_opt,
                chunk_shape=streamed_chunk_shape(parent.state))
            self.feedback_sync_bytes += (source.host_to_device_bytes
                                         + source.device_to_host_bytes)
            self._prepared_feedback = None
            return
        update_diagnostics(
            parent.state, parent.cfg.run.hypsometric_opt, window=window)
        # The resident view outside this rectangle may still contain
        # attach-time diagnostics. Only the recomputed columns belong to
        # this transaction; preserve the live store everywhere else.
        # Diagnostics takes (j0, i0, nj, ni); the store seam takes
        # inclusive (j0, j1, i0, i1). These are mass-grid outputs.
        _sync_out(parent.state, _DIAGNOSTIC_OUTPUTS,
                  window=(cj_lo, cj_hi, ci_lo, ci_hi))
        self._prepared_feedback = None


__all__ = [
    "MISMATCHED_MICROPHYSICS_FEEDBACK_BLOCKER", "NestCoupler",
]
