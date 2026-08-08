"""Real-data child rebuild at a new placement (statics-on-move, leg 3).

The 2026-08-06 requirement (Drew, via the WRF moving-nest discussion): a
relocated nest's STATIC fields -- terrain, landuse, soil categories --
must be REBUILT for the new footprint from the nest's own static source
at nest resolution, never inherited parent-interpolated, because
over-land storm-following at 2 km lives on resolved terrain.  This
module is the initializer that satisfies it on the real-data routes, and
it is deliberately assembled from the SAME machinery the t=0 nest cold
start uses (:mod:`gpuwm.ingest.nest_init`) rather than a re-derivation:

1. **Footprint-rebuilt statics.**  :func:`gpuwm.static.build.
   build_static_for_domain` is footprint-parametric and is invoked for
   the new footprint (plus the ``[static.highres]`` overlay when the
   case enables it, whose tile cache makes repeat moves cheap).  The
   footprint's grid comes from :meth:`ProjectedGrid.translated` -- the
   reference grid moved by an exact whole number of its own cells -- so
   a cell two placements share evaluates to bitwise-identical
   coordinates and therefore bitwise-identical statics (identical
   source + identical cells = identical bytes).  That equality is what
   lets the bitwise overlap transplant survive a statics rebuild, and
   the route's preparer ASSERTS it on every move rather than assuming
   it.

2. **Fresh-strip atmosphere on the new terrain.**  The atmosphere at
   move time exists only in the live parent, so the child is filled by
   the ordinary full-parent SINT (:func:`~gpuwm.ingest.nest_init.
   parent_only_init`) and then adjusted to the rebuilt fine terrain by
   the SAME sequence the t=0 real child runs after its own fine-terrain
   analysis: the analytic fine base (``_make_real_base``), WRF's
   three-operand terrain blend (``blend_terrain`` on ht/mub/phb against
   the parent SINT captures), ``adjust_tempqv`` for the base-column-mass
   change, the ``start_domain`` base/EOS re-derivation, and the
   real-nest ``press_adj`` MU correction -- one call,
   ``_adjust_and_rederive``, reused verbatim.  On overlap cells outside
   both placements' blend frames this reproduces the t=0 base bitwise
   (the analytic base is a per-column pure function of the terrain
   bytes), which is what keeps the donor-alignment instrument armed.

3. **Blend frames are rebased, not resplit silently.**  Inside the two
   placements' blend frames the child base states genuinely differ, so
   the bitwise transplant of perturbation fields there would silently
   change totals.  :func:`rebase_transplanted_perturbations` (installed
   as the initializer's ``post_transplant`` hook) preserves the
   outgoing child's TOTALS on exactly the cells whose base bytes
   differ, and leaves every other cell's bitwise stamp untouched.

4. **Land state moves by donor fill.**  Soil, snow and surface fields
   live on the physics driver, outside the serialised-state transplant;
   leg 1's contract moves them with a second transplant against the
   same plan, and the newly exposed strip is filled from its nearest
   same-landmask-class donor inside the overlap
   (:func:`donor_fill_plan`), with the counts recorded on the move
   receipt.  The route-side driver rebuild lives in
   :class:`gpuwm.runtime.RealRelocationChildPreparer`.

Nothing here decides when or where to move, and a restart across a move
still promises nothing (Drew, 2026-08-06).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from gpuwm.core.nest_relocation import RelocationRefusal

#: The receipt-stated provenance for statics rebuilt by this initializer.
REAL_DATA_FOOTPRINT_REBUILT_STATICS = (
    "footprint-rebuilt own-source statics: build_static_for_domain on the "
    "placement-translated child grid (30s baseline, [static.highres] "
    "overlay when the case enables it); terrain adjustment via the t=0 "
    "nest cold-start sequence (blend_terrain + adjust_tempqv + "
    "start_domain re-derivation + press_adj)")

#: What fills the strip a move exposes under this initializer.
REAL_DATA_STRIP_FILL_SOURCE = (
    "full-parent SINT adjusted to footprint-rebuilt fine terrain "
    "(t=0 nest cold-start lineage); overlap then stamped bitwise from "
    "the outgoing child, blend-frame perturbations rebased to preserve "
    "totals; land-surface state donor-filled per leg-1's contract")

#: Driver-held land-surface continuation state a move carries: the Noah
#: prognostics, snow state, and the seeded surface diagnostics.  Fields a
#: configuration does not allocate are skipped by name, exactly like the
#: serialized-state transplant.  Accumulators (RAINC/RAINNC lineage) are
#: deliberately NOT here: they are re-initialised at the new placement --
#: leg 1's documented contract -- and the receipt says so.
LAND_SURFACE_CONTINUATION_FIELDS = (
    "tsk", "tslb", "smois", "sh2o", "smcrel",
    "canwat", "snow", "snowh", "snowc", "snotime",
    "xice", "qsfc", "ust",
    "psfc", "t2", "q2", "th2", "u10", "v10",
)

#: Statics the overlap-equality assertion refuses on.  Everything the
#: model or driver consumes; compared bitwise on shared ground.
OVERLAP_STATIC_EQUALITY_FIELDS = (
    "HGT_M", "LANDMASK", "LU_INDEX", "LANDUSEF",
    "SOILCTOP", "SCT_DOM", "SOILCBOT", "SCB_DOM",
    "GREENFRAC", "LAI12M", "ALBEDO12M", "SNOALB", "SOILTEMP", "TMN",
)

#: (perturbation, base) pairs whose split changes when the base changes.
#: Everything else the transplant carries is a physical total.
_REBASE_PAIRS = (("thp", "thb"), ("php", "phb"), ("mup", "mub2d"))


def _host(value, dtype=None) -> np.ndarray:
    if hasattr(value, "get") and hasattr(value, "__cuda_array_interface__"):
        value = value.get()
    array = np.asarray(value)
    return array if dtype is None else np.asarray(array, dtype=dtype)


def _write_window(target, window, values) -> None:
    """Assign host ``values`` into ``target[..., dst_j, dst_i]``."""
    (dst_j, _src_j), (dst_i, _src_i) = window
    if hasattr(target, "__cuda_array_interface__"):
        import cupy as cp

        values = cp.asarray(np.ascontiguousarray(values))
    target[..., dst_j, dst_i] = values


# ---------------------------------------------------------------------------
# Overlap statics equality: asserted, never assumed
# ---------------------------------------------------------------------------

def overlap_statics_mismatches(old_fields, new_fields, plan,
                               names=OVERLAP_STATIC_EQUALITY_FIELDS
                               ) -> dict[str, object]:
    """Count per-field bitwise mismatches on the shared ground.

    ``old_fields`` are the outgoing footprint's statics, ``new_fields``
    the rebuilt ones; ``plan`` supplies the index-space windows
    (``new[dst] <-> old[src]``).  Identical source + identical cells
    must give identical bytes; any nonzero count is a defect of the
    statics build, not an input error, and the caller refuses on it.
    """
    fields: dict[str, int] = {}
    compared_cells = 0
    for name in names:
        old = old_fields.get(name)
        new = new_fields.get(name)
        if old is None or new is None:
            continue
        old = np.asarray(old)
        new = np.asarray(new)
        if old.shape != new.shape:
            fields[name] = max(int(old.size), int(new.size), 1)
            continue
        window = plan.window(old.shape)
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        actual = np.ascontiguousarray(new[..., dst_j, dst_i])
        expected = np.ascontiguousarray(old[..., src_j, src_i])
        if actual.dtype.itemsize == 8 and actual.dtype.kind == "f":
            mismatches = int(np.count_nonzero(
                actual.view(np.uint64) != expected.view(np.uint64)))
        elif actual.dtype.kind == "f":
            mismatches = int(np.count_nonzero(
                actual.view(np.uint32) != expected.view(np.uint32)))
        else:
            mismatches = int(np.count_nonzero(actual != expected))
        fields[name] = mismatches
        compared_cells += int(actual.size)
    return {
        "fields": fields,
        "compared_cells": int(compared_cells),
        "mismatched_fields": {name: count for name, count in fields.items()
                              if count},
        "pass": bool(compared_cells) and not any(fields.values()),
    }


# ---------------------------------------------------------------------------
# Donor fill: land-surface state for the strip
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DonorFillPlan:
    """Nearest-donor assignment for every strip cell, with provenance.

    ``donor_flat`` maps each cell of the new footprint to the flat index
    of the overlap cell that donates its land-surface state; overlap
    cells map to themselves.  Donors are exact-Euclidean nearest within
    the overlap, matched on the NEW footprint's landmask class (the
    donor's class equals the outgoing child's on shared ground by the
    overlap-statics equality this move already asserted); a strip cell
    whose class has no donor anywhere in the overlap falls back to the
    nearest overlap cell of any class, and that fallback is COUNTED,
    never silent.
    """

    donor_flat: np.ndarray
    counts: dict[str, int]

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Fill one field's strip cells from their donors (trailing y, x)."""
        values = np.asarray(values)
        ny, nx = self.donor_flat.shape
        if values.shape[-2:] != (ny, nx):
            raise ValueError(
                f"donor fill built for (ny, nx) = {(ny, nx)}, got a field "
                f"with trailing shape {values.shape[-2:]}")
        flat = values.reshape(values.shape[:-2] + (ny * nx,))
        return flat[..., self.donor_flat.ravel()].reshape(values.shape)


def _nearest_donor(target_yx: np.ndarray, donor_yx: np.ndarray) -> np.ndarray:
    """Exact nearest donor per target (first index breaks ties)."""
    out = np.empty(target_yx.shape[0], dtype=np.int64)
    chunk = max(1, int(4e6) // max(1, donor_yx.shape[0]))
    for start in range(0, target_yx.shape[0], chunk):
        stop = min(start + chunk, target_yx.shape[0])
        block = target_yx[start:stop]
        d2 = ((block[:, None, 0] - donor_yx[None, :, 0]) ** 2
              + (block[:, None, 1] - donor_yx[None, :, 1]) ** 2)
        out[start:stop] = np.argmin(d2, axis=1)
    return out


def donor_fill_plan(*, overlap_mask: np.ndarray,
                    landmask: np.ndarray) -> DonorFillPlan:
    """Build the strip's donor assignment from the overlap and landmask."""
    overlap_mask = np.asarray(overlap_mask, dtype=bool)
    land = np.asarray(landmask) >= 0.5
    if overlap_mask.shape != land.shape:
        raise ValueError(
            f"overlap mask {overlap_mask.shape} and landmask "
            f"{land.shape} disagree")
    ny, nx = overlap_mask.shape
    if not overlap_mask.any():
        raise RelocationRefusal(
            "donor fill requires a non-empty overlap; a disjoint move is "
            "refused upstream and cannot reach here")
    donor_flat = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    counts = {
        "strip_cells": int(np.count_nonzero(~overlap_mask)),
        "overlap_cells": int(np.count_nonzero(overlap_mask)),
        "land_donor_filled": 0,
        "water_donor_filled": 0,
        "class_fallback_filled": 0,
    }
    yy, xx = np.mgrid[0:ny, 0:nx]
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    flat_overlap = overlap_mask.ravel()
    flat_land = land.ravel()
    unresolved = np.zeros(ny * nx, dtype=bool)
    for is_land, label in ((True, "land_donor_filled"),
                           (False, "water_donor_filled")):
        targets = ~flat_overlap & (flat_land == is_land)
        if not targets.any():
            continue
        donors = np.flatnonzero(flat_overlap & (flat_land == is_land))
        if donors.size == 0:
            unresolved |= targets
            continue
        picked = donors[_nearest_donor(coords[targets], coords[donors])]
        donor_flat.ravel()[np.flatnonzero(targets)] = picked
        counts[label] = int(np.count_nonzero(targets))
    if unresolved.any():
        donors = np.flatnonzero(flat_overlap)
        picked = donors[_nearest_donor(coords[unresolved], coords[donors])]
        donor_flat.ravel()[np.flatnonzero(unresolved)] = picked
        counts["class_fallback_filled"] = int(np.count_nonzero(unresolved))
    return DonorFillPlan(donor_flat=donor_flat, counts=counts)


def overlap_mask_for_plan(plan, shape) -> np.ndarray:
    """Boolean (ny, nx) mask of the new footprint's overlap cells."""
    mask = np.zeros(tuple(int(n) for n in shape), dtype=bool)
    window = plan.window(mask.shape)
    if window is not None:
        (dst_j, _sj), (dst_i, _si) = window
        mask[dst_j, dst_i] = True
    return mask


# ---------------------------------------------------------------------------
# Blend-frame rebase: totals survive where the base bytes differ
# ---------------------------------------------------------------------------

def rebase_transplanted_perturbations(*, source_state, target_state,
                                      plan, cfg) -> dict[str, object]:
    """Preserve totals where the outgoing and incoming bases differ.

    The transplant stamps perturbation fields bitwise; against an equal
    base that preserves the total, and the doubly-interior overlap has
    equal bases by construction.  Inside either placement's blend frame
    the bases differ, so on exactly the cells whose base bytes differ
    (a per-cell bitwise comparison, not a geometric guess) the stamped
    perturbation is corrected to ``(pert_out + base_out) - base_in`` in
    the field's own float32 arithmetic -- the same split arithmetic the
    t=0 path uses to set ``thp`` -- and the EOS diagnostics are then
    re-derived so ``p``/``al``/``alt`` agree with the corrected split.
    On every cell whose base bytes are equal the bitwise stamp is left
    untouched, so the certified bulk behaviour is preserved exactly.
    """
    from gpuwm.core.diagnostics import update_diagnostics

    rebased: dict[str, int] = {}
    for pert_name, base_name in _REBASE_PAIRS:
        pert_target = getattr(target_state, pert_name, None)
        base_target = getattr(target_state, base_name, None)
        pert_source = getattr(source_state, pert_name, None)
        base_source = getattr(source_state, base_name, None)
        if any(value is None for value in (pert_target, base_target,
                                           pert_source, base_source)):
            continue
        window = plan.window(np.shape(pert_target))
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        base_out = _host(base_source, np.float32)[..., src_j, src_i]
        base_in = _host(base_target, np.float32)[..., dst_j, dst_i]
        differs = base_out.view(np.uint32) != base_in.view(np.uint32)
        count = int(np.count_nonzero(differs))
        rebased[pert_name] = count
        if not count:
            continue
        pert = np.array(_host(pert_target, np.float32)[..., dst_j, dst_i],
                        copy=True)
        pert[differs] = ((pert[differs] + base_out[differs])
                         - base_in[differs])
        _write_window(pert_target, window, pert)
    update_diagnostics(target_state, cfg.hypsometric_opt)
    return {
        "pairs": {pert: base for pert, base in _REBASE_PAIRS},
        "rebased_cells": rebased,
        "diagnostics_rederived": True,
    }


def clamp_interpolation_undershoot(state) -> dict[str, int]:
    """Zero-floor the interpolation-filled nonnegative fields, counted.

    The fresh strip is filled by SINT of the live parent, and SINT's
    stencil rings: near a sharp hydrometeor edge it can undershoot a
    nonnegative field by an epsilon-scale negative (first seen on the
    first real-data move with active microphysics -- one Morrison rain-
    number cell at -1.8e-15 -- which the idealized proofs could never
    expose because their hydrometeors were identically zero at move
    time).  A rebuilt child must satisfy the state's own health
    invariants BEFORE its first step, and WRF's own microphysics drivers
    clamp these fields internally anyway, so the defined behaviour is an
    exact zero floor on exactly the fields whose shared health rule says
    ``>= 0`` (non-strict): the authority is
    :func:`gpuwm.core.health.rule_for_field`, never a private list, so
    a field added to the model reaches both the gate and this floor at
    once.  Strict-lower fields (coupled mass, total specific volume) are
    deliberately untouched -- zero is illegal for them and a violation
    there must FAIL the gate, not be papered over.  Counts are returned
    for the receipt, never silent.
    """
    from gpuwm.core.health import rule_for_field
    from gpuwm.core.nest_relocation import relocatable_attrs

    clamped: dict[str, int] = {}
    for name in relocatable_attrs():
        value = getattr(state, name, None)
        if value is None:
            continue
        rule = rule_for_field(name)
        if rule.lower != 0.0 or rule.strict_lower:
            continue
        negative = value < 0
        count = int(negative.sum())
        if count:
            value[negative] = 0
            clamped[name] = count
    return clamped


# ---------------------------------------------------------------------------
# The initializer factory
# ---------------------------------------------------------------------------

def _build_footprint_statics(grid, catalog, child_dc):
    """The t=0 static build, verbatim inputs, on the supplied grid."""
    from gpuwm.ingest.nest_init import _static_catalog
    from gpuwm.static.build import (build_static_for_domain,
                                    geog_selection_from_catalog)

    static_catalog = _static_catalog(catalog)
    fields = build_static_for_domain(grid, static_catalog, child_dc.grid_id)
    selection = geog_selection_from_catalog(static_catalog, child_dc.grid_id)
    landuse_attrs = selection.landuse_global_attrs()
    highres = getattr(catalog, "static_highres", None)
    highres_applied = False
    if highres is not None and getattr(highres, "enabled", False):
        from gpuwm.static.highres_production import apply_highres_statics

        # Same case_date input as the t=0 build: statics are a
        # time-invariant property of the domain, and a drifting date here
        # would break the overlap equality against the t=0 footprint.
        fields, _ = apply_highres_statics(
            fields, grid, config=highres, domain_id=child_dc.grid_id,
            case_date=child_dc.start_time.date(),
            landuse_attrs=landuse_attrs)
        highres_applied = True
    return fields, selection, landuse_attrs, highres_applied


def real_relocation_initializer(*, catalog=None, vertical, child_config,
                                reference_grid, reference_i_parent_start,
                                reference_j_parent_start,
                                drift_tolerance_deg: float = 1.0e-8,
                                statics_builder=None):
    """Build the per-footprint rebuild seam for one real-data child.

    ``reference_grid`` is the child's grid as originally placed
    (``reference_i/j_parent_start``); every relocated footprint is that
    grid translated by an exact whole number of child cells, which is
    the construction that makes footprint-rebuilt statics bitwise-equal
    on shared ground (see the module docstring).  The returned
    ``initialize(new_dc, parent_node, ...)`` satisfies
    :func:`gpuwm.core.nest_relocation.relocate_child`'s initializer
    contract, states its provenance, scopes the donor-alignment
    instrument to the region where its invariant is defined
    (``donor_alignment_frame_width``), and installs the blend-frame
    rebase as its ``post_transplant`` hook.

    WHERE THE STATICS COME FROM is the one route-owned seam.  The
    case-data route holds its input ``catalog`` (the GEOG source) and
    rebuilds per footprint through ``build_static_for_domain``; the
    prepared routes hold no ingest inputs and pass ``statics_builder``
    instead -- a callable ``(grid, new_dc) -> fields`` (the sealed
    statics-corridor crop, :func:`gpuwm.static.corridor
    .corridor_footprint_statics_builder`) carrying ``static_provenance``
    and ``source_label`` attributes for the receipt.  Everything after
    the statics -- the SINT fill, the t=0 terrain adjustment, the
    rebase hook -- is one implementation, never forked per route.
    """
    if not hasattr(child_config, "run") or not hasattr(child_config,
                                                       "grid_id"):
        raise TypeError(
            "child_config must be the child's DomainConfig (an object "
            "with grid_id, parent_grid_ratio and a run config)")
    if (catalog is None) == (statics_builder is None):
        raise TypeError(
            "real_relocation_initializer takes exactly one statics "
            "source: the case-data route's input catalog OR a "
            "statics_builder (the prepared routes' sealed-corridor "
            "crop); a rebuild with neither has nothing to build "
            "footprint statics from, and with both one would silently "
            "shadow the other")
    cfg = child_config.run
    ratio = int(child_config.parent_grid_ratio)
    ref_i = int(reference_i_parent_start)
    ref_j = int(reference_j_parent_start)
    spec_width = int(cfg.spec_bdy_width)
    blend_width = int(getattr(child_config, "blend_width", 5))

    def initialize(new_dc, parent_node, *, scratch_arena=None,
                   dycore_state_workspace=None):
        from gpuwm.ingest.nest_init import (_adjust_and_rederive, _as_like,
                                            _capture_parent_blend_fields,
                                            _child_grid,
                                            _shared_vertical_coord,
                                            parent_only_init,
                                            seed_rk_time_t_copies)
        from gpuwm.core.nest_interp import blend_terrain
        from gpuwm.ingest.real import _make_real_base

        if int(new_dc.grid_id) != int(child_config.grid_id):
            raise RelocationRefusal(
                f"this initializer serves grid_id {child_config.grid_id}, "
                f"asked to rebuild grid_id {new_dc.grid_id}")
        shift_i = (int(new_dc.i_parent_start) - ref_i) * ratio
        shift_j = (int(new_dc.j_parent_start) - ref_j) * ratio
        grid = reference_grid.translated(shift_i, shift_j)
        # The translated frame and the parent's nest arithmetic must
        # describe the same physical corner; drift means the reference
        # this factory holds is not this tree's child, which is a wiring
        # defect and refuses before any state is touched.
        resolved = _child_grid(new_dc, parent_node)
        lat_t, lon_t = grid.ij_to_latlon(1.0, 1.0)
        lat_r, lon_r = resolved.ij_to_latlon(1.0, 1.0)
        drift = max(abs(float(lat_t) - float(lat_r)),
                    abs(float(lon_t) - float(lon_r)))
        if drift > drift_tolerance_deg:
            raise RelocationRefusal(
                f"placement-translated grid drifts {drift:.3e} deg from the "
                f"parent-resolved placement at ({new_dc.i_parent_start}, "
                f"{new_dc.j_parent_start}); the initializer's reference "
                "grid does not belong to this tree")

        started = time.perf_counter()
        if statics_builder is None:
            static_fields, selection, landuse_attrs, highres_applied = \
                _build_footprint_statics(grid, catalog, new_dc)
            static_source = str(selection.root)
        else:
            static_fields = statics_builder(grid, new_dc)
            static_source = str(getattr(
                statics_builder, "source_label", "statics_builder"))
            highres_applied = bool(getattr(
                statics_builder, "highres_applied", False))
        statics_seconds = time.perf_counter() - started

        extra = {}
        if scratch_arena is not None:
            extra["scratch_arena"] = scratch_arena
        if dycore_state_workspace is not None:
            extra["dycore_state_workspace"] = dycore_state_workspace
        sint_started = time.perf_counter()
        initialized = parent_only_init(new_dc, parent_node, grid=grid,
                                       **extra)
        sint_seconds = time.perf_counter() - sint_started

        # The t=0 terrain-adjustment sequence, on the parent-interpolated
        # state.  parent_only_init already ran update_diagnostics, so
        # state.p is the pre-change pressure adjust_tempqv requires --
        # the same ordering finalize_prepared_child establishes at t=0.
        adjust_started = time.perf_counter()
        state = initialized.state
        coord = _shared_vertical_coord(vertical, cfg.nz)
        save_mub = state.mub2d.copy()
        fine = _make_real_base(
            coord, np.asarray(static_fields["HGT_M"], dtype=np.float64),
            float(vertical.p_top), float(cfg.base_temp),
            int(cfg.hypsometric_opt))
        state.ht[...] = _as_like(fine.terrain_z, state.ht)
        state.mub2d[...] = _as_like(fine.mub, state.mub2d)
        state.phb[...] = _as_like(fine.phb, state.phb)
        ht_int, mub_int, phb_int = _capture_parent_blend_fields(
            new_dc, parent_node)
        ht_int = _as_like(ht_int, state.ht)
        mub_int = _as_like(mub_int, state.mub2d)
        phb_int = _as_like(phb_int, state.phb)
        blend_terrain(ht_int, state.ht, spec_bdy_width=spec_width,
                      blend_width=blend_width)
        blend_terrain(mub_int, state.mub2d, spec_bdy_width=spec_width,
                      blend_width=blend_width)
        blend_terrain(phb_int, state.phb, spec_bdy_width=spec_width,
                      blend_width=blend_width)
        _adjust_and_rederive(state, cfg, coord, save_mub,
                             static_fields["HGT_M"])
        undershoot = clamp_interpolation_undershoot(state)
        seed_rk_time_t_copies(state)
        del ht_int, mub_int, phb_int, save_mub
        adjust_seconds = time.perf_counter() - adjust_started

        receipt = {
            "initializer": "real_relocation_initializer",
            "static_source": static_source,
            "static_provenance": initialize.static_provenance,
            "highres_applied": bool(highres_applied),
            "placement_translation_child_cells": [shift_i, shift_j],
            "translation_drift_deg": float(drift),
            "interpolation_undershoot_clamped": undershoot,
            "timings_seconds": {
                "static_rebuild": statics_seconds,
                "atmosphere_sint": sint_seconds,
                "terrain_adjustment": adjust_seconds,
            },
            "terrain_adjustment": (
                "blend_terrain(ht/mub/phb) + adjust_tempqv + start_domain "
                "base/EOS re-derivation + press_adj (t=0 lineage, "
                f"spec_bdy_width={spec_width}, blend_width={blend_width})"),
        }
        from dataclasses import replace as _replace

        return _replace(initialized, grid=grid,
                        static_fields=static_fields,
                        preprocess_receipt=receipt)

    def post_transplant(*, source_state, target_state, plan):
        return rebase_transplanted_perturbations(
            source_state=source_state, target_state=target_state,
            plan=plan, cfg=cfg)

    if statics_builder is None:
        initialize.static_provenance = REAL_DATA_FOOTPRINT_REBUILT_STATICS
        initialize.strip_fill_source = REAL_DATA_STRIP_FILL_SOURCE
    else:
        initialize.static_provenance = str(getattr(
            statics_builder, "static_provenance",
            REAL_DATA_FOOTPRINT_REBUILT_STATICS))
        from gpuwm.static.corridor import CORRIDOR_STRIP_FILL_SOURCE
        initialize.strip_fill_source = CORRIDOR_STRIP_FILL_SOURCE
    # The donor-alignment instrument compares SINT/analytic base fields
    # that are placement-invariant only OUTSIDE both placements' blend
    # frames; inside them the t=0 machinery blends toward the parent at
    # each footprint's own edge and the invariant is genuinely undefined.
    initialize.donor_alignment_frame_width = spec_width + blend_width
    initialize.post_transplant = post_transplant
    return initialize


__all__ = [
    "DonorFillPlan", "LAND_SURFACE_CONTINUATION_FIELDS",
    "OVERLAP_STATIC_EQUALITY_FIELDS",
    "REAL_DATA_FOOTPRINT_REBUILT_STATICS", "REAL_DATA_STRIP_FILL_SOURCE",
    "clamp_interpolation_undershoot", "donor_fill_plan",
    "overlap_mask_for_plan",
    "overlap_statics_mismatches", "real_relocation_initializer",
    "rebase_transplanted_perturbations",
]
