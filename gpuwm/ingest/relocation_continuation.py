"""Global host-side continuation decisions shared by resident and slab rebuilds."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelocationContinuation:
    captured: dict
    plan: object
    statics_verdict: dict
    fill: object
    land: dict


def stage_relocation_continuation(captured, new_dc, static, *, plan=None):
    """Shift on the full domain and choose global same-class donor columns.

    The result may be sliced for bounded device construction afterwards.
    Choosing donors separately within slabs would change scientific fill at
    slab boundaries, so no window enters this operation.
    """
    from gpuwm.core.nest_relocation import Placement, RelocationRefusal, plan_relocation
    from gpuwm.ingest.relocation_init import (
        donor_fill_plan, overlap_mask_for_plan, overlap_statics_mismatches)

    if captured is None or captured["grid_id"] != int(new_dc.grid_id):
        raise RelocationRefusal(
            "RealRelocationChildPreparer.__call__ without a matching "
            "capture_outgoing: the runner drives both seams, and a "
            "rebuild that never saw the outgoing child has no land "
            "state to move")
    if static is None:
        raise RelocationRefusal(
            "the relocation initializer produced no static fields; "
            "the real-data route requires footprint-rebuilt statics")
    cfg = new_dc.run
    # Descendants can move in ground coordinates without changing their
    # placement relative to their parent. The route's override is authority.
    if plan is None:
        plan = plan_relocation(
            placement_from=Placement(
                grid_id=captured["grid_id"],
                i_parent_start=captured["i_parent_start"],
                j_parent_start=captured["j_parent_start"]),
            placement_to=Placement(
                grid_id=int(new_dc.grid_id),
                i_parent_start=int(new_dc.i_parent_start),
                j_parent_start=int(new_dc.j_parent_start), generation=1),
            parent_grid_ratio=int(new_dc.parent_grid_ratio),
            child_nx=int(cfg.nx), child_ny=int(cfg.ny))
    if captured["static_fields"] is None:
        raise RelocationRefusal(
            "the outgoing child's statics are not on record, so the "
            "overlap-statics equality cannot be asserted; a move "
            "whose load-bearing claim cannot be checked is refused")
    verdict = overlap_statics_mismatches(captured["static_fields"], static, plan)
    if not verdict["pass"]:
        raise RelocationRefusal(
            "footprint-rebuilt statics differ from the outgoing "
            "child's on shared ground (identical source + identical "
            "cells must give identical bytes); this is a statics-"
            "build defect, not an input error: "
            f"{verdict['mismatched_fields'] or verdict}")
    fill = donor_fill_plan(overlap_mask=overlap_mask_for_plan(plan, (cfg.ny, cfg.nx)),
                           landmask=np.asarray(static["LANDMASK"]))
    moved = {}
    for name, old in captured["fields"].items():
        window = plan.window(old.shape)
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        staged = np.zeros_like(old)
        staged[..., dst_j, dst_i] = old[..., src_j, src_i]
        moved[name] = fill.apply(staged)
    return RelocationContinuation(captured, plan, verdict, fill, moved)
