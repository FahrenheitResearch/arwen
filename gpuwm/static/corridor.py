"""Sealed child-resolution statics corridor for prepared moving nests.

The prepared (digest-bound) tree routes deliberately run WITHOUT their
ingest inputs, so a relocating nest had nothing to rebuild its statics
from and the tree runner refused ``[relocation]`` follow sources
outright.  The corridor closes that gap at PREPARATION time, when the
geography source IS on hand: statics are pre-built at CHILD resolution
over the ground the nest can REACH, sealed beside the other hierarchy
artifacts, and their digest is bound into the preparation document
exactly like every other sealed artifact.  At runtime a relocation CROPS
the new footprint's statics out of the corridor -- no runtime ingest, no
loosening of the digest relay.

WHICH GROUND.  The reach (:mod:`gpuwm.core.nest_reach`): the child's
declared footprint widened by the most every mover above it (and the
child itself) can travel over the run -- from its follow settings, its
itinerary, the run length and the ``reach_speed_m_s`` bound -- and clipped
to the frame.  Before 2.8 every corridor covered the whole frame, which
made a storm-following 500 m nest over a 2,700 x 3,000 km parent a 25 GB
artifact and a 62 GB preparation.  A nest that can reach the whole frame
still gets the whole frame, and its receipt says why.

WHY A CROP IS EXACT.  The corridor grid is the child's reference grid
``translated`` onto the parent's first cell and re-extented to cover the
parent (:meth:`gpuwm.static.projection.ProjectedGrid.translated`), so
every corridor cell evaluates its coordinates through the reference
grid's own float arithmetic at the reference's own index -- the same
bytes a placement-translated footprint build evaluates for that cell.
The static build is per-cell on those coordinates (accumulation order is
absolute-source-row-major, interpolation is native-tile-scoped, and the
terrain smoother's dependency cone lies inside the shared halo), so a
footprint cropped from the corridor is BITWISE the statics built
directly for that footprint from the same geography.  That equality is
the same invariant the relocation machinery already asserts on every
move (``identical source + identical cells = identical bytes``), and
tests/test_statics_corridor.py proves it against the build rather than
assuming it.

It also needs the SOURCE side to be placement-independent, which is what
the corridor's own extent put at risk: a corridor over a parent that
spans the antimeridian reads a source window crossing the wrap seam
while its child's footprint window does not.  The build bins every
source pixel at its canonical column for exactly that reason, and each
sealed corridor records the contract it was built under
(:data:`STATICS_CORRIDOR_BUILD_CONTRACT`) because the digest relay
proves WHICH BYTES preparation wrote and not which build wrote them.

MEMORY POSTURE.  The corridor is a DISK artifact loaded into HOST memory
by the runner's preflight; crops are host arrays consumed by the same
rebuild path the case-data route uses.  It adds no GPU residency: the
rebuilt child has the same device footprint at every placement.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from gpuwm.native_wrf_contract import (
    NATIVE_LANDUSE_IDENTITY,
    NATIVE_STATIC_REQUIRED,
    _NATIVE_STATIC_CATEGORY_COUNT,
    _NATIVE_STATIC_MONTHLY,
)

#: One corridor set per prepared hierarchy: ``receipt.json`` plus one
#: ``dNN.npz`` per covered child, under ``hierarchy-artifacts/<DIRNAME>``.
STATICS_CORRIDOR_SET_SCHEMA = "gpuwm-statics-corridor-set-v1"
STATICS_CORRIDOR_SCHEMA = "gpuwm-child-statics-corridor-v1"
STATICS_CORRIDOR_DIRNAME = "statics-corridor"
STATICS_CORRIDOR_RECEIPT = "receipt.json"

#: The preparation flag that emits a corridor; refusals name it so the
#: remedy is one re-preparation away, and the spelling lives in exactly
#: one place.
STATICS_CORRIDOR_FLAG = "--statics-corridor"

#: What the sealed FIELD BYTES were produced by, recorded per corridor and
#: checked at load.  Older receipts are re-verified by field content when
#: the tree supplies its sealed child statics. The first crop must agree
#: with those fields under the same rule the first relocation applies.
#: The digest binding proves which bytes preparation wrote, regardless
#: of how a newer builder would represent the same ground.
#:
#: ``canonical-source-column-binning-v1``: every source pixel was binned at
#: its CANONICAL column.  Before 2.7.5 a build whose source window crossed
#: the x-wrap seam kept the unwrapped column index, and a WPS_GEOG index
#: declares a truncated decimal (30-arcsec trees say dx = 0.00833333, whose
#: 43200-fold is 359.999856 deg), so those pixels were placed 1.44e-4 deg
#: west of the ground whose bytes they carried.  A corridor over a
#: dateline-spanning parent therefore disagreed with the footprint build on
#: the categorical fields, and the first relocation refused on the
#: overlap-statics equality.
STATICS_CORRIDOR_BUILD_CONTRACT = "canonical-source-column-binning-v1"

#: Receipt-stated provenance for statics a relocation crops from the
#: corridor (the prepared-route counterpart of
#: :data:`gpuwm.ingest.relocation_init.REAL_DATA_FOOTPRINT_REBUILT_STATICS`).
CORRIDOR_REBUILT_STATICS = (
    "footprint statics cropped from the sealed child-resolution statics "
    "corridor (build_static over the nest's reach, emitted at preparation time, "
    "digest-bound into the preparation document and verified before "
    "use); terrain adjustment via the t=0 nest cold-start sequence "
    "(blend_terrain + adjust_tempqv + start_domain re-derivation + "
    "press_adj)")

#: What fills the strip a move exposes under the corridor initializer.
CORRIDOR_STRIP_FILL_SOURCE = (
    "full-parent SINT adjusted to corridor-cropped fine terrain (t=0 "
    "nest cold-start lineage); overlap then stamped bitwise from the "
    "outgoing child, blend-frame perturbations rebased to preserve "
    "totals; land-surface state donor-filled per the relocation "
    "contract")


class CorridorRefusal(ValueError):
    """A corridor that cannot be verified refuses; it never degrades to a
    silently static nest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def config_declares_follow_source(exp) -> bool:
    """Does this experiment move a nest during the run?

    THE predicate.  Every door that has to react to a moving nest asks
    this one function: ``gpuwm go``'s prepare stage and the printed
    ``rw-wps`` line both derive ``--statics-corridor`` from it, and
    ``gpuwm run-plan`` decides from it whether the chain it is about to
    dispatch can supply a moving nest's statics at all.  Three readings
    of the same sentence used to be three copies of it; a config that
    one door thought was following and another did not would prepare a
    bundle its own forecast stage refuses.

    Bounds-only ``[relocation]`` (enabled, but naming neither a follow
    source nor an explicit itinerary) is not a moving nest: it
    constrains where a nest may sit, and the nest stays put.
    """
    relocation = exp.relocation
    return bool((relocation.enabled
                 and (relocation.follow is not None or relocation.moves))
                or any(getattr(dc, "follow", None) is not None
                       for dc in getattr(exp, "domains", ())))


# ---------------------------------------------------------------------------
# Geometry: where the corridor sits on the child lattice
# ---------------------------------------------------------------------------

def ancestor_chain(domains_by_id, grid_id: int, frame_id: int) -> list:
    """``[target, ..., direct child of frame]`` -- target-first."""
    chain = []
    gid = int(grid_id)
    seen = set()
    while gid != int(frame_id):
        if gid in seen or gid not in domains_by_id:
            raise ValueError(
                f"grid {grid_id} does not descend from frame {frame_id}")
        seen.add(gid)
        dc = domains_by_id[gid]
        chain.append(dc)
        gid = int(dc.parent_id)
    return chain


def origin_in_frame_cells(domains_by_id, grid_id: int, frame_id: int
                          ) -> tuple[int, int]:
    """The child's origin in ITS OWN cells, counted from frame cell 1.

    THE WHOLE REASON A MID-TREE MOVE IS EXPRESSIBLE.  A placement is a
    whole number of PARENT cells, which is why the parent-anchored
    corridor could index itself as ``(ip-1)*ratio``.  Two levels down
    that breaks: d03 under a 27/9/3 km tree sits 1005 cells of 3 km from
    d01's cell 1, which is 111.67 cells of d01 -- not an integer, so no
    parent-cell index can name it and a corridor addressed that way is
    off by up to a whole parent cell of terrain.

    Counted in the CHILD's own cells it is exactly 1005.  Each ancestor
    contributes ``(i_parent_start - 1)`` of its parent's cells, converted
    to child cells by the product of the ratios from that ancestor down
    to the child -- every factor an integer, so the sum is exact.  This
    is the same arithmetic WRF's moving nests rely on when they index a
    pre-staged high-resolution static field covering the roam area.
    """
    oi = oj = 0
    prod = 1
    for dc in ancestor_chain(domains_by_id, grid_id, frame_id):
        prod *= int(dc.parent_grid_ratio)
        oi += (int(dc.i_parent_start) - 1) * prod
        oj += (int(dc.j_parent_start) - 1) * prod
    return oi, oj


def corridor_geometry(child_dc, parent_run, *, frame_run=None,
                      ratio_to_frame: int | None = None,
                      frame_grid_id: int | None = None,
                      reference_origin: tuple[int, int] | None = None,
                      window: tuple[int, int, int, int] | None = None
                      ) -> dict[str, object]:
    """The corridor's placement-independent geometry for one child.

    The corridor covers a WINDOW of a FRAME domain's mass extent at child
    resolution -- the whole extent when ``window`` is ``None`` -- and is
    addressed in CHILD cells from frame cell 1.  ``window`` is
    ``(x0, y0, nx, ny)`` in those cells.  A whole-frame window adds no
    key, so a whole-frame corridor's geometry, receipt and sealed bytes
    are exactly what they were before windows existed; a smaller one adds
    ``window_origin_child_cells`` and sets ``corridor_nx``/``corridor_ny``
    to its own size.

    Frame = the child's own parent (the default, and every corridor
    before mid-tree moves existed): the frame is stationary, so the
    footprint at ``i_parent_start = ip`` sits at child cell
    ``(ip-1)*ratio`` and the bundle is byte-for-byte what it was.

    Frame = the ROOT: required when an ANCESTOR of this child moves.
    The child's parent is then not a fixed frame at all -- its cells
    describe different ground after every move -- so a corridor anchored
    to it would hand back the wrong terrain.  Anchoring to the root, the
    one domain that never moves, keeps every crop exact; see
    :func:`origin_in_frame_cells` for why the addressing stays integral.
    """
    ratio = int(child_dc.parent_grid_ratio)
    if ratio < 1:
        raise ValueError(f"parent_grid_ratio must be >= 1, got {ratio}")
    frame_run = parent_run if frame_run is None else frame_run
    ratio_to_frame = ratio if ratio_to_frame is None else int(ratio_to_frame)
    frame_grid_id = (int(child_dc.parent_id) if frame_grid_id is None
                     else int(frame_grid_id))
    ref_i = int(child_dc.i_parent_start)
    ref_j = int(child_dc.j_parent_start)
    if reference_origin is None:
        reference_origin = ((ref_i - 1) * ratio, (ref_j - 1) * ratio)
    origin_i, origin_j = (int(reference_origin[0]), int(reference_origin[1]))
    frame_nx = int(frame_run.nx) * ratio_to_frame
    frame_ny = int(frame_run.ny) * ratio_to_frame
    if window is None:
        window = (0, 0, frame_nx, frame_ny)
    x0, y0, nx, ny = (int(value) for value in window)
    if x0 < 0 or y0 < 0 or nx < 1 or ny < 1 or (
            x0 + nx > frame_nx or y0 + ny > frame_ny):
        raise ValueError(
            f"corridor window {nx}x{ny} child cells at ({x0}, {y0}) does "
            f"not lie inside the {frame_nx}x{frame_ny}-cell frame "
            f"d{frame_grid_id:02d}")
    geometry = {
        "grid_id": int(child_dc.grid_id),
        "parent_id": int(child_dc.parent_id),
        "parent_grid_ratio": ratio,
        "frame_grid_id": frame_grid_id,
        "ratio_to_frame": ratio_to_frame,
        "reference_i_parent_start": ref_i,
        "reference_j_parent_start": ref_j,
        "reference_origin_child_cells": [origin_i, origin_j],
        "child_nx": int(child_dc.run.nx),
        "child_ny": int(child_dc.run.ny),
        "parent_nx": int(frame_run.nx),
        "parent_ny": int(frame_run.ny),
        "corridor_nx": nx,
        "corridor_ny": ny,
        # The exact whole-cell translation from the child's reference
        # grid to the corridor origin (the window's first cell).
        "origin_translation_child_cells": [x0 - origin_i, y0 - origin_j],
    }
    if (x0, y0, nx, ny) != (0, 0, frame_nx, frame_ny):
        geometry["window_origin_child_cells"] = [x0, y0]
    return geometry


def geometry_window(geometry: Mapping[str, object]
                    ) -> tuple[int, int, int, int]:
    """``(x0, y0, nx, ny)`` of a corridor in frame child cells."""
    x0, y0 = geometry.get("window_origin_child_cells", (0, 0))
    return (int(x0), int(y0), int(geometry["corridor_nx"]),
            int(geometry["corridor_ny"]))


def moving_grid_ids(exp) -> frozenset[int]:
    """Grid ids the legacy relocation and per-domain followers may move.

    The tracked mover, plus the ``[relocation.containment]`` ancestor
    when one is configured -- the ancestor slides in whole cells of ITS
    parent, so everything downstream of this answer (corridor coverage,
    root-anchored frames for children of a mover) must count it as a
    mover in its own right.
    """
    movers = {int(dc.grid_id) for dc in getattr(exp, "domains", ())
              if getattr(dc, "follow", None) is not None}
    relocation = getattr(exp, "relocation", None)
    if (relocation is not None and getattr(relocation, "enabled", False)
            and (getattr(relocation, "follow", None) is not None
                 or getattr(relocation, "moves", ()))):
        grid_id = getattr(relocation, "grid_id", None)
        if grid_id is not None:
            movers.add(int(grid_id))
            containment = getattr(relocation, "containment", None)
            if containment is not None:
                movers.add(int(containment.grid_id))
    return frozenset(movers)


def relocating_subtree_grid_ids(exp, *, moving_roots=None) -> tuple[int, ...]:
    """Every grid whose GROUND changes when this experiment moves.

    The mover plus all of its descendants, ascending.  A mid-tree move
    re-grounds the whole subtree -- each member rebuilds its statics for
    new ground -- so each member needs its own corridor, and this is the
    one place that says which.  For a leaf mover it is the single grid id
    it has always been, so nothing about a leaf bundle changes.
    """
    movers = moving_grid_ids(exp) if moving_roots is None else moving_roots
    if not movers:
        return ()
    by_parent: dict[int, list[int]] = {}
    for d in exp.domains:
        by_parent.setdefault(int(d.parent_id), []).append(int(d.grid_id))
    out, stack = set(), list(movers)
    while stack:
        gid = stack.pop()
        if gid in out:
            continue
        out.add(gid)
        stack.extend(by_parent.get(gid, ()))
    return tuple(sorted(out))


def corridor_frame_kwargs(exp, child_dc) -> dict[str, object]:
    """Which frame this child's corridor is anchored to.

    ONE function, called by the emission and by the acceptance, because
    they must agree exactly: the loader re-derives the geometry and
    compares it key by key, so a frame chosen two ways is a refusal on
    every run that should have worked.

    The rule: anchor to the child's own parent unless a STRICT ANCESTOR
    of the child moves, in which case anchor to the root.  A parent that
    moves is not a frame -- its cell 1 describes different ground after
    every relocation -- so a corridor addressed against it hands back
    terrain from wherever the parent used to be.  When nothing above the
    child moves this returns ``{}`` and the geometry, the receipt and
    the sealed bytes are exactly what they were before mid-tree moves
    existed.
    """
    domains_by_id = {int(d.grid_id): d for d in exp.domains}
    movers = moving_grid_ids(exp)
    root_id = next(int(d.grid_id) for d in exp.domains
                   if int(d.parent_id) in (0, int(d.grid_id)))
    ancestors = []
    gid = int(child_dc.parent_id)
    while gid in domains_by_id and gid != root_id:
        ancestors.append(gid)
        gid = int(domains_by_id[gid].parent_id)
    if not (movers & set(ancestors)):
        return {}
    ratio_to_frame = 1
    for dc in ancestor_chain(domains_by_id, int(child_dc.grid_id), root_id):
        ratio_to_frame *= int(dc.parent_grid_ratio)
    return {
        "frame_run": domains_by_id[root_id].run,
        "frame_grid_id": root_id,
        "ratio_to_frame": ratio_to_frame,
        "reference_origin": origin_in_frame_cells(
            domains_by_id, int(child_dc.grid_id), root_id),
    }


def corridor_reach(exp, child_dc) -> dict[str, object]:
    """The part of the frame a child's footprint can reach over the run.

    ONE function for every reader -- the emission that builds the
    corridor, the run-plan estimate that prices it, the terrain survey
    that reads its ground, the loader that checks a sealed one covers
    it, and the tests -- so the window a run is priced at is the window
    it is built and verified at.

    The child's declared footprint in the frame, widened on each axis by
    the displacement range of every mover in its chain (itself and each
    ancestor that moves), each scaled from that mover's parent cells to
    the child's own cells, then clipped to the frame.  A mover that is
    the ``[relocation.containment]`` ancestor of the child's chain is
    skipped for the tracked mover's subtree: the slide carries that
    subtree earth-fixed.  A dormant nest anywhere in the chain, or a
    mover nothing bounds, reaches the whole frame and the record says so.

    Returns ``window_child_cells`` ``[x0, y0, nx, ny]``,
    ``frame_child_cells`` ``[nx, ny]``, ``whole_frame``, the per-mover
    ``movers`` bases and, when the whole frame is taken because nothing
    bounds the reach, the ``unbounded`` reason.
    """
    from gpuwm.core.nest_reach import _contained_mover, mover_reach

    by_id = {int(d.grid_id): d for d in exp.domains}
    parent_run = by_id[int(child_dc.parent_id)].run
    frame_kwargs = corridor_frame_kwargs(exp, child_dc)
    full = corridor_geometry(child_dc, parent_run, **frame_kwargs)
    frame_nx, frame_ny = int(full["corridor_nx"]), int(full["corridor_ny"])
    origin_x, origin_y = full["reference_origin_child_cells"]
    nx, ny = int(child_dc.run.nx), int(child_dc.run.ny)
    root_id = next(int(d.grid_id) for d in exp.domains
                   if int(d.parent_id) in (0, int(d.grid_id)))
    relocation = getattr(exp, "relocation", None)
    contained = _contained_mover(exp)
    containment = (None if contained is None
                   else int(relocation.containment.grid_id))
    record = {
        "frame_grid_id": int(full["frame_grid_id"]),
        "frame_child_cells": [frame_nx, frame_ny],
        "movers": [],
    }

    def whole(reason: str) -> dict[str, object]:
        record.update({"window_child_cells": [0, 0, frame_nx, frame_ny],
                       "whole_frame": True, "unbounded": reason})
        return record

    lo_x = hi_x = lo_y = hi_y = 0
    scale = 1
    seen_contained = False
    dc = child_dc
    while int(dc.grid_id) != root_id:
        gid = int(dc.grid_id)
        scale *= int(dc.parent_grid_ratio)
        if getattr(dc, "spawn", None) is not None:
            return whole(
                f"d{gid:02d} is dormant: its placement is chosen when it "
                "fires, so no declared start bounds where this footprint "
                "can be")
        if gid == contained:
            seen_contained = True
        reach = mover_reach(exp, gid)
        if reach is not None and not (gid == containment
                                      and seen_contained):
            if not reach.bounded:
                return whole(str(reach.basis.get("reason")))
            lo_x += int(reach.lo_i) * scale
            hi_x += int(reach.hi_i) * scale
            lo_y += int(reach.lo_j) * scale
            hi_y += int(reach.hi_j) * scale
            record["movers"].append({
                "grid_id": gid,
                "range_parent_cells": [int(reach.lo_i), int(reach.hi_i),
                                       int(reach.lo_j), int(reach.hi_j)],
                "child_cells_per_parent_cell": scale,
                **reach.basis})
        dc = by_id[int(dc.parent_id)]
    x0 = max(0, int(origin_x) + lo_x)
    y0 = max(0, int(origin_y) + lo_y)
    x1 = min(frame_nx, int(origin_x) + hi_x + nx)
    y1 = min(frame_ny, int(origin_y) + hi_y + ny)
    record.update({
        "window_child_cells": [x0, y0, x1 - x0, y1 - y0],
        "whole_frame": (x0, y0, x1, y1) == (0, 0, frame_nx, frame_ny),
    })
    return record


@dataclass(frozen=True)
class PlannedCorridor:
    """One child's corridor as the experiment asks for it: frame, window,
    geometry and the reach record that sized it."""

    frame_kwargs: dict
    window: tuple[int, int, int, int]
    geometry: dict
    reach: dict


def planned_corridor(exp, child_dc) -> PlannedCorridor:
    """The frame, reach window and geometry of one child's corridor."""
    by_id = {int(d.grid_id): d for d in exp.domains}
    frame_kwargs = corridor_frame_kwargs(exp, child_dc)
    reach = corridor_reach(exp, child_dc)
    window = tuple(int(v) for v in reach["window_child_cells"])
    geometry = corridor_geometry(
        child_dc, by_id[int(child_dc.parent_id)].run, **frame_kwargs,
        window=window)
    return PlannedCorridor(frame_kwargs=frame_kwargs, window=window,
                           geometry=geometry, reach=reach)


def _planes_per_cell() -> int:
    """Float64 planes one corridor cell carries.

    Counted off the native static contract itself -- the SAME inventory
    :func:`_validate_corridor_fields` shape-checks a built corridor
    against -- rather than restated as a literal, so a field added to
    the contract reprices the corridor instead of silently making the
    quoted size wrong.
    """
    planes = 0
    for name in NATIVE_STATIC_REQUIRED:
        if name in _NATIVE_STATIC_CATEGORY_COUNT:
            planes += int(_NATIVE_STATIC_CATEGORY_COUNT[name])
        elif name in _NATIVE_STATIC_MONTHLY:
            planes += 12
        else:
            planes += 1
    return planes


#: Float64 planes, and therefore bytes, one corridor cell costs.  The
#: build's own dtype is float64 (``_validate_corridor_fields`` refuses
#: anything else), so the itemsize is asked of numpy rather than
#: assumed to be 8.
CORRIDOR_PLANES_PER_CELL = _planes_per_cell()
CORRIDOR_BYTES_PER_CELL = (CORRIDOR_PLANES_PER_CELL
                           * int(np.dtype(np.float64).itemsize))


def corridor_cost(child_dc, parent_run, frame_kwargs=None,
                  window=None) -> dict[str, int]:
    """What one child's corridor will cost, WITHOUT building it.

    Pure arithmetic on the geometry and the field inventory: no GEOG
    source, no ``build_static`` call, nothing on disk.  That is what
    lets a front door price a corridor before the preparation runs --
    the figure a caller sees ahead of launch and the ``host_bytes`` the
    sealed receipt reports afterwards come out of the same two facts
    (cell count and plane count), and
    ``tests/test_statics_corridor.py`` holds them equal against a real
    build rather than against each other.

    ``frame_kwargs`` and ``window`` are the build's own: a child under a
    moving ancestor is framed on the root, and pricing it on its parent
    (as this did before 2.8, ignoring the frame it was handed) quoted the
    1 km corridor of a 9/3/1 km storm-following tree at a tenth, and a
    500 m corridor below it at a ninetieth, of what the build wrote.
    :func:`planned_corridor_cost` passes both from the experiment.

    ``host_bytes`` is the loaded footprint AND, to within the container
    headers, the on-disk one: the cache is an uncompressed
    (``ZIP_STORED``) NPZ of exactly these arrays.
    """
    geometry = corridor_geometry(child_dc, parent_run,
                                 **(frame_kwargs or {}), window=window)
    cells = int(geometry["corridor_nx"]) * int(geometry["corridor_ny"])
    return {
        "grid_id": int(geometry["grid_id"]),
        "parent_id": int(geometry["parent_id"]),
        "corridor_nx": int(geometry["corridor_nx"]),
        "corridor_ny": int(geometry["corridor_ny"]),
        "cells": cells,
        "planes_per_cell": CORRIDOR_PLANES_PER_CELL,
        "bytes_per_cell": CORRIDOR_BYTES_PER_CELL,
        "host_bytes": cells * CORRIDOR_BYTES_PER_CELL,
    }


def planned_corridor_cost(exp, child_dc) -> dict[str, object]:
    """:func:`corridor_cost` for the corridor this experiment will seal:
    its own frame and reach window, plus whether that is the whole frame."""
    plan = planned_corridor(exp, child_dc)
    by_id = {int(d.grid_id): d for d in exp.domains}
    cost = corridor_cost(child_dc, by_id[int(child_dc.parent_id)].run,
                         plan.frame_kwargs, plan.window)
    cost["frame_grid_id"] = int(plan.geometry["frame_grid_id"])
    cost["window_child_cells"] = list(plan.window)
    cost["frame_child_cells"] = list(plan.reach["frame_child_cells"])
    cost["whole_frame"] = bool(plan.reach["whole_frame"])
    if plan.reach.get("unbounded"):
        cost["unbounded"] = plan.reach["unbounded"]
    return cost


def corridor_summary_line(label: str, entry: Mapping[str, object]) -> str:
    """The one line a preparation door prints per sealed corridor.

    Size accuracy at the door: the cost is stated where it is paid, and so
    is the ground it covers -- the reach window, or the whole frame and
    the reason nothing narrower holds.
    """
    frame = int(entry.get("frame_grid_id", entry.get("parent_id", 0)))
    reach = entry.get("reach") or {}
    if "window_origin_child_cells" not in entry:
        cover = f"over the whole d{frame:02d} extent"
        why = reach.get("unbounded")
        if why:
            cover += f" ({why})"
        elif reach.get("movers"):
            cover += " (its reach covers all of it)"
    else:
        frame_nx = int(entry["parent_nx"]) * int(entry["ratio_to_frame"])
        frame_ny = int(entry["parent_ny"]) * int(entry["ratio_to_frame"])
        share = (int(entry["corridor_nx"]) * int(entry["corridor_ny"])
                 / float(frame_nx * frame_ny))
        cover = (f"covering the ground it can reach, {share:.0%} of the "
                 f"d{frame:02d} extent")
    return (f"  statics corridor {label}: "
            f"{entry['corridor_nx']}x{entry['corridor_ny']} child cells "
            f"{cover}, {entry['cache']['bytes'] / 1.0e6:.1f} MB on disk, "
            f"{entry['host_bytes'] / 1.0e6:.1f} MB host when loaded by a "
            "relocating run (no GPU residency)")


def corridor_grid(reference_grid, geometry: Mapping[str, object]):
    """The corridor's grid: the reference child grid, translated and
    re-extented on the SAME lattice (per-cell reference arithmetic)."""
    di, dj = geometry["origin_translation_child_cells"]
    return reference_grid.translated(
        int(di), int(dj),
        e_we=int(geometry["corridor_nx"]) + 1,
        e_sn=int(geometry["corridor_ny"]) + 1)


def grid_identity_probes(grid) -> dict[str, list[float]]:
    """Float64 lat/lon probes pinning where the corridor grid lies.

    Recorded at preparation and required of the runner's reconstructed
    corridor grid by :func:`grid_probe_drift`, within
    :data:`gpuwm.static.grid_identity.GRID_POSITION_TOLERANCE_CELLS` of a
    cell: the same posture as the per-domain MAPFAC regeneration gate,
    which admits math-library rounding and nothing more.  The corridor's
    bitwise claim is about its own sealed bytes, which every crop reads
    whatever machine runs it; what the probes guard is a runner whose
    grid lies somewhere else.
    """
    # Default route: the Rust seam renders the same five probes with
    # shortest-round-trip formatting, parsing back to the exact float64
    # bits (fixed-means-default; GPUWM_STATIC_PYTHON=1 falls back to
    # the arithmetic below as a reported workaround).
    from . import rust_bridge
    bridge = rust_bridge.route("grid_identity_probes")
    if bridge is not None and hasattr(grid, "_rust_handle"):
        return json.loads(bridge.grid_identity_probes_json(
            grid._rust_handle(bridge)))
    nx = int(grid.e_we) - 1
    ny = int(grid.e_sn) - 1
    points = {
        "sw": (1.0, 1.0),
        "se": (float(nx), 1.0),
        "nw": (1.0, float(ny)),
        "ne": (float(nx), float(ny)),
        "center": (grid.e_we / 2.0, grid.e_sn / 2.0),
    }
    probes = {}
    for name, (x, y) in points.items():
        lat, lon = grid.ij_to_latlon(x, y)
        probes[name] = [float(lat), float(lon)]
    return probes


def grid_probe_drift(recorded, grid) -> dict[str, dict[str, object]]:
    """Probes on which ``recorded`` places the corridor grid elsewhere.

    Each probe is a computed position, so it is compared within
    :data:`~gpuwm.static.grid_identity.GRID_POSITION_TOLERANCE_CELLS` of a
    cell of ``grid`` rather than to the bit: the machine that prepared the
    corridor and the one that runs it may round the same projection
    differently in the last digit.  A missing, extra or malformed probe
    is drift.
    """
    from gpuwm.static.grid_identity import grid_record_drift

    expected = grid_identity_probes(grid)
    if not isinstance(recorded, Mapping):
        return {"grid_identity_probes": {"recorded": recorded,
                                         "expected": expected}}

    def flat(probes):
        out = {}
        for name, value in probes.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                out[f"{name}.lat"], out[f"{name}.lon"] = value
            else:
                out[f"{name}"] = value
        return out

    names = sorted(set(recorded) | set(expected))
    return grid_record_drift(
        flat(recorded), flat(expected), dx_m=grid.dx, dy_m=grid.dy,
        points=tuple((f"{name}.lat", f"{name}.lon") for name in names))


# ---------------------------------------------------------------------------
# Field validation (the corridor carries the GEOG-derived inventory only;
# geometry fields regenerate from the footprint grid, as everywhere else)
# ---------------------------------------------------------------------------

def _validate_corridor_fields(fields: Mapping[str, np.ndarray],
                              geometry: Mapping[str, object]) -> None:
    ny = int(geometry["corridor_ny"])
    nx = int(geometry["corridor_nx"])
    names = set(fields)
    if names != set(NATIVE_STATIC_REQUIRED):
        raise CorridorRefusal(
            f"statics corridor field inventory differs from the native "
            f"static contract: missing "
            f"{sorted(set(NATIVE_STATIC_REQUIRED) - names)}, unexpected "
            f"{sorted(names - set(NATIVE_STATIC_REQUIRED))}")
    for name, value in fields.items():
        value = np.asarray(value)
        if name in _NATIVE_STATIC_CATEGORY_COUNT:
            expected = (_NATIVE_STATIC_CATEGORY_COUNT[name], ny, nx)
        elif name in _NATIVE_STATIC_MONTHLY:
            expected = (12, ny, nx)
        else:
            expected = (ny, nx)
        if value.shape != expected:
            raise CorridorRefusal(
                f"statics corridor field {name} has shape {value.shape}, "
                f"expected {expected}")
        if value.dtype != np.float64:
            raise CorridorRefusal(
                f"statics corridor field {name} must be float64 (the "
                f"static build's own dtype), got {value.dtype}")
        if not np.isfinite(value).all():
            raise CorridorRefusal(
                f"statics corridor field {name} contains non-finite values")


# ---------------------------------------------------------------------------
# Preparation side: build and seal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorridorBuild:
    """One child's corridor as built (fields + its receipt entry sans
    cache binding, which the writer adds)."""

    grid_id: int
    fields: Mapping[str, np.ndarray]
    entry: dict


def build_child_statics_corridor(*, child_dc, parent_run, reference_grid,
                                 static_catalog,
                                 frame_kwargs=None, static_highres=None,
                                 window=None, reach=None) -> CorridorBuild:
    """Build one child's statics corridor from the GEOG source, through
    the same builder the domain statics come from.

    ``reference_grid`` is the CHILD's own grid at its declared placement:
    the corridor is that grid translated and re-extented, so every cell
    evaluates through the child's own arithmetic.  ``window`` is the
    reach window (``None``: the whole frame) and ``reach`` the record
    that sized it, sealed into the entry so the receipt says why the
    corridor is the size it is."""
    from gpuwm.static.build import build_static, geog_selection_from_catalog

    geometry = corridor_geometry(child_dc, parent_run, **(frame_kwargs or {}),
                                 window=window)
    grid = corridor_grid(reference_grid, geometry)
    selection = geog_selection_from_catalog(
        static_catalog, int(child_dc.grid_id))
    coverage: dict[str, object] = {}
    fields = build_static(grid, selection.root, selection=selection,
                          source_coverage_report=coverage)
    landuse = selection.landuse_global_attrs()
    highres_receipt = None
    if static_highres is not None and static_highres.enabled:
        from gpuwm.static.highres_production import apply_highres_statics
        fields, highres_receipt = apply_highres_statics(
            fields, grid, config=static_highres, domain_id=child_dc.grid_id,
            case_date=child_dc.start_time.date(), landuse_attrs=landuse)
    _validate_corridor_fields(fields, geometry)
    entry = {
        "schema": STATICS_CORRIDOR_SCHEMA,
        "status": "READY",
        "build_contract": STATICS_CORRIDOR_BUILD_CONTRACT,
        **geometry,
        "grid_identity_probes": grid_identity_probes(grid),
        "landuse": {name: landuse[name]
                    for name in ("MMINLU", "ISWATER", "ISLAKE", "ISICE")},
        "geog": {
            "root": str(selection.root),
            "resolution_tokens": list(selection.resolution_tokens),
        },
        "fields": sorted(fields),
        "cells": geometry["corridor_nx"] * geometry["corridor_ny"],
        "host_bytes": int(sum(np.asarray(value).nbytes
                              for value in fields.values())),
        # Tile-presence proof over the corridor's whole window, summarized
        # (the full per-tile inventories would swell the preparation
        # document without adding a verifiable byte).
        "source_coverage": {
            name: {
                "required_cells": report["required_cells"],
                "coverage_fraction": report["coverage_fraction"],
                "required_tile_count": report["required_tile_count"],
            }
            for name, report in sorted(coverage.items())
        },
    }
    if highres_receipt is not None:
        entry["highres"] = highres_receipt
    if reach is not None:
        entry["reach"] = json.loads(json.dumps(reach, allow_nan=False))
    return CorridorBuild(grid_id=int(child_dc.grid_id), fields=fields,
                         entry=entry)


def validated_corridor_selection(exp, statics_corridor) -> tuple[int, ...]:
    """Resolve the corridor opt-in to an ordered tuple of child grid ids.

    ``None`` selects nothing, ``"all"`` every child domain, and a
    sequence exactly those children.  It lives here rather than beside
    one chain's preparation because THREE readers need the same answer:
    the GFS hierarchy join, the HRRR hierarchy stage, and
    ``run-plan --estimate``, which prices what a bare flag will build.
    A selection resolved one way at pricing time and another at
    preparation time would quote a corridor set that is not the one
    written.
    """
    if statics_corridor is None:
        return ()
    children = [int(domain.grid_id) for domain in exp.domains
                if int(domain.parent_id) != 0]
    if not children:
        raise ValueError(
            "statics_corridor was requested but this experiment has no "
            "child domain; a corridor is child-resolution statics over a "
            "parent, so a single-domain preparation has nothing to emit")
    if isinstance(statics_corridor, str):
        token = statics_corridor.strip().lower()
        if token != "all":
            raise ValueError(
                f"statics_corridor accepts 'all' or child grid ids, got "
                f"{statics_corridor!r}")
        return tuple(children)
    try:
        requested = tuple(int(value) for value in statics_corridor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"statics_corridor accepts 'all' or child grid ids, got "
            f"{statics_corridor!r}") from exc
    unknown = sorted(set(requested) - set(children))
    if unknown:
        raise ValueError(
            f"statics_corridor names grid ids {unknown} that are not "
            f"child domains of this experiment (children: {children})")
    if len(set(requested)) != len(requested):
        raise ValueError(
            f"statics_corridor repeats grid ids: {list(requested)}")
    return tuple(sorted(set(requested)))


def emit_statics_corridor_set(*, exp, grids, static_catalog, directory,
                              statics_corridor, static_highres=None):
    """Build and seal the selected children's corridors, or emit nothing.

    THE emission.  Every preparation chain that can seal a corridor
    calls this one function -- the GFS/ERA5 hierarchy join
    (:func:`gpuwm.source_hierarchy
    .initialize_and_export_regular_source_hierarchy`) and the HRRR
    hierarchy stage (:func:`gpuwm.hrrr_hierarchy_direct
    .prepare_hrrr_hierarchy`) -- rather than each walking its own
    children and calling the builder itself.  The two chains reach it
    from opposite ends of the codebase and their bundles are consumed by
    ONE runner, so a forked emission loop would be two corridor formats
    wearing one schema.

    ``grids`` is positionally aligned with ``exp.domains``, which is the
    convention both callers already hold for the reference grids they
    pass to the artifact writer.  Returns the set receipt for the caller
    to bind into its preparation document, or ``None`` when the
    selection is empty -- in which case nothing is written and the
    bundle is byte-for-byte what it would have been.
    """
    grid_ids = validated_corridor_selection(exp, statics_corridor)
    if not grid_ids:
        return None
    index_by_id = {int(domain.grid_id): index
                   for index, domain in enumerate(exp.domains)}

    def builds():
        # A GENERATOR, so the writer seals each child's corridor before
        # the next is built and host memory holds one corridor at a
        # time.  A list here held every child's fields until the last
        # was built: a 3 km, a 1 km and a 500 m corridor at once.
        for grid_id in grid_ids:
            child = exp.domains[index_by_id[grid_id]]
            parent_run = exp.domains[index_by_id[int(child.parent_id)]].run
            plan = planned_corridor(exp, child)
            yield build_child_statics_corridor(
                child_dc=child, parent_run=parent_run,
                reference_grid=grids[index_by_id[grid_id]],
                static_catalog=static_catalog,
                frame_kwargs=plan.frame_kwargs, window=plan.window,
                reach=plan.reach,
                **({} if static_highres is None
                   else {"static_highres": static_highres}))
    return write_statics_corridor_set(Path(directory), builds())


def _write_deterministic_npz(path: Path,
                             fields: Mapping[str, np.ndarray]) -> None:
    """A byte-deterministic NPZ: same arrays in, same file bytes out.

    ``np.savez`` stamps each member with the wall clock, so two
    preparations of identical inputs would carry different digests --
    unacceptable for an artifact whose determinism is itself a test
    surface.  Members are written STORED with a pinned timestamp; the
    result reads back through ``np.load`` unchanged.
    """
    from numpy.lib import format as npy_format

    path = Path(path)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}.npz")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for name in sorted(fields):
                buffer = io.BytesIO()
                npy_format.write_array(
                    buffer,
                    np.ascontiguousarray(np.asarray(fields[name],
                                                    dtype=np.float64)),
                    allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy",
                                       date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_statics_corridor_set(directory: Path,
                               builds) -> dict[str, object]:
    """Seal one corridor per build under ``directory`` and return the
    set receipt (also written as ``receipt.json`` beside the caches).

    The returned document is what the preparation embeds in its proof;
    the on-disk copy must equal it byte-for-semantic-byte, which the
    runner verifies before any corridor byte is trusted.

    ``builds`` may be a generator: each build is sealed and released
    before the next is drawn, so a set costs host memory for its largest
    corridor rather than for all of them.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(
            f"refusing to overwrite statics corridor set {directory}")
    domains: dict[str, dict] = {}
    for build in builds:
        label = f"d{int(build.grid_id):02d}"
        if label in domains:
            raise ValueError(f"duplicate statics corridor for {label}")
        directory.mkdir(parents=True, exist_ok=True)
        cache_path = directory / f"{label}.npz"
        _write_deterministic_npz(cache_path, build.fields)
        entry = dict(build.entry)
        del build
        entry["cache"] = {
            "path": cache_path.name,
            "bytes": cache_path.stat().st_size,
            "sha256": _sha256(cache_path),
        }
        domains[label] = entry
    if not domains:
        raise ValueError("a statics corridor set requires at least one "
                         "child corridor")
    receipt = {
        "schema": STATICS_CORRIDOR_SET_SCHEMA,
        "status": "READY",
        "purpose": (
            "child-resolution statics over the ground each child can "
            "reach, sealed at preparation so the prepared tree route can honor "
            "[relocation] follow sources by cropping footprint statics "
            "instead of refusing them"),
        "runtime_memory": (
            "disk artifact, loaded to host memory by the runner "
            "preflight; cropped per relocation; zero GPU residency"),
        "domains": domains,
    }
    receipt_path = directory / STATICS_CORRIDOR_RECEIPT
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
        + "\n", encoding="utf-8")
    return receipt


# ---------------------------------------------------------------------------
# Runtime side: verify, load, crop
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChildStaticsCorridor:
    """One verified, loaded corridor; ``crop`` is its only consumer API."""

    geometry: Mapping[str, object]
    fields: Mapping[str, np.ndarray]
    cache_sha256: str
    highres_applied: bool = False

    @property
    def host_bytes(self) -> int:
        return int(sum(np.asarray(value).nbytes
                       for value in self.fields.values()))

    def crop(self, i_parent_start: int, j_parent_start: int
             ) -> dict[str, np.ndarray]:
        """The child-footprint statics at one placement, cropped.

        The slice arithmetic mirrors :func:`corridor_geometry`: a
        placement at ``(ip, jp)`` starts at corridor 0-based cell
        ``((ip-1)*ratio, (jp-1)*ratio)``.  Off-corridor placements
        refuse -- they name a footprint outside the parent, which the
        move admissibility walk should never produce.
        """
        ratio = int(self.geometry["parent_grid_ratio"])
        # A sub-1 placement lands at a negative origin, which crop_at
        # refuses by the same bounds test as an over-run: one refusal for
        # "not inside the corridor", whichever edge it left.
        return self.crop_at((int(i_parent_start) - 1) * ratio,
                            (int(j_parent_start) - 1) * ratio)

    def crop_at(self, x0: int, y0: int) -> dict[str, np.ndarray]:
        """The footprint statics at a CHILD-CELL origin in the corridor.

        The general address, and the only one a domain under a moving
        ancestor can use: its origin is a whole number of its OWN cells
        from the frame, never a whole number of its parent's (see
        :func:`origin_in_frame_cells`).  :meth:`crop` is this with the
        parent-cell address converted for it.
        """
        geometry = self.geometry
        nx = int(geometry["child_nx"])
        ny = int(geometry["child_ny"])
        x0, y0 = int(x0), int(y0)
        wx0, wy0, wnx, wny = geometry_window(geometry)
        lx, ly = x0 - wx0, y0 - wy0
        if lx < 0 or ly < 0 or lx + nx > wnx or ly + ny > wny:
            raise CorridorRefusal(
                f"footprint origin ({x0}, {y0}) + {nx}x{ny} child cells "
                f"of frame d{int(geometry.get('frame_grid_id', -1)):02d} "
                f"lies outside the statics corridor, which covers child "
                f"cells {wx0}..{wx0 + wnx - 1} x {wy0}..{wy0 + wny - 1}; "
                "no statics exist for that footprint.  The runner keeps a "
                "nest inside its reach, so this is a wiring defect or a "
                "corridor prepared for a different reach")
        result: dict[str, np.ndarray] = {}
        for name, value in self.fields.items():
            cropped = np.ascontiguousarray(
                np.asarray(value)[..., ly:ly + ny, lx:lx + nx])
            cropped.setflags(write=False)
            result[name] = cropped
        return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorridorRefusal(message)


def _verify_legacy_child_statics(corridor, child_dc, sealed_child_statics,
                                 label):
    """Check the first move's shared ground against this tree's own bytes.

    Later moves compare crops from the same corridor. A newer builder is
    not an authority for fields sealed together by an older preparation.
    """
    from gpuwm.core.nest_relocation import Placement, plan_relocation
    from gpuwm.ingest.relocation_init import (
        OVERLAP_STATIC_EQUALITY_FIELDS, overlap_statics_mismatches,
    )

    remedy = (
        "Rebuild with the original source input arguments and a new output "
        "directory: rw-wps --source <source> --wps-namelist <namelist.wps> "
        "--geog-root <WPS_GEOG> --experiment-config <experiment.toml> "
        f"--output-root <new-prepared-root> {STATICS_CORRIDOR_FLAG}")

    def refuse(reason):
        raise CorridorRefusal(
            f"{label} statics corridor cannot preserve the first relocation's "
            f"statics on shared ground: {reason}. {remedy}")

    if not isinstance(sealed_child_statics, Mapping):
        refuse("sealed child statics are missing; the prepared footprint "
               "has no reference fields to compare")
    missing = sorted(set(OVERLAP_STATIC_EQUALITY_FIELDS)
                     - set(sealed_child_statics))
    if missing:
        refuse(f"sealed child statics are missing fields {missing}")
    geometry = corridor.geometry
    if int(geometry["frame_grid_id"]) == int(child_dc.parent_id):
        crop = corridor.crop(child_dc.i_parent_start, child_dc.j_parent_start)
    else:
        # corridor_frame_kwargs derives this validated prepared origin via
        # origin_in_frame_cells, as the mover does for its live placements.
        crop = corridor.crop_at(*geometry["reference_origin_child_cells"])
    placement = Placement(
        grid_id=int(child_dc.grid_id),
        i_parent_start=int(child_dc.i_parent_start),
        j_parent_start=int(child_dc.j_parent_start))
    identity = plan_relocation(
        placement_from=placement, placement_to=placement,
        parent_grid_ratio=int(child_dc.parent_grid_ratio),
        child_nx=int(child_dc.run.nx), child_ny=int(child_dc.run.ny))
    comparison = overlap_statics_mismatches(
        sealed_child_statics, crop, identity,
        names=OVERLAP_STATIC_EQUALITY_FIELDS)
    if not comparison["pass"]:
        refuse("corridor crop at the prepared placement differs from "
               "the sealed child statics; mismatched cell counts by field "
               f"{comparison['mismatched_fields']}")
    if comparison["within_one_ulp"]:
        print(f"{label} statics corridor matches sealed child statics "
              f"within one ULP: {comparison['within_one_ulp']}",
              file=sys.stderr)



def load_child_statics_corridor(
        directory: Path, *, expected_set_receipt: Mapping[str, object],
        grid_id: int, child_dc, parent_run,
        reference_grid, frame_kwargs=None,
        sealed_child_statics: Mapping[str, np.ndarray] | None = None,
        required_window: tuple[int, int, int, int] | None = None,
        reach: Mapping[str, object] | None = None) -> ChildStaticsCorridor:
    """Verify one child's corridor against the preparation document and
    load it.

    ``expected_set_receipt`` is the proof-embedded set receipt, already
    covered by the preparation digest the runner pinned; the on-disk
    receipt must equal it, the cache digest must match the entry, and
    the entry's geometry must be the experiment's own -- each failure is
    a loud refusal, never a fall-back to a static nest.

    A legacy entry is checked against ``sealed_child_statics`` at the
    prepared placement using the first relocation's overlap rule. No
    geography or fresh static build is needed, including for overlays.

    ``required_window`` is the ground this run's nest can reach
    (:func:`planned_corridor`); ``None`` requires the whole frame.  The
    sealed corridor must COVER it, not equal it: a corridor prepared for
    a longer run, or a whole-frame one sealed before reach windows
    existed, serves a run whose reach it contains, and every crop reads
    the same bytes either would (the window build is the frame build's
    cells, bit for bit).  One that does not cover it is refused with both
    windows named, before the run starts rather than at the move that
    would have found no statics.
    """
    directory = Path(directory)
    label = f"d{int(grid_id):02d}"
    _require(directory.is_dir(),
             f"statics corridor directory is missing: {directory}")
    receipt_path = directory / STATICS_CORRIDOR_RECEIPT
    _require(receipt_path.is_file(),
             f"statics corridor receipt is missing: {receipt_path}")
    try:
        on_disk = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorridorRefusal(
            f"statics corridor receipt is not readable JSON: "
            f"{receipt_path}") from exc
    if on_disk != expected_set_receipt:
        raise CorridorRefusal(
            "statics corridor receipt differs from the copy sealed in "
            "the preparation document; the corridor cannot be trusted "
            "and the move is refused (re-prepare the tree)")
    domains = expected_set_receipt.get("domains")
    _require(isinstance(domains, Mapping) and label in domains,
             f"the preparation's statics corridor covers "
             f"{sorted(domains) if isinstance(domains, Mapping) else '?'} "
             f"but [relocation] names {label}; re-prepare with "
             f"{STATICS_CORRIDOR_FLAG} covering {label}")
    entry = domains[label]
    _require(entry.get("schema") == STATICS_CORRIDOR_SCHEMA
             and entry.get("status") == "READY",
             f"{label} statics corridor entry is not a READY "
             f"{STATICS_CORRIDOR_SCHEMA} document")
    try:
        sealed_window = geometry_window(entry)
        geometry = corridor_geometry(child_dc, parent_run,
                                     **(frame_kwargs or {}),
                                     window=sealed_window)
    except (KeyError, TypeError, ValueError) as exc:
        raise CorridorRefusal(
            f"{label} statics corridor window is not one this experiment's "
            f"frame can hold ({exc}); the corridor was prepared for a "
            "different tree") from exc
    for key, value in geometry.items():
        _require(entry.get(key) == value,
                 f"{label} statics corridor {key} = {entry.get(key)!r} "
                 f"differs from the experiment's {value!r}; the corridor "
                 "was prepared for a different tree")
    full = corridor_geometry(child_dc, parent_run, **(frame_kwargs or {}))
    need = (tuple(int(v) for v in required_window)
            if required_window is not None else geometry_window(full))
    sx0, sy0, snx, sny = sealed_window
    rx0, ry0, rnx, rny = need
    if not (sx0 <= rx0 and sy0 <= ry0 and rx0 + rnx <= sx0 + snx
            and ry0 + rny <= sy0 + sny):
        bounded_by = sorted({str(m.get("bounded_by"))
                             for m in (reach or {}).get("movers", ())})
        raise CorridorRefusal(
            f"{label} statics corridor covers child cells "
            f"{sx0}..{sx0 + snx - 1} x {sy0}..{sy0 + sny - 1} of frame "
            f"d{int(geometry['frame_grid_id']):02d}, but this run's nest "
            f"can reach {rx0}..{rx0 + rnx - 1} x {ry0}..{ry0 + rny - 1}"
            + (f" (bounded by {', '.join(bounded_by)})" if bounded_by
               else "")
            + "; a move there would find no statics.  It was prepared "
            "for a shorter run or a smaller reach: re-prepare the tree "
            f"with {STATICS_CORRIDOR_FLAG} for this experiment")
    _require(entry.get("landuse") == dict(NATIVE_LANDUSE_IDENTITY),
             f"{label} statics corridor land-use identity "
             f"{entry.get('landuse')!r} differs from the native contract "
             f"{dict(NATIVE_LANDUSE_IDENTITY)!r}")
    grid = corridor_grid(reference_grid, geometry)
    probe_drift = grid_probe_drift(entry.get("grid_identity_probes"), grid)
    _require(not probe_drift,
             f"{label} statics corridor grid probes differ from this "
             f"run's reconstructed corridor grid ({probe_drift}); the "
             "corridor was prepared on a grid that lies somewhere else")

    cache = entry.get("cache")
    _require(isinstance(cache, Mapping) and isinstance(
        cache.get("path"), str), f"{label} statics corridor entry lacks "
        "its cache binding")
    cache_path = directory / cache["path"]
    _require(cache_path.is_file(),
             f"{label} statics corridor cache is missing: {cache_path}")
    observed = _sha256(cache_path)
    if observed != cache.get("sha256"):
        raise CorridorRefusal(
            f"{label} statics corridor cache digest mismatch: "
            f"{cache_path} has sha256 {observed}, the preparation sealed "
            f"{cache.get('sha256')}; the corridor is corrupt or "
            "substituted, and the move is refused rather than run "
            "silently static")
    with np.load(cache_path, allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name], dtype=np.float64)
                  for name in archive.files}
    _validate_corridor_fields(fields, geometry)
    _require(sorted(fields) == list(entry.get("fields", ())),
             f"{label} statics corridor field inventory differs from its "
             "receipt")
    for value in fields.values():
        value.setflags(write=False)
    corridor = ChildStaticsCorridor(
        geometry=geometry, fields=fields, cache_sha256=observed,
        highres_applied=entry.get("highres", {}).get("status") == "APPLIED")
    if entry.get("build_contract") != STATICS_CORRIDOR_BUILD_CONTRACT:
        _verify_legacy_child_statics(
            corridor, child_dc, sealed_child_statics, label)
    return corridor


def corridor_footprint_statics_builder(corridor: ChildStaticsCorridor):
    """The relocation initializer's statics seam, corridor-backed.

    Satisfies :func:`gpuwm.ingest.relocation_init
    .real_relocation_initializer`'s ``statics_builder`` contract: called
    with the footprint's translated grid and DomainConfig, returns the
    footprint's static fields.  The grid is not consulted for values --
    every byte comes from the sealed corridor -- but its placement must
    agree with the crop, which the initializer's own drift gate already
    pinned to the parent-resolved placement.
    """

    def build(grid, new_dc):
        del grid  # placement equality is enforced by the drift gate
        return corridor.crop(int(new_dc.i_parent_start),
                             int(new_dc.j_parent_start))

    build.static_provenance = CORRIDOR_REBUILT_STATICS
    build.source_label = (
        f"statics-corridor d{int(corridor.geometry['grid_id']):02d} "
        f"sha256:{corridor.cache_sha256[:12]}")
    build.highres_applied = corridor.highres_applied
    return build


__all__ = [
    "CORRIDOR_BYTES_PER_CELL", "CORRIDOR_PLANES_PER_CELL",
    "CORRIDOR_REBUILT_STATICS", "CORRIDOR_STRIP_FILL_SOURCE",
    "ChildStaticsCorridor", "CorridorBuild", "CorridorRefusal",
    "STATICS_CORRIDOR_BUILD_CONTRACT",
    "STATICS_CORRIDOR_DIRNAME", "STATICS_CORRIDOR_FLAG",
    "STATICS_CORRIDOR_RECEIPT", "STATICS_CORRIDOR_SCHEMA",
    "STATICS_CORRIDOR_SET_SCHEMA", "build_child_statics_corridor",
    "config_declares_follow_source", "corridor_cost",
    "ancestor_chain", "corridor_frame_kwargs",
    "corridor_footprint_statics_builder", "corridor_geometry",
    "corridor_reach", "corridor_summary_line", "geometry_window",
    "PlannedCorridor", "planned_corridor", "planned_corridor_cost",
    "moving_grid_ids", "origin_in_frame_cells",
    "relocating_subtree_grid_ids",
    "corridor_grid", "emit_statics_corridor_set", "grid_identity_probes",
    "grid_probe_drift",
    "load_child_statics_corridor", "validated_corridor_selection",
    "write_statics_corridor_set",
]
