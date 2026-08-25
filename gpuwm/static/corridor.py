"""Sealed child-resolution statics corridor for prepared moving nests.

The prepared (digest-bound) tree routes deliberately run WITHOUT their
ingest inputs, so a relocating nest had nothing to rebuild its statics
from and the tree runner refused ``[relocation]`` follow sources
outright.  The corridor closes that gap at PREPARATION time, when the
geography source IS on hand: statics are pre-built at CHILD resolution
over the WHOLE PARENT EXTENT (a discrete relocation cannot leave its
parent), sealed beside the other hierarchy artifacts, and their digest
is bound into the preparation document exactly like every other sealed
artifact.  At runtime a relocation CROPS the new footprint's statics out
of the corridor -- no runtime ingest, no loosening of the digest relay.

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

#: Receipt-stated provenance for statics a relocation crops from the
#: corridor (the prepared-route counterpart of
#: :data:`gpuwm.ingest.relocation_init.REAL_DATA_FOOTPRINT_REBUILT_STATICS`).
CORRIDOR_REBUILT_STATICS = (
    "footprint statics cropped from the sealed child-resolution statics "
    "corridor (parent-extent build_static emitted at preparation time, "
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
    return bool(relocation.enabled
                and (relocation.follow is not None or relocation.moves))


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
                      reference_origin: tuple[int, int] | None = None
                      ) -> dict[str, object]:
    """The corridor's placement-independent geometry for one child.

    The corridor covers a FRAME domain's full mass extent at child
    resolution, and is addressed in CHILD cells from frame cell 1.

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
    return {
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
        "corridor_nx": int(frame_run.nx) * ratio_to_frame,
        "corridor_ny": int(frame_run.ny) * ratio_to_frame,
        # The exact whole-cell translation from the child's reference
        # grid to the corridor origin (frame cell 1).
        "origin_translation_child_cells": [-origin_i, -origin_j],
    }


def moving_grid_ids(exp) -> frozenset[int]:
    """Grid ids a ``[relocation]`` block authorises to move.

    The tracked mover, plus the ``[relocation.containment]`` ancestor
    when one is configured -- the ancestor slides in whole cells of ITS
    parent, so everything downstream of this answer (corridor coverage,
    root-anchored frames for children of a mover) must count it as a
    mover in its own right.
    """
    relocation = getattr(exp, "relocation", None)
    if relocation is None or not getattr(relocation, "enabled", False):
        return frozenset()
    grid_id = getattr(relocation, "grid_id", None)
    if grid_id is None:
        return frozenset()
    movers = {int(grid_id)}
    containment = getattr(relocation, "containment", None)
    if containment is not None:
        movers.add(int(containment.grid_id))
    return frozenset(movers)


def relocating_subtree_grid_ids(exp) -> tuple[int, ...]:
    """Every grid whose GROUND changes when this experiment moves.

    The mover plus all of its descendants, ascending.  A mid-tree move
    re-grounds the whole subtree -- each member rebuilds its statics for
    new ground -- so each member needs its own corridor, and this is the
    one place that says which.  For a leaf mover it is the single grid id
    it has always been, so nothing about a leaf bundle changes.
    """
    movers = moving_grid_ids(exp)
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


def corridor_cost(child_dc, parent_run, frame_kwargs=None) -> dict[str, int]:
    """What one child's corridor will cost, WITHOUT building it.

    Pure arithmetic on the geometry and the field inventory: no GEOG
    source, no ``build_static`` call, nothing on disk.  That is what
    lets a front door price a corridor before the preparation runs --
    the figure a caller sees ahead of launch and the ``host_bytes`` the
    sealed receipt reports afterwards come out of the same two facts
    (cell count and plane count), and
    ``tests/test_statics_corridor.py`` holds them equal against a real
    build rather than against each other.

    ``host_bytes`` is the loaded footprint AND, to within the container
    headers, the on-disk one: the cache is an uncompressed
    (``ZIP_STORED``) NPZ of exactly these arrays.
    """
    geometry = corridor_geometry(child_dc, parent_run)
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


def corridor_grid(reference_grid, geometry: Mapping[str, object]):
    """The corridor's grid: the reference child grid, translated and
    re-extented on the SAME lattice (per-cell reference arithmetic)."""
    di, dj = geometry["origin_translation_child_cells"]
    return reference_grid.translated(
        int(di), int(dj),
        e_we=int(geometry["corridor_nx"]) + 1,
        e_sn=int(geometry["corridor_ny"]) + 1)


def grid_identity_probes(grid) -> dict[str, list[float]]:
    """Exact float64 lat/lon probes pinning the grid arithmetic.

    Recorded at preparation and required to match the runner's
    reconstructed corridor grid to the BYTE, the same posture as the
    per-domain MAPFAC regeneration gate: preparation-machine and
    run-machine grid arithmetic must be identical, or the corridor's
    bitwise claim has no floor under it.
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
                                 frame_kwargs=None) -> CorridorBuild:
    """Build one child's frame-extent statics corridor from the GEOG
    source, through the same builder the domain statics come from."""
    from gpuwm.static.build import build_static, geog_selection_from_catalog

    geometry = corridor_geometry(child_dc, parent_run, **(frame_kwargs or {}))
    grid = corridor_grid(reference_grid, geometry)
    selection = geog_selection_from_catalog(
        static_catalog, int(child_dc.grid_id))
    coverage: dict[str, object] = {}
    fields = build_static(grid, selection.root, selection=selection,
                          source_coverage_report=coverage)
    _validate_corridor_fields(fields, geometry)
    landuse = selection.landuse_global_attrs()
    entry = {
        "schema": STATICS_CORRIDOR_SCHEMA,
        "status": "READY",
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
        # Tile-presence proof over the whole parent extent, summarized
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
                              statics_corridor):
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
    builds = []
    for grid_id in grid_ids:
        child = exp.domains[index_by_id[grid_id]]
        parent_run = exp.domains[index_by_id[int(child.parent_id)]].run
        builds.append(build_child_statics_corridor(
            child_dc=child, parent_run=parent_run,
            reference_grid=grids[index_by_id[grid_id]],
            static_catalog=static_catalog,
            frame_kwargs=corridor_frame_kwargs(exp, child)))
    return write_statics_corridor_set(Path(directory), builds)


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
    """
    directory = Path(directory)
    builds = list(builds)
    if not builds:
        raise ValueError("a statics corridor set requires at least one "
                         "child corridor")
    if directory.exists():
        raise FileExistsError(
            f"refusing to overwrite statics corridor set {directory}")
    directory.mkdir(parents=True)
    domains: dict[str, dict] = {}
    for build in builds:
        label = f"d{int(build.grid_id):02d}"
        if label in domains:
            raise ValueError(f"duplicate statics corridor for {label}")
        cache_path = directory / f"{label}.npz"
        _write_deterministic_npz(cache_path, build.fields)
        entry = dict(build.entry)
        entry["cache"] = {
            "path": cache_path.name,
            "bytes": cache_path.stat().st_size,
            "sha256": _sha256(cache_path),
        }
        domains[label] = entry
    receipt = {
        "schema": STATICS_CORRIDOR_SET_SCHEMA,
        "status": "READY",
        "purpose": (
            "child-resolution statics over the whole parent extent, "
            "sealed at preparation so the prepared tree route can honor "
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
        if (x0 < 0 or y0 < 0
                or x0 + nx > int(geometry["corridor_nx"])
                or y0 + ny > int(geometry["corridor_ny"])):
            raise CorridorRefusal(
                f"footprint origin ({x0}, {y0}) + {nx}x{ny} child cells "
                f"lies outside the statics corridor "
                f"({geometry['corridor_nx']}x{geometry['corridor_ny']} "
                f"child cells over grid "
                f"d{int(geometry.get('frame_grid_id', -1)):02d}); an "
                "admissible move cannot reach here, so this is a wiring "
                "defect or a corridor prepared for a narrower frame")
        result: dict[str, np.ndarray] = {}
        for name, value in self.fields.items():
            cropped = np.ascontiguousarray(
                np.asarray(value)[..., y0:y0 + ny, x0:x0 + nx])
            cropped.setflags(write=False)
            result[name] = cropped
        return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorridorRefusal(message)


def load_child_statics_corridor(
        directory: Path, *, expected_set_receipt: Mapping[str, object],
        grid_id: int, child_dc, parent_run,
        reference_grid, frame_kwargs=None) -> ChildStaticsCorridor:
    """Verify one child's corridor against the preparation document and
    load it.

    ``expected_set_receipt`` is the proof-embedded set receipt, already
    covered by the preparation digest the runner pinned; the on-disk
    receipt must equal it, the cache digest must match the entry, and
    the entry's geometry must be the experiment's own -- each failure is
    a loud refusal, never a fall-back to a static nest.
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

    geometry = corridor_geometry(child_dc, parent_run, **(frame_kwargs or {}))
    for key, value in geometry.items():
        _require(entry.get(key) == value,
                 f"{label} statics corridor {key} = {entry.get(key)!r} "
                 f"differs from the experiment's {value!r}; the corridor "
                 "was prepared for a different tree")
    _require(entry.get("landuse") == dict(NATIVE_LANDUSE_IDENTITY),
             f"{label} statics corridor land-use identity "
             f"{entry.get('landuse')!r} differs from the native contract "
             f"{dict(NATIVE_LANDUSE_IDENTITY)!r}")
    grid = corridor_grid(reference_grid, geometry)
    _require(entry.get("grid_identity_probes") == grid_identity_probes(grid),
             f"{label} statics corridor grid probes differ from this "
             "run's reconstructed corridor grid; preparation-time and "
             "run-time grid arithmetic disagree, so the corridor's "
             "bitwise contract does not hold here")

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
    return ChildStaticsCorridor(
        geometry=geometry, fields=fields, cache_sha256=observed)


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
    build.highres_applied = False
    return build


__all__ = [
    "CORRIDOR_BYTES_PER_CELL", "CORRIDOR_PLANES_PER_CELL",
    "CORRIDOR_REBUILT_STATICS", "CORRIDOR_STRIP_FILL_SOURCE",
    "ChildStaticsCorridor", "CorridorBuild", "CorridorRefusal",
    "STATICS_CORRIDOR_DIRNAME", "STATICS_CORRIDOR_FLAG",
    "STATICS_CORRIDOR_RECEIPT", "STATICS_CORRIDOR_SCHEMA",
    "STATICS_CORRIDOR_SET_SCHEMA", "build_child_statics_corridor",
    "config_declares_follow_source", "corridor_cost",
    "ancestor_chain", "corridor_frame_kwargs",
    "corridor_footprint_statics_builder", "corridor_geometry",
    "moving_grid_ids", "origin_in_frame_cells",
    "relocating_subtree_grid_ids",
    "corridor_grid", "emit_statics_corridor_set", "grid_identity_probes",
    "load_child_statics_corridor", "validated_corridor_selection",
    "write_statics_corridor_set",
]
