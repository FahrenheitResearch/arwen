"""Multi-GPU domain decomposition: each GPU owns a sub-domain, halos move.

This is the sibling of :mod:`tilestream.driver`.  ``run_tiled`` streams every
tile through one GPU; here each GPU PERMANENTLY OWNS a slab of the domain and
only the ``halo``-wide seams cross the PCIe link.  Halo traffic scales with the
perimeter, compute with the area, so the byte count is roughly ``2*halo/T`` of
what streaming the whole tile costs.

THE HEADLINE, so it is not buried in the tables below: this is BIT-EXACT
against a monolithic run, and on 2x RTX 4090 it is **1.99x at 1536^2** and
**1.62x at 768^2**.  The seam is not the problem at two cards -- the exposed
exchange is 4.2 ms/step at 768^2 and 8.1 ms at 1536^2, which is 6% and 3% of
the two-GPU step respectively, a low single-digit percentage at every size
measured.  What limits the small end is halo recompute and a fixed ~3.8 ms
host-side concurrency loss, both of which shrink as the domain grows; see the
attribution table below, which factors all three effects.

THREE FACTS A READER NEEDS BEFORE PLANNING ANYTHING WITH THIS
-------------------------------------------------------------
1. **GeForce has no working P2P.**  ``canAccessPeer == 0`` on 40-series, so
   every halo byte routes through host memory whether you ask it to or not.
   ``cudaMemcpyPeerAsync`` still works and CUDA stages it through its own host
   buffer; that is the transport used here and nothing in this module
   hand-rolls device->host->device.  The measured seam rate, ~10 GB/s per
   direction, IS that staged path -- it is not a P2P number and must not be
   quoted as one, and a datacentre card with real NVLink would not be
   predicted by it in either direction.

2. **Ring mode does NOT parallelise across GPUs.**  :mod:`tilestream.rings`
   removes the shadow store's second copy of the domain, which is the right
   trade for the single-GPU out-of-core lane and is worth nothing here: it was
   measured at 1.00x across all 12 geometries tried, because the ring is a
   memory optimisation on the WRITE path of one sweep and the thing multi-GPU
   needs is two sweeps at once.  Worse, a ring plan makes every already-written
   tile's band a dependency of the next tile, which is exactly the serialising
   structure a decomposition is trying to get rid of.  So multi-GPU wants
   ``write_mode="shadow"``, and ``run_tiled`` refusing a caller-supplied
   shadow under ``write_mode="ring"`` (asserted in the gate) is the guard rail
   that keeps the two from being confused.

3. **At eight cards the exchange IS the scaling loss, and it is 6.1x off the
   PCIe floor.**  An independent reimplementation of this decomposition --
   written from scratch on a rented 8-GPU box, because this file had never
   been committed and could not be found there -- measured 3.72x at 8 GPUs
   with 18/18 configurations bit-exact.  It agrees with this module on the
   PHYSICS and is not a measurement of this code.  What it establishes that
   two cards cannot is where the ceiling comes from: attributing its loss put
   essentially all of it in the exchange, running 6.1x slower than the PCIe
   bandwidth floor for the bytes moved.  That is a transport result, not a
   decomposition result -- the halo volume is what the geometry says it is --
   and it is the number to attack first before anything about the split is
   changed.

PROVENANCE.  This file was never committed.  It lived as a loose file on a
rented box's disk until it was recovered and committed byte-identical; the
commit immediately before this docstring is that rescue.  Every number below
was measured with THIS code on the box named in the heading.

Nothing about the numerics changes.  A sub-domain is stepped exactly the way a
tile is stepped -- as a smaller periodic domain, under its own ``RunConfig``,
with a ``halo``-cell surround whose contents are thrown away and refreshed --
so every fact the single-GPU lane established still holds and is not re-argued
here:

* the halo is ``harness.halo_radius(cfg)`` (16 at ``time_step_sound=4``);
* exactly ONE step happens between exchanges (two on a 16-cell halo fails);
* only the persisted inventory has to move.


THE GEOMETRY, drawn.  ``nx = 8T``-ish, two GPUs, periodic in x
--------------------------------------------------------------
The domain's mass columns, and who owns them::

    logical x:  0 ....................... T ....................... nx
                |<------ GPU 0 owns ----->|<------ GPU 1 owns ----->|

Each GPU's ARRAY is its interior grown by ``h`` on both sides.  Because the
domain is periodic there are TWO seams, not one -- the interior seam at ``T``
and the wrap seam at ``0``/``nx`` -- and each GPU's two halos BOTH come from
the other GPU, from opposite ends of its interior::

    GPU 0 array   [ left halo ][    interior [0,T)    ][ right halo ]
    covers        [ nx-h, nx )                          [ T, T+h )
                        ^                                    ^
                        |  wrap seam                         |  interior seam
                        |                                    |
    GPU 1 array   [ left halo ][   interior [T,nx)    ][ right halo ]
    covers        [ T-h, T )                            [ nx, nx+h ) = [0, h)
                                 ^                  ^
                                 |                  |
                    feeds GPU 0's right halo    feeds GPU 0's left halo

So GPU 0's LEFT halo is fed by the RIGHT end of GPU 1's interior, and GPU 0's
RIGHT halo by the LEFT end of GPU 1's interior.  Getting those two crossed is
the failure this module's ``verify_geometry`` exists to catch: it compares
every sub-domain array, cell for cell, against the window of a monolithic run,
so a crossed seam shows up as a mismatch confined to one halo band instead of
as a slightly wrong forecast.

The vertical is NEVER split.  Whether y is split is up to ``grid=(gy, gx)``:
with ``gy == 1`` a sub-domain has the parent's full ``ny``, so its own periodic
wrap in y IS the domain's and no y halo exists, and only ``nx`` differs between
the parent config and a sub-config.  ``grid=(n, 1)`` is the same picture rotated
a quarter turn -- the seam bands become whole rows instead of column strips and
the staggered variant at the seam is ``v`` (``ny+1`` rows) instead of ``u``
(``nx+1`` columns), which is a genuinely different set of strides and a
different alias slot, so the gate runs both.

``grid=(gy, gx)`` with BOTH above 1 adds corners: a sub-domain's corner block
(inside its x halo and its y halo at once) belongs to the DIAGONAL neighbour and
no single band reaches it.  Rather than a fifth band per corner, the exchange
runs in two rounds -- x seams over the sub-array's full height, then y seams
over its full width, which by then includes the x halo the first round made
exact.  The corner arrives in two hops.  ``seam_plan`` documents the ordering
and every exchange mode enforces it.


STAGGERING AT THE SEAM
----------------------
``u`` has ``nx+1`` faces and the last one is the periodic alias of face 0
(:mod:`tilestream.spec` calls it the alias slot).  A sub-domain of width
``W = T + 2h`` therefore carries ``W+1`` u slots, and the halo refresh must
cover ``h+1`` of them on the right (slots ``h+T .. W`` inclusive), not ``h``.
That extra slot is filled from LOGICAL face ``ci0 + W`` reduced mod ``nx``,
which is exactly what ``TileSpec.gather("u")`` puts there in the single-GPU
lane -- the scheme that is bit-exact.  ``v`` is staggered in y only, so at an
x seam it moves the same x-range as a mass field, over ``ny+1`` rows.

Reads never touch a source array's own alias slot: the source index is derived
from a logical index reduced mod ``nx`` and always lands inside the neighbour's
INTERIOR, which is the only part of the neighbour that is trustworthy after a
step.


WHAT IS REUSED, AND THE ONE THING THAT IS NOT
---------------------------------------------
The geometry is :class:`tilestream.spec.TileSpec`, built directly rather than
through ``plan_tiles`` because the split is x-only and a sub-domain must keep
the parent's full ``ny`` (``plan_tiles`` grows the halo on both axes).
``spec.validate_plan``, ``TileSpec.gather`` and ``TileSpec.scatter`` then apply
unchanged, and they are what loads the sub-domains and reassembles them.  The
config is ``harness.tile_config``; the buffer is ``driver.make_tile_state``;
the stagger classification is ``gather.classify``; the halo is
``harness.halo_radius``.

The SEAM transport is the exception.  ``gather.gather_tile`` moves a
full-domain array to a tile array and ``build_plan`` checks the two sides
against exactly that relationship -- a seam has a sub-domain on BOTH ends, so
that check rejects it before any bytes move.  And a halo band is
``nz*ny`` rows of ``halo*4 = 64`` bytes: issued as-is over a link that stages
through host, it is a latency test, not a transfer.  So the seam packs into
one contiguous staging buffer per band with device-local ``cudaMemcpy2DAsync``
(0.35-0.50 ms, measured), crosses in ONE ``cudaMemcpyPeerAsync``, and unpacks
the same way.


ORDERING: why the exchange cannot be moved
------------------------------------------
After a step, a sub-domain's interior is exact and its halo is garbage (it is
the sub-domain's own periodic wrap-around, not the neighbour).  The refresh
must therefore read the neighbour's interior AT THE SAME TIME LEVEL -- i.e.
after the neighbour has stepped too.  That is a barrier, and it is not
negotiable: the read-at-time-t rule from :mod:`tilestream.driver` applies here
verbatim.  What CAN overlap is pack/transfer/unpack against the other GPU's
remaining compute, and the two directions against each other; see
``exchange_mode="events"``.

There is no interior/halo split of the step itself, so "exchange while the
interiors compute" is NOT available without carving up ``dycore.step``.  This
module does not pretend otherwise.


MEASURED ON THIS BOX (2x RTX 4090, driver 580.95.05, CUDA 13.0, cupy 14.1.1).
Reproduce any line with ``python -m tilestream.multigpu <what>``.
--------------------------------------------------------------------------
``gate`` is green for all four (step, exchange) pipelines and all three
transports; ``controls`` fires on all four deliberate defects -- at its own
step default, which is longer than the positive gates' and is not negotiable
downwards, see ``SHORT_HALO_VISIBLE_AT``; ``geometry`` shows every sub-domain
array, halo included, identical to the monolithic window.  Wall clock for a whole domain, two GPUs versus one -- median of 7,
``events``/``events`` (the ``threads`` pipeline is 1-3% quicker but its spread
reaches 8% at 768^2, so the steady one is quoted)::

    domain     1 GPU     2 GPU      vs 1 GPU   no-exchange   exposed exchange
    768^2    120.20 ms   74.38 ms    x1.616     x1.711        4.2 ms /  74 MiB
    1024^2   218.77 ms  124.99 ms    x1.750     x1.836        5.8 ms /  99 MiB
    1280^2   364.93 ms  187.75 ms    x1.944     x2.023        7.4 ms / 124 MiB
    1536^2   550.02 ms  276.17 ms    x1.992     x2.052        8.1 ms / 149 MiB

THREE effects, and they do not all point the same way.  Factored by measuring
one sub-domain ALONE on one card and comparing (``run_phase2.py attrib``)::

    domain   full 1-GPU   1 sub-domain alone   halo    concurrency  exchange
    768^2      120.18 ms       66.16 ms       1.0833x    1.0564x     1.1048x
    1024^2     218.92 ms      114.76 ms       1.0625x    1.0346x     1.0334x
    1280^2     363.59 ms      176.72 ms       1.0500x    1.0223x     1.0259x
    1536^2     548.97 ms      264.78 ms       1.0417x    1.0131x     1.0223x

* HALO RECOMPUTE, against: each GPU steps ``(T + 2h)/T`` cells it needs, and
  throws the overlap away.  Shrinks as the domain grows.
* CONCURRENCY LOSS, against: two cards busy at once cost more than one card
  alone by a FIXED ~3.8 ms per step (3.73 / 3.97 / 3.94 / 3.46 ms across the
  four sizes), not a fixed fraction.  That constant is the host side -- CPython
  serialises the two step-launch sequences whichever way they are driven --
  so it fades as the step gets longer, and it is why ``threads`` barely beats
  ``interleaved``.
* PER-CELL COST, FOR: the step costs more per cell on a bigger domain
  (4.16 ns/cell at 512^2, 4.26 at 1024^2, 4.75 at 1536^2), so halving the
  domain also moves both cards into a cheaper regime.  At 1280^2 one
  672x1280 sub-domain alone runs in 176.72 ms against 363.59 for the whole
  thing -- better than half, despite computing 5% more cells; at 1536^2 it is
  264.78 against 548.97, a 2.07x edge before concurrency is even involved.

Below ~1024^2 the first two win and the pair gives ~1.6-1.75x; by 1536^2 the
third has taken over and it reaches ~2.0x.  Do not extrapolate either end:
the crossover is a property of this card's cache and occupancy, not of the
decomposition.

The seam itself costs ~10 GB/s per direction, measured by ``exchange``
(74 MiB each way at 1536^2 in 7.75 ms) -- P2P is off, so CUDA stages through
host, and this is that path, not a hand-rolled one.


TRAPS CHECKED
-------------
* ``gpuwm.core.kernels.load_module`` is ``@lru_cache``d on the source, so both
  devices share one ``RawModule`` object.  CuPy keeps a per-device module map
  inside ``RawModule``, and this was MEASURED: a state compiled on device 0 and
  stepped on device 1 gives a hash identical to device 0's.  No per-device
  cache key is needed.
* CuPy memory pools are per device.  Every allocation here happens inside an
  explicit ``cp.cuda.Device`` context, including the seam staging buffers.
* P2P is unavailable on GeForce 40-series (``canAccessPeer == 0``).
  ``cudaMemcpyPeerAsync`` still works -- CUDA stages through its own host
  buffer -- and that is the transport used.  Nothing here hand-rolls
  device->host->device.
* Every cross-device copy is fenced with events on both devices before any
  result is read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
import gc
import hashlib
import threading
import time

import numpy as np

from tilestream import driver as _driver
from tilestream import gather as _gather
from tilestream import harness as _harness
from tilestream import spec as _spec


__all__ = [
    "MultiGPUDomain",
    "MultiGPUError",
    "SeamCopy",
    "bench",
    "bench_exchange",
    "bench_monolithic",
    "compare_hosts",
    "describe_plan",
    "download_state",
    "forced_config",
    "forced_domain_inputs",
    "forced_halo",
    "forced_state_factory",
    "gate",
    "gate_forced",
    "hash_host",
    "negative_controls",
    "plan_split",
    "plan_split_x",
    "plan_split_y",
    "seam_plan",
    "validate_forced_plan",
    "verify_geometry",
    "window_geography",
]


class MultiGPUError(RuntimeError):
    """A decomposition that cannot be right."""


# ==========================================================================
# geometry
# ==========================================================================

def plan_split(nx: int, ny: int, halo: int, gx: int = 2, gy: int = 1,
               periodic: bool = True, *,
               periodic_x: bool | None = None,
               periodic_y: bool | None = None) -> list[_spec.TileSpec]:
    """One :class:`tilestream.spec.TileSpec` per sub-domain, on a ``gy x gx`` grid.

    Row-major order (``ty`` outer, ``tx`` inner), the same order
    ``spec.plan_tiles`` uses.

    An axis with ONE sub-domain is NOT SPLIT: its compute window is the whole
    domain extent (``ci0 = 0``, ``cnx = nx``), so the sub-domain's own periodic
    wrap on that axis IS the domain's wrap and there is neither a halo nor a
    seam there.  An axis with several sub-domains gets interiors that partition
    it as evenly as the division allows, each grown by ``halo`` on both sides.
    ``gx=ngpu, gy=1`` is the x split; ``gx=1, gy=ngpu`` the y split; both > 1
    is a 2-D decomposition, whose corners are filled by the two-phase exchange
    described in :func:`seam_plan`.

    ``periodic`` sets both axes; ``periodic_x`` / ``periodic_y`` override one,
    because the axes are independent in the model itself
    (``dycore._boundary_x`` / ``_boundary_y``) and a plan that clamps an axis
    the kernels wrap is the measured mgstream defect.  On a NON-PERIODIC axis
    a sub-domain that touches the domain edge gets NO halo on that side -- the
    array edge IS the domain edge, which is what lets the resident lateral-
    boundary machinery (specified or open) find the real boundary at the real
    place -- and there is no wrap seam, so the edge sides exchange nothing.
    Interior seams are unchanged.  This is :mod:`tilestream.realcase`'s
    clamped-plan rule, carried onto the resident decomposition.

    Reusing ``TileSpec`` rather than inventing a second geometry type is the
    point: ``spec.gather`` / ``spec.scatter`` / ``spec.validate_plan`` then
    apply unchanged, and the staggering and periodic wrap are handled by the
    code the single-GPU gate already validated.
    """
    nx, ny, halo = int(nx), int(ny), int(halo)
    gx, gy = int(gx), int(gy)
    px = bool(periodic if periodic_x is None else periodic_x)
    py = bool(periodic if periodic_y is None else periodic_y)
    if gx < 1 or gy < 1:
        raise ValueError(f"grid must be positive, got gy={gy} gx={gx}")
    # ``gx == gy == 1`` is the IDENTITY PLAN: one sub-domain, no halo, no
    # seam, compute window == domain.  It is not a decomposition and it buys
    # no parallelism, and refusing it looked like a way to catch a caller who
    # had miscounted their devices.  It is not: it removes the ONE arm that
    # separates "the decomposition is wrong" from "driving this state through
    # MultiGPUDomain at all is wrong", because a 1x1 domain steps the SAME
    # cells under the SAME config as the monolithic reference and must
    # therefore reproduce it bit for bit.  Every branch below already handles
    # it -- the per-axis rule two paragraphs up leaves an unsplit axis whole,
    # ``seam_plan`` emits no band when neither axis is split, and
    # ``validate_plan`` covers the domain exactly once with a single spec.
    if gx * gy < 1:
        raise ValueError(f"grid must be positive, got gy={gy} gx={gx}")
    if halo < 0:
        raise ValueError(f"halo must be >= 0, got {halo}")

    xb = [(t * nx) // gx for t in range(gx)] + [nx]
    yb = [(t * ny) // gy for t in range(gy)] + [ny]
    for axis, bounds, count in (("x", xb, gx), ("y", yb, gy)):
        if count == 1:
            continue
        widths = [bounds[k + 1] - bounds[k] for k in range(count)]
        if min(widths) < halo + 1:
            raise MultiGPUError(
                f"sub-domain interiors {widths} along {axis} are too narrow "
                f"for halo={halo}: a seam must be served by ONE neighbour's "
                f"interior, which needs every interior >= halo+1 = "
                f"{halo + 1} cells.  Use a wider domain or fewer GPUs.")

    def _pads(t: int, count: int, wraps: bool) -> tuple[int, int]:
        """Halo widths (low side, high side) for sub-domain ``t`` of ``count``.

        A split periodic axis grows every interior by ``halo`` on both sides.
        A split NON-PERIODIC axis grows only the sides that face a
        neighbour: the domain-edge side keeps zero halo, so the array edge
        coincides with the domain edge and the boundary treatment lands on
        real boundary cells.  An unsplit axis has no halo at all.
        """
        if count == 1:
            return 0, 0
        if wraps:
            return halo, halo
        return (halo if t > 0 else 0), (halo if t < count - 1 else 0)

    specs: list[_spec.TileSpec] = []
    for ty in range(gy):
        j0, j1 = yb[ty], yb[ty + 1]
        lo_y, hi_y = _pads(ty, gy, py)
        for tx in range(gx):
            i0, i1 = xb[tx], xb[tx + 1]
            lo_x, hi_x = _pads(tx, gx, px)
            specs.append(_spec.TileSpec(
                ty=ty, tx=tx, nx=nx, ny=ny,
                tile_nx=i1 - i0, tile_ny=j1 - j0,
                halo=halo, periodic_x=px, periodic_y=py,
                i0=i0, i1=i1, j0=j0, j1=j1,
                ci0=(i0 - lo_x) if gx > 1 else 0,
                cj0=(j0 - lo_y) if gy > 1 else 0,
                cnx=((i1 - i0) + lo_x + hi_x) if gx > 1 else nx,
                cny=((j1 - j0) + lo_y + hi_y) if gy > 1 else ny,
            ))
    return specs


def forced_halo(cfg) -> int:
    """The halo a SPECIFIED (boundary-forced) decomposition needs.

    :func:`tilestream.realcase.halo_for`'s quarantine arithmetic, reused
    rather than restated: the per-step dependency radius plus
    ``max(spec_zone, relax_zone)``, because on a seam side the boundary
    kernel writes that many cells of fiction into the window before the step
    starts to propagate, and the halo is what keeps every owned cell out of
    its reach.  See realcase.py's module docstring for the proof shape.
    """
    from tilestream import realcase as _realcase

    return int(_realcase.halo_for(cfg))


def _boundary_forced_cfg(cfg) -> bool:
    return bool(getattr(cfg, "specified", False))


def validate_forced_plan(cfg, specs, halo: int, boundaries, *,
                         enforce_halo: bool = True) -> None:
    """Refuse a forced decomposition that cannot be right.  Pure host logic.

    Raises :class:`MultiGPUError` with the defect NAMED and the remedy in the
    message; called by ``MultiGPUDomain.__init__`` and directly testable
    without a GPU.
    """
    forced = _boundary_forced_cfg(cfg)
    if getattr(cfg, "nested", False):
        raise MultiGPUError(
            "cfg.nested=True: nested forcing is not wired through the "
            "multi-GPU decomposition; run the child resident or streamed, "
            "or use external specified forcing")
    if forced and boundaries is None:
        raise MultiGPUError(
            "cfg.specified=True but no lateral forcing was given: pass "
            "boundaries=<gpuwm.ingest.lateral_bc.LateralBoundaries> so each "
            "rank can be attached to its windowed tables.  A specified "
            "domain without forcing has no defined edge values.")
    if boundaries is not None and not forced:
        raise MultiGPUError(
            "boundaries were given but cfg.specified is False: the ranks "
            "would attach forcing the step never applies.  Set "
            "cfg.specified=True or drop boundaries.")
    if not forced:
        return
    need = forced_halo(cfg)
    if enforce_halo and int(halo) < need:
        raise MultiGPUError(
            f"halo={int(halo)} is below the forced-decomposition radius "
            f"{need} (= dependency radius {int(_harness.halo_radius(cfg))} + "
            f"max(spec_zone={int(cfg.spec_zone)}, "
            f"relax_zone={int(cfg.relax_zone)})): a seam-side boundary "
            "application writes that many cells of fiction before the step "
            "propagates it, so a narrower halo lets it reach owned cells.  "
            "Use halo=None (the default derives the padded radius).")
    if (int(boundaries.spec_zone) != int(cfg.spec_zone)
            or int(boundaries.relax_zone) != int(cfg.relax_zone)):
        raise MultiGPUError(
            f"boundaries carry spec_zone={boundaries.spec_zone}/"
            f"relax_zone={boundaries.relax_zone} but cfg says "
            f"{cfg.spec_zone}/{cfg.relax_zone}; the Davies weights would be "
            "built for one geometry and applied to the other.  Rebuild the "
            "forcing with the config's zones.")
    active = max(int(cfg.spec_zone), int(cfg.relax_zone))
    for g, s in enumerate(specs):
        if min(int(s.cny), int(s.cnx)) <= 2 * active:
            raise MultiGPUError(
                f"sub-domain {g} array is {s.cny}x{s.cnx}, too narrow for "
                f"the specified/relaxation frame width {active} (needs both "
                "extents > twice that): the boundary kernel would have no "
                "unique interior.  Use fewer ranks or a wider domain.")


def plan_split_x(nx: int, ny: int, halo: int, ngpu: int = 2,
                 periodic: bool = True) -> list[_spec.TileSpec]:
    """``ngpu`` sub-domains side by side in x; y is not split.  See :func:`plan_split`."""
    if int(ngpu) < 2:
        raise ValueError(f"ngpu must be >= 2, got {ngpu}")
    return plan_split(nx, ny, halo, gx=int(ngpu), gy=1, periodic=periodic)


def plan_split_y(nx: int, ny: int, halo: int, ngpu: int = 2,
                 periodic: bool = True) -> list[_spec.TileSpec]:
    """``ngpu`` sub-domains stacked in y; x is not split.

    The transpose of :func:`plan_split_x`, and worth having as its own gate
    row: the seam bands are contiguous runs of whole rows rather than strided
    column strips, and the staggered variant at the seam is ``v``
    (``ny+1`` rows) rather than ``u`` (``nx+1`` columns).  Different strides,
    different alias slot, same arithmetic -- which is exactly the kind of pair
    where a transposed index survives one orientation and not the other.
    """
    if int(ngpu) < 2:
        raise ValueError(f"ngpu must be >= 2, got {ngpu}")
    return plan_split(nx, ny, halo, gx=1, gy=int(ngpu), periodic=periodic)


def _owner_x(specs, ty: int, logical: int, nx: int) -> int:
    """Index of the sub-domain in grid row ``ty`` owning mass column ``logical``."""
    c = logical % nx
    for g, s in enumerate(specs):
        if s.ty == ty and s.i0 <= c < s.i1:
            return g
    raise MultiGPUError(f"logical column {logical} (={c} mod {nx}) is unowned "
                        f"in grid row {ty}")


def _owner_y(specs, tx: int, logical: int, ny: int) -> int:
    """Index of the sub-domain in grid column ``tx`` owning mass row ``logical``."""
    c = logical % ny
    for g, s in enumerate(specs):
        if s.tx == tx and s.j0 <= c < s.j1:
            return g
    raise MultiGPUError(f"logical row {logical} (={c} mod {ny}) is unowned "
                        f"in grid column {tx}")


@dataclass(frozen=True)
class SeamCopy:
    """One halo band: ``dst_gpu``'s ``side`` halo, filled from ``src_gpu``.

    ``dst_x`` / ``src_x`` (``axis == "x"``) or ``dst_y`` / ``src_y``
    (``axis == "y"``) are per-variant slices into the two sub-domain ARRAYS,
    not the full domain.  The OTHER axis is never sliced: an x band spans every
    row the sub-array has and a y band spans every column, which is what makes
    the two-phase order fill the corners (see :func:`seam_plan`) and what keeps
    every band a single pitched 2-D copy.
    """

    dst_gpu: int
    src_gpu: int
    side: str                        # left|right (x) or south|north (y)
    dst_x: dict | None               # variant -> slice, None = whole extent
    src_x: dict | None
    logical: tuple                   # (first, last) logical mass index
    dst_y: dict | None = None
    src_y: dict | None = None
    axis: str = "x"
    phase: int = 0                   # exchange round; x is 0, y is 1

    @property
    def dst_sel(self) -> dict:
        return self.dst_x if self.axis == "x" else self.dst_y

    @property
    def src_sel(self) -> dict:
        return self.src_x if self.axis == "x" else self.src_y

    def describe(self) -> str:
        stag = "u" if self.axis == "x" else "v"
        m_d, m_s = self.dst_sel["mass"], self.src_sel["mass"]
        s_d, s_s = self.dst_sel[stag], self.src_sel[stag]
        return (f"gpu{self.dst_gpu}.{self.side:<5s} halo <- gpu{self.src_gpu}  "
                f"[{self.axis} phase {self.phase}]  "
                f"mass dst[{m_d.start}:{m_d.stop}] src[{m_s.start}:{m_s.stop}]"
                f"  {stag} dst[{s_d.start}:{s_d.stop}] "
                f"src[{s_s.start}:{s_s.stop}]  "
                f"logical mass [{self.logical[0]}, {self.logical[1]})")


def _axis_seam(specs, a, side: str, halo: int, nx: int, ny: int) -> SeamCopy:
    """One band of sub-domain ``a``, on whichever axis ``side`` names.

    Halo widths are read off the SPEC (``halo_left``/``halo_right``/
    ``halo_south``/``halo_north``) rather than assumed symmetric, because a
    non-periodic edge sub-domain has NO halo on its domain-edge side and its
    array is narrower by exactly that much.  A side whose halo is zero has no
    band; :func:`seam_plan` never asks for one.
    """
    sa = specs[a]
    x_axis = side in ("left", "right")
    sel_d: dict = {}
    sel_s: dict = {}
    src_gpus = set()
    logical = None
    n_axis = nx if x_axis else ny
    for variant in _spec.VARIANTS:
        ey, ex = _spec.stagger(variant)
        extra = ex if x_axis else ey            # 1 for u on x, v on y
        if side in ("left", "south"):
            width = sa.halo_left if x_axis else sa.halo_south
            d0, d1 = 0, width
            first = sa.ci0 if x_axis else sa.cj0        # logical index of d0
        else:
            width = sa.halo_right if x_axis else sa.halo_north
            interior = sa.interior_nx if x_axis else sa.interior_ny
            off = sa.halo_left if x_axis else sa.halo_south
            d0, d1 = off + interior, off + interior + width + extra
            first = sa.i1 if x_axis else sa.j1
        if width <= 0:
            raise MultiGPUError(
                f"seam gpu{a}.{side} has a zero-width halo: that side is a "
                "domain edge, not a seam, and no band exists there")
        n = d1 - d0
        b = (_owner_x(specs, sa.ty, first, nx) if x_axis
             else _owner_y(specs, sa.tx, first, ny))
        sb = specs[b]
        base = sb.i0 if x_axis else sb.j0
        b_off = sb.halo_left if x_axis else sb.halo_south
        s0 = ((first - base) % n_axis) + b_off          # local index in b
        b_interior = sb.interior_nx if x_axis else sb.interior_ny
        if s0 < b_off or s0 + n > b_off + b_interior:
            raise MultiGPUError(
                f"seam gpu{a}.{side} variant {variant!r} wants gpu{b} local "
                f"{'x' if x_axis else 'y'}[{s0}:{s0 + n}] but only "
                f"[{b_off}:{b_off + b_interior}] (its interior) is exact "
                f"after a step; the halo is too wide for these sub-domains")
        sel_d[variant] = slice(d0, d1)
        sel_s[variant] = slice(s0, s0 + n)
        src_gpus.add(b)
        if variant == "mass":
            logical = (first, first + n)
    if len(src_gpus) != 1:
        raise MultiGPUError(
            f"seam gpu{a}.{side} spans several neighbours {src_gpus}")
    b = src_gpus.pop()
    if x_axis:
        return SeamCopy(a, b, side, sel_d, sel_s, logical, axis="x", phase=0)
    return SeamCopy(a, b, side, None, None, logical, dst_y=sel_d, src_y=sel_s,
                    axis="y", phase=1)


def seam_plan(specs, halo: int, nx: int | None = None,
              ny: int | None = None) -> list[SeamCopy]:
    """Every halo band in the decomposition, two per split axis per sub-domain.

    Derived from LOGICAL coordinates and reduced mod ``nx``/``ny`` at the end,
    so the periodic wrap is arithmetic rather than a special case.  Raises if a
    band would need data from more than one neighbour or from a neighbour's
    halo (only a neighbour's INTERIOR is trustworthy after a step).

    TWO PHASES, AND THE ORDER IS WHAT FILLS THE CORNERS.  With both axes split,
    a sub-domain's corner (its x halo AND its y halo at once) is owned by the
    DIAGONAL neighbour, and no direct band reaches it.  Rather than add a fifth
    band per corner, phase 0 exchanges the x seams over the FULL height of the
    sub-array and phase 1 exchanges the y seams over its FULL width::

        phase 0:  ...x halo... <- neighbour's interior columns, all rows
                  (a corner is written here from data that is still garbage:
                   the source's own y halo has not been refreshed yet)
        phase 1:  ...y halo... <- neighbour's interior rows, ALL columns
                  including the neighbour's x halo, which phase 0 has just
                  made exact -- so the corner is now the diagonal neighbour's
                  interior, arrived in two hops.

    Phase 1 must therefore not start packing until phase 0 has finished
    UNPACKING on the source device.  Every exchange mode enforces that; the
    event pipeline does it with a per-device phase barrier rather than a sync.
    A 1-D split has only one phase and none of this applies.
    """
    halo = int(halo)
    nx = int(specs[0].nx if nx is None else nx)
    ny = int(specs[0].ny if ny is None else ny)
    gx = 1 + max(s.tx for s in specs)
    gy = 1 + max(s.ty for s in specs)
    out: list[SeamCopy] = []
    # A side with a ZERO halo is a true domain edge of a non-periodic axis:
    # nothing lies beyond it, nothing is exchanged across it, and the lateral
    # boundary (specified or open) owns its cells.  On a periodic axis every
    # side of a split has a halo, so this filter changes nothing there.
    if gx > 1:
        for a in range(len(specs)):
            if specs[a].halo_left > 0:
                out.append(_axis_seam(specs, a, "left", halo, nx, ny))
            if specs[a].halo_right > 0:
                out.append(_axis_seam(specs, a, "right", halo, nx, ny))
    if gy > 1:
        for a in range(len(specs)):
            if specs[a].halo_south > 0:
                out.append(_axis_seam(specs, a, "south", halo, nx, ny))
            if specs[a].halo_north > 0:
                out.append(_axis_seam(specs, a, "north", halo, nx, ny))
    return out


def cross_seams(seams, specs, halo: int) -> list[SeamCopy]:
    """THE NEGATIVE CONTROL: every band reads the WRONG END of its neighbour.

    A left/south halo should come from the FAR end of the neighbour's interior
    and a right/north halo from its NEAR end.  Swapping them keeps every shape,
    every byte count and every device pairing identical -- so nothing but the
    arithmetic notices -- which is exactly why it is the control worth running:
    it is the first bug anyone writes on a periodic split, and a gate that
    cannot see it cannot see anything.
    """
    halo = int(halo)
    out: list[SeamCopy] = []
    for sc in seams:
        sp = specs[sc.src_gpu]
        interior = sp.interior_nx if sc.axis == "x" else sp.interior_ny
        off = sp.halo_left if sc.axis == "x" else sp.halo_south
        flipped = {}
        for v, sl in sc.src_sel.items():
            n = sl.stop - sl.start
            flipped[v] = (slice(off, off + n)
                          if sc.side in ("left", "south")
                          else slice(off + interior - n, off + interior))
        key = "src_x" if sc.axis == "x" else "src_y"
        out.append(_dc_replace(sc, **{key: flipped}))
    return out


def describe_plan(specs, seams) -> str:
    lines = ["sub-domains:"]
    for g, s in enumerate(specs):
        lines.append(f"  gpu{g}: grid({s.ty},{s.tx}) interior "
                     f"y[{s.j0}:{s.j1}) x[{s.i0}:{s.i1}) "
                     f"({s.interior_ny}x{s.interior_nx} cells)  array "
                     f"y[{s.cj0}:{s.cj0 + s.cny}) x[{s.ci0}:{s.ci0 + s.cnx}) "
                     f"(ny={s.cny}, nx={s.cnx})")
    lines.append("seams:")
    for sc in seams:
        lines.append("  " + sc.describe())
    return "\n".join(lines)


# ==========================================================================
# the seam transport
# ==========================================================================

@dataclass
class _Band:
    """One field's contribution to one seam band, as raw pitched bytes."""
    name: str
    variant: str
    rows: int
    width: int          # bytes per row
    src_pitch: int      # bytes
    dst_pitch: int      # bytes
    src_off: int        # bytes from the source array base
    dst_off: int        # bytes from the destination array base
    buf_off: int        # bytes into the staging buffer

    @property
    def nbytes(self) -> int:
        return self.rows * self.width


def _variant_of(name: str, arr, nz: int, ny: int, nx: int) -> str:
    return _gather.classify(arr.shape, nz, ny, nx, layers_ok=True)


def _bands(seam: SeamCopy, src_arrays, dst_arrays, nz,
           src_spec, dst_spec, names) -> tuple[list[_Band], int]:
    """Pitched-copy descriptors for one seam, one per field.

    Both orientations are a single 2-D pitched copy over a C-contiguous
    ``(..., H, W)`` array, because each band is full-extent on the axis it does
    NOT cut:

    * an x band cuts x and spans every row, so its unit is a row: ``rows`` is
      every leading index times ``H``, the run is ``n * itemsize`` wide and the
      pitch is one row, ``W * itemsize``;
    * a y band cuts y and spans every column, so its unit is a whole
      HORIZONTAL SLICE: ``rows`` is the leading count alone, the run is
      ``n * W * itemsize`` (n contiguous rows) and the pitch is one slice,
      ``H * W * itemsize``.

    So the y seam moves the same bytes in far fewer, far longer runs, and a
    transposed index cannot survive both -- which is why the gate runs both.
    """
    bands: list[_Band] = []
    off = 0
    x_axis = seam.axis == "x"
    for name in names:
        s = src_arrays[name]
        d = dst_arrays[name]
        if s.ndim < 2:
            continue                    # vertical-only: identical on every GPU
        v_s = _variant_of(name, s, nz, src_spec.cny, src_spec.cnx)
        v_d = _variant_of(name, d, nz, dst_spec.cny, dst_spec.cnx)
        if v_s != v_d:
            raise MultiGPUError(
                f"{name!r} classifies as {v_s!r} on gpu{seam.src_gpu} but "
                f"{v_d!r} on gpu{seam.dst_gpu}")
        if s.dtype != d.dtype:
            raise MultiGPUError(f"{name!r} dtype {s.dtype} vs {d.dtype}")
        if not (s.flags.c_contiguous and d.flags.c_contiguous):
            raise MultiGPUError(f"{name!r} is not C-contiguous; the pitched "
                                "copy assumes it is")
        if s.shape[:-2] != d.shape[:-2]:
            raise MultiGPUError(
                f"{name!r} leading shape {s.shape[:-2]} != {d.shape[:-2]}; "
                "only the horizontal extents may differ between sub-domains")
        # The axis a band does NOT cut must match on both ends, because the
        # band spans all of it.  It always does -- an x seam joins two
        # sub-domains in the same grid ROW (same cny), a y seam two in the same
        # grid COLUMN (same cnx) -- so this is an assertion, not a fixup.
        keep = -2 if x_axis else -1
        if s.shape[keep] != d.shape[keep]:
            raise MultiGPUError(
                f"{name!r} spans {s.shape[keep]} on gpu{seam.src_gpu} but "
                f"{d.shape[keep]} on gpu{seam.dst_gpu} along the axis the "
                f"{seam.axis} seam does not cut")
        it = s.dtype.itemsize
        sl_s, sl_d = seam.src_sel[v_s], seam.dst_sel[v_d]
        n = sl_d.stop - sl_d.start
        if sl_s.stop - sl_s.start != n:
            raise MultiGPUError("seam slice lengths disagree")
        cut = -1 if x_axis else -2
        if sl_s.stop > s.shape[cut] or sl_d.stop > d.shape[cut]:
            raise MultiGPUError(
                f"{name!r} seam slice src[{sl_s.start}:{sl_s.stop}] / "
                f"dst[{sl_d.start}:{sl_d.stop}] escapes shapes {s.shape} / "
                f"{d.shape} along {seam.axis}")
        if x_axis:
            rows = int(np.prod(s.shape[:-1]))
            width = n * it
            src_pitch, dst_pitch = int(s.shape[-1]) * it, int(d.shape[-1]) * it
            src_off, dst_off = int(sl_s.start) * it, int(sl_d.start) * it
        else:
            rows = int(np.prod(s.shape[:-2])) if s.ndim > 2 else 1
            width = n * int(s.shape[-1]) * it
            src_pitch = int(s.shape[-2]) * int(s.shape[-1]) * it
            dst_pitch = int(d.shape[-2]) * int(d.shape[-1]) * it
            src_off = int(sl_s.start) * int(s.shape[-1]) * it
            dst_off = int(sl_d.start) * int(d.shape[-1]) * it
        bands.append(_Band(
            name=name, variant=v_s, rows=rows, width=width,
            src_pitch=src_pitch, dst_pitch=dst_pitch,
            src_off=src_off, dst_off=dst_off, buf_off=off))
        off += rows * width
    return bands, off


def peer_access_matrix(devices) -> dict:
    """``cudaDeviceCanAccessPeer`` for every ordered pair of ``devices``.

    The driver's answer, asked rather than assumed.  It is the only thing that
    decides whether ``cudaMemcpyPeerAsync`` crosses the link or is staged
    through host memory by CUDA, and both cases return success, so a transport
    arm that does not ask this cannot say which path it measured.
    """
    from cupy.cuda import runtime as rt

    out = {}
    for a in devices:
        for b in devices:
            if int(a) == int(b):
                continue
            out[(int(a), int(b))] = bool(
                rt.deviceCanAccessPeer(int(a), int(b)))
    return out


# What each transport does when P2P is available and when it is not.  Keyed by
# (transport, canAccessPeer).  ``legs`` counts host round trips, which is what
# separates a one-leg measurement from an end-to-end one.
_TRANSPORT_PATHS = {
    ("peer", True): ("peer-to-peer over the link", 1),
    ("peer", False): ("no P2P between these devices, CUDA-staged through host",
                      1),
    ("default", True): ("peer-to-peer over the link, chosen by memcpyDefault",
                        1),
    ("default", False): ("no P2P between these devices, CUDA-staged through "
                         "host by memcpyDefault", 1),
    ("host", True): ("explicit pinned-host staging, D2H then H2D", 2),
    ("host", False): ("explicit pinned-host staging, D2H then H2D", 2),
}


class _SeamChannel:
    """Staging buffers plus the pack/transfer/unpack for one seam band."""

    def __init__(self, seam: SeamCopy, src_dev: int, dst_dev: int,
                 bands: list[_Band], nbytes: int, transport: str):
        import cupy as cp

        self.seam = seam
        self.src_dev = int(src_dev)
        self.dst_dev = int(dst_dev)
        self.bands = bands
        self.nbytes = int(nbytes)
        self.transport = transport
        with cp.cuda.Device(self.src_dev):
            self.send = cp.empty(self.nbytes, dtype=cp.uint8)
            self.send_ptr = int(self.send.data.ptr)
        with cp.cuda.Device(self.dst_dev):
            self.recv = cp.empty(self.nbytes, dtype=cp.uint8)
            self.recv_ptr = int(self.recv.data.ptr)
        self.host = None
        if transport == "host":
            self.host = _gather.pinned_empty(self.nbytes, np.uint8)
            self.host_ptr = int(self.host.ctypes.data)

    # -- the three phases ---------------------------------------------------

    def pack(self, src_arrays, stream_ptr: int) -> None:
        """Gather the band out of the source arrays into one contiguous run.

        Device-local, on the source GPU.  Without it the cross-device copy
        would be ``nz*ny`` rows of ``halo*4 = 64`` bytes, which is not a
        transfer, it is a latency test.
        """
        from cupy.cuda import runtime as rt

        for b in self.bands:
            rt.memcpy2DAsync(
                self.send_ptr + b.buf_off, b.width,
                int(src_arrays[b.name].data.ptr) + b.src_off, b.src_pitch,
                b.width, b.rows, rt.memcpyDeviceToDevice, stream_ptr)

    def transfer(self, stream_ptr: int) -> None:
        from cupy.cuda import runtime as rt

        if self.transport == "peer":
            rt.memcpyPeerAsync(self.recv_ptr, self.dst_dev,
                               self.send_ptr, self.src_dev,
                               self.nbytes, stream_ptr)
        elif self.transport == "default":
            rt.memcpyAsync(self.recv_ptr, self.send_ptr, self.nbytes,
                           rt.memcpyDefault, stream_ptr)
        elif self.transport == "host":
            rt.memcpyAsync(self.host_ptr, self.send_ptr, self.nbytes,
                           rt.memcpyDeviceToHost, stream_ptr)
        else:
            raise ValueError(f"unknown transport {self.transport!r}")

    def transfer_finish(self, stream_ptr: int) -> None:
        """Second leg of the ``host`` transport; a no-op for the direct ones."""
        from cupy.cuda import runtime as rt

        if self.transport == "host":
            rt.memcpyAsync(self.recv_ptr, self.host_ptr, self.nbytes,
                           rt.memcpyHostToDevice, stream_ptr)

    def unpack(self, dst_arrays, stream_ptr: int) -> None:
        from cupy.cuda import runtime as rt

        for b in self.bands:
            rt.memcpy2DAsync(
                int(dst_arrays[b.name].data.ptr) + b.dst_off, b.dst_pitch,
                self.recv_ptr + b.buf_off, b.width,
                b.width, b.rows, rt.memcpyDeviceToDevice, stream_ptr)

    # -- the unpacked reference -------------------------------------------

    def direct_send(self, src_arrays, dst_arrays, stream_ptr: int) -> None:
        """Move this seam WITHOUT the staging buffer: one copy per field.

        The reference the packed path is measured and checked against.  It
        moves exactly the same bytes to exactly the same places -- the only
        difference is the transfer COUNT, ``len(self.bands)`` crossings of the
        link instead of one.  That is deliberate: pinning, transport and band
        geometry are all held fixed so a before/after difference can only be
        the count.

        NOT a shipping path.  It exists so "the pack is faster" and "the pack
        is bit-exact" are claims about a second implementation rather than
        about the same code timed twice.
        """
        from cupy.cuda import runtime as rt

        for b in self.bands:
            if self.transport == "host":
                # Two legs per FIELD, through the same pinned buffer the packed
                # path uses, at the same offset.  Pinning is held fixed.
                rt.memcpy2DAsync(
                    self.host_ptr + b.buf_off, b.width,
                    int(src_arrays[b.name].data.ptr) + b.src_off, b.src_pitch,
                    b.width, b.rows, rt.memcpyDeviceToHost, stream_ptr)
            else:
                # UVA makes a 2-D copy between two devices' pointers legal with
                # ``memcpyDefault``; this is the cross-device crossing itself.
                rt.memcpy2DAsync(
                    int(dst_arrays[b.name].data.ptr) + b.dst_off, b.dst_pitch,
                    int(src_arrays[b.name].data.ptr) + b.src_off, b.src_pitch,
                    b.width, b.rows, rt.memcpyDefault, stream_ptr)

    def direct_recv(self, dst_arrays, stream_ptr: int) -> None:
        """Second leg of ``direct_send`` under ``host``; a no-op otherwise."""
        from cupy.cuda import runtime as rt

        if self.transport != "host":
            return
        for b in self.bands:
            rt.memcpy2DAsync(
                int(dst_arrays[b.name].data.ptr) + b.dst_off, b.dst_pitch,
                self.host_ptr + b.buf_off, b.width,
                b.width, b.rows, rt.memcpyHostToDevice, stream_ptr)

    @property
    def n_transfers_packed(self) -> int:
        """Link crossings per exchange for this seam, packed."""
        return 2 if self.transport == "host" else 1

    @property
    def n_transfers_direct(self) -> int:
        """Link crossings per exchange for this seam, unpacked."""
        return len(self.bands) * (2 if self.transport == "host" else 1)


# ==========================================================================
# host-side helpers (setup, assembly, comparison)
# ==========================================================================

def download_state(state) -> dict:
    """``{name: numpy array}`` for the persisted inventory of ``state``."""
    import cupy as cp

    return {k: cp.asnumpy(v) for k, v in _harness.state_arrays(state).items()}


def hash_host(arrays: dict) -> str:
    """``harness.hash_state``'s digest, over a host-resident inventory.

    Same protocol -- name, dtype string, shape, C-order bytes, in
    ``STATE_SERIALIZED_ATTRS`` order -- so a host hash is directly comparable
    with a device one.
    """
    if not arrays:
        raise MultiGPUError("no arrays to hash")
    digest = hashlib.sha256()
    for name, arr in arrays.items():
        digest.update(_harness._digest_payload(f"state.{name}", arr))
    return digest.hexdigest()


def compare_hosts(a: dict, b: dict, nx: int) -> dict:
    """Per-field ``max|a-b|`` plus the x columns where they differ.

    The established differential: differences ringing the seam mean the halo
    geometry is wrong, differences everywhere mean a config or device-context
    problem, differences in one sub-domain only mean an asymmetric seam.
    """
    report: dict = {}
    for name in a:
        if name not in b:
            report[name] = {"missing": True}
            continue
        x, y = np.asarray(a[name]), np.asarray(b[name])
        if x.shape != y.shape:
            report[name] = {"shape": (x.shape, y.shape)}
            continue
        if x.dtype.kind not in "fc":
            if not np.array_equal(x, y):
                report[name] = {"exact": False}
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        m = float(d.max()) if d.size else 0.0
        if m == 0.0:
            continue
        axes = tuple(range(d.ndim - 1))
        percol = d.max(axis=axes) if d.ndim > 1 else d
        cols = np.nonzero(percol > 0)[0]
        report[name] = {
            "max_abs": m,
            "bad_cols": int(cols.size),
            "col_span": (int(cols.min()), int(cols.max())),
            "first_cols": cols[:12].tolist(),
            "last_cols": cols[-12:].tolist(),
        }
    return report


# ==========================================================================
# the decomposition
# ==========================================================================

class MultiGPUDomain:
    """``ngpu`` sub-domains, one per device, stepped in lock-step with halos.

    ``exchange_mode``
        ``"blocking"``   full device syncs around every phase.  Slowest, and
                         the reference the other two are checked against.
        ``"stream"``     per-device streams, one sync at the end.
        ``"events"``     each seam's pack/transfer/unpack chained by CUDA
                         events so a GPU's outbound copy starts the instant it
                         finishes stepping, without waiting for its neighbour.

    ``step_mode``
        ``"sequential"`` device 0's step is synchronised before device 1's is
                         launched.  No concurrency at all -- the STEP A
                         configuration, and the one that isolates decomposition
                         bugs from concurrency bugs.
        ``"interleaved"`` both steps launched from one thread, then both
                         synced.  ``dycore.step`` only launches, so the GPUs
                         run at the same time.
        ``"threads"``    one thread per device.
    """

    def __init__(self, cfg, *, ngpu: int = 2, grid=None, devices=None,
                 halo=None, state_factory=None, inventory_fn=None,
                 transport: str = "peer", _unsafe_short_halo: bool = False,
                 exchange_names=None, boundaries=None, seam: str = "zeros",
                 boundary_snapshot=None):
        import cupy as cp

        from gpuwm.core.streaming import _periodic_axes

        self.cfg = cfg
        self.nz, self.ny, self.nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
        # The plan's periodicity is the MODEL's, per axis, never a guess:
        # a plan that clamps an axis the kernels wrap (or wraps an axis the
        # lateral boundary owns) is the measured mgstream defect.
        self.periodic_x, self.periodic_y = _periodic_axes(cfg)
        self.forced = _boundary_forced_cfg(cfg)
        need = forced_halo(cfg) if self.forced \
            else int(_harness.halo_radius(cfg))
        self.halo = need if halo is None else int(halo)
        if self.halo < need and not _unsafe_short_halo:
            raise MultiGPUError(
                f"halo={self.halo} is below the per-step dependency radius "
                f"{need} for time_step_sound={cfg.time_step_sound}"
                + (" (forced: dependency radius + boundary frame width)"
                   if self.forced else "")
                + ".  A short halo is silently wrong AND faster; it is not "
                  "tunable on a short test.")
        # ``grid`` is (gy, gx).  The default is the x split the phase-one
        # lane built and measured; (n, 1) is the y split and (m, n) is 2-D.
        self.grid = (1, int(ngpu)) if grid is None \
            else (int(grid[0]), int(grid[1]))
        nsub = self.grid[0] * self.grid[1]
        self.devices = list(range(nsub)) if devices is None \
            else [int(d) for d in devices]
        self.ngpu = len(self.devices)
        if self.ngpu != nsub:
            raise MultiGPUError(
                f"grid {self.grid} needs {nsub} sub-domains but "
                f"{self.ngpu} devices were given: {self.devices}")
        avail = cp.cuda.runtime.getDeviceCount()
        for d in self.devices:
            if d >= avail:
                raise MultiGPUError(f"device {d} does not exist ({avail} seen)")

        self.specs = plan_split(self.nx, self.ny, self.halo,
                                gx=self.grid[1], gy=self.grid[0],
                                periodic_x=self.periodic_x,
                                periodic_y=self.periodic_y)
        _spec.validate_plan(self.specs, self.ny, self.nx)
        validate_forced_plan(cfg, self.specs, self.halo, boundaries,
                             enforce_halo=not _unsafe_short_halo)
        self.boundaries = boundaries
        self.seam = str(seam)
        self.seams = seam_plan(self.specs, self.halo, self.nx, self.ny)
        self.sub_cfgs = [_harness.tile_config(cfg, s.cnx, s.cny)
                         for s in self.specs]

        factory = state_factory or _driver.make_tile_state
        self.inventory_fn = inventory_fn or _harness.state_arrays
        # Re-derive the inventory every step only when it can MOVE.  The dry
        # persisted set never does (measured: identical pointers across 8
        # steps), and the phase-one timings were taken without the re-derive,
        # so the default path is left exactly as it was measured.
        self.volatile_inventory = (inventory_fn is not None)
        import inspect

        takes_spec = False
        try:
            takes_spec = "spec" in inspect.signature(factory).parameters
        except (TypeError, ValueError):        # builtins, C callables
            takes_spec = False
        self.states = []
        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                self.states.append(
                    factory(self.sub_cfgs[g], spec=self.specs[g])
                    if takes_spec else factory(self.sub_cfgs[g]))

        # A FORCED rank owns a WINDOW of the domain's lateral forcing: the
        # sides that are true domain edges keep the domain's own tables,
        # sliced tangentially to the rank's array (staggering included), and
        # interior-seam sides get INERT tables whose fiction the padded halo
        # quarantines.  This is gpuwm.core.streaming.window_boundaries --
        # the machinery the streamed specified lane measured bit-exact --
        # applied to the resident decomposition unchanged.
        self.rank_boundaries = None
        if boundaries is not None:
            from gpuwm.core.streaming import window_boundaries
            from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

            self.rank_boundaries = []
            for g, dev in enumerate(self.devices):
                windowed = window_boundaries(
                    boundaries, self.specs[g], seam=self.seam,
                    snapshot=boundary_snapshot)
                with cp.cuda.Device(dev):
                    attach_lateral_boundaries(self.states[g], windowed)
                self.rank_boundaries.append(windowed)

        # MEASURED HAZARD (tilestream.driver:562).  A non-blocking stream does
        # NOT synchronise with the legacy default stream, and DomainState's
        # constructor uploads every setup array on the default stream.  Without
        # this barrier the first stepped sub-domain reads setup arrays that
        # have not landed and comes out all-NaN.  Once per construction.
        for dev in self.devices:
            with cp.cuda.Device(dev):
                cp.cuda.runtime.deviceSynchronize()

        self.arrays = [self.inventory_fn(s) for s in self.states]
        self.names = list(self.arrays[0].keys())
        for g, inv in enumerate(self.arrays):
            if list(inv.keys()) != self.names:
                raise MultiGPUError(
                    f"gpu{g} inventory {sorted(inv)} differs from gpu0's "
                    f"{sorted(self.names)}")
            # A "two-GPU" run whose second sub-domain quietly allocated on
            # device 0 would still be bit-exact and would still look like a
            # speedup for a while.  CuPy's pools are per device and the
            # allocating device is whatever ``Device`` context was current, so
            # this is a real failure mode, not a theoretical one.  Check it.
            for name, arr in inv.items():
                if int(arr.device.id) != int(self.devices[g]):
                    raise MultiGPUError(
                        f"sub-domain {g} field {name!r} lives on device "
                        f"{arr.device.id}, not the device {self.devices[g]} "
                        "it was built under")
            got = (int(self.sub_cfgs[g].ny), int(self.sub_cfgs[g].nx))
            want = (int(self.specs[g].cny), int(self.specs[g].cnx))
            if got != want:
                raise MultiGPUError(
                    f"sub-domain {g} config says (ny, nx)={got} but its array "
                    f"is {want}; ~15 sites in the step path launch from "
                    "cfg.nx/cfg.ny, so this would step the wrong grid")

        # WHICH carriers get a seam.  ``None`` means every one of them,
        # which is what this module has always done and what every
        # measured row in the docstring was taken under.
        #
        # It is a parameter because the runner that produced the 154 Mcell
        # imagery selected a SMALLER set -- 3-D carriers only -- and a gate
        # that cannot express that set cannot certify it.  A gate must be
        # able to run the configuration that ships, not a safer neighbour
        # of it; see tilestream/SEAM-IN-SURFACE-PRESSURE.md.
        if exchange_names is None:
            self.exchange_names = list(self.names)
        else:
            self.exchange_names = [str(n) for n in exchange_names]
            unknown = sorted(set(self.exchange_names) - set(self.names))
            if unknown:
                raise MultiGPUError(
                    f"exchange_names names {unknown}, which are not in "
                    f"the inventory {sorted(self.names)}")
        self.transport = transport
        self._build_channels()

        self.compute_streams = []
        self.copy_streams = []
        self.unpack_streams = []
        for dev in self.devices:
            with cp.cuda.Device(dev):
                self.compute_streams.append(cp.cuda.Stream(non_blocking=True))
                self.copy_streams.append(cp.cuda.Stream(non_blocking=True))
                self.unpack_streams.append(cp.cuda.Stream(non_blocking=True))
        self._events = None

    # -- construction helpers ----------------------------------------------

    def _build_channels(self) -> None:
        self.channels = []
        for seam in self.seams:
            bands, nbytes = _bands(
                seam,
                self.arrays[seam.src_gpu], self.arrays[seam.dst_gpu],
                self.nz,
                self.specs[seam.src_gpu], self.specs[seam.dst_gpu],
                self.exchange_names)
            self.channels.append(_SeamChannel(
                seam, self.devices[seam.src_gpu], self.devices[seam.dst_gpu],
                bands, nbytes, self.transport))
        # Channels grouped by exchange round, with their index into the
        # per-channel event list kept alongside.  A 1-D split has one round; a
        # 2-D split has two and the SECOND one is what fills the corners, so
        # the rounds may never be merged or reordered (see ``seam_plan``).
        rounds = sorted({c.seam.phase for c in self.channels})
        self.channel_phases = [
            [(k, c) for k, c in enumerate(self.channels)
             if c.seam.phase == p]
            for p in rounds]
        self._events = None

    def refresh_arrays(self) -> None:
        """Re-derive each sub-domain's inventory, because SOME CARRIERS MOVE.

        MEASURED on this box (``+YSU PBL`` rung, 96x80x49): the six
        ``driver/pbl_tendencies/*`` carriers are a NEW device allocation after
        EVERY step -- the driver rebinds the attribute instead of writing in
        place, and the pointer changes again on the next step.  Everything the
        seam does is built on raw device pointers, and ``assemble_host`` reads
        whatever array the inventory dict holds, so an inventory captured once
        at construction packs, ships and returns a buffer that nothing has
        written since the buffer was built.

        That failure is silent and it does NOT look like a seam bug: it is
        UNIFORM, every column of those six carriers differing by up to 2.3e3,
        while the other 133 carriers stay bit-exact.  ``tilestream.driver``
        never sees it because it re-derives the inventory per tile.

        Shapes and dtypes must not change -- asserted here, because a changed
        shape would quietly move the wrong bytes instead of failing.
        """
        for g, state in enumerate(self.states):
            fresh = self.inventory_fn(state)
            old = self.arrays[g]
            if list(fresh.keys()) != self.names:
                raise MultiGPUError(
                    f"sub-domain {g}'s inventory changed identity mid-run: "
                    f"{sorted(set(fresh) ^ set(old))} appeared or vanished.  "
                    "The seam plan is built once and cannot follow that.")
            for name, arr in fresh.items():
                ref = old[name]
                if arr.shape != ref.shape or arr.dtype != ref.dtype:
                    raise MultiGPUError(
                        f"sub-domain {g} carrier {name!r} changed from "
                        f"{ref.shape}/{ref.dtype} to {arr.shape}/{arr.dtype} "
                        "mid-run; the seam's pitches were computed from the "
                        "old shape and would move the wrong bytes")
            self.arrays[g] = fresh

    @property
    def seam_bytes(self) -> int:
        """Bytes crossing the link per exchange, summed over every band."""
        return sum(c.nbytes for c in self.channels)

    def vram_bytes(self) -> list:
        return [sum(a.nbytes for a in inv.values()) for inv in self.arrays]

    # -- device bookkeeping -------------------------------------------------

    def sync_all(self) -> None:
        import cupy as cp

        for dev in self.devices:
            with cp.cuda.Device(dev):
                cp.cuda.runtime.deviceSynchronize()

    def impose_clock(self, seconds: float) -> None:
        """Set every rank's model clock to the same instant.

        ``dtbc`` -- the position inside the current forcing interval -- is a
        function of ``state.elapsed_seconds``, and each rank advances its own
        copy by ``cfg.dt`` per step, so the ranks stay aligned once they
        start aligned.  ``attach_lateral_boundaries`` zeroes the clock at
        attach time; a caller loading a warmed state imposes the matching
        instant here, once, before the first step.
        """
        for state in self.states:
            state.elapsed_seconds = float(seconds)

    def close(self) -> None:
        """Drop every device allocation and return the pools to the driver.

        ``del dom`` is NOT enough: the sub-states, the inventory dicts and the
        seam channels form reference cycles, so the arrays survive until the
        cyclic collector happens to run and the next size OOMs against memory
        that is logically free.  Measured: a 1536^2 monolithic run that fits in
        a fresh process failed with 14.1 GB still allocated after a 1024^2
        two-GPU run in the same process.
        """
        import gc

        import cupy as cp

        self.sync_all()
        self.states = []
        self.arrays = []
        self.channels = []
        self._events = None
        self.compute_streams = []
        self.copy_streams = []
        self.unpack_streams = []
        gc.collect()
        for dev in self.devices:
            with cp.cuda.Device(dev):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- loading and unloading ---------------------------------------------

    def load_from_host(self, host: dict) -> None:
        """Fill every sub-domain from a full-domain host inventory.

        Uses ``TileSpec.gather`` -- the same rectangles, with the same periodic
        wrap and the same never-read-the-alias-slot rule, that the single-GPU
        lane gathers a tile with.
        """
        import cupy as cp

        missing = set(self.names) - set(host)
        if missing:
            raise MultiGPUError(f"host inventory is missing {sorted(missing)}")
        for g, dev in enumerate(self.devices):
            sp = self.specs[g]
            with cp.cuda.Device(dev):
                for name in self.names:
                    dst = self.arrays[g][name]
                    src = host[name]
                    if dst.ndim < 2:
                        dst[...] = cp.asarray(src)
                        continue
                    v = _variant_of(name, src, self.nz, self.ny, self.nx)
                    for t in sp.gather(v):
                        dst[t.tile_key] = cp.asarray(
                            np.ascontiguousarray(src[t.full_key]))
                cp.cuda.runtime.deviceSynchronize()

    def assemble_host(self) -> dict:
        """Full-domain host inventory, from every sub-domain's INTERIOR.

        ``TileSpec.scatter`` writes interiors plus exactly one alias slot per
        staggered axis, and ``spec.validate_plan`` (run in ``__init__``) has
        already proved that covers the full array exactly once -- so nothing
        here is left at its initial value, and the check below proves it.
        """
        import cupy as cp

        self.sync_all()
        if self.volatile_inventory:
            self.refresh_arrays()
        out: dict = {}
        for name in self.names:
            a0 = self.arrays[0][name]
            if a0.ndim < 2:
                with cp.cuda.Device(self.devices[0]):
                    out[name] = cp.asnumpy(a0)
                continue
            v = _variant_of(name, a0, self.nz, self.specs[0].cny,
                            self.specs[0].cnx)
            ey, ex = _spec.stagger(v)
            shape = tuple(a0.shape[:-2]) + (self.ny + ey, self.nx + ex)
            out[name] = np.full(shape, np.nan, dtype=a0.dtype)
        written = {name: np.zeros(out[name].shape[-2:], dtype=np.int64)
                   for name in out if out[name].ndim >= 2}
        for g, dev in enumerate(self.devices):
            sp = self.specs[g]
            with cp.cuda.Device(dev):
                for name in self.names:
                    src = self.arrays[g][name]
                    if src.ndim < 2:
                        continue
                    v = _variant_of(name, src, self.nz, sp.cny, sp.cnx)
                    for t in sp.scatter(v):
                        out[name][t.full_key] = cp.asnumpy(
                            cp.ascontiguousarray(src[t.tile_key]))
                        written[name][t.full_y, t.full_x] += 1
        for name, count in written.items():
            if not np.array_equal(count, np.ones_like(count)):
                raise MultiGPUError(
                    f"assembly wrote {name!r} {sorted(np.unique(count))} "
                    "times per point; expected exactly 1")
        return out

    def hash(self) -> str:
        return hash_host(self.assemble_host())

    # -- the loop -----------------------------------------------------------

    def _step_sequential(self) -> None:
        import cupy as cp
        from gpuwm.core.dycore import step

        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                step(self.states[g], self.sub_cfgs[g])
                cp.cuda.runtime.deviceSynchronize()

    def _step_interleaved(self) -> None:
        import cupy as cp
        from gpuwm.core.dycore import step

        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                with self.compute_streams[g]:
                    step(self.states[g], self.sub_cfgs[g])
        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                self.compute_streams[g].synchronize()

    def _step_threads(self) -> None:
        import cupy as cp
        from gpuwm.core.dycore import step

        errors: list = []

        def work(g, dev):
            try:
                with cp.cuda.Device(dev):
                    with self.compute_streams[g]:
                        step(self.states[g], self.sub_cfgs[g])
                    self.compute_streams[g].synchronize()
            except BaseException as exc:            # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(g, d))
                   for g, d in enumerate(self.devices)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]

    def _step_launch_only(self) -> None:
        """Launch every device's step on its compute stream; do NOT sync."""
        import cupy as cp
        from gpuwm.core.dycore import step

        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                with self.compute_streams[g]:
                    step(self.states[g], self.sub_cfgs[g])

    # -- the exchange -------------------------------------------------------

    def exchange_blocking(self) -> None:
        """Pack, copy, unpack with a full device sync between every step.

        Rounds run in order (x seams, then y seams) and every round is fully
        drained before the next one packs -- which is what lets the y round
        read the x halo the previous round just filled, and so reach the
        corner data that lives on the diagonal neighbour.
        """
        import cupy as cp

        for group in self.channel_phases:
            for _k, ch in group:
                with cp.cuda.Device(ch.src_dev):
                    ch.pack(self.arrays[ch.seam.src_gpu], 0)
                    cp.cuda.runtime.deviceSynchronize()
            for _k, ch in group:
                with cp.cuda.Device(ch.src_dev):
                    ch.transfer(0)
                    cp.cuda.runtime.deviceSynchronize()
                with cp.cuda.Device(ch.dst_dev):
                    ch.transfer_finish(0)
                    cp.cuda.runtime.deviceSynchronize()
            for _k, ch in group:
                with cp.cuda.Device(ch.dst_dev):
                    ch.unpack(self.arrays[ch.seam.dst_gpu], 0)
                    cp.cuda.runtime.deviceSynchronize()

    def exchange_stream(self) -> None:
        """Same three phases on per-sub-domain streams, one sync per round."""
        import cupy as cp

        for group in self.channel_phases:
            for _k, ch in group:
                g = ch.seam.src_gpu
                with cp.cuda.Device(ch.src_dev):
                    ch.pack(self.arrays[g], self.copy_streams[g].ptr)
                    ch.transfer(self.copy_streams[g].ptr)
            self.sync_all()
            for _k, ch in group:
                d = ch.seam.dst_gpu
                with cp.cuda.Device(ch.dst_dev):
                    ch.transfer_finish(self.unpack_streams[d].ptr)
                    ch.unpack(self.arrays[d], self.unpack_streams[d].ptr)
            self.sync_all()

    def exchange_direct(self) -> None:
        """The UNPACKED reference exchange: no staging, one copy per field.

        Same rounds, same order, same bytes, same destinations as
        :meth:`exchange_stream` -- and ``sum(len(ch.bands))`` link crossings
        per exchange where the packed path makes one per seam.  Keep it: it is
        the second implementation the packed path's digest is checked against,
        and the "before" arm of the pack measurement.  It is not a shipping
        path and ``run`` names it ``direct`` so a caller cannot reach it by
        accident.
        """
        import cupy as cp

        for group in self.channel_phases:
            for _k, ch in group:
                s, d = ch.seam.src_gpu, ch.seam.dst_gpu
                with cp.cuda.Device(ch.src_dev):
                    ch.direct_send(self.arrays[s], self.arrays[d],
                                   self.copy_streams[s].ptr)
            self.sync_all()
            for _k, ch in group:
                d = ch.seam.dst_gpu
                with cp.cuda.Device(ch.dst_dev):
                    ch.direct_recv(self.arrays[d], self.unpack_streams[d].ptr)
            self.sync_all()

    def transfer_counts(self) -> dict:
        """Link crossings per exchange, packed vs unpacked.

        The evidence this lane acts on says the exchange is bound by transfer
        COUNT, so the count is reported as a measured property of the plan
        rather than left to be inferred from a docstring.
        """
        return {
            "seams": len(self.channels),
            "bands_total": sum(len(c.bands) for c in self.channels),
            "packed": sum(c.n_transfers_packed for c in self.channels),
            "direct": sum(c.n_transfers_direct for c in self.channels),
            "transport": self.transport,
        }

    def transport_report(self) -> list:
        """The path each device pair ACTUALLY takes, per ordered pair.

        ``transport`` is a REQUEST.  ``cudaMemcpyPeerAsync`` succeeds whether
        or not the two devices can reach each other, and when they cannot,
        CUDA stages the bytes through host memory and says nothing.  A report
        that prints the request as though it were the outcome is how a
        host-staged copy comes to be labelled "peer": the number is real, the
        name on it is not.  So the driver is asked, once, and the answer
        travels with every measurement.

        Ordered pairs, not unordered: ``canAccessPeer`` is directional and a
        box can answer differently each way.
        """
        pairs = peer_access_matrix(self.devices)
        seen, rows = set(), []
        for ch in self.channels:
            key = (ch.src_dev, ch.dst_dev)
            if key in seen:
                continue
            seen.add(key)
            p2p = pairs.get(key, False)
            actual, legs = _TRANSPORT_PATHS[(self.transport, p2p)]
            rows.append({"src_dev": ch.src_dev, "dst_dev": ch.dst_dev,
                         "requested": self.transport, "can_access_peer": p2p,
                         "actual": actual, "host_legs": legs})
        return sorted(rows, key=lambda r: (r["src_dev"], r["dst_dev"]))

    def describe_transport(self) -> str:
        """:meth:`transport_report` as the block every transport arm prints."""
        rows = self.transport_report()
        out = [f"  TRANSPORT PATH (requested {self.transport!r}):"]
        for r in rows:
            out.append(f"    gpu{r['src_dev']} -> gpu{r['dst_dev']}  "
                       f"canAccessPeer={int(r['can_access_peer'])}  "
                       f"ACTUAL: {r['actual']}")
        if not any(r["can_access_peer"] for r in rows) \
                and self.transport in ("peer", "default"):
            out.append("    No pair on this box has P2P, so every byte below "
                       "went through host memory.")
        return "\n".join(out)

    def _ensure_events(self):
        import cupy as cp

        if self._events is not None:
            return self._events
        arrived = []
        stepped = []
        unpacked = []
        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                stepped.append(cp.cuda.Event(block=False, disable_timing=True))
                unpacked.append(cp.cuda.Event(block=False,
                                              disable_timing=True))
        for ch in self.channels:
            # An event is created on the device that RECORDS it; the waiter may
            # live on the other device (cudaStreamWaitEvent is cross-device).
            with cp.cuda.Device(ch.src_dev):
                arrived.append(cp.cuda.Event(block=False, disable_timing=True))
        # One per-device event per round BOUNDARY: round p+1's pack reads what
        # round p's unpack wrote, on the same device but a different stream.
        round_done = []
        for _p in range(max(len(self.channel_phases) - 1, 0)):
            evs = []
            for dev in self.devices:
                with cp.cuda.Device(dev):
                    evs.append(cp.cuda.Event(block=False,
                                             disable_timing=True))
            round_done.append(evs)
        self._events = dict(stepped=stepped, unpacked=unpacked,
                            arrived=arrived, round_done=round_done)
        return self._events

    def step_events(self) -> None:
        """One step + exchange with the phases chained by CUDA events.

        The ordering that has to hold, and why each edge exists:

        * pack(seam) after step(src)      -- it reads src's fresh interior;
        * unpack(seam) after step(dst)    -- it writes dst's halo, which dst's
          own step is still writing until it finishes;
        * unpack(seam) after transfer(seam);
        * next step(dst) after every unpack into dst.

        Note what is NOT required: pack on GPU a does not wait for GPU b.  So
        a's outbound copy starts the moment a finishes stepping and overlaps
        whatever b has left to do.

        With a 2-D decomposition there is a second round, and it needs one more
        edge -- round 1's pack after round 0's UNPACK on the same device, since
        the y band it packs includes the x halo round 0 just wrote.  That edge
        is a per-device event, not a sync: the two devices still do not wait for
        each other.
        """
        import cupy as cp

        ev = self._ensure_events()
        self._step_launch_only()
        if self.volatile_inventory:
            # Host-side rebinding happens during the LAUNCH, so the fresh
            # pointers are already correct even though the step has not run.
            self.refresh_arrays()
        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                ev["stepped"][g].record(self.compute_streams[g])
        ready = ev["stepped"]           # per device: "your data is packable"
        last = len(self.channel_phases) - 1
        for p, group in enumerate(self.channel_phases):
            for k, ch in group:
                s, d = ch.seam.src_gpu, ch.seam.dst_gpu
                with cp.cuda.Device(ch.src_dev):
                    self.copy_streams[s].wait_event(ready[s])
                    ch.pack(self.arrays[s], self.copy_streams[s].ptr)
                    ch.transfer(self.copy_streams[s].ptr)
                    ev["arrived"][k].record(self.copy_streams[s])
                with cp.cuda.Device(ch.dst_dev):
                    self.unpack_streams[d].wait_event(ev["arrived"][k])
                    self.unpack_streams[d].wait_event(ready[d])
                    ch.transfer_finish(self.unpack_streams[d].ptr)
                    ch.unpack(self.arrays[d], self.unpack_streams[d].ptr)
            if p < last:
                for g, dev in enumerate(self.devices):
                    with cp.cuda.Device(dev):
                        ev["round_done"][p][g].record(self.unpack_streams[g])
                ready = ev["round_done"][p]
        for g, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                ev["unpacked"][g].record(self.unpack_streams[g])
                self.compute_streams[g].wait_event(ev["unpacked"][g])

    # -- public driver ------------------------------------------------------

    def run(self, nsteps: int, *, step_mode: str = "sequential",
            exchange_mode: str = "blocking", exchange: bool = True) -> None:
        """Advance every sub-domain ``nsteps`` steps, exchanging each step.

        ``exchange=False`` is a TIMING CONTROL ONLY -- it measures the compute
        without the seam and its answer is not a forecast.
        """
        import cupy as cp

        nsteps = int(nsteps)
        if step_mode == "events" or exchange_mode == "events":
            if not (step_mode == "events" and exchange_mode == "events"):
                raise ValueError("the 'events' pipeline owns both the step and "
                                 "the exchange; pass it for both")
            for _ in range(nsteps):
                if exchange:
                    self.step_events()
                else:
                    # The timing control has to actually drop the exchange.
                    # It did not, the first time this was written, and the
                    # control then reported the exchange as free because it
                    # was still running it.
                    self._step_launch_only()
                    for g, dev in enumerate(self.devices):
                        with cp.cuda.Device(dev):
                            self.compute_streams[g].synchronize()
            self.sync_all()
            return

        stepper = {"sequential": self._step_sequential,
                   "interleaved": self._step_interleaved,
                   "threads": self._step_threads}[step_mode]
        exchanger = {"blocking": self.exchange_blocking,
                     "stream": self.exchange_stream,
                     "direct": self.exchange_direct}[exchange_mode]
        for _ in range(nsteps):
            stepper()
            if self.volatile_inventory:
                self.refresh_arrays()
            if exchange:
                exchanger()
        self.sync_all()


# ==========================================================================
# verification
# ==========================================================================

def _window_of(host: dict, sp, nz, ny, nx, names) -> dict:
    """The sub-domain window a monolithic host inventory implies for ``sp``."""
    out: dict = {}
    for name in names:
        src = host[name]
        if src.ndim < 2:
            out[name] = np.array(src, copy=True)
            continue
        v = _variant_of(name, src, nz, ny, nx)
        ey, ex = _spec.stagger(v)
        dst = np.full(tuple(src.shape[:-2]) + (sp.cny + ey, sp.cnx + ex),
                      np.nan, dtype=src.dtype)
        for t in sp.gather(v):
            dst[t.tile_key] = src[t.full_key]
        out[name] = dst
    return out


def verify_geometry(cfg, *, ngpu: int = 2, grid=None, devices=None,
                    steps: int = 1, seed: int = _harness.DEFAULT_SEED,
                    verbose: bool = True):
    """Compare EVERY sub-domain array against the monolithic run's window.

    Stronger than a hash of the assembled domain: it scores the halos too, so
    a crossed or mis-sized seam is localised to a band of x columns instead of
    being smeared over the whole answer.  Returns ``(ok, report)``.
    """
    import cupy as cp

    dev0 = 0 if devices is None else int(devices[0])
    with cp.cuda.Device(dev0):
        ref = _harness.make_state(cfg, seed=seed)
        start = download_state(ref)
        _harness.run_steps(ref, cfg, steps)
        truth = download_state(ref)
        del ref
        cp.get_default_memory_pool().free_all_blocks()

    dom = MultiGPUDomain(cfg, ngpu=ngpu, grid=grid, devices=devices)
    dom.load_from_host(start)
    dom.run(steps, step_mode="sequential", exchange_mode="blocking")

    report: dict = {"seams": describe_plan(dom.specs, dom.seams)}
    ok = True
    for g, dev in enumerate(dom.devices):
        want = _window_of(truth, dom.specs[g], dom.nz, dom.ny, dom.nx,
                          dom.names)
        with cp.cuda.Device(dev):
            got = {k: cp.asnumpy(v) for k, v in dom.arrays[g].items()}
        diff = compare_hosts(want, got, dom.specs[g].cnx)
        report[f"gpu{g}"] = diff
        if diff:
            ok = False
        if verbose:
            print(f"  gpu{g} sub-array vs monolithic window: "
                  f"{'IDENTICAL' if not diff else 'DIFFERS'}")
            for name, d in list(diff.items())[:8]:
                print(f"    {name}: {d}")
    return ok, report


def gate(nx: int = 256, ny: int = 128, nz: int = _harness.DEFAULT_NZ,
         steps: int = 3, *, ngpu: int = 2, grid=None, devices=None,
         seed: int = _harness.DEFAULT_SEED, modes=None,
         transport: str = "peer", verbose: bool = True):
    """The bit-exact gate: 2-GPU decomposition vs a 1-GPU monolithic run.

    Every entry of ``modes`` -- ``(step_mode, exchange_mode)`` -- must produce
    the SAME digest as the monolithic run.  A decomposition bug that skips work
    looks exactly like a speedup, so nothing here is timed without also being
    hashed.
    """
    import cupy as cp

    modes = modes or [("sequential", "blocking"),
                      ("interleaved", "stream"),
                      ("threads", "stream"),
                      ("events", "events")]
    cfg = _harness.make_config(nx, ny, nz)
    dev0 = 0 if devices is None else int(devices[0])

    with cp.cuda.Device(dev0):
        ref = _harness.make_state(cfg, seed=seed)
        start = download_state(ref)
        h_start = _harness.hash_state(ref)
        _harness.run_steps(ref, cfg, steps)
        h_mono = _harness.hash_state(ref)
        truth = download_state(ref)
        del ref
        cp.get_default_memory_pool().free_all_blocks()

    results = {"config": (nx, ny, nz, steps),
               "hash_initial": h_start, "hash_monolithic": h_mono, "runs": {}}
    if verbose:
        # ``grid`` OVERRIDES ``ngpu`` in MultiGPUDomain, so printing ``ngpu``
        # here labelled a --grid 2x2 run "2 GPUs" while it ran four
        # sub-domains on four cards.  The header is the line a reader quotes;
        # it states the geometry that was built.
        nsub = ngpu if grid is None else int(grid[0]) * int(grid[1])
        gy, gx = (1, ngpu) if grid is None else (int(grid[0]), int(grid[1]))
        print(f"domain {nx}x{ny}x{nz}, {steps} steps, {nsub} sub-domains "
              f"({gy}x{gx}) on {nsub} GPUs, "
              f"halo={_harness.halo_radius(cfg)}")
        print(f"  monolithic (1 GPU): {h_mono}")

    ok = True
    for step_mode, exchange_mode in modes:
        dom = MultiGPUDomain(cfg, ngpu=ngpu, grid=grid, devices=devices,
                             transport=transport)
        if verbose and not results["runs"]:
            print(describe_plan(dom.specs, dom.seams))
            print(f"  seam bytes/exchange: {dom.seam_bytes / 2**20:.2f} MiB")
        dom.load_from_host(start)
        dom.run(steps, step_mode=step_mode, exchange_mode=exchange_mode)
        host = dom.assemble_host()
        h = hash_host(host)
        same = (h == h_mono)
        ok = ok and same
        results["runs"][f"{step_mode}/{exchange_mode}"] = {
            "hash": h, "match": same}
        if verbose:
            print(f"  {step_mode:12s}/{exchange_mode:9s}: {h} "
                  f"{'MATCH' if same else 'DIFFER'}")
        if not same:
            results["runs"][f"{step_mode}/{exchange_mode}"]["diff"] = \
                compare_hosts(truth, host, nx)
            if verbose:
                for name, d in list(
                        results["runs"][f"{step_mode}/{exchange_mode}"]
                        ["diff"].items())[:10]:
                    print(f"      {name}: {d}")
        dom.close()
    results["ok"] = ok
    return ok, results


# ==========================================================================
# the FORCED rung: specified lateral boundaries through the decomposition
# ==========================================================================

def window_geography(geo, spec):
    """One rank's window of the DOMAIN's geography, per variant.

    Built by ``TileSpec.apply_gather`` on the host arrays, so the map
    factors, Coriolis, rotation, lat/lon and terrain a rank is constructed
    with are the domain's own values at the rank's window -- never a per-rank
    Lambert REBUILD, which is displaced by the window offset and is the
    measured geography defect the streamed lane exists to avoid.
    """
    def cut(arr, variant):
        ey, ex = _spec.stagger(variant)
        out = np.empty((spec.cny + ey, spec.cnx + ex), dtype=np.float64)
        spec.apply_gather(np.asarray(arr, dtype=np.float64), out, variant)
        return out

    return _harness.Geography(
        grid=None,
        msft=cut(geo.msft, "mass"), msfu=cut(geo.msfu, "u"),
        msfv=cut(geo.msfv, "v"),
        f=cut(geo.f, "mass"), e=cut(geo.e, "mass"),
        sina=cut(geo.sina, "mass"), cosa=cut(geo.cosa, "mass"),
        lat=cut(geo.lat, "mass"), lon=cut(geo.lon, "mass"),
        terrain=None if geo.terrain is None else cut(geo.terrain, "mass"))


def forced_config(nx: int, ny: int, nz: int = _harness.DEFAULT_NZ):
    """A SPECIFIED dry config on the real Lambert projection with terrain.

    ``periodic=False`` so :func:`tilestream.harness.make_config` does not
    force the lateral flags off; ``specified=True`` plus the WRF-standard
    ``spec_zone=1, relax_zone=4`` then makes every axis non-periodic and the
    boundary machinery live.  Geography comes from
    ``harness.GEOGRAPHY_OVERRIDES`` (Lambert, terrain, dx=12 km) because a
    flat identity-map domain cannot gate a boundary: its edge columns carry
    no signal a wrong window would change.
    """
    return _harness.make_config(
        nx, ny, nz, periodic=False,
        specified=True, nested=False, open_x=False, open_y=False,
        **_harness.GEOGRAPHY_OVERRIDES)


def forced_domain_inputs(cfg, *, seed: int = _harness.DEFAULT_SEED,
                         bdy_seconds: float = 3600.0, device: int = 0):
    """``(geo, start, boundaries, h_start)`` for one specified gate case.

    The forcing is built from TWO differently seeded states so the time
    tendency is nonzero -- a zero tendency quietly disarms the ``dtbc``
    clock and would pass on a decomposition that never carried it
    (tilestream.test_route's own trap).  The start state is downloaded
    BEFORE any stepping, with its clock at zero, and the same host inventory
    feeds both the monolithic arm and every decomposed arm.
    """
    import cupy as cp

    from gpuwm.ingest.lateral_bc import build_state_lateral_boundaries

    geo = _harness.make_geography(cfg, terrain=True, periodic_faces=False)
    with cp.cuda.Device(int(device)):
        state = _harness.make_state(cfg, seed=seed, geography=geo)
        other = _harness.make_state(cfg, seed=seed + 1, geography=geo)
        boundaries = build_state_lateral_boundaries(
            [state, other], [0.0, float(bdy_seconds)],
            spec_bdy_width=int(cfg.spec_bdy_width),
            spec_zone=int(cfg.spec_zone), relax_zone=int(cfg.relax_zone))
        del other
        start = download_state(state)
        h_start = hash_host(start)
        del state
        cp.get_default_memory_pool().free_all_blocks()
    return geo, start, boundaries, h_start


def forced_state_factory(geo):
    """A rank state factory carrying the DOMAIN's geography, windowed."""
    def make(sub_cfg, *, spec):
        return _harness.make_state(sub_cfg, geography=window_geography(
            geo, spec))
    return make


def _scaled_boundaries(boundaries, factor: float):
    """Every true side table scaled -- the control that proves forcing is live.

    A gate whose forced arms matched because the boundary never reached the
    answer would still match here; a gate whose forcing is live must differ.
    """
    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         LateralBoundaries, SideBoundary)

    intervals = []
    for iv in boundaries.intervals:
        fields = {}
        for name, fb in iv.fields.items():
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                side = getattr(fb, side_name)
                sides[side_name] = SideBoundary(
                    np.asarray(side.value) * float(factor),
                    np.asarray(side.tendency) * float(factor))
            fields[name] = FieldBoundary(**sides)
        intervals.append(BoundaryInterval(iv.start_seconds, iv.end_seconds,
                                          fields))
    return LateralBoundaries(tuple(intervals), boundaries.spec_bdy_width,
                             boundaries.spec_zone, boundaries.relax_zone)


#: The FORCED rung's rank geometries: the identity arm that separates
#: "decomposition wrong" from "forced state through MultiGPUDomain wrong",
#: the 2-rank x split, and the 2x2 whose ranks each own ONE domain corner --
#: the geometry where corner ownership (Y sides own corners, and a rank's
#: local frame must coincide with the global one at a real corner) can fail.
FORCED_GRIDS: tuple = ((1, 1), (1, 2), (2, 2))


def gate_forced(nx: int = 220, ny: int = 168, nz: int = _harness.DEFAULT_NZ,
                steps: int = 6, *, grids=FORCED_GRIDS, devices=None,
                seed: int = _harness.DEFAULT_SEED, seam: str = "zeros",
                bdy_seconds: float = 3600.0, controls: bool = True,
                verbose: bool = True):
    """The FORCED bit-exact gate: specified BCs, N ranks vs the resident run.

    Every grid in ``grids`` must reproduce the resident (monolithic)
    specified run's digest bit for bit -- the same standard the periodic
    :func:`gate` holds, on the case the periodic gate cannot see: real
    Lambert geography, real terrain, external specified forcing with a
    nonzero tendency, non-periodic plan, per-rank windowed boundary tables.

    ``devices=None`` places rank ``r`` on card ``r % ndev`` over the visible
    cards, so a one-card box still gates the full geometry and forcing
    arithmetic (the transport degenerates to same-device copies).

    Two controls, run unless ``controls=False``:

    * ``poison`` seam tables MUST STILL MATCH -- the quarantine proof: the
      halo pad, not luck, is what keeps seam fiction out of owned cells.
    * true-edge tables scaled by 1.000001 MUST DIFFER -- the treatment
      proof: an arm that matches because the forcing never reached the
      answer is a green light on nothing, and this is the arm that catches
      it.
    """
    import cupy as cp

    cfg = forced_config(nx, ny, nz)
    ndev_avail = cp.cuda.runtime.getDeviceCount()
    dev0 = 0 if not devices else int(devices[0])

    geo, start, boundaries, h_start = forced_domain_inputs(
        cfg, seed=seed, bdy_seconds=bdy_seconds, device=dev0)

    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

    with cp.cuda.Device(dev0):
        ref = _harness.make_state(cfg, seed=seed, geography=geo)
        attach_lateral_boundaries(ref, boundaries)
        ref.elapsed_seconds = 0.0
        _harness.run_steps(ref, cfg, steps)
        h_mono = _harness.hash_state(ref)
        truth = download_state(ref)
        del ref
        cp.get_default_memory_pool().free_all_blocks()

    results = {"config": (nx, ny, nz, steps), "hash_initial": h_start,
               "hash_monolithic": h_mono, "halo": forced_halo(cfg),
               "runs": {}, "controls": {}}
    if verbose:
        print(f"FORCED domain {nx}x{ny}x{nz}, {steps} steps, specified BCs "
              f"(spec_zone={cfg.spec_zone}, relax_zone={cfg.relax_zone}), "
              f"halo={results['halo']} "
              f"(= {_harness.halo_radius(cfg)} + "
              f"{max(cfg.spec_zone, cfg.relax_zone)})")
        print(f"  resident (1 GPU): {h_mono}")

    def _arm(grid, *, bnd, seam_mode):
        gy, gx = int(grid[0]), int(grid[1])
        nsub = gy * gx
        pool = list(devices) if devices else list(range(ndev_avail))
        devs = [pool[d % len(pool)] for d in range(nsub)]
        dom = MultiGPUDomain(cfg, grid=(gy, gx), devices=devs,
                             boundaries=bnd, seam=seam_mode,
                             state_factory=forced_state_factory(geo))
        dom.load_from_host(start)
        dom.impose_clock(0.0)
        dom.run(steps, step_mode="sequential", exchange_mode="blocking")
        host = dom.assemble_host()
        h = hash_host(host)
        dom.close()
        return h, host

    ok = True
    for grid in grids:
        h, host = _arm(grid, bnd=boundaries, seam_mode=seam)
        same = (h == h_mono)
        ok = ok and same
        key = f"{int(grid[0])}x{int(grid[1])}"
        results["runs"][key] = {"hash": h, "match": same}
        if verbose:
            print(f"  {key} ({int(grid[0]) * int(grid[1])} rank(s)): {h} "
                  f"{'MATCH' if same else 'DIFFER'}")
        if not same:
            results["runs"][key]["diff"] = compare_hosts(truth, host, nx)
            if verbose:
                for name, d in list(results["runs"][key]["diff"].items())[:8]:
                    print(f"      {name}: {d}")

    if controls:
        multi = next((g for g in grids if g[0] * g[1] > 1), (1, 2))
        h_poison, _ = _arm(multi, bnd=boundaries, seam_mode="poison")
        poison_ok = (h_poison == h_mono)
        results["controls"]["poison_seam_matches"] = poison_ok
        h_scaled, _ = _arm(multi, bnd=_scaled_boundaries(boundaries,
                                                         1.000001),
                           seam_mode=seam)
        scaled_fired = (h_scaled != h_mono)
        results["controls"]["scaled_edge_differs"] = scaled_fired
        ok = ok and poison_ok and scaled_fired
        if verbose:
            print(f"  control poison seams ({multi[0]}x{multi[1]}): "
                  f"{'MATCH (quarantine holds)' if poison_ok else 'DIFFER -- SEAM FICTION REACHED OWNED CELLS'}")
            print(f"  control scaled edges ({multi[0]}x{multi[1]}): "
                  f"{'DIFFER (forcing is live)' if scaled_fired else 'MATCH -- FORCING NEVER REACHED THE ANSWER'}")

    results["ok"] = ok
    return ok, results


#: The first step at which the ``short_halo`` control becomes VISIBLE, at the
#: geometry :func:`negative_controls` runs.  Measured, twice, on two different
#: architectures: ``halo = 13`` reproduces the monolithic digest bit for bit
#: through step 7 and first differs at step 8 (2x RTX 4090, and again on 4x
#: RTX 3090 sm_86).  A run shorter than this cannot see the defect, so a
#: shorter run is not a weaker gate -- it is no gate, and it is refused rather
#: than reported.
SHORT_HALO_VISIBLE_AT = 8

#: Steps :func:`negative_controls` runs when nobody says otherwise: the
#: visibility floor plus margin, so the control is not sitting on the first
#: step that happens to work.
CONTROLS_DEFAULT_STEPS = 10


def negative_controls(nx: int = 256, ny: int = 128,
                      nz: int = _harness.DEFAULT_NZ,
                      steps: int = CONTROLS_DEFAULT_STEPS, *,
                      devices=None, seed: int = _harness.DEFAULT_SEED,
                      verbose: bool = True):
    """Break the decomposition on purpose and check the gate NOTICES.

    A gate that has never seen a failure is not a gate.  Four defects, each
    the shape of a plausible bug:

    ``crossed``     the two seams' sources swapped -- GPU 0's left halo fed
                    from the LEFT of GPU 1's interior instead of its right.
                    The classic first bug on a periodic split.
    ``no_exchange`` sub-domains never see each other.  Fastest of all, and the
                    one a timing-only test would happily certify.
    ``short_halo``  ``halo = 13``.  MEASURED HERE (256x128x49, 2 GPUs): the
                    assembled digest is bit-identical to the monolithic one for
                    SEVEN consecutive steps and then goes 7.8e-3 / 2.5e-1 /
                    4.8e-1 at steps 8 / 9 / 10, while ``halo = 16`` stays
                    bit-exact for at least 12.  That is why ``steps`` defaults
                    to 10 here: at 5 this control passes silently, which is the
                    exact trap the halo is not allowed to be tuned on.
    ``stale``       exchange every OTHER step, i.e. two steps on one halo.
    """
    import cupy as cp

    cfg = _harness.make_config(nx, ny, nz)
    dev0 = 0 if devices is None else int(devices[0])
    with cp.cuda.Device(dev0):
        ref = _harness.make_state(cfg, seed=seed)
        start = download_state(ref)
        _harness.run_steps(ref, cfg, steps)
        h_mono = _harness.hash_state(ref)
        del ref
        cp.get_default_memory_pool().free_all_blocks()

    out: dict = {"hash_monolithic": h_mono}

    def _run(dom, nsteps, every=1):
        dom.load_from_host(start)
        for i in range(nsteps):
            dom.run(1, step_mode="sequential", exchange_mode="blocking",
                    exchange=((i + 1) % every == 0))
        return hash_host(dom.assemble_host())

    # 1. crossed seams: every band reads the OTHER end of the neighbour's
    #    interior, at the same length (so nothing but the physics notices).
    dom = MultiGPUDomain(cfg, devices=devices)
    dom.seams = cross_seams(dom.seams, dom.specs, dom.halo)
    dom._build_channels()
    out["crossed"] = _run(dom, steps)
    dom.close()

    # 2. no exchange at all
    dom = MultiGPUDomain(cfg, devices=devices)
    out["no_exchange"] = _run(dom, steps, every=10 ** 9)
    dom.close()

    # 3. halo below the dependency radius
    dom = MultiGPUDomain(cfg, devices=devices, halo=13,
                         _unsafe_short_halo=True)
    out["short_halo"] = _run(dom, steps)
    dom.close()

    # 4. two steps per exchange
    dom = MultiGPUDomain(cfg, devices=devices)
    out["stale_halo"] = _run(dom, steps, every=2)
    dom.close()

    out["detected"] = {k: (v != h_mono) for k, v in out.items()
                       if k not in ("hash_monolithic", "detected")}
    out["ok"] = all(out["detected"].values())
    if verbose:
        print(f"negative controls at {nx}x{ny}x{nz}, {steps} steps "
              f"(all four MUST differ from the monolithic hash):")
        for k, v in out["detected"].items():
            print(f"  {k:12s}: {'DETECTED' if v else 'MISSED -- GATE IS BLIND'}")
    return out["ok"], out


# ==========================================================================
# measurement
# ==========================================================================

def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def bench(nx: int, ny: int, nz: int = _harness.DEFAULT_NZ, *,
          reps: int = 7, steps: int = 1, ngpu: int = 2, grid=None,
          devices=None,
          step_mode: str = "interleaved", exchange_mode: str = "stream",
          exchange: bool = True, seed: int = _harness.DEFAULT_SEED,
          warmup: int = 2, verbose: bool = True) -> dict:
    """Median ms/step for the decomposition, with a full sync on BOTH devices.

    Timed with ``time.perf_counter`` around a ``run`` that ends in a
    ``deviceSynchronize`` on every device, which is the only honest barrier
    when two devices are involved (a CUDA event is per-device).

    Setup is slow and the GPU is idle for all of it: ``make_state`` draws
    ~21 host-side normal arrays per sub-domain, which is minutes at 2000^2.
    That is startup, not measurement -- ``warmup`` steps run before the first
    timed rep.
    """
    import cupy as cp

    cfg = _harness.make_config(nx, ny, nz)
    # Seed each sub-domain directly.  These values are NOT a decomposition of
    # any one domain -- this function times, it does not verify -- but they are
    # realistic magnitudes, which matters: an at-rest state is full of exact
    # zeros and does not exercise the same arithmetic.  Correctness is
    # ``gate``'s job and every timed shape is also gated.
    dom = MultiGPUDomain(cfg, ngpu=ngpu, grid=grid, devices=devices,
                         state_factory=lambda c: _harness.make_state(
                             c, seed=seed))

    dom.run(warmup, step_mode=step_mode, exchange_mode=exchange_mode,
            exchange=exchange)
    times = []
    for _ in range(int(reps)):
        dom.sync_all()
        t0 = time.perf_counter()
        dom.run(steps, step_mode=step_mode, exchange_mode=exchange_mode,
                exchange=exchange)
        times.append((time.perf_counter() - t0) * 1e3 / steps)
    med = _median(times)
    spread = (max(times) - min(times)) / med if med else 0.0
    out = {"ms_per_step": med, "spread": spread, "times": times,
           "seam_MiB": dom.seam_bytes / 2**20,
           "vram_MiB": [b / 2**20 for b in dom.vram_bytes()],
           "step_mode": step_mode, "exchange_mode": exchange_mode,
           "exchange": exchange, "domain": (nx, ny, nz),
           "grid": dom.grid,
           "sub_shape": [(s.cny, s.cnx) for s in dom.specs]}
    if verbose:
        print(f"  {nx}x{ny}x{nz} {step_mode}/{exchange_mode} "
              f"exchange={exchange}: {med:.2f} ms/step "
              f"(spread {spread * 100:.1f}%)"
              + ("  <-- SPREAD > 10%" if spread > 0.10 else ""))
    dom.close()
    return out


def bench_exchange(nx: int, ny: int, nz: int = _harness.DEFAULT_NZ, *,
                   reps: int = 9, ngpu: int = 2, grid=None, devices=None,
                   transport: str = "peer",
                   seed: int = _harness.DEFAULT_SEED,
                   verbose: bool = True) -> dict:
    """The exchange with nothing else running, split into its three phases.

    Measured cumulatively -- pack, then pack+transfer, then the whole thing --
    because the phases are chained on one stream and cannot be timed
    independently without breaking that chain.  The differences attribute the
    cost; the ``transfer`` line is the only one that touches the link.

    ``link_GB_s`` divides by the bytes in ONE direction, since the two
    directions run on separate devices' copy engines and overlap.  Reporting
    the sum against the same wall time would be the "1902 GB/s against a 1396
    GB/s ceiling" mistake in a new hat.

    CAVEAT for ``transport="host"``: its copy has two legs and only the D2H one
    lands in ``transfer_ms``; the H2D leg is inside ``unpack_ms``.  So its
    ``link_GB_s`` is NOT comparable with peer's -- compare ``full``.  MEASURED
    at 768^2: peer 4.72 ms vs host 6.68 ms full, i.e. hand-rolling the staging
    costs 1.41x, which is the same trap the single-GPU lane already hit.
    """
    import cupy as cp

    cfg = _harness.make_config(nx, ny, nz)
    dom = MultiGPUDomain(cfg, ngpu=ngpu, grid=grid, devices=devices,
                         transport=transport,
                         state_factory=lambda c: _harness.make_state(c,
                                                                     seed=seed))

    def phases(do_transfer: bool, do_unpack: bool) -> None:
        # TIMING ONLY.  Every round is issued at once here rather than in
        # order, so the ANSWER this leaves in the arrays is not a forecast --
        # only its byte count and its wall clock mean anything.  Correctness
        # is ``gate``'s job.
        for ch in dom.channels:
            g = ch.seam.src_gpu
            with cp.cuda.Device(ch.src_dev):
                ch.pack(dom.arrays[g], dom.copy_streams[g].ptr)
                if do_transfer:
                    ch.transfer(dom.copy_streams[g].ptr)
        dom.sync_all()
        if do_transfer and do_unpack:
            for ch in dom.channels:
                d = ch.seam.dst_gpu
                with cp.cuda.Device(ch.dst_dev):
                    ch.transfer_finish(dom.unpack_streams[d].ptr)
                    ch.unpack(dom.arrays[d], dom.unpack_streams[d].ptr)
            dom.sync_all()

    out: dict = {"seam_MiB": dom.seam_bytes / 2**20, "transport": transport,
                 "domain": (nx, ny, nz)}
    for tag, args in (("pack", (False, False)), ("pack+xfer", (True, False)),
                      ("full", (True, True))):
        phases(*args)                                   # warm
        ts = []
        for _ in range(int(reps)):
            dom.sync_all()
            t0 = time.perf_counter()
            phases(*args)
            ts.append((time.perf_counter() - t0) * 1e3)
        out[tag] = _median(ts)
        out[tag + "_spread"] = (max(ts) - min(ts)) / out[tag]
    out["transfer_ms"] = out["pack+xfer"] - out["pack"]
    out["unpack_ms"] = out["full"] - out["pack+xfer"]
    one_way = dom.seam_bytes / 2 / 1e9
    out["link_GB_s"] = one_way / (out["transfer_ms"] / 1e3) \
        if out["transfer_ms"] > 0 else float("nan")
    # The transport is a request; this is what the driver actually did with
    # it.  Carried in the result, not only printed, so a caller writing JSON
    # cannot lose the attribution the number depends on.
    out["transport_report"] = dom.transport_report()
    out["p2p_any"] = any(r["can_access_peer"] for r in out["transport_report"])
    # ``transfer_ms`` times ONE memcpy call.  For the host transport that call
    # is the D2H leg alone and the H2D leg is inside ``unpack_ms``, so its
    # link figure counts half the journey and is not comparable with peer's.
    # The end-to-end figure divides the same bytes by the whole pack-transfer-
    # unpack wall time and IS comparable across all three arms.
    out["link_GB_s_counts"] = ("the D2H leg only; the H2D leg is inside "
                               "unpack_ms" if transport == "host"
                               else "the whole crossing, one memcpy")
    out["end_to_end_GB_s"] = one_way / (out["full"] / 1e3) \
        if out["full"] > 0 else float("nan")
    if verbose:
        print(f"  exchange {nx}x{ny}x{nz} transport={transport}: "
              f"{out['seam_MiB']:.1f} MiB total "
              f"({out['seam_MiB'] / 2:.1f} MiB each way)")
        print(dom.describe_transport())
        for tag in ("pack", "pack+xfer", "full"):
            print(f"    {tag:10s} {out[tag]:7.3f} ms "
                  f"(spread {out[tag + '_spread'] * 100:.1f}%)"
                  + ("  <-- SPREAD > 10%"
                     if out[tag + "_spread"] > 0.10 else ""))
        print(f"    -> pack {out['pack']:.3f} ms, transfer "
              f"{out['transfer_ms']:.3f} ms, unpack {out['unpack_ms']:.3f} ms")
        print(f"    -> link {out['link_GB_s']:.1f} GB/s per direction, "
              f"counting {out['link_GB_s_counts']}")
        print(f"    -> end to end {out['end_to_end_GB_s']:.1f} GB/s per "
              "direction, pack through unpack; this is the arm-to-arm "
              "comparable figure")
    dom.close()
    return out


def bench_monolithic(nx: int, ny: int, nz: int = _harness.DEFAULT_NZ, *,
                     reps: int = 7, device: int = 0,
                     seed: int = _harness.DEFAULT_SEED, warmup: int = 2,
                     verbose: bool = True) -> dict:
    """Median ms/step for the same domain on ONE GPU, for the ratio."""
    import cupy as cp

    cfg = _harness.make_config(nx, ny, nz)
    with cp.cuda.Device(device):
        state = _harness.make_state(cfg, seed=seed)
        _harness.run_steps(state, cfg, warmup)
        times = []
        for _ in range(int(reps)):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            _harness.run_steps(state, cfg, 1)
            times.append((time.perf_counter() - t0) * 1e3)
        med = _median(times)
        spread = (max(times) - min(times)) / med if med else 0.0
        vram = sum(a.nbytes for a in _harness.state_arrays(state).values())
        del state
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    if verbose:
        print(f"  {nx}x{ny}x{nz} monolithic on gpu{device}: {med:.2f} ms/step "
              f"(spread {spread * 100:.1f}%)"
              + ("  <-- SPREAD > 10%" if spread > 0.10 else ""))
    return {"ms_per_step": med, "spread": spread, "times": times,
            "vram_MiB": vram / 2**20, "domain": (nx, ny, nz)}


# ==========================================================================
# CLI
# ==========================================================================

def _main(argv=None) -> int:
    """``python -m tilestream.multigpu {gate,controls,geometry,bench,exchange}``."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=("gate", "forced", "controls", "geometry",
                                    "bench", "exchange", "plan"))
    p.add_argument("--devices", default=None,
                   help="comma-separated card list ranks round-robin over "
                        "(forced only); default = every visible card")
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=_harness.DEFAULT_NZ)
    # NOT one shared default.  ``controls`` needs a longer run than the
    # positive gates do -- its short-halo defect is bit-invisible before step
    # SHORT_HALO_VISIBLE_AT -- and a single ``default=6`` handed that
    # subcommand a run in which the control CANNOT fire, so the module's own
    # documented reproduce line, ``python -m tilestream.multigpu controls``,
    # printed "short_halo: MISSED -- GATE IS BLIND" and FAILED on a healthy
    # tree.  Each subcommand now carries the default its own measurement
    # justifies, and ``None`` is what distinguishes "the caller chose 6" from
    # "nobody said".
    p.add_argument("--steps", type=int, default=None,
                   help="steps to run; defaults per subcommand -- 6 for "
                        f"gate/geometry, {CONTROLS_DEFAULT_STEPS} for "
                        "controls, whose short-halo defect first differs at "
                        f"step {SHORT_HALO_VISIBLE_AT}")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--ngpu", type=int, default=2)
    p.add_argument("--grid", default=None,
                   help="sub-domain grid as GYxGX, e.g. 1x2 (x split), "
                        "2x1 (y split), 2x2.  Overrides --ngpu.")
    p.add_argument("--transport", default="peer",
                   choices=("peer", "default", "host"))
    p.add_argument("--step-mode", default="events",
                   choices=("sequential", "interleaved", "threads", "events"))
    p.add_argument("--exchange-mode", default="events",
                   choices=("blocking", "stream", "events"))
    a = p.parse_args(argv)

    steps = a.steps
    if steps is None:
        steps = CONTROLS_DEFAULT_STEPS if a.what == "controls" else 6
    if a.what == "controls" and steps < SHORT_HALO_VISIBLE_AT:
        print(f"controls REFUSED: --steps {steps} is below "
              f"{SHORT_HALO_VISIBLE_AT}, the step at which the short-halo "
              "defect first differs from the monolithic digest.  A shorter "
              "run cannot fire that control, so the set would report three "
              "results and one artefact of the run length.")
        return 2

    grid = None if a.grid is None else tuple(
        int(v) for v in a.grid.lower().split("x"))
    if a.what == "plan":
        halo = _harness.halo_radius(_harness.make_config(a.nx, a.ny, a.nz))
        gy, gx = grid or (1, a.ngpu)
        specs = plan_split(a.nx, a.ny, halo, gx=gx, gy=gy)
        _spec.validate_plan(specs, a.ny, a.nx)
        print(describe_plan(specs, seam_plan(specs, halo, a.nx, a.ny)))
        return 0
    if a.what == "geometry":
        ok, _ = verify_geometry(_harness.make_config(a.nx, a.ny, a.nz),
                                ngpu=a.ngpu, grid=grid, steps=steps)
        print("geometry:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    if a.what == "gate":
        ok, _ = gate(a.nx, a.ny, a.nz, steps, ngpu=a.ngpu, grid=grid,
                     transport=a.transport)
        print("gate:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    if a.what == "forced":
        devices = None if a.devices is None else [
            int(v) for v in a.devices.split(",") if v.strip() != ""]
        grids = FORCED_GRIDS if grid is None else (grid,)
        # The forced rung has its own domain default: non-square, divisible
        # by 2 on both axes, wide enough for interior >= halo + 1 at 2x2
        # under the padded forced halo.
        nx = 220 if a.nx == 256 else a.nx
        ny = 168 if a.ny == 128 else a.ny
        ok, _ = gate_forced(nx, ny, a.nz, steps, grids=grids,
                            devices=devices)
        print("forced gate:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    if a.what == "controls":
        ok, _ = negative_controls(a.nx, a.ny, a.nz, steps)
        print("negative controls:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    if a.what == "exchange":
        bench_exchange(a.nx, a.ny, a.nz, reps=a.reps, ngpu=a.ngpu,
                       transport=a.transport)
        return 0
    bench_monolithic(a.nx, a.ny, a.nz, reps=a.reps)
    bench(a.nx, a.ny, a.nz, reps=a.reps, ngpu=a.ngpu, step_mode=a.step_mode,
          exchange_mode=a.exchange_mode)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(_main())
