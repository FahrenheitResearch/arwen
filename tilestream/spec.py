"""Tile geometry for the out-of-core streaming prototype: pure index arithmetic.

This module owns every slice the streaming pipeline uses.  It never touches a
GPU, imports neither cupy nor gpuwm, and has no state -- a :class:`TileSpec` is
a frozen description of *where* a tile lives in the full domain and *where* its
cells land in the tile array.  ``gather.py`` / ``hoststore.py`` / ``driver.py``
consume it; if the arithmetic here is wrong, everything downstream is wrong in
a way that looks like a working speedup.


The grid and the staggering
---------------------------
Arrays are ``(nz, ny, nx)`` with x fastest.  Only the trailing two axes are
tiled; the vertical is never split.  WRF-ARW staggering means the four
horizontal *variants* have different array shapes for the same domain:

===========  ===================  ===========================================
variant      full-array shape     what an index means
===========  ===================  ===========================================
``"mass"``   ``(.., ny,   nx)``   cell centre ``i`` spans ``[i, i+1) * dx``
``"u"``      ``(.., ny,   nx+1)`` face ``i`` is the LEFT face of mass cell i
``"v"``      ``(.., ny+1, nx)``   face ``j`` is the SOUTH face of mass cell j
``"w"``      ``(.., ny,   nx)``   same horizontal layout as mass (``w``/``phi``
                                  stagger in z only)
===========  ===================  ===========================================

A tile covering mass columns ``[i0, i1)`` therefore needs ``u`` over the faces
``[i0, i1]`` -- one MORE than the mass count -- and ``v`` over ``[j0, j1]``.
Slicing every array identically is the single easiest way to produce a subtly
wrong forecast instead of a crash, so this module never lets a caller do it:
you ask for a variant and you get that variant's slices.


Periodicity, and the alias slot
-------------------------------
On a fully periodic domain there are only ``nx`` INDEPENDENT u faces.  The
array still carries ``nx+1`` of them and the convention -- the one
``harness.make_state`` seeds (``vals[..., -1] = vals[..., 0]``) and the dycore
maintains -- is that the last slot is a duplicate::

    u[..., nx] == u[..., 0]        v[..., ny, :] == v[..., 0, :]

This module calls slot ``nx`` the ALIAS SLOT.  Two consequences, both
deliberate:

* **Gathers never read the alias slot.**  A logical face index is reduced
  ``mod nx`` and read from ``[0, nx)``.  Reading slot ``nx`` instead would be
  numerically identical *only if* the duplicate invariant currently holds;
  reducing mod nx is correct by the definition of periodicity and holds even on
  a store whose duplicate has gone stale.  The cost is at most one extra copy
  segment per axis.
* **Exactly one tile writes the alias slot.**  Under ``periodic=True`` it is
  the tile whose interior contains mass column 0 (it copies its own logical
  face 0 into full slot ``nx``), which keeps the invariant true after every
  scatter.  Under ``periodic=False`` slot ``nx`` is a real boundary face and is
  owned by the last tile in x.  Either way the coverage count over the whole
  ``nx+1``-wide array is exactly one.


Who owns a shared face
----------------------
Adjacent tiles share the face between them.  **A tile owns the LEFT/SOUTH face
of every mass cell it owns** -- interior u faces are ``[i0, i1)``, interior v
faces are ``[j0, j1)`` -- plus the alias/boundary slot described above.  Face
``i1`` belongs to the next tile, as its own left face.  This gives no
double-writes and no unwritten faces, which the coverage test asserts by
counting.

The owned faces are far enough from the tile edge to be trustworthy: interior
u face ``i0`` sits at tile-local index ``halo``, i.e. ``halo`` cells from the
gathered array's edge, and the last owned face sits ``halo+1`` from the other
edge.


Ragged edges: pad the COMPUTE window, ragged the INTERIOR
---------------------------------------------------------
``nx`` need not divide by ``tile_nx``.  The policy here, chosen so that one
CUDA graph can serve every tile:

* the **compute window is always exactly** ``(tile_ny + 2*halo, tile_nx +
  2*halo)`` mass cells -- every tile, including the trailing ones.  Tile array
  shapes are uniform for a given variant, always.
* the **interior is ragged**: the trailing tile's interior is
  ``[i0, min(i0 + tile_nx, nx))``, which can be narrower than ``tile_nx``.  The
  compute window keeps its full width, so a short trailing tile simply carries
  a wider right halo (which, being periodic, is a redundant recomputation of
  cells that tile 0 owns).  Those cells are computed and thrown away.

So: uniform shapes, ragged ownership.  ``TileSpec.ragged_x`` /
``ragged_y`` say which tiles are short, and ``interior_nx`` / ``interior_ny``
give the real widths.  Nothing outside ``[i0, i1) x [j0, j1)`` is ever written
back, so the wasted work is bounded by one tile row/column of overlap.


Non-periodic domains
--------------------
``periodic=False`` is index arithmetic only -- milestone one is periodic and
the lateral-BC question is deferred.  What this module does under it: the
compute window is CLAMPED into ``[0, nx]`` instead of wrapping, by sliding the
window inward (``ci0 = clip(i0 - halo, 0, nx - cnx)``) rather than truncating
it.  Sliding keeps the uniform shape and keeps every gathered cell real; the
price is that the interior is off-centre in edge tiles (``halo_left`` /
``halo_right`` report the actual margins, and the margin on a domain-edge side
is smaller than ``halo`` because the domain boundary is right there).  It
requires ``tile_nx + 2*halo <= nx`` and raises otherwise.  A caller must still
decide what boundary condition an edge tile is stepped with; this module does
not know and does not pretend to.


Tiles wider than the domain
---------------------------
Under ``periodic=True`` a compute window wider than ``nx`` is supported: the
gather wraps onto itself and the tile array holds the domain repeated.  The
interior is still exact (its ``halo``-cell surround is a faithful periodic
image either way), but the tile contains duplicated columns and the work is
mostly wasted -- ``TileSpec.wraps_onto_itself`` flags it.  Under
``periodic=False`` it is a ``ValueError``.


Periodicity is PER AXIS, and it had to become so
------------------------------------------------
``periodic=`` sets both axes and is still the ordinary way to call
:func:`plan_tiles`; ``periodic_x=`` / ``periodic_y=`` override one axis.
That is not generality for its own sake.  ``cfg.open_x=True, open_y=False``
is a real WRF configuration -- the classic squall-line domain, radiative-open
across the flow and PERIODIC along it -- and for it ``dycore._boundary_x`` is
true while ``_boundary_y`` is false, so every y stencil in the model wraps.

A single plan-wide flag has no way to say that.  ``autoplan.is_periodic``
returned False (correctly: x is not periodic), the whole plan clamped, and
the south-most tile's row 0 became an owned cell with NO halo beneath it
while the kernels stepping it wrapped in y to row ``cny-1`` OF THE WINDOW --
a row 64 cells away instead of the domain's row ``ny-1``.

MEASURED, 256x192x49 at dx=12 km, tile 32x32, halo 16, one dry step,
``open_x=True, open_y=False``: all nine carriers differ, and the difference
is exactly rows 0-11 and 180-191 at every x -- the two y-boundary tile rows,
nothing else.  The x axis, where the open boundary actually lives, was clean.
With ``periodic_y=True`` the same case is bit-exact at N = 1, 3 and 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

#: The horizontal variants, and their staggering as ``(extra_y, extra_x)``
#: points relative to the mass grid.  ``"phi"`` is an alias of ``"w"``.
VARIANT_STAGGER: dict[str, tuple[int, int]] = {
    "mass": (0, 0),
    "u": (0, 1),
    "v": (1, 0),
    "w": (0, 0),
    "phi": (0, 0),
}

#: Canonical variant list for coverage checks -- ``phi`` is omitted because it
#: is bit-for-bit the same geometry as ``w``.
VARIANTS: tuple[str, ...] = ("mass", "u", "v", "w")

#: Default per-step horizontal dependency radius, in mass cells.  Correct for
#: ``time_step_sound=4`` ONLY; use ``tilestream.harness.halo_radius(cfg)``
#: (``10 + 3*ns//2``) whenever a config is in hand.
DEFAULT_HALO = 16


def stagger(variant: str) -> tuple[int, int]:
    """``(extra_y, extra_x)`` for ``variant``; raises on an unknown name."""
    try:
        return VARIANT_STAGGER[variant]
    except KeyError:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of "
            f"{sorted(VARIANT_STAGGER)}") from None


def full_shape(ny: int, nx: int, variant: str) -> tuple[int, int]:
    """Horizontal shape of the FULL-domain array for ``variant``."""
    ey, ex = stagger(variant)
    return (ny + ey, nx + ex)


def tile_shape(tile_ny: int, tile_nx: int, halo: int,
               variant: str) -> tuple[int, int]:
    """Horizontal shape of the TILE array for ``variant``.

    Depends only on the plan parameters, not on which tile: shapes are
    uniform by construction (see the ragged-edge policy in the module
    docstring), which is what lets one CUDA graph serve every tile.
    """
    ey, ex = stagger(variant)
    return (tile_ny + 2 * halo + ey, tile_nx + 2 * halo + ex)


# ---------------------------------------------------------------------------
# 1-D segments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """One contiguous run along one axis: ``full[full] <-> tile[tile]``.

    Both slices have unit stride and identical length.  ``full`` indexes the
    full-domain array, ``tile`` indexes the tile array, whichever direction the
    copy actually goes.
    """

    full: slice
    tile: slice

    def __post_init__(self) -> None:
        fl = self.full.stop - self.full.start
        tl = self.tile.stop - self.tile.start
        if self.full.step not in (None, 1) or self.tile.step not in (None, 1):
            raise ValueError("segments must have unit stride")
        if fl != tl or fl <= 0:
            raise ValueError(f"segment length mismatch: {self.full} vs {self.tile}")

    @property
    def length(self) -> int:
        return self.full.stop - self.full.start


def _merge(segments: Sequence[Segment]) -> tuple[Segment, ...]:
    """Join runs that are contiguous on BOTH sides.  Order preserved."""
    out: list[Segment] = []
    for seg in segments:
        if out:
            prev = out[-1]
            if (prev.full.stop == seg.full.start
                    and prev.tile.stop == seg.tile.start):
                out[-1] = Segment(slice(prev.full.start, seg.full.stop),
                                  slice(prev.tile.start, seg.tile.stop))
                continue
        out.append(seg)
    return tuple(out)


def _wrap_segments(origin: int, count: int, n: int) -> tuple[Segment, ...]:
    """Walk ``count`` points from logical ``origin``, reduced mod ``n``.

    Emits an ordered run of segments whose ``tile`` slices tile ``[0, count)``
    exactly once and whose ``full`` slices stay inside ``[0, n)``.  Handles
    ``count > n`` (the window wraps onto itself and re-reads the domain).
    """
    if count <= 0:
        return ()
    segments: list[Segment] = []
    pos = 0
    k = origin % n
    while pos < count:
        take = min(n - k, count - pos)
        segments.append(Segment(slice(k, k + take), slice(pos, pos + take)))
        pos += take
        k = 0 if k + take >= n else k + take
    return _merge(segments)


# ---------------------------------------------------------------------------
# 2-D transfers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transfer:
    """One rectangular copy between a full-domain array and a tile array.

    Both arrays are ``(..., y, x)``; the leading axes (z, and anything else)
    are copied whole via ``Ellipsis``.  Use :attr:`full_key` / :attr:`tile_key`
    with any array library -- numpy and cupy both accept them.
    """

    full_y: slice
    full_x: slice
    tile_y: slice
    tile_x: slice

    @property
    def full_key(self) -> tuple:
        return (Ellipsis, self.full_y, self.full_x)

    @property
    def tile_key(self) -> tuple:
        return (Ellipsis, self.tile_y, self.tile_x)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.full_y.stop - self.full_y.start,
                self.full_x.stop - self.full_x.start)

    @property
    def count(self) -> int:
        h, w = self.shape
        return h * w

    def copy_to_tile(self, full_array, tile_array) -> None:
        """``tile[...] = full[...]`` for this rectangle."""
        tile_array[self.tile_key] = full_array[self.full_key]

    def copy_to_full(self, tile_array, full_array) -> None:
        """``full[...] = tile[...]`` for this rectangle."""
        full_array[self.full_key] = tile_array[self.tile_key]


def _rectangles(y_segments: Iterable[Segment],
                x_segments: Iterable[Segment]) -> tuple[Transfer, ...]:
    ys = tuple(y_segments)
    xs = tuple(x_segments)
    return tuple(Transfer(y.full, x.full, y.tile, x.tile)
                 for y in ys for x in xs)


# ---------------------------------------------------------------------------
# TileSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileSpec:
    """Geometry of one tile.

    All coordinates are MASS-cell coordinates unless a name says otherwise.

    Interior (``i0:i1``, ``j0:j1``) is the set of mass columns this tile is
    responsible for.  It never wraps: ``0 <= i0 < i1 <= nx``.  It is what the
    scatter writes and nothing else.

    Compute window (``ci0``, ``cnx``) is the interior grown by ``halo``.  On a
    PERIODIC axis ``ci0`` may be negative or run past ``nx`` -- it is a
    LOGICAL coordinate, reduced mod ``nx`` when a gather is built.  On a
    non-periodic one it is clamped so that ``0 <= ci0`` and
    ``ci0 + cnx <= nx``.

    ``periodic_x`` and ``periodic_y`` are independent because the model's own
    ``dycore._boundary_x`` / ``_boundary_y`` are (see the module docstring).
    ``periodic`` remains readable and means "both axes wrap".
    """

    ty: int
    tx: int
    nx: int
    ny: int
    tile_nx: int
    tile_ny: int
    halo: int
    periodic_x: bool
    periodic_y: bool
    i0: int
    i1: int
    j0: int
    j1: int
    ci0: int
    cj0: int
    cnx: int
    cny: int

    # -- basic geometry ----------------------------------------------------

    @property
    def periodic(self) -> bool:
        """Both axes wrap.  The old plan-wide flag, kept as a reader."""
        return self.periodic_x and self.periodic_y

    @property
    def index(self) -> tuple[int, int]:
        """``(ty, tx)`` position in the tile grid."""
        return (self.ty, self.tx)

    @property
    def interior_nx(self) -> int:
        return self.i1 - self.i0

    @property
    def interior_ny(self) -> int:
        return self.j1 - self.j0

    @property
    def ragged_x(self) -> bool:
        """True if this tile owns fewer than ``tile_nx`` columns."""
        return self.interior_nx < self.tile_nx

    @property
    def ragged_y(self) -> bool:
        return self.interior_ny < self.tile_ny

    @property
    def halo_left(self) -> int:
        """Mass cells between the compute-window edge and the interior."""
        return self.i0 - self.ci0

    @property
    def halo_right(self) -> int:
        return (self.ci0 + self.cnx) - self.i1

    @property
    def halo_south(self) -> int:
        return self.j0 - self.cj0

    @property
    def halo_north(self) -> int:
        return (self.cj0 + self.cny) - self.j1

    @property
    def wraps_onto_itself(self) -> bool:
        """True if the compute window is wider/taller than the domain."""
        return self.cnx > self.nx or self.cny > self.ny

    @property
    def owns_x_alias(self) -> bool:
        """True if this tile writes u's ``nx`` slot (see module docstring)."""
        return self.i0 == 0 if self.periodic_x else self.i1 == self.nx

    @property
    def owns_y_alias(self) -> bool:
        """True if this tile writes v's ``ny`` slot."""
        return self.j0 == 0 if self.periodic_y else self.j1 == self.ny

    def shape(self, variant: str) -> tuple[int, int]:
        """Horizontal shape of this tile's array for ``variant``."""
        ey, ex = stagger(variant)
        return (self.cny + ey, self.cnx + ex)

    # -- gather ------------------------------------------------------------

    def _axis_gather(self, origin: int, count: int, n: int,
                     extra: int, periodic: bool) -> tuple[Segment, ...]:
        """Segments filling ``count + extra`` tile points from ``origin``.

        ``count`` is the mass-cell count of the window; ``extra`` is 1 on the
        axis this variant is staggered along, which asks for one more point
        (the closing face).  ``periodic`` is THIS AXIS's flag: the two axes
        are independent (see the module docstring), and this method is called
        once per axis, so the split costs nothing here.
        """
        total = count + extra
        if periodic:
            # Reduce mod n and never touch the alias slot; see module docstring.
            return _wrap_segments(origin, total, n)
        if origin < 0 or origin + count > n:
            raise AssertionError(  # pragma: no cover - guarded at plan time
                f"non-periodic window [{origin}, {origin + count}) escapes "
                f"[0, {n}]")
        return (Segment(slice(origin, origin + total), slice(0, total)),)

    def gather(self, variant: str) -> tuple[Transfer, ...]:
        """Rectangles copying the full-domain array INTO the tile array.

        Together they fill the whole tile array for ``variant`` -- every point,
        halo included -- exactly once.  On a periodic axis a window that runs
        off an edge produces several rectangles; issue them in order.
        """
        ey, ex = stagger(variant)
        ys = self._axis_gather(self.cj0, self.cny, self.ny, ey,
                               self.periodic_y)
        xs = self._axis_gather(self.ci0, self.cnx, self.nx, ex,
                               self.periodic_x)
        return _rectangles(ys, xs)

    # -- scatter -----------------------------------------------------------

    def _axis_scatter(self, a0: int, a1: int, c0: int, n: int,
                      extra: int, owns_alias: bool,
                      periodic: bool) -> tuple[Segment, ...]:
        off = a0 - c0                       # tile-local index of logical a0
        width = a1 - a0
        segments = [Segment(slice(a0, a1), slice(off, off + width))]
        if extra and owns_alias:
            if periodic:
                # logical face 0 -> alias slot n.  This tile owns column 0, so
                # its own left face sits at tile index `off`.
                segments.append(Segment(slice(n, n + 1),
                                        slice(off, off + 1)))
            else:
                # a1 == n: the real closing boundary face, one past the last
                # owned mass cell, contiguous with the run above.
                segments.append(Segment(slice(n, n + 1),
                                        slice(off + width, off + width + 1)))
        return _merge(segments)

    def scatter(self, variant: str) -> tuple[Transfer, ...]:
        """Rectangles copying the tile array's INTERIOR back to the domain.

        The halo ring is contaminated by the tile's own wrap-around and is
        never written.  Shared faces follow the left/south ownership rule, so
        across a whole plan every point of the full array is written exactly
        once (see :func:`coverage_counts`).
        """
        ey, ex = stagger(variant)
        ys = self._axis_scatter(self.j0, self.j1, self.cj0, self.ny, ey,
                                self.owns_y_alias, self.periodic_y)
        xs = self._axis_scatter(self.i0, self.i1, self.ci0, self.nx, ex,
                                self.owns_x_alias, self.periodic_x)
        return _rectangles(ys, xs)

    def interior_in_tile(self, variant: str) -> tuple[slice, slice]:
        """``(y_slice, x_slice)`` of the owned points within the tile array.

        Contiguous by construction.  Note this EXCLUDES the alias slot, which
        lives at the far end of the full array and has no matching contiguous
        home in the tile; use :meth:`scatter` for the complete write set.
        """
        ey, ex = stagger(variant)
        oy = self.j0 - self.cj0
        ox = self.i0 - self.ci0
        return (slice(oy, oy + self.interior_ny + (ey if self.owns_y_alias
                                                   and not self.periodic_y
                                                   else 0)),
                slice(ox, ox + self.interior_nx + (ex if self.owns_x_alias
                                                   and not self.periodic_x
                                                   else 0)))

    # -- convenience -------------------------------------------------------

    def apply_gather(self, full_array, tile_array, variant: str) -> None:
        """Run every gather rectangle.  Works with numpy or cupy arrays."""
        for transfer in self.gather(variant):
            transfer.copy_to_tile(full_array, tile_array)

    def apply_scatter(self, tile_array, full_array, variant: str) -> None:
        """Run every scatter rectangle.  Works with numpy or cupy arrays."""
        for transfer in self.scatter(variant):
            transfer.copy_to_full(tile_array, full_array)

    def describe(self) -> str:
        return (f"tile({self.ty},{self.tx}) "
                f"interior y[{self.j0}:{self.j1}) x[{self.i0}:{self.i1}) "
                f"compute y[{self.cj0}:{self.cj0 + self.cny}) "
                f"x[{self.ci0}:{self.ci0 + self.cnx})"
                + (" ragged" if self.ragged_x or self.ragged_y else ""))


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def plan_tiles(nx: int, ny: int, tile_nx: int, tile_ny: int,
               halo: int = DEFAULT_HALO,
               periodic: bool = True, *,
               periodic_x: bool | None = None,
               periodic_y: bool | None = None) -> list[TileSpec]:
    """Cover an ``ny x nx`` mass grid with tiles of interior ``tile_ny x tile_nx``.

    Returns row-major order (``ty`` outer, ``tx`` inner).  Every returned tile
    has the same compute shape, ``(tile_ny + 2*halo, tile_nx + 2*halo)`` mass
    cells; trailing tiles may own a narrower interior (see the ragged-edge
    policy in the module docstring).  The interiors partition the domain
    exactly.

    ``periodic`` sets both axes; ``periodic_x`` / ``periodic_y`` override one.
    They must come from the CONFIG -- ``autoplan.is_periodic_x`` /
    ``is_periodic_y``, which are ``dycore._boundary_x`` / ``_boundary_y``
    negated -- and never from a guess: a plan that clamps an axis the model
    wraps hands the boundary tile an owned row with no halo under it, and the
    kernels then wrap it to the far side of the WINDOW.  See the module
    docstring for the measurement.

    ``halo`` must be the per-step dependency radius for the config that will
    step the tiles -- ``tilestream.harness.halo_radius(cfg)``, which is 16 only
    when ``time_step_sound == 4``.  A halo that is too small silently corrupts
    tile interiors and looks like a speedup.
    """
    nx, ny = int(nx), int(ny)
    tile_nx, tile_ny = int(tile_nx), int(tile_ny)
    halo = int(halo)
    px = bool(periodic if periodic_x is None else periodic_x)
    py = bool(periodic if periodic_y is None else periodic_y)

    for name, value in (("nx", nx), ("ny", ny),
                        ("tile_nx", tile_nx), ("tile_ny", tile_ny)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if halo < 0:
        raise ValueError(f"halo must be >= 0, got {halo}")

    cnx = tile_nx + 2 * halo
    cny = tile_ny + 2 * halo
    for axis, wide, span, flag in (("x", cnx, nx, px), ("y", cny, ny, py)):
        if not flag and wide > span:
            raise ValueError(
                f"a non-periodic {axis} axis needs the compute window to fit "
                f"inside the domain: tile+2*halo = {wide} exceeds "
                f"n{axis} = {span}.  Use a smaller tile or halo, or a "
                f"periodic {axis} axis (which wraps instead of clamping).")

    n_tx = -(-nx // tile_nx)                # ceil
    n_ty = -(-ny // tile_ny)

    specs: list[TileSpec] = []
    for ty in range(n_ty):
        j0 = ty * tile_ny
        j1 = min(j0 + tile_ny, ny)
        cj0 = j0 - halo if py else min(max(j0 - halo, 0), ny - cny)
        for tx in range(n_tx):
            i0 = tx * tile_nx
            i1 = min(i0 + tile_nx, nx)
            ci0 = i0 - halo if px else min(max(i0 - halo, 0), nx - cnx)
            specs.append(TileSpec(
                ty=ty, tx=tx, nx=nx, ny=ny,
                tile_nx=tile_nx, tile_ny=tile_ny,
                halo=halo, periodic_x=px, periodic_y=py,
                i0=i0, i1=i1, j0=j0, j1=j1,
                ci0=ci0, cj0=cj0, cnx=cnx, cny=cny,
            ))
    return specs


def coverage_counts(specs: Sequence[TileSpec], ny: int, nx: int,
                    variant: str):
    """Per-point scatter-write count over the full array, as a numpy array.

    A correct plan gives all ones.  This is the assertion the test suite makes
    for every variant; it is exported so a driver can self-check a plan it was
    handed rather than trusting it.
    """
    import numpy as np

    shape = full_shape(ny, nx, variant)
    counts = np.zeros(shape, dtype=np.int64)
    for spec in specs:
        for transfer in spec.scatter(variant):
            counts[transfer.full_y, transfer.full_x] += 1
    return counts


def validate_plan(specs: Sequence[TileSpec], ny: int, nx: int) -> None:
    """Raise unless the plan writes every point of every variant exactly once."""
    import numpy as np

    if not specs:
        raise ValueError("empty plan")
    for variant in VARIANTS:
        counts = coverage_counts(specs, ny, nx, variant)
        if not np.array_equal(counts, np.ones_like(counts)):
            bad = np.unique(counts)
            raise ValueError(
                f"variant {variant!r} coverage is not exactly 1 "
                f"(observed counts {bad.tolist()}, "
                f"{int((counts != 1).sum())} bad points of {counts.size})")


def plan_efficiency(specs: Sequence[TileSpec]) -> dict[str, float]:
    """How much work a plan wastes, in mass cells.

    ``redundancy`` is compute cells divided by domain cells: 1.0 would be a
    free tiling, 4.0 means the halos quadruple the arithmetic.  It is the
    number that decides tile size -- big tiles waste less compute but need more
    VRAM, and the halo cost grows as ``(1 + 2h/T)**2``.  ``gathered`` counts
    the mass cells pulled over PCIe per sweep and ``scattered`` the cells
    pushed back; with no full duplex on this card their SUM is what the link
    budget sees.
    """
    if not specs:
        raise ValueError("empty plan")
    domain = float(specs[0].nx * specs[0].ny)
    compute = float(sum(s.cnx * s.cny for s in specs))
    interior = float(sum(s.interior_nx * s.interior_ny for s in specs))
    return {
        "tiles": float(len(specs)),
        "domain_cells": domain,
        "compute_cells": compute,
        "redundancy": compute / domain,
        "gathered_cells": compute,
        "scattered_cells": interior,
    }


def iter_tiles(nx: int, ny: int, tile_nx: int, tile_ny: int,
               halo: int = DEFAULT_HALO,
               periodic: bool = True, *,
               periodic_x: bool | None = None,
               periodic_y: bool | None = None) -> Iterator[TileSpec]:
    """Streaming form of :func:`plan_tiles`."""
    yield from plan_tiles(nx, ny, tile_nx, tile_ny, halo, periodic,
                          periodic_x=periodic_x, periodic_y=periodic_y)


__all__ = [
    "DEFAULT_HALO", "VARIANTS", "VARIANT_STAGGER",
    "Segment", "TileSpec", "Transfer",
    "coverage_counts", "full_shape", "iter_tiles", "plan_efficiency",
    "plan_tiles", "stagger", "tile_shape", "validate_plan",
]
