"""Tests for :mod:`tilestream.rings` -- the geometry, exhaustively.

The ring scheme's correctness reduces to one statement:

    every cell a LATER tile reads out of an EARLIER tile's write set is in
    the saved bands.

:func:`tilestream.rings.assert_ring_covers_reads` checks that with rectangle
arithmetic.  This module checks the same thing a second, independent way --
by simulating the whole sweep cell by cell on a full-domain numpy array -- so
that a shared mistake in the rectangle algebra cannot make both agree.  The
two must agree on every plan in :data:`PLANS`, which spans periodic and not,
ragged and not, tiles narrower than the halo, and windows that wrap onto
themselves.

Everything here is pure index arithmetic: no GPU, no gpuwm, seconds to run.
The bit-exact end-to-end checks live in ``test_gate``.
"""

from __future__ import annotations

import numpy as np

from tilestream import rings
from tilestream import spec as tspec
from tilestream.rings import Rect


#: ``(label, kwargs for plan_tiles)``.  Each exercises a different way the
#: naive "band of width halo" answer is wrong.
PLANS: list[tuple[str, dict]] = [
    ("3x3 periodic", dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16)),
    ("2x1 x seam", dict(nx=192, ny=192, tile_nx=96, tile_ny=192, halo=16)),
    ("1x2 y seam", dict(nx=192, ny=192, tile_nx=192, tile_ny=96, halo=16)),
    ("2x2", dict(nx=192, ny=192, tile_nx=96, tile_ny=96, halo=16)),
    ("6x6 tile == 2*halo", dict(nx=192, ny=192, tile_nx=32, tile_ny=32, halo=16)),
    ("1x1 wraps onto itself", dict(nx=192, ny=192, tile_nx=192, tile_ny=192,
                                   halo=16)),
    ("ragged x", dict(nx=200, ny=192, tile_nx=64, tile_ny=64, halo=16)),
    ("ragged both, odd", dict(nx=173, ny=149, tile_nx=48, tile_ny=40, halo=16)),
    ("gate size 2x2", dict(nx=96, ny=80, tile_nx=48, tile_ny=40, halo=16)),
    ("gate 3x3 ragged", dict(nx=96, ny=80, tile_nx=40, tile_ny=30, halo=16)),
    ("4x4 non-square", dict(nx=256, ny=192, tile_nx=64, tile_ny=48, halo=16)),
    ("fat tile 0, self-wrap", dict(nx=100, ny=100, tile_nx=90, tile_ny=90,
                                   halo=16)),
    ("tiny domain, self-wrap", dict(nx=40, ny=40, tile_nx=20, tile_ny=20,
                                    halo=16)),
    ("halo 14 (the tight one)", dict(nx=192, ny=192, tile_nx=64, tile_ny=64,
                                     halo=14)),
    ("NON-PERIODIC 3x3", dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16,
                              periodic=False)),
    ("NON-PERIODIC ragged", dict(nx=200, ny=192, tile_nx=64, tile_ny=64,
                                 halo=16, periodic=False)),
]


def _plan(kw):
    specs = tspec.plan_tiles(**kw)
    tspec.validate_plan(specs, kw["ny"], kw["nx"])
    return specs


def _masks(specs, kind):
    """``(writer, saved_or_None)`` full-array maps for one variant."""
    ny, nx = specs[0].ny, specs[0].nx
    shape = tspec.full_shape(ny, nx, kind)
    writer = np.full(shape, -1, dtype=np.int64)
    for i, s in enumerate(specs):
        for t in s.scatter(kind):
            writer[t.full_y, t.full_x] = i
    return writer, shape


def _needed_mask(specs, kind, writer, shape):
    """Cells read by a tile processed AFTER the tile that writes them."""
    need = np.zeros(shape, dtype=bool)
    for j, s in enumerate(specs):
        read = np.zeros(shape, dtype=bool)
        for t in s.gather(kind):
            read[t.full_y, t.full_x] = True
        need |= read & (writer >= 0) & (writer < j)
    return need


def _saved_mask(plan, kind, shape):
    saved = np.zeros(shape, dtype=bool)
    for band in plan.bands:
        if band.kind == kind:
            r = band.rect
            saved[r.y0:r.y1, r.x0:r.x1] = True
    return saved


# --------------------------------------------------------------------------
# the invariant, twice, independently
# --------------------------------------------------------------------------

def test_saved_bands_cover_every_later_read():
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)         # calls the rect checker
        for kind in rings.GEOMETRY_KINDS:
            writer, shape = _masks(specs, kind)
            need = _needed_mask(specs, kind, writer, shape)
            saved = _saved_mask(plan, kind, shape)
            missing = int((need & ~saved).sum())
            assert missing == 0, (
                f"{label} / {kind}: {missing} cells that a later tile reads "
                "are not saved")


def test_bands_of_one_tile_are_disjoint():
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        for tile in range(plan.ntiles):
            for kind in rings.GEOMETRY_KINDS:
                bands = [b for b in plan.bands
                         if b.tile == tile and b.kind == kind]
                for i, a in enumerate(bands):
                    for b in bands[:i]:
                        assert rings._intersect(a.rect, b.rect) is None, (
                            f"{label}: tile {tile} bands {a.rect} and "
                            f"{b.rect} overlap; a cell would be saved twice "
                            "and the arena offsets would double-count")


def test_bands_lie_inside_the_tile_that_owns_them():
    """A band must be part of its own tile's write set, never a neighbour's."""
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        for band in plan.bands:
            owner = specs[band.tile]
            covered = 0
            for t in owner.scatter(band.kind):
                overlap = rings._intersect(
                    band.rect,
                    Rect(t.full_y.start, t.full_y.stop,
                         t.full_x.start, t.full_x.stop))
                if overlap is not None:
                    covered += rings._area(overlap)
            assert covered == band.cells, (
                f"{label}: band {band.rect} of tile {band.tile} is not "
                "entirely inside that tile's write set")


def test_arena_offsets_tile_the_plane_exactly():
    """Every band gets its own storage, and none of it is wasted or shared."""
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        per_kind: dict[str, list] = {}
        for band in plan.bands:
            per_kind.setdefault(band.kind, []).append(band)
        for kind, bands in per_kind.items():
            spans = sorted((b.plane_offset, b.plane_offset + b.cells)
                           for b in bands)
            cursor = 0
            for start, stop in spans:
                assert start == cursor, (
                    f"{label}/{kind}: arena has a hole or an overlap at "
                    f"{start} (expected {cursor})")
                cursor = stop
            assert cursor == plan.plane_cells[kind], (
                f"{label}/{kind}: arena is {plan.plane_cells[kind]} cells but "
                f"the bands use {cursor}")


def test_patches_restore_exactly_the_cells_the_gather_got_wrong():
    """For every tile: patched cells == (window cells written by an earlier tile).

    Simulated cell by cell in the TILE's own coordinates, which is where the
    error would actually land -- a patch aimed at the right domain cell but
    the wrong tile-local offset passes every rectangle check and still
    corrupts the halo.
    """
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        for kind in rings.GEOMETRY_KINDS:
            writer, shape = _masks(specs, kind)
            for j, s in enumerate(specs):
                th, tw = s.shape(kind)
                # what the tile array holds after the base gather: the domain
                # index each tile cell came from, and who wrote it
                came_from = np.full((th, tw), -1, dtype=np.int64)
                for t in s.gather(kind):
                    src = (writer[t.full_y, t.full_x])
                    came_from[t.tile_y, t.tile_x] = src
                stale = (came_from >= 0) & (came_from < j)
                patched = np.zeros((th, tw), dtype=bool)
                for p in plan.patches[j]:
                    if p.band.kind != kind:
                        continue
                    patched[p.ty0:p.ty0 + p.h, p.tx0:p.tx0 + p.w] = True
                assert not (stale & ~patched).any(), (
                    f"{label}/{kind}: tile {j} leaves "
                    f"{int((stale & ~patched).sum())} stale cells unpatched")
                assert not (patched & ~stale).any(), (
                    f"{label}/{kind}: tile {j} patches "
                    f"{int((patched & ~stale).sum())} cells that were never "
                    "stale; the patch is aimed at the wrong tile offset")


def test_patch_sources_are_the_right_cells_of_the_right_band():
    """The value a patch moves must be the one that domain cell had at time t.

    Follows every patch block from its arena coordinates back to a domain
    cell (through the band) and forward to a tile cell (through the gather),
    and demands the two name the same domain point.  This is the check that a
    transposed or mis-signed offset fails.
    """
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        for j, s in enumerate(specs):
            # tile cell -> domain cell, per kind, from the gather itself
            maps = {}
            for kind in rings.GEOMETRY_KINDS:
                th, tw = s.shape(kind)
                dom = np.full((th, tw, 2), -1, dtype=np.int64)
                for t in s.gather(kind):
                    ys = np.arange(t.full_y.start, t.full_y.stop)
                    xs = np.arange(t.full_x.start, t.full_x.stop)
                    dom[t.tile_y, t.tile_x, 0] = ys[:, None]
                    dom[t.tile_y, t.tile_x, 1] = xs[None, :]
                maps[kind] = dom
            for p in plan.patches[j]:
                dom = maps[p.band.kind]
                for dy in (0, p.h - 1):
                    for dx in (0, p.w - 1):
                        want = (p.band.rect.y0 + p.by0 + dy,
                                p.band.rect.x0 + p.bx0 + dx)
                        got = tuple(dom[p.ty0 + dy, p.tx0 + dx])
                        assert got == want, (
                            f"{label}: tile {j} patch of band "
                            f"{p.band.rect} puts domain cell {want} at tile "
                            f"cell {(p.ty0 + dy, p.tx0 + dx)}, which the "
                            f"gather filled from {got}")


def test_saves_read_the_right_tile_cells():
    """A save must read the band's own cells out of the tile that owns them."""
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        for k, s in enumerate(specs):
            maps = {}
            for kind in rings.GEOMETRY_KINDS:
                th, tw = s.shape(kind)
                dom = np.full((th, tw, 2), -1, dtype=np.int64)
                for t in s.gather(kind):
                    ys = np.arange(t.full_y.start, t.full_y.stop)
                    xs = np.arange(t.full_x.start, t.full_x.stop)
                    dom[t.tile_y, t.tile_x, 0] = ys[:, None]
                    dom[t.tile_y, t.tile_x, 1] = xs[None, :]
                maps[kind] = dom
            for sv in plan.saves[k]:
                dom = maps[sv.band.kind]
                for dy in (0, sv.h - 1):
                    for dx in (0, sv.w - 1):
                        want = (sv.band.rect.y0 + dy, sv.band.rect.x0 + dx)
                        got = tuple(dom[sv.ty0 + dy, sv.tx0 + dx])
                        assert got == want, (
                            f"{label}: tile {k} saves tile cell "
                            f"{(sv.ty0 + dy, sv.tx0 + dx)} as domain {want}, "
                            f"but the gather filled it from {got}")


# --------------------------------------------------------------------------
# the dependency lists the loop orders on
# --------------------------------------------------------------------------

def test_war_deps_name_every_earlier_reader():
    """``war_deps[k]`` must list every j < k whose window reads tile k's interior.

    That list is what stops tile k's scatter from overtaking tile j's gather.
    A missing entry is a race that this card's single DMA queue would hide
    completely.
    """
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        for k in range(plan.ntiles):
            expect = set()
            for kind in rings.GEOMETRY_KINDS:
                writes = rings._transfer_rects(specs[k], "scatter", kind)
                for j in range(k):
                    reads = rings._transfer_rects(specs[j], "gather", kind)
                    for w, _wt in writes:
                        for r, _rt in reads:
                            if rings._intersect(r, w) is not None:
                                expect.add(j)
            assert set(plan.war_deps[k]) == expect, (
                f"{label}: tile {k} war_deps {plan.war_deps[k]} != {sorted(expect)}")


def test_patch_deps_match_the_patches():
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs)
        for j in range(plan.ntiles):
            from_patches = {p.band.tile for p in plan.patches[j]}
            assert set(plan.patch_deps[j]) == from_patches, label
            assert all(k < j for k in plan.patch_deps[j]), (
                f"{label}: tile {j} patches from a tile that has not been "
                "saved yet")


# --------------------------------------------------------------------------
# the two deliberately broken modes
# --------------------------------------------------------------------------

def test_halo_margin_mode_is_unsound_and_says_so():
    """``margin_mode='halo'`` must be REFUSED by the invariant, not silently OK.

    It is the mistake that looks right, and the gate cannot see it: a
    halo-wide ring produces bit-exact answers at every halo that produces a
    correct answer at all (MEASURED at 14, 15 and 16), because the cells it
    drops sit at the outer edge of the reading window, outside the influence
    cone.  The rectangle invariant is the only instrument that catches it.
    """
    unsound = 0
    for label, kw in PLANS:
        specs = _plan(kw)
        plan = rings.build_ring_plan(specs, margin_mode="halo")
        try:
            rings.assert_ring_covers_reads(plan)
        except rings.RingError:
            unsound += 1
    assert unsound >= 8, (
        f"only {unsound} of {len(PLANS)} plans expose the halo-wide ring as "
        "unsound; this control has stopped controlling anything")


def test_x_only_mode_drops_the_y_bands():
    specs = _plan(dict(nx=192, ny=192, tile_nx=64, tile_ny=64, halo=16))
    plan = rings.build_ring_plan(specs, margin_mode="x_only")
    full = rings.build_ring_plan(specs)
    assert sum(b.cells for b in plan.bands) < sum(b.cells for b in full.bands)
    try:
        rings.assert_ring_covers_reads(plan)
    except rings.RingError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("x_only should not cover the y reads")


def test_unknown_margin_mode_raises():
    specs = _plan(dict(nx=64, ny=64, tile_nx=32, tile_ny=32, halo=8))
    try:
        rings.build_ring_plan(specs, margin_mode="whatever")
    except ValueError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("an unknown margin mode must raise")


# --------------------------------------------------------------------------
# size
# --------------------------------------------------------------------------

def test_ring_is_far_smaller_than_a_shadow_at_a_realistic_tile():
    """The whole point: the arena must be a few per cent, not 100%."""
    for nx, tile, want_max in ((1950, 650, 0.06), (3276, 546, 0.07),
                               (4096, 512, 0.08)):
        specs = tspec.plan_tiles(nx, nx, tile, tile, 16, True)
        plan = rings.build_ring_plan(specs)
        report = rings.ring_report(plan)
        assert report["ring_fraction"] < want_max, (
            f"{nx}^2 tile {tile}: ring is "
            f"{100 * report['ring_fraction']:.2f}% of the domain, expected "
            f"under {100 * want_max:.0f}%")


def test_lazy_saving_beats_the_four_sided_estimate():
    """A band only an EARLIER tile reads is not saved -- worth about 2x."""
    specs = tspec.plan_tiles(1950, 1950, 650, 650, 16, True)
    plan = rings.build_ring_plan(specs)
    saved = sum(b.cells for b in plan.bands) / len(plan.kinds)
    h = 16.0
    four_sided = sum(
        s.interior_ny * s.interior_nx
        - max(s.interior_ny - 2 * h, 0) * max(s.interior_nx - 2 * h, 0)
        for s in specs)
    assert saved < 0.62 * four_sided, (
        f"lazy saving keeps {saved} cells against the four-sided "
        f"{four_sided}; the ordering argument has stopped paying")


def test_single_tile_plan_needs_no_ring_at_all():
    specs = tspec.plan_tiles(192, 192, 192, 192, 16, True)
    plan = rings.build_ring_plan(specs)
    assert plan.bands == []
    assert plan.patches == [[]]
    assert plan.war_deps == [()]


def _run_all():
    fns = [(name, obj) for name, obj in sorted(globals().items())
           if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:                     # pragma: no cover
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
