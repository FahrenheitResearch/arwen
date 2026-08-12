"""Tests for tilestream.spec -- numpy only, no GPU, no gpuwm import.

The three gates the task names are :func:`test_coverage_exactly_once`,
:func:`test_round_trip` and :func:`test_halo_matches_direct_indexing`; the rest
guard the traps around them (alias slot, shared-face ownership, uniform shapes,
tiles wider than the domain, zero halo, error paths).

Run with ``pytest tilestream/test_spec.py`` or directly with ``python
tilestream/test_spec.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilestream.spec import (  # noqa: E402
    DEFAULT_HALO, VARIANTS, Segment, TileSpec, Transfer, coverage_counts,
    full_shape, plan_efficiency, plan_tiles, stagger, tile_shape,
    validate_plan,
)

NZ = 3

# (nx, ny, tile_nx, tile_ny, halo)
PERIODIC_CASES = [
    (128, 128, 32, 32, 16),     # clean divide, the milestone-one shape
    (100, 80, 32, 24, 16),      # ragged in both x and y
    (96, 96, 48, 48, 16),       # 2x2 tiles, halo reaches most of the domain
    (64, 64, 32, 32, 16),       # compute window == domain width
    (48, 48, 16, 16, 16),       # compute window == domain width, 3x3 tiles
    (40, 40, 16, 16, 16),       # compute window WIDER than domain (48 > 40)
    (37, 29, 7, 5, 3),          # odd, ragged, small halo
    (16, 16, 16, 16, 4),        # single tile
    (9, 9, 4, 4, 0),            # zero halo
    (5, 7, 2, 3, 6),            # tiny domain, halo several times the domain
    (128, 64, 128, 8, 16),      # one tile wide, many tall
]

NONPERIODIC_CASES = [
    (128, 128, 32, 32, 16),
    (100, 80, 32, 24, 16),      # ragged
    (64, 64, 16, 16, 16),       # compute window 48 of 64
    (128, 96, 64, 32, 16),      # compute window exactly fills y (64 of 96)
    (64, 64, 32, 32, 16),       # compute window exactly == domain
    (20, 20, 8, 8, 2),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_full(nx, ny, variant, periodic, seed=0):
    """A ``(NZ, ny+ey, nx+ex)`` array of unique values.

    Under ``periodic`` the alias slot carries the duplicate the model
    maintains (``u[..., nx] == u[..., 0]``), so it is NOT unique there -- that
    is the real convention and the round trip has to reproduce it.
    """
    ey, ex = stagger(variant)
    shape = (NZ, ny + ey, nx + ex)
    base = 1000.0 * (seed + 1)
    arr = base + np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    if periodic:
        if ex:
            arr[..., :, nx] = arr[..., :, 0]
        if ey:
            arr[..., ny, :] = arr[..., 0, :]
    return arr


def independent(arr, nx, ny):
    """The non-alias part of a full array: the first ``ny``/``nx`` points."""
    return arr[..., :ny, :nx]


def expected_tile(full, spec, variant):
    """Reference tile contents, built by direct numpy indexing.

    Deliberately does NOT reuse any spec slice: it recomputes each tile point's
    source from ``(cj0 + p) % ny``, ``(ci0 + q) % nx`` for periodic domains and
    from a plain window for non-periodic ones.
    """
    ey, ex = stagger(variant)
    ny_pts = spec.cny + ey
    nx_pts = spec.cnx + ex
    if spec.periodic:
        indep = independent(full, spec.nx, spec.ny)
        rows = (spec.cj0 + np.arange(ny_pts)) % spec.ny
        cols = (spec.ci0 + np.arange(nx_pts)) % spec.nx
        return indep[..., rows, :][..., :, cols]
    return full[..., spec.cj0:spec.cj0 + ny_pts, spec.ci0:spec.ci0 + nx_pts]


def gather_tile(full, spec, variant, sentinel=np.nan):
    """Gather into a fresh sentinel-filled tile array and check it is full."""
    ny_pts, nx_pts = spec.shape(variant)
    tile = np.full((NZ, ny_pts, nx_pts), sentinel, dtype=full.dtype)
    written = np.zeros((ny_pts, nx_pts), dtype=np.int64)
    for transfer in spec.gather(variant):
        transfer.copy_to_tile(full, tile)
        written[transfer.tile_y, transfer.tile_x] += 1
    assert np.array_equal(written, np.ones_like(written)), (
        f"gather wrote some tile points {sorted(np.unique(written).tolist())} "
        f"times for {variant} on {spec.describe()}")
    assert not np.isnan(tile).any()
    return tile


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_shapes_are_staggered():
    assert full_shape(8, 10, "mass") == (8, 10)
    assert full_shape(8, 10, "u") == (8, 11)
    assert full_shape(8, 10, "v") == (9, 10)
    assert full_shape(8, 10, "w") == (8, 10)
    assert full_shape(8, 10, "phi") == (8, 10)
    assert tile_shape(4, 6, 2, "mass") == (8, 10)
    assert tile_shape(4, 6, 2, "u") == (8, 11)
    assert tile_shape(4, 6, 2, "v") == (9, 10)


def test_interiors_partition_the_mass_grid():
    for nx, ny, tnx, tny, halo in PERIODIC_CASES + NONPERIODIC_CASES:
        for periodic in (True, False):
            if not periodic and (tnx + 2 * halo > nx or tny + 2 * halo > ny):
                continue
            specs = plan_tiles(nx, ny, tnx, tny, halo, periodic)
            seen = np.zeros((ny, nx), dtype=np.int64)
            for spec in specs:
                assert 0 <= spec.i0 < spec.i1 <= nx
                assert 0 <= spec.j0 < spec.j1 <= ny
                seen[spec.j0:spec.j1, spec.i0:spec.i1] += 1
            assert np.array_equal(seen, np.ones_like(seen)), (
                f"mass interiors not a partition for "
                f"{(nx, ny, tnx, tny, halo, periodic)}")


def test_compute_shapes_are_uniform():
    """One CUDA graph per plan requires identical tile shapes."""
    for nx, ny, tnx, tny, halo in PERIODIC_CASES:
        specs = plan_tiles(nx, ny, tnx, tny, halo)
        for variant in VARIANTS:
            shapes = {spec.shape(variant) for spec in specs}
            assert shapes == {tile_shape(tny, tnx, halo, variant)}, (
                f"non-uniform {variant} shapes {shapes} for "
                f"{(nx, ny, tnx, tny, halo)}")


def test_halo_margins_are_at_least_halo_when_periodic():
    for nx, ny, tnx, tny, halo in PERIODIC_CASES:
        for spec in plan_tiles(nx, ny, tnx, tny, halo):
            assert spec.halo_left == halo
            assert spec.halo_south == halo
            # trailing (ragged) tiles get a WIDER right/north halo, never a
            # narrower one.
            assert spec.halo_right >= halo
            assert spec.halo_north >= halo


def test_nonperiodic_window_stays_inside_the_domain():
    for nx, ny, tnx, tny, halo in NONPERIODIC_CASES:
        for spec in plan_tiles(nx, ny, tnx, tny, halo, periodic=False):
            assert 0 <= spec.ci0 and spec.ci0 + spec.cnx <= nx
            assert 0 <= spec.cj0 and spec.cj0 + spec.cny <= ny
            # the interior is always inside its own compute window
            assert spec.ci0 <= spec.i0 and spec.i1 <= spec.ci0 + spec.cnx
            assert spec.cj0 <= spec.j0 and spec.j1 <= spec.cj0 + spec.cny


def test_ragged_policy_is_ragged_interior_not_ragged_shape():
    specs = plan_tiles(100, 80, 32, 24, 16)
    assert len({(s.cny, s.cnx) for s in specs}) == 1
    last_x = [s for s in specs if s.tx == max(t.tx for t in specs)]
    assert all(s.ragged_x for s in last_x)
    assert all(s.interior_nx == 100 - 96 for s in last_x)
    assert all(not s.ragged_x for s in specs if s.tx == 0)


# ---------------------------------------------------------------------------
# gate 1: coverage
# ---------------------------------------------------------------------------

def test_coverage_exactly_once():
    """Every point of every variant's full array is scattered exactly once."""
    checked = 0
    for periodic, cases in ((True, PERIODIC_CASES),
                            (False, NONPERIODIC_CASES)):
        for nx, ny, tnx, tny, halo in cases:
            specs = plan_tiles(nx, ny, tnx, tny, halo, periodic)
            for variant in VARIANTS:
                counts = coverage_counts(specs, ny, nx, variant)
                assert counts.shape == full_shape(ny, nx, variant)
                assert np.array_equal(counts, np.ones_like(counts)), (
                    f"{variant} coverage {sorted(np.unique(counts).tolist())} "
                    f"for {(nx, ny, tnx, tny, halo, periodic)}")
                checked += 1
            validate_plan(specs, ny, nx)
    assert checked == 4 * (len(PERIODIC_CASES) + len(NONPERIODIC_CASES))


def test_shared_u_faces_have_one_owner():
    """Neighbouring tiles must not both write the face between them."""
    specs = plan_tiles(128, 64, 32, 32, 16)
    row = sorted((s for s in specs if s.ty == 0), key=lambda s: s.tx)
    owned = []
    for spec in row:
        xs = set()
        for transfer in spec.scatter("u"):
            xs.update(range(transfer.full_x.start, transfer.full_x.stop))
        owned.append(xs)
    for a, b in zip(owned, owned[1:]):
        assert not (a & b), "adjacent tiles double-write a u face"
    # tile tx owns exactly its own left faces
    assert owned[0] == set(range(0, 32)) | {128}   # + the periodic alias slot
    assert owned[1] == set(range(32, 64))
    assert set().union(*owned) == set(range(129))


def test_alias_slot_is_written_by_exactly_one_tile():
    specs = plan_tiles(96, 96, 32, 32, 16)
    writers_x = [s for s in specs if s.owns_x_alias]
    assert {(s.ty, s.tx) for s in writers_x} == {(ty, 0) for ty in range(3)}
    writers_y = [s for s in specs if s.owns_y_alias]
    assert {(s.ty, s.tx) for s in writers_y} == {(0, tx) for tx in range(3)}

    nonp = plan_tiles(96, 96, 32, 32, 16, periodic=False)
    assert {(s.ty, s.tx) for s in nonp if s.owns_x_alias} == \
        {(ty, 2) for ty in range(3)}
    assert {(s.ty, s.tx) for s in nonp if s.owns_y_alias} == \
        {(2, tx) for tx in range(3)}


def test_scatter_never_reads_the_halo_ring():
    """Owned tile points sit at least `halo` from the gathered array's edge."""
    for nx, ny, tnx, tny, halo in PERIODIC_CASES:
        for spec in plan_tiles(nx, ny, tnx, tny, halo):
            for variant in VARIANTS:
                ey, ex = stagger(variant)
                ny_pts, nx_pts = spec.shape(variant)
                for transfer in spec.scatter(variant):
                    assert transfer.tile_x.start >= halo
                    assert transfer.tile_y.start >= halo
                    assert transfer.tile_x.stop <= nx_pts - halo
                    assert transfer.tile_y.stop <= ny_pts - halo


# ---------------------------------------------------------------------------
# gate 2: round trip
# ---------------------------------------------------------------------------

def _round_trip(nx, ny, tnx, tny, halo, periodic):
    specs = plan_tiles(nx, ny, tnx, tny, halo, periodic)
    for variant in VARIANTS:
        full = make_full(nx, ny, variant, periodic)
        out = np.zeros_like(full)
        for spec in specs:
            tile = gather_tile(full, spec, variant)
            spec.apply_scatter(tile, out, variant)
        assert np.array_equal(out, full), (
            f"round trip lost {variant} for "
            f"{(nx, ny, tnx, tny, halo, periodic)}: "
            f"{int((out != full).sum())} of {full.size} points differ")


def test_round_trip():
    for nx, ny, tnx, tny, halo in PERIODIC_CASES:
        _round_trip(nx, ny, tnx, tny, halo, True)
    for nx, ny, tnx, tny, halo in NONPERIODIC_CASES:
        _round_trip(nx, ny, tnx, tny, halo, False)


def test_round_trip_is_not_vacuous():
    """A deliberately wrong plan must fail the round trip we just passed."""
    nx = ny = 64
    specs = plan_tiles(nx, ny, 32, 32, 16)
    full = make_full(nx, ny, "u", True)
    out = np.zeros_like(full)
    for spec in specs:
        tile = gather_tile(full, spec, "u")
        # WRONG: slice u exactly like mass, dropping the staggered face.
        for transfer in spec.scatter("mass"):
            transfer.copy_to_full(tile, out)
    assert not np.array_equal(out, full)


def test_round_trip_on_2d_arrays():
    """Transfers use Ellipsis, so a bare (ny, nx) array must work too."""
    nx, ny = 40, 40
    specs = plan_tiles(nx, ny, 16, 16, 16)
    for variant in VARIANTS:
        ey, ex = stagger(variant)
        full = np.arange((ny + ey) * (nx + ex), dtype=np.float64
                         ).reshape(ny + ey, nx + ex)
        if ex:
            full[:, nx] = full[:, 0]
        if ey:
            full[ny, :] = full[0, :]
        out = np.zeros_like(full)
        for spec in specs:
            tile = np.zeros(spec.shape(variant), dtype=full.dtype)
            spec.apply_gather(full, tile, variant)
            spec.apply_scatter(tile, out, variant)
        assert np.array_equal(out, full)


def test_periodic_duplicate_survives_the_round_trip():
    nx, ny = 100, 80
    specs = plan_tiles(nx, ny, 32, 24, 16)
    for variant, ey, ex in (("u", 0, 1), ("v", 1, 0)):
        full = make_full(nx, ny, variant, True)
        out = np.zeros_like(full)
        for spec in specs:
            spec.apply_scatter(gather_tile(full, spec, variant), out, variant)
        if ex:
            assert np.array_equal(out[..., :, nx], out[..., :, 0])
        if ey:
            assert np.array_equal(out[..., ny, :], out[..., 0, :])


# ---------------------------------------------------------------------------
# gate 3: the halo really contains the neighbours
# ---------------------------------------------------------------------------

def test_halo_matches_direct_indexing():
    for periodic, cases in ((True, PERIODIC_CASES),
                            (False, NONPERIODIC_CASES)):
        for nx, ny, tnx, tny, halo in cases:
            specs = plan_tiles(nx, ny, tnx, tny, halo, periodic)
            for variant in VARIANTS:
                full = make_full(nx, ny, variant, periodic)
                for spec in specs:
                    got = gather_tile(full, spec, variant)
                    want = expected_tile(full, spec, variant)
                    assert got.shape == want.shape
                    assert np.array_equal(got, want), (
                        f"{variant} halo wrong on {spec.describe()} "
                        f"({(nx, ny, tnx, tny, halo, periodic)})")


def test_interior_tile_halo_is_the_literal_neighbourhood():
    """Spelled out by hand for one interior tile, no modular arithmetic."""
    nx = ny = 128
    halo = 16
    specs = plan_tiles(nx, ny, 32, 32, halo)
    spec = next(s for s in specs if (s.ty, s.tx) == (1, 1))
    assert (spec.i0, spec.i1, spec.j0, spec.j1) == (32, 64, 32, 64)
    assert (spec.ci0, spec.cj0) == (16, 16)

    full = make_full(nx, ny, "mass", True)
    tile = gather_tile(full, spec, "mass")
    assert len(spec.gather("mass")) == 1          # no wrap for an interior tile
    assert np.array_equal(tile, full[..., 16:80, 16:80])
    assert np.array_equal(tile[..., halo:halo + 32, halo:halo + 32],
                          full[..., 32:64, 32:64])

    ufull = make_full(nx, ny, "u", True)
    utile = gather_tile(ufull, spec, "u")
    assert utile.shape == (NZ, 64, 65)
    assert np.array_equal(utile, ufull[..., 16:80, 16:81])
    # the owned faces are the LEFT faces of the owned mass cells
    assert np.array_equal(utile[..., halo:halo + 32, halo:halo + 32],
                          ufull[..., 32:64, 32:64])

    vfull = make_full(nx, ny, "v", True)
    vtile = gather_tile(vfull, spec, "v")
    assert vtile.shape == (NZ, 65, 64)
    assert np.array_equal(vtile, vfull[..., 16:81, 16:80])


def test_corner_tile_halo_wraps_to_the_far_side():
    nx = ny = 128
    halo = 16
    spec = plan_tiles(nx, ny, 32, 32, halo)[0]
    assert (spec.ci0, spec.cj0) == (-16, -16)
    full = make_full(nx, ny, "mass", True)
    tile = gather_tile(full, spec, "mass")
    assert len(spec.gather("mass")) == 4          # wraps in both x and y
    # the tile's south-west corner is the domain's north-east corner
    assert np.array_equal(tile[..., :halo, :halo], full[..., 112:, 112:])
    assert np.array_equal(tile[..., :halo, halo:], full[..., 112:, :48])
    assert np.array_equal(tile[..., halo:, :halo], full[..., :48, 112:])
    assert np.array_equal(tile[..., halo:, halo:], full[..., :48, :48])


def test_gather_never_touches_the_alias_slot_when_periodic():
    for nx, ny, tnx, tny, halo in PERIODIC_CASES:
        for spec in plan_tiles(nx, ny, tnx, tny, halo):
            for transfer in spec.gather("u"):
                assert transfer.full_x.stop <= nx
            for transfer in spec.gather("v"):
                assert transfer.full_y.stop <= ny


def test_gather_uses_the_boundary_face_when_not_periodic():
    spec = plan_tiles(64, 64, 32, 32, 16, periodic=False)[-1]
    assert spec.ci0 + spec.cnx == 64
    xs = [t.full_x.stop for t in spec.gather("u")]
    assert max(xs) == 65, "the real boundary face must be gathered"


# ---------------------------------------------------------------------------
# wider than the domain
# ---------------------------------------------------------------------------

def test_tile_wider_than_domain_wraps_onto_itself():
    nx = ny = 40
    specs = plan_tiles(nx, ny, 16, 16, 16)
    assert all(s.wraps_onto_itself for s in specs)   # 16 + 32 = 48 > 40
    full = make_full(nx, ny, "mass", True)
    for spec in specs:
        tile = gather_tile(full, spec, "mass")
        # the tile holds the periodic extension, so column q and column q+nx
        # (both inside the window) hold the same data
        assert np.array_equal(tile[..., :, :8], tile[..., :, 40:48])
    validate_plan(specs, ny, nx)


def test_single_tile_covering_everything():
    nx, ny = 24, 24
    specs = plan_tiles(nx, ny, 64, 64, 16)
    assert len(specs) == 1
    spec = specs[0]
    assert (spec.i0, spec.i1, spec.j0, spec.j1) == (0, 24, 0, 24)
    assert spec.ragged_x and spec.ragged_y
    validate_plan(specs, ny, nx)
    _round_trip(nx, ny, 64, 64, 16, True)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

def test_rejects_nonperiodic_window_larger_than_domain():
    try:
        plan_tiles(63, 63, 32, 32, 16, periodic=False)
    except ValueError as exc:
        assert "fit inside" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("expected ValueError")


def test_rejects_bad_arguments():
    for args in ((0, 10, 4, 4), (10, 0, 4, 4), (10, 10, 0, 4), (10, 10, 4, 0)):
        try:
            plan_tiles(*args, 1)
        except ValueError:
            pass
        else:                                        # pragma: no cover
            raise AssertionError(f"expected ValueError for {args}")
    try:
        plan_tiles(10, 10, 4, 4, -1)
    except ValueError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("expected ValueError for negative halo")


def test_rejects_unknown_variant():
    spec = plan_tiles(32, 32, 16, 16, 4)[0]
    for bad in ("U", "theta", "mass_point", ""):
        try:
            spec.gather(bad)
        except ValueError as exc:
            assert "unknown variant" in str(exc)
        else:                                        # pragma: no cover
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_segment_rejects_mismatched_lengths():
    for full_s, tile_s in ((slice(0, 4), slice(0, 5)),
                           (slice(0, 0), slice(0, 0)),
                           (slice(0, 4, 2), slice(0, 4))):
        try:
            Segment(full_s, tile_s)
        except ValueError:
            pass
        else:                                        # pragma: no cover
            raise AssertionError(f"expected ValueError for {full_s}/{tile_s}")


def test_validate_plan_catches_a_broken_plan():
    specs = plan_tiles(64, 64, 32, 32, 16)
    broken = list(specs[:-1])                        # drop a tile
    try:
        validate_plan(broken, 64, 64)
    except ValueError as exc:
        assert "coverage is not exactly 1" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def test_interior_in_tile_agrees_with_scatter():
    for periodic, cases in ((True, PERIODIC_CASES),
                            (False, NONPERIODIC_CASES)):
        for nx, ny, tnx, tny, halo in cases:
            for spec in plan_tiles(nx, ny, tnx, tny, halo, periodic):
                for variant in VARIANTS:
                    ys, xs = spec.interior_in_tile(variant)
                    covered = set()
                    for t in spec.scatter(variant):
                        covered.update(
                            (y, x)
                            for y in range(t.tile_y.start, t.tile_y.stop)
                            for x in range(t.tile_x.start, t.tile_x.stop))
                    named = {(y, x)
                             for y in range(ys.start, ys.stop)
                             for x in range(xs.start, xs.stop)}
                    assert named <= covered, (
                        f"interior_in_tile claims points scatter never "
                        f"writes: {variant} {spec.describe()}")


def test_transfer_counts_match_the_tile_area():
    for nx, ny, tnx, tny, halo in PERIODIC_CASES:
        for spec in plan_tiles(nx, ny, tnx, tny, halo):
            for variant in VARIANTS:
                ny_pts, nx_pts = spec.shape(variant)
                total = sum(t.count for t in spec.gather(variant))
                assert total == ny_pts * nx_pts


def test_plan_efficiency_counts_the_halo_waste():
    # 4 tiles of 32 interior + 16 halo each side -> 64x64 compute per tile.
    eff = plan_efficiency(plan_tiles(64, 64, 32, 32, 16))
    assert eff["tiles"] == 4
    assert eff["domain_cells"] == 64 * 64
    assert eff["compute_cells"] == 4 * 64 * 64
    assert eff["redundancy"] == 4.0
    assert eff["scattered_cells"] == 64 * 64      # interiors partition exactly
    # scattered cells always equal the domain, for every plan
    for nx, ny, tnx, tny, halo in PERIODIC_CASES:
        e = plan_efficiency(plan_tiles(nx, ny, tnx, tny, halo))
        assert e["scattered_cells"] == nx * ny
        assert e["redundancy"] >= 1.0


def test_default_halo_is_the_documented_16():
    assert DEFAULT_HALO == 16
    assert plan_tiles(64, 64, 16, 16)[0].halo == 16


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
