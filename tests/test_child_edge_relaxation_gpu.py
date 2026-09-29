"""The GPU halves of the downscaled-child lateral zone.

* A specified domain with ``relax_w`` relaxes ``w`` in its zone toward the
  table and forces the table's ``w`` back on the specified rows, as a nest
  does -- where WRF's root-domain rule copies the first interior row onto
  the boundary and never relaxes ``w`` at all, which is how an updraft that
  forms in the zone reached the child's edge.
* A streamed tile's seams relax nothing (``LateralBoundaries.seam_sides``):
  a zone sized in parent cells is wider than the tile halo, and a seam
  relaxing toward its zero placeholder would reach owned cells.
* So a tile whose interior, widened by its halo, reaches a zone must own
  that edge, and a zone cell further out, which a clamped window still
  holds, changes nothing: tilings on both sides of that line are run
  streamed against the resident domain.

CPU halves: tests/test_child_lateral_zone.py.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu


def _setup(*, relax_w: bool, with_w_table: bool = True, nx=40, ny=36,
           relax_zone=8):
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=nx, ny=ny, nz=6, dx=250.0, dy=250.0, ztop=12000.0,
                    dt=1.25, run_seconds=60.0, moist=False, specified=True,
                    spec_zone=1, relax_zone=relax_zone,
                    spec_bdy_width=relax_zone + 1, relax_timescale_s=150.0,
                    relax_w=relax_w)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    forcing = [init_at_rest(cfg, coord, base) for _ in range(2)]
    rng = np.random.default_rng(2609)
    for name in ("u", "v", "w", "thp", "php"):
        values = rng.normal(0.0, 1.0, getattr(state, name).shape)
        getattr(state, name)[...] = cp.asarray(values.astype(np.float32))
    state.w[0] = 0.0
    snapshots = []
    for n, f in enumerate(forcing):
        f.u += cp.float32(2.0 + n)
        f.thp += cp.float32(3.0 + n)
        f.w[1:] = cp.float32(0.5 + n)
        snapshot = dict(lbc.domain_boundary_snapshot(f))
        if with_w_table:
            mu = f.total_mu()
            chf = f.c1f[:, None, None] * mu[None] + f.c2f[:, None, None]
            snapshot["w"] = lbc._host(chf * f.w)
        snapshots.append(snapshot)
    boundaries = lbc.build_lateral_boundaries(
        snapshots, [0.0, 60.0], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    lbc.attach_lateral_boundaries(state, boundaries)
    return cfg, state, boundaries


def _distances(ny, nx):
    jj, ii = np.mgrid[0:ny, 0:nx]
    return jj, ny - 1 - jj, ii, nx - 1 - ii


@requires_gpu
@pytest.mark.gpu
def test_relax_w_relaxes_w_in_the_zone_and_specifies_the_rows():
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(relax_w=True)
    for name in ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t"):
        getattr(state, name)[...] = 0
    lbc.apply_state_lateral_boundaries(state, cfg, rk_stage=0)
    rw = cp.asnumpy(state.rw_t)
    ds, dn, dw, de = _distances(cfg.ny, cfg.nx)
    edge = np.minimum.reduce([ds, dn, dw, de])
    zone = (edge >= cfg.spec_zone) & (edge < cfg.relax_zone)
    # every relaxed column moved w; nothing beyond the zone did
    assert np.all(np.abs(rw[1:-1][:, zone]).max(axis=0) > 0.0)
    assert np.all(rw[:, edge >= cfg.relax_zone] == 0.0)

    w_before = cp.asnumpy(state.w).copy()
    lbc.apply_state_boundary_values(state, cfg, elapsed_seconds=0.0)
    w_after = cp.asnumpy(state.w)
    spec = edge < cfg.spec_zone
    # the specified rows now hold the table's w (0.5 m/s aloft), not a copy
    # of the first interior row
    np.testing.assert_allclose(w_after[2:-1][:, spec], 0.5, rtol=1e-4)
    # elsewhere w only takes the couple/uncouple round trip every forced
    # field takes (spec_bdy_final), which is rounding, not forcing
    np.testing.assert_allclose(w_after[:, ~spec], w_before[:, ~spec],
                               rtol=1e-6, atol=1e-7)
    # and the root-domain copy stands down
    lbc.apply_specified_w_zero_gradient(state, cfg)
    np.testing.assert_array_equal(cp.asnumpy(state.w), w_after)


@requires_gpu
@pytest.mark.gpu
def test_without_relax_w_the_specified_domain_keeps_wrfs_rule():
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(relax_w=False)
    for name in ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t"):
        getattr(state, name)[...] = 0
    lbc.apply_state_lateral_boundaries(state, cfg, rk_stage=0)
    assert float(cp.abs(state.rw_t).max()) == 0.0
    w_before = cp.asnumpy(state.w).copy()
    lbc.apply_state_boundary_values(state, cfg, elapsed_seconds=0.0)
    np.testing.assert_array_equal(cp.asnumpy(state.w), w_before)


@requires_gpu
@pytest.mark.gpu
def test_relax_w_without_a_w_table_is_refused_in_plain_words():
    import gpuwm.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(relax_w=True, with_w_table=False)
    with pytest.raises(RuntimeError, match="needs a w boundary table"):
        lbc.apply_state_lateral_boundaries(state, cfg, rk_stage=0)


@requires_gpu
@pytest.mark.gpu
def test_a_seam_side_relaxes_nothing_and_leaves_the_true_edges_alone():
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc

    cfg, state, boundaries = _setup(relax_w=False, relax_zone=8)
    device, dtbc, dt, spec_exp = lbc._active_device_interval(state, cfg)

    def held(bnd):
        state.lateral_boundaries = bnd
        out = cp.zeros_like(state.thp)
        lbc.apply_specified_relaxation(
            state.thp, out, device.fields["theta"], dtbc=dtbc, dt=dt,
            spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone,
            spec_exp=spec_exp, apply_relax=True, state=state,
            field_name="theta", clear_specified=True, timescale_s=150.0)
        return cp.asnumpy(out)

    whole = held(boundaries)
    east_seam = held(dataclasses.replace(boundaries, seam_sides=("east",)))
    ds, dn, dw, de = _distances(cfg.ny, cfg.nx)
    r = cfg.relax_zone
    # the east band proper (away from the south/north zones) relaxes nothing
    east_band = (de < r) & (ds >= r) & (dn >= r)
    assert np.abs(whole[:, east_band]).max() > 0.0
    assert np.all(east_seam[:, east_band] == 0.0)
    # everything the east edge never owned is bit-identical
    far = de >= r
    np.testing.assert_array_equal(east_seam[:, far], whole[:, far])
    # the corner cells the east edge used to own (nearer east than south)
    # go to the south edge, the only true edge near them in the domain
    corner = (de >= cfg.spec_zone) & (de < ds) & (ds >= cfg.spec_zone) & (
        ds < r)
    assert corner.any()
    assert np.abs(east_seam[:, corner]).min() > 0.0
    # a whole domain is the default: all four sides relax
    np.testing.assert_array_equal(held(boundaries), whole)


@requires_gpu
@pytest.mark.gpu
def test_single_precision_host_tables_hand_the_card_the_same_bits():
    """The offline child holds its boundary tables at FP32 on the host
    (a zone sized in parent cells is 41 rows at ratio 20, for every parent
    frame).  The device mirror reads FP32 either way, so the rounding moved
    from the upload to the build and the card sees identical bits."""
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc
    from gpuwm.offline_child import _single_precision_interval

    cfg, state, boundaries = _setup(relax_w=True)
    wide = boundaries.intervals[0]
    rng = np.random.default_rng(4)
    for boundary in wide.fields.values():
        for side in ("west", "east", "south", "north"):
            got = getattr(boundary, side)
            assert got.value.dtype == np.float64
            # values off the FP32 grid, so the rounding really happens
            object.__setattr__(got, "value", got.value + rng.uniform(
                -1e-3, 1e-3, got.value.shape))
    narrow = _single_precision_interval(wide)
    for boundary in narrow.fields.values():
        assert boundary.west.value.dtype == np.float32
    lbc.attach_streaming_lateral_boundaries(
        state, dataclasses.replace(boundaries, intervals=(wide,)))
    from_wide = cp.asnumpy(state._lateral_boundary_device.packed_forcing)
    lbc.attach_streaming_lateral_boundaries(
        state, dataclasses.replace(boundaries, intervals=(narrow,)))
    from_narrow = cp.asnumpy(state._lateral_boundary_device.packed_forcing)
    np.testing.assert_array_equal(from_wide, from_narrow)


#: Appended to lbc_state.cu's own source, so the probe runs the very
#: frame_point and frame_offset the relaxation kernel is compiled with.
_FRAME_PROBE = r"""
extern "C" __global__
void frame_point_probe(int* hits, int* inverse, int ny, int nx, int width,
                       int rings, int frame_count)
{
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= frame_count) return;
    int j, i, d;
    if (!frame_point(p, ny, nx, width, &j, &i, &d)) return;
    atomicAdd(&hits[j*nx + i], 1);
    inverse[p] = frame_offset(j, i, ny, nx, rings);
}
"""


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("ny,nx,width", [
    (77, 120, 40), (120, 77, 40), (41, 41, 24), (43, 90, 24), (76, 120, 40),
    (60, 64, 8)])
def test_frame_point_lists_each_cell_of_a_narrow_window_once(ny, nx, width):
    """state_specified_relaxation hands one thread to each frame entry, and
    each thread reads, modifies and writes its cell.  A 77 x 120 tile
    window under a 40-cell zone listed its middle row twice (9284 entries
    for 9240 cells), so two threads raced on each of its 44 cells and the
    step could land twice.  Every cell is listed once now, and frame_offset
    still inverts frame_point."""
    import cupy as cp

    from gpuwm.core.kernels import module_source
    from gpuwm.ingest.lateral_bc import _frame_rings, _perimeter_count

    module = cp.RawModule(code=module_source("lbc_state") + _FRAME_PROBE,
                          options=("-std=c++17",))
    probe = module.get_function("frame_point_probe")
    rings = _frame_rings(ny, nx, width)
    count = _perimeter_count(ny, nx, rings)
    hits = cp.zeros((ny, nx), dtype=cp.int32)
    inverse = cp.full(count, -1, dtype=cp.int32)
    # the relaxation kernel passes the zone width, not the capped ring count
    probe(((count + 127) // 128,), (128,), (
        hits, inverse, np.int32(ny), np.int32(nx), np.int32(width),
        np.int32(rings), np.int32(count)))
    hits = cp.asnumpy(hits)
    expected = np.zeros((ny, nx), dtype=bool)
    for d in range(rings):
        expected[d, d:nx - d] = True
        expected[ny - 1 - d, d:nx - d] = True
        expected[d + 1:ny - d - 1, d] = True
        expected[d + 1:ny - d - 1, nx - 1 - d] = True
    assert hits.max() == 1, np.argwhere(hits > 1)[:4]
    np.testing.assert_array_equal(hits.astype(bool), expected)
    np.testing.assert_array_equal(cp.asnumpy(inverse), np.arange(count))


#: A ratio-20 child's zone under an 18-cell halo.
_ZONE, _HALO = 40, 18
_ZONE_NZ, _ZONE_STEPS = 20, 8


def _zone_domain(n: int):
    """An ``n`` x ``n`` specified domain with a ratio-20 child's zone: 40
    linearly ramped rows, and ``w`` relaxed toward, and specified from,
    its own table.  The table is made from two seeds other than the
    state's own, so from the first step the zone pulls every field it
    holds toward a different state."""
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc
    from tilestream import test_join as join

    cfg = join.join_cfg(n, n, nz=_ZONE_NZ, rung="dry", spec_zone=1,
                        relax_zone=_ZONE, spec_bdy_width=_ZONE + 1,
                        relax_w=True)
    snapshots = []
    for seed in (join.SEED + 1, join.SEED + 2):
        state, _ = join.build_domain(cfg, seed=seed, warmup=0)
        snapshot = dict(lbc.domain_boundary_snapshot(state))
        mu = state.total_mu()
        chf = state.c1f[:, None, None] * mu[None] + state.c2f[:, None, None]
        snapshot["w"] = lbc._host(chf * state.w)
        snapshots.append(snapshot)
        del state
    cp.get_default_memory_pool().free_all_blocks()
    boundaries = lbc.build_lateral_boundaries(
        snapshots, [0.0, join.BDY_SECONDS],
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)
    return cfg, boundaries


def _zone_run(cfg, boundaries, tile=None):
    """The domain stepped resident (``tile`` None) or streamed at a pinned
    square tiling through the offline child's own builder, the production
    boundary clock bound either way; the carriers after the run."""
    import warnings

    import cupy as cp

    from gpuwm.core import streaming
    from gpuwm.core.dycore import step as dycore_step
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
    from gpuwm.offline_child_run import _child_boundary_clock
    from tilestream import physics_inventory as physinv
    from tilestream import test_join as join

    state, _ = join.build_domain(cfg, seed=join.SEED, boundaries=boundaries,
                                 warmup=0)
    clock = _child_boundary_clock(
        cfg, lbc_interval_seconds=float(join.BDY_SECONDS),
        steps=_ZONE_STEPS, output_steps=_ZONE_STEPS)
    bind_lateral_boundary_clock(state, clock)
    if tile is None:
        stepper = dycore_step
    else:
        options = streaming.StreamingOptions(
            mode="on", tile_nx=int(tile), tile_ny=int(tile), nbuffers=2,
            halo=_HALO, store="host")
        with warnings.catch_warnings():
            # the arms read the store, never the stale resident state
            warnings.simplefilter("ignore", RuntimeWarning)
            stepper = streaming.make_stepper(
                state, cfg, options,
                build=streaming.standalone_domain_builder(
                    grid_id=int(cfg.grid_id)))
        assert streaming.is_streaming(stepper)
    for _ in range(_ZONE_STEPS):
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        stepper(state, cfg, refl_10cm_due=False)
        clock.advance()
    cp.cuda.runtime.deviceSynchronize()
    if tile is None:
        out = {name: join._as_numpy(arr) for name, arr in
               physinv.carrier_inventory(state, None).items()}
        bad = sum(int(np.count_nonzero(~np.isfinite(v)))
                  for v in out.values() if v.dtype.kind == "f")
        assert bad == 0, f"the resident run has {bad} non-finite cells"
        return out
    return dict(stepper.store)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("n,tile", [(200, 125), (200, 142), (160, 71)])
def test_a_tiling_the_rule_admits_matches_the_resident_run(n, tile):
    """200 cells in 125- or 142-cell tiles: each seam lies at least
    zone + halo = 58 cells from both edges, but the second window, clamped
    against its own edge, starts at column 39 or 22, inside the other
    edge's zone, where it runs those columns unrelaxed.  They lie past
    the halo of the tile's interior, reach none of its cells within a
    step, and the run is bit-identical to the resident one.  160 cells in
    71-cell tiles: the last seam is 18 cells from the east edge, and the
    tile before it has its window clamped out to that edge, so it owns
    the edge and relaxes the zone itself."""
    from tilestream import spec as tspec
    from tilestream import test_join as join

    assert not tspec.edge_band_unowned(n, tile, _HALO, _ZONE)
    cfg, boundaries = _zone_domain(n)
    resident = _zone_run(cfg, boundaries)
    tiled = _zone_run(cfg, boundaries, tile)
    got = join.compare(resident, tiled)
    assert got["nonfinite"] == 0
    assert got["bitexact"], (
        f"{n} cells in {tile}-cell tiles differ from the resident run on "
        f"{got['ndiff']} of {got['ntotal']} carriers "
        f"(max |d| {got['max_abs']:.6g}, first {got['differing']})")


@requires_gpu
@pytest.mark.gpu
def test_a_zone_cell_within_the_halo_is_what_the_rule_refuses(monkeypatch):
    """The control that shows the comparison above can see the defect.
    200 cells in 160-cell tiles: the first tile's halo reads columns 160
    to 177, the innermost 18 columns of the east zone, from a window that
    stops at 196 and so does not own that edge.  The rule refuses the
    tiling; with the refusal taken out, those columns run unrelaxed, feed
    the tile's own cells, and the run differs from the resident one."""
    from gpuwm.core import streaming
    from tilestream import spec as tspec
    from tilestream import test_join as join

    assert tspec.edge_band_unowned(200, 160, _HALO, _ZONE)
    cfg, boundaries = _zone_domain(200)
    with pytest.raises(streaming.StreamingRefused, match="east"):
        _zone_run(cfg, boundaries, 160)

    def unguarded(bnd, spec):
        return tuple(side for side, owned in
                     streaming.owned_edges(spec).items() if not owned)

    monkeypatch.setattr(streaming, "tile_seam_sides", unguarded)
    resident = _zone_run(cfg, boundaries)
    tiled = _zone_run(cfg, boundaries, 160)
    got = join.compare(resident, tiled)
    assert got["nonfinite"] == 0
    assert not got["bitexact"], (
        "a tile running east-zone cells unrelaxed inside its halo matched "
        "the resident run, so this comparison cannot see the defect")
