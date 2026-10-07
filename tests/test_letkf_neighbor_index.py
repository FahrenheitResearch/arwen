"""The once-per-analysis neighbour index reproduces the forward walk exactly.

The host-staged LETKF route used to find each gridpoint's localised
observations by walking gridpoints x stencil slots per chunk on the host.
``neighbor_search="index"`` walks observations x stencil slots once
(gpuwm.da.neighbor_index).  The breakage these tests prevent: an index that
drops, adds, reorders or reweights a neighbour, which would change the
packed transform inputs and so the analysis.  The comparison is bitwise.
"""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.da import letkf
from gpuwm.da.letkf import GridGeometry, LetkfDiagnostics

from test_letkf_bounded_staging import problem
from test_letkf_sparse_neighbors import full_depth_problem


def geodesic_problem(*, batches=5, members=10):
    """Terrain, lat/lon (haversine metric) and windowed batches together."""
    prior, obs, grid, cfg = full_depth_problem(fields=3, batches=batches,
                                               members=members, chunk=17)
    ny, nx = grid.heights_m.shape[1:]
    lat = 31.0 + np.arange(ny)[:, None] * 0.027 + np.arange(nx)[None, :] * 1e-3
    lon = -89.0 + np.arange(nx)[None, :] * 0.0315 - np.arange(ny)[:, None] * 1e-3
    grid = GridGeometry(dx_m=3000., dy_m=3000., heights_m=grid.heights_m,
                        lat_deg=lat, lon_deg=lon)
    return prior, obs, grid, cfg


CASES = {
    "bounded": lambda: problem(fields=3, batches=4),
    "many-batches": lambda: problem(fields=2, batches=20),
    "full-depth": lambda: full_depth_problem(fields=3, batches=4, chunk=13),
    "geodesic": geodesic_problem,
}


def _both(prior, obs, grid, cfg, namespace):
    old_diag, new_diag = LetkfDiagnostics(), LetkfDiagnostics()
    old = letkf.analyze(prior, obs, grid, replace(cfg, neighbor_search="forward"),
                        old_diag, solve_namespace=namespace)
    new = letkf.analyze(prior, obs, grid, replace(cfg, neighbor_search="index"),
                        new_diag, solve_namespace=namespace)
    return old, new, old_diag, new_diag


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_index_is_bitwise_the_forward_walk_on_the_host(case, dtype):
    prior, obs, grid, cfg = CASES[case]()
    cfg = replace(cfg, solve_dtype=dtype)
    old, new, old_diag, new_diag = _both(prior, obs, grid, cfg, np)
    for name in prior:
        np.testing.assert_array_equal(new[name], old[name])
    assert new_diag.active_points == old_diag.active_points > 0
    assert new_diag.max_local_obs == old_diag.max_local_obs
    assert new_diag.batches == old_diag.batches
    assert new_diag.neighbor_search == "index"
    assert old_diag.neighbor_search == "forward"
    receipt = new_diag.neighbor_index
    assert receipt["observations"] == sum(int(np.count_nonzero(o.mask)) for o in obs)
    assert 0 < receipt["neighbours"] <= receipt["pairs_evaluated"]


def test_index_roster_matches_the_forward_roster_entry_by_entry():
    """Every gridpoint's (batch, observation, weight) list, in order."""
    from gpuwm.da.neighbor_index import build_neighbor_index
    prior, obs, grid, cfg = geodesic_problem(batches=3, members=6)
    # The forward roster, rebuilt independently gridpoint by gridpoint from
    # the public helpers.
    nz, ny, nx = grid.heights_m.shape
    zflat = grid.height_field(ny, nx).reshape(-1)
    lat = np.radians(grid.lat_deg).reshape(-1)
    lon = np.radians(grid.lon_deg).reshape(-1)

    def hdist(a, b):
        return letkf._geodesic_m(lat[a], lon[a], lat[b], lon[b],
                                 float(grid.earth_radius_m))

    stencils = []
    for o in obs:
        dj, di = letkf._horizontal_stencil(cfg.localization, grid, nx, ny)
        dk = letkf._vertical_stencil(cfg.localization, grid)
        j0, i0 = (o.window[0], o.window[2]) if o.window else (0, 0)
        mask = np.asarray(o.mask)
        stencils.append(dict(
            mask=mask.reshape(-1), values=np.where(mask, o.values, 0).reshape(-1),
            err2=np.where(mask, o.errors, 1).reshape(-1).astype(np.float64) ** 2,
            sim=np.where(mask[None], o.simulated, 0).reshape(o.simulated.shape[0], -1),
            dk=dk, dj=dj, di=di, hcut=cfg.localization.horizontal_m,
            vcut=cfg.localization.vertical_m, j0=j0, i0=i0,
            nj=mask.shape[1], ni=mask.shape[2]))
    index = build_neighbor_index(stencils, shape=(nz, ny, nx), zflat=zflat,
                                 horizontal_distance=hdist, members=6,
                                 solve_dtype=np.float64, xp=np,
                                 pair_budget=257)
    assert index.receipt["slabs"] > 1        # the small budget forces slabs
    for g in range(nz * ny * nx):
        k, rem = divmod(g, ny * nx)
        j, i = divmod(rem, nx)
        expected = []
        for b, st in enumerate(stencils):
            for a, d in enumerate(st["dk"]):
                for h, (oj, oi) in enumerate(zip(st["dj"], st["di"])):
                    k2, j2, i2 = k + d, j + oj, i + oi
                    if not (0 <= k2 < nz and 0 <= j2 < ny and 0 <= i2 < nx):
                        continue
                    jw, iw = j2 - st["j0"], i2 - st["i0"]
                    if not (0 <= jw < st["nj"] and 0 <= iw < st["ni"]):
                        continue
                    local = (k2 * st["nj"] + jw) * st["ni"] + iw
                    if not st["mask"][local]:
                        continue
                    n = (k2 * ny + j2) * nx + i2
                    wh = letkf.gaspari_cohn(hdist(np.array([[j * nx + i]]),
                                                  np.array([[j2 * nx + i2]])),
                                            st["hcut"])[0, 0]
                    wv = letkf.gaspari_cohn(np.abs(zflat[[n]] - zflat[[g]]),
                                            st["vcut"])[0]
                    w = wv * wh
                    if w > 0:
                        expected.append((b, local, w))
        lo, hi = index.row_ptr[g], index.row_ptr[g + 1]
        got_obs = index.obs[lo:hi]
        got_w = index.weight[lo:hi]
        assert hi - lo == len(expected), g
        for (b, local, w), oid, gw in zip(expected, got_obs, got_w):
            start = index.batch_offsets[b]
            lidx = np.flatnonzero(stencils[b]["mask"])
            assert lidx[oid - start] == local
            assert gw == w


def test_neighbor_search_mode_is_validated():
    _, _, _, cfg = problem()
    with pytest.raises(letkf.LetkfError, match="neighbor_search"):
        replace(cfg, neighbor_search="kdtree")


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(CASES))
def test_index_is_bitwise_the_forward_walk_on_the_card(case):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no card")
    prior, obs, grid, cfg = CASES[case]()
    old, new, old_diag, new_diag = _both(prior, obs, grid, cfg, cp)
    for name in prior:
        np.testing.assert_array_equal(new[name], old[name])
    assert new_diag.neighbor_index["namespace"] == "cupy"
    assert new_diag.active_points == old_diag.active_points
    again = letkf.analyze(prior, obs, grid, cfg, solve_namespace=cp)
    for name in prior:
        np.testing.assert_array_equal(again[name], new[name])


def _on_card(cp, prior, obs):
    from dataclasses import replace as _replace
    return ({k: cp.asarray(v) for k, v in prior.items()},
            [_replace(o, values=cp.asarray(o.values), errors=cp.asarray(o.errors),
                      simulated=cp.asarray(o.simulated), mask=cp.asarray(o.mask))
             for o in obs])


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(CASES))
def test_resident_card_route_reads_the_same_roster(case):
    """The card-resident route packs from the roster too.

    Against the dense resident transform the answer agrees to rounding
    (the dense route sums the same positive terms with zero-weight slots
    in between); against the bounded route on the same roster it agrees to
    the rounding of the prior mean, which each storage forms in its own
    namespace; and it is the same bytes run to run.
    """
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no card")
    prior, obs, grid, cfg = CASES[case]()
    d_prior, d_obs = _on_card(cp, prior, obs)
    dense_diag, index_diag = LetkfDiagnostics(), LetkfDiagnostics()
    dense = letkf.analyze(d_prior, d_obs, grid,
                          replace(cfg, neighbor_search="forward"), dense_diag)
    indexed = letkf.analyze(d_prior, d_obs, grid, cfg, index_diag)
    again = letkf.analyze(d_prior, d_obs, grid, cfg)
    staged = letkf.analyze(prior, obs, grid, cfg, solve_namespace=cp)
    assert dense_diag.neighbor_search == "dense"
    assert index_diag.neighbor_search == "index"
    assert index_diag.active_points == dense_diag.active_points
    assert index_diag.max_local_obs == dense_diag.max_local_obs
    for name in prior:
        got = cp.asnumpy(indexed[name])
        np.testing.assert_array_equal(got, cp.asnumpy(again[name]))
        np.testing.assert_allclose(got, cp.asnumpy(dense[name]),
                                   atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(got, staged[name], atol=1e-12, rtol=1e-11)


def test_roster_route_skips_the_whole_batch_placeholder_copies(monkeypatch):
    """QC on the roster route checks observed points and copies nothing.

    The forward walk gathers rectangular stencils, so validation replaced
    every unobserved value, error and member H(x) with a finite placeholder
    -- three whole-batch copies per batch (2 GB per surface batch on the
    recent case).  The roster reads observed points only; the breakage this
    prevents is those copies coming back on the route that never reads them.
    """
    prior, obs, grid, cfg = problem(fields=2, batches=4)
    batch_shapes = ({np.shape(o.simulated) for o in obs}
                    | {np.shape(o.values) for o in obs})
    real_where = np.where
    copies = {"n": 0}

    def counting_where(*args, **kwargs):
        out = real_where(*args, **kwargs)
        if len(args) == 3 and np.shape(out) in batch_shapes:
            copies["n"] += 1
        return out

    monkeypatch.setattr(np, "where", counting_where)
    letkf.analyze(prior, obs, grid, cfg, solve_namespace=np)
    assert copies["n"] == 0
    letkf.analyze(prior, obs, grid, replace(cfg, neighbor_search="forward"),
                  solve_namespace=np)
    assert copies["n"] >= len(obs)


def test_host_resident_roster_packs_the_same_bits():
    """A roster kept on the host packs exactly what a resident one packs.

    Breakage prevented: the 3 km CONUS roster (about 70 GB) cannot stay on
    a 32 GB card, so it is kept on the host and sliced per chunk; a slice
    that drops, reorders or reweights a neighbour would change the
    analysis.
    """
    from gpuwm.da.neighbor_index import build_neighbor_index
    prior, obs, grid, cfg = geodesic_problem(batches=3, members=6)
    nz, ny, nx = grid.heights_m.shape
    zflat = grid.height_field(ny, nx).reshape(-1)
    lat = np.radians(grid.lat_deg).reshape(-1)
    lon = np.radians(grid.lon_deg).reshape(-1)

    def hdist(a, b):
        return letkf._geodesic_m(lat[a], lon[a], lat[b], lon[b],
                                 float(grid.earth_radius_m))

    stencils = []
    for o in obs:
        dj, di = letkf._horizontal_stencil(cfg.localization, grid, nx, ny)
        dk = letkf._vertical_stencil(cfg.localization, grid)
        j0, i0 = (o.window[0], o.window[2]) if o.window else (0, 0)
        mask = np.asarray(o.mask)
        stencils.append(dict(
            mask=mask.reshape(-1), values=np.where(mask, o.values, 0).reshape(-1),
            err2=np.where(mask, o.errors, 1).reshape(-1).astype(np.float64) ** 2,
            sim=np.where(mask[None], o.simulated, 0).reshape(o.simulated.shape[0], -1),
            dk=dk, dj=dj, di=di, hcut=cfg.localization.horizontal_m,
            vcut=cfg.localization.vertical_m, j0=j0, i0=i0,
            nj=mask.shape[1], ni=mask.shape[2]))
    kw = dict(shape=(nz, ny, nx), zflat=zflat, horizontal_distance=hdist,
              members=6, solve_dtype=np.float64, xp=np, pair_budget=257)
    device = build_neighbor_index(stencils, resident="device", **kw)
    host = build_neighbor_index(stencils, resident="host", **kw)
    assert host.host_resident and not device.host_resident
    assert host.receipt["resident"] == "host"
    np.testing.assert_array_equal(host.row_ptr, device.row_ptr)
    active = np.flatnonzero(device.counts > 0)
    assert active.size > 10
    local_max = int(device.counts.max())
    for gpts in (active, active[::3], active[-7:]):
        for a, b in zip(device.pack(gpts, local_max, 6, np.float64),
                        host.pack(gpts, local_max, 6, np.float64)):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="resident"):
        build_neighbor_index(stencils, resident="card", **kw)


def test_row_bands_build_the_same_roster():
    """A level cut into row bands gives the whole-level roster, bit for bit.

    Breakage prevented: one analysis level with more candidate pairs than
    the slab budget (3 km CONUS, 100 km surface localisation: about a
    billion) was sorted as one slab and exhausted a 32 GB card.  Such a
    level is now cut into row bands; a band edge that dropped, repeated or
    reordered a neighbour would change the analysis.
    """
    from gpuwm.da.neighbor_index import build_neighbor_index
    prior, obs, grid, cfg = geodesic_problem(batches=3, members=6)
    nz, ny, nx = grid.heights_m.shape
    zflat = grid.height_field(ny, nx).reshape(-1)
    lat = np.radians(grid.lat_deg).reshape(-1)
    lon = np.radians(grid.lon_deg).reshape(-1)

    def hdist(a, b):
        return letkf._geodesic_m(lat[a], lon[a], lat[b], lon[b],
                                 float(grid.earth_radius_m))

    stencils = []
    for o in obs:
        dj, di = letkf._horizontal_stencil(cfg.localization, grid, nx, ny)
        dk = letkf._vertical_stencil(cfg.localization, grid)
        j0, i0 = (o.window[0], o.window[2]) if o.window else (0, 0)
        mask = np.asarray(o.mask)
        stencils.append(dict(
            mask=mask.reshape(-1), values=np.where(mask, o.values, 0).reshape(-1),
            err2=np.where(mask, o.errors, 1).reshape(-1).astype(np.float64) ** 2,
            sim=np.where(mask[None], o.simulated, 0).reshape(o.simulated.shape[0], -1),
            dk=dk, dj=dj, di=di, hcut=cfg.localization.horizontal_m,
            vcut=cfg.localization.vertical_m, j0=j0, i0=i0,
            nj=mask.shape[1], ni=mask.shape[2]))
    kw = dict(shape=(nz, ny, nx), zflat=zflat, horizontal_distance=hdist,
              members=6, solve_dtype=np.float64, xp=np)
    whole = build_neighbor_index(stencils, pair_budget=1 << 40, **kw)
    assert whole.receipt["row_band_slabs"] == 0
    for budget in (257, 31, 1):
        banded = build_neighbor_index(stencils, pair_budget=budget, **kw)
        assert banded.receipt["row_band_slabs"] > 0
        np.testing.assert_array_equal(banded.counts, whole.counts)
        np.testing.assert_array_equal(banded.row_ptr, whole.row_ptr)
        np.testing.assert_array_equal(banded.obs, whole.obs)
        np.testing.assert_array_equal(banded.weight, whole.weight)
        assert banded.receipt["neighbours"] == whole.receipt["neighbours"]
