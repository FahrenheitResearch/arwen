# tests/test_terrain_init.py  (Phase 2 Task 3: terrain module + terrain-aware
# DomainState init; general-form EOS diagnostics)
import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import make_base_state, make_vertical_coord


def theta_const(z):
    return np.full_like(np.asarray(z, float), 300.0)


def hill_profile(cfg):
    """The plan's bell ridge on domain-centered cell-center x, float64."""
    x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
    h = cfg.hill_height / (1.0 + (x / cfg.hill_halfwidth) ** 2)
    return np.broadcast_to(h, (cfg.ny, cfg.nx)).copy()


def terrain_cfg(**kw):
    base = dict(nx=64, ny=1, nz=48, dx=2000.0, dy=2000.0, ztop=16000.0,
                dt=1.0, run_seconds=0.0, terrain_opt=1,
                hill_height=1000.0, hill_halfwidth=10000.0)
    base.update(kw)
    return RunConfig(**base)


# ---- terrain.bell_hill (CPU) --------------------------------------------------

def test_bell_hill_profile():
    from gpuwm.core.terrain import bell_hill
    cfg = terrain_cfg()
    h = bell_hill(cfg)
    assert h.shape == (cfg.ny, cfg.nx) and h.dtype == np.float64
    # exact formula on domain-centered cell-center x, hill centered at x = 0
    assert np.array_equal(h, hill_profile(cfg))
    # symmetric about the domain center (even nx: mirror pairs match)
    assert np.allclose(h[:, :], h[:, ::-1])
    # peak at the two central cells, decaying outward
    ic = cfg.nx // 2
    assert h[0, ic] == h.max() and h[0, ic - 1] == h.max()
    assert np.all(np.diff(h[0, :ic]) > 0) and np.all(np.diff(h[0, ic:]) < 0)
    # decays into the far field: under 10% of the peak at the domain edge
    # (edge cells sit ~6.3 halfwidths out for this cfg)
    assert h[0, 0] < 0.1 * cfg.hill_height and h[0, -1] < 0.1 * cfg.hill_height


def test_bell_hill_y_uniform():
    from gpuwm.core.terrain import bell_hill
    cfg = terrain_cfg(ny=3)
    h = bell_hill(cfg)
    assert h.shape == (3, cfg.nx)
    assert np.array_equal(h[0], h[1]) and np.array_equal(h[0], h[2])


# ---- DomainState plumbing -----------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_height_half_raises_before_load_base():
    # final-review carry-over (ledger minor T6): today this silently returns
    # zeros; it must raise until load_base has filled the base geopotential.
    from gpuwm.core.state import DomainState
    s = DomainState(terrain_cfg())
    with pytest.raises(RuntimeError, match="load_base"):
        s.height_half()


@pytest.mark.gpu
@requires_gpu
def test_flat_state_gains_general_fields():
    import cupy as cp
    from gpuwm.core.state import init_at_rest
    cfg = RunConfig(nx=16, ny=2, nz=8, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=0.0)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_const, p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    # 2-D column mass broadcast of the flat scalar; zero terrain height
    assert s.mub2d.shape == (2, 16) and s.mub2d.dtype == cp.float32
    assert np.all(cp.asnumpy(s.mub2d) == np.float32(b.mub))
    assert s.ht.shape == (2, 16) and float(cp.abs(s.ht).max()) == 0.0
    # hybrid coefficient arrays ride along (identity for hybrid_opt=0)
    assert s.c1h.shape == (8,) and s.c2h.shape == (8,)
    assert s.c1f.shape == (9,) and s.c2f.shape == (9,)
    for arr in (s.c1h, s.c2h, s.c1f, s.c2f):
        assert arr.dtype == cp.float32
    assert np.all(cp.asnumpy(s.c1h) == 1.0) and np.all(cp.asnumpy(s.c2h) == 0.0)
    assert np.all(cp.asnumpy(s.c1f) == 1.0) and np.all(cp.asnumpy(s.c2f) == 0.0)
    # base profiles keep their Phase 1 column shapes
    assert s.thb.shape == (8,) and s.phb.shape == (9,)


@pytest.mark.gpu
@requires_gpu
def test_terrain_state_allocation_shapes_dtypes():
    import cupy as cp
    from gpuwm.core.state import init_at_rest
    cfg = terrain_cfg(ny=2)
    terrain = hill_profile(cfg)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_const, p_surf=cfg.p_surf, ztop=cfg.ztop,
                        terrain_z=terrain)
    s = init_at_rest(cfg, vc, b, terrain_z=terrain)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    # terrain height and 2-D dry mass on device, FP32
    assert s.ht.shape == (ny, nx) and s.ht.dtype == cp.float32
    assert np.array_equal(cp.asnumpy(s.ht), terrain.astype(np.float32))
    assert s.mub2d.shape == (ny, nx) and s.mub2d.dtype == cp.float32
    assert np.array_equal(cp.asnumpy(s.mub2d), b.mub.astype(np.float32))
    # base profiles are per-column with terrain
    assert s.thb.shape == (nz, ny, nx) and s.thb.dtype == cp.float32
    assert s.pb.shape == (nz, ny, nx)
    assert s.alb.shape == (nz, ny, nx)
    assert s.phb.shape == (nz + 1, ny, nx)
    assert np.array_equal(cp.asnumpy(s.phb[0]),
                          (c.G * terrain).astype(np.float32))
    # hybrid coefficients loaded from the coord
    assert np.array_equal(cp.asnumpy(s.c1h), vc.c1h.astype(np.float32))
    assert np.array_equal(cp.asnumpy(s.c2h), vc.c2h.astype(np.float32))
    # perturbations start at rest
    assert float(cp.abs(s.thp).max()) == 0.0
    assert float(cp.abs(s.php).max()) == 0.0
    # column-dependent half-level heights sit above the terrain
    z = s.height_half()
    assert z.shape == (nz, ny, nx)
    assert np.all(z[0] > terrain) and np.all(np.diff(z, axis=0) > 0)


@pytest.mark.gpu
@requires_gpu
def test_load_base_terrain_cfg_mismatch_raises():
    from gpuwm.core.state import init_at_rest
    # terrain base into a flat-allocated state
    cfg_flat = terrain_cfg(terrain_opt=0)
    terrain = hill_profile(cfg_flat)
    vc = make_vertical_coord(cfg_flat.nz)
    b_terr = make_base_state(vc, theta_const, p_surf=cfg_flat.p_surf,
                             ztop=cfg_flat.ztop, terrain_z=terrain)
    with pytest.raises(ValueError, match="terrain_opt"):
        init_at_rest(cfg_flat, vc, b_terr)
    # flat base into a terrain-allocated state
    cfg_terr = terrain_cfg()
    b_flat = make_base_state(make_vertical_coord(cfg_terr.nz), theta_const,
                             p_surf=cfg_terr.p_surf, ztop=cfg_terr.ztop)
    with pytest.raises(ValueError, match="terrain_opt"):
        init_at_rest(cfg_terr, make_vertical_coord(cfg_terr.nz), b_flat)


@pytest.mark.gpu
@requires_gpu
def test_init_terrain_z_crosscheck():
    from gpuwm.core.state import init_at_rest
    cfg = terrain_cfg()
    terrain = hill_profile(cfg)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_const, p_surf=cfg.p_surf, ztop=cfg.ztop,
                        terrain_z=terrain)
    # disagreeing terrain_z is rejected
    with pytest.raises(ValueError, match="terrain_z"):
        init_at_rest(cfg, vc, b, terrain_z=terrain + 1.0)
    # terrain_z against a flat base is rejected
    cfg_flat = terrain_cfg(terrain_opt=0)
    b_flat = make_base_state(make_vertical_coord(cfg_flat.nz), theta_const,
                             p_surf=cfg_flat.p_surf, ztop=cfg_flat.ztop)
    with pytest.raises(ValueError, match="terrain_z"):
        init_at_rest(cfg_flat, make_vertical_coord(cfg_flat.nz), b_flat,
                     terrain_z=np.zeros((cfg_flat.ny, cfg_flat.nx)))
    # omitting terrain_z still fills ht from the base state
    import cupy as cp
    s = init_at_rest(cfg, vc, b)
    assert np.array_equal(cp.asnumpy(s.ht), terrain.astype(np.float32))


# ---- diagnostics over terrain -------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_diagnostics_hydrostatic_pressure_over_terrain():
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import init_at_rest
    cfg = terrain_cfg()
    terrain = hill_profile(cfg)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_const, p_surf=cfg.p_surf, ztop=cfg.ztop,
                        terrain_z=terrain)
    s = init_at_rest(cfg, vc, b, terrain_z=terrain)
    update_diagnostics(s)
    p = cp.asnumpy(s.p)
    # at rest the diagnosed pressure reproduces the base pressure per column
    np.testing.assert_allclose(p, b.pb, rtol=3e-4)

    # valley-column surface p exceeds the ridge-top column's by the
    # hydrostatic weight of the intervening air, within 1%
    ic = int(np.argmax(terrain[0]))            # ridge column
    i0 = int(np.argmin(terrain[0]))            # valley column
    zh = 0.5 * (b.phb[0] + b.phb[1]) / c.G     # (ny, nx) first half-level z
    dp_gpu = float(p[0, 0, i0] - p[0, 0, ic])
    # reference hydrostatic p(z): the same fine-grid Exner integration the
    # base state was built from (pi is exactly piecewise linear in z)
    zf = np.append(np.arange(0.0, cfg.ztop, 1.0), cfg.ztop)
    dz = np.diff(zf)
    z_mid = 0.5 * (zf[:-1] + zf[1:])
    pi_surf = (cfg.p_surf / c.P0) ** c.RCP
    pi = np.concatenate(([pi_surf], pi_surf - np.cumsum(
        c.G * dz / (c.CP * theta_const(z_mid)))))
    pi_at = np.interp([zh[0, i0], zh[0, ic]], zf, pi)
    p_ref = c.P0 * pi_at ** (1.0 / c.RCP)
    dp_ref = float(p_ref[0] - p_ref[1])
    assert dp_gpu > 0.0
    assert abs(dp_gpu - dp_ref) < 0.01 * dp_ref


@pytest.mark.gpu
@requires_gpu
def test_diagnostics_kernel_matches_mirror_terrain_hybrid():
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import DTYPE, init_at_rest
    from gpuwm.verify.npref import np_calc_p_alpha
    cfg = terrain_cfg(nx=32, ny=2, nz=40, hybrid_opt=2, etac=0.2)
    terrain = hill_profile(cfg)
    vc = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2)
    b = make_base_state(vc, theta_const, p_surf=cfg.p_surf, ztop=cfg.ztop,
                        terrain_z=terrain)
    s = init_at_rest(cfg, vc, b, terrain_z=terrain)
    # c1h/c2h are non-trivial for the true hybrid coordinate
    assert np.any(vc.c1h != 1.0) and np.any(vc.c2h != 0.0)
    rng = np.random.default_rng(3)
    for name, amp in (("thp", 0.5), ("php", 20.0), ("mup", 20.0)):
        arr = getattr(s, name)
        arr[...] = cp.asarray(amp * rng.standard_normal(arr.shape),
                              dtype=DTYPE)
    update_diagnostics(s)
    p_ref, al_ref, alt_ref = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), b, vc)
    np.testing.assert_allclose(cp.asnumpy(s.p), p_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.alt), alt_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.al), al_ref, rtol=2e-5, atol=2e-5)


# ---- flat path stays bitwise Phase 1 ------------------------------------------

#: Verbatim Phase 1 diagnostics kernel (pre-Task 3), the bit-compatibility
#: anchor for the flat path: scalar mub, 1-D base columns, alt = -dph*rdnw/mu.
_PHASE1_DIAG_SRC = r"""
extern "C" __global__
void calc_p_alpha_phase1(const real* __restrict__ thp,
                         const real* __restrict__ php,
                         const real* __restrict__ mup,
                         const real* __restrict__ thb,
                         const real* __restrict__ phb,
                         const real* __restrict__ alb,
                         const real* __restrict__ rdnw,
                         real mub,
                         int nz, int ny, int nx,
                         real* __restrict__ p,
                         real* __restrict__ al,
                         real* __restrict__ alt)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;

    real mu = mub + mup[(size_t)j * nx + i];
    for (int k = 0; k < nz; ++k) {
        real th  = thb[k] + thp[IDX3(k, j, i)];
        real dph = (phb[k + 1] + php[IDX3(k + 1, j, i)])
                 - (phb[k]     + php[IDX3(k,     j, i)]);
        real a = -dph * rdnw[k] / mu;
        alt[IDX3(k, j, i)] = a;
        al[IDX3(k, j, i)]  = a - alb[k];
        p[IDX3(k, j, i)]   = P0 * powf((RD * th) / (P0 * a), GAMMA);
    }
}
"""


@pytest.mark.gpu
@requires_gpu
def test_flat_init_and_diagnostics_bitwise_phase1():
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.kernels import _preamble
    from gpuwm.core.state import init_theta_perturbation
    cfg = RunConfig(nx=32, ny=1, nz=16, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=0.0)
    vc = make_vertical_coord(cfg.nz)
    sounding = lambda z: 300.0 + 0.003 * np.asarray(z, float)
    b = make_base_state(vc, sounding, p_surf=cfg.p_surf, ztop=cfg.ztop)
    rng = np.random.default_rng(11)
    vals = rng.normal(0.0, 1.0, (cfg.nz, cfg.ny, cfg.nx))
    s = init_theta_perturbation(cfg, vc, b, lambda x, z: vals)

    # --- thp/php: bitwise against the verbatim Phase 1 init algorithm
    th_total = b.thb[:, None, None] + vals
    p_top = cfg.p_surf - b.mub
    p_col = (vc.znu * b.mub + p_top)[:, None, None]
    alpha = c.RD * th_total * (p_col / c.P0) ** c.RCP / p_col
    ph = np.zeros((cfg.nz + 1, cfg.ny, cfg.nx))
    for k in range(cfg.nz):
        ph[k + 1] = ph[k] - vc.dnw[k] * b.mub * alpha[k]
    php_ref = ph - b.phb[:, None, None]
    assert np.array_equal(cp.asnumpy(s.thp), vals.astype(np.float32))
    assert np.array_equal(cp.asnumpy(s.php), php_ref.astype(np.float32))

    # --- p/al/alt: bitwise against the verbatim Phase 1 kernel
    update_diagnostics(s)
    mod = cp.RawModule(code=_preamble() + _PHASE1_DIAG_SRC,
                       options=("-std=c++17",))
    k1 = mod.get_function("calc_p_alpha_phase1")
    p1 = cp.zeros_like(s.p)
    al1 = cp.zeros_like(s.al)
    alt1 = cp.zeros_like(s.alt)
    ncol = cfg.ny * cfg.nx
    k1(((ncol + 255) // 256,), (256,),
       (s.thp, s.php, s.mup, s.thb, s.phb, s.alb, s.rdnw,
        np.float32(b.mub), np.int32(cfg.nz), np.int32(cfg.ny),
        np.int32(cfg.nx), p1, al1, alt1))
    assert bool((s.p == p1).all())
    assert bool((s.al == al1).all())
    assert bool((s.alt == alt1).all())


# ---- theta perturbation over terrain ------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_theta_perturbation_over_terrain_at_rest_phi_stays_zero():
    # phi' init stays 0 with terrain: the base state already carries the
    # terrain in phb, so a zero-theta' init must leave php identically zero.
    import cupy as cp
    from gpuwm.core.state import init_theta_perturbation
    cfg = terrain_cfg()
    terrain = hill_profile(cfg)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_const, p_surf=cfg.p_surf, ztop=cfg.ztop,
                        terrain_z=terrain)
    seen = {}

    def thp_zero(x, z):
        seen["x"], seen["z"] = x, z
        return np.zeros((cfg.nz, cfg.ny, cfg.nx))

    s = init_theta_perturbation(cfg, vc, b, thp_zero, terrain_z=terrain)
    # the height argument is per-column over terrain
    assert seen["x"].shape == (cfg.nx,)
    assert seen["z"].shape == (cfg.nz, cfg.ny, cfg.nx)
    assert np.all(seen["z"][0] > terrain)
    assert float(cp.abs(s.thp).max()) == 0.0
    assert float(cp.abs(s.php).max()) == 0.0

    # a nonzero theta' rebalances phi' per column: surface pinned, air above
    def thp_warm(x, z):
        out = np.zeros((cfg.nz, cfg.ny, cfg.nx))
        out[:8] = 2.0
        return out

    s2 = init_theta_perturbation(cfg, vc, b, thp_warm, terrain_z=terrain)
    php = cp.asnumpy(s2.php)
    assert float(np.abs(php[0]).max()) == 0.0      # surface geopotential fixed
    assert float(np.abs(php[1:]).max()) > 0.0      # column re-inflated above
