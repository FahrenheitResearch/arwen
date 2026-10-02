"""A wrfinput start reaches WRF's own cold-start state before the first step.

real.exe writes zero Noah liquid soil water (SH2O) and zero vertical
velocity (W).  WRF does not run from either: LSMINIT sets SH2O from SMOIS
and TSLB (module_sf_noahdrv.F), and start_domain_em diagnoses W from the
terrain slope and the lowest three wind levels (start_em.F, set_w_surface
in module_bc_em.F). Native restoration must run both initializers before
the first forecast step and before a later checkpoint replaces the state.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest import wrfinput as wi

_COORD_NAMES = ('znw', 'znu', 'dnw', 'rdnw', 'dn', 'rdn', 'fnp', 'fnm',
                'c1f', 'c2f', 'c3f', 'c4f', 'c1h', 'c2h', 'c3h', 'c4h')
f32 = np.float32


def _wrf_set_w_surface(u, v, ht, msftx, msfty, znw, cf1, cf2, cf3, rdx, rdy,
                       periodic_x, periodic_y):
    """module_bc_em.F set_w_surface with fill_w_flag, one FP32 rounding per
    operation in the Fortran evaluation order (no contraction)."""
    nz = u.shape[0]
    ny, nx = ht.shape
    w = np.zeros((nz + 1, ny, nx), np.float32)

    def column(a, j, i):
        return f32(f32(f32(cf1 * a[0, j, i]) + f32(cf2 * a[1, j, i]))
                   + f32(cf3 * a[2, j, i]))

    for j in range(ny):
        jm1 = j - 1 if j > 0 else (ny - 1 if periodic_y else 0)
        jp1 = j + 1 if j < ny - 1 else (0 if periodic_y else ny - 1)
        for i in range(nx):
            im1 = i - 1 if i > 0 else (nx - 1 if periodic_x else 0)
            ip1 = i + 1 if i < nx - 1 else (0 if periodic_x else nx - 1)
            y = f32(f32(f32(msfty[j, i] * f32(.5)) * rdy)
                    * f32(f32(f32(ht[jp1, i] - ht[j, i]) * column(v, j + 1, i))
                          + f32(f32(ht[j, i] - ht[jm1, i]) * column(v, j, i))))
            x = f32(f32(f32(msftx[j, i] * f32(.5)) * rdx)
                    * f32(f32(f32(ht[j, ip1] - ht[j, i]) * column(u, j, i + 1))
                          + f32(f32(ht[j, i] - ht[j, im1]) * column(u, j, i))))
            w[0, j, i] = f32(y + x)
            for k in range(1, nz + 1):
                w[k, j, i] = f32(f32(w[0, j, i] * znw[k]) * znw[k])
    return w


def _terrain_input(cfg, *, file_w):
    """A small real.exe-shaped cold start over sloped terrain."""
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    jj, ii = np.mgrid[0:ny, 0:nx]
    hgt = (40. * ii + 25. * jj + 15. * np.sin(ii * 1.3 + jj * .7)).astype(np.float64)
    coord = make_vertical_coord(nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.), 100000.,
                           cfg.ztop, terrain_z=hgt)
    rng = np.random.default_rng(7)
    raw = {name.upper(): np.asarray(getattr(coord, name), np.float32)
           for name in _COORD_NAMES}
    raw.update(
        P_TOP=np.float32(base.p_top), MUB=np.asarray(base.mub, np.float32),
        PB=np.asarray(base.pb, np.float32), ALB=np.asarray(base.alb, np.float32),
        T_INIT=(np.asarray(base.thb, np.float32) - f32(300.)),
        PHB=np.asarray(base.phb, np.float32), HGT=hgt.astype(np.float32),
        U=rng.uniform(-12., 15., (nz, ny, nx + 1)).astype(np.float32),
        V=rng.uniform(-9., 11., (nz, ny + 1, nx)).astype(np.float32),
        W=np.full((nz + 1, ny, nx), file_w, np.float32),
        T=np.zeros((nz, ny, nx), np.float32), PH=np.zeros((nz + 1, ny, nx), np.float32),
        MU=np.zeros((ny, nx), np.float32), P=np.zeros((nz, ny, nx), np.float32),
        AL=np.zeros((nz, ny, nx), np.float32),
        CF1=np.array([1.53], np.float32), CF2=np.array([-.61], np.float32),
        CF3=np.array([.08], np.float32),
        MAPFAC_M=np.full((ny, nx), 1.01, np.float32),
        MAPFAC_U=np.full((ny, nx + 1), 1.01, np.float32),
        MAPFAC_V=np.full((ny + 1, nx), 1.01, np.float32),
        MAPFAC_MX=rng.uniform(.98, 1.03, (ny, nx)).astype(np.float32),
        MAPFAC_MY=rng.uniform(.98, 1.03, (ny, nx)).astype(np.float32),
        F=np.zeros((ny, nx), np.float32), E=np.zeros((ny, nx), np.float32),
        SINALPHA=np.zeros((ny, nx), np.float32), COSALPHA=np.ones((ny, nx), np.float32))
    return SimpleNamespace(raw=raw, global_attributes={
        'HYBRID_OPT': coord.hybrid_opt, 'ETAC': coord.etac})


@pytest.mark.gpu
@pytest.mark.parametrize('boundaries', ['specified', 'periodic'])
@pytest.mark.parametrize('file_w', [0., 3.])
def test_restored_start_diagnoses_wrf_vertical_velocity(boundaries, file_w):
    cp = pytest.importorskip('cupy')
    from gpuwm.config import RunConfig
    edges = (dict(specified=True, spec_bdy_width=2) if boundaries == 'specified' else {})
    cfg = RunConfig(nx=7, ny=6, nz=5, ztop=6000., dx=3000., dy=3000., dt=12.,
                    run_seconds=12., moist=False, mp_physics=0, terrain_opt=1, **edges)
    restored = _terrain_input(cfg, file_w=file_w)
    state = wi.restore_domain_state(restored, cfg)
    raw = restored.raw
    periodic = boundaries == 'periodic'
    expected = _wrf_set_w_surface(
        raw['U'], raw['V'], raw['HGT'], raw['MAPFAC_MX'], raw['MAPFAC_MY'],
        raw['ZNW'], raw['CF1'][0], raw['CF2'][0], raw['CF3'][0],
        f32(f32(1.) / f32(cfg.dx)), f32(f32(1.) / f32(cfg.dy)), periodic, periodic)
    # WRF's default use_input_w=.false. replaces a nonzero file W as well.
    assert np.abs(expected[0]).max() > 0.05
    np.testing.assert_array_equal(cp.asnumpy(state.w).view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(cp.asnumpy(state.w0).view(np.uint32), expected.view(np.uint32))


def _noah_case():
    """Land columns with SH2O written as zeros, one water column and one
    column frozen in its top three layers."""
    ny, nx, layers = 4, 5, 4
    shape = ny, nx
    soil = layers, ny, nx
    landmask = np.ones(shape, np.float32)
    landmask[0, 0] = 0.
    isltyp = (np.arange(ny * nx, dtype=np.int32).reshape(shape) % 12) + 1
    isltyp[0, 0] = 14
    tslb = np.full(soil, 289.25, np.float32)
    tslb[:, 2, 3] = [266.5, 270.25, 272.75, 276.5]
    tslb[:, 1, 1] = [f32(273.149), 273.5, 274., 280.]   # FP32 guard word stays liquid
    smois = (np.linspace(.12, .41, layers * ny * nx, dtype=np.float32)
             .reshape(soil))
    tsk = np.full(shape, 291., np.float32)
    raw = dict(
        LANDMASK=landmask, XLAND=np.where(landmask > .5, 1., 2.).astype(np.float32),
        TSK=tsk, TSLB=tslb, SMOIS=smois, SH2O=np.zeros(soil, np.float32),
        LU_INDEX=np.where(landmask > .5, 10, 17).astype(np.int32), ISLTYP=isltyp,
        VEGFRA=np.full(shape, 60., np.float32), TMN=np.full(shape, 287., np.float32),
        XICE=np.zeros(shape, np.float32), SNOW=np.zeros(shape, np.float32),
        SNOWH=np.zeros(shape, np.float32), ALBBCK=np.full(shape, .2, np.float32),
        LAI=np.full(shape, 2., np.float32), GLW=np.full(shape, 300., np.float32))
    return raw


@pytest.mark.gpu
def test_noah_cold_start_takes_lsminit_liquid_water_not_the_file_zeros():
    cp = pytest.importorskip('cupy')
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.noah import load_tables, pack_params, sh2o_init
    from gpuwm.core.state import DomainState
    cfg = RunConfig(nx=5, ny=4, nz=4, ztop=3000., dx=1000., dy=1000., dt=1.,
                    run_seconds=1., moist=True, mp_physics=6,
                    sf_surface_physics=2, sf_sfclay_physics=1, bl_pbl_physics=1)
    coord = make_vertical_coord(cfg.nz)
    state = DomainState(cfg)
    state.load_base(coord, make_base_state(coord, lambda z: np.full_like(z, 300.),
                                           100000., cfg.ztop))
    raw = _noah_case()
    restored = SimpleNamespace(raw=raw, global_attributes={
        'MMINLU': 'MODIFIED_IGBP_MODIS_NOAH'})
    driver = wi.initialize_wrfinput_physics(state, restored, cfg)
    liquid = cp.asnumpy(driver.fields['sh2o'])
    smois, tslb = raw['SMOIS'], raw['TSLB'].copy()
    water = raw['LANDMASK'] < .5
    tslb[:, water] = raw['TSK'][water]
    warm = tslb >= f32(273.149)
    assert warm.sum() == warm.size - 3
    # Unfrozen layers copy SMOIS word for word, as LSMINIT does.
    np.testing.assert_array_equal(liquid[warm].view(np.uint32), smois[warm].view(np.uint32))
    # Frozen layers take the Flerchinger/FRH2O partition: some, not all,
    # of the water is liquid.  The FP64 helper is the independent check;
    # FRH2O's 0.005 Newton stop leaves FP32 and FP64 a few words apart.
    reference = sh2o_init(smois, tslb, raw['ISLTYP'],
                          pack_params(load_tables(mminlu='MODIFIED_IGBP_MODIS_NOAH')))
    frozen = ~warm
    assert np.all(liquid[frozen] > 0.) and np.all(liquid[frozen] < smois[frozen])
    np.testing.assert_allclose(liquid[frozen], reference[frozen], rtol=1e-4, atol=0)
