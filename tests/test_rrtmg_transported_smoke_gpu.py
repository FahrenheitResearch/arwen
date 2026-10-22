"""simple_dir_fdb = 1: the forecast's own smoke dims the legacy RRTMG shortwave.

HRRR v4.1.21 (WRFV3.9 fork): smoke/module_add_emiss_burn.F:194 forms the layer
AOD 1e-6 * 4.5 m2/g * smoke (ug/kg) * 1/ALT * dz8w, and
phys/module_radiation_driver.F:979-984 adds MIN(3, it) to the aer_opt = 3
layer AOD.  Zero smoke is word-for-word the run without it; a dense plume
dims the surface shortwave; a NaN or out-of-range word is taken as 1e-16 the
way the fork clamps it.  The column comes from the SW oracle deck, as in
tests/test_rrtmg_smoke_adapter_gpu.py.
"""
from datetime import datetime

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

START = datetime(2026, 10, 2, 21)


def _env():
    import cupy as cp
    from test_rrtmg_legacy_wiring import _bundle, _env_from, profile
    ny = nx = 2
    lat = np.array([[40., 40.], [40.02, 40.02]], np.float32)
    lon = np.array([[-100., -99.98], [-100., -99.98]], np.float32)
    bundle = _bundle(profile.__wrapped__(), ny * nx, seed=20261004)
    env = _env_from(bundle, np.arange(ny * nx), ny, nx, lat.ravel(), lon.ravel())
    for name in ("qc", "qr", "qi", "qs", "qg"):
        getattr(env.state, name).fill(cp.float32(0))
    env.state.nwfa = cp.full_like(env.state.qv, cp.float32(1.e8))
    env.state.nifa = cp.full_like(env.state.qv, cp.float32(1.e4))
    env.state.elapsed_seconds = 0.0
    env.state.alt = cp.full_like(env.state.qv, cp.float32(1.0))
    env.fields["glw"] = cp.full((ny, nx), cp.float32(300.))
    env.cfg.o3input = 0
    env.cfg.swint_opt = 0
    env.cfg.rrtmg_smoke_manifest = ""
    env.cfg.mp_physics = 28
    env.cfg.aer_opt = 3
    return env, lat, lon


def test_transported_smoke_dims_the_surface_shortwave():
    import cupy as cp
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    from test_rrtmg_legacy_wiring import _call
    env, lat, lon = _env()

    def run(smoke):
        env.cfg.simple_dir_fdb = 1 if smoke else 0
        adapter = RRTMGLegacyRadiation(
            START, lat, lon, p_top=env.p_top, o3input=0, longwave=False,
            shortwave=True, aer_opt=3,
            transported_smoke="chem_smoke" if smoke else None)
        return cp.asnumpy(_call(adapter, env).swdown)

    baseline = run(False)
    env.state.chem_smoke = cp.zeros_like(env.state.qv)
    np.testing.assert_array_equal(run(True).view(np.uint32), baseline.view(np.uint32))
    # NaN and negative words are 1e-16 in the fork: still the clean sun.
    env.state.chem_smoke[...] = cp.float32(np.nan)
    env.state.chem_smoke[0] = cp.float32(-5.0)
    np.testing.assert_array_equal(run(True).view(np.uint32), baseline.view(np.uint32))
    env.state.chem_smoke = cp.full_like(env.state.qv, cp.float32(200.0))
    smoky = run(True)
    assert (baseline > 0).all(), "the numerical fixture must be sunlit"
    assert np.isfinite(smoky).all()
    assert (baseline - smoky).min() > 5.0, (baseline, smoky)


def test_the_adapter_refuses_double_counts_and_other_shortwaves():
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    lat = np.zeros((1, 1), np.float32)
    with pytest.raises(ValueError, match="aer_opt=3"):
        RRTMGLegacyRadiation(START, lat, lat, longwave=False, shortwave=True,
                             aer_opt=0, transported_smoke="chem_smoke")
    with pytest.raises(ValueError, match="twice"):
        RRTMGLegacyRadiation(START, lat, lat, longwave=False, shortwave=True, aer_opt=3,
                             smoke_provider=object(), transported_smoke="chem_smoke")


def test_the_fused_transported_aod_is_the_two_kernel_words():
    """One pass, no temporaries, the same words as rrtmg_smoke_dry_density +
    rrtmg_smoke_aod over the fork's clamp (NaN, negative, >= 1.1e4 -> 1e-16)."""
    import cupy as cp
    from gpuwm.core import rrtmg_aerosol_optics as aer
    rng = np.random.default_rng(1)
    smoke = cp.asarray(rng.uniform(-10, 12000, (5, 7, 9)).astype(np.float32))
    smoke[0, 0, 0] = np.nan
    alt = cp.asarray(rng.uniform(0.8, 3, (5, 7, 9)).astype(np.float32))
    dz = cp.asarray(rng.uniform(20, 800, (5, 7, 9)).astype(np.float32))
    got = aer.smoke_aod_from_transported_device(smoke, alt, dz)
    clamped = cp.where((smoke >= 0) & (smoke < np.float32(1.1e4)), smoke,
                       np.float32(1e-16))
    want = aer.smoke_aod_from_dry_mixing_ratio_device(
        clamped, aer.smoke_dry_air_density_device(alt), dz)
    assert bool((got.view(cp.uint32) == want.view(cp.uint32)).all())
