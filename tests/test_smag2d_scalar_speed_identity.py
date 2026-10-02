"""Lazy scalar interpolation keeps the original eager helper words.

The helper literal is from commit 9896aa48f7f611cdb0130c87c3a9ac7f56be5621.
Its whole smag2d.cu body SHA-256 was
eedf1fb33b0b8daa1582311bb667f006571a5cfa3d0f83c4b3107c310b4aa7df.
The archived-base probe and real forecast comparisons cover the whole source;
this focused test isolates the two scalar interpolation helper changes.
"""
from dataclasses import replace
from functools import lru_cache
import hashlib

import numpy as np
import pytest

from conftest import requires_gpu


EAGER_HELPERS = r'''__device__ __forceinline__
real wrf_scalar_w_xface(const WrfSmagGrid& q, const WrfScalarField& s,
                        int kw, int j, int i)
{
    real p0 = wrf_scalar(q, s, 0, j, i - 1) + wrf_scalar(q, s, 0, j, i);
    real p1 = wrf_scalar(q, s, 1, j, i - 1) + wrf_scalar(q, s, 1, j, i);
    real p2 = wrf_scalar(q, s, 2, j, i - 1) + wrf_scalar(q, s, 2, j, i);
    real pl = wrf_scalar(q, s, q.nz - 1, j, i - 1)
            + wrf_scalar(q, s, q.nz - 1, j, i);
    real pp = wrf_scalar(q, s, q.nz - 2, j, i - 1)
            + wrf_scalar(q, s, q.nz - 2, j, i);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_scalar(q, s, kc, j, i - 1) + wrf_scalar(q, s, kc, j, i);
    real pb = wrf_scalar(q, s, kb, j, i - 1) + wrf_scalar(q, s, kb, j, i);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

__device__ __forceinline__
real wrf_scalar_w_yface(const WrfSmagGrid& q, const WrfScalarField& s,
                        int kw, int j, int i)
{
    real p0 = wrf_scalar(q, s, 0, j - 1, i) + wrf_scalar(q, s, 0, j, i);
    real p1 = wrf_scalar(q, s, 1, j - 1, i) + wrf_scalar(q, s, 1, j, i);
    real p2 = wrf_scalar(q, s, 2, j - 1, i) + wrf_scalar(q, s, 2, j, i);
    real pl = wrf_scalar(q, s, q.nz - 1, j - 1, i)
            + wrf_scalar(q, s, q.nz - 1, j, i);
    real pp = wrf_scalar(q, s, q.nz - 2, j - 1, i)
            + wrf_scalar(q, s, q.nz - 2, j, i);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_scalar(q, s, kc, j - 1, i) + wrf_scalar(q, s, kc, j, i);
    real pb = wrf_scalar(q, s, kb, j - 1, i) + wrf_scalar(q, s, kb, j, i);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

'''
EAGER_HELPERS_SHA256 = "94c40e79f0d201f17ffabff96d04fbf2451d230738131c1bf7960c15feb08808"


def eager_reference_source(source):
    assert hashlib.sha256(EAGER_HELPERS.encode()).hexdigest() == EAGER_HELPERS_SHA256
    for name in ("wrf_scalar_w_xface", "wrf_scalar_w_yface"):
        signature = "__device__ __forceinline__\nreal " + name
        wrapper = source.index("#if defined(GPUWM_WRF_EXACT) || "
                               "defined(GPUWM_WRF_EXACT_C_DIFFUSION)\n" + signature)
        begin = source.index("\n#else\n", wrapper) + len("\n#else\n")
        end = source.index("\n#endif", begin)
        eager_begin = EAGER_HELPERS.index(signature)
        eager_end = EAGER_HELPERS.find("__device__ __forceinline__", eager_begin + 1)
        if eager_end < 0:
            eager_end = len(EAGER_HELPERS)
        source = (source[:begin] + EAGER_HELPERS[eager_begin:eager_end].rstrip()
                  + source[end:])
    return source


def test_original_eager_interpolation_literal_is_pinned():
    assert hashlib.sha256(EAGER_HELPERS.encode()).hexdigest() == EAGER_HELPERS_SHA256


@lru_cache(maxsize=1)
def reference_module():
    import cupy as cp
    from gpuwm.core.kernels import module_source

    return cp.RawModule(code=eager_reference_source(module_source("smag2d")),
                        options=("-std=c++17",))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("boundary_x,boundary_y", [
    (False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("moist", [False, True])
@pytest.mark.parametrize("time_t", [False, True])
@pytest.mark.parametrize("terrain", [False, True])
@pytest.mark.parametrize("full_theta", [False, True])
@pytest.mark.parametrize("shape", [(8, 9, 17), (6, 7, 129)])
def test_production_scalar_faces_match_eager_words(
        boundary_x, boundary_y, moist, time_t, terrain, full_theta, shape):
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import (
        _TPB, _save_time_t, _wrf_smag_grid_args, launch_wrf_smag2d_hd)
    from gpuwm.verify.npref import random_acoustic_state

    nz, ny, nx = shape
    state, cfg = random_acoustic_state(
        seed=734, nx=nx, ny=ny, nz=nz, stretch=1.4,
        hybrid_opt=2 if terrain else 0,
        hill_height=300.0 if terrain else 0.0,
        msf_amp=0.09, moist=moist)
    cfg = replace(cfg, km_opt=4, bl_pbl_physics=1, dx=900.0, dy=1100.0,
                  open_x=boundary_x, open_y=boundary_y)
    random = np.random.default_rng(908)
    if moist:
        state.qv[...] = cp.asarray(random.uniform(0.001, 0.025, shape), cp.float32)
        update_diagnostics(state, cfg.hypsometric_opt)
    _save_time_t(state)
    state.u[...] *= cp.float32(1.5)
    state.v[...] *= cp.float32(0.75)
    state.w[...] *= cp.float32(1.25)
    state.php[...] += cp.float32(0.03125)
    if moist:
        state.qv[...] *= cp.float32(0.8)
    field = cp.asarray(random.normal(0.0, 0.5, shape), cp.float32)
    coefficient = cp.asarray(random.uniform(0.0, 700.0, shape), cp.float32)
    coefficient[:, :, ::5] = cp.float32(0.0)
    fx = cp.full((nz, ny, nx + 1), cp.nan, cp.float32)
    fy = cp.full((nz, ny + 1, nx), cp.nan, cp.float32)
    initial = cp.asarray(random.normal(0.0, 0.2, shape), cp.float32)
    reference, actual = initial.copy(), initial.copy()
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3), np.int32(boundary_x), np.int32(boundary_y)]
    module = reference_module()
    module.get_function("wrf_smag_flux_s")(
        ((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz), (_TPB, 1, 1),
        tuple(common + [field, coefficient, state.thb, np.int32(full_theta),
                        np.int32(state.thb.ndim == 3), fx, fy] + tail))
    module.get_function("wrf_smag_hd_s")(
        ((nx + _TPB - 1) // _TPB, ny, nz), (_TPB, 1, 1),
        tuple(common + [fx, fy, reference] + tail))
    state.scratch(fx.shape, "diff6_x").fill(cp.float32(np.nan))
    state.scratch(fy.shape, "diff6_y").fill(cp.float32(np.nan))
    launch_wrf_smag2d_hd(state, cfg, field, coefficient, actual,
                         stagger="", time_t=time_t, full_theta=full_theta)
    for old, new in ((fx, state.scratch(fx.shape, "diff6_x")),
                     (fy, state.scratch(fy.shape, "diff6_y")),
                     (reference, actual)):
        np.testing.assert_array_equal(cp.asnumpy(new).view(np.uint32),
                                      cp.asnumpy(old).view(np.uint32))
    assert bool(cp.any(actual != initial))
