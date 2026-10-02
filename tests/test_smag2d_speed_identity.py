"""Cached w stresses preserve the direct production kernel's output words."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("boundary_x,boundary_y", [
    (False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("moist", [False, True])
@pytest.mark.parametrize("time_t", [False, True])
@pytest.mark.parametrize("terrain", [False, True])
@pytest.mark.parametrize("shape", [(8, 9, 17), (6, 7, 129)])
def test_cached_w_stress_matches_direct_kernel_words(
        boundary_x, boundary_y, moist, time_t, terrain, shape, monkeypatch):
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import (
        _TPB, _save_time_t, _wrf_smag_grid_args, launch_wrf_smag2d_hd)
    from gpuwm.core.kernels import get_kernel, get_kernel_int_defines
    from gpuwm.core import dycore
    from gpuwm.verify.npref import random_acoustic_state

    nz, ny, nx = shape
    state, config = random_acoustic_state(
        seed=734, nx=nx, ny=ny, nz=nz, stretch=1.4,
        hybrid_opt=2 if terrain else 0,
        hill_height=300.0 if terrain else 0.0,
        msf_amp=0.09, moist=moist)
    config = replace(config, km_opt=4, bl_pbl_physics=1,
                     dx=900.0, dy=1100.0,
                     open_x=boundary_x, open_y=boundary_y)
    random = np.random.default_rng(908)
    if moist:
        state.qv[...] = cp.asarray(
            random.uniform(0.001, 0.025, state.qv.shape), dtype=cp.float32)
        update_diagnostics(state, config.hypsometric_opt)
    _save_time_t(state)
    # Distinct saved and live inputs make selecting the wrong carrier visible.
    state.u[...] *= cp.float32(1.5)
    state.v[...] *= cp.float32(0.75)
    state.w[...] *= cp.float32(1.25)
    state.php[...] += cp.float32(0.03125)
    if moist:
        state.qv[...] *= cp.float32(0.8)
    coefficient = cp.asarray(
        random.uniform(0.0, 700.0, (nz, ny, nx)), dtype=cp.float32)
    coefficient[:, :, ::5] = cp.float32(0.0)
    initial = cp.asarray(
        random.normal(0.0, 0.2, state.w.shape), dtype=cp.float32)
    reference = initial.copy()
    actual = initial.copy()
    common = _wrf_smag_grid_args(state, config, time_t=time_t)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(boundary_x), np.int32(boundary_y)]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz + 1)
    get_kernel_int_defines(
        "smag2d", "wrf_smag_hd_w", (("GPUWM_SMAG_DIRECT_W_REFERENCE", 1),))(
        grid, (_TPB, 1, 1), tuple(common + [coefficient, reference] + tail))
    # Shared face buffers contain hostile values before the production route.
    state.scratch((nz, ny, nx + 1), "diff6_x").fill(cp.float32(np.nan))
    state.scratch((nz, ny + 1, nx), "diff6_y").fill(cp.float32(np.nan))
    launches = []

    def traced_kernel(module, name):
        launches.append((module, name))
        return get_kernel(module, name)

    monkeypatch.setattr(dycore, "get_kernel", traced_kernel)
    field = state.w0 if time_t else state.w
    launch_wrf_smag2d_hd(
        state, config, field, coefficient, actual,
        stagger="z", time_t=time_t)
    if dycore.WRF_EXACT:
        assert launches == [("smag2d", "wrf_smag_hd_w")]
    elif str(state.w.device.compute_capability) == "120":
        assert launches == [("smag2d", "wrf_smag_w_stress"),
                            ("smag2d", "wrf_smag_hd_w_stress")]
    else:
        assert ("smag2d", "wrf_smag_w_primitives") in launches
        assert ("smag2d", "wrf_smag_hd_w_cached") in launches
    np.testing.assert_array_equal(
        cp.asnumpy(actual).view(np.uint32),
        cp.asnumpy(reference).view(np.uint32))
    # The top and bottom source rows are deliberately untouched, including bits.
    for row in (0, nz):
        np.testing.assert_array_equal(
            cp.asnumpy(actual[row]).view(np.uint32),
            cp.asnumpy(initial[row]).view(np.uint32))
    assert bool(cp.any(actual[1:-1] != initial[1:-1]))
