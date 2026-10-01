"""The DA cycle admission's analysis and FFT terms, against the card.

``tests/test_da_cycle_memory.py`` pins the admission's arithmetic on the
CPU.  These cells run the priced work on a card with the device pool
instrumented, and hold each price to what the pool actually reached: the
leg analysis on its resident route, and one perturbation draw under the
cuFFT plans the admission measured.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@contextlib.contextmanager
def _pool_high_water():
    """The device pool's largest used and total bytes over the block."""
    import cupy as cp

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    peak = {"used": pool.used_bytes(), "total": pool.total_bytes(),
            "base_used": pool.used_bytes()}

    def malloc(size):
        memory = pool.malloc(size)
        peak["used"] = max(peak["used"], pool.used_bytes())
        peak["total"] = max(peak["total"], pool.total_bytes())
        return memory

    cp.cuda.set_allocator(malloc)
    try:
        yield peak
    finally:
        cp.cuda.set_allocator(pool.malloc)


def _synthetic_leg(*, members=10, nz=10, ny=64, nx=64, seed=7):
    """A prior, one whole-grid batch and two windowed radar batches."""
    from gpuwm.da.letkf import GriddedObs, GridGeometry, Localization

    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    grid = GridGeometry(dx_m=3000.0, dy_m=3000.0,
                        heights_m=np.linspace(200.0, 12000.0, nz))
    fields = ("u", "v")
    prior = {name: rng.standard_normal((members,) + shape) + 3.0 * index
             for index, name in enumerate(fields)}
    batches = []
    mask = rng.random(shape) < 0.05
    sim = rng.standard_normal((members,) + shape)
    batches.append(GriddedObs(
        name="whole", values=np.where(mask, sim.mean(axis=0) + 0.3, 0.0),
        errors=np.full(shape, 1.0), simulated=sim, mask=mask))
    for index, (j0, i0) in enumerate(((4, 4), (30, 26))):
        window = (j0, j0 + 23, i0, i0 + 23)
        wshape = (nz, 24, 24)
        wmask = rng.random(wshape) < 0.2
        wsim = rng.standard_normal((members,) + wshape)
        batches.append(GriddedObs(
            name=f"radar:{index}",
            values=np.where(wmask, wsim.mean(axis=0) - 0.2, 0.0),
            errors=np.full(wshape, 2.0), simulated=wsim, mask=wmask,
            localization=Localization(horizontal_m=9000.0,
                                      vertical_m=3000.0),
            window=window))
    return grid, prior, batches, fields


def test_the_analysis_price_covers_the_resident_solve_on_the_card():
    import cupy as cp

    from gpuwm.da.letkf import (LetkfConfig, LetkfDiagnostics, Localization,
                                analysis_device_price, analyze)
    from gpuwm.da.radar_assimilation import _execute_analysis

    grid, prior, batches, fields = _synthetic_leg()
    members = next(iter(prior.values())).shape[0]
    shape = next(iter(prior.values())).shape[1:]
    config = LetkfConfig(
        localization=Localization(horizontal_m=12000.0, vertical_m=4000.0),
        analysis_fields=fields, rtps_alpha=0.5, memory_budget_mib=64.0)
    price = analysis_device_price(
        members=members, shape=shape, fields=len(fields), prior_itemsize=8,
        batches=[(int(np.prod(b.mask.shape)), b.localization)
                 for b in batches],
        grid=grid, config=config, obs_itemsize=8)
    with _pool_high_water() as peak:
        increments, diagnostics, _stage, _unstage, storage, _attempts = \
            _execute_analysis(analyze, prior, batches, grid, config,
                              namespace=cp, device="cuda",
                              diagnostics=LetkfDiagnostics())
    assert storage == "cuda-resident"
    assert diagnostics.chunk_oom_shrinks == 0
    # The solve sized its chunk at or above the priced one: it sizes on
    # the slots a row can reach, never more than the priced sum.
    assert diagnostics.chunk_points_initial >= price.chunk_points
    assert diagnostics.chunk_points_initial * \
        diagnostics.solve_bytes_per_point <= price.scratch_bytes
    grown = peak["total"] - peak["base_used"]
    assert 0 < grown <= price.resident_bytes, (grown, price)
    for name in fields:
        assert np.all(np.isfinite(increments[name]))


def test_a_draw_runs_under_uncached_plans_and_draws_the_same_field(
        monkeypatch):
    import cupy as cp

    from gpuwm.da import perturb

    shape = (12, 96, 130)
    kwargs = dict(seed=11, name="u", dx_km=3.0, dy_km=3.0,
                  length_scale_km=30.0, vertical_scale_levels=2.0, xp=cp)
    # The one-time availability probe transforms a 2x2x2 array through
    # cupy's cache; it runs before the draw's own plans are looked at.
    assert perturb._device_fft_available(cp)
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    field, info = perturb.gaussian_random_field(shape, **kwargs)
    assert info["fft_backend"] == "cupy"
    # Neither plan stayed behind in cupy's cache, work area and all.
    assert cache.get_curr_size() == 0
    assert cache.get_curr_memsize() == 0
    # cupy's own cached plans draw the same bytes.
    monkeypatch.setattr(perturb, "_device_fft_plan",
                        lambda *_a, **_k: contextlib.nullcontext())
    cached, _info = perturb.gaussian_random_field(shape, **kwargs)
    assert cache.get_curr_size() > 0
    cache.clear()
    assert cp.asnumpy(field).tobytes() == cp.asnumpy(cached).tobytes()


def test_the_draw_stays_inside_its_price_with_the_measured_plans():
    import cupy as cp

    from gpuwm.da import perturb

    config = perturb.PerturbationConfig.from_mapping({
        "dx_km": 3.0, "dy_km": 3.0, "rim_width": 5,
        "fields": [{"name": "u", "amplitude": 1.0,
                    "length_scale_km": 30.0}]})
    mass = (24, 160, 200)
    plans = perturb.fft_plan_work_bytes(config, mass, cp)
    u_shape = (24, 160, 201)
    assert set(plans) == {u_shape}
    assert all(size > 0 for size in plans[u_shape])
    price = perturb.device_working_bytes(config, mass,
                                         plan_work_bytes=plans)
    bare = perturb.device_working_bytes(config, mass)
    assert perturb._device_fft_available(cp)
    cp.fft.config.get_plan_cache().clear()
    with _pool_high_water() as peak:
        field, _info = perturb.gaussian_random_field(
            u_shape, seed=3, name="u", dx_km=3.0, dy_km=3.0,
            length_scale_km=30.0, xp=cp)
        del field
    # Live bytes: the census counts arrays, and the pool rounds each of
    # the at most eight arrays alive at the inverse transform up to its
    # 512-byte unit.  (The pool's idle blocks are not a requirement: cupy
    # frees them and retries before it reports out of memory.)
    live = peak["used"] - peak["base_used"]
    assert live <= price + 8 * 512, (live, price, plans)
    # And the plans' work areas are what the census without them missed.
    assert live > bare, (live, bare, plans)
