"""Actual CUDA equality and reservation controls for bounded reconstruction."""
import numpy as np
import pytest
from gpuwm.core.nest_interp import blend_terrain, register_nest, sint


@pytest.mark.gpu
def test_gpu_window_sint_and_terrain_are_bit_exact():
    import cupy as cp
    rng = np.random.default_rng(91)
    for ratio in (1, 2, 3, 4, 5):
        for stagger in ("", "x", "y"):
            reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                                j_parent_start=8, child_nx=17, child_ny=19,
                                parent_nx=40, parent_ny=42, stagger=stagger)
            source = cp.asarray(rng.normal(size=(4, reg.nyp, reg.nxp)).astype(np.float32))
            full = sint(source, reg)
            window = (slice(2, 14), slice(3, 16))
            cropped = sint(source, reg, window=window)
            np.testing.assert_array_equal(cp.asnumpy(cropped).view(np.uint32),
                                          cp.asnumpy(full[(...,)+window]).view(np.uint32))
    coarse = cp.asarray(rng.normal(size=(5, 29, 31)).astype(np.float32))
    fine = cp.asarray(rng.normal(size=(5, 29, 31)).astype(np.float32))
    full = fine.copy()
    blend_terrain(coarse, full)
    for j0, j1 in ((0, 7), (7, 21), (21, 29)):
        window = (..., slice(j0, j1), slice(3, 19))
        chunk = fine[window].copy()
        blend_terrain(coarse[window].copy(), chunk,
                      domain_shape=(29, 31), origin=(j0, 3))
        np.testing.assert_array_equal(cp.asnumpy(chunk).view(np.uint32),
                                      cp.asnumpy(full[window]).view(np.uint32))


def _real_window_case(monkeypatch, target_mp=10, size=48, complete_statics=False,
                      return_model=False, forecast_steps=2):
    from dataclasses import replace
    import cupy as cp
    from tilestream import test_moving_nest as moving, test_nest_executor as executor
    from gpuwm.experiment import VerticalConfig
    from gpuwm.ingest.relocation_init import real_relocation_initializer
    monkeypatch.setattr(executor, "I_PARENT_START", 41)
    monkeypatch.setattr(executor, "J_PARENT_START", 41)
    pcfg = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6., moist_cq=True)
    ccfg = moving.child_cfg(pcfg, nx=size, ny=size)
    boundaries = moving.domain_boundaries(pcfg, seconds=60.) if return_model else None
    parent, geo = moving.build_parent(pcfg, warmup=0, boundaries=boundaries)
    exp = executor.experiment(pcfg, ccfg, steps=forecast_steps)
    model = executor.assemble(exp, parent, geo.grid, feedback=0)
    child = model.node(2)
    child_dc = replace(child.cfg, run=replace(
        child.cfg.run, mp_physics=target_mp,
        nest_microphysics_transition=("same-scheme-only" if target_mp == 10
                                      else "mp-edge-mass-diagnosed-v1")))
    vertical = VerticalConfig(
        eta_levels=tuple(float(v) for v in cp.asnumpy(parent.znw)),
        p_top=float(parent.p_top), hybrid_opt=pcfg.hybrid_opt, etac=pcfg.etac)
    def statics(grid, dc):
        lat, lon = grid.latlon_mass()
        shape = (dc.run.ny, dc.run.nx)
        result = {"HGT_M": 100. + 20.*np.sin(lat)*np.cos(lon),
                  "LANDMASK": np.ones(shape)}
        if complete_statics:
            result.update(LU_INDEX=np.ones(shape, np.int32), SCT_DOM=np.full(shape, 6, np.int32),
                          GREENFRAC=np.full((12,)+shape, .5), LAI12M=np.full((12,)+shape, 2.),
                          TMN=np.full(shape, 285.), SNOALB=np.full(shape, .5))
        return result
    initialize = real_relocation_initializer(
        vertical=vertical, child_config=child_dc,
        reference_grid=child.grid, reference_i_parent_start=child.cfg.i_parent_start,
        reference_j_parent_start=child.cfg.j_parent_start, statics_builder=statics)
    moved = replace(child_dc, i_parent_start=child.cfg.i_parent_start+1)
    return ((initialize, moved, child.parent, model) if return_model
            else (initialize, moved, child.parent))


@pytest.mark.gpu
@pytest.mark.parametrize("target_mp", [10, 8, 18, 50])
@pytest.mark.parametrize("host_parent", [False, True])
def test_gpu_real_initializer_windows_match_full(monkeypatch, target_mp, host_parent):
    import cupy as cp
    initialize, moved, parent = _real_window_case(monkeypatch, target_mp)
    full = initialize(moved, parent)
    if host_parent:
        from types import SimpleNamespace
        from gpuwm.core.streamed_state import CanonicalStoreState
        from gpuwm.core.streaming import streamed_store_inventory
        from tilestream.driver import geography_inventory
        inventory = streamed_store_inventory()(parent.state)
        geography = geography_inventory(parent.state)
        facade = CanonicalStoreState(parent.state, parent.cfg.run,
            store={key: cp.asnumpy(value) for key, value in inventory.items()},
            geography={key: cp.asnumpy(value) for key, value in geography.items()},
            scalars={}, inventory=inventory, geography_inventory=geography)
        parent = SimpleNamespace(state=facade, cfg=parent.cfg, grid=parent.grid)
    count = 0
    for j0, j1 in ((0, 13), (13, 31), (31, 48)):
        window = (slice(j0, j1), slice(3, 43))
        slab = initialize(moved, parent, window=window)
        for name, value in vars(full.state).items():
            if not isinstance(value, cp.ndarray):
                continue
            target = getattr(slab.state, name)
            if value.ndim >= 2 and value.shape[-2] in (48, 49) and value.shape[-1] in (48, 49):
                expected = value[..., j0:j1+(value.shape[-2]-48),
                                 3:43+(value.shape[-1]-48)]
            else:
                expected = value
            np.testing.assert_array_equal(cp.asnumpy(target), cp.asnumpy(expected),
                                          err_msg=f"{name} at rows {j0}:{j1}")
            count += 1
        del slab
    assert count > 100


@pytest.mark.gpu
def test_gpu_reserved_reconstruction_store_matches_full(monkeypatch):
    import gc
    import cupy as cp
    from gpuwm.core.streaming import prime_lazy_carriers, streamed_store_inventory
    from gpuwm.ingest.reconstruction_store import store_from_reconstruction
    from tilestream.driver import geography_inventory
    initialize, moved, parent = _real_window_case(monkeypatch, size=96)
    full = initialize(moved, parent)
    prime_lazy_carriers(full.state, moved.run)
    expected = [{k: cp.asnumpy(v) for k, v in fn(full.state).items()}
                for fn in (streamed_store_inventory(), geography_inventory)]
    resident_bytes = sum(v.nbytes for v in vars(full.state).values()
                         if isinstance(v, cp.ndarray))
    del full
    gc.collect()
    budget = 32*1024*1024
    assert resident_bytes > budget
    built = store_from_reconstruction(
        initialize, moved, parent, rows_per_slab=12,
        device_budget_bytes=budget, host_budget_bytes=256*1024*1024)
    for actual, reference in zip((built.store, built.geography), expected):
        assert set(actual) == set(reference)
        for name in actual:
            np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)
    assert built.receipt["device_reservation_bytes"] == budget
    assert built.receipt["retained_pool_bytes"] <= budget
    assert built.template.p.shape[1] == 12


@pytest.mark.gpu
def test_gpu_reconstruction_refuses_to_exceed_reserved_budget(monkeypatch):
    import cupy as cp
    from gpuwm.ingest.reconstruction_store import store_from_reconstruction
    initialize, moved, parent = _real_window_case(monkeypatch)
    original = cp.asnumpy(parent.state.thp)
    with pytest.raises(cp.cuda.memory.OutOfMemoryError):
        store_from_reconstruction(
            initialize, moved, parent, rows_per_slab=12,
            device_budget_bytes=512, host_budget_bytes=256*1024*1024)
    np.testing.assert_array_equal(cp.asnumpy(parent.state.thp), original)
