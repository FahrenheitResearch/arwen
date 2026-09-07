"""Price the actual per-grid CFL ring once, including streamed tile buffers."""
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.cfl_inventory import WRF_CFL_BYTES, WRF_CFL_SHAPE
from gpuwm.core import preflight as pf
from gpuwm.experiment import experiment_from_run_config
from tilestream import autoplan


def _config(**kwargs):
    return RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                     dt=1., run_seconds=10., **kwargs)


def _experiment(cfg):
    return experiment_from_run_config(cfg, datetime(2026, 1, 1))


def test_ring_is_persistent_and_not_shared_scratch_or_state(monkeypatch):
    monkeypatch.delenv('GPUWM_WRF_CFL_PROBE', raising=False)
    domain = _experiment(_config()).root
    off = pf.estimate_domain(domain)
    on = pf.estimate_domain(domain, cfl_recording=True)
    ring, = [item for item in on.items if item.category == 'diagnostic']
    assert (ring.shape, ring.dtype, ring.nbytes) == (WRF_CFL_SHAPE, 'uint32', 4718592)
    assert on.resident_bytes - off.resident_bytes == WRF_CFL_BYTES
    assert on.transient_bytes == off.transient_bytes
    assert on.arena_scratch_bytes == off.arena_scratch_bytes
    assert on.rebuilt_state_bytes == off.rebuilt_state_bytes


def test_experiment_prices_every_grid_when_root_enables_recording(monkeypatch):
    monkeypatch.delenv('GPUWM_WRF_CFL_PROBE', raising=False)
    exp = _experiment(_config())
    # Construct a pricing-only tree with deliberately different inherited
    # flags: the actual model driver enables recording for the whole tree.
    child = replace(exp.root, grid_id=2, run=replace(exp.root.run, grid_id=2))
    exp = replace(exp, domains=(exp.root, child))
    on = replace(exp, domains=(replace(exp.root, run=replace(
        exp.root.run, use_adaptive_time_step=True)), child))
    before, after = pf.estimate_experiment(exp), pf.estimate_experiment(on)
    assert after.resident_bytes - before.resident_bytes == 2 * WRF_CFL_BYTES
    assert all(d.category_bytes('diagnostic') == WRF_CFL_BYTES for d in after.domains)
    assert after.scratch_arena_bytes == before.scratch_arena_bytes


@pytest.mark.parametrize('switch,enabled', [('0', False), ('false', False),
    ('off', False), ('no', False), ('', False), ('1', True), ('true', True)])
def test_explicit_probe_has_same_activation_in_both_estimators(monkeypatch, switch, enabled):
    monkeypatch.setenv('GPUWM_WRF_CFL_PROBE', switch)
    cfg = _config()
    assert pf.estimate_domain(_experiment(cfg).root).category_bytes('diagnostic') == enabled * WRF_CFL_BYTES
    assert autoplan.footprint_for(cfg).domain_fixed_bytes == enabled * WRF_CFL_BYTES


def test_tile_ring_is_once_per_grid_never_per_buffer(monkeypatch):
    monkeypatch.delenv('GPUWM_WRF_CFL_PROBE', raising=False)
    cfg = _config()
    off = autoplan.footprint_for(cfg)
    on = autoplan.footprint_for(replace(cfg, use_adaptive_time_step=True))
    assert off is autoplan.FOOTPRINTS[autoplan.rung_of(cfg)]
    for buffers in (1, 2, 4):
        assert on.vram_bytes(500000, buffers) - off.vram_bytes(500000, buffers) == pytest.approx(WRF_CFL_BYTES * autoplan.VRAM_SAFETY)
        assert on.marginal_bytes(500000, buffers) - off.marginal_bytes(500000, buffers) == pytest.approx(WRF_CFL_BYTES * autoplan.VRAM_SAFETY)
        assert on.buffer_bytes(500000) == off.buffer_bytes(500000)
        budget = 2 * 1024**3
        cells = autoplan._max_window_cells(on, buffers, budget)
        assert on.vram_bytes(cells, buffers) <= budget
        assert on.vram_bytes(cells + 1, buffers) > budget


@pytest.mark.gpu
def test_actual_ring_allocation_is_reused_after_state_replacement(monkeypatch):
    cp = pytest.importorskip('cupy')
    from gpuwm.core import dycore
    monkeypatch.delenv('GPUWM_WRF_CFL_PROBE', raising=False)
    dycore.reset_wrf_cfl_recording()
    cfg = _config()
    z = lambda shape: cp.zeros(shape, dtype=cp.float32)
    state = SimpleNamespace(mup=z((6, 8)), mub2d=cp.ones((6, 8), dtype=cp.float32),
        c1f=cp.ones(5, dtype=cp.float32), c2f=z(5), rdnw=cp.ones(4, dtype=cp.float32),
        u=z((4, 6, 9)), v=z((4, 7, 8)),
        msfu=cp.ones((6, 9), dtype=cp.float32), msfv=cp.ones((7, 8), dtype=cp.float32))
    ww = z((5, 6, 8))
    try:
        dycore.record_wrf_vertical_cfl(state, cfg, ww)
        assert not dycore._WRF_CFL_STAT
        dycore.enable_wrf_cfl_recording()
        dycore.record_wrf_vertical_cfl(state, cfg, ww)
        ring = dycore._WRF_CFL_STAT[cfg.grid_id]
        assert (ring.shape, ring.dtype, ring.nbytes) == (WRF_CFL_SHAPE, np.dtype('uint32'), WRF_CFL_BYTES)
        for _ in range(5):
            dycore.record_wrf_vertical_cfl(SimpleNamespace(**vars(state)), cfg, ww)
        assert dycore._WRF_CFL_STAT[cfg.grid_id] is ring
        dycore.record_wrf_vertical_cfl(state, replace(cfg, grid_id=cfg.grid_id + 1), ww)
        cp.cuda.get_current_stream().synchronize()
        assert sum(a.nbytes for a in dycore._WRF_CFL_STAT.values()) == 2 * WRF_CFL_BYTES
    finally:
        dycore.reset_wrf_cfl_recording()
    assert not dycore._WRF_CFL_STAT
