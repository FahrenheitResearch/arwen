"""Classic LW executable cap, shape pricing and real allocation controls."""
from dataclasses import replace
from datetime import datetime
import gc
import sys

import numpy as np
import pytest

from gpuwm.core.rrtm_inventory import (allocation_bytes, auto_column_chunk,
    chunk_workspace_bytes, chunk_workspace_phases, effective_column_chunk)
from gpuwm.core.preflight import classic_rrtm_column_shapes
from test_rrtm_longwave import _atmosphere_and_state, _column_block, _radiation_config


@pytest.mark.parametrize('cap', [True, 0, -1, 2.5])
def test_column_cap_rejects_values_that_are_not_column_counts(cap):
    with pytest.raises(ValueError, match='positive integer'):
        effective_column_chunk(cap, 20)


def test_auto_sizing_obeys_inventory_at_both_sides_of_a_boundary():
    for layers in (3, 43, 65, 127):
        exact = chunk_workspace_bytes(7, layers)
        assert auto_column_chunk(1000, layers, exact) == 7
        assert auto_column_chunk(1000, layers, exact-1) == 6
        assert auto_column_chunk(3, layers, exact) == 3
        with pytest.raises(MemoryError, match='one column'):
            auto_column_chunk(1000, layers, chunk_workspace_bytes(1, layers)-1)


@pytest.mark.parametrize('lw,sw', [(1, 0), (1, 1), (1, 4)])
def test_actual_window_cap_prices_classic_spectrum_independently(lw, sw):
    cfg = replace(_radiation_config(lw, sw), nx=30, ny=20, nz=40)
    small = classic_rrtm_column_shapes(cfg, 5000, column_chunk=17)
    large = classic_rrtm_column_shapes(cfg, 5000, column_chunk=40)
    assert allocation_bytes(large) > allocation_bytes(small)
    assert small['classic_rrtm/transfer/gpoint/itr'] == ((17, 53, 140), 4)
    tile = replace(cfg, nx=4, ny=3)
    capped = classic_rrtm_column_shapes(tile, 1000, column_chunk=400)
    assert capped['classic_rrtm/transfer/gpoint/itr'] == ((12, 43, 140), 4)
    assert capped['classic_rrtm/packed/temperature'] == ((12, 40), 4)
    assert not classic_rrtm_column_shapes(replace(cfg, ra_lw_physics=0), column_chunk=17)


@pytest.mark.parametrize('sw', [0, 1, 4])
def test_factory_forwards_selected_cap_to_classic_leaf(monkeypatch, sw):
    from gpuwm.core import radiation_composition as factory
    from gpuwm.core import rrtm_lw
    class Leaf:
        publishes_olr = True
        glw_provenance = 'scheme'
        def __init__(self, *args, **kwargs):
            self.column_chunk = kwargs.get('column_chunk')
    monkeypatch.setattr(rrtm_lw, 'RRTMLongwaveRadiation', Leaf)
    monkeypatch.setattr(rrtm_lw, 'RRTMDudhiaRadiation', Leaf)
    monkeypatch.setitem(sys.modules, 'cupy', np)
    # The selected other leaf is irrelevant to this constructor seam.
    import gpuwm.core.rrtmg_legacy as legacy
    monkeypatch.setattr(legacy, 'RRTMGLegacyRadiation', Leaf)
    cfg = replace(_radiation_config(1, sw), ra_rrtmg_variant='legacy')
    scheme = factory.make_radiation(cfg, datetime(2021, 1, 1), np.zeros((20, 20)),
                                    np.zeros((20, 20)), p_top=1000, column_chunk=19)
    leaves = factory.radiation_adapters(scheme)
    assert any(item.column_chunk == 19 for item in leaves)


@pytest.mark.gpu
@pytest.mark.parametrize('columns,top', [(1, 1000), (17, 5000), (256, 10000)])
def test_actual_cuda_allocation_peak_is_inside_named_envelope(columns, top):
    import cupy as cp
    from gpuwm.core import rrtm_lw
    kwargs = _column_block(ncol=columns)
    nz = kwargs['t'].shape[1]
    levels = np.linspace(1013, top*0.01, nz+1, dtype=np.float32)
    kwargs['pw_mb'] = np.tile(levels[::-1], (columns, 1))
    kwargs['p_mb'] = np.tile((0.5*(levels[:-1]+levels[1:]))[::-1], (columns, 1))
    kwargs['nlayers'] = rrtm_lw.rrtm_layer_count(nz, top)
    inputs = {k: cp.asarray(v) if isinstance(v, np.ndarray) else v for k,v in kwargs.items()}
    warm = rrtm_lw.rrtm_longwave_columns(**inputs)
    cp.cuda.get_current_stream().synchronize()
    del warm
    gc.collect()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline = pool.used_bytes()
    baseline_total = pool.total_bytes()
    class Probe(cp.cuda.MemoryHook):
        name = 'rrtm_inventory_peak'
        peak = 0
        def malloc_postprocess(self, **kw):
            self.peak = max(self.peak, pool.used_bytes()-baseline)
    probe = Probe()
    with probe:
        output = rrtm_lw.rrtm_longwave_columns(**inputs)
        cp.cuda.get_current_stream().synchronize()
    assert all(bool(cp.isfinite(v).all()) for v in output.values())
    envelope = chunk_workspace_bytes(columns, kwargs['nlayers'])
    assert probe.peak <= envelope, (probe.peak, envelope)
    # Global preflight already applies its allocator headroom separately.
    from gpuwm.core.preflight import ALLOCATOR_HEADROOM
    # Live arrays from an earlier test can retain split pool blocks that
    # free_all_blocks cannot return. Compare held capacity with held capacity,
    # so that pre-existing fragmentation is not charged to this column call.
    assert pool.total_bytes()-baseline_total <= int(envelope*ALLOCATOR_HEADROOM)


@pytest.mark.gpu
def test_actual_cuda_cap_changes_no_radiation_output():
    import cupy as cp
    from gpuwm.core.rrtm_lw import RRTMLongwaveRadiation
    atmosphere, fields, state, cfg = _atmosphere_and_state(ny=4, nx=5)
    atmosphere = {k:cp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in atmosphere.items()}
    fields = {k:cp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in fields.items()}
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            setattr(state, name, cp.asarray(value))
    ny, nx = fields['tsk'].shape
    results = []
    for cap in (1, 7, ny*nx, None):
        adapter = RRTMLongwaveRadiation(datetime(2021,1,1),
            cp.full((ny,nx),39,dtype=cp.float32), cp.full((ny,nx),-87,dtype=cp.float32),
            p_top=state.p_top, column_chunk=cap)
        result = adapter(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        results.append({name:cp.asnumpy(value) for name,value in vars(result).items()
                        if isinstance(value,cp.ndarray)})
    assert results[0]
    for actual in results[1:]:
        assert actual.keys() == results[0].keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], results[0][name], err_msg=name)


def _classic_experiment(cap=17):
    from gpuwm.experiment import experiment_from_run_config, VerticalConfig
    from gpuwm.core.streaming import StreamingOptions
    cfg = replace(_radiation_config(1, 1), nx=400, ny=400, nz=40)
    exp = experiment_from_run_config(cfg, datetime(2021, 1, 1))
    return replace(exp, column_chunk=cap,
                   vertical=replace(exp.vertical, eta_levels=tuple(np.linspace(1,0,cfg.nz+1)), p_top=10000),
                   tiles=StreamingOptions(mode='on', tile_nx=128, tile_ny=128))


def test_resolved_experiment_cap_reaches_global_and_overridden_tile_options():
    from gpuwm.core import streaming
    exp = _classic_experiment()
    own = replace(exp.root, tiles=streaming.StreamingOptions(mode='off'))
    exp = replace(exp, domains=(own,), column_chunk=71)
    assert exp.tiles.radiation_context == streaming.RadiationMemoryContext(71, 10000)
    assert streaming.options_for_domain(exp.root, exp.tiles).radiation_context == streaming.RadiationMemoryContext(71, 10000)
    assert 'radiation_context' not in exp.tiles.to_json()
    assert streaming.identity_payload_entry(exp.tiles) == {}


def test_pinned_tile_prices_its_window_and_only_excess_over_reserved_radiation():
    from gpuwm.core import streaming
    from tilestream import autoplan
    from gpuwm.core.rrtm_inventory import call_workspace_bytes
    small = _classic_experiment(17)
    large = replace(small, column_chunk=15000)
    cfg = small.root.run
    decision = streaming.decide(cfg, small.tiles)
    window_columns = (decision.tile_nx+2*decision.halo)*(decision.tile_ny+2*decision.halo)
    cells = window_columns*cfg.nz
    fps = [streaming.radiation_footprint(cfg,e.tiles) for e in (small,large)]
    for fp,exp in zip(fps,(small,large)):
        assert fp.classic_call_bytes(cells) == call_workspace_bytes(
            window_columns,cfg.nz,10000,exp.column_chunk)
    base = replace(fps[0],classic_lw_context=None)
    assert fps[0].vram_bytes(cells,2) == base.vram_bytes(cells,2)
    assert fps[1].classic_call_bytes(cells)*autoplan.VRAM_SAFETY > base.radiation_transient_bytes
    extra = fps[1].classic_call_bytes(cells)*autoplan.VRAM_SAFETY-base.radiation_transient_bytes
    assert fps[1].vram_bytes(cells,2) == base.vram_bytes(cells,2)+extra
    assert fps[1].vram_bytes(cells,3) == base.vram_bytes(cells,3)+extra
    expected = int(fps[1].vram_bytes(cells,2))
    actual = streaming.streamed_envelope(cfg,large.tiles,decision=decision)
    assert actual.vram_bytes == expected
    from types import SimpleNamespace
    node = SimpleNamespace(cfg=large.root,parent=None)
    assert streaming._decision_claim_bytes(node,decision,large.tiles) == int(
        fps[1].marginal_bytes(cells,2))


def test_unknown_legacy_pressure_top_remains_deferred_not_refused():
    from gpuwm.experiment import experiment_from_run_config
    from gpuwm.core import streaming
    exp = experiment_from_run_config(_radiation_config(1,1),datetime(2021,1,1))
    assert exp.vertical.p_top == 0
    assert classic_rrtm_column_shapes(exp.root.run,exp.vertical.p_top) == {}
    footprint = streaming.radiation_footprint(exp.root.run,exp.tiles)
    assert footprint.classic_lw_context is None
    assert 'unresolved' in footprint.source


def test_public_tile_transport_preserves_budgets_and_rebinds_experiment_context():
    import json
    from gpuwm.prepared_single_domain_forecast import _streaming_options_argument
    from gpuwm.core.streaming import STREAMING_KEYS, RadiationMemoryContext
    exp = _classic_experiment(31)
    options = replace(exp.tiles, vram_budget_bytes=9_000_000_000,
                       host_budget_bytes=40_000_000_000)
    public = options.to_mapping()
    assert set(public) == STREAMING_KEYS
    parsed = _streaming_options_argument(json.dumps(public, sort_keys=True))
    assert parsed.vram_budget_bytes == options.vram_budget_bytes
    assert parsed.host_budget_bytes == options.host_budget_bytes
    assert parsed.radiation_context is None
    rebound = replace(exp, tiles=parsed)
    assert rebound.tiles.radiation_context == RadiationMemoryContext(31,10000)
