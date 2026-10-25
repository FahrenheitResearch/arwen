"""Real exchange, transfer, MC and history on CUDA; manufactured runoff, not basin skill."""
import importlib.util
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip('cupy')

from gpuwm.hydro import driver, output, runoff, troute_mc
from gpuwm.hydro.noahowp import coupling
from gpuwm.numerics import mode
from test_hydro_routing_clock import intervals
import test_nextgen_mc_initial_cuda as mc_fixture


def baseline(name, relative):
    root = Path(os.environ['HYDRO_CLOCK_BASELINE'])
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(dt, duration, accumulated, legacy=False):
    modules = (driver, output, runoff, coupling)
    if legacy:
        modules = tuple(baseline('gpuwm.hydro._clock_baseline_' + name, path) for name, path in
                        [('driver', 'driver.py'), ('output', 'output.py'), ('runoff', 'runoff.py'),
                         ('coupling', 'noahowp/coupling.py')])
    dm, om, rm, cm = modules
    surface, sub = cp.zeros((1, 2), cp.float32), cp.zeros((1, 2), cp.float32)
    zero = cp.zeros_like(surface)
    soil = cp.full((4, 1, 2), .3, cp.float32)
    args = dict(offsets=cp.asarray([0, 1, 2], cp.int64), indices=cp.asarray([0, 1], cp.int32),
                weights=cp.ones(2, cp.float32), smois=soil, sh2o=soil,
                dt_seconds=300. if accumulated else dt, weights_sha256='test', fabric_sha256='test',
                numerics=mode())
    if accumulated:
        exchange = cm.PreparedAccumulatedExchange(surface_total=surface, subsurface_total=sub,
                                                  qfx=zero, **args)
    else:
        exchange = cm.PreparedCatchmentExchange(runsrf=surface, runsub=sub, soldrain=zero,
                                                sfcheadrt=zero, ecan=zero, etran=zero, edir=zero, **args)
    transfer_args = dict(depth=exchange.arrays['runoff_depth_m'], area=cp.asarray([8., 13.], cp.float64),
        catchment_to_node=cp.asarray([0, 1], cp.int32), node_count=2,
        period_seconds=300. if accumulated else dt, area_units='m2', routing_period_seconds=300., identity={})
    if not legacy:
        transfer_args['routing_clock'] = True
    transfer = rm.PreparedRunoffRouting(**transfer_args)
    network = mc_fixture.PreparedInitialTests().packet()
    router = troute_mc.prepare(settings={'interval_seconds': 300., 'time_origin': '2024-04-26T17:00:00Z'},
        options={'network_component': 'network', 'lateral_component': 'runoff', 'dt_seconds': 300.},
        state=None, physics=None, device_id=0, mode='standalone',
        components={'network': network, 'runoff': transfer})
    history_options = dict(capacity_steps=math.ceil(duration / 300), cadence_steps=12,
                           origin='2024-04-26T17:00:00Z', types=['flowpath'] * 2)
    if not legacy:
        history_options['routing_clock'] = True
    history = om.History(settings={'interval_seconds': 300.}, options=history_options,
                          source=router, device_id=0)
    received, produced = cp.zeros(2, cp.float64), cp.zeros(2, cp.float64)
    native_received = cp.zeros(2, cp.float64)
    def route(span, *, stream=None):
        router.launch(span, stream=stream)
        cp.add(received, transfer.arrays['routing_volume_m3'], out=received)
        # MC consumes the native f32 lateral discharge, not the f64 audit
        # volume. Include its conversion rounding in the delivered-water proof.
        cp.add(native_received, transfer.arrays['routing_lateral'].astype(cp.float64) * 300., out=native_received)
    components = {'exchange': exchange, 'runoff': transfer, 'routing': router, 'history': history}
    stages = []
    for name, component in components.items():
        callback = route if name == 'routing' else component.advance
        cadence = 25 if name in ('routing', 'history') or accumulated else 1
        stages.append(dm.Stage(name, component, callback, cadence_steps=cadence if legacy else 1,
                              interval_seconds=300. if legacy and cadence == 25 else None))
    kw = {} if legacy else {'routing_clock': {'period_seconds': 300., 'duration_seconds': duration,
                                             'accumulated_parent': accumulated}}
    resident = dm.ResidentHydroDriver(device_id=0, interval_seconds=dt, identity={}, after=stages, **kw)
    resident._proof_output = om
    resident._proof_native_received = native_received
    def parent(step, i):
        # Exact binary parent depths, with time-varying runoff and nonzero subsurface.
        value = ((i % 7) + 1) / 8.
        if accumulated:
            surface[:] += value
            sub[:] += value / 4.
        else:
            surface.fill(value)
            sub.fill(value / 4.)
        produced[:] += cp.asarray([8., 13.]) * (value * 1.25 / 1000.)
    return resident, parent, received, produced


@pytest.mark.parametrize('dt', [11.25, 128.571, [15., 18., 24., 401.]])
@pytest.mark.parametrize('accumulated', [True, False])
def test_real_cuda_mass_tail_and_restart(dt, accumulated, tmp_path):
    steps = intervals(dt) if isinstance(dt, list) else [dt] * (89 if dt == 11.25 else 8)
    duration = sum(steps) if isinstance(dt, list) else dt * len(steps)
    resident, parent, received, produced = build(steps[0], duration, accumulated)
    if isinstance(dt, list):
        resident.allow_variable_interval()
    for i, step in enumerate(steps):
        resident.before_lsm()
        parent(step, i)
        resident.advance(step)
        if i == len(steps) // 2:
            saved = resident.checkpoint_metadata()
            incoming = {key: value.copy() for key, value in resident.arrays.items()}
            fresh, fresh_parent, fresh_received, fresh_produced = build(steps[0], duration, accumulated)
            if isinstance(dt, list):
                fresh.allow_variable_interval()
            fresh.validate_checkpoint(saved, incoming)
            for key, value in fresh.arrays.items():
                cp.copyto(value, incoming[key])
            fresh.restore_metadata(saved)
            cp.copyto(fresh_received, received)
            cp.copyto(fresh_produced, produced)
            cp.copyto(fresh._proof_native_received, resident._proof_native_received)
            resident, parent, received, produced = fresh, fresh_parent, fresh_received, fresh_produced
    resident.check_health()
    # Native parent/exchange depths are f32, transfer and time splitting f64.
    cp.testing.assert_allclose(received, produced, rtol=2e-7, atol=1e-12)
    cp.testing.assert_allclose(resident._proof_native_received, produced, rtol=2e-7, atol=1e-12)
    history = resident.components['history']
    assert output._history_steps(history) == 4
    assert history.final_time_seconds == duration
    assert resident.stage_launches['routing'] == 4
    block = dict(dt=dt, accumulated=accumulated, parent_m3=produced.get().tolist(),
                 routed_m3=received.get().tolist(),
                 relative_error=cp.max(cp.abs(received - produced) / produced).item(),
                 native_routing_received_m3=resident._proof_native_received.get().tolist(),
                 native_relative_error=cp.max(cp.abs(resident._proof_native_received - produced) / produced).item(),
                 launches=resident.stage_launches)
    print('MASS_PROOF', json.dumps(block))
    (tmp_path / 'mass.json').write_text(json.dumps(block))
    writer = output.prepare(settings={}, driver=resident, outdir=tmp_path / 'series')
    receipt = writer.finish(resident, {})
    request = json.loads((Path(receipt['output_directory']) / 'history/renderer-request.json').read_text())
    final = datetime.fromisoformat(request['frames'][-1]['valid_time'].replace('Z', '+00:00'))
    assert (final - datetime.fromisoformat('2024-04-26T17:00:00+00:00')).total_seconds() == pytest.approx(duration)
    if accumulated and dt == 11.25:
        # The partial end must be consumable by the installed Rust renderer,
        # including its fractional-second timestamp, not merely valid JSON.
        from gpuwm.nextgen_products import run_nextgen_products
        request_path = Path(receipt['output_directory']) / 'history/renderer-request.json'
        request['out_dir'] = str(tmp_path / 'products')
        # Select manufactured catchments for model-only native hydrographs.
        request['gauges'] = request['node_ids']
        request_path.write_text(json.dumps(request))
        render_receipt = run_nextgen_products(request_path)
        assert render_receipt.is_file()
        images = [p for p in (tmp_path / 'products').rglob('*') if p.suffix in ('.svg', '.png')]
        print('PARTIAL_RENDER', render_receipt.read_text(), [str(p.relative_to(tmp_path)) for p in images])
        assert images
        if target := os.environ.get('HYDRO_CLOCK_RENDER_EVIDENCE'):
            import shutil
            shutil.copytree(tmp_path / 'products', Path(target) / 'fractional-tail-products', dirs_exist_ok=True)


@pytest.mark.parametrize('accumulated', [True, False])
def test_dt12_is_bit_identical_to_base_exchange_transfer_mc_history(accumulated, tmp_path):
    current, parent, _, _ = build(12., 3600., accumulated)
    old, old_parent, _, _ = build(12., 3600., accumulated, legacy=True)
    for i in range(300):
        for resident, produce in [(current, parent), (old, old_parent)]:
            resident.before_lsm()
            produce(12., i)
            resident.advance(12.)
    for key, array in old.arrays.items():
        assert current.arrays[key].get().tobytes() == array.get().tobytes(), key
    for label, resident in [('old', old), ('current', current)]:
        resident._proof_output.prepare(settings={}, driver=resident, outdir=tmp_path / label).finish(resident, {})
    assert (tmp_path / 'old/history/streamflow.csv').read_bytes() == (tmp_path / 'current/history/streamflow.csv').read_bytes()
    for path in (tmp_path / 'old/history/renderer-history').iterdir():
        assert path.read_bytes() == (tmp_path / 'current/history/renderer-history' / path.name).read_bytes()
    print('BIT_IDENTITY', accumulated, len(old.arrays), 'resident arrays including MC and history', mode())


@pytest.mark.parametrize('accumulated', [True, False])
@pytest.mark.parametrize('steps', [7, 25, 31])
def test_stopped_parent_flushes_actual_water_and_capture_time(accumulated, steps, tmp_path):
    resident, parent, received, produced = build(12., 1001., accumulated)
    for i in range(steps):
        resident.before_lsm()
        parent(12., i)
        resident.advance(12.)
    resident.finish_clock()
    resident.finish_clock()  # Output retries do not deliver the same water twice.
    cp.testing.assert_allclose(received, produced, rtol=2e-7)
    history = resident.components['history']
    assert history.final_time_seconds == steps * 12.
    assert output._history_steps(history) == math.ceil(steps * 12. / 300.)
    writer = output.prepare(settings={}, driver=resident, outdir=tmp_path / 'series')
    receipt = writer.finish(resident, {})
    assert receipt['products']
