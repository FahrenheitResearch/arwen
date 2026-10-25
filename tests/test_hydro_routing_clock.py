"""Clock orchestration checks; resident numerical proof is in the CUDA test."""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from gpuwm.hydro.driver import ResidentHydroDriver, Stage
from gpuwm.hydro import door, provider


def intervals(dt, duration=1001.0):
    result, elapsed = [], 0.0
    while elapsed < duration:
        step = dt[len(result) % len(dt)] if isinstance(dt, list) else dt
        step = min(step, duration - elapsed)
        result.append(step)
        elapsed += step
    return result


def clock_driver(dt, accumulated, duration, components):
    return ResidentHydroDriver(device_id=0, interval_seconds=dt, identity={},
        after=[Stage(name, component, component.advance)
               for name, component in components.items()],
        routing_clock={'period_seconds': 300., 'duration_seconds': duration,
                       'accumulated_parent': accumulated})


@pytest.mark.parametrize('dt', [11.25, 128.571, [15., 18., 24., 401.]])
@pytest.mark.parametrize('accumulated', [True, False])
def test_clock_carries_time_and_water_and_publishes_tail(dt, accumulated):
    water = SimpleNamespace(depth=0., produced=0., bucket=0., received=[], samples=[])
    class Component:
        arrays, restart_identity = {}, {}
        def __init__(self, callback):
            self.advance = callback
    def exchange(span, **kw):
        water.samples.append(span)
        if accumulated:
            water.depth = water.produced - getattr(water, 'previous', 0.)
            water.previous = water.produced
    def runoff(span, fraction, reset, **kw):
        water.bucket = (0. if reset else water.bucket) + water.depth * fraction
    def routing(span, **kw):
        assert span == 300.
        water.received.append(water.bucket)
    frames = []
    components = {name: Component(callback) for name, callback in
                  [('exchange', exchange), ('runoff', runoff), ('routing', routing),
                   ('history', lambda span, **kw: frames.append(kw))]}
    steps = intervals(dt)
    driver = clock_driver(steps[0], accumulated, sum(steps), components)
    driver.allow_variable_interval()
    for i, step in enumerate(steps):
        water.depth = step * (i + 1) * .003
        water.produced += water.depth
        driver.before_lsm()
        driver.advance(step)
    assert sum(water.received) == pytest.approx(water.produced, rel=2e-15)
    assert [f['time_seconds'] for f in frames] == pytest.approx([300., 600., 900., 1001.], abs=1e-10)
    assert frames[-1]['final']
    assert len(water.samples) <= 4 if accumulated else len(water.samples) == len(steps)
    saved = copy.deepcopy(driver.checkpoint_metadata())
    fresh = clock_driver(steps[0], accumulated, sum(steps), components)
    fresh.allow_variable_interval()
    fresh.validate_checkpoint(saved, {})
    fresh.restore_metadata(saved)
    assert fresh.checkpoint_metadata() == saved


@pytest.mark.parametrize('key,value', [('interval_seconds', 0), ('duration_seconds', -1),
                                       ('interval_seconds', float('inf')), ('duration_seconds', None)])
def test_invalid_clock_gap_precedes_any_availability_fetch(key, value):
    row = json.loads(door.TABLE.read_text())['providers'][0]
    geometry = door.geometry_from_points([40.], [-96.], center_lon=-96., parent_lsm=2,
                                          grid_key='test', interval_seconds=11.25,
                                          duration_seconds=1001.)
    geometry[key] = value
    verdict = door.evaluate(row, geometry, probe_row=lambda row: pytest.fail('tool/source fetch'))
    assert any('coupling_interval_invalid' in why for why in verdict['reasons'])


@pytest.mark.parametrize('dt', [0., float('nan'), 37.])
def test_invalid_routing_or_hourly_spinup_clock_precedes_fetch(dt):
    row = next(r for r in json.loads(door.TABLE.read_text())['providers'] if r.get('preparation'))
    row['preparation']['routing_interval_seconds'] = dt
    geometry = door.geometry_from_points([40.], [-96.], center_lon=-96., parent_lsm=2,
        grid_key='test', interval_seconds=128.571, duration_seconds=101.)
    verdict = door.evaluate(row, geometry, probe_row=lambda row: pytest.fail('tool/source fetch'))
    assert any('coupling_' in why for why in verdict['reasons'])


@pytest.mark.parametrize('dt', [11.25, 128.571, 401.])
def test_composition_has_no_whole_parent_or_duration_requirement(tmp_path, dt):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'parent_grid_contract': 'parent',
        'routing': {'dt_seconds': 300., 'initial_state_array': 'state'},
        'input': {'sha256': 'fabric'},
        'files': {'exchange_weights': {'sha256': 'weights'}, 'node_ids': {'shape': [2]}}}))
    result = provider.composition(manifest=manifest,
        footprint={'parent_grid_contract': 'parent', 'interval_seconds': dt,
                   'duration_seconds': 101., 'time_origin': '2024-04-26T17:00:00Z'},
        realization='parent-noah-routed')
    assert result['routing_clock']['accumulated_parent'] is True
    history = result['stages'][-1]['options']
    assert history['capacity_steps'] == 1 and history['routing_clock']
    # Exercise the default provider's actual TOML/process hand-off and table
    # expansion: a correct direct driver is insufficient if the door drops it.
    from gpuwm.hydro.runtime import read_settings
    settings = tmp_path / 'hydro-door.toml'
    result.update(enabled=True, realization='parent-noah-routed',
                  settings_bindings={'fabric_manifest': str(manifest)})
    settings.write_text(door.to_toml(result))
    resolved = read_settings(settings)
    assert resolved['routing_clock'] == result['routing_clock']
    assert resolved['stages'] == result['stages']
