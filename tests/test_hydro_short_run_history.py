"""A forecast shorter than one hour keeps default hydro (breakage 2026-10-22 14:59Z:
a 600 s run stopped at start on 'history capacity must contain at least one chosen
output interval', gpuwm/hydro/output.py)."""
import json

import pytest

from gpuwm.hydro.provider import composition


def _history(tmp_path, duration):
    manifest = tmp_path / 'routing.json'
    manifest.write_text(json.dumps({
        'parent_grid_contract': 'grid-a', 'input': {'sha256': 'f' * 64},
        'routing': {'dt_seconds': 300.0, 'initial_state_array': 'q0', 'output_reach_ids': 'node_ids'},
        'files': {'exchange_weights': {'sha256': 'e' * 64}, 'node_ids': {'shape': [7]}}}))
    footprint = {'parent_grid_contract': 'grid-a', 'time_origin': '2026-10-22T00:00:00Z',
                 'interval_seconds': 60.0, 'duration_seconds': float(duration)}
    plan = composition(manifest=manifest, footprint=footprint)
    return next(row for row in plan['stages'] if row['name'] == 'history')['options']


@pytest.mark.parametrize('duration,frames', [(600, 1), (300, 1), (3600, 1), (5400, 1), (21600, 6)])
def test_history_holds_at_least_one_frame_for_any_whole_routing_duration(tmp_path, duration, frames):
    options = _history(tmp_path, duration)
    assert options['capacity_steps'] >= options['cadence_steps'] >= 1
    assert options['capacity_steps'] // options['cadence_steps'] == frames


def test_runs_of_an_hour_or_more_keep_hourly_frames(tmp_path):
    assert _history(tmp_path, 7200)['cadence_steps'] == 12
