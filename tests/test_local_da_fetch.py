"""Window ownership and immutable observation recovery, with no fetches."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import pytest

from gpuwm.local_da_fetch import assigned_document, choose_document, observation_window
from gpuwm.local_da import PlanError

T = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
SCHEDULE = [T, T + timedelta(minutes=15)]


def test_radar_scan_is_used_in_only_one_cycle():
    document = {'valid_time': (T - timedelta(minutes=1)).isoformat()}
    assert assigned_document(document, T, SCHEDULE, 3600.)
    assert not assigned_document(document, SCHEDULE[1], SCHEDULE, 3600.)


def test_future_contributing_radar_is_not_hidden_by_nominal_product_time():
    document = dict(valid_time=T.isoformat(), provenance={'per_radar': [
        {'volume_valid_time': (T + timedelta(seconds=1)).isoformat()}]})
    assert not assigned_document(document, T, SCHEDULE, 900.)


def test_a_radar_still_scanning_at_the_analysis_is_not_hidden_by_its_start():
    # The volume started five minutes before the analysis and its last
    # radial was collected one second after it: the product holds the
    # future, whatever its header start says.
    document = dict(valid_time=T.isoformat(), provenance={'per_radar': [
        {'volume_valid_time': (T - timedelta(minutes=5)).isoformat(),
         'volume_end_time': (T + timedelta(seconds=1)).isoformat()}]})
    assert not assigned_document(document, T, SCHEDULE, 900.)
    # Complete a second before the analysis, it is this cycle's and no
    # later one's.
    document['provenance']['per_radar'][0]['volume_end_time'] = (
        T - timedelta(seconds=1)).isoformat()
    assert assigned_document(document, T, SCHEDULE, 900.)
    assert not assigned_document(document, SCHEDULE[1], SCHEDULE, 900.)


def test_newest_complete_product_is_selected_without_merging_alternatives():
    records = {'a': {'valid_time': (T - timedelta(minutes=2)).isoformat()},
               'b': {'valid_time': (T - timedelta(minutes=1)).isoformat()}}
    selected, report = choose_document(['a', 'b'], lambda p, **kw: records[p],
        grid=None, when=T, schedule=SCHEDULE, max_age=900.)
    assert selected == Path('b')
    assert [r['status'] for r in report] == ['superseded_product', 'selected']


def backend(tmp_path, paths=()):
    return SimpleNamespace(root=tmp_path, analysis_times=SCHEDULE,
        grid=SimpleNamespace(identity_sha256=lambda: 'a' * 64),
        plan={'review_sha256': 'b' * 64, 'selected': {'cadence_seconds': 900.},
              'request': {'obs_tables': list(paths)}, 'observations': []})


def test_empty_window_is_frozen_and_recovery_does_not_fetch(tmp_path):
    b = backend(tmp_path)
    first = observation_window(b, 0, T, {})
    assert first['rows'] == [] and first['radar'] is None
    b.plan['observations'] = [{'id': 'injected', 'route': 'not-a-route', 'status': 'ready'}]
    second = observation_window(b, 0, T, {})
    assert second == first


def test_window_binds_review_clock_and_geometry(tmp_path):
    b = backend(tmp_path)
    observation_window(b, 0, T, {})
    b.grid.identity_sha256 = lambda: 'c' * 64
    with pytest.raises(PlanError, match='different review, clock or grid'):
        observation_window(b, 0, T, {})


def test_neutral_table_bytes_cannot_change_after_partial_analysis(tmp_path):
    shared = pytest.importorskip('arwen_global.obs_table')
    table = tmp_path / 'obs.csv'
    with table.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(shared.TABLE_HEADER)
        writer.writerow(['fixture', 'station', 0, 0, 0, 80000, T.isoformat(), 'temperature_k', 270., 2.,
                         'sonde_level', '', '', '', ''])
    b = backend(tmp_path, [table])
    assert len(observation_window(b, 0, T, {})['rows']) == 1
    table.write_text(table.read_text().replace('270.0', '271.0'))
    with pytest.raises(PlanError, match='Frozen observation bytes changed'):
        observation_window(b, 0, T, {})
