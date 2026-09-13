"""Neutral-table adapter tests over the shared observation table.

No network, device or weather integration.  Every test here needs the
pinned global observation-table package for its row type and its shared
operators, so the dependency is a top-of-file ``skipif`` rather than an
``importorskip``: the items are still COLLECTED, so
``tools/battery/no_silent_deselection.py`` and the census floor keep
seeing this file, and an installation without the package reports skips
with a reason instead of a silently empty leg.  The adapter units that
need none of it are in ``tests/test_local_da_observations.py``.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json
import numpy as np
import pytest

from gpuwm.da.obs_point import point_batches, read_tables

try:
    from arwen_global import obs_table as shared
except ImportError:                                  # pragma: no cover
    shared = None

pytestmark = pytest.mark.skipif(
    shared is None,
    reason="the shared observation-table package (arwen_global.obs_table) "
           "is not installed in this environment; it owns the row type and "
           "the shared operators these tests adapt")

ObsRow = getattr(shared, 'ObsRow', None)
T = datetime(2026, 9, 10, 12, 15, tzinfo=timezone.utc)


def row(**kw):
    args = dict(source='fixture', station_id='s1', latitude_deg=1., longitude_deg=1.,
                elevation_m=0., level_pa=80000., valid_time=T - timedelta(minutes=2),
                variable='temperature_k', value=281., error=2., measurement='sonde_level',
                nominal_time=None, published_time=None, received_time=None, revision='')
    args.update(kw)
    return ObsRow(**args)


def grid():
    return SimpleNamespace(nz=2, ny=3, nx=3, terrain_m=np.zeros((3, 3)),
        z_w=np.broadcast_to(np.array([0., 1000., 3000.])[:, None, None], (3, 3, 3)),
        mass_index=lambda lat, lon: (np.asarray(lon), np.asarray(lat)))


def columns():
    result = []
    for dt, dp in ((-1., 0.), (1., 1000.)):
        full = lambda data: np.broadcast_to(np.array(data)[:, None, None], (2, 3, 3)).copy()
        result.append(dict(p=full([90000. + dp, 70000. + dp]),
                           t=full([290. + dt, 270. + dt]), q=full([.008, .004]),
                           u=full([10., 20.]), v=full([0., 5.])))
    return result


def adapt(rows, **kw):
    defaults = dict(states=[{}, {}], setup={}, grid=grid(), analysis_time=T,
                    analysis_times=[T, T + timedelta(minutes=15)], columns=columns())
    defaults.update(kw)
    return point_batches(rows, **defaults)


def test_member_specific_pressure_interpolation():
    batches, receipt = adapt([row()])
    assert receipt['counts']['accepted'] == 1
    batch = batches[0]
    index = tuple(np.argwhere(batch.mask)[0])
    # Analytic two-level log-pressure interpolation, independent of the
    # production interpolator. Each member has its own pressure coordinate.
    expected = [290. + dt - 20. * np.log(80000. / (90000. + dp)) /
                np.log((70000. + dp) / (90000. + dp)) for dt, dp in ((-1., 0.), (1., 1000.))]
    np.testing.assert_allclose(batch.simulated[(slice(None), *index)], expected)
    assert not receipt['vertical_extrapolation']


@pytest.mark.parametrize('changes,key', [
    ({'received_time': T + timedelta(seconds=1)}, 'received_after_cutoff'),
    ({'published_time': T + timedelta(seconds=1)}, 'published_after_cutoff'),
    ({'valid_time': T + timedelta(seconds=1)}, 'future_observation'),
    ({'valid_time': T - timedelta(hours=1)}, 'outside_assigned_window'),
    ({'longitude_deg': 10.}, 'outside_domain'),
    ({'level_pa': 40000.}, 'outside_vertical_column_or_invalid_operator'),
    ({'level_pa': 100000.}, 'outside_vertical_column_or_invalid_operator'),
    ({'variable': 'brightness_temperature_k'}, 'requires_column_radiance_operator'),
    ({'value': 400.}, 'gross_physical_range'),
    ({'value': 310.}, 'background_check'),
    ({'error': 0.}, 'invalid_value_or_error'),
])
def test_counted_rejections(changes, key):
    batches, receipt = adapt([row(**changes)])
    assert not batches
    assert receipt['counts'][key] == 1


def test_duplicate_and_latest_eligible_revision():
    r = row(received_time=T - timedelta(minutes=2), revision='a')
    corrected = replace(r, value=282., received_time=T - timedelta(minutes=1), revision='b')
    late = replace(r, value=283., received_time=T + timedelta(minutes=1), revision='c')
    batches, receipt = adapt([r, corrected, late, r])
    assert batches[0].values[batches[0].mask].tolist() == [282.]
    assert receipt['counts']['duplicate_or_superseded'] == 2
    assert receipt['counts']['received_after_cutoff'] == 1


def test_observation_not_assigned_to_two_cycles():
    _, first = adapt([row()])
    second, receipt = adapt([row()], analysis_time=T + timedelta(minutes=15), max_age_seconds=3600.)
    assert not second and receipt['counts']['outside_assigned_window'] == 1
    second, receipt = adapt([row()], used_identities=first['accepted_identities'])
    assert not second and receipt['counts']['already_assimilated'] == 1


def test_late_arrival_assigned_once_to_next_eligible_boundary():
    r = row(received_time=T + timedelta(seconds=1))
    batches, receipt = adapt([r], analysis_time=T + timedelta(minutes=15), max_age_seconds=1800.)
    assert batches and receipt['counts']['accepted'] == 1


def test_missing_arrival_is_not_claimed_causal():
    _, receipt = adapt([row()])
    assert receipt['counts']['latency_unverified'] == 1


def test_surface_uses_real_screen_diagnostic_not_lowest_layer():
    r = row(level_pa=None, measurement='screen_temperature_2m', value=281.)
    t2 = np.full((2, 3, 3), 280.)
    t2[1] += 2
    batches, receipt = adapt([r], surface={'t2': t2})
    np.testing.assert_array_equal(batches[0].simulated[:, 0, 1, 1], [280., 282.])
    missing, receipt = adapt([r])
    assert not missing and receipt['counts']['missing_surface_operator_or_diagnostic'] == 1


def test_station_pressure_and_sea_level_pressure_are_not_lowest_level_pressure():
    for label in ('station_pressure', 'station_pressure_from_altimeter', 'sea_level_pressure'):
        batches, receipt = adapt([row(variable='surface_pressure_pa', level_pa=None, measurement=label, value=100000., error=100.)])
        assert not batches
        assert receipt['counts']['missing_surface_operator_or_diagnostic'] == 1


def test_refractivity_uses_tangent_height_and_shared_operator():
    from arwen_global.obs_operators import refractivity_n
    col = columns()[0]
    height = 1000.
    z = np.array([500., 2000.])
    p = np.exp(np.interp(height, z, np.log(col['p'][:, 1, 1])))
    t = np.interp(height, z, col['t'][:, 1, 1])
    q = np.interp(height, z, col['q'][:, 1, 1])
    expected = float(refractivity_n(p, t, q))
    batches, _ = adapt([row(variable='refractivity_n', measurement='ro_refractivity_tangent_point',
                           elevation_m=height, value=expected, error=5.)])
    idx = tuple(np.argwhere(batches[0].mask)[0])
    assert batches[0].simulated[(0, *idx)] == pytest.approx(expected)


def test_surface_sensor_height_semantics_not_guessed():
    batches, receipt = adapt([row(level_pa=None, measurement='platform_temperature')],
                            surface={'t2': np.full((2, 3, 3), 281.)})
    assert not batches and receipt['counts']['missing_surface_operator_or_diagnostic'] == 1


def test_source_decoder_v2_and_receipt(tmp_path):
    import csv
    path = tmp_path / 'observations.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(shared.TABLE_HEADER)
        writer.writerow(['fixture', 's1', 1, 1, 0, 80000, T.isoformat(), 'temperature_k', 281, 2,
                         'sonde_level', '', '', '', 'a'])
    rows, receipts = read_tables([path])
    assert len(rows) == 1 and rows[0].measurement == 'sonde_level'
    assert receipts[0]['counts']['table_version'] == 2
    assert len(receipts[0]['sha256']) == 64


def test_dewpoint_above_reported_temperature_rejected():
    r = row(variable='dewpoint_k', value=285.)
    batches, receipt = adapt([row(value=281.), r])
    assert receipt['counts']['supersaturated_report'] == 1
    assert all('dewpoint' not in b.name for b in batches)
