"""Behavioral controls shared with the prior finite execution interface."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pytest


def test_default_review_discovers_satellite_without_local_files(tmp_path, monkeypatch):
    import gpuwm.local_da_observations as review
    from gpuwm.local_da import derive_rung, configuration
    from gpuwm.obs import nexrad
    from test_local_da_plan import request
    binary = tmp_path / 'native-observation-door'
    binary.write_bytes(b'fixed executable fixture')
    monkeypatch.setattr(review, '_door_record', lambda *args: (str(binary), 'available'))
    monkeypatch.setattr(nexrad, 'find_nexrad_bin', lambda: None)
    monkeypatch.setattr(review.importlib.util, 'find_spec', lambda name: None)
    req = request()
    rung = derive_rung(req, req.scale)
    _, _, exp, _ = configuration(req, rung)
    cwp = next(row for row in review.inspect_routes(req, rung, exp)
               if row['route'] == 'cloud-water-path')
    assert cwp['status'] == 'candidate' and cwp['files'] == []
    assert cwp['binary_sha256'] == hashlib.sha256(binary.read_bytes()).hexdigest()


@pytest.mark.parametrize('future_seconds', [1., .1])
def test_nominal_grid_time_cannot_hide_future_satellite_measurements(future_seconds):
    from gpuwm.local_da_fetch import assigned_document
    when = datetime(2026, 9, 12, 18, 15, tzinfo=timezone.utc)
    document = {'schema': 'gpuwm-obs.goes-grid.v1', 'valid_time': when.isoformat(),
        'provenance': {'pack': {'scan_start': (when - timedelta(minutes=3)).isoformat(),
                              'scan_end': (when + timedelta(seconds=future_seconds)).isoformat()}}}
    assert assigned_document(document, when, [when], 900.) is False


def test_published_analysis_usage_survives_resume_without_counting_fetched_columns(tmp_path):
    from gpuwm.da.letkf import GriddedObs
    from gpuwm.da.radar_assimilation import innovation_summary
    from gpuwm.local_da_runtime import launch
    from test_local_da_runtime import saved, Backend

    class CountedBackend(Backend):
        def assimilate(self, cycle_index, member_states):
            increments, report = super().assimilate(cycle_index, member_states)
            mask = np.zeros((2, 3, 3), bool)
            mask[0, 1, 1] = True
            batches = [GriddedObs('cwp', np.zeros(mask.shape), 50.,
                np.zeros((2, *mask.shape)), mask)]
            report.update(innovations=innovation_summary(batches),
                routes=[dict(id='cloud-water-path', status='fetched', observed_columns=9),
                        dict(id='surface', status='unavailable', reason='No report arrived')])
            return increments, report

    path, _ = saved(tmp_path)
    first = launch(path, backend=CountedBackend())
    usage = first['observation_usage']
    assert usage[0]['accepted_for_analysis'] == usage[0]['cwp_accepted'] == 1
    assert usage[0]['routes'][0]['observed_columns'] == 9
    assert usage[0]['routes'][1]['status'] == 'unavailable'
    backend = CountedBackend()
    second = launch(path, backend=backend)
    assert second['observation_usage'] == usage
    assert not [call for call in backend.calls if call[0] == 'analysis']
    assert json.loads((path.parent / 'execution.json').read_text())['observation_usage'] == usage


def test_usage_does_not_infer_assimilation_from_enabled_or_downloaded_streams():
    from gpuwm.local_da_runtime import observation_usage
    for method, expected in [('forecast-only', 0), ('legacy-unreported', None)]:
        result = observation_usage(dict(method=method, cwp_assimilated=True,
            routes=[dict(id='cloud-water-path', status='fetched', observed_columns=100)]))
        assert result['accepted_for_analysis'] == expected
        assert result['cwp_accepted'] == expected
    result = observation_usage(dict(innovations=[dict(name='cwp', observations=0)]))
    assert result['accepted_for_analysis'] == result['cwp_accepted'] == 0
