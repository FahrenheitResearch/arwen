"""A two-position nest must retain both ends of its wind swath."""
import sys
from types import SimpleNamespace

import numpy as np

from gpuwm.ensemble.cyclone_products import CommonWind


def test_moving_wind_union_keeps_departed_and_new_strips(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', np)
    common = CommonWind({'common_grid': dict(south=0, north=1, west=0,
                                           east=2, ny=2, nx=3)})
    common.bind = lambda node, model: None
    common.sample = lambda value: value
    common.rotation = (0., 1.)
    common.normal = (1., 0.)
    common.coast = np.zeros((2, 3), bool)
    fields = {'u10': np.array([[25., 20., 0.]] * 2),
              'v10': np.zeros((2, 3))}
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=2),
                           state=SimpleNamespace(elapsed_seconds=900.,
                               physics=SimpleNamespace(fields=fields)))
    common.inside = np.array([[True, True, False]] * 2)
    common.observe(node=node, model=None)
    node.state.elapsed_seconds = 1800.
    fields['u10'] = np.array([[0., 30., 35.]] * 2)
    common.inside = np.array([[False, True, True]] * 2)
    common.observe(node=node, model=None)
    np.testing.assert_array_equal(common.maximum, [[25., 30., 35.]] * 2)
    assert common.visited.all()
    np.testing.assert_array_equal(common.coverage, [[False, True, False]] * 2)
    np.testing.assert_array_equal(common.sample_count, [[1, 2, 1]] * 2)
    # A repeated callback at the same model clock must not double-count.
    common.observe(node=node, model=None)
    np.testing.assert_array_equal(common.sample_count, [[1, 2, 1]] * 2)


def test_ensemble_step_swaths_keep_members_separate(tmp_path, monkeypatch):
    from datetime import datetime
    from threading import RLock
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    from gpuwm import tc_accumulator

    class Capture:
        def __init__(self, request):
            self.request, self.samples = request, []
        def committed_step(self, *, node, model):
            self.samples.append(node.state.elapsed_seconds)

    monkeypatch.setattr(tc_accumulator, 'Capture', Capture)
    owner = HeadlineDiagnosticCollector.__new__(HeadlineDiagnosticCollector)
    owner.cyclone_request = {'common_grid': dict(south=0, north=1, west=0, east=2, ny=2, nx=3)}
    owner.cyclone_captures = {}
    owner._lock, owner.root, owner.start_time = RLock(), tmp_path, datetime(2024, 10, 7)
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=2), state=SimpleNamespace(elapsed_seconds=0.))
    for member, clock in ((0, 900), (1, 900), (0, 1800), (1, 2700)):
        with member_output_scope(MemberOutputCapture(owner, member)):
            node.state.elapsed_seconds = clock
            owner.committed_step(node=node, model=None)
    assert owner.cyclone_captures[0].samples == [900, 1800]
    assert owner.cyclone_captures[1].samples == [900, 2700]
    assert owner.cyclone_captures[0].root != owner.cyclone_captures[1].root
    assert owner.cyclone_captures[0].request['analysis_time'] == '2024-10-07T00:00:00+00:00'


def test_ordinary_analysis_handoff_keeps_first_7p5_second_rain_interval(tmp_path, monkeypatch):
    """The real arm-B first step must not shorten the exposure window."""
    from datetime import datetime
    from threading import RLock
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    from gpuwm.ensemble.runtime_context import (
        MemberOutputCapture, member_output_scope, observe_initial_diagnostics)
    from gpuwm import tc_accumulator

    monkeypatch.setitem(sys.modules, 'cupy', np)
    request = {'common_grid': dict(south=0, north=2, west=0, east=2, ny=3, nx=3),
               'products': ['rain']}
    rain = np.zeros((3, 3), np.float32)
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=2, run=SimpleNamespace()),
        state=SimpleNamespace(elapsed_seconds=0.,
            physics=SimpleNamespace(fields={'u10': rain+25., 'v10': rain.copy()}),
            existing_scratch=lambda key: rain if key == 'mp_rainnc' else None))
    model = SimpleNamespace(walk_parent_first=lambda: iter([node]))

    class Capture:
        def __init__(self, options):
            self.common = CommonWind(options)
            self.common.bind = lambda node, model: None
            self.common.sample = lambda value: value
            self.common.rotation = (0., 1.)
            self.common.normal = (1., 0.)
            self.common.coast = np.zeros((3, 3), bool)
            self.common.inside = np.ones((3, 3), bool)
            self.common.j, self.common.i = np.indices((3, 3))
        def committed_step(self, *, node, model):
            self.common.observe(node=node, model=model)

    monkeypatch.setattr(tc_accumulator, 'Capture', Capture)
    owner = HeadlineDiagnosticCollector.__new__(HeadlineDiagnosticCollector)
    owner.cyclone_request, owner.cyclone_captures = request, {}
    owner._lock, owner.root, owner.start_time = RLock(), tmp_path, datetime(2024, 10, 7)
    with member_output_scope(MemberOutputCapture(owner, 0)):
        observe_initial_diagnostics(model)
        observe_initial_diagnostics(model)
        for seconds in (7.5, 15.):
            node.state.elapsed_seconds = seconds
            rain[:] += 1.
            owner.committed_step(node=node, model=model)
    common = owner.cyclone_captures[0].common
    assert common.samples == 3
    assert common.rain_seconds[0, 0] == 15.  # formerly 7.5
    assert common.rain[0, 0] == 2.
    owner.cyclone_request = None
    with member_output_scope(MemberOutputCapture(owner, 1)):
        observe_initial_diagnostics(object())
    assert set(owner.cyclone_captures) == {0}
