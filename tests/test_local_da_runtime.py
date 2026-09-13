"""CPU orchestration checks with the real regional cycle and restart owners."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest

from gpuwm.local_da import publish, PlanError, build_plan
from gpuwm.local_da_runtime import launch, read_plan, PreparedBackend
from gpuwm.ensemble.member import MemberOutcome
from gpuwm.ensemble.state_sha import checkpoint_state_sha256
from test_local_da_plan import request, price, availability


def saved(tmp_path, **kwargs):
    plan = build_plan(request(**kwargs), availability=availability, price=price)
    result = publish(plan, tmp_path / 'local')
    return Path(result['plan_path']), plan


class Backend:
    mp_physics = 6
    def __init__(self, *, fail_preflight=False, fail_analysis=False):
        self.calls, self.fail_preflight, self.fail_analysis = [], fail_preflight, fail_analysis
    def preflight(self):
        self.calls.append(('preflight',))
        if self.fail_preflight:
            raise PlanError('A required runtime dependency is absent; install it before launch.', code='MISSING_DEPENDENCY')
    def prepare(self):
        self.calls.append(('prepare',))
    def member_runner(self, **kwargs):
        self.calls.append(('member', kwargs['index'], kwargs['run_seconds'], kwargs.get('restart')))
        root = Path(kwargs['member_dir'])
        root.mkdir(parents=True, exist_ok=True)
        restart = kwargs.get('restart')
        value = np.zeros((2, 3, 3), np.float32)
        initial = '0' * 64
        if restart is not None:
            with np.load(restart, allow_pickle=False) as file:
                value = file['state/thp'].copy()
            initial = checkpoint_state_sha256(restart)
        checkpoint = root / 'gpuwmrst_d01_end.npz'
        np.savez(checkpoint, **{'state/thp': value + 1., 'meta/elapsed_seconds': np.asarray(kwargs['run_seconds'])})
        return MemberOutcome(index=kwargs['index'], seed=kwargs['seed'], member_dir=root,
            initial_state_sha256=initial, final_state_sha256=checkpoint_state_sha256(checkpoint),
            wall_seconds=.01, sim_seconds=kwargs['run_seconds'], wrfout_count=0,
            last_checkpoint=str(checkpoint), perturbation={})
    def assimilate(self, cycle_index, member_states):
        self.calls.append(('analysis', cycle_index, tuple(sorted(member_states))))
        if self.fail_analysis:
            raise ValueError('Analysis input changed; restore the reviewed observations before resuming.')
        return {i: {'thp': np.full((2, 3, 3), .25)} for i in member_states}, {'method': 'cpu-test-increment'}
    def products(self, forecast):
        self.calls.append(('products',))
        return {'status': 'test-no-output'}
    def close(self):
        self.calls.append(('close',))


def test_one_trajectory_real_cycle_publication_then_short_forecast(tmp_path):
    path, plan = saved(tmp_path)
    backend = Backend()
    result = launch(path, backend=backend)
    assert result['status'] == 'COMPLETE' and result['forecast_started']
    members = [c for c in backend.calls if c[0] == 'member']
    assert len(members) == 2
    assert {c[1] for c in members} == {0}
    assert members[0][2] == plan['selected']['cadence_seconds']
    assert members[1][2] == plan['selected']['cadence_seconds'] + plan['selected']['forecast_seconds']
    assert members[0][3] is None and members[1][3].name == 'analysis.npz'
    with np.load(path.parent / 'forecast/member_000/gpuwmrst_d01_end.npz', allow_pickle=False) as file:
        np.testing.assert_array_equal(file['state/thp'], 2.25)
    assert backend.calls[-1] == ('close',)
    assert json.loads((path.parent / 'execution.json').read_text())['status'] == 'COMPLETE'


def test_resume_does_not_repeat_completed_members_or_analysis(tmp_path):
    path, _ = saved(tmp_path)
    launch(path, backend=Backend())
    backend = Backend()
    result = launch(path, backend=backend)
    assert result['status'] == 'COMPLETE'
    assert not [c for c in backend.calls if c[0] in ('member', 'analysis')]


def test_multi_member_multiple_cycles_restart_complete_roster(tmp_path):
    path, plan = saved(tmp_path, scale=2)
    backend = Backend()
    result = launch(path, backend=backend)
    assert result['status'] == 'COMPLETE'
    assert [c[2] for c in backend.calls if c[0] == 'analysis'] == [tuple(range(plan['selected']['members']))] * plan['selected']['cycles']
    assert len([c for c in backend.calls if c[0] == 'member']) == plan['selected']['members'] * (plan['selected']['cycles'] + 1)


def test_missing_dependency_refused_without_runtime_writes(tmp_path):
    path, _ = saved(tmp_path)
    before = set(path.parent.rglob('*'))
    backend = Backend(fail_preflight=True)
    with pytest.raises(PlanError, match='install'):
        launch(path, backend=backend)
    assert set(path.parent.rglob('*')) == before
    assert backend.calls == [('preflight',)]


def test_analysis_failure_never_launches_short_forecast(tmp_path):
    path, _ = saved(tmp_path)
    backend = Backend(fail_analysis=True)
    with pytest.raises(ValueError) as caught:
        launch(path, backend=backend)
    assert caught.value.forecast_started
    report = json.loads((path.parent / 'execution.json').read_text())
    assert report['status'] == 'FAILED' and report['forecast_started']
    assert not (path.parent / 'forecast').exists()
    assert backend.calls[-1] == ('close',)


def test_incomplete_analysis_roster_cannot_start_forecast(tmp_path):
    path, _ = saved(tmp_path)
    with pytest.raises(PlanError, match='roster'):
        launch(path, backend=Backend(), roster_reader=lambda *a, **kw: {})
    assert not (path.parent / 'forecast').exists()


@pytest.mark.parametrize('filename', ['experiment.toml', 'ensemble.toml', 'experiment.namelist.wps'])
def test_generated_authorities_are_bound_before_any_work(tmp_path, filename):
    path, _ = saved(tmp_path)
    with (path.parent / filename).open('a') as file:
        file.write('\n# changed\n')
    backend = Backend()
    with pytest.raises(PlanError, match='reviewed'):
        launch(path, backend=backend)
    assert not backend.calls


def test_review_digest_tamper_refused(tmp_path):
    path, _ = saved(tmp_path)
    doc = json.loads(path.read_text())
    doc['selected']['members'] += 1
    path.write_text(json.dumps(doc))
    with pytest.raises(PlanError, match='digest'):
        read_plan(path)


def test_surface_diagnostics_are_bound_to_member_checkpoint(tmp_path):
    backend = PreparedBackend({'analysis_times': []}, tmp_path)
    receipt = {'fields': ['t2'], 'checkpoint_sha256': 'a' * 64}
    np.savez(tmp_path / 'surface-end.npz', t2=np.zeros((2, 2)),
             receipt=np.frombuffer(json.dumps(receipt).encode(), dtype=np.uint8))
    with pytest.raises(PlanError, match='diagnostics'):
        backend._surface_for_members({0: {'member_dir': tmp_path, 'state_sha256': 'b' * 64}})
