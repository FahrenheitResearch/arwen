"""Storage retries preserve inputs and never retry numerical failure."""
from concurrent.futures import CancelledError
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from gpuwm.da import radar_assimilation as owner
from gpuwm.da.letkf import GriddedObs, LetkfCapacityError, LetkfDiagnostics, LetkfError


class DeviceArray(np.ndarray):
    pass


class Namespace:
    def __init__(self, fail_stage=False, fail_unstage=False):
        self.references = []
        self.releases = 0
        self.stages = 0
        self.fail_stage = fail_stage
        self.fail_unstage = fail_unstage
        self.cuda = SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None))
    def asarray(self, values):
        self.stages += 1
        if self.fail_stage and self.stages == 2:
            raise MemoryError('device staging allocation failed')
        result = np.array(values, copy=True).view(DeviceArray)
        self.references.append(weakref.ref(result))
        return result
    def asnumpy(self, values):
        if self.fail_unstage:
            raise MemoryError('result transfer allocation failed')
        return np.array(values, copy=True)
    def get_default_memory_pool(self):
        return self
    def free_all_blocks(self):
        # This executes before the host retry, not after it has succeeded.
        assert all(reference() is None for reference in self.references)
        self.releases += 1


def inputs():
    values = np.arange(12., dtype=np.float64).reshape(3, 1, 2, 2)
    prior = {'u': values}
    batch = GriddedObs('observation', values[0].copy(), 1., values.copy(),
                       np.ones((1, 2, 2), dtype=bool))
    return prior, [batch]


def test_builtin_cuda_prefers_resident_and_forwards_progress():
    prior, batches = inputs()
    namespace = Namespace()
    messages = []
    def solver(p, b, geometry, config, diagnostics, **options):
        assert isinstance(p['u'], DeviceArray)
        assert 'solve_namespace' not in options
        options['progress'](dict(phase='solve', gridpoints_done=3))
        diagnostics.active_points = 3
        return {'u': namespace.asarray(p['u']*.1)}
    solver.supports_host_staging = True
    result = owner._execute_analysis(solver, prior, batches, None, None,
        namespace=namespace, device='cuda', progress=messages.append)
    increments, diagnostics, _, _, storage, attempts = result
    assert storage == 'cuda-resident' and len(attempts) == 1
    assert diagnostics.active_points == 3
    assert messages == [dict(phase='solve', gridpoints_done=3, attempt=1, storage='cuda-resident')]
    assert not isinstance(increments['u'], DeviceArray)


@pytest.mark.parametrize('failure', ['stage', 'solve', 'unstage', 'capacity'])
def test_memory_retry_releases_actual_attempt_references_and_resets_diagnostics(failure):
    prior, batches = inputs()
    original = prior['u'].copy()
    namespace = Namespace(failure == 'stage', failure == 'unstage')
    messages = []
    def solver(p, b, geometry, config, diagnostics, **options):
        if isinstance(p['u'], DeviceArray):
            diagnostics.active_points = 999
            options['progress'](dict(phase='solve', gridpoints_done=3))
            if failure == 'solve':
                raise MemoryError('matrix allocation failed')
            if failure == 'capacity':
                raise LetkfCapacityError('resident scratch cannot fit')
            return {'u': namespace.asarray(p['u']*.1)}
        assert namespace.releases == 1
        assert all(reference() is None for reference in namespace.references)
        assert diagnostics.active_points == 0
        assert options['solve_namespace'] is namespace
        diagnostics.active_points = 4
        options['progress'](dict(phase='complete', gridpoints_done=4))
        return {'u': p['u']*.1}
    solver.supports_host_staging = True
    result = owner._execute_analysis(solver, prior, batches, None, None,
        namespace=namespace, device='cuda', progress=messages.append)
    increments, diagnostics, _, _, storage, attempts = result
    assert storage == 'host-staged-cuda'
    assert diagnostics.active_points == 4
    assert [row['status'] for row in attempts] == ['memory-failed', 'computed']
    assert not any(row['committed'] for row in attempts)
    assert any(row['phase'] == 'retry' and row['attempt'] == 2 for row in messages)
    assert messages[-1]['attempt'] == 2
    np.testing.assert_array_equal(prior['u'], original)
    np.testing.assert_array_equal(increments['u'], original*.1)


@pytest.mark.parametrize('error', [LetkfError('nonfinite covariance'),
    FloatingPointError('invalid arithmetic'), CancelledError('stop'), KeyboardInterrupt()])
def test_numerical_errors_and_cancellation_do_not_retry_even_with_oom_context(error):
    prior, batches = inputs()
    namespace = Namespace()
    calls = []
    error.__context__ = MemoryError('earlier unrelated allocation')
    def solver(*args, **kwargs):
        calls.append(1)
        raise error
    solver.supports_host_staging = True
    with pytest.raises(type(error)):
        owner._execute_analysis(solver, prior, batches, None, None,
            namespace=namespace, device='cuda')
    assert len(calls) == 1 and namespace.releases == 0


@pytest.mark.parametrize('device', ['host', 'cuda'])
def test_custom_callback_keeps_positional_contract_and_diagnostics_identity(device):
    prior, batches = inputs()
    diagnostics = LetkfDiagnostics()
    def solver(p, b, geometry, config, actual_diagnostics):
        assert actual_diagnostics is diagnostics
        actual_diagnostics.active_points = 4
        return {'u': p['u']*.1}
    result = owner._execute_analysis(solver, prior, batches, None, None,
        namespace=Namespace() if device == 'cuda' else np,
        device=device, diagnostics=diagnostics)
    assert result[1] is diagnostics


def test_unknown_custom_callback_memory_error_is_not_retried():
    prior, batches = inputs()
    calls = []
    def solver(*args):
        calls.append(1)
        raise MemoryError('custom allocation failed')
    with pytest.raises(MemoryError):
        owner._execute_analysis(solver, prior, batches, None, None,
            namespace=Namespace(), device='cuda')
    assert len(calls) == 1
