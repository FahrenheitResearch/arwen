"""The local-device opt-out governs preparation before any CUDA probe."""
from types import SimpleNamespace
import pytest
from gpuwm.ingest import preprocess_backend as module
from gpuwm.local_gpu import NO_LOCAL_GPU_ENV


@pytest.fixture
def forbidden_device(monkeypatch):
    monkeypatch.setenv(NO_LOCAL_GPU_ENV, '1')
    # An accidental caller visibility reset must not defeat the policy.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(module, '_ANNOUNCED_AUTO_REASONS', set())
    def touched(*args, **kwargs):
        raise AssertionError('local CUDA boundary was reached')
    monkeypatch.setattr(module, 'CudaPreprocessBackend', touched)
    monkeypatch.setattr(module, '_gpu_runtime_installed', touched)
    monkeypatch.setattr(module, '_device_reading', touched)
    monkeypatch.setattr(module, 'ParallelCpuPreprocessBackend',
        lambda workers=None, bridge=None: SimpleNamespace(name='cpu', workers=workers))
    return touched


def test_auto_uses_cpu_before_constructing_cuda(forbidden_device):
    backend = module.resolve_preprocess_backend('auto', workers=1)
    assert backend.name == 'cpu'
    assert backend.workers == 1
    assert backend.selection['requested'] == 'auto'
    assert 'GPUWM_NO_LOCAL_GPU' in backend.selection['reason']


def test_explicit_cuda_refuses_before_probe(forbidden_device):
    with pytest.raises(ValueError, match='GPUWM_NO_LOCAL_GPU'):
        module.resolve_preprocess_backend('cuda')


def test_named_device_auto_never_prices_or_probes(forbidden_device):
    backend, selection = module.decide_preparation_device(
        'auto', forbidden_device, probe=forbidden_device)
    assert backend == 'cpu'
    assert selection['backend'] == 'cpu'
    assert 'GPUWM_NO_LOCAL_GPU' in selection['reason']


def test_named_explicit_cuda_never_prices_or_probes(forbidden_device):
    with pytest.raises(ValueError, match='GPUWM_NO_LOCAL_GPU'):
        module.decide_preparation_device('cuda', forbidden_device, probe=forbidden_device)


def test_direct_cuda_backend_cannot_import_cupy(forbidden_device, monkeypatch):
    import builtins
    from gpuwm.ingest import horiz
    original = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == 'cupy':
            raise AssertionError('CuPy import boundary was reached')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    with pytest.raises(RuntimeError, match='GPUWM_NO_LOCAL_GPU'):
        horiz._cupy()


def test_identity_probe_does_not_import_cupy(forbidden_device, monkeypatch):
    import builtins
    from gpuwm.core import device_probe
    original = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == 'cupy':
            raise AssertionError('CuPy identity probe boundary was reached')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    assert device_probe.cuda_device_identity(0) is None
