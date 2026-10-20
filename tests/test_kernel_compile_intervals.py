"""Diagnostic compiler intervals never stand in for NVRTC/GPU timing."""
import concurrent.futures
import sys
from types import ModuleType, SimpleNamespace
import pytest
from gpuwm import kernel_compile_worker as worker, kernel_plan as plan, numerics


def fake_compiler(monkeypatch, fails=False):
    selector = lambda arch: ('original', 'cubin')
    calls = []
    def compile(*args):
        calls.append(args)
        assert numerics._mode == 'strict'
        if fails:
            raise RuntimeError('original compile failure')
        return b'image', None
    compiler = SimpleNamespace(_use_ptx=False,
        _get_arch_for_options_for_nvrtc=selector,
        _compile_using_nvrtc_no_warning=compile)
    cupy = ModuleType('cupy'); cuda = ModuleType('cupy.cuda'); cuda.compiler = compiler
    nvrtc = SimpleNamespace(getSupportedArchs=lambda: [120])
    libs = ModuleType('cupy_backends.cuda.libs'); libs.nvrtc = nvrtc
    monkeypatch.setitem(sys.modules, 'cupy', cupy)
    monkeypatch.setitem(sys.modules, 'cupy.cuda', cuda)
    monkeypatch.setitem(sys.modules, 'cupy_backends.cuda.libs', libs)
    monkeypatch.setattr(worker, '_identity', lambda: {'compiler': 'bound'})
    monkeypatch.setattr(numerics, '_mode', 'fast')
    args = dict(source='exact', options=['ordered'], filename='owned.cu',
                name_expressions=['k'], cache_in_memory=False, method=None)
    return compiler, selector, calls, args


@pytest.mark.parametrize('trace', [False, True])
def test_worker_interval_is_optional_and_preserves_compile(monkeypatch, trace):
    compiler, selector, calls, args = fake_compiler(monkeypatch)
    stamps = iter([123000, 456000])
    def mono():
        assert trace, 'ordinary compiler path must not read diagnostic clocks'
        return next(stamps)
    monkeypatch.setattr(worker.time, 'monotonic_ns', mono)
    result, timing = worker.compile_request(args, '120', {'compiler':'bound'}, trace=trace)
    assert result == (b'image', None) and len(calls)==1
    assert calls[0] == ('exact', ('ordered',), '120', 'owned.cu', ['k'], None, False, None, None)
    assert compiler._get_arch_for_options_for_nvrtc is selector and numerics._mode=='fast'
    if trace:
        assert timing['compile_begin_mono_ns']==123000 and timing['compile_end_mono_ns']==456000
    else:
        assert set(timing)=={'pid','seconds','cpu_seconds'}


def test_failed_compile_keeps_original_error_and_restores_globals(monkeypatch):
    compiler, selector, calls, args = fake_compiler(monkeypatch, fails=True)
    with pytest.raises(RuntimeError, match='original compile failure'):
        worker.compile_request(args, '120', {'compiler':'bound'}, trace=True)
    assert compiler._get_arch_for_options_for_nvrtc is selector and numerics._mode=='fast'


def record():
    return dict(pool_compiled=0,pool_failed=0,pool_seconds=0.,pool_cpu_seconds=0.,
                started=0.,last_done=None,pool_processes=[],compile_intervals=[])


def test_proxy_keeps_request_target_and_success_interval(monkeypatch):
    r=record(); pending=concurrent.futures.Future()
    proxy=plan._ProcessCompileFuture(pending,r,request_sha256='bound-request',arch='120')
    pending.set_result(((b'image',None),dict(pid=123,seconds=.2,cpu_seconds=.1,
        compile_begin_mono_ns=100,compile_end_mono_ns=300)))
    proxy.completed(pending)
    assert proxy.result()==(b'image',None)
    assert r['compile_intervals']==[dict(request_sha256='bound-request',arch='120',pid=123,
        begin_mono_ns=100,end_mono_ns=300,cpu_seconds=.1)]
    monkeypatch.setitem(plan._state,'replay',r)
    published=plan.receipt()['replay']['compile_intervals']
    published[0]['pid']=999;published.clear()
    assert r['compile_intervals'][0]['pid']==123


def test_failed_proxy_counts_failure_without_invented_interval():
    r=record(); pending=concurrent.futures.Future()
    proxy=plan._ProcessCompileFuture(pending,r)
    pending.set_exception(RuntimeError('failed'))
    proxy.completed(pending)
    with pytest.raises(RuntimeError,match='failed'):proxy.result()
    assert r['pool_failed']==1 and r['compile_intervals']==[]


@pytest.mark.parametrize('trace', [False, True])
def test_submit_binds_final_request_only_when_traced(monkeypatch, trace):
    from gpuwm import kernel_aot
    r=record()
    if not trace:r.pop('compile_intervals')
    calls=[]
    pending=concurrent.futures.Future()
    def submit(*args,**kwargs):
        calls.append((args,kwargs));return pending
    monkeypatch.setitem(plan._state,'compiler_pool',SimpleNamespace(submit=submit))
    monkeypatch.setitem(plan._state,'worker_identity',{'compiler':'bound'})
    monkeypatch.setattr(kernel_aot,'effective_options',lambda flags: (*flags,'final'))
    args=dict(source='exact',options=['ordered'],arch='device',filename='owned.cu',
              name_expressions=None,cache_in_memory=False,method=None)
    future=plan._submit_compile(args,'120',r)
    assert calls[0][1]==({'trace':True} if trace else {})
    submitted=calls[0][0][1]
    assert submitted['options']==['ordered','final'] and args['options']==['ordered']
    assert future.request_sha256==(plan._key(submitted) if trace else None)
    pending.cancel()
