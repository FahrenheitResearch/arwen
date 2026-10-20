"""Real-driver recorder contract checks, not a weather/performance proof."""
import ctypes as C
import importlib.util
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native_step',root/'gpuwm/native_step/__init__.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
class Frame(C.Structure):
    _fields_=[('delta',C.c_int32)]
kernel=cp.RawKernel('extern "C" __global__ void add32(int*x,int v){x[threadIdx.x]+=v;}','add32')
x=cp.zeros(32,cp.int32); y=cp.zeros_like(x)
s1=cp.cuda.Stream(non_blocking=True);s2=cp.cuda.Stream(non_blocking=True)
event=cp.cuda.Event(disable_timing=True)
kernel((1,),(32,),(x,np.int32(0)))
cp.cuda.Device().synchronize()
receipt={'purpose':__doc__,'checks':[],'runtime_version':cp.cuda.runtime.runtimeGetVersion(),'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
started=time.perf_counter()
plan=native.Plan()
try:
    with plan.record(kernel,x,y,s1,s2,event):
        with s1:
            x.fill(0)
            kernel((1,),(32,),(x,np.int32(1)))
            with plan.when(require=1):
                kernel((1,),(32,),(x,np.int32(10)))
            event.record()
        with s2:
            s2.wait_event(event)
            cp.copyto(y,x)
            cp.add(y,np.int32(2),out=y)
    cp.cuda.Device().synchronize()
    assert np.array_equal(cp.asnumpy(y),np.full(32,13,np.int32))
    description=plan.describe()
    (root/'receipts/cuda-contract-sequence.json').write_text(json.dumps(description,indent=2)+'\n')
    op=next(r['index'] for r in description['operations'] if r.get('kernel')=='add32')
    plan.bind(op,1,Frame,'delta')
    for flags,expected in ((0,5),(1,15),(0,5)):
        plan.submit(Frame(3),flags=flags)
        cp.cuda.Device().synchronize()
        actual=cp.asnumpy(y)
        assert np.array_equal(actual,np.full(32,expected,np.int32)),(flags,cp.asnumpy(x),actual)
        receipt['checks'].append({'kind':'replay','flags':flags,'expected':expected,'words':32,'differences':0})
    # Missing bound bytes must fail before issuing any command.
    before=cp.asnumpy(y)
    try:
        plan.submit()
        raise AssertionError('missing parameter frame accepted')
    except native.RecordingError as exc:
        assert 'rc=-1002' in str(exc),str(exc)
    cp.cuda.Device().synchronize()
    assert np.array_equal(before,cp.asnumpy(y))
    receipt['checks'].append({'kind':'missing_frame','result':'rejected before side effects'})
    receipt['sequence']=description
finally:
    cp.cuda.Device().synchronize();plan.close()
# Host writes/reads cannot be silently frozen or omitted.
for kind in ('upload','readback'):
    plan=native.Plan();plan.trace_failures()
    try:
        try:
            with plan.record(kernel,x):
                kernel((1,),(32,),(x,np.int32(1)))
                if kind=='upload': uploaded=cp.asarray(np.arange(32,dtype=np.int32))
                else: cp.asnumpy(x)
            raise AssertionError('host transfer accepted without native contract')
        except native.RecordingError as exc:
            assert 'rc=-1004' in str(exc),str(exc)
        try:
            plan.submit()
            raise AssertionError('failed recording replayed')
        except native.RecordingError:
            pass
        receipt['checks'].append({'kind':kind,'result':'recording and replay rejected','detail':plan.describe()})
    finally:
        cp.cuda.Device().synchronize();plan.close()
# Reuse temporary storage while keeping its backing allocation private.
plan=native.Plan()
try:
    with plan.record(x):
        x.fill(0)
        for increment in range(1,33):
            temporary=cp.add(x,np.int32(increment))
            cp.copyto(x,temporary)
            del temporary
    cp.cuda.Device().synchronize()
    reserved=plan.describe()['pool_reserved_bytes']
    assert reserved <= 4096, reserved
    for _ in range(3):
        plan.submit()
        cp.cuda.Device().synchronize()
        assert np.array_equal(cp.asnumpy(x),np.full(32,528,np.int32))
    receipt['checks'].append({'kind':'reused_temporaries','iterations':32,
                              'replays':3,'words_per_replay':32,'differences':0,
                              'reserved_bytes':reserved})
finally:
    cp.cuda.Device().synchronize();plan.close()
receipt.update(status='PASS',wall_s=time.perf_counter()-started)
(root/'receipts/cuda-contract.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'checks':len(receipt['checks']),'wall_s':receipt['wall_s']}))
