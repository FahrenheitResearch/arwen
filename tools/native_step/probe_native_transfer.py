"""New native-produced H2D and live-count CUDA cases; no model timing."""
import ctypes as C
import json
from pathlib import Path
import cupy as cp
import numpy as np
from gpuwm.native_step import Plan
ROOT=Path(__file__).resolve().parents[2]
class Args(C.Structure):
    _fields_=[('input',C.POINTER(C.c_int64)),('output',C.POINTER(C.c_int64)),('delta',C.c_int64)]
class Frame(C.Structure):
    _fields_=[('args',Args)]
k=cp.RawKernel('extern "C" __global__ void add(long long* x){x[0]+=1;}','add')
other=cp.zeros(1,dtype=cp.int64);k((1,),(1,),(other,));cp.cuda.get_current_stream().synchronize()
cases=[]
for label,stream in [('default',cp.cuda.Stream.null),('nonblocking',cp.cuda.Stream(non_blocking=True))]:
    plan=Plan();source=C.c_int64(5);ordinary=np.zeros(1,np.int64)
    def error():return ValueError('original fixture rejection')
    with stream:
        with plan.record(other,k,stream,ordinary):
            args=Args(C.pointer(source),ordinary.ctypes.data_as(C.POINTER(C.c_int64)),0)
            producer=plan.native_host(ROOT/'receipts/libnative-host-fixture.so','fixture_consumer',args,source,ordinary,error=error)
            with plan.native_upload() as operations:
                device=cp.asarray(ordinary)
            address,size=plan.host_buffer(operations[0]);assert size==8
            count=C.c_uint64.from_address(address)
            plan.count_predicate(count,greater_than=2,bit=0)
            with plan.when(require=1):k((1,),(1,),(device,))
            with plan.when(require=8):k((1,),(1,),(other,))
        stream.synchronize();assert int(device.get()[0])==6
        plan.resources.append(device)
        plan.bind(producer,0,Frame,'args')
        frame=Frame(Args(C.pointer(source),C.cast(address,C.POINTER(C.c_int64)),0))
        ordinary.fill(999)
        for value,flags,want in [(0,9,0),(3,8,4),(2,9,2)]:
            source.value=value;other.fill(0);plan.submit(frame,flags=flags);stream.synchronize()
            assert int(device.get()[0])==want and int(other.get()[0])==1 and ordinary[0]==999
            cases.append({'case':label+' count '+str(value),'status':'PASS','source':value,'device':want,'unrelated_group':1})
        # No caller synchronization between these submissions: upload scope
        # must complete before the following producer overwrites its storage.
        for value in range(4,20):
            source.value=value;plan.submit(frame,flags=8)
        stream.synchronize();assert int(device.get()[0])==20
        cases.append({'case':label+' back-to-back buffer lifetime','status':'PASS','submissions':16})
        source.value=-1;other.fill(0);device.fill(77)
        try:plan.submit(frame,flags=9)
        except ValueError as exc:assert str(exc)=='original fixture rejection'
        else:raise AssertionError('missing original error')
        stream.synchronize();assert int(device.get()[0])==77 and int(other.get()[0])==0
        cases.append({'case':label+' failure before upload and groups','status':'PASS'})
        plan.close()
receipt={'status':'PASS','cases':cases,'case_count':len(cases),'cupy':cp.__version__,'model_claim':False,'timing_claim':False}
(ROOT/'receipts/native-transfer-gpu.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
