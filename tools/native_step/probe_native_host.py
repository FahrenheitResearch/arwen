"""Real CUDA readback plus compiled host consumer; no model/timing claim."""
import ctypes as C
import json
from pathlib import Path
import cupy as cp
from gpuwm.native_step import Plan
ROOT=Path(__file__).resolve().parents[2]
class Args(C.Structure):
    _fields_=[('input',C.POINTER(C.c_int64)),('output',C.POINTER(C.c_int64)),('delta',C.c_int64)]
class Frame(C.Structure):
    _fields_=[('args',Args)]
k=cp.RawKernel('extern "C" __global__ void add(long long* x){x[0]+=1;}','add')
d=cp.array([10],dtype=cp.int64);count=cp.zeros(1,dtype=cp.int64)
k((1,),(1,),(count,));cp.cuda.get_current_stream().synchronize();count.fill(0)
cases=[]
for label,stream in [('default',cp.cuda.Stream.null),('nonblocking',cp.cuda.Stream(non_blocking=True))]:
    plan=Plan();output=C.c_int64();errors=[]
    def error():
        errors.append(True)
        return ValueError('original fixture rejection')
    with stream:
        d.fill(10);count.fill(0)
        with plan.record(d,count,k,stream):
            k((1,),(1,),(d,))
            with plan.readback() as operations:
                ordinary=cp.asnumpy(d)
            address,size=plan.host_buffer(operations[0]);assert size==8
            args=Args(C.cast(address,C.POINTER(C.c_int64)),C.pointer(output),7)
            op=plan.native_host(ROOT/'receipts/libnative-host-fixture.so','fixture_consumer',args,output,error=error)
            k((1,),(1,),(count,))
        stream.synchronize();assert ordinary[0]==11 and output.value==18
        cases.append({'case':label+' capture','status':'PASS'})
        args.delta=99;d.fill(20);count.fill(0);plan.submit();stream.synchronize()
        assert output.value==28 and int(count.get()[0])==1 and ordinary[0]==11 and not errors
        cases.append({'case':label+' live readback and argument snapshot','status':'PASS'})
        plan.bind(op,0,Frame,'args');frame=Frame(Args(C.cast(address,C.POINTER(C.c_int64)),C.pointer(output),9))
        d.fill(30);count.fill(0);plan.submit(frame);stream.synchronize()
        assert output.value==40 and int(count.get()[0])==1
        cases.append({'case':label+' changed frame','status':'PASS'})
        d.fill(-3);count.fill(0)
        try:
            plan.submit(frame)
        except ValueError as exc:
            assert str(exc)=='original fixture rejection'
        else:
            raise AssertionError('missing original error')
        stream.synchronize();assert errors==[True] and output.value==40 and int(count.get()[0])==0
        cases.append({'case':label+' exact error and stopped downstream','status':'PASS'})
        plan.close()
cp.cuda.get_current_stream().synchronize()
receipt={'status':'PASS','cases':cases,'case_count':len(cases),'cupy':cp.__version__,'timing_claim':False,'model_claim':False}
(ROOT/'receipts/native-host-gpu.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
