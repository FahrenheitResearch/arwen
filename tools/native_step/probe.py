"""Real CUDA seam check before a model recording, not a weather benchmark."""
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import time
import cupy as cp
import numpy as np
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("native_step",root/"gpuwm/native_step/__init__.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
kernel=cp.RawKernel('extern "C" __global__ void add(float*x,float v){x[threadIdx.x]+=v;}',"add")
x=cp.zeros(32,cp.float32)
kernel((1,),(32,),(x,np.float32(0)))
cp.cuda.get_current_stream().synchronize()
p=m.Plan()
class Frame(C.Structure):
    _fields_=[("increment",C.c_float)]
t=time.perf_counter()
try:
    with p.record(kernel,x):
        kernel((1,),(32,),(x,np.float32(1)))
    record=p.describe()
    p.bind(0,1,Frame,"increment")
    p.submit(Frame(3))
    cp.cuda.get_current_stream().synchronize()
    result=cp.asnumpy(x)
    assert np.array_equal(result,np.full(32,4,np.float32)),result
    record.update(status="PASS",word_differences=0,words=32,wall_s=time.perf_counter()-t)
finally:
    record=locals().get("record",p.describe())
    (root/"receipts/cuda-probe.json").write_text(json.dumps(record,indent=2)+"\n")
    p.close()
print(json.dumps(record))
