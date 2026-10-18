"""Production OML versus the pinned WRF v4.6.1 column oracle."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
N, NF, STEPS = 112, 22, 16
STATE = ("tml","t0ml","hml","h0ml","huml","hvml","tmoml","tsk")
FORCING = ("hfx","lh","gsw","glw","uair","vair","ust","f","emiss","xland")

def ordered(a):
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    return monotone_fp32_key(a)

def metrics(a,b):
    assert np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite oracle or GPU state"
    return {"max_ulp":int(np.abs(ordered(a)-ordered(b)).max(initial=0)),
            "changed_words":int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
            "max_abs":float(np.abs(a.astype(float)-b.astype(float)).max(initial=0))}

def compare(root,mutate=False):
    import cupy as cp
    from gpuwm.core.surface_ocean import launch_ocean_init, launch_ocean_step
    from gpuwm.core import kernels
    if mutate:
        original=kernels._unit_text
        def changed(name,kernel_dir=kernels._KDIR):
            source=original(name,kernel_dir)
            if name=="oml":
                assert "float ust2=0.5f*ust[i];" in source
                source=source.replace("float ust2=0.5f*ust[i];","float ust2=__int_as_float(0x3f000001)*ust[i];")
            return source
        kernels._unit_text=changed
        kernels.get_kernel.cache_clear(); kernels.load_module.cache_clear()
    report={}; hashes={}
    receipt=json.loads((root / "oracle.json").read_text())
    pin=json.loads(Path(__file__).with_name("PIN.json").read_text())
    assert receipt["source_sha256"] == pin["source_sha256"], "oracle source pin mismatch"
    words=receipt["arrays"]
    def read_words(label):
        return np.frombuffer(bytes.fromhex(words[label]),dtype="<f4").copy()
    for b,hml0 in enumerate((50.,0.,-1.),1):
        def load(label):
            return read_words(f"branch{b}-{label}").reshape((N,NF),order="F")
        a=load("input")
        d={name:cp.array(a[:,k]) for k,name in enumerate((*STATE,*FORCING))}
        launch_ocean_init(d,hml0)
        got=np.stack([cp.asnumpy(d[name]) for name in STATE],axis=1)
        expected=load("init")[:,:8]
        for k,name in enumerate(STATE): report[f"init{b}/{name}"]=metrics(got[:,k].copy(),expected[:,k].copy())
        before={name:cp.asnumpy(d[name]).tobytes() for name in STATE}
        launch_ocean_init(d,hml0,start=False)
        assert all(cp.asnumpy(d[name]).tobytes()==before[name] for name in STATE),"restart init overwrote state"
        a=load("start")
        frames=read_words(f"branch{b}-steps").reshape((STEPS,NF,N)).transpose(0,2,1)
        groups={}
        for i in range(N): groups.setdefault(tuple(float(v) for v in a[i,18:21]),[]).append(i)
        gotframes=np.empty((STEPS,N,8),dtype=np.float32)
        for (gamma,relax,dt),indices in groups.items():
            d={name:cp.array(a[indices,k]) for k,name in enumerate((*STATE,*FORCING))}
            for step in range(STEPS):
                launch_ocean_step(d,dt=dt,gamma=gamma,relaxation_time=relax)
                for k,name in enumerate(STATE): gotframes[step,indices,k]=cp.asnumpy(d[name])
        for k,name in enumerate(STATE): report[f"steps{b}/{name}"]=metrics(gotframes[:,:,k].copy(),frames[:,:,k].copy())
        hashes[f"branch{b}"]={"gpu":hashlib.sha256(gotframes.tobytes()).hexdigest(),"wrf":hashlib.sha256(frames[:,:,:8].copy().tobytes()).hexdigest()}
    return {"fields":report,"hashes":hashes,"max_ulp":max(x["max_ulp"] for x in report.values())}

def main():
    p=argparse.ArgumentParser(); p.add_argument("oracle",type=Path); p.add_argument("--out",type=Path,required=True); p.add_argument("--mutation",action="store_true")
    a=p.parse_args(); r=compare(a.oracle,a.mutation)
    r.update(strict=os.environ.get("GPUWM_WRF_EXACT")=="1",columns=N,water_columns=96,steps=STEPS,default_tolerance={"max_ulp":0,"max_abs":0.0},mutation=a.mutation)
    r["bitwise"]=all(row["changed_words"] == 0 for row in r["fields"].values())
    r["pass"]=not r["bitwise"] if a.mutation else r["bitwise"]
    a.out.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({k:v for k,v in r.items() if k!="fields"}))
    return 0 if r["pass"] else 1
if __name__=="__main__": raise SystemExit(main())
