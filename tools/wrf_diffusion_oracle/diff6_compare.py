"""Measure the product launch path against every compiled-Fortran output word."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import numpy as np


def measured_words(got, want):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    got=np.asarray(got,dtype=np.float32);want=np.asarray(want,dtype=np.float32)
    gu=got.view(np.uint32);wu=want.view(np.uint32)
    distance=fp32_ulp_distance(got,want)
    changed=gu!=wu
    flat=np.flatnonzero(changed)
    return {"words":int(got.size),"different_words":int(changed.sum()),"max_ulp":int(distance.max(initial=0)),
            "max_absolute":float(np.max(np.abs(got.astype(np.float64)-want.astype(np.float64)),initial=0)),
            "first_differences":[{"index":[int(x) for x in np.unravel_index(int(i),got.shape)],"gpu_word":hex(int(gu.flat[i])),"wrf_word":hex(int(wu.flat[i]))} for i in flat[:4]]}


def compare(directory, result):
    import cupy as cp
    from gpuwm.core.dycore import launch_diff6
    metadata=json.loads((directory/"diff6-wrf471.json").read_text())
    fixture=np.load(directory/"diff6-wrf471.npz")
    rows=[]
    for settings in metadata["cases"]:
        case=settings["case"];prefix=f"c{case:03d}__"
        inputs={key[len(prefix):]:fixture[key] for key in fixture.files if key.startswith(prefix)}
        dev={key:cp.asarray(value) for key,value in inputs.items() if key not in ("reference","latitude")}
        tendency=dev["tendency"].copy()
        stagger={"u":"x","v":"y","w":"z"}.get(settings["name"],"")
        launch_diff6(dev["field"],tendency,dev["mut"],dev["c1"],dev["c2"],settings["factor"],settings["dt"],settings["opt"],
                     stagger,phb=dev["phb"],msfu=dev["mapu"],msfv=dev["mapv"],msft=dev["mapm"],slopeopt=settings["slopeopt"],
                     thresh=settings["thresh"],dx=settings["dx"],dy=settings["dy"],bnd_x=bool(settings["boundary_mode"]),bnd_y=bool(settings["boundary_mode"]))
        if settings["boundary_mode"]:
            tendency[:,:,:3]=0;tendency[:,:,-3:]=0;tendency[:,:3,:]=0;tendency[:,-3:,:]=0
        got=tendency.get()
        row={**settings,**measured_words(got,inputs["reference"])}
        rows.append(row)
    from gpuwm.core.kernels import module_source
    summary={"cases":len(rows),"words":sum(r["words"] for r in rows),"different_words":sum(r["different_words"] for r in rows),
             "max_ulp":max(r["max_ulp"] for r in rows),"gpu":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),"rows":rows}
    summary["fixture_sha256"]={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ("diff6-wrf471.npz","diff6-wrf471.json")}
    summary["assembled_cuda_sha256"]={n:hashlib.sha256(module_source(n).encode()).hexdigest() for n in ("diff6","diff6_seam")}
    summary["launcher_sha256"]=hashlib.sha256(inspect.getsource(launch_diff6).encode()).hexdigest()
    summary["cuda_options"]=["-std=c++17"]
    summary["cupy_version"]=cp.__version__
    summary["cuda_runtime_version"]=cp.cuda.runtime.runtimeGetVersion()
    summary["cuda_driver_version"]=cp.cuda.runtime.driverGetVersion()
    result.write_text(json.dumps(summary,indent=2)+"\n", encoding="utf-8", newline="\n")
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        selected=[r for r in rows if r["scenario"]==scenario]
        print(f'{scenario}: cases={len(selected)} differing={sum(r["different_words"] for r in selected)} max_ulp={max(r["max_ulp"] for r in selected)}')
    print(f'cases={summary["cases"]} words={summary["words"]} different={summary["different_words"]} max_ulp={summary["max_ulp"]}')


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("fixture",type=Path);p.add_argument("result",type=Path)
    a=p.parse_args();compare(a.fixture,a.result)
