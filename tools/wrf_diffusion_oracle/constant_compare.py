"""Compare every native constant-K tendency word and retain all differences."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def compare(directory,result):
    import cupy as cp
    from gpuwm.core.diffusion import launch_add_diff2
    from gpuwm.core.kernels import module_source
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    meta=json.loads((directory/"constant-wrf471.json").read_text())
    fixture=np.load(directory/"constant-wrf471.npz")
    rows=[];outputs={}
    for case in meta["cases"]:
        prefix=f'c{case["case"]:03d}__'
        fx={key[len(prefix):]:fixture[key] for key in fixture.files if key.startswith(prefix)}
        f=cp.asarray(fx["field"]);t=cp.zeros_like(f)
        launch_add_diff2(f,t,case["kh"],case["kv"],case["dx"],case["dy"],fx["zf"],case["stagger"])
        # Explicit unit adapter: the primitive is uncoupled, WRF returns a
        # dry-mass-coupled tendency. No boundary word is masked or repaired.
        got=(t*cp.asarray(fx["coupling_mass"])[None]).get()
        want=fx["reference"]
        changed=got.view(np.uint32)!=want.view(np.uint32)
        ulps=fp32_ulp_distance(got,want)
        indices=np.flatnonzero(changed)
        row={**case,"words":int(got.size),"different_words":int(changed.sum()),"max_ulp":int(ulps.max(initial=0)),
             "max_absolute":float(np.max(np.abs(got.astype(np.float64)-want.astype(np.float64)),initial=0)),
             "first_differences":[{"index":[int(x) for x in np.unravel_index(int(i),got.shape)],"gpu_word":hex(int(got.view(np.uint32).flat[i])),"wrf_word":hex(int(want.view(np.uint32).flat[i]))} for i in indices[:6]]}
        rows.append(row)
        outputs[prefix+"gpu"]=got
        outputs[prefix+"different_mask"]=changed
    summaries=[]
    for routine in dict.fromkeys(r["routine"] for r in rows):
        selected=[r for r in rows if r["routine"]==routine]
        summary={"routine":routine,"cases":len(selected),"words":sum(r["words"] for r in selected),
                 "different_words":sum(r["different_words"] for r in selected),"max_ulp":max(r["max_ulp"] for r in selected)}
        summary["status"]="BIT-IDENTICAL" if not summary["different_words"] else "DIFFERENT"
        summaries.append(summary);print(json.dumps(summary))
    receipt={"gpu":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),"rows":rows,"summary":summaries,
             "fixture_sha256":{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ("constant-wrf471.npz","constant-wrf471.json")},
             "assembled_cuda_sha256":hashlib.sha256(module_source("diffusion").encode()).hexdigest(),
             "cuda_options":["-std=c++17"],"normalization":meta["normalization"]}
    result.write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    pin_path=result.with_name(result.stem.replace("-comparison","-words")+".npz")
    np.savez_compressed(pin_path,**outputs)
    if result.name=="constant-product-comparison.json" and result.parent.resolve()==directory.resolve():
        names=("constant-wrf471.npz","constant-wrf471.json",result.name,pin_path.name)
        (directory/"constant-oracle-sha256sums.txt").write_text("".join(hashlib.sha256((directory/n).read_bytes()).hexdigest()+"  "+n+"\n" for n in names), encoding="utf-8", newline="\n")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("fixture",type=Path);p.add_argument("result",type=Path)
    a=p.parse_args();compare(a.fixture,a.result)
