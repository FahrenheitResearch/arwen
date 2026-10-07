"""Unchanged native deposition words and exact PTX for named division sites."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np


def ptx_control(baseline,destination):
    from gpuwm.core.kernels import module_source
    from tools.kernel_ptx_identity.compare import compile_ptx,ptx_functions
    from tools.literal_division_census import production_units,census_unit,rewrite_sites
    root=Path(__file__).resolve().parents[2]
    current=(root/"gpuwm/core/kernels/chem_drydep_wesely.cu").read_text(encoding="utf-8")
    source=module_source("chem_drydep_wesely")
    assert source.endswith(current)
    old=source[:-len(current)]+Path(baseline).read_text(encoding="utf-8")
    records=[]
    for arch in ("compute_89","compute_120"):
        for ftz in ("false","true"):
            options=("-std=c++17","-ftz="+ftz)
            before=compile_ptx(old,options,arch,"chem_drydep_wesely")
            after=compile_ptx(source,options,arch,"chem_drydep_wesely")
            a,b=ptx_functions(before),ptx_functions(after)
            changed={kind:[key for key in set(a[kind])|set(b[kind]) if a[kind].get(key)!=b[kind].get(key)]
                     for kind in ("entry","func")}
            records.append(dict(arch=arch,ftz=ftz,whole_ptx_identical=before==after,
                changed_functions=changed,before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                after_sha256=hashlib.sha256(after.encode()).hexdigest()))
            assert before==after,(arch,ftz,changed)
    unit=next(unit for unit in production_units(root) if unit.key=="kernels:chem_drydep_wesely")
    receipt=dict(reason="WFloat overloaded division already emits unflushed div.rn.f32; named wd_div preserves that operator",
        original_file_sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
        current_file_sha256=hashlib.sha256(current.encode()).hexdigest(),records=records,
        census_89=census_unit(unit,"compute_89"),rewrite_120=rewrite_sites(unit,"compute_120"))
    Path(destination).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"whole_ptx_identical":all(record["whole_ptx_identical"] for record in records),"comparisons":len(records)}))


def native_control(destination):
    import cupy as cp
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/"tests"))
    spec=importlib.util.spec_from_file_location("wesely_native_controls",root/"tests/test_chem_wesely_wrf471_parity.py")
    controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
    original=controls.max_ulp
    records=[];case_name=""
    def capture(actual,expected):
        result=original(actual,expected)
        a,b=np.ascontiguousarray(actual).view(np.uint32),np.ascontiguousarray(expected).view(np.uint32)
        records.append(dict(case=case_name,words=a.size,different_words=int(np.count_nonzero(a!=b)),max_ulp=result))
        return result
    controls.max_ulp=capture
    for case in controls.CASES:
        case_name=case.name
        controls.test_gpu_parity(case)
    receipt=dict(reference="unmodified compiled WRF-Chem v4.7.1 Wesely driver",
        device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        native_cases=len(controls.CASES),words=sum(row["words"] for row in records),
        different_words=sum(row["different_words"] for row in records),max_ulp=max(row["max_ulp"] for row in records),
        records=records,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ("gpuwm/core/kernels/chem_drydep_wesely.cu","gpuwm/core/chem_drydep_gas.py","tests/test_chem_wesely_wrf471_parity.py")})
    Path(destination).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("native_cases","words","different_words","max_ulp")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("native","ptx"));parser.add_argument("destination")
    parser.add_argument("--baseline")
    args=parser.parse_args()
    native_control(args.destination) if args.mode=="native" else ptx_control(args.baseline,args.destination)
