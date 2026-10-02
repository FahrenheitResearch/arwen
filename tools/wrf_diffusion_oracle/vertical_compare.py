"""Measure every output word against the compiled WRF fixtures, no tolerance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gpuwm.verify.diffusion_oracle import word_comparison
from vertical_gpu import vertical_gpu,tke_gpu,vertical_driver_gpu


def compare(fixture_dir,output,diagnostic_no_fma=False,diagnostic_reference_metrics=False,
            diagnostic_glibc_power=False):
    import cupy as cp
    fixture_dir = Path(fixture_dir)
    manifest = json.loads((fixture_dir/"vertical-fixtures.json").read_text())
    diagnostic_module = None
    if diagnostic_no_fma or diagnostic_reference_metrics or diagnostic_glibc_power:
        import gpuwm.core.dycore as dycore
        import gpuwm.core.kernels as kernels
        source = kernels.module_source("smag2d")
        if diagnostic_reference_metrics:
            from horizontal_compare import reference_metric_source
            source = reference_metric_source(source)
        if diagnostic_glibc_power:
            header = (Path(kernels.__file__).parent/"glibc_flt32.cuh").read_text()
            source = source.replace("// gpuwm/core/kernels/smag2d.cu",header+"\n// gpuwm/core/kernels/smag2d.cu",1)
            original_power = "chm * coefc * tketmp * sqrtf(tketmp) / l"
            if source.count(original_power)!=1:
                raise ValueError("TKE dissipation power site changed")
            source = source.replace(original_power,"chm * coefc * gfk_pow(tketmp, 1.5f) / l")
        options = ("-std=c++17","--fmad=false") if diagnostic_no_fma else ("-std=c++17",)
        diagnostic_module = cp.RawModule(code=source,options=options)
        original_kernel = kernels.get_kernel
        controlled_kernel = lambda name,symbol: diagnostic_module.get_function(symbol) if name=="smag2d" else original_kernel(name,symbol)
        dycore.get_kernel = controlled_kernel
        kernels.get_kernel = controlled_kernel
    cases = []
    for case in manifest["cases"]:
        path = fixture_dir/case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=case["sha256"]:
            raise ValueError(f"Fixture changed: {path.name}")
        with np.load(path,allow_pickle=False) as payload:
            if diagnostic_reference_metrics:
                metric_values = [cp.asarray(payload["metric_"+k]) for k in ("rdz","rdzw","rho","zx","zy")]
                diagnostic_module.get_function("oracle_set_metrics")((1,),(1,),tuple(metric_values))
            arrays = {k[3:]:payload[k] for k in payload.files if k.startswith("in_")}
            meta = case["metadata"]
            coef = {k:payload["coefficient_"+k] for k in ("kmh","kmv","khv")}
            deform = {k:payload["deformation_"+k] for k in ("d11","d22","d12")}
            got = vertical_gpu(arrays,meta,kmv=coef["kmv"],khv=coef["khv"],var=arrays["qv"])
            got["vertical_s_theta"] = vertical_gpu(arrays,meta,kmv=coef["kmv"],khv=coef["khv"],
                var=arrays["thp"],full_theta=True)["vertical_s"]
            got["vertical_s_tke"] = vertical_gpu(arrays,meta,kmv=coef["kmv"],khv=coef["kmv"],
                var=arrays["tke"],doing_tke=True)["vertical_s"]
            for isfflx in (0,1,2):
                terms = tke_gpu(arrays,meta,deform,coef,payload["bn2"],isfflx=isfflx,
                                c_k=meta["c_k"],dt=meta["dt"])
                got.update({f"{k}_flux{isfflx}":v for k,v in terms.items() if k.startswith("tke_")})
            if not meta["bx"] and not meta["by"]:
                for km_opt in (2,4):
                    driver_coef = {k:payload[f"driver{km_opt}_coefficient_{k}"] for k in ("kmh","kmv","khv")}
                    for isfflx in (0,1,2):
                        driver = vertical_driver_gpu(arrays,meta,driver_coef,km_opt=km_opt,isfflx=isfflx)
                        got.update({f"driver{km_opt}_{k}_flux{isfflx}":v for k,v in driver.items()})
            fields = {k:word_comparison(v,payload["ref_"+k]) for k,v in got.items()}
            # Exact observed words allow independent diagnostics of every
            # mismatch. The JSON statistics do not substitute for arrays.
            observed = Path(output).parent / (path.stem+"-gpu.npz")
            np.savez_compressed(observed,**got)
            cases.append({"case":case["case"],"fields":fields,
                          "array_sha256":{k:hashlib.sha256(np.ascontiguousarray(v,dtype="<f4").tobytes()).hexdigest()
                                          for k,v in got.items()},"gpu_file":observed.name,
                          "gpu_sha256":hashlib.sha256(observed.read_bytes()).hexdigest()})
    result = {"device":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
              "cupy_version":cp.__version__,"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),
              "cases":cases,"diagnostic_no_fma":diagnostic_no_fma,
              "diagnostic_reference_metrics":diagnostic_reference_metrics,
              "diagnostic_glibc_power":diagnostic_glibc_power}
    from gpuwm.core.kernels import module_source
    result["kernel_source_sha256"] = hashlib.sha256(module_source("smag2d").encode()).hexdigest()
    result["fixture_manifest_sha256"] = hashlib.sha256((fixture_dir/"vertical-fixtures.json").read_bytes()).hexdigest()
    if diagnostic_module is not None:
        kernels.get_kernel = original_kernel
        dycore.get_kernel = original_kernel
    Path(output).write_text(json.dumps(result,indent=2)+"\n", encoding="utf-8", newline="\n")
    aggregate = {}
    for case in cases:
        for key,value in case["fields"].items():
            bucket = aggregate.setdefault(key,{"cases":0,"words":0,"different_words":0,"max_ulp":0,"nonfinite_words":0})
            bucket["cases"] += 1
            for k in ("words","different_words","nonfinite_words"):
                bucket[k] += value[k]
            bucket["max_ulp"] = max(bucket["max_ulp"],value["max_ulp"])
    print(json.dumps(aggregate,indent=2))
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--diagnostic-no-fma",action="store_true")
    parser.add_argument("--diagnostic-reference-metrics",action="store_true")
    parser.add_argument("--diagnostic-glibc-power",action="store_true")
    args = parser.parse_args()
    compare(**vars(args))
