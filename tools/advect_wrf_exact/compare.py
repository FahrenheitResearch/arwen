"""Every defined total tendency word against compiled WRF advection fixtures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import ModuleType

PIN_RECEIPT={}


def verify_fixture_pins():
    """Reject comparison against altered input or compiled-reference archives."""
    from gpuwm.verify.advect_oracle import ADVECT_ORACLE_DIR
    root=Path(__file__).resolve().parents[2]
    manifest=ADVECT_ORACLE_DIR/"oracle-sha256sums.txt"
    checked={}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected,relative=line.split(maxsplit=1)
        if not relative.startswith("tests/data/wrf471_advect/"):
            continue
        if not (relative.endswith(".npz") or relative.endswith("/cases.json")):
            continue
        path=root/relative
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError("Advection input/reference archive changed: "+relative)
        checked[relative]=actual
    if not checked:
        raise ValueError("No pinned advection input/reference archives were checked")
    return {"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"files":checked}


def exact_pd_diagnostics(case,a,coord,fluxes,variant):
    """Expose defined optional channels without changing ordinary output words."""
    import cupy as cp
    import numpy as np
    from unittest.mock import patch
    from gpuwm.core import moist
    from gpuwm.core.kernels import module_source
    from gpuwm.verify.advect_oracle import _initial,SENTINEL
    source=module_source("pd_advection")
    start=source.index("void pd_renorm_apply(")
    body=source.index("{",start)
    signature=source[start:body]
    position=signature.rfind(")")
    signature=signature[:position]+",real* diag_h,real* diag_z"+signature[position:]
    source=source[:start]+signature+source[body:]
    stores='''
    diag_z[IDX3(k,j,i)]=0.0f-rdnw[k]*zbracket;
    if(!open_x || (i>=1 && i<=nx-2)) {
        real bracket=sx_r*fxc_r-sx_l*fxc_l+fxl[I3(k,j,i+1,ny,nx+1)]-fxl[I3(k,j,i,ny,nx+1)];
        diag_h[IDX3(k,j,i)]=0.0f-m*(dx_inv*bracket);
    }
    if(!open_y || (j>=1 && j<=ny-2)) {
        real bracket=sy_r*fyc_r-sy_l*fyc_l+fyl[I3(k,j+1,i,ny+1,nx)]-fyl[I3(k,j,i,ny+1,nx)];
        diag_h[IDX3(k,j,i)]=diag_h[IDX3(k,j,i)]-m*(dy_inv*bracket);
    }
'''
    marker='    tend_out[IDX3(k,j,i)]=result;'
    assert source.count(marker)==1
    source=source.replace(marker,marker+stores)
    h,z=(cp.full(case.shape,SENTINEL,cp.float32) for _ in range(2))
    m=case.metadata
    actual=cp.asarray(_initial(case,"tend_pd",case.shape))
    observed=actual.copy()
    def launch(tend):
        moist.launch_pd_renorm_apply(a["q0"],a["mu_old"],*fluxes,tend=tend,coord=coord,
            dx=m["dx"],dy=m["dy"],dt=m["dt"],msft=a["msftx"],has_msf=True,
            open_x=m["open_x"],open_y=m["open_y"])
    launch(actual)
    captured=[]
    with patch.object(moist,"get_kernel",lambda module,name:
         lambda grid,block,args:captured.append((grid,block,args))):
        launch(observed)
    grid,block,args=captured[0]
    module=cp.RawModule(code=source,options=("-std=c++17",))
    module.get_function("pd_renorm_apply")(grid,block,(*args,h,z))
    if not bool(cp.array_equal(actual.view(cp.uint32),observed.view(cp.uint32))):
        raise AssertionError("Optional channel observation changed native total tendency words")
    return h,z


def compare():
    global PIN_RECEIPT
    from gpuwm.wrf_exact import ADVECTION_ENABLED
    if not ADVECTION_ENABLED:
        raise RuntimeError("Set GPUWM_WRF_EXACT=1 and GPUWM_WRF_EXACT_ADVECTION=1 before import")
    PIN_RECEIPT=verify_fixture_pins()
    from gpuwm.verify import advect_oracle as original
    source=Path(original.__file__).read_text(encoding="utf-8")
    source=source.replace('coord = SimpleNamespace(**{key: a[key] for key in ("rdnw", "rdn", "fnm", "fnp", "c1h", "c2h")})',
        'coord = SimpleNamespace(**{key: a[key] for key in ("rdnw", "rdn", "fnm", "fnp", "c1h", "c2h")},\n'
        '                            mub2d=a["mub"],mup0=a["mu_perturbation"])')
    adapter=ModuleType("exact_advect_adapter")
    adapter.__file__=original.__file__
    adapter.__name__=original.__name__
    # Keep dataclass registration under the real module name for the reuse
    # of pinned fixture classes; no numerical reference code is replaced.
    exec(compile(source,original.__file__,"exec"),adapter.__dict__)
    adapter._pd_diagnostics=exact_pd_diagnostics
    result={}
    for case in adapter.load_advect_cases():
        outputs=adapter.advect_port_outputs(case)
        result[case.name]=adapter.measure_advect_parity(case,outputs)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=compare()
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    a.output.with_suffix(".pins.json").write_text(json.dumps(PIN_RECEIPT,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({case:{routine:{k:v for k,v in metric.items() if k in
        ("different_words","max_ulp","max_abs_difference")} for routine,metric in routines.items()}
        for case,routines in result.items()},indent=2))


if __name__=="__main__":main()
