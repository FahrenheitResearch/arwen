"""Direct word comparisons of exact-mode native launches and compiled WRF.

The original oracle adapters explicitly translate full-theta fixtures.
This runner supplies their canonical perturbation words identically to both
compiled routines and retains every original stress case and output metric.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np


def exact_workspace_source(source):
    """Store already computed workspace words in every preprocessor branch."""
    prototype = "real cf1, real cf2, real cf3, real rdx, real rdy,"
    assert source.count(prototype) == 2
    source = source.replace(prototype,
        "real* __restrict__ oracle_t2, real* __restrict__ oracle_mu,\n" + prototype)
    counts = tuple(source.count(marker) for marker in
                   ("real t2_dn =", "real t2_up =", "real muave ="))
    assert (all(count in (2, 4) for count in counts)
            or counts in ((3, 3, 3), (5, 5, 3))), counts
    for marker, store in (("real t2_dn =", "oracle_t2[c] = t2_dn;"),
                          ("real t2_up =", "oracle_t2[h] = t2_up;"),
                          ("real muave =", "oracle_mu[c] = muts; oracle_mu[st + c] = muave;")):
        cursor = 0
        count = source.count(marker)
        for _ in range(count):
            start = source.index(marker, cursor)
            end = source.index(";", start) + 1
            source = source[:end] + "\n        " + store + source[end:]
            cursor = end + len(store) + 9
    return source


def exact_adapters():
    """Reuse pinned packing/reference calls, selecting native representation."""
    from gpuwm.verify import smallstep_horizontal_oracle as h
    from gpuwm.verify import smallstep_vertical_oracle as v
    hs = Path(h.__file__).read_text(encoding="utf-8")
    vs = Path(v.__file__).read_text(encoding="utf-8")
    hs = hs.replace("muu=_face_mass(mu,1,periodic); muv=_face_mass(mu,0,periodic)",
        "from tools.smallstep_wrf471_oracle.bookkeeping import wrf_face_mass\n"
        "    muu=wrf_face_mass(data['mup'],data['mub2d'],1,periodic)\n"
        "    muv=wrf_face_mass(data['mup'],data['mub2d'],0,periodic)")
    hs = hs.replace('mass["ww_1"][...]=ref["ww"]',
        'mass["ww_1"][...]=ref["ww"]\n'
        '    data["ww_ref"]=oracle.extract(ref["ww"],data["ww_pp"].shape)')
    hs = hs.replace('data["rth_t"] = data["native_ft"] + F(300)*data["c1h"][:,None,None]*dmdt_ref[None]/data["msft"][None]',
                    'data["rth_t"] = data["native_ft"].copy()')
    hs = hs.replace('    state=_device_state(data)\n',
        '    state=_device_state(data)\n'
        '    offset=F(300)*data["c1h"][:,None,None]*data["mu_pp"][None]\n'
        '    state.th_pp[...] = cp.asarray(data["th_pp"]-offset)\n'
        '    state.buffers["rk_ww"] = cp.asarray(data["ww_ref"])\n')
    hs = hs.replace('    if not full_theta_probe:\n        got["th_pp"] -=',
                    '    if False:\n        got["th_pp"] -=')
    hs = hs.replace('        acoustic_substep_explicit(state,cfg,.25,getattr(cfg,"first",True),\n'
                    '                                 cq=cq,mudf=mudf)',
        '        from gpuwm.core import acoustic\n'
        '        from unittest.mock import patch\n'
        '        native_get=acoustic.get_kernel\n'
        '        def select(module,name):\n'
        '            if name=="advance_exact_frame_mu_t":\n'
        '                return lambda grid,block,args: None\n'
        '            return native_get(module,name)\n'
        '        with patch.object(acoustic,"get_kernel",select):\n'
        '            acoustic_substep_explicit(state,cfg,.25,getattr(cfg,"first",True),\n'
        '                                     cq=cq,mudf=mudf)')
    # Exact-mode fixtures start with the same canonical words the original
    # reference adapter supplied, before launching either implementation.
    vs = vs.replace('    return s, cfg\n',
        '    offset=s.c1h[:nz,None,None]*s.mu_pp[None]*np.float32(300)\n'
        '    s.th_pp[...] -= offset\n'
        '    old=s.scratch((nz,ny,nx),"acoustic_th_pp_old")\n'
        '    old_mu=s.scratch((ny,nx),"acoustic_mu_pp_old")\n'
        '    old[...] -= s.c1h[:nz,None,None]*old_mu[None]*np.float32(300)\n'
        '    return s, cfg\n')
    vs = vs.replace('t2 = before["th_pp"] - (c1h * before["mu_pp"]) * theta_offset',
                    't2 = before["th_pp"].copy()')
    vs = vs.replace('t2old = old_th - (c1h * old_mu) * theta_offset',
                    't2old = old_th.copy()')
    hm, vm = ModuleType("smallstep_exact_horizontal"), ModuleType("smallstep_exact_vertical")
    hm.__file__, vm.__file__ = h.__file__, v.__file__
    exec(compile(hs, h.__file__, "exec"), hm.__dict__)
    exec(compile(vs, v.__file__, "exec"), vm.__dict__)
    return hm, vm


def exact_bookkeeping_adapter():
    from gpuwm.verify import smallstep_bookkeeping_oracle as b
    source=Path(b.__file__).read_text(encoding="utf-8")
    source=source.replace('dx=float(fixture["dx"]), dy=float(fixture["dy"]),',
        'dx=float(fixture["dx"]), dy=float(fixture["dy"]),\n'
        '        rk_step=int(fixture.get("rk_step",1)),')
    source=source.replace('values["native_th_pp"] = values["th_pp"] - offset',
                          'values["native_th_pp"] = values["th_pp"].copy()')
    source=source.replace('    return state\n',
        '    if "hdiab_dt" in fixture:\n'
        '        state.th_pp[...] = cp.asarray(fixture["in_th_pp"]\n'
        '          -(fixture["in_c1h"][:,None,None]*fixture["in_mu_pp"][None])*np.float32(300))\n'
        '    return state\n')
    # Observe the final masses used by production, after exact-mode overrides.
    source=source.replace('needle = "real mtf = rn_mul(0.5f, rn_add(mt_a, mt_b));"',
        'needle = "real ct = rn_add(rn_mul(c1h[k], mtf), c2h[k]);"')
    source=source.replace('needle = "real mnf = rn_mul(0.5f, rn_add(mna, mnb));"',
        'needle = "real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);"')
    module=ModuleType("smallstep_exact_bookkeeping")
    module.__file__=b.__file__
    exec(compile(source,b.__file__,"exec"),module.__dict__)
    return module


def compare_frame_driver(library):
    """The extra driver launch is checked against compiled spec_bdyupdate."""
    import ctypes
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    from gpuwm.verify.smallstep_oracle import word_metrics
    from tools.smallstep_wrf471_oracle.horizontal_measure import horizontal_cases
    h,_=exact_adapters()
    native=ctypes.CDLL(str(library)).oracle_frame
    native.argtypes=[ctypes.c_void_p]*8
    result={}
    for name,raw,metadata,options in horizontal_cases():
        if any(options.get(x) for x in ("full_theta_probe","no_fma_probe")):
            continue
        data,cfg=h.make_horizontal_state(raw,metadata,**options)
        if not cfg.specified:
            continue
        theta=data["th_pp"]-np.float32(300)*data["c1h"][:,None,None]*data["mu_pp"][None]
        # Include nonzero boundary tendencies and signed-zero corners.
        rmu=np.linspace(-.03125,.03125,data["mu_pp"].size,dtype=np.float32).reshape(data["mu_pp"].shape)
        rth=np.linspace(-.0625,.0625,theta.size,dtype=np.float32).reshape(theta.shape)
        rmu[[0,-1],:]=np.float32(0);rth[:,[0,-1],:]=np.float32(0)
        mu_device,theta_device=cp.asarray(data["mu_pp"]),cp.asarray(theta)
        get_kernel("acoustic","advance_exact_frame_mu_t")(
            ((cfg.ny*cfg.nx+255)//256,),(256,),
            (mu_device,theta_device,cp.asarray(rmu),cp.asarray(rth),np.float32(.25),
             np.int32(cfg.spec_zone),np.int32(0),np.int32(cfg.nz),np.int32(cfg.ny),np.int32(cfg.nx)))
        result[name]={}
        for label,field,tendency,got in (("mu",data["mu_pp"][None],rmu[None],mu_device),
                                           ("theta",theta,rth,theta_device)):
            arg=np.asfortranarray(field.transpose(2,0,1))
            tend=np.asfortranarray(tendency.transpose(2,0,1))
            scalars=[np.asarray(value,dtype=dtype) for value,dtype in
                     ((.25,np.float32),(cfg.nx,np.int32),(cfg.ny,np.int32),
                      (field.shape[0],np.int32),(cfg.spec_zone,np.int32),(0,np.int32))]
            args=[arg,tend,*scalars]
            native(*[ctypes.c_void_p(x.ctypes.data) for x in args])
            expected=arg.transpose(1,2,0)
            actual=cp.asnumpy(got)
            if label=="mu":expected=expected[0]
            result[name][label]=word_metrics(actual,expected)
    return result


def compare(library):
    from gpuwm.wrf_exact import ENABLED
    if not ENABLED:
        raise RuntimeError("Set GPUWM_WRF_EXACT=1 before starting this comparison")
    from tools.smallstep_wrf471_oracle.horizontal_measure import horizontal_cases
    from tools.smallstep_wrf471_oracle import vertical_workspace
    h, v = exact_adapters()
    result = {"horizontal": {}, "vertical": {}, "bookkeeping": {}}
    with patch.object(vertical_workspace, "workspace_source", exact_workspace_source):
        for name, raw, metadata, options in horizontal_cases():
            if options.get("full_theta_probe") or options.get("no_fma_probe"):
                continue
            result["horizontal"][name] = h.measure_horizontal_case(raw, metadata, library, **options)
        for name, raw, metadata in v.vertical_cases():
            result["vertical"][name] = v.measure_vertical_case(raw, metadata, library)
    b=exact_bookkeeping_adapter()
    manifest=json.loads((b.BOOKKEEPING_DIR/"manifest.json").read_text())
    for name, item in manifest["files"].items():
        fixture=b.load_bookkeeping(b.BOOKKEEPING_DIR/name)
        result["bookkeeping"][name]=b.measure_bookkeeping(fixture,item["routine"])
    frame_library=os.environ.get("GPUWM_SMALLSTEP_FRAME_LIB")
    if frame_library:
        result["frame_driver"]=compare_frame_driver(frame_library)
    return result


def differences(value, prefix=""):
    if isinstance(value, dict):
        if "different_words" in value or "differing_words" in value:
            count=value.get("different_words",value.get("differing_words"))
            if count:
                yield prefix, count, value["max_ulp"], value["max_abs"]
        else:
            for key, child in value.items():
                yield from differences(child, prefix+"/"+key)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--library",default=os.environ.get("GPUWM_SMALLSTEP_ORACLE_LIB"),required=False)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=compare(a.library)
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    remaining=list(differences(result))
    print(json.dumps({"remaining":remaining,"cases":{k:len(v) for k,v in result.items()}},indent=2))
    return int(bool(remaining))


if __name__=="__main__":
    raise SystemExit(main())
