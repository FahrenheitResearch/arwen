"""Emit matching C++ and native Fortran wrappers around complete WRF drivers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re

from translate import ROOT, Variable, bounds, capacity, declaration, define, statements, VALUES

spec = importlib.util.spec_from_file_location("lake_schema", ROOT / "gpuwm/core/lake_schema.py")
schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema)


def routines():
    result, current = {}, None
    for n, line in statements():
        m = re.match(r"subroutine (lake|lakeini)\s*\((.*)\)", line)
        if m:
            from translate import split
            current = result[m[1]] = dict(args=split(m[2]), vars={})
        elif line.startswith("end subroutine"):
            current = None
        elif current is not None:
            vs = declaration(line)
            if vs:
                current["vars"].update((v.name, v) for v in vs)
    return result


def indices(var):
    from itertools import product
    ranges = []
    for dim in var.dims:
        lo, hi = bounds(dim)
        ranges.append(range(eval(lo, {"__builtins__":{}}, VALUES), eval(hi, {"__builtins__":{}}, VALUES)+1))
    return [tuple(reversed(v)) for v in product(*reversed(ranges))]


def generate(kind):
    routine = routines()["lakeini" if kind == "init" else "lake"]
    vs = {k:v for k,v in routine["vars"].items() if k in routine["args"]}
    cpp, ftn = [], []
    if kind == "init":
        signature = "int n,const float* seed,float* columns,float* statics,int use_depth,int depth_flag,float default_depth,int* errors"
        fparams = "n,seed,columns,statics,use_depth,depth_flag,default_depth,errors"
    else:
        signature = "int n,const float* forcing,float* columns,const float* statics,float* output,float dt,int* errors"
        fparams = "n,forcing,columns,statics,output,dt,errors"
    cpp += [f'extern "C" __global__ void lake_{kind}_columns({signature}) {{',
            "#ifdef __CUDACC__", "int col = blockDim.x*blockIdx.x+threadIdx.x; if(col>=n)return;", "#else", "for(int col=0;col<n;++col) {", "#endif", "LakeColumn model;"]
    ftn += [f"subroutine lake_{kind}_columns({fparams}) bind(c)", "use iso_c_binding",
            "use module_sf_lake, only: wrf_lake=>lake,wrf_lakeini=>lakeini", "implicit none",
            "integer(c_int),value :: n", "integer(c_int) :: errors(n)",
            f"real(c_float) :: columns(n,{schema.LAKE_STATE_WORDS}), statics(n,{schema.LAKE_STATIC_WORDS})"]
    if kind=="init":
        ftn += ["integer(c_int),value :: use_depth,depth_flag", "real(c_float),value :: default_depth", "real(c_float) :: seed(n,5)"]
    else:
        ftn += ["real(c_float),value :: dt", f"real(c_float) :: forcing(n,{schema.LAKE_FORCING_WORDS}),output(n,{schema.LAKE_OUTPUT_WORDS})"]
    ftn += ["integer :: col"]
    for v in vs.values():
        # Bound constants are local values rather than module imports.
        dims = [str(eval(b,{"__builtins__":{}},VALUES)) for d in v.dims for b in bounds(d)]
        cppv = Variable(v.name, v.typ, [f"{dims[i]}:{dims[i+1]}" for i in range(0,len(dims),2)])
        cpp.append(define(cppv,vs))
        dtype = {"float":"real", "double":"real(kind=8)","int":"integer", "bool":"logical"}[v.typ]
        shape = "(" + ",".join(cppv.dims) + ")" if v.dims else ""
        ftn.append(f"{dtype} :: {v.name}{shape}")
    ftn += ["do col=1,n"]

    def put(c, f):
        cpp.append(c); ftn.append(f)

    for name,v in vs.items():
        value = "0" if v.typ != "bool" else "false"
        put(f"{name}.fill({value});" if v.dims else f"{name}={value};",f"{name}="+(".false." if value=="false" else value))
    for name in vs:
        if name in VALUES and not vs[name].dims:
            put(f"{name}={VALUES[name]};",f"{name}={VALUES[name]}")
    values = dict(lakemask=1,ivgtyp=17,iswater=17,xland=2,ht=100,lake_min_elev=5,xice_threshold=0.5)
    if kind == "init": values.update(lakeflag=1,use_lakedepth="use_depth",lake_depth_flag="depth_flag",lakedepth_default="default_depth")
    else: values.update(dtbl="dt")
    for name,val in values.items():
        put(f"{name}.fill({val});" if vs[name].dims else f"{name}={val};",f"{name}={val}")

    def transfer(layout, packed, load):
        offset = 0
        for name,size in layout:
            v = vs[name]
            sub = indices(v)
            for j in range(size):
                cname=f"{name}.data[{j}]"
                fname=name+"("+",".join(str(x) for x in sub[j])+")"
                cp=f"{packed}[{offset+j}*n+col]"
                fp=f"{packed}(col,{offset+j+1})"
                put(f"{cname}={cp};" if load else f"{cp}={cname};", f"{fname}={fp}" if load else f"{fp}={fname}")
            offset += size
    if kind=="init":
        transfer(schema.LAKE_SEED_LAYOUT,"seed",True)
    else:
        transfer(schema.LAKE_STATE_LAYOUT,"columns",True)
        transfer(schema.LAKE_STATIC_LAYOUT,"statics",True)
        transfer(schema.LAKE_FORCING_LAYOUT,"forcing",True)
    cpp.append("model."+("lakeini" if kind=="init" else "lake")+"("+",".join(routine["args"])+");")
    cpp.append("errors[col]=model.error; if(!model.error) {")
    ftn.append("call wrf_"+("lakeini" if kind=="init" else "lake")+"("+", &\n".join(routine["args"])+")")
    transfer(schema.LAKE_STATE_LAYOUT,"columns",False)
    if kind=="init": transfer(schema.LAKE_STATIC_LAYOUT,"statics",False)
    else: transfer(schema.LAKE_OUTPUT_LAYOUT,"output",False)
    cpp += ["}", "#ifndef __CUDACC__", "}", "#endif", "}"]
    ftn += ["errors(col)=0", "enddo", "end subroutine"]
    return "\n".join(cpp), "\n".join(ftn)


if __name__=="__main__":
    pairs = [generate("init"),generate("step")]
    cpp = "// Generated complete WRF driver column entry points.\n"
    (ROOT / "gpuwm/core/kernels/lake.cu").write_text(cpp+"\n".join(p[0] for p in pairs)+"\n",encoding="utf-8",newline="\n")
    Path(__file__).with_name("column_wrapper.F90").write_text("\n".join(p[1] for p in pairs)+"\n",encoding="utf-8",newline="\n")
