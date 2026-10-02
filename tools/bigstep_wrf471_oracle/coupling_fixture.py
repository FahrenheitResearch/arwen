"""Feed real-state/edge inputs to the actual compiled Fortran routines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from gpuwm.verify.bigstep_coupling_oracle import coupling_cases, SPECIES


def pad2(a, nx, ny, periodic):
    out = np.zeros((ny+2, nx+2), np.float32)
    h,w = a.shape
    out[1:h+1,1:w+1] = a
    if w == nx:
        out[1:h+1,0] = a[:,-1 if periodic else 0]
        out[1:h+1,-1] = a[:,0 if periodic else -1]
    else:
        out[1:h+1,0] = a[:,-2 if periodic else 0]
    if h == ny:
        out[0] = out[ny if periodic else 1]
        out[-1] = out[1 if periodic else ny]
    else:
        out[0] = out[ny if periodic else 1]
    return out


def pad3(a,nx,ny,nz,periodic):
    out = np.zeros((ny+2,nz+1,nx+2),np.float32)
    for k in range(a.shape[0]):
        out[:,k,:] = pad2(a[k],nx,ny,periodic)
    if a.shape[0] == nz:
        out[:,nz,:] = out[:,nz-1,:]
    return out


def pad1(a,nz):
    return np.pad(a,(0,nz+1-len(a)),mode="edge").astype(np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("build",type=Path)
    a=p.parse_args()
    arrays={}
    receipt={"schema":"bigstep-coupling-wrf471-v1","cases":[],"build":json.loads((a.build/"coupling-build-receipt.json").read_text())}
    for case in coupling_cases(a.directory):
        f,c=case["fields"],case["control"]
        nz,ny,nx=f["T"].shape
        name=case["name"]
        def p2(v): return pad2(v,nx,ny,c["periodic"])
        def p3(v): return pad3(v,nx,ny,nz,c["periodic"])
        inp=a.build/(name+".input.bin")
        out=a.build/(name+".output.bin")
        with inp.open("wb") as stream:
            np.array([nx,ny,nz,7,int(c["periodic"]),c["ieva"],c["w_damping"]],np.int32).tofile(stream)
            np.array([c[k] for k in ("rdx","rdy","dt","dampcoef","zdamp","w_crit_cfl")],np.float32).tofile(stream)
            for k in ("MU","MUB","MAPFAC_MX","MAPFAC_MY","MAPFAC_UX","MAPFAC_UY","MAPFAC_VX","MAPFAC_VY","MF_VX_INV"):
                p2(f[k]).tofile(stream)
            for k in ("C1H","C2H","C1F","C2F","DNW","RDNW"):
                pad1(f[k],nz).tofile(stream)
            for k in ("ub","vb","tb","zb"):
                case["bases"][k].tofile(stream)
            for k in ("U","V","W"):
                p3(f[k]).tofile(stream)
            p3(case["wwd"]).tofile(stream)
            for k in ("PH","PHB","T","T_INIT"):
                p3(f[k]).tofile(stream)
            np.zeros((ny+2,nz+1,nx+2),np.float32).tofile(stream)  # Registry unused first slot
            for k in SPECIES:
                p3(f[k]).tofile(stream)
            for unused in range(5):
                np.zeros((ny+2,nz+1,nx+2),np.float32).tofile(stream)
        subprocess.run([str((a.build/"coupling_run").resolve()),str(inp.resolve()),str(out.resolve())],check=True)
        raw=np.fromfile(out,np.float32)
        index=0
        def take(count):
            nonlocal index
            value=raw[index:index+count]; index+=count
            return value
        for k,shape in (("muu",(ny,nx+1)),("muv",(ny+1,nx))):
            value=take((ny+2)*(nx+2)).reshape(ny+2,nx+2)
            h,w=shape
            arrays[name+"/"+k]=value[1:h+1,1:w+1].copy()
        for k,shape in (("cqu",(nz,ny,nx+1)),("cqv",(nz,ny+1,nx)),("cqw",(nz+1,ny,nx)),("cqwr",(nz+1,ny,nx)),("ww",(nz+1,ny,nx)),("php",(nz,ny,nx)),("rwd",(nz+1,ny,nx))):
            value=take((ny+2)*(nz+1)*(nx+2)).reshape(ny+2,nz+1,nx+2).transpose(1,0,2)
            z,h,w=shape
            arrays[name+"/"+k]=value[:z,1:h+1,1:w+1].copy()
        arrays[name+"/maxv"]=take(1).reshape(()).copy()
        arrays[name+"/maxh"]=take(1).reshape(()).copy()
        for k,shape in (("rayleigh_ru",(nz,ny,nx+1)),("rayleigh_rv",(nz,ny+1,nx)),("rayleigh_rw",(nz+1,ny,nx)),("rayleigh_rt",(nz,ny,nx)),("php_ru",(nz,ny,nx+1)),("php_rv",(nz,ny+1,nx))):
            value=take((ny+2)*(nz+1)*(nx+2)).reshape(ny+2,nz+1,nx+2).transpose(1,0,2)
            z,h,w=shape
            arrays[name+"/"+k]=value[:z,1:h+1,1:w+1].copy()
        assert index==len(raw),(index,len(raw))
        receipt["cases"].append({"name":name,"input_sha256":hashlib.sha256(inp.read_bytes()).hexdigest(),"output_sha256":hashlib.sha256(out.read_bytes()).hexdigest()})
    np.savez_compressed(a.directory/"coupling.npz",**arrays)
    receipt["fixture_sha256"]=hashlib.sha256((a.directory/"coupling.npz").read_bytes()).hexdigest()
    receipt["arrays"]={k:{"shape":list(v.shape),"sha256":hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()}
    # Build paths are reproducibility inputs but stay out of packaged text.
    receipt["build"].pop("commands")
    (a.directory/"coupling-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    main()
