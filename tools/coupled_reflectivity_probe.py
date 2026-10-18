"""Measure WSM6 radar intermediates from the already captured real call.

Only diagnostic builds replace the final output with a preceding value.
The original build is run too. No probe enters the forecast launcher.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coupled_dycore_receipt import read_wrf
from coupled_trace_receipt import digest

DRIVER = """program radar_probe
 use ccpp_kind_types, only: kind_phys
 use module_model_constants, only: rhoair0,rhowater,rhosnow,cliq,cpv
 use mp_wsm6, only: mp_wsm6_init,refl10cm_wsm6
 implicit none
 integer :: u,v,ncol,nz,c,err
 character(1024) :: inp,outp
 character(256) :: msg
 real(kind_phys),allocatable :: a(:,:,:),dbz(:,:)
 call get_command_argument(1,inp)
 call get_command_argument(2,outp)
 open(newunit=u,file=trim(inp),access='stream',form='unformatted',status='old')
 read(u) ncol,nz
 allocate(a(nz,ncol,6),dbz(nz,ncol))
 read(u) a
 close(u)
 call mp_wsm6_init(rhoair0,rhowater,rhosnow,cliq,cpv,0,msg,err)
 if(err/=0) error stop 'initialization failed'
 do c=1,ncol
  call refl10cm_wsm6(a(:,c,1),a(:,c,2),a(:,c,3),a(:,c,4),a(:,c,5),a(:,c,6),dbz(:,c),1,nz)
 enddo
 open(newunit=v,file=trim(outp),access='stream',form='unformatted',status='replace')
 write(v) dbz
 close(v)
end program
"""

STOCK = "dBZ(k) = 10.*log10((ze_rain(k)+ze_snow(k)+ze_graupel(k))*1.d18)"
GPU = "const float dbz = (float)(10.0 * log10((double)zsum * 1.0e18));"
EXPRESSIONS = {
    "reflectivity": (None, None),
    "zsum": ("ze_rain(k)+ze_snow(k)+ze_graupel(k)", "zsum"),
    "ze_rain": ("ze_rain(k)", "ze_rain"),
    "ze_snow": ("ze_snow(k)", "ze_snow[k]"),
    "ze_graupel": ("ze_graupel(k)", "ze_graupel[k]"),
}


def capture(directory, arm, shape):
    if arm == "wrf":
        get = lambda name: read_wrf(directory, 1, 0, "mp_wsm6_out", name, shape)
        fields = [get(n) for n in ("qv_curr", "qr_curr", "qs_curr", "qg_curr")]
        fields += [(get("th") * get("pi_phy")).astype(np.float32), get("p")]
    else:
        paths = list(directory.glob("*_s001_*_mp_wsm6_out.npz"))
        if len(paths) != 1:
            raise ValueError("one captured WSM6 output is required")
        with np.load(paths[0], allow_pickle=False) as z:
            fields = [z["x_"+n] for n in ("qv", "qr", "qs", "qg")]
            fields += [(z["x_theta"] * z["x_pii"]).astype(np.float32), z["x_pressure"]]
    if any(v.shape != shape or v.dtype != np.float32 for v in fields):
        raise ValueError("captured radar fields must have the complete float32 domain shape")
    return fields


def stock_outputs(fields, build, out):
    nz, ny, nx = fields[0].shape
    (out/"driver.f90").write_text(DRIVER)
    with (out/"inputs.bin").open("wb") as f:
        f.write(np.asarray([ny*nx, nz], "<i4").tobytes())
        for a in fields:
            f.write(a.transpose(1,2,0).copy().tobytes())
    phys = build/"phys"
    includes = ["-I"+str(phys), "-I"+str(build/"main"), "-I"+str(build/"share")]
    flags = ["-O2", "-fno-fast-math", "-ffp-contract=off", "-fno-tree-vectorize",
             "-cpp", "-DRWORDSIZE=4", "-ffree-form", "-ffree-line-length-none", *includes]
    source = (phys/"physics_mmm/mp_wsm6.F90").read_text()
    if source.count(STOCK) != 1:
        raise ValueError("one stock radar final assignment is required")
    results = {}
    for name, (expr, _) in EXPRESSIONS.items():
        folder = out/name
        folder.mkdir()
        obj = phys/"physics_mmm/mp_wsm6.o"
        if expr is not None:
            path = folder/"mp_wsm6.F90"
            path.write_text(source.replace(STOCK, "dBZ(k) = "+expr))
            obj = folder/"mp_wsm6.o"
            subprocess.run(["gfortran", *flags, "-J"+str(folder), "-c", str(path), "-o", str(obj)], check=True)
        exe = folder/"probe"
        subprocess.run(["gfortran", *flags, str(out/"driver.f90"), str(obj),
                        str(phys/"physics_mmm/mp_radar.o"),
                        str(phys/"physics_mmm/module_libmassv.o"),
                        str(build/"share/module_model_constants.o"), "-o", str(exe)], check=True)
        raw = folder/"output.bin"
        subprocess.run([str(exe), str(out/"inputs.bin"), str(raw)], check=True)
        results[name] = np.frombuffer(raw.read_bytes(), "<f4").reshape(ny,nx,nz).transpose(2,0,1).copy()
    return results


def device_outputs(fields):
    import cupy as cp
    from gpuwm.core.kernels import module_source, module_options
    from gpuwm.core.wsm6 import wsm6_radar_words_on_device
    nz,ny,nx = fields[0].shape
    source = module_source("wsm6")
    if source.count(GPU) != 1:
        raise ValueError("one radar final assignment is required")
    args = [cp.asarray(v) for v in fields]
    rf,rd = wsm6_radar_words_on_device(0)
    results = {}
    for name, (_,expr) in EXPRESSIONS.items():
        text = source if expr is None else source.replace(GPU,"const float dbz = "+expr+";")
        mod = cp.RawModule(code=text,options=module_options("wsm6"))
        result = cp.empty(fields[0].shape, cp.float32)
        mod.get_function("wsm6_refl10cm_column")(((ny*nx+31)//32,), (32,),
            (*args,rf,rd,result,np.int32(1),np.int32(nz),np.int32(ny),np.int32(nx)))
        results[name] = cp.asnumpy(result)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--arm",choices=("wrf","woof"),required=True)
    p.add_argument("--shape",type=int,nargs=3,required=True)
    p.add_argument("--wrf-build",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    fields=capture(a.directory,a.arm,tuple(a.shape))
    outputs=stock_outputs(fields,a.wrf_build,a.out) if a.arm=="wrf" else device_outputs(fields)
    rows={k:digest(v) for k,v in zip(("qv","qr","qs","qg","temperature","pressure"),fields)}
    for name,v in outputs.items():
        rows[name]=digest(v)
        # The active tile is reported alongside the full domain; stock
        # never calls this routine on the specified boundary ring.
        rows[name]["active"]=digest(v[:,1:-1,1:-1])
        np.save(a.out/(name+".npy"),v)
    (a.out/"receipt.json").write_text(json.dumps(dict(arm=a.arm,rows=rows),indent=2)+"\n")


if __name__=="__main__":
    main()
