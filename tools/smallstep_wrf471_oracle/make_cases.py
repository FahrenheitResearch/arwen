"""Extract pinned real WRF crops through the engine's Rust NetCDF reader."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gpuwm.netcdf_bridge import open_dataset

FIELDS = ("U V W T P PB PH PHB MU MUB AL ALB HGT QVAPOR QCLOUD QRAIN QICE QSNOW QGRAUP "
          "C1H C2H C1F C2F C3H C4H C3F C4F RDN RDNW DN DNW FNM FNP ZNU ZNW "
          "CF1 CF2 CF3 MAPFAC_M MAPFAC_U MAPFAC_V MAPFAC_MX MAPFAC_MY MAPFAC_UX "
          "MAPFAC_UY MAPFAC_VX MAPFAC_VY XLAT XLONG F E").split()


def extract(path, x, y, nx, ny, base=None):
    result = {}
    with open_dataset(path) as dataset:
        for name in FIELDS:
            if name not in dataset.variables:
                continue
            var = dataset.variables[name]
            a = np.asarray(var[:],dtype=np.float32)
            dims = tuple(var.dimensions)
            if dims and dims[0] == "Time":
                a, dims = a[0], dims[1:]
            slices = []
            for dim in dims:
                slices.append(slice(x,x+nx+int(dim.endswith("_stag"))) if dim.startswith("west_east")
                              else slice(y,y+ny+int(dim.endswith("_stag"))) if dim.startswith("south_north")
                              else slice(None))
            result[name] = np.ascontiguousarray(a[tuple(slices)])
        metadata = {"nx": nx, "ny": ny, "nz": int(result["T"].shape[0]),
                    "dx": float(dataset.getncattr("DX")), "dy": float(dataset.getncattr("DY")),
                    "x":x,"y":y,"start_date": str(dataset.getncattr("START_DATE"))}
    if "ALB" not in result:
        if base is None:
            raise ValueError("Initial WRF state must include its own AL and ALB")
        result["ALB"] = base["ALB"].copy()
        # The history file omits density. This supplies an input diagnostic,
        # never a reference output, from its recorded dry theta and pressure.
        p = result["PB"].astype(np.float64)+result["P"].astype(np.float64)
        theta = result["T"].astype(np.float64)+300.0
        alt = np.asarray((287.0/100000.0)*theta*(p/100000.0)**(-5.0/7.0),dtype=np.float32)
        result["AL"] = np.asarray(alt-result["ALB"],dtype=np.float32)
        metadata["derived_fields"] = {"ALB":"unchanged base density from initial file",
                                     "AL":"dry EOS from history P+PB and T+300, then subtract initial ALB; input only"}
    for key in FIELDS:
        if key not in result and base is not None and key in base:
            result[key] = base[key].copy()
    result["ALT"] = np.asarray(result["AL"]+result["ALB"], dtype=np.float32)
    return result, metadata


def make(initial, evolved, output):
    output.mkdir(parents=True,exist_ok=True)
    arrays, cases = {}, []
    base,_ = extract(initial,20,20,8,7)
    for index, (name, source, stress) in enumerate((
        ("real_initial", initial, "none"), ("real_evolved", evolved,"none"),
        ("steep_terrain", initial,"terrain"), ("map_extremes", initial,"maps"),
        ("zero_near_zero", initial,"zero"), ("southern_hemisphere", initial,"south"),
    )):
        raw, meta = extract(source,20,20,8,7,base=base)
        if stress == "terrain":
            j,i = np.indices((7,8),dtype=np.float32)
            height = np.asarray(1500*np.sin(i*np.float32(.65))*np.cos(j*np.float32(.55)),dtype=np.float32)
            raw["PHB"] += np.float32(9.81)*height[None]
            raw["HGT"] += height
        elif stress == "maps":
            for key in tuple(raw):
                if key.startswith("MAPFAC_"):
                    j,i = np.indices(raw[key].shape[-2:],dtype=np.float32)
                    raw[key][...] = np.float32(.35)+np.float32(2.15)*(i+np.float32(.7)*j)/(i.max()+np.float32(.7)*j.max())
        elif stress == "zero":
            for key in ("U","V","W","P","PH","MU","T","AL","QVAPOR","QCLOUD","QRAIN","QICE","QSNOW","QGRAUP"):
                if key in raw:
                    raw[key].fill(0)
                    raw[key].reshape(-1)[1::5] = np.float32(1.e-30)
            raw["ALT"] = np.asarray(raw["AL"]+raw["ALB"],dtype=np.float32)
        elif stress == "south":
            raw["XLAT"] *= -1
            raw["F"] *= -1
            raw["V"] *= -1
        prefix = f"case{index}_"
        meta.update(name=name,prefix=prefix,fields=sorted(raw),stress=stress,
                    dtau=2.0,periodic=False,specified=True,spec_zone=1,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        cases.append(meta)
        arrays.update({prefix+k:v for k,v in raw.items()})
    np.savez_compressed(output/"real-state.npz",**arrays)
    (output/"cases.json").write_text(json.dumps({"schema":"gpuwm-smallstep-wrf471-cases-v1","cases":cases},indent=2)+"\n")
    print(json.dumps({"cases":len(cases),"nz":cases[0]["nz"],"fields":cases[0]["fields"],
                      "fixture_bytes":(output/"real-state.npz").stat().st_size}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("initial",type=Path)
    p.add_argument("evolved",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    make(a.initial,a.evolved,a.output)
