"""Feed realistic and branch-probing states to compiled WRF, without a mirror.

NetCDF decoding is performed by rw_netcdf. This script only assembles standalone
test cases from its exact raw binary exports and orchestrates Fortran execution.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import struct
import subprocess
import numpy as np

NX, NY, NZ = 17, 15, 7


def read_dump(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    return {v["name"]: np.fromfile(directory / v["filename"], dtype="<f8").reshape(v["shape"])[0].astype(np.float32)
            for v in metadata["variables"]}


def crop(raw, level=0, x=60, y=70):
    state = {name.lower(): raw[name][level:level + NZ + (name == "W"), y:y + NY + (name == "V"), x:x + NX + (name == "U")].copy()
             for name in ("U", "V", "W", "T", "QVAPOR")}
    state["mut"] = (raw["MU"][y:y+NY, x:x+NX] + raw["MUB"][y:y+NY, x:x+NX]).astype(np.float32)
    state["phb"] = raw["PHB"][level:level+NZ+1,y:y+NY,x:x+NX].copy()
    for name in ("C1H", "C2H", "C1F", "C2F"):
        state[name.lower()] = raw[name][level:level+NZ+(name.endswith("F"))].copy()
    for name, target, sy, sx in (("MAPFAC_U","mapu",NY,NX+1),("MAPFAC_V","mapv",NY+1,NX),("MAPFAC_M","mapm",NY,NX)):
        state[target] = raw[name][y:y+sy,x:x+sx].copy()
    state["latitude"] = raw["XLAT"][y:y+NY,x:x+NX].copy()
    return state


def scenarios(raw):
    lower = crop(raw)
    upper = crop(raw, 32, 25, 35)
    identity = copy.deepcopy(lower)
    for name in ("mapu", "mapv", "mapm"):
        identity[name][:] = 1
    steep = copy.deepcopy(lower)
    ridge = np.float32(9.81) * (np.maximum(0, 2400-600*np.abs(np.arange(NX)-8))[None,:] + 250*np.arange(NY)[:,None])
    steep["phb"] += ridge[None].astype(np.float32)
    extreme = copy.deepcopy(lower)
    for name in ("mapu", "mapv", "mapm"):
        a = extreme[name]
        a[:] = np.linspace(.25, 2.75, a.shape[1], dtype=np.float32)[None]
    zero = copy.deepcopy(identity)
    for name in ("u", "v", "w", "t", "qvapor"):
        zero[name][:] = 0
    near = copy.deepcopy(identity)
    for name in ("u", "v", "w", "t", "qvapor"):
        a = near[name]
        kk,jj,ii = np.indices(a.shape)
        a[:] = ((-1.)**(ii+jj)) * np.float32(2.**-50) * (1+kk).astype(np.float32)
    south = copy.deepcopy(lower)
    south["latitude"] *= -1
    south["v"] *= -1
    south["u"] *= -1
    hybrid = copy.deepcopy(upper)
    reference_pressure=np.mean(hybrid["mut"],dtype=np.float32)
    for family,n in (("h",NZ),("f",NZ+1)):
        hybrid["c1"+family] = np.linspace(1,0,n,dtype=np.float32)
        hybrid["c2"+family] = (np.float32(1)-hybrid["c1"+family])*reference_pressure
    periodic=copy.deepcopy(lower)
    periodic["u"][:,:,-1]=periodic["u"][:,:,0]
    periodic["v"][:,-1,:]=periodic["v"][:,0,:]
    periodic["mapu"][:,-1]=periodic["mapu"][:,0]
    periodic["mapv"][-1,:]=periodic["mapv"][0,:]
    return [("real_lower_projected",lower),("real_lower_identity",identity),("real_upper",upper),
            ("steep_terrain",steep),("map_factor_extremes",extreme),("zero",zero),
            ("near_zero",near),("southern_wind_reversal",south),
            ("hybrid_transition",hybrid),("real_periodic_halos",periodic)]


def padded(a, shape):
    a = np.asarray(a, dtype="<f4")
    pads = [(0,n-s) for s,n in zip(a.shape,shape)]
    return np.pad(a,pads,mode="edge")


def fbytes(a):
    if a.ndim == 3:
        a = a.transpose(1,0,2)
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def halo(a, mode):
    if a.ndim==1:return padded(a,(NZ+1,))
    if a.ndim==3:a=padded(a,(NZ+1,a.shape[1],a.shape[2]))
    if mode==0:
        return a[...,np.arange(-3,NY+4)%NY,:][...,np.arange(-3,NX+4)%NX]
    shape=(NZ+1,NY+1,NX+1) if a.ndim==3 else (NY+1,NX+1)
    return np.pad(padded(a,shape),[(0,0)]*(a.ndim-2)+[(3,3),(3,3)],mode="edge")


def generate(dump, build, output, real_file=None):
    output.mkdir(parents=True, exist_ok=True)
    raw = read_dump(dump)
    records=[]
    arrays={}
    for scenario,state in scenarios(raw):
        for name,variable in (("u","u"),("v","v"),("w","w"),("t","t"),("q","qvapor")):
            for opt,slope in ((1,0),(2,0),(1,1),(2,1)):
                index=len(records)
                mode=0 if scenario=="real_periodic_halos" else (index%3)+1
                settings={"case":index,"scenario":scenario,"name":name,"opt":opt,"slopeopt":slope,"boundary_mode":mode,
                          "factor":[.125,.12,.08][index%3],"dt":[8.,3.7,2.,2./3][index%4],
                          "dx":[3000.,1200.,900.,4500.][index%4],"dy":[3000.,1750.,900.,2500.][index%4],"thresh":.1}
                prefix=f"c{index:03d}__"
                field=state[variable]
                full=name=="w"
                initial=np.full_like(field,.25) if mode==0 else np.zeros_like(field)
                inputs={"field":field,"tendency":initial,"mut":state["mut"],"c1":state["c1f" if full else "c1h"],
                        "c2":state["c2f" if full else "c2h"],"phb":state["phb"],
                        "mapu":state["mapu"],"mapv":state["mapv"],"mapm":state["mapm"],"latitude":state["latitude"]}
                blob=struct.pack("<6i",NX,NY,NZ,opt,slope,mode)+name.encode()+struct.pack("<5f",settings["dt"],settings["factor"],1/settings["dx"],1/settings["dy"],settings["thresh"])
                for a in (field,initial,inputs["mut"],inputs["c1"],inputs["c2"],inputs["phb"],
                          inputs["mapm"],inputs["mapm"],inputs["mapu"],inputs["mapu"],inputs["mapv"],inputs["mapv"]):
                    blob+=fbytes(halo(a,mode))
                input_file=build/"diff6-case-input.bin"
                result_file=build/"diff6-case-result.bin"
                input_file.write_bytes(blob)
                subprocess.run([str((build/"diff6_driver").resolve()),str(input_file.resolve()),str(result_file.resolve())],check=True)
                reference=np.fromfile(result_file,dtype="<f4").reshape(NY+7,NZ+1,NX+7).transpose(1,0,2)
                inputs["reference"]=reference[:field.shape[0],3:3+field.shape[1],3:3+field.shape[2]].copy()
                for key,value in inputs.items():arrays[prefix+key]=value
                records.append(settings)
    fixture=output/"diff6-wrf471.npz"
    np.savez_compressed(fixture,**arrays)
    metadata=json.loads((build/"diff6-build.json").read_text())
    metadata.update({"schema":"wrf471-diff6-oracle-v1","cases":records,"nx":NX,"ny":NY,"nz":NZ,
                     "state_time":"2024-05-25T18:00:00Z","state_source":"WRF wrfinput",
                     "raw_decoder":"rw_netcdf dump --raw", "hemisphere_probe":"winds and latitude reversed; routine does not use latitude",
                     "initial_tendency":"zero for nonperiodic cases; 0.25 for periodic accumulated tendencies", "every_output_array":["tendency"],
                     "hybrid_transition":"coefficient edge probe on real upper state; C1 spans 1 to 0; C2=(1-C1)*mean(real MUT)",
                     "dimensions":"Fortran i,k,j; ids=0 ide=nx jds=0 jde=ny kds=1 kde=nz+1; memory ims=-3 ime=nx+3 jms=-3 jme=ny+3; tile physical domain; field stagger retained; periodic halos wrap the mass core"})
    if real_file: metadata["real_state_sha256"]=hashlib.sha256(real_file.read_bytes()).hexdigest()
    (output/"diff6-wrf471.json").write_text(json.dumps(metadata,indent=2)+"\n", encoding="utf-8", newline="\n")
    metadata["tools_sha256"]={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in ("diff6_driver.f90","diff6_build.py","diff6_cases.py")}
    (output/"diff6-wrf471.json").write_text(json.dumps(metadata,indent=2)+"\n", encoding="utf-8", newline="\n")
    receipt="".join(hashlib.sha256((output/n).read_bytes()).hexdigest()+"  "+n+"\n" for n in ("diff6-wrf471.npz","diff6-wrf471.json"))
    (output/"diff6-oracle-sha256sums.txt").write_text(receipt, encoding="utf-8", newline="\n")
    print(f"compiled WRF cases={len(records)} output words={sum(v.size for k,v in arrays.items() if k.endswith('__reference'))}")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("dump",type=Path);p.add_argument("build",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--real-file",type=Path)
    a=p.parse_args();generate(a.dump,a.build,a.output,a.real_file)
