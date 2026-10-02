"""Build flat constant-K cases and call the original WRF routines directly."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess
import numpy as np
from diff6_cases import read_dump,crop,halo,fbytes,NX,NY,NZ


def generate(dump,build,output):
    output.mkdir(parents=True,exist_ok=True)
    raw=read_dump(dump);real=crop(raw)
    mass=np.mean(real["mut"],dtype=np.float32)
    real_zf=np.mean(real["phb"].astype(np.float64),axis=(1,2))/9.81
    rows=[];arrays={}
    for scenario in ("real_flat_uniform","real_flat_stretched","linear_shear","zero","near_zero","variable_mass","pure_meridional_checker"):
        zf=real_zf if scenario=="real_flat_stretched" else np.arange(NZ+1,dtype=np.float64)*250.
        zh=.5*(zf[:-1]+zf[1:])
        alpha=np.float32(1.)
        rdnw=np.zeros(NZ+1,dtype=np.float32);rdn=np.zeros(NZ+1,dtype=np.float32)
        rdnw[:NZ]=(-float(mass)*float(alpha)/9.81/np.diff(zf)).astype(np.float32)
        rdn[1:NZ]=(-float(mass)*float(alpha)/9.81/np.diff(zh)).astype(np.float32)
        for name,variable,stag in (("u","u","x"),("v","v","y"),("w","w","z"),("m","t","")):
            f=real[variable].copy()
            if scenario=="zero":f[:]=0.
            if scenario=="near_zero":
                kk,jj,ii=np.indices(f.shape)
                f[:]=np.float32(2.**-45)*(1+kk).astype(np.float32)*((-1.)**(ii+jj)).astype(np.float32)
            if scenario=="linear_shear":
                kk,jj,ii=np.indices(f.shape)
                f[:]=(.15*kk*kk+.01*ii+.02*jj).astype(np.float32)
                if name=="w":f[:]=(.2*kk*(NZ-kk)).astype(np.float32)
            if scenario=="pure_meridional_checker":
                kk,jj,ii=np.indices(f.shape)
                f[:]=np.float32(5.)*((-1.)**jj).astype(np.float32)
            if name=="u":f[:,:,-1]=f[:,:,0]
            if name=="v":f[:,-1,:]=f[:,0,:]
            mut=np.full((NY,NX),mass,dtype=np.float32)
            if scenario=="variable_mass":mut+=np.linspace(-1000,1000,NX,dtype=np.float32)[None]
            # The WRF u/v vertical routines receive their face masses.
            muf=mut
            if name=="u":muf=.5*(mut+np.roll(mut,1,axis=1));muf=np.concatenate((muf,muf[:,:1]),axis=1)
            if name=="v":muf=.5*(mut+np.roll(mut,1,axis=0));muf=np.concatenate((muf,muf[:1,:]),axis=0)
            for op in (0,1):
                case=len(rows);prefix=f"c{case:03d}__"
                kh,kv=(20.,0.) if op==0 else (0.,5.)
                native_mass=muf if op==1 and name in ("u","v") else mut
                dx,dy=(2048.,2048.) if scenario=="pure_meridional_checker" else (3000.,1750.)
                blob=struct.pack("<5i",NX,NY,NZ,op,0)+name.encode()+struct.pack("<4f",kh,kv,1./dx,1./dy)
                alpha3d=np.full((NZ+1,NY,NX),alpha,dtype=np.float32)
                for a in (f,native_mass,alpha3d,rdn,rdnw,np.ones(NZ+1,dtype=np.float32),np.zeros(NZ+1,dtype=np.float32)):
                    blob+=fbytes(halo(a,0))
                infile=build/"constant-case-input.bin";outfile=build/"constant-case-result.bin"
                infile.write_bytes(blob)
                subprocess.run([str((build/"constant_driver").resolve()),str(infile.resolve()),str(outfile.resolve())],check=True)
                reference=np.fromfile(outfile,dtype="<f4").reshape(NY+7,NZ+1,NX+7).transpose(1,0,2)
                reference=reference[:f.shape[0],3:3+f.shape[1],3:3+f.shape[2]].copy()
                fields={"field":f,"mut":mut,"coupling_mass":muf,"zf":zf,"alt":alpha3d,"rdn":rdn,"rdnw":rdnw,"reference":reference}
                for key,value in fields.items():arrays[prefix+key]=value
                routine=("horizontal_diffusion_3dmp" if name=="m" else "horizontal_diffusion") if op==0 else {"u":"vertical_diffusion_u","v":"vertical_diffusion_v","w":"vertical_diffusion","m":"vertical_diffusion_3dmp"}[name]
                rows.append({"case":case,"scenario":scenario,"name":name,"stagger":stag,"op":op,"routine":routine,"kh":kh,"kv":kv,"dx":dx,"dy":dy})
    np.savez_compressed(output/"constant-wrf471.npz",**arrays)
    metadata=json.loads((build/"constant-build.json").read_text())
    metadata.update({"schema":"wrf471-constant-diffusion-v1","cases":rows,"state_time":"2024-05-25T18:00:00Z","raw_decoder":"rw_netcdf dump --raw",
                     "normalization":"GPU primitive output multiplied by field face mass, matching the WRF coupled tendency; every stored output word remains compared",
                     "coefficient_contract":"matched per-field K; the WRF production scalar Prandtl factor is an additional declared caller divergence",
                     "vertical_geometry":"flat hydrostatic eta metrics linked to full-level heights with specific volume 1 m3/kg; real field values retained except named edge probes",
                     "periodic_halos":"three cells wrapped around the mass core; redundant u/v boundary faces stored explicitly",
                     "tools_sha256":{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ("constant_build.py","constant_driver.f90","constant_cases.py","diff6_cases.py")}})
    (output/"constant-wrf471.json").write_text(json.dumps(metadata,indent=2)+"\n", encoding="utf-8", newline="\n")
    (output/"constant-oracle-sha256sums.txt").write_text("".join(hashlib.sha256((output/n).read_bytes()).hexdigest()+"  "+n+"\n" for n in ("constant-wrf471.npz","constant-wrf471.json")), encoding="utf-8", newline="\n")
    print(f"compiled constant-K cases={len(rows)} output words={sum(v.size for k,v in arrays.items() if k.endswith('__reference'))}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dump",type=Path);p.add_argument("build",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();generate(a.dump,a.build,a.output)
