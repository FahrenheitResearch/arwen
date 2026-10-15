"""Run the untouched WRF driver or grade the shared CUDA runtime launcher."""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, struct, subprocess, tempfile
from pathlib import Path
import numpy as np

OPTIONS=(1,2,3,11)
SCENARIOS=("deep","no_cells","marine","warm_top","input_month","thresholds")
OUTPUTS=("ic_flashrate","cg_flashrate","ic_flashcount","cg_flashcount","lpi")
TIMES=(0.,10.,20.,30.,40.,50.)

def fixture(scenario):
    nx,ny,nz=3,2,8
    shape=(nz,ny,nx)
    f32=np.float32
    ht=np.array([[0.,127.25,410.5],[152.125,20.,35.]],dtype=f32)
    k=np.arange(nz,dtype=f32)[:,None,None]
    c=np.arange(nx*ny,dtype=f32).reshape(ny,nx)
    t=np.ascontiguousarray(np.broadcast_to(f32(295.)-k*f32(7.5),shape))
    z=np.ascontiguousarray((k+f32(.5))*f32(1800.)+ht)
    w=np.ascontiguousarray(np.broadcast_to(np.array([1.,3.25,12.125,19.4,20.75,6.125,4.2,0.],dtype=f32)[:,None,None],shape))
    refl=np.full(shape,-35.,dtype=f32)
    refl[1:6]=30.;refl[5,:,1]=20.;refl[6,0,0]=25.
    fields=dict(t=t,z=z,w=w,refl=refl,dz=np.full(shape,1800.25,dtype=f32),
                theta=np.ascontiguousarray(t/f32(.75)),pi=np.full(shape,.75,dtype=f32),
                ht=ht,land=np.array([[1.,1.,2.],[1.,2.,2.]],dtype=f32),
                lat=np.array([[25.,55.,55.01],[24.99,40.,40.]],dtype=f32),
                lon=np.array([[-105.,-90.,-70.],[-120.,-120.01,-69.99]],dtype=f32),
                top=np.full((ny,nx),7,dtype=np.int32))
    for n,scale in (("qv",.01),("qc",.001),("qr",.0015),("qi",.0005),("qs",.0008),("qg",.0012)):
        fields[n]=np.ascontiguousarray(np.broadcast_to(f32(scale)*(f32(1.)+c/f32(13.)),shape))
    fields["inputnum"]=np.zeros((12,ny,nx),dtype=f32)
    fields["inputden"]=np.ones((12,ny,nx),dtype=f32)
    fields["inputnum"][5]=np.array([[0.,4.,6.],[0.,1.,25.]],dtype=f32)
    fields["inputden"][5]=np.array([[0.,0.,2.],[4.,1.,10.]],dtype=f32)
    if scenario=="no_cells":
        fields["refl"].fill(20.)
        fields["top"].fill(1)
        for n in ("qc","qr","qi","qs","qg"):fields[n].fill(0.)
    elif scenario=="marine":
        fields["land"].fill(2.)
    elif scenario=="warm_top":
        fields["t"][6:]=280.
    elif scenario=="input_month":
        fields["lat"].fill(40.);fields["lon"].fill(-95.)
    elif scenario=="thresholds":
        fields["refl"][6,0,0]=20.
        fields["refl"][5,0,1]=np.nextafter(f32(20.),f32(100.))
        fields["theta"][3]=f32(273.16)/f32(.75)
        fields["theta"][4]=f32(253.16)/f32(.75)
        fields["qi"][3]=f32(1.e-10)
        fields["qg"][4]=f32(1.e-10)
    return fields

def ordered_bytes(a):
    # Fortran [i,k,j] arrays, with i the fastest varying index.
    return np.ascontiguousarray(a.transpose(1,0,2) if a.ndim==3 else a).tobytes()

def run_oracle(executable,out):
    rows=[]
    with tempfile.TemporaryDirectory(dir=out.parent,prefix="columns-") as folder:
        folder=Path(folder)
        for option,method,cell,scenario in itertools.product(OPTIONS,range(5),range(3),SCENARIOS):
            f=fixture(scenario);nz,ny,nx=f["t"].shape
            path=folder/"input.bin";result=folder/"output.bin"
            with path.open("wb") as stream:
                stream.write(struct.pack("<8i",nx,ny,nz,option,method,cell,6,len(TIMES)))
                stream.write(struct.pack("<9f",10.,3000.,4000.,15.,20.,1.25,3.,1.,.125))
                stream.write(np.array(TIMES,dtype="<f8").tobytes())
                for name in ("lat","lon","land","ht","top","t","z","w","refl","dz","theta","pi"):
                    stream.write(ordered_bytes(f[name]))
                for name in ("qv","qc","qr","qi","qs","qg"):
                    stream.write(ordered_bytes(f[name]))
                for name in ("inputnum","inputden"):
                    stream.write(f[name].tobytes())
            subprocess.run([str(executable),str(path),str(result)],check=True)
            bits=np.fromfile(result,dtype="<u4").tolist()
            rows.append(dict(option=option,method=method,cell=cell,scenario=scenario,bits=bits))
    sources=(executable.parent/"source.sha256").read_text()
    doc=dict(wrf_commit="d66e442fccc04111067e29274c9f9eaccc3cef28",source_sha256=sources,
             compiler=subprocess.check_output(["gfortran","--version"],text=True).splitlines()[0],
             flags="-O0 -ffree-form -ffp-contract=off -fno-tree-vectorize -fcheck=bounds",
             rows=rows,steps=len(TIMES),shape=[2,3],times=TIMES)
    out.write_text(json.dumps(doc,separators=(",",":"))+"\n")
    print(json.dumps(dict(oracle_rows=len(rows),words=len(rows)*len(rows[0]["bits"]),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))

def check_words(actual,expected):
    a=np.asarray(actual,dtype=np.uint32)
    e=np.asarray(expected,dtype=np.uint32)
    if a.shape!=e.shape:
        raise AssertionError(f"shape mismatch: {a.shape} != {e.shape}")
    delta=np.abs(a.astype(np.int64)-e.astype(np.int64))
    maxulp=int(delta.max(initial=0))
    return maxulp,int(np.count_nonzero(delta))

def grade(golden,out):
    import cupy as cp
    from gpuwm.core.lightning import launch_lightning,next_activation
    from gpuwm.certify.kernel_manifest import kernel_manifest
    doc=json.loads(golden.read_text())
    report=[]
    all_words=0
    for row in doc["rows"]:
        host=fixture(row["scenario"]);f={k:cp.asarray(v) for k,v in host.items()}
        output={n:cp.full((2,3),v,dtype=cp.float32) for n,v in zip(OUTPUTS,(.03125,.0625,.125,.25,.75))}
        activation=20.
        actual=[]
        for current in TIMES:
            updated=next_activation(activation,current,10.,15.)
            if updated is not None:
                launch_lightning(f,output,option=row["option"],method=row["method"],
                                 cellcount_method=row["cell"],dx=3000.,dy=4000.,interval=15.,
                                 factor=1.25,num=3.,den=1.,adjust=.125,month=6)
                activation=updated
            for name in OUTPUTS:
                actual.extend(cp.asnumpy(output[name]).view(np.uint32).ravel().tolist())
            actual.append(int(np.float32(activation).view(np.uint32)))
        ulp,misses=check_words(actual,row["bits"])
        floats=np.asarray(actual,np.uint32).view(np.float32)
        expected=np.asarray(row["bits"],np.uint32).view(np.float32)
        absolute=float(np.max(np.abs(floats-expected),initial=0.))
        relative=float(np.max(np.abs(floats-expected)/np.maximum(np.abs(expected),np.float32(1.e-30)),initial=0.))
        report.append({k:row[k] for k in ("option","method","cell","scenario")}|
                      dict(max_ulp=ulp,mismatched_words=misses,max_abs=absolute,max_rel=relative))
        all_words+=len(actual)
    planted=np.array(doc["rows"][0]["bits"],dtype=np.uint32)
    planted[0]^=np.uint32(1)
    plant_ulp,plant_misses=check_words(planted,doc["rows"][0]["bits"])
    assert plant_misses==1 and plant_ulp==1,"one-bit negative control was not caught"
    result=dict(arm="strict" if os.environ.get("GPUWM_WRF_EXACT")=="1" else "default",
                oracle_sha256=hashlib.sha256(golden.read_bytes()).hexdigest(),
                rows=len(report),words=all_words,max_ulp=max(r["max_ulp"] for r in report),
                max_abs=max(r["max_abs"] for r in report),max_rel=max(r["max_rel"] for r in report),
                mismatched_words=sum(r["mismatched_words"] for r in report),
                planted_one_bit_caught=True,fields=OUTPUTS,results=report,kernel_manifest=kernel_manifest())
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("results","kernel_manifest")}))
    if result["mismatched_words"]:
        raise SystemExit(1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--oracle",type=Path)
    parser.add_argument("--golden",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.oracle:run_oracle(args.oracle,args.golden)
    else:grade(args.golden,args.out)

if __name__=="__main__":main()
