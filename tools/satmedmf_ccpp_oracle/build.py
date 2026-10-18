"""Build only the pinned, byte-verified CCPP reference on a sprint CPU box."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent

def build(source:Path,out:Path)->Path:
    pin=json.loads((HERE/"source.json").read_text())
    actual=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if actual != pin["commit"]:
        raise ValueError(f"CCPP commit mismatch: {actual}, expected {pin['commit']}")
    for relative,digest in pin["source_sha256"].items():
        actual=hashlib.sha256((source/relative).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"CCPP source changed: {relative}")
    out.mkdir(parents=True,exist_ok=True)
    flags=["-O0","-ffp-contract=off","-fno-tree-vectorize","-fno-fast-math","-fPIC","-fcheck=all"]
    units=list(pin["source_sha256"])[1:-1]
    objects=[]
    for relative in units:
        path=source/relative
        fmt=["-cpp","-ffixed-form","-ffixed-line-length-72"] if path.suffix in (".F",".f") else ["-ffree-form"]
        obj=out/(path.stem+".o")
        cmd=["gfortran",*flags,*fmt,"-c",str(path),"-o",str(obj)]
        print(" ".join(cmd),flush=True)
        subprocess.run(cmd,cwd=out,check=True)
        objects.append(obj)
    obj=out/"column_oracle.o"
    subprocess.run(["gfortran",*flags,"-ffree-form","-c",str(HERE/"column_oracle.F90"),"-o",str(obj)],cwd=out,check=True)
    target=out/"libsatmedmf_oracle.so"
    subprocess.run(["gfortran","-shared","-o",str(target),*[str(o) for o in objects],str(obj)],cwd=out,check=True)
    receipt={"source":pin,"compiler":subprocess.check_output(["gfortran","--version"],text=True).splitlines()[0],
             "flags":flags,"library_sha256":hashlib.sha256(target.read_bytes()).hexdigest()}
    (out/"build-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return target

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    print(build(a.source.resolve(),a.out.resolve()))
