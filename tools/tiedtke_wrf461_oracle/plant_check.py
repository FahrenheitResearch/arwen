"""Compile a 1 ULP planted change in a diagnostic copy of the real driver.

The original driver file is not edited. The CPU native-tangent control removes
the already known math delta so the detector must isolate exactly one word.
"""
from __future__ import annotations
import ctypes as c
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from check import fixture,fortran,host,FIELDS,compare_words

HERE=Path(__file__).resolve().parent
KERNELS=HERE.parents[1]/"gpuwm/core/kernels"


def main():
    build=Path(sys.argv[1]).resolve()
    source=(KERNELS/"tiedtke.cu").read_text()
    anchor="rthcuten[a] = tk_mul(tk_div(tk_sub(t1(zz), t3d[a]), pi3d[a]), rdelt);"
    if source.count(anchor)!=1:
        raise RuntimeError("the driver planting boundary moved")
    planted=source.replace(anchor,anchor+"\n        if (k == 1 && c == 0) rthcuten[a] = nextafterf(rthcuten[a], INFINITY);")
    (build/"planted-driver.cu").write_text(planted)
    cpp=(HERE/"host.cpp").read_text()
    for line in cpp.splitlines():
        if line.startswith('#include "'):
            target=(HERE/line.split('"')[1]).resolve()
            if target.name=="tiedtke.cu":
                target=build/"planted-driver.cu"
            cpp=cpp.replace(line,f'#include "{target}"')
    (build/"planted-host.cpp").write_text(cpp)
    command=["g++","-std=c++17","-O2","-fno-fast-math","-ffp-contract=off",
             "-DTK_NATIVE_TAN_CONTROL","-fPIC","-shared","planted-host.cpp","-o","libcu6_planted.so"]
    subprocess.run(command,cwd=build,check=True)
    reference=c.CDLL(str(build/"libcu6_oracle.so"))
    inputs,_=fixture(reference,72,50)
    config=dict(dt=60.,itimestep=2,stepcu=1,fqc=1,fqi=1)
    expected=fortran(reference,inputs,**config)
    actual=host(c.CDLL(str(build/"libcu6_planted.so")),inputs,**config)
    result={name:compare_words(expected[name],actual[name]) for name in FIELDS}
    success=(result["rthcuten"]["differing_words"]==1 and result["rthcuten"]["max_ulp"]==1
             and all(v["differing_words"]==0 for name,v in result.items() if name!="rthcuten"))
    receipt={"pass":success,"backend":"cpu-compiled-driver-with-native-tan-control",
             "gpu_checked":False,"columns":72,"nz":50,"config":config,
             "planted_source_sha256":hashlib.sha256(planted.encode()).hexdigest(),
             "plant":"one nextafterf step in rthcuten at bottom level, first column",
             "build_command":command,"fields":result}
    Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!="fields"}))
    return int(not success)


if __name__=="__main__":
    raise SystemExit(main())
