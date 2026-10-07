"""Make the CPU parity gates fail, then restore each reference byte for byte."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def check(output):
    root=Path(__file__).resolve().parents[2]
    changes=(
        ("wetdep bounds","gpuwm/verify/chem_wetdep_ls_ref.py","range(nz - 2)","range(nz - 1)",
         "tests/test_chem_wetdep_ls_wrf471_parity.py::test_cpu_parity",2),
        ("fire clamp","gpuwm/verify/chem_fire_ref.py","min(F(5000), max(F(0)","min(F(4999), max(F(0)",
         "tests/test_chem_fire_emissions.py::test_injection_cpu",2),
        ("PBL average divisor","gpuwm/verify/chem_fire_ref.py","avg = F(avg/F(kpbl))","avg = F(avg/F(kpbl+1))",
         "tests/test_chem_fire_emissions.py::test_prep_cpu",4),
        ("classification strict edge","gpuwm/verify/chem_fire_ref.py",'> F(rule["gt"])','>= F(rule["gt"])',
         "tests/test_chem_fire_emissions.py::test_classification_oracle",1),
        ("scaled cache floor","gpuwm/verify/chem_fire_ref.py","max(F(1e-4),coef[ix])","max(F(1e-3),coef[ix])",
         "tests/test_chem_fire_emissions.py::test_injection_cpu[2]",1),
        ("source mass conversion","gpuwm/verify/chem_fire_ref.py","F(1e9)","F(1e8)",
         "tests/test_chem_fire_emissions.py::test_source_mass_identity",1),
    )
    results={}
    env=dict(os.environ,GPUWM_NO_LOCAL_GPU="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    for name,relative,before,after,test,expected in changes:
        path=root/relative
        original=path.read_bytes()
        assert original.count(before.encode())==1,(name,"mutation target is not unique")
        cache=Path(importlib.util.cache_from_source(str(path)))
        try:
            path.write_bytes(original.replace(before.encode(),after.encode()))
            cache.unlink(missing_ok=True)
            run=subprocess.run([sys.executable,"-m","pytest","-q",test],cwd=root,env=env,
                               capture_output=True,text=True)
            failed=[line for line in run.stdout.splitlines() if line.startswith("FAILED ")]
            results[name]={"exit_code":run.returncode,"failed":failed}
            assert run.returncode==1 and len(failed)==expected,(name,run.stdout,run.stderr)
        finally:
            path.write_bytes(original)
            cache.unlink(missing_ok=True)
        assert path.read_bytes()==original
    output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print("Six deliberate mutations fired all nine CPU oracle case gates and the source mass test; references restored byte for byte")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    check(parser.parse_args().output)
