"""Build and grade root uptake, then record full-column release controls.

Reference sources must already be fetched by build.py. All builds/tests are
CPU-only and are intended for the lane CPU box. No simulation data is returned.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from build import FLAGS, HERE, LEGACY, URBAN480_SHA256, run, sha


def grade_root(sources, out):
    out.mkdir(parents=True, exist_ok=True)
    source = sources / "v4.8.0"
    if sha(source / "phys/module_sf_urban.F") != URBAN480_SHA256:
        raise ValueError("urban 4.8 source hash differs")
    commands = []
    for file in [LEGACY / "stub_ucm.F90", source / "share/module_model_constants.F",
                 source / "frame/module_wrf_error.F", source / "phys/module_sf_urban.F",
                 HERE / "root_ref.F90"]:
        command = ["gfortran", "-c", *FLAGS, "-fcheck=all", "-I.", file]
        commands.append(command)
        run(command, cwd=out)
    commands.append(["gfortran", "-shared", "-o", out / "root-ref.so", "stub_ucm.o",
                     "module_model_constants.o", "module_wrf_error.o", "module_sf_urban.o", "root_ref.o"])
    run(commands[-1], cwd=out)
    cflags = ["-O0", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-shared", "-std=c++17"]
    commands.append(["g++", *cflags, HERE / "root_host.cpp", "-o", out / "root-host.so"])
    run(commands[-1])
    helper = HERE.parents[1] / "gpuwm/core/kernels/urban_root_uptake_480.cuh"
    listing = helper.read_text(encoding="utf-8-sig")
    # Change one bit of Lv in the new term, then compile and run that code.
    before = np.float32(2.26e6)
    bits = before.view(np.uint32) ^ np.uint32(1)
    after = bits.view(np.float32)
    term = "FDIV(ele, 2.26e6f)"
    if listing.count(term) != 1:
        raise ValueError("mutation anchor must occur exactly once")
    (out / "root-mutant.cuh").write_text(listing.replace(term, f"FDIV(ele, {float(after).hex()}f)"))
    host = (HERE / "root_host.cpp").read_text()
    (out / "root-mutant.cpp").write_text(host.replace(
        '../../gpuwm/core/kernels/urban_root_uptake_480.cuh', 'root-mutant.cuh'))
    commands.append(["g++", *cflags, out / "root-mutant.cpp", "-o", out / "root-mutant.so"])
    run(commands[-1])
    tests = HERE.parents[1] / "tests/test_urban48_root_uptake.py"
    spec = importlib.util.spec_from_file_location("root_test", tests)
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    fields, failures, mutation_failures, indices = [], [], [], []
    words = 0
    for index, values in enumerate(test.cases()):
        want = test.invoke(out / "root-ref.so", "root_ref", values)
        got = test.invoke(out / "root-host.so", "root_host", values)
        mutant = test.invoke(out / "root-mutant.so", "root_host", values)
        bad = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
        mbad = np.flatnonzero(mutant.view(np.uint32) != want.view(np.uint32))
        words += len(want)
        fields.append({"case": index, "layers": len(want), "sha256": hashlib.sha256(want.tobytes()).hexdigest()})
        for cell in bad:
            failures.append({"case":index,"layer":int(cell),"got":int(got.view(np.uint32)[cell]),
                             "want":int(want.view(np.uint32)[cell])})
        if mbad.size:
            mutation_failures.append(index)
            if len(indices)<12:
                indices.append({"case":index,"layers":mbad.tolist(),
                                "got":mutant.view(np.uint32)[mbad].tolist(),
                                "want":want.view(np.uint32)[mbad].tolist()})
    receipt = {"cases":len(fields),"words":words,"differences":failures,
               "mutation": {"constant_before":int(before.view(np.uint32)), "constant_after":int(bits),
                            "rejected_cases":len(mutation_failures),"first_cells":indices},
               "hashes":fields,"helper_sha256":sha(helper),"commands":commands,
               "compiler":run(["g++","--version"]).stdout.splitlines()[0],
               "fortran":run(["gfortran","--version"]).stdout.splitlines()[0]}
    if failures or not mutation_failures:
        (out / "root-receipt.json").write_text(json.dumps(receipt,indent=2,default=str))
        raise ValueError(f"root comparison failed: {len(failures)} differing words; receipt at {out / 'root-receipt.json'}")
    env = dict(os.environ, GPUWM_URBAN480_ORACLE_BUILD=str(out),CUDA_VISIBLE_DEVICES="",GPUWM_NO_LOCAL_GPU="1")
    command = ["python","-m","pytest","-q",tests]
    p = subprocess.run(list(map(str,command)),env=env,text=True,capture_output=True)
    receipt["pytest_command"]=list(map(str,command))
    receipt["pytest_exit"]=p.returncode
    (out / "pytest.log").write_text(p.stdout+p.stderr)
    (out / "root-receipt.json").write_text(json.dumps(receipt,indent=2,default=str))
    if p.returncode:
        raise ValueError(p.stdout+p.stderr)
    return receipt


def read_numeric(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    keys=[k for k in rows[0] if k!='variant']
    values=np.asarray([[float(r[k]) for k in keys] for r in rows],np.float32)
    return keys,values


def grade_water(out):
    command=["gfortran","-c",*FLAGS,"-fcheck=all","-I.",HERE/"water_ref.F90"]
    run(command,cwd=out)
    commands=[command,["gfortran","-shared","-o",out/"water-ref.so","stub_ucm.o",
                      "module_model_constants.o","module_wrf_error.o","module_sf_urban.o","water_ref.o"],
              ["g++","-O0","-fPIC","-fno-fast-math","-ffp-contract=off","-shared","-std=c++17",
               HERE/"water_host.cpp","-o",out/"water-host.so"]]
    run(commands[1],cwd=out);run(commands[2])
    tests=HERE.parents[1]/"tests/test_urban48_ground_water.py"
    spec=importlib.util.spec_from_file_location("water_test",tests)
    test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
    hashes={k:hashlib.sha256() for k in test.FIELDS}
    differences=[];steps=0;ncases=0
    for index,initial in enumerate(test.cases()):
        ncases+=1
        hg=[v.copy() for v in initial];wg=[v.copy() for v in initial]
        for step in range(6):
            want=test.invoke(out/"water-ref.so","water_ref",wg)
            got=test.invoke(out/"water-host.so","water_host",hg)
            bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
            for cell in bad:
                differences.append({"case":index,"step":step,"field":test.FIELDS[cell],
                                    "got":int(got.view(np.uint32)[cell]),"want":int(want.view(np.uint32)[cell])})
            for cell,key in enumerate(test.FIELDS):hashes[key].update(want[cell].tobytes())
            hg[0][:3]=got[12:15];wg[0][:3]=want[12:15]
            hg[4][11]=got[15];wg[4][11]=want[15]
            steps+=1
    env=dict(os.environ,GPUWM_URBAN480_ORACLE_BUILD=str(out),CUDA_VISIBLE_DEVICES="",GPUWM_NO_LOCAL_GPU="1")
    p=subprocess.run(["python","-m","pytest","-q",str(tests)],env=env,text=True,capture_output=True)
    (out/"water-pytest.log").write_text(p.stdout+p.stderr)
    receipt={"cases":ncases,"steps":steps,"words":steps*len(test.FIELDS),"differences":differences,
             "field_hashes":{k:h.hexdigest() for k,h in hashes.items()},"commands":commands,
             "pytest_exit":p.returncode,
             "undefined_not_graded": ["SRT RHSTT(4), AI(4), BI(4), CI(4)","SMFLX SMC(4)"],
             "scope":"CPU compilation of CUDA-source helpers only; no forecast coupling qualification"}
    (out/"water-receipt.json").write_text(json.dumps(receipt,indent=2,default=str))
    if differences or p.returncode:
        raise ValueError(f"water comparison failed: {len(differences)} differing words; "
                         f"receipt at {out / 'water-receipt.json'}; pytest at {out / 'water-pytest.log'}")
    return receipt


def grade_columns(build, out):
    out.mkdir(parents=True,exist_ok=True)
    runs=[]
    configurations=[("control",0,0,0,0), ("fraction_only",0,0.25,0,0),
                    ("tree_diagnostic",1,0.25,0,0), ("tree_prognostic",1,0.25,1,0),
                    ("tree_fvg_zero",1,0,0,0), ("tree_greenroof",1,0.25,0,1)]
    for tag in ("v4.6.1","v4.7.0","v4.7.1","v4.8.0"):
        for mode in ("ref","snan","zero"):
            work=build/tag/mode
            configs=configurations if tag=="v4.8.0" else configurations[:1]
            for name,tree,fvg,tt,green in configs:
                prefix=(out/f"{tag}-{mode}-{name}").resolve()
                command=[work/"run_ucm","nlcd",name,prefix,"-9","-9","-9","-9","-9","-9",str(green),"0.5","-9"]
                if tag=="v4.8.0":command += [str(tree),str(fvg),str(tt)]
                p=run(command,cwd=work,allow_fail=True)
                log=out/f"{tag}-{mode}-{name}.log"
                log.write_text(p.stdout+p.stderr)
                row={"tag":tag,"mode":mode,"config":name,"command":list(map(str,command)),
                     "exit":p.returncode,"log":str(log)}
                if p.returncode==0:
                    csv_path=Path(str(prefix)+".csv")
                    keys,values=read_numeric(csv_path)
                    row.update(rows=len(values), words=values.size,
                               nonfinite=int(np.sum(~np.isfinite(values))),sha256=sha(csv_path))
                runs.append(row)
    comparisons=[]
    pairs=[("v4.7.1-ref-control","v4.8.0-ref-control"),
           ("v4.8.0-ref-control","v4.8.0-ref-fraction_only"),
           ("v4.8.0-ref-tree_diagnostic","v4.8.0-snan-tree_diagnostic"),
           ("v4.8.0-ref-tree_diagnostic","v4.8.0-zero-tree_diagnostic")]
    for left,right in pairs:
        a,b=out/f"{left}.csv",out/f"{right}.csv"
        if not a.exists() or not b.exists():continue
        ak,av=read_numeric(a);bk,bv=read_numeric(b)
        if ak!=bk or av.shape!=bv.shape:raise ValueError("column ABI mismatch")
        bad=av.view(np.uint32)!=bv.view(np.uint32)
        first=np.argwhere(bad)[:20]
        comparisons.append({"left":left,"right":right,"differing_words":int(bad.sum()),
                            "fields":{k:int(bad[:,i].sum()) for i,k in enumerate(ak) if bad[:,i].any()},
                            "first_cells":[{"row":int(r),"field":ak[c],"left":int(av.view(np.uint32)[r,c]),
                                            "right":int(bv.view(np.uint32)[r,c])} for r,c in first]})
    receipt={"runs":runs,"comparisons":comparisons,
             "scope":"unmodified Fortran column reference only; no CUDA generation comparison"}
    work=build/"v4.8.0"/"snan"
    prefix=(out/"invalid-trap").resolve()
    command=[work/"run_ucm_trap","nlcd","trap",prefix,"-9","-9","-9","-9","-9","-9","0","0.5","-9","1","0.25","0"]
    if command[0].exists():
        p=run(command,cwd=work,allow_fail=True)
        (out/"invalid-trap.log").write_text(p.stdout+p.stderr)
        receipt["checked_invalid_trap"]={"command":list(map(str,command)),"exit":p.returncode,
                                         "log":str(out/"invalid-trap.log")}
    (out/"column-receipt.json").write_text(json.dumps(receipt,indent=2))
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sources",type=Path,required=True)
    p.add_argument("--build",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--component",choices=("all","root","water","columns"),default="all")
    a=p.parse_args()
    a.sources=a.sources.resolve();a.build=a.build.resolve();a.out=a.out.resolve()
    if a.component in ("all","root"):
        root=grade_root(a.sources,a.out/"root")
        print(f"ROOT_UPTAKE: {root['cases']} cases, {root['words']} words, 0 differences; "
              f"changed-term mutation rejected in {root['mutation']['rejected_cases']} cases")
    if a.component in ("all","water"):
        water=grade_water(a.out/"root")
        print(f"Root-aware water solve: {water['steps']} steps, {water['words']} words, 0 differences")
    if a.component in ("all","columns"):
        columns=grade_columns(a.build,a.out/"columns")
        print(f"Fortran column configurations recorded: {len(columns['runs'])}")


if __name__=="__main__":main()
