"""Bounded original-MPI large-packet control with explicit process ownership."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time


def descendants(pid):
    rows=subprocess.check_output(["ps","-eo","pid=,ppid="],text=True).splitlines()
    pairs=[tuple(map(int,row.split())) for row in rows]
    owned={pid}
    while True:
        extended=owned|{child for child,parent in pairs if parent in owned}
        if extended==owned:
            return owned
        owned=extended


def run(original_build,native_build,sdk,build,destination):
    original_build,native_build,sdk,build=map(Path,(original_build,native_build,sdk,build))
    build.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).parent
    raw=(script/"run_spotting_mpi.F90").read_bytes()
    changed=raw.replace(b"integer,parameter::n=18",b"integer,parameter::n=18000")
    if raw==changed:
        raise ValueError("large-packet control requires the pinned small-packet harness declaration")
    harness=build/"run_spotting_mpi_large.F90"
    harness.write_bytes(changed)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    env["LIBRARY_PATH"]=str(sdk/"usr/lib/x86_64-linux-gnu/openmpi/lib")
    flags=["-O0","-cpp","-ffp-contract=off","-fcheck=all","-fbacktrace","-ffree-form","-ffree-line-length-none","-DDM_PARALLEL"]
    includes=["-I",str(original_build),"-I",str(native_build),
        "-I",str(sdk/"usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-16/openmpi"),
        "-I",str(sdk/"usr/lib/x86_64-linux-gnu/openmpi/include")]
    subprocess.run(["mpifort",*flags,*includes,"-c",str(harness),"-o",str(build/"large.o")],check=True,env=env)
    executable=build/"run"
    objects=(native_build/"oracle_io.o",original_build/"spotting_mpi_services.o",
             original_build/"module_firebrand_spotting_mpi.o",build/"large.o")
    subprocess.run(["mpifort","-o",str(executable),*map(str,objects)],check=True,env=env)
    command=["nice","-n","10","taskset","-c","0,1","mpirun","--oversubscribe","--bind-to","none","-np","9",str(executable),str(build/"fixtures")]
    started=time.monotonic()
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    timed_out=False;terminated=0
    try:
        output,_=process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        timed_out=True
        owned=descendants(process.pid)
        for pid in sorted(owned-{process.pid},reverse=True):
            try:
                os.kill(pid,signal.SIGTERM);terminated+=1
            except ProcessLookupError:
                pass
        process.terminate()
        output,_=process.communicate(timeout=10)
    result=dict(original_native_helper=True,brands_per_rank=18000,ranks=9,
        deadline_seconds=15,timed_out=timed_out,terminated_owned_processes=terminated,
        exit_code=process.returncode,elapsed_seconds=time.monotonic()-started,
        helper_object_sha256=hashlib.sha256(objects[2].read_bytes()).hexdigest(),
        original_harness_sha256=hashlib.sha256(raw).hexdigest(),
        large_harness_sha256=hashlib.sha256(changed).hexdigest(),
        transformation="particle capacity 18 -> 18000; original native MPI routines unchanged")
    Path(destination).write_text(json.dumps(result,indent=2)+"\n")
    Path(destination).with_suffix(".log").write_text(output)
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("original_build","native_build","sdk","build","destination"):
        parser.add_argument(name)
    args=parser.parse_args()
    run(**vars(args))
