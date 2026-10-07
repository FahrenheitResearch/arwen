"""Count preserved disk checkpoint words and record exercised source bytes."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def collect(artifacts, engine, destination):
    artifacts, engine = Path(artifacts), Path(engine)
    references = sorted(p for p in artifacts.glob("test_actual_coupled_fire_disk_*/reference.npz") if not p.parent.is_symlink())
    spotting_references=sorted(p for p in artifacts.glob("test_actual_firebrand_disk_con*/reference.npz") if not p.parent.is_symlink())
    inactive_references=sorted(p for p in artifacts.glob("test_inactive_moisture_and_spo*/reference.npz") if not p.parent.is_symlink())
    references.extend(spotting_references)
    references.extend(inactive_references)
    if not references:
        raise ValueError("coupled disk artifacts are missing")
    cases, runs, hashes = {}, {}, {}
    for index, reference in enumerate(references):
        resumed = reference.with_name("resumed.npz")
        spotting=reference in spotting_references
        label = "spotting" if spotting else "inactive" if reference in inactive_references else "mapped" if index else "unmapped"
        run = {}
        with np.load(reference, allow_pickle=False) as a, np.load(resumed, allow_pickle=False) as b:
            keys = set(a.files) - {"__header__"}
            if reference in inactive_references:
                label="inactive-nfmc"+str(a["fire/grid.moisture.fmc_gc"].shape[0])
            if keys != set(b.files) - {"__header__"}:
                raise ValueError("coupled checkpoint inventories differ")
            for key in sorted(keys):
                x, y = a[key], b[key]
                if key.startswith("__") or x.dtype.itemsize != 4:
                    continue
                if x.shape != y.shape or x.dtype != y.dtype:
                    raise ValueError("coupled checkpoint type or shape differs")
                aa, bb = np.ascontiguousarray(x).view(np.uint32), np.ascontiguousarray(y).view(np.uint32)
                entry = dict(words=x.size, different_words=int(np.count_nonzero(aa != bb)))
                cases[label + "/" + key] = entry
                run[key] = entry
        runs[label] = dict(arrays=len(run), words=sum(v["words"] for v in run.values()),
                          different_words=sum(v["different_words"] for v in run.values()),
                          continuation="8 vs 4+disk+4" if spotting else "6 vs 3+disk+3")
        for p in (reference, resumed, reference.with_name("mid.npz")):
            hashes[label + "/" + p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    source_names = (
        "gpuwm/io/restart.py", "gpuwm/io/history_layout.py", "gpuwm/io/wrf_output_schema.py",
        "gpuwm/io/wrfout.py", "gpuwm/io/sfire_schema.py", "gpuwm/io/nc_writer_bridge.py",
        "gpuwm/core/physics.py", "gpuwm/core/physics_inventory.py", "gpuwm/core/state.py",
        "gpuwm/core/sfire.py", "gpuwm/core/sfire_coupler.py", "gpuwm/core/sfire_wind.py",
        "gpuwm/core/sfire_moisture.py", "gpuwm/core/sfire_core.py", "gpuwm/core/sfire_phys.py",
        "gpuwm/config.py", "gpuwm/sfire_config.py", "gpuwm/core/dycore.py",
        "gpuwm/core/kernels/dycore.cu", "gpuwm/core/kernels/sfire_coupling.cu",
        "gpuwm/core/kernels/sfire_wind.cu", "gpuwm/core/kernels/sfire_core.cu",
        "gpuwm/core/kernels/sfire_atm.cu", "gpuwm/core/kernels/sfire_moisture.cu",
        "gpuwm/core/kernels/sfire_phys.cu", "gpuwm/core/kernels/sfire_phys.cuh",
        "gpuwm/core/kernels/glibc_flt32.cuh", "gpuwm/core/kernels/sfire_libm.cuh",
        "tests/test_sfire_restart_coupled_gpu.py",
    )
    if spotting_references:
        source_names+= ("gpuwm/core/sfire_spotting.py","gpuwm/core/kernels/sfire_spotting.cu",
                        "gpuwm/core/kernels/glibc_flt64.cuh")
    receipt = dict(continuation="6 coupled steps versus 3 + disk restore + 3; active spotting 8 versus 4 + disk restore + 4" if spotting_references else "6 coupled steps versus 3 + disk restore + 3",
                   physical_steps=6, step_seconds=0.25, fire_refinement=[4, 3],
                   assertions=["nonzero fire area", "nonzero sensible flux", "nonzero held atmosphere heating",
                               "modeled moisture initialized", "every restored fire array remains a device array",
                               "malformed moisture clock refuses before model mutation", "all published fire fields use Rust history"],
                   arrays=len(cases), words=sum(v["words"] for v in cases.values()),
                   different_words=sum(v["different_words"] for v in cases.values()),
                   fire_arrays=sum("/fire/" in k for k in cases),
                   fire_words=sum(v["words"] for k, v in cases.items() if "/fire/" in k),
                   cases=cases, runs=runs, artifact_sha256=hashes,
                   source_sha256={name: hashlib.sha256((engine / name).read_bytes()).hexdigest()
                                  for name in source_names})
    Path(destination).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("arrays", "words", "different_words", "fire_arrays", "fire_words")}))
    if receipt["different_words"]:
        raise ValueError("coupled disk continuation is not byte exact")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts")
    parser.add_argument("engine")
    parser.add_argument("destination")
    args = parser.parse_args()
    collect(args.artifacts, args.engine, args.destination)
