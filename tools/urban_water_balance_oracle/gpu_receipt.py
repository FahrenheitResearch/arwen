"""Hash a real surface-driver replay and record its restart algorithm identity."""
from pathlib import Path
import dataclasses
import hashlib
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tests"))
import cupy as cp
import numpy as np
from test_urban_default_off_identity import _small_driver
from gpuwm.core import kernels
from gpuwm.io.restart import physics_setup_identity

root = Path(sys.argv[1]).resolve()
engine = Path(sys.argv[2])
base = "4d1079249ad0cc2553a9632ac9ef267553913dd1"
old = subprocess.check_output(["git","-C",str(engine),"show",f"{base}:gpuwm/core/kernels/urban_ucm.cu"]).decode()
from gpuwm.cuda_subnormals import MARKER,install
install()
baseline_source = MARKER+kernels._preamble()+kernels._extra_header_text("urban_ucm")+old
baseline_module = cp.RawModule(code=baseline_source,options=("-std=c++17","--ftz=false"))
original_get = kernels.get_kernel
def baseline_get(name,function):
    return baseline_module.get_function(function) if name == "urban_ucm" else original_get(name,function)

def hashes(mapping):
    return {name:hashlib.sha256(cp.asnumpy(value).tobytes()).hexdigest()
            for name,value in mapping.items() if isinstance(value,cp.ndarray)}

def replay(lsm,generation,baseline=False):
    kernels.get_kernel = baseline_get if baseline else original_get
    try:
        state,cfg,driver = _small_driver(sf_urban_physics=1,sf_surface_physics=lsm,
                                        urban_water_balance=generation)
        urban = driver.urban
        values = dict(urban.params.values)
        values["IMP_SCHEME"] = np.asarray(2,np.int32)
        urban.params = dataclasses.replace(urban.params,values=values)
        for surface in "rbg":
            urban.fields[f"drel{surface}_urb2d"].fill(cp.float32(0.0001))
            urban.fields[f"flxhum{surface}_urb2d"].fill(cp.float32(0.00003))
        steps = []
        state.physics = driver
        identity = physics_setup_identity(state,cfg)
        for step in range(3):
            driver.fields["rainbl"].fill(cp.float32(1.e-10))
            driver.compute(state,cfg)
            steps.append({"state":hashes(vars(state)),"surface":hashes(driver.fields),
                          "urban":hashes(urban.fields)})
        return {"config":dataclasses.asdict(cfg),"identity":identity,"steps":steps}
    finally:
        kernels.get_kernel = original_get

records = {}
for lsm in (2,4):
    baseline = replay(lsm,"wrf_471",True)
    legacy = replay(lsm,"wrf_471")
    corrected = replay(lsm,"wrf_480")
    assert baseline["steps"] == legacy["steps"],f"legacy full-field hashes moved for LSM {lsm}"
    assert corrected["steps"] != legacy["steps"],f"new generation is not active for LSM {lsm}"
    assert legacy["identity"]["algorithms"]["urban"] == "urban-slucm-wrf-v4.7.1-v1"
    assert corrected["identity"]["algorithms"]["urban"] == "urban-slucm-wrf-v4.7.1-water-balance-v4.8.0-v1"
    records[str(lsm)] = {"baseline":baseline,"legacy":legacy,"corrected":corrected}
(root/"gpu-replay.json").write_text(json.dumps({"base":base,"base_kernel_sha256":hashlib.sha256(old.encode()).hexdigest(),
    "scope":"three real physics-driver surface steps, controlled IMP_SCHEME=2 and rain; no dynamics RK replay",
    "cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),"records":records},indent=2)+"\n")
print("legacy full-field hashes unchanged for both LSMs at all three steps; corrected hashes differ")
