"""Capture the second natural AC root step; diagnostic only, no replay."""
import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument("--diagnostic",action="store_true")
parser.add_argument("--host-contracts",action="store_true")
parser.add_argument("--binding",type=Path,required=True)
options=parser.parse_args()
if options.host_contracts and not options.diagnostic:
 parser.error("host annotations are audit-only")
binding=json.loads(options.binding.read_text())
old_mask=os.environ["CUDA_VISIBLE_DEVICES"]
os.environ.update(binding["runtime_env"])
os.environ["CUDA_VISIBLE_DEVICES"]=old_mask
for key,relative in {"TMPDIR":"tmp","XDG_CACHE_HOME":"cache","CUPY_CACHE_DIR":"cache/cupy",
 "CUDA_CACHE_PATH":"cache/cuda","GPUWM_SURFACE_OBS_CACHE":"cache/surface",
 "GPUWM_GENERATED_TABLE_ROOT":"cache/generated"}.items():
 p=root/relative;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
if binding.get("warm_cache"):
 shutil.copytree(binding["warm_cache"],root/"cache/cupy",dirs_exist_ok=True)
sys.path[:0]=binding["runtime_env"]["PYTHONPATH"].split(":")
spec=importlib.util.spec_from_file_location("native_step",root/"gpuwm/native_step/__init__.py")
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
import cupy as cp
from gpuwm.core import dycore
from gpuwm.prepared_domain_tree_forecast import main as run
class CaptureDone(BaseException): pass
original=dycore.step
count=0
record={"purpose":"AC natural second root recording, no replay or throughput","binding":binding,
        "base_source":str(Path(dycore.__file__).resolve()),"cuda_visible_devices":old_mask,
        "shapes":[], "host_annotations":[]}
def step(state,cfg,**kwargs):
 global count
 record["shapes"].append([cfg.nx,cfg.ny,cfg.nz,float(state.elapsed_seconds)])
 if (cfg.nx,cfg.ny,cfg.nz)!=(120,70,44):
  return original(state,cfg,**kwargs)
 count+=1
 if count!=2:return original(state,cfg,**kwargs)
 plan=native.Plan(diagnostic=options.diagnostic)
 plan.trace_failures()
 error=None
 try:
  with plan.record(state,cfg):
   if options.host_contracts:
    from host_annotations import annotate
    context=annotate(plan,record["host_annotations"])
   else:context=nullcontext()
   with context:original(state,cfg,**kwargs)
 except native.RecordingError as exc:
  error=str(exc)
 cp.cuda.get_current_stream().synchronize()
 record.update(capture=plan.describe(),error=error,root_step=count)
 plan.close()
 raise CaptureDone()
dycore.step=step
command=["--prepared-root",binding["prepared_root"],
 binding.get("proof_option","--prepared-head-sha256"),binding["prepared_head_sha256"],
 "--experiment-config",binding["config"],"--experiment-config-sha256",binding["config_sha256"],
 "--progress-format","jsonl","--render-products","none","--io-mode","none",
 "--outdir",str(root/("ac-capture-"+str(time.time_ns())))]
record["argv"]=command
started=time.perf_counter()
try:
 run(command)
 raise RuntimeError("runner did not reach capture")
except CaptureDone:
 if options.diagnostic and record.get("capture",{}).get("error")=="diagnostic capture cannot replay":
  record["status"]="RECORDER_AUDIT_COMPLETE_NOT_REPLAYABLE"
 else:
  record["status"]="CAPTURE_REJECTED" if record.get("error") else "CAPTURED_NOT_FORECAST_COMPLETE"
except BaseException as exc:
 record.update(status="FAILED",error=repr(exc))
 raise
finally:
 record["wall_s"]=time.perf_counter()-started
 (root/("receipts/ac-host-contract-audit.json" if options.host_contracts else "receipts/ac-capture.json")).write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("binding","capture")}))
