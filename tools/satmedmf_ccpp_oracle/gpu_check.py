"""Real GPU field, table, mutation and compile receipts for the pinned column."""
import argparse,json,ctypes
from pathlib import Path
import numpy as np
from gpuwm.core import satmedmf
from gpuwm.core.kernels import load_module
from gpuwm.certify.kernel_manifest import kernel_manifest
from gpuwm.certify.compile_platform import compile_platform_fingerprint
from tools.satmedmf_ccpp_oracle.check import Oracle,columns,compare,advance,FIELDS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--library",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    module=load_module("satmedmf")
    attrs={name:module.get_function(name).attributes for name in ("satmedmf_column","satmedmf_fpvs_table","satmedmf_exner")}
    print("FRAME_RECEIPT",json.dumps({"attributes":attrs,"compile_platform":compile_platform_fingerprint()}),flush=True)
    oracle=Oracle(args.library)
    table=satmedmf.build_table().get()
    expected=np.empty(7501,dtype=np.float64)
    oracle.library.satmedmf_table_oracle(ctypes.c_void_p(expected.ctypes.data))
    table_receipt=compare({"fpvs":expected},{"fpvs":table})
    all_steps=[]
    for strict in (False,True):
        x=columns()
        for step in range(3):
            reference=oracle(x)
            actual=satmedmf.run_columns(x,dt=60.,strict=strict)
            tolerance={field:(0.,0.) for field in FIELDS}
            fields=compare(reference,actual,strict=strict,tolerances=tolerance)
            all_steps.append({"strict":strict,"step":step,"fields":fields,"default_tolerance":{"absolute":0.,"relative":0.}})
            x=advance(x,reference,60.)
    mutation=[]
    for field in FIELDS:
        y=oracle(columns());changed={k:v.copy() for k,v in y.items()}
        words=changed[field].view(np.uint64 if changed[field].dtype==np.float64 else np.uint32)
        words.flat[0]^=np.array(1,dtype=words.dtype)
        receipt=compare(y,changed)
        mutation.append({"field":field,"caught":not receipt[field]["pass"],"max_ulp":receipt[field]["max_ulp"]})
    module=load_module("satmedmf")
    attrs={name:module.get_function(name).attributes for name in ("satmedmf_column","satmedmf_fpvs_table","satmedmf_exner")}
    result={"verdict":"PASS" if all(all(v["pass"] for v in row["fields"].values()) for row in all_steps) and table_receipt["fpvs"]["pass"] and all(row["caught"] for row in mutation) else "FAIL",
      "columns":64,"levels":48,"steps_per_mode":3,"table":table_receipt,"modes":all_steps,"mutations":mutation,
      "kernel_attributes":attrs,"compile_platform":compile_platform_fingerprint(),
      "kernel_manifest":kernel_manifest()}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(result["verdict"],"max_local_size_bytes",max(a["local_size_bytes"] for a in attrs.values()),flush=True)
    if result["verdict"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()
