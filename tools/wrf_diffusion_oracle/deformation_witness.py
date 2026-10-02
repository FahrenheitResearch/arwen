"""Record the largest cancellation witnesses without changing acceptance pins."""
import argparse
import json
import numpy as np
from deformation_compare import port_outputs
from gpuwm.core.fp32_ulp import fp32_ulp_distance

parser=argparse.ArgumentParser();parser.add_argument("fixtures");parser.add_argument("output")
args=parser.parse_args();out=[]
from pathlib import Path
for name,field in (("evolved_real_open","div"),("real_open","d11")):
    with np.load(Path(args.fixtures)/f"deformation-{name}.npz") as data:
        arrays={k.removeprefix("input__"):data[k] for k in data.files if k.startswith("input__")}
        meta=json.loads(str(data["meta_json"]))
        metrics={k.removeprefix("metric__"):data[k] for k in data.files if k.startswith("metric__")}
        actual=port_outputs(arrays,meta)[field]
        expected=data["ref__km4_iso0__"+field]
        distance=fp32_ulp_distance(actual,expected)
        index=tuple(int(i) for i in np.unravel_index(np.argmax(distance),distance.shape))
        diagnostic=port_outputs(arrays,meta,contract=False,metrics=metrics,reference_order=True)[field]
        out.append({"case":name,"field":field,"index_k_j_i":index,"ulp":int(distance[index]),
                    "gpu":float(actual[index]),"wrf":float(expected[index]),
                    "gpu_word":f"{actual[index].view(np.uint32):08x}",
                    "wrf_word":f"{expected[index].view(np.uint32):08x}",
                    "restored_order_word":f"{diagnostic[index].view(np.uint32):08x}"})
Path(args.output).write_text(json.dumps(out,indent=2)+"\n", encoding="utf-8", newline="\n")
print(json.dumps(out,indent=2))
