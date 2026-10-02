"""Seal portable source, fixture and arithmetic-attribution receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def seal(root):
    root=Path(root).resolve()
    folder=root/"tests/data/wrf471_diffusion"
    receipt_path=folder/"deformation-build-receipt.json"
    receipt=json.loads(receipt_path.read_text())
    receipt["commands"]=[[Path(arg).name if arg.startswith("/") else arg for arg in command]
                         for command in receipt["commands"]]
    receipt_path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8", newline="\n")
    paths=sorted(folder.glob("deformation-*.npz"))+sorted(folder.glob("deformation-*.json"))
    paths+=sorted(folder.glob("rounding-*.json"))
    paths+=sorted((folder/"producer-snapshots").glob("*"))
    paths+=sorted((root/"tools/wrf_diffusion_oracle").glob("deformation_*.py"))
    paths+=[root/"tools/wrf_diffusion_oracle/deformation_wrappers.F90",
            root/"tools/wrf_diffusion_oracle/deformation_probe.cu",
            root/"tools/wrf_diffusion_oracle/build_common.py",root/"tools/wrf_diffusion_oracle/cases.py",
            root/"tools/wrf_diffusion_oracle/horizontal_compare.py",root/"tools/wrf_diffusion_oracle/vertical_evolved.py"]
    paths+=[root/"tools/wrf_diffusion_oracle/rounding_attribution.py",
            root/"tools/wrf_diffusion_oracle/rounding_intermediate_probe.F90",
            root/"gpuwm/core/kernels/glibc_flt32.cuh"]
    lines=[hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.relative_to(root).as_posix()
           for path in paths]
    (folder/"deformation-sha256sums.txt").write_text("\n".join(lines)+"\n",encoding="utf-8", newline="\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("root",type=Path)
    seal(parser.parse_args().root)
