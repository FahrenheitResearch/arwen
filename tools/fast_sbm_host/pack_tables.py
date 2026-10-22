"""Pack FAST_HUCMINIT output from source tables, without oracle fixtures.

The C++ initialiser is verification tooling. The forecast reads only the
packed tables, whose manifest pins all input and implementation files.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from . import host as H
from .tables import load_tables
from gpuwm.core.fast_sbm_contract import TABLE_ORDER


def pack(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    receipt={"source_files":{},"outputs":{}}
    for prec in ("float32","float64"):
        dtype=np.float32 if prec=="float32" else np.float64
        t=load_tables(prec)
        raw=[]
        for name in TABLE_ORDER:
            a=getattr(t,name)
            if name in ("ywll","ywls","ywlg","ywlh","ywss"):
                a=a.transpose(1,2,0)
            raw.append(np.ascontiguousarray(a.ravel(order="F"),dtype=dtype))
        h=H.Host(raw,H.flags_corrected(),"f32" if prec=="float32" else "f64")
        arrays=dict(zip(TABLE_ORDER,raw)); arrays.update(h.derived())
        path=out/f"tables-{prec}.npz"
        np.savez_compressed(path,**arrays)
        receipt["outputs"][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for p in [H.KERNEL_DIR/name for name in H.HEADERS] + [H.SOURCE, Path(__file__)]:
        receipt["source_files"][str(p.relative_to(H.REPO_ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((out/'SBM_input_33').glob('*')):
        if p.is_file(): receipt["source_files"][str(p.relative_to(H.REPO_ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'packed-manifest.json').write_text(json.dumps(receipt,indent=2)+"\n")

if __name__=="__main__":
    pack(H.REPO_ROOT/'gpuwm/data/fast_sbm')
