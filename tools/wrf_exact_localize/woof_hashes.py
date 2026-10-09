"""woof_hashes.py WRFOUT_DIR OUT: hashes.json (recording form) of WOOF's per-step history files, plus arrays at steps 0-3 in OUT/steps.npz."""
import json, sys, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, "/work/pverify/combo/replay/src")
from gpuwm.verify_exact.replay import read_frame
d, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
files = sorted(d.glob("wrfout_d01_*"))
fields = {}; keep = {}
for f, p in enumerate(files):
    arrs = read_frame(p)
    for k, v in arrs.items():
        a = np.ascontiguousarray(v.astype(v.dtype.newbyteorder("<")))
        fields.setdefault(k, {"sha256": [None] * len(files)})["sha256"][f] = hashlib.sha256(a.tobytes()).hexdigest()
        if f <= 3:
            keep[f"{k}@{f}"] = a
json.dump({"fields": fields, "files": [p.name for p in files]}, open(out / "hashes.json", "w"))
np.savez(out / "steps.npz", **keep)
print(len(files), "frames")
