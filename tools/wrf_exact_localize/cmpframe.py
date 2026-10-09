"""cmpframe.py WRF_REC WOOF_REC STEP [FIELD...]: where history fields differ at one frame (word level)."""
import json
import sys

import numpy as np

wrec, orec, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
names = sys.argv[4:]
meta = json.load(open(f"{wrec}/step{step:04d}.json"))
raw = open(f"{wrec}/step{step:04d}.bin", "rb").read()
wf = {f["name"]: np.frombuffer(raw[f["offset"]:f["offset"] + f["nbytes"]], f["dtype"]).reshape(f["shape"])
      for f in meta["fields"]}
o = np.load(f"{orec}/steps.npz")
for n in names or sorted(set(wf) & {k.split("@")[0] for k in o.files}):
    key = f"{n}@{step}"
    if key not in o.files or n not in wf:
        continue
    a = wf[n].astype(np.float32)
    b = o[key].astype(np.float32).reshape(a.shape)
    bad = a.view(np.int32) != b.view(np.int32)
    if not bad.any():
        continue
    idx = np.argwhere(bad)
    if a.ndim >= 2:
        jj, ii = idx[:, -2], idx[:, -1]
        ny, nx = a.shape[-2:]
        edge = np.minimum.reduce([jj, ii, ny - 1 - jj, nx - 1 - ii])
        where = f"min edge {edge.min()}, edge<5 {np.mean(edge < 5) * 100:.0f}%"
    else:
        where = ""
    ks = sorted(set(int(x) for x in idx[:, 0])) if a.ndim == 3 else []
    print(f"{n:10s} differ {len(idx)}/{a.size} first {tuple(int(x) for x in idx[0])} "
          f"WRF {a[tuple(idx[0])]!r} WOOF {b[tuple(idx[0])]!r} {where}" + (f" k {ks[0]}..{ks[-1]}" if ks else ""))
