"""Per-field distance of the Grell 3D port from the WRF oracle, one arm.

usage: GPUWM_WRF_EXACT=0|1 python tools/g3_wrf461_oracle/g3_arm_report.py OUT.csv

Writes one row per (configuration, field): graded words, differing words,
max ULP distance, max |diff|, the field's max |WRF|, and the number of
band columns whose integer level or trigger decision differs.  The default
arm's tolerance in tests/test_g3_cu5_parity.py is set from this table and
the port report quotes it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gpuwm import wrf_exact  # noqa: E402
import test_g3_cu5_parity as t  # noqa: E402


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-2**31) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2**31) - ib, ib)
    return np.abs(ia - ib)


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    for name in t.CONFIGS:
        fx = t._fixture(name)
        got = t.run_fixture(fx)
        trig_w = fx["out_raincv"] > 0
        trig_g = got["raincv"] > 0
        for key in sorted(fx):
            if not key.startswith("out_") or key[4:] in t.NOT_GRADED:
                continue
            field = key[4:]
            want = fx[key]
            have = got[field].reshape(want.shape)
            if want.dtype.kind == "i":
                nd = int(np.sum(want != have))
                rows.append((name, field, want.size, nd, "",
                             int(np.max(np.abs(want - have))) if nd else 0,
                             int(np.max(np.abs(want))), ""))
                continue
            d = want.view(np.int32) != have.astype(np.float32).view(np.int32)
            ulp = ulp_distance(want, have)
            diff = np.abs(want.astype(np.float64) - have.astype(np.float64))
            rows.append((name, field, want.size, int(d.sum()),
                         int(ulp.max()), float(diff.max()),
                         float(np.abs(want).max()), ""))
        rows.append((name, "deep-trigger-flips", int(trig_w.size),
                     int(np.sum(trig_w != trig_g)), "", "", "", ""))
    import cupy as cp
    from gpuwm.core.g3 import run_g3_diag
    for path in sorted((ROOT / "tests/data/oracles/g3_diag").glob("diag-*.npz")):
        with np.load(path) as z:
            fx = {key: z[key] for key in z.files}
        diag = {}
        metrics = {}
        for n in range(len(fx["dt"])):
            def volume(key):
                return cp.asarray(np.ascontiguousarray(fx["in_"+key][n].transpose(1,0,2)))
            run_g3_diag(diag,cloud=volume("cloud"),ice=volume("ice"),
                rain=cp.asarray(fx["in_rain"][n]),temperature=volume("t"),pressure=volume("p"),
                qv=volume("qv"),qc=volume("qc"),qi=volume("qi"),fqi=int(fx["fqi"]),
                dt=float(fx["dt"][n]),curr_secs=float(fx["curr"][n]),avg_minutes=float(fx["avg"]),
                adaptive=bool(fx["adapt"]))
            for key in sorted(key for key in fx if key.startswith("out_")):
                field = key[4:]
                want=fx[key][n]; have=diag[field].get()
                if have.ndim==3: have=have.transpose(1,0,2)
                row=metrics.setdefault(field,[0,0,0,0.,0.])
                row[0]+=want.size
                row[1]+=int(np.sum(want.view("u4")!=have.view("u4")))
                row[2]=max(row[2],int(ulp_distance(want,have).max()))
                row[3]=max(row[3],float(np.abs(want.astype("f8")-have.astype("f8")).max()))
                row[4]=max(row[4],float(np.abs(want).max()))
        rows.extend((path.stem,field,*values,"") for field,values in metrics.items())
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "field", "words", "differing", "max_ulp",
                    "max_abs_diff", "max_abs_wrf", "note"])
        w.writerows(rows)
    arm = "strict" if wrf_exact.ENABLED else "default"
    total = sum(r[3] for r in rows if r[1] != "deep-trigger-flips")
    print(f"{arm} arm: {total} differing words over {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
