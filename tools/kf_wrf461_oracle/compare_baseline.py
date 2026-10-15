"""Grade the PRE-PORT Kain-Fritsch kernel (lane/verify-base, 91de806e2)
against the same WRF v4.6.1 column oracle, for the before/after record.

Run it with PYTHONPATH naming a tree that still has the old kernel; it uses
that tree's API exactly as its KainFritsch adapter did:

  * W0AVG: ``(w0avg*(TST-1) + 0.5*(w(k)+w(k+1)))/TST`` as a CuPy expression;
  * ``launch_kf(u, v, T, qv, qc, p, exner, dz, w0avg, dx, dt, cudt_seconds,
    phase_mode)`` with ``cudt_seconds`` the adapter's
    ``steps*clock_dt`` (``clock_dt`` at cudt = 0);
  * PRATEC recovered as the adapter did, ``rainc/min(cudt, nca)``.

The old kernel had no CLDFRA/QC_KF outputs and no F_QR fold, so those are
not graded here; everything else is compared exactly as compare_woof.py
does.  qc is passed as zeros (the old kernel did not read it).

    python compare_baseline.py COLUMNS.bin ORACLE.out REPORT.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from . import kf_columns as K
    from .compare_woof import ulp_distance
except ImportError:                       # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kf_columns as K  # noqa: E402
    from compare_woof import ulp_distance  # noqa: E402


def main(columns_path, oracle_path, report_path):
    import cupy as cp
    from gpuwm import wrf_exact
    wrf_exact.install()
    from gpuwm.core.kf import KFPhaseMode, launch_kf

    cols = K.read_columns(columns_path)
    outs = K.read_outputs(oracle_path)
    groups = defaultdict(list)
    for i, c in enumerate(cols):
        nz, warm, fqr, fqi, fqs, stepcu, adapt, trig = c["ints"]
        groups[(nz, warm, fqr, fqi, fqs, stepcu, adapt,
                *(float(x) for x in c["scal"][:3]))].append(i)
    fields3 = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
               "rqscuten", "w0avg")
    fields2 = ("raincv", "pratec", "nca", "cutop", "cubot", "shall")
    stats = {f: dict(words=0, mismatched=0, max_ulp=0) for f in fields3 + fields2}
    exact_cols = 0
    graded = 0
    by_shall = {}
    for key, idx in groups.items():
        nz, warm, fqr, fqi, fqs, stepcu, adapt, dt, dx, cudt = key
        if not fqr:
            continue                  # the old kernel had no F_QR fold
        mode = (KFPhaseMode.WARM_RAIN if warm else
                KFPhaseMode.NO_SEPARATE_SNOW if not fqs else
                KFPhaseMode.SEPARATE_SNOW if not fqi else
                KFPhaseMode.SEPARATE_ICE_SNOW)
        n = len(idx)

        def stack(name, levels=nz):
            arr = np.stack([cols[i][name] for i in idx], axis=1)
            return cp.asarray(arr.reshape(levels, 1, n).astype(np.float32))

        w = stack("w", nz + 1)
        w0 = stack("w0avg")
        tst = np.float32(2 * stepcu)
        inst = cp.ascontiguousarray(np.float32(0.5) * (w[:-1] + w[1:]))
        w0[...] = (w0 * (tst - np.float32(1.0)) + inst) / tst
        steps = 1 if cudt <= 0 else max(int(math.floor(cudt * 60.0 / dt + 0.5)), 1)
        cudt_s = dt if cudt <= 0 else steps * dt
        qc = cp.zeros_like(w0)
        res = launch_kf(stack("u"), stack("v"), stack("t"), stack("qv"), qc,
                        stack("p"), stack("pi"), stack("dz"), w0,
                        dx=dx, dt=dt, cudt=cudt_s, phase_mode=mode)
        h = {k: cp.asnumpy(v) for k, v in res.items()}
        w0h = cp.asnumpy(w0)
        nca = h["nca_seconds"]
        div = np.minimum(np.float32(cudt_s), nca)
        pratec = np.where(div > 0, h["rainc"] / np.where(div > 0, div, 1),
                          0).astype(np.float32)
        for j, i in enumerate(idx):
            o = outs[i]
            graded += 1
            bad = False
            got = {f: h[f][:, 0, j] for f in fields3 if f != "w0avg"}
            got["w0avg"] = w0h[:, 0, j]
            for f in fields3:
                if (f == "rqicuten" and not fqi) or (f == "rqscuten" and not fqs):
                    continue
                d = ulp_distance(o[f], got[f])
                stats[f]["words"] += d.size
                m = int(np.count_nonzero(d))
                stats[f]["mismatched"] += m
                if m:
                    stats[f]["max_ulp"] = max(stats[f]["max_ulp"], int(d.max()))
                    bad = True
            trig = bool(h["triggered"][0, j])
            got2 = dict(
                pratec=pratec[0, j], raincv=np.float32(dt) * pratec[0, j],
                nca=nca[0, j],
                cutop=np.float32(h["cloud_top"][0, j] + 1) if trig else np.float32(1),
                cubot=np.float32(h["cloud_base"][0, j] + 1) if trig else np.float32(nz + 1),
                shall=np.float32(h["shallow"][0, j] if trig else 2))
            for f in fields2:
                ref = np.float32(o[f])
                if f == "nca" and ref.tobytes() == np.float32(cols[i]["scal"][3]).tobytes():
                    ref = np.float32(0.0)
                d = ulp_distance([ref], [got2[f]])
                stats[f]["words"] += 1
                if d[0]:
                    stats[f]["mismatched"] += 1
                    stats[f]["max_ulp"] = max(stats[f]["max_ulp"], int(d[0]))
                    bad = True
            exact_cols += int(not bad)
            row = by_shall.setdefault(str(int(o["shall"])), [0, 0])
            row[0] += 1
            row[1] += int(not bad)
    rep = dict(strict=wrf_exact.ENABLED, graded=graded, exact_columns=exact_cols,
               by_shall={k: dict(columns=v[0], exact=v[1])
                         for k, v in sorted(by_shall.items())},
               per_field=stats,
               words=sum(v["words"] for v in stats.values()),
               mismatched_words=sum(v["mismatched"] for v in stats.values()))
    json.dump(rep, open(report_path, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("strict", "graded", "exact_columns",
                                          "words", "mismatched_words",
                                          "by_shall")}))
    for f, v in stats.items():
        print(f"  {f}: {v['mismatched']}/{v['words']} max_ulp {v['max_ulp']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
