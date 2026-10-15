"""Grade WOOF's Kain-Fritsch kernel against the WRF v4.6.1 column oracle.

Runs WOOF's real GPU path -- ``gpuwm.core.kf.advance_w0avg`` (the W0AVG
running mean of KF_eta_CPS) and ``gpuwm.core.kf.launch_kf`` (KF_eta_PARA
plus KF_eta_CPS's tendency mapping) -- on the exact float32 words the
Fortran read, then compares every output word.

    python compare_woof.py COLUMNS.bin ORACLE.out REPORT.json [--label L]

Arithmetic is whatever the process was started with: GPUWM_WRF_EXACT=1 for
the strict build, unset for the default build.  The report states which.

Compared, per column: RTHCUTEN, RQVCUTEN, RQCCUTEN, RQRCUTEN, RQICUTEN
(when F_QI), RQSCUTEN (when F_QS), the advanced W0AVG, RAINCV, PRATEC, NCA
(where WRF set it; WOOF reports 0 where WRF leaves the driver's timer
alone), CUTOP, CUBOT, SHALL, CLDFRA_DP_KF, CLDFRA_SH_KF, QC_KF, QI_KF.
Columns on which WRF called wrf_error_fatal are reported and never counted
as matches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from . import kf_columns as K
except ImportError:                       # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kf_columns as K  # noqa: E402

FIELDS_3D = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
             "rqscuten", "w0avg", "cldfra_dp", "cldfra_sh", "qc_kf",
             "qi_kf")
FIELDS_2D = ("raincv", "pratec", "nca", "cutop", "cubot", "shall")


def ordered(bits):
    bits = bits.astype(np.int64)
    return np.where(bits < 0x80000000, bits, 0x80000000 - bits)


def ulp_distance(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    ia = ordered(a.view(np.uint32))
    ib = ordered(b.view(np.uint32))
    return np.abs(ia - ib)


def run(columns_path, oracle_path, label=None, meta_path=None):
    import cupy as cp
    from gpuwm import wrf_exact
    wrf_exact.install()
    from gpuwm.core.kf import KFPhaseMode, advance_w0avg, launch_kf, \
        w0avg_factors

    cols = K.read_columns(columns_path)
    outs = K.read_outputs(oracle_path)
    if len(cols) != len(outs):
        raise SystemExit("column/oracle count mismatch")
    groups = defaultdict(list)
    for i, c in enumerate(cols):
        nz, warm, fqr, fqi, fqs, stepcu, adapt, trig = c["ints"]
        if trig != 1:
            raise SystemExit("only trigger=1 is ported")
        key = (nz, warm, fqr, fqi, fqs, stepcu, adapt,
               *(float(x) for x in c["scal"][:3]))
        groups[key].append(i)

    woof = [None] * len(cols)
    for key, idx in groups.items():
        nz, warm, fqr, fqi, fqs, stepcu, adapt, dt, dx, cudt = key
        if warm:
            mode = KFPhaseMode.WARM_RAIN
        elif not fqs:
            mode = KFPhaseMode.NO_SEPARATE_SNOW
        elif not fqi:
            mode = KFPhaseMode.SEPARATE_SNOW
        else:
            mode = KFPhaseMode.SEPARATE_ICE_SNOW
        n = len(idx)

        def stack(name, levels=nz):
            arr = np.stack([cols[i][name] for i in idx], axis=1)
            return cp.asarray(arr.reshape(levels, 1, n).astype(np.float32))

        w = stack("w", nz + 1)
        w0 = stack("w0avg")
        factors = w0avg_factors(cudt_minutes=cudt, dt=dt, stepcu=stepcu,
                                adaptive=bool(adapt))
        advance_w0avg(w0, w, factors=factors)
        res = launch_kf(stack("u"), stack("v"), stack("t"), stack("qv"),
                        stack("p"), stack("rho"), stack("pi"), stack("dz"),
                        w0, dx=dx, dt=dt, cudt_minutes=cudt,
                        phase_mode=mode, f_qr=bool(fqr),
                        cloud_diagnostics=True)
        host = {k: cp.asnumpy(v) for k, v in res.items()}
        w0h = cp.asnumpy(w0)
        for j, i in enumerate(idx):
            rec = {name: host[name][:, 0, j].copy() for name in
                   ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten",
                    "rqicuten", "rqscuten", "cldfra_dp", "cldfra_sh",
                    "qc_kf", "qi_kf")}
            rec["w0avg"] = w0h[:, 0, j].copy()
            rec["raincv"] = host["raincv"][0, j]
            rec["pratec"] = host["pratec"][0, j]
            rec["nca"] = host["nca_seconds"][0, j]
            rec["cutop"] = host["cutop"][0, j]
            rec["cubot"] = host["cubot"][0, j]
            rec["shall"] = host["shall"][0, j]
            woof[i] = rec

    meta_path = meta_path or str(columns_path) + ".json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    else:
        meta = [{} for _ in cols]
    per_field = {f: dict(words=0, mismatched=0, max_ulp=0, max_abs=0.0)
                 for f in FIELDS_3D + FIELDS_2D}
    col_rows = []
    fatal = 0
    for i, (c, o, w) in enumerate(zip(cols, outs, woof)):
        if o["fatal"]:
            fatal += 1
            col_rows.append(dict(index=i, fatal=True, **meta[i]))
            continue
        nz, warm, fqr, fqi, fqs = c["ints"][:5]
        bad = []
        for f in FIELDS_3D:
            ref = o[f]
            got = w[f]
            if f == "rqicuten" and not fqi:
                continue
            if f == "rqscuten" and not fqs:
                continue
            d = ulp_distance(ref, got)
            st = per_field[f]
            st["words"] += d.size
            m = int(np.count_nonzero(d))
            st["mismatched"] += m
            if m:
                st["max_ulp"] = max(st["max_ulp"], int(d.max()))
                st["max_abs"] = max(st["max_abs"], float(np.max(np.abs(
                    ref.astype(np.float64) - got.astype(np.float64)))))
                bad.append(f)
        for f in FIELDS_2D:
            ref = np.float32(o[f])
            got = np.float32(w[f])
            if f == "nca":
                nca_in = np.float32(c["scal"][3])
                if ref.tobytes() == nca_in.tobytes():
                    ref = np.float32(0.0)      # WRF left the timer alone
            d = ulp_distance([ref], [got])
            st = per_field[f]
            st["words"] += 1
            if d[0]:
                st["mismatched"] += 1
                st["max_ulp"] = max(st["max_ulp"], int(d[0]))
                st["max_abs"] = max(st["max_abs"], float(abs(
                    np.float64(ref) - np.float64(got))))
                bad.append(f)
        col_rows.append(dict(index=i, shall=int(o["shall"]),
                             woof_shall=int(w["shall"]),
                             exact=not bad, fields=bad, **meta[i]))
    graded = [r for r in col_rows if not r.get("fatal")]
    by_regime = defaultdict(lambda: [0, 0])
    by_shall = defaultdict(lambda: [0, 0])
    for r in graded:
        by_regime[r.get("regime", "?")][0] += 1
        by_regime[r.get("regime", "?")][1] += int(r["exact"])
        by_shall[r["shall"]][0] += 1
        by_shall[r["shall"]][1] += int(r["exact"])
    total_words = sum(v["words"] for v in per_field.values())
    total_bad = sum(v["mismatched"] for v in per_field.values())
    from gpuwm.core.kf import KF_MODULE_OPTIONS
    return dict(
        label=label,
        strict=wrf_exact.ENABLED,
        compile_options=list(wrf_exact.effective_options(KF_MODULE_OPTIONS)
                             if wrf_exact.ENABLED else KF_MODULE_OPTIONS),
        columns=len(cols), graded=len(graded), wrf_fatal=fatal,
        exact_columns=sum(int(r["exact"]) for r in graded),
        words=total_words, mismatched_words=total_bad,
        max_ulp=max(v["max_ulp"] for v in per_field.values()),
        max_abs=max(v["max_abs"] for v in per_field.values()),
        per_field=per_field,
        by_regime={k: dict(columns=v[0], exact=v[1])
                   for k, v in sorted(by_regime.items())},
        by_shall={str(k): dict(columns=v[0], exact=v[1])
                  for k, v in sorted(by_shall.items())},
        mismatched_columns=[r for r in col_rows if not r.get("exact", True)],
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("columns")
    ap.add_argument("oracle")
    ap.add_argument("report")
    ap.add_argument("--label")
    a = ap.parse_args(argv)
    rep = run(a.columns, a.oracle, a.label)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(json.dumps({k: rep[k] for k in (
        "label", "strict", "columns", "graded", "wrf_fatal", "exact_columns",
        "words", "mismatched_words", "max_ulp", "max_abs")}, default=float))
    print("by_shall", json.dumps(rep["by_shall"]))
    for f, v in rep["per_field"].items():
        if v["mismatched"]:
            print(f"  {f}: {v['mismatched']}/{v['words']} words, "
                  f"max {v['max_ulp']} ulp, max |d| {v['max_abs']:.3e}")
    return 0 if rep["mismatched_words"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
