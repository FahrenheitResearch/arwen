"""Find Kain-Fritsch columns whose capped downdraft leaves a negative rain flux.

WRF caps DDINC = TRPPT/TDER when the trial downdraft would evaporate more than
the updraft rains, then forms PPTFLX = TRPPT-TDER (module_cu_kfeta.F:1810-1821).
In binary32 that difference rounds to either side of zero.  kernels/kf.cu
floors it at zero (a declared divergence, d547dc8c3).  This probe compiles a
CONTROL copy of kf.cu with WRF's bare subtraction restored, runs it and the
production kernel on the same columns, and writes the columns where the
control's PRATEC is negative -- the inputs of the floor's regression test.

    python floor_probe.py OUT.npz [--blocks B --per-block C --seed N]
                                  [--columns COLS.bin ...]

One card.  Arithmetic is whatever the process was started with.
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

FLOORED = "pptflx = fmaxf(trppt - tder, 0.0f);"
WRF_BARE = "pptflx = trppt - tder;"
OUT_2D = ("raincv", "pratec", "nca_seconds", "cutop", "cubot", "shall")
OUT_3D = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
          "rqscuten")
INPUTS = ("u", "v", "t", "qv", "p", "rho", "pi", "dz")


def control_source() -> str:
    from gpuwm.core.kernels import module_source
    source = module_source("kf")
    if source.count(FLOORED) != 1:
        raise SystemExit("kf.cu no longer carries exactly one floored PPTFLX")
    return source.replace(FLOORED, WRF_BARE)


def run_columns(cols, *, control: bool):
    """Run every column group through launch_kf; return per-column outputs."""
    import cupy as cp
    from gpuwm.core import kf as kfmod
    from gpuwm.core.kf import (KFPhaseMode, advance_w0avg, launch_kf,
                               w0avg_factors)

    import gpuwm.core.kernels as kernels
    saved = kernels.module_source
    if control:
        text = control_source()
        kernels.module_source = (lambda name, *a, **k:
                                 text if name == "kf" else saved(name, *a, **k))
    kfmod._kf_module.cache_clear()
    try:
        groups = defaultdict(list)
        for i, c in enumerate(cols):
            nz, warm, fqr, fqi, fqs, stepcu, adapt, trig = c["ints"]
            groups[(nz, warm, fqr, fqi, fqs, stepcu, adapt,
                    *(float(x) for x in c["scal"][:3]))].append(i)
        out = [None] * len(cols)
        for key, idx in groups.items():
            nz, warm, fqr, fqi, fqs, stepcu, adapt, dt, dx, cudt = key
            mode = (KFPhaseMode.WARM_RAIN if warm else
                    KFPhaseMode.NO_SEPARATE_SNOW if not fqs else
                    KFPhaseMode.SEPARATE_SNOW if not fqi else
                    KFPhaseMode.SEPARATE_ICE_SNOW)
            n = len(idx)

            def stack(name, levels=nz):
                arr = np.stack([cols[i][name] for i in idx], axis=1)
                return cp.asarray(arr.reshape(levels, 1, n).astype(np.float32))

            w0 = stack("w0avg")
            advance_w0avg(w0, stack("w", nz + 1),
                          factors=w0avg_factors(cudt_minutes=cudt, dt=dt,
                                                stepcu=stepcu,
                                                adaptive=bool(adapt)))
            res = launch_kf(*(stack(k) for k in ("u", "v", "t", "qv", "p",
                                                  "rho", "pi", "dz")),
                            w0, dx=dx, dt=dt, cudt_minutes=cudt,
                            phase_mode=mode, f_qr=bool(fqr))
            host = {k: cp.asnumpy(res[k]) for k in OUT_2D + OUT_3D}
            for j, i in enumerate(idx):
                rec = {k: host[k][0, j] for k in OUT_2D}
                rec.update({k: host[k][:, 0, j].copy() for k in OUT_3D})
                out[i] = rec
        return out
    finally:
        kernels.module_source = saved
        kfmod._kf_module.cache_clear()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--per-block", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--columns", nargs="*", default=())
    args = ap.parse_args(argv)
    from gpuwm import wrf_exact
    wrf_exact.install()
    sets = []
    for path in args.columns:
        sets.append((path, K.read_columns(path)))
    if args.blocks:
        import tempfile
        cols = K.sweep(args.seed, args.blocks, args.per_block)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sweep.bin")
            K.write_columns(path, cols)
            sets.append((f"sweep(seed={args.seed},{args.blocks}x"
                         f"{args.per_block})", K.read_columns(path)))
    found = []
    summary = {"arithmetic": "strict" if wrf_exact.ENABLED else "default",
               "sets": []}
    for label, cols in sets:
        prod = run_columns(cols, control=False)
        ctrl = run_columns(cols, control=True)
        neg = [i for i, r in enumerate(ctrl) if r["pratec"] < 0.0]
        differ = []
        for i, (a, b) in enumerate(zip(prod, ctrl)):
            names = [k for k in OUT_2D + OUT_3D
                     if np.asarray(a[k]).view(np.uint32).tolist()
                     != np.asarray(b[k]).view(np.uint32).tolist()]
            if names:
                differ.append((i, names))
        summary["sets"].append({
            "label": label, "columns": len(cols),
            "control_negative_pratec": len(neg),
            "control_min_pratec": float(min(r["pratec"] for r in ctrl)),
            "production_min_pratec": float(min(r["pratec"] for r in prod)),
            "columns_where_production_and_control_differ": len(differ),
            "differing_outputs": sorted({n for _, ns in differ for n in ns}),
            "differ_only_on_negative_columns": all(i in neg for i, _ in differ),
        })
        for i in neg:
            found.append((cols[i], ctrl[i]["pratec"], prod[i]["pratec"]))
    print(json.dumps(summary, indent=1))
    if found:
        nz = {c["ints"][0] for c, _, _ in found}
        if len(nz) != 1:
            found = [f for f in found if f[0]["ints"][0] == max(nz)]
        rec = {k: np.stack([c[k] for c, _, _ in found], axis=1)
               for k in INPUTS + ("w", "w0avg")}
        rec["ints"] = np.stack([np.asarray(c["ints"]) for c, _, _ in found])
        rec["scal"] = np.stack([np.asarray(c["scal"]) for c, _, _ in found])
        rec["control_pratec"] = np.asarray([f[1] for f in found], np.float32)
        rec["production_pratec"] = np.asarray([f[2] for f in found], np.float32)
        np.savez_compressed(args.out, **rec)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
