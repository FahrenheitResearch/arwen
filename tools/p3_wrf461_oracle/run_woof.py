#!/usr/bin/env python3
"""Run WOOF's P3 device kernels over the oracle fixtures, FORMAT.md dumps.

    python run_woof.py <fixdir> <outdir> [--arms unfused,fused,levels,sedlevels]

Calls ``gpuwm.core.p3_device.run_p3_device`` -- the function the forecast
step (``gpuwm.core.p3.apply``) calls -- with the wrapper's own locals
(nc and ssat zero, mp=50 specified Nc) and the tables ``device_tables()``
uploads in a run.  Every arm in ``ARM_KERNELS`` can be driven; the forecast
selects ``CONFIG_ARM['cuda']``.

Strict build: run with GPUWM_WRF_EXACT=1 in the environment; gpuwm's import
hook then rewrites every NVRTC compile to the strict options, and this
script writes the compile receipt next to the dumps so the run proves which
options actually compiled the kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

POISON = np.float32(-8.888e30)
IN3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
        "th_old", "qv_old", "pii", "p", "dz", "w"]
IN2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
OUT3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
         "th_old", "qv_old", "zdbz", "effc", "effi", "vmi", "di", "rhopo"]
OUT2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixdir")
    ap.add_argument("outdir")
    ap.add_argument("--arms", default="unfused,fused,levels,sedlevels")
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    import cupy as cp

    import gpuwm
    from gpuwm.core import p3_device as PD
    from gpuwm import wrf_exact

    fixdir = pathlib.Path(args.fixdir)
    outroot = pathlib.Path(args.outdir)
    index = json.loads((fixdir / "index.json").read_text())
    tables = PD.device_tables()
    module = PD.p3_module(PD.DEFAULT_OPTIONS)
    src = PD.p3_source()
    want = [w for w in args.cases.split(",") if w]

    receipt = {
        "gpuwm_file": gpuwm.__file__,
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "wrf_exact_enabled": wrf_exact.ENABLED,
        "requested_options": list(PD.DEFAULT_OPTIONS),
        "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "p3_cu_sha256": hashlib.sha256(
            pathlib.Path(PD._P3_SOURCE).read_bytes()).hexdigest(),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "compute_capability": "".join(
            str(x) for x in cp.cuda.Device().compute_capability),
        "cupy": cp.__version__,
        "nvrtc": ".".join(str(x) for x in cp.cuda.nvrtc.getVersion()),
        "arm_for_forecast": PD.CONFIG_ARM["cuda"],
        "cases": {},
    }
    for arm in args.arms.split(","):
        outdir = outroot / arm
        outdir.mkdir(parents=True, exist_ok=True)
        for case in index["cases"]:
            name = case["name"]
            if want and not any(name.startswith(w) for w in want):
                continue
            ncol, nk = int(case["ncol"]), int(case["nk"])
            nsteps, dt = int(case["nsteps"]), float(case["dt"])
            buf = np.fromfile(fixdir / (name + ".bin"), dtype="<f4")
            assert buf.size == len(IN3D) * ncol * nk + len(IN2D) * ncol
            h3, off = {}, 0
            for f in IN3D:          # (ncol,nk) Fortran == (nk,ncol) C order
                h3[f] = buf[off:off + ncol * nk].reshape((nk, ncol)).copy()
                off += ncol * nk
            h2 = {}
            for f in IN2D:
                h2[f] = buf[off:off + ncol].copy()
                off += ncol
            fields = {f: cp.asarray(h3[f]) for f in
                      ("qc", "qr", "nr", "qi", "qir", "ni", "qib", "th",
                       "qv", "th_old", "qv_old")}
            fields["pres"] = cp.asarray(h3["p"])
            fields["dz"] = cp.asarray(h3["dz"])
            fields["nc"] = cp.zeros((nk, ncol), dtype=cp.float32)    # :847
            fields["ssat"] = cp.zeros((nk, ncol), dtype=cp.float32)  # :851
            diag = {d: cp.full((nk, ncol), POISON, dtype=cp.float32)
                    for d in PD.DIAG_SLOTS}
            surf = {"prt_liq": cp.zeros(ncol, dtype=cp.float32),
                    "prt_sol": cp.zeros(ncol, dtype=cp.float32)}
            for s in OUT2D:
                surf[s] = cp.asarray(h2[s])
            ws = PD.make_workspace(ncol, nk)
            frames = []

            def snap():
                parts = [cp.asnumpy(fields[f] if f in fields else diag[f])
                         .ravel() for f in OUT3D]
                parts += [cp.asnumpy(surf[s]).ravel() for s in OUT2D]
                frames.append(np.concatenate(parts).astype("<f4"))

            snap()
            t0 = time.perf_counter()
            for it in range(1, nsteps + 1):
                PD.run_p3_device(fields, diag, surf, workspace=ws,
                                 tables=tables, dt=dt, it=it, arm=arm,
                                 module=module)
                cp.cuda.Stream.null.synchronize()
                snap()
            path = outdir / (name + ".ref")
            np.concatenate(frames).tofile(path)
            receipt["cases"][f"{arm}/{name}"] = {
                "seconds": time.perf_counter() - t0,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            print(f"{arm:9s} {name}: {nsteps} steps", flush=True)
    receipt["compile"] = wrf_exact.compile_receipt()
    (outroot / "receipt.json").write_text(json.dumps(receipt, indent=1,
                                                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
