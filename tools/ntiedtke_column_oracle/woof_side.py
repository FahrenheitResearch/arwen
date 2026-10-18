"""THE WOOF SIDE of the New Tiedtke column oracle.

Runs the shipped ``cu_physics = 16`` adapter, ``gpuwm.core.ntiedtke.
NewTiedtke``, on the GPU over a case written by make_columns.py, through
its own call signature: the atmosphere mapping, the surface fields, the
domain state and the run config the physics driver hands it, with the six
forcing lanes on a stand-in driver object exactly where PhysicsDriver keeps
them (``rthratenlw``, ``rthratensw``, ``gf_rthdynten``, ``gf_rthblten``,
``gf_rqvdynten``, ``gf_rqvblten``).  The fold, the chunking, the pad and
the fused walk are therefore the production code, not a re-assembly.

    python woof_side.py CASE OUT --dx DX [--replicate R] [--stepcu N]
                        [--elapsed SECONDS] [--no-bl-lanes]

--replicate R tiles the case R times along x.  Every replica must give the
same words as the unreplicated case: this is how the fused walk (more than
NT_FUSED_MIN_COLUMNS columns) and the multi-chunk walk with a padded tail
are reached with the same columns.

--stepcu N builds the pipeline with stepcu = N.  WOOF pins cudt_minutes = 0
for cu_physics = 16, so stepcu is 1 in every reachable run; N > 1 grades
the kernels' stepcu arithmetic only.

--elapsed is the model time of the call.  The adapter derives WRF's
ITIMESTEP from it; at 0 the call is itimestep = 1, where
cu_ntiedtke_pre_run zeroes the forcing (module_cu_ntiedtke.F:451-457).

Outputs are written in the case's layout (C order (ny, nz, nx*R)).  PRATEC
is read from the pipeline workspace when the run is one chunk, because the
adapter computes it and deliberately does not return it (PORT-RECORD 38).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_subset import load  # noqa: E402

LANES = {
    "rthratenlw": "rthratenlw", "rthratensw": "rthratensw",
    "gf_rthdynten": "rthdynten", "gf_rthblten": "rthblten",
    "gf_rqvdynten": "rqvdynten", "gf_rqvblten": "rqvblten",
}
OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten",
           "rvcuten")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--stepcu", type=int, default=1)
    ap.add_argument("--elapsed", type=float, default=3600.0)
    ap.add_argument("--no-bl-lanes", action="store_true")
    a = ap.parse_args()

    import cupy as cp

    import gpuwm.core.ntiedtke as ntm
    from gpuwm import wrf_exact

    meta, lev, iface, sfc = load(a.case)
    R = a.replicate

    def dev3(x):                      # (ny, nz, nx) -> (nz, ny, nx*R)
        y = np.ascontiguousarray(x.transpose(1, 0, 2))
        if R > 1:
            y = np.tile(y, (1, 1, R))
        return cp.asarray(y)

    def dev2(x):
        return cp.asarray(np.tile(x, (1, R)) if R > 1 else x)

    atmosphere = {
        "temperature": dev3(lev["t3d"]), "qv": dev3(lev["qv3d"]),
        "qc": dev3(lev["qc3d"]), "qi": dev3(lev["qi3d"]),
        "u": dev3(lev["u3d"]), "v": dev3(lev["v3d"]),
        "pressure": dev3(lev["pcps"]), "dz": dev3(lev["dz8w"]),
        "rho": dev3(lev["rho3d"]), "exner": dev3(lev["exner"]),
        "p_interface": dev3(iface["p8w"]),
    }
    driver = SimpleNamespace(**{
        attr: dev3(lev[name]) for attr, name in LANES.items()
        if not (a.no_bl_lanes and attr in ("gf_rthblten", "gf_rqvblten"))})
    fields = {n: dev2(sfc[n]) for n in ("xland", "hfx", "qfx")}
    dt = float(meta["dt"])
    state = SimpleNamespace(p=atmosphere["pressure"], w=dev3(iface["w"]),
                            elapsed_seconds=float(a.elapsed))
    cfg = SimpleNamespace(dx=float(a.dx), dt=dt, clock_dt=0.0,
                          ntiedtke_tiedtke_closure=False)

    if a.stepcu != 1:
        base_pipeline = ntm.NtPipeline

        class SteppedPipeline(base_pipeline):
            def __init__(self, **kw):
                kw["stepcu"] = a.stepcu
                super().__init__(**kw)

        ntm.NtPipeline = SteppedPipeline

    adapter = ntm.NewTiedtke()
    adapter.bind_driver(driver)
    t0 = time.time()
    result = adapter(atmosphere=atmosphere, fields=fields, state=state,
                     cfg=cfg)
    cp.cuda.Device().synchronize()
    elapsed = time.time() - t0

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in OUT_LEV:
        arr = cp.asnumpy(getattr(result, name))       # (nz, ny, nx*R)
        np.ascontiguousarray(arr.transpose(1, 0, 2)).astype("<f4").tofile(
            out / f"{name}.f32")
    cp.asnumpy(result.rainc).astype("<f4").tofile(out / "raincv.f32")

    nz, ny, nxr = atmosphere["pressure"].shape
    ncol = ny * nxr
    width = ntm.nt_tile_columns(ncol, ntm._nt_multiprocessor_count())
    chunks = -(-ncol // width)
    pipeline = adapter._pipeline[1]
    if chunks == 1:
        cp.asnumpy(pipeline.w.bind("pratec", 1)[:ncol]).reshape(
            ny, nxr).astype("<f4").tofile(out / "pratec.f32")
    info = {
        "ncol": ncol, "chunk_width": width, "chunks": chunks,
        "padded_columns": chunks * width - ncol,
        "fused_walk": width > ntm.NT_FUSED_MIN_COLUMNS,
        "multiprocessors": ntm._nt_multiprocessor_count(),
        "replicate": R, "stepcu": int(pipeline.scalars["stepcu"]),
        "itimestep_scalar": int(pipeline.scalars["itimestep"]),
        "elapsed_seconds": a.elapsed, "dx": a.dx, "dt": dt,
        "bl_lanes": not a.no_bl_lanes,
        "strict": bool(wrf_exact.ENABLED),
        "env_GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "pratec_graded": chunks == 1,
        "seconds": elapsed,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    if wrf_exact.ENABLED:
        receipt = wrf_exact.compile_receipt()
        info["strict_options"] = receipt["strict_options"]
        info["nvrtc_compiles"] = sum(
            1 for r in receipt["records"] if r["kind"] == "nvrtc_compile")
        info["compile_requests"] = len(receipt["records"])
        info["requests_missing_strict"] = sum(
            1 for r in receipt["records"]
            if "--fmad=false" not in r["options"])
    (out / "woof-run.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
