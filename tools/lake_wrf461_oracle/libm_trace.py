"""Which binary64 library calls separate the CUDA lake from native WRF?

Recompiles the production lake source with every binary64 intrinsic
(lake_support.cuh's lake_exp, lake_log, lake_log10, lake_sin, lake_atan and
lake_pow_real) routed through a recorder that keeps the argument and the
result, steps the regime inputs, and evaluates each recorded argument with
the host's libm (the library the native WRF oracle links) to count the calls
whose results differ.  The recorder returns the value the function returned,
so it changes no arithmetic.  ``--libm cuda`` records CUDA's own binary64
functions (the lake before the correctly rounded fix); ``--libm woof``
records f64cr.cuh's.

    python libm_trace.py OUT BUILD [--campaign dt60] [--steps N] [--libm cuda|woof] [--dump]

``BUILD`` must hold libm_host.so (built by build.sh).  ``--dump`` also saves
every distinct argument for cr_check.py.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

FUNCS = ("exp", "log", "log10", "sin", "atan", "pow")
CAP = 1 << 22

RECORDER = r"""
// ---- libm_trace recorder (diagnostic build only) ----
__device__ unsigned long long ltr_count;
__device__ int ltr_fn[%(cap)d];
__device__ double ltr_a[%(cap)d];
__device__ double ltr_b[%(cap)d];
__device__ double ltr_r[%(cap)d];
__device__ inline double ltr_put(int f,double a,double b,double r) {
    unsigned long long k=atomicAdd(&ltr_count,1ULL);
    if(k<%(cap)dULL){ltr_fn[k]=f;ltr_a[k]=a;ltr_b[k]=b;ltr_r[k]=r;}
    return r;
}
__device__ inline double LTR_exp(double x){return ltr_put(0,x,0.,%(exp)s(x));}
__device__ inline double LTR_log(double x){return ltr_put(1,x,0.,%(log)s(x));}
__device__ inline double LTR_log10(double x){return ltr_put(2,x,0.,%(log10)s(x));}
__device__ inline double LTR_sin(double x){return ltr_put(3,x,0.,%(sin)s(x));}
__device__ inline double LTR_atan(double x){return ltr_put(4,x,0.,%(atan)s(x));}
__device__ inline double LTR_pow(double x,double y){return ltr_put(5,x,y,%(pow)s(x,y));}
// ---- end recorder ----
"""


def traced_source(source: str, impl: dict[str, str]) -> str:
    """Route the lake's binary64 intrinsics (lake_support.cuh's lake_exp,
    ..., lake_pow_real) through LTR_*, which call ``impl``."""
    for f in FUNCS[:-1]:
        site = f"{{return wcr_{f}(x);}}"
        if site not in source:
            raise SystemExit(f"lake_support.cuh lake_{f} site moved; update libm_trace")
        source = source.replace(site, f"{{return LTR_{f}(x);}}")
    site = "{return wcr_pow(a,b);}"
    if site not in source:
        raise SystemExit("lake_support.cuh lake_pow_real site moved; update libm_trace")
    source = source.replace(site, "{return LTR_pow(a,b);}")
    anchor = "LAKE_HD double lake_exp(double x)"
    return source.replace(anchor, RECORDER % dict(cap=CAP, **impl) + anchor, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("build", type=Path)
    p.add_argument("--campaign", default="dt60")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--dump", action="store_true", help="also save every distinct recorded argument")
    p.add_argument("--libm", choices=("cuda", "woof"), default="woof",
                   help="cuda: CUDA's binary64 functions (the lake before the fix); woof: f64cr.cuh")
    a = p.parse_args()

    import cupy as cp
    import gpuwm  # noqa: F401  strict compiler hook when GPUWM_WRF_EXACT=1
    from gpuwm import wrf_exact
    from gpuwm.core.kernels import module_options, module_source
    from gpuwm.core.lake import lake_arena, launch_lake_columns
    from gpuwm.core.lake_schema import LAKE_OUTPUT_WORDS, LAKE_STATE_WORDS, LAKE_STATIC_WORDS

    impl = {f: (f if a.libm == "cuda" else f"wcr_{f}") for f in FUNCS}
    src = traced_source(module_source("lake"), impl)
    mod = cp.RawModule(code=src, options=module_options("lake"))
    init, step = mod.get_function("lake_init_columns"), mod.get_function("lake_step_columns")
    count_ptr = mod.get_global("ltr_count")
    bufs = {k: mod.get_global(f"ltr_{k}") for k in ("fn", "a", "b", "r")}
    count = cp.ndarray((1,), cp.uint64, count_ptr)
    fn = cp.ndarray((CAP,), cp.int32, bufs["fn"])
    xa = cp.ndarray((CAP,), cp.float64, bufs["a"])
    xb = cp.ndarray((CAP,), cp.float64, bufs["b"])
    xr = cp.ndarray((CAP,), cp.float64, bufs["r"])

    host = ctypes.CDLL(str(a.build / "libm_host.so"))
    D = ctypes.POINTER(ctypes.c_double)
    host.libm_host_eval.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_int), D, D, D]

    data = np.load(a.out / f"inputs-{a.campaign}.npz")
    seed, forcing, dt = data["seed"], data["forcing"], np.float32(data["dt"])
    n = seed.shape[1]
    names = json.loads((a.out / "columns.json").read_text())["names"]
    state = cp.zeros((LAKE_STATE_WORDS, n), cp.float32)
    static = cp.zeros((LAKE_STATIC_WORDS, n), cp.float32)
    output = cp.zeros((LAKE_OUTPUT_WORDS, n), cp.float32)
    errors = cp.zeros(n, cp.int32)
    arena = lake_arena(n)
    thr = np.float32(0.5)
    tally = {f: dict(calls=0, differing=0, max_ulp=0, examples=[]) for f in FUNCS}
    seen = {f: [] for f in FUNCS}

    def drain(label):
        k = int(count.get()[0])
        if k > CAP:
            raise SystemExit(f"recorder overflow at {label}: {k}")
        if k == 0:
            return
        f_h, a_h, b_h, r_h = (cp.asnumpy(v[:k]) for v in (fn, xa, xb, xr))
        ref = np.empty(k, np.float64)
        host.libm_host_eval(k, f_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), a_h.ctypes.data_as(D),
                            b_h.ctypes.data_as(D), ref.ctypes.data_as(D))
        ri, gi = ref.view(np.int64), r_h.view(np.int64)
        diff = ri != gi
        ulp = np.abs(np.where(ri < 0, -0x8000000000000000 - ri, ri).astype(object)
                     - np.where(gi < 0, -0x8000000000000000 - gi, gi).astype(object)) if diff.any() else None
        for i, f in enumerate(FUNCS):
            sel = f_h == i
            if a.dump and sel.any():
                pair = np.stack([a_h[sel].view(np.uint64), b_h[sel].view(np.uint64)], 1)
                seen[f].append(np.unique(pair, axis=0))
            t = tally[f]
            t["calls"] += int(sel.sum())
            d = diff & sel
            if d.any():
                t["differing"] += int(d.sum())
                t["max_ulp"] = max(t["max_ulp"], int(max(ulp[d])))
                for j in np.flatnonzero(d)[: max(0, 6 - len(t["examples"]))]:
                    t["examples"].append(dict(at=label, x=a_h[j].hex(), y=b_h[j].hex(),
                                              device=r_h[j].hex(), host=ref[j].hex()))
        count[...] = 0

    count[...] = 0
    launch_lake_columns(init, n, (n, cp.asarray(seed), state, static, 1, 1, np.float32(50), thr, errors), arena)
    drain("init")
    df = cp.asarray(forcing)
    steps = a.steps or forcing.shape[0]
    for s in range(steps):
        launch_lake_columns(step, n, (n, df[s], state, static, output, dt, thr, errors), arena)
        drain(f"step{s + 1}")
    assert not cp.asnumpy(errors).any()
    report = dict(campaign=a.campaign, steps=steps, columns=n, strict=wrf_exact.ENABLED,
                  libm=a.libm, impl=impl, functions=tally)
    tag = "strict" if wrf_exact.ENABLED else "default"
    name = f"libm-trace-{a.libm}-{tag}-{a.campaign}"
    if a.dump:
        np.savez_compressed(a.out / f"libm-inputs-{a.campaign}.npz", **{
            f: (np.unique(np.concatenate(v), axis=0) if v else np.zeros((0, 2), np.uint64)) for f, v in seen.items()})
    (a.out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({f: {k: v for k, v in t.items() if k != "examples"} for f, t in tally.items()}, indent=1))


if __name__ == "__main__":
    main()
