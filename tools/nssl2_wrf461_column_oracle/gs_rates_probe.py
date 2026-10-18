"""WOOF half of the GS variable attribution (diagnostic, never a grade).

  gs_rates_probe.py PREFIX COLUMNS OUT.npz NAMES.json --dt DT

Runs WOOF's fused GS from WRF's stage-1 state exactly as
run_gpu_isolated.py does, on a copy of the nssl2_fused_gs source whose
NSSL2_GS_DUMP_RATES hook (a no-op in the shipped kernel) is defined to write
the named variables, the same names instrument_gs.py dumps from WRF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_stages import WOOF_FIELDS, read_dumps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix", type=Path)
    p.add_argument("columns", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("names", type=Path, help="names.json written by instrument_gs.py")
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--stages", type=Path, default=None,
                   help="WRF stage dump to start from (default PREFIX.wrf_stages.bin)")
    a = p.parse_args()

    import cupy as cp
    import gpuwm.core.kernels as kernels
    names = json.loads(a.names.read_text())["names"]
    nd = len(names) + 1
    original = kernels.module_source

    def patched(name, **kw):
        src = original(name, **kw)
        if name != "nssl2_fused_gs":
            return src
        stub = "#define NSSL2_GS_DUMP_RATES(dump) ((void)0)"
        assert src.count(stub) == 1
        body = " ".join(f"d__[{q}] = (float)({n});" for q, n in enumerate(names))
        macro = ("#define NSSL2_GS_DUMP_RATES(dump) do { if (g_n2m_dump) { "
                 "float* d__ = g_n2m_dump + (size_t)(blockDim.x * blockIdx.x + threadIdx.x)"
                 f" * {nd}; {body} d__[{nd - 1}] = 1.0f; }} }} while (0)")
        return "__device__ float* g_n2m_dump;\n" + src.replace(stub, macro)

    kernels.module_source = patched
    from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
    from gpuwm.core.nssl2_fused_gs import launch_fused_gs

    regime = np.load(a.columns)["regime"]
    ncol = regime.size
    fin = np.load(a.prefix.with_name(a.prefix.name + ".fin.npz"))
    nz = fin["theta"].shape[1]
    wrf = read_dumps(a.stages or a.prefix.with_name(a.prefix.name + ".wrf_stages.bin"), ncol)
    shape = (nz, 1, ncol)

    def dev(x):
        return cp.asarray(np.ascontiguousarray(np.asarray(x, np.float32).T[:, None, :]))

    rho, pres, pii, dz = (dev(fin[n]) for n in ("rho", "pressure", "exner", "dz"))
    w = dev(fin["w"])
    ws = NSSL2DriverWorkspace(cp.empty((16, *shape), dtype=cp.float32),
                              cp.zeros((5, 1, ncol), dtype=cp.float32), shape,
                              ignored_accumulator=cp.zeros((1, ncol), dtype=cp.float32))
    for i, name in enumerate(WOOF_FIELDS):
        ws.state[i] = dev(wrf[1][name])
    theta = dev(wrf[1]["theta"])
    buf = cp.zeros((nz * ncol, nd), dtype=cp.float32)
    mod = kernels.load_module("nssl2_fused_gs")
    g = mod.get_global("g_n2m_dump")
    pointer = np.array([buf.data.ptr], dtype=np.uint64)
    cp.cuda.runtime.memcpy(g.ptr, pointer.ctypes.data, 8, cp.cuda.runtime.memcpyHostToDevice)
    temperature = cp.empty(shape, dtype=cp.float32)
    target = cp.empty(shape, dtype=cp.float32)
    launch_fused_gs(ws, theta, rho, pres, pii, w, temperature, target, dz,
                    np.float32(a.dt), hail_on=True)
    cp.cuda.Device().synchronize()
    host = cp.asnumpy(buf).reshape(nz, ncol, nd)
    out = {n: host[:, :, q].T for q, n in enumerate(names)}
    out["__gathered"] = host[:, :, nd - 1].T
    np.savez(a.out, **out)
    print("wrote", a.out, len(out), "fields")


if __name__ == "__main__":
    main()
