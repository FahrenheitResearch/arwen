"""Compare captured coordinate-theta operands using the production kernel.

Reference data is used only in this isolated diagnostic, never in a forecast.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coupled_dycore_receipt import read_wrf
from gpuwm.core.gf import _IN_SCA


def measure(a, b):
    bad = a.view(np.uint32) != b.view(np.uint32)
    where = np.argwhere(bad)
    return dict(words=a.size, unequal_words=int(bad.sum()),
                max_abs=float(np.max(np.abs(a.astype(np.float64)-b))),
                first=where[0].tolist() if len(where) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("woof", type=Path)
    p.add_argument("wrf", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--host-library", type=Path)
    p.add_argument("--gf-launch", type=Path)
    a = p.parse_args()
    if a.host_library:
        import ctypes
        from types import SimpleNamespace
        cp = SimpleNamespace(asarray=np.ascontiguousarray, asnumpy=np.asarray,
                             zeros_like=np.zeros_like)
        fn = ctypes.CDLL(str(a.host_library.resolve())).coordinate_host
        pointer = ctypes.POINTER(ctypes.c_float)
        fn.argtypes = [pointer]*9 + [ctypes.c_float]*2 + [pointer] + [ctypes.c_int]*3
        fn.restype = None
        def launch_coordinate_horizontal(field, kh, mu, c1, c2, mt, mfu,
                                         mfv, dx, dy, out, **kwargs):
            arrays = (field, kh, mu, c1, c2, kwargs["theta_initial"], mt, mfu, mfv)
            for value in (*arrays, out):
                assert value.dtype == np.float32 and value.flags.c_contiguous
            nz, ny, nx = field.shape
            fn(*(v.ctypes.data_as(pointer) for v in arrays),
               np.float32(1/dx), np.float32(1/dy), out.ctypes.data_as(pointer),
               nz, ny, nx)
    else:
        import cupy as cp
        from gpuwm.core.dycore import launch_coordinate_horizontal
    def load(tag):
        paths = list(a.woof.glob(f"*_s{a.step:03d}_*_{tag}.npz"))
        if len(paths) != 1:
            raise ValueError((tag, len(paths)))
        return np.load(paths[0])
    with load("c00_deform") as z:
        kh = cp.asarray(z["x_xkhh"])
        fields = {n: cp.asarray(z[n]) for n in
                  ("thp", "thb", "mub2d", "mup", "c1h", "c2h",
                   "msft", "msfu", "msfv")}
    with load("c_phys") as z:
        physics = cp.asarray(z["x_rtheta"])
    with load("c_fixed") as z:
        original = z["x_smag_rth"].copy()
    shape = tuple(fields["thp"].shape)
    initial = cp.asarray(read_wrf(a.wrf, 1, 0, "c0_reference",
                                 "t_init", shape, step=a.step))
    expected = read_wrf(a.wrf, 1, 0, "rk_add_inputs", "t_tendf",
                        shape, step=a.step)
    launches = [a.gf_launch] if a.gf_launch else list(a.woof.glob("gf_launch_*.npz"))
    if len(launches) != 1:
        raise ValueError("select one captured GF launch with --gf-launch")
    with np.load(launches[0]) as z:
        dx = float(z["scin"][0, _IN_SCA.index("dx")])
    base = fields["thb"]
    if base.ndim == 1:
        base = base[:, None, None]
    rounded = (fields["thp"] + base) - np.float32(300)
    result = {"roundtrip_operands": measure(cp.asnumpy(fields["thp"]),
                                            cp.asnumpy(rounded))}
    for name, theta in (("roundtrip", rounded), ("direct", fields["thp"])):
        for seeded in (False, True):
            out = physics.copy() if seeded else cp.zeros_like(physics)
            launch_coordinate_horizontal(theta, kh,
                fields["mub2d"]+fields["mup"], fields["c1h"], fields["c2h"],
                fields["msft"], fields["msfu"], fields["msfv"], dx, dx, out,
                boundary_x=True, boundary_y=True, theta_initial=initial)
            if not seeded:
                out += physics
            value = cp.asnumpy(out)
            key = name + ("_seeded" if seeded else "_separate")
            result[key] = measure(expected, value)
            if name == "roundtrip" and not seeded:
                result["reproduced_original"] = measure(original, value)
    a.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
