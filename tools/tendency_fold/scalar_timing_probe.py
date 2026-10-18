"""Measure coordinate scalar mixing at the two captured mass boundaries.

This is an isolated kernel diagnostic. Reference arrays are used for comparison
only; they never supply a forecast tendency or coefficient.
"""
import argparse
import json
from pathlib import Path
import sys

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coupled_dycore_receipt import read_wrf
from gpuwm.core.dycore import launch_coordinate_horizontal
from gpuwm.core.gf import _IN_SCA


def measure(a, b):
    bad = a.view(np.uint32) != b.view(np.uint32)
    indices = np.argwhere(bad)
    return dict(words=a.size, unequal_words=int(bad.sum()),
                max_abs=float(np.max(np.abs(a.astype(np.float64)-b))),
                first=indices[0].tolist() if len(indices) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("woof", type=Path)
    p.add_argument("wrf", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    def load(tag):
        paths = list(a.woof.glob(f"*_s001_*_{tag}.npz"))
        if len(paths) != 1:
            raise ValueError((tag, len(paths)))
        return np.load(paths[0])
    with load("c00_deform") as z:
        kh = cp.asarray(z["x_xkhh"])
        fields = {n:cp.asarray(z[n]) for n in
                  ("qv", "c1h", "c2h", "msft", "msfu", "msfv")}
    with load("c_phys") as z:
        physics = cp.asarray(z["x_rqv"])
    with np.load(next(a.woof.glob("*_s001_rk1_it01_g_ss.npz"))) as z:
        mass = {"before": cp.asarray(z["mub2d"]+z["mup"]),
                "after": cp.asarray((z["mub2d"]+z["mup"])+z["mu_pp"])}
    with np.load(a.woof/"gf_launch_0000.npz") as z:
        dx = float(z["scin"][0,_IN_SCA.index("dx")])
    shape = tuple(fields["qv"].shape)
    expected = read_wrf(a.wrf,1,0,"q1_sctend","moist_tend",shape)
    result = {}
    for name, mut in mass.items():
        out = physics.copy()
        launch_coordinate_horizontal(fields["qv"],kh,mut,
            fields["c1h"],fields["c2h"],fields["msft"],fields["msfu"],
            fields["msfv"],dx,dx,out,boundary_x=True,boundary_y=True)
        result[name] = measure(expected,cp.asnumpy(out))
    expected_mu = read_wrf(a.wrf,1,1,"g_ss","muts",shape[1:])
    result["after_mass"] = measure(expected_mu,cp.asnumpy(mass["after"]))
    result["dx"] = dx
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
