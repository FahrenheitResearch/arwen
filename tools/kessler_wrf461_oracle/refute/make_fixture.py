#!/usr/bin/env python3
"""Pack the refute sets into committed fixtures (a call subset of A and B).

    python make_fixture.py SETS_ROOT FIXTURE_ROOT
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

from gpuwm.verify import kessler_oracle as ko

# set -> (cases kept by index, WRF reference files to pack)
KEEP = {"setA": ((1, 4), ("kessler-wrf-O0.bin",)),
        "setB": ((2,), ("kessler-wrf-O0.bin",)),
        "setC": (None, ("kessler-wrf-O0.bin",)),
        "setD": (None, ("kessler-wrf-O0.bin",)),
        "setF": (None, ("kessler-wrf-O0.bin",))}


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    lines = []
    for s, (keep, refs) in KEEP.items():
        d = root / s
        inp = ko.read_inputs_bin(d / "kessler-inputs.bin")
        with np.load(d / "kessler-inputs.npz") as f:
            regime = f["regime"]
        nz, ncol = inp["t"].shape
        ns = inp["nsteps"]
        starts = np.concatenate([[0], np.cumsum(ns)[:-1]])
        idx = range(len(ns)) if keep is None else keep
        sel = np.concatenate([np.arange(starts[i], starts[i] + ns[i])
                              for i in idx])
        dst = out / f"refute-{s}"
        dst.mkdir(parents=True, exist_ok=True)
        fx = dict(inp)
        fx["dt"] = inp["dt"][list(idx)]
        fx["nsteps"] = ns[list(idx)]
        fx["regime"] = regime
        np.savez_compressed(dst / "kessler-inputs.npz", **fx)
        wrf = ko.read_outputs_bin(d / "kessler-wrf-O0.bin", nz, ncol,
                                  int(ns.sum()))
        np.savez_compressed(dst / "kessler-wrf461.npz",
                            **{k: v[sel] for k, v in wrf.items()})
        if s == "setF":
            g = ko.read_outputs_bin(root / "setF_nofma" / "kessler-wrf-O0.bin",
                                    nz, ncol, int(ns.sum()))
            np.savez_compressed(dst / "kessler-wrf461-generic-libm.npz",
                                **{k: v[sel] for k, v in g.items()})
        for p in sorted(dst.glob("*.npz")):
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  "
                         f"{p.relative_to(out)}  {p.stat().st_size}")
        print(s, ncol, "columns x", nz, "levels,", len(sel), "calls")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
