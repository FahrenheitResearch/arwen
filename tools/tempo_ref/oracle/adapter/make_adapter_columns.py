"""Reuse the 127 original plus 73 refute columns, with float32 host state.

The binary uses the existing 17-field format, t now holding theta perturbation
and qb holding m3/kg, followed per set by geopotential faces and nwfa2d.
These are test inputs, not a data decoder or model input preparation path.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_columns as mc
from refute import make_refute_columns as rc


def build(out):
    original = mc.build()
    for s in original:
        s["nz"] = mc.NZ
    sets = original + rc.build()
    catalog = []
    out.mkdir(parents=True, exist_ok=True)
    with (out / "adapter_in.bin").open("wb") as f:
        np.array([len(sets)], dtype="<i8").tofile(f)
        for si, s in enumerate(sets, 1):
            ncol, nz = len(s["cols"]), s["nz"]
            np.array([ncol, nz, s["nsteps"]], dtype="<i8").tofile(f)
            np.array([s["dt"]], dtype="<f8").tofile(f)
            a = {n: np.stack([c[n] for c in s["cols"]]).astype(np.float32)
                 for n in mc.FIELDS}
            # Initial theta is input metadata. Both callers receive these exact
            # words; each independently computes the Exner used by the call.
            pi = ((a["p"] / np.float32(1e5)).astype(np.float64)
                  ** float(np.float32(2.0 / 7.0))).astype(np.float32)
            a["t"] = a["t"] / pi - np.float32(300.0)
            a["qb"] = (np.stack([c["qb"] for c in s["cols"]]) * 1e-3).astype(np.float32)
            for n in mc.FIELDS:
                a[n].astype("<f8").tofile(f)
            phi = (np.concatenate([np.zeros((ncol, 1)),
                    np.cumsum(a["dz"].astype(np.float64), axis=1)], axis=1)
                   * np.float32(9.81)).astype(np.float32)
            phi.astype("<f8").tofile(f)
            # Nonzero and non-integral emissions exercise placement and rounding.
            emission = (np.arange(ncol, dtype=np.float64) + 0.37) * 123456.789
            emission.astype(np.float32).astype("<f8").tofile(f)
            for ci, (regime, variant) in enumerate(s["cat"], 1):
                catalog.append(dict(set=si, column=ci, regime=regime, variant=variant,
                                    dt=s["dt"], nsteps=s["nsteps"], nz=nz))
    (out / "adapter_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    assert len(catalog) == 200


if __name__ == "__main__":
    build(Path(sys.argv[1]))
