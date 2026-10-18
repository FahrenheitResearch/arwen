"""Wall time of one fork Thompson call (mp_physics = 28, thompson_version =
"wrf_39_noaa") on a large grid, for comparing two trees' implementations.

usage (on a CUDA host, through the GPU mutex):
    PYTHONPATH=<tree> python time_fork_column.py FIXTURE.npz --nx 1024 \
        --ny 1024 [--calls 5]

The grid tiles the fixture's columns (tests/data/
thompson_fork_column_oracle.npz inputs) to ny x nx.  Each call starts from
a fresh copy of the tiled state; the first call (compile and table upload)
is reported separately and the rest are timed with CUDA events.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
           "nifa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("--nx", type=int, default=1024)
    ap.add_argument("--ny", type=int, default=1024)
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--dt", type=float, default=20.0)
    args = ap.parse_args()

    import cupy as cp
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    z = np.load(args.fixture)
    cols = {k[3:]: z[k] for k in z.files if k.startswith("in_")}
    ncol_src = cols["p"].shape[0]
    n = args.nx * args.ny
    pick = np.arange(n) % ncol_src

    def vol(a):
        a = np.ascontiguousarray(np.asarray(a, np.float32)[pick].T)
        return cp.asarray(a.reshape(a.shape[0], args.ny, args.nx))

    base = {name: vol(cols[name]) for name in ("p", "th", "geop", "w")
            + SPECIES}
    base["nwfa2d"] = cp.asarray(np.asarray(cols["nwfa2d"], np.float32)[pick]
                                .reshape(args.ny, args.nx))
    base["nifa2d"] = cp.asarray(np.asarray(cols["nifa2d"], np.float32)[pick]
                                .reshape(args.ny, args.nx))
    nz = base["p"].shape[0]

    class State:
        pass

    def fresh():
        s = State()
        s.p = base["p"]
        s.thb = cp.zeros((nz,), cp.float32)
        s.thp = base["th"].copy()
        s.phb = cp.zeros((nz + 1,), cp.float32)
        s.php = base["geop"]
        s.w = base["w"]
        for name in SPECIES:
            setattr(s, name, base[name].copy())
        s.nwfa2d = base["nwfa2d"]
        s.nifa2d = base["nifa2d"]
        for name in ("effc", "effi", "effs", "h_diabatic"):
            setattr(s, name, cp.zeros((nz, args.ny, args.nx), cp.float32))
        s.physics = SimpleNamespace(refl_10cm=None, state=s)
        s._scratch = {}

        def scratch(shape, slot, dtype=None, _s=s):
            value = _s._scratch.get(slot)
            if value is None:
                value = cp.zeros(tuple(shape), dtype=np.float32
                                 if dtype is None else dtype)
                _s._scratch[slot] = value
            return value

        s.scratch = scratch
        s.existing_scratch = lambda slot, _s=s: _s._scratch.get(slot)
        return s

    cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0,
                          thompson_version="wrf_39_noaa",
                          thompson_fork_snow_fall="wrf_39_noaa")
    t0 = time.perf_counter()
    _apply_thompson_aerosol(fresh(), cfg, args.dt, refl_10cm_due=True)
    cp.cuda.Device().synchronize()
    first = time.perf_counter() - t0
    times = []
    for _ in range(args.calls):
        s = fresh()
        cp.cuda.Device().synchronize()
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        _apply_thompson_aerosol(s, cfg, args.dt, refl_10cm_due=True)
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(json.dumps({"grid": [nz, args.ny, args.nx], "columns": n,
                      "first_call_s": round(first, 2),
                      "call_ms": [round(t, 2) for t in times],
                      "median_ms": round(float(np.median(times)), 2),
                      "gpu": dev["name"].decode()
                      if isinstance(dev["name"], bytes) else dev["name"]}))


if __name__ == "__main__":
    main()
