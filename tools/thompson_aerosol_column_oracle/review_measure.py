#!/usr/bin/env python3
"""Measure default mp=28 fallout and full adapter on seeded oracle columns.

Run through the GPU queue. Timing excludes compilation, setup and transfers.
Every repeated launch starts from identical inputs. Hashes cover whole outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from gpu_run import ColumnState
import oracle_io as io


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--serial-fallout", action="store_true")
    ap.add_argument("--no-pass-timers", action="store_true",
                    help="time the whole adapter without per-launch synchronization")
    a = ap.parse_args()
    import cupy as cp
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol
    if a.serial_fallout:
        from gpuwm.core import thompson_aerosol_sed as sed
        original_kernel = sed.aerosol_kernel
        def serial_kernel(module, name):
            if "_levels_" not in name:
                return original_kernel(module, name)
            kernel = original_kernel(module, name.replace("_levels", ""))
            def launch(grid, block, args):
                columns = int(args[-2]) * int(args[-1])
                kernel(((columns + 31)//32,), (32,), args)
            return launch
        sed.aerosol_kernel = serial_kernel
    from types import SimpleNamespace
    gate = np.load(HERE.parents[1] / "tests/data/mp28_column_oracle_wrf461.npz")
    rows = {}
    for case in ("mixed", "clear", "rain_only"):
        cols = {k: np.tile(gate[k], (a.tiles,) + (1,) * (gate[k].ndim - 1))
                for k in io.SPECIES + ("p", "th", "geop", "w", "nwfa2d", "nifa2d")}
        if case != "mixed":
            for k in ("qc", "qi", "qs", "qg", "ni", "nc"):
                cols[k] = np.zeros_like(cols[k])
            if case == "clear":
                for k in ("qr", "nr"):
                    cols[k] = np.zeros_like(cols[k])
            cols["qv"] *= np.float32(0.001)
        state = ColumnState(cols, cp)
        originals = {k: getattr(state, k).copy() for k in io.SPECIES + ("thp",)}
        cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.,
                              thompson_version="wrf_461", thompson_fork_snow_fall="blend")
        timings = []
        passes = []
        from gpuwm.core import thompson_aerosol_sed as sed
        launchers = {}
        for kind in ("snow", "graupel"):
            attr = f"launch_aa_{kind}_sedimentation"
            orig = getattr(sed, attr)
            launchers[attr] = orig
            def timed(*args, _orig=orig, _kind=kind, **kwargs):
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                result = _orig(*args, **kwargs)
                end.record(); end.synchronize()
                passes.append((_kind, float(cp.cuda.get_elapsed_time(start, end))))
                return result
            if not a.no_pass_timers:
                setattr(sed, attr, timed)
        try:
            for iteration in range(12):
                state.physics.refl_10cm = None
                for k, v in originals.items():
                    getattr(state, k)[...] = v
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                diag = _apply_thompson_aerosol(state, cfg, 20., refl_10cm_due=True)
                end.record(); end.synchronize()
                if iteration >= 3:
                    timings.append(float(cp.cuda.get_elapsed_time(start, end)))
        finally:
            for attr, orig in launchers.items():
                setattr(sed, attr, orig)
        hashes = {k: hashlib.sha256(cp.asnumpy(getattr(state, k)).tobytes()).hexdigest()
                  for k in io.SPECIES + ("thp", "effc", "effi", "effs")}
        hashes["refl"] = hashlib.sha256(cp.asnumpy(state.physics.refl_10cm).tobytes()).hexdigest()
        for k in io.OUT2:
            hashes[k] = hashlib.sha256(cp.asnumpy(getattr(diag, k)).tobytes()).hexdigest()
        rows[case] = {"columns": len(cols["p"]), "levels": cols["p"].shape[1],
                      "adapter_median_ms": float(np.median(timings)),
                      "adapter_samples_ms": timings,
                      "fallout_median_ms": {k: float(np.median([v for key, v in passes[6:] if key == k]))
                                            for k in ("snow", "graupel")} if passes else {},
                      "hashes": hashes}
    receipt = {"rows": rows, "serial_fallout": a.serial_fallout,
               "pass_timers": not a.no_pass_timers,
               "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "cupy": cp.__version__, "commit": __import__("subprocess").check_output(
                   ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()}
    Path(a.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
