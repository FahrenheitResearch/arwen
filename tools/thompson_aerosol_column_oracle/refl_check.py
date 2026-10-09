"""mp=28's 10 cm reflectivity against WRF v4.6.1's calc_refl10cm, bitwise.

The column oracle grades REFL_10CM after a whole microphysics call, where a
state difference anywhere upstream shows in the echo.  This check feeds
WRF's own routine and WOOF's ``launch_aa_refl10cm`` the SAME state -- the
oracle's WRF end states, each column again 2 K and 5 K warmer so more
levels melt -- and compares every word.

usage (on the oracle host, GPU work through the box's mutex):
  python refl_check.py build WRF_PHYS_DIR STUB_F90 BUILD_DIR
  python refl_check.py inputs RUN_DIR BUILD_DIR      # columns + WRF ends
  python refl_check.py wrf BUILD_DIR TABLE_RUN_DIR   # calc_refl10cm
  python refl_check.py gpu BUILD_DIR                 # WOOF, then compare
  python refl_check.py fixture BUILD_DIR OUT.npz [NCOL]
  python refl_check.py constants BUILD_DIR TABLE_RUN_DIR   # init state

``build`` compiles a copy of the pristine module_mp_thompson.F whose
PRIVATE attributes are dropped (calc_refl10cm has no PUBLIC statement)
with the oracle's flags, and refl_driver.F90 against it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = ["-O2", "-fno-tree-vectorize", "-ffree-form",
         "-ffree-line-length-none"]
FIELDS = ("qv", "qc", "qr", "nr", "qs", "qg", "ng", "t", "p")


def build(phys, stub, out):
    os.makedirs(out, exist_ok=True)
    src = open(os.path.join(phys, "module_mp_thompson.F")).read()
    src = re.sub(r",\s*PRIVATE", "", src, flags=re.IGNORECASE)
    with open(os.path.join(out, "thompson_public.F"), "w") as f:
        f.write(src)
    run = lambda *a: subprocess.run(list(a), cwd=out, check=True)
    run("gfortran", "-c", *FLAGS, stub)
    run("gfortran", "-c", *FLAGS, os.path.join(phys, "module_mp_radar.F"),
        "-o", "module_mp_radar.o")
    run("gfortran", "-c", *FLAGS, "-cpp", "-DWRF_CHEM=0", "thompson_public.F",
        "-o", "module_mp_thompson.o")
    run("gfortran", "-c", *FLAGS, os.path.join(HERE, "refl_driver.F90"))
    run("gfortran", "-O2", "-o", "refl_driver", "stub_wrf.o",
        "module_mp_radar.o", "module_mp_thompson.o", "refl_driver.o")
    run("gfortran", "-c", *FLAGS, os.path.join(HERE, "constdump.F90"))
    run("gfortran", "-O2", "-o", "constdump", "stub_wrf.o",
        "module_mp_radar.o", "module_mp_thompson.o", "constdump.o")


def constants(build_dir, table_dir):
    env = dict(os.environ, GFORTRAN_CONVERT_UNIT="big_endian:20")
    out = subprocess.run([os.path.join(build_dir, "constdump")], cwd=table_dir,
                         env=env, check=True, capture_output=True, text=True)
    sys.stdout.write("".join(line + "\n" for line in out.stdout.splitlines()
                             if not line.startswith("ThompMP")))


def inputs(run_dir, out):
    cols = np.load(os.path.join(run_dir, "columns.npz"))
    wrf = np.load(os.path.join(run_dir, "summary-strict-dt20.wrf.npz"))
    gpu = np.load(os.path.join(run_dir, "gpu-strict-dt20.npz"))
    t = (wrf["th"] * gpu["in_pii"]).astype(np.float32)
    ncol, nz = t.shape
    k = np.arange(nz)[None, :]
    c = np.arange(ncol)[:, None]
    # Any graupel number will do, provided both codes see the same one;
    # this spreads the median volume diameter over two decades.
    spread = np.float32(3.5) + np.float32(2.0) * (
        ((k * 7 + c * 3) % 10).astype(np.float32) / np.float32(10.0))
    ng = (wrf["qg"] * np.power(np.float32(10.0), spread)).astype(np.float32)
    base = {"qv": wrf["qv"], "qc": wrf["qc"], "qr": wrf["qr"],
            "nr": wrf["nr"], "qs": wrf["qs"], "qg": wrf["qg"], "ng": ng,
            "t": t, "p": cols["p"]}
    stacked = {}
    for name in FIELDS:
        parts = []
        for warm in (0.0, 2.0, 5.0):
            v = base[name]
            if name == "t":
                v = (v + np.float32(warm)).astype(np.float32)
            parts.append(np.asarray(v, dtype=np.float32))
        stacked[name] = np.ascontiguousarray(np.concatenate(parts, axis=0))
    np.savez(os.path.join(out, "refl_inputs.npz"), **stacked)
    n = stacked["t"].shape[0]
    with open(os.path.join(out, "refl_inputs.bin"), "wb") as f:
        np.array([n, nz], dtype="<i4").tofile(f)
        for name in FIELDS:
            # column index fastest: Fortran (ncol, nz) order.
            stacked[name].T.astype("<f4").tofile(f)
    print(f"{n} columns x {nz} levels")


def wrf(build_dir, table_dir):
    env = dict(os.environ, GFORTRAN_CONVERT_UNIT="big_endian:20")
    subprocess.run([os.path.join(build_dir, "refl_driver"),
                    os.path.join(build_dir, "refl_inputs.bin"),
                    os.path.join(build_dir, "refl_wrf.bin")],
                   cwd=table_dir, env=env, check=True)


def _wrf_answer(build_dir, ncol, nz):
    raw = np.fromfile(os.path.join(build_dir, "refl_wrf.bin"), dtype="<f4")
    return raw.reshape(nz, ncol).T.copy()


def _melting_cells(x):
    t, qr, qs, qg = x["t"], x["qr"], x["qs"], x["qg"]
    ncol, nz = t.shape
    melt = np.zeros((ncol, nz), dtype=bool)
    for c in range(ncol):
        k0 = None
        for k in range(nz - 2, -1, -1):
            if (t[c, k] > np.float32(273.15) and qr[c, k] > np.float32(1e-12)
                    and (qs[c, k + 1] > np.float32(1e-6)
                         or qg[c, k + 1] > np.float32(1e-6))):
                k0 = k + 1
                break
        if k0 is None or k0 < 1 or not qs[c, k0] > np.float32(1e-6):
            continue
        for k in range(k0):
            melt[c, k] = qs[c, k] > np.float32(1e-6)
    return melt


def gpu(build_dir):
    import cupy as cp
    from gpuwm.core.thompson_aerosol_state import launch_aa_refl10cm

    x = dict(np.load(os.path.join(build_dir, "refl_inputs.npz")))
    ncol, nz = x["t"].shape
    dev = {k: cp.asarray(v.T.reshape(nz, 1, ncol).copy())
           for k, v in x.items()}
    refl = cp.zeros((nz, 1, ncol), dtype=cp.float32)
    launch_aa_refl10cm(dev["qv"], dev["qr"], dev["nr"], dev["qs"], dev["qg"],
                       dev["ng"], dev["t"], dev["p"], refl)
    woof = cp.asnumpy(refl).reshape(nz, ncol).T
    ref = _wrf_answer(build_dir, ncol, nz)
    differ = woof.view(np.int32) != ref.view(np.int32)
    melt = _melting_cells(x)
    ulp = np.abs(woof.view(np.int32).astype(np.int64)
                 - ref.view(np.int32).astype(np.int64))
    print(f"refl: {differ.sum()} of {differ.size} cells differ "
          f"(max {ulp.max()} ulp); melting-snow cells {melt.sum()}, "
          f"of which {np.logical_and(differ, melt).sum()} differ")
    np.save(os.path.join(build_dir, "refl_woof.npy"), woof)
    return int(differ.sum())


def fixture(build_dir, out, ncol_keep=60):
    """A small committed fixture of ``ncol_keep`` columns: half with a
    melting-snow cell, half without, each half evenly spaced through the
    columns that qualify."""
    x = dict(np.load(os.path.join(build_dir, "refl_inputs.npz")))
    ncol, nz = x["t"].shape
    ref = _wrf_answer(build_dir, ncol, nz)
    melt = _melting_cells(x).any(axis=1)
    keep = []
    for group, count in ((np.flatnonzero(melt), int(ncol_keep) // 2),
                         (np.flatnonzero(~melt),
                          int(ncol_keep) - int(ncol_keep) // 2)):
        count = min(count, group.size)
        picks = np.linspace(0, group.size - 1, count).round().astype(int)
        keep.extend(group[np.unique(picks)])
    keep = np.array(sorted(keep))
    data = {k: v[keep] for k, v in x.items()}
    data["refl_wrf"] = ref[keep]
    np.savez_compressed(out, **data)
    print(f"{len(keep)} columns ({int(melt[keep].sum())} with melting snow)"
          f" -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(*sys.argv[2:5])
    elif cmd == "inputs":
        inputs(*sys.argv[2:4])
    elif cmd == "wrf":
        wrf(*sys.argv[2:4])
    elif cmd == "gpu":
        sys.exit(1 if gpu(sys.argv[2]) else 0)
    elif cmd == "constants":
        constants(*sys.argv[2:4])
    elif cmd == "fixture":
        fixture(sys.argv[2], sys.argv[3],
                *(int(a) for a in sys.argv[4:5]))
    else:
        raise SystemExit(__doc__)
