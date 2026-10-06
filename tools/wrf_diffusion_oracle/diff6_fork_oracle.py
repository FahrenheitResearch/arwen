"""Compiled oracle: the NOAA-EMC WRFV3.9 fork's sixth_order_diffusion.

Builds the fork's routine byte for byte (NOAA-EMC/HRRR tag v4.1.21,
sorc/hrrr_wrfarw.fd/WRFV3.9/dyn_em/module_big_step_utilities_em.F) with the
file's own hybrid-coordinate macro block, preprocessed as the operational
build does (-DHYBRID_COORD=1, configure.wrf.useme:236), and compiled with
gfortran -O0 -ffp-contract=off, 4-byte REAL.  The existing stream driver
(diff6_driver.f90) calls it with the 4.x argument list, which the fork's
routine shares; the fork reads only config_flags%diff_6th_slopeopt and
config_flags%diff_6th_thresh.

Cases: the fork loops to the domain edge and reads the halo.  Every input is
given set_physical_bc3d's specified/nested halo (share/module_bc.F open_xs/
open_xe and the y analogues: zero-gradient copies of the edge datum), the
field, the column mass, the base geopotential and the map factors alike.
The GPU side gets the storage arrays only and runs the port's production
launcher (gpuwm.core.dycore.launch_diff6_to_edge).  Every tendency word of
the storage region is compared as uint32.

    python diff6_fork_oracle.py build FORK_FILE BUILD_DIR
    python diff6_fork_oracle.py compare BUILD_DIR RESULT_JSON   (GPU)

Analysis and verification tooling only; nothing here is on a data path.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

FORK_TAG = "v4.1.21"
FORK_PATH = "sorc/hrrr_wrfarw.fd/WRFV3.9/dyn_em/module_big_step_utilities_em.F"
HERE = Path(__file__).resolve().parent


def build(fork_file: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw = fork_file.read_bytes()
    macro = re.match(rb"(#if \( HYBRID_COORD==1 \).*?\n#endif\n)", raw, re.S)
    assert macro, "the fork file's hybrid macro block"
    routine = re.search(rb"(?im)^ *SUBROUTINE sixth_order_diffusion\(.*?^ *END SUBROUTINE sixth_order_diffusion[^\n]*\n",
                        raw, re.S)
    assert routine
    config = b"""module module_configure
  implicit none
  type grid_config_rec_type
    logical :: specified=.false., nested=.false.
    logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
    integer :: diff_6th_slopeopt=0
    real :: diff_6th_thresh=0.1
  end type
end module
module module_big_step_utilities_em
  use module_configure
  implicit none
contains
"""
    src = out / "fork_sixth_order.F"
    src.write_bytes(macro[1] + config + routine[0] + b"end module\n")
    pre = out / "fork_sixth_order.f90"
    subprocess.run(["cpp", "-P", "-DHYBRID_COORD=1", str(src), str(pre)], check=True)
    text = pre.read_text()
    assert "(c1(k)*MUT(i-1,j)+c2(k))" in text.replace(" ", ""), "MUT macro expansion"
    flags = ["-O0", "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds",
             "-ffree-form", "-ffree-line-length-none"]
    subprocess.run(["gfortran", *flags, pre.name, str(HERE / "diff6_driver.f90"), "-o", "diff6_fork_driver"],
                   cwd=out, check=True)
    meta = {"fork_tag": FORK_TAG, "fork_path": FORK_PATH,
            "fork_file_sha256": hashlib.sha256(raw).hexdigest(),
            "routine_slice_sha256": hashlib.sha256(routine[0]).hexdigest(),
            "routine_source_lines": [raw[:routine.start()].count(b"\n") + 1, raw[:routine.end()].count(b"\n")],
            "macro_block_sha256": hashlib.sha256(macro[1]).hexdigest(),
            "cpp": "cpp -P -DHYBRID_COORD=1",
            "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
            "flags": flags, "real_kind_bytes": 4}
    (out / "diff6-fork-build.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _pad(a, lo_y, hi_y, lo_x, hi_x):
    pad = [(0, 0)] * (a.ndim - 2) + [(lo_y, hi_y), (lo_x, hi_x)]
    return np.pad(a, pad, mode="edge")


def _case(seed, stagger, nx=24, ny=20, nz=6, slope=1, opt=2):
    rng = np.random.default_rng(seed)
    nlev = nz + 1 if stagger == "z" else nz
    shape = {"": (nlev, ny, nx), "z": (nlev, ny, nx), "x": (nlev, ny, nx + 1), "y": (nlev, ny + 1, nx)}[stagger]
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    base = 1.0e-2 * np.exp(-((xx - shape[2] * rng.random()) ** 2 + (yy - shape[1] * rng.random()) ** 2) / 30.0)
    f = (base[None] * (1 + 0.3 * rng.standard_normal(shape)) + 2e-3 * rng.random(shape)).astype(np.float32)
    f[:, :, :2] += np.float32(4e-3)          # edge structure the halo read sees
    mut = (8.0e4 + 1.5e4 * rng.random((ny, nx))).astype(np.float32)
    c1 = np.linspace(1.0, 0.0, nlev).astype(np.float32)
    c2 = np.linspace(0.0, 1.2e4, nlev).astype(np.float32)
    ridge = np.exp(-((np.arange(nx)[None, :] - nx / 2) ** 2) / 8.0) * np.ones((ny, 1))
    phb = (9.81 * (300.0 * ridge[None] + 400.0 * np.arange(nz + 1)[:, None, None]
                   + 50.0 * rng.random((nz + 1, ny, nx)))).astype(np.float32)
    msfu = (1.0 + 0.03 * rng.random((ny, nx + 1))).astype(np.float32)
    msfv = (1.0 + 0.03 * rng.random((ny + 1, nx))).astype(np.float32)
    msft = (1.0 + 0.03 * rng.random((ny, nx))).astype(np.float32)
    return dict(f=f, mut=mut, c1=c1, c2=c2, phb=phb, msfu=msfu, msfv=msfv, msft=msft,
                nx=nx, ny=ny, nz=nz, slope=slope, opt=opt, stagger=stagger)


def _fortran_input(c, factor, dt, dx, dy, thresh, path):
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    st = c["stagger"]
    # Memory -3..n+3 on each axis; a staggered axis stores n+1 points.
    hx = 3 if st == "x" else 4
    hy = 3 if st == "y" else 4
    nlev = c["f"].shape[0]
    field = np.zeros((nz + 1, ny + 7, nx + 7), np.float32)
    field[:nlev] = _pad(c["f"], 3, hy, 3, hx)
    phb = _pad(c["phb"], 3, 4, 3, 4)
    mut = _pad(c["mut"], 3, 4, 3, 4)
    mtx = _pad(c["msft"], 3, 4, 3, 4)
    mux = _pad(c["msfu"], 3, 4, 3, 3)
    mvx = _pad(c["msfv"], 3, 3, 3, 4)
    c1 = np.zeros(nz + 1, np.float32); c1[:nlev] = c["c1"]
    c2 = np.zeros(nz + 1, np.float32); c2[:nlev] = c["c2"]
    name = {"": "m", "x": "u", "y": "v", "z": "w"}[st]
    rdx, rdy = np.float32(1.0 / dx), np.float32(1.0 / dy)
    tend = np.zeros_like(field)
    # Fortran (i, k, j) column-major == C (j, k, i).
    fk = lambda a: np.ascontiguousarray(np.transpose(a, (1, 0, 2)))
    with open(path, "wb") as fh:
        np.array([nx, ny, nz, c["opt"], c["slope"], 1], np.int32).tofile(fh)
        fh.write(name.encode())
        np.array([dt, factor, rdx, rdy, thresh], np.float32).tofile(fh)
        for a in (fk(field), fk(tend), mut, c1, c2, fk(phb), mtx, mtx, mux, mux, mvx, mvx):
            np.asarray(a, np.float32).tofile(fh)
    return name


def compare(build_dir: Path, result: Path) -> dict:
    import cupy as cp
    from gpuwm.core.dycore import launch_diff6_to_edge
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    rows = []
    for seed in range(4):
        for st in ("", "x", "y", "z"):
            for slope, opt in ((1, 2), (0, 2), (1, 1)):
                c = _case(100 * seed + len(st) + 7 * slope + opt, st, slope=slope, opt=opt)
                factor, dt, dx, dy, thresh = 0.04, 20.0, 3000.0, 3000.0, 0.05
                inp = build_dir / "case.bin"
                _fortran_input(c, factor, dt, dx, dy, thresh, inp)
                outp = build_dir / "case.out"
                subprocess.run([str(build_dir / "diff6_fork_driver"), str(inp), str(outp)], check=True)
                nx, ny, nz = c["nx"], c["ny"], c["nz"]
                want = np.fromfile(outp, np.float32).reshape(ny + 7, nz + 1, nx + 7).transpose(1, 0, 2)
                nlev, nys, nxs = c["f"].shape
                want = np.ascontiguousarray(want[:nlev, 3:3 + nys, 3:3 + nxs])
                tend = cp.zeros(c["f"].shape, cp.float32)
                launch_diff6_to_edge(cp.asarray(c["f"]), tend, cp.asarray(c["mut"]), cp.asarray(c["c1"]),
                                     cp.asarray(c["c2"]), factor, dt, opt, stagger=st,
                                     phb=cp.asarray(c["phb"]), msfu=cp.asarray(c["msfu"]),
                                     msfv=cp.asarray(c["msfv"]), msft=cp.asarray(c["msft"]),
                                     slopeopt=slope, thresh=thresh, dx=dx, dy=dy)
                got = tend.get()
                diff = got.view(np.uint32) != want.view(np.uint32)
                rows.append({"seed": seed, "stagger": st or "mass", "slopeopt": slope, "opt": opt,
                             "words": int(got.size), "different_words": int(diff.sum()),
                             "max_ulp": int(fp32_ulp_distance(got, want).max(initial=0)),
                             "edge_rows_nonzero": bool(np.abs(want[:, 1:3, :]).max() > 0)})
    summary = {"cases": len(rows), "words": sum(r["words"] for r in rows),
               "different_words": sum(r["different_words"] for r in rows),
               "max_ulp": max(r["max_ulp"] for r in rows),
               "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "build": json.loads((build_dir / "diff6-fork-build.json").read_text()), "rows": rows}
    result.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"cases={summary['cases']} words={summary['words']} different={summary['different_words']} "
          f"max_ulp={summary['max_ulp']}")
    return summary


if __name__ == "__main__":
    if sys.argv[1] == "build":
        print(json.dumps(build(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
