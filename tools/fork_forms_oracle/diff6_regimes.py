"""diff_6th_form = noaa_wrf39 + diff_6th_factor2 on regime columns.

Two checks, both against the operational RAP/HRRR branch (NOAA-EMC/HRRR
v4.1.21 WRFV3.9):

1. Per-row factor and step.  The branch hands sixth_order_diffusion
   ``diff_6th_factor`` and ``grid%dt`` for u, v, w and theta (rk_tendency,
   module_em.F:884-925), and for the moist and scalar arrays
   ``diff_6th_factor2`` (solve_em.F:2418, :2680, :2974) with the full step
   rk_scalar_tend rebuilds, ``dt = dt_rk*(rk_order - rk_step + 1)``
   (module_em.F:1216-1217) where ``dt_rk = grid%dt/3.`` at rk_step 1;
   TKE takes ``diff_6th_factor`` (:2587) with that same step.  A gfortran
   program evaluates the branch's two REAL expressions for every time step
   tried; WOOF's ``gpuwm.core.dycore._diff6_factor`` and ``_diff6_dt`` for a
   fork configuration must give the same words for every row.
2. The routine on regime fields.  tools/wrf_diffusion_oracle/
   diff6_fork_oracle.py builds the branch's sixth_order_diffusion verbatim
   (hybrid macro block, -DHYBRID_COORD=1, gfortran -O0
   -ffp-contract=off); here it runs on 24 x 20 patches (480 columns each)
   of physically shaped fields per regime, with HRRR's slope taper
   (diff_6th_slopeopt = 1, diff_6th_thresh = 0.05) and the factor and step
   each row takes, against WOOF's production launcher
   ``gpuwm.core.dycore.launch_diff6_to_edge`` on the device.

    python diff6_regimes.py build FORK_FILE BUILD_DIR      (CPU)
    python diff6_regimes.py run BUILD_DIR OUT_DIR           (GPU)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "wrf_diffusion_oracle"))
F = np.float32
NX, NY, NZ = 24, 20, 12
DX = 3000.0
#: Time steps tried for the per-row step (HRRR runs 20 s; adaptive and
#: other operational steps besides).
DTS = (20.0, 18.0, 15.0, 12.0, 10.0, 7.5, 6.0, 4.0, 3.0, 25.0, 30.0, 36.0,
       60.0, 16.6666667, 13.3333333, 22.5)

STEP_PROGRAM = """program steps
implicit none
integer :: n, i, rk_order, rk_step
real :: dt(64), dt_rk, full
character(len=1024) :: a, b
call get_command_argument(1, a)
call get_command_argument(2, b)
open(11, file=trim(a), access='stream', form='unformatted', status='old')
read(11) n
read(11) dt(1:n)
close(11)
open(12, file=trim(b), access='stream', form='unformatted', status='replace')
rk_order = 3
rk_step = 1
do i = 1, n
  ! solve_em.F:596-600 (rk_step 1 of rk_ord 3), then module_em.F:1217.
  dt_rk = dt(i)/3.
  full = dt_rk * (rk_order - rk_step + 1)
  write(12) full
end do
close(12)
end program
"""

#: WOOF slot, WRF array, stagger, regime field generator key.
ROWS = (("smag_ru", "u (rk_tendency)", "x"), ("smag_rv", "v (rk_tendency)", "y"),
        ("smag_rw", "w (rk_tendency)", "z"), ("smag_rth", "theta (rk_tendency)", ""),
        ("smag_rtke", "tke_2 (rk_scalar_tend)", ""),
        ("smag_rqv", "moist qv", ""), ("smag_rqc", "moist qc", ""),
        ("smag_rqr", "moist qr", ""), ("smag_rqi", "moist qi", ""),
        ("smag_rqs", "moist qs", ""), ("smag_rqg", "moist qg", ""),
        ("smag_rni", "scalar qni", ""), ("smag_rnr", "scalar qnr", ""),
        ("smag_rnc", "scalar qnc", ""), ("smag_rnwfa", "scalar qnwfa", ""),
        ("smag_rnifa", "scalar qnifa", ""))
REGIMES = ("convective", "stable_night", "snow", "ocean_stratocu",
           "high_terrain", "edges")


def build(fork_file: Path, build_dir: Path) -> dict:
    import diff6_fork_oracle as d6
    build_dir = build_dir.resolve()
    meta = d6.build(fork_file.resolve(), build_dir)
    (build_dir / "steps.f90").write_text(STEP_PROGRAM, encoding="utf-8")
    for arm, flags in (("o0", ["-O0", "-ffp-contract=off"]),
                       ("wrf_o2", ["-O2", "-ftree-vectorize", "-funroll-loops"])):
        subprocess.run(["gfortran", *flags, "steps.f90", "-o", f"steps_{arm}"],
                       cwd=build_dir, check=True)
    return meta


def _fork_cfg(dt):
    from gpuwm.config import RunConfig
    return RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DX, ztop=20000., dt=dt,
                     run_seconds=60., diff_6th_opt=2, diff_6th_factor=0.12,
                     diff_6th_form="noaa_wrf39", diff_6th_factor2=0.04,
                     diff_6th_slopeopt=1, diff_6th_thresh=0.05,
                     specified=True)


def check_steps(build_dir: Path) -> dict:
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor
    inp = build_dir / "steps.in"
    with inp.open("wb") as fh:
        np.array([len(DTS)], "<i4").tofile(fh)
        np.asarray(DTS, "<f4").tofile(fh)
    out = {}
    for arm in ("o0", "wrf_o2"):
        subprocess.run([str(build_dir / f"steps_{arm}"), str(inp),
                        str(build_dir / f"steps_{arm}.out")], check=True)
        out[arm] = np.fromfile(build_dir / f"steps_{arm}.out", "<f4")
    rows = []
    bad = 0
    for i, dt in enumerate(DTS):
        cfg = _fork_cfg(float(F(dt)))
        for slot, array, _ in ROWS:
            dry = slot in ("smag_ru", "smag_rv", "smag_rw", "smag_rth")
            want_dt = F(F(dt)) if dry else out["o0"][i]
            want_factor = F(0.12) if slot in ("smag_ru", "smag_rv", "smag_rw",
                                              "smag_rth", "smag_rtke") else F(0.04)
            got_dt = F(_diff6_dt(cfg, slot))
            got_factor = F(_diff6_factor(cfg, slot))
            ok = (got_dt.view(np.uint32) == want_dt.view(np.uint32)
                  and got_factor.view(np.uint32) == want_factor.view(np.uint32))
            bad += not ok
            rows.append({"dt": float(F(dt)), "slot": slot, "wrf_array": array,
                         "fork_dt": float(want_dt), "woof_dt": float(got_dt),
                         "fork_factor": float(want_factor),
                         "woof_factor": float(got_factor), "match": bool(ok)})
    same_o2 = bool(np.array_equal(out["o0"].view(np.uint32), out["wrf_o2"].view(np.uint32)))
    return {"rows": len(rows), "mismatches": bad, "o0_equals_o2": same_o2,
            "full_steps": {str(float(F(d))): float(v) for d, v in zip(DTS, out["o0"])},
            "detail": rows}


def _field(regime, slot, stagger, rng):
    """A physically shaped field for one row and regime, (nlev, ny, nx)."""
    nlev = NZ + 1 if stagger == "z" else NZ
    ny = NY + 1 if stagger == "y" else NY
    nx = NX + 1 if stagger == "x" else NX
    z = np.linspace(0.0, 1.0, nlev)[:, None, None]
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cell = np.exp(-((xx - nx * 0.55) ** 2 + (yy - ny * 0.45) ** 2) / 12.0)[None]
    front = np.tanh((xx - 0.4 * nx - 0.3 * yy) / 2.0)[None]
    noise = rng.standard_normal((nlev, ny, nx))
    if regime == "edges":
        # Sharp gradients against the domain edge, the halo the branch reads.
        cell = np.zeros_like(cell)
        cell[:, :, :2] = 1.0
        cell[:, :2, :] = 1.0
    base = {
        "smag_ru": 10.0 + 25.0 * z + 8.0 * front + 3.0 * cell,
        "smag_rv": -4.0 + 10.0 * z * front + 2.0 * noise * 0.3,
        "smag_rw": 0.05 * noise + 6.0 * cell * np.sin(np.pi * z),
        "smag_rth": 295.0 + 40.0 * z + 2.0 * front - 3.0 * cell * (z < .3),
        "smag_rtke": np.maximum(0.0, 1.5 * cell * (1 - z) + 0.01 * noise),
        "smag_rqv": 0.016 * np.exp(-3 * z) * (1 + 0.2 * cell - 0.1 * front),
        "smag_rqc": 1.5e-3 * cell * np.exp(-((z - .3) / .15) ** 2),
        "smag_rqr": 3e-3 * cell ** 2 * (z < .4),
        "smag_rqi": 1e-4 * cell * (z > .6),
        "smag_rqs": 1.5e-3 * cell * (z > .4),
        "smag_rqg": 2e-3 * cell ** 3 * (z > .3) * (z < .7),
        "smag_rni": 1e5 * cell * (z > .6),
        "smag_rnr": 1e4 * cell ** 2 * (z < .4),
        "smag_rnc": 1e8 * cell * (z < .5),
        "smag_rnwfa": 3e8 * (1 + 0.5 * front) * np.exp(-2 * z),
        "smag_rnifa": 1e6 * (1 + 0.3 * front) * np.exp(-3 * z),
    }[slot]
    if regime == "stable_night":
        base = base * (0.2 if slot not in ("smag_rth", "smag_rqv", "smag_ru",
                                            "smag_rnwfa", "smag_rnifa") else 1.0)
    elif regime == "snow" and slot in ("smag_rqs", "smag_rqi", "smag_rni"):
        base = base * 3.0 + 1e-6 * (slot != "smag_rni")
    elif regime == "ocean_stratocu" and slot in ("smag_rqc", "smag_rnc"):
        base = base + (1e-4 if slot == "smag_rqc" else 1e8) * ((z > .05) & (z < .15))
    if regime == "edges" and slot in ("smag_rqc", "smag_rqr"):
        base = base + 1e-12 * noise  # threshold-scale words, signed
    return np.ascontiguousarray(np.broadcast_to(base, (nlev, ny, nx)).astype(F))


def _case(regime, slot, stagger, seed):
    rng = np.random.default_rng(seed)
    f = _field(regime, slot, stagger, rng)
    nlev = f.shape[0]
    mut = (8.0e4 + 1.0e4 * rng.random((NY, NX))).astype(F)
    c1 = np.linspace(1.0, 0.0, nlev).astype(F)
    c2 = np.linspace(0.0, 1.2e4, nlev).astype(F)
    if regime == "high_terrain":
        ridge = 2500.0 * np.exp(-((np.arange(NX)[None, :] - NX / 2) ** 2) / 6.0) * np.ones((NY, 1))
        mut = (mut - 2.5e4 * ridge / 2500.0).astype(F)
    elif regime == "ocean_stratocu":
        ridge = np.zeros((NY, NX))
    else:
        ridge = 300.0 * np.exp(-((np.arange(NX)[None, :] - NX / 2) ** 2) / 8.0) * np.ones((NY, 1))
    phb = (9.81 * (ridge[None] + 600.0 * np.arange(NZ + 1)[:, None, None]
                   + 20.0 * rng.random((NZ + 1, NY, NX)))).astype(F)
    msfu = (1.0 + 0.03 * rng.random((NY, NX + 1))).astype(F)
    msfv = (1.0 + 0.03 * rng.random((NY + 1, NX))).astype(F)
    msft = (1.0 + 0.03 * rng.random((NY, NX))).astype(F)
    return dict(f=f, mut=mut, c1=c1, c2=c2, phb=phb, msfu=msfu, msfv=msfv,
                msft=msft, nx=NX, ny=NY, nz=NZ, slope=1, opt=2, stagger=stagger)


def run(build_dir: Path, out_dir: Path) -> dict:
    import cupy as cp
    import diff6_fork_oracle as d6
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor, launch_diff6_to_edge
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    build_dir, out_dir = build_dir.resolve(), out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default"
    steps = check_steps(build_dir)
    cfg = _fork_cfg(20.0)
    rows = []
    seed = 0
    for regime in REGIMES:
        for slot, array, st in ROWS:
            seed += 1
            c = _case(regime, slot, st, seed)
            factor, dt = float(F(_diff6_factor(cfg, slot))), float(F(_diff6_dt(cfg, slot)))
            inp = out_dir / "case.bin"
            d6._fortran_input(c, factor, dt, DX, DX, 0.05, inp)
            outp = out_dir / "case.out"
            subprocess.run([str(build_dir / "diff6_fork_driver"), str(inp), str(outp)],
                           check=True)
            want = np.fromfile(outp, F).reshape(NY + 7, NZ + 1, NX + 7).transpose(1, 0, 2)
            nlev, nys, nxs = c["f"].shape
            want = np.ascontiguousarray(want[:nlev, 3:3 + nys, 3:3 + nxs])
            tend = cp.zeros(c["f"].shape, cp.float32)
            launch_diff6_to_edge(cp.asarray(c["f"]), tend, cp.asarray(c["mut"]),
                                 cp.asarray(c["c1"]), cp.asarray(c["c2"]), factor, dt, 2,
                                 stagger=st, phb=cp.asarray(c["phb"]),
                                 msfu=cp.asarray(c["msfu"]), msfv=cp.asarray(c["msfv"]),
                                 msft=cp.asarray(c["msft"]), slopeopt=1, thresh=0.05,
                                 dx=DX, dy=DX)
            got = tend.get()
            diff = got.view(np.uint32) != want.view(np.uint32)
            rows.append({"regime": regime, "slot": slot, "wrf_array": array,
                         "stagger": st or "mass", "factor": factor, "dt": dt,
                         "columns": int(nys * nxs), "words": int(got.size),
                         "nonzero_tendency_words": int(np.count_nonzero(want)),
                         "different_words": int(diff.sum()),
                         "max_ulp": int(fp32_ulp_distance(got, want).max(initial=0))})
    summary = {"mode": mode, "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "patch": [NX, NY, NZ], "cases": len(rows),
               "columns": sum(r["columns"] for r in rows),
               "words": sum(r["words"] for r in rows),
               "different_words": sum(r["different_words"] for r in rows),
               "max_ulp": max(r["max_ulp"] for r in rows),
               "per_row_steps": {k: v for k, v in steps.items() if k != "detail"},
               "per_row_steps_detail": steps["detail"], "rows": rows,
               "build": json.loads((build_dir / "diff6-fork-build.json").read_text())}
    (out_dir / f"diff6-{mode}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"diff6 {mode}: cases={summary['cases']} words={summary['words']} "
          f"different={summary['different_words']} max_ulp={summary['max_ulp']}; "
          f"per-row factor/step mismatches={steps['mismatches']} of {steps['rows']}")
    return summary


if __name__ == "__main__":
    if sys.argv[1] == "build":
        print(json.dumps(build(Path(sys.argv[2]), Path(sys.argv[3])), indent=1))
    else:
        r = run(Path(sys.argv[2]), Path(sys.argv[3]))
        sys.exit(1 if r["different_words"] or r["per_row_steps"]["mismatches"] else 0)
