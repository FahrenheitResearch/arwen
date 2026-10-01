"""Capture one IEVA substep of a gpuwm forecast for the WRF oracle.

Usage::

    IEVA_CAPTURE_DIR=DIR python capture.py $(which gpuwm) sim ...   # capture
    python capture.py --compare DIR                                  # grade

``gpuwm sim`` integrates in a worker process, so the capture also installs
itself from ``sitecustomize``: put ``tools/ieva_wrf_oracle/site`` first on
``PYTHONPATH`` with ``IEVA_CAPTURE_DIR`` set and every Python process of
the run carries it; the first IEVA substep over the threshold is written
once (``O_EXCL`` on ``DIR/claimed``).

The first form runs the command unchanged except that the first IEVA
substep whose WRF vertical Courant number (w_damp's
``|ww/(c1f*mut+c2f)*rdnw*dt|``, interior w levels) exceeds
``IEVA_CAPTURE_MIN_CFL`` (default 1) has every input of the dynamics
solves, the tendencies before and after each solve, and the first moist
species' scalar split and solve written to DIR as float32 C-order files
plus ``meta.txt``.  ``ieva_oracle`` (WRF 4.7.1's compiled routines) then
reads DIR and writes ``wrf_*.bin``; ``--compare`` grades gpuwm's words
against WRF's.
"""

from __future__ import annotations

import json
import os
import runpy
import sys

import numpy as np


def _compare(directory: str) -> int:
    meta = open(os.path.join(directory, "meta.txt")).read().split()
    nx, ny, nz = int(meta[0]), int(meta[1]), int(meta[2])

    def load(name, shape):
        return np.fromfile(os.path.join(directory, name + ".bin"),
                           dtype=np.float32).reshape(shape)

    fl = (nz + 1, ny, nx)
    rows = []
    # (gpuwm file, WRF file, shape, region WRF computes, label)
    interior_w = (slice(1, nz), slice(None), slice(None))
    ring_free_w = (slice(1, nz), slice(1, ny - 1), slice(1, nx - 1))
    cases = [
        ("wwE", "wrf_wwE", fl, interior_w, "WW_SPLIT wwE (dynamics)"),
        ("wwI", "wrf_wwI", fl, interior_w, "WW_SPLIT wwI (dynamics)"),
        ("mut_new", "wrf_mut_new", (ny, nx), (slice(None), slice(None)),
         "CALC_MUT_NEW"),
        ("ru_t_ieva", "wrf_ru_t", (nz, ny, nx + 1),
         (slice(None), slice(None), slice(1, nx)), "advect_u_implicit"),
        ("rv_t_ieva", "wrf_rv_t", (nz, ny + 1, nx),
         (slice(None), slice(1, ny), slice(None)), "advect_v_implicit"),
        ("rth_t_ieva", "wrf_rth_t", (nz, ny, nx),
         (slice(None), slice(None), slice(None)),
         "advect_s_implicit (theta)"),
        ("rph_t_ieva", "wrf_rph_t", fl, interior_w, "advect_ph_implicit"),
        ("rw_t_ieva", "wrf_rw_t", fl, ring_free_w,
         "advect_w_implicit (outer row excluded, see ieva.py (3))"),
        ("wwE_m", "wrf_wwE_m", fl, interior_w, "WW_SPLIT wwE (scalars)"),
        ("wwI_m", "wrf_wwI_m", fl, interior_w, "WW_SPLIT wwI (scalars)"),
        ("q_tend_ieva", "wrf_q_tend", (nz, ny, nx),
         (slice(None), slice(None), slice(None)),
         "advect_s_implicit (first moist species)"),
    ]
    failed = 0
    for mine, theirs, shape, region, label in cases:
        a = load(mine, shape)[region]
        b = load(theirs, shape)[region]
        ia, ib = a.view(np.uint32), b.view(np.uint32)
        differ = ia != ib
        zero_sign = differ & (a == 0) & (b == 0)
        words = int(differ.sum() - zero_sign.sum())
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        ulp = np.abs(ia.astype(np.int64) - ib.astype(np.int64))
        ulp[zero_sign] = 0
        row = {"array": label, "words": int(a.size),
               "bit_differences": words,
               "signed_zero_only": int(zero_sign.sum()),
               "max_abs_diff": float(diff.max()) if diff.size else 0.0,
               "max_ulp": int(ulp.max()) if ulp.size else 0,
               "nonzero_words": int(np.count_nonzero(b))}
        rows.append(row)
        failed += words != 0
    extra = json.load(open(os.path.join(directory, "capture.json")))
    report = {"capture": extra, "rows": rows,
              "all_bit_identical": failed == 0}
    json.dump(report, open(os.path.join(directory, "compare.json"), "w"),
              indent=1)
    for row in rows:
        print(f"{row['array']:<56} {row['words']:>9} words  "
              f"{row['bit_differences']:>7} differ  "
              f"(+{row['signed_zero_only']} signed zero)  "
              f"max ulp {row['max_ulp']}")
    print("ALL BIT-IDENTICAL" if failed == 0 else f"{failed} ARRAYS DIFFER")
    return 0 if failed == 0 else 1


def _install(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    import cupy as cp
    from gpuwm.core import ieva

    min_cfl = float(os.environ.get("IEVA_CAPTURE_MIN_CFL", "1.0"))
    status = {"armed": False, "done_dyn": False, "done_scalar": False,
              "substeps_seen": 0, "max_cfl_seen": 0.0}
    info: dict = {}

    import atexit

    def _report():
        if status["substeps_seen"]:
            path = os.path.join(directory, f"seen-{os.getpid()}.json")
            with open(path, "w") as fh:
                json.dump({k: v for k, v in status.items()}, fh)
    atexit.register(_report)

    def save(name, arr):
        np.ascontiguousarray(cp.asnumpy(arr), dtype=np.float32).tofile(
            os.path.join(directory, name + ".bin"))

    orig_prepare = ieva.prepare_dynamics

    def prepare_dynamics(state, cfg, ww):
        ctx = orig_prepare(state, cfg, ww)
        status["substeps_seen"] += 1
        if status["done_dyn"]:
            return ctx
        nz = state.p.shape[0]
        mass = (state.c1f[1:nz, None, None] * ctx.mut[None]
                + state.c2f[1:nz, None, None])
        cfl = cp.abs(ww[1:nz] / mass * state.rdnw[1:nz, None, None]
                     * np.float32(cfg.dt))
        vmax = float(cfl.max())
        status["max_cfl_seen"] = max(status["max_cfl_seen"], vmax)
        if vmax <= min_cfl:
            return ctx
        try:
            os.close(os.open(os.path.join(directory, "claimed"),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            status["done_dyn"] = status["done_scalar"] = True
            return ctx
        status["armed"] = True
        info.update(
            step_substep=status["substeps_seen"],
            elapsed_seconds=float(state.elapsed_seconds),
            dt=float(cfg.dt), dx=float(cfg.dx), dy=float(cfg.dy),
            max_vertical_cfl=vmax,
            cells_over_1=int((cfl > 1.0).sum()),
            cells_over_1_5=int((cfl > 1.5).sum()),
            has_msf=bool(state.has_msf),
            implicit_share_nonzero=int(cp.count_nonzero(ctx.wwI)),
            nz=int(nz), ny=int(state.p.shape[1]), nx=int(state.p.shape[2]))
        mux, muy = ieva.stage_face_masses(state, cfg, ctx.mut)
        for name, arr in (("ww", ww), ("u", state.u), ("v", state.v),
                          ("u0", state.u0), ("v0", state.v0),
                          ("w0", state.w0), ("php", state.php),
                          ("php0", state.php0),
                          ("mut", ctx.mut), ("mut_old", ctx.mut_old),
                          ("mut_new", ctx.mut_new), ("wwE", ctx.wwE),
                          ("wwI", ctx.wwI), ("mux", mux), ("muy", muy),
                          ("msft", state.msft), ("msfu", state.msfu),
                          ("msfv", state.msfv), ("ht", state.ht),
                          ("rdnw", state.rdnw), ("rdn", state.rdn),
                          ("c1f", state.c1f), ("c2f", state.c2f),
                          ("c1h", state.c1h), ("c2h", state.c2h),
                          ("fnm", state.fnm), ("fnp", state.fnp)):
            save(name, arr)
        phb = state.phb
        if phb.ndim == 1:
            phb = cp.broadcast_to(phb[:, None, None], state.php.shape)
        save("phb", phb)
        # WRF's t_1, as solve_theta's kernel forms it: (thb - t0) + thp0.
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        save("theta_old", (thb - ieva.THETA_OFFSET) + state.thp0)
        info["cf"] = [float(state.cf1), float(state.cf2), float(state.cf3)]
        return ctx

    def wrap(name, field):
        orig = getattr(ieva, name)

        def solve(state, cfg, ctx):
            if status["armed"] and not status["done_dyn"]:
                save(field + "_explicit", getattr(state, field))
            orig(state, cfg, ctx)
            if status["armed"] and not status["done_dyn"]:
                save(field + "_ieva", getattr(state, field))
                if name == "solve_w":
                    status["done_dyn"] = True
        setattr(ieva, name, solve)

    for name, field in (("solve_u", "ru_t"), ("solve_v", "rv_t"),
                        ("solve_theta", "rth_t"), ("solve_ph", "rph_t"),
                        ("solve_w", "rw_t")):
        wrap(name, field)
    ieva.prepare_dynamics = prepare_dynamics

    orig_split = ieva.split_scalar_omega

    def split_scalar_omega(state, cfg, ww_m, mut, dt):
        wwE, wwI = orig_split(state, cfg, ww_m, mut, dt)
        if status["done_dyn"] and not status["done_scalar"]:
            status["scalar_armed"] = True
            info["dt_scalar"] = float(np.float32(dt))
            save("ww_m", ww_m)
            save("muts", mut)
            save("mu0s", state.mub2d + state.mup0)
            save("wwE_m", wwE)
            save("wwI_m", wwI)
        return wwE, wwI

    orig_solve_scalar = ieva.solve_scalar

    def solve_scalar(state, tend, q_old, wwI, mu_old, mu_new, dt):
        grab = status.get("scalar_armed") and not status["done_scalar"]
        if grab:
            save("q_tend_explicit", tend)
            save("q_old", q_old)
        orig_solve_scalar(state, tend, q_old, wwI, mu_old, mu_new, dt)
        if grab:
            save("q_tend_ieva", tend)
            status["done_scalar"] = True
            with open(os.path.join(directory, "meta.txt"), "w") as fh:
                fh.write(f"{info['nx']} {info['ny']} {info['nz']} "
                         f"{int(info['has_msf'])}\n")
                def f32(x):
                    return repr(float(np.float32(x)))
                fh.write(f"{f32(info['dt'])} {f32(info['dx'])} "
                         f"{f32(info['dy'])} {f32(info['dt_scalar'])}\n")
                fh.write(" ".join(f32(x) for x in info["cf"]) + "\n")
            json.dump(info, open(os.path.join(directory, "capture.json"), "w"),
                      indent=1)
            print(f"IEVA_CAPTURED {json.dumps(info)}", flush=True)
    ieva.solve_scalar = solve_scalar
    ieva.split_scalar_omega = split_scalar_omega


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--compare":
        return _compare(sys.argv[2])
    directory = os.environ["IEVA_CAPTURE_DIR"]
    os.makedirs(directory, exist_ok=True)
    _install(directory)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
