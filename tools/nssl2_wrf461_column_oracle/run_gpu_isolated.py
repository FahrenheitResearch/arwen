"""Stage-isolated WOOF runs from WRF's own intermediate states (attribution).

The end-to-end oracle (run_gpu_columns.py) feeds both engines the same
inputs; any early difference is then amplified by every later stage.  This
runner removes the amplification: each WOOF stage starts from WRF's dumped
state at the stage's entry (the diagnostic stage build, instrument.py) and
is graded against WRF's state at its exit:

  gs       WRF stage 1 -> launch_fused_gs              -> vs WRF stage 2
  nucond   WRF stage 2 -> launch_nucond (default hook) -> vs WRF stage 3
  diag     WRF stage 3 -> radar, radii, scatter        -> vs WRF final output

The launchers are called with exactly the arguments the production hooks
and coordinator pass (gpuwm/core/nssl2_default_hooks.py,
nssl2_fused_gs.NSSL2FusedGS, nssl2_production_coordinator).  Sedimentation
needs no isolation: it is the first stage and both engines enter it with
identical bytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_bitwise import OUT2, OUT3, field_stats, read_fortran
from compare_stages import WOOF_FIELDS, read_dumps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix", type=Path, help="run prefix, e.g. runs/r2/dt12.0_it2_exact")
    p.add_argument("columns", type=Path)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--final", type=Path, default=None,
                   help="WRF final output to grade the diag/scatter stage against "
                        "(default PREFIX.wrf_scalar.bin)")
    a = p.parse_args()

    import cupy as cp
    from gpuwm.core.nssl2_diagnostics import launch_radardd02
    from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
    from gpuwm.core.nssl2_fused_gs import launch_fused_gs
    from gpuwm.core.nssl2_nucond import launch_nucond
    from gpuwm.core.nssl2_production_coordinator import scatter_nssl2_driver_workspace
    from gpuwm.core.nssl2_radiation import launch_effective_radius_concentration

    regime = np.load(a.columns)["regime"]
    ncol = regime.size
    fin = np.load(a.prefix.with_name(a.prefix.name + ".fin.npz"))
    nz = fin["theta"].shape[1]
    wrf = read_dumps(a.prefix.with_name(a.prefix.name + ".wrf_stages.bin"), ncol)
    final = read_fortran(a.final or a.prefix.with_name(a.prefix.name + ".wrf_scalar.bin"),
                         ncol, nz)
    shape = (nz, 1, ncol)

    def dev(x):  # (ncol, nzx) -> (nzx, 1, ncol)
        return cp.asarray(np.ascontiguousarray(np.asarray(x, np.float32).T[:, None, :]))

    def host(x):
        return np.ascontiguousarray(cp.asnumpy(x)[:, 0, :].T)

    rho, pres, pii, dz = (dev(fin[n]) for n in ("rho", "pressure", "exner", "dz"))
    w = dev(fin["w"])
    step = np.float32(a.dt)

    def workspace_from(stage):
        ws = NSSL2DriverWorkspace(cp.empty((16, *shape), dtype=cp.float32),
                                  cp.zeros((5, 1, ncol), dtype=cp.float32), shape,
                                  ignored_accumulator=cp.zeros((1, ncol), dtype=cp.float32))
        for i, name in enumerate(WOOF_FIELDS):
            ws.state[i] = dev(wrf[stage][name])
        return ws, dev(wrf[stage]["theta"])

    saved = {}

    def grade(label, got, want_stage=None, want=None):
        for name, value in got.items():
            saved[f"{label.split()[0]}__{name}"] = np.asarray(value, np.float32)
        rows, total = {}, 0
        for name, value in got.items():
            ref = want[name] if want is not None else wrf[want_stage][name]
            st = field_stats(np.asarray(ref, np.float32), np.asarray(value, np.float32), regime)
            if st["bit_mismatches"]:
                rows[name] = {k: st[k] for k in ("bit_mismatches", "max_ulp", "max_abs_diff")}
                rows[name]["worst"] = st["worst"]
                total += st["bit_mismatches"]
        print(f"[{label}] {'EXACT' if not total else f'{total} mismatches'}")
        for name, r in rows.items():
            print(f"   {name:7s} n={r['bit_mismatches']:5d} max_ulp={r['max_ulp']:<11d} "
                  f"worst={r['worst']}")
        return {"total_bit_mismatches": total, "fields": rows}

    report = {}
    # GS
    ws, theta = workspace_from(1)
    temperature = cp.empty(shape, dtype=cp.float32)
    target = cp.empty(shape, dtype=cp.float32)
    # WRF diagnoses t7 from the vapour as loaded (before the first-step
    # calcnfromq and sedimentation): the Fortran input qv.
    launch_fused_gs(ws, theta, rho, pres, pii, w, temperature, target, dz, step, hail_on=True,
                    loaded_vapor=dev(fin["qv"]))
    got = {name: host(ws.field(name)) for name in WOOF_FIELDS}
    got["theta"] = host(theta)
    report["gs"] = grade("gs from WRF stage 1", got, want_stage=2)
    report["gs_t7"] = grade("t7 prepass from WRF stage 1", {"t7": host(target)},
                            want={"t7": wrf[1]["t7"]})

    # NUCOND (+ its internal QVEXCESS), as the default hook calls it
    ws, theta = workspace_from(2)
    ss = cp.empty(shape, dtype=cp.float32)
    launch_nucond(theta, rho, pres, pii, w, ws.field("qv"), ws.field("qc"), ws.field("qr"),
                  ws.field("qi"), ws.field("qs"), ws.field("qndrop"), ws.field("qnr"),
                  ws.field("qni"), ws.field("qns"), ws.field("qnn"), step,
                  supersaturation_scratch=ss, concentration_space=True,
                  predicted_ccn=True, validate_values=False,
                  frozen=tuple(ws.field(n) for n in
                               ("qg", "qh", "qng", "qnh", "qvolg", "qvolh")))
    got = {name: host(ws.field(name)) for name in WOOF_FIELDS}
    got["theta"] = host(theta)
    report["nucond"] = grade("nucond from WRF stage 2", got, want_stage=3)

    # radar, radii and scatter, as the coordinator calls them
    ws, theta = workspace_from(3)
    temperature = theta * pii
    refl = cp.empty(shape, dtype=cp.float32)
    launch_radardd02(rho, temperature, ws.field("qr"), ws.field("qi"), ws.field("qs"),
                     ws.field("qg"), ws.field("qh"), ws.field("qnr"), ws.field("qni"),
                     ws.field("qns"), ws.field("qng"), ws.field("qnh"), ws.field("qvolg"),
                     ws.field("qvolh"), refl, output_due=True, concentration_space=True,
                     validate_values=False)
    rc, ri, rs = (cp.empty(shape, dtype=cp.float32) for _ in range(3))
    launch_effective_radius_concentration(rho, ws.field("qc"), ws.field("qndrop"),
                                          ws.field("qi"), ws.field("qni"), ws.field("qs"),
                                          ws.field("qns"), rc, ri, rs, validate_values=False)
    reg = {name: cp.empty(shape, dtype=cp.float32) for name in WOOF_FIELDS}
    scatter_nssl2_driver_workspace(ws, rho, *(reg[n] for n in WOOF_FIELDS), predicted_ccn=True)
    got = {name: host(reg[name]) for name in WOOF_FIELDS}
    got.update(refl_10cm=host(refl), re_cloud=host(rc), re_ice=host(ri), re_snow=host(rs),
               theta=host(theta))
    report["diag_scatter"] = grade("radar/radii/scatter from WRF stage 3", got,
                                   want={k: final[k] for k in got})
    np.savez(a.prefix.with_name(a.prefix.name + ".isolated.npz"), **saved)
    a.prefix.with_name(a.prefix.name + ".isolated.json").write_text(
        json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
