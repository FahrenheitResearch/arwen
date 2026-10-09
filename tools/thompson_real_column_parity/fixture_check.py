#!/usr/bin/env python3
"""Grade the port's host build against a committed WRF v4.6.1 fixture
(``make_fixture.py``): every process rate, the final state, reflectivity
and the surface accumulations, classified the way ``real_column_parity``
classifies them.

A cell whose relative difference from WRF exceeds 2e-6 is counted as
rounding when the port's own response to a one-unit nudge of every input
(four draws) explains it, or, for the final state, when it is within four
float32 units of the largest value the cell held.  Under mp=8 two rates
carry one more named rule each, because rounding decides them there:

* rain evaporation (``prv_rev``, ``pnr_rev``) where the saturation
  adjustment has just brought the air to saturation: WRF evaporates where
  the post-adjustment ``ssatw`` is below -1e-15 (:3501), and that ``ssatw``
  is a residual of a few float32 units, so its sign is rounding's.  A cell
  is explained when WRF's ``ssatw`` there is within 16 float32 epsilons of
  zero after an adjustment, or when the difference moves at most 1e-5 of
  the cell's rain;
* rain self-collection and break-up (``pnr_rcr``) near the 1950 micron
  crossing (:2159-2176), whose relative sensitivity to the rain mean
  diameter is ``kappa``; explained when the relative gap over ``kappa`` is
  at most 1e-6.

Under mp=28 neither rule applies.  Both were transcription differences, not
rounding: the mp=28 rain distribution divided the DOUBLE literal 3.672 where
WRF divides the REAL sum 3.0 + mu_r + 0.672 (pnr_rcr through mvd_r), and the
condensation and rain evaporation added their increments to the running
vapour and temperature where WRF re-forms qv1d + DT*qvten and t1d + DT*tten
(the sign of the post-adjustment ssatw).  Both were repaired on
lane/mp28fix-warm-network, after which the mp=28 host build gives pnr_rcr
bit for bit on every active cell of the 153-column oracle and prv_rev
differs only where an upstream rate does.

A fixture written by ``make_fixture.py --mp8`` carries ``mp_physics`` 8
and grades the classic adapter (``gpuwm.core.microphysics._apply_thompson``)
on the same columns; the rates classic Thompson does not carry
(``port_rates.NOT_CARRIED_MP8``) are reported with WRF's activity only.

Needs a C++ compiler and the Thompson table set, nothing else.  Prints one
JSON object; ``tests/test_thompson_real_column_host_parity.py`` asserts on
it, in process on a POSIX box and through WSL on Windows.

usage: fixture_check.py FIXTURE.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import port_rates  # noqa: E402
import real_column_parity as R  # noqa: E402

FAR = 1.0e-2
EPS32 = float(np.finfo(np.float32).eps)


def _rel(a, b, floor):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    big = np.maximum(np.abs(a), np.abs(b))
    ok = np.isfinite(a) & np.isfinite(b) & (big > floor)
    rel = np.zeros_like(big)
    rel[ok] = np.abs(a - b)[ok] / big[ok]
    return rel, ok


def _unexplained_far(port, wrf, sens, floor=0.0, scale=None):
    """Cells beyond FAR that neither the sensitivity nor the cell scale
    explains, as a boolean array."""
    a = np.asarray(port, np.float64)
    b = np.asarray(wrf, np.float64)
    rel, ok = _rel(a, b, floor)
    gap = np.abs(a - b)
    far = ok & (rel > FAR)
    within = gap <= R.SENSITIVITY_FACTOR * np.nan_to_num(sens)
    if scale is not None:
        cell = np.maximum(np.nan_to_num(np.asarray(scale, np.float64)),
                          np.maximum(np.abs(a), np.abs(b)))
        within |= gap <= R.CELL_SCALE_UNITS * R.ulp32(cell)
    return far & ~within


def check(fixture):
    z = np.load(fixture)
    mp = int(z["mp_physics"]) if "mp_physics" in z.files else 28
    cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
    dt = float(z["dt"])
    cols["dt"] = np.float32(dt)
    inp = R.prepare(cols, mp)
    ncol, nz = inp["p"].shape
    port = R.run_port(inp, dt, rates=True, mp=mp)
    sens = R.sensitivity(cols, dt, port, mp)
    out = {"mp_physics": mp, "ncol": ncol, "nz": nz, "rates": {},
           "final": {}, "rates_not_carried": {},
           "labels": {str(k): int(v) for k, v in zip(
               *np.unique(z["labels"], return_counts=True))}}

    # Rain-evaporation rule inputs.
    ssatw = z["cp2_ssatw"].astype(np.float64)
    adjusted = z["cp2_prw_vcd"] != 0
    qr_pre = np.maximum(port["stages"]["sources"]["qr"], 1.0e-30)
    # Break-up rule inputs (:2159-2176).
    rho = z["cp1_rho"].astype(np.float64)
    mvd = R.rain_mean_volume_diameter(z["cp1_qr1d"], z["cp1_nr1d"], rho)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = 2300.0 * (mvd - 1950.0e-6)
        kappa = np.abs(2300.0 * mvd * np.exp(x) / np.where(
            np.abs(1.0 - np.exp(x)) > 0, 1.0 - np.exp(x), 1.0e-30))

    for name in R.RATES:
        wrf_v = z[f"rate_{name}"].astype(np.float64)
        if name in port_rates.NOT_CARRIED[mp]:
            out["rates_not_carried"][name] = int(
                (np.isfinite(wrf_v) & (wrf_v != 0)).sum())
            continue
        port_v = np.where(np.isfinite(wrf_v), port["rates"][name], np.nan)
        cls = R.classify(port_v, wrf_v, sens["rates"][name],
                         mask=np.isfinite(wrf_v))
        un = _unexplained_far(port_v, wrf_v, sens["rates"][name])
        rel, _ok = _rel(port_v, wrf_v, 0.0)
        # mp=8 only: see the module docstring for why mp=28 has no such rule.
        if mp == 8 and name in ("prv_rev", "pnr_rev"):
            moved = (np.abs(port["rates"]["prv_rev"]
                            - z["rate_prv_rev"].astype(np.float64))
                     * dt / qr_pre)
            un &= ~(((np.abs(ssatw) <= 16 * EPS32) & adjusted)
                    | (moved <= 1.0e-5))
        if mp == 8 and name == "pnr_rcr":
            un &= ~(rel / np.maximum(np.nan_to_num(kappa), 1.0) <= 1.0e-6)
        entry = {k: cls[k] for k in (
            "n_active", "n_beyond_rounding", "n_beyond_unexplained",
            "n_beyond_1e-2")}
        entry["n_far_unexplained"] = int(un.sum())
        out["rates"][name] = entry

    def scale(var):
        s = np.abs(inp[var].astype(np.float64)) if var in inp else 0.0
        for snap in port["stages"].values():
            if var in snap:
                s = np.maximum(s, np.nan_to_num(np.abs(snap[var])))
        return s

    finals = [(var, z[f"wrf_{var}"], R.FLOOR[var])
              for var in R.SPECIES_BY_MP[mp]]
    finals.append(("ng", z["cpx_ng1d"], R.FLOOR["ng"]))
    for var, wrf_v, floor in finals:
        cls = R.classify(port["final"][var], wrf_v, sens["final"][var],
                         floor=floor, scale=scale(var))
        un = _unexplained_far(port["final"][var], wrf_v, sens["final"][var],
                              floor=floor, scale=scale(var))
        entry = {k: cls[k] for k in (
            "n_active", "n_beyond_rounding", "n_beyond_unexplained",
            "n_beyond_1e-2")}
        entry["n_far_unexplained"] = int(un.sum())
        out["final"][var] = entry
    rel, ok = _rel(port["final"]["T"], z["cpx_t1d"], 0.0)
    out["final"]["T_exit"] = {"n_active": int(ok.sum()),
                              "rel_max": float(rel.max())}
    for name in ("rainnc", "snownc", "graupelnc"):
        rel, ok = _rel(port["final"][name], z[f"wrf_{name}"], 1.0e-9)
        out["final"][name] = {"n_active": int(ok.sum()),
                              "rel_max": float(rel.max())}
    d = np.abs(port["final"]["refl"] - z["wrf_refl"].astype(np.float64))
    out["refl"] = {"max_abs_db": float(d.max()),
                   "n_gt_0p1_db": int((d > 0.1).sum())}
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    print(json.dumps(check(sys.argv[1]), indent=1))
