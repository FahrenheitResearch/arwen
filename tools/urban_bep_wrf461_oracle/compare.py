#!/usr/bin/env python
"""Grade WOOF's BEP GPU port against the WRF v4.6.1 column fixtures.

    python tools/urban_bep_wrf461_oracle/compare.py FIXTURE_ROOT [--json OUT]

FIXTURE_ROOT holds the four directories build.sh writes
(``bep461_{nlcd,lcz}/steps`` and ``bep461_couple_{nlcd,lcz}/steps``).  Run it
once with ``GPUWM_WRF_EXACT=1`` (strict build) and once without (default
arithmetic); the arithmetic mode is recorded in the output.

Four arms, every output word compared as a float32 bit pattern:

* ``bep_step``   SUBROUTINE BEP from each step's recorded inputs (state,
  14 PBL source-term profiles over all nz+1 rows including the words BEP must
  leave at the sentinel, the 4 column diagnostics);
* ``bep_chain``  the same, free running: WOOF carries its own state from the
  step-1 input through every step and each step is graded against WRF's;
* ``couple``     zeroing + BEP + the Noah / Noah-MP coupling block end to end
  (``urban_bep.after_lsm``) from each step's recorded inputs;
* ``sfcdiag``    the surface-diagnostic override.

Per field: words compared, words whose bits differ, max ULP (signed zeros
count as 0 ULP, NaN opposite anything as a sentinel distance), max |diff|.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SENTINEL = np.float32(-777.25)
STATE = ("trb_urb4d", "tw1_urb4d", "tw2_urb4d", "tgb_urb4d", "sfw1_urb3d",
         "sfw2_urb3d", "sfr_urb3d", "sfg_urb3d")
PROFILES = ("a_u", "a_v", "a_t", "a_e", "b_u", "b_v", "b_t", "b_e", "b_q",
            "dlg", "dl_u", "sf", "vl")
COLUMN = ("rl_up", "rs_abs", "emiss", "grdflx_urb")
COUPLE_PBL = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
              "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
              "dlg_bep", "dl_u_bep", "sf_bep", "vl_bep")
COUPLE_FIELDS = ("ust", "tsk", "hfx", "qfx", "lh", "grdflx", "albedo",
                 "emiss")
COUPLE_DIAG = ("ts_urb2d", "sh_urb2d", "lh_urb2d", "g_urb2d", "rn_urb2d")
BEP_OUT = ("rl_up_urb", "rs_abs_urb", "emiss_urb", "grdflx_urb")


def load(directory: Path) -> dict[str, np.ndarray]:
    """oracle_io case, C order (Fortran dims reversed), scalars as 1-element."""
    from gpuwm.verify.urban_oracle import load_case

    return {name: (np.ascontiguousarray(np.asarray(value).T)
                   if np.ndim(value) else np.atleast_1d(value))
            for name, value in load_case(directory).items()}


def table_params(fx) -> dict:
    params = {"ICATE": int(fx["tbl_ICATE"][0])}
    for key, value in fx.items():
        if key.startswith("tbl_") and key != "tbl_ICATE":
            params[key[4:]] = value.T if value.ndim == 2 else value
    return params


class Tally:
    """Per (arm, field) worst case over every case graded."""

    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}
        self.where: dict[tuple[str, str], list] = {}

    def add(self, arm: str, field: str, got, want, label: str = "",
            calm=None):
        """``calm``: words on WRF's calm-column defect (rural stress divided
        by a zero level-1 speed).  Where WRF wrote a non-finite word there,
        the word is not graded bitwise: it is counted under
        ``calm_wrf_nonfinite`` and WOOF's word must be finite."""
        from gpuwm.core.fp32_ulp import fp32_ulp_distance

        got = np.ascontiguousarray(got, dtype=np.float32).reshape(-1)
        want = np.ascontiguousarray(want, dtype=np.float32).reshape(-1)
        assert got.shape == want.shape, (arm, field, got.shape, want.shape)
        skip = np.zeros(got.shape, dtype=bool)
        if calm is not None:
            skip = np.ascontiguousarray(calm).reshape(-1) & ~np.isfinite(want)
        calm_row = self.rows.setdefault((arm, field), {
            "words": 0, "bits_differ": 0, "max_ulp": 0, "max_abs": 0.0,
            "nonfinite_wrf": 0, "nonfinite_woof": 0})
        calm_row["calm_wrf_nonfinite"] = (calm_row.get("calm_wrf_nonfinite", 0)
                                          + int(skip.sum()))
        calm_row["calm_woof_nonfinite"] = (calm_row.get("calm_woof_nonfinite", 0)
                                           + int((skip & ~np.isfinite(got)).sum()))
        got, want = got[~skip], want[~skip]
        differ = got.view(np.uint32) != want.view(np.uint32)
        ulp = fp32_ulp_distance(got, want)
        with np.errstate(invalid="ignore"):
            absd = np.abs(got.astype(np.float64) - want.astype(np.float64))
        both_nan = np.isnan(got) & np.isnan(want)
        absd = np.where(both_nan, 0.0, absd)
        absd = np.where(np.isnan(absd), np.inf, absd)
        row = self.rows.setdefault((arm, field), {
            "words": 0, "bits_differ": 0, "max_ulp": 0, "max_abs": 0.0,
            "nonfinite_wrf": 0, "nonfinite_woof": 0})
        row["words"] += int(got.size)
        row["bits_differ"] += int(differ.sum())
        row["max_ulp"] = max(row["max_ulp"], int(ulp.max()) if ulp.size else 0)
        row["max_abs"] = max(row["max_abs"], float(absd.max()) if absd.size else 0.0)
        row["nonfinite_wrf"] += int((~np.isfinite(want)).sum())
        row["nonfinite_woof"] += int((~np.isfinite(got)).sum())
        if differ.any():
            self.where.setdefault((arm, field), []).append(
                (label, np.flatnonzero(differ)[:8].tolist()))

    def summary(self) -> dict:
        out = {}
        for (arm, field), row in sorted(self.rows.items()):
            out.setdefault(arm, {})[field] = row
        return out


def _cp():
    import cupy as cp
    return cp


def bep_launch(fx, state_np: dict):
    """One BEP call from recorded atmosphere + the given state."""
    cp = _cp()
    from gpuwm.core.urban_bep import launch_bep_columns

    nz, ncol = int(fx["meta_nz"][0]), int(fx["meta_ncol"][0])

    def lev(a, rows):
        return cp.asarray(np.ascontiguousarray(a[:rows].reshape(rows, 1, ncol)))

    def sfc(a, dtype=np.float32):
        return cp.asarray(np.ascontiguousarray(a.reshape(1, ncol), dtype=dtype))

    state = {n: lev(state_np[n], state_np[n].shape[0]) for n in STATE}
    # WOOF's profiles carry the nz mass levels (sf: nz+1 faces), each preset
    # to the sentinel so words the port must not write stay visible.
    outputs = {n: cp.full((nz + 1 if n == "sf" else nz, 1, ncol), SENTINEL,
                          dtype=cp.float32) for n in PROFILES}
    outputs["b_q"][:] = 0.0                                       # :1611
    for n in COLUMN:                                              # :1607-1610
        outputs[n] = cp.zeros((1, ncol), dtype=cp.float32)
    outputs["error_flags"] = cp.zeros((1, ncol), dtype=cp.int32)
    nhi = fx["hi_urb2d"].shape[0]
    res = launch_bep_columns(
        params=table_params(fx), frc_urb2d=sfc(fx["frc_urb2d"]),
        utype_urb2d=sfc(fx["utype_urb2d"], np.int32),
        dz8w=lev(fx["dz8w"], nz), u_phy=lev(fx["u_phy"], nz),
        v_phy=lev(fx["v_phy"], nz), th_phy=lev(fx["th_phy"], nz),
        rho=lev(fx["rho"], nz), p_phy=lev(fx["p_phy"], nz),
        swdown=sfc(fx["swdown"]), glw=sfc(fx["glw"]),
        cosz_urb2d=sfc(fx["cosz_urb2d"]), omg_urb2d=sfc(fx["omg_urb2d"]),
        declin_urb=float(fx["meta_declin_urb"][0]),
        dt=float(fx["meta_dt"][0]), lp_urb2d=sfc(fx["lp_urb2d"]),
        lb_urb2d=sfc(fx["lb_urb2d"]), hgt_urb2d=sfc(fx["hgt_urb2d"]),
        hi_urb2d=lev(fx["hi_urb2d"], nhi), outputs=outputs, **state)
    got = {n: cp.asnumpy(v).reshape(v.shape[0] if v.ndim == 3 else 1, ncol)
           for n, v in res.items()}
    got.update({n: cp.asnumpy(v).reshape(-1, ncol) for n, v in state.items()})
    return got


def grade_bep(tally: Tally, fx, tset: str):
    nstep = int(fx["meta_nstep"][0])
    chain = {n: fx["s1_in_" + n].copy() for n in STATE}
    for step in range(1, nstep + 1):
        tag = f"s{step}_out_"
        for arm, state in (("bep_step", {n: fx[f"s{step}_in_" + n] for n in STATE}),
                           ("bep_chain", chain)):
            got = bep_launch(fx, state)
            label = f"{tset} step {step}"
            flags = int(np.count_nonzero(got["error_flags"]))
            tally.add(arm, "error_flags", np.float32([flags]), np.float32([0]), label)
            for n in STATE:
                tally.add(arm, n, got[n], fx[tag + n], label)
            for n in PROFILES:
                rows = got[n].shape[0]
                tally.add(arm, n, got[n], fx[tag + n][:rows], label)
            for n in COLUMN:
                tally.add(arm, n, got[n][0], fx[tag + n], label)
            if arm == "bep_chain":
                chain = {n: got[n].copy() for n in STATE}


def grade_couple(tally: Tally, fx, params, tset: str):
    cp = _cp()
    from gpuwm.core import urban_bep

    nz, ncol = int(fx["meta_nz"][0]), int(fx["meta_ncol"][0])
    nstep = int(fx["meta_nstep"][0])

    def lev(a, rows):
        return cp.asarray(np.ascontiguousarray(a[:rows].reshape(rows, 1, ncol)))

    def sfc(a, dtype=np.float32):
        return cp.asarray(np.ascontiguousarray(a.reshape(1, ncol), dtype=dtype))

    for lsm in ("noah", "noahmp"):
        for step in range(1, nstep + 1):
            tag = f"{lsm}_s{step}_"
            label = f"{tset} {lsm} step {step}"
            state = SimpleNamespace(
                frc_urb2d=sfc(fx["frc_urb2d"]),
                utype_urb2d=sfc(fx["utype_urb2d"], np.int32),
                lp_urb2d=sfc(fx["lp_urb2d"]), lb_urb2d=sfc(fx["lb_urb2d"]),
                hgt_urb2d=sfc(fx["hgt_urb2d"]),
                hi_urb2d=lev(fx["hi_urb2d"], fx["hi_urb2d"].shape[0]),
                pbl_terms={n: lev(fx[tag + "pre_" + n],
                                  nz + 1 if n == "sf_bep" else nz)
                           for n in COUPLE_PBL},
                **{n: cp.full((1, ncol), -1.0, dtype=cp.float32)
                   for n in COUPLE_DIAG})
            for n in STATE:
                a = fx[f"s{step}_in_" + n]
                setattr(state, n, lev(a, a.shape[0]))
            fields = {n: sfc(fx["lsm_" + n]) for n in COUPLE_FIELDS if n != "lh"}
            fields["lh"] = sfc(fx["lsm_qfx"] * np.float32(2.5e6))
            fields["swdown"], fields["glw"] = sfc(fx["swdown"]), sfc(fx["glw"])
            atmosphere = {"dz": lev(fx["dz8w"], nz), "u": lev(fx["u_phy"], nz),
                          "v": lev(fx["v_phy"], nz),
                          "theta": lev(fx["th_phy"], nz),
                          "rho": lev(fx["rho"], nz),
                          "pressure": lev(fx["p_phy"], nz)}
            solar = SimpleNamespace(coszen=sfc(fx["cosz_urb2d"]),
                                    hrang=sfc(fx["omg_urb2d"]),
                                    declin=float(fx["meta_declin_urb"][0]))
            urban_bep.after_lsm(state, params, lsm=2 if lsm == "noah" else 4,
                                fields=fields, atmosphere=atmosphere,
                                dt=float(fx["meta_dt"][0]), itimestep=step,
                                solar=solar)
            u1, v1 = fx["u_phy"][0], fx["v_phy"][0]
            calm1 = (u1 * u1 + v1 * v1) == np.float32(0)    # spd == 0
            for n in COUPLE_PBL:
                rows = nz + 1 if n == "sf_bep" else nz
                calm = np.zeros((rows, ncol), dtype=bool)
                if n in ("a_u_bep", "a_v_bep"):
                    calm[0] = calm1
                tally.add("couple", n,
                          cp.asnumpy(state.pbl_terms[n]).reshape(rows, ncol),
                          fx[tag + "out_" + n][:rows], label, calm=calm)
            for n in COUPLE_FIELDS:
                tally.add("couple", n, cp.asnumpy(fields[n]),
                          fx[tag + "out_" + n], label,
                          calm=calm1 if n == "ust" else None)
            for n in COUPLE_DIAG:
                tally.add("couple", n, cp.asnumpy(getattr(state, n)),
                          fx[tag + "out_" + n], label)
            for n in STATE:
                want = fx[f"s{step}_out_" + n]
                tally.add("couple", n, cp.asnumpy(getattr(state, n)), want, label)
            for n in BEP_OUT:
                tally.add("couple", n, cp.asnumpy(state.bep_out[n]),
                          fx[tag + "bep_" + n], label)


def grade_sfcdiag(tally: Tally, fx, tset: str):
    cp = _cp()
    from gpuwm.core.urban_bep_couple import launch_bep_sfcdiag

    nz, ncol = int(fx["meta_nz"][0]), int(fx["meta_ncol"][0])

    def lev(a):
        return cp.asarray(np.ascontiguousarray(a[:nz].reshape(nz, 1, ncol)))

    out = {n: cp.full((1, ncol), -1.0, dtype=cp.float32)
           for n in ("t2", "th2", "q2", "u10", "v10")}
    launch_bep_sfcdiag(
        utype_urb2d=cp.asarray(fx["utype_urb2d"].reshape(1, ncol)),
        th_phy=lev(fx["th_phy"]), qv=lev(fx["sfcdiag_qv"]),
        u_phy=lev(fx["u_phy"]), v_phy=lev(fx["v_phy"]),
        psfc=cp.asarray(fx["sfcdiag_psfc"].reshape(1, ncol)), **out)
    for n, arr in out.items():
        tally.add("sfcdiag", n, cp.asnumpy(arr), fx["sfcdiag_" + n], tset)


def column_census(root: Path) -> dict:
    out = {}
    for tset in ("nlcd", "lcz"):
        fx = load(root / f"bep461_{tset}" / "steps")
        frc, reg = fx["frc_urb2d"], fx["regime"]
        out[tset] = {"columns": int(frc.size),
                     "urban_columns": int((frc > 0).sum()),
                     "regimes": sorted(set(int(r) for r in reg[frc > 0])),
                     "classes": sorted(set(int(u) for u in fx["utype_urb2d"][frc > 0])),
                     "steps": int(fx["meta_nstep"][0]),
                     "dt": float(fx["meta_dt"][0]),
                     "declin_urb": float(fx["meta_declin_urb"][0])}
    return out


def run(root: Path) -> dict:
    from gpuwm import wrf_exact

    tally = Tally()
    for tset in ("nlcd", "lcz"):
        fx = load(root / f"bep461_{tset}" / "steps")
        grade_bep(tally, fx, tset)
        cfx = load(root / f"bep461_couple_{tset}" / "steps")
        grade_couple(tally, cfx, table_params(fx), tset)
        grade_sfcdiag(tally, cfx, tset)
    summary = tally.summary()
    worst = max(r["max_ulp"] for arm in summary.values() for r in arm.values())
    calm = sum(r.get("calm_wrf_nonfinite", 0) for arm in summary.values()
               for r in arm.values())
    calm_bad = sum(r.get("calm_woof_nonfinite", 0) for arm in summary.values()
                   for r in arm.values())
    differ = sum(r["bits_differ"] for arm in summary.values() for r in arm.values())
    words = sum(r["words"] for arm in summary.values() for r in arm.values())
    return {"gpuwm_wrf_exact": bool(wrf_exact.ENABLED),
            "census": column_census(root),
            "words": words, "bits_differ": differ, "max_ulp": worst,
            "calm_wrf_nonfinite_words": calm,
            "calm_woof_nonfinite_words": calm_bad,
            "arms": summary,
            "first_differences": {f"{a}/{f}": v[:4]
                                  for (a, f), v in tally.where.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    result = run(args.root)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    mode = "STRICT (GPUWM_WRF_EXACT=1)" if result["gpuwm_wrf_exact"] else "DEFAULT"
    print(f"{mode}: {result['words']} words graded, {result['bits_differ']} differ, "
          f"max ULP {result['max_ulp']}; calm-column words WRF left "
          f"non-finite {result['calm_wrf_nonfinite_words']} (WOOF non-finite "
          f"there: {result['calm_woof_nonfinite_words']})")
    for arm, fields in result["arms"].items():
        bad = {f: (r["bits_differ"], r["max_ulp"], r["max_abs"])
               for f, r in fields.items() if r["bits_differ"]}
        print(f"  {arm}: {sum(r['words'] for r in fields.values())} words; "
              f"differing fields {bad if bad else 'none'}")
    return 0 if (result["bits_differ"] == 0
                 and result["calm_woof_nonfinite_words"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
