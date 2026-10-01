"""MYJURB (MYJ under BEP/BEP+BEM) against WRF v4.7.1, word for word.

``module_pbl_driver.F:1462-1492`` calls ``MYJURB`` instead of ``MYJPBL``
whenever ``sf_urban_physics`` is 2 or 3 (``idiff`` is hard-wired to 0 at
:997).  ``tools/urban_wrf471_oracle/run_myjurb.F90`` drives the
byte-unmodified ``module_bl_myjurb.F`` over 18 columns for two consecutive
steps (TKE, THZ0, QZ0, QSFC and EXCH_H/M carried as WRF carries them);
this module runs :func:`gpuwm.core.myjurb.launch_myjurb` from each step's
recorded inputs and compares every word.

Unlike the engine's MYJPBL port, MYJURB seeds the interface heights with the
terrain height (``ZINT(KTE+1) = HT``, :283) exactly as WRF does, so columns
at HT = 350 m are in the fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance

FIXTURE = (Path(__file__).resolve().parents[1] / "gpuwm" / "data" / "urban"
           / "oracle" / "bep" / "myjurb"
           / "columns")

BEP = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep", "b_u_bep",
       "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep", "dlg_bep", "dl_u_bep",
       "vl_bep", "sf_bep")
COLUMN_OUT = {"rublten": "rublten", "rvblten": "rvblten",
              "rthblten": "rthblten", "rqvblten": "rqvblten",
              "rqcblten": "rqcblten", "el_myj": "el", "exch_h": "exch_h",
              "exch_m": "exch_m"}
SURFACE_OUT = ("pblh",)
STATE = ("thz0", "qz0", "qsfc", "ct")

#: Every word is required to be identical except the lanes where WRF wrote
#: a SUBNORMAL: CuPy appends -ftz=true unconditionally, so the kernel writes
#: exactly zero there (the class tests/test_ysu_wrf461_parity.py pins as
#: SUBNORMAL_LANES).  Measured on an RTX 4090 (NVRTC 13.4): three
#: rqcblten lanes per step, WRF 1.35e-42, i.e. a cloud-free level whose CWM
#: diffusion leaves a denormal residue.  Pinned as a count, and each lane is
#: checked to be exactly that case.
SUBNORMAL_LANES = {"rqcblten": 3}


def load_bin_fixture(directory: Path) -> dict[str, np.ndarray]:
    """One oracle case through gpuwm.verify.urban_oracle.load_case, with each
    array transposed from Fortran index order to C order (dims reversed, so a
    Fortran ``(ncol, nz)`` array comes back ``(nz, ncol)``) and scalars as
    one-element arrays."""
    from gpuwm.verify.urban_oracle import load_case

    return {name: (np.ascontiguousarray(np.asarray(value).T)
                   if np.ndim(value) else np.atleast_1d(value))
            for name, value in load_case(directory).items()}


def _run(fx, step):
    import cupy as cp

    from gpuwm.core.myjurb import launch_myjurb

    nz, ncol = int(fx["meta_nz"][0]), int(fx["meta_ncol"][0])

    def lev(a, n=nz):
        return cp.asarray(np.ascontiguousarray(a[:n].reshape(n, 1, ncol)))

    def sfc(a):
        return cp.asarray(np.ascontiguousarray(a.reshape(1, ncol)))

    tag = f"s{step}_in_"
    columns = {"dz": lev(fx["dz"]), "u": lev(fx["u"]), "v": lev(fx["v"]),
               "t": lev(fx["t"]), "th": lev(fx["th"]),
               "exner": lev(fx["exner"]), "qv": lev(fx["qv"]),
               "qc": lev(fx["cwm"]), "p": lev(fx["pmid"])}
    surface = {"psfc": sfc(fx["pint"][0]), "ust": sfc(fx["ustar"]),
               "tsk": sfc(fx["tsk"]), "chklowq": sfc(fx["chklowq"]),
               "xland": sfc(fx["xland"]), "sice": sfc(fx["sice"]),
               "snow": sfc(fx["snow"]), "akhs": sfc(fx["akhs"]),
               "akms": sfc(fx["akms"]), "elflx": sfc(fx["elflx"]),
               "uz0": sfc(fx[tag + "uz0"]), "vz0": sfc(fx[tag + "vz0"])}
    state = {"thz0": sfc(fx[tag + "thz0"]), "qz0": sfc(fx[tag + "qz0"]),
             "qsfc": sfc(fx[tag + "qsfc"]), "ct": sfc(fx["ct"])}
    tke = lev(fx[tag + "tke"])
    bep = {n: lev(fx[n], nz + 1 if n == "sf_bep" else nz) for n in BEP}
    outputs = {n: cp.zeros((nz, 1, ncol), dtype=cp.float32)
               for n in COLUMN_OUT}
    outputs["exch_h"] = lev(fx[tag + "exch_h"])
    outputs["exch_m"] = lev(fx[tag + "exch_m"])
    outputs.update(pblh=cp.zeros((1, ncol), dtype=cp.float32),
                   mixht=cp.zeros((1, ncol), dtype=cp.float32),
                   kpbl=cp.zeros((1, ncol), dtype=cp.int32))
    out = launch_myjurb(columns, surface, state, tke,
                        dtturbl=float(fx["meta_dt"][0]), bep=bep,
                        frc_urb2d=sfc(fx["frc_urb2d"]), ht=sfc(fx["ht"]),
                        outputs=outputs)
    got = {n: cp.asnumpy(out[n]).reshape(nz, ncol) for n in COLUMN_OUT}
    got["tke"] = cp.asnumpy(tke).reshape(nz, ncol)
    got["pblh"] = cp.asnumpy(out["pblh"]).reshape(ncol)
    got["kpbl"] = cp.asnumpy(out["kpbl"]).reshape(ncol)
    for n in STATE:
        got[n] = cp.asnumpy(state[n]).reshape(ncol)
    return got


def _measure(fx, step):
    nz = int(fx["meta_nz"][0])
    got = _run(fx, step)
    tag = f"s{step}_out_"
    ulp = {}
    subnormal = {}
    tiny = np.float32(np.finfo(np.float32).tiny)
    for n, w in COLUMN_OUT.items():
        want = np.ascontiguousarray(fx[tag + w][:nz])
        have = np.ascontiguousarray(got[n])
        flushed = (want != 0) & (np.abs(want) < tiny) & (have == 0)
        subnormal[n] = int(flushed.sum())
        d = fp32_ulp_distance(have, want)
        d[flushed] = 0
        ulp[n] = int(d.max())
    ulp["tke"] = int(fp32_ulp_distance(
        np.ascontiguousarray(got["tke"]),
        np.ascontiguousarray(fx[tag + "tke"][:nz])).max())
    for n in ("pblh",) + STATE:
        ulp[n] = int(fp32_ulp_distance(
            np.ascontiguousarray(got[n]),
            np.ascontiguousarray(fx[tag + n])).max())
    ulp["kpbl"] = int(np.abs(got["kpbl"] - fx[tag + "kpbl"]).max())
    return ulp, {k: v for k, v in subnormal.items() if v}


def test_fixture_reaches_the_arms():
    fx = load_bin_fixture(FIXTURE)
    assert (fx["ht"] > 0).any() and (fx["ht"] == 0).any()
    assert (fx["xland"] == 2).any() and (fx["snow"] > 0).any()
    assert (fx["cwm"] > 0).any()
    frc = fx["frc_urb2d"]
    assert (frc == 0).any() and (frc == 1).any() and ((frc > 0) & (frc < 1)).any()
    assert np.abs(fx["b_e_bep"]).max() > 0 and np.abs(fx["dl_u_bep"]).max() > 0


@requires_gpu
@pytest.mark.parametrize("step", [1, 2])
def test_myjurb_matches_wrf(step):
    fx = load_bin_fixture(FIXTURE)
    measured, flushed = _measure(fx, step)
    print("measured", step, measured, flushed)
    assert measured == {k: 0 for k in measured}, measured
    assert flushed == SUBNORMAL_LANES


@requires_gpu
def test_the_memory_check_prices_what_myjurb_allocates():
    """launch_myjurb's per-call bundle is MYJ's roster plus EXCH_M (and the
    published-zero RQIBLTEN MYJ's roster already carries); the fit gate
    prices exactly that under sf_urban_physics 2/3 with MYJ, and nothing
    extra without an urban model."""
    import dataclasses

    from gpuwm.config import RunConfig
    from gpuwm.core import preflight as pf
    from gpuwm.core.myjurb import (MYJURB_COLUMN_OUTPUTS,
                                   MYJURB_SURFACE_OUTPUTS)

    base = RunConfig(nx=8, ny=6, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0,
                     dt=10.0, run_seconds=60.0, sf_surface_physics=2,
                     bl_pbl_physics=2, sf_sfclay_physics=2)
    for option in (2, 3):
        cfg = dataclasses.replace(base, sf_urban_physics=option)
        priced = {k.split("/", 1)[1]
                  for k in pf.myj_output_transient_shapes(cfg)}
        assert priced == (set(MYJURB_COLUMN_OUTPUTS) | {"rqiblten"}
                          | set(MYJURB_SURFACE_OUTPUTS)), option
    plain = {k.split("/", 1)[1] for k in pf.myj_output_transient_shapes(base)}
    assert "exch_m" not in plain
