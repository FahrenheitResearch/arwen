"""The BEP surface coupling block against WRF v4.7.1, bit for bit.

``tools/urban_wrf471_oracle/run_bep_couple.F90`` runs, per step, the
zeroing, byte-unmodified ``BEP``, and then the coupling block INCLUDEd
verbatim from the pinned sources (``module_sf_noahdrv.F:1679-1776`` and
``module_sf_noahmpdrv.F:3363-3372 + 3689-3776``), for two steps with each
LSM's chain carrying its own coupled arrays forward.  This module feeds
:func:`gpuwm.core.urban_bep_couple.launch_bep_couple` the oracle's own
post-BEP words (so the gate grades the coupling block alone, independently
of the BEP column kernel) and requires every output word to be identical.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "data" / "oracles" / "urban" / "bep"
TOOLS = ROOT / "tools" / "urban_wrf471_oracle"

#: sha256 of the two INCLUDE files as cut from the pinned WRF v4.7.1 files
#: (PROVENANCE.md has the sed commands).  An edited copy would grade the
#: port against something that is not WRF.
INCLUDE_SHA256 = {
    "couple_noahdrv_1679_1776.inc":
        "16636ac58bebd3127dd8ea70d8ee227ad37d5d68788f3770a6c989f03d7e3f6c",
    "couple_noahmpdrv_3363_3372_3689_3776.inc":
        "18fedbabe18f63853ef4e0ea547823511c025b795d449f33b4e1132970e8aa0a",
    "sfcdiag_surface_driver_3028_3032.inc":
        "a3690623abcf14e26ef4b68950e9114577b0a91a130dd749dc7ace0075496e83",
}

PBL = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep", "b_u_bep",
       "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep", "sf_bep", "vl_bep")
FIELDS = ("ust", "tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "emiss")
DIAG = ("ts_urb2d", "sh_urb2d", "lh_urb2d", "g_urb2d", "rn_urb2d")


def load_bin_fixture(directory: Path) -> dict[str, np.ndarray]:
    """One oracle case through gpuwm.verify.urban_oracle.load_case, with each
    array transposed from Fortran index order to C order (dims reversed, so a
    Fortran ``(ncol, nz)`` array comes back ``(nz, ncol)``) and scalars as
    one-element arrays."""
    from gpuwm.verify.urban_oracle import load_case

    return {name: (np.ascontiguousarray(np.asarray(value).T)
                   if np.ndim(value) else np.atleast_1d(value))
            for name, value in load_case(directory).items()}


def test_include_files_are_the_pinned_wrf_lines():
    for name, digest in INCLUDE_SHA256.items():
        assert hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() \
            == digest, name


@pytest.mark.parametrize("tset", ["nlcd", "lcz"])
def test_fixture_reaches_both_frc_arms_and_both_lsm_blocks(tset):
    fx = load_bin_fixture(ORACLE / f"bep_couple_{tset}" / "steps")
    frc = fx["frc_urb2d"]
    assert (frc == 0).any() and (frc == 1).any() and ((frc > 0) & (frc < 1)).any()
    assert (fx["swdown"] == 0).any() and (fx["swdown"] > 0).any()
    # the two blocks differ (grdflx sign, lh_urb2d / frc): the fixture must
    # show it, or the lsm switch would be ungraded
    urban = frc > 0
    assert not np.array_equal(fx["noah_s1_out_grdflx"][urban],
                              fx["noahmp_s1_out_grdflx"][urban])


@requires_gpu
@pytest.mark.parametrize("tset", ["nlcd", "lcz"])
@pytest.mark.parametrize("lsm", ["noah", "noahmp"])
@pytest.mark.parametrize("step", [1, 2])
def test_coupling_block_is_bitwise_wrf(tset, lsm, step):
    import cupy as cp

    from gpuwm.core.urban_bep_couple import launch_bep_couple

    fx = load_bin_fixture(ORACLE / f"bep_couple_{tset}" / "steps")
    nz, ncol = int(fx["meta_nz"][0]), int(fx["meta_ncol"][0])
    tag = f"{lsm}_s{step}_"

    def lev(a):          # Fortran (ncol, nz+1) -> (nz, 1, ncol)
        return cp.asarray(np.ascontiguousarray(a[:nz].reshape(nz, 1, ncol)))

    def sfc(a):
        return cp.asarray(np.ascontiguousarray(a.reshape(1, ncol)))

    pbl = {}
    for n in PBL:
        full = fx[tag + "bep_" + n]
        if n == "sf_bep":
            pbl[n] = cp.asarray(np.ascontiguousarray(
                full.reshape(nz + 1, 1, ncol)))
        else:
            pbl[n] = lev(full)
    fields = {n: sfc(fx["lsm_" + n]) for n in FIELDS if n != "lh"}
    fields["lh"] = sfc(fx["lsm_qfx"] * np.float32(2.5e6))
    diag = {n: cp.full((1, ncol), -1.0, dtype=cp.float32) for n in DIAG}
    bep_out = {n: sfc(fx[tag + "bep_" + n])
               for n in ("rl_up_urb", "rs_abs_urb", "emiss_urb",
                         "grdflx_urb")}
    launch_bep_couple(
        lsm=2 if lsm == "noah" else 4, frc_urb2d=sfc(fx["frc_urb2d"]),
        dz8w=lev(fx["dz8w"]), rho=lev(fx["rho"]), u_phy=lev(fx["u_phy"]),
        v_phy=lev(fx["v_phy"]), glw=sfc(fx["glw"]),
        swdown=sfc(fx["swdown"]), bep_out=bep_out, pbl=pbl, fields=fields,
        diag=diag)
    bad = []
    for n in PBL:
        want = fx[tag + "out_" + n]
        got = cp.asnumpy(pbl[n]).reshape(-1, ncol)
        rows = nz + 1 if n == "sf_bep" else nz
        if not np.array_equal(got[:rows].view(np.uint32),
                              want[:rows].view(np.uint32)):
            bad.append(n)
    for n in FIELDS + DIAG:
        want = fx[tag + "out_" + n]
        got = cp.asnumpy((fields if n in FIELDS else diag)[n]).reshape(-1)
        if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
            bad.append(n)
    assert not bad, f"{tset} {lsm} step {step}: not bitwise WRF: {bad}"


@requires_gpu
@pytest.mark.parametrize("tset", ["nlcd", "lcz"])
def test_surface_diagnostics_override_is_bitwise_wrf(tset):
    """module_surface_driver.F:3028-3032 (== :3414-3418 for Noah-MP): on
    urban-category columns T2/TH2/Q2/U10/V10 come from level 1; elsewhere
    the words are untouched."""
    import cupy as cp

    from gpuwm.core.urban_bep_couple import launch_bep_sfcdiag

    fx = load_bin_fixture(ORACLE / f"bep_couple_{tset}" / "steps")
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
    assert (fx["utype_urb2d"] == 0).any() and (fx["utype_urb2d"] > 0).any()
    for n, arr in out.items():
        got = cp.asnumpy(arr).reshape(-1)
        want = fx["sfcdiag_" + n]
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), n
