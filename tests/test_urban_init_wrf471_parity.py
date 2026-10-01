"""The urban cold start is WRF v4.7.1's ``urban_var_init``, bit for bit.

Reference: ``urban_var_init`` itself, compiled at -O0 from the pinned tree
and run over one row of columns by ``tools/urban_wrf471_oracle/run_init.F90``
for all six ``(sf_urban_physics, use_wudapt_lcz)`` pairs, after the same
``urban_param_init`` call module_physics_init.F makes.  The row carries every
urban category of both legends, NATURAL, grassland and water, and input
urban fractions inside (0, 1], at zero, negative, above one, and on a
non-urban column -- the arms of the UTYPE mapping and the FRC_URB2D table
fallback.

Every WRF inout was primed with a -7 sentinel, so an array WRF leaves alone
is visibly left alone; gpuwm then has to leave it at its allocation zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.urban_state import (option_spec, resolve_dimensions,
                                    urban_var_init_host, PBL_TERM_NAMES)
from gpuwm.core.urban_tables import load_urban_params, urban_category_set
from gpuwm.verify.urban_oracle import ORACLE_ROOT, load, ulp_table

INIT = ORACLE_ROOT / "infra" / "init"
CASES = [(opt, lcz) for opt in (1, 2, 3) for lcz in (0, 1)]
SENTINEL = np.float32(-7.0)


@pytest.fixture(scope="module")
def oracle():
    return load(INIT)


def _to_gpuwm(array: np.ndarray) -> np.ndarray:
    """WRF (i, [k,] j) with j = 1 -> gpuwm ([k,] j, i)."""
    if array.ndim == 2:
        return array.T
    return array.transpose(1, 2, 0)


def _port(ref, opt, lcz):
    params = load_urban_params(opt, lcz)
    cats = urban_category_set(isurban=13)
    dims = resolve_dimensions(opt)
    nz = ref["a_u_bep"].shape[1] - 1
    host = urban_var_init_host(
        option=opt, use_wudapt_lcz=lcz, params=params, categories=cats,
        ivgtyp=_to_gpuwm(ref["ivgtyp"]), tsk=_to_gpuwm(ref["tsk_in"]),
        tslb=_to_gpuwm(ref["tslb_in"]), tmn=_to_gpuwm(ref["tmn_in"]),
        smois=_to_gpuwm(ref["smois_in"]), frc_urb2d=_to_gpuwm(ref["frc_in"]),
        num_urban_hi=15, nz=nz, dims=dims, spec=option_spec(opt))
    return host, dims


@pytest.mark.parametrize("opt,lcz", CASES)
def test_the_urban_dimensions_are_wrfs(oracle, opt, lcz):
    ref = oracle[f"opt{opt}_lcz{lcz}"]
    (ndm, nz, ng, nwr, nf, ngb, nbui, ngr, zrd, zwd, gd, zd, zdf, bd, wd,
     gbd, fbd, zgrd) = (int(v) for v in ref["dims"])
    if opt == 1:
        assert resolve_dimensions(1) == {}
        return
    dims = resolve_dimensions(opt)
    assert (dims["ndm"], dims["nz_um"], dims["ng_u"], dims["nwr_u"]) == (
        ndm, nz, ng, nwr)
    assert (dims["nf_u"], dims["ngb_u"], dims["nbui_max"], dims["ngr_u"]) == (
        nf, ngb, nbui, ngr)
    assert {k: dims[f"urban_map_{k}"] for k in (
        "zrd", "zwd", "gd", "zd", "zdf", "bd", "wd", "gbd", "fbd",
        "zgrd")} == dict(zrd=zrd, zwd=zwd, gd=gd, zd=zd, zdf=zdf, bd=bd,
                         wd=wd, gbd=gbd, fbd=fbd, zgrd=zgrd)


@pytest.mark.parametrize("opt,lcz", CASES)
def test_urban_var_init_is_wrfs_bit_for_bit(oracle, opt, lcz):
    ref = oracle[f"opt{opt}_lcz{lcz}"]
    host, _ = _port(ref, opt, lcz)
    compared = []
    for name, got in host.items():
        if name not in ref:
            # A gpuwm-only array (tsk_rural, z0/zd/lf_urb2d_s): the input
            # has no gridded morphology, so these hold the allocation zero,
            # which is what WRF's Registry allocation holds.
            assert name in ("tsk_rural", "tsk_rural_bep", "z0_urb2d",
                            "zd_urb2d", "lf_urb2d_s", "uc_urb2d",
                            "psim_urb2d", "psih_urb2d", "gz1oz0_urb2d",
                            "u10_urb2d", "v10_urb2d", "th2_urb2d",
                            "q2_urb2d", "ust_urb2d", "akms_urb2d"), name
            assert not np.any(got), name
            continue
        want = _to_gpuwm(ref[name])
        if name in PBL_TERM_NAMES:
            want = want[: got.shape[0]]
        untouched = want == SENTINEL
        if np.all(untouched):
            assert not np.any(got), f"{name}: WRF leaves it, gpuwm wrote it"
            continue
        assert not np.any(untouched), f"{name}: WRF wrote only part of it"
        table = ulp_table(got, want)
        assert table["max_ulp"] == 0, (name, table)
        compared.append(name)
    # The comparison must have reached the arms that matter for the option.
    must = {"frc_urb2d", "utype_urb2d", "ts_urb2d", "sh_urb2d", "hi_urb2d"
            if opt > 1 else "lf_urb2d"}
    if opt == 1:
        must |= {"trl_urb3d", "tgrl_urb3d", "smr_urb3d", "tgl_urb3d",
                 "qc_urb2d", "tgr_urb2d"}
    else:
        must |= {"trb_urb4d", "tw1_urb4d", "tgb_urb4d", "sf_bep", "vl_bep"}
    if opt == 3:
        must |= {"trv_urb4d", "qr_urb4d", "qlev_urb3d", "tflev_urb3d",
                 "tgr_urb3d"}
    assert must <= set(compared), must - set(compared)


def test_the_fraction_takes_every_arm(oracle):
    ref = oracle["opt1_lcz0"]
    host, _ = _port(ref, 1, 0)
    frc_in = _to_gpuwm(ref["frc_in"])[0]
    frc = host["frc_urb2d"][0]
    utype = host["utype_urb2d"][0]
    table = load_urban_params(1, 0).FRC_URB_TBL
    kept = (utype > 0) & (frc_in > 0) & (frc_in <= 1)
    from_table = (utype > 0) & ~kept
    assert kept.any() and from_table.any() and (utype == 0).any()
    assert np.array_equal(frc[kept], frc_in[kept])
    assert np.array_equal(frc[from_table], table[utype[from_table] - 1])
    assert np.all(frc[utype == 0] == 0)


def test_wrfs_legend_mismatch_fatals_are_kept():
    cats = urban_category_set(isurban=13)
    params0 = load_urban_params(1, 0)
    params1 = load_urban_params(1, 1)
    one = np.ones((1, 2), dtype=np.float32)
    soil = np.ones((4, 1, 2), dtype=np.float32)
    kw = dict(tsk=one * 290, tslb=soil * 285, tmn=one * 280, smois=soil * 0.3)
    with pytest.raises(ValueError, match="WITHOUT URBPARM_LCZ"):
        urban_var_init_host(option=1, use_wudapt_lcz=0, params=params0,
                            categories=cats,
                            ivgtyp=np.array([[55, 10]]), **kw)
    with pytest.raises(ValueError, match="OLD 3 URBAN CLASSES"):
        urban_var_init_host(option=1, use_wudapt_lcz=1, params=params1,
                            categories=cats,
                            ivgtyp=np.array([[10, 10]]), **kw)
