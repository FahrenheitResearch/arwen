"""BEP (sf_urban_physics = 2) column kernel against WRF v4.7.1.

``tools/urban_wrf471_oracle/run_bep.F90`` calls ``SUBROUTINE BEP`` from the
byte-unmodified ``phys/module_sf_bep.F`` exactly as
``module_sf_noahdrv.F:1603-1631`` does, over two table sets --
``URBPARM.TBL`` (use_wudapt_lcz = 0: three classes) and ``URBPARM_LCZ.TBL``
(= 1: eleven LCZ classes) -- with every class in several regimes (day,
night, low sun, near calm, gridded morphology with a building-height
histogram, dawn; urban fraction 0.02 .. 1, and a zero-fraction column BEP
must skip), for four consecutive steps.  Every input BEP reads and every
word it writes is recorded per step.

This module runs :func:`gpuwm.core.urban_bep.launch_bep_columns` from each
step's recorded inputs (so each step is graded on its own, not through the
chain) and requires every word -- the wall/roof/road layer temperatures, the
surface fluxes, the fourteen PBL source-term profiles and the four column
diagnostics -- to be identical.  The table words fed to the port are the
ones ``urban_param_init`` left in module_sf_urban, dumped by the oracle, so
the gate grades BEP and not the table reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance

ORACLE = Path(__file__).resolve().parents[1] / "gpuwm" / "data" / "urban" \
    / "oracle" / "bep"

STATE = ("trb_urb4d", "tw1_urb4d", "tw2_urb4d", "tgb_urb4d", "sfw1_urb3d",
         "sfw2_urb3d", "sfr_urb3d", "sfg_urb3d")
PROFILES = ("a_u", "a_v", "a_t", "a_e", "b_u", "b_v", "b_t", "b_e", "b_q",
            "dlg", "dl_u", "vl")
COLUMN = ("rl_up", "rs_abs", "emiss", "grdflx_urb")
SENTINEL = np.float32(-777.25)          # run_bep.F90's untouched marker

#: Per-output max ULP where the kernel is not (yet) word-identical, with the
#: measured cause beside each; {} = every word identical.
KNOWN_MAX_ULP: dict[str, int] = {}


def load_bin_fixture(directory: Path) -> dict[str, np.ndarray]:
    """One oracle case through gpuwm.verify.urban_oracle.load_case, with each
    array transposed from Fortran index order to C order (dims reversed, so a
    Fortran ``(ncol, nz)`` array comes back ``(nz, ncol)``) and scalars as
    one-element arrays."""
    from gpuwm.verify.urban_oracle import load_case

    return {name: (np.ascontiguousarray(np.asarray(value).T)
                   if np.ndim(value) else np.atleast_1d(value))
            for name, value in load_case(directory).items()}


def table_params(fx) -> dict:
    """The module_sf_urban words, in pack_bep_table's Fortran shapes."""
    params = {"ICATE": int(fx["tbl_ICATE"][0])}
    for key, value in fx.items():
        if key.startswith("tbl_") and key != "tbl_ICATE":
            # loader returns C order with Fortran dims reversed; rank-two
            # tables go back to (MAXDIRS|MAXHGTS, ICATE)
            params[key[4:]] = value.T if value.ndim == 2 else value
    return params


def run_step(fx, step: int):
    import cupy as cp

    from gpuwm.core.urban_bep import launch_bep_columns

    nz, ncol = int(fx["meta_nz"][0]), int(fx["meta_ncol"][0])

    def lev(a, rows):
        return cp.asarray(np.ascontiguousarray(a[:rows].reshape(rows, 1, ncol)))

    def sfc(a, dtype=np.float32):
        return cp.asarray(np.ascontiguousarray(a.reshape(1, ncol), dtype=dtype))

    tag = f"s{step}_in_"
    state = {n: lev(fx[tag + n], fx[tag + n].shape[0]) for n in STATE}
    outputs = {n: cp.full((nz, 1, ncol), SENTINEL, dtype=cp.float32)
               for n in PROFILES}
    outputs["b_q"] = cp.zeros((nz, 1, ncol), dtype=cp.float32)   # :1611
    outputs["sf"] = cp.full((nz + 1, 1, ncol), SENTINEL, dtype=cp.float32)
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


def measure(fx, step: int) -> tuple[dict, dict]:
    nz = int(fx["meta_nz"][0])
    got = run_step(fx, step)
    tag = f"s{step}_out_"
    ulp, flagged = {}, {}
    for n in STATE:
        ulp[n] = int(fp32_ulp_distance(np.ascontiguousarray(got[n]),
                                       fx[tag + n]).max())
    for n in PROFILES:
        ulp[n] = int(fp32_ulp_distance(np.ascontiguousarray(got[n][:nz]),
                                       np.ascontiguousarray(fx[tag + n][:nz])).max())
    ulp["sf"] = int(fp32_ulp_distance(np.ascontiguousarray(got["sf"]),
                                      np.ascontiguousarray(fx[tag + "sf"])).max())
    for n in COLUMN:
        ulp[n] = int(fp32_ulp_distance(np.ascontiguousarray(got[n][0]),
                                       fx[tag + n]).max())
    flagged["error_flags"] = int(np.count_nonzero(got["error_flags"]))
    return ulp, flagged


def test_fixture_covers_the_arms():
    for tset, nclass in (("nlcd", 3), ("lcz", 11)):
        fx = load_bin_fixture(ORACLE / f"bep_{tset}" / "steps")
        frc, ut = fx["frc_urb2d"], fx["utype_urb2d"]
        assert set(ut[frc > 0].tolist()) == set(range(1, nclass + 1))
        assert (frc == 0).any() and (frc == 1).any()
        assert (fx["swdown"] == 0).any() and (fx["cosz_urb2d"] < 0).any()
        assert (fx["hgt_urb2d"] > 0).any() and (fx["hi_urb2d"] > 0).any()
        assert int(fx["tbl_ICATE"][0]) == nclass
        # the zero-fraction column is left alone by WRF: the gate below
        # checks the port leaves the same sentinel words
        skipped = fx["s1_out_a_u"][:, frc == 0]
        assert np.all(skipped == SENTINEL)


@requires_gpu
@pytest.mark.parametrize("tset", ["nlcd", "lcz"])
@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_bep_column_matches_wrf(tset, step):
    fx = load_bin_fixture(ORACLE / f"bep_{tset}" / "steps")
    ulp, flagged = measure(fx, step)
    print("measured", tset, step, ulp, flagged)
    assert flagged == {"error_flags": 0}
    expected = {k: KNOWN_MAX_ULP.get(k, 0) for k in ulp}
    assert ulp == expected


# ---------------------------------------------------------------------------
# The model-lane entry point, end to end: zeroing + BEP + coupling block
# (urban_bep.after_lsm) against run_bep_couple.F90, both LSM blocks, both
# steps, from each step's recorded inputs.
# ---------------------------------------------------------------------------

COUPLE_PBL = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
              "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
              "dlg_bep", "dl_u_bep", "sf_bep", "vl_bep")
COUPLE_FIELDS = ("ust", "tsk", "hfx", "qfx", "lh", "grdflx", "albedo",
                 "emiss")
COUPLE_DIAG = ("ts_urb2d", "sh_urb2d", "lh_urb2d", "g_urb2d", "rn_urb2d")


@requires_gpu
@pytest.mark.parametrize("tset", ["nlcd", "lcz"])
@pytest.mark.parametrize("lsm", ["noah", "noahmp"])
@pytest.mark.parametrize("step", [1, 2])
def test_after_lsm_end_to_end_matches_wrf(tset, lsm, step):
    from types import SimpleNamespace

    import cupy as cp

    from gpuwm.core import urban_bep

    fx = load_bin_fixture(ORACLE / f"bep_couple_{tset}" / "steps")
    params = table_params(load_bin_fixture(ORACLE / f"bep_{tset}" / "steps"))
    nz, ncol = int(fx["meta_nz"][0]), int(fx["meta_ncol"][0])

    def lev(a, rows):
        return cp.asarray(np.ascontiguousarray(a[:rows].reshape(rows, 1, ncol)))

    def sfc(a, dtype=np.float32):
        return cp.asarray(np.ascontiguousarray(a.reshape(1, ncol), dtype=dtype))

    tag = f"{lsm}_s{step}_"
    state = SimpleNamespace(
        frc_urb2d=sfc(fx["frc_urb2d"]),
        utype_urb2d=sfc(fx["utype_urb2d"], np.int32),
        lp_urb2d=sfc(fx["lp_urb2d"]), lb_urb2d=sfc(fx["lb_urb2d"]),
        hgt_urb2d=sfc(fx["hgt_urb2d"]),
        hi_urb2d=lev(fx["hi_urb2d"], fx["hi_urb2d"].shape[0]),
        pbl_terms={n: lev(fx[tag + "pre_" + n], nz + 1 if n == "sf_bep" else nz)
                   for n in COUPLE_PBL},
        **{n: cp.full((1, ncol), -1.0, dtype=cp.float32) for n in COUPLE_DIAG})
    for n in STATE:
        a = fx[f"s{step}_in_" + n]
        setattr(state, n, lev(a, a.shape[0]))
    fields = {n: sfc(fx["lsm_" + n]) for n in COUPLE_FIELDS if n != "lh"}
    fields["lh"] = sfc(fx["lsm_qfx"] * np.float32(2.5e6))
    fields["swdown"], fields["glw"] = sfc(fx["swdown"]), sfc(fx["glw"])
    atmosphere = {"dz": lev(fx["dz8w"], nz), "u": lev(fx["u_phy"], nz),
                  "v": lev(fx["v_phy"], nz), "theta": lev(fx["th_phy"], nz),
                  "rho": lev(fx["rho"], nz), "pressure": lev(fx["p_phy"], nz)}
    solar = SimpleNamespace(coszen=sfc(fx["cosz_urb2d"]),
                            hrang=sfc(fx["omg_urb2d"]),
                            declin=float(fx["meta_declin_urb"][0]))
    urban_bep.after_lsm(state, params, lsm=2 if lsm == "noah" else 4,
                        fields=fields, atmosphere=atmosphere,
                        dt=float(fx["meta_dt"][0]), itimestep=step,
                        solar=solar)
    bad = []

    def same(got, want):
        return np.array_equal(np.ascontiguousarray(got).view(np.uint32),
                              np.ascontiguousarray(want).view(np.uint32))

    for n in COUPLE_PBL:
        rows = nz + 1 if n == "sf_bep" else nz
        if not same(cp.asnumpy(state.pbl_terms[n]).reshape(rows, ncol),
                    fx[tag + "out_" + n][:rows]):
            bad.append(n)
    for n in COUPLE_FIELDS:
        if not same(cp.asnumpy(fields[n]).reshape(-1), fx[tag + "out_" + n]):
            bad.append(n)
    for n in COUPLE_DIAG:
        if not same(cp.asnumpy(getattr(state, n)).reshape(-1),
                    fx[tag + "out_" + n]):
            bad.append(n)
    for n in STATE:
        want = fx[f"s{step}_out_" + n]
        if not same(cp.asnumpy(getattr(state, n)).reshape(want.shape), want):
            bad.append(n)
    for n in ("rl_up_urb", "rs_abs_urb", "emiss_urb", "grdflx_urb"):
        if not same(cp.asnumpy(state.bep_out[n]).reshape(-1),
                    fx[tag + "bep_" + n]):
            bad.append(n)
    assert not bad, f"{tset} {lsm} step {step}: not bitwise WRF: {bad}"


# ---------------------------------------------------------------------------
# In the real physics driver: option 2 wired through infra's UrbanCoupler,
# YSU's flag_bep arm and MYJURB.  Not a parity gate (the column gates above
# are); this proves the entry points meet the driver's own call shapes and
# that a city column comes out different from its rural neighbour.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("pbl", ["ysu", "myj"])
def test_option2_runs_in_the_physics_driver(pbl):
    import sys

    import cupy as cp

    sys.path.insert(0, str(Path(__file__).parent))
    from test_urban_default_off_identity import _small_driver

    extra = ({} if pbl == "ysu"
             else {"bl_pbl_physics": 2, "sf_sfclay_physics": 2})
    state, cfg, driver = _small_driver(sf_urban_physics=2, **extra)
    urban = driver.urban
    assert urban is not None and urban.option == 2
    assert urban.pbl_terms is not None
    for _ in range(3):
        driver.compute(state, cfg)
    f = driver.fields
    city = cp.asnumpy(f["utype_urb2d"]) > 0
    assert city[:, :2].all() and not city[:, 2:].any()
    for name in ("tsk", "hfx", "qfx", "ust", "t2", "q2", "u10", "v10",
                 "ts_urb2d", "sh_urb2d", "rn_urb2d"):
        assert np.isfinite(cp.asnumpy(f[name])).all(), name
    for name, arr in urban.pbl_terms.items():
        assert np.isfinite(cp.asnumpy(arr)).all(), name
    # the whole canopy sits in this coarse grid's first level: BEP's drag and
    # the rural drag both land there, and the buildings take volume only on
    # the city
    a_u = cp.asnumpy(urban.pbl_terms["a_u_bep"])
    assert (a_u[0] < 0).all() and (a_u[1:] == 0).all()
    vl = cp.asnumpy(urban.pbl_terms["vl_bep"])
    assert (vl[0][city] < 1).all() and (vl[0][~city] == 1).all()
    # the skin temperature and heat flux of a city column are the urban blend
    tsk, hfx = cp.asnumpy(f["tsk"]), cp.asnumpy(f["hfx"])
    assert not np.array_equal(tsk[city], tsk[~city][: city.sum()])
    assert (cp.asnumpy(f["sh_urb2d"])[city] != 0).all()
    assert (cp.asnumpy(f["sh_urb2d"])[~city] == 0).all()
    # module_surface_driver.F:3028-3032: city T2/TH2 are level-1 theta
    th1 = cp.asnumpy(state.total_theta())[0]
    assert np.array_equal(cp.asnumpy(f["th2"])[city], th1[city])
