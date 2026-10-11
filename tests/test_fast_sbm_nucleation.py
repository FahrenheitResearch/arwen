"""Droplet activation and the initial aerosol of the fast SBM host reference.

``gpuwm/core/fast_sbm/nucleation.py`` transcribes WRF v4.8.0's warm
``JERNUCL01_KS``, ``WATER_NUCLEATION``, ``Cloud_Base_Super`` and
``supmax_COEFF``, the cloud-base detection and flag of ``FAST_SBM`` and its
first-step aerosol fill.  These gates run it against the Fortran oracle in
``tests/data/fast_sbm_oracle`` (read only through
``tools/fast_sbm_wrf480_oracle/fixtures.py``) and against the published
physics.  What each one prevents:

* a transcription error in the activation loop, the Koehler radius, the
  cloud-base crossing or the partial split: the O64 sweep and KC-1 probe
  records disagree beyond relative 1e-10 (the negative control shows a
  one-part-per-million error in the solute density is caught);
* a port that departs from WRF's native arithmetic by more than float32
  storage: the O32 envelope and the native probe bound;
* the wrong cloud-base rule (finding D9) or flag: the probe records' flags;
* an initial aerosol on the wrong spectrum, scaling or units: the KC-1
  aerosol after the first call;
* a sink that loses or invents water or heat: the budget closure of the
  corrected preset, and D3 reproduced number for number under ``wrf_v480``;
* an ice process run by the warm reference: the named refusals.

Run: ``C:/Python313/python -m pytest -p no:cacheprovider
tests/test_fast_sbm_nucleation.py`` with ``PYTHONPATH`` the worktree root.
"""

from __future__ import annotations

import dataclasses
import math
import types
from functools import lru_cache

import numpy as np
import pytest

from gpuwm.core.fast_sbm import nucleation as N
from gpuwm.core.fast_sbm.constants import (
    AA1_MY,
    AA2_MY,
    AL1,
    BB1_MY,
    BB2_MY,
    COL,
    IONS,
    MWAERO,
    NKR,
    NKR_AEROSOL,
    RO_SOLUTE,
    FsbmBehavior,
    FsbmRefusal,
)
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from tools.fast_sbm_wrf480_oracle import fixtures as fx

WRF = FsbmBehavior.wrf_v480()
CORRECTED = FsbmBehavior.corrected()
#: Design 5.5, process sweeps and probe records.
REL = 1e-10
FLOOR = 1e-12
#: Design 5.5, native envelope.
ENVELOPE_FLOOR = 1e-9
#: Budget closure the item spec asks of one activation step.
BUDGET = 1e-13
#: The outputs of JERNUCL01_KS a record carries, host field -> record field.
OUTPUTS = (("psi1", "psi1"), ("fccnr", "fccnr"), ("tt", "tt"), ("qq", "qq"),
           ("sup1", "sup1"), ("sup2", "sup2"))


# ---------------------------------------------------------------- fixtures
@lru_cache(maxsize=None)
def _tables(build: str):
    t = fx.load_fixture(f"tables_{build}")
    return {
        "xl": t["read.xl"][0].astype(np.float64),
        "rccn": t["derived.rccn"][0].astype(np.float64),
        "fccnr_con": t["derived.fccnr_con"][0].astype(np.float64),
        "fccnr_mar": t["derived.fccnr_mar"][0].astype(np.float64),
    }


@lru_cache(maxsize=None)
def _sweep(build: str):
    return fx.load_fixture(f"sweep_nucleation_{build}")


def _sweep_record(build: str, i: int) -> dict:
    d = _sweep(build)
    return {key.split(".", 1)[1]: d[key][i] for key in d}


def _sweep_count(build: str) -> int:
    return int(_sweep(build)["case.tt_in"].shape[0])


def _supersaturations(tt: float, qq: float, pp: float) -> tuple[float, float]:
    """Water and ice supersaturation of a state, written here from WRF:
    ``OPER3(QQ, PP)`` (:3068) over ``POLYSVP(TT, 0)`` and ``POLYSVP(TT,
    1)`` (:1716-1742), minus one, in the operation order of :3170-3176, so a
    state and the supersaturations computed from it agree bit for bit."""

    ew = qq * pp / (0.622 + 0.378 * qq)
    es_water = AA1_MY * math.exp(-BB1_MY / tt)
    es_ice = AA2_MY * math.exp(-BB2_MY / tt)
    return ew / es_water - 1.0, ew / es_ice - 1.0


def _run_sweep_record(rec: dict, tabs: dict, behavior=WRF, **kw):
    return N.jernucl01_warm(
        rec["psi1_in"], rec["fccnr_in"], tabs["xl"], rec["tt_in"],
        rec["qq_in"], rec["ror"], rec["pp"], rec["sup1_in"], rec["sup2_in"],
        rec["win"], int(rec["is_cloud_base"]), tabs["rccn"],
        behavior=behavior, **kw)


@lru_cache(maxsize=None)
def _probe_pairs(build: str):
    recs = list(fx.iter_probe(f"kc1_probe_{build}",
                              types=["nucl_before", "nucl_after"]))
    before = [r for r in recs if r.type == "nucl_before"]
    after = [r for r in recs if r.type == "nucl_after"]
    assert len(before) == len(after) > 0
    for r0, r1 in zip(before, after):
        assert (int(r0["step"]), int(r0["i"]), int(r0["k"]), int(r0["j"])) == \
            (int(r1["step"]), int(r1["i"]), int(r1["k"]), int(r1["j"]))
    return tuple(zip(before, after))


def _run_probe(r0, tabs: dict, behavior=WRF):
    return N.jernucl01_warm(
        r0["ff1"], r0["fccn"], tabs["xl"], r0["tt"], r0["qq"], r0["rhocgs"],
        r0["pcgs"], r0["del1in"], r0["del2in"], r0["w_stag"],
        int(r0["is_cloud_base"]), tabs["rccn"], behavior=behavior)


def _close(host, ref, rel=REL, floor=FLOOR) -> np.ndarray:
    host = np.asarray(host, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    return np.abs(host - ref) <= rel * np.abs(ref) + floor * scale


def _sweep_gate_failures(build: str = "o64", **kw) -> list[tuple[int, str]]:
    tabs = _tables(build)
    failures = []
    for i in range(_sweep_count(build)):
        rec = _sweep_record(build, i)
        res = _run_sweep_record(rec, tabs, **kw)
        for host_name, rec_name in OUTPUTS:
            if not np.all(_close(getattr(res, host_name), rec[rec_name + "_out"])):
                failures.append((i, host_name))
        oracle_activated = not np.array_equal(rec["psi1_in"], rec["psi1_out"])
        if res.activated != oracle_activated:
            failures.append((i, "activated"))
        expected = "cloud_base" if int(rec["is_cloud_base"]) == 1 else "in_cloud"
        if res.branch != expected:
            failures.append((i, "branch"))
    return failures


@lru_cache(maxsize=None)
def _kc1_statics(build: str):
    ini = next(iter(fx.iter_records(f"kc1_summary_{build}", types=["initial"])))
    return {name: np.asarray(ini[name], dtype=np.float64)
            for name in ("dz8w", "rho", "p", "pii", "xland")}


@lru_cache(maxsize=None)
def _kc1_pre(build: str):
    return {step: pre for step, pre, _post in fx.iter_replay(f"kc1_replay_{build}")}


def _zcgs(dz8w_col: np.ndarray) -> np.ndarray:
    """``zcgs`` as FAST_SBM builds it (:4269-4276)."""

    z = np.empty(dz8w_col.shape[0])
    z_full = 0.0
    for k in range(dz8w_col.shape[0]):
        z[k] = z_full + 0.5 * float(dz8w_col[k]) * 100
        z_full = z_full + float(dz8w_col[k]) * 100.0
    return z


def _kc1_column_inputs(build: str, step: int, c: int):
    s = _kc1_statics(build)
    pre = _kc1_pre(build)[step]
    t_new = np.asarray(pre["th"][c], dtype=np.float64) * s["pii"][c]
    return (t_new, np.asarray(pre["qv"][c], dtype=np.float64),
            s["p"][c] * 10.0, np.asarray(pre["w"][c], dtype=np.float64),
            _zcgs(s["dz8w"][c]))


# ------------------------------------------------------------ oracle gates
def test_every_o64_sweep_record_matches_the_host():
    """270 JERNUCL01_KS warm records: 3 temperatures, 5 supersaturations,
    both branches, 3 updrafts, fresh continental, depleted continental with
    drops, fresh maritime; outputs within relative 1e-10, branch and
    activation exact."""

    assert _sweep_count("o64") == 270
    assert _sweep_gate_failures("o64") == []


def test_the_sweep_covers_what_it_gates():
    """A sweep that never split a bin or never reached the cloud-base branch
    would pass the gate above without testing those paths."""

    tabs = _tables("o64")
    seen = {"cloud_base": 0, "in_cloud": 0, "activated": 0, "idle": 0,
            "partial": 0, "whole": 0, "with_drops": 0, "regimes": set()}
    for i in range(_sweep_count("o64")):
        rec = _sweep_record("o64", i)
        res = _run_sweep_record(rec, tabs)
        seen[res.branch] += 1
        seen["activated" if res.activated else "idle"] += 1
        seen["regimes"].add(int(rec["aerosol"]))
        if np.any(rec["psi1_in"] > 0):
            seen["with_drops"] += 1
        for step in res.water.steps:
            seen[step.split] = seen.get(step.split, 0) + 1
    assert seen["cloud_base"] > 0 and seen["in_cloud"] > 0
    assert seen["activated"] > 0 and seen["idle"] > 0
    assert seen["partial"] > 0 and seen["whole"] > 0
    assert seen["with_drops"] > 0
    assert seen["regimes"] == {1, 2, 3}


def test_the_o32_sweep_sits_inside_the_native_envelope():
    """Host on the native build's own inputs and tables, against the native
    outputs: ``abs(host - O32) <= abs(O64 - O32) + 1e-9 scale``.

    Departure from design 5.5, recorded for the integration record.  The
    design's envelope takes O64x, the O64 build run on the O32 states.  The
    sweep fixtures carry no such cross-run, so ``O64`` here is the O64 build
    on its own sweep inputs, which differ from the O32 inputs (``sup1_in`` by
    up to 3.4e-4 relative).  The envelope is therefore wider in places and
    narrower in others than the design's.  To keep the comparison exact where
    it is narrower, the native build's float32 storage of PSI1 and FCCNR is
    applied to the host value before the comparison, and an element whose
    float64 result sits inside the envelope but rounds to the neighbouring
    float32 word also passes when it is at most one float32 ULP from the O32
    word (``gpuwm.core.fp32_ulp``).  Measured on the 270 records: 40
    elements fail the literal envelope without the rounding, 20 with it, and
    all 20 are within one float32 ULP.  The primary gates are the O64 sweep
    and probe records at relative 1e-10; this one is secondary.
    """

    tabs = _tables("o32")
    d32 = _sweep("o32")
    assert _sweep_count("o32") == _sweep_count("o64")
    failures = []
    for i in range(_sweep_count("o32")):
        rec = _sweep_record("o32", i)
        rec64 = _sweep_record("o64", i)
        assert int(rec["aerosol"]) == int(rec64["aerosol"])
        assert int(rec["is_cloud_base"]) == int(rec64["is_cloud_base"])
        res = _run_sweep_record(rec, tabs)
        for host_name, rec_name in OUTPUTS:
            raw = d32[f"case.{rec_name}_out"][i]
            stored_f4 = np.asarray(raw).dtype == np.float32
            o32 = np.asarray(raw, dtype=np.float64)
            o64 = np.asarray(rec64[rec_name + "_out"], dtype=np.float64)
            host = np.asarray(getattr(res, host_name), dtype=np.float64)
            if stored_f4:
                host = host.astype(np.float32).astype(np.float64)
            scale = max(float(np.max(np.abs(o64))), float(np.max(np.abs(o32))))
            ok = np.abs(host - o32) <= np.abs(o64 - o32) + ENVELOPE_FLOOR * scale
            if stored_f4:
                ok = ok | (fp32_ulp_distance(host.astype(np.float32),
                                             np.asarray(raw, dtype=np.float32)) <= 1)
            if not np.all(ok):
                failures.append((i, host_name))
        if res.activated != (not np.array_equal(rec["psi1_in"], rec["psi1_out"])):
            failures.append((i, "activated"))
    assert failures == []


def test_every_o64_nucleation_probe_record_matches_the_host():
    """KC-1's 49 recorded JERNUCL01_KS calls inside FAST_SBM, the one
    cloud-base activation (step 40, column c4, level 40) among them."""

    tabs = _tables("o64")
    pairs = _probe_pairs("o64")
    assert len(pairs) == 49
    failures = []
    branches = set()
    for r0, r1 in pairs:
        res = _run_probe(r0, tabs)
        for host_name, rec_name in (("psi1", "ff1"), ("fccnr", "fccn"),
                                    ("tt", "tt"), ("qq", "qq"),
                                    ("sup1", "del1in"), ("sup2", "del2in")):
            if not np.all(_close(getattr(res, host_name), r1[rec_name])):
                failures.append((int(r0["step"]), int(r0["i"]), int(r0["k"]),
                                 host_name))
        if res.activated != (not np.array_equal(r0["ff1"], r1["ff1"])):
            failures.append((int(r0["step"]), int(r0["i"]), int(r0["k"]),
                             "activated"))
        expected = ("cloud_base" if int(r0["is_cloud_base"]) == 1
                    else "in_cloud")
        if res.branch != expected:
            failures.append((int(r0["step"]), int(r0["i"]), int(r0["k"]),
                             "branch"))
        branches.add(res.branch)
    assert failures == []
    assert branches == {"cloud_base", "in_cloud"}


def test_the_o32_probe_records_are_within_two_float32_ulp():
    """The host on the native build's own KC-1 states lands within two
    float32 ULP of the native outputs: a branch the native build did not take
    at a state it met (a bin activated wholly instead of split, a bin it
    never reached) would be far outside."""

    tabs = _tables("o32")
    worst = 0
    for r0, r1 in _probe_pairs("o32"):
        res = _run_probe(r0, tabs)
        for host, ref in ((res.psi1, r1["ff1"]), (res.fccnr, r1["fccn"])):
            d = fp32_ulp_distance(np.asarray(host).astype(np.float32),
                                  np.asarray(ref, dtype=np.float32))
            worst = max(worst, int(np.max(d)))
        assert res.tt == float(r1["tt"]) and res.qq == float(r1["qq"])
    assert worst <= 2


@pytest.mark.parametrize("build", ["o64", "o32"])
def test_cloud_base_level_reproduces_the_probe_records(build):
    """The recorded ``KZ_Cloud_Base`` of every probed column and step is
    what ``cloud_base_level`` gives under WRF's rule on the replayed state;
    the intended rule (D9 corrected) does not reproduce it."""

    recorded: dict[tuple[int, int], set[int]] = {}
    for r0, _r1 in _probe_pairs(build):
        recorded.setdefault((int(r0["step"]), int(r0["i"])), set()).add(
            int(r0["kz_cloud_base"]))
    assert recorded
    differs = 0
    for (step, i), kzs in sorted(recorded.items()):
        assert len(kzs) == 1
        args = _kc1_column_inputs(build, step, i - 1)
        kz_wrf = N.cloud_base_level(*args, behavior=WRF)
        assert kz_wrf is not None and kz_wrf + 1 == next(iter(kzs)), (step, i)
        kz_first = N.cloud_base_level(*args, rule="first_level")
        if kz_first is None or kz_first + 1 != next(iter(kzs)):
            differs += 1
    assert differs > 0


@pytest.mark.parametrize("build", ["o64", "o32"])
def test_is_this_cloud_base_reproduces_the_probe_flags(build):
    xl = _tables(build)["xl"]
    flags = []
    for r0, _r1 in _probe_pairs(build):
        k = int(r0["k"]) - 1
        kz = int(r0["kz_cloud_base"]) - 1
        got = N.is_this_cloud_base(k, kz, np.asarray(r0["ff1"], dtype=np.float64), xl)
        assert got == (int(r0["is_cloud_base"]) == 1), (int(r0["step"]), int(r0["i"]), k)
        flags.append(got)
    assert any(flags) and not all(flags)


def test_initial_aerosol_column_is_the_kc1_aerosol_after_step_1():
    """The first call fills the aerosol and activates nothing (``th_old`` is
    set to ``th`` first, so the forcing is zero).  The replay's pre-call
    state of step 2 is that aerosol after one advection step; a uniform
    mixing ratio passes the upwind operator unchanged up to rounding, so the
    still column c3 and column c4 match exactly and c1, c2 to 1e-15."""

    tabs = _tables("o64")
    s = _kc1_statics("o64")
    pre = _kc1_pre("o64")[2]
    chem = np.asarray(pre["chem"], dtype=np.float64)
    exact = []
    for c in range(chem.shape[0]):
        rho_cgs = s["rho"][c] * 0.001
        fccn = N.initial_aerosol_column(tabs["fccnr_con"], tabs["fccnr_mar"],
                                        rho_cgs, s["xland"][c] == 1)
        assert fccn.shape == (rho_cgs.shape[0], NKR_AEROSOL)
        advected = fccn / rho_cgs[:, None] * 1000.0          # exit, :5297
        ref = chem[c][:, 99:142]                              # ff8i01..ff8i43
        np.testing.assert_allclose(advected, ref, rtol=1e-15, atol=0.0)
        exact.append(np.array_equal(advected, ref))
    assert exact[2] and exact[3]
    # land starts continental, water maritime (:4299-4305)
    assert s["xland"][0] == 1 and s["xland"][1] == 2
    assert not np.allclose(chem[0][:, 99:142], chem[1][:, 99:142])


# ---------------------------------------------------------- analytic checks
@pytest.mark.parametrize("tt", [263.15, 278.15, 290.15, 300.15, 313.15])
@pytest.mark.parametrize("s", [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1])
def test_the_in_cloud_critical_radius_is_koehlers(tt, s):
    """``r* = (A/3) (4/(B S^2))^(1/3)`` with ``A = 3.3e-5/T`` and ``B`` from
    the solute (Pruppacher and Klett 1997, chapter 6): the dry radius whose
    Koehler critical supersaturation ``(4 A^3 / (27 B r^3))^(1/2)`` is S."""

    a = 3.3e-5 / tt
    b = IONS * 4.3 / MWAERO * (4.0 / 3.0) * 3.141593 * RO_SOLUTE
    r = N.koehler_rcriti_in_cloud(tt, s)
    assert math.isclose(r, (4.0 * a**3 / (27.0 * b * s * s)) ** (1.0 / 3.0),
                        rel_tol=1e-14)
    assert math.isclose(math.sqrt(4.0 * a**3 / (27.0 * b * r**3)), s,
                        rel_tol=1e-13)
    # the radius WATER_NUCLEATION uses in cloud is this one
    tabs = _tables("o64")
    res = N.water_nucleation(tabs["fccnr_con"], np.zeros(NKR), tabs["xl"], tt,
                             0.01, 1.1e-3, s, 9.0e5, 0, 100.0, tabs["rccn"])
    for step in res.steps:
        assert step.rcriti == r
    if res.exit_rcriti is not None:
        assert res.exit_rcriti == r


#: Surface tension of water against air (dyn/cm) and water density (g/cm3)
#: at 0, 20 and 30 C, standard tabulated values.
_WATER_TABLE = ((273.15, 75.64, 0.99984), (293.15, 72.75, 0.99821),
                (303.15, 71.20, 0.99565))


@pytest.mark.parametrize("tt, sigma, rho_w", _WATER_TABLE)
def test_supmax_coeff_uses_the_same_solute_and_a_physical_curvature_term(
        tt, sigma, rho_w):
    """``BKOE`` is the in-cloud solute term exactly; ``AKOE`` is the Koehler
    curvature term ``2 sigma / (Rv rho_w T)`` within 0.5 percent of the
    tabulated surface tension and density; the fixed in-cloud ``3.3e-5/T`` is
    that term at 0 C to 1 percent; ``C3`` is positive and finite."""

    akoe, bkoe, c3 = N.supmax_coeff(9.0e5, tt)
    assert bkoe == IONS * 4.3 / MWAERO * (4.0 / 3.0) * 3.141593 * RO_SOLUTE
    assert abs(akoe / (2.0 * sigma / (461.5e4 * rho_w * tt)) - 1.0) < 0.005
    if tt == 273.15:
        assert abs(akoe / (3.3e-5 / tt) - 1.0) < 0.01
    assert math.isfinite(c3) and c3 > 0.0


@pytest.mark.parametrize("regime", ["fccnr_con", "fccnr_mar"])
def test_activated_number_does_not_fall_as_supersaturation_rises(regime):
    tabs = _tables("o64")
    prev = -1.0
    for s in np.geomspace(1e-5, 0.3, 1500):
        res = N.water_nucleation(tabs[regime], np.zeros(NKR), tabs["xl"],
                                 293.15, 0.01, 1.1e-3, float(s), 9.0e5, 0,
                                 100.0, tabs["rccn"])
        assert res.activated_number >= prev, s
        prev = res.activated_number
    assert prev > 0.0


@pytest.mark.parametrize("regime", ["fccnr_con", "fccnr_mar"])
def test_cloud_base_activation_does_not_fall_as_the_updraft_rises(regime):
    """Pinsky et al. (2012): the cloud-base droplet number grows with w."""

    tabs = _tables("o64")
    prev = -1.0
    for w in np.geomspace(1.0, 2000.0, 600):
        res = N.water_nucleation(tabs[regime], np.zeros(NKR), tabs["xl"],
                                 293.15, 0.01, 1.1e-3, 0.01, 9.0e5, 1,
                                 float(w), tabs["rccn"])
        assert res.branch == "cloud_base"
        assert res.activated_number >= prev, w
        prev = res.activated_number
    assert prev > 0.0


def test_the_partial_split_conserves_aerosol_number():
    """Aerosol number before equals aerosol number after plus activated drop
    number, for the whole call and for every whole or partial bin (:3286-3302),
    over the O64 sweep's inputs in both presets."""

    tabs = _tables("o64")
    partial = 0
    for i in range(_sweep_count("o64")):
        rec = _sweep_record("o64", i)
        for behavior in (WRF, CORRECTED):
            res = _run_sweep_record(rec, tabs, behavior=behavior)
            before = COL * np.asarray(rec["fccnr_in"], dtype=np.float64)
            after = COL * res.fccnr
            total_before = float(np.sum(before))
            assert math.isclose(total_before,
                                float(np.sum(after)) + res.water.activated_number,
                                rel_tol=1e-13)
            for step in res.water.steps:
                assert step.split in ("whole", "partial")
                b = before[step.imax - 1]
                a = after[step.imax - 1]
                assert math.isclose(b, a + step.ccnconc, rel_tol=1e-13,
                                    abs_tol=1e-15 * total_before)
                partial += step.split == "partial"
    assert partial > 0


def test_the_bin1_partial_split_is_out_of_physical_reach():
    """WRF splits aerosol bin 2 with bin 1's geometry when the critical
    radius falls there (:3293-3297), which does not conserve number.  Even at
    100 percent supersaturation and 40 C the critical radius lies above
    RCCN(5), so no physical state reaches that path."""

    rccn = _tables("o64")["rccn"]
    assert N.koehler_rcriti_in_cloud(313.15, 1.0) > rccn[4]


def test_activation_into_drop_bins_2_to_8():
    """Aerosol bins 37-43 feed drop bins 2-8 (``8-(NKR-IMAX)``, :3310) and
    bins up to 36 drop bin 1; the drop f-form gains ``N/(3 COL x)``."""

    tabs = _tables("o64")
    xl = tabs["xl"]
    fccnr = np.zeros(NKR_AEROSOL)
    fccnr[33:43] = 1.0                     # aerosol bins 34-43
    psi1 = np.zeros(NKR)
    res = N.water_nucleation(fccnr, psi1, xl, 290.15, 0.01, 1.1e-3, 0.05,
                             9.0e5, 0, 100.0, tabs["rccn"])
    # WRF walks on through the empty bins below 34 until the critical bin
    imaxes = [s.imax for s in res.steps]
    assert imaxes == list(range(43, 43 - len(imaxes), -1)) and len(imaxes) > 10
    for s in res.steps:
        assert s.drop_bin == (1 if s.imax <= 36 else 8 - (43 - s.imax))
        if s.imax >= 34:
            assert s.split == "whole" and s.ccnconc == COL
        else:
            assert s.ccnconc == 0.0
    expected = np.zeros(NKR)
    expected[0] = 3 * COL                  # bins 34, 35, 36
    expected[1:8] = COL                    # bins 37..43 -> drop bins 2..8
    np.testing.assert_allclose(res.dropconcn, expected, rtol=1e-15)
    np.testing.assert_allclose(res.psi1[:8], expected[:8] / (3.0 * COL * xl[:8]),
                               rtol=1e-15)
    assert np.all(res.psi1[8:] == 0.0)
    assert np.all(res.fccnr == 0.0)


def test_cloud_base_crossing_and_its_log_interpolation():
    """``NN`` is the first bin with ``PL <= PR`` (every earlier bin from 2 has
    ``PL > PR``), ``PL = supmax sqrt(N_act)`` with ``supmax`` the Koehler
    critical supersaturation of the bin's dry radius and ``N_act`` the
    aerosol number at and above it, ``PR = C3 w^(3/4)``, and the critical
    radius sits at the crossing's fraction of the bin in ln r."""

    tabs = _tables("o64")
    rccn = tabs["rccn"]
    cases = 0
    for regime in ("fccnr_con", "fccnr_mar"):
        for w in (10.0, 100.0, 500.0):
            for tt in (278.15, 290.15, 300.15):
                fccnr = tabs[regime]
                r = N.cloud_base_super(fccnr, rccn, tt, 9.0e5, w)
                akoe, bkoe, c3 = N.supmax_coeff(9.0e5, tt)
                assert math.isclose(r.pr, c3 * w**0.75, rel_tol=1e-15)
                nn = r.nn
                assert nn >= 3
                for nr in range(2, nn + 1):
                    n_act = COL * float(np.sum(fccnr[nr - 1:]))
                    smax = math.sqrt(4.0 * akoe**3 / (27.0 * bkoe * rccn[nr - 1]**3))
                    assert math.isclose(r.pl[nr - 1], smax * math.sqrt(n_act),
                                        rel_tol=1e-13)
                    if nr < nn:
                        assert r.pl[nr - 1] > r.pr
                assert r.pl[nn - 1] <= r.pr
                frac = (r.pl[nn - 2] - r.pr) / (r.pl[nn - 2] - r.pl[nn - 1])
                assert 0.0 < frac <= 1.0
                assert math.isclose(math.log(r.rcriti / rccn[nn - 2]),
                                    COL * frac, rel_tol=1e-12)
                assert N.cloud_base_rcriti(fccnr, rccn, tt, 9.0e5, w) == r.rcriti
                cases += 1
    assert cases == 18


@pytest.mark.parametrize("build", ["o64", "o32"])
def test_a_cloud_base_without_a_crossing_is_refused(build):
    """Refusal case 5: WRF stops with 'NN is not defined in cloud base
    routine, model stop' (:3548); the host raises FsbmRefusal on the same
    inputs, through JERNUCL01_KS as WRF reached it."""

    cases = [r for r in fx.iter_refusals(build=build) if r.case == 5]
    assert len(cases) == 1
    ref = cases[0]
    assert ref.exit_status == 99
    assert ref.message == "NN is not defined in cloud base routine, model stop"
    rec = ref.records[0]
    tabs = _tables(build)
    with pytest.raises(FsbmRefusal) as err:
        N.jernucl01_warm(np.zeros(NKR), rec["fccnr"], tabs["xl"], rec["tt"],
                         rec["qq"], rec["ror"], rec["pp"], rec["sup1"],
                         rec["sup2"], rec["win"], int(rec["is_cloud_base"]),
                         tabs["rccn"], behavior=WRF)
    assert "Cloud_Base_Super" in err.value.where
    assert "NN is not defined in cloud base routine" in err.value.condition


def test_a_downdraft_at_cloud_base_reaches_the_same_stop():
    """``PR = C3 w^(3/4)`` is NaN for w < 0 and no ``PL <= NaN`` holds."""

    tabs = _tables("o64")
    with pytest.raises(FsbmRefusal):
        N.cloud_base_super(tabs["fccnr_con"], tabs["rccn"], 290.15, 9.0e5, -5.0)


def test_the_warm_reference_refuses_the_ice_processes_by_name():
    tabs = _tables("o64")
    xl, rccn, con = tabs["xl"], tabs["rccn"], tabs["fccnr_con"]
    z = np.zeros(NKR)
    # homogeneous freezing transfer (:3101), drop branch at -38 C
    with pytest.raises(FsbmRefusal) as err:
        N.jernucl01_warm(z, con, xl, 273.15 - 38.0, 1e-4, 6e-4, 4e5, 0.01,
                         0.4, 100.0, 0, rccn)
    assert ":3101" in err.value.where and "milestone 3" in err.value.condition
    # crystal nucleation branch (:3117), ice supersaturated below 0 C
    with pytest.raises(FsbmRefusal) as err:
        N.jernucl01_warm(z, con, xl, 268.15, 1e-3, 1e-3, 7e5, -0.02, 0.03,
                         100.0, 0, rccn)
    assert ":3117" in err.value.where and "ICE_NUCL" in err.value.condition
    # below 0 C with neither water nor ice supersaturation WRF does nothing
    res = N.jernucl01_warm(z, con, xl, 268.15, 1e-3, 1e-3, 7e5, -0.2, -0.1,
                           100.0, 0, rccn)
    assert res.branch is None and not res.activated
    np.testing.assert_array_equal(res.fccnr, con)
    # between -38 C and -37 C the ice branch is not entered (:3117)
    res = N.jernucl01_warm(z, con, xl, 273.15 - 37.5, 1e-4, 6e-4, 4e5, 0.01,
                           0.4, 100.0, 0, rccn, behavior=WRF)
    assert res.activated


# ------------------------------------------------------------------ budgets
def _budget(rec: dict, res, xl) -> tuple[float, float, float, float]:
    l_in = N.liquid_mass_mixing_ratio(rec["psi1_in"], xl, rec["ror"])
    l_out = N.liquid_mass_mixing_ratio(res.psi1, xl, rec["ror"])
    water_in = float(rec["qq_in"]) + l_in
    water_out = res.qq + l_out
    lse_in = float(rec["tt_in"]) - AL1 * l_in
    lse_out = res.tt - AL1 * l_out
    return water_in, water_out - water_in, lse_in, lse_out - lse_in


def test_corrected_activation_closes_total_water_and_liquid_static_energy():
    """With ``nucleation_water_sink`` the drops are made of vapour and their
    latent heat warms the air: qv plus liquid, and T minus AL1 times liquid,
    close to 1e-13 over every activation of the sweep, and the recomputed
    supersaturation falls.

    The supersaturations returned are those of the state returned, bit for
    bit: the warm column hands ``sup1`` to ``ONECOND1`` as ``DEL1N``, its
    starting supersaturation, which decides condensation or evaporation
    (:6905-6945).  Breakage prevented: a sink that recomputes them from the
    vapour or the temperature before the sink, or leaves ``sup2`` at its
    input, which the budget alone does not see (the drop spectrum moved by
    up to 0.98 percent of its maximum per call on the KC-1 replay states
    with the pre-sink vapour).

    The sink and the budget both measure liquid with
    ``liquid_mass_mixing_ratio``, so the sink's mass is also checked against
    the activated drops summed directly, ``sum(DROPCONCN XL) / ROR``: the
    two agree to within 8 epsilon of the liquid present (1.5 measured on the
    O64 sweep, where the liquid already present reaches 130 times the
    activated mass)."""

    tabs = _tables("o64")
    xl = tabs["xl"]
    eps = np.finfo(np.float64).eps
    n = 0
    for i in range(_sweep_count("o64")):
        rec = _sweep_record("o64", i)
        res = _run_sweep_record(rec, tabs, behavior=CORRECTED)
        if not res.activated:
            assert res.tt == float(rec["tt_in"]) and res.qq == float(rec["qq_in"])
            assert (res.sup1, res.sup2) == (float(rec["sup1_in"]),
                                            float(rec["sup2_in"])), i
            continue
        assert (res.sup1, res.sup2) == _supersaturations(
            res.tt, res.qq, float(rec["pp"])), i
        water, dwater, lse, dlse = _budget(rec, res, xl)
        assert abs(dwater) <= BUDGET * water, i
        assert abs(dlse) <= BUDGET * lse, i
        direct = 0.0
        for k in range(NKR):
            direct = direct + float(res.water.dropconcn[k]) * float(xl[k])
        direct = direct / float(rec["ror"])
        l_in = N.liquid_mass_mixing_ratio(rec["psi1_in"], xl, rec["ror"])
        assert abs(res.delmassl - direct) <= 8.0 * eps * (l_in + direct), i
        assert res.sink_applied and res.sup1 < float(rec["sup1_in"])
        # the activation itself is WRF's
        wrf = _run_sweep_record(rec, tabs, behavior=WRF)
        np.testing.assert_array_equal(res.psi1, wrf.psi1)
        np.testing.assert_array_equal(res.fccnr, wrf.fccnr)
        n += 1
    assert n > 200


def test_wrf_v480_activation_creates_the_activated_mass_from_nothing():
    """D3 reproduced: under ``wrf_v480`` the water residual of an activation
    step is the activated drop mass, and the temperature misses AL1 times it.
    Numbers (O64 sweep inputs; KC-1 probe states)."""

    tabs = _tables("o64")
    masses = []
    for i in range(_sweep_count("o64")):
        rec = _sweep_record("o64", i)
        res = _run_sweep_record(rec, tabs, behavior=WRF)
        if not res.activated:
            continue
        water, dwater, lse, dlse = _budget(rec, res, tabs["xl"])
        assert res.delmassl > 0.0 and not res.sink_applied
        assert abs(dwater - res.delmassl) <= BUDGET * water, i
        assert abs(dlse + AL1 * res.delmassl) <= BUDGET * lse, i
        masses.append(res.delmassl)
    probe = [_run_probe(r0, tabs).delmassl for r0, _ in _probe_pairs("o64")]
    probe = [m for m in probe if m > 0.0]
    report = (f"D3: {len(masses)} sweep activations create "
              f"{min(masses) * 1e3:.3e} to {max(masses) * 1e3:.3e} g/kg of "
              f"liquid from nothing (heating missed up to "
              f"{AL1 * max(masses):.3e} K); {len(probe)} KC-1 probe "
              f"activations up to {max(probe) * 1e3:.3e} g/kg")
    print(report)
    assert len(masses) > 200 and len(probe) > 0


def test_rmassl_nucl_is_not_the_activated_mass():
    """WRF's discarded RMASSL_NUCL holds only the last bin's value and is the
    activated mass times ``3 COL x_k``; a sink built on it would remove about
    2e-11 of the drops' mass."""

    tabs = _tables("o64")
    xl = tabs["xl"]
    checked = 0
    for i in range(_sweep_count("o64")):
        rec = _sweep_record("o64", i)
        if np.any(rec["psi1_in"] > 0):
            continue
        res = _run_sweep_record(rec, tabs)
        water = res.water
        assert water.rmassl_nucl == (water.rmassl_nucl_iterations[-1]
                                     if water.steps else 0.0)
        if len(water.steps) == 1 and res.activated:
            k = water.steps[0].drop_bin - 1
            assert math.isclose(water.rmassl_nucl,
                                res.delmassl * 3.0 * COL * xl[k], rel_tol=1e-12)
            assert water.rmassl_nucl < 1e-10 * res.delmassl
            checked += 1
    assert checked > 0


# --------------------------------------------------------- negative control
def test_ro_solute_perturbed_by_one_part_per_million_fails_the_sweep_gate():
    failures = _sweep_gate_failures("o64", ro_solute=RO_SOLUTE * (1.0 + 1e-6))
    assert len(failures) > 20
    assert {name for _, name in failures} >= {"fccnr"}


# ---------------------------------------------------------- the D9 rule
def _synthetic_column(cloudy: list[bool], w_up: float = 1.0):
    """Levels saturated (qv at 101 percent) or not (80 percent) with a
    uniform updraft, 100 m apart."""

    nz = len(cloudy)
    t = np.full(nz, 290.0)
    p = np.full(nz, 9.0e5)
    es = 2.53e12 * np.exp(-5.42e3 / t)
    e = np.where(cloudy, 1.01, 0.80) * es
    qv = 0.622 * e / (p - 0.378 * e)
    w = np.full(nz + 1, w_up)
    z = (np.arange(nz) + 0.5) * 1.0e4
    return t, qv, p, w, z


def test_the_cloud_base_rules_on_a_column_they_disagree_on():
    """Levels 3-5 (one based) are saturated updraft: the intended rule picks
    level 3; WRF's evaluation of :4387 picks the highest level that is not a
    saturated updraft, here the top."""

    cloudy = [False, False, True, True, True, False, False, False]
    col = _synthetic_column(cloudy)
    assert N.cloud_base_level(*col, rule="first_level") == 2
    assert N.cloud_base_level(*col, rule="wrf_v480") == 7
    # every level above the first saturated: WRF ends on level 1
    col = _synthetic_column([True] * 8)
    assert N.cloud_base_level(*col, rule="wrf_v480") == 0
    assert N.cloud_base_level(*col, rule="first_level") == 2
    # top levels saturated: WRF lands below them
    col = _synthetic_column([False] * 5 + [True] * 3)
    assert N.cloud_base_level(*col, rule="wrf_v480") == 4
    # no updraft: nothing qualifies under the intended rule
    col = _synthetic_column(cloudy, w_up=0.05)
    assert N.cloud_base_level(*col, rule="first_level") is None
    assert N.cloud_base_level(*col, rule="wrf_v480") == 7
    # the intended rule ignores levels 1 and 2 and heights of 3 km and above
    col = _synthetic_column([True, True, False, False])
    assert N.cloud_base_level(*col, rule="first_level") is None
    t, qv, p, w, z = _synthetic_column([False, False, False, True])
    assert N.cloud_base_level(t, qv, p, w, z, rule="first_level") == 3
    assert N.cloud_base_level(t, qv, p, w, z + 3.0e5, rule="first_level") is None


def test_the_cloud_base_rule_follows_the_preset():
    assert N.cloud_base_rule(WRF) == "wrf_v480"
    assert N.cloud_base_rule(CORRECTED) == "first_level"
    assert N.cloud_base_rule(None) == "first_level"
    col = _synthetic_column([False, False, True, True, False])
    assert N.cloud_base_level(*col) == 2
    assert N.cloud_base_level(*col, behavior=WRF) == 4
    assert N.cloud_base_level(*col, behavior=WRF, rule="first_level") == 2
    with pytest.raises(FsbmRefusal):
        N.cloud_base_level(*col, rule="lowest")


def test_a_mixed_behaviour_keeps_its_presets_cloud_base_rule():
    """A preset with one unrelated field changed (an A/B arm isolating D1,
    D3, D5 or the breakup cap) keeps the preset's D9 rule, because
    ``FsbmBehavior.cloud_base_rule`` has no default and ``replace`` carries
    it, so the arm never switches the cloud-base detection with it and
    measures one correction at a time.  An explicit ``rule=`` still wins."""

    col = _synthetic_column([False, False, True, True, False])
    mixed = (
        (dataclasses.replace(WRF, rainnc_velocity="own_column"), 4),   # D1
        (dataclasses.replace(WRF, rainnc_source="bottom_outflow"), 4),  # D2
        (dataclasses.replace(WRF, nucleation_water_sink=True), 4),      # D3
        (dataclasses.replace(WRF, dtcond_carry=False), 4),              # D5
        (dataclasses.replace(WRF, breakup_retry_cap=128), 4),           # D7
        (dataclasses.replace(CORRECTED, rainnc_velocity="tile_last"), 2),
        (dataclasses.replace(CORRECTED, nucleation_water_sink=False), 2),
    )
    for behavior, level in mixed:
        assert N.cloud_base_level(*col, behavior=behavior) == level
        assert N.cloud_base_level(*col, behavior=behavior,
                                  rule="wrf_v480") == 4
        assert N.cloud_base_level(*col, behavior=behavior,
                                  rule="first_level") == 2
    assert N.cloud_base_rule(FsbmBehavior(**dataclasses.asdict(CORRECTED))) \
        == "first_level"


def test_a_behaviour_decides_its_cloud_base_rule_and_one_without_is_refused():
    """``FsbmBehavior.cloud_base_rule`` is taken at its word, mixed or not;
    a value outside the two rules is refused when the behaviour is built;
    an object that carries no ``cloud_base_rule`` is refused until the
    caller names the rule, so its ``Cloud_Base_Super`` points are never
    picked for it silently."""

    arm = dataclasses.replace(WRF, nucleation_water_sink=True)
    assert N.cloud_base_rule(arm) == "wrf_v480"
    arm = dataclasses.replace(WRF, cloud_base_rule="first_level")
    assert N.cloud_base_rule(arm) == "first_level"
    with pytest.raises(FsbmRefusal, match="cloud_base_rule"):
        dataclasses.replace(WRF, cloud_base_rule="lowest")
    col = _synthetic_column([False, False, True, True, False])
    legacy = types.SimpleNamespace(
        **{k: v for k, v in dataclasses.asdict(WRF).items()
           if k != "cloud_base_rule"})
    with pytest.raises(FsbmRefusal, match="rule= explicitly"):
        N.cloud_base_rule(legacy)
    with pytest.raises(FsbmRefusal, match="rule= explicitly"):
        N.cloud_base_level(*col, behavior=legacy)
    assert N.cloud_base_level(*col, behavior=legacy, rule="wrf_v480") == 4


def test_is_this_cloud_base_needs_the_level_and_few_drops():
    xl = _tables("o64")["xl"]
    few = np.zeros(NKR)
    few[0] = 4.9 / (COL * xl[0])           # COL sum(f x) just below 5
    many = np.zeros(NKR)
    many[0] = 5.0 / (COL * xl[0]) * 1.001
    assert N.is_this_cloud_base(3, 3, few, xl)
    assert not N.is_this_cloud_base(3, 3, many, xl)
    assert not N.is_this_cloud_base(2, 3, few, xl)
    assert not N.is_this_cloud_base(3, None, few, xl)


# ------------------------------------------------------------- the rest
def test_the_polysvp_copy_matches_the_oracle():
    """The private POLYSVP the corrected sink uses, against every O64
    POLYSVP record (water and ice)."""

    d = fx.load_fixture("sweep_polysvp_o64")
    n = d["case.tt"].shape[0]
    assert n > 600
    for i in range(n):
        got = N._polysvp(float(d["case.tt"][i]), int(d["case.itype"][i]))
        assert math.isclose(got, float(d["case.es"][i]), rel_tol=1e-14)


def test_the_functions_do_not_modify_their_inputs():
    tabs = _tables("o64")
    rec = _sweep_record("o64", 50)
    psi1 = np.array(rec["psi1_in"], dtype=np.float64)
    fccnr = np.array(rec["fccnr_in"], dtype=np.float64)
    keep = (psi1.copy(), fccnr.copy(), tabs["xl"].copy(), tabs["rccn"].copy())
    for behavior in (WRF, CORRECTED):
        N.jernucl01_warm(psi1, fccnr, tabs["xl"], rec["tt_in"], rec["qq_in"],
                         rec["ror"], rec["pp"], rec["sup1_in"], rec["sup2_in"],
                         rec["win"], 1, tabs["rccn"], behavior=behavior)
    for a, b in zip((psi1, fccnr, tabs["xl"], tabs["rccn"]), keep):
        np.testing.assert_array_equal(a, b)


def test_wrong_shapes_are_refused():
    tabs = _tables("o64")
    with pytest.raises(FsbmRefusal):
        N.jernucl01_warm(np.zeros(NKR - 1), tabs["fccnr_con"], tabs["xl"],
                         290.0, 0.01, 1e-3, 9e5, 0.01, -0.1, 100.0, 0,
                         tabs["rccn"])
    with pytest.raises(FsbmRefusal):
        N.initial_aerosol_column(tabs["fccnr_con"][:40], tabs["fccnr_mar"],
                                 np.ones(5), True)
