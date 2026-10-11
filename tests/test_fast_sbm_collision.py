"""Drop collision-coalescence, collisional breakup and spontaneous breakup.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
* A Bott update that is not sequential and in place, or that visits the
  pairs in another order: every later pair reads what the earlier ones left,
  so a vectorised or reordered update gives a different spectrum with the
  same total mass.  The O64 sweep and probe records catch it.
* A breakup retry that doubles in the wrong place, restarts from the wrong
  spectrum, or applies a breakup WRF abandoned: the flags (``ndiv``, path)
  are compared exactly and the spectrum at label 11 is compared too.
* WRF's unbounded mass-gain retry (design finding D7) turning into a hang
  instead of a named refusal.
* A tally for the emulator lane that does not add up to the mass the
  coalescence actually moved, or that changes the result.
* A kernel, table or rate that is slightly wrong and still "looks right":
  each oracle gate carries a negative control that must fail.

Tolerances (design 5.5): host against the promoted O64 build within
relative 1e-10 with an absolute floor of 1e-12 of the record's scale, flags
exact.  The native rung of design 5.5 (the host on the native build's
pre-call states, within ``abs(O64x - O32) + 1e-9 scale`` where O64x is the
promoted build run on those same states) has no sweep records to run on:
the oracle holds O64x only for the column, ``kc1_crossreplay_o64_on_o32``,
and the warm-column gate 3 carries the rung there.  The sweep O32 records
are the same case definitions built in float32, with inputs that are not
the O64 inputs, so they cannot stand in for O64x (see "The native rung"
below).  The largest O64 distance measured on this desktop (Windows, Python 3.13) is
7.6e-11, all of it in ``coll_xxx_lwf``: Bott's flux takes
``exp(x1/2) - exp(x1 (1/2 - c))``, which cancels when the Courant number
``c`` is near zero (``chucm`` is 2e-5 on some diagonal pairs), so a one-ULP
difference between the C runtime's ``exp``/``log`` and glibc's is amplified
by up to about 1e5.  With correctly rounded ``exp`` and ``log`` the distance
falls to 2.9e-11 or less; the calls whose kernel the probe recorded agree to
1.2e-15.

Kernels.  ``COAL_BOTT_NEW`` builds ``cwll`` itself (``Kernals_KS``,
:6256-6278, ``ckern_z`` :6544-6558), which belongs to the derived-table item.
Where the oracle recorded the ``cwll`` a call used, the test passes that
array.  Elsewhere (the 60 s stress sweep cases and the 102 probe calls past
the probe's 40-kernel cap) it passes ``_cwll``, this file's transcription of
those lines on the O64 dump's tables, which the first test proves byte-equal
to every ``cwll`` the oracle recorded.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import math
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.fast_sbm import collision as C
from gpuwm.core.fast_sbm.constants import COL, NKR, FsbmBehavior, FsbmRefusal
from gpuwm.core.fast_sbm.tables import load_tables
from tools.fast_sbm_wrf480_oracle import fixtures as fx

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "gpuwm" / "core" / "fast_sbm" / "collision.py"

REL = 1.0e-10
FLOOR = 1.0e-12
#: KC-1's time step (manifest kc1.definition.call): the spontaneous breakup
#: probe calls pass DT = 12 s.
KC1_DT = 12.0


# ---------------------------------------------------------------------------
# Oracle inputs
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _tables():
    return load_tables("float64")


@functools.lru_cache(maxsize=None)
def _dump(build: str) -> dict:
    return fx.load_fixture(f"tables_{build}")


@functools.lru_cache(maxsize=None)
def _derived(build: str = "o64"):
    t = _dump(build)
    return types.SimpleNamespace(
        xl_mg=t["derived.xl_mg_6078"][0],
        ima=t["derived.ima"][0],
        chucm=t["derived.chucm"][0],
        ecoalmassm=t["derived.ecoalmassm"][0],
        prob=t["derived.prob"][0],
        gain_var_new=t["derived.gain_var_new"][0],
        ikr_spon_break=int(t["derived.ikr_spon_break"][0]),
    )


def _cwll(p_z: float, dtime_coal: float) -> np.ndarray:
    """``Kernals_KS`` (:6256-6278) with ``ckern_z`` (:6544-6558), promoted.

    Test-side reconstruction of the drop-drop kernel on the O64 dump's
    tables, in WRF's order: interpolate in pressure, times ``dtime_coal``,
    times ``dlnr``, then times ``ECOALMASSM``.
    """
    t = _dump("o64")
    y1 = t["read.ywll_1000mb"][0]
    y2 = t["read.ywll_750mb"][0]
    y3 = t["read.ywll_500mb"][0]
    ecoal = _derived().ecoalmassm
    p1, p2, p3 = 1.0e6, 0.75e6, 0.50e6
    scal = 1.0
    dlnr = math.log(2.0) / (3.0 * scal)
    out = np.empty((NKR, NKR))
    for i in range(NKR):
        for j in range(NKR):
            c1, c2, c3 = float(y1[i, j]), float(y2[i, j]), float(y3[i, j])
            ck = math.nan
            if p_z >= p1:
                ck = c1
            if p_z <= p3:
                ck = c3
            if p1 > p_z >= p2:
                ck = c2 + (c1 - c2) * (p_z - p2) / (p1 - p2)
            if p2 > p_z > p3:
                ck = c3 + (c2 - c3) * (p_z - p3) / (p2 - p3)
            out[i, j] = ck * dtime_coal * dlnr
    return ecoal * out


@functools.lru_cache(maxsize=None)
def _records(name: str, rtype: str | None = None) -> tuple:
    types_ = None if rtype is None else [rtype]
    return tuple(fx.iter_records(name, types=types_))


@functools.lru_cache(maxsize=None)
def _sweep_kernels(build: str) -> dict:
    return {(float(r["p_z"]), float(r["dt_coll"])): r["cwll"]
            for r in _records(f"sweep_coll_xxx_{build}", "kernel")}


@functools.lru_cache(maxsize=None)
def _probe_points(build: str = "o64") -> tuple:
    """Probe records grouped by call, in the order WRF reached them."""
    wanted = ["coal_before", "xxx_before", "kernel", "xxx_after", "breakup",
              "coal_after", "spont_before", "spont_after"]
    groups: dict = defaultdict(dict)
    order = []
    for rec in fx.iter_records(f"kc1_probe_{build}", types=wanted):
        key = (int(rec["step"]), int(rec["i"]), int(rec["k"]), int(rec["j"]))
        if key not in groups:
            order.append(key)
        assert rec.type not in groups[key], f"two {rec.type} records at {key}"
        groups[key][rec.type] = rec
    return tuple((key, groups[key]) for key in order)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _excess(host, ref, rel=REL, floor=FLOOR) -> float:
    """Largest ``|host - ref| / max(rel |ref|, floor scale)``; <= 1 passes."""
    host = np.asarray(host, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert host.shape == ref.shape
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    diff = np.abs(host - ref)
    if scale == 0.0:
        return 0.0 if np.all(diff == 0.0) else math.inf
    allowed = np.maximum(rel * np.abs(ref), floor * scale)
    return float(np.max(diff / allowed))


def _check(host, ref, what: str) -> None:
    e = _excess(host, ref)
    assert e <= 1.0, (f"{what}: host differs from the O64 oracle by "
                      f"{e:.3g} times the allowance (relative {REL}, floor "
                      f"{FLOOR} of scale)")


def _seqsum(values) -> float:
    total = 0.0
    for v in np.asarray(values, dtype=np.float64):
        total = total + float(v)
    return total


def _coal(rec_ff1, tt, qq, pp, rho, dt, cwll, *, tables=None, behavior=None,
          **kw):
    return C.coal_bott_warm(rec_ff1, float(tt), float(qq), float(pp), float(rho),
                            float(dt), tables or _tables(), _derived(), cwll,
                            behavior, **kw)


def _ndiv(value) -> int:
    return -1 if value is None else int(value)


# ---------------------------------------------------------------------------
# The test's own kernel builder is WRF's cwll
# ---------------------------------------------------------------------------

def test_the_test_kernel_builder_is_wrfs_cwll_byte_for_byte():
    seen = 0
    for (p_z, dt), cwll in _sweep_kernels("o64").items():
        assert np.array_equal(_cwll(p_z, dt), cwll), (p_z, dt)
        seen += 1
    for rec in _records("kernels_o64"):
        assert np.array_equal(_cwll(float(rec["p_z"]), float(rec["dtime_coal"])),
                              rec["cwll"]), float(rec["p_hpa"])
        seen += 1
    probe_kernels = 0
    for _key, group in _probe_points():
        if "kernel" in group:
            kern = group["kernel"]
            dt = float(group["coal_before"]["dt_coll"])
            assert np.array_equal(_cwll(float(kern["pp"]), dt), kern["cwll"])
            assert np.array_equal(kern["xl_mg"], _derived().xl_mg)
            probe_kernels += 1
    assert probe_kernels == 40
    assert seen == 12 + 10


# ---------------------------------------------------------------------------
# O64 sweeps
# ---------------------------------------------------------------------------

def test_coll_xxx_lwf_reproduces_every_o64_sweep_record():
    cases = _records("sweep_coll_xxx_o64", "case")
    assert len(cases) == 36
    for rec in cases:
        cwll = _sweep_kernels("o64")[(float(rec["p_z"]), float(rec["dt_coll"]))]
        d = _derived()
        g, fl = C.coll_xxx_lwf(rec["g_in"], np.ones(NKR), cwll, rec["x"], d.chucm,
                               d.ima, 1.0)
        what = f"coll_xxx p={float(rec['p_z'])} dt={float(rec['dt_coll'])} " \
               f"spectrum {int(rec['spectrum'])}"
        _check(g, rec["g_out"], what + " g")
        _check(fl, rec["fl_out"], what + " fl")


def test_coll_breakup_reproduces_every_o64_sweep_record():
    t = _tables()
    xt = _derived().xl_mg * 1.0e-3
    singles = _records("sweep_breakup_o64", "case")
    sequences = _records("sweep_breakup_o64", "sequence")
    assert len(singles) == 16 and len(sequences) == 16
    for rec in singles:
        out = C.coll_breakup(rec["gt_in"], rec["xt"], float(rec["dt_call"]),
                             t.pkij, t.qkj)
        _check(out, rec["gt_out"], f"breakup spectrum {int(rec['spectrum'])} "
                                   f"dt={float(rec['dt_call'])}")
    for rec in sequences:
        out = rec["gt_in"]
        for _ in range(int(rec["ndiv"])):
            out = C.coll_breakup(out, xt, float(rec["dt_call"]), t.pkij, t.qkj)
        what = f"breakup sequence spectrum {int(rec['spectrum'])} ndiv {int(rec['ndiv'])}"
        _check(out, rec["gt_out"], what)
        _check(_seqsum(out) / _seqsum(rec["gt_in"]), rec["mass_ratio"], what + " mass")


def test_coal_bott_warm_reproduces_every_o64_sweep_record():
    cases = _records("sweep_coal_bott_o64")
    assert len(cases) == 45
    paths = set()
    for rec in cases:
        dt = float(rec["dt_coll"])
        cwll = _sweep_kernels("o64").get((float(rec["pp"]), dt))
        if cwll is None:
            cwll = _cwll(float(rec["pp"]), dt)
        res = _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"],
                    dt, cwll, del1in=float(rec["del1in_in"]),
                    del2in=float(rec["del2in_in"]))
        what = f"coal p={float(rec['p_hpa'])} dt={dt} spectrum {int(rec['spectrum'])}"
        flags = (res.icol_drop, res.icol_drop_brk, _ndiv(res.ndiv), res.breakup_path)
        want = (int(rec["icol_drop"]), int(rec["icol_drop_brk"]), int(rec["ndiv"]),
                int(rec["breakup_path"]))
        assert flags == want, f"{what}: flags {flags}, WRF {want}"
        paths.add((res.breakup_path, _ndiv(res.ndiv)))
        _check(res.ff1r, rec["ff1_out"], what + " ff1")
        _check(res.g1_before_xxx, rec["g1_before_xxx"], what + " g1 before xxx")
        _check(res.g1_after_xxx, rec["g1_after_xxx"], what + " g1 after xxx")
        _check(res.g1_after_breakup, rec["g1_after_breakup"], what + " g1 at 11")
        per = 0.0 if res.break_drop_per is None else res.break_drop_per
        _check(per, rec["break_drop_per"], what + " break_drop_per")
        assert (res.tt, res.del1in, res.del2in) == (
            float(rec["tt_out"]), float(rec["del1in_out"]), float(rec["del2in_out"]))
        assert res.colleff is None
    # The sweep reaches no breakup, breakup at ndiv 1, and both retry depths.
    assert {(0, -1), (1, 1), (1, 2), (1, 4)} <= paths


def test_spont_rain_breakup_reproduces_every_o64_sweep_record():
    d = _derived()
    cases = _records("sweep_spont_o64")
    assert len(cases) == 12
    early = 0
    for rec in cases:
        out = C.spont_rain_breakup(float(rec["dt"]), rec["ff_in"], rec["xl"], d.prob,
                                   d.gain_var_new, d.ikr_spon_break)
        _check(out, rec["ff_out"], f"spont spectrum {int(rec['spectrum'])} "
                                   f"dt={float(rec['dt'])}")
        if int(rec["taken"]) == 0:
            early += 1
            assert np.array_equal(out, rec["ff_in"])
    assert early == 4  # both early-return paths, at two time steps each


# ---------------------------------------------------------------------------
# KC-1 probe records
# ---------------------------------------------------------------------------

def test_every_kc1_coal_probe_record():
    calls = with_kernel = breakups = 0
    for key, group in _probe_points():
        if "coal_before" not in group:
            continue
        calls += 1
        before, after, brk = group["coal_before"], group["coal_after"], group["breakup"]
        dt = float(before["dt_coll"])
        if "kernel" in group:
            cwll = group["kernel"]["cwll"]
            with_kernel += 1
        else:
            cwll = _cwll(float(before["pp"]), dt)
        res = _coal(before["ff1"], before["tt"], before["qq"], before["pp"],
                    before["rhocgs"], dt, cwll, del1in=float(before["del1in"]),
                    del2in=float(before["del2in"]))
        what = f"KC-1 COAL at step, i, k, j = {key}"
        flags = (res.icol_drop, res.icol_drop_brk, _ndiv(res.ndiv), res.breakup_path)
        want = (int(brk["icol_drop"]), int(brk["icol_drop_brk"]), int(brk["ndiv"]),
                int(brk["path"]))
        assert flags == want, f"{what}: flags {flags}, WRF {want}"
        breakups += res.breakup_path == 1
        _check(res.ff1r, after["ff1"], what + " ff1")
        _check(res.g1_after_breakup, brk["g1"], what + " g1 at 11")
        per = 0.0 if res.break_drop_per is None else res.break_drop_per
        _check(per, brk["break_drop_per"], what + " break_drop_per")
        if "xxx_before" in group:
            _check(res.g1_before_xxx, group["xxx_before"]["g1"], what + " g1 before xxx")
            _check(res.g1_after_xxx, group["xxx_after"]["g1"], what + " g1 after xxx")
        else:
            assert res.icol_drop == 0 and res.g1_before_xxx is None
        # The warm path leaves T, qv and the supersaturations alone.
        for name in ("tt", "qq", "del1in", "del2in"):
            assert float(after[name]) == float(before[name])
        assert (res.tt, res.del1in, res.del2in) == (
            float(after["tt"]), float(after["del1in"]), float(after["del2in"]))
    assert calls == 142 and with_kernel == 40
    assert breakups > 0


def test_every_kc1_spontaneous_breakup_probe_record():
    d = _derived()
    calls = changed = 0
    for key, group in _probe_points():
        if "spont_before" not in group:
            continue
        calls += 1
        before, after = group["spont_before"], group["spont_after"]
        assert int(before["ikr_spon_break"]) == d.ikr_spon_break
        out = C.spont_rain_breakup(KC1_DT, before["ff1"], _tables().xl, d.prob,
                                   d.gain_var_new, int(before["ikr_spon_break"]))
        _check(out, after["ff1"], f"KC-1 spontaneous breakup at {key}")
        changed += not np.array_equal(before["ff1"], after["ff1"])
    assert calls == 142 and changed > 0


# ---------------------------------------------------------------------------
# The native rung
#
# Design 5.5 bounds the host on the native build's pre-call states by the
# promoted build run on those same states (O64x).  The sweeps have no O64x:
# each sweep case is built from one definition in both builds, so the O32
# record's inputs are that definition rounded through float32, not the O64
# record's inputs, and the O64 record is not O64x.  An envelope test that
# ran the host on the O64 inputs could not fail: the O64 gate above holds
# the host to 1e-10 of the O64 output, so |host - O32| <= |O64 - O32| +
# 1e-9 scale follows from the triangle inequality (the four such tests
# this file carried until the post-review fix were removed for that
# reason).  Run on the O32 inputs, that bound is the wrong one (the host
# lands up to 1.8e-8 of scale outside it on coal_bott_warm and 2.8e-10 on
# coll_xxx_lwf), because |O64 - O32| mixes the input rounding into the
# precision distance.  The rung is therefore the column's: warm-column gate
# 3 runs the host, which calls every function of this file, on the O32
# pre-call states of the 16 replay calls against kc1_crossreplay_o64_on_o32.
# Sweep O64x records would need a read-back mode in run_sweeps.F90 and a
# fixture rebuild, which is the oracle item's.
# ---------------------------------------------------------------------------

#: Record fields that are the inputs of each sweep family.
_SWEEP_INPUTS = {
    "coll_xxx": ("g_in",),
    "breakup": ("gt_in",),
    "coal_bott": ("ff1_in", "tt_in", "qq", "pp", "rho"),
    "spont": ("ff_in",),
}


def test_the_sweep_o32_records_are_not_o64x_so_the_native_rung_is_the_columns():
    """Breakage prevented: the O64 sweep records relabelled as O64x, and an
    envelope that cannot fail reported as native evidence again.

    The O32 sweep inputs equal the O64 inputs in 0 of 36 coll_xxx cases, 0
    of 32 breakup records, 0 of 45 coal_bott cases and 2 of 12 spont cases,
    so no sweep record is the promoted build on a native state.  The column
    carries the rung: its gate 3 must exist and read the cross-replay."""
    equal = {}
    for family, names in _SWEEP_INPUTS.items():
        same = total = 0
        for a, b in zip(_records(f"sweep_{family}_o64"),
                        _records(f"sweep_{family}_o32")):
            if family == "coll_xxx" and a.type != "case":
                continue
            assert a.type == b.type and int(a["spectrum"]) == int(b["spectrum"])
            total += 1
            same += all(np.array_equal(np.asarray(a[n], dtype=np.float64),
                                       np.asarray(b[n], dtype=np.float64))
                        for n in names)
        equal[family] = (same, total)
    assert equal == {"coll_xxx": (0, 36), "breakup": (0, 32),
                     "coal_bott": (0, 45), "spont": (2, 12)}, equal
    column = ROOT / "tests" / "test_fast_sbm_warm_column.py"
    tree = ast.parse(column.read_text(encoding="utf-8"))
    gate = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "test_native_envelope_on_the_native_builds_states")
    strings = {node.value for node in ast.walk(gate)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {"kc1_crossreplay_o64_on_o32", "kc1_replay_o32"} <= strings


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", ["o64", "o32"])
def test_negative_drops_after_coalescence_refuse_as_wrf_stops(build):
    refusals = [r for r in fx.iter_refusals(build=build) if r.routine == "COAL_BOTT_NEW"]
    assert len(refusals) == 1
    ref = refusals[0]
    assert ref.exit_status == 99
    (rec,) = ref.records
    ff1 = np.asarray(rec["ff1"], dtype=np.float64)
    assert ff1[19] < 0.0  # bin 20, which no collision pair targets
    with pytest.raises(FsbmRefusal) as info:
        _coal(ff1, rec["tt"], rec["qq"], rec["pp"], rec["rho"], rec["dt_coll"],
              _cwll(float(rec["pp"]), float(rec["dt_coll"])))
    wrf_condition = ref.message.split(", model stop")[0]
    assert wrf_condition == "stop at end coal_bott - FF1R NaN or FF1R < 0.0"
    assert wrf_condition in info.value.condition
    assert "bin 20" in info.value.condition
    assert ":8606-8611" in info.value.where


def _with_nan(rec, bin_number: int) -> np.ndarray:
    ff1 = np.array(rec["ff1_in"], dtype=np.float64)
    ff1[bin_number - 1] = np.nan
    return ff1


def _coal_rec(rec, ff1):
    dt = float(rec["dt_coll"])
    return _coal(ff1, rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], dt,
                 _cwll(float(rec["pp"]), dt))


def test_a_nan_drop_bin_is_refused_on_the_no_breakup_path():
    """With no breakup, the NaN half of the final check (:8606-8611) is the
    only guard between a NaN drop bin and a NaN ``FF1R`` handed back.

    Breakage prevented: the check reduced to ``FF1R < 0`` (NaN compares
    false), which returned bin 17 as NaN with every other test green."""
    rec = next(r for r in _records("sweep_coal_bott_o64")
               if int(r["icol_drop"]) == 1 and int(r["icol_drop_brk"]) == 0)
    assert (int(rec["spectrum"]), float(rec["dt_coll"])) == (1, 3.0)
    with pytest.raises(FsbmRefusal) as info:
        _coal_rec(rec, _with_nan(rec, 17))
    assert ":8606-8611" in info.value.where
    assert "FF1R NaN" in info.value.condition
    assert "bin 17 is nan" in info.value.condition


def test_a_nan_drop_bin_is_refused_on_the_breakup_path():
    """On the breakup path WRF stops at two NaN checks before the final one:
    a NaN breakup rate inside ``coll_breakup_KS`` (:1578-1593), reached by a
    NaN in bin 2 or above, which coalescence carries into the breakup bins;
    and a NaN ``FF1R`` after breakup (:8409-8415), reached by a NaN in bin
    1, which coalescence leaves where it is, below the breakup bins.  Each
    must refuse at its own check.

    Breakage prevented: either check dropped, the NaN then refused (or not)
    somewhere later, with no test naming the check WRF stops at."""
    rec = next(r for r in _records("sweep_coal_bott_o64")
               if int(r["breakup_path"]) == 1)
    assert (int(rec["spectrum"]), float(rec["dt_coll"]), int(rec["ndiv"])) == (
        2, 3.0, 1)
    with pytest.raises(FsbmRefusal) as info:
        _coal_rec(rec, _with_nan(rec, 17))
    assert info.value.where == "coll_breakup_KS (:1578-1593)"
    assert "drop bin 16" in info.value.condition
    with pytest.raises(FsbmRefusal) as info:
        _coal_rec(rec, _with_nan(rec, 1))
    assert info.value.where == "COAL_BOTT_NEW (:8409-8415)"
    assert "NaN in bin 1 after collisional breakup with ndiv=1" in \
        info.value.condition
    # coll_breakup_KS on its own: a NaN among the breakup bins.
    t = _tables()
    gt = np.array(rec["g1_after_xxx"], dtype=np.float64) * 1.0e-3
    gt[NKR - 1] = np.nan
    with pytest.raises(FsbmRefusal, match="breakup rate is NaN"):
        C.coll_breakup(gt, _derived().xl_mg * 1.0e-3, 3.0, t.pkij, t.qkj)


def test_the_retry_cap_refuses_a_runaway_breakup():
    # PKIJ raised by 10 percent makes breakup gain mass at every substep
    # count, so WRF's doubling at :8423-8425 would never end (D7).
    rec = next(r for r in _records("sweep_coal_bott_o64")
               if (int(r["spectrum"]), float(r["dt_coll"]), float(r["p_hpa"]))
               == (6, 12.0, 1000.0))
    runaway = dataclasses.replace(_tables(), pkij=_tables().pkij * 1.1)
    for preset in (FsbmBehavior.wrf_v480(), FsbmBehavior.corrected()):
        with pytest.raises(FsbmRefusal) as info:
            _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"],
                  12.0, _cwll(1.0e6, 12.0), tables=runaway, behavior=preset)
        assert "D7" in info.value.where and ":8423-8425" in info.value.where
        assert "ndiv=64" in info.value.condition
        assert "breakup_retry_cap=64" in info.value.condition
    small = dataclasses.replace(FsbmBehavior.wrf_v480(), breakup_retry_cap=4)
    with pytest.raises(FsbmRefusal) as info:
        _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0,
              _cwll(1.0e6, 12.0), tables=runaway, behavior=small)
    assert "ndiv=4" in info.value.condition
    # Unscaled, the same spectrum is accepted at ndiv 1.
    res = _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0,
                _cwll(1.0e6, 12.0), behavior=small)
    assert (res.breakup_path, res.ndiv, res.ndiv_trials) == (1, 1, (1,))


def test_d7_the_mass_gain_retry_starts_just_above_1_001():
    """The mass-gain criterion is WRF's ``break_drop_per > 1.001`` (:8423).

    No oracle record reaches that retry (the sweep's ndiv 2 and 4 come from
    the negativity retry, with accepted ratios 0.9985 and 0.9962), and the
    PKIJ x 1.1 runaway above sits far past any threshold, so this test
    places the ratio on each side of 1.001.  On sweep spectrum 6 at 12 s
    and 1000 hPa, PKIJ scaled by 1.003 gives a ratio of 1.00090 at ndiv 1,
    which WRF accepts; scaled by 1.004 it gives 1.00122 at ndiv 1, and the
    ratio grows with ndiv on this spectrum (1.00125 at ndiv 64), so the
    doubling never brings it back and the cap refuses.

    Breakage prevented: a criterion moved by orders of magnitude with the
    suite still green (``per > 1.01``, ``> 1.0001`` and ``> 1.00001`` each
    passed every test of this file before).  ``per >= 1.001`` differs from
    WRF only at a ratio of exactly 1.001 and is not reachable this way.
    """
    rec = next(r for r in _records("sweep_coal_bott_o64")
               if (int(r["spectrum"]), float(r["dt_coll"]), float(r["p_hpa"]))
               == (6, 12.0, 1000.0))
    args = (rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0,
            _cwll(1.0e6, 12.0))

    def scaled(factor):
        return dataclasses.replace(_tables(), pkij=_tables().pkij * factor)

    for preset in (FsbmBehavior.wrf_v480(), FsbmBehavior.corrected()):
        res = _coal(*args, tables=scaled(1.003), behavior=preset)
        assert (res.breakup_path, res.ndiv, res.ndiv_trials) == (1, 1, (1,))
        assert 1.0008 < res.break_drop_per < C.BREAKUP_MASS_GAIN_LIMIT
        # The first attempt at 1.004 already gains more than 0.1 percent:
        # with a cap of 1 the doubling is refused at once.
        one = dataclasses.replace(preset, breakup_retry_cap=1)
        with pytest.raises(FsbmRefusal) as info:
            _coal(*args, tables=scaled(1.004), behavior=one)
        assert "ndiv=1;" in info.value.condition
        assert "by 1.0012" in info.value.condition
        # ... and every doubling up to the cap gains more still.
        with pytest.raises(FsbmRefusal) as info:
            _coal(*args, tables=scaled(1.004), behavior=preset)
        assert "ndiv=64;" in info.value.condition
        assert "D7" in info.value.where
    assert C.BREAKUP_MASS_GAIN_LIMIT == 1.001


def test_breakup_that_stays_negative_at_ndiv_8_is_abandoned_as_wrf_does():
    # The stress spectrum at a 300 s collision step: every substep count up
    # to 8 leaves a negative bin, so WRF jumps to label 11 (:8403-8405) and
    # keeps the coalesced spectrum.
    rec = next(r for r in _records("sweep_coal_bott_o64")
               if (int(r["spectrum"]), float(r["dt_coll"])) == (14, 60.0))
    res = _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 300.0,
                _cwll(1.0e6, 300.0))
    assert (res.breakup_path, res.ndiv, res.ndiv_trials) == (2, 8, (1, 2, 4, 8))
    assert res.break_drop_per is None
    assert np.array_equal(res.g1_after_breakup, res.g1_after_xxx)
    xl = _tables().xl
    assert np.array_equal(res.ff1r, res.g1_after_xxx / (3.0 * xl * xl * 1.0e3))


def test_cold_points_and_ice_bins_are_refused_as_m3_scope():
    rec = _records("sweep_coal_bott_o64")[1]
    args = (rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0,
            _cwll(float(rec["pp"]), 12.0))
    for tt in (273.15, 250.0):
        with pytest.raises(FsbmRefusal, match="M3"):
            _coal(args[0], tt, *args[2:])
    ice = np.zeros(NKR)
    ice[5] = 1.0e-3
    for name in ("ff2r", "ff3r", "ff4r", "ff5r"):
        with pytest.raises(FsbmRefusal, match="M3"):
            _coal(*args, **{name: ice})
    zero = _coal(*args, ff3r=np.zeros(NKR), ff5r=np.zeros(NKR))
    assert np.array_equal(zero.ff1r, _coal(*args).ff1r)
    # Just above the cold block's bound the warm path runs.
    assert _coal(args[0], np.nextafter(273.15, 300.0), *args[2:]).icol_drop == 1


def test_functions_leave_their_inputs_alone_and_accept_a_kernels_object():
    rec = _records("sweep_coal_bott_o64")[1]
    ff = np.array(rec["ff1_in"])
    keep = ff.copy()
    cwll = _cwll(float(rec["pp"]), 12.0)
    cw_keep = cwll.copy()
    a = _coal(ff, rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0, cwll)
    b = _coal(ff, rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0,
              types.SimpleNamespace(cwll=cwll))
    assert np.array_equal(ff, keep) and np.array_equal(cwll, cw_keep)
    assert np.array_equal(a.ff1r, b.ff1r)
    g = np.array(a.g1_before_xxx)
    g_keep = g.copy()
    d = _derived()
    C.coll_xxx_lwf(g, np.ones(NKR), cwll, d.xl_mg, d.chucm, d.ima, 1.0)
    assert np.array_equal(g, g_keep)
    gt = g * 1.0e-3
    gt_keep = gt.copy()
    C.coll_breakup(gt, d.xl_mg * 1.0e-3, 12.0, _tables().pkij, _tables().qkj)
    assert np.array_equal(gt, gt_keep)


def test_the_module_imports_neither_derived_nor_cupy():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    assert not any("derived" in n or "cupy" in n for n in names), names


# ---------------------------------------------------------------------------
# Negative controls: each oracle gate can fail
# ---------------------------------------------------------------------------

def test_a_kernel_scaled_by_1_001_fails_the_probe_gate():
    worst = 0.0
    for _key, group in _probe_points():
        if "kernel" not in group:
            continue
        before, after = group["coal_before"], group["coal_after"]
        res = _coal(before["ff1"], before["tt"], before["qq"], before["pp"],
                    before["rhocgs"], before["dt_coll"], group["kernel"]["cwll"] * 1.001)
        worst = max(worst, _excess(res.ff1r, after["ff1"]))
    assert worst > 1.0e3, worst


def test_breakup_coefficients_scaled_by_1_001_fail_the_sweep_gate():
    t = _tables()
    worst = 0.0
    for rec in _records("sweep_breakup_o64", "case"):
        out = C.coll_breakup(rec["gt_in"], rec["xt"], float(rec["dt_call"]),
                             t.pkij * 1.001, t.qkj)
        worst = max(worst, _excess(out, rec["gt_out"]))
    assert worst > 1.0e3, worst


def test_a_breakup_probability_scaled_by_1_001_fails_the_spont_gate():
    d = _derived()
    worst = 0.0
    for rec in _records("sweep_spont_o64"):
        out = C.spont_rain_breakup(float(rec["dt"]), rec["ff_in"], rec["xl"],
                                   d.prob * 1.001, d.gain_var_new, d.ikr_spon_break)
        worst = max(worst, _excess(out, rec["ff_out"]))
    assert worst > 1.0e3, worst


# ---------------------------------------------------------------------------
# Analytic and conservation checks
# ---------------------------------------------------------------------------

#: Golovin (1963) kernel K = b (x + y) with b = 1500 cm3/(g s), liquid water
#: 1 g/m3 and an exponential spectrum of mean radius 10 um: the standard
#: case of Bott (1998, section 4).
GOLOVIN_B = 1.5e3
GOLOVIN_L = 1.0e-6        # g/cm3
GOLOVIN_R0 = 10.0e-4      # cm
GOLOVIN_DT = 12.0         # s, KC-1's collision step
GOLOVIN_STEPS = 300       # one hour
#: Pinned number-decay error, relative to the analytic exp(-b L t).
#: Measured on this grid: -0.076 at 10 min, -0.110 at 20, -0.127 at 30,
#: -0.136 at 40, -0.140 at 50 and -0.142 at 60 min, the same with 1 s and
#: 10 s steps to within 0.016.  The reason: the grid doubles mass per bin,
#: the coarsest Bott (1998) tested, and his flux method conserves mass but
#: not number; spreading each collected mass over bins k and k+1 diffuses
#: the spectrum toward large drops, so number falls faster than the
#: analytic solution.  0.15 leaves 0.008 above the measured hour.
GOLOVIN_NUMBER_TOLERANCE = 0.15


def _golovin_run(floored: bool):
    d = _derived()
    x_g = d.xl_mg * 1.0e-3
    x0 = 4.0 / 3.0 * math.pi * GOLOVIN_R0 ** 3 * 1.0
    n0 = GOLOVIN_L / x0
    # g(ln r) = 3 x^2 n(x) with n(x) = (N0/x0) exp(-x/x0), in mg/cm3.
    g = 3.0 * x_g ** 2 * (n0 / x0) * np.exp(-x_g / x0) * 1.0e3
    if floored:
        g = np.maximum(g, C.G_LIM_COLLISION * 1.0e3)
    dlnr = math.log(2.0) / 3.0
    ck = GOLOVIN_B * (x_g[:, None] + x_g[None, :]) * GOLOVIN_DT * dlnr
    mass0 = _seqsum(g)
    number0 = _seqsum(g / d.xl_mg)
    liquid = mass0 * dlnr * 1.0e-3          # g/cm3 on the grid
    mass_residual = 0.0
    number_error = []
    for n in range(1, GOLOVIN_STEPS + 1):
        g, _ = C.coll_xxx_lwf(g, np.ones(NKR), ck, d.xl_mg, d.chucm, d.ima, 1.0)
        mass_residual = max(mass_residual, abs(_seqsum(g) / mass0 - 1.0))
        if n % 50 == 0:
            t = n * GOLOVIN_DT
            ratio = _seqsum(g / d.xl_mg) / number0
            number_error.append(ratio / math.exp(-GOLOVIN_B * liquid * t) - 1.0)
    return mass_residual, number_error


def test_golovin_kernel_conserves_mass_and_follows_the_analytic_number_decay():
    residual, errors = _golovin_run(floored=True)
    # Every bin starts at or above WRF's gmin, so the floors at :736 and
    # :743 never add mass and the flux method's own conservation shows.
    assert residual <= 1.0e-12, residual
    assert len(errors) == 6
    for e in errors:
        assert -GOLOVIN_NUMBER_TOLERANCE <= e <= 0.0, errors
    assert errors == sorted(errors, reverse=True)  # the error grows with time


def test_golovin_without_the_floor_shows_wrfs_gmin_floors_adding_mass():
    # The same run from the plain exponential: the advancing edge enters
    # empty bins with fluxes below gmin, which :743 raises to gmin.  The
    # added mass is WRF's (measured 4.8e-10 of the total after an hour);
    # it never removes mass, and this is the check that the conservation
    # gate above can fail.
    residual, errors = _golovin_run(floored=False)
    assert 1.0e-12 < residual < 1.0e-9, residual
    floored = _golovin_run(floored=True)[1]
    assert np.allclose(errors, floored, atol=1.0e-3)


def _drop_mass(ff, xl) -> float:
    return _seqsum(3.0 * COL * np.asarray(xl) ** 2 * np.asarray(ff, dtype=np.float64))


def test_spontaneous_breakup_conserves_mass():
    d = _derived()
    xl = _tables().xl
    checked = 0
    inputs = [(float(r["dt"]), r["ff_in"]) for r in _records("sweep_spont_o64")]
    inputs += [(KC1_DT, g["spont_before"]["ff1"]) for _k, g in _probe_points()
               if "spont_before" in g]
    for dt, ff in inputs:
        out = C.spont_rain_breakup(dt, ff, xl, d.prob, d.gain_var_new, d.ikr_spon_break)
        before = _drop_mass(ff, xl)
        if before == 0.0:
            continue
        assert abs(_drop_mass(out, xl) / before - 1.0) <= 1.0e-12
        checked += 1
    assert checked > 100
    # Negative control: fragment weights 0.1 percent heavy break the balance.
    rec = next(r for r in _records("sweep_spont_o64") if int(r["taken"]) == 1)
    out = C.spont_rain_breakup(float(rec["dt"]), rec["ff_in"], xl, d.prob,
                               d.gain_var_new * 1.001, d.ikr_spon_break)
    assert abs(_drop_mass(out, xl) / _drop_mass(rec["ff_in"], xl) - 1.0) > 1.0e-12


def test_collisional_breakup_mass_residual_against_wrfs_1_001_criterion():
    """WRF accepts a breakup step when mass after over mass before is at most
    1.001 (:8423).  The residuals it actually produces, measured here on the
    host over the breakup and coalescence sweeps and every KC-1 breakup:

    * largest gain: 1.5e-6 (KC-1, step 80, column c2, level 25, ndiv 1),
      660 times inside the criterion; on the sweeps 1.8e-7 (cloud plus rain
      at a 30 s step);
    * losses up to 3.8e-3 (the ten-times stress spectrum at 60 s, accepted
      at ndiv 4); a loss never triggers a retry;
    * halving the substep moves the residual by at most 1.6e-4 (stress
      spectrum, ndiv 1 to 8), which is why a gain above 1.001 is not cured
      by WRF's doubling and D7's cap is needed.
    """
    t = _tables()
    ratios = []
    for rec in _records("sweep_coal_bott_o64"):
        dt = float(rec["dt_coll"])
        res = _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"],
                    dt, _cwll(float(rec["pp"]), dt))
        if res.break_drop_per is not None:
            ratios.append(res.break_drop_per)
    for _key, group in _probe_points():
        if "coal_before" not in group:
            continue
        before = group["coal_before"]
        cwll = group["kernel"]["cwll"] if "kernel" in group else             _cwll(float(before["pp"]), float(before["dt_coll"]))
        res = _coal(before["ff1"], before["tt"], before["qq"], before["pp"],
                    before["rhocgs"], before["dt_coll"], cwll)
        if res.break_drop_per is not None:
            ratios.append(res.break_drop_per)
    residual = np.asarray(ratios) - 1.0
    assert residual.size >= 80
    assert np.all(residual <= C.BREAKUP_MASS_GAIN_LIMIT - 1.0)
    assert 1.0e-6 < residual.max() < 2.0e-6
    assert -4.0e-3 < residual.min() < -3.5e-3
    spread = 0.0
    xt = _derived().xl_mg * 1.0e-3
    by_spectrum = defaultdict(dict)
    for rec in _records("sweep_breakup_o64", "sequence"):
        out = rec["gt_in"]
        for _ in range(int(rec["ndiv"])):
            out = C.coll_breakup(out, xt, float(rec["dt_call"]), t.pkij, t.qkj)
        by_spectrum[int(rec["spectrum"])][int(rec["ndiv"])] = \
            _seqsum(out) / _seqsum(rec["gt_in"])
    for per_ndiv in by_spectrum.values():
        spread = max(spread, max(per_ndiv.values()) - min(per_ndiv.values()))
    assert 1.0e-4 < spread < 2.0e-4, spread


# ---------------------------------------------------------------------------
# The tally for the emulator lane
# ---------------------------------------------------------------------------

def test_the_tally_changes_nothing_and_sums_to_what_each_bin_lost():
    d = _derived()
    for rec in _records("sweep_coll_xxx_o64", "case"):
        cwll = _sweep_kernels("o64")[(float(rec["p_z"]), float(rec["dt_coll"]))]
        plain = C.coll_xxx_lwf(rec["g_in"], np.ones(NKR), cwll, rec["x"], d.chucm,
                               d.ima, 1.0)
        tally = np.zeros((NKR, NKR))
        tallied = C.coll_xxx_lwf(rec["g_in"], np.ones(NKR), cwll, rec["x"], d.chucm,
                                 d.ima, 1.0, tally=tally)
        assert np.array_equal(plain[0], tallied[0])
        assert np.array_equal(plain[1], tallied[1])
        assert np.all(tally >= 0.0)
        g_in, g_out = np.asarray(rec["g_in"]), tallied[0]
        scale = float(np.max(np.abs(g_in)))
        removed = tally.sum(axis=1)
        received = g_out - g_in + removed
        # What a bin gained can only have come from coalescence into it.
        assert np.all(received >= -1.0e-15 * scale)
        # Bin 1 is nobody's target (ima >= 2): what it lost is all it changed.
        assert abs(removed[0] - (g_in[0] - g_out[0])) <= 1.0e-15 * scale
        # Pairs above bin nkr-1 never collide (:672-684).
        assert np.all(tally[NKR - 1, :] == 0.0) and np.all(tally[:, NKR - 1] == 0.0)


@pytest.mark.parametrize("i, j, fraction", [(3, 20, 0.01), (12, 12, 0.01),
                                             (3, 20, 100.0), (12, 12, 100.0)])
def test_the_tally_records_the_mass_each_bin_gives_one_pair(i, j, fraction):
    """One colliding pair, so every term of :693-710 can be written down.

    ``fraction`` sets the kernel so that ``x01`` is that fraction of the
    limiter ``g(i) x(j)`` (:694).  At 100 the limiters bind: a
    self-collection then takes ``gsi = gsj = g(i)``, twice what the bin
    holds, the clip at :705-709 removes only what is there, and the tally
    records that decrement while WRF adds the full ``gsk`` to the target.
    """
    d = _derived()
    x = d.xl_mg
    g_in = np.zeros(NKR)
    g_in[i - 1] = 2.0e-4
    g_in[j - 1] = 3.0e-4 if i != j else 2.0e-4
    ck = np.zeros((NKR, NKR))
    ck[i - 1, j - 1] = fraction * x[j - 1] / g_in[j - 1]   # only this pair
    tally = np.zeros((NKR, NKR))
    g_out, _ = C.coll_xxx_lwf(g_in, np.ones(NKR), ck, x, d.chucm, d.ima, 1.0,
                              tally=tally)
    k = int(d.ima[i - 1, j - 1])
    # :693-698 by hand
    x01 = ck[i - 1, j - 1] * g_in[i - 1] * g_in[j - 1] * 1.0
    x02 = min(x01, g_in[i - 1] * x[j - 1])
    x03 = min(x02, g_in[j - 1] * x[i - 1]) if j != k else x02
    gsi, gsj = x03 / x[j - 1], x03 / x[i - 1]
    expected = np.zeros((NKR, NKR))
    if i == j:
        assert k == i + 1
        expected[i - 1, i - 1] = min(gsi + gsj, g_in[i - 1])
        # The bin is nobody else's target, so its loss is observable.
        assert tally[i - 1, i - 1] == g_in[i - 1] - g_out[i - 1]
    else:
        assert k == j        # a smaller drop collected by a larger one
        expected[i - 1, j - 1] = min(gsi, g_in[i - 1])
        expected[j - 1, i - 1] = gsj
        assert tally[i - 1, j - 1] == g_in[i - 1] - g_out[i - 1]
    # The tally is the decrement WRF applies, g_before - g_after, so it
    # carries the rounding of that subtraction (1e-14 here) against the
    # hand-computed gsi and gsj.
    assert np.allclose(tally, expected, rtol=1.0e-12, atol=0.0)
    assert np.count_nonzero(tally) == (1 if i == j else 2)


def test_coalescence_conserves_mass_apart_from_wrfs_gmin_floors():
    """``coll_xxx_lwf`` moves mass between bins and adds only what the
    floors at gmin add (:736, :743): on the sweep spectra that is one floor
    on the cloud spectrum's advancing edge (0.92 gmin per call) and rounding
    elsewhere.  The Golovin tests carry the long-run version."""
    d = _derived()
    gmin = C.G_LIM_COLLISION * 1.0e3
    for rec in _records("sweep_coll_xxx_o64", "case"):
        cwll = _sweep_kernels("o64")[(float(rec["p_z"]), float(rec["dt_coll"]))]
        g, _ = C.coll_xxx_lwf(rec["g_in"], np.ones(NKR), cwll, rec["x"], d.chucm,
                              d.ima, 1.0)
        total = _seqsum(rec["g_in"])
        change = _seqsum(g) - total
        assert -1.0e-12 * total <= change <= gmin + 1.0e-12 * total, change


def test_coal_bott_warm_passes_the_tally_to_the_drop_drop_pairs():
    rec = next(r for r in _records("sweep_coal_bott_o64")
               if (int(r["spectrum"]), float(r["dt_coll"]), float(r["p_hpa"]))
               == (2, 12.0, 1000.0))
    cwll = _cwll(1.0e6, 12.0)
    tally = np.zeros((NKR, NKR))
    res = _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0,
                cwll, tally=tally)
    plain = _coal(rec["ff1_in"], rec["tt_in"], rec["qq"], rec["pp"], rec["rho"], 12.0,
                  cwll)
    assert np.array_equal(res.ff1r, plain.ff1r)
    direct = np.zeros((NKR, NKR))
    d = _derived()
    C.coll_xxx_lwf(res.g1_before_xxx, np.ones(NKR), cwll, d.xl_mg, d.chucm, d.ima,
                   1.0, tally=direct)
    assert np.array_equal(tally, direct) and tally.sum() > 0.0
    with pytest.raises(FsbmRefusal):
        C.coll_xxx_lwf(res.g1_before_xxx, np.ones(NKR), cwll, d.xl_mg, d.chucm,
                       d.ima, 1.0, tally=np.zeros((NKR, NKR), dtype=np.float32))
