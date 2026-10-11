"""Warm diffusional growth and evaporation (WRF FSBM-2 ``ONECOND1``) against WRF.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
* A transcription slip in the condensation path (a wrong literal, a lost
  term in the supersaturation solution, a remap index off by one, a sum
  reordered) that leaves drops growing plausibly but not as WRF grows them.
  Every host routine is compared with WRF's own Fortran promoted to float64
  (the O64 oracle build) on the oracle's sweeps, and ``ONECOND1`` on every
  call WRF made inside the KC-1 column at the probed points, to 1e-10
  relative, with the branch taken and ``IDROP`` compared exactly.
* A branch of ``JERSUPSAT_KS`` or ``JERNEWF_KS`` that the sweeps never
  reach, so a slip in it would pass unseen.  The coverage tests fail if the
  oracle records stop exercising any of them.
* An O32 sweep stream whose record n is not O64 record n's case, or that
  is not WRF's native float32 build at all.  Later comparisons with the
  native build (M2's precision measurement) pair records by index, so the
  pairing is checked here and the native spread recorded.  The host is not
  held to an O32 envelope here: run on the O64 inputs it is already within
  1e-10 of O64, so such an envelope could not fail.  The native envelope
  of design 5.5 (the host on O32 pre-call states against the O64 build on
  the same states) belongs to the column replay item, whose records carry
  those states.
* A WRF fatal check that the port lets through: each warm-path refusal case
  of the oracle, in both builds, must raise ``FsbmRefusal`` naming WRF's
  condition.
* Gates that cannot fail: latent heat 2501 instead of 2500 must fail the
  probe gate.
* WRF behaviour no oracle record reaches, pinned by construction so a slip
  in it cannot pass: the Kovetz-Olund step assigning a bin whose mass did
  not move instead of accumulating into it (:2541-2543), the 3-point step
  doing the same (:2592-2594), the ventilation law switching at Reynolds
  number 2.5 with ``<`` (:1882), and ``ONECOND1`` handing the final remap
  the ``IDROP`` its in-loop remap left (:6977-6979, :7087-7090).

Beside the oracle, analytic checks catch a transcription that happens to
match a mistranscribed oracle: the supersaturation solver against the closed
form and against ``scipy.integrate.solve_ivp`` of its linear system, the
series branches against the exact solution, monodisperse growth against the
``m**(2/3)`` law, the Kovetz-Olund remap against number and mass
conservation, and ``ONECOND1`` against water conservation with ``T``
changing by 2500 times the liquid change.  Two WRF behaviours of the
mixed-phase solver found by those checks are pinned as findings for the ice
milestone (M3), listed by id in fast-sbm design 3.7: S1, the dimensionally
inconsistent ``ALFA`` expansion of branches 8 and 9, and S2, the ``GAMA``
clamp of branch 7 that stops the slow mode decaying.  ``ONECOND1`` never
reaches the mixed-phase branches.

The analytic references run in 50-digit mpmath.  gpuwm does not declare
mpmath (pyproject lists neither it nor sympy, which brings it), so those
tests skip by name where it is absent; the oracle, refusal, coverage,
pinned-behaviour and hygiene gates never import it.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gpuwm.core.fast_sbm import condensation as C
from gpuwm.core.fast_sbm.constants import AL1, COL, NKR, FsbmRefusal
from gpuwm.core.fast_sbm.tables import load_tables
from tools.fast_sbm_wrf480_oracle import fixtures as fx

MODULE_PATH = Path(C.__file__)
REL = 1.0e-10
FLOOR = 1.0e-12

WATER_ONLY = (1, 0, 0, 0, 0, 0, 0)
BOTH_PHASES = (1, 1, 0, 0, 1, 0, 0)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _worst(got, exp, rel=REL, floor=FLOOR) -> float:
    """Largest ``|got - exp| / (rel |exp| + floor * scale)`` of one field.

    ``scale`` is the field's largest magnitude in the record (for a scalar,
    its own magnitude), so the floor only matters for the near-zero entries
    of an array.  A value above 1 fails the gate.
    """
    g = np.atleast_1d(np.asarray(got, dtype=np.float64))
    e = np.atleast_1d(np.asarray(exp, dtype=np.float64))
    assert g.shape == e.shape, (g.shape, e.shape)
    scale = float(np.max(np.abs(e))) if e.size else 0.0
    lim = rel * np.abs(e) + floor * scale
    err = np.abs(g - e)
    ratio = np.where(lim > 0, err / np.where(lim > 0, lim, 1.0),
                     np.where(err > 0, np.inf, 0.0))
    return float(np.max(ratio)) if ratio.size else 0.0


def _mpmath():
    """mpmath for a 50-digit analytic reference, or a skip that names why.

    mpmath is not a declared gpuwm dependency (the same position as
    ``tests/test_correctly_rounded_libm.py``), so a fresh ``pip install
    .[dev]`` venv lacks it.  Precision is set per use with ``workdps``,
    never globally.
    """
    return pytest.importorskip(
        "mpmath", reason="mpmath is not a declared gpuwm dependency; this "
        "50-digit analytic reference needs it")


def _records(name):
    return list(fx.iter_records(name))


def _liquid(f, xl, ror):
    # ONECOND1's liquid mixing ratio, COL3*RORI*sum(f x**2) (:7038-7054)
    s = 0.0
    for k in range(NKR):
        s = s + float(f[k]) * float(xl[k]) * float(xl[k])
    return s * (3.0 * COL) * (1.0 / float(ror))


@pytest.fixture(scope="module")
def tables():
    return load_tables("float64")


# Host callers, one per sweep family: record -> {field: host value}.

def _host_polysvp(r, tables):
    return {"es": C.polysvp(r["tt"], int(r["itype"]))}


def _host_jerrate(r, tables):
    return {"b11": C.jerrate_drops(r["xl"], r["tp"], r["pp"], r["vxl"],
                                   tables.rlec, tables.ro1bl)}


def _host_jersupsat(r, tables):
    s = C.jersupsat(r["del1"], r["del2"], r["rw_in"], r["pw_in"], r["ri_in"],
                    r["pi_in"], r["dt"], r["dyn1"], r["dyn2"], r["isym_in"])
    return {"del1n": s.del1n, "del2n": s.del2n, "del1int": s.del1int,
            "del2int": s.del2int, "rw_out": s.rw, "pw_out": s.pw,
            "ri_out": s.ri, "pi_out": s.pi}


def _host_jerdfun(r, tables):
    psi, xin, _ = C.jerdfun(r["xi"], r["b21"], r["fi"], r["del2n"], ind=1,
                            itype=1, tpn=r["tpn"], idrop=int(r["idrop_in"]))
    return {"psi": psi, "xin": xin}


def _host_jernewf(r, tables):
    psi, _ = C.jernewf(r["rr"], r["fi"], r["rn"], i3point=C.ISIGN_3POINT,
                       tpn=r["tpn"], idrop=int(r["idrop_in"]))
    return {"psi": psi}


def _sweep_isym(r):
    """The ISYM flags run_sweeps.F90 passes ``ONECOND1`` for a sweep record.

    ``sweep_onecond1`` sets ``ISYM1 = 1`` when the spectrum holds any drops
    and 0 otherwise, and every other flag to 0 (:532-534).  The record's own
    ``isym1`` is written after the call (:554), so it is ``ISYM1`` as
    ``ONECOND1`` left it, not the flag the harness passed.
    """
    return (1 if float(np.max(r["ff1_in"])) > 0.0 else 0, 0, 0, 0, 0, 0, 0)


def _host_onecond1(r, tables):
    # The sweep harness passes PSI1 = 0 (run_sweeps.F90, sweep_onecond1).
    s = C.onecond1(r["tt_in"], r["qq_in"], r["pp"], r["ror"], r["vr1"],
                   r["del1n_in"], r["del2n_in"], r["ff1_in"], tables,
                   r["dtcond"], psi1=np.zeros(NKR), isym=_sweep_isym(r))
    return {"tt_out": s.tt, "qq_out": s.qq, "del1n_out": s.del1n,
            "del2n_out": s.del2n, "ff1_out": s.ff1}


HOSTS = {
    "polysvp": _host_polysvp,
    "jerrate": _host_jerrate,
    "jersupsat": _host_jersupsat,
    "jerdfun": _host_jerdfun,
    "jernewf": _host_jernewf,
    "onecond1": _host_onecond1,
}

#: Inputs that define each family's cases, used to prove O32 record n is the
#: same case as O64 record n (to float32 rounding).  ONECOND1's qv and
#: supersaturations are left out: the harness derives them with WRF's
#: default-real constants, so O32 holds a different (float32-consistent)
#: state for the same target supersaturation.
INPUTS = {
    "polysvp": ("tt", "itype"),
    "jerrate": ("tp", "pp", "xl", "vxl"),
    "jersupsat": ("case_id", "del1", "del2", "rw_in", "pw_in", "ri_in",
                  "pi_in", "dt", "dyn1", "dyn2", "isym_in"),
    "jerdfun": ("spectrum", "del2n", "idrop_in", "tpn", "xi", "b21", "fi"),
    "jernewf": ("spectrum", "mode", "idrop_in", "tpn", "rr", "fi"),
    "onecond1": ("spectrum", "target_sat", "tt_in", "pp", "ror", "vr1",
                 "dtcond", "ff1_in"),
}

#: The WRF outputs each family's records hold, which the host callers return.
OUTPUTS = {
    "polysvp": ("es",),
    "jerrate": ("b11",),
    "jersupsat": ("del1n", "del2n", "del1int", "del2int", "rw_out", "pw_out",
                  "ri_out", "pi_out"),
    "jerdfun": ("psi", "xin"),
    "jernewf": ("psi",),
    "onecond1": ("tt_out", "qq_out", "del1n_out", "del2n_out", "ff1_out"),
}


def _gate(family, tables):
    """(worst ratio, where) of a family's O64 sweep over every record."""
    worst, where = 0.0, None
    for r in _records(f"sweep_{family}_o64"):
        host = HOSTS[family](r, tables)
        assert tuple(host) == OUTPUTS[family], (family, tuple(host))
        for field, value in host.items():
            q = _worst(value, r[field])
            if q > worst:
                worst, where = q, (r.index, field)
    return worst, where


# --------------------------------------------------------------------------
# O64 sweeps: the tight gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("family", sorted(HOSTS))
def test_o64_sweep_values(family, tables):
    worst, where = _gate(family, tables)
    assert worst <= 1.0, (
        f"sweep_{family}_o64 record {where[0]} field {where[1]}: error is "
        f"{worst:.3g} times the 1e-10 relative gate")


def test_jerrate_sweep_uses_the_packaged_drop_masses(tables):
    for r in _records("sweep_jerrate_o64"):
        assert np.array_equal(r["xl"], tables.xl)


def test_o64_jersupsat_branch_flags_and_isym(tables):
    for r in _records("sweep_jersupsat_o64"):
        s = C.jersupsat(r["del1"], r["del2"], r["rw_in"], r["pw_in"],
                        r["ri_in"], r["pi_in"], r["dt"], r["dyn1"], r["dyn2"],
                        r["isym_in"])
        assert s.branch == int(r["classified_branch"]), r.index
        assert int(s.g2_clamp) == int(r["g2_clamp"]), r.index
        assert s.isym == tuple(int(v) for v in r["isym_out"]), r.index


def test_jersupsat_sweep_reaches_every_branch_and_both_clamps():
    records = _records("sweep_jersupsat_o64")
    branches = {int(r["classified_branch"]) for r in records}
    assert branches == set(range(10)), sorted(branches)
    # The harness's intended branch agrees with the routine's predicates.
    for r in records:
        assert int(r["classified_branch"]) == int(r["intended_branch"]), r.index
    assert {int(r["g2_clamp"]) for r in records} == {0, 1}
    gama = [C.jersupsat(r["del1"], r["del2"], r["rw_in"], r["pw_in"],
                        r["ri_in"], r["pi_in"], r["dt"], r["dyn1"], r["dyn2"],
                        r["isym_in"]).gama_clamp for r in records]
    assert any(gama) and not all(gama)
    # Water-only records with a stale ice flag have it cleared.
    assert any(int(r["isym_in"][4]) == 1 and int(r["isym_out"][4]) == 0
               for r in records)


def test_o64_jerdfun_idrop(tables):
    seen = set()
    for r in _records("sweep_jerdfun_o64"):
        _, _, idrop = C.jerdfun(r["xi"], r["b21"], r["fi"], r["del2n"], ind=1,
                                itype=1, tpn=r["tpn"], idrop=int(r["idrop_in"]))
        assert idrop == int(r["idrop_out"]), r.index
        seen.add((int(r["idrop_in"]), idrop))
    # Evaporation turns IDROP off; growth keeps it.
    assert {(1, 0), (1, 1), (0, 0)} <= seen


def test_o64_jernewf_idrop():
    seen = set()
    for r in _records("sweep_jernewf_o64"):
        _, idrop = C.jernewf(r["rr"], r["fi"], r["rn"], tpn=r["tpn"],
                             idrop=int(r["idrop_in"]))
        assert idrop == int(r["idrop_out"]), r.index
        seen.add((int(r["idrop_in"]), idrop))
    assert {(1, 0), (1, 1), (0, 0)} <= seen


def test_jernewf_sweep_exercises_every_remap_path(monkeypatch):
    """The sweep would catch a slip in the 3-point step or the tail remap."""
    records = _records("sweep_jernewf_o64")

    def mismatches(**kw):
        n = 0
        for r in records:
            psi, _ = C.jernewf(r["rr"], r["fi"], r["rn"], tpn=r["tpn"],
                               idrop=int(r["idrop_in"]), **kw)
            n += _worst(psi, r["psi"]) > 1.0
        return n

    assert mismatches() == 0
    # The 3-point step changes the answer in some records ...
    assert mismatches(i3point=0) > 0
    # ... and so does the bins 6-12 tail merge.
    monkeypatch.setattr(C, "COEFF_REMAPING", 0.0)
    assert mismatches() > 0
    monkeypatch.undo()
    # Evaporation below bin 1 and a no-change record are both present.
    assert any(float(r["rn"][0]) < float(r["rr"][0]) * 0.5 for r in records)
    assert any(np.array_equal(r["rn"], r["rr"]) for r in records)


def test_o64_onecond1_branches_and_isym(tables):
    """Each sweep call as the harness made it, with ``ISYM1`` as WRF left it.

    The host is handed the flags the harness passed (:func:`_sweep_isym`),
    and its ``ISYM1`` on return must be the record's post-call ``isym1``.
    """
    kconds = set()
    for r in _records("sweep_onecond1_o64"):
        s = C.onecond1(r["tt_in"], r["qq_in"], r["pp"], r["ror"], r["vr1"],
                       r["del1n_in"], r["del2n_in"], r["ff1_in"], tables,
                       r["dtcond"], psi1=np.zeros(NKR), isym=_sweep_isym(r))
        assert s.isym[0] == int(r["isym1"]), r.index
        kconds.add(s.kcond)
    assert kconds == {C.KCOND_CONDENSATION, C.KCOND_EVAPORATION}


# --------------------------------------------------------------------------
# ONECOND1 inside FAST_SBM: the KC-1 probe records
# --------------------------------------------------------------------------

def _cond1_pairs():
    recs = list(fx.iter_probe("kc1_probe_o64",
                              types=["cond1_before", "cond1_after"]))
    pairs = []
    for before, after in zip(recs[0::2], recs[1::2]):
        assert before.type == "cond1_before" and after.type == "cond1_after"
        for key in ("step", "i", "k", "j"):
            assert int(before[key]) == int(after[key])
        pairs.append((before, after))
    return pairs


def _probe_gate(tables, al1=AL1):
    """Per probe record, the worst ratio over tt, qq, DEL1N, DEL2N and ff1."""
    out = []
    for b, a in _cond1_pairs():
        # FAST_SBM passes PP = pcgs (:4586), ROR = rhocgs, PSI1 = FF1IN, a
        # copy of FF1R (:4683, :4718): the default psi1.
        s = C.onecond1(b["tt"], b["qq"], b["pcgs"], b["rhocgs"], b["vr1"],
                       b["del1in"], b["del2in"], b["ff1"], tables, b["dtcond"],
                       al1=al1)
        out.append((max(_worst(s.tt, a["tt"]), _worst(s.qq, a["qq"]),
                        _worst(s.del1n, a["del1in"]),
                        _worst(s.del2n, a["del2in"]),
                        _worst(s.ff1, a["ff1"])), s, b))
    return out


def test_kc1_probe_onecond1_records(tables):
    gate = _probe_gate(tables)
    assert len(gate) == 57
    for worst, _, b in gate:
        assert worst <= 1.0, (
            f"ONECOND1 at step {int(b['step'])} column {int(b['i'])} level "
            f"{int(b['k'])}: {worst:.3g} times the gate")
    kinds = {(s.kcond, s.supsat.branch) for _, s, _ in gate}
    # Condensation and evaporation both occur, and so does the call where
    # JERSUPSAT_KS clears ISYM1 and PSI1 (= FF1IN) comes back unchanged.
    assert (C.KCOND_CONDENSATION, C.SUPSAT_WATER) in kinds
    assert any(k == C.KCOND_EVAPORATION for k, _ in kinds)
    assert any(br == C.SUPSAT_ALL_BELOW_MIN for _, br in kinds)


def test_negative_control_al1_2501_fails_the_probe_gate(tables):
    gate = _probe_gate(tables, al1=2501.0)
    failing = sum(worst > 1.0 for worst, _, _ in gate)
    # Only calls with no liquid change (ISYM1 cleared) can pass.
    assert failing >= 50, failing


# --------------------------------------------------------------------------
# O32: the same cases, from the native build
# --------------------------------------------------------------------------

@pytest.mark.parametrize("family", sorted(HOSTS))
def test_o32_sweep_is_the_o64_sweep_from_the_native_build(family, request):
    """O32 record n is O64 record n's case, and O32 is not O64's output.

    The breakage: an O32 stream whose records are not the O64 cases in the
    same order, or that holds the promoted build's output (the two builds
    swapped, or one binary run twice).  Comparisons with WRF's native build
    pair records by index, so either would compare the wrong thing without
    failing.  The inputs must agree to float32 rounding, at least one output
    must differ, and the largest output spread ``|O64 - O32| / scale`` is
    recorded in the test report's user properties as ``o32_spread_<family>``
    (with the record and field as ``o32_spread_<family>_at``).

    The host is not compared with O32 here (module docstring).
    """
    r64 = _records(f"sweep_{family}_o64")
    r32 = _records(f"sweep_{family}_o32")
    assert len(r64) == len(r32)
    spread, where = 0.0, None
    for a, b in zip(r64, r32):
        for name in INPUTS[family]:
            x, y = np.asarray(a[name], dtype=np.float64), np.asarray(b[name], dtype=np.float64)
            tol = 1.0e-6 * float(np.max(np.abs(x))) if x.size else 0.0
            assert np.allclose(x, y, rtol=1.0e-6, atol=tol), (family, a.index, name)
        for field in OUTPUTS[family]:
            x = np.atleast_1d(np.asarray(a[field], dtype=np.float64))
            y = np.atleast_1d(np.asarray(b[field], dtype=np.float64))
            scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
            if scale > 0.0:
                q = float(np.max(np.abs(x - y))) / scale
                if q > spread:
                    spread, where = q, (a.index, field)
    request.node.user_properties.append((f"o32_spread_{family}", spread))
    request.node.user_properties.append((f"o32_spread_{family}_at", repr(where)))
    assert spread > 0.0, (
        f"sweep_{family}_o32 equals sweep_{family}_o64 in every output: it is "
        "not the native build")


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

WARM_ROUTINES = ("ONECOND1", "JERDFUN_KS/JERNEWF_KS")


@pytest.mark.parametrize("build", ["o64", "o32"])
def test_warm_path_refusal_fixtures(build, tables):
    cases = [ref for ref in fx.iter_refusals(build=build)
             if ref.routine in WARM_ROUTINES]
    assert len(cases) == 3
    for ref in cases:
        assert ref.exit_status == 99 and ref.message
        (rec,) = ref.records
        with pytest.raises(FsbmRefusal) as info:
            if rec.type == "onecond1":
                C.onecond1(rec["tt"], rec["qq"], rec["pp"], rec["ror"],
                           rec["vr1"], rec["del1n"], rec["del2n"], rec["ff1"],
                           tables, rec["dtcond"], psi1=np.zeros(NKR))
            else:
                # JERDFUN_KS with IND 2: new masses xi + DEL2N*B21 (:2395-2398)
                C.jerdfun(rec["rr"], rec["b21"], rec["fi"], rec["del2n"],
                          ind=2, itype=1, tpn=288.15, idrop=int(rec["idrop"]))
        assert info.value.condition.startswith(ref.message), (
            ref.case, ref.message, info.value.condition)


def test_polysvp_refuses_an_undefined_type():
    with pytest.raises(FsbmRefusal, match="ITYPE"):
        C.polysvp(290.0, 2)


def test_jersupsat_refuses_a_negative_discriminant():
    with pytest.raises(FsbmRefusal) as info:
        C.jersupsat(0.01, 0.05, 0.5, 0.4, -10.0, 0.6, 4.0, 0.0, 0.0, BOTH_PHASES)
    assert info.value.condition.startswith("fatal error: STOP 1905:A < 0, model stop")


def test_jersupsat_refuses_a_malformed_flag_set():
    with pytest.raises(FsbmRefusal, match="7"):
        C.jersupsat(0.01, 0.05, 0.5, 0.4, 0.0, 0.0, 4.0, 0.0, 0.0, (1, 0, 0))


def _cloud(tables):
    r = _records("sweep_onecond1_o64")[0]
    return dict(tt=r["tt_in"], qq=r["qq_in"], pp=r["pp"], ror=r["ror"],
                vr1=r["vr1"], del1n=r["del1n_in"], del2n=r["del2n_in"],
                ff1=r["ff1_in"], tables=tables, isym=_sweep_isym(r))


@pytest.mark.parametrize("del1n, text", [
    (0.005, "fatal error in ONECOND1-DEL1N>0:(DTT<0), model stop"),
    (-0.005, "fatal error in ONECOND1-DEL1N<0:(DTT<0), model stop"),
])
def test_onecond1_refuses_a_negative_substep(del1n, text, tables):
    args = _cloud(tables)
    args["del1n"] = del1n
    with pytest.raises(FsbmRefusal) as info:
        C.onecond1(dtcond=-1.0, **args)
    assert info.value.condition.startswith(text)


def test_onecond1_refuses_nan_rates(tables):
    args = _cloud(tables)
    args["tt"] = float("nan")
    with pytest.raises(FsbmRefusal) as info:
        C.onecond1(dtcond=4.0, **args)
    assert info.value.condition.startswith(
        "fatal error in ONECOND1 (RW or PW are NaN), model stop")


@pytest.mark.parametrize("del1n, flipped, text", [
    (0.005, -0.01, "fatal error in ONECOND1-1 (DEL1.GT.0.AND.DEL1N.LT.0), model stop"),
    (-0.005, 0.01, "fatal error in ONECOND1-2 (DEL1.LT.0.AND.DEL1N.GT.0), model stop"),
])
def test_onecond1_refuses_a_supersaturation_sign_flip(monkeypatch, del1n,
                                                      flipped, text, tables):
    # With DYN = 0 the relaxation cannot flip the sign (module docstring), so
    # the check is driven through a stand-in solver that does.
    real = C.jersupsat

    def flipping(*a, **k):
        res = real(*a, **k)
        return C.JersupsatResult(**{**res.__dict__, "del1n": flipped})

    monkeypatch.setattr(C, "jersupsat", flipping)
    args = _cloud(tables)
    args["del1n"] = del1n
    with pytest.raises(FsbmRefusal) as info:
        C.onecond1(dtcond=4.0, **args)
    assert info.value.condition.startswith(text)
    # Below EPSDEL the flip passes.
    monkeypatch.setattr(C, "jersupsat", lambda *a, **k: C.JersupsatResult(
        **{**real(*a, **k).__dict__, "del1n": math.copysign(0.5e-4, flipped)}))
    C.onecond1(dtcond=4.0, **args)


def test_onecond1_refuses_the_final_temperature_jump(monkeypatch, tables):
    # The 5 K end check cannot fire unless the 3 K in-loop check already has
    # (module docstring); a stand-in final remap that doubles the spectrum
    # drives it.
    real = C.jerdfun_new
    monkeypatch.setattr(C, "jerdfun_new",
                        lambda *a, **k: (lambda p: (p[0] * 40.0, p[1]))(real(*a, **k)))
    with pytest.raises(FsbmRefusal) as info:
        C.onecond1(dtcond=4.0, **_cloud(tables))
    assert info.value.condition.startswith(
        "fatal error in ONECOND1-out (ABS(DAL1*DELMASSL1) > 5.0), model stop")


# --------------------------------------------------------------------------
# Analytic checks of the supersaturation solver
# --------------------------------------------------------------------------

def _exact(d1, d2, rw, pw, ri, pi, dt, y1, y2):
    """Exact (D1, D2, int D1, int D2) of dD/dt = Y - M D, M = [[RW, RI], [PW, PI]].

    The matrix exponential of the augmented 5x5 system in 50-digit mpmath,
    independent of every formula in the routine.
    """
    mp = _mpmath()
    with mp.workdps(50):
        a = mp.zeros(5, 5)
        a[0, 0], a[0, 1], a[0, 4] = -mp.mpf(rw), -mp.mpf(ri), mp.mpf(y1)
        a[1, 0], a[1, 1], a[1, 4] = -mp.mpf(pw), -mp.mpf(pi), mp.mpf(y2)
        a[2, 0] = 1
        a[3, 1] = 1
        z = mp.expm(a * mp.mpf(dt)) * mp.matrix([mp.mpf(d1), mp.mpf(d2), 0, 0, 1])
        return [float(z[i]) for i in range(4)]


def _outputs(s):
    return [s.del1n, s.del2n, s.del1int, s.del2int]


@pytest.mark.parametrize("dt, dyn1, dyn2", [
    (4.0, 0.0, 0.0), (4.0, 1.0e-4, -2.0e-4), (1.0, 3.0e-3, 1.0e-3),
])
@pytest.mark.parametrize("d1, d2", [(0.01, 0.05), (-0.2, -0.15)])
def test_jersupsat_water_branch_matches_the_closed_form(d1, d2, dt, dyn1, dyn2):
    """Water only: D1 relaxes exponentially, D2 follows from PW D1 (closed form)."""
    mp = _mpmath()
    rw, pw = 0.5, 0.4
    s = C.jersupsat(d1, d2, rw, pw, 0.0, 0.0, dt, dyn1, dyn2, WATER_ONLY)
    assert s.branch == C.SUPSAT_WATER
    with mp.workdps(50):
        R, P, T = mp.mpf(rw), mp.mpf(pw), mp.mpf(dt)
        D1, D2, Y1, Y2 = mp.mpf(d1), mp.mpf(d2), mp.mpf(dyn1), mp.mpf(dyn2)
        e = mp.exp(-R * T)
        n1 = D1 * e + Y1 / R * (1 - e)
        i1 = D1 * (1 - e) / R + Y1 / R * (T - (1 - e) / R)
        ii1 = D1 / R * (T - (1 - e) / R) + Y1 / R * (T * T / 2 - (T - (1 - e) / R) / R)
        n2 = D2 + Y2 * T - P * i1
        i2 = D2 * T + Y2 * T * T / 2 - P * ii1
        wants = [float(v) for v in (n1, n2, i1, i2)]
    for got, want in zip(_outputs(s), wants):
        assert abs(got - want) <= 1.0e-12 * abs(want), (got, want)


@pytest.mark.parametrize("dt", [4.0, 1.0])
@pytest.mark.parametrize("dyn1, dyn2", [(0.0, 0.0), (1.0e-4, -2.0e-4)])
@pytest.mark.parametrize("rates", [
    (0.5, 0.4, 0.3, 0.6),     # G2 > 0: both eigenvalues decay (physical)
    (1.2, 0.2, 0.05, 0.3),
])
def test_jersupsat_mixed_branch_matches_solve_ivp(rates, dyn1, dyn2, dt):
    rw, pw, ri, pi = rates
    d1, d2 = 0.01, 0.05
    s = C.jersupsat(d1, d2, rw, pw, ri, pi, dt, dyn1, dyn2, BOTH_PHASES)
    assert s.branch == C.SUPSAT_MIXED and not s.g2_clamp

    def rhs(t, z):
        return [dyn1 - rw * z[0] - ri * z[1], dyn2 - pw * z[0] - pi * z[1],
                z[0], z[1]]

    sol = solve_ivp(rhs, (0.0, dt), [d1, d2, 0.0, 0.0], method="DOP853",
                    rtol=1.0e-13, atol=1.0e-18)
    assert sol.success
    for got, want in zip(_outputs(s), sol.y[:, -1]):
        assert abs(got - want) <= 1.0e-10 * abs(want), (got, want)


def _branch7_rates(gama):
    # Rates with eigenvalues beta ~ 1.1 and -gama: G2 = RW PI - RI PW = -beta gama
    rw, ri, pi = 0.5, 0.3, 0.6
    return rw, (rw * pi + 1.1 * gama) / ri, ri, pi


@pytest.mark.parametrize("case, branch", [
    ((0.01, 0.05, 1.0e-8, 2.0e-8, 0.0, 0.0), C.SUPSAT_WATER_SERIES),
    ((-0.2, -0.1, 3.0e-9, 1.0e-8, 0.0, 0.0), C.SUPSAT_WATER_SERIES),
    ((-0.01, 0.05, 0.0, 0.0, 1.0e-8, 2.0e-8), C.SUPSAT_ICE_SERIES),
    ((0.01, 0.05) + _branch7_rates(1.0e-8), C.SUPSAT_MIXED_SERIES),
    ((0.01, 0.05) + _branch7_rates(3.0e-8), C.SUPSAT_MIXED_SERIES),
])
def test_jersupsat_series_branches_match_the_exact_solution(case, branch):
    """Small |rate DT| branches (fifth-order EXPM1) against the exact solution.

    Every caller of JERSUPSAT_KS in WRF passes DYN1 = DYN2 = 0 (:6965,
    :7008, :7462, :7545, :8059), which these cases use.  (With forcing,
    the water series branch loses about 1e-10 relative to cancellation
    between its DYN1*DT/RW and DYN1*EXPR/RW**2 terms.)
    """
    d1, d2, rw, pw, ri, pi = case
    isym = WATER_ONLY if branch == C.SUPSAT_WATER_SERIES else BOTH_PHASES
    s = C.jersupsat(d1, d2, rw, pw, ri, pi, 4.0, 0.0, 0.0, isym)
    assert s.branch == branch and not s.gama_clamp
    for got, want in zip(_outputs(s), _exact(d1, d2, s.rw, s.pw, s.ri, s.pi,
                                             4.0, 0.0, 0.0)):
        assert abs(got - want) <= 1.0e-12 * abs(want), (got, want)


@pytest.mark.parametrize("rates, branch", [
    ((1.0e-19, 1.0e-9, 1.0e-6, 0.9e-5), C.SUPSAT_MIXED_SMALL_RW),
    ((3.0e-19, 2.0e-9, 5.0e-6, 2.0e-5), C.SUPSAT_MIXED_SMALL_RW),
    ((1.0e-6, 1.0e-9, 1.0e-19, 2.0e-13), C.SUPSAT_MIXED_SMALL_RI),
    ((4.0e-6, 3.0e-9, 2.0e-19, 1.0e-12), C.SUPSAT_MIXED_SMALL_RI),
])
def test_finding_small_ratio_branches_expand_alfa_in_the_wrong_variable(rates, branch):
    """Finding S1 for M3 (design 3.7): branches 8 and 9 (:2252-2254, :2286-2288).

    ``ALFA = sqrt(PI**2 + X)`` is expanded as ``PI*(1 + (X/PI)/2 - ...)``,
    where the expansion variable is ``X/PI**2``; branch 9 does the same with
    ``RW``.  The terms do not share units, so ALFA is wrong by about
    ``X (1/PI - 1)/(2 PI)`` relative: 2.5e-5 on the oracle's own branch 8
    case.  Written out, WRF has ``PI + X/2`` where ``PI + X/(2 PI)`` is
    right; the first-order terms of ``BETA`` and ``GAMA`` are right and only
    their second-order terms carry the slip.  The port reproduces WRF (the sweep gate above holds it to O64);
    this pins that the whole departure from the exact solution is that ALFA,
    so M3 can decide the fix with the number in hand.  ONECOND1 never
    reaches these branches: its ice rates are zero.
    """
    mp = _mpmath()
    rw, pw, ri, pi = rates
    s = C.jersupsat(0.01, 0.05, rw, pw, ri, pi, 4.0, 0.0, 0.0, BOTH_PHASES)
    assert s.branch == branch
    exact = _exact(0.01, 0.05, s.rw, s.pw, s.ri, s.pi, 4.0, 0.0, 0.0)
    with mp.workdps(50):
        RW, PW, RI, PI = (mp.mpf(v) for v in (s.rw, s.pw, s.ri, s.pi))
        alfa_true = mp.sqrt((RW - PI) ** 2 + 4 * PW * RI)
        if branch == C.SUPSAT_MIXED_SMALL_RW:
            x = -2 * RW * PI + RW * RW + 4 * PW * RI
            alfa_wrf = PI * (1 + (x / PI) / 2 - (x / PI) ** 2 / 8)
        else:
            x = -2 * RW * PI + PI * PI + 4 * PW * RI
            alfa_wrf = RW * (1 + (x / RW) / 2 - (x / RW) ** 2 / 8)
        ratio = float(alfa_wrf / alfa_true)
    assert abs(ratio - 1.0) > 1.0e-7
    for got, want in zip(_outputs(s), exact):
        assert abs(got - want) > 1.0e-8 * abs(want)          # WRF's departure
        assert abs(got * ratio - want) <= 1.0e-8 * abs(want)  # all of it ALFA


def test_finding_branch7_gama_clamp_stops_the_slow_mode_decaying():
    """Finding S2 for M3 (design 3.7): the (KS) 2016 clamp at :2202.

    With both phases present the physical rates give ``G2 > 0`` (the
    ``OPER2`` terms make ``RW PI - RI PW`` proportional to
    ``(B5L - B5I)(AL1 - AL2) > 0``), so ``GAMA`` is negative and the clamp
    fires every time branch 7 (``|GAMA DT| <= 1e-6``) runs: ``GAMA``
    becomes +5e-14, so the slow mode neither decays nor grows over the step,
    an error of order ``|GAMA| DT`` relative, at most 1e-6.  ONECOND1 never
    reaches this branch.
    """
    gama = -1.0e-8
    rw, pw, ri, pi = _branch7_rates(gama)
    s = C.jersupsat(0.01, 0.05, rw, pw, ri, pi, 4.0, 0.0, 0.0, BOTH_PHASES)
    assert s.branch == C.SUPSAT_MIXED_SERIES and s.gama_clamp
    assert not s.g2_clamp
    for got, want in zip(_outputs(s), _exact(0.01, 0.05, rw, pw, ri, pi, 4.0,
                                             0.0, 0.0)):
        rel = abs(got - want) / abs(want)
        assert 0.1 * abs(gama) * 4.0 < rel < 2.0 * abs(gama) * 4.0, rel


# --------------------------------------------------------------------------
# Analytic checks of growth and remap
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bin_", [1, 5, 14, 22, 30])
@pytest.mark.parametrize("integral", [2.0e-3, 0.5, -1.0e-3, -0.3])
def test_monodisperse_growth_follows_the_m23_law(bin_, integral, tables):
    """dm/dt = k m**(1/3) S gives m**(2/3) = m0**(2/3) + (2/3) k int S.

    ``B = k m**(1/3)`` is the growth coefficient; the host must return
    ``(m**(2/3) + (2/3) B I / m**(1/3))**(3/2)``, evaluated here in 50-digit
    arithmetic, and the Kovetz-Olund remap must put the one bin's number and
    mass on the two grid masses around the new mass.
    """
    mp = _mpmath()
    xl = np.asarray(tables.xl, dtype=np.float64)
    k = 1.0e-7
    with mp.workdps(50):
        b = np.array([k * float(mp.cbrt(mp.mpf(float(x)))) for x in xl])
    fi = np.zeros(NKR)
    fi[bin_ - 1] = 1.0e3
    psi, xin, _ = C.jerdfun(xl, b, fi, integral, ind=1, itype=1, tpn=290.0,
                            idrop=0)
    with mp.workdps(50):
        for kr in range(NKR):
            m = mp.mpf(float(xl[kr]))
            bracket = m ** (mp.mpf(2) / 3) + (mp.mpf(2) / 3) * mp.mpf(float(b[kr])) \
                * mp.mpf(integral) / m ** (mp.mpf(1) / 3)
            want = 1.0e-50 if bracket < 0 else float(bracket ** mp.mpf(1.5))
            assert abs(xin[kr] - want) <= 1.0e-14 * want, (kr, xin[kr], want)
    m_new = xin[bin_ - 1]
    number0 = fi[bin_ - 1] * xl[bin_ - 1]
    if xl[0] <= m_new <= xl[-1]:
        assert np.count_nonzero(psi) <= 2
        assert abs(float(np.dot(psi, xl)) - number0) <= 1.0e-13 * number0
        assert (abs(float(np.dot(psi, xl * xl)) - number0 * m_new)
                <= 1.0e-13 * number0 * m_new)


def test_kovetz_olund_remap_conserves_number_and_mass(tables):
    """The Kovetz-Olund step keeps number and mass (Kovetz and Olund 1969).

    With IDROP = 0 neither the 3-point step nor the tail merge runs, so the
    remap is the pure two-bin split: number sum(psi x) equals sum(f x) and
    mass sum(psi x**2) equals sum(f x m_new), for every new mass inside the
    grid, growth and evaporation alike.
    """
    xl = np.asarray(tables.xl, dtype=np.float64)
    rng = np.random.default_rng(20260923)
    for trial in range(200):
        fi = rng.lognormal(mean=0.0, sigma=3.0, size=NKR) * 10.0 ** rng.uniform(-8, 4, NKR)
        fi[rng.random(NKR) < 0.2] = 0.0
        factor = rng.uniform(0.3, 3.5, NKR) if trial % 2 else rng.uniform(0.6, 1.9, NKR)
        rn = np.clip(xl * factor, xl[0], xl[-1])
        # A bin whose mass moved by less than 1e-16 g is assigned, not
        # accumulated (PSINEW(K) = FI(K)*RR(K), :2541-2543), which drops what
        # lower bins moved into it; clipping can create such bins at the
        # grid ends, so they are left empty here.
        fi[np.abs(rn - xl) < 1.0e-16] = 0.0
        psi, idrop = C.jernewf(xl, fi, rn, i3point=C.ISIGN_3POINT, idrop=0)
        assert idrop == 0
        number_before = float(np.dot(fi, xl))
        mass_moved = float(np.dot(fi * xl, rn))
        assert abs(float(np.dot(psi, xl)) - number_before) <= 1.0e-13 * number_before
        assert abs(float(np.dot(psi, xl * xl)) - mass_moved) <= 1.0e-13 * mass_moved


def _onecond1_cases(tables):
    for r in _records("sweep_onecond1_o64"):
        yield (dict(tt=r["tt_in"], qq=r["qq_in"], pp=r["pp"], ror=r["ror"],
                    vr1=r["vr1"], del1n=r["del1n_in"], del2n=r["del2n_in"],
                    ff1=r["ff1_in"], tables=tables, dtcond=r["dtcond"],
                    isym=_sweep_isym(r)),
               np.zeros(NKR))
    for b, _ in _cond1_pairs():
        yield (dict(tt=b["tt"], qq=b["qq"], pp=b["pcgs"], ror=b["rhocgs"],
                    vr1=b["vr1"], del1n=b["del1in"], del2n=b["del2in"],
                    ff1=b["ff1"], tables=tables, dtcond=b["dtcond"]), None)


def test_onecond1_conserves_water_and_heats_by_2500_times_the_liquid(tables):
    """qv plus liquid is unchanged and T moves by AL1 = 2500 times the liquid change.

    Liquid is ONECOND1's own ``3 COL / ROR sum(f x**2)`` (:7093-7117); the
    tolerances are a few units in the last place of qv + liquid and of T.
    """
    n = 0
    for args, psi1 in _onecond1_cases(tables):
        s = C.onecond1(**args, psi1=psi1)
        l0 = _liquid(args["ff1"], tables.xl, args["ror"])
        l1 = _liquid(s.ff1, tables.xl, args["ror"])
        total0 = float(args["qq"]) + l0
        assert abs((s.qq + l1) - total0) <= 4.0 * np.spacing(total0), (s.qq, l1)
        dt_expected = 2500.0 * (l1 - l0)
        assert abs((s.tt - float(args["tt"])) - dt_expected) <= (
            4.0 * np.spacing(float(args["tt"])) + 1.0e-12 * abs(dt_expected))
        n += 1
    assert n == 40 + 57


# --------------------------------------------------------------------------
# WRF behaviour no oracle record reaches, pinned by construction
# --------------------------------------------------------------------------

def _split(rn, lo, hi):
    """Kovetz-Olund shares ``(GMAT, GMAT2)`` of a new mass between two grid
    masses: ``GMAT`` to ``lo`` and ``GMAT2`` to ``hi``, keeping number and
    mass (Kovetz and Olund 1969; WRF :2560-2566)."""
    return (hi - rn) / (hi - lo), (rn - lo) / (hi - lo)


def test_kovetz_olund_assigns_a_bin_whose_mass_did_not_move(tables):
    """WRF assigns ``PSINEW(K) = FI(K)*RR(K)`` for an unmoved bin (:2541-2543).

    The breakage: the assignment turned into an accumulation (or the reverse
    edit elsewhere), which every sweep record would pass because none has a
    lower bin moving into an unmoved one.  Bin 20 grows a quarter of the way
    to bin 21; bin 21 does not move; bin 22 evaporates a third of the way
    back to bin 21.  In WRF's loop order bin 20's share into bin 21 is added
    first and then overwritten, so it is lost; bin 22's share comes after
    the assignment and is kept.  ``IDROP = 0`` keeps the 3-point step and
    the tail merge out, so the result is the Kovetz-Olund step alone.
    """
    xl = np.asarray(tables.xl, dtype=np.float64)
    j, k, u = 19, 20, 21                       # bins 20, 21, 22 (zero-based)
    fi = np.zeros(NKR)
    fi[j], fi[k], fi[u] = 3.0e-2, 5.0e-3, 2.0e-3
    rn = xl.copy()
    rn[j] = xl[j] + 0.25 * (xl[k] - xl[j])
    rn[u] = xl[u] - (xl[u] - xl[k]) / 3.0
    psi, idrop = C.jernewf(xl, fi, rn, i3point=C.ISIGN_3POINT, idrop=0)
    assert idrop == 0

    gj, gj2 = _split(rn[j], xl[j], xl[k])
    gu, gu2 = _split(rn[u], xl[k], xl[u])
    lost = fi[j] * xl[j] * gj2                 # bin 20's share into bin 21
    expected = np.zeros(NKR)
    expected[j] = fi[j] * xl[j] * gj / xl[j]
    expected[k] = (fi[k] * xl[k] + fi[u] * xl[u] * gu) / xl[k]
    expected[u] = fi[u] * xl[u] * gu2 / xl[u]
    assert np.allclose(psi, expected, rtol=1.0e-14, atol=0.0), (psi[j:u + 1],
                                                                expected[j:u + 1])
    # Number is short by exactly the share the assignment dropped.
    number_lost = float(np.dot(fi, xl)) - float(np.dot(psi, xl))
    assert abs(number_lost - lost) <= 1.0e-12 * lost, (number_lost, lost)


def test_three_point_step_reassigns_a_bin_whose_mass_did_not_move(tables):
    """The 3-point step writes ``PSI(K) = FI(K)*RR(K)`` again (:2592-2594).

    The breakage: that line dropped, or turned into an accumulation; no
    sweep record reaches it with anything already in ``PSI(K)``.  Bin 21
    does not move and bin 22 evaporates a third of the way back to it, so
    the Kovetz-Olund step leaves bin 22's share in bin 21.  With
    ``IDROP = 1`` the 3-point step runs, reassigns bin 21 and so drops that
    share; at bin 22 its lower weight is negative with nothing below, so it
    leaves the loop (:2638-2663) without changing anything.  The bins are
    above 12, so the tail merge over bins 6-12 changes nothing either.
    """
    xl = np.asarray(tables.xl, dtype=np.float64)
    k, u = 20, 21                              # bins 21, 22 (zero-based)
    fi = np.zeros(NKR)
    fi[k], fi[u] = 5.0e-3, 2.0e-3
    rn = xl.copy()
    rn[u] = xl[u] - (xl[u] - xl[k]) / 3.0
    psi, idrop = C.jernewf(xl, fi, rn, i3point=C.ISIGN_3POINT, idrop=1)
    assert idrop == 1

    gu, gu2 = _split(rn[u], xl[k], xl[u])
    lost = fi[u] * xl[u] * gu                  # bin 22's share into bin 21
    expected = np.zeros(NKR)
    expected[k] = fi[k] * xl[k] / xl[k]
    expected[u] = fi[u] * xl[u] * gu2 / xl[u]
    assert np.allclose(psi, expected, rtol=1.0e-14, atol=0.0), (psi[k:u + 1],
                                                                expected[k:u + 1])
    number_lost = float(np.dot(fi, xl)) - float(np.dot(psi, xl))
    assert abs(number_lost - lost) <= 1.0e-12 * lost, (number_lost, lost)
    # Without the 3-point step the same spectrum keeps that share.
    psi_ko, _ = C.jernewf(xl, fi, rn, i3point=0, idrop=1)
    assert abs(float(np.dot(psi_ko, xl)) - float(np.dot(fi, xl))) <= (
        1.0e-13 * float(np.dot(fi, xl)))


def test_ventilation_switches_law_at_reynolds_2_5_with_less_than(tables):
    """``Re < 2.5`` takes the low law and ``Re = 2.5`` the high one (:1882).

    The breakage: ``<=`` written for ``<``.  The two ventilation laws,
    ``1 + 0.108 Re Sc**(2/3)`` and ``0.78 + 0.308 Re**(1/2) Sc**(1/3)``
    (:1883, :1886), do not meet at 2.5 (about 1e-3 apart here), so the slip
    moves exactly that Reynolds number onto the wrong law, and no sweep
    record sits on it.  For each bin where one exists, the slowest fall
    speed is found at which WRF's Reynolds expression ``B*V*(X/RO)**(1/3)``
    (:1878-1879, with ``B`` from :1847-1848) is 2.5 exactly in float64; the
    growth rate there must continue the high law from the next speed up and
    jump from the next speed down, where the Reynolds number is below 2.5.
    """
    r = _records("sweep_jerrate_o64")[0]
    xl = [float(v) for v in r["xl"]]
    ro = [float(v) for v in tables.ro1bl]
    deg03 = 1.0 / 3.0
    b = 2.0 * math.pow(3.0 / 4.0 / 3.141593, deg03) / 0.13

    def rate(kr, v):
        vxl = np.array(r["vxl"], dtype=np.float64)
        vxl[kr] = v
        return float(C.jerrate_drops(r["xl"], r["tp"], r["pp"], vxl,
                                     tables.rlec, tables.ro1bl)[kr])

    checked = 0
    for kr in range(NKR):
        p = math.pow(xl[kr] / ro[kr], deg03)
        v0 = 2.5 / (b * p)
        hits = [v for v in (v0, *(math.nextafter(v0, d) for d in (math.inf, -math.inf)))
                if b * v * p == 2.5]
        if not hits:
            continue
        # Neighbouring speeds can round to the same Reynolds number, so step
        # down to the slowest speed that still gives 2.5 exactly; the next
        # speed down then gives less than 2.5.
        v = hits[0]
        while b * math.nextafter(v, -math.inf) * p == 2.5:
            v = math.nextafter(v, -math.inf)
        assert b * math.nextafter(v, -math.inf) * p < 2.5
        assert b * math.nextafter(v, math.inf) * p >= 2.5
        at = rate(kr, v)
        above = rate(kr, math.nextafter(v, math.inf))
        below = rate(kr, math.nextafter(v, -math.inf))
        assert abs(at - above) <= 1.0e-14 * abs(above), (kr, at, above)
        assert abs(at - below) > 1.0e-4 * abs(below), (kr, at, below)
        checked += 1
    assert checked >= 10, checked


def test_onecond1_hands_the_final_remap_the_in_loop_idrop(monkeypatch, tables):
    """``JERDFUN_NEW_KS`` gets the ``IDROP`` the in-loop remap left (:7087-7090).

    ``ONECOND1`` sets ``IDROP = ISYM1`` and lets ``JERDFUN_KS`` turn it off
    on evaporation (:6977-6979, :7020-7023), then hands that variable to the
    final remap.  The breakage: the final remap handed 1 (or anything else)
    instead, which would run the 3-point step and the tail merge after the
    in-loop remap had turned them off.  Every oracle record gives the same
    answer either way, because the final remap sees the same evaporation
    and turns ``IDROP`` off again itself, so the hand-over is checked
    directly.
    """
    real_dfun, real_new = C.jerdfun, C.jerdfun_new
    in_loop, final = [], []

    def spy_dfun(*a, **k):
        out = real_dfun(*a, **k)
        in_loop.append(out[2])
        return out

    def spy_new(*a, **k):
        final.append(k["idrop"])
        return real_new(*a, **k)

    monkeypatch.setattr(C, "jerdfun", spy_dfun)
    monkeypatch.setattr(C, "jerdfun_new", spy_new)
    seen = set()
    for args, psi1 in _onecond1_cases(tables):
        in_loop.clear()
        final.clear()
        C.onecond1(**args, psi1=psi1)
        assert final == in_loop, (in_loop, final)
        seen.update(final)
    assert seen == {0, 1}, seen


# --------------------------------------------------------------------------
# Contract and hygiene
# --------------------------------------------------------------------------

def test_named_parameters_are_wrfs_literals():
    """The named WRF parameters hold WRF's literals exactly.

    A shift in the merge criterion, the ventilation cap, the rate floors or
    ``EPSDEL`` can pass the data gates when no oracle record sits near the
    threshold, so each is compared with the value WRF writes (transcribed
    here independently, with its line).
    """
    expected = {
        "ISIGN_KO_1": 0, "ISIGN_KO_2": 0, "ISIGN_3POINT": 1,        # :1704
        "COEFF_REMAPING": 0.0066667,                                # :1706
        "VENTPL_MAX": 5.0,                                          # :1707
        "RW_PW_MIN": 1.0e-10, "RI_PI_MIN": 1.0e-10,                 # :1709-1710
        "RW_PW_RI_PI_MIN": 1.0e-10, "RATIO_ICEW_MIN": 1.0e-4,       # :1711-1712
        "KRDROP_REMAPING_MIN": 6, "KRDROP_REMAPING_MAX": 12,        # :2465
        "EPSDEL": 1.0e-4,                                           # :6843
        "ONECOND1_IN_LIMIT": 3.0, "ONECOND1_OUT_LIMIT": 5.0,        # :7061, :7119
        "RV_MY": 461.5e4, "CF_MY": 2.4e3, "D_MYIN": 0.211,          # :1768
        "PZERO": 1.013e6, "TZERO": 273.15,                          # :1782, :1786
        "AL1_MY_WATER": 2.5e10, "CONST_4PI": 12.566372,             # :1816, :1829
        "COEFF_VISCOUS": 0.13,                                      # :1839
    }
    for name, value in expected.items():
        got = getattr(C, name)
        assert type(got) is type(value) and got == value, (name, got, value)
    assert AL1 == 2500.0                                            # :6830


def test_routines_do_not_modify_their_inputs(tables):
    r = _records("sweep_onecond1_o64")[3]
    ff1 = np.array(r["ff1_in"])
    vr1 = np.array(r["vr1"])
    keep = (ff1.copy(), vr1.copy())
    C.onecond1(r["tt_in"], r["qq_in"], r["pp"], r["ror"], vr1, r["del1n_in"],
               r["del2n_in"], ff1, tables, r["dtcond"])
    assert np.array_equal(ff1, keep[0]) and np.array_equal(vr1, keep[1])
    j = _records("sweep_jernewf_o64")[20]
    fi, rn = np.array(j["fi"]), np.array(j["rn"])
    fi[0], rn[0] = 1.0, -1.0          # exercises JERNEWF's in-place reset
    keep = (fi.copy(), rn.copy())
    C.jernewf(j["rr"], fi, rn, idrop=1)
    assert np.array_equal(fi, keep[0]) and np.array_equal(rn, keep[1])


def test_jerdfun_leaves_psi_and_xin_as_the_caller_had_them_without_drops():
    psi_in, xin_in = np.full(NKR, 7.0), np.full(NKR, 3.0)
    psi, xin, idrop = C.jerdfun(np.ones(NKR), np.ones(NKR), np.ones(NKR), 0.1,
                                ind=1, itype=1, tpn=290.0, idrop=1, isym2=0,
                                psi_in=psi_in, xin_in=xin_in)
    assert np.array_equal(psi, psi_in) and np.array_equal(xin, xin_in) and idrop == 1


def test_module_imports_nothing_but_numpy_math_and_the_shared_layer():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert imported <= {"__future__", "math", "dataclasses", "typing", "numpy",
                        ".constants"}, imported


def test_docstring_cites_every_routine_and_its_lines():
    doc = " ".join(C.__doc__.split())
    for routine, lines in (("POLYSVP", ":1716-1742"), ("JERRATE_KS", ":1744-1921"),
                           ("JERTIMESC_KS", ":1925-1964"),
                           ("JERSUPSAT_KS", ":1966-2330"), ("JERDFUN_KS", ":2334-2442"),
                           ("JERNEWF_KS", ":2444-2757"),
                           ("JERDFUN_NEW_KS", ":2759-2834"), ("ONECOND1", ":6748-7143")):
        assert f"``{routine}``, {lines}" in doc, routine
    assert "fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e" in doc
    for source in ("Khain and Sednev (1996)", "Kovetz and Olund (1969)",
                   "Pruppacher and Klett (1997"):
        assert source in doc, source
