"""Bitwise gate for the Noah-MP soil-water port.

Replays ``gpuwm/data/noahmp/oracle/noahmp-soilwater.csv`` through
:mod:`gpuwm.core.noahmp_soilwater` and requires ``max_ulp 0`` on every emitted
column of every case, then re-runs the fixture's own structural validator so
the CSV cannot drift underneath the port.

The fixture is byte-pinned here as well as validated, because a fixture that
can be regenerated silently is not a gate.
"""

from __future__ import annotations

import csv
import hashlib
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.noahmp_soilwater import (
    SoilParameters, canwater, infil, soilwater, srt, sstep,
)

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-soilwater.csv"
PROBE = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-soilwater-libm.csv"
VALIDATOR = (REPO / "tools" / "noahmp_wrf461_oracle"
             / "validate_soilwater_oracle.py")

PINNED = {
    "noahmp-soilwater.csv":
        "17cb1be26852fc025d5b3e3fe904a2b16cc3e3699233ace43d0292805266e1a0",
    "noahmp-soilwater-libm.csv":
        "68b4c7d89ad9e69e6b9bf167cf0f1cae9003d5b1c5fefd69dfb506b1c7c6bead",
}

NSOIL = 4
NSNOW = 3


# ---------------------------------------------------------------------------

def _bits(x) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


def _load():
    table = defaultdict(dict)
    for r in csv.DictReader(FIXTURE.open(newline="")):
        key = (r["field"], int(r["index"]))
        v = (int(r["value"]) if r["dtype"] == "int"
             else np.float32(struct.unpack(">f", bytes.fromhex(r["bits"]))[0]))
        table[(r["leaf"], r["case"], r["stage"])][key] = v
    return table


TABLE = _load()
CASES = defaultdict(list)
for _leaf, _case, _stage in TABLE:
    if _stage == "input":
        CASES[_leaf].append(_case)
for _leaf in CASES:
    CASES[_leaf] = sorted(CASES[_leaf])


def _params(leaf, case) -> SoilParameters:
    p = TABLE[(leaf, case, "param")]
    return SoilParameters(
        smcmax=[p[("smcmax", k)] for k in range(1, NSOIL + 1)],
        smcwlt=[p[("smcwlt", k)] for k in range(1, NSOIL + 1)],
        bexp=[p[("bexp", k)] for k in range(1, NSOIL + 1)],
        dksat=[p[("dksat", k)] for k in range(1, NSOIL + 1)],
        dwsat=[p[("dwsat", k)] for k in range(1, NSOIL + 1)],
        kdt=p[("kdt", 0)], frzx=p[("frzx", 0)], slope=p[("slope", 0)],
        ch2op=p[("ch2op", 0)], urban_flag=bool(p[("urban_flag", 0)]),
        timean=p[("timean", 0)], fsatmx=p[("fsatmx", 0)],
    )


def _vec(d, name, lo=1, hi=NSOIL):
    return [d[(name, k)] for k in range(lo, hi + 1)]


def _check(leaf, case, got: dict):
    """Every produced column must match the fixture bit for bit."""
    want = TABLE[(leaf, case, "output")]
    bad = []
    for key, value in got.items():
        if key not in want:
            raise AssertionError(f"{leaf}/{case}: {key} is not an oracle column")
        if _bits(value) != _bits(want[key]):
            bad.append(f"{key[0]}[{key[1]}] port={_bits(value)} "
                       f"oracle={_bits(want[key])}")
    missing = sorted(set(want) - set(got))
    assert not bad, f"{leaf}/{case}: " + "; ".join(bad)
    assert not missing, f"{leaf}/{case}: port produced nothing for {missing}"


# ---------------------------------------------------------------------------
# Per-leaf replays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES["canwater"])
def test_canwater_matches_oracle(case):
    x = TABLE[("canwater", case, "input")]
    p = _params("canwater", case)
    (canliq, canice, tv, cmc, ecan, etran, fwet,
     qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc) = canwater(
        p, x[("dt", 0)], x[("fcev", 0)], x[("fctr", 0)], x[("elai", 0)],
        x[("esai", 0)], x[("fveg", 0)], x[("bdfall", 0)],
        bool(x[("frozen_canopy", 0)]),
        x[("canliq", 0)], x[("canice", 0)], x[("tv", 0)])
    _check("canwater", case, {
        ("canliq", 0): canliq, ("canice", 0): canice, ("tv", 0): tv,
        ("cmc", 0): cmc, ("ecan", 0): ecan, ("etran", 0): etran,
        ("fwet", 0): fwet, ("qsubc", 0): qsubc, ("qfroc", 0): qfroc,
        ("qfrzc", 0): qfrzc, ("qmeltc", 0): qmeltc, ("qevac", 0): qevac,
        ("qdewc", 0): qdewc,
    })


@pytest.mark.parametrize("case", CASES["infil"])
def test_infil_matches_oracle(case):
    x = TABLE[("infil", case, "input")]
    p = _params("infil", case)
    pddum, runsrf = infil(
        p, x[("dt", 0)], _vec(x, "zsoil"), _vec(x, "sh2o"), _vec(x, "sice"),
        x[("sicemax", 0)], x[("qinsur", 0)], x[("pddum", 0)], x[("runsrf", 0)])
    _check("infil", case, {("pddum", 0): pddum, ("runsrf", 0): runsrf})


@pytest.mark.parametrize("case", CASES["srt"])
def test_srt_matches_oracle(case):
    x = TABLE[("srt", case, "input")]
    p = _params("srt", case)
    rhstt, ai, bi, ci, qdrain, wcnd = srt(
        p, _vec(x, "zsoil"), x[("pddum", 0)], _vec(x, "etrani"),
        x[("qseva", 0)], _vec(x, "smc"), _vec(x, "fcr"))
    got = {("qdrain", 0): qdrain}
    for k in range(NSOIL):
        got[("rhstt", k + 1)] = rhstt[k]
        got[("ai", k + 1)] = ai[k]
        got[("bi", k + 1)] = bi[k]
        got[("ci", k + 1)] = ci[k]
        got[("wcnd", k + 1)] = wcnd[k]
    _check("srt", case, got)


@pytest.mark.parametrize("case", CASES["sstep"])
def test_sstep_matches_oracle(case):
    x = TABLE[("sstep", case, "input")]
    p = _params("sstep", case)
    dzsnso = [x[("dzsnso", k)] for k in range(-NSNOW + 1, NSOIL + 1)]
    sh2o, smc, ai, bi, ci, rhstt, wplus = sstep(
        p, x[("dt", 0)], dzsnso, _vec(x, "sice"), _vec(x, "sh2o"),
        _vec(x, "ai"), _vec(x, "bi"), _vec(x, "ci"), _vec(x, "rhstt"))
    got = {("wplus", 0): wplus}
    for k in range(NSOIL):
        got[("sh2o", k + 1)] = sh2o[k]
        got[("smc", k + 1)] = smc[k]
        got[("ai", k + 1)] = ai[k]
        got[("bi", k + 1)] = bi[k]
        got[("ci", k + 1)] = ci[k]
        got[("rhstt", k + 1)] = rhstt[k]
    # SMCWTD/QDRAIN/DEEPRECH are INOUT but written only under OPT_RUN==5, so
    # the port never touches them; the fixture must echo the entry value.
    for name in ("smcwtd", "qdrain", "deeprech"):
        got[(name, 0)] = x[(name, 0)]
    _check("sstep", case, got)


@pytest.mark.parametrize("case", CASES["soilwater"])
def test_soilwater_matches_oracle(case):
    x = TABLE[("soilwater", case, "input")]
    p = _params("soilwater", case)
    dzsnso = [x[("dzsnso", k)] for k in range(-NSNOW + 1, NSOIL + 1)]
    sh2o, smc, runsrf, qdrain, runsub, wcnd, fcrmax = soilwater(
        p, x[("dt", 0)], _vec(x, "zsoil"), dzsnso, x[("qinsur", 0)],
        x[("qseva", 0)], _vec(x, "etrani"), _vec(x, "sice"),
        _vec(x, "sh2o"), _vec(x, "smc"), x[("runsub", 0)])
    got = {("runsrf", 0): runsrf, ("qdrain", 0): qdrain,
           ("runsub", 0): runsub, ("fcrmax", 0): fcrmax}
    for k in range(NSOIL):
        got[("sh2o", k + 1)] = sh2o[k]
        got[("smc", k + 1)] = smc[k]
        got[("wcnd", k + 1)] = wcnd[k]
    # Arguments no opt_run=3 statement writes.  The port does not take them at
    # all, so it can only be right if the oracle echoed them unchanged.
    for name in ("zwt", "smcwtd", "deeprech", "qtldrn"):
        got[(name, 0)] = x[(name, 0)]
    got[("vegtyp", 0)] = x[("vegtyp", 0)]
    _check("soilwater", case, got)


# ---------------------------------------------------------------------------
# Properties of the port that max_ulp 0 alone does not establish
# ---------------------------------------------------------------------------

def test_niter_probe_agrees_with_the_port():
    """The port's own NITER must match the oracle's independent derivation.

    SOILWATER's iteration count is a local, so bitwise agreement on the outputs
    could in principle be reached with the wrong count on a case where the
    extra passes are near-idempotent.  The fixture carries the count and the
    two sides of its predicate, derived in the harness from the module's own
    INFIL, and this pins the port against them.
    """
    from gpuwm.core.noahmp_soilwater import _f, _soil, infil as _infil

    seen = set()
    for case in CASES["soilwater"]:
        x = TABLE[("soilwater", case, "input")]
        pr = TABLE[("soilwater", case, "probe")]
        p = _params("soilwater", case)
        dz = _soil([x[("dzsnso", k)] for k in range(-NSNOW + 1, NSOIL + 1)],
                   NSNOW)
        sice = np.asarray(_vec(x, "sice"), dtype=np.float32)
        sh2o = np.asarray(_vec(x, "sh2o"), dtype=np.float32).copy()
        for k in range(NSOIL):
            epore = max(np.float32(1.0e-4), _f(p.smcmax[k] - sice[k]))
            sh2o[k] = min(epore, sh2o[k])
        sicemax = np.float32(0.0)
        for k in range(NSOIL):
            if sice[k] > sicemax:
                sicemax = sice[k]
        pddum, _rs = _infil(p, x[("dt", 0)], _vec(x, "zsoil"), sh2o, sice,
                            sicemax, x[("qinsur", 0)], np.float32(0.0),
                            np.float32(0.0))
        assert _bits(pddum) == _bits(pr[("niter_pddum", 0)]), case
        assert _bits(sicemax) == _bits(pr[("probe_sicemax", 0)]), case
        niter = 3
        if _f(pddum * _f(x[("dt", 0)])) > _f(dz[0] * p.smcmax[0]):
            niter *= 2
        assert niter == pr[("niter", 0)], case
        seen.add(niter)
    assert seen == {3, 6}, f"the fixture only exercises NITER in {sorted(seen)}"


def test_soilwater_smc_is_not_sh2o_plus_sice_on_the_watmin_path():
    """A negative control for the SMC/SH2O inconsistency the port reproduces.

    The WATMIN fixup at 7546-7574 rewrites SH2O and never touches SMC, so a
    port that "tidies up" by returning ``sh2o + sice`` would be wrong.  At
    least one fixture case must show the two disagreeing, or this test is
    guarding nothing.
    """
    disagree = []
    for case in CASES["soilwater"]:
        out = TABLE[("soilwater", case, "output")]
        inp = TABLE[("soilwater", case, "input")]
        for k in range(1, NSOIL + 1):
            tidy = np.float32(out[("sh2o", k)] + inp[("sice", k)])
            if _bits(tidy) != _bits(out[("smc", k)]):
                disagree.append((case, k))
    assert disagree, ("no case distinguishes SMC from SH2O+SICE, so the "
                      "fixture cannot catch a port that recomputes it")


def test_inert_probes_are_reproduced_by_the_port():
    """The port drops arguments the fixture proves inert; check both ends."""
    pairs = [("srt", "srt_wet_baseline", "srt_inert_probe"),
             ("sstep", "sstep_baseline", "sstep_inert_probe"),
             ("soilwater", "slw_moderate_rain", "slw_inert_probe"),
             ("canwater", "canw_unfrozen_evap", "canw_inert_probe")]
    for leaf, base, probe in pairs:
        b = TABLE[(leaf, base, "output")]
        p = TABLE[(leaf, probe, "output")]
        # Only the pass-through slots may differ.
        passthrough = {"zwt", "smcwtd", "deeprech", "qtldrn", "vegtyp",
                       "qdrain"} if leaf in ("soilwater", "sstep") else set()
        for key in b:
            if key[0] in passthrough:
                continue
            assert _bits(b[key]) == _bits(p[key]), f"{leaf}/{probe}/{key}"


def test_fixture_is_the_pinned_bytes():
    for name, digest in PINNED.items():
        path = FIXTURE.parent / name
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == digest, f"{name} changed: {got}"


def test_fixture_still_validates():
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), "--fixture", str(FIXTURE),
         "--probe", str(PROBE), "--quiet"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_probe_pins_the_module_constants():
    want = {"tfrz": 273.16, "hvap": 2.5104e06, "hsub": 2.8440e06,
            "hfus": 0.3336e06, "cwat": 4.188e06, "cice": 2.094e06,
            "denh2o": 1000.0, "denice": 917.0}
    import gpuwm.core.noahmp_soilwater as sw
    rows = {r["name"].strip(): r for r in csv.DictReader(PROBE.open(newline=""))}
    for name, value in want.items():
        assert rows[name]["bits"].upper() == _bits(value)
        assert _bits(getattr(sw, name.upper())) == rows[name]["bits"].upper()


def test_libm_probe_matches_the_transcriptions():
    """The port's expf/powf must reproduce the compiled module's own calls."""
    from gpuwm.core.noahmp_libm import expf, powf

    rows = {r["name"].strip(): r["bits"].upper()
            for r in csv.DictReader(PROBE.open(newline=""))}
    checked = 0
    for i in range(257):
        x = np.float32(-4.0 * (1.0 - np.float32(i) / np.float32(256.0)))
        assert _bits(expf(x)) == rows[f"expf_fcr_{i}"], f"expf_fcr_{i}"
        checked += 1
    for i in range(257):
        x = np.float32(np.float32(i) / np.float32(256.0))
        assert _bits(powf(x, np.float32(0.667))) == rows[f"powf_fwet_{i}"], i
        checked += 1
    for i in range(129):
        x = np.float32(np.float32(0.01)
                       + np.float32(np.float32(1.0 - 0.01)
                                    * np.float32(np.float32(i)
                                                 / np.float32(128.0))))
        assert _bits(powf(x, np.float32(6.74))) == rows[f"powf_wdf_{i}"], i
        assert _bits(powf(x, np.float32(12.48))) == rows[f"powf_wcnd_{i}"], i
        checked += 2
    assert checked == 772


def test_exp_neg_a_is_the_runtime_call_not_the_compile_time_fold():
    """`EXP(-A)` with A a PARAMETER is a compile-time fold; measure it.

    gfortran folds it with MPFR (correctly rounded) while `EXP(-A*(1.-FICE))`
    lowers to glibc's expf.  On this domain the two agree, which is why the
    port may use one function for both; the probe is what makes that a
    measurement rather than an assumption.
    """
    from gpuwm.core.noahmp_libm import expf

    rows = {r["name"].strip(): r["bits"].upper()
            for r in csv.DictReader(PROBE.open(newline=""))}
    assert rows["exp_neg_a_folded"] == rows["exp_neg_a_runtim"]
    assert _bits(expf(np.float32(-4.0))) == rows["exp_neg_a_folded"]
