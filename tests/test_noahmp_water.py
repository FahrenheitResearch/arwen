"""Bitwise gate for the Noah-MP WATER assembly.

Replays ``gpuwm/data/noahmp/oracle/noahmp-water.csv`` through
:mod:`gpuwm.core.noahmp_water` and requires ``max_ulp 0`` on every emitted
column of every case, then re-runs the fixture's own structural validator so
the CSV cannot drift underneath the port.

Three things beyond the replay:

1. The fixture's `probe` stage pins four locals that never reach a WRF output
   -- QSNSUB/QSNFRO's gates, QSEVA/QSDEW, and SNOWWATER's glacier SNOFLOW.
   Holding the port to those too means a compensating pair of errors either
   side of SNOWWATER cannot pass on the composed output alone.
2. QIN and QDIS are INTENT(OUT) scalars that no live statement writes.  The
   port omits them; ``test_qin_qdis_are_never_written`` proves the omission is
   right by requiring entry == exit on every case, including the two cases
   that enter with different values.
3. The fixture is byte-pinned as well as validated, because a fixture that can
   be regenerated silently is not a gate.
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

from gpuwm.core.noahmp_snow import SnowColumn
from gpuwm.core.noahmp_water import WaterParameters, water

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-water.csv"
PROBE = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-water-libm.csv"
VALIDATOR = (REPO / "tools" / "noahmp_wrf461_oracle"
             / "validate_water_oracle.py")

PINNED = {
    "noahmp-water.csv":
        "07fcd12e4e45571e3d8735907e322a35fa86eaf038fccd019723986ec2af534d",
    "noahmp-water-libm.csv":
        "a7960673073979528b41d855a55c0ce75d6d7d18908dd73fdcc6506963796913",
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
        table[(r["case"], r["stage"])][key] = v
    return table


TABLE = _load()
CASES = sorted({c for c, stage in TABLE if stage == "input"})


def _params(case) -> WaterParameters:
    p = TABLE[(case, "param")]
    return WaterParameters(
        smcmax=[p[("smcmax", k)] for k in range(1, NSOIL + 1)],
        smcwlt=[p[("smcwlt", k)] for k in range(1, NSOIL + 1)],
        bexp=[p[("bexp", k)] for k in range(1, NSOIL + 1)],
        dksat=[p[("dksat", k)] for k in range(1, NSOIL + 1)],
        dwsat=[p[("dwsat", k)] for k in range(1, NSOIL + 1)],
        kdt=p[("kdt", 0)], frzx=p[("frzx", 0)], slope=p[("slope", 0)],
        ch2op=p[("ch2op", 0)], urban_flag=bool(p[("urban_flag", 0)]),
        timean=p[("timean", 0)], fsatmx=p[("fsatmx", 0)],
        nroot=p[("nroot", 0)], ssi=p[("ssi", 0)],
        snow_ret_fac=p[("snow_ret_fac", 0)],
    )


def _vec(d, name, lo=1, hi=NSOIL):
    return np.asarray([d[(name, k)] for k in range(lo, hi + 1)],
                      dtype=np.float32)


def _ivec(d, name, lo, hi):
    return [int(d[(name, k)]) for k in range(lo, hi + 1)]


def _column(d) -> SnowColumn:
    return SnowColumn(
        nsnow=NSNOW, nsoil=NSOIL,
        isnow=int(d[("isnow", 0)]),
        snowh=d[("snowh", 0)], sneqv=d[("sneqv", 0)],
        snice=_vec(d, "snice", -NSNOW + 1, 0),
        snliq=_vec(d, "snliq", -NSNOW + 1, 0),
        stc=_vec(d, "stc", -NSNOW + 1, NSOIL),
        zsnso=_vec(d, "zsnso", -NSNOW + 1, NSOIL),
        dzsnso=_vec(d, "dzsnso", -NSNOW + 1, NSOIL),
        sh2o=_vec(d, "sh2o"), sice=_vec(d, "sice"),
    )


def _call(case):
    """The physical positional WATER argument list for one fixture case."""
    x = TABLE[(case, "input")]
    p = _params(case)
    col = _column(x)
    return (
        p, col, _vec(x, "smc"), x[("dt", 0)], int(x[("ist", 0)]),
        _ivec(x, "imelt", -NSNOW + 1, 0), _vec(x, "ficeold", -NSNOW + 1, 0),
        x[("fcev", 0)], x[("fctr", 0)], x[("elai", 0)], x[("esai", 0)],
        x[("fveg", 0)], x[("bdfall", 0)], bool(x[("frozen_canopy", 0)]),
        bool(x[("frozen_ground", 0)]), x[("sfctmp", 0)], x[("qvap", 0)],
        x[("qdew", 0)], _vec(x, "zsoil"), _vec(x, "btrani"),
        x[("qsnow", 0)], x[("qrain", 0)], x[("snowhin", 0)],
        x[("ponding", 0)], x[("canliq", 0)], x[("canice", 0)], x[("tv", 0)],
        x[("wslake", 0)], x[("acc_qinsur", 0)], x[("acc_qseva", 0)],
        _vec(x, "acc_etrani"))


def _run(case):
    args = _call(case)
    col = args[1]
    return col, water(*args)


def _produced(col, out) -> dict:
    """Every column the port claims, keyed exactly as the fixture keys it."""
    got = {
        ("isnow", 0): col.isnow,
        ("snowh", 0): col.snowh,
        ("sneqv", 0): col.sneqv,
        ("canliq", 0): out.canliq,
        ("canice", 0): out.canice,
        ("tv", 0): out.tv,
        ("zwt", 0): None,      # filled below from the pass-through set
    }
    del got[("zwt", 0)]
    for k in range(NSOIL):
        got[("sh2o", k + 1)] = col.sh2o[k]
        got[("sice", k + 1)] = col.sice[k]
        got[("smc", k + 1)] = out.smc[k]
        got[("acc_etrani", k + 1)] = out.acc_etrani[k]
    for k in range(-NSNOW + 1, 1):
        got[("snice", k)] = col.snice[k + NSNOW - 1]
        got[("snliq", k)] = col.snliq[k + NSNOW - 1]
    for k in range(-NSNOW + 1, NSOIL + 1):
        j = k + NSNOW - 1
        got[("stc", k)] = col.stc[j]
        got[("zsnso", k)] = col.zsnso[j]
        got[("dzsnso", k)] = col.dzsnso[j]
    got.update({
        ("wslake", 0): out.wslake,
        ("acc_qinsur", 0): out.acc_qinsur,
        ("acc_qseva", 0): out.acc_qseva,
        ("cmc", 0): out.cmc,
        ("ecan", 0): out.ecan,
        ("etran", 0): out.etran,
        ("fwet", 0): out.fwet,
        ("runsrf", 0): out.runsrf,
        ("runsub", 0): out.runsub,
        ("qtldrn", 0): out.qtldrn,
        ("ponding1", 0): out.ponding1,
        ("ponding2", 0): out.ponding2,
        ("qsnbot", 0): out.qsnbot,
        ("qsnsub", 0): out.qsnsub,
        ("qsnfro", 0): out.qsnfro,
        ("qsubc", 0): out.qsubc,
        ("qfroc", 0): out.qfroc,
        ("qfrzc", 0): out.qfrzc,
        ("qmeltc", 0): out.qmeltc,
        ("qevac", 0): out.qevac,
        ("qdewc", 0): out.qdewc,
    })
    return got


# Fields the oracle emits at the `output` stage that WATER provably does not
# write under the pinned identity.  Each is checked separately, by requiring
# the exit row to equal the entry row, rather than being produced by the port.
PASS_THROUGH = {
    "vegtyp", "ist", "iloc", "jloc", "imelt", "nsnow", "nsoil",
    "dt", "uu", "vv", "fcev", "fctr", "qprecc", "qprecl", "elai", "esai",
    "sfctmp", "qvap", "qdew", "tg", "fveg", "bdfall", "fp", "rain", "snow",
    "qsnow", "qrain", "snowhin", "latheav", "latheag", "dx", "tdfracmp",
    "frozen_canopy", "frozen_ground", "croplu", "irrfra", "mifac", "fifac",
    "zsoil", "btrani", "smceq", "ficeold", "ponding",
    "zwt", "wa", "wt", "smcwtd", "deeprech", "rech",
    "iramtfi", "iramtmi", "irfirate", "irmirate",
    "qin", "qdis",
}


@pytest.mark.parametrize("case", CASES)
def test_water_matches_oracle(case):
    col, out = _run(case)
    got = _produced(col, out)
    want = TABLE[(case, "output")]

    bad = []
    for key, value in got.items():
        if key not in want:
            raise AssertionError(f"{case}: {key} is not an oracle column")
        if _bits(value) != _bits(want[key]):
            bad.append(f"{key[0]}[{key[1]}] port={_bits(value)} "
                       f"oracle={_bits(want[key])}")
    missing = sorted(k for k in want
                     if k not in got and k[0] not in PASS_THROUGH)
    assert not bad, f"{case}: " + "; ".join(bad)
    assert not missing, f"{case}: port produced nothing for {missing}"


@pytest.mark.parametrize("case", CASES)
def test_water_matches_probe_stage(case):
    """The four locals that decide a branch but never reach a WRF output."""
    col, out = _run(case)
    want = TABLE[(case, "probe")]
    for name, value in (("qsnsub", out.qsnsub), ("qsnfro", out.qsnfro),
                        ("snoflow", out.snoflow), ("qsnbot", out.qsnbot),
                        ("ponding1", out.ponding1),
                        ("ponding2", out.ponding2)):
        assert _bits(value) == _bits(want[(name, 0)]), \
            f"{case}: probe {name} port={_bits(value)} " \
            f"oracle={_bits(want[(name, 0)])}"


@pytest.mark.parametrize("case", CASES)
def test_qseva_and_qsdew_reach_the_soil_unmodified(case):
    """QSEVA/QSDEW before 6145, which is where the frozen branch forks.

    The probe's values are the pre-FROZEN_GROUND ones, so the comparison only
    applies where that block is off; where it is on, the port must have zeroed
    both, and 6147-6148 say so.
    """
    x = TABLE[(case, "input")]
    col, out = _run(case)
    want = TABLE[(case, "probe")]
    if bool(x[("frozen_ground", 0)]):
        assert _bits(out.qseva) == _bits(np.float32(0.0))
        assert _bits(out.qsdew) == _bits(np.float32(0.0))
    else:
        # 6167 scales QSEVA by 0.001 after the fork; undo nothing, compare the
        # probe against the pre-scaling value the port also reports.
        assert _bits(np.float32(out.qseva / np.float32(0.001))) == \
            _bits(want[("qseva", 0)])
        assert _bits(out.qsdew) == _bits(want[("qsdew", 0)])


@pytest.mark.parametrize("case", CASES)
def test_qin_qdis_are_never_written(case):
    """Under opt_run=3 nothing assigns either, so entry must equal exit.

    This is what licenses the port to omit them.  If a future option identity
    ever made GROUNDWATER live, this test fails before the omission becomes a
    silent wrong answer.
    """
    x = TABLE[(case, "input")]
    y = TABLE[(case, "output")]
    for name in ("qin", "qdis"):
        assert _bits(x[(name, 0)]) == _bits(y[(name, 0)]), \
            f"{case}: {name} moved, so opt_run=3 does write it after all"


@pytest.mark.parametrize("case", CASES)
def test_every_dead_argument_is_returned_untouched(case):
    """The oracle's own record that WATER does not write its inert arguments."""
    x = TABLE[(case, "input")]
    y = TABLE[(case, "output")]
    for key in x:
        if key[0] in PASS_THROUGH and key[0] not in ("qin", "qdis"):
            assert x[key] == y[key], f"{case}: {key} moved"


def test_qtldrn_is_identically_zero():
    """opt_tdrn=0: 6112 zeroes it and 6254 scales the zero."""
    for case in CASES:
        y = TABLE[(case, "output")]
        assert _bits(y[("qtldrn", 0)]) == _bits(np.float32(0.0)), case


def test_inert_probes_reproduce_their_baseline():
    """Every perturbed dead argument must move nothing the port produces."""
    for probe, base in (("wat_inert_probe", "wat_bare_rain"),
                        ("wat_lake_inert_probe", "wat_lake_filling")):
        a, b = TABLE[(probe, "output")], TABLE[(base, "output")]
        moved = sorted(k for k in a
                       if k[0] not in PASS_THROUGH and a[k] != b[k])
        assert not moved, f"{probe}: {moved} moved against {base}"
        col_a, out_a = _run(probe)
        col_b, out_b = _run(base)
        ga, gb = _produced(col_a, out_a), _produced(col_b, out_b)
        moved = sorted(k for k in ga if _bits(ga[k]) != _bits(gb[k]))
        assert not moved, f"{probe}: the port moved {moved} against {base}"


def test_out_entry_probes_prove_which_entry_values_are_dead():
    """RUNSRF/RUNSUB/QSNSUB/QSNFRO entry values must not survive; QIN/QDIS must."""
    for probe, base in (("wat_out_entry_probe", "wat_bare_rain"),
                        ("wat_snow_out_entry_probe", "wat_snow_sublimation")):
        xa, xb = TABLE[(probe, "input")], TABLE[(base, "input")]
        ya, yb = TABLE[(probe, "output")], TABLE[(base, "output")]
        for name in ("runsrf", "runsub", "qsnsub", "qsnfro"):
            assert xa[(name, 0)] != xb[(name, 0)], \
                f"{probe}: {name} is not actually perturbed"
            assert _bits(ya[(name, 0)]) == _bits(yb[(name, 0)]), \
                f"{probe}: {name}'s entry value survived"
        for name in ("qin", "qdis"):
            assert _bits(ya[(name, 0)]) == _bits(xa[(name, 0)])


def test_branch_coverage_is_assertable_from_the_inputs():
    """Every live branch of WATER is taken by some case, read off the inputs."""
    def inp(c):
        return TABLE[(c, "input")]

    def prb(c):
        return TABLE[(c, "probe")]

    zero = np.float32(0.0)
    seen = defaultdict(list)
    for c in CASES:
        x, p = inp(c), prb(c)
        seen["sneqv>0" if x[("sneqv", 0)] > zero else "sneqv==0"].append(c)
        if x[("sneqv", 0)] > zero:
            lhs = x[("qvap", 0)]
            rhs = np.float32(x[("sneqv", 0)] / x[("dt", 0)])
            seen["qsnsub=qvap" if lhs < rhs else "qsnsub=sneqv/dt"].append(c)
        seen["qdew>0" if x[("qdew", 0)] > zero else "qdew==0"].append(c)
        seen["frozen" if x[("frozen_ground", 0)] else "unfrozen"].append(c)
        seen["isnow==0" if x[("isnow", 0)] == 0 else "isnow<0"].append(c)
        seen["lake" if x[("ist", 0)] == 2 else "soil"].append(c)
        if x[("ist", 0)] == 2:
            seen["wslake>=wslmax" if x[("wslake", 0)] >= np.float32(5000.0)
                 else "wslake<wslmax"].append(c)
        seen["snoflow>0" if p[("snoflow", 0)] > zero
             else "snoflow==0"].append(c)
        seen["qsnbot>0" if p[("qsnbot", 0)] > zero else "qsnbot==0"].append(c)
        seen["ponding1>0" if p[("ponding1", 0)] > zero
             else "ponding1==0"].append(c)
        seen["ponding>0" if x[("ponding", 0)] > zero else "ponding==0"].append(c)
        seen[f"nroot={TABLE[(c, 'param')][('nroot', 0)]}"].append(c)

    required = [
        "sneqv>0", "sneqv==0", "qsnsub=qvap", "qsnsub=sneqv/dt",
        "qdew>0", "qdew==0", "frozen", "unfrozen", "isnow==0", "isnow<0",
        "lake", "soil", "wslake>=wslmax", "wslake<wslmax",
        "snoflow>0", "snoflow==0", "qsnbot>0", "qsnbot==0",
        "ponding1>0", "ponding1==0", "ponding>0", "ponding==0",
        "nroot=3", "nroot=4",
    ]
    uncovered = [k for k in required if not seen[k]]
    assert not uncovered, f"no case reaches {uncovered}"


def test_the_sice_deficit_branch_is_reached():
    """6149-6152 fires only when the frozen-ground debit exceeds SICE(1)."""
    hits = []
    for case in CASES:
        x = TABLE[(case, "input")]
        if not x[("frozen_ground", 0)]:
            continue
        p = TABLE[(case, "probe")]
        dz = TABLE[(case, "input")][("dzsnso", 1)]
        delta = np.float32(np.float32(np.float32(p[("qsdew", 0)]
                                                 - p[("qseva", 0)])
                                      * x[("dt", 0)])
                           / np.float32(dz * np.float32(1000.0)))
        if np.float32(x[("sice", 1)] + delta) < np.float32(0.0):
            hits.append(case)
    assert hits, "no case drives SICE(1) negative under FROZEN_GROUND"


def test_the_gate_can_fail():
    """A one-ULP nudge to any live input must break the comparison."""
    case = "wat_bare_rain"
    col, out = _run(case)
    good = _produced(col, out)
    want = TABLE[(case, "output")]
    assert all(_bits(good[k]) == _bits(want[k]) for k in good)

    x = dict(TABLE[(case, "input")])
    x[("qrain", 0)] = np.float32(
        np.uint32(np.float32(x[("qrain", 0)]).view(np.uint32) + 1)
        .view(np.float32))
    p = _params(case)
    col2 = _column(x)
    out2 = water(
        p, col2, _vec(x, "smc"), x[("dt", 0)], int(x[("ist", 0)]),
        _ivec(x, "imelt", -NSNOW + 1, 0), _vec(x, "ficeold", -NSNOW + 1, 0),
        x[("fcev", 0)], x[("fctr", 0)], x[("elai", 0)], x[("esai", 0)],
        x[("fveg", 0)], x[("bdfall", 0)], bool(x[("frozen_canopy", 0)]),
        bool(x[("frozen_ground", 0)]), x[("sfctmp", 0)], x[("qvap", 0)],
        x[("qdew", 0)], _vec(x, "zsoil"), _vec(x, "btrani"),
        x[("qsnow", 0)], x[("qrain", 0)], x[("snowhin", 0)],
        x[("ponding", 0)], x[("canliq", 0)], x[("canice", 0)], x[("tv", 0)],
        x[("wslake", 0)], x[("acc_qinsur", 0)], x[("acc_qseva", 0)],
        _vec(x, "acc_etrani"))
    bad = _produced(col2, out2)
    assert any(_bits(bad[k]) != _bits(want[k]) for k in bad), \
        "a one-ULP change to QRAIN did not move a single output"


def test_fixture_is_byte_pinned():
    for name, digest in PINNED.items():
        path = FIXTURE.parent / name
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == digest, f"{name} changed: {got}"


def test_fixture_passes_its_own_validator():
    r = subprocess.run(
        [sys.executable, str(VALIDATOR), "--fixture", str(FIXTURE),
         "--probe", str(PROBE)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("label,edit", [
    # A pass-through argument that moved: the whole `QIN/QDIS are never
    # written` claim rests on this check being able to fire.
    ("qin moved",
     lambda t: t.replace("water,wat_bare_rain,output,qin,0,real,-3.125000000E+01,C1FA0000",
                         "water,wat_bare_rain,output,qin,0,real,-3.125100040E+01,C1FA0347")),
    # A decimal column that no longer round-trips to its bit pattern.
    ("decimal/bits disagree",
     lambda t: t.replace("water,wat_bare_rain,output,fwet,0,real,1.000000000E+00,3F800000",
                         "water,wat_bare_rain,output,fwet,0,real,1.100000000E+00,3F800000")),
    # QTLDRN non-zero, which opt_tdrn=0 forbids.
    ("qtldrn non-zero",
     lambda t: t.replace("water,wat_bare_rain,output,qtldrn,0,real,0.000000000E+00,00000000",
                         "water,wat_bare_rain,output,qtldrn,0,real,1.000000000E+00,3F800000")),
])
def test_the_validator_can_fail(tmp_path, label, edit):
    """A fixture checker that accepts anything is not a checker.

    Each edit below is one class of corruption the validator claims to catch.
    The edits are applied to a copy; the committed fixture is never touched.
    """
    text = FIXTURE.read_text()
    bad = edit(text)
    assert bad != text, f"{label}: the edit did not match any row"
    p = tmp_path / "noahmp-water.csv"
    p.write_text(bad, newline="")
    r = subprocess.run(
        [sys.executable, str(VALIDATOR), "--fixture", str(p),
         "--probe", str(PROBE)],
        capture_output=True, text=True)
    assert r.returncode != 0, f"{label}: the validator accepted a corrupt fixture"


# --------------------------------------------------------------------------
# the runtime device packing
# --------------------------------------------------------------------------
def _fixture_rows_packed():
    """Build the flat device rows straight off the fixture, independently.

    This is the reference the packer is checked against.  It reads the oracle
    CSV in the same order ``tests/test_noahmp_water_cuda.py`` does -- an
    authority outside :mod:`gpuwm.core.noahmp_water_gpu` -- so a transposition
    inside the packer cannot agree with it by construction.
    """
    from gpuwm.core.noahmp_water_gpu import (INT_STRIDE, IN_STRIDE, NSTATE,
                                             P_STRIDE)

    n = len(CASES)
    par = np.zeros((n, P_STRIDE), dtype=np.float32)
    fin = np.zeros((n, IN_STRIDE), dtype=np.float32)
    iin = np.zeros((n, INT_STRIDE), dtype=np.int32)
    for i, case in enumerate(CASES):
        p = TABLE[(case, "param")]
        for name, base in (("smcmax", 0), ("smcwlt", 4), ("bexp", 8),
                           ("dksat", 12), ("dwsat", 16)):
            for k in range(NSOIL):
                par[i, base + k] = p[(name, k + 1)]
        for name, slot in (("kdt", 20), ("frzx", 21), ("slope", 22),
                           ("ch2op", 23), ("ssi", 24), ("snow_ret_fac", 25)):
            par[i, slot] = p[(name, 0)]

        x = TABLE[(case, "input")]
        fin[i, 0] = x[("snowh", 0)]
        fin[i, 1] = x[("sneqv", 0)]
        for k in range(NSNOW):
            fin[i, 2 + k] = x[("snice", k - NSNOW + 1)]
            fin[i, 5 + k] = x[("snliq", k - NSNOW + 1)]
        for k in range(NSNOW + NSOIL):
            j = k - NSNOW + 1
            fin[i, 8 + k] = x[("stc", j)]
            fin[i, 15 + k] = x[("zsnso", j)]
            fin[i, 22 + k] = x[("dzsnso", j)]
        for k in range(NSOIL):
            fin[i, 29 + k] = x[("sh2o", k + 1)]
            fin[i, 33 + k] = x[("sice", k + 1)]
        for name, slot in (
            ("dt", 0), ("fcev", 1), ("fctr", 2), ("elai", 3), ("esai", 4),
            ("fveg", 5), ("bdfall", 6), ("sfctmp", 7), ("qvap", 8),
            ("qdew", 9), ("qsnow", 10), ("qrain", 11), ("snowhin", 12),
            ("ponding", 13), ("canliq", 14), ("canice", 15), ("tv", 16),
            ("wslake", 17), ("acc_qinsur", 18), ("acc_qseva", 19),
        ):
            fin[i, NSTATE + slot] = x[(name, 0)]
        for name, base, lo, cnt in (("zsoil", 20, 1, NSOIL),
                                    ("btrani", 24, 1, NSOIL),
                                    ("acc_etrani", 28, 1, NSOIL),
                                    ("smc", 32, 1, NSOIL),
                                    ("ficeold", 36, -NSNOW + 1, NSNOW)):
            for k in range(cnt):
                fin[i, NSTATE + base + k] = x[(name, lo + k)]

        iin[i, 0] = x[("isnow", 0)]
        iin[i, 1] = p[("urban_flag", 0)]
        iin[i, 2] = p[("nroot", 0)]
        iin[i, 3] = x[("ist", 0)]
        iin[i, 4] = x[("frozen_canopy", 0)]
        iin[i, 5] = x[("frozen_ground", 0)]
        for k in range(NSNOW):
            iin[i, 6 + k] = x[("imelt", k - NSNOW + 1)]
    return par, fin, iin


def test_the_runtime_packing_matches_the_kernel_layout():
    """The device batch's rows must be the rows ``noahmp_water.cu`` reads.

    WATER has the widest argument surface of any leaf here -- five per-layer
    parameter vectors, a nine-array snow/soil column, twenty scalars and five
    more vectors -- and none of the correspondence between the physical call
    and the flat row is checkable by a compiler.
    """
    from gpuwm.core.noahmp_water_gpu import pack_water_calls

    calls = [(_call(case), {}) for case in CASES]
    par, fin, iin = pack_water_calls(calls)
    want_par, want_fin, want_iin = _fixture_rows_packed()
    for got, want, what in ((par, want_par, "par"), (fin, want_fin, "fin")):
        bad = np.flatnonzero(got.ravel().view(np.uint32)
                             != want.ravel().view(np.uint32))
        assert bad.size == 0, (
            f"{what}: {bad.size} words differ, first at flat slot "
            f"{int(bad[0])} (row {int(bad[0]) // got.shape[1]}, "
            f"slot {int(bad[0]) % got.shape[1]})")
    np.testing.assert_array_equal(iin, want_iin)


def test_the_water_packing_gate_can_fail(monkeypatch):
    """Show the failing form first.

    ``sh2o`` and ``sice`` are adjacent blocks of the packed column and are the
    transposition a reader is most likely to make; both are live, so swapping
    them would change the forecast rather than be absorbed.
    """
    from gpuwm.core import noahmp_water_gpu as gpu

    slots = dict(gpu.STATE_SLOT)
    slots["sh2o"], slots["sice"] = slots["sice"], slots["sh2o"]
    monkeypatch.setattr(gpu, "STATE_SLOT", slots)

    calls = [(_call(case), {}) for case in CASES]
    _par, fin, _iin = gpu.pack_water_calls(calls)
    _wp, want_fin, _wi = _fixture_rows_packed()
    assert (fin.ravel().view(np.uint32)
            != want_fin.ravel().view(np.uint32)).any(), (
        "transposing SH2O and SICE in the packer went undetected")
