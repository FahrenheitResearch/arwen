#!/usr/bin/env python3
"""Validate the Noah-MP WATER oracle fixture.

Runs inside ``build_water.sh`` before the CSV is allowed out of the build
directory, and again from ``tests/test_noahmp_water.py`` against the committed
copy.  It knows nothing about the port; everything it checks is a property of
the fixture itself.

What it enforces
----------------

1. **Shape.**  Every case carries exactly the field/index set the harness
   declares, at every stage, and no others.

2. **Bit/decimal agreement.**  The ``value`` column must round-trip to the
   ``bits`` column in binary32.  This catches a hand-edited row.

3. **No sentinel survives.**  ``-999.0`` is the harness's uninitialised
   marker.  Every INTENT(OUT) argument WATER is obliged to write must have
   overwritten it.

4. **Pass-through, measured.**  Every argument the pinned identity does not
   write must come back bit-identical to its entry value in *every* case --
   not only in the inert probes.  ``QIN`` and ``QDIS`` are in that set: they
   are INTENT(OUT) at 6052-6053 and the only statement that assigns either is
   inside the ``OPT_RUN==1`` GROUNDWATER call, so under ``opt_run=3`` the
   caller's value stands.  This check is what turns that into a fact.

5. **Dead entry values.**  ``RUNSRF``/``RUNSUB`` (zeroed at 6109-6110),
   ``QTLDRN`` (6112) and ``QSNSUB``/``QSNFRO`` (6126/6132) must *not* survive.
   The ``*_out_entry_probe`` cases perturb each and require the paired
   baseline's outputs bit for bit.

6. **Discrimination.**  No input slot may be zero in every case unless it is
   declared inert, because a slot that is always zero cannot tell a port that
   reads it from one that ignores it.  An inert slot that is zero everywhere
   is rejected too: its inertness would be vacuous.

7. **Inertness, measured.**  Each ``*_inert_probe`` case names a baseline and
   the arguments it perturbs.  Every output must be bit-identical to the
   baseline's except the pass-through slots, which must echo the probe's own
   entry value.

8. **Branch coverage, from the inputs.**  Each declared branch has a predicate
   evaluated on the *input* and *probe* columns only -- never on an output --
   and must be taken by at least one case and not taken by at least one case.
   A branch whose two sides produce identical outputs is called out rather
   than silently counted.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from collections import defaultdict
from pathlib import Path

NSOIL = 4
NSNOW = 3
WSLMAX = 5000.0

SENTINEL_BITS = "C479C000"          # -999.0


def _f(spec):
    out = []
    for name, lo, hi in spec:
        out.extend((name, k) for k in range(lo, hi + 1))
    return tuple(out)


def _s(*names):
    return tuple((n, 0) for n in names)


PARAM_FIELDS = (
    _f((("smcmax", 1, NSOIL), ("smcwlt", 1, NSOIL), ("bexp", 1, NSOIL),
        ("dksat", 1, NSOIL), ("dwsat", 1, NSOIL)))
    + _s("kdt", "frzx", "slope", "ch2op", "ssi", "snow_ret_fac",
         "timean", "fsatmx", "nroot", "urban_flag")
)

# The emit order of run_water.F90's emit_state, which is used for both the
# input and the output stage.
STATE_FIELDS = (
    _s("nsnow", "nsoil", "vegtyp", "ist", "iloc", "jloc")
    + _f((("imelt", -NSNOW + 1, 0),))
    + _s("dt", "uu", "vv", "fcev", "fctr", "qprecc", "qprecl", "elai", "esai",
         "sfctmp", "qvap", "qdew", "tg", "fveg", "bdfall", "fp", "rain",
         "snow", "qsnow", "qrain", "snowhin", "latheav", "latheag", "dx",
         "tdfracmp", "frozen_canopy", "frozen_ground", "croplu",
         "irrfra", "mifac", "fifac")
    + _f((("zsoil", 1, NSOIL), ("btrani", 1, NSOIL), ("smceq", 1, NSOIL),
          ("ficeold", -NSNOW + 1, 0)))
    + _s("isnow", "ponding", "canliq", "canice", "tv", "snowh", "sneqv")
    + _f((("snice", -NSNOW + 1, 0), ("snliq", -NSNOW + 1, 0),
          ("stc", -NSNOW + 1, NSOIL), ("zsnso", -NSNOW + 1, NSOIL),
          ("dzsnso", -NSNOW + 1, NSOIL),
          ("sh2o", 1, NSOIL), ("sice", 1, NSOIL), ("smc", 1, NSOIL)))
    + _s("zwt", "wa", "wt", "wslake", "smcwtd", "deeprech", "rech", "qtldrn",
         "iramtfi", "iramtmi", "irfirate", "irmirate",
         "acc_qinsur", "acc_qseva")
    + _f((("acc_etrani", 1, NSOIL),))
    + _s("cmc", "ecan", "etran", "fwet", "runsrf", "runsub", "qin", "qdis",
         "ponding1", "ponding2", "qsnbot", "qsnsub", "qsnfro", "qsubc",
         "qfroc", "qfrzc", "qmeltc", "qevac", "qdewc")
)

PROBE_FIELDS = _s(
    "qsnsub", "qseva", "qsnfro", "qsdew", "ponding1_in", "ponding2_in",
    "isnow_post_snow", "sneqv_post_snow", "snoflow", "qsnbot",
    "ponding1", "ponding2")

STAGES = {"param": PARAM_FIELDS, "input": STATE_FIELDS,
          "probe": PROBE_FIELDS, "output": STATE_FIELDS}

# ---------------------------------------------------------------------------
# What the pinned identity does not consume.  Each reason is the WRF statement
# that would have read it, so the claim is falsifiable against the source.
# ---------------------------------------------------------------------------

INERT = {
    "uu":       "declared 5960, never referenced in WATER's body",
    "vv":       "declared 5960, never referenced in WATER's body",
    "qprecc":   "declared 6003 group, never referenced in WATER's body",
    "qprecl":   "declared 6003 group, never referenced in WATER's body",
    "fp":       "declared 6009, never referenced in WATER's body",
    "rain":     "declared 6010, never referenced in WATER's body",
    "snow":     "declared 6011, never referenced in WATER's body",
    "latheav":  "declared 6056, never referenced in WATER's body",
    "latheag":  "declared 6057, never referenced in WATER's body",
    "irrfra":   "declared 6071, never referenced in WATER's body",
    "smceq":    "read only by SHALLOWWATERTABLE at 6244 (OPT_RUN==5)",
    "wa":       "written only by GROUNDWATER at 6228 and at 6249 (OPT_RUN 1/5)",
    "wt":       "written only by GROUNDWATER at 6228 (OPT_RUN==1)",
    "rech":     "written only by SHALLOWWATERTABLE at 6245 (OPT_RUN==5)",
    "mifac":    "passed only to MICRO_IRRIGATION at 6199 (opt_irr=0)",
    "fifac":    "passed only to FLOOD_IRRIGATION at 6190 (opt_irr=0)",
    "iramtfi":  "gate at 6188 is false for every state opt_irr=0 can produce",
    "iramtmi":  "gate at 6196 is false for every state opt_irr=0 can produce",
    "irfirate": "written only by FLOOD_IRRIGATION at 6191 (opt_irr=0)",
    "irmirate": "written only by MICRO_IRRIGATION at 6200 (opt_irr=0)",
    "croplu":   "only guards the two irrigation gates at 6188/6196",
    "vegtyp":   "forwarded to CANWATER (6268) and SOILWATER (7268); inert in both",
    "tg":       "forwarded to CANWATER at 6118; declared 6289, never referenced",
    "iloc":     "forwarded only; inert in CANWATER, SNOWWATER, SOILWATER",
    "jloc":     "forwarded only; inert in CANWATER, SNOWWATER, SOILWATER",
    "dx":       "forwarded to SOILWATER; read only by TILE_HOOGHOUDT (opt_tdrn=0)",
    "tdfracmp": "forwarded to SOILWATER; read only in the OPT_TDRN gates",
    "zwt":      "forwarded to SOILWATER; read only in its OPT_RUN 1/2/5 blocks",
    "smcwtd":   "forwarded to SOILWATER; written only in its OPT_RUN==5 block",
    "deeprech": "forwarded to SOILWATER; written only in its OPT_RUN==5 block",
    "qin":      "INTENT(OUT) at 6052; assigned only by GROUNDWATER (OPT_RUN==1)",
    "qdis":     "INTENT(OUT) at 6053; assigned only by GROUNDWATER (OPT_RUN==1)",
    "zsnso":    "write-only: SNOWWATER rebuilds it at 6522-6533 from ISNOW, "
                "DZSNSO and ZSOIL and never reads the entry value",
}

# IRAMTFI/IRAMTMI cannot be positive under opt_irr=0, so the fixture drives
# them negative in the probes.  That is a measurement of the 6188/6196 gates'
# operator, not a forecast state, and it is the only non-zero value the pinned
# identity permits.  Recorded here so the choice is visible rather than buried.
NEGATIVE_ONLY = {"iramtfi", "iramtmi"}

# Arguments no live statement writes.  Checked on every case, not just probes.
PASS_THROUGH = {
    "nsnow", "nsoil", "vegtyp", "ist", "iloc", "jloc", "imelt",
    "dt", "uu", "vv", "fcev", "fctr", "qprecc", "qprecl", "elai", "esai",
    "sfctmp", "qvap", "qdew", "tg", "fveg", "bdfall", "fp", "rain", "snow",
    "qsnow", "qrain", "snowhin", "latheav", "latheag", "dx", "tdfracmp",
    "frozen_canopy", "frozen_ground", "croplu", "irrfra", "mifac", "fifac",
    "zsoil", "btrani", "smceq", "ficeold", "ponding",
    "zwt", "wa", "wt", "smcwtd", "deeprech", "rech",
    "iramtfi", "iramtmi", "irfirate", "irmirate",
    "qin", "qdis",
}

# INTENT(OUT) arguments WATER is obliged to write on every path.
MUST_WRITE = ("cmc", "ecan", "etran", "fwet", "ponding1", "ponding2",
              "qsnbot", "qsubc", "qfroc", "qfrzc", "qmeltc", "qevac", "qdewc")

# Entry values that must NOT survive, and the statement that kills them.
DEAD_ENTRY = {
    "runsrf": "RUNSRF = 0.0 at 6110",
    "runsub": "RUNSUB = 0.0 at 6109",
    "qtldrn": "QTLDRN = 0.0 at 6112",
    "qsnsub": "QSNSUB = 0.0 at 6126",
    "qsnfro": "QSNFRO = 0.0 at 6132",
}

INERT_PROBES = {
    "wat_inert_probe": ("wat_bare_rain", tuple(sorted(
        set(INERT) - {"qin", "qdis"}))),
    "wat_lake_inert_probe": ("wat_lake_filling", tuple(sorted(
        set(INERT) - {"qin", "qdis"}))),
}

OUT_ENTRY_PROBES = {
    "wat_out_entry_probe": "wat_bare_rain",
    "wat_snow_out_entry_probe": "wat_snow_sublimation",
}


# ---------------------------------------------------------------------------
# Branch coverage.  Every predicate reads inputs and probes only, so coverage
# is assertable from the fixture's own entry state and cannot be satisfied by
# coincidence in an output.
# ---------------------------------------------------------------------------

def _b(x, f, i=0):
    return x[(f, i)]


def _sice1_after_frozen(x):
    """SICE(1) after 6146, from the inputs and the probe alone."""
    delta = _f32(_f32(_f32(_b(x, "qsdew") - _b(x, "qseva")) * _b(x, "dt"))
                 / _f32(_b(x, "dzsnso", 1) * 1000.0))
    return _f32(_b(x, "sice", 1) + delta)


BRANCHES = [
    ("SNEQV > 0.0, so QSNSUB/QSNFRO take the pack (6127/6133)",
     lambda p, x: _b(x, "sneqv") > 0.0),
    ("MIN(QVAP, SNEQV/DT) picks SNEQV/DT (6128)",
     lambda p, x: _b(x, "sneqv") > 0.0
     and _f32(_b(x, "sneqv") / _b(x, "dt")) < _b(x, "qvap")),
    ("QDEW > 0, so QSNFRO/QSDEW are discriminated (6134/6136)",
     lambda p, x: _b(x, "qdew") > 0.0),
    ("FROZEN_GROUND, the SICE(1) surface exchange (6145)",
     lambda p, x: _b(x, "frozen_ground") == 1),
    ("FROZEN_GROUND drives SICE(1) below zero, the SH2O fixup (6149)",
     lambda p, x: _b(x, "frozen_ground") == 1 and _sice1_after_frozen(x) < 0.0),
    ("ISNOW == 0, so QRAIN enters QINSUR (6161)",
     lambda p, x: _b(x, "isnow") == 0),
    ("PONDING non-zero, so the 6159 term is live",
     lambda p, x: _b(x, "ponding") != 0.0),
    ("QSNBOT non-zero out of SNOWH2O's percolation (6162/6164)",
     lambda p, x: _b(x, "qsnbot") > 0.0),
    ("PONDING1 non-zero out of SNOWWATER's COMBINE (6159)",
     lambda p, x: _b(x, "ponding1") > 0.0),
    ("SNOFLOW non-zero, so RUNSUB += SNOFLOW*DT is live (6259)",
     lambda p, x: _b(x, "snoflow") > 0.0),
    ("the pack disappears inside the call, ISNOW < 0 -> 0",
     lambda p, x: _b(x, "isnow") < 0 and _b(x, "isnow_post_snow") == 0),
    ("IST == 2, the lake balance (6209)",
     lambda p, x: _b(x, "ist") == 2),
    ("WSLAKE >= WSLMAX, the lake spills (6211)",
     lambda p, x: _b(x, "ist") == 2 and _b(x, "wslake") >= WSLMAX),
    ("NROOT < NSOIL, so the ETRANI loop stops short (6169)",
     lambda p, x: p[("nroot", 0)] < NSOIL),
    ("BTRANI(NSOIL) non-zero, so the loop bound is observable (6170)",
     lambda p, x: _b(x, "btrani", NSOIL) != 0.0),
    ("the accumulators enter non-zero, the 6178/6205 aliasing",
     lambda p, x: _b(x, "acc_qinsur") != 0.0 or _b(x, "acc_qseva") != 0.0
     or any(_b(x, "acc_etrani", k) != 0.0 for k in range(1, NSOIL + 1))),
    ("urban, SOILWATER overwrites FCR(1) at 7361",
     lambda p, x: p[("urban_flag", 0)] == 1),
    ("frozen soil column, SOILWATER's FCR EXP is live (7334)",
     lambda p, x: any(_b(x, "sice", k) > 0.0 for k in range(1, NSOIL + 1))),
    ("a step other than 1800 s, so DT is discriminated",
     lambda p, x: _b(x, "dt") != 1800.0),
    ("CROPLU true with both irrigation amounts at zero (6188/6196)",
     lambda p, x: _b(x, "croplu") == 1),
    ("SNOWFALL creates the first layer, so SFCTMP reaches STC(0) (6570-6577)",
     lambda p, x: _b(x, "isnow") == 0 and _b(x, "isnow_post_snow") < 0),
    ("COMPACT's melt term is live: FICEOLD above the current ice fraction "
     "with IMELT==1 (7052-7056)",
     lambda p, x: any(
         _b(x, "imelt", j) == 1
         and _b(x, "snice", j) + _b(x, "snliq", j) > 0.0
         and _b(x, "ficeold", j) > _f32(_b(x, "snice", j)
                                        / _f32(_b(x, "snice", j)
                                               + _b(x, "snliq", j)))
         for j in range(_b(x, "isnow") + 1, 1))),
]


# ---------------------------------------------------------------------------

class Failure(Exception):
    pass


def _f32(v: float) -> float:
    return struct.unpack(">f", struct.pack(">f", v))[0]


def _from_bits(bits: str) -> float:
    return struct.unpack(">f", bytes.fromhex(bits))[0]


def load(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise Failure(f"{path}: empty")
    table = defaultdict(dict)
    raw = defaultdict(dict)
    order = []
    for r in rows:
        if r["leaf"] != "water":
            raise Failure(f"unexpected leaf {r['leaf']!r}")
        case, stage = r["case"], r["stage"]
        if case not in order:
            order.append(case)
        key = (r["field"], int(r["index"]))
        if r["dtype"] == "int":
            table[(case, stage)][key] = int(r["value"])
            raw[(case, stage)][key] = r["value"]
        else:
            bits = r["bits"].upper()
            want = float(r["value"])
            if struct.pack(">f", want).hex().upper() != bits:
                raise Failure(
                    f"{case}/{stage}/{key}: decimal {r['value']} does not "
                    f"round-trip to bits {bits}")
            table[(case, stage)][key] = _from_bits(bits)
            raw[(case, stage)][key] = bits
    return table, raw, order, len(rows)


def check_shape(table, order):
    for case in order:
        for stage, want in STAGES.items():
            got = table.get((case, stage))
            if got is None:
                raise Failure(f"{case}: stage {stage!r} is missing")
            if tuple(sorted(got)) != tuple(sorted(want)):
                missing = sorted(set(want) - set(got))
                extra = sorted(set(got) - set(want))
                raise Failure(f"{case}/{stage}: field set differs; "
                              f"missing={missing} extra={extra}")
    for (case, stage) in table:
        if stage not in STAGES:
            raise Failure(f"{case}: unexpected stage {stage!r}")


def check_sentinels(raw, order):
    bad = []
    for case in order:
        d = raw[(case, "output")]
        for name in MUST_WRITE:
            if d[(name, 0)] == SENTINEL_BITS:
                bad.append(f"{case}/{name}")
    if bad:
        raise Failure("the -999.0 sentinel survived into an output WATER is "
                      f"obliged to write: {bad}")


def check_pass_through(raw, order):
    bad = []
    for case in order:
        xi, xo = raw[(case, "input")], raw[(case, "output")]
        for (f, i) in STATE_FIELDS:
            if f in PASS_THROUGH and xi[(f, i)] != xo[(f, i)]:
                bad.append(f"{case}/{f}[{i}] {xi[(f, i)]} -> {xo[(f, i)]}")
    if bad:
        raise Failure("arguments the pinned identity does not write moved: "
                      + "; ".join(bad))
    return [f"pass-through: {len(PASS_THROUGH)} argument names bit-identical "
            f"across all {len(order)} cases, QIN/QDIS among them"]


def check_qtldrn_zero(raw, order):
    bad = [c for c in order if raw[(c, "output")][("qtldrn", 0)] != "00000000"]
    if bad:
        raise Failure(f"opt_tdrn=0 but QTLDRN is non-zero on {bad}")
    return ["qtldrn: identically 0.0 on every case (6112 then 6254)"]


def check_dead_entry(table, raw, order):
    report = []
    for probe, base in OUT_ENTRY_PROBES.items():
        if (probe, "input") not in table:
            raise Failure(f"missing out-entry probe {probe}")
        pi, bi = table[(probe, "input")], table[(base, "input")]
        for name in DEAD_ENTRY:
            if name == "qtldrn":
                continue
            if pi[(name, 0)] == bi[(name, 0)]:
                raise Failure(f"{probe}: {name} is not actually perturbed "
                              f"against {base}")
        for (f, i) in STATE_FIELDS:
            if f in PASS_THROUGH or f in DEAD_ENTRY:
                continue
            if pi[(f, i)] != bi[(f, i)]:
                raise Failure(f"{probe}: {f}[{i}] differs from {base} but is "
                              f"not a declared dead-entry slot")
        po, bo = raw[(probe, "output")], raw[(base, "output")]
        moved = [f"{f}[{i}]" for (f, i) in STATE_FIELDS
                 if f not in PASS_THROUGH and po[(f, i)] != bo[(f, i)]]
        if moved:
            raise Failure(
                f"{probe}: perturbing {sorted(DEAD_ENTRY)} moved {moved}, so "
                f"at least one entry value survives")
        report.append(f"{probe}: entry values of {sorted(DEAD_ENTRY)} are dead "
                      f"against {base}")
    return report


def check_discrimination(table, order):
    problems = []
    for (f, i) in STATE_FIELDS:
        if f in INERT:
            continue
        vals = [table[(c, "input")][(f, i)] for c in order]
        if all(v == 0 for v in vals):
            problems.append(f"input {f}[{i}] is zero in every case")
    for name, reason in INERT.items():
        slots = [(f, i) for (f, i) in STATE_FIELDS if f == name]
        if name in NEGATIVE_ONLY:
            vals = [table[(c, "input")][s] for c in order for s in slots]
            if not any(v < 0 for v in vals):
                problems.append(
                    f"inert slot {name} is never driven non-zero, so its "
                    f"inertness is vacuous ({reason})")
            continue
        vals = [table[(c, "input")][s] for c in order for s in slots]
        if all(v == 0 for v in vals):
            problems.append(
                f"inert slot {name} is zero in every case, so its inertness "
                f"is vacuous ({reason})")
    if problems:
        raise Failure("\n".join(problems))


def check_inert_probes(table, raw):
    report = []
    for probe, (base, perturbed) in INERT_PROBES.items():
        if (probe, "input") not in table:
            raise Failure(f"missing inert probe {probe}")
        pi, bi = table[(probe, "input")], table[(base, "input")]
        for name in perturbed:
            slots = [(f, i) for (f, i) in STATE_FIELDS if f == name]
            if all(bi[s] == pi[s] for s in slots):
                raise Failure(f"{probe}: claims to perturb {name} but every "
                              f"slot matches {base}")
        for (f, i) in STATE_FIELDS:
            if f in perturbed or f in DEAD_ENTRY:
                continue
            if bi[(f, i)] != pi[(f, i)]:
                raise Failure(f"{probe}: {f}[{i}] differs from {base} but is "
                              f"not declared perturbed")
        po, bo = raw[(probe, "output")], raw[(base, "output")]
        moved = [f"{f}[{i}]" for (f, i) in STATE_FIELDS
                 if f not in PASS_THROUGH and po[(f, i)] != bo[(f, i)]]
        if moved:
            raise Failure(
                f"{probe}: perturbing {len(perturbed)} arguments moved "
                f"{moved}; they are not inert under the pinned identity")
        report.append(f"{probe}: every output bit-identical to {base} while "
                      f"{len(perturbed)} dead arguments were perturbed")
    return report


def check_branch_coverage(table, raw, order):
    report = []
    problems = []
    for name, pred in BRANCHES:
        taken, nottaken = [], []
        for c in order:
            x = dict(table[(c, "input")])
            x.update(table[(c, "probe")])
            p = table[(c, "param")]
            try:
                hit = bool(pred(p, x))
            except (KeyError, ZeroDivisionError) as exc:
                problems.append(f"`{name}`: predicate failed on {c}: {exc}")
                continue
            (taken if hit else nottaken).append(c)
        if not taken:
            problems.append(f"no case takes `{name}`")
        elif not nottaken:
            problems.append(f"every case takes `{name}`, so the branch is not "
                            f"discriminated")
        else:
            def outs(cs):
                return {tuple(sorted(raw[(c, "output")].items())) for c in cs}
            note = ("" if outs(taken) - outs(nottaken)
                    else "  [!] every taken-side output is reproduced by a "
                         "not-taken case")
            report.append(f"`{name}` taken by {len(taken)}, not taken by "
                          f"{len(nottaken)}{note}")
    if problems:
        raise Failure("\n".join(problems))
    return report


def check_probe_consistency(table, order):
    """The probe stage must agree with the entry state it claims to describe."""
    bad = []
    for c in order:
        x, pr = table[(c, "input")], table[(c, "probe")]
        qsnsub = 0.0
        if x[("sneqv", 0)] > 0.0:
            qsnsub = min(x[("qvap", 0)], _f32(x[("sneqv", 0)] / x[("dt", 0)]))
        qsnfro = x[("qdew", 0)] if x[("sneqv", 0)] > 0.0 else 0.0
        for name, want in (("qsnsub", qsnsub),
                           ("qseva", _f32(x[("qvap", 0)] - qsnsub)),
                           ("qsnfro", qsnfro),
                           ("qsdew", _f32(x[("qdew", 0)] - qsnfro))):
            if struct.pack(">f", pr[(name, 0)]) != struct.pack(">f", want):
                bad.append(f"{c}/{name}")
        if pr[("ponding1_in", 0)] != 0.0 or pr[("ponding2_in", 0)] != 0.0:
            bad.append(f"{c}: the SNOWWATER probe did not enter with zeroed "
                       f"PONDING1/PONDING2")
    if bad:
        raise Failure("probe stage disagrees with the entry state: "
                      + ", ".join(bad))
    return ["probe: the four 6126-6136 scalars re-derive from the inputs"]


def check_probe_csv(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    by = {r["name"].strip(): r for r in rows}
    for need in ("tfrz", "hvap", "hsub", "hfus", "cwat", "cice", "denh2o",
                 "denice", "exp_neg_a_folded", "exp_neg_a_runtim"):
        if need not in by:
            raise Failure(f"{path}: missing probe {need!r}")
    for r in rows:
        if struct.pack(">f", float(r["value"])).hex().upper() != r["bits"].upper():
            raise Failure(f"{path}: {r['name']} does not round-trip")
    folded = by["exp_neg_a_folded"]["bits"].upper()
    runtime = by["exp_neg_a_runtim"]["bits"].upper()
    note = ("identical" if folded == runtime
            else f"DIFFER: folded {folded} vs runtime {runtime}")
    return [f"probe: {len(rows)} rows; EXP(-A) compile-folded vs runtime "
            f"expf: {note}"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        table, raw, order, nrows = load(args.fixture)
        check_shape(table, order)
        check_sentinels(raw, order)
        lines = check_pass_through(raw, order)
        lines += check_qtldrn_zero(raw, order)
        check_discrimination(table, order)
        lines += check_dead_entry(table, raw, order)
        lines += check_inert_probes(table, raw)
        lines += check_probe_consistency(table, order)
        lines += check_branch_coverage(table, raw, order)
        lines += check_probe_csv(args.probe)
    except Failure as exc:
        print(f"validate_water_oracle: FAIL\n{exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"  water: {len(order)} cases")
        for line in lines:
            print(f"  {line}")
        print(f"  {nrows} data rows validated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
