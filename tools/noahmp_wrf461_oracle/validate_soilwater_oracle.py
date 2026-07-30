#!/usr/bin/env python3
"""Validate the Noah-MP soil-water oracle fixture.

Runs inside ``build_soilwater.sh`` before the CSV is allowed out of the build
directory, and again from ``tests/test_noahmp_soilwater.py`` against the
committed copy.  It knows nothing about the port; everything it checks is a
property of the fixture itself.

What it enforces
----------------

1. **Shape.**  Every case of every leaf carries exactly the field/index set
   that leaf declares, at every stage the leaf declares, and no others.

2. **Bit/decimal agreement.**  The ``value`` column must round-trip to the
   ``bits`` column in binary32.  This catches a hand-edited row.

3. **No sentinel survives.**  ``-999.0`` is the harness's uninitialised marker.
   It may appear only where the fixture explicitly pins a pass-through of an
   ``INTENT(OUT)`` argument that WRF leaves unassigned; anywhere else it means
   a routine failed to write an output it is supposed to write.

4. **Discrimination.**  No input slot may be zero in every case unless it is
   declared inert, because a slot that is always zero cannot tell a port that
   reads it from one that ignores it.

5. **Inertness, measured.**  Each ``*_inert_probe`` case names a baseline case
   and a set of arguments it perturbs.  Every output must be bit-identical to
   the baseline's except the pass-through slots, which must equal the probe's
   own entry value.  This turns "these arguments are dead under the pinned
   option identity" from a claim into a row in the CSV.

6. **Branch coverage, from the inputs.**  Each declared branch has a predicate
   evaluated on the *input* columns only, and must be taken by at least one
   case and not taken by at least one case.  A branch whose two sides produce
   identical outputs is called out rather than silently counted.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Leaf declarations.  Field sets are the harness's emit order; see
# run_soilwater.F90.
# ---------------------------------------------------------------------------

PARAM_FIELDS = (
    ("smcmax", 4), ("smcwlt", 4), ("bexp", 4), ("dksat", 4), ("dwsat", 4),
    ("kdt", 1), ("frzx", 1), ("slope", 1), ("ch2op", 1),
    ("timean", 1), ("fsatmx", 1), ("urban_flag", 1),
)


def _fields(spec):
    out = []
    for name, n in spec:
        if n == 1:
            out.append((name, 0))
        else:
            out.extend((name, k) for k in range(1, n + 1))
    return tuple(out)


def _snowsoil(name):
    return tuple((name, k) for k in range(-2, 5))


LEAVES = {
    "canwater": {
        "input": _fields((("vegtyp", 1), ("iloc", 1), ("jloc", 1),
                          ("frozen_canopy", 1), ("dt", 1), ("fcev", 1),
                          ("fctr", 1), ("elai", 1), ("esai", 1), ("tg", 1),
                          ("fveg", 1), ("bdfall", 1), ("canliq", 1),
                          ("canice", 1), ("tv", 1))),
        "output": _fields((("canliq", 1), ("canice", 1), ("tv", 1),
                           ("cmc", 1), ("ecan", 1), ("etran", 1), ("fwet", 1),
                           ("qsubc", 1), ("qfroc", 1), ("qfrzc", 1),
                           ("qmeltc", 1), ("qevac", 1), ("qdewc", 1))),
        # WRF line numbers are the executable statement of what this option
        # identity does not consume.
        "inert": {
            "vegtyp": "declared 6268, never referenced in the body",
            "iloc": "declared 6280, never referenced in the body",
            "jloc": "declared 6279, never referenced in the body",
            "tg": "declared 6289, never referenced in the body",
        },
        "passthrough": {},
        "probes": {"canw_inert_probe": ("canw_unfrozen_evap",
                                        ("vegtyp", "tg"))},
    },
    "infil": {
        "input": (_fields((("nsoil", 1), ("dt", 1)))
                  + tuple(("zsoil", k) for k in range(1, 5))
                  + tuple(("sh2o", k) for k in range(1, 5))
                  + tuple(("sice", k) for k in range(1, 5))
                  + _fields((("sicemax", 1), ("qinsur", 1),
                             ("pddum", 1), ("runsrf", 1)))),
        "output": _fields((("pddum", 1), ("runsrf", 1))),
        "inert": {},
        # INTENT(OUT) but assigned only inside IF (QINSUR > 0.0) at 7655.
        "passthrough": {"pddum": "IF (QINSUR > 0.0) at 7655",
                        "runsrf": "IF (QINSUR > 0.0) at 7655"},
        "probes": {},
    },
    "srt": {
        "input": (_fields((("nsoil", 1), ("iloc", 1), ("jloc", 1), ("dt", 1)))
                  + tuple(("zsoil", k) for k in range(1, 5))
                  + (("pddum", 0),)
                  + tuple(("etrani", k) for k in range(1, 5))
                  + (("qseva", 0),)
                  + tuple(("sh2o", k) for k in range(1, 5))
                  + tuple(("smc", k) for k in range(1, 5))
                  + (("zwt", 0),)
                  + tuple(("fcr", k) for k in range(1, 5))
                  + _fields((("sicemax", 1), ("fcrmax", 1), ("smcwtd", 1)))),
        "output": (tuple(("rhstt", k) for k in range(1, 5))
                   + tuple(("ai", k) for k in range(1, 5))
                   + tuple(("bi", k) for k in range(1, 5))
                   + tuple(("ci", k) for k in range(1, 5))
                   + (("qdrain", 0),)
                   + tuple(("wcnd", k) for k in range(1, 5))),
        "inert": {
            "dt": "declared 7739, never referenced in the body",
            "iloc": "declared 7731, never referenced in the body",
            "jloc": "declared 7732, never referenced in the body",
            "sh2o": "read only in the OPT_INF==2 loop at 7783 and at 7789",
            "zwt": "read only in the OPT_RUN==5 block at 7812",
            "sicemax": "read only in the OPT_INF==2 loop at 7783",
            "fcrmax": "read only in the OPT_RUN==4 block at 7810",
            "smcwtd": "read only at 7779 and 7789, both OPT_RUN==5",
        },
        "passthrough": {},
        "probes": {"srt_inert_probe": ("srt_wet_baseline",
                                       ("dt", "iloc", "jloc", "sh2o", "zwt",
                                        "sicemax", "fcrmax", "smcwtd"))},
    },
    "sstep": {
        "input": (_fields((("nsoil", 1), ("nsnow", 1), ("iloc", 1),
                           ("jloc", 1), ("dt", 1)))
                  + tuple(("zsoil", k) for k in range(1, 5))
                  + _snowsoil("dzsnso")
                  + tuple(("sice", k) for k in range(1, 5))
                  + (("zwt", 0),)
                  + tuple(("sh2o", k) for k in range(1, 5))
                  + tuple(("smc", k) for k in range(1, 5))
                  + tuple(("ai", k) for k in range(1, 5))
                  + tuple(("bi", k) for k in range(1, 5))
                  + tuple(("ci", k) for k in range(1, 5))
                  + tuple(("rhstt", k) for k in range(1, 5))
                  + _fields((("smcwtd", 1), ("qdrain", 1), ("deeprech", 1)))),
        "output": (tuple(("sh2o", k) for k in range(1, 5))
                   + tuple(("smc", k) for k in range(1, 5))
                   + tuple(("ai", k) for k in range(1, 5))
                   + tuple(("bi", k) for k in range(1, 5))
                   + tuple(("ci", k) for k in range(1, 5))
                   + tuple(("rhstt", k) for k in range(1, 5))
                   + _fields((("smcwtd", 1), ("qdrain", 1), ("deeprech", 1),
                              ("wplus", 1)))),
        "inert": {
            "iloc": "declared 7859, never referenced in the body",
            "jloc": "declared 7860, never referenced in the body",
            "zsoil": "read only in the OPT_RUN==5 block at 7927",
            "zwt": "read only in the OPT_RUN==5 block at 7927",
            "smc": "INOUT, unconditionally overwritten by SMC = SH2O + SICE "
                   "at 7971",
        },
        # INOUT but written only inside the OPT_RUN==5 block (7929-7944).
        "passthrough": {"smcwtd": "written only under OPT_RUN==5, 7929-7944",
                        "qdrain": "written only under OPT_RUN==5, 7943",
                        "deeprech": "written only under OPT_RUN==5, 7930/7944"},
        "probes": {"sstep_inert_probe": ("sstep_baseline",
                                         ("zsoil", "zwt", "smc", "iloc",
                                          "jloc", "dzsnso", "smcwtd",
                                          "qdrain", "deeprech"))},
        # The snow slots of DZSNSO only; the soil slots are live.
        "inert_index": {("dzsnso", -2), ("dzsnso", -1), ("dzsnso", 0)},
    },
    "soilwater": {
        "input": (_fields((("nsoil", 1), ("nsnow", 1), ("iloc", 1),
                           ("jloc", 1), ("vegtyp", 1), ("dt", 1)))
                  + tuple(("zsoil", k) for k in range(1, 5))
                  + _snowsoil("dzsnso")
                  + _fields((("qinsur", 1), ("qseva", 1)))
                  + tuple(("etrani", k) for k in range(1, 5))
                  + tuple(("sice", k) for k in range(1, 5))
                  + _fields((("tdfracmp", 1), ("dx", 1)))
                  + tuple(("sh2o", k) for k in range(1, 5))
                  + tuple(("smc", k) for k in range(1, 5))
                  + _fields((("zwt", 1), ("smcwtd", 1), ("deeprech", 1),
                             ("qtldrn", 1), ("runsub", 1)))),
        "probe": _fields((("niter_pddum_in", 1), ("niter_runsrf_in", 1),
                          ("niter_pddum", 1), ("niter_lhs", 1),
                          ("niter_rhs", 1), ("probe_sicemax", 1),
                          ("niter", 1))),
        "output": (tuple(("sh2o", k) for k in range(1, 5))
                   + tuple(("smc", k) for k in range(1, 5))
                   + _fields((("zwt", 1), ("vegtyp", 1), ("smcwtd", 1),
                              ("deeprech", 1), ("qtldrn", 1), ("runsrf", 1),
                              ("qdrain", 1), ("runsub", 1)))
                   + tuple(("wcnd", k) for k in range(1, 5))
                   + (("fcrmax", 0),)),
        "inert": {
            "iloc": "declared 7253, only forwarded to SRT/SSTEP where it is "
                    "inert too",
            "jloc": "declared 7254, likewise",
            "vegtyp": "declared 7268 INOUT, never referenced in the body",
            "dx": "declared 7267, read only by TILE_HOOGHOUDT at 7527",
            "tdfracmp": "declared 7266, read only in the OPT_TDRN gates at "
                        "7521 and 7524",
            "zwt": "read at 7355/7367/7378/7389 (OPT_RUN 1/2/5) and forwarded "
                   "to SRT, where it is OPT_RUN==5 only",
            "smcwtd": "forwarded to SRT/SSTEP, OPT_RUN==5 only in both",
            "deeprech": "written only in SSTEP's OPT_RUN==5 branch and at 7544",
            "qtldrn": "written only by TILE_DRAIN/TILE_HOOGHOUDT, OPT_TDRN/=0",
        },
        # RUNSUB is INTENT(OUT) yet read at 7549 before assignment.
        "passthrough": {"zwt": "no OPT_RUN==3 statement writes it",
                        "smcwtd": "no OPT_RUN==3 statement writes it",
                        "deeprech": "no OPT_RUN==3 statement writes it",
                        "qtldrn": "no OPT_TDRN==0 statement writes it",
                        "vegtyp": "INOUT, never referenced"},
        "probes": {"slw_inert_probe": ("slw_moderate_rain",
                                       ("iloc", "jloc", "vegtyp", "dx",
                                        "tdfracmp", "zwt", "smcwtd",
                                        "deeprech", "qtldrn"))},
        "inert_index": {("dzsnso", -2), ("dzsnso", -1), ("dzsnso", 0)},
    },
}

SENTINEL_BITS = "C479C000"          # -999.0
HALF_SENTINEL_BITS = "C3F9C000"     # -499.5

# Slots where the sentinel is a pinned result, not a failure: an INTENT(OUT)
# argument the routine leaves standing on the path the case takes.
SENTINEL_ALLOWED = {
    ("infil", "infil_qinsur_zero", "output", "pddum", 0),
    ("infil", "infil_qinsur_zero", "output", "runsrf", 0),
}


# ---------------------------------------------------------------------------
# Branch coverage.  Each entry is (leaf, name, predicate over the input dict).
# Predicates read inputs and parameters only, never outputs, so coverage cannot
# be satisfied by coincidence.
# ---------------------------------------------------------------------------

def _b(d, f, i=0):
    return d[(f, i)]


BRANCHES = [
    ("canwater", "frozen_canopy true (6329)",
     lambda p, x: _b(x, "frozen_canopy") == 1),
    ("canwater", "CANLIQ <= 1.0E-06 zeroing (6353)",
     lambda p, x: _b(x, "canliq") <= 1.0e-6 and _b(x, "fcev") == 0.0),
    ("canwater", "CANICE <= 1.0E-6 zeroing (6363)",
     lambda p, x: _b(x, "canice") <= 1.0e-6 and _b(x, "fcev") == 0.0),
    ("canwater", "FWET from the ice branch (6367)",
     lambda p, x: _b(x, "frozen_canopy") == 1 and _b(x, "canice") > 0.0
     and _b(x, "canice") >= _b(x, "canliq")),
    ("canwater", "MAXLIQ = 0, so MAX(MAXLIQ,1.0E-06) floors (6371)",
     lambda p, x: _b(x, "elai") + _b(x, "esai") == 0.0),
    ("canwater", "melt block, CANICE>1e-6 and TV>TFRZ (6381)",
     lambda p, x: _b(x, "canice") > 1.0e-6 and _b(x, "tv") > 273.16
     and _b(x, "frozen_canopy") == 0),
    ("canwater", "freeze block, CANLIQ>1e-6 and TV<TFRZ (6388)",
     lambda p, x: _b(x, "canliq") > 1.0e-6 and _b(x, "tv") < 273.16
     and _b(x, "frozen_canopy") == 0),
    ("canwater", "dew/frost, FCEV < 0 (6331/6337)",
     lambda p, x: _b(x, "fcev") < 0.0),

    ("infil", "QINSUR > 0.0 (7655)", lambda p, x: _b(x, "qinsur") > 0.0),
    ("infil", "DICE > 1.0E-2 frozen correction (7687)",
     lambda p, x: _b(x, "qinsur") > 0.0 and _dice(x) > 1.0e-2),
    ("infil", "SICEMAX > 0, WDFCND2 VKWGT branch (9222)",
     lambda p, x: _b(x, "qinsur") > 0.0 and _b(x, "sicemax") > 0.0),

    ("srt", "FCR non-zero, so WDFCND1's (1-FCR) is live (9180)",
     lambda p, x: any(_b(x, "fcr", k) > 0.0 for k in range(1, 5))),
    ("srt", "SMC/SMCMAX below the 0.01 FACTR floor (9177)",
     lambda p, x: any(_b(x, "smc", k) / p[("smcmax", k)] < 0.01
                      for k in range(1, 5))),
    ("srt", "FACTR at 1.0, WDF=DWSAT and WCND=DKSAT (9177)",
     lambda p, x: all(_b(x, "smc", k) >= p[("smcmax", k)]
                      for k in range(1, 5))),
    ("srt", "negative source terms",
     lambda p, x: _b(x, "qseva") < 0.0
     or any(_b(x, "etrani", k) < 0.0 for k in range(1, 5))),

    ("sstep", "SICE above SMCMAX, EPORE takes the 1.0E-4 floor (7952)",
     lambda p, x: all(p[("smcmax", k)] - _b(x, "sice", k) <= 1.0e-4
                      for k in range(1, 5))),
    ("sstep", "a longer step, DT = 1800 (7896-7901)",
     lambda p, x: _b(x, "dt") >= 1800.0),

    ("soilwater", "NITER doubled to 6 (7443)",
     lambda p, x: _b(x, "niter") == 6),
    ("soilwater", "QINSUR > 0, so INFIL runs inside the loop (7458)",
     lambda p, x: _b(x, "qinsur") > 0.0),
    ("soilwater", "urban FCR(1) = 0.95 (7361)",
     lambda p, x: p[("urban_flag", 0)] == 1),
    ("soilwater", "supersaturated entry, RSAT accumulates (7327)",
     lambda p, x: any(_b(x, "sh2o", k)
                      > max(1.0e-4, p[("smcmax", k)] - _b(x, "sice", k))
                      for k in range(1, 5))),
    ("soilwater", "frozen soil, FICE and the FCR EXP are live (7334)",
     lambda p, x: any(_b(x, "sice", k) > 0.0 for k in range(1, 5))),
    ("soilwater", "FICE saturates at 1.0 (7334)",
     lambda p, x: all(_b(x, "sice", k) >= p[("smcmax", k)]
                      for k in range(1, 5))),
    ("soilwater", "RUNSUB entered non-zero, the 7549 aliasing (7280)",
     lambda p, x: _b(x, "runsub") != 0.0),
]


def _dice(x):
    """INFIL's DICE at 7658-7666, from the input columns alone."""
    zsoil = [x[("zsoil", k)] for k in range(1, 5)]
    sice = [x[("sice", k)] for k in range(1, 5)]
    d = -zsoil[0] * sice[0]
    for k in range(1, 4):
        d += (zsoil[k - 1] - zsoil[k]) * sice[k]
    return d


# ---------------------------------------------------------------------------

class Failure(Exception):
    pass


def _f32(bits: str) -> float:
    return struct.unpack(">f", bytes.fromhex(bits))[0]


def load(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise Failure(f"{path}: empty")
    table = defaultdict(dict)
    raw = defaultdict(dict)
    for r in rows:
        leaf, case, stage = r["leaf"], r["case"], r["stage"]
        key = (r["field"], int(r["index"]))
        if r["dtype"] == "int":
            table[(leaf, case, stage)][key] = int(r["value"])
            raw[(leaf, case, stage)][key] = r["value"]
        else:
            bits = r["bits"]
            got = _f32(bits)
            want = float(r["value"])
            if struct.pack(">f", want).hex().upper() != bits.upper():
                raise Failure(
                    f"{leaf}/{case}/{stage}/{key}: decimal {r['value']} does "
                    f"not round-trip to bits {bits}")
            table[(leaf, case, stage)][key] = got
            raw[(leaf, case, stage)][key] = bits.upper()
    return table, raw, len(rows)


def check_shape(table):
    seen = defaultdict(set)
    for (leaf, case, stage) in table:
        if stage == "param":
            continue
        if leaf not in LEAVES:
            raise Failure(f"unknown leaf {leaf!r}")
        seen[leaf].add(case)
        want = LEAVES[leaf].get(stage)
        if want is None:
            raise Failure(f"{leaf}/{case}: unexpected stage {stage!r}")
        got = tuple(sorted(table[(leaf, case, stage)]))
        if got != tuple(sorted(want)):
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            raise Failure(
                f"{leaf}/{case}/{stage}: field set differs; "
                f"missing={missing} extra={extra}")
    for leaf, spec in LEAVES.items():
        if leaf not in seen:
            raise Failure(f"leaf {leaf!r} has no cases")
        for case in seen[leaf]:
            pk = tuple(sorted(table[(leaf, case, "param")]))
            if pk != tuple(sorted(_fields(PARAM_FIELDS))):
                raise Failure(f"{leaf}/{case}: parameter block differs")
    return {leaf: sorted(cases) for leaf, cases in seen.items()}


def check_sentinels(raw):
    bad = []
    for (leaf, case, stage), d in raw.items():
        if stage != "output":
            continue
        for (f, i), bits in d.items():
            if bits in (SENTINEL_BITS, HALF_SENTINEL_BITS):
                if (leaf, case, stage, f, i) not in SENTINEL_ALLOWED:
                    bad.append(f"{leaf}/{case}/{f}[{i}]")
    if bad:
        raise Failure("sentinel survived into an output that WRF must "
                      f"write: {bad}")


def check_discrimination(table, cases):
    problems = []
    for leaf, spec in LEAVES.items():
        inert = spec.get("inert", {})
        inert_index = spec.get("inert_index", set())
        for (f, i) in spec["input"]:
            if f in inert or (f, i) in inert_index:
                continue
            vals = [table[(leaf, c, "input")][(f, i)] for c in cases[leaf]]
            if all(v == 0 for v in vals):
                problems.append(f"{leaf}: input {f}[{i}] is zero in every case")
        # An inert slot that is zero everywhere proves nothing either.
        for name, reason in inert.items():
            vals = [table[(leaf, c, "input")][(f, i)]
                    for c in cases[leaf] for (f, i) in spec["input"] if f == name]
            if all(v == 0 for v in vals):
                problems.append(
                    f"{leaf}: inert slot {name} is zero in every case, so its "
                    f"inertness is vacuous ({reason})")
    if problems:
        raise Failure("\n".join(problems))


def check_inert_probes(table, raw):
    """Every declared probe must reproduce its baseline bit for bit."""
    report = []
    for leaf, spec in LEAVES.items():
        for probe, (base, perturbed) in spec.get("probes", {}).items():
            bi = table[(leaf, base, "input")]
            pi = table[(leaf, probe, "input")]
            if not bi or not pi:
                raise Failure(f"{leaf}: probe {probe} or baseline {base} missing")
            # The probe must actually perturb every argument it claims to.
            for name in perturbed:
                slots = [(f, i) for (f, i) in spec["input"] if f == name]
                inert_index = spec.get("inert_index", set())
                slots = [s for s in slots
                         if name not in spec.get("inert", {}) or True]
                if name == "dzsnso":
                    slots = [s for s in slots if s in inert_index]
                if all(bi[s] == pi[s] for s in slots):
                    raise Failure(
                        f"{leaf}/{probe}: claims to perturb {name} but every "
                        f"slot matches the baseline")
            # Everything the leaf reads must be identical.
            for (f, i) in spec["input"]:
                if f in perturbed:
                    continue
                if f == "dzsnso" and "dzsnso" in perturbed:
                    continue
                if bi[(f, i)] != pi[(f, i)]:
                    raise Failure(
                        f"{leaf}/{probe}: {f}[{i}] differs from {base} but is "
                        f"not declared perturbed")
            bo = raw[(leaf, base, "output")]
            po = raw[(leaf, probe, "output")]
            passthrough = set(spec.get("passthrough", {}))
            moved = []
            for (f, i) in spec["output"]:
                if f in passthrough:
                    # Must echo the probe's own entry value.
                    if raw[(leaf, probe, "input")].get((f, i)) is not None:
                        if po[(f, i)] != raw[(leaf, probe, "input")][(f, i)]:
                            raise Failure(
                                f"{leaf}/{probe}: pass-through {f}[{i}] "
                                f"changed")
                    continue
                if bo[(f, i)] != po[(f, i)]:
                    moved.append(f"{f}[{i}]")
            if moved:
                raise Failure(
                    f"{leaf}/{probe}: perturbing {perturbed} moved {moved}; "
                    f"they are not inert under the pinned identity")
            report.append(f"{leaf}/{probe}: {len(spec['output'])} outputs "
                          f"bit-identical to {base} while {len(perturbed)} "
                          f"arguments were perturbed")
    return report


def check_branch_coverage(table, raw, cases):
    report = []
    problems = []
    for leaf, name, pred in BRANCHES:
        taken, nottaken = [], []
        for c in cases[leaf]:
            x = dict(table[(leaf, c, "input")])
            if (leaf, c, "probe") in table:
                x.update(table[(leaf, c, "probe")])
            p = table[(leaf, c, "param")]
            try:
                hit = bool(pred(p, x))
            except (KeyError, ZeroDivisionError) as exc:
                problems.append(f"{leaf}/{name}: predicate failed on {c}: {exc}")
                continue
            (taken if hit else nottaken).append(c)
        if not taken:
            problems.append(f"{leaf}: no case takes `{name}`")
        elif not nottaken:
            problems.append(f"{leaf}: every case takes `{name}`, so the "
                            f"branch is not discriminated")
        else:
            def outs(cs):
                return {tuple(sorted(raw[(leaf, c, "output")].items()))
                        for c in cs}
            note = ("" if outs(taken) - outs(nottaken)
                    else "  [!] every taken-side output is reproduced by a "
                         "not-taken case")
            report.append(f"{leaf}: `{name}` taken by {len(taken)}, not taken "
                          f"by {len(nottaken)}{note}")
    if problems:
        raise Failure("\n".join(problems))
    return report


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
        table, raw, nrows = load(args.fixture)
        cases = check_shape(table)
        check_sentinels(raw)
        check_discrimination(table, cases)
        lines = check_inert_probes(table, raw)
        lines += check_branch_coverage(table, raw, cases)
        lines += check_probe_csv(args.probe)
    except Failure as exc:
        print(f"validate_soilwater_oracle: FAIL\n{exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        for leaf in sorted(cases):
            print(f"  {leaf}: {len(cases[leaf])} cases")
        for line in lines:
            print(f"  {line}")
        print(f"  {nrows} data rows validated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
