#!/usr/bin/env python3
"""Mutation and branch-coverage study of the gpuwm RUC ``snowtemp`` port.

Bitwise parity against a fixture is only as strong as the fixture's ability
to notice the port being wrong.  This asks two separate questions, because
one mutation operator cannot answer both:

1. **Value mutation.**  Replace exactly one read of one argument-derived
   local with a very different value, then ask whether the fixtures still
   reproduce bit for bit.  A survivor at an *arithmetic* site means the
   fixtures do not pin that arithmetic at all.  At a *branch predicate* a
   value change is only observable if it flips the branch, so survivors
   there are reported separately and answered by (2).
2. **Branch coverage.**  Is each arm of each ``if`` in the per-column body
   executed by some fixture row?  An unexecuted arm is arithmetic no
   fixture can pin, whatever the mutation operator.

Run against the pinned six-regime fixture alone and it reports 113
survivors and 11 unexecuted arms; run against both fixtures and it reports
60 survivors and 5 unexecuted arms, all of which
``EXPECTED_SURVIVORS``/``UNREACHABLE_ARMS`` below account for.  Exits
non-zero when anything falls outside those sets.

usage: mutation_study_snowtemp.py [ORACLE.csv ...]
       (default: gpuwm/data/ruc/oracle/snowtemp{,_contract}.csv)
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "gpuwm" / "core" / "ruc.py"
ORACLE_DIR = ROOT / "gpuwm" / "data" / "ruc" / "oracle"
FUNCTIONS = ("_ruc_snow_thermal_diffusivity", "ruc_snow_temperature_step")

TARGETS = {
    "thdif", "cap", "tranf", "tso",
    "snwe", "snwepr", "snhei", "newsnow", "snowfrac", "beta", "deltsn",
    "snth", "rhosn", "rhonewsn", "meltfactor", "prcpms", "rainf", "patm",
    "tabs", "qvatm", "emiss", "rnet", "qkms", "tkms", "rho", "vegfrac",
    "drycan", "wetcan", "transum", "dew", "soilt", "soilt1", "qvg",
    "timestep", "constant_flux_depth", "latent_heat", "water_heat_capacity",
    "root_count", "snow_layers", "zsmain", "zshalf", "dtdzs", "table",
}

# Arithmetic-site survivors that remain against both fixtures, keyed by the
# stripped source line and the mutated name so they survive renumbering.
# Every one is accounted for; nothing here is merely untested.
#
# (a) WRF dead stores -- the value written is unconditionally overwritten
#     before any read, so no fixture can observe it:
#       :5076 soiltfrac=soilt         -> :5416 or :5649
#       :5159-5160 tsob/soilt1        -> :5357-5358
#       :5164-5165, :5192-5194, :5191-5192 tsnav -> :5715-5725
#       :5172 tsob=soilt1             -> :5353
#       :5180-5181 soilt1/tsob        -> :5364-5365
#       :5207 epot                    -> :5422 or :5559, else never read
# (b) unreachable because h == 1 exactly (:5209), so tx2 == 0 and q1 == qs1:
#       :5299 tq2=qvatm, :5317 the second vilka call
# (c) unreachable because snhei == 0 is rejected -- WRF reads the
#     uninitialised snprim and tsob there (:5459, :5626):
#       :5221 r22 (read only at :5264), :5369-5371, :5626
# (d) semantically equivalent mutants that no fixture can kill:
#       :5339 soilt=min(273.15,soilt) is guarded by soilt > 273.15, so
#             raising soilt leaves the minimum at 273.15
#       :5436 the transp loop bound -- WRF transf leaves tranf zero below
#             the root zone, so extra iterations add zero to ett1
EXPECTED_SURVIVORS = frozenset({
    ("soiltfrac = soilt", "soilt"),
    ("tsob = tso[0]", "tso"),
    ("soilt1 = tso[0]", "tso"),
    ("np.float32(half * np.float32(soilt + tso[0])) - freeze", "soilt"),
    ("np.float32(half * np.float32(soilt + tso[0])) - freeze", "tso"),
    ("tsob = soilt1", "soilt1"),
    ("tsnav = np.float32(np.float32(soilt + soilt1) * deltsn)", "soilt"),
    ("tsnav = np.float32(np.float32(soilt + soilt1) * deltsn)", "soilt1"),
    ("tsnav = np.float32(np.float32(soilt + soilt1) * deltsn)", "deltsn"),
    ("np.float32(soilt1 + tso[0]) * np.float32(snhei - deltsn)", "soilt1"),
    ("np.float32(soilt1 + tso[0]) * np.float32(snhei - deltsn)", "tso"),
    ("np.float32(soilt1 + tso[0]) * np.float32(snhei - deltsn)", "snhei"),
    ("np.float32(soilt1 + tso[0]) * np.float32(snhei - deltsn)", "deltsn"),
    ("np.float32(np.float32(half / snhei) * tsnav) - freeze", "snhei"),
    ("tsob = tso[1]", "tso"),
    (
        "tsnav = np.float32(np.float32(half * np.float32(soilt + tso[0])) - freeze)",
        "soilt",
    ),
    (
        "tsnav = np.float32(np.float32(half * np.float32(soilt + tso[0])) - freeze)",
        "tso",
    ),
    ("epot = np.float32(-np.float32(qkms * np.float32(qvatm - qgold)))", "qkms"),
    ("epot = np.float32(-np.float32(qkms * np.float32(qvatm - qgold)))", "qvatm"),
    (
        "np.float32(thdif[0] * timestep) * np.float32(dzstop * dzstop)",
        "thdif",
    ),
    (
        "np.float32(thdif[0] * timestep) * np.float32(dzstop * dzstop)",
        "timestep",
    ),
    ("tq2 = qvatm", "qvatm"),
    ("qs1, ts1 = _ruc_vilka(tn, aa, bb, pp, table)", "table"),
    ("soilt = min(freeze, soilt)", "soilt"),
    ("tso[0] = soilt", "tso"),
    ("tso[0] = soilt", "soilt"),
    ("soilt1 = soilt", "soilt"),
    ("for level in range(root_count):", "root_count"),
    ("s = np.float32(d9sn * np.float32(soilt - tsob))", "soilt"),
})

# See (b) and (c) above; the non-finite guard is a contract check.
UNREACHABLE_ARMS = frozenset({
    ("if not saturated:", "then"),
    ("if saturated:", "else"),
    ("elif snhei > zero and snhei < snth:", "else"),
    ("elif snhei < snth and snhei > zero:", "else"),
    ("if not np.all(np.isfinite(array)):", "then"),
})

PROLOGUE = '''
def _MUTATE(value):
    import numpy as _np
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    array = _np.asarray(value)
    if array.dtype.kind in "iu":
        return (array + 1).astype(array.dtype)
    shifted = (
        array.astype(_np.float32) * _np.float32(2.0) + _np.float32(3.0)
    ).astype(_np.float32)
    return shifted if array.ndim else _np.float32(shifted)
'''

BEFORE = {
    "snwe": "snwe_before", "snhei": "snhei_before", "beta": "beta_before",
    "rhosn": "rhosn_before", "soilt": "soilt_before",
    "soilt1": "soilt1_before", "qvg": "qvg_before", "dew": "dew_before",
}
AFTER = {
    "soilt": "soilt_after", "soilt1": "soilt1_after", "tsnav": "tsnav_after",
    "qvg": "qvg_after", "qsg": "qsg_after", "qcg": "qcg_after",
    "dew": "dew_after", "snwe": "snwe_after", "snhei": "snhei_after",
    "rhosn": "rhosn_after", "beta": "beta_after", "smelt": "smelt",
    "snoh": "snoh", "snflx": "snflx", "s": "s", "rsm": "rsm",
    "snweprint": "snweprint", "snheiprint": "snheiprint", "storage": "x",
}


class Wrap(ast.NodeTransformer):
    """Wrap exactly the Name load at one (lineno, col_offset, id)."""

    def __init__(self, target):
        self.target = target
        self.hits = 0

    def visit_Name(self, node):
        key = (node.lineno, node.col_offset, node.id)
        if isinstance(node.ctx, ast.Load) and key == self.target:
            self.hits += 1
            return ast.Call(
                func=ast.Name(id="_MUTATE", ctx=ast.Load()),
                args=[node],
                keywords=[],
            )
        return node


def load(source: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = "gpuwm.core"
    sys.modules[name] = module
    try:
        exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def read_oracle(path: Path):
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if len(rows) != len(cases) * 9:
        raise SystemExit(f"{path.name}: {len(rows)} rows for {len(cases)} cases")
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(len(cases), 9)
        .T
        for key in rows[0]
        if key != "case"
    }
    return cases, fields


def reproduces(module, fixtures, trace=None) -> bool:
    columns = module.RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    step = module.ruc_snow_temperature_step
    for cases, fields in fixtures:
        for case in range(len(cases)):
            sel = [case]
            values = {
                name: np.ascontiguousarray(fields[name][:, sel])
                for name in ("cap", "thdif", "tranf")
            }
            values["tso"] = np.ascontiguousarray(fields["tso_before"][:, sel])
            values.update({
                name: np.ascontiguousarray(
                    fields[BEFORE.get(name, name)][0, sel]
                )
                for name in columns
            })
            if trace is not None:
                sys.settrace(trace)
            try:
                result = step(
                    values,
                    delt=float(fields["delt"][0, case]),
                    conflx=float(fields["conflx"][0, case]),
                    nroot=int(fields["nroot"][0, case]),
                    ilnb=int(fields["ilnb_before"][0, case]),
                    xlvm=float(fields["xlvm"][0, case]),
                    cvw=float(fields["cvw"][0, case]),
                )
            except Exception:
                return False
            finally:
                if trace is not None:
                    sys.settrace(None)
            if not np.array_equal(
                result.tso[:, 0].view(np.uint32),
                fields["tso_after"][:, case].view(np.uint32),
            ):
                return False
            for name, column in AFTER.items():
                if not np.array_equal(
                    getattr(result, name).view(np.uint32),
                    fields[column][0, case : case + 1].view(np.uint32),
                ):
                    return False
            if int(result.ilnb[0]) != int(fields["ilnb_after"][0, case]):
                return False
    return True


def main(paths: list[Path]) -> None:
    fixtures = [read_oracle(path) for path in paths]
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    if len(functions) != len(FUNCTIONS):
        raise SystemExit(f"expected {FUNCTIONS} in {SOURCE}")
    step = next(n for n in functions if n.name == "ruc_snow_temperature_step")
    loop = next(
        n for n in ast.walk(step)
        if isinstance(n, ast.For) and getattr(n.target, "id", "") == "column"
    )

    predicates = set()
    for function in functions:
        for node in ast.walk(function):
            if isinstance(node, (ast.If, ast.While, ast.IfExp)):
                for name in ast.walk(node.test):
                    if (
                        isinstance(name, ast.Name)
                        and isinstance(name.ctx, ast.Load)
                        and name.id in TARGETS
                    ):
                        predicates.add(
                            (name.lineno, name.col_offset, name.id)
                        )
    keys = sorted({
        (n.lineno, n.col_offset, n.id)
        for function in functions
        for n in ast.walk(function)
        if isinstance(n, ast.Name)
        and isinstance(n.ctx, ast.Load)
        and n.id in TARGETS
    })

    baseline = load(text + PROLOGUE, "ruc_snowtemp_baseline")
    if not reproduces(baseline, fixtures):
        raise SystemExit("the unmutated port does not reproduce the fixtures")
    print(f"fixtures: {', '.join(p.name for p in paths)} "
          f"({sum(len(c) for c, _ in fixtures)} regimes)")
    print(f"{len(keys)} argument read sites in {', '.join(FUNCTIONS)}")

    survivors = []
    for index, key in enumerate(keys):
        wrap = Wrap(key)
        mutated = wrap.visit(ast.parse(text))
        if wrap.hits != 1:
            raise SystemExit(f"mutation key {key} matched {wrap.hits} sites")
        ast.fix_missing_locations(mutated)
        module = load(
            ast.unparse(mutated) + PROLOGUE, f"ruc_snowtemp_mutant_{index}"
        )
        if reproduces(module, fixtures):
            survivors.append(key)

    guard = [k for k in survivors if k[0] < loop.lineno]
    predicate = [
        k for k in survivors if k[0] >= loop.lineno and k in predicates
    ]
    arithmetic = [
        (lines[k[0] - 1].strip(), k[2])
        for k in survivors
        if k[0] >= loop.lineno and k not in predicates
    ]
    print(f"\nsurvivors: {len(survivors)} of {len(keys)}")
    print(f"  {len(guard)} in the input-validation prologue "
          f"(the contract-drift tests cover these, not the fixtures)")
    print(f"  {len(predicate)} at branch predicates "
          f"(answered by the coverage report below)")
    print(f"  {len(arithmetic)} at arithmetic sites")

    unexpected = sorted(set(arithmetic) - EXPECTED_SURVIVORS)
    fixed = sorted(EXPECTED_SURVIVORS - set(arithmetic))
    for name, site in unexpected:
        print(f"  UNEXPECTED SURVIVOR [{site}]  {name}")
    for name, site in fixed:
        print(f"  no longer a survivor (update EXPECTED_SURVIVORS) [{site}]"
              f"  {name}")

    executed: set[int] = set()

    def tracer(frame, event, arg):
        if frame.f_code.co_filename == str(SOURCE):
            if event == "line":
                executed.add(frame.f_lineno)
            return tracer
        return None

    if not reproduces(baseline, fixtures, trace=tracer):
        raise SystemExit("the traced baseline diverged")
    unexecuted = set()
    for node in ast.walk(step):
        if not isinstance(node, ast.If) or node.lineno < loop.lineno:
            continue
        test = lines[node.lineno - 1].strip()
        for label, body in (("then", node.body), ("else", node.orelse)):
            if body and body[0].lineno not in executed:
                unexecuted.add((test, label))
    print(f"\nunexecuted if-arms in the physics body: {len(unexecuted)}")
    for test, label in sorted(unexecuted):
        mark = "" if (test, label) in UNREACHABLE_ARMS else "  UNEXPECTED"
        print(f"  {label:4s} arm of  {test}{mark}")
    surprises = sorted(unexecuted - UNREACHABLE_ARMS)
    covered = sorted(UNREACHABLE_ARMS - unexecuted)
    for test, label in covered:
        print(f"  now reachable (update UNREACHABLE_ARMS)  {label} of {test}")

    if unexpected or surprises:
        raise SystemExit(
            "the fixtures leave arithmetic or a reachable branch unpinned"
        )
    print("\nRUC snowtemp mutation study: PASS "
          "(every surviving mutant is a WRF dead store, unreachable on the "
          "supported domain, or semantically equivalent)")


if __name__ == "__main__":
    argv = [Path(a) for a in sys.argv[1:]]
    main(argv or [
        ORACLE_DIR / "snowtemp.csv",
        ORACLE_DIR / "snowtemp_contract.csv",
    ])
