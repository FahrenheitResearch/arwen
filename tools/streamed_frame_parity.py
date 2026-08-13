"""Is a streamed wrfout the SAME FILE a resident run of that config writes?

THE CHECK THAT WOULD HAVE CAUGHT IT.  The 2.2.0 headline claim was that a
streamed forecast's output is bit-identical to a resident one.  Every gate
the release had compared VALUES of fields both files carried, and the
defect was a field only ONE of them carried: a streamed run published 73 of
the resident run's 74 variables, omitting ``OLR``, with forecast validity
PASS and health ``status_bits`` 0.  A value comparison over the
intersection cannot see that, and neither can a health check, a validity
check, or a NaN sweep.

So this compares three things in this order, and the FIRST is the one that
was missing:

1. **The field set.**  Names present in one file and not the other, both
   directions.  A frame missing rows a resident run writes is not the same
   frame -- and the direction matters too, since a streamed run that
   invented a row would be just as wrong.
2. **Every shared field, byte for byte.**  ``tobytes()`` equality, not
   ``allclose``: the claim under test is bit-identity, and a tolerance
   would quietly certify the reduction-ordering differences the claim
   exists to deny.  Differences are reported with ``maxabs`` and the COUNT
   of differing cells, because one cell at 1e-10 and every cell at 1e-10
   are different findings.
3. **Global attributes.**  Which is how the second defect surfaced:
   ``GPUWM_CARRIER_GLW_SOURCE`` read ``unwritten`` on a streamed file whose
   ``GLW`` array was byte-identical to the resident one and correct.
   Provenance metadata that contradicts the data beside it is worse than
   none, so attribute equality is part of parity rather than a footnote.

USAGE, and it is a GATE -- exit 0 only on full parity::

    python -m tools.streamed_frame_parity RESIDENT.nc STREAMED.nc
    python -m tools.streamed_frame_parity A.nc B.nc --json receipt.json
    python -m tools.streamed_frame_parity A.nc B.nc --allow-attr SIMULATION_START_DATE

``--allow-attr`` is deliberately not populated by default.  An attribute
that legitimately differs between two runs of one configuration is a claim
that needs making explicitly at the call site, not a list that grows in
this file until the check means nothing.

VALIDATE THE INSTRUMENT.  ``tests/test_streamed_frame_parity.py`` runs it
in BOTH directions -- a pair it must call identical, and four seeded
corruptions (a dropped variable, an added variable, one flipped bit, one
changed attribute) it must catch -- because an equality checker that
always says "equal" passes every parity gate ever written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def compare(resident, streamed, *, allow_attrs=()) -> dict:
    """Compare two wrfout files and return the finding, never raising.

    ``resident`` is the reference and ``streamed`` the file under test;
    the names are not symmetric in the REPORT (``missing`` versus
    ``extra``) even though the comparison itself is, because "the streamed
    run dropped a field" and "the streamed run invented one" are different
    defects and a symmetric report makes the reader work out which.
    """
    import numpy as np
    from netCDF4 import Dataset

    allow = {str(a) for a in allow_attrs}
    with Dataset(str(resident)) as a, Dataset(str(streamed)) as b:
        names_a, names_b = set(a.variables), set(b.variables)
        missing = sorted(names_a - names_b)
        extra = sorted(names_b - names_a)

        identical, differing = 0, []
        for name in sorted(names_a & names_b):
            xa = np.ma.filled(np.asanyarray(a.variables[name][:]), 0)
            ya = np.ma.filled(np.asanyarray(b.variables[name][:]), 0)
            if xa.shape != ya.shape:
                differing.append({"name": name, "kind": "shape",
                                  "resident": str(xa.shape),
                                  "streamed": str(ya.shape)})
                continue
            if xa.tobytes() == ya.tobytes():
                identical += 1
                continue
            row = {"name": name, "kind": "bytes"}
            if xa.dtype.kind in "fc":
                delta = np.abs(xa.astype("f8") - ya.astype("f8"))
                row["maxabs"] = float(np.nanmax(delta))
                row["ndiff"] = int((delta != 0).sum())
                row["ncells"] = int(delta.size)
            differing.append(row)

        attrs_a = {k: str(getattr(a, k)) for k in a.ncattrs()}
        attrs_b = {k: str(getattr(b, k)) for k in b.ncattrs()}
        attr_diff = [
            {"name": k, "resident": attrs_a.get(k), "streamed": attrs_b.get(k)}
            for k in sorted(set(attrs_a) | set(attrs_b))
            if k not in allow and attrs_a.get(k) != attrs_b.get(k)]

    return {
        "resident": str(resident),
        "streamed": str(streamed),
        "field_set_equal": not missing and not extra,
        "missing_from_streamed": missing,
        "extra_in_streamed": extra,
        "identical_fields": identical,
        "differing_fields": differing,
        "attrs_differing": attr_diff,
        "allowed_attrs": sorted(allow),
        "parity": (not missing and not extra and not differing
                   and not attr_diff),
    }


def render(result: dict) -> str:
    """The finding as text, leading with whichever half failed."""
    lines = [f"streamed frame parity: {Path(result['streamed']).name}",
             f"  reference: {result['resident']}"]
    if result["missing_from_streamed"]:
        lines.append(
            "  FIELD SET: the streamed frame is MISSING "
            f"{result['missing_from_streamed']} -- a frame short of rows a "
            "resident run writes is not the same frame")
    if result["extra_in_streamed"]:
        lines.append(
            "  FIELD SET: the streamed frame carries "
            f"{result['extra_in_streamed']}, which the resident run does not")
    if not result["missing_from_streamed"] and not result["extra_in_streamed"]:
        lines.append(f"  field set: EQUAL "
                     f"({result['identical_fields'] + len(result['differing_fields'])} "
                     "variables both sides)")
    lines.append(f"  bit-identical fields: {result['identical_fields']}")
    for row in result["differing_fields"][:25]:
        if row["kind"] == "shape":
            lines.append(f"  DIFFERS {row['name']}: shape "
                         f"{row['resident']} vs {row['streamed']}")
        else:
            extra = ""
            if "maxabs" in row:
                extra = (f" maxabs={row['maxabs']:.6g} "
                         f"ndiff={row['ndiff']}/{row['ncells']}")
            lines.append(f"  DIFFERS {row['name']}:{extra}")
    for row in result["attrs_differing"][:25]:
        lines.append(f"  ATTR {row['name']}: resident={row['resident']!r} "
                     f"streamed={row['streamed']!r}")
    lines.append("  PARITY" if result["parity"] else "  NO PARITY")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a streamed wrfout against a resident one.")
    parser.add_argument("resident", type=Path)
    parser.add_argument("streamed", type=Path)
    parser.add_argument("--allow-attr", action="append", default=[],
                        help="a global attribute permitted to differ; state "
                             "each one explicitly, they are never defaulted")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the finding as a receipt")
    args = parser.parse_args(argv)

    result = compare(args.resident, args.streamed,
                     allow_attrs=args.allow_attr)
    print(render(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True),
                             encoding="utf-8")
    return 0 if result["parity"] else 1


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
