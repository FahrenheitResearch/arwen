"""Run the surface-moisture ledger over a column A/B and account for the gap.

WHAT THIS ANSWERS.  When two arms of a run disagree about 2 m moisture, this
says WHICH TERM the disagreement lives in, with a number against each, and
refuses to let the difference vanish into a rounding story.

THE ACCOUNTING.  Q2 is a function of the provider's inputs and nothing else.
For the SFCLAY-family identity ``q2 = qsfc + (qv1 - qsfc)*r`` with
``r = chs/cqs2``, the exact first-order decomposition of a change is

    dQ2  =  (1-r)*dQSFC  +  r*dQV1  +  (QV1-QSFC)*dR  +  (cross terms)

and every term on the right is computed from ledger columns that were
recorded, not inferred.  The cross term is carried explicitly rather than
dropped, so the four named terms plus the cross term reproduce the measured
dQ2 to the FP32 budget.  What is left over after all five is the
UNEXPLAINED REMAINDER, and a remainder above the reporting threshold is
itself a finding: it means the published Q2 did not come from the inputs the
ledger recorded, which is a different and worse defect than any of the terms.

THE ARMS.  ``--arm-a`` and ``--arm-b`` each name a ledger JSON written by a
run.  Arms must share their column set and their times, because an
accounting between different columns is not an accounting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

#: Below this, a remainder is FP32 bookkeeping.  Above it, the accounting is
#: incomplete and the tool says so in its exit status, not only in its text.
REMAINDER_THRESHOLD_KG_KG = 1.0e-4          # 0.1 g/kg


def _load(path: Path) -> dict:
    rows = json.loads(Path(path).read_text())
    if isinstance(rows, dict):
        rows = rows["rows"]
    keyed = {}
    for row in rows:
        keyed[(row["model_time"], row["j"], row["i"])] = row
    return keyed


#: rho = psfc/(RD*tsk) in the Noah SFCDIAGS identity.  The value the
#: engine uses (gpuwm/core/constants.py); restated here because this tool
#: runs standalone on two JSON files.
RD_J_KG_K = 287.0


def decompose(a: dict, b: dict) -> dict:
    """Attribute ``b['q2'] - a['q2']`` to named terms, BY PROVIDER.

    Returns the four physical terms, the cross term, and the remainder.
    Every value is in kg/kg so they add up without a unit conversion in the
    middle, where sign errors hide.

    THE PROVIDER DECIDES THE TERMS.  The SFCLAY-family identity is
    ``q2 = qsfc + (qv1 - qsfc)*r`` with ``r = chs/cqs2``; Noah's SFCDIAGS
    identity is ``q2 = qsfc - qfx*s`` with ``s = 1/(rho*cqs2)`` and
    ``rho = psfc/(RD*tsk)``.  These are different functions of different
    recorded inputs, so a decomposition written for one is not exact for
    the other: run the SFCLAY terms over Noah rows and the whole QFX route
    lands in the "unexplained" remainder, which is exactly what happened
    the first time this tool met a land A/B.  Each branch below is the
    algebraically exact expansion of its own identity, cross term carried,
    so the remainder measures only FP32 publication rounding (and the
    out-of-range guard branch, which substitutes ``qv1`` and is a
    different function again -- a remainder on a guard-branch column is
    reported, not hidden).
    """
    provider_a = a.get("q2_provider")
    dq2 = b["q2"] - a["q2"]
    dqsfc = b["qsfc"] - a["qsfc"]
    dqv1 = b["qv1"] - a["qv1"]

    if (provider_a == "NOAH_SFCDIAGS"
            and b.get("q2_provider") == "NOAH_SFCDIAGS"):
        # q2 = qsfc - qfx*s.  Exact: dq2 = dqsfc - sa*dqfx - qfxa*ds
        #                                   - dqfx*ds.
        def s_of(row):
            cqs2 = row.get("cqs2") or 0.0
            rho = row["psfc"] / (RD_J_KG_K * row["tsk"])
            return (1.0 / (rho * cqs2)) if cqs2 else math.nan

        sa, sb = s_of(a), s_of(b)
        dqfx = b["qfx"] - a["qfx"]
        ds = sb - sa
        term_qsfc = dqsfc
        term_qv1 = -sa * dqfx          # the flux route: QFX through CQS2
        term_exchange = -a["qfx"] * ds  # rho and cqs2 moving under the flux
        term_cross = -dqfx * ds
    else:
        def r_of(row):
            cqs2 = row.get("cqs2") or 0.0
            chs = row.get("chs") or 0.0
            return (chs / cqs2) if cqs2 else math.nan

        ra, rb = r_of(a), r_of(b)
        dr = rb - ra

        # Evaluated at ARM A, with the cross term carried explicitly so the
        # decomposition is exact rather than first-order-approximate.
        term_qsfc = (1.0 - ra) * dqsfc
        term_qv1 = ra * dqv1
        term_exchange = (a["qv1"] - a["qsfc"]) * dr
        term_cross = (dqv1 - dqsfc) * dr
    named = term_qsfc + term_qv1 + term_exchange + term_cross

    # The provider residual is not part of dQ2; it is the check that the
    # identity closed in each arm at all.  Carried alongside because a
    # decomposition of a quantity whose identity did not close is fiction.
    resid_a = a.get("q2_residual", math.nan)
    resid_b = b.get("q2_residual", math.nan)

    noah = (provider_a == "NOAH_SFCDIAGS"
            and b.get("q2_provider") == "NOAH_SFCDIAGS")
    return {
        "model_time": a["model_time"], "j": a["j"], "i": a["i"],
        # What the second and third named terms ARE, per provider, so a
        # receipt reader is never left mapping a SFCLAY label onto a
        # Noah number.
        "term_qv1_meaning": ("QFX flux route (-s*dQFX)" if noah
                             else "QV1 route (r*dQV1)"),
        "term_exchange_meaning": (
            "rho*cqs2 route (-QFX*ds)" if noah
            else "exchange ratio route ((QV1-QSFC)*dr)"),
        "provider_a": a.get("q2_provider"), "provider_b": b.get("q2_provider"),
        "provider_changed": a.get("q2_provider") != b.get("q2_provider"),
        "q2_a": a["q2"], "q2_b": b["q2"], "dq2": dq2,
        "term_qsfc": term_qsfc,
        "term_qv1": term_qv1,
        "term_exchange": term_exchange,
        "term_cross": term_cross,
        "named_total": named,
        "remainder": dq2 - named,
        "residual_a": resid_a, "residual_b": resid_b,
        "identities_closed": bool(
            a.get("q2_within_budget") and b.get("q2_within_budget")),
        "td2_a": a.get("td2_from_q2"), "td2_b": b.get("td2_from_q2"),
        "dtd2": (b.get("td2_from_q2", math.nan)
                 - a.get("td2_from_q2", math.nan)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", required=True, type=Path)
    parser.add_argument("--arm-b", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threshold", type=float,
                        default=REMAINDER_THRESHOLD_KG_KG)
    args = parser.parse_args(argv)

    a_rows, b_rows = _load(args.arm_a), _load(args.arm_b)
    shared = sorted(set(a_rows) & set(b_rows))
    if not shared:
        parser.error("the two arms share no (time, column) key; an "
                     "accounting between different columns is not an "
                     "accounting")
    missing = (set(a_rows) ^ set(b_rows))
    results = [decompose(a_rows[k], b_rows[k]) for k in shared]

    worst = max(results, key=lambda r: abs(r["dq2"]))
    unexplained = [r for r in results
                   if abs(r["remainder"]) > args.threshold]
    unclosed = [r for r in results if not r["identities_closed"]]

    report = {
        "columns_compared": len(results),
        "keys_in_one_arm_only": len(missing),
        "threshold_kg_kg": args.threshold,
        "largest_change": worst,
        "unexplained_remainder_count": len(unexplained),
        "identity_did_not_close_count": len(unclosed),
        "rows": results,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))

    w = worst
    print(f"columns compared        : {len(results)}")
    print(f"largest dQ2             : {w['dq2']*1000:+.4f} g/kg "
          f"at t={w['model_time']:.0f}s ({w['j']},{w['i']})")
    print(f"  QSFC term             : {w['term_qsfc']*1000:+.4f} g/kg")
    print(f"  {w['term_qv1_meaning']:<22}: {w['term_qv1']*1000:+.4f} g/kg")
    print(f"  {w['term_exchange_meaning']:<22}: "
          f"{w['term_exchange']*1000:+.4f} g/kg")
    print(f"  cross term            : {w['term_cross']*1000:+.4f} g/kg")
    print(f"  named total           : {w['named_total']*1000:+.4f} g/kg")
    print(f"  UNEXPLAINED remainder : {w['remainder']*1000:+.4f} g/kg")
    print(f"  provider a/b          : {w['provider_a']} / {w['provider_b']}"
          f"{'  CHANGED' if w['provider_changed'] else ''}")
    print(f"  dTD2                  : {w['dtd2']:+.3f} K")
    print(f"identity did not close  : {len(unclosed)} column-times")
    print(f"remainder over threshold: {len(unexplained)} column-times")

    # A remainder above threshold is a FINDING, so it must reach the caller
    # as a non-zero exit and not only as a line of text someone may skim.
    return 1 if (unexplained or unclosed) else 0


if __name__ == "__main__":
    sys.exit(main())
