"""Pair coverage of the combo list, counted over what actually ran.

The combo list is built so that every valid pair of option values sits in
at least one combo of its stratum.  That is a property of the list.  What a
sweep proves is narrower: a pair is exercised only by a run that WOOF
integrated and scored against a sound, designated referee (PASS or FAIL),
and proved only by a run that passed.  WRF refusals, WOOF refusals,
unsound or unknown referees and informational builds exercise nothing.
CPU only, no numpy.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping

#: Outcomes in which WOOF integrated and was scored against a sound,
#: designated referee.
INTEGRATED_SOUND = frozenset({"PASS", "FAIL"})


def _pairs(factors: Mapping[str, object]) -> set[frozenset]:
    items = sorted((str(k), _key(v)) for k, v in factors.items())
    return {frozenset(pair) for pair in combinations(items, 2)}


def _key(value):
    # JSON values: keep 1 and True apart, and make lists hashable.
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, list):
        return ("list", tuple(_key(v) for v in value))
    return value


def _plain(value):
    if isinstance(value, tuple) and value and value[0] == "bool":
        return value[1]
    if isinstance(value, tuple) and value and value[0] == "list":
        return [_plain(v) for v in value[1]]
    return value


def _order(value):
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    return (1, 0.0, repr(value))


def pair_coverage(combos: Mapping[str, Mapping], outcomes: Mapping[tuple[str, str], str]
                  ) -> list[dict]:
    """One row per stratum present in ``outcomes``.

    ``combos`` maps combo id to its record (``stratum``, ``factors``);
    ``outcomes`` maps (combo id, case) to the run's outcome.  ``pairs_in_list``
    counts the distinct option-value pairs over every combo of the stratum in
    the list; the other counts are over the scored runs only.
    """

    strata = sorted({str(combos.get(cid, {}).get("stratum", "")) for cid, _ in outcomes})
    rows = []
    for stratum in strata:
        listed = {cid: rec for cid, rec in combos.items()
                  if str(rec.get("stratum", "")) == stratum}
        design: set[frozenset] = set()
        values: dict[str, set] = {}
        for rec in listed.values():
            factors = rec.get("factors") or {}
            design |= _pairs(factors)
            for k, v in factors.items():
                values.setdefault(str(k), set()).add(_key(v))
        sound: set[frozenset] = set()
        passed: set[frozenset] = set()
        seen_values: dict[str, set] = {}
        runs = {"scored": 0, "integrated_sound": 0, "passed": 0}
        for (cid, _case), outcome in sorted(outcomes.items()):
            if cid not in listed:
                continue
            runs["scored"] += 1
            if outcome not in INTEGRATED_SOUND:
                continue
            factors = listed[cid].get("factors") or {}
            runs["integrated_sound"] += 1
            sound |= _pairs(factors)
            for k, v in factors.items():
                seen_values.setdefault(str(k), set()).add(_key(v))
            if outcome == "PASS":
                runs["passed"] += 1
                passed |= _pairs(factors)
        never = {k: sorted((_plain(v) for v in vs - seen_values.get(k, set())), key=_order)
                 for k, vs in sorted(values.items()) if vs - seen_values.get(k, set())}
        rows.append({
            "stratum": stratum, "combos_in_list": len(listed), "runs": runs,
            "pairs_in_list": len(design),
            "pairs_integrated_sound": len(sound & design),
            "pairs_passed": len(passed & design),
            "values_never_integrated_sound": never,
        })
    return rows


def describe(row: Mapping) -> str:
    """One line for the command's tail."""

    total = row["pairs_in_list"]

    def share(n: int) -> str:
        return f"{n} ({100.0 * n / total:.1f} %)" if total else f"{n}"

    line = (f"coverage {row['stratum'] or '-'}: pairs {total} in the list; "
            f"{share(row['pairs_integrated_sound'])} integrated against a sound referee; "
            f"{share(row['pairs_passed'])} passed")
    never = row["values_never_integrated_sound"]
    if never:
        shown = "; ".join(f"{k} {', '.join(str(v) for v in vs)}" for k, vs in never.items())
        line += f"; values never integrated against a sound referee: {shown}"
    return line
