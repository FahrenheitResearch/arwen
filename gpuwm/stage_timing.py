"""What a stage receipt's ``timing_seconds`` block has to add up to.

Every long stage in this tree publishes a ``timing_seconds`` mapping in
its receipt: a handful of named children and a ``total``.  The block is
only worth reading if the children account for the total -- a receipt
whose named work is 60% of its own wall clock tells a reader where 60%
of the time went and lies by omission about the rest.

MEASURED, 2026-08-16, which is why this module exists: the GFS prepare
stage at the documented normal size took 36.0 s and its proof.json
attributed decode 1.8 + init 9.9 + cache-write 1.1 + export 7.9 =
20.7 s.  The missing 14.1 s -- 40%, the single largest item -- was the
static-field build, timed only behind an opt-in environment variable.
Under this project's fixed-means-default law that is not instrumentation
at all, and the number a reader needed most was the one they could not
get without knowing to ask for it.

The rule this module states is deliberately loose in one direction and
strict in the other.  Children may overlap or nest slightly and may miss
a few milliseconds of bookkeeping between phases, so the bar is a
FRACTION rather than equality; but a child that is not there at all
cannot be recovered by any reader, so the bar exists.
"""

from __future__ import annotations

from typing import Any, Mapping

#: The key every receipt's ``timing_seconds`` uses for its own wall.
TOTAL_KEY = "total"

#: The fraction of ``total`` the named children must account for.
#:
#: 0.9 leaves room for the process bookkeeping between phases -- receipt
#: assembly, digests, the atomic publish -- while refusing the case this
#: exists to catch, a whole phase that is simply not named.
MINIMUM_COVERAGE = 0.9


def timing_children(timing: Mapping[str, Any]) -> dict[str, float]:
    """The named children of one ``timing_seconds`` block.

    Everything except ``total`` and anything that is not a number: a
    receipt may carry a nested breakdown or a note beside its numbers,
    and a coverage check that choked on one would push call sites into
    keeping a second, parallel list of "the real keys".
    """

    return {
        str(key): float(value)
        for key, value in timing.items()
        if key != TOTAL_KEY and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def timing_coverage(timing: Mapping[str, Any]) -> float | None:
    """What fraction of ``total`` the named children account for.

    ``None`` when the question cannot be asked -- no ``total``, a
    non-numeric one, or a zero one (a stage that took no measurable time
    has nothing to attribute).  ``None`` is never a failure: a caller
    that cannot tell "uncovered" from "unaskable" would refuse receipts
    from the fastest runs.
    """

    total = timing.get(TOTAL_KEY)
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        return None
    total = float(total)
    if total <= 0.0:
        return None
    return sum(timing_children(timing).values()) / total


def timing_coverage_shortfall(timing: Mapping[str, Any], *, what: str,
                              minimum: float = MINIMUM_COVERAGE
                              ) -> str | None:
    """Why this receipt does not account for itself, or ``None``.

    The sentence names the fraction, the seconds that are unattributed
    and the keys that ARE there, because the reader who meets it is
    about to go looking for the phase nobody timed and the existing
    keys are the map of where they have already looked.
    """

    coverage = timing_coverage(timing)
    if coverage is None or coverage >= minimum:
        return None
    total = float(timing[TOTAL_KEY])
    children = timing_children(timing)
    named = ", ".join(sorted(children)) or "(none)"
    return (
        f"{what} attributes {coverage * 100:.0f}% of its own "
        f"{total:.1f} s: {total - sum(children.values()):.1f} s is not "
        f"named by any key in timing_seconds, which carries {named}.  A "
        "receipt whose largest item is the unattributed remainder cannot "
        "answer the question it exists to answer.")


__all__ = [
    "MINIMUM_COVERAGE", "TOTAL_KEY", "timing_children", "timing_coverage",
    "timing_coverage_shortfall",
]
