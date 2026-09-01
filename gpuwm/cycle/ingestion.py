"""Did the analysis actually land, or did the cycle quietly drop it?

Those two outcomes look identical from outside: both produce a forecast,
both produce receipts, both run to completion.  The only difference is
that one of them is assimilating radar and the other is a very expensive
free forecast wearing a DA cycle's clothes.  This module is what tells
them apart, and it does it with three hashes taken at three different
places by two different processes:

1. ``background_sha256`` -- the parent state the SPINE wrote into the
   anchor, hashed by the spine.
2. the increment itself -- its sha, its nonzero cell count, and a per
   field L2, so a refusal can say what was thrown away.
3. ``analysis_sha256`` -- the state after application, hashed by the
   FORECAST process off its own rehydrated device state, downloaded to
   host.  Taken anywhere else it proves nothing: the interesting failure
   is precisely that the increment never reached the device.

**Both directions are gated, because an exact-zero delta means the
experiment never ran.**  A nonzero increment that leaves the state
bit-identical is a dropped analysis.  A zero increment that changes the
state means rehydration itself is perturbing the atmosphere, which is a
different and worse bug wearing the same clothes.  Neither is allowed to
pass silently, and both arms are exercised in the tests.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gpuwm.cycle.anchor import prognostic_sha256
from gpuwm.cycle.contracts import CycleRefusal, INGESTION_SCHEMA


def increment_summary(increment: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Hash, count and size an analysis increment.

    ``increment_nonzero_cells`` is the count of cells the analysis
    actually touched, summed over fields -- the number a refusal quotes
    when it says the increment was not free.
    """
    fields = sorted(increment)
    nonzero = 0
    l2: dict[str, float] = {}
    for name in fields:
        array = np.asarray(increment[name])
        nonzero += int(np.count_nonzero(array))
        l2[name] = float(np.sqrt(np.sum(np.square(
            array.astype(np.float64, copy=False)))))
    return {
        "schema": INGESTION_SCHEMA,
        "increment_sha256": prognostic_sha256(increment),
        "increment_nonzero_cells": nonzero,
        "increment_l2": l2,
        "fields": fields,
    }


def verify_ingestion(*, background_sha256: str,
                     increment: Mapping[str, np.ndarray],
                     analysis_sha256: str, label: str) -> dict[str, Any]:
    """The gate.  Returns the receipt block, or refuses in either direction.

    The returned keys are exactly ``background_sha256``,
    ``increment_sha256``, ``increment_nonzero_cells``, ``increment_l2``,
    ``analysis_sha256`` and ``state`` (plus ``schema`` and ``fields`` for
    the reader's benefit); that block is what the supervisor writes into
    the transition receipt as ``analysis["ingestion"]``.
    """
    summary = increment_summary(increment)
    nonzero = summary["increment_nonzero_cells"]
    unchanged = str(background_sha256) == str(analysis_sha256)

    if nonzero > 0 and unchanged:
        raise CycleRefusal(
            "analysis was not ingested",
            label=label,
            increment_nonzero_cells=nonzero,
            increment_l2=summary["increment_l2"],
            increment_sha256=summary["increment_sha256"],
            shared_sha256=str(background_sha256),
            fields=summary["fields"],
            remedy="check that the increment reached the device state "
                   "before the first step, not the host copy that was "
                   "discarded at rehydration")

    if nonzero == 0 and not unchanged:
        raise CycleRefusal(
            "null-increment arm is not bit-stable",
            label=label,
            increment_nonzero_cells=nonzero,
            background_sha256=str(background_sha256),
            analysis_sha256=str(analysis_sha256),
            fields=summary["fields"],
            remedy="rehydration itself perturbed the state; compare the "
                   "restored prognostics field by field against the "
                   "anchor before assimilating anything")

    return {
        "schema": INGESTION_SCHEMA,
        "background_sha256": str(background_sha256),
        "increment_sha256": summary["increment_sha256"],
        "increment_nonzero_cells": nonzero,
        "increment_l2": summary["increment_l2"],
        "analysis_sha256": str(analysis_sha256),
        "fields": summary["fields"],
        "state": "APPLIED" if nonzero > 0 else "NULL_ARM",
    }
