"""Shared seam for gpuwm's cycling spine: schemas, ticks, refusals.

Three producers build the spine in parallel; this module is the one place
their vocabularies meet.  It carries schema strings, the tick type, and
the refusal base class -- no model state, no I/O, no cupy.  It is safe to
import from a receipt reader, from a test, and from a process that will
never open a GPU.

The tick is one millisecond of MODEL time.  Every time in the cycling
system is an integer number of ticks since the epoch anchor, and the
epoch anchor is the only ``datetime`` in the spine.  A dt that is not a
whole number of milliseconds is refused by name rather than rounded,
because a rounded dt is how a child drifts from its parent over a long
cycle with nothing reporting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

CONTRACTS_VERSION = "gpuwm-cycle.contracts/v1"

CLOCK_SCHEMA = "gpuwm-cycle.clock/v1"
ANCHOR_SCHEMA = "gpuwm-cycle.anchor/v1"
LEDGER_SCHEMA = "gpuwm-cycle.ledger/v1"
TRANSITION_SCHEMA = "gpuwm-cycle.transition/v1"
PLACEMENT_SCHEMA = "gpuwm-cycle.placement/v1"
INGESTION_SCHEMA = "gpuwm-cycle.analysis-ingestion/v1"
CONSISTENCY_SCHEMA = "gpuwm-cycle.state-consistency/v1"

TICK_HZ = 1000

#: The parents an anchor may be stamped with.  Each name is a CLAIM about
#: what produced the state, and the vocabulary is closed so that an
#: unrecognised claim is refused rather than written.
#:
#: ``mpas-cuda``
#:     The port's device stack integrated this cycle IN PROCESS, from the
#:     analysed state.  A closed loop: the analysis re-entered the dycore.
#: ``mpas-cuda-frames``
#:     The REAL frozen CUDA dycore produced this cycle's forecast
#:     tendency, but OUT OF PROCESS -- the leg composed a recorded real
#:     history series with the analysis instead of re-integrating from
#:     it.  Everything here is real model output; what is missing is the
#:     anchor -> device-stack round trip, so the analysis did NOT feed
#:     back into the dycore.  Distinct from ``mpas-cuda`` on purpose: a
#:     reader must never mistake an open loop for a closed one.
#: ``arwen``
#:     The ArWen child engine.
#: ``replay``
#:     The LOUD STUB.  No dycore integrated anything.
PARENT_KINDS = ("mpas-cuda", "mpas-cuda-frames", "arwen", "replay")

ANALYSIS_STATES = ("APPLIED", "SKIPPED_NO_OBS", "REJECTED", "NULL_ARM")

HALT_REASONS = ("ANALYSIS_REJECTED", "ANALYSIS_NOT_INGESTED",
                "PARENT_DIVERGED", "STALE_ANALYSIS_BUDGET_EXHAUSTED",
                "CLOCK_UNARMED")

CHILD_STATES = ("PLANNED", "LIVE", "RETIRED", "REFUSED", "DIVERGED")


class CycleRefusal(RuntimeError):
    """A fail-closed refusal that is required to name what it observed.

    Constructing one without observations is itself an error: a refusal
    that says only "invalid placement" costs an hour of somebody's night,
    and this program has paid that hour more than once.
    """

    def __init__(self, message: str, **observed: object) -> None:
        if not observed:
            raise ValueError(
                "a cycle refusal must name what was observed; "
                f"got bare message {message!r}")
        rendered = ", ".join(f"{key}={observed[key]!r}"
                             for key in sorted(observed))
        super().__init__(f"{message} | observed: {rendered}")
        self.observed = dict(observed)


def seconds_to_ticks(seconds: float, *, label: str) -> int:
    """Exact seconds -> integer model ticks, or a refusal naming ``label``."""
    scaled = (Fraction(str(float(seconds))).limit_denominator(1_000_000)
              * TICK_HZ)
    if scaled.denominator != 1:
        raise CycleRefusal(
            "time is not a whole number of model ticks",
            label=label, seconds=float(seconds), tick_hz=TICK_HZ,
            remainder_ticks=float(scaled - int(scaled)))
    if scaled <= 0:
        raise CycleRefusal("time must be positive",
                           label=label, seconds=float(seconds))
    return int(scaled)


def ticks_to_seconds(ticks: int) -> float:
    return int(ticks) / float(TICK_HZ)


@dataclass(frozen=True, slots=True)
class TickRatio:
    """A child's step expressed against its parent's, exactly."""

    parent_ticks: int
    child_ticks: int

    def __post_init__(self) -> None:
        if self.child_ticks <= 0 or self.parent_ticks <= 0:
            raise CycleRefusal("tick ratio needs positive steps",
                               parent_ticks=self.parent_ticks,
                               child_ticks=self.child_ticks)
        if self.parent_ticks % self.child_ticks:
            raise CycleRefusal(
                "child step does not divide the parent step exactly",
                parent_ticks=self.parent_ticks,
                child_ticks=self.child_ticks,
                remainder=self.parent_ticks % self.child_ticks)

    @property
    def ratio(self) -> int:
        return self.parent_ticks // self.child_ticks
