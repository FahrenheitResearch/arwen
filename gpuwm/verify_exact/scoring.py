"""One verdict per replayed run, from its comparisons.  CPU only.

Outcomes (CASE.md's list, in the command's spelling):

``PASS``            0 ULP on every reference field at every step against a sound,
                    designated referee (no field skipped: see ``ALLOWED_ABSENT``).
``FAIL``            a difference against a sound, designated referee, or a
                    reference field WOOF did not write or wrote in another shape.
``WOOF-REFUSED``    WOOF's door or loader refused the combo, or the runner raised a
                    named ValueError before writing its first history frame.
``WOOF-CRASH``      WOOF stopped before the referee's last good step, every
                    frame it wrote matching; or, with no referee, WOOF did not
                    write every planned step.
``WOOF-NONFINITE``  no referee, and WOOF's state went NaN or infinite.
``REPEAT-MISS``     the combo asks for a WOOF repeat run and the two differ, or
                    the repeat run is missing.
``UNSOUND-REF``     the referee is not a sound 0 ULP reference, whether WOOF
                    matches it or not: the recording flags it (no repeat,
                    out-of-bounds read, unfixed Thompson), it has no
                    ``clean_reference`` entry (unknown), or it did not finish
                    (a crashed or non-finite referee).  Never a pass.
``INFO-MATCH``/``INFO-MISS``  scored against stock 4.6.1 for information only:
                    the combo's designated reference is another build.
``WOOF-ONLY-OK``    no referee exists and WOOF finished finite (and repeated).
``WRF-REFUSED``     WRF refused the combo; there was nothing to replay.

The referee is the scalar build by default (WOOF's own float32 math is scalar,
RECORD.md), or its Thompson-fixed variant where one was recorded.  Every other
recorded build is scored too and reported on the same line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gpuwm.verify_exact.compare import Comparison

#: Outcomes that make the command exit nonzero.
FAIL_OUTCOMES = frozenset({"FAIL", "WOOF-REFUSED", "WOOF-CRASH", "WOOF-NONFINITE",
                           "REPEAT-MISS"})


@dataclass
class Verdict:
    combo: str
    case: str
    stratum: str
    outcome: str
    referee: str | None = None
    referee_sound: bool = True
    #: ``sound``, or why not: ``unsound: ...``, ``unknown: ...``, ``incomplete: ...``.
    referee_soundness: str = ""
    designated: bool = True
    woof_status: str = ""
    woof_error: str = ""
    woof_nonfinite_step: int | None = None
    wall_s: float = 0.0
    phases: dict = field(default_factory=dict)
    wrf_note: str = ""
    comparisons: dict[str, Comparison] = field(default_factory=dict)
    repeat: Comparison | None = None
    declared_divergences: tuple[str, ...] = ()
    factors: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.outcome in FAIL_OUTCOMES

    def detail(self) -> str:
        if self.outcome == "WRF-REFUSED":
            return self.wrf_note
        if self.outcome == "WOOF-REFUSED":
            return self.woof_error
        parts = []
        if self.referee and self.referee in self.comparisons:
            parts.append(f"vs {self.referee}: {self.comparisons[self.referee].summary()}")
        elif not self.referee:
            parts.append("no WRF reference" + (f" ({self.wrf_note})" if self.wrf_note else ""))
        others = [f"{build}: {_short(c)}" for build, c in self.comparisons.items()
                  if build != self.referee]
        if others:
            parts.append("also " + ", ".join(others))
        if self.woof_status == "CRASH":
            parts.append("WOOF " + self.woof_error[:160])
        if self.woof_nonfinite_step is not None:
            parts.append(f"WOOF non-finite from step {self.woof_nonfinite_step}")
        if self.repeat is not None:
            parts.append("repeat " + _short(self.repeat))
        if self.referee and not self.referee_sound:
            parts.append("referee not sound (" + (self.referee_soundness or "unsound") + ")")
        return " | ".join(parts)

    def line(self) -> str:
        detail = " / ".join(part.strip() for part in self.detail().splitlines() if part.strip())
        return f"{self.combo:<16} {self.case:<22} {self.outcome:<14} {detail}"

    def as_json(self) -> dict:
        return {
            "combo": self.combo, "case": self.case, "stratum": self.stratum,
            "outcome": self.outcome, "failed": self.failed, "referee": self.referee,
            "referee_sound": self.referee_sound,
            "referee_soundness": self.referee_soundness, "designated": self.designated,
            "woof": {"status": self.woof_status, "error": self.woof_error,
                     "nonfinite_step": self.woof_nonfinite_step, "wall_s": self.wall_s,
                     "phases": dict(self.phases)},
            "wrf_note": self.wrf_note,
            "comparisons": {b: c.as_json() for b, c in self.comparisons.items()},
            "repeat": None if self.repeat is None else self.repeat.as_json(),
            "declared_divergences": list(self.declared_divergences),
            "factors": dict(self.factors), "line": self.line(),
        }


def _short(comparison: Comparison) -> str:
    if not comparison.steps_compared:
        return "no frame to compare"
    if comparison.first_step is None:
        if comparison.missing_steps:
            return "match until WOOF stopped"
        return "match" if not comparison.uncompared else (
            f"match, {len(comparison.uncompared)} fields not compared")
    text = f"step {comparison.first_step} {comparison.lead_field}"
    state = comparison.state_first
    if state is not None and state[1][0] != comparison.lead_field:
        text += f", state step {state[0]} {state[1][0]}"
    return text


def decide(*, woof_status: str, nonfinite_step: int | None, referee: str | None,
           comparisons: dict[str, Comparison], referee_sound: bool, designated: bool,
           repeat: Comparison | None, repeat_required: bool = False,
           planned_last_step: int | None = None, woof_last_step: int | None = None) -> str:
    """The outcome for one run.  See the module docstring.

    Soundness is checked before any PASS: a match against a referee that is
    not a sound 0 ULP reference proves nothing, so it is UNSOUND-REF.  A
    match needs every reference field compared (``Comparison.matched``).
    With no referee, WOOF must have written every planned step
    (``planned_last_step``, the last history step the plan reaches) to be
    WOOF-ONLY-OK.  A combo that asks for a repeat run needs one.
    """

    if woof_status == "REFUSED":
        return "WOOF-REFUSED"
    repeat_missed = (repeat is None and repeat_required) or (
        repeat is not None and not repeat.matched)
    if referee is None or referee not in comparisons:
        if woof_status == "CRASH":
            return "WOOF-CRASH"
        if nonfinite_step is not None:
            return "WOOF-NONFINITE"
        if planned_last_step is not None and (woof_last_step is None
                                              or woof_last_step < planned_last_step):
            return "WOOF-CRASH"
        if repeat_missed:
            return "REPEAT-MISS"
        return "WOOF-ONLY-OK"
    scored = comparisons[referee]
    if scored.first_step is None and (scored.missing_steps or not scored.steps_compared):
        return "WOOF-CRASH"
    if not designated:
        return "INFO-MATCH" if scored.matched else "INFO-MISS"
    if not referee_sound:
        return "UNSOUND-REF"
    if not scored.matched:
        return "FAIL"
    if repeat_missed:
        return "REPEAT-MISS"
    return "PASS"
