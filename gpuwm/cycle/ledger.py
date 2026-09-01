"""The cycling spine's control plane: an append-only ledger of transitions.

Crash recovery is a replay of this file, so it is written the way
:mod:`gpuwm.ensemble.cycle` publishes an analysis -- declare, fsync,
then rename -- because that is the one place in this program that
already gets atomicity right.

Three phases, per record:

  1. the record is serialised into ``cycle_ledger.jsonl.staged`` and
     fsynced, so the bytes exist on the platter before anything claims
     they do;
  2. the staged line is appended to the live log and the log is fsynced;
  3. the staged file is removed.

A crash between 1 and 2 leaves a staged file; :meth:`CycleLedger.settle`
rolls it forward, because the staged bytes were proved before phase two
began.  A crash between 2 and 3 leaves a staged file whose ``seq`` is
already live; settle drops it rather than double-counting.  Rolling
forward is safe in both directions and neither is a guess.

**Reading a directory listing, or the JSONL itself, to discover a
cycle's state is OUT OF CONTRACT.** :meth:`CycleLedger.settle` is the
only supported reader: it settles any in-flight publication first, and a
raw read cannot, so a raw reader sees a half-published ledger and takes
it for a whole one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from gpuwm.cycle.contracts import (CycleRefusal, HALT_REASONS, LEDGER_SCHEMA,
                                   TRANSITION_SCHEMA)

LEDGER_NAME = "cycle_ledger.jsonl"

#: Records the fold in :meth:`CycleLedger.state` understands.  An event
#: outside this set is carried in the log and ignored by the fold -- the
#: ledger is a record of what happened, not only of what it can score.
_FOLDED_EVENTS = ("cycle-completed", "analysis", "child", "halt")


class CycleLedger:
    """Append-only JSONL of cycle transitions, with a settled reader."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.root / LEDGER_NAME

    @property
    def staged_path(self) -> Path:
        return self.root / (LEDGER_NAME + ".staged")

    # -- writing ---------------------------------------------------------

    def append(self, record: Mapping) -> dict:
        """Stamp, stage, fsync, append.  Returns the stamped record."""
        if not isinstance(record, Mapping):
            raise CycleRefusal("a ledger record must be a mapping",
                               got_type=type(record).__name__)
        missing = [key for key in ("event", "cycle") if key not in record]
        if missing:
            raise CycleRefusal(
                "a ledger record must name its event and its cycle",
                missing_keys=missing, got_keys=sorted(record))
        settled = self.settle()
        payload = dict(record)
        payload["schema"] = LEDGER_SCHEMA
        payload["seq"] = len(settled)
        payload["cycle"] = int(record["cycle"])
        payload["written_at"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"

        # Phase one: the bytes exist before anything claims they do.
        with open(self.staged_path, "w", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        # Phase two: the append is the transaction.
        self._commit_line(line)
        # Phase three: the staging slot is free again.
        self.staged_path.unlink(missing_ok=True)
        return payload

    def _commit_line(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    # -- reading ---------------------------------------------------------

    def settle(self) -> list[dict]:
        """Recover any staged write, then return the whole record list.

        This is the only supported reader.  A raw read of
        :attr:`path` sees whatever a crash left behind.
        """
        records = self._read_committed()
        staged = self._read_staged()
        if staged is not None:
            live_seqs = {record.get("seq") for record in records}
            if staged.get("seq") not in live_seqs:
                # Crash between phase one and phase two: roll forward.
                self._commit_line(
                    json.dumps(staged, sort_keys=True, default=str) + "\n")
                records.append(staged)
            self.staged_path.unlink(missing_ok=True)
        return records

    def _read_committed(self) -> list[dict]:
        if not self.path.is_file():
            return []
        records: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except ValueError as exc:
                    raise CycleRefusal(
                        "the cycle ledger carries a line this build cannot "
                        "read; the run's state is unknown and nothing here "
                        "will guess it",
                        path=str(self.path), line_number=lineno,
                        error=str(exc)) from exc
                if record.get("schema") != LEDGER_SCHEMA:
                    raise CycleRefusal(
                        "the cycle ledger declares a schema this build does "
                        "not speak",
                        path=str(self.path), line_number=lineno,
                        expected_schema=LEDGER_SCHEMA,
                        observed_schema=record.get("schema"))
                records.append(record)
        return records

    def _read_staged(self) -> dict | None:
        if not self.staged_path.is_file():
            return None
        raw = self.staged_path.read_text(encoding="utf-8").strip()
        if not raw:
            # Phase one died mid-write; nothing was ever promised.
            self.staged_path.unlink(missing_ok=True)
            return None
        try:
            record = json.loads(raw)
        except ValueError as exc:
            raise CycleRefusal(
                "an interrupted ledger write is unreadable; the last "
                "transition's fate is unknown and rolling forward would be "
                "a guess",
                path=str(self.staged_path), error=str(exc)) from exc
        return record

    # -- the fold --------------------------------------------------------

    def state(self) -> dict:
        """Fold the ledger into the resume state the supervisor needs."""
        last_completed = -1
        halted = False
        halt_reason = None
        streak = 0
        children: dict[str, str] = {}
        for record in self.settle():
            event = record.get("event")
            if event == "cycle-completed":
                last_completed = max(last_completed, int(record["cycle"]))
            elif event == "analysis":
                if record.get("state") == "SKIPPED_NO_OBS":
                    streak += 1
                else:
                    streak = 0
            elif event == "child":
                children[str(record.get("grid_id"))] = str(
                    record.get("state"))
            elif event == "halt":
                halted = True
                halt_reason = record.get("reason")
        return {
            "last_completed_cycle": last_completed,
            "halted": halted,
            "halt_reason": halt_reason,
            "forecast_only_streak": streak,
            "live_children": children,
        }

    # -- halting ---------------------------------------------------------

    def halt(self, *, cycle: int, reason: str, **observed) -> NoReturn:
        """Record a halt and refuse.  An unknown reason never reaches disk."""
        if reason not in HALT_REASONS:
            raise CycleRefusal(
                "halt reason is not one this contract declares",
                reason=reason, known_reasons=list(HALT_REASONS),
                cycle=int(cycle))
        record = {"event": "halt", "cycle": int(cycle), "reason": reason}
        record.update({key: _jsonable(value)
                       for key, value in observed.items()})
        self.append(record)
        raise CycleRefusal(f"cycle halted: {reason}",
                           cycle=int(cycle), reason=reason, **observed)


def transition_receipt(*, clock, cycle_index: int, arming: Mapping,
                       ingestion: Mapping | None,
                       placements: Sequence[Mapping],
                       children: Sequence[Mapping],
                       refusals: Sequence[Mapping],
                       ingestion_evidence: Mapping | None = None) -> dict:
    """The per-cycle receipt: arming triple, ingestion, every decision.

    ``ingestion_evidence`` always says where the three-hash gate's
    evidence for this cycle lives, including when ``ingestion`` is
    ``None``.  A bare ``null`` cannot distinguish "did not assimilate"
    from "evidence is in the anchor", and a reader who cannot tell those
    apart has no way to audit the most important gate in the system.
    """
    return {
        "schema": TRANSITION_SCHEMA,
        "cycle_index": int(cycle_index),
        "clock": clock.to_json(),
        "valid_time": clock.valid_time(cycle_index).isoformat(),
        "boundary_ticks": clock.boundary_ticks(cycle_index),
        "arming": dict(arming),
        "ingestion": dict(ingestion) if ingestion is not None else None,
        "ingestion_evidence": (dict(ingestion_evidence)
                               if ingestion_evidence is not None else None),
        "placements": [dict(item) for item in placements],
        "children": [dict(item) for item in children],
        "refusals": [dict(item) for item in refusals],
    }


def write_receipt_atomically(path: str | Path, payload: Mapping) -> Path:
    """Write a receipt through a staged file and an atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".staged")
    with open(staged, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, target)
    return target


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)
