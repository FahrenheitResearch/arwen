"""Settled suite journal. All writers share one OS mutex.

Use the cycle spine's fsync and atomic receipt publisher; suit cycles are
UTC strings and have row identities, so the cycle-specific fold is not reused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from gpuwm.cycle.ledger import write_receipt_atomically
from gpuwm.mcp.gpulock import _os_lock


class SuiteLedger:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "suite-ledger.jsonl"
        self.staged = self.root / "suite-ledger.jsonl.staged"
        self.mutex = self.root / "ledger.mutex"
        self.snapshot = self.root / "ledger-snapshot.json"

    def _snapshot(self):
        return json.loads(self.snapshot.read_text()) if self.snapshot.exists() else {"seq": -1, "jobs": [], "halted": {}, "configuration": None, "watermarks": {}}

    def _read(self):
        if not self.path.exists():
            return [], b""
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        records = []
        first = None
        base = self._snapshot()["seq"]
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
            except ValueError:
                if i == len(lines) - 1 and not line.endswith(b"\n"):
                    return [r for r in records if r["seq"] > base], line
                raise ValueError(f"ledger_corrupt: {self.path}:{i + 1}; cannot infer launch state")
            if first is None:
                first = record.get("seq")
                if not isinstance(first, int) or first > base + 1:
                    raise ValueError("ledger_snapshot_gap: journal starts after its durable snapshot")
            if record.get("seq") != first + len(records):
                raise ValueError(f"ledger_sequence: {self.path}:{i + 1}")
            records.append(record)
        return [r for r in records if r["seq"] > base], b""

    def _commit(self, line):
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _settle(self):
        records, tail = self._read()
        staged = json.loads(self.staged.read_text()) if self.staged.exists() else None
        if tail:
            expected = (json.dumps(staged, sort_keys=True) + "\n").encode() if staged else b""
            if not expected.startswith(tail):
                raise ValueError("ledger_torn_append: no staged evidence for unfinished transition")
            with self.path.open("r+b") as handle:
                handle.truncate(self.path.stat().st_size - len(tail))
                handle.flush()
                os.fsync(handle.fileno())
        if staged:
            seq = staged["seq"]
            next_seq = records[-1]["seq"] + 1 if records else self._snapshot()["seq"] + 1
            if seq == next_seq:
                self._commit((json.dumps(staged, sort_keys=True) + "\n").encode())
                records.append(staged)
            elif not any(r == staged for r in records):
                raise ValueError("ledger_stage_conflict: refusing to overwrite a different transition")
            self.staged.unlink()
        return records

    def settle(self):
        with _os_lock(self.mutex):
            return self._settle()

    def append(self, event, *, row=None, cycle=None, suite_hash=None, **fields):
        if cycle is not None and not suite_hash:
            raise ValueError("cycle transition must carry suite-file hash")
        with _os_lock(self.mutex):
            records = self._settle()
            next_seq = records[-1]["seq"] + 1 if records else self._snapshot()["seq"] + 1
            record = dict(fields, schema="gpuwm.suite.ledger.v1", seq=next_seq,
                          written_unix=time.time(), event=event, row=row,
                          cycle=cycle, suite_hash=suite_hash)
            write_receipt_atomically(self.staged, record)
            self._commit((json.dumps(record, sort_keys=True) + "\n").encode())
            self.staged.unlink()
            return record

    def _fold(self, records):
        snapshot = self._snapshot()
        jobs = {(x["row"], x["cycle"]): x for x in snapshot["jobs"]}
        halted, active = dict(snapshot["halted"]), snapshot["configuration"]
        for entry in records:
            if entry["event"] == "configuration":
                active = entry
            elif entry["event"] == "halt":
                halted[entry["row"]] = entry
            elif entry["event"] == "resume":
                halted.pop(entry["row"], None)
            if entry.get("cycle") is not None and entry.get("row") is not None:
                key = (entry["row"], entry["cycle"])
                jobs[key] = dict(jobs.get(key, {}), **entry)
        return {"jobs": jobs, "halted": halted, "configuration": active, "watermarks": snapshot.get("watermarks", {})}

    def state(self):
        with _os_lock(self.mutex):
            return self._fold(self._settle())

    def compact(self, *, max_bytes=125000000, keep_events=2048):
        """Publish recovery state before replacing old events. Global cursors survive.

        A crash after snapshot publication sees the old journal and filters the
        already folded prefix. A crash after replacement sees the same suffix.
        Old terminal cycles become watermarks; unfinished cycles stay explicit.
        """
        with _os_lock(self.mutex):
            records = self._settle()
            if not self.path.exists() or len(records) < 2 or (self.path.stat().st_size <= max_bytes and len(records) <= keep_events * 2):
                return
            split = max(1, len(records) - keep_events)
            while split < len(records) - 1 and sum(len(json.dumps(r)) + 1 for r in records[split:]) > max_bytes:
                split += 1
            state = self._fold(records[:split])
            latest = {}
            watermarks = dict(state["watermarks"])
            for (row, cycle), entry in state["jobs"].items():
                if entry.get("phase") in {"completed", "skipped", "halted"}:
                    watermarks[row] = max(watermarks.get(row, ""), cycle)
                    if cycle > latest.get(row, {}).get("cycle", ""):
                        latest[row] = entry
            jobs = [entry for entry in state["jobs"].values() if entry.get("phase") not in {"completed", "skipped", "halted"}]
            jobs.extend(latest.values())
            write_receipt_atomically(self.snapshot, dict(seq=records[split-1]["seq"], jobs=jobs,
                halted=state["halted"], configuration=state["configuration"], watermarks=watermarks))
            temporary = self.root / "suite-ledger.compacting"
            with temporary.open("wb") as handle:
                for record in records[split:]:
                    handle.write((json.dumps(record, sort_keys=True) + "\n").encode())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)


def halt_row(root, row, reason, suite_hash=None):
    """Stop only identity-recorded suite workers, then persist terminal phases."""
    import signal
    from gpuwm import proc_identity
    ledger = SuiteLedger(root)
    ledger.append("halt", row=row, reason=reason, suite_hash=suite_hash)
    for (row_id, cycle), state in ledger.state()["jobs"].items():
        if row not in {"*", row_id} or state.get("phase") in {"completed", "skipped", "halted"}:
            continue
        stopped = False
        if state.get("work"):
            work = Path(state["work"])
            if not work.resolve().is_relative_to(ledger.root):
                raise ValueError("suite_worker_outside_owner: refusing to signal an external worker")
            path = work / "worker.json"
            info = json.loads(path.read_text()) if path.exists() else {}
            proc_identity.signal_process(info.get("child"), signal.SIGTERM, tree=True)
            stopped = proc_identity.signal_process(info.get("worker", state.get("worker")), signal.SIGTERM, tree=True)
        ledger.append("phase", row=row_id, cycle=cycle, suite_hash=state["suite_hash"],
                      phase="halted", reason=reason, stopped=stopped)
