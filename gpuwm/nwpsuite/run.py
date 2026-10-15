"""Unattended orchestration of existing doors, with durable launch tokens."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from gpuwm import proc_identity
from gpuwm.cycle.ledger import write_receipt_atomically
from gpuwm.mcp.gpulock import _os_lock
from .ledger import SuiteLedger, halt_row
from .recovery import disk_victim, next_action, replan, lost_box
from .retention import duration, bytes_limit, prune, register

TERMINAL = {"completed", "skipped", "halted"}


@contextmanager
def scheduler_lock(root):
    """Refuse competing controllers immediately; the OS releases a killed owner."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler.mutex").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            raise ValueError("suite_scheduler_active: another controller owns this state directory; prevents competing cycle launch decisions") from None
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def utc(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def stamp(value):
    return utc(value).isoformat().replace("+00:00", "Z")


def load_plan(path, bindings=None):
    """Resolve the spec-owned suite, or read an admitted plan document."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    if path.suffix == ".json":
        plan = json.loads(raw)
    else:
        # The spec lane owns validation and resolution. Never parse a suite here.
        from .planner import plan_suite
        from .validator import load_suite
        plan = plan_suite(load_suite(path), base=path.parent)
    if not plan.get("admitted"):
        raise ValueError("suite_not_admitted: " + json.dumps(plan.get("refusals", [])))
    if plan.get("schema") == "gpuwm.suite.plan.v1":
        deployment = json.loads(Path(bindings).read_text()) if bindings else None
        if not plan.get("execution_ready") and deployment is None:
            raise ValueError("suite_execution_unbound: planner reservations lack prepared, DA or CPU-stage door bindings")
        from .bindings import bind_plan
        plan = bind_plan(plan, deployment)
    elif plan.get("schema") != "gpuwm.suite.runtime-fixture.v1":
        raise ValueError("suite_plan_schema: expected admitted gpuwm.suite.plan.v1")
    if not isinstance(plan, dict) or not isinstance(plan.get("rows"), list):
        raise ValueError("suite_plan_not_executable: planner must provide rows with door argv")
    ids = set()
    for row in plan["rows"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["id"]) or row["id"] in ids:
            raise ValueError(f"suite_row_identity: duplicate or unsafe row {row['id']!r}")
        ids.add(row["id"])
        if not row.get("argv"):
            raise ValueError(f"suite_door_missing: {row['id']} has no launch argv")
        if int(row.get("cards", 0)) < 0:
            raise ValueError("suite_card_count: must be nonnegative")
    for row in plan["rows"]:
        unknown = set(row.get("dependencies", [])) - ids
        if unknown:
            raise ValueError(f"suite_dependency_unknown: {row['id']}: {sorted(unknown)}")
    # Dependency cycles otherwise leave an unattended loop waiting forever.
    pending = {row["id"]: set(row.get("dependencies", [])) for row in plan["rows"]}
    while pending:
        ready = {key for key, deps in pending.items() if not deps & pending.keys()}
        if not ready:
            raise ValueError(f"suite_dependency_cycle: {sorted(pending)}")
        pending = {key: deps for key, deps in pending.items() if key not in ready}
    plan["suite_hash"] = hashlib.sha256(raw).hexdigest()
    if bindings:
        plan["bindings_hash"] = hashlib.sha256(Path(bindings).read_bytes()).hexdigest()
        plan["plan_hash"] = hashlib.sha256((plan["suite_hash"] + plan["bindings_hash"]).encode()).hexdigest()
    plan["suite_file"] = str(path)
    return plan


def expand(values, context):
    # Only explicit placeholders; braces in inline JSON are ordinary data.
    def binding(match):
        if match[1] not in context:
            raise ValueError(f"suite_binding_missing: {match[1]}")
        return str(context[match[1]])
    return [re.sub(r"\{([a-z_][a-z0-9_.]*)\}", binding, str(value)) for value in values]


def bind_reservations(plan):
    from .bindings import bind_plan
    return bind_plan(plan)


class SuiteRunner:
    def __init__(self, plan, root, *, clock=time.time, phase_hook=None, rehearsal=False):
        self.root = Path(root).resolve()
        self.ledger = SuiteLedger(self.root)
        self.plan = plan
        self.clock = clock
        self.phase_hook = phase_hook or (lambda phase: None)
        self.rehearsal = rehearsal
        self.children = []
        self._install(plan)

    def _install(self, plan):
        with _os_lock(self.root / "apply.mutex"):
            self._install_locked(plan)

    def _install_locked(self, plan):
        digest = plan.get("plan_hash", plan["suite_hash"])
        path = self.root / "plans" / (digest + ".json")
        if not path.exists():
            write_receipt_atomically(path, plan)
        config = self.ledger.state()["configuration"]
        if config is None:
            self.ledger.append("configuration", suite_hash=plan["suite_hash"], plan_hash=digest, plan=str(path))
        elif config.get("plan_hash", config["suite_hash"]) != digest:
            # Resume uses the pinned old plan. apply is the only configuration change.
            self.plan = json.loads(Path(config["plan"]).read_text())

    def record(self, row, cycle, phase, **fields):
        entry = self.ledger.append("phase", row=row["id"], cycle=cycle,
            suite_hash=self.plan["suite_hash"], phase=phase, box=row.get("box", "local"),
            bindings_hash=self.plan.get("bindings_hash"),
            priority=row.get("priority", 2), cards=row.get("cards", 0), card_ids=row.get("card_ids"), **fields)
        self.phase_hook(phase)
        return entry

    def context(self, row, cycle, attempt, source_index=0):
        folder = self.root / "runs" / row["id"] / cycle.replace(":", "").replace("+", "") / str(attempt)
        sources = row.get("sources", [])
        context = dict(root=str(self.root), row=row["id"], cycle=cycle,
                    cycle_compact=utc(cycle).strftime("%Y%m%d%H"), attempt=attempt,
                    out=str(folder / "output"), work=str(folder), cards=int(row.get("cards", 0)),
                    source=sources[source_index] if sources else "")
        from .bindings import resolve_context
        return resolve_context(row, context)

    def _apply_boundary(self, state):
        with _os_lock(self.root / "apply.mutex"):
            self._apply_boundary_locked(state)

    def _apply_boundary_locked(self, state):
        pending = self.root / "pending-plan.json"
        if pending.exists() and not any(x.get("phase") not in TERMINAL for x in state["jobs"].values()):
            self.plan = json.loads(pending.read_text())
            digest = self.plan.get("plan_hash", self.plan["suite_hash"])
            target = self.root / "plans" / (digest + ".json")
            write_receipt_atomically(target, self.plan)
            self.ledger.append("configuration", suite_hash=self.plan["suite_hash"], plan_hash=digest, plan=str(target), reason="apply_boundary")
            for row in self.plan["rows"]:
                if state["halted"].get(row["id"], {}).get("reason") in {"box_lost", "card_lost_priority"}:
                    self.ledger.append("resume", row=row["id"], reason="profile_boundary")
            pending.unlink()

    def _consume(self, row, cycle, state, result):
        cutoff = state["deadline_unix"] - row.get("runtime_seconds", 0)
        action = next_action(row, result, state, now=self.clock(), cutoff=cutoff)
        self.ledger.append("recovery", row=row["id"], cycle=cycle, suite_hash=self.plan["suite_hash"],
                           result=result, decision=action)
        name = action.pop("action")
        if name == "complete":
            phase = "completed"
        elif name == "skip":
            phase = "skipped"
        elif name == "halt":
            phase = "halted"
            self.ledger.append("halt", row=row["id"], cycle=cycle,
                               suite_hash=self.plan["suite_hash"], **action)
        elif name == "wait":
            phase = "waiting"
        else:
            phase = "planned"
            action["attempt"] = state.get("attempt", 0) + 1
            action["launch_mode"] = name
        self.record(row, cycle, phase, result=result, **action)

    def _launch(self, row, cycle, state):
        context = self.context(row, cycle, state.get("attempt", 0), state.get("source_index", 0))
        folder = Path(context["work"])
        path = folder / "intent.json"
        if path.exists():
            # Recover the already committed token without rereading prepared data
            # or checking that an active door's output is still empty.
            command = json.loads(path.read_text())["argv"]
            self._start_worker(row, cycle, folder, command)
            return
        from .bindings import materialize
        context = materialize(row, context, self.plan)
        folder.mkdir(parents=True, exist_ok=True)
        argv = row["argv"]
        if state.get("launch_mode") == "checkpoint":
            argv = row["resume_argv"]
        elif state.get("launch_mode") == "degrade":
            argv = row["degrade"][state["degrade_index"]]["argv"]
        command = expand(argv, context)
        box = next((b for b in self.plan.get("boxes", []) if b["id"] == row.get("box", "local")), {})
        if row.get("cards", 0):
            prefix = box.get("launcher_prefix", [])
            if not prefix:
                raise ValueError("suite_launcher_missing: GPU rows require the box's admission launcher")
            command = expand(prefix, context) + [sys.executable, "-m", "gpuwm.nwpsuite.admitted", str(folder / "admitted.txt"), *command]
        environment = {k: expand([v], context)[0] for k, v in row.get("env", {}).items()}
        intent = dict(argv=command, cwd=row.get("cwd"), env=environment,
            receipts=expand(row.get("receipts", []), context),
            events=expand([row["events"]], context)[0] if row.get("events") else None)
        intent["door_kind"] = row.get("door_kind", "command")
        intent["admission_marker"] = str(folder / "admitted.txt") if row.get("cards", 0) else None
        intent["receipt_globs"] = row.get("receipt_globs", [])
        intent["score_files"] = expand(row.get("score_files", []), context)
        intent["receipt_budget_bytes"] = max(1, bytes_limit(self.plan.get("retention", {}).get("receipt_cap", "500MB")) // max(1, len(self.plan["rows"])))
        intent["receipt_scope"] = str(self.root)
        intent["retention"] = self.plan.get("retention", {})
        write_receipt_atomically(path, intent)
        self._start_worker(row, cycle, folder, command)

    def _start_worker(self, row, cycle, folder, command):
        self.record(row, cycle, "launch_intent", work=str(folder), argv=command)
        with (folder / "worker.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, "-m", "gpuwm.nwpsuite.worker", str(folder)],
                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        self.children.append(child)
        self.record(row, cycle, "running", worker=proc_identity.identify(child.pid), work=str(folder))

    def tick(self, cycles):
        for child in self.children[:]:
            if child.poll() is not None:
                self.children.remove(child)
        state = self.ledger.state()
        self._apply_boundary(state)
        self._health()
        now = self.clock()
        policy = self.plan.get("retention", {})
        floor = float(policy.get("disk_floor_gb", 0)) * 1e9
        if shutil.disk_usage(self.root).free < floor:
            prune(self.root, policy, ledger=self.ledger)
            if shutil.disk_usage(self.root).free < floor:
                victim = disk_victim([r for r in self.plan["rows"] if r["id"] not in state["halted"]])
                halt_row(self.root, victim or "*", "disk_floor", self.plan["suite_hash"])
                return
        for cycle in cycles:
            from .bindings import is_due, row_at
            for definition in sorted(self.plan["rows"], key=lambda r: r.get("priority", 2)):
                if not is_due(definition, cycle):
                    continue
                row = row_at(definition, cycle)
                state = self.ledger.state()
                if row["id"] in state["halted"] or "*" in state["halted"]:
                    continue
                current = state["jobs"].get((row["id"], cycle))
                if not current and cycle <= state.get("watermarks", {}).get(row["id"], ""):
                    continue
                if current and current.get("phase") in TERMINAL:
                    continue
                if not current:
                    deadline = (now if self.rehearsal else utc(cycle).timestamp() + row.get("ready_offset_s", 0)) + duration(row.get("deadline", "6h"))
                    current = self.record(row, cycle, "planned", attempt=0, deadline_unix=deadline,
                        not_before_unix=now if self.rehearsal else utc(cycle).timestamp() + row.get("start_offset_s", 0))
                phase = current.get("phase")
                if phase in {"launch_intent", "running"}:
                    folder = Path(current["work"])
                    result_path = folder / "result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        self.record(row, cycle, "receipted", result=result)
                        self._consume(row, cycle, current, result)
                        # Publish the recoverable result before making launch
                        # anchors eligible for concurrent worker pruning.
                        with _os_lock(self.root / "artifacts.mutex"):
                            for item in result.get("receipts", []):
                                if Path(item["path"]).is_relative_to(self.root):
                                    register(self.root, item["path"], "receipts", row=row["id"], cycle=cycle)
                            self._artifacts(row, cycle, folder)
                    else:
                        info = json.loads((folder / "worker.json").read_text()) if (folder / "worker.json").exists() else {}
                        alive = proc_identity.alive(info.get("worker")) or proc_identity.alive(current.get("worker"))
                        if not alive:
                            self._launch_checked(row, cycle, current)
                    continue
                if phase == "receipted":
                    self._consume(row, cycle, current, current["result"])
                    continue
                dependencies = [state["jobs"].get((dep, cycle), {}) for dep in row.get("dependencies", [])]
                for dependency in row.get("previous_dependencies", []):
                    previous = stamp(utc(cycle).timestamp() - duration(dependency["every"]))
                    background = state["jobs"].get((dependency["row"], previous), {})
                    if background.get("phase") != "completed":
                        # Shadow DA lacks a qualified background. Public analysis
                        # remains the forecast authority; never invent a new run.
                        self.record(row, cycle, "skipped", reason="da_failed", background_cycle=previous,
                                    remedy="previous-cycle background unavailable; public analysis retained")
                        break
                else:
                    background = None
                if background is not None:
                    continue
                if any(dep.get("phase") == "halted" or dep_id in state["halted"] for dep_id, dep in zip(row.get("dependencies", []), dependencies)):
                    self.record(row, cycle, "skipped", reason="dependency_halted")
                    continue
                if any(dep.get("phase") == "skipped" and dep.get("reason") != "da_failed" for dep in dependencies):
                    self.record(row, cycle, "skipped", reason="dependency_skipped")
                    continue
                if any(dep.get("phase") not in {"completed", "skipped"} for dep in dependencies):
                    continue
                if max(current.get("retry_at", 0), current.get("not_before_unix", 0)) > now:
                    continue
                if row.get("readiness_argv") and not self.rehearsal:
                    try:
                        context = self.context(row, cycle, current.get("attempt", 0), current.get("source_index", 0))
                        from .bindings import materialize
                        context = materialize(row, context, self.plan, check_source=False)
                        ready = subprocess.run(expand(row["readiness_argv"], context), capture_output=True, text=True,
                                               timeout=row.get("readiness_timeout", 120), cwd=row.get("cwd"),
                                               env=dict(os.environ, **{k: expand([v], context)[0] for k, v in row.get("env", {}).items()}))
                    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                        self._consume(row, cycle, current, dict(exit_code=2 if isinstance(error, ValueError) else 1, error_class=type(error).__name__,
                                                               message=str(error), readiness=True))
                        continue
                    if ready.returncode:
                        try:
                            document = json.loads(ready.stdout)
                        except ValueError:
                            document = {}
                        self._consume(row, cycle, current, dict(document, exit_code=ready.returncode, readiness=True))
                        continue
                # Launch only when planned box capacity is free. CPU rows use zero slots.
                active = [x for x in state["jobs"].values() if x.get("box") == row.get("box", "local")
                          and x.get("phase") in {"running", "launch_intent"}]
                occupied = sum(x.get("cards", 0) for x in active)
                box = next((b for b in self.plan.get("boxes", []) if b["id"] == row.get("box", "local")), {})
                if occupied + row.get("cards", 0) > box.get("cards", 0):
                    continue
                if row.get("card_ids") and any(set(row["card_ids"]) & set(x.get("card_ids") or []) for x in active):
                    continue
                self.record(row, cycle, "ready")
                self._launch_checked(row, cycle, current)
        write_receipt_atomically(self.root / "heartbeat.json", {"updated_unix": self.clock(), "scheduler": proc_identity.identify(os.getpid())})

    def _launch_checked(self, row, cycle, state):
        from .bindings import SourcePending
        try:
            self._launch(row, cycle, state)
        except SourcePending as error:
            self._consume(row, cycle, state, dict(exit_code=75, error_class="source_behind", message=str(error)))
        except (OSError, ValueError) as error:
            self._consume(row, cycle, state, dict(exit_code=2, error_class="binding_refused", message=str(error)))

    def _artifacts(self, row, cycle, folder):
        """Inventory only this launch's files, never external prepared inputs."""
        for base, dirs, files in os.walk(folder, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
            for filename in files:
                path = Path(base) / filename
                if path.is_symlink():
                    continue
                kind = None
                for category, patterns in row.get("artifact_globs", {}).items():
                    if any(path.relative_to(folder).match(pattern) for pattern in patterns):
                        kind = category
                        break
                if kind is None and path.suffix in {".json", ".jsonl", ".log", ".toml", ".csv"}:
                    kind = "receipts"
                if kind:
                    register(self.root, path, kind, row=row["id"], cycle=cycle)

    def _health(self):
        """Box-config probes report facts; recovery remains policy data."""
        path = self.root / "box-health.json"
        prior = json.loads(path.read_text()) if path.exists() else {}
        health = {}
        for box in self.plan.get("boxes", []):
            command = box.get("health_argv")
            if not command:
                continue
            try:
                done = subprocess.run(command, capture_output=True, text=True, timeout=10)
                info = json.loads(done.stdout) if done.returncode == 0 else {"alive": False}
            except (OSError, ValueError, subprocess.TimeoutExpired):
                info = {"alive": False}
            health[box["id"]] = dict(info, checked_unix=self.clock())
            old = prior.get(box["id"], {})
            if info.get("alive", True) is False and old.get("alive", True):
                survivors = [b["id"] for b in self.plan["boxes"] if b["id"] != box["id"]]
                decision = lost_box(self.plan, survivors[0] if survivors else None)
                self.ledger.append("health_recovery", box=box["id"], decision=decision)
                if decision.get("action") == "profile":
                    profile = decision["plan"]
                    profile.setdefault("suite_hash", hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest())
                    apply_plan(self.root, profile)
                    for row in self.plan["rows"]:
                        if row.get("box") == box["id"]:
                            halt_row(self.root, row["id"], "box_lost", self.plan["suite_hash"])
                else:
                    halt_row(self.root, "*", decision["halt"], self.plan["suite_hash"])
            cards = info.get("cards")
            box.setdefault("configured_cards", box["cards"])
            if isinstance(cards, int) and cards < box["cards"] and cards != old.get("cards"):
                decision = replan([r for r in self.plan["rows"] if r.get("box") == box["id"]], cards)
                self.ledger.append("health_recovery", box=box["id"], reason="card_lost", decision=decision)
                if decision.get("halt"):
                    halt_row(self.root, "*", decision["halt"], self.plan["suite_hash"])
                else:
                    for row_id in decision["drop"]:
                        halt_row(self.root, row_id, "card_lost_priority", self.plan["suite_hash"])
                    box["cards"] = cards
            if isinstance(cards, int):
                box["cards"] = max(0, min(cards, box["configured_cards"]))
        if health:
            write_receipt_atomically(path, health)

    def run(self, *, cycles=None, once=False, poll=1):
        # Only the scheduler owns this mutex. Readers and halt/apply stay responsive.
        with scheduler_lock(self.root):
            while True:
                selected = cycles
                if selected is None:
                    state = self.ledger.state()
                    due = {c for (_, c), entry in state["jobs"].items() if entry.get("phase") not in TERMINAL}
                    for row in self.plan["rows"]:
                        from .bindings import cadence
                        period = int(cadence(row))
                        previous = [c for (rid, c) in state["jobs"] if rid == row["id"]]
                        previous.append(state.get("watermarks", {}).get(row["id"], ""))
                        latest = max(previous)
                        start = int(utc(latest).timestamp()) + period if latest else int(self.clock() // period) * period
                        due.update(stamp(t) for t in range(start, int(self.clock()) + 1, period))
                    selected = sorted(due)
                self.tick(selected)
                from .status import publish_status
                self.ledger.compact(max_bytes=max(1024, bytes_limit(self.plan.get("retention", {}).get("receipt_cap", "500MB")) // 4))
                retained = prune(self.root, self.plan.get("retention", {}), ledger=self.ledger)
                if retained["cap_unmet"]:
                    state = self.ledger.state()
                    active = any(job.get("phase") in {"running", "launch_intent"} for job in state["jobs"].values())
                    # Active workers enforce the same combined cap and publish
                    # their typed result. Control-only pressure must also stop
                    # new work, rather than growing forever without a worker.
                    if not active and "*" not in state["halted"]:
                        eligible = [r for r in self.plan["rows"] if r["id"] not in state["halted"]]
                        if eligible:
                            halt_row(self.root, disk_victim(eligible) or "*", "receipt_cap_exceeded", self.plan["suite_hash"])
                publish_status(self.plan, self.root, now=self.clock())
                if once:
                    return
                if cycles is not None:
                    jobs = self.ledger.state()["jobs"]
                    from .bindings import is_due
                    expected = [(r["id"], c) for c in cycles for r in self.plan["rows"] if is_due(r, c)]
                    halted = self.ledger.state()["halted"]
                    marks = self.ledger.state().get("watermarks", {})
                    if all(jobs.get(key, {}).get("phase") in TERMINAL or key[0] in halted or key[1] <= marks.get(key[0], "") for key in expected):
                        return
                time.sleep(poll)


def apply_plan(root, plan):
    with _os_lock(Path(root) / "apply.mutex"):
        return _apply_plan_locked(root, plan)


def _apply_plan_locked(root, plan):
    ledger = SuiteLedger(root)
    old = ledger.state()["configuration"]
    previous = json.loads(Path(old["plan"]).read_text()) if old else {"rows": []}
    before = {r["id"]: r for r in previous["rows"]}
    after = {r["id"]: r for r in plan["rows"]}
    diff = {"added": sorted(after.keys() - before.keys()), "removed": sorted(before.keys() - after.keys()),
            "changed": sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
            "settings_changed": sorted(k for k in set(previous) | set(plan)
                                       if k not in {"rows", "suite_hash", "suite_file"} and previous.get(k) != plan.get(k)),
            "before_hash": previous.get("suite_hash"), "after_hash": plan["suite_hash"]}
    write_receipt_atomically(Path(root) / "pending-plan.json", plan)
    ledger.append("apply_requested", diff=diff, suite_hash=plan["suite_hash"])
    return diff
