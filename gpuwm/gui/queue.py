"""New forecast's queue: forecasts that wait for the card and start themselves, in order.

A forecast queued from New forecast is written as its own run folder at
once (the plan, the box, the physics choice) with a ``gui-queued.json``
marker, and its place in line is kept in ``<root>/.arwen-gui/queue.json``.
Both are files in the forecasts folder, so the queue survives closing the
page and restarting the page server; the folder shows in My forecasts as
"queued".

One scheduler thread in the page server looks at the queue every few
seconds.  Per machine, the first queued forecast starts when that
machine's card is free: on this computer when no forecast holds the job
manager's card lock, no live line is in the card-sharing OWNER file (when
this server is told of one) and no other program's compute process leaves
too little free memory for the forecast's own fit; on a Machines node
when its probe says it is idle.  Only one forecast starts per machine at a
time, so one card runs one queued forecast.

Starting takes the card through the OWNER file the same way a Machines
node does: one test-and-append under the file's locks, a line naming this
server's tag, handed to the run's own job wrapper once it has started.
The wrapper removes the line when the run ends, whether or not this page
server is still running; the scheduler removes a line whose process ended
without doing so (a wrapper killed outright, or a page server that stopped
between claiming the card and handing the line on), and a line whose
number now names another process than the one recorded for it, but never
a line of another tag.  A forecast that no longer fits (the disk
has too little room, or the card too little memory for its fit) is held
with the reason instead of started into a failure; the forecasts behind
it keep their turn, and it starts by itself once the reason is gone.

A forecast is also held while its start is one Start would not take yet
(:mod:`.availability`): Queue it takes a start from the last day that no
check has confirmed, and the forecast waits, saying so, until one does.
The scheduler asks the page's own checks, which ask again every minute or
two, and the forecast then starts in its turn.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Callable

from . import runs
from .files import SERVER_DIR, read_json, write_json
from .jobs import runtime_install
from .machines import LOCAL

QUEUE_FILE = "queue.json"
#: A queued forecast's marker while the scheduler starts it: not in the line to a page, still in its place in the
#: order file, so a start that does not happen puts it back where it was.
STARTING = runs.QUEUED + ".starting"
QUEUE_SCHEMA = "gpuwm.gui-queue.v1"
MARKER_SCHEMA = "gpuwm.gui-queued.v1"
#: The scheduler looks at the queue this often.
TICK_S = 5.0
#: What a forecast behind the first one on its machine waits for.
AHEAD = "Waiting for the forecasts ahead of it."
#: The scheduler looks through the whole forecasts folder for a queued folder the order file does not list (a crash
#: between its two writes, or a folder copied in) when it starts and then this often.  A page never does: with
#: 1,500 run folders the walk took one to two seconds, and the Review step asks for the queue every 3 s.
SCAN_EVERY_S = 60.0
#: A Machines node is asked how its card is doing at most this often (one SSH call).
REMOTE_EVERY_S = 30.0
#: A queued forecast needs at least this much free disk to start.  New forecast's fit gives no estimate of what a
#: run writes, so for most forecasts this floor is the whole disk check; a storm-following Customise also carries the
#: engine's own projection of what its run writes (the marker's ``disk_gib``), and waits for that much.
DISK_FLOOR_GIB = 5.0
#: The OWNER file this server takes the card through, and the tag its lines carry.
OWNER_FILE_ENV = "GPUWM_GPU_OWNER_FILE"
OWNER_TAG_ENV = "GPUWM_GPU_OWNER_TAG"
DEFAULT_OWNER_TAG = "gpuwm-gui"
#: The bound written on an OWNER line: how long the line's holder expects to keep the card, in minutes.
OWNER_BOUND_MIN = 720
#: A remote queued forecast asks its machine to wait this long for the card, so a race at the start is a wait.
REMOTE_WAIT_MIN = 30


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def started_in(rundir: Path) -> bool:
    """Something started in a queued forecast's folder: the page's job record, a machine's mirror, or the
    engine's own manifest or event log.  A start that wrote one of these is a start, whatever failed after it."""

    return any((rundir / name).exists() for name in (runs.JOB, runs.REMOTE, runs.MANIFEST, runs.EVENTS))


def _gib(mib: Any) -> float | None:
    try:
        return round(float(mib) / 1024.0, 1)
    except (TypeError, ValueError):
        return None


class _QueueOrderChanged(Exception):
    """The line changed (a Move, a Remove, a new forecast) while the scheduler read a card or checked data."""


class ForecastQueue:
    """The queue's files, its card checks and its scheduler thread."""

    def __init__(self, api: Any, *, owner_file: str | None = None, owner_tag: str | None = None,
                 cards: Callable[[], dict[str, Any]] | None = None,
                 disk: Callable[[], float | None] | None = None,
                 tick_s: float = TICK_S, pid: int | None = None) -> None:
        self.api = api
        self.root = Path(api.root)
        self.owner_file = owner_file if owner_file is not None else (os.environ.get(OWNER_FILE_ENV) or None)
        self.owner_tag = owner_tag or os.environ.get(OWNER_TAG_ENV) or DEFAULT_OWNER_TAG
        self._cards = cards
        self._disk = disk
        self.tick_s = tick_s
        self.pid = pid or os.getpid()
        # Held only while the queue's files are read or written: never across a card reading, a start or SSH,
        # so a page asking for the queue never waits for the scheduler.
        self._lock = threading.RLock()
        # One tick at a time.
        self._ticking = threading.Lock()
        self._stop = threading.Event()
        # Set by a change a page makes to the line (a reorder), so the scheduler looks again at once.
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._remote: dict[str, tuple[float, dict[str, Any]]] = {}
        self._remote_asking: set[str] = set()
        self._local: tuple[float, dict[str, Any]] | None = None
        self._local_asking = False
        self._recovered = False
        self._scanned: float | None = None
        self.last_tick_utc: str | None = None

    # ------------------------------------------------------------ files

    def _file(self) -> Path:
        return self.root / SERVER_DIR / QUEUE_FILE

    def _document(self) -> dict[str, Any]:
        document = read_json(self._file(), default=None)
        if not isinstance(document, dict) or document.get("schema") != QUEUE_SCHEMA:
            document = {"schema": QUEUE_SCHEMA, "order": [], "owned": []}
        document.setdefault("order", [])
        document.setdefault("owned", [])
        return document

    def _save(self, document: dict[str, Any]) -> None:
        self._file().parent.mkdir(parents=True, exist_ok=True)
        write_json(self._file(), document)

    def _marker(self, run_id: str) -> tuple[Path, dict[str, Any]] | None:
        try:
            rundir = runs.run_path(self.root, run_id)
        except Exception:  # noqa: BLE001 - a bad id in the file is dropped, never served
            return None
        marker = read_json(rundir / runs.QUEUED, default=None)
        if not isinstance(marker, dict):
            return None
        return rundir, marker

    def _starting(self, run_id: str) -> bool:
        try:
            return (runs.run_path(self.root, run_id) / STARTING).is_file()
        except Exception:  # noqa: BLE001
            return False

    def _orders(self, scan: bool = False) -> tuple[list[str], list[tuple[str, Path, dict[str, Any]]]]:
        """(the order file's line, starting forecasts kept in their places; the line a page sees, each forecast
        with its folder and marker).  ``scan`` also walks the forecasts folder for markers the file does not
        list; only the scheduler asks for that (:data:`SCAN_EVERY_S`)."""

        with self._lock:
            document = self._document()
            kept: list[str] = []
            line: dict[str, tuple[Path, dict[str, Any]]] = {}
            for run_id in dict.fromkeys(document["order"]):
                found = self._marker(run_id)
                if found is not None:
                    line[run_id] = found
                if found is not None or self._starting(run_id):
                    kept.append(run_id)
            loose = []
            if scan:
                for rundir in runs.iter_runs(self.root):
                    if (rundir / runs.QUEUED).is_file():
                        run_id = rundir.resolve().relative_to(self.root.resolve()).as_posix()
                        if run_id not in kept:
                            found = self._marker(run_id)
                            if found is not None:
                                line[run_id] = found
                                loose.append((str(found[1].get("queued_utc") or ""), run_id))
            full = kept + [run_id for _, run_id in sorted(loose)]
            if full != document["order"]:
                document["order"] = full
                self._save(document)
            return full, [(run_id, *line[run_id]) for run_id in full if run_id in line]

    def order(self, scan: bool = False) -> list[str]:
        """The queued run ids in order.  A marker the order file does not list (a crash between the two writes)
        joins the end in the time it was queued, once the scheduler's walk of the folder finds it (``scan``); a
        listed run whose marker is gone leaves the line, except one being started, which keeps its place until
        its start is known."""

        return [run_id for run_id, _, _ in self._orders(scan)[1]]

    def add(self, run_id: str, rundir: Path, marker: dict[str, Any]) -> int:
        """Queue a written run folder at the end of the line, saying at once why it waits; its place in the line.

        The place is counted under the lock the forecast joins the line with: the scheduler is woken at once and
        can start it before the caller answers, and a place read from the line after that found it gone.  The
        marker and the line are written together: a line that cannot be saved takes the marker back before the
        lock is let go, so a folder this raises for is never found by the scheduler's walk and started later.
        Nothing after that save can raise, so a raise always means the forecast is not in line.
        """

        where = str(marker.get("machine") or LOCAL)
        # The page's last reading of this computer's card, as GET /api/queue serves it; a Machines node's card is
        # an SSH call, left to the scheduler's look.
        card = self.card(LOCAL) if where == LOCAL else None
        with self._lock:
            # The line it joins the end of, read first: reading it can save the order file.
            ahead_of_it = [other for other_id, _, other in self._orders()[1] if other_id != run_id]
            document = {"schema": MARKER_SCHEMA, "queued_utc": _utc(), "machine": LOCAL, "held": None,
                        "waiting": None, **marker}
            if document["waiting"] is None and card is not None:
                # Until the scheduler's look (a fresh card reading) the listing said nothing of why it waits: behind
                # another forecast for this card it waits for the ones ahead, and first in line for what holds the
                # card now.  A hold for its data or its size is written after this, and replaces it.
                ahead = any(str(other.get("machine") or LOCAL) == where and not other.get("held")
                            for other in ahead_of_it)
                document["waiting"] = AHEAD if ahead else self.blocking(document, card)
            write_json(rundir / runs.QUEUED, document)
            queue = self._document()
            queue["order"] = [item for item in queue["order"] if item != run_id] + [run_id]
            try:
                self._save(queue)
            except BaseException:
                (rundir / runs.QUEUED).unlink(missing_ok=True)
                raise
            # The scheduler looks at once, so the new forecast says why it waits (or starts) now, not a tick later.
            self._wake.set()
            # Numbered as My forecasts numbers the line: every forecast waiting to start, on any machine.
            return len(ahead_of_it) + 1

    def move(self, run_id: str, step: int) -> list[str]:
        """Swap a forecast with its neighbour in the line (``step`` -1 up, 1 down); one being started stays put."""

        with self._lock:
            full, line = self._orders()
            order = [item for item, _, _ in line]
            if run_id not in order:
                raise KeyError(run_id)
            index = order.index(run_id)
            target = max(0, min(len(order) - 1, index + (1 if step > 0 else -1)))
            if target != index:
                a, b = full.index(order[index]), full.index(order[target])
                full[a], full[b] = full[b], full[a]
                queue = self._document()
                queue["order"] = full
                self._save(queue)
                line[index], line[target] = line[target], line[index]
                self._rewrite_waiting(line)
                self._wake.set()
            return [item for item in full if item in order]

    def _rewrite_waiting(self, line: list[tuple[str, Path, dict[str, Any]]]) -> None:
        """After a reorder, say at once why each forecast waits, as the scheduler's next look will.

        Per machine, the first forecast that is not held takes the card's reason (the one the first in line was
        given at the last look) and each one behind it waits for the forecasts ahead of it; a held forecast is
        passed over and keeps its own reason.
        """

        card: dict[str, str] = {}
        for _, _, marker in line:
            where = str(marker.get("machine") or LOCAL)
            waiting = marker.get("waiting")
            if not marker.get("held") and waiting and waiting != AHEAD:
                card.setdefault(where, waiting)
        seen: set[str] = set()
        for _, rundir, marker in line:
            where = str(marker.get("machine") or LOCAL)
            if marker.get("held") or where not in card:
                continue
            waiting = AHEAD if where in seen else card[where]
            seen.add(where)
            if marker.get("waiting") != waiting and (rundir / runs.QUEUED).is_file():
                try:
                    write_json(rundir / runs.QUEUED, {**marker, "waiting": waiting})
                except OSError:
                    pass

    def remove(self, run_id: str) -> str | None:
        """Take a forecast out of the line and delete the folder the queue wrote for it (nothing ran in it).

        Its marker goes first: the marker is what puts a folder in line, and the scheduler's walk of the forecasts
        folder puts back any folder that still has one.  Left in a folder that could not be deleted, it came back
        on the next walk or restart and started, after Remove had said it was removed.  A marker that cannot be
        deleted raises :class:`OSError` with the forecast still in line; a folder that cannot be deleted once the
        marker is gone is returned as the reason (it never starts), else None.
        """

        with self._lock:
            found = self._marker(run_id)
            if found is None:
                raise KeyError(run_id)
            rundir, _ = found
            state = runs.status(rundir)["state"]
            if state != "queued":
                raise KeyError(run_id)
            (rundir / runs.QUEUED).unlink(missing_ok=True)
            queue = self._document()
            queue["order"] = [item for item in queue["order"] if item != run_id]
            try:
                self._save(queue)
            except OSError as error:
                # Out of line all the same: a listed forecast with no marker leaves the line when it is next read.
                print(f"gpuwm gui: {run_id} was taken out of the queue, but the queue's file could not be saved: "
                      f"{error!r}", file=sys.stderr, flush=True)
            try:
                shutil.rmtree(rundir)
            except OSError as error:
                # What can go goes; what is left is said, never passed over.
                shutil.rmtree(rundir, ignore_errors=True)
                if rundir.exists():
                    print(f"gpuwm gui: {run_id} left the queue, but its folder could not be deleted: {error!r}",
                          file=sys.stderr, flush=True)
                    return error.strerror or type(error).__name__
            return None

    # ------------------------------------------------------------ the card

    def _card_info(self) -> dict[str, Any]:
        if self._cards is not None:
            return self._cards()
        from gpuwm.machine_agent import cards

        return cards()

    def _running_run(self, holder: Any) -> dict[str, Any] | None:
        """The run folder whose job holds this computer's card, with its status, or None.

        ``holder`` is :meth:`Runner.card_holder`'s answer: the folder its job
        writes, which for a forecast started here is the run's own folder.
        """

        folder = holder.get("folder") if isinstance(holder, dict) else None
        if not folder:
            return None
        rundir = Path(str(folder))
        if not rundir.is_dir():
            return None
        return {"run": runs.run_id_of(self.root, rundir) or rundir.name, "status": runs.status(rundir)}

    def local_card(self, fresh: bool = False) -> dict[str, Any]:
        """This computer's card as the queue sees it.

        ``fresh`` reads it now: the scheduler does, every tick, and so does a
        start.  A page is served the last reading at once, and one older
        than a tick is read again in the background, so a page never waits
        on nvidia-smi; only a read before any other reads it in the request.
        """

        if not fresh:
            with self._lock:
                hit = self._local
                if hit is not None:
                    if time.monotonic() - hit[0] >= self.tick_s and not self._local_asking:
                        self._local_asking = True
                        threading.Thread(target=self._refresh_local, daemon=True,
                                         name="arwen-gui-queue-card").start()
                    return hit[1]
        return self._read_local()

    def _refresh_local(self) -> None:
        try:
            self._read_local()
        except Exception:  # noqa: BLE001 - the last reading stays until the next one comes
            pass
        finally:
            with self._lock:
                self._local_asking = False

    def _read_local(self) -> dict[str, Any]:
        holder = self.api.runner.card_holder()
        info = self._card_info()
        devices = info.get("devices") or []
        device = devices[0] if devices else {}
        total = _gib(device.get("memory_total_mib"))
        used = _gib(device.get("memory_used_mib"))
        free = None if total is None or used is None else round(total - used, 1)
        owners = []
        if self.owner_file:
            from gpuwm.machine_agent import owner_lines

            owners = [row for row in owner_lines(self.owner_file) if row["live"]]
        running = self._running_run(holder) if holder else None
        processes = info.get("processes") or []
        state = {"machine": LOCAL, "name": device.get("name"), "total_gib": total, "free_gib": free,
                 "processes": len(processes), "pids": [row.get("pid") for row in processes],
                 "owners": [row["line"] for row in owners], "owner_file": self.owner_file,
                 "running": running, "held_by_lock": bool(holder), "checked_utc": _utc()}
        why = None
        if isinstance(holder, dict) and holder.get("unreadable"):
            why = "The card counts as in use: its lock file cannot be read."
        elif holder:
            name = (running or {}).get("run")
            why = f"The card is running {name}." if name else "The card is running another forecast."
        elif owners:
            tag = str(owners[0]["line"]).split()[0]
            why = f"The card is held by {tag}."
        state["why"] = why
        state["busy"] = why is not None
        with self._lock:
            self._local = (time.monotonic(), state)
        return state

    def remote_card(self, machine: str) -> dict[str, Any]:
        """A Machines node's card, from its last probe; a probe older than :data:`REMOTE_EVERY_S` is asked again
        in the background, so this never waits on SSH."""

        with self._lock:
            hit = self._remote.get(machine)
            asking = machine in self._remote_asking
            if (hit is None or time.monotonic() - hit[0] >= REMOTE_EVERY_S) and not asking:
                self._remote_asking.add(machine)
                threading.Thread(target=self._probe_remote, args=(machine,), daemon=True,
                                 name=f"arwen-gui-queue-probe-{machine}").start()
        if hit is None:
            return {"machine": machine, "busy": True, "why": f"Checking {machine}.", "checking": True}
        return hit[1]

    def _probe_remote(self, machine: str) -> None:
        try:
            probe = self.api.machines.probe(machine, fresh=True)
            state = str(probe.get("state") or "")
            busy = state in ("running", "busy") or not probe.get("reachable", True)
            why = None
            if busy:
                why = (f"{machine} is not reachable." if not probe.get("reachable", True)
                       else f"{machine}'s card is {probe.get('detail') or state}.")
            result = {"machine": machine, "busy": busy, "why": why, "total_gib": probe.get("vram_gib"),
                      "disk_free_gib": probe.get("disk_free_gib"), "version_matches": probe.get("version_matches"),
                      "checked_utc": _utc()}
        except Exception as error:  # noqa: BLE001 - a machine that cannot be asked keeps its forecasts waiting
            result = {"machine": machine, "busy": True, "why": f"{machine} could not be checked: {error}",
                      "checked_utc": _utc()}
        with self._lock:
            self._remote[machine] = (time.monotonic(), result)
            self._remote_asking.discard(machine)

    def card(self, machine: str) -> dict[str, Any]:
        return self.local_card() if machine == LOCAL else self.remote_card(machine)

    def _disk_free(self, machine: str, card: dict[str, Any]) -> float | None:
        if machine != LOCAL:
            return card.get("disk_free_gib")
        if self._disk is not None:
            return self._disk()
        from .api import disk_free_gib

        return disk_free_gib(self.root)

    def _fits_beside(self, marker: dict[str, Any], card: dict[str, Any]) -> bool:
        """Another program's compute process leaves room for this forecast's own fit."""

        need = marker.get("need_gib")
        free = card.get("free_gib")
        return need is not None and free is not None and float(free) >= float(need)

    def data_reason(self, marker: dict[str, Any]) -> str | None:
        """Why a queued forecast's start cannot be taken yet, or None: the rule Start follows, from the page's own
        checks (:meth:`.availability.Board.row`), which never wait on a server."""

        return self.data_hold(marker)[0]

    def data_hold(self, marker: dict[str, Any]) -> tuple[str | None, bool]:
        """:meth:`data_reason`, and whether the forecast only waits for its start to be confirmed published (an
        ordinary wait, which ends by itself) rather than being held on a start its source does not offer."""

        data = marker.get("data") or {}
        if not data.get("source") or not data.get("cycle"):
            return None, False
        try:
            row = self.api.availability_of(str(data["source"]), str(data["cycle"]), float(data.get("hours") or 6))
        except Exception:  # noqa: BLE001 - no answer is no reason to hold: the download still checks the bytes
            return None, False
        starts = row.get("starts", "now")
        if starts == "now":
            return None, False
        if starts == "queue":
            names = self.api.offered_sources() or {}
            name = names.get(str(data["source"])) or str(data["source"]).upper()
            when = str(data["cycle"]).replace("T", " ") + ":00 UTC"
            return (f"Held until a check confirms the {name} {when} start is published (checked every minute or "
                    "two). It then starts in its turn."), True
        return f"Held: {str(row.get('why') or '').strip()} Remove it and queue another start.", False

    def note_data(self, run_id: str) -> None:
        """Say at once why a forecast just queued waits for its data, rather than at the scheduler's next look."""

        found = self._marker(run_id)
        if found is None:
            return
        rundir, marker = found
        held, awaits = self.data_hold(marker)
        if held:
            self._note(rundir, marker, held=held, waiting=None, awaits_data=awaits)

    def hold_reason(self, marker: dict[str, Any], card: dict[str, Any], machine: str) -> str | None:
        """Why a queued forecast cannot start on this machine even once the card is free, or None."""

        missing = self.api.runner.runtime_gap() if machine == LOCAL else None
        if missing:
            # Started, it would stop at its first step for want of the runtime.
            return f"Held: {missing} {runtime_install()}, and it starts by itself."
        free = self._disk_free(machine, card)
        if free is not None and free < DISK_FLOOR_GIB:
            return (f"Held: the disk that holds the forecasts has {free:.1f} GiB free and a forecast needs at least "
                    f"{DISK_FLOOR_GIB:.0f} GiB to start. Free some space and it starts by itself.")
        writes = marker.get("disk_gib")
        if free is not None and writes and float(writes) > free:
            # Started now, it would stop partway when the disk fills, with nothing usable.
            return (f"Held: this forecast writes about {float(writes):.0f} GiB and the disk that holds the forecasts "
                    f"has {free:.1f} GiB free. Free some space and it starts by itself.")
        need = marker.get("need_gib")
        total = card.get("total_gib")
        if need is not None and total is not None and float(need) > float(total):
            return (f"Held: this forecast needs about {float(need):.1f} GiB of card memory and the card has "
                    f"{float(total):.1f} GiB. Remove it and make a smaller one.")
        return None

    def blocking(self, marker: dict[str, Any], card: dict[str, Any]) -> str | None:
        """Why a queued forecast must wait for the card right now, or None when it may start."""

        if card.get("busy"):
            return card.get("why") or "The card is busy."
        if card.get("machine") == LOCAL and card.get("processes") and not self._fits_beside(marker, card):
            free = card.get("free_gib")
            return ("Another program is using the card"
                    + (f" and {free:.1f} GiB is free." if isinstance(free, (int, float)) else "."))
        return None

    # ------------------------------------------------------------ starting

    def _claim(self, marker: dict[str, Any], card: dict[str, Any], at: str) -> tuple[bool, str]:
        """(claimed, why).  ``at`` is the time the line is written with, recorded before the claim."""

        if not self.owner_file:
            return True, "no OWNER file"
        from gpuwm.machine_agent import claim_card

        # A compute process that leaves room for this forecast's fit does not stop the claim; one that does not
        # was already a reason to wait.
        beside = tuple(pid for pid in card.get("pids") or [] if pid) if self._fits_beside(marker, card) else ()
        answer = claim_card(self.owner_file, self.owner_tag, self.pid, OWNER_BOUND_MIN, own_pids=beside,
                            info=self._card_info(), at=at)
        return bool(answer.get("claimed")), str(answer.get("why") or "")

    def _swap_owned(self, old: dict[str, Any] | None, new: dict[str, Any] | None) -> None:
        """Replace the queue's ``owned`` record ``old`` with ``new`` (either may be None), under the lock."""

        with self._lock:
            queue = self._document()
            rows = [row for row in queue["owned"] if old is None or row != old]
            if new is not None:
                rows.append(new)
            if rows != queue["owned"]:
                queue["owned"] = rows
                self._save(queue)

    def launch_holding_card(self, run_id: str, launch: Callable[[], Any], need_gib: float | None = None) -> Any:
        """Start a forecast on this computer's card through the OWNER file, when this server has one.

        One test-and-append takes the card under this server's tag; the
        launch runs; the line is handed to the run's own job wrapper, which
        removes it when the run ends (so closing ``gpuwm gui`` mid-run leaves
        no line behind), and the scheduler removes one whose process ended
        without doing so.  A card another owner holds refuses the start with
        who holds it.  Every local
        start goes through here, a queued one or Start pressed on a free
        card, so the OWNER file always names what runs on the card.  An
        install that cannot run a forecast here (:meth:`.jobs.Runner.runtime_gap`)
        is refused before anything is launched: launched, the forecast
        stopped at its first step and was listed as failed.
        """

        from .api import ApiError

        missing = self.api.runner.runtime_gap()
        if missing:
            raise ApiError(409, f"{missing} Nothing started.",
                           f"{runtime_install()}, or run it on a machine added on the Machines page.")
        if not self.owner_file:
            return launch()
        from gpuwm import proc_identity
        from gpuwm.machine_agent import release_card, retag_card

        card = self.local_card(fresh=True)
        # This server's own claim is recorded before it is made, with this server's identity: a page server that
        # stops between claiming the card and handing the line on leaves this record, and the next sweep removes
        # the line once this process is gone or its number names another process (after a reboot, say).  The
        # record carries the line's time from the start and the claim writes that time, so the record names
        # that one line: one left by a server stopped before its line was written names no line at all, never
        # another line of this tag that a later process under the same number holds.
        stamp = _utc()
        me = proc_identity.identify(self.pid)
        claim = None if me is None else {"run": run_id, "pid": self.pid, "utc": stamp, "process": me,
                                         "line": {"tag": self.owner_tag, "utc": stamp, "pid": self.pid}}
        if claim is not None:
            self._swap_owned(None, claim)
        try:
            claimed, why = self._claim({"need_gib": need_gib}, card, stamp)
        except BaseException:
            self._swap_owned(claim, None)
            raise
        if not claimed:
            self._swap_owned(claim, None)
            raise ApiError(409, f"The card is taken: {why}.",
                           "Queue it instead, and it starts by itself when the card is free.")
        handed = False
        try:
            reply = launch()
            job = (reply.body or {}).get("job") or {}
            wrapper = int(job.get("wrapper_pid") or 0)
            identity = job.get("wrapper_process")
            if wrapper:
                # The line and the queue's record of it change together: the line's pid is handed to the run's
                # wrapper only while that pid is still the wrapper (its PID and creation time); a run that
                # already ended has its line removed here instead of handed to a number that may be another
                # program's.  The record keeps the wrapper's identity and the exact line, so the scheduler's
                # cleanup removes that line and no other.
                with self._lock:
                    line = retag_card(self.owner_file, self.pid, wrapper, tag=self.owner_tag,
                                      keep=lambda: proc_identity.alive(identity, wrapper))
                    handed = True
                    self._swap_owned(claim, None if line is None else {
                        "run": run_id, "pid": wrapper, "utc": _utc(), "process": identity, "line": line})
            return reply
        finally:
            if not handed:
                release_card(self.owner_file, self.pid)
                self._swap_owned(claim, None)
            self._local = None

    def _start(self, run_id: str, rundir: Path, marker: dict[str, Any], machine: str,
               *, expected_order: list[str] | None = None) -> tuple[bool, str | None]:
        """Start one queued forecast, here or on its Machines node: (started, why it keeps waiting).

        Its marker is renamed to :data:`STARTING` under the lock first, so a
        Remove or a move meanwhile finds it starting rather than queued; a
        start that does not happen renames it back, in its place in line.
        (False, None) when it was removed while this tick looked at the card.

        ``expected_order`` is the line the tick chose this forecast from.  The
        card and data checks run outside the lock, so a Move can land between
        the tick's reading of the line and this start; the line is compared
        in the same locked step that marks the forecast starting, and a
        changed line raises :class:`_QueueOrderChanged` so the tick looks
        again instead of starting the forecast that used to be first.
        """

        from .api import ApiError
        from .jobs import engine_argv, log_command

        queued, starting = rundir / runs.QUEUED, rundir / STARTING
        with self._lock:
            if expected_order is not None and self._orders()[0] != expected_order:
                raise _QueueOrderChanged
            if not queued.is_file():
                return False, None
            os.replace(queued, starting)
        try:
            if machine == LOCAL:
                self.api._launch(run_id, rundir, engine_argv("run-plan", str(rundir / runs.PLAN)),
                                 need_gib=marker.get("need_gib"))
            else:
                payload = dict(marker.get("payload") or {})
                payload.update({"machine": machine, "wait_min": REMOTE_WAIT_MIN})
                self.api.create_start_remote(machine, payload, False, queued=rundir)
        except Exception as error:  # noqa: BLE001 - whatever stopped the start, the forecast keeps its place
            if not started_in(rundir):
                with self._lock:
                    if starting.is_file():
                        os.replace(starting, queued)
                return False, error.message if isinstance(error, ApiError) else f"The start failed: {error}"
            # Something failed after the forecast had started: it leaves the line all the same.  Put back, it
            # was started a second time once the card was free.
            print(f"gpuwm gui: {run_id} started from the queue, but a step after its start failed: {error!r}",
                  file=sys.stderr, flush=True)
        with self._lock:
            starting.unlink(missing_ok=True)
            queue = self._document()
            queue["order"] = [item for item in queue["order"] if item != run_id]
            self._save(queue)
            self._local = None
        if machine == LOCAL:
            try:
                log_command(rundir, "# started from the queue when the card was free", "Queue")
            except OSError as error:
                print(f"gpuwm gui: {run_id} started from the queue; its commands.log line could not be written: "
                      f"{error!r}", file=sys.stderr, flush=True)
        return True, None

    def _recover(self) -> None:
        """Once per page server: a start a crash cut short.  A forecast left with its :data:`STARTING` marker goes
        back in line in its place when nothing started in its folder, and leaves the line when something did."""

        if self._recovered:
            return
        self._recovered = True
        self._orders(scan=True)
        self._scanned = time.monotonic()
        queue = self._document()
        for run_id in list(queue["order"]):
            if not self._starting(run_id):
                continue
            rundir = runs.run_path(self.root, run_id)
            if started_in(rundir):
                (rundir / STARTING).unlink(missing_ok=True)
                queue["order"] = [item for item in queue["order"] if item != run_id]
            else:
                os.replace(rundir / STARTING, rundir / runs.QUEUED)
        self._save(queue)

    def _release_ended(self) -> None:
        """The scheduler's sweep of this server's OWNER lines whose holder is gone.

        A recorded line (a run's wrapper, or a page server's own claim before
        the hand-over) goes once its process is not the one recorded: ended,
        or its number given to another process.  Any other line of this
        server's tag goes once no process holds its number.  A line of
        another tag is never touched.
        """

        if not self.owner_file:
            return
        from gpuwm import proc_identity
        from gpuwm.machine_agent import release_dead_lines, release_line

        queue = self._document()
        keep = []
        for row in queue["owned"]:
            # The run's own wrapper, not whatever holds its PID now: a PID reused after the run ended kept the
            # OWNER line (and so the card) held for a program that never asked for it.
            if proc_identity.alive(row.get("process"), row.get("pid")):
                keep.append(row)
            else:
                # Only the line this record names, by its tag, time and pid: another line under the same number,
                # of this tag or another, is another claim and stays.  A record naming no line removes nothing.
                release_line(self.owner_file, row.get("line"))
        if keep != queue["owned"]:
            queue["owned"] = keep
            self._save(queue)
        # A line of this tag that no record names (a server that could not read its own identity to record its
        # claim, or one from before claims were recorded) goes once no process holds its PID.
        release_dead_lines(self.owner_file, self.owner_tag)

    def tick(self) -> list[str]:
        """One look at the queue: start what may start.  Returns the run ids started.

        The queue's files are read and written under the lock; the card
        readings, the starts and a Machines node's SSH call run outside it,
        so a page asking for the queue, or queuing another forecast, never
        waits for a start under way.
        """

        started: list[str] = []
        with self._ticking:
            with self._lock:
                self._recover()
                self._release_ended()
                moment = time.monotonic()
                scan = self._scanned is None or moment - self._scanned >= SCAN_EVERY_S
                if scan:
                    self._scanned = moment
                expected_order, line = self._orders(scan=scan)
            decided: set[str] = set()
            cards: dict[str, dict[str, Any]] = {}
            for run_id, rundir, marker in line:
                machine = str(marker.get("machine") or LOCAL)
                if machine in decided:
                    # One that can never start on this card or disk says so now, not once it reaches the front.
                    room = self.hold_reason(marker, cards[machine], machine)
                    data, awaits = (None, False) if room else self.data_hold(marker)
                    self._note(rundir, marker, held=room or data,
                               waiting=None if room or data else AHEAD, awaits_data=awaits)
                    continue
                if machine not in cards:
                    cards[machine] = self.local_card(fresh=True) if machine == LOCAL else self.remote_card(machine)
                card = cards[machine]
                room = self.hold_reason(marker, card, machine)
                data, awaits = (None, False) if room else self.data_hold(marker)
                if room or data:
                    self._note(rundir, marker, held=room or data, waiting=None, awaits_data=awaits)
                    continue
                waiting = self.blocking(marker, card)
                if waiting is None:
                    try:
                        began, waiting = self._start(
                            run_id, rundir, marker, machine,
                            expected_order=expected_order)
                    except _QueueOrderChanged:
                        # The next look reads the line as it is now; the loop wakes for it at once.
                        self._wake.set()
                        break
                    if began:
                        # This start took the forecast out of the line itself: the line to compare against next.
                        expected_order = [item for item in expected_order if item != run_id]
                        started.append(run_id)
                        decided.add(machine)
                        continue
                    if waiting is None:
                        continue          # removed while this tick looked at its card
                decided.add(machine)
                self._note(rundir, marker, held=None, waiting=waiting)
            self.last_tick_utc = _utc()
        return started

    def _note(self, rundir: Path, marker: dict[str, Any], *, held: str | None, waiting: str | None,
              awaits_data: bool = False) -> None:
        # ``awaits_data``: held only until its start is confirmed published, an ordinary wait rather than a
        # forecast that cannot start as it stands (disk or card), so the page does not show it as a fault.
        if (marker.get("held") == held and marker.get("waiting") == waiting
                and bool(marker.get("awaits_data")) == awaits_data):
            return
        marker = {**marker, "held": held, "waiting": waiting, "awaits_data": awaits_data}
        with self._lock:
            # A forecast removed (or started) meanwhile has no marker to write, and never gets one back.
            if (rundir / runs.QUEUED).is_file():
                try:
                    write_json(rundir / runs.QUEUED, marker)
                except OSError:
                    pass

    # ------------------------------------------------------------ what the page reads

    def listing(self, machine: str | None = None, need_gib: float | None = None,
                data: dict[str, Any] | None = None) -> dict[str, Any]:
        """The queue in order, each forecast with its machine, place, and why it waits or is held.

        With ``machine``, also ``expect`` (:meth:`expect`); the queue's files are read once for both.
        """

        line = self._orders()[1]
        items = []
        places: dict[str, int] = {}
        for run_id, _, marker in line:
            where = str(marker.get("machine") or LOCAL)
            places[where] = places.get(where, 0) + 1
            items.append({"run": run_id, "machine": where, "place": places[where],
                          "queued_utc": marker.get("queued_utc"), "held": marker.get("held"),
                          "awaits_data": bool(marker.get("held") and marker.get("awaits_data")),
                          "waiting": marker.get("waiting"), "need_gib": marker.get("need_gib"),
                          "title": marker.get("title") or run_id})
        listing = {"items": items, "local": self.local_card(), "owner_file": self.owner_file,
                   "owner_tag": self.owner_tag, "last_tick_utc": self.last_tick_utc}
        if machine is not None:
            listing["expect"] = self.expect(machine, need_gib, line=line, data=data)
        return listing

    def expect(self, machine: str, need_gib: float | None,
               line: list[tuple[str, Path, dict[str, Any]]] | None = None,
               data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Where and when a forecast started now would run: at once, or after the ones ahead of it.

        With ``data`` (the source, start and length on the page), also what takes that start (``data.starts``:
        "now" Start, "queue" only Queue it, which holds the forecast until the start is confirmed, "no"
        neither) and why, from the page's own checks.
        """

        card = self.card(machine)
        line = self._orders()[1] if line is None else line
        ahead = sum(1 for _, _, marker in line
                    if str(marker.get("machine") or LOCAL) == machine and not marker.get("held"))
        marker = {"need_gib": need_gib}
        waiting = self.blocking(marker, card)
        running = card.get("running") or {}
        left = ((running.get("status") or {}).get("seconds_left")) if running else None
        held = self.hold_reason(marker, card, machine)
        answer = None
        if data and data.get("source") and data.get("cycle"):
            try:
                row = self.api.availability_of(str(data["source"]), str(data["cycle"]),
                                               float(data.get("hours") or 6))
                answer = {"starts": row.get("starts", "now"), "why": row.get("why") or "",
                          "checking": bool(row.get("checking"))}
            except Exception:  # noqa: BLE001 - the Start request checks again, and says why
                answer = None
        return {"machine": machine, "busy": waiting is not None, "why": waiting, "ahead": ahead,
                "held": held, "running": running.get("run") if running else None, "data": answer,
                "seconds_left": left, "card_name": card.get("name"), "total_gib": card.get("total_gib"),
                "free_gib": card.get("free_gib"), "checking": bool(card.get("checking")),
                # Start now takes the card beside what runs there only when the card lock and OWNER allow it, and
                # never a start the queue would hold: that one fails, or cannot run, however free the card is.
                "start_now": held is None and (waiting is None
                                               or (not card.get("busy") and self._fits_beside(marker, card)))}

    # ------------------------------------------------------------ the thread

    def start(self) -> threading.Thread:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="arwen-gui-queue", daemon=True)
            self._thread.start()
            return self._thread

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - the queue keeps going; the log says what failed
                print(f"gpuwm gui: the queue check failed: {error}", file=sys.stderr, flush=True)
            self._wake.wait(self.tick_s)
            self._wake.clear()


__all__ = ["DISK_FLOOR_GIB", "ForecastQueue", "LOCAL", "OWNER_FILE_ENV", "OWNER_TAG_ENV", "QUEUE_FILE"]
