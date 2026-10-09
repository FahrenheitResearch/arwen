"""The machine side of ``gpuwm gui``'s Machines: one small file, standard library only.

The page's server copies this file to a machine over SSH (into the
machine's workspace, named by its own hash) and runs it with the
machine's ``python3``.  It never listens on a port and never runs a
command it was sent: every verb below is fixed, and each answers with
one JSON document on stdout (``pack`` answers with a tar stream).

It does four jobs on the machine:

- ``probe``: what the machine is and what it is doing now (cards and
  memory, free disk, the gpuwm install, the card-sharing OWNER file if
  the machine uses one, and the forecasts and renders this workspace
  holds);
- ``launch``/``snapshot``/``stop``: a forecast, run detached by a
  supervisor that claims the card through the OWNER file first (one
  test-and-append under a lock) and removes its own line when it ends;
- ``render-start``/``render-feed``/``render-end``/``render-list``: a
  render worker that draws frames with ``gpuwm render`` (the Rust
  renderer) as they are fed to it;
- ``pack``/``unpack``: move files as one tar stream, so frames and
  pictures travel without a second copy tool.

A workspace looks like::

    <workspace>/runs/<run>/            plan.json, the run's own files, engine.log
    <workspace>/runs/<run>/gui-job.json
    <workspace>/renders/<job>/inbox/   frames fed to a render worker
    <workspace>/renders/<job>/run-<stamp>/<domain>/<product>/<day>/*.png
    <workspace>/renders/<job>/render-job.json
"""

from __future__ import annotations

import argparse
import base64
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

AGENT_SCHEMA = "gpuwm.machine-agent.v1"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
EVENTS_CHUNK = 1024 * 1024
LOG_TAIL = 6000
#: How often a waiting forecast looks at the card again.
CARD_POLL_S = 20.0
RENDER_POLL_S = 3.0
#: A render worker that has been fed nothing and told nothing for this
#: long stops by itself, because a page server that went away would
#: otherwise leave it polling on the machine forever.
RENDER_IDLE_S = 6 * 3600.0
STOP_ESCALATE_S = 60.0


# ------------------------------------------------------------------ small parts

def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, document) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_json(path: Path, **fields) -> dict:
    document = read_json(path, default={}) or {}
    document.update(fields)
    write_json(path, document)
    return document


def pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a liveness probe on Windows: signal 0 is CTRL_C_EVENT there, sent to a console
        # group, and it succeeds for a process that has already exited, so a dead worker read as alive forever.
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        # A zombie has exited; only its parent has not collected it yet.
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def process_start(pid) -> str | None:
    """When the process holding ``pid`` now was created (with the boot it belongs to), or None when none does.

    A PID names a process only until that process ends: after a reboot or
    a crash another program can hold the same number, and a record read
    by PID alone showed a dead forecast as running and let Stop signal
    that program.  So every PID this agent writes has its start beside
    it, and :func:`process_alive` compares the two.
    """

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or not pid_alive(pid):
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel32.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
                return None
            return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel32.CloseHandle(handle)
    try:
        # Field 22 of /proc/<pid>/stat is the start in clock ticks since boot; the boot id makes it unique.
        ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot}:{ticks}"
    except (OSError, IndexError):
        pass
    try:
        done = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return " ".join(done.stdout.split()) or None


def process_alive(pid, start) -> bool:
    """``pid`` is running and is still the process whose start was recorded; a record without one is not."""

    return bool(start) and process_start(pid) == str(start)


def name(value: str, what: str) -> str:
    if not NAME.match(str(value or "")):
        raise SystemExit(f"{what} {value!r} is not a plain name")
    return str(value)


def tail(path: Path, size: int = LOG_TAIL) -> str:
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end - size))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def engine_env(python: str | None, extra: dict | None) -> dict:
    """This environment, the row's additions, and the engine's own bin folder first on PATH
    (the engine prints commands that start with its console script)."""

    env = dict(os.environ)
    env.update(extra or {})
    if python:
        folder = str(Path(python).parent)
        env["PATH"] = folder + os.pathsep + env.get("PATH", "")
    return env


def expand_home(value):
    """Every string starting with ~/ in a document, expanded on this machine."""

    if isinstance(value, dict):
        return {key: expand_home(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_home(item) for item in value]
    if isinstance(value, str) and value.startswith("~/"):
        return os.path.expanduser(value)
    return value


def spawn(argv: list[str], *, cwd: Path, log: Path, env: dict | None = None) -> int:
    """Start one process in its own session, detached from this SSH call."""

    with open(log, "ab") as out:
        process = subprocess.Popen(argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env=env, close_fds=True)
    return process.pid


# ------------------------------------------------------------------ cards and OWNER

def cards() -> dict:
    """The machine's NVIDIA cards and compute processes, from nvidia-smi."""

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {"devices": [], "processes": [], "cuda_major": None,
                "error": "nvidia-smi is not on this machine's PATH"}
    try:
        rows = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        apps = subprocess.run(
            [smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        header = subprocess.run([smi], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return {"devices": [], "processes": [], "cuda_major": None, "error": str(error)}
    devices = []
    for line in rows.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append({"index": int(parts[0]), "name": parts[1],
                            "memory_total_mib": int(float(parts[2])),
                            "memory_used_mib": int(float(parts[3])), "driver": parts[4]})
        except ValueError:
            continue
    processes = []
    for line in apps.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                used = int(float(parts[1]))
            except ValueError:
                used = None
            processes.append({"pid": int(parts[0]), "used_mib": used})
    found = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", header)
    return {"devices": devices, "processes": processes,
            "cuda": None if not found else f"{found.group(1)}.{found.group(2)}",
            "cuda_major": None if not found else int(found.group(1)), "error": None}


OWNER_LINE = re.compile(r"^(?P<tag>\S+)\s+(?P<utc>\S+)\s+pid\s+(?P<pid>\d+)\s+bounded\s+(?P<bound>\d+)\s+min")


def owner_lines(owner_file: str | None) -> list[dict]:
    """The OWNER file's lines, each marked live when its pid still runs here."""

    if not owner_file:
        return []
    try:
        text = Path(owner_file).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        found = OWNER_LINE.match(raw.strip())
        row = {"line": raw.strip(), "live": False}
        if found:
            row.update(tag=found.group("tag"), utc=found.group("utc"), pid=int(found.group("pid")),
                       bound_min=int(found.group("bound")))
            row["live"] = pid_alive(row["pid"])
        else:
            # A line this reader cannot parse is treated as a live owner:
            # guessing it is dead would put two runs on one card.
            row["live"] = True
        out.append(row)
    return out


def _locked(stream):
    try:
        import fcntl
    except ImportError:  # not Linux: nothing to lock against
        return None
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return fcntl


class _OwnerLock:
    """Both locks a machine's OWNER convention may use, taken in one order.

    Some machines lock the OWNER file itself; others keep a sibling
    ``OWNER.lock`` that every writer flocks (a writer of that kind never
    locks OWNER).  Taking the sibling first, when it exists, then the
    file, excludes writers of either kind, so an append here can never
    interleave with theirs.
    """

    def __init__(self, owner_file: str) -> None:
        self.path = Path(owner_file)
        self.sibling = self.path.with_name(self.path.name + ".lock")
        self._held = []

    def __enter__(self):
        try:
            import fcntl
        except ImportError:  # not Linux: nothing to lock against
            return self
        if self.sibling.exists():
            handle = open(self.sibling, "a")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._held.append(handle)
        return self

    def __exit__(self, *exc) -> None:
        if not self._held:
            return
        import fcntl

        while self._held:
            handle = self._held.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def claim_card(owner_file: str | None, tag: str, pid: int, bound_min: int,
               own_pids: tuple[int, ...] = (), info: dict | None = None, at: str | None = None) -> dict:
    """One test-and-append: claim the card when no compute process and no live owner line.

    Returns {"claimed": bool, "why": str}, and the written ``line`` when
    one is.  With no OWNER file the card is claimed when nvidia-smi shows
    no compute process.  ``info`` is a :func:`cards` reading the caller
    just took; without it one is taken.  ``at`` is the line's time when the
    caller recorded it before claiming, so that its record names this one
    line from the start; without it the line takes the time of the claim.
    """

    info = info if info is not None else cards()
    busy = [row for row in info["processes"] if row["pid"] not in own_pids]
    if not owner_file:
        if busy:
            return {"claimed": False, "why": f"the card has {len(busy)} compute process(es) running"}
        return {"claimed": True, "why": "no compute process on the card"}
    path = Path(owner_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _OwnerLock(owner_file), open(path, "a+", encoding="utf-8") as stream:
        locker = _locked(stream)
        try:
            live = [row for row in owner_lines(owner_file) if row["live"]]
            if live:
                return {"claimed": False, "why": "the card is held by " + live[0]["line"]}
            if busy:
                return {"claimed": False,
                        "why": f"the card has {len(busy)} compute process(es) and no owner line"}
            stream.seek(0, os.SEEK_END)
            line = f"{tag} {at or utc()} pid {pid} bounded {bound_min} min\n"
            stream.write(line)
            stream.flush()
            return {"claimed": True, "why": "claimed", "line": line.strip()}
        finally:
            if locker is not None:
                locker.flock(stream.fileno(), locker.LOCK_UN)


def release_card(owner_file: str | None, pid: int) -> None:
    """Remove this pid's own line and nobody else's."""

    if not owner_file:
        return
    _rewrite(owner_file, lambda found, raw: None if int(found.group("pid")) == pid else raw)


def retag_card(owner_file: str | None, old_pid: int, new_pid: int, *, tag: str | None = None,
               keep=None) -> dict | None:
    """Hand this process's own line to another pid (the run it just started), under the same locks.

    The line keeps its tag, time and bound; only its pid changes, so the
    card stays held across the hand-over and the line lives exactly as
    long as the run does.  ``tag``, when given, is the only tag a line of
    ours has.  ``keep`` is asked under the locks whether the new pid is
    still the process the line is handed to; when it is not (the run
    already ended, and its number may be another program's), the line is
    removed instead, never handed to a number that names someone else.
    Returns the handed line's ``{tag, utc, pid}``, or None when none was.
    """

    if not owner_file:
        return None
    handed: list[dict] = []

    def swap(found, raw):
        if int(found.group("pid")) != old_pid or (tag is not None and found.group("tag") != tag):
            return raw
        if keep is not None and not keep():
            return None
        handed.append({"tag": found.group("tag"), "utc": found.group("utc"), "pid": int(new_pid)})
        return raw.replace(f" pid {old_pid} ", f" pid {new_pid} ", 1)

    _rewrite(owner_file, swap)
    return handed[0] if handed else None


def release_line(owner_file: str | None, line: dict) -> None:
    """Remove the one OWNER line ``line`` records (its tag, its time and its pid), and no other.

    A line whose pid matches but whose tag or time does not is another
    claim, whoever holds that number now, and stays.  A record with no time
    names no line and removes nothing: tag and pid alone also match the
    line of a later process of that tag given the same number.
    """

    if not owner_file or not isinstance(line, dict) or not line.get("utc"):
        return

    def drop(found, raw):
        same = (found.group("tag") == line.get("tag") and int(found.group("pid")) == int(line.get("pid") or 0)
                and found.group("utc") == line.get("utc"))
        return None if same else raw

    _rewrite(owner_file, drop)


def release_dead_lines(owner_file: str | None, tag: str) -> list[dict]:
    """Remove every line of ``tag`` whose PID no process holds now, and no line of any other tag.

    Such a line is no one's claim any more: its holder ended without
    removing it (a page server that stopped between claiming the card and
    handing the line on, a wrapper killed outright).  Left in place it is
    harmless only until its number is given to another process, which then
    holds the card as far as every reader of the file can tell.  Each line
    is judged again under the file's locks.  Returns the removed lines'
    ``{tag, utc, pid}``.
    """

    if not owner_file or not any(row.get("tag") == tag and not row["live"] for row in owner_lines(owner_file)):
        return []
    removed: list[dict] = []

    def drop(found, raw):
        if found.group("tag") != tag or pid_alive(int(found.group("pid"))):
            return raw
        removed.append({"tag": found.group("tag"), "utc": found.group("utc"), "pid": int(found.group("pid"))})
        return None

    _rewrite(owner_file, drop)
    return removed


def _rewrite(owner_file: str, change) -> None:
    """Rewrite each parsed OWNER line through ``change(match, raw) -> raw | None``; others stay as they are."""

    path = Path(owner_file)
    try:
        with _OwnerLock(owner_file), open(path, "r+", encoding="utf-8") as stream:
            locker = _locked(stream)
            try:
                lines = stream.read().splitlines(keepends=True)
                keep = []
                for raw in lines:
                    found = OWNER_LINE.match(raw.strip())
                    if found:
                        raw = change(found, raw)
                        if raw is None:
                            continue
                    keep.append(raw)
                if keep != lines:
                    stream.seek(0)
                    stream.truncate()
                    stream.write("".join(keep))
            finally:
                if locker is not None:
                    locker.flock(stream.fileno(), locker.LOCK_UN)
    except OSError:
        pass


# ------------------------------------------------------------------ probe

#: Run in the machine's interpreter: its gpuwm, and the runtime requirements that gpuwm declares and that are not
#: installed there.  The installer puts gpuwm in before its requirements (``cmd_install``), and an install that
#: stops between the two leaves a gpuwm that imports and reports its version with numpy or the rest missing.
#: Requirements behind an extra or a marker are not asked about; a gpuwm that is not an installed distribution (a
#: source tree on the path) declares none here.
VERSION_PROBE = """\
import json, re, sys
try:
    import gpuwm; v = gpuwm.__version__; f = gpuwm.__file__
except Exception as e:
    v = None; f = str(e)
missing = []
try:
    from importlib import metadata
    for line in metadata.requires("gpuwm") or []:
        if ";" in line:
            continue
        wanted = re.split(r"[\\s<>=!~\\[(]", line.strip(), maxsplit=1)[0]
        try:
            metadata.version(wanted)
        except metadata.PackageNotFoundError:
            missing.append(wanted)
except Exception:
    pass
print(json.dumps({"version": v, "file": f, "py": sys.version.split()[0], "missing": missing}))
"""


def gpuwm_version(python: str | None) -> dict:
    if not python:
        return {"python": None, "version": None, "error": "no interpreter named"}
    if not Path(python).exists() and shutil.which(python) is None:
        return {"python": python, "version": None, "error": f"{python} does not exist on this machine"}
    try:
        done = subprocess.run([python, "-I", "-c", VERSION_PROBE], capture_output=True, text=True, timeout=60)
        document = json.loads(done.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"python": python, "version": None, "error": f"could not ask {python}: {error}"}
    missing = document.get("missing")
    return {"python": python, "version": document.get("version"), "file": document.get("file"),
            "python_version": document.get("py"),
            "missing": [str(name) for name in missing] if isinstance(missing, list) else [],
            "error": None if document.get("version") else f"gpuwm is not importable: {document.get('file')}"}


def workspace_jobs(workspace: Path) -> dict:
    forecasts, renders = [], []
    for job_file in sorted((workspace / "runs").glob("*/gui-job.json")):
        job = read_json(job_file, default={}) or {}
        alive = process_alive(job.get("supervisor_pid"), job.get("supervisor_start"))
        forecasts.append({"run": job_file.parent.name, "state": job.get("state"), "alive": alive,
                          "started_utc": job.get("started_utc"), "ended_utc": job.get("ended_utc")})
    for job_file in sorted((workspace / "renders").glob("*/render-job.json")):
        job = read_json(job_file, default={}) or {}
        renders.append({"job": job_file.parent.name, "run": job.get("run"), "state": job.get("state"),
                        "alive": process_alive(job.get("pid"), job.get("pid_start")),
                        "rendered": job.get("rendered", 0)})
    return {"forecasts": forecasts, "renders": renders}


def cmd_probe(args) -> dict:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(workspace)
    info = cards()
    owners = owner_lines(args.owner_file)
    jobs = workspace_jobs(workspace)
    own_supervisors = set()
    for job_file in (workspace / "runs").glob("*/gui-job.json"):
        job = read_json(job_file, default={}) or {}
        for key in ("supervisor", "engine"):
            if process_alive(job.get(f"{key}_pid"), job.get(f"{key}_start")):
                own_supervisors.add(int(job[f"{key}_pid"]))
    others = [row for row in owners if row["live"] and row.get("pid") not in own_supervisors]
    running = [row for row in jobs["forecasts"] if row["alive"] and row["state"] != "waiting-for-card"]
    waiting = [row for row in jobs["forecasts"] if row["alive"] and row["state"] == "waiting-for-card"]
    rendering = [row for row in jobs["renders"] if row["alive"] and row["state"] == "rendering"]
    foreign = [row for row in info["processes"] if row["pid"] not in own_supervisors]
    if running:
        state, detail = "running", "running " + ", ".join(row["run"] for row in running)
    elif others:
        state, detail = "busy", "the card is held by " + others[0]["line"]
        if waiting:
            detail += "; waiting for it: " + ", ".join(row["run"] for row in waiting)
    elif foreign and not running:
        state, detail = "busy", f"the card has {len(foreign)} compute process(es) of another user"
    elif rendering:
        state, detail = "rendering", "rendering " + ", ".join(str(row["run"]) for row in rendering)
    else:
        state, detail = "idle", "idle"
    if rendering and state != "rendering":
        detail += "; also rendering " + ", ".join(str(row["run"]) for row in rendering)
    return {
        "schema": AGENT_SCHEMA, "ok": True, "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "utc": utc(), "state": state, "detail": detail, "cards": info,
        "disk": {"path": str(workspace), "free_gib": round(usage.free / 2**30, 1),
                 "total_gib": round(usage.total / 2**30, 1)},
        "gpuwm": gpuwm_version(args.python),
        "owner_file": args.owner_file or None,
        "owner_file_exists": bool(args.owner_file) and Path(args.owner_file).is_file(),
        "owners": owners, "jobs": jobs,
        "install": install_record(workspace),
    }


# ------------------------------------------------------------------ forecasts

def run_dir(workspace: Path, run: str) -> Path:
    return workspace / "runs" / name(run, "run")


#: Files a launch writes or the supervisor keeps in a run folder; a sent file never takes one of these names.
LAUNCH_RESERVED = frozenset({"gui-job.json", "gui-stop-request", "engine.log", "supervisor.log"})
#: Windows device names, which open the device instead of a file of that name in any folder.
DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                          *(f"lpt{i}" for i in range(1, 10))})


def _launch_files(request: dict) -> tuple[dict[str, bytes] | None, str | None]:
    """Every file a launch writes, as bytes, or the reason one cannot be written; read before anything is.

    ``files`` are JSON documents (the plan, the region); ``text_files`` are
    files a plan's configuration names beside itself (a storm-following
    layout's TOML, Vtable and WPS namelist), written as the text they are:
    passed through write_json they were quoted into one JSON string, which
    no reader of a TOML or a namelist can read.
    """

    files, text_files = request.get("files", {}), request.get("text_files", {})
    if not isinstance(files, dict) or not isinstance(text_files, dict):
        return None, "The forecast's files did not arrive as a list of named files."
    if "plan.json" not in files:
        return None, "The request carried no plan.json."
    encoded: dict[str, bytes] = {}
    for mapping, plain_text in ((files, False), (text_files, True)):
        for file_name, document in mapping.items():
            spelling = str(file_name).lower()
            if (not NAME.match(str(file_name)) or spelling.endswith(".") or spelling in LAUNCH_RESERVED
                    or spelling.split(".", 1)[0] in DEVICE_NAMES or spelling in {key.lower() for key in encoded}):
                return None, f"{file_name!r} is not a plain file name this forecast can use."
            if plain_text and (not isinstance(document, str) or "\0" in document):
                return None, f"{file_name} did not arrive as text."
            try:
                text = document if plain_text else json.dumps(expand_home(document), indent=2, sort_keys=True) + "\n"
                encoded[str(file_name)] = text.encode("utf-8")
            except (UnicodeError, TypeError, ValueError):
                return None, f"{file_name} could not be written as UTF-8 text."
    return encoded, None


def cmd_launch(args) -> dict:
    """Write the plan and start the detached supervisor; refuse a busy card unless told to wait.

    Two launches of one run name on this machine are decided one at a time
    under the workspace's launch lock for that name: the second finds the
    first's supervisor and is refused, instead of both writing the folder
    and both starting. A folder a launch left that never started (the card
    refused it, or its supervisor could not start) is taken by the next
    launch of that name, which is how a queued forecast tries again; a run
    that is going or finished there is never written over.
    """

    workspace = Path(args.workspace)
    rundir = run_dir(workspace, args.run)
    request = json.loads(sys.stdin.read() or "{}")
    if not isinstance(request, dict):
        return {"ok": False, "message": "The forecast request did not arrive whole.", "fix": "Start it again."}
    encoded, why = _launch_files(request)
    if encoded is None:
        return {"ok": False, "message": why, "fix": "Start it again from the page."}
    wait_min = int(request.get("wait_min") or 0)
    if wait_min <= 0:
        verdict = card_verdict(args.owner_file)
        if verdict:
            return {"ok": False, "message": f"This machine's card is busy: {verdict}.",
                    "fix": "Wait until it is idle, or start it with a wait so it begins when the card frees up."}
    rundir.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(rundir.parent / f".{args.run}.launch.lock"):
        if rundir.is_symlink():
            return {"ok": False, "message": f"{args.run} on this machine is a link to another folder.",
                    "fix": "Give the forecast another name."}
        if (rundir / "gui-job.json").is_file():
            job = read_json(rundir / "gui-job.json", default={}) or {}
            supervised = process_alive(job.get("supervisor_pid"), job.get("supervisor_start"))
            if supervised or job.get("state") in ("finished", "running"):
                return {"ok": False, "message": f"{args.run} already exists on this machine ({job.get('state')}).",
                        "fix": "Give the forecast another name."}
        rundir.mkdir(parents=True, exist_ok=True)
        for file_name, content in encoded.items():
            target = rundir / file_name
            tmp = target.with_name(f"{file_name}.{os.getpid()}.tmp")
            tmp.write_bytes(content)
            os.replace(tmp, target)
        job = {"schema": AGENT_SCHEMA, "run": args.run, "state": "starting", "created_utc": utc(),
               "python": args.python, "owner_file": args.owner_file or None, "owner_tag": args.owner_tag,
               "bound_min": int(request.get("bound_min") or 120), "wait_min": wait_min,
               "env": {str(k): str(v) for k, v in (request.get("env") or {}).items()},
               "argv": [args.python, "-m", "gpuwm", "run-plan", str(rundir / "plan.json")]}
        write_json(rundir / "gui-job.json", job)
        pid = spawn([sys.executable, str(Path(__file__).resolve()), "supervise", "--workspace", str(workspace),
                     "--run", args.run], cwd=rundir, log=rundir / "supervisor.log")
        job = update_json(rundir / "gui-job.json", supervisor_pid=pid, supervisor_start=process_start(pid))
    return {"ok": True, "run": args.run, "rundir": str(rundir), "job": job,
            "command": " ".join(job["argv"])}


def card_verdict(owner_file: str | None) -> str | None:
    live = [row for row in owner_lines(owner_file) if row["live"]]
    if live:
        return "held by " + live[0]["line"]
    busy = cards()["processes"]
    if busy:
        return f"{len(busy)} compute process(es) running"
    return None


#: What a job whose engine cannot start ends with, whether that is found before or after the card wait.
ENGINE_GONE = ("The forecast engine could not start on this machine: its Python is missing or cannot run. "
               "Install this version on it again.")


def engine_missing(argv) -> bool:
    """True when the engine's program is not there to run: a venv removed since the install."""

    program = str((argv or [""])[0] or "")
    return not program or shutil.which(program) is None


def cmd_supervise(args) -> int:
    """Wait for the card (bounded), claim it, run the engine, release, record the end."""

    workspace = Path(args.workspace)
    rundir = run_dir(workspace, args.run)
    job_path = rundir / "gui-job.json"
    job = read_json(job_path, default={}) or {}
    me = os.getpid()
    owner_file = job.get("owner_file")
    if engine_missing(job.get("argv")):
        # Found before the card wait: a job that cannot start must not hold its place in the card's queue for
        # up to its wait, only to fail when the card frees up.
        print("gpuwm machine agent: the engine is missing: " + str((job.get("argv") or [""])[0]),
              file=sys.stderr, flush=True)
        update_json(job_path, state="failed", exit_code=127, ended_utc=utc(), message=ENGINE_GONE,
                    supervisor_pid=me, supervisor_start=process_start(me))
        return 1
    deadline = time.monotonic() + 60.0 * float(job.get("wait_min") or 0)
    update_json(job_path, state="waiting-for-card", supervisor_pid=me, supervisor_start=process_start(me))
    while True:
        if (rundir / "gui-stop-request").exists():
            update_json(job_path, state="stopped", ended_utc=utc(), message="Stopped before the card was free.")
            return 130
        verdict = claim_card(owner_file, job.get("owner_tag") or "gpuwm-gui", me, int(job.get("bound_min") or 120))
        if verdict["claimed"]:
            break
        update_json(job_path, waiting_because=verdict["why"])
        if time.monotonic() >= deadline:
            update_json(job_path, state="refused", ended_utc=utc(),
                        message=f"The card did not free up within {job.get('wait_min') or 0} min: {verdict['why']}.")
            return 75
        time.sleep(CARD_POLL_S)
    env = engine_env(job.get("python"), job.get("env"))
    started = time.monotonic()
    try:
        try:
            with open(rundir / "engine.log", "ab") as out:
                engine = subprocess.Popen(job["argv"], cwd=str(rundir), stdin=subprocess.DEVNULL, stdout=out,
                                          stderr=subprocess.STDOUT, start_new_session=True, env=env)
        except OSError as error:
            # The engine's environment is gone or cannot run (a venv removed since the install). The job ends
            # here, failed, and the card is released below; without an end the page's follower waited on a
            # supervisor that had died with a traceback.
            print(f"gpuwm machine agent: the engine did not start: {error}", file=sys.stderr, flush=True)
            update_json(job_path, state="failed", exit_code=127, ended_utc=utc(), message=ENGINE_GONE,
                        wall_seconds=round(time.monotonic() - started, 1))
            return 1
        update_json(job_path, state="running", engine_pid=engine.pid, engine_start=process_start(engine.pid),
                    started_utc=utc(), owner_line=verdict.get("line"))
        bound_s = 60.0 * float(job.get("bound_min") or 120)
        interrupted_at = None
        # Ctrl+C, then after a grace a SIGTERM, then after another a SIGKILL: an engine that answered neither
        # held the card for another hour past its bound.
        escalation = [signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)]
        while engine.poll() is None:
            now = time.monotonic()
            if interrupted_at is None and (now - started > bound_s or (rundir / "gui-stop-request").exists()):
                why = "bound" if now - started > bound_s else "stop"
                update_json(job_path, stop_reason=why)
                try:
                    os.killpg(engine.pid, signal.SIGINT)
                except OSError:
                    pass
                interrupted_at = now
            elif interrupted_at is not None and escalation and now - interrupted_at > STOP_ESCALATE_S:
                # The engine is this process's own child, not yet reaped, so its group cannot have been handed on.
                try:
                    os.killpg(engine.pid, escalation.pop(0))
                except OSError:
                    pass
                interrupted_at = now
            time.sleep(1.0)
        code = engine.returncode
    finally:
        release_card(owner_file, me)
    final = read_json(job_path, default={}) or {}
    # A run told to stop, by its bound or by Stop, reads stopped however its engine exited.
    state = "stopped" if final.get("stop_reason") else ("finished" if code == 0 else "failed")
    update_json(job_path, state=state, exit_code=code, ended_utc=utc(),
                wall_seconds=round(time.monotonic() - started, 1), owner_released_utc=utc())
    return 0


def cmd_snapshot(args) -> dict:
    rundir = run_dir(Path(args.workspace), args.run)
    if not rundir.is_dir():
        return {"ok": False, "message": f"No forecast called {args.run} on this machine.", "fix": ""}
    job = read_json(rundir / "gui-job.json", default=None)
    events = rundir / "events.jsonl"
    chunk, size = b"", 0
    try:
        size = events.stat().st_size
        if size > args.offset:
            with open(events, "rb") as stream:
                stream.seek(args.offset)
                chunk = stream.read(EVENTS_CHUNK)
    except OSError:
        pass
    alive = isinstance(job, dict) and (process_alive(job.get("supervisor_pid"), job.get("supervisor_start"))
                                       or process_alive(job.get("engine_pid"), job.get("engine_start")))
    return {"ok": True, "run": args.run, "utc": utc(), "job": job, "alive": bool(alive),
            "manifest": read_json(rundir / "run-manifest.json", default=None),
            "heartbeat": read_json(rundir / "run-progress.json", default=None),
            "events_size": size, "events_offset": args.offset,
            "events_b64": base64.b64encode(chunk).decode("ascii"),
            "engine_log": tail(rundir / "engine.log")}


def cmd_stop(args) -> dict:
    rundir = run_dir(Path(args.workspace), args.run)
    job = read_json(rundir / "gui-job.json", default=None)
    if not isinstance(job, dict):
        return {"ok": False, "message": f"No forecast called {args.run} on this machine.", "fix": ""}
    (rundir / "gui-stop-request").write_text(utc() + "\n")
    engine = job.get("engine_pid")
    sent = False
    # Only the engine this job started: a PID given to another program since (after a reboot) is not signalled.
    if process_alive(engine, job.get("engine_start")):
        try:
            os.killpg(int(engine), signal.SIGINT)
            sent = True
        except OSError:
            pass
    return {"ok": True, "method": "interrupt" if sent else "request",
            "message": "Sent Ctrl+C to the run. It stops at its next step and records that it was stopped."
            if sent else "The run had not started on the card yet; it will not start."}


def cmd_runs(args) -> dict:
    return {"ok": True, **workspace_jobs(Path(args.workspace))}


# ------------------------------------------------------------------ renders

def render_dir(workspace: Path, job: str) -> Path:
    return workspace / "renders" / name(job, "render job")


def _end_marker(folder: Path) -> Path:
    return folder / "inbox" / "END"


def _ended_attempt(folder: Path) -> int | None:
    """The attempt an END marker ends, or None when there is no marker.

    A marker written before attempts were numbered reads as -1 and ends
    whichever attempt reads it: a new attempt always clears old markers first.
    """

    try:
        text = _end_marker(folder).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(json.loads(text).get("attempt"))
    except (ValueError, TypeError, AttributeError):
        return -1


def _newer_attempt(job_path: Path, attempt: int) -> int | None:
    """The attempt a fresh start has handed this worker since ``attempt``, or None.

    Attempts only ever count up, so a job file that cannot be read this
    instant is not mistaken for a new attempt that throws away every frame
    already drawn.
    """

    try:
        now = int((read_json(job_path, default={}) or {}).get("attempt") or 0)
    except (TypeError, ValueError):
        return None
    return now if now > attempt else None


@contextlib.contextmanager
def _job_lock(folder: Path):
    """Hold one render job's hand-over lock (``render-job.lock``).

    A Draw again that lands while the worker is finishing must either be
    taken over by that worker or find it gone and start a new one. Without
    one lock around both decisions the worker could read "no new attempt",
    the start could read "worker alive" and hand it the attempt, and the
    worker could then exit: the new attempt waited for a worker that was
    gone and drew nothing. The worker takes this lock to record a batch and
    to finish; a start takes it to decide and write the new attempt.
    """

    folder.mkdir(parents=True, exist_ok=True)
    with _file_lock(folder / "render-job.lock"):
        yield


@contextlib.contextmanager
def _file_lock(path: Path):
    """Hold an exclusive lock on ``path`` (created when absent) across processes."""

    with open(path, "a+b") as handle:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        import msvcrt

        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                break
            except OSError:  # LK_LOCK gives up after 10 s; the holder only ever holds it for a file write
                continue
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _worker_live(job) -> bool:
    """Whether the worker a job file names is still taking attempts.

    A worker that wrote ``finished`` or ``stopped`` under the job lock has
    returned and takes nothing more, even while its process is exiting.
    """

    return (isinstance(job, dict) and process_alive(job.get("pid"), job.get("pid_start"))
            and job.get("state") not in ("finished", "stopped"))


def cmd_render_start(args) -> dict:
    """Start a render worker, or keep the one that is going.

    Every start is one attempt, numbered in render-job.json. Its completion
    state (the END marker, the frames it drew) belongs to that attempt: a
    fresh start clears what the last attempt left, so a Draw again after a
    finished draw waits for its own end instead of finishing at once on the
    previous attempt's marker.
    """

    workspace = Path(args.workspace)
    folder = render_dir(workspace, args.job)
    request = json.loads(sys.stdin.read() or "{}")
    with _job_lock(folder):
        return _render_start_locked(args, workspace, folder, request)


def _render_start_locked(args, workspace: Path, folder: Path, request: dict) -> dict:
    job_path = folder / "render-job.json"
    existing = read_json(job_path, default=None)
    fresh = bool(request.get("fresh"))
    attempt = int((existing or {}).get("attempt") or 0) + 1
    asked = {"run": request.get("run"), "python": args.python, "products": request.get("products") or None,
             "render_section": request.get("render_section") or None,
             "extra_args": [str(item) for item in request.get("extra_args") or []],
             "env": {str(k): str(v) for k, v in (request.get("env") or {}).items()}}
    if _worker_live(existing):
        if not fresh:
            return {"ok": True, "job": existing, "already": True}
        # The worker is still up, winding down on the last attempt's end: it takes this attempt over (it reads
        # the number each pass, with what this attempt asks for) and draws every frame again, instead of
        # finishing and leaving the new frames queued for a worker that is gone.
        _end_marker(folder).unlink(missing_ok=True)
        job = update_json(job_path, attempt=attempt, done=[], rendered=0, batches=[], state="waiting",
                          ended_utc=None, **asked)
        return {"ok": True, "job": job, "already": True}
    (folder / "inbox").mkdir(parents=True, exist_ok=True)
    # No worker is alive, so an END in the inbox was written for an attempt that is over.
    _end_marker(folder).unlink(missing_ok=True)
    out = (existing or {}).get("out") or str(folder / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")))
    Path(out).mkdir(parents=True, exist_ok=True)
    job = {"schema": AGENT_SCHEMA, "job": args.job, "state": "starting", **asked,
           "out": out, "created_utc": (existing or {}).get("created_utc") or utc(),
           "attempt": attempt, "done": [], "rendered": 0, "batches": []}
    if existing and not fresh:
        job.update(done=existing.get("done") or [], rendered=existing.get("rendered", 0),
                   batches=existing.get("batches") or [])
    write_json(job_path, job)
    pid = spawn([sys.executable, str(Path(__file__).resolve()), "render-loop", "--workspace", str(workspace),
                 "--job", args.job], cwd=folder, log=folder / "render.log")
    job = update_json(job_path, pid=pid, pid_start=process_start(pid))
    return {"ok": True, "job": job}


def cmd_render_feed(args) -> dict:
    """Queue frames: each is a path on this machine (a relayed copy or the run's own file)."""

    folder = render_dir(Path(args.workspace), args.job)
    paths = json.loads(sys.stdin.read() or "[]")
    inbox = folder / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    queued = []
    for entry in paths:
        path = Path(str(entry)).expanduser()
        if not path.is_absolute():
            path = inbox / path
        if not path.is_file():
            continue
        key = re.sub(r"[^A-Za-z0-9._-]", "_", path.name)
        write_json(inbox / f"{key}.frame.json", {"path": str(path), "queued_utc": utc()})
        queued.append(str(path))
    return {"ok": True, "queued": queued}


def cmd_render_end(args) -> dict:
    folder = render_dir(Path(args.workspace), args.job)
    (folder / "inbox").mkdir(parents=True, exist_ok=True)
    job = read_json(folder / "render-job.json", default={}) or {}
    # The marker names the attempt it ends, so it can never end a later one.
    write_json(_end_marker(folder), {"attempt": int(job.get("attempt") or 0), "utc": utc()})
    return {"ok": True}


def _domain_key(path: str) -> str:
    found = re.search(r"wrfout_(d\d+)", Path(path).name)
    return found.group(1) if found else ""


def prune_inbox(inbox: Path, keep: set, drawn: set) -> int:
    """Delete relayed frames in ``drawn``, except ``keep`` (each domain's newest, its context).

    Only frames already drawn go: frames keep arriving while a batch draws,
    and one fed during the batch is in the inbox but not yet drawn, so it
    must survive until the next batch draws it.  Only files directly inside
    the inbox go: a render on the forecast's own machine queues the run's
    own wrfouts by path, and those are never touched.  A relayed frame is a
    whole wrfout (40 MB at 3 km, far more at 1 km), so keeping them would
    fill the render machine's disk.
    """

    freed = 0
    root = inbox.resolve()
    kept = {str(Path(item).resolve()) for item in keep}
    finished = {str(Path(item).resolve()) for item in drawn}
    for entry in sorted(inbox.glob("*.frame.json")):
        record = read_json(entry, default={}) or {}
        target = Path(str(record.get("path") or ""))
        try:
            resolved = target.resolve()
        except OSError:
            continue
        if str(resolved) in kept:
            continue
        if resolved.exists() and str(resolved) not in finished:
            continue
        inside = resolved.parent == root
        if inside and resolved.is_file():
            freed += resolved.stat().st_size
            resolved.unlink()
        if inside or not resolved.exists():
            entry.unlink(missing_ok=True)
    return freed


def cmd_render_loop(args) -> int:
    workspace = Path(args.workspace)
    folder = render_dir(workspace, args.job)
    job_path = folder / "render-job.json"
    # Every write of the job file is under the job lock, as a start's are: a write outside it read the file, a
    # fresh start then handed this worker a new attempt, and the write put the old attempt back over it.
    with _job_lock(folder):
        job = update_json(job_path, state="waiting", pid=os.getpid(), pid_start=process_start(os.getpid()))
    env = engine_env(job.get("python"), job.get("env"))
    attempt = int(job.get("attempt") or 0)
    done = list(job.get("done") or [])
    last_by_domain: dict[str, str] = {}
    for path in done:
        last_by_domain[_domain_key(path)] = path
    idle_since = time.monotonic()
    while True:
        # A fresh start while this worker was still up hands it the new attempt with nothing drawn yet, and with
        # what that attempt asked for (its products, its engine).
        newer = _newer_attempt(job_path, attempt)
        if newer is not None:
            job = read_json(job_path, default=None) or job
            env = engine_env(job.get("python"), job.get("env"))
            attempt, done, last_by_domain = newer, [], {}
            idle_since = time.monotonic()
        # END is read before the inbox, so a frame fed just before END was
        # written is seen by this scan and drawn before the worker finishes.
        # Only this attempt's END ends it.
        marker = _ended_attempt(folder)
        ended = marker is not None and marker in (attempt, -1)
        entries = sorted((folder / "inbox").glob("*.frame.json"))
        pending = []
        for entry in entries:
            record = read_json(entry, default={}) or {}
            path = record.get("path")
            if path and path not in done and Path(path).is_file():
                pending.append(path)
        if pending:
            idle_since = time.monotonic()
            pending.sort(key=lambda item: (Path(item).name, item))
            # --series: frames are drawn as one timeline, which is what lets the
            # previous frame of each domain be context for windowed products.
            argv = [job["python"], "-m", "gpuwm", "render", *pending, "--series", "--out", job["out"]]
            if job.get("products"):
                argv += ["--products", str(job["products"])]
            if job.get("render_section"):
                argv += ["--section=" + str(job["render_section"])]
            for key in sorted({_domain_key(path) for path in pending}):
                previous = last_by_domain.get(key)
                if previous and Path(previous).is_file():
                    argv += ["--context-wrfout", previous]
            argv += list(job.get("extra_args") or [])
            with _job_lock(folder):
                if _newer_attempt(job_path, attempt) is not None:
                    continue  # a fresh start arrived since this pass read the inbox: the next pass takes it
                update_json(job_path, state="rendering", current=pending, command=" ".join(argv))
            began = time.monotonic()
            with open(folder / "render.log", "ab") as out:
                out.write(f"\n# {utc()} {' '.join(argv)}\n".encode())
                out.flush()
                code = subprocess.run(argv, cwd=str(folder), stdin=subprocess.DEVNULL, stdout=out,
                                      stderr=subprocess.STDOUT, env=env).returncode
            seconds = round(time.monotonic() - began, 1)
            with _job_lock(folder):
                if _newer_attempt(job_path, attempt) is not None:
                    # A fresh start arrived while this batch drew. The frames it queues again belong to the new
                    # attempt, which draws them from nothing, so nothing is recorded against them and none is
                    # pruned.
                    continue
                done.extend(pending)
                for path in pending:
                    last_by_domain[_domain_key(path)] = path
                batches = (read_json(job_path, default={}) or {}).get("batches") or []
                batches.append({"frames": [Path(p).name for p in pending], "exit_code": code,
                                "seconds": seconds, "ended_utc": utc()})
                update_json(job_path, done=done, rendered=len(done), batches=batches, current=[],
                            state="waiting", last_exit_code=code)
                prune_inbox(folder / "inbox", keep=set(last_by_domain.values()), drawn=set(done))
            continue
        if ended:
            with _job_lock(folder):
                if _newer_attempt(job_path, attempt) is not None:
                    continue  # a fresh start arrived during this pass: the new attempt is not over
                # Written under the lock, so a start that comes after this finds the worker done and starts
                # its own, and one that came before has already handed this worker its attempt.
                prune_inbox(folder / "inbox", keep=set(), drawn=set(done))
                update_json(job_path, state="finished", ended_utc=utc())
                return 0
        if time.monotonic() - idle_since > RENDER_IDLE_S:
            with _job_lock(folder):
                if _newer_attempt(job_path, attempt) is not None:
                    continue
                update_json(job_path, state="stopped", ended_utc=utc(),
                            message="Nothing was fed for six hours, so the render worker stopped.")
                return 0
        time.sleep(RENDER_POLL_S)


def cmd_render_list(args) -> dict:
    folder = render_dir(Path(args.workspace), args.job)
    job = read_json(folder / "render-job.json", default=None)
    if not isinstance(job, dict):
        return {"ok": False, "message": f"No render job {args.job} on this machine.", "fix": ""}
    out = Path(job["out"])
    files = []
    if out.is_dir():
        for path in sorted(out.rglob("*.png")):
            relative = path.relative_to(out)
            # Only filed pictures (<domain>/<product>/<valid-day>/<file>): the
            # renderer draws into the folder's top level first and files each
            # picture afterwards, so a top-level PNG is one still moving.
            if len(relative.parts) < 4 or any(part.startswith(".") for part in relative.parts):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append([relative.as_posix(), stat.st_size])
    # The renderer's record of where each picture sits on the map, beside the tree: without it the forecast's
    # map listed every picture drawn here and placed none of them.
    manifests = []
    for record in ("render-georef.json",):
        try:
            manifests.append([record, (out / record).stat().st_size])
        except OSError:
            pass
    job = dict(job)
    job.pop("done", None)
    return {"ok": True, "job": job, "alive": process_alive(job.get("pid"), job.get("pid_start")), "files": files,
            "manifests": manifests, "log": tail(folder / "render.log", 3000)}


# ------------------------------------------------------------------ bytes

#: The exit of a pack or unpack that stopped for a reason it states.
COPY_REFUSED = 3


class _Output:
    """stdout for the tar stream, keeping the error if the pipe itself fails (the reader went away)."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.broken = None

    def write(self, data: bytes) -> int:
        try:
            return self.stream.write(data)
        except OSError as error:
            self.broken = error
            raise

    def flush(self) -> None:
        try:
            self.stream.flush()
        except OSError as error:
            self.broken = error
            raise


def _pack_stopped(file_name: str, reason: str, missing: list[str] | None = None) -> int:
    """The last stderr line of a pack that stopped: which file (its name only) and why, as JSON.

    ``missing`` names every listed file that is not there, so a caller
    that knows files can go (a forecast's frames) can copy the rest
    without asking once per file.
    """

    answer = {"ok": False, "file": file_name, "reason": reason}
    if missing:
        answer["missing"] = missing
    sys.stderr.write(json.dumps(answer) + "\n")
    sys.stderr.flush()
    return COPY_REFUSED


def cmd_pack(args) -> int:
    """A tar of the named files (paths relative to --base, or absolute) on stdout.

    A file that is missing or cannot be read stops the copy with exit 3,
    and the last line on stderr names it and says why.  It is never
    skipped: the receiver takes a tar that ends between two files as
    complete, so a skipped file was a copy that looked finished without it.
    """

    base = Path(args.base)
    names = json.loads(sys.stdin.read() or "[]")
    entries = []
    gone = []
    for entry in names:
        path = Path(str(entry))
        source = path if path.is_absolute() else base / path
        arcname = path.name if path.is_absolute() else path.as_posix()
        if ".." in Path(arcname).parts:
            return _pack_stopped(path.name, "it is outside the folder being copied")
        if not source.is_file():
            gone.append(path.name)
            continue
        if not os.access(source, os.R_OK):
            return _pack_stopped(path.name, "it may not be read")
        entries.append((source, arcname))
    if gone:
        return _pack_stopped(gone[0], "it is not there", gone)
    out = _Output(sys.stdout.buffer)
    current = ""
    try:
        with tarfile.open(fileobj=out, mode="w|") as archive:
            for source, arcname in entries:
                current = PurePath(arcname).name
                archive.add(str(source), arcname=arcname, recursive=False)
        out.flush()
    except OSError as error:
        if out.broken is None:
            # A file vanished or could not be opened after the check above.
            if isinstance(error, FileNotFoundError):
                return _pack_stopped(current, "it is not there", [current])
            return _pack_stopped(current, error.strerror or "it could not be read")
        # The receiving side stopped reading; its own answer says why.  stdout is pointed at nothing so the
        # interpreter's last flush does not add a second error to stderr.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        sys.stderr.write(json.dumps({"ok": False, "reason": "the receiving side stopped reading"}) + "\n")
        return COPY_REFUSED
    return 0


def _discard(path) -> None:
    if path is None:
        return
    try:
        Path(path).unlink()
    except OSError:
        pass


def cmd_unpack(args) -> dict:
    """Save the tar stream on stdin under --dest.

    A copy that cannot be saved (a full disk, a folder this machine may
    not write, a folder under a plain file, a stream that ends inside a
    file) answers ok false with the reason in plain words and no path,
    removes the part it was writing, and exits 3 (see :func:`main`).
    """

    dest = Path(args.dest)
    names = []
    tmp = None
    try:
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    continue
                target = dest / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name("." + target.name + ".part")
                source = archive.extractfile(member)
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
                os.replace(tmp, target)
                tmp = None
                names.append(str(target))
    except OSError as error:
        _discard(tmp)
        return {"ok": False, "reason": error.strerror or "it could not write there", "saved": len(names)}
    except (tarfile.TarError, EOFError):
        _discard(tmp)
        return {"ok": False, "reason": "the copy arrived incomplete", "saved": len(names)}
    return {"ok": True, "files": names}


# ------------------------------------------------------------------ install

#: The extra that names a CUDA major, and so the one CuPy build it installs.
CUPY_EXTRA = re.compile(r"^gpu-cu(\d+)$")


def one_cupy_lines(py: str, extra: str) -> list[str]:
    """Install-script lines that leave one CuPy build for the extra about to be installed.

    ``cupy-cuda12x`` and ``cupy-cuda13x`` install the same ``cupy`` files
    and pip keeps both registered.  The extra follows the machine's driver,
    so after a driver moved from CUDA 12 to 13 the reused venv got the new
    build beside the old one, the install said installed, and whichever
    wrote last answered ``import cupy``.  When any CuPy other than the
    extra's is present, every CuPy is removed first (the rule install.sh
    uses).  The list is read into a variable so a pip that cannot list
    the venv stops the install (``set -e``) instead of reading as no CuPy.
    """

    match = CUPY_EXTRA.match(extra or "")
    if not match:
        return []
    wanted = f"cupy-cuda{match.group(1)}x"
    return [
        f"listed=$({py} -m pip list --format=freeze --disable-pip-version-check)",
        "have=$(printf '%s\\n' \"$listed\" | tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz' | "
        "sed -n 's/^\\(cupy\\(-cuda[0-9][0-9]*x\\)\\{0,1\\}\\) *[=@].*/\\1/p')",
        "other=0",
        f"for dist in $have; do [ \"$dist\" = {wanted} ] || other=1; done",
        "if [ \"$other\" = 1 ]; then",
        f"  echo \"this venv holds another CUDA major's CuPy ($(echo $have)); removing every CuPy build so "
        f"{wanted} installs clean\"",
        f"  {py} -m pip uninstall -y $have",
        "fi",
    ]


#: The free-threading probe: exits 0 only on a build without the GIL (PEP 703).
FT_PROBE = 'import sys, sysconfig; sys.exit(0 if sysconfig.get_config_var("Py_GIL_DISABLED") else 1)'

#: Said in the install log when the venv is made on a Python with the lock.
LOCKED_WARNING = ("WARNING: this venv runs on a Python with the interpreter lock, not free-threaded "
                  "CPython 3.14t: multi-card [devices] forecasts run about 2x slower on it (4 cards "
                  "measured 289 s per forecast hour against 130 s on 3.14t); one-card forecasts are "
                  "unaffected. To fix: put python3.14t on PATH (or install uv), remove the venv and "
                  "install again.")


def interpreter_lines(venv: str) -> list[str]:
    """Shell lines that make ``venv`` on free-threaded CPython 3.14t when one can be had.

    THE BREAKAGE THIS PREVENTS: the rank threads of a multi-card forecast take
    turns on the interpreter lock, so a machine installed from the Machines page
    ran multi-card forecasts about 2x slower than the published benchmarks
    (every one ran 3.14t).  The order is a python3.14t on PATH, then the one
    uv finds or installs, then python3 with a warning in the install log.
    Sets ``made_ft=1`` when the venv this run made is free-threaded.

    A 3.14t that cannot make a venv (a distro build without its venv package
    has no ensurepip) gives way to python3 instead of ending the install
    under ``set -e``, and any half-made venv is removed: one left with its
    bin/python would be taken as already made by the next install, which
    then stopped at its pip upgrade.
    """
    q = shlex.quote
    could_not = ("could not make the venv (a Python without its venv package, "
                 "python3.14-venv on deadsnakes, does this); making it on python3. ")
    return [
        f"FT_PROBE={q(FT_PROBE)}",
        "made_ft=0",
        f"if [ ! -x {q(venv + '/bin/python')} ]; then",
        "  base=",
        '  if command -v python3.14t >/dev/null 2>&1 && python3.14t -c "$FT_PROBE"; then',
        "    base=python3.14t",
        "  else",
        '    for uv in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do',
        '      command -v "$uv" >/dev/null 2>&1 || continue',
        '      found=$("$uv" python find 3.14t 2>/dev/null) || {',
        '        UV_PYTHON_INSTALL_BIN=0 "$uv" python install 3.14t && found=$("$uv" python find 3.14t 2>/dev/null); } || found=',
        '      if [ -n "$found" ] && "$found" -c "$FT_PROBE"; then base=$found; fi',
        "      break",
        "    done",
        "  fi",
        '  if [ -n "$base" ]; then',
        '    echo "making the venv on free-threaded $base"',
        f'    if "$base" -m venv {q(venv)}; then',
        "      made_ft=1",
        "    else",
        f'      echo "$base "{q(could_not + LOCKED_WARNING)}',
        f"      rm -rf {q(venv)}",
        "    fi",
        "  else",
        f"    echo {q(LOCKED_WARNING)}",
        "  fi",
        '  if [ "$made_ft" != 1 ]; then',
        f"    python3 -m venv {q(venv)} || {{ rm -rf {q(venv)}; exit 1; }}",
        "  fi",
        "fi",
    ]


def cmd_install(args) -> dict:
    """Start a detached install of the given wheels into <workspace>/venv."""

    workspace = Path(args.workspace)
    folder = workspace / "install"
    folder.mkdir(parents=True, exist_ok=True)
    request = json.loads(sys.stdin.read() or "{}")
    current = read_json(folder / "install.json", default=None)
    if isinstance(current, dict) and process_alive(current.get("pid"), current.get("pid_start")):
        return {"ok": True, "install": current, "already": True}
    wheels = [os.path.expanduser(str(item)) for item in request.get("wheels") or []]
    for wheel in wheels:
        if not Path(wheel).is_file():
            return {"ok": False, "message": f"{wheel} is not on this machine.", "fix": "Copy it again."}
    venv = workspace / "venv"
    extra = request.get("extra") or ""
    main = wheels[0] + (f"[{extra}]" if extra else "")
    q = shlex.quote
    py = q(str(venv / "bin" / "python"))
    steps = [
        # --prefer-binary on a free-threaded venv: cftime 1.6.6 publishes no
        # cp314t wheel (1.6.5 does), and pip would build the newest from source.
        f"if {py} -c \"$FT_PROBE\" 2>/dev/null; then prefer=--prefer-binary; else prefer=; fi",
        f"{py} -m pip install --upgrade pip",
        f"{py} -m pip install --force-reinstall --no-deps " + " ".join(q(w) for w in wheels),
        *one_cupy_lines(py, extra),
        f"{py} -m pip install $prefer {q(main)} " + " ".join(q(w) for w in wheels[1:]),
    ]
    script = "\n".join([
        "set -e",
        f"export PIP_CACHE_DIR={q(str(folder / 'pip-cache'))} TMPDIR={q(str(folder / 'tmp'))}",
        f"mkdir -p {q(str(folder / 'tmp'))}",
        *interpreter_lines(str(venv)),
        # The steps run in their own `bash -e`: a shell testing a function's
        # status ignores `set -e` inside it, and a failed `pip list` must stop.
        f"set +e; bash -e {q(str(folder / 'install-steps.sh'))}; status=$?; set -e",
        # A dependency that will not install under 3.14t must not cost the
        # whole install: a venv this run made on 3.14t is made again on python3.
        "if [ \"$status\" -ne 0 ]; then",
        "  [ \"$made_ft\" = 1 ] || exit \"$status\"",
        f"  echo {q('a dependency did not install under 3.14t; making the venv again on python3. ' + LOCKED_WARNING)}",
        f"  rm -rf {q(str(venv))}",
        f"  python3 -m venv {q(str(venv))} || {{ rm -rf {q(str(venv))}; exit 1; }}",
        f"  bash -e {q(str(folder / 'install-steps.sh'))}",
        "fi",
    ])
    (folder / "install-steps.sh").write_text("\n".join([f"FT_PROBE={q(FT_PROBE)}", *steps]) + "\n")
    (folder / "install.sh").write_text(script + "\n")
    document = {"state": "installing", "started_utc": utc(), "wheels": wheels, "extra": extra,
                "python": str(venv / "bin" / "python"), "log": str(folder / "install.log")}
    write_json(folder / "install.json", document)
    pid = spawn([sys.executable, str(Path(__file__).resolve()), "install-run", "--workspace", str(workspace)],
                cwd=folder, log=folder / "install.log")
    document = update_json(folder / "install.json", pid=pid, pid_start=process_start(pid))
    return {"ok": True, "install": document}


def cmd_install_run(args) -> int:
    folder = Path(args.workspace) / "install"
    began = time.monotonic()
    with open(folder / "install.log", "ab") as out:
        code = subprocess.run(["bash", str(folder / "install.sh")], stdin=subprocess.DEVNULL,
                              stdout=out, stderr=subprocess.STDOUT).returncode
    update_json(folder / "install.json", state="installed" if code == 0 else "failed", exit_code=code,
                ended_utc=utc(), seconds=round(time.monotonic() - began, 1),
                log_tail=tail(folder / "install.log", 2000))
    return code


#: How long a record may say installing before its process is recorded beside it (cmd_install writes the
#: record, then starts the install, then adds the pid).
INSTALL_PID_GRACE_S = 120.0


def install_record(workspace: Path):
    """The last install's record, read as failed when it still says installing but its process is gone.

    The install writes its own end (installed or failed).  A record left at
    installing whose process no longer runs was ended from outside, by a
    restart or a kill, and read as installing for ever.
    """

    document = read_json(Path(workspace) / "install" / "install.json", default=None)
    if not isinstance(document, dict) or document.get("state") != "installing":
        return document
    if document.get("pid") is not None:
        if pid_alive(document.get("pid")):
            return document
    else:
        try:
            started = datetime.strptime(str(document.get("started_utc")), "%Y-%m-%dT%H:%M:%SZ")
            age = (datetime.now(timezone.utc) - started.replace(tzinfo=timezone.utc)).total_seconds()
        except ValueError:
            age = INSTALL_PID_GRACE_S
        if age < INSTALL_PID_GRACE_S:
            return document
    return {**document, "state": "failed", "stopped": True,
            "detail": "The install stopped before it finished."}


# ------------------------------------------------------------------ entry

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="machine_agent")
    sub = parser.add_subparsers(dest="verb", required=True)

    def verb(word: str, *extra: str):
        command = sub.add_parser(word)
        command.add_argument("--workspace", required=True)
        for flag in extra:
            command.add_argument(flag, default=None)
        return command

    verb("probe", "--python", "--owner-file")
    launch = verb("launch", "--python", "--owner-file", "--owner-tag", "--run")
    launch.set_defaults(owner_tag=None)
    verb("supervise", "--run")
    snap = verb("snapshot", "--run")
    snap.add_argument("--offset", type=int, default=0)
    verb("stop", "--run")
    verb("runs")
    verb("render-start", "--job", "--python")
    verb("render-feed", "--job")
    verb("render-end", "--job")
    verb("render-loop", "--job")
    verb("render-list", "--job")
    verb("pack", "--base")
    verb("unpack", "--dest")
    verb("install")
    verb("install-run")
    args = parser.parse_args(argv)
    for key in ("workspace", "base", "dest", "python", "owner_file"):
        if getattr(args, key, None):
            setattr(args, key, os.path.expanduser(getattr(args, key)))
    handlers = {
        "probe": cmd_probe, "launch": cmd_launch, "snapshot": cmd_snapshot, "stop": cmd_stop,
        "runs": cmd_runs, "render-start": cmd_render_start, "render-feed": cmd_render_feed,
        "render-end": cmd_render_end, "render-list": cmd_render_list, "unpack": cmd_unpack,
        "install": cmd_install,
    }
    looping = {"supervise": cmd_supervise, "render-loop": cmd_render_loop, "pack": cmd_pack,
               "install-run": cmd_install_run}
    if args.verb in looping:
        return int(looping[args.verb](args) or 0)
    try:
        document = handlers[args.verb](args)
    except SystemExit as error:
        document = {"ok": False, "message": str(error), "fix": ""}
    sys.stdout.write(json.dumps(document, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()
    # A copy that was not saved also says so in its exit, so nothing reading only the exit takes it as done.
    return COPY_REFUSED if args.verb == "unpack" and document.get("ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
