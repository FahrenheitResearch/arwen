"""Machines: this computer plus any number of SSH hosts and cloud machines.

A machine is one row of the user's machines file (``~/.gpuwm/machines.toml``,
or ``$GPUWM_MACHINES_FILE``), so adding a host is adding a row::

    [[machine]]
    name = "gpu-box"
    host = "me@gpu-box.local"
    workspace = "/data/gpuwm"
    owner_file = "/data/gpu-mutex/OWNER"

SSH is key authentication only: every call runs ``ssh -o BatchMode=yes``,
which never asks for a password, and a row has no password field to
store one in.  An unknown host key is refused (``StrictHostKeyChecking``)
rather than accepted silently, because accepting it would let another
machine answering on that address receive your forecasts.

On the machine, :mod:`gpuwm.machine_agent` does the work: this module
copies it there (named by its own hash) and calls its fixed verbs.  A
forecast started on a machine keeps a mirror folder under the runs root
(``<root>/<run>/`` with ``gui-machine.json``), which a detached follower
(``gpuwm machines follow``) keeps up to date: the event log, the
heartbeat, and the pictures a render worker draws, so the run lists in
My forecasts and its pictures appear on the page as they are written.

Render workers: a forecast on any machine can be drawn on any machine.
Frames travel through this computer (``pack`` on the source, ``unpack``
on the render machine), because this computer is the one place that
holds a key for both; a render on the forecast's own machine reads the
frames where they are.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Iterable

from .files import NAME_RE, long_path, read_json, write_json

CONFIG_ENV = "GPUWM_MACHINES_FILE"
LOCAL = "this-computer"
MACHINE = "gui-machine.json"
RENDER = "gui-render.json"
FOLLOW = "gui-follow.json"
REMOTE_MANIFEST = "remote-run-manifest.json"
DEFAULT_OWNER_TAG = "gpuwm-gui"
DEFAULT_WORKSPACE = "~/gpuwm-machine"
AGENT_SOURCE = Path(__file__).resolve().parent.parent / "machine_agent.py"
HOST_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]{0,254}$")
ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
CARDS = ("12gb", "16gb", "24gb", "32gb")
PROBE_CACHE_S = 5.0
CALL_TIMEOUT_S = 90.0
FOLLOW_TICK_S = 5.0
#: Keys a row may carry.  Anything else is refused by name, so a typo
#: (``owner-file``) is not silently a machine with no OWNER convention.
SSH_KEYS = ("name", "kind", "host", "port", "identity", "workspace", "python", "agent_python",
            "owner_file", "owner_tag", "geog_root", "data_dir", "env", "note",
            "known_hosts", "host_key_alias")
CLOUD_KEYS = ("provider", "image", "instance_type", "region", "profile", "key_name",
              "security_group", "subnet", "ssh_user", "volume_gib", "spend_cap_usd",
              "idle_stop_min", "price_per_hour_usd", "instance_id", "spent_usd",
              "started_utc", "cap_minutes_granted", "guard_minutes_charged")


class MachineError(Exception):
    """A machine call that failed; ``message`` says what, ``fix`` what to do.

    ``missing``: for a copy that stopped because listed files are not
    there, their names, so a caller whose files can go (a forecast's
    frames) can set them aside and copy the rest.
    """

    def __init__(self, message: str, fix: str = "", status: int = 409, *,
                 missing: list[str] | tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix
        self.status = status
        self.missing = tuple(missing)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_version() -> str:
    try:
        from gpuwm import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------ the table

def config_path() -> Path:
    env = os.environ.get(CONFIG_ENV)
    return Path(env).expanduser() if env else Path.home() / ".gpuwm" / "machines.toml"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, dict):
        inner = ", ".join(f"{key} = {_toml_value(item)}" for key, item in sorted(value.items()))
        return "{ " + inner + " }" if inner else "{}"
    return json.dumps(str(value))


def load_rows(path: Path | None = None) -> list[dict[str, Any]]:
    import tomllib

    path = path or config_path()
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as error:
        raise MachineError(f"The machines file {path} could not be read: {error}.",
                           "Fix the file by hand or move it aside.", 500) from error
    rows = document.get("machine") or []
    return [dict(row) for row in rows if isinstance(row, dict)]


def save_rows(rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = ["# Machines for gpuwm gui: one [[machine]] row per computer.",
              "# SSH rows use key authentication only; nothing here is a password.", ""]
    for row in rows:
        blocks.append("[[machine]]")
        for key in [*SSH_KEYS, *CLOUD_KEYS]:
            if key in row and row[key] not in (None, "", {}):
                blocks.append(f"{key} = {_toml_value(row[key])}")
        blocks.append("")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


def check_row(row: dict[str, Any], *, providers: dict[str, Any] | None = None) -> dict[str, Any]:
    """A row as it will be stored, or MachineError naming the field."""

    if any(key in row for key in ("password", "passphrase", "secret")):
        raise MachineError("Passwords are never stored for a machine.",
                           "Put your SSH key on the machine (ssh-copy-id user@host) and add it again.", 400)
    kind = str(row.get("kind") or "ssh")
    allowed = set(SSH_KEYS) | (set(CLOUD_KEYS) if kind != "ssh" else set())
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise MachineError(f"A machine row has no field {', '.join(unknown)}.",
                           f"Fields are: {', '.join(sorted(allowed))}.", 400)
    name = str(row.get("name") or "")
    if not NAME_RE.match(name) or name == LOCAL:
        raise MachineError(f"The machine name {name!r} is not allowed.",
                           "Use letters, digits, '.', '_' or '-', and not this-computer.", 400)
    out = {key: value for key, value in row.items() if value not in (None, "")}
    out["kind"] = kind
    if kind == "ssh":
        host = str(row.get("host") or "")
        if not HOST_RE.match(host):
            raise MachineError("Give the machine's host as a name, an address, or user@host.",
                               "For example me@192.0.2.10 or an alias from ~/.ssh/config.", 400)
    else:
        from .cloud import provider

        provider(kind, providers)  # refuses an unknown provider by name
        out.setdefault("provider", kind)
    if "port" in out:
        try:
            port = int(out["port"])
        except (TypeError, ValueError) as error:
            raise MachineError("port must be a number.", "", 400) from error
        if not 1 <= port <= 65535:
            raise MachineError("port must be between 1 and 65535.", "", 400)
        out["port"] = port
    if "known_hosts" in out:
        known = Path(str(out["known_hosts"])).expanduser()
        if not known.is_absolute():
            raise MachineError("known_hosts must be the full path of a file on this computer.", "", 400)
        out["known_hosts"] = str(known)
    if "host_key_alias" in out and not NAME_RE.match(str(out["host_key_alias"])):
        raise MachineError("host_key_alias must be one plain word.", "", 400)
    if "identity" in out:
        identity = Path(str(out["identity"])).expanduser()
        if not identity.is_absolute() or not identity.is_file():
            raise MachineError("identity must be the full path of an SSH key file on this computer.", "", 400)
        out["identity"] = str(identity)
    for key in ("workspace", "python", "owner_file", "geog_root", "data_dir"):
        if key in out:
            text = str(out[key])
            if not (text.startswith("/") or text.startswith("~/")) or any(ord(c) < 32 for c in text):
                raise MachineError(f"{key} must be a full path on the machine (starting with / or ~/).", "", 400)
            out[key] = text
    env = out.get("env") or {}
    if not isinstance(env, dict) or not all(ENV_KEY_RE.match(str(k)) for k in env):
        raise MachineError("env must be a table of NAME = \"value\" settings.", "", 400)
    out["env"] = {str(k): str(v) for k, v in env.items()}
    if "owner_tag" in out and not NAME_RE.match(str(out["owner_tag"])):
        raise MachineError("owner_tag must be one plain word.", "", 400)
    return out


# ------------------------------------------------------------------ transport

@dataclass
class Machine:
    row: dict[str, Any]
    _agent_sent: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return str(self.row["name"])

    @property
    def kind(self) -> str:
        return str(self.row.get("kind") or "ssh")

    @property
    def is_local(self) -> bool:
        return self.kind == "local"

    @property
    def workspace(self) -> str:
        return str(self.row.get("workspace") or DEFAULT_WORKSPACE)

    def ws(self) -> str:
        """The workspace as the agent is given it (``~`` left for the remote shell to expand)."""

        return self.workspace

    @property
    def python(self) -> str:
        if self.row.get("python"):
            return str(self.row["python"])
        if self.is_local:
            # This computer runs the gpuwm this page runs on; it has no workspace venv of its own, and the
            # workspace default below is a Linux path that no Windows computer has.
            return sys.executable
        return f"{self.workspace.rstrip('/')}/venv/bin/python"

    def ssh_argv(self, remote_command: str) -> list[str]:
        from gpuwm.remote_cli import ssh_executable

        ssh = ssh_executable()
        if ssh is None:
            raise MachineError("This computer has no OpenSSH client.",
                               "Install OpenSSH (Windows: Settings, Optional features, OpenSSH Client).", 500)
        host = str(self.row.get("host") or "")
        if not HOST_RE.match(host):
            raise MachineError(f"{self.name} has no host yet.", "Start it first.")
        argv = [ssh, "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
        if self.row.get("port"):
            argv += ["-p", str(self.row["port"])]
        if self.row.get("identity"):
            argv += ["-i", str(self.row["identity"])]
        if self.row.get("known_hosts"):
            # A cloud machine's address changes on every start; its host key
            # is checked under the instance's own name in a gpuwm-owned file.
            argv += ["-o", f"UserKnownHostsFile={self.row['known_hosts']}"]
        if self.row.get("host_key_alias"):
            argv += ["-o", f"HostKeyAlias={self.row['host_key_alias']}"]
        return argv + ["--", host, remote_command]

    def _expand(self, word: str) -> str:
        # ~ must reach the remote shell unquoted to expand there.
        if word.startswith("~/"):
            return '"$HOME"/' + shlex.quote(word[2:])
        return shlex.quote(word)

    def command(self, words: list[str]) -> str:
        return " ".join(self._expand(str(word)) for word in words)

    def popen(self, words: list[str], *, stdin: Any = subprocess.PIPE, stdout: Any = subprocess.PIPE) -> subprocess.Popen:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        if self.is_local:
            argv = [os.path.expanduser(w) if w.startswith("~/") else w for w in words]
            return subprocess.Popen(argv, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, creationflags=flags)
        return subprocess.Popen(self.ssh_argv(self.command(words)), stdin=stdin, stdout=stdout,
                                stderr=subprocess.PIPE, creationflags=flags)

    def run(self, words: list[str], *, data: bytes = b"", timeout: float = CALL_TIMEOUT_S) -> subprocess.CompletedProcess:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        argv = ([os.path.expanduser(w) if w.startswith("~/") else w for w in words] if self.is_local
                else self.ssh_argv(self.command(words)))
        try:
            return subprocess.run(argv, input=data, capture_output=True, timeout=timeout, creationflags=flags)
        except subprocess.TimeoutExpired as error:
            raise MachineError(f"{self.name} did not answer within {int(timeout)} s.",
                               "Check that the machine is on and reachable.", 504) from error

    # -------------------------------------------------------------- the agent

    def agent_path(self) -> str:
        digest = hashlib.sha256(AGENT_SOURCE.read_bytes()).hexdigest()[:12]
        return f"{self.workspace.rstrip('/')}/.agent/machine_agent-{digest}.py"

    def agent_words(self, verb: str, *args: str) -> list[str]:
        python = sys.executable if self.is_local else str(self.row.get("agent_python") or "python3")
        return [python, str(AGENT_SOURCE) if self.is_local else self.agent_path(), verb,
                "--workspace", self.ws(), *[str(arg) for arg in args]]

    def ensure_agent(self) -> None:
        if self.is_local:
            return
        target = self.agent_path()
        if target in self._agent_sent:
            return
        folder = target.rsplit("/", 1)[0]
        script = (f"test -f {self._expand(target)} && echo have || "
                  f"{{ mkdir -p {self._expand(folder)} && cat > {self._expand(target + '.part')} && "
                  f"mv {self._expand(target + '.part')} {self._expand(target)} && echo sent; }}")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            done = subprocess.run(self.ssh_argv(script), input=AGENT_SOURCE.read_bytes(), capture_output=True,
                                  timeout=CALL_TIMEOUT_S, creationflags=flags)
        except subprocess.TimeoutExpired as error:
            raise MachineError(f"{self.name} did not answer within {int(CALL_TIMEOUT_S)} s.",
                               "Check that the machine is on and reachable.", 504) from error
        if done.returncode == SSH_FAILED:
            raise ssh_failure(self, done.stderr, done.returncode)
        if done.returncode != 0:
            # SSH got through; the machine could not keep the helper in the workspace (a folder it may not
            # write, a full disk).
            raise MachineError(f"{self.name} could not keep gpuwm's helper in its workspace "
                               f"({plain_reason(done.stderr)}).",
                               f"Pick a workspace folder {self.name} can write, or free some space there, "
                               "then check again.", 502)
        self._agent_sent.add(target)

    def call(self, verb: str, *args: str, payload: Any = None, timeout: float = CALL_TIMEOUT_S) -> dict[str, Any]:
        """One agent verb; its JSON answer, or MachineError."""

        self.ensure_agent()
        data = b"" if payload is None else json.dumps(payload).encode("utf-8")
        done = self.run(self.agent_words(verb, *args), data=data, timeout=timeout)
        if done.returncode != 0:
            # The page gets plain words; the raw ones (a helper's traceback) go to the server's terminal, where
            # whoever looks after the machine can read what "gpuwm's helper stopped with an error" was.
            _log_failure(self, verb, done.returncode, done.stderr)
            raise ssh_failure(self, done.stderr, done.returncode)
        lines = [line for line in done.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
        try:
            document = json.loads(lines[-1])
        except (IndexError, ValueError) as error:
            raise MachineError(f"{self.name} answered {verb} with something that is not JSON.",
                               "Run gpuwm machines check " + self.name + " in a terminal.", 502) from error
        if not document.get("ok", True):
            raise MachineError(str(document.get("message") or f"{verb} failed on {self.name}."),
                               str(document.get("fix") or ""))
        return document


#: ssh's own exit when it could not connect or was refused; any other exit is the remote command's.
SSH_FAILED = 255
_ERRNO_RE = re.compile(r"\[Errno -?\d+\] ([^:'\"\[\]]+)")
_EXCEPTION_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit)\b")
#: A path in quotes, as coreutils prints one (straight quotes, or curly ones in a UTF-8 locale).
_QUOTES = "'\"\u2018\u2019\u201c\u201d"
_QUOTED_PATH_RE = re.compile(f"[{_QUOTES}][/~][^{_QUOTES}]*[{_QUOTES}]")


def plain_reason(stderr: bytes | str) -> str:
    """The last line of a machine's error output in plain words: an OS error's own words, never a traceback's
    class name or a path on the machine."""

    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "no answer"
    last = lines[-1]
    found = _ERRNO_RE.search(last)
    if found:
        return found.group(1).strip()
    if _EXCEPTION_RE.match(last) or last.startswith("Traceback"):
        return "gpuwm's helper stopped with an error"
    # A tool's own line naming a path ("mkdir: cannot create directory '/x': Permission denied") ends in its reason.
    if _QUOTED_PATH_RE.search(last) and ": " in last:
        return last.rsplit(": ", 1)[1]
    # The tools and shells the helper's copy runs: "mkdir: Not a directory", or a shell's own line naming a path
    # ("sh: 1: cannot create /x/y.part: Permission denied"), which keeps only its reason.
    tool, colon, rest = last.partition(": ")
    if colon and tool in ("mkdir", "cat", "mv", "sh", "bash") and rest:
        return rest.rsplit(": ", 1)[-1] if "/" in rest else rest
    return last


def ssh_failure(machine: Machine, stderr: bytes, code: int | None = None) -> MachineError:
    """Why a call to ``machine`` failed.

    Only ssh's own failures (exit 255, or no exit given) are read as SSH's.
    Any other exit is the machine's command answering, and its words say
    "Permission denied" about a folder as readily as ssh says it about a
    key: read as SSH's, a folder the machine may not write was reported as
    a refused key.
    """

    text = stderr.decode("utf-8", "replace").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else "no answer"
    if machine.is_local or (code is not None and code != SSH_FAILED):
        return MachineError(f"{machine.name}: {plain_reason(stderr)}", "", 502)
    lowered = text.lower()
    host = machine.row.get("host")
    if "host key verification failed" in lowered or "no matching host key" in lowered \
            or "remote host identification has changed" in lowered:
        return MachineError(f"{machine.name}: this computer does not know {host}'s host key.",
                            f"Run ssh {host} once in a terminal and confirm the key, then check again.", 502)
    if "permission denied" in lowered:
        return MachineError(f"{machine.name}: {host} refused this computer's SSH key.",
                            f"Put your key on it (ssh-copy-id {host}); passwords are never used.", 502)
    if "could not resolve" in lowered or "timed out" in lowered or "no route" in lowered \
            or "connection refused" in lowered:
        return MachineError(f"{machine.name}: {host} cannot be reached ({last}).",
                            "Check the address and that the machine is on.", 502)
    return MachineError(f"{machine.name}: {last}", "", 502)


# ------------------------------------------------------------------ the registry

def local_row() -> dict[str, Any]:
    return {"name": LOCAL, "kind": "local", "workspace": str(Path.home() / ".gpuwm" / "machine")}


class Registry:
    """The machines table plus a short cache of each machine's last probe."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        #: Held across every read, change and save of the file, so two
        #: threads (the parallel listing adopting a python) never race.
        self._write_lock = threading.RLock()
        self._probes: dict[str, tuple[float, dict[str, Any]]] = {}
        self._asking: set[str] = set()
        self._machines: dict[str, Machine] = {}

    def file(self) -> Path:
        return self.path or config_path()

    def rows(self) -> list[dict[str, Any]]:
        return load_rows(self.file())

    def get(self, name: str) -> Machine:
        if name in ("", LOCAL, None):
            return self._machine(local_row())
        for row in self.rows():
            if row.get("name") == name:
                return self._machine(row)
        raise MachineError(f"There is no machine called {name!r}.", "Add it on the Machines list.", 404)

    def _machine(self, row: dict[str, Any]) -> Machine:
        with self._lock:
            cached = self._machines.get(row["name"])
            if cached is None or cached.row != row:
                # A changed row can name another host (a new address, a cloud machine's new instance) under the
                # same workspace path, and that host has not been sent the agent: the next call checks again
                # instead of calling an agent that is not there.
                cached = self._machines[row["name"]] = Machine(dict(row))
            return cached

    def put(self, row: dict[str, Any]) -> None:
        with self._write_lock:
            rows = [item for item in self.rows() if item.get("name") != row["name"]]
            rows.append(row)
            save_rows(rows, self.file())

    def update(self, name: str, **fields: Any) -> dict[str, Any]:
        with self._write_lock:
            rows = self.rows()
            for row in rows:
                if row.get("name") == name:
                    row.update({k: v for k, v in fields.items()})
                    save_rows(rows, self.file())
                    return row
        raise MachineError(f"There is no machine called {name!r}.", "", 404)

    def remove(self, name: str) -> None:
        with self._write_lock:
            rows = self.rows()
            kept = [row for row in rows if row.get("name") != name]
            if len(kept) == len(rows):
                raise MachineError(f"There is no machine called {name!r}.", "", 404)
            save_rows(kept, self.file())
        with self._lock:
            self._probes.pop(name, None)

    # -------------------------------------------------------------- checks

    def probe(self, name: str, *, fresh: bool = False) -> dict[str, Any]:
        with self._lock:
            hit = self._probes.get(name)
        if hit is not None and not fresh and time.monotonic() - hit[0] < PROBE_CACHE_S:
            return hit[1]
        machine = self.get(name)
        document = check(machine)
        if document.get("adopt_python"):
            self.update(name, python=document["adopt_python"])
        with self._lock:
            self._probes[name] = (time.monotonic(), document)
        return document

    def _probe_or_offline(self, name: str, fresh: bool) -> dict[str, Any]:
        try:
            document = self.probe(name, fresh=fresh)
        except MachineError as error:
            document = {"name": name, "state": "offline", "detail": error.message, "fix": error.fix}
        with self._lock:
            self._probes[name] = (time.monotonic(), document)
        return document

    def _refresh(self, name: str) -> None:
        try:
            self._probe_or_offline(name, True)
        except Exception:  # noqa: BLE001 - the last answer stays until the next one comes
            pass
        finally:
            with self._lock:
                self._asking.discard(name)

    def listing(self, *, fresh: bool = False) -> list[dict[str, Any]]:
        """Every machine's last probe, at once.

        ``fresh`` asks each machine now and waits for the answers (the
        Check button).  Otherwise a probe older than :data:`PROBE_CACHE_S`
        is served as it is while one background call asks again, and a
        machine never asked says "checking": a page listing machines
        never waits on SSH.
        """

        names = [str(row["name"]) for row in self.rows()]
        if fresh:
            with ThreadPoolExecutor(max_workers=max(1, min(8, len(names)))) as pool:
                return list(pool.map(lambda name: self._probe_or_offline(name, True), names))
        out = []
        for name in names:
            with self._lock:
                hit = self._probes.get(name)
                stale = hit is None or time.monotonic() - hit[0] >= PROBE_CACHE_S
                if stale and name not in self._asking:
                    self._asking.add(name)
                    threading.Thread(target=self._refresh, args=(name,), daemon=True,
                                     name=f"arwen-gui-machine-{name}").start()
            out.append(hit[1] if hit is not None else {"name": name, "state": "checking", "detail": "checking"})
        return out


def card_class(total_mib: int | None) -> str | None:
    if not total_mib:
        return None
    gib = total_mib / 1024
    fitting = [c for c in CARDS if int(c[:-2]) <= gib + 0.6]
    return fitting[-1] if fitting else CARDS[0]


def check(machine: Machine) -> dict[str, Any]:
    """Reachable, cards and VRAM, gpuwm version against this one, free disk, OWNER, state."""

    row = machine.row
    base = {"name": machine.name, "kind": machine.kind, "host": row.get("host"),
            "workspace": machine.workspace, "owner_file": row.get("owner_file"),
            "checked_utc": _utc()}
    if machine.kind not in ("ssh", "local"):
        from .cloud import cloud_state

        base.update(cloud_state(row))
        if not row.get("host"):
            return base
    began = time.monotonic()
    probe = machine.call("probe", "--python", machine.python,
                         *(["--owner-file", str(row["owner_file"])] if row.get("owner_file") else []))
    base["round_trip_s"] = round(time.monotonic() - began, 2)
    devices = (probe.get("cards") or {}).get("devices") or []
    mine = local_version()
    interpreter = probe.get("gpuwm") or {}
    theirs = interpreter.get("version")
    install = probe.get("install") or None
    unready = not_ready(interpreter, install)
    adopt = None
    if theirs != mine and isinstance(install, dict) and install.get("state") == "installed" \
            and install.get("python") and install.get("python") != machine.python:
        adopt = install["python"]
    from gpuwm.doctor import _GPU_EXTRA_BY_MAJOR

    cuda_major = (probe.get("cards") or {}).get("cuda_major")
    total = max((d.get("memory_total_mib") or 0 for d in devices), default=0)
    base.update({
        "reachable": True,
        "hostname": probe.get("hostname"),
        "state": probe.get("state"),
        "detail": probe.get("detail"),
        "cards": devices,
        "card": card_class(total),
        "vram_gib": round(total / 1024, 1) if total else None,
        "cuda": (probe.get("cards") or {}).get("cuda"),
        "compute_processes": len((probe.get("cards") or {}).get("processes") or []),
        "gpuwm": probe.get("gpuwm"),
        "version_here": mine,
        "version_there": theirs,
        "version_matches": theirs == mine,
        "install_extra": _GPU_EXTRA_BY_MAJOR.get(cuda_major) if cuda_major else None,
        "install": install,
        # Why this interpreter cannot take work yet although its gpuwm answers (:func:`not_ready`), or None.
        "not_ready": unready,
        "disk_free_gib": (probe.get("disk") or {}).get("free_gib"),
        "owner_convention": bool(probe.get("owner_file_exists")),
        "owners": probe.get("owners") or [],
        "jobs": probe.get("jobs") or {},
        "adopt_python": adopt,
    })
    if theirs != mine:
        base["offer"] = {"action": "install",
                         "words": f"{machine.name} has gpuwm {theirs or 'not installed'}; this computer has {mine}. "
                                  f"Install the same {mine} wheel there."}
    elif unready and unready["state"] == "incomplete":
        base["offer"] = {"action": "install", "words": f"gpuwm {theirs} on {machine.name} is {unready['label']}. "
                                                       f"Install the same {mine} wheel there again."}
    # The row is where an install's end is read ("the machine's row shows the result"), so one still going, or
    # one that did not finish where it left something to put right, is said there beside what the machine is
    # doing, whether or not the version it installs already answers.
    said = install_words(install)
    if said and (theirs != mine or unready or install.get("state") == "installing"):
        base["detail"] = f"{base['detail']}; {said}" if base.get("detail") else said
    return base


def not_ready(interpreter: dict[str, Any], install: Any) -> dict[str, Any] | None:
    """Why the machine's interpreter cannot take a forecast or a draw although its gpuwm imports, or None.

    An install puts gpuwm in before its requirements (``machine_agent.cmd_install``), so the version answers
    while pip is still at work, and an install that stopped between the two leaves numpy and the rest missing: a
    draw accepted then failed with "No module named 'numpy'".  Only an install into this interpreter counts; a
    machine configured with a Python of its own keeps working while its workspace venv is installed.
    """

    python = interpreter.get("python")
    if isinstance(install, dict) and install.get("state") == "installing" and python and install.get("python") \
            and posixpath.normpath(str(install["python"])) == posixpath.normpath(str(python)):
        return {"state": "installing", "label": "still installing its requirements"}
    missing = [str(name) for name in interpreter.get("missing") or []]
    if interpreter.get("version") and missing:
        return {"state": "incomplete", "missing": missing, "label": f"missing {', '.join(missing)}"}
    return None


def require_ready(name: str, report: dict[str, Any]) -> None:
    """Refuse work on a machine whose interpreter :func:`not_ready` names, before anything is sent to it.

    The breakage it prevents: the work is accepted, then fails there with "No module named 'numpy'".
    """

    unready = report.get("not_ready")
    if not unready:
        return
    if unready.get("state") == "installing":
        raise MachineError(f"{name} is still installing gpuwm's requirements, so it cannot run this yet.",
                           "Wait for the install to finish (its Machines row says when), then try again.", 409)
    raise MachineError(f"gpuwm on {name} is {unready.get('label')}, so it cannot run this.",
                       f"Use Install on {name}'s Machines row, then try again.", 409)


def install_words(install: Any) -> str:
    """The machine's last install in a few words, while it runs or when it did not finish; else ''."""

    if not isinstance(install, dict):
        return ""
    wheels = install.get("wheels") or []
    parts = wheel_name(wheels[0]).split("-") if wheels else []
    what = f"gpuwm {parts[1]}" if len(parts) > 2 else "gpuwm"
    if install.get("state") == "installing":
        return f"installing {what}"
    if install.get("state") == "failed":
        return f"the install of {what} did not finish"
    return ""


# ------------------------------------------------------------------ wheels

def wheel_name(path: str) -> str:
    """A wheel's file name from its full path, whichever separator the path uses.

    A folder listed on Windows gives a path with backslashes; one
    listed over SSH gives ``/srv/wheels/gpuwm-2.8.0-...whl``.
    """

    return re.split(r"[\\/]", str(path))[-1]


def find_wheels(folder: Iterable[str], version: str) -> list[str]:
    """The gpuwm Linux wheel and its gpuwm-data companion of exactly ``version``.

    Names are matched on each path's file name; the full paths come back,
    since those are what the copy reads.
    """

    names = list(folder)
    main = [n for n in names if re.match(rf"^gpuwm-{re.escape(version)}-.*linux.*x86_64\.whl$", wheel_name(n))]
    data = [n for n in names if re.match(rf"^gpuwm_data-{re.escape(version)}-py3-none-any\.whl$", wheel_name(n))]
    if not main:
        raise MachineError(f"No gpuwm {version} Linux wheel is in that folder.",
                           f"Build or download gpuwm-{version} for Linux; the versions must match, or a "
                           "plan made here would be read by different code there.")
    if not data:
        raise MachineError(f"No gpuwm_data {version} wheel is in that folder.",
                           "It ships beside the gpuwm wheel; the engine refuses a mismatched pair.")
    return [max(main, key=wheel_name), max(data, key=wheel_name)]


def install(registry: Registry, name: str, *, wheelhouse: str, from_machine: str | None = None,
            dry: bool = False) -> dict[str, Any]:
    """Copy this version's wheels to the machine and start a detached install into its workspace."""

    machine = registry.get(name)
    if machine.is_local:
        raise MachineError("This computer already runs this gpuwm.", "", 400)
    version = local_version()
    probe = registry.probe(name, fresh=True)
    extra = probe.get("install_extra")
    if not extra:
        raise MachineError(f"{name} reports no CUDA driver, so there is no CuPy wheel to pick.",
                           "Install the NVIDIA driver there first.")
    source = registry.get(from_machine) if from_machine else registry.get(LOCAL)
    if source.is_local:
        folder = Path(wheelhouse).expanduser()
        listing = [str(p) for p in folder.glob("*.whl")] if folder.is_dir() else []
    else:
        done = source.run(["sh", "-c", f"ls -1 {source._expand(wheelhouse)}/*.whl"], timeout=30)
        if done.returncode == SSH_FAILED:
            raise ssh_failure(source, done.stderr, done.returncode)
        # One path per line: a folder whose name has a space is one path, not two.
        listing = [line for line in done.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    wheels = find_wheels(listing, version)
    dest = f"{machine.workspace.rstrip('/')}/install/wheels"
    if source.name == machine.name:
        remote_wheels = list(wheels)
    else:
        remote_wheels = [f"{dest}/{wheel_name(w)}" for w in wheels]
    plan = {"machine": name, "version": version, "extra": extra, "wheels": wheels,
            "copy_to": dest, "venv": f"{machine.workspace.rstrip('/')}/venv",
            "from": source.name}
    if dry:
        return {"ok": True, "dry_run": True, **plan,
                "message": "Nothing ran. This copies the two wheels and installs them into the machine's own venv."}
    began = time.monotonic()
    moved = 0
    if source.name != machine.name:
        moved = relay(source, machine, wheels, dest)
    answer = machine.call("install", payload={"wheels": remote_wheels, "extra": extra}, timeout=120)
    return {"ok": True, **plan, "bytes_copied": moved, "copy_seconds": round(time.monotonic() - began, 1),
            "install": answer.get("install"),
            "message": f"Installing gpuwm {version} on {name}. It takes a few minutes; the machine's row "
                       "shows the result."}


# ------------------------------------------------------------------ moving files

def relay(source: Machine, dest: Machine, paths: list[str], dest_dir: str, *, base: str = "/") -> int:
    """Copy files from one machine to another through this computer; returns bytes moved.

    ``paths`` are absolute on the source (or relative to ``base``); they
    land in ``dest_dir`` under their own names (or relative paths).
    """

    if not paths:
        return 0
    if source.is_local:
        # A relative path is under ``base`` and keeps its folders (a render worker's <domain>/<product>/<day>
        # tree); an absolute one lands under its own name.  Both ends on this computer is a plain copy: this
        # computer drawing a forecast of its own used to tar relative paths from the wrong folder.
        entries = [(Path(entry), Path(entry).name) if Path(entry).is_absolute()
                   else (Path(base) / PurePosixPath(entry), PurePosixPath(entry).as_posix()) for entry in paths]
        # Every file that is not there is named at once, as the machine side does, before anything is sent.
        gone = [PurePosixPath(arcname).name for path, arcname in entries if not long_path(path).is_file()]
        if gone:
            raise MachineError(f"{gone[0]} could not be read on this computer, so the copy to {_where(dest)} "
                               "stopped.", "Check that the file is still there and readable, then try again.",
                               missing=gone)
        if dest.is_local:
            import shutil

            moved = 0
            for path, arcname in entries:
                target = Path(dest_dir) / PurePosixPath(arcname)
                if long_path(target).exists() and long_path(target).resolve() == long_path(path).resolve():
                    continue
                long_path(target.parent).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(long_path(path), long_path(target))
                moved += long_path(target).stat().st_size
            return moved
        dest.ensure_agent()
        receiver = dest.popen(dest.agent_words("unpack", "--dest", dest_dir), stdin=subprocess.PIPE)
        count = _CountingWriter(receiver.stdin)
        arcname = ""
        try:
            with tarfile.open(fileobj=count, mode="w|") as archive:
                for path, arcname in entries:
                    archive.add(str(long_path(path)), arcname=arcname, recursive=False)
        except OSError as error:
            if count.broken is None:
                # A file on this computer could not be read.  The receiver has a tar that simply stops, which
                # it would take as complete, so it is ended here and the file is named instead.
                receiver.kill()
                _finish(receiver, dest, 60)
                name = PurePosixPath(arcname).name
                raise MachineError(f"{name} could not be read on this computer, so the "
                                   f"copy to {dest.name} stopped.",
                                   "Check that the file is still there and readable, then try again.",
                                   missing=[name] if isinstance(error, FileNotFoundError) else []) from error
            # Otherwise the receiver stopped reading (a closed pipe: EPIPE, or EINVAL on Windows); its exit says why.
        out, err = _finish(receiver, dest, 3600)
        if receiver.returncode != 0:
            raise receiver_failure(dest, receiver.returncode, out, err)
        return count.count
    source.ensure_agent()
    dest.ensure_agent()
    sender = source.popen(source.agent_words("pack", "--base", base), stdin=subprocess.PIPE)
    try:
        sender.stdin.write(json.dumps(list(paths)).encode())
    except OSError:
        pass  # the sender ended early; its exit status says why
    _close(sender)
    if dest.is_local:
        # The sender's words are read beside its tar stream, so a sender that says a lot cannot stall on a
        # full pipe while this side waits for the tar to end.
        errors = _drain_later(sender.stderr)
        try:
            written = extract(sender.stdout, Path(dest_dir))
        except tarfile.TarError:
            written = -1
        except OSError as error:
            # This computer could not save what arrived (a full disk, a folder it may not write).  The sender
            # is ended and reaped rather than left writing into a pipe nobody reads.
            sender.kill()
            _finish(sender, source, 60, stderr_taken=True)
            raise MachineError(f"This computer could not save the files from {source.name}: "
                               f"{error.strerror or error}.",
                               "Free some space or pick a folder this computer can write, then try again.",
                               500) from error
        _finish(sender, source, 600, stderr_taken=True)
        if sender.returncode != 0:
            raise sender_failure(source, dest, sender.returncode, errors())
        if written < 0:
            raise MachineError(f"The copy from {source.name} arrived incomplete.",
                               f"Check that {source.name} is still reachable, then try again.", 502)
        return written
    count_box = [0]
    receiver_stopped = [False]
    receiver = dest.popen(dest.agent_words("unpack", "--dest", dest_dir), stdin=subprocess.PIPE)
    sender_errors = _drain_later(sender.stderr)

    def pump() -> None:
        try:
            while True:
                block = sender.stdout.read(1024 * 1024)
                if not block:
                    break
                count_box[0] += len(block)
                receiver.stdin.write(block)
        except (OSError, ValueError):
            receiver_stopped[0] = True  # the receiver stopped reading; its answer says why
        finally:
            _close(receiver)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    thread.join(timeout=3600)
    if thread.is_alive():
        for proc in (sender, receiver):
            proc.kill()
        thread.join(timeout=60)
        _finish(sender, source, 60, stderr_taken=True)
        _finish(receiver, dest, 60)
        raise MachineError(f"Copying from {source.name} to {dest.name} did not finish within an hour.",
                           "Check that both machines are on and reachable, then try again.", 504)
    if receiver_stopped[0]:
        # The receiver stopped, so the rest of the sender's stream has nowhere to go: the sender is ended
        # rather than read into memory to its end, and the receiver's reason is the one given.
        sender.kill()
    _finish(sender, source, 600, stderr_taken=True)
    out, err = _finish(receiver, dest, 600)
    if receiver_stopped[0] and receiver.returncode != 0:
        raise receiver_failure(dest, receiver.returncode, out, err)
    if sender.returncode != 0:
        raise sender_failure(source, dest, sender.returncode, sender_errors())
    if receiver.returncode != 0:
        raise receiver_failure(dest, receiver.returncode, out, err)
    return count_box[0]


def _agent_answer(text: bytes) -> dict[str, Any]:
    """The agent's own JSON on the last line of ``text``, or {} when it gave none (a crash, ssh's words)."""

    lines = [line for line in text.decode("utf-8", "replace").splitlines() if line.strip()]
    try:
        document = json.loads(lines[-1]) if lines else {}
    except ValueError:
        return {}
    return document if isinstance(document, dict) else {}


def _where(machine: Machine) -> str:
    return "this computer" if machine.is_local else machine.name


def receiver_failure(dest: Machine, code: int, out: bytes, err: bytes) -> MachineError:
    """Why ``dest`` did not save a copy, in plain words.

    Only ssh's own exit (255) is read as an SSH problem.  Otherwise the
    machine's unpack answered: its reason (a full disk, a folder it may
    not write, a folder under a plain file) is its OS error's words with no
    path.  A crash is said without its traceback, which names classes and
    the machine's paths.
    """

    if code == SSH_FAILED and not dest.is_local:
        return ssh_failure(dest, err, code)
    reason = str(_agent_answer(out).get("reason") or "").strip()
    fix = f"Free some space on {dest.name} or pick a folder it can write, then try again."
    if reason:
        return MachineError(f"{dest.name} could not save the files: {reason}.", fix, 502)
    _log_failure(dest, "unpack", code, err)
    return MachineError(f"{dest.name} stopped before the files were saved.", fix, 502)


def sender_failure(source: Machine, dest: Machine, code: int, err: bytes) -> MachineError:
    """Why ``source`` did not send a copy, in plain words; a file it could not read is named, never skipped."""

    if code == SSH_FAILED and not source.is_local:
        return ssh_failure(source, err, code)
    answer = _agent_answer(err)
    if answer.get("file"):
        missing = [str(name) for name in answer.get("missing") or []]
        return MachineError(f"{answer['file']} could not be read on {source.name}, so the copy to "
                            f"{_where(dest)} stopped.",
                            "Check that the file is still there and readable, then try again.", 502,
                            missing=missing)
    _log_failure(source, "pack", code, err)
    return MachineError(f"{source.name} stopped before it sent the files to {_where(dest)}.",
                        f"Run gpuwm machines check {source.name} in a terminal, then try again.", 502)


def _log_failure(machine: Machine, verb: str, code: int, err: bytes) -> None:
    """A copy step's raw words for whoever reads the server's terminal; the page gets plain words only."""

    tail = err.decode("utf-8", "replace").strip()[-600:]
    print(f"gpuwm: {verb} on {machine.name} ended with exit {code}" + (f":\n{tail}" if tail else ""),
          file=sys.stderr, flush=True)


def _close(proc: subprocess.Popen) -> None:
    """Close a child's stdin once and forget it, so nothing later flushes the closed stream."""

    stream, proc.stdin = proc.stdin, None
    if stream is not None:
        try:
            stream.close()
        except OSError:
            pass


def _drain_later(stream: Any):
    """Read ``stream`` to its end on a thread; the returned call waits for it and gives the bytes."""

    box: list[bytes] = []

    def read() -> None:
        try:
            box.append(stream.read() if stream is not None else b"")
        except (OSError, ValueError):
            box.append(b"")

    thread = threading.Thread(target=read, daemon=True)
    thread.start()

    def result() -> bytes:
        thread.join(timeout=60)
        return box[0] if box else b""

    return result


def _finish(proc: subprocess.Popen, machine: Machine, timeout: float, *,
            stderr_taken: bool = False) -> tuple[bytes, bytes]:
    """Wait for a child whose stdin this side has finished with; what is left of its stdout, and its stderr.

    ``communicate()`` flushes a stdin it still holds, and Python 3.12 and
    older raise "flush of closed file" when this side closed it first, after
    the files had already arrived: stdin is closed and detached here before
    it runs.  ``stderr_taken`` leaves stderr to the thread already reading
    it.  A child that outlives ``timeout`` is ended and reaped.
    """

    _close(proc)
    if stderr_taken:
        proc.stderr = None
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        proc.kill()
        proc.communicate()
        raise MachineError(f"{machine.name} did not finish the copy within {int(timeout)} s.",
                           "Check that the machine is on and reachable, then try again.", 504) from error
    return out or b"", err or b""


class _CountingWriter:
    """The receiver's stdin, counting what goes in and keeping the error if the pipe itself fails.

    ``broken`` tells a receiver that stopped reading apart from a file on
    this side that could not be read: both reach the tar writer as OSError.
    """

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.count = 0
        self.broken: OSError | None = None

    def write(self, data: bytes) -> int:
        try:
            self.stream.write(data)
        except OSError as error:
            self.broken = error
            raise
        self.count += len(data)
        return len(data)

    def flush(self) -> None:
        try:
            self.stream.flush()
        except OSError as error:
            self.broken = error
            raise


def extract(stream: Any, dest: Path) -> int:
    """Unpack a tar stream of plain files into ``dest``; returns bytes written."""

    written = 0
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or member.name.startswith("/") or ".." in parts \
                    or any(part.startswith(".") for part in parts):
                continue
            target = dest.joinpath(*parts)
            long_path(target.parent).mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            source = archive.extractfile(member)
            with open(long_path(tmp), "wb") as out:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
            os.replace(long_path(tmp), long_path(target))
    return written


__all__ = ["CONFIG_ENV", "LOCAL", "MACHINE", "Machine", "MachineError", "RENDER", "Registry",
           "card_class", "check", "check_row", "config_path", "extract", "find_wheels", "install",
           "load_rows", "local_version", "not_ready", "relay", "require_ready", "save_rows", "wheel_name"]
