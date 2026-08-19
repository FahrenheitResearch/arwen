"""Record what a user actually experiences, one real command at a time.

A release is judged by what happens on someone else's machine: the
commands they type in the order they type them, how long each one sits
there saying nothing, which one exits 1 on a fresh box, and how long it
is before a picture appears.  None of that is visible from inside the
source tree, and none of it survives being remembered afterwards.  What
usually gets written down instead is a summary written by the person
who already knows the answer.

This harness removes the remembering.  It takes a *walk script* -- an
ordered list of shell commands with a plain-language intent on each one
-- runs them for real in a working directory you choose, and records,
per step: the exact argv, the full stdout and stderr streamed to files,
the exit code, the wall time, when output first appeared, the longest
stretch of silence, and every file created or modified under the walk
root.  Then it renders one self-contained HTML page and a terse Markdown
summary so the experience can be *seen* rather than argued about.

Nothing here fakes, mocks, replays or reorders anything.  The only
numbers in the report are measurements of the commands as executed.  A
step that fails is recorded as failed; a step declared ``expect_failure``
is recorded as an expected failure and the walk carries on; a step
declared ``expect_failure`` that *succeeds* is friction, because the
walk's premise about the product is out of date.

The harness is generic on purpose.  It knows nothing about any product,
any case, any persona.  A persona is a walk script -- a fresh user on
Windows, an experienced researcher on Linux, someone upgrading from an
older release are three files, not three code paths.

Usage::

    python -m tools.walk_capture run WALK.toml --out CAPTURE_DIR
    python -m tools.walk_capture report CAPTURE_DIR --html PAGE.html
    python -m tools.walk_capture validate WALK.toml

``docs/dev/walk-capture.md`` documents the walk-script format;
``tests/test_walk_capture.py`` holds the promises this module makes.
"""

from __future__ import annotations

import argparse
import html
import json
import locale
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - the fallback exists only below 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

SCHEMA_VERSION = 1

#: Suffixes counted as a rendered plot for the time-to-first-plot metric.
DEFAULT_IMAGE_SUFFIXES = (".png",)

#: Directory names whose image files are package payload, never product
#: output.  A walk that installs into a venv under its own root unpacks
#: thousands of shipped PNGs -- matplotlib's toolbar icons, scipy's
#: test-fixture images -- and every one of them lands under one of these.
#: Counting them turned a walk that rendered nothing into a reported
#: "time to first plot: 13.4 s".
DEFAULT_PLOT_SKIP_DIRS = ("site-packages", "dist-packages")

#: Directory names never walked when tracking file activity.  Neither
#: carries user-visible product output and both churn on every step.
DEFAULT_SKIP_DIRS = (".git", "__pycache__")

DEFAULT_SKIP_SUFFIXES = (".pyc", ".pyo")

#: A snapshot bigger than this stops being a measurement and starts
#: being the measurement's own cost.  Exceeding it is reported, not
#: refused: a walk that installs into a venv under its own root
#: legitimately creates tens of thousands of files.
DEFAULT_MAX_TRACKED_FILES = 60000

_TRACEBACK_MARKERS = (
    "Traceback (most recent call last)",
)
_EXCEPTION_LINE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Exit)\s*:", re.M)


class WalkRefusal(Exception):
    """A refusal that names the concrete breakage it prevents.

    Every raise site spells out what would go wrong if the harness went
    ahead, because a refusal that only says "invalid" teaches the author
    of the walk script nothing.
    """


# --------------------------------------------------------------------------
# command tokenizing
# --------------------------------------------------------------------------

def tokenize_command(command: str) -> list[str]:
    """Split a command line into argv without letting a shell near it.

    The rule is small enough to hold in your head and identical on every
    platform: whitespace separates arguments, double quotes group, and
    every other character -- backslash included -- is literal.  That last
    clause is the point.  ``shlex.split`` in POSIX mode eats the
    backslashes out of ``C:\\walks\\case``, silently turning a Windows
    path into a different one; and handing the line to ``cmd.exe``
    instead would give ``&``, ``|``, ``^`` and ``>`` inside a directory
    name their shell meanings.  Neither happens here: the tokens go
    straight to :func:`subprocess.Popen` as a list, so a path containing
    ``a & b`` arrives at the child as ``a & b``.

    A step needing a literal double quote inside an argument uses the
    ``argv`` form instead, which skips this function entirely.
    """

    tokens: list[str] = []
    current: list[str] = []
    in_quote = False
    started = False
    for ch in command:
        if ch == '"':
            in_quote = not in_quote
            started = True
            continue
        if not in_quote and ch in " \t\r\n":
            if started:
                tokens.append("".join(current))
                current = []
                started = False
            continue
        current.append(ch)
        started = True
    if in_quote:
        raise WalkRefusal(
            "unbalanced double quote in command {!r}: the rest of the line "
            "would silently collapse into one argument, so the step would "
            "run a command you did not write".format(command))
    if started:
        tokens.append("".join(current))
    if not tokens:
        raise WalkRefusal(
            "empty command: a step with nothing to run cannot be a "
            "measurement of anything")
    return tokens


_CMD_METACHARACTERS = frozenset('&|<>^"')


def refuse_batch_metacharacters(argv: Sequence[str]) -> None:
    """Refuse a ``.bat``/``.cmd`` target carrying shell metacharacters.

    Windows has no way to execute a batch file except by handing a
    command *line* to ``cmd.exe``, which re-parses it.  So for this one
    target class the argv list is not the boundary it is everywhere
    else, and an argument containing ``&`` -- a directory named
    ``a & b``, a password, a pasted URL -- executes as a second command.
    The harness does not have a safe quoting for that, so it declines
    rather than running something the walk script does not say.
    """

    if not argv:
        return
    target = argv[0].lower()
    if not (target.endswith(".bat") or target.endswith(".cmd")):
        return
    for arg in argv[1:]:
        bad = sorted(_CMD_METACHARACTERS.intersection(arg))
        if bad:
            raise WalkRefusal(
                "argument {!r} carries {} and the target {!r} is a batch "
                "file: Windows runs batch files through cmd.exe, which "
                "re-parses the line, so that character would execute as "
                "shell syntax instead of arriving as text.  Point the step "
                "at the real executable, or set shell = true and own the "
                "quoting deliberately.".format(
                    arg, " ".join(repr(c) for c in bad), argv[0]))


# --------------------------------------------------------------------------
# walk script model
# --------------------------------------------------------------------------

@dataclass
class Settings:
    slow_seconds: float = 20.0
    quiet_seconds: float = 15.0
    timeout_seconds: float = 900.0
    stop_on_unexpected_failure: bool = True
    output_bytes_in_report: int = 20000
    image_suffixes: tuple[str, ...] = DEFAULT_IMAGE_SUFFIXES
    track_files: bool = True
    max_tracked_files: int = DEFAULT_MAX_TRACKED_FILES
    skip_dirs: tuple[str, ...] = DEFAULT_SKIP_DIRS
    skip_suffixes: tuple[str, ...] = DEFAULT_SKIP_SUFFIXES
    plot_skip_dirs: tuple[str, ...] = DEFAULT_PLOT_SKIP_DIRS


@dataclass
class Step:
    index: int
    step_id: str
    intent: str = ""
    command: str = ""
    argv: list[str] = field(default_factory=list)
    shell: bool = False
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)
    expect_failure: bool = False
    expect_exit_code: int | None = None
    allow_failure: bool = False
    timeout_seconds: float | None = None
    slow_seconds: float | None = None
    tags: list[str] = field(default_factory=list)
    kind: str = "command"  # or "bootstrap"


@dataclass
class VenvSpec:
    path: str
    system_site_packages: bool = False
    recreate: bool = True
    intent: str = ""


@dataclass
class Walk:
    name: str
    description: str = ""
    persona: str = ""
    script_path: Path | None = None
    root: Path = field(default_factory=Path)
    settings: Settings = field(default_factory=Settings)
    env: dict[str, str] = field(default_factory=dict)
    env_remove: list[str] = field(default_factory=list)
    venv: VenvSpec | None = None
    steps: list[Step] = field(default_factory=list)


_STEP_KEYS = {
    "id", "intent", "command", "argv", "shell", "cwd", "env",
    "expect_failure", "expect_exit_code", "allow_failure",
    "timeout_seconds", "slow_seconds", "tags",
}
_WALK_KEYS = {
    "name", "description", "persona", "root", "settings", "env",
    "env_remove", "venv", "steps",
}
_SETTINGS_KEYS = {f.name for f in Settings.__dataclass_fields__.values()}
_VENV_KEYS = {"path", "system_site_packages", "recreate", "intent"}


def _refuse_unknown(kind: str, where: str, given: Iterable[str],
                    known: Iterable[str]) -> None:
    extra = sorted(set(given) - set(known))
    if extra:
        raise WalkRefusal(
            "{} {} has unknown key(s) {}: a misspelled key is silently "
            "ignored by a permissive loader, so the walk would run without "
            "the setting you wrote and the report would describe a run you "
            "did not ask for.  Known keys: {}".format(
                kind, where, ", ".join(repr(e) for e in extra),
                ", ".join(sorted(known))))


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise WalkRefusal(
            "{} must be a string, got {!r}".format(where, value))
    return value


def _expand(text: str, mapping: dict[str, str]) -> str:
    """Substitute ``{token}`` placeholders literally.

    ``str.format`` is not used on purpose: command lines carry JSON,
    f-string-looking text and shell braces, and a formatter would either
    raise on them or eat them.
    """

    for key, value in mapping.items():
        text = text.replace("{" + key + "}", value)
    return text


def load_walk(path: str | os.PathLike[str]) -> Walk:
    """Parse a walk script from JSON or TOML into a :class:`Walk`."""

    script = Path(path).resolve()
    if not script.is_file():
        raise WalkRefusal(
            "walk script {} does not exist: with nothing to run there is no "
            "experience to capture".format(script))
    raw_text = script.read_text(encoding="utf-8")
    suffix = script.suffix.lower()
    if suffix == ".json":
        document = json.loads(raw_text)
    elif suffix == ".toml":
        if tomllib is None:  # pragma: no cover
            raise WalkRefusal(
                "TOML walk scripts need Python 3.11+ (tomllib); this "
                "interpreter is {}".format(platform.python_version()))
        document = tomllib.loads(raw_text)
    else:
        raise WalkRefusal(
            "walk script {} has suffix {!r}: the loader picks the parser by "
            "suffix, so an unrecognised one would be parsed by guesswork "
            "and a mis-parsed walk runs the wrong commands.  Use .json or "
            ".toml".format(script, script.suffix))
    if not isinstance(document, dict):
        raise WalkRefusal(
            "walk script {} is a {}, not a table/object: there is nowhere "
            "for the step list to live".format(
                script, type(document).__name__))
    return build_walk(document, script_path=script)


def build_walk(document: dict[str, Any], *,
               script_path: Path | None = None) -> Walk:
    _refuse_unknown("walk script", str(script_path or "<document>"),
                    document.keys(), _WALK_KEYS)

    name = _as_str(document.get("name", ""), "name")
    if not name:
        raise WalkRefusal(
            "walk script has no `name`: the report and the capture "
            "directory are both keyed on it, so two unnamed walks would "
            "overwrite each other's evidence")

    settings_doc = document.get("settings", {}) or {}
    if not isinstance(settings_doc, dict):
        raise WalkRefusal("[settings] must be a table")
    _refuse_unknown("[settings] in", str(script_path or "<document>"),
                    settings_doc.keys(), _SETTINGS_KEYS)
    settings = Settings()
    for key, value in settings_doc.items():
        if key in ("image_suffixes", "skip_dirs", "skip_suffixes",
                   "plot_skip_dirs"):
            value = tuple(str(item) for item in value)
        setattr(settings, key, value)

    base_dir = script_path.parent if script_path else Path.cwd()
    tokens = {
        "walk_dir": str(base_dir),
        "python": sys.executable,
        "repo_root": str(Path(__file__).resolve().parents[1]),
        "home": str(Path.home()),
    }
    root_text = _expand(_as_str(document.get("root", "."), "root"), tokens)
    root = Path(root_text)
    if not root.is_absolute():
        root = base_dir / root
    root = Path(os.path.normpath(str(root)))
    tokens["root"] = str(root)

    env_doc = document.get("env", {}) or {}
    if not isinstance(env_doc, dict):
        raise WalkRefusal("[env] must be a table of NAME = \"value\"")
    walk_env = {str(k): _expand(_as_str(v, "env." + str(k)), tokens)
                for k, v in env_doc.items()}

    env_remove = [str(item) for item in document.get("env_remove", []) or []]

    venv_doc = document.get("venv") or None
    venv: VenvSpec | None = None
    if venv_doc is not None:
        if not isinstance(venv_doc, dict):
            raise WalkRefusal("[venv] must be a table")
        _refuse_unknown("[venv] in", str(script_path or "<document>"),
                        venv_doc.keys(), _VENV_KEYS)
        venv = VenvSpec(
            path=_expand(_as_str(venv_doc.get("path", ".venv"), "venv.path"),
                         tokens),
            system_site_packages=bool(
                venv_doc.get("system_site_packages", False)),
            recreate=bool(venv_doc.get("recreate", True)),
            intent=_as_str(venv_doc.get("intent", ""), "venv.intent"),
        )

    steps_doc = document.get("steps", []) or []
    if not isinstance(steps_doc, list) or not steps_doc:
        raise WalkRefusal(
            "walk script has no `steps`: a capture with no commands "
            "measures nothing and would render an empty report that reads "
            "like a clean run")

    steps: list[Step] = []
    seen_ids: set[str] = set()
    for position, step_doc in enumerate(steps_doc, start=1):
        if not isinstance(step_doc, dict):
            raise WalkRefusal(
                "step {} is a {}, not a table".format(
                    position, type(step_doc).__name__))
        _refuse_unknown("step {}".format(position),
                        str(script_path or "<document>"),
                        step_doc.keys(), _STEP_KEYS)
        command = _as_str(step_doc.get("command", ""), "command")
        argv_doc = step_doc.get("argv") or []
        if command and argv_doc:
            raise WalkRefusal(
                "step {} sets both `command` and `argv`: only one of them "
                "can be what actually runs, so the report would name a "
                "command line the child never saw".format(position))
        if not command and not argv_doc:
            raise WalkRefusal(
                "step {} has neither `command` nor `argv`: there is nothing "
                "to execute".format(position))
        argv = [_expand(_as_str(item, "argv item"), tokens)
                for item in argv_doc]
        command = _expand(command, tokens)
        step_id = _as_str(step_doc.get("id", ""), "id") or "step-{:02d}".format(
            position)
        if step_id in seen_ids:
            raise WalkRefusal(
                "duplicate step id {!r}: per-step log files are named by id, "
                "so the second step would overwrite the first one's "
                "output".format(step_id))
        seen_ids.add(step_id)
        step_env_doc = step_doc.get("env", {}) or {}
        if not isinstance(step_env_doc, dict):
            raise WalkRefusal(
                "step {} `env` must be a table".format(position))
        expect_exit = step_doc.get("expect_exit_code")
        if expect_exit is not None and not isinstance(expect_exit, int):
            raise WalkRefusal(
                "step {} `expect_exit_code` must be an integer".format(
                    position))
        steps.append(Step(
            index=len(steps) + 1,
            step_id=step_id,
            intent=_as_str(step_doc.get("intent", ""), "intent"),
            command=command,
            argv=argv,
            shell=bool(step_doc.get("shell", False)),
            cwd=_expand(_as_str(step_doc.get("cwd", "."), "cwd"), tokens),
            env={str(k): _expand(_as_str(v, "step env"), tokens)
                 for k, v in step_env_doc.items()},
            expect_failure=bool(step_doc.get("expect_failure", False)),
            expect_exit_code=expect_exit,
            allow_failure=bool(step_doc.get("allow_failure", False)),
            timeout_seconds=step_doc.get("timeout_seconds"),
            slow_seconds=step_doc.get("slow_seconds"),
            tags=[str(t) for t in step_doc.get("tags", []) or []],
        ))

    return Walk(
        name=name,
        description=_as_str(document.get("description", ""), "description"),
        persona=_as_str(document.get("persona", ""), "persona"),
        script_path=script_path,
        root=root,
        settings=settings,
        env=walk_env,
        env_remove=env_remove,
        venv=venv,
        steps=steps,
    )


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        fallback = locale.getpreferredencoding(False) or "latin-1"
        return raw.decode(fallback, errors="replace")


class _Pump:
    """Read one child stream to a file while timestamping the arrivals.

    The timestamps are the whole reason this is not
    ``subprocess.run(capture_output=True)``: "printed nothing for
    forty seconds" is a fact about *when* bytes arrived, and a buffered
    capture throws that away.
    """

    def __init__(self, stream, sink_path: Path, head_limit: int,
                 tail_limit: int, clock) -> None:
        self.stream = stream
        self.sink_path = sink_path
        self.head_limit = max(head_limit, 0)
        self.tail_limit = max(tail_limit, 0)
        self.clock = clock
        self.total = 0
        self.events: list[float] = []
        self._head = bytearray()
        self._tail = bytearray()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)

    def _run(self) -> None:
        with open(self.sink_path, "wb") as sink:
            while True:
                try:
                    chunk = self.stream.read(4096)
                except (OSError, ValueError):
                    # The pipe was closed under us -- which happens when a
                    # grandchild outlives the killed child and the reader
                    # is being torn down.  Everything read so far is
                    # already on disk; a traceback here would be noise in
                    # the middle of someone else's report.
                    break
                if not chunk:
                    break
                self.events.append(self.clock())
                self.total += len(chunk)
                sink.write(chunk)
                sink.flush()
                room = self.head_limit - len(self._head)
                if room > 0:
                    self._head.extend(chunk[:room])
                self._tail.extend(chunk)
                if len(self._tail) > self.tail_limit:
                    del self._tail[:len(self._tail) - self.tail_limit]

    @property
    def truncated(self) -> bool:
        return self.total > self.head_limit + self.tail_limit

    def excerpt(self) -> str:
        """The head, the tail, and an honest count of what is missing.

        Below the limit the two buffers overlap and are stitched back
        into the exact stream, so a short run's report shows every byte
        the user saw.  Above it the middle is dropped with the omitted
        byte count named, and the per-step log file still holds all of
        it -- the report is bounded, the evidence is not.
        """

        head = bytes(self._head)
        tail = bytes(self._tail)
        if self.total <= self.head_limit:
            return _decode(head)
        tail_start = self.total - len(tail)
        if not self.truncated:
            return _decode(head + tail[self.head_limit - tail_start:])
        omitted = tail_start - self.head_limit
        return (_decode(head)
                + "\n\n... [{} bytes omitted from the middle -- the full "
                  "stream is in the per-step log file] ...\n\n".format(omitted)
                + _decode(tail))


def _snapshot(root: Path, settings: Settings) -> tuple[dict[str, tuple[int, int]], bool]:
    """Map every tracked file under ``root`` to ``(mtime_ns, size)``."""

    found: dict[str, tuple[int, int]] = {}
    truncated = False
    if not root.is_dir():
        return found, truncated
    skip_dirs = set(settings.skip_dirs)
    skip_suffixes = tuple(settings.skip_suffixes)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            if filename.endswith(skip_suffixes):
                continue
            full = Path(dirpath) / filename
            try:
                stat = full.stat()
            except OSError:
                continue
            try:
                relative = full.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - defensive
                relative = str(full)
            found[relative] = (stat.st_mtime_ns, stat.st_size)
            if len(found) >= settings.max_tracked_files:
                return found, True
    return found, truncated


def is_installed_package_payload(root: Path, relative: str,
                                 skip_dirs: Iterable[str],
                                 venv_cache: dict[str, bool] | None = None,
                                 ) -> bool:
    """True when ``relative`` is a file an installer put there.

    Two rules, both keyed on what is on disk rather than on a name we
    hope nobody uses for their own output:

    * a ``site-packages``/``dist-packages`` directory anywhere above the
      file -- that is where wheels are unpacked, including by
      ``pip install --target`` outside any environment;
    * a ``pyvenv.cfg`` in any ancestor directory -- the file that
      *defines* a virtual environment, so everything beneath it belongs
      to the interpreter and not to the walk.

    The walk still tracks these files; only the plot metric ignores
    them.  A report that answers "time to first plot" with the moment
    pip unpacked matplotlib's toolbar icons is a confident wrong number,
    and nobody re-walks a walk by hand to catch it.
    """

    names = {name.lower() for name in skip_dirs}
    parts = relative.replace("\\", "/").split("/")[:-1]
    if any(part.lower() in names for part in parts):
        return True
    if venv_cache is None:
        venv_cache = {}
    for depth in range(1, len(parts) + 1):
        prefix = "/".join(parts[:depth])
        marked = venv_cache.get(prefix)
        if marked is None:
            marked = root.joinpath(*parts[:depth], "pyvenv.cfg").is_file()
            venv_cache[prefix] = marked
        if marked:
            return True
    return False


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False, timeout=30)
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            pass
    try:
        proc.kill()
    except OSError:  # pragma: no cover
        pass


@dataclass
class StepResult:
    record: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.record["status"] in ("ok", "expected-failure")


def _step_status(step: Step, exit_code: int | None, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if exit_code is None:  # pragma: no cover - defensive
        return "failed"
    if step.expect_exit_code is not None:
        return "ok" if exit_code == step.expect_exit_code else "failed"
    if step.expect_failure:
        return "expected-failure" if exit_code != 0 else "failed"
    return "ok" if exit_code == 0 else "failed"


def _flags_for(step: Step, record: dict[str, Any],
               settings: Settings) -> list[dict[str, str]]:
    """Plain-language friction, derived only from what was measured."""

    flags: list[dict[str, str]] = []
    slow_seconds = (step.slow_seconds if step.slow_seconds is not None
                    else settings.slow_seconds)
    if record["timed_out"]:
        flags.append({
            "code": "timeout",
            "label": "timed out",
            "detail": "The command was still running after {:.0f} s and was "
                      "killed.  A user watching this has no way to tell a "
                      "hung command from a slow one.".format(
                          record["seconds"]),
        })
    if record["status"] == "failed":
        if step.expect_failure and record["exit_code"] == 0:
            flags.append({
                "code": "unexpected-success",
                "label": "was supposed to fail",
                "detail": "This step is declared as an expected failure and "
                          "it succeeded.  The walk's description of the "
                          "product is out of date.",
            })
        else:
            flags.append({
                "code": "nonzero",
                "label": "exit {}".format(record["exit_code"]),
                "detail": "The user's command failed here.  Everything below "
                          "this line is what they see after a failure, not "
                          "after a success.",
            })
    if record["seconds"] >= slow_seconds:
        flags.append({
            "code": "slow",
            "label": "{:.1f} s".format(record["seconds"]),
            "detail": "Over the {:.0f} s threshold for this walk.  This is "
                      "time the user spends waiting at a prompt.".format(
                          slow_seconds),
        })
    stderr_text = record["stderr_excerpt"]
    if any(marker in stderr_text for marker in _TRACEBACK_MARKERS) or \
            _EXCEPTION_LINE.search(stderr_text):
        flags.append({
            "code": "traceback",
            "label": "traceback on stderr",
            "detail": "The user was shown a Python traceback.  Whatever the "
                      "underlying problem is, the message they got is "
                      "addressed to us, not to them.",
        })
    silence = record["max_silence_seconds"]
    if silence >= settings.quiet_seconds:
        if record["stdout_bytes"] == 0 and record["stderr_bytes"] == 0:
            flags.append({
                "code": "silent",
                "label": "{:.0f} s with no output at all".format(silence),
                "detail": "Nothing was printed for the whole step.  The user "
                          "cannot tell it is working.",
            })
        else:
            flags.append({
                "code": "silent",
                "label": "{:.0f} s gap in output".format(silence),
                "detail": "The longest stretch where the terminal showed "
                          "nothing new.  The user cannot tell it is working.",
            })
    return flags


def _run_one(step: Step, *, root: Path, env: dict[str, str],
             settings: Settings, capture_dir: Path,
             walk_started: float) -> StepResult:
    step_dir = capture_dir / "steps" / "{:02d}-{}".format(
        step.index, re.sub(r"[^A-Za-z0-9_.-]", "_", step.step_id))
    step_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = step_dir / "stdout.txt"
    stderr_path = step_dir / "stderr.txt"

    if step.argv:
        argv = list(step.argv)
    elif step.shell:
        argv = []
    else:
        argv = tokenize_command(step.command)
    if argv and not step.shell:
        refuse_batch_metacharacters(argv)

    cwd = Path(step.cwd)
    if not cwd.is_absolute():
        cwd = root / step.cwd
    cwd = Path(os.path.normpath(str(cwd)))
    if not cwd.is_dir():
        raise WalkRefusal(
            "step {!r} names working directory {} which does not exist: the "
            "command would run somewhere else and the report would attribute "
            "its output to a directory it never saw".format(
                step.step_id, cwd))

    step_env = dict(env)
    step_env.update(step.env)

    # Windows resolves an executable name against the *parent* process's
    # PATH, not the one handed to CreateProcess, so a step that follows a
    # fresh-venv bootstrap would silently run the outer interpreter and
    # the whole capture would describe the wrong install.  Resolving a
    # bare name here -- against the PATH this step actually declares --
    # is what makes the bootstrap mean anything.
    if argv and not step.shell and not os.path.dirname(argv[0]):
        resolved = shutil.which(argv[0], path=step_env.get("PATH"))
        if resolved:
            argv = [resolved] + list(argv[1:])

    timeout = (step.timeout_seconds if step.timeout_seconds is not None
               else settings.timeout_seconds)
    head_limit = max(settings.output_bytes_in_report // 2, 512)
    tail_limit = head_limit

    before, before_truncated = (
        _snapshot(root, settings) if settings.track_files else ({}, False))

    started_wall = time.time()
    started = time.monotonic()

    def clock() -> float:
        return time.monotonic() - started

    popen_target: Any = step.command if step.shell else argv
    try:
        proc = subprocess.Popen(
            popen_target,
            cwd=str(cwd),
            env=step_env,
            shell=step.shell,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        seconds = time.monotonic() - started
        stdout_path.write_bytes(b"")
        stderr_path.write_text(str(exc), encoding="utf-8")
        record = _base_record(
            step, argv, cwd, root, started_wall, started - walk_started)
        record.update({
            "seconds": seconds,
            "exit_code": None,
            "timed_out": False,
            "launch_error": str(exc),
            "status": "failed",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_bytes": 0,
            "stderr_bytes": len(str(exc)),
            "stdout_excerpt": "",
            "stderr_excerpt": str(exc),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "first_output_seconds": None,
            "max_silence_seconds": seconds,
            "files_created": [],
            "files_modified": [],
            "files_created_count": 0,
            "files_modified_count": 0,
            "tracking_truncated": before_truncated,
            "plots": [],
            "installed_images_skipped": 0,
        })
        record["flags"] = [{
            "code": "not-found",
            "label": "command did not start",
            "detail": "The user typed this and the operating system could "
                      "not run it: {}".format(exc),
        }]
        return StepResult(record)

    out_pump = _Pump(proc.stdout, stdout_path, head_limit, tail_limit, clock)
    err_pump = _Pump(proc.stderr, stderr_path, head_limit, tail_limit, clock)
    out_pump.start()
    err_pump.start()

    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            exit_code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover
            exit_code = None
    seconds = time.monotonic() - started
    out_pump.join(30)
    err_pump.join(30)
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:  # pragma: no cover
            pass

    # The pumps are joined after the process wall time is taken, so an
    # arrival can be stamped a few milliseconds past the end.  Clamping
    # keeps every gap non-negative; without it a late arrival makes one
    # interval negative and the "longest silence" answer depends on
    # thread scheduling.
    events = sorted(min(max(event, 0.0), seconds)
                    for event in out_pump.events + err_pump.events)
    first_output = events[0] if events else None
    marks = [0.0] + events + [seconds]
    max_silence = max(b - a for a, b in zip(marks, marks[1:]))

    after, after_truncated = (
        _snapshot(root, settings) if settings.track_files else ({}, False))
    created = sorted(set(after) - set(before))
    modified = sorted(p for p in set(after) & set(before)
                      if after[p] != before[p])

    step_start_offset = started - walk_started
    step_end_offset = step_start_offset + seconds
    walk_start_wall = walk_started_wall_offset(walk_started, started,
                                               started_wall)
    plots = []
    skipped_payload = 0
    venv_cache: dict[str, bool] = {}
    image_suffixes = tuple(s.lower() for s in settings.image_suffixes)
    for relative in created:
        if not relative.lower().endswith(image_suffixes):
            continue
        if is_installed_package_payload(root, relative,
                                        settings.plot_skip_dirs, venv_cache):
            skipped_payload += 1
            continue
        # The file's mtime is wall clock and everything else here is
        # monotonic, so pair the two and then hold the answer inside the
        # step that created the file: a copied-in file can carry an
        # mtime from last year, and a plot cannot honestly be reported
        # as having appeared before the command that wrote it started.
        when = after[relative][0] / 1e9 - walk_start_wall
        plots.append({
            "path": relative,
            "seconds": max(step_start_offset, min(when, step_end_offset)),
        })

    record = _base_record(step, argv, cwd, root, started_wall,
                          started - walk_started)
    record.update({
        "seconds": seconds,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "status": _step_status(step, exit_code, timed_out),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_bytes": out_pump.total,
        "stderr_bytes": err_pump.total,
        "stdout_excerpt": out_pump.excerpt(),
        "stderr_excerpt": err_pump.excerpt(),
        "stdout_truncated": out_pump.truncated,
        "stderr_truncated": err_pump.truncated,
        "first_output_seconds": first_output,
        "max_silence_seconds": max_silence,
        "files_created": created[:200],
        "files_modified": modified[:200],
        "files_created_count": len(created),
        "files_modified_count": len(modified),
        "tracking_truncated": before_truncated or after_truncated,
        "plots": plots,
        "installed_images_skipped": skipped_payload,
    })
    record["flags"] = _flags_for(step, record, settings)
    return StepResult(record)


def walk_started_wall_offset(walk_started_monotonic: float,
                             step_started_monotonic: float,
                             step_started_wall: float) -> float:
    """Wall-clock instant, in seconds, of the walk's start.

    File mtimes are wall clock; every other timing here is monotonic.
    Pairing the two clocks once per step -- rather than sampling
    ``time.time()`` at the start of the walk and trusting it for an hour
    -- keeps a plot's offset correct even if the system clock is
    adjusted mid-walk.
    """

    return step_started_wall - (step_started_monotonic
                                - walk_started_monotonic)


def _base_record(step: Step, argv: Sequence[str], cwd: Path, root: Path,
                 started_wall: float, offset: float) -> dict[str, Any]:
    try:
        shown_cwd = str(cwd.relative_to(root))
    except ValueError:
        shown_cwd = str(cwd)
    return {
        "index": step.index,
        "id": step.step_id,
        "kind": step.kind,
        "intent": step.intent,
        "tags": list(step.tags),
        "command": step.command or subprocess.list2cmdline(list(argv)),
        "argv": list(argv),
        "shell": step.shell,
        "cwd": shown_cwd or ".",
        "env_overrides": dict(step.env),
        "expect_failure": step.expect_failure,
        "expect_exit_code": step.expect_exit_code,
        "allow_failure": step.allow_failure,
        "started_utc": datetime.fromtimestamp(
            started_wall, timezone.utc).isoformat(timespec="seconds"),
        "offset_seconds": offset,
    }


def _bootstrap_step(venv: VenvSpec, root: Path) -> Step:
    target = Path(venv.path)
    if not target.is_absolute():
        target = root / venv.path
    argv = [sys.executable, "-m", "venv"]
    if venv.system_site_packages:
        argv.append("--system-site-packages")
    if venv.recreate:
        argv.append("--clear")
    argv.append(str(target))
    return Step(
        index=1,
        step_id="bootstrap-venv",
        intent=venv.intent or (
            "Start from a clean interpreter, the way someone who has never "
            "installed this before does."),
        argv=argv,
        kind="bootstrap",
        tags=["bootstrap"],
    )


def _apply_venv(env: dict[str, str], venv: VenvSpec, root: Path) -> Path:
    target = Path(venv.path)
    if not target.is_absolute():
        target = root / venv.path
    bin_dir = target / ("Scripts" if os.name == "nt" else "bin")
    env["VIRTUAL_ENV"] = str(target)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    return bin_dir


def run_walk(walk: Walk, capture_dir: str | os.PathLike[str], *,
             label: str = "") -> dict[str, Any]:
    """Execute every step of ``walk`` for real and return the capture."""

    capture = Path(capture_dir).resolve()
    capture.mkdir(parents=True, exist_ok=True)
    walk.root.mkdir(parents=True, exist_ok=True)
    walk.root = walk.root.resolve()

    settings = walk.settings
    if settings.track_files:
        try:
            capture.relative_to(walk.root)
        except ValueError:
            pass
        else:
            raise WalkRefusal(
                "the capture directory {} is inside the walk root {}: every "
                "step would see the previous step's log files as newly "
                "created user output, and the file-activity column would "
                "report the harness measuring itself.  Put --out "
                "elsewhere, or set settings.track_files = false".format(
                    capture, walk.root))

    env = os.environ.copy()
    for name in walk.env_remove:
        env.pop(name, None)
    env.update(walk.env)

    steps = list(walk.steps)
    if walk.venv is not None:
        bootstrap = _bootstrap_step(walk.venv, walk.root)
        steps = [bootstrap] + steps
    for position, step in enumerate(steps, start=1):
        step.index = position

    # Tokenize and screen every step before running the first one.  A
    # walk-script authoring error found at step six would otherwise have
    # already executed five real commands, leaving side effects on the
    # machine and a half-capture that renders as an aborted walk rather
    # than as the mistake it is.
    for step in steps:
        if step.argv or step.shell:
            candidate = list(step.argv)
        else:
            candidate = tokenize_command(step.command)
        if candidate and not step.shell:
            refuse_batch_metacharacters(candidate)

    started_wall = time.time()
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    notes: list[str] = []
    aborted = False

    for step in steps:
        if aborted:
            record = _base_record(step, step.argv, walk.root, walk.root,
                                  time.time(), time.monotonic() - started)
            record.update({
                "seconds": 0.0, "exit_code": None, "timed_out": False,
                "status": "skipped", "stdout_path": "", "stderr_path": "",
                "stdout_bytes": 0, "stderr_bytes": 0,
                "stdout_excerpt": "", "stderr_excerpt": "",
                "stdout_truncated": False, "stderr_truncated": False,
                "first_output_seconds": None, "max_silence_seconds": 0.0,
                "files_created": [], "files_modified": [],
                "files_created_count": 0, "files_modified_count": 0,
                "tracking_truncated": False, "plots": [],
                "installed_images_skipped": 0, "flags": [{
                    "code": "skipped",
                    "label": "never reached",
                    "detail": "The walk stopped before this step, so the "
                              "user never got here.",
                }],
            })
            records.append(record)
            continue

        result = _run_one(step, root=walk.root, env=env, settings=settings,
                          capture_dir=capture, walk_started=started)
        records.append(result.record)

        if step.kind == "bootstrap" and walk.venv is not None:
            if result.record["status"] == "ok":
                bin_dir = _apply_venv(env, walk.venv, walk.root)
                notes.append(
                    "every step after the bootstrap runs with {} first on "
                    "PATH".format(bin_dir))
            else:
                aborted = True
                notes.append(
                    "the fresh-environment bootstrap failed, so every later "
                    "step would have run against the wrong interpreter and "
                    "measured the wrong install; the walk stopped here")
                continue

        if not result.ok and not step.allow_failure and \
                settings.stop_on_unexpected_failure:
            aborted = True
            notes.append(
                "step {!r} failed and settings.stop_on_unexpected_failure is "
                "on, so the remaining steps were not run".format(step.step_id))

    total_seconds = time.monotonic() - started
    plots: list[dict[str, Any]] = []
    for record in records:
        for plot in record.get("plots", []):
            plots.append({"path": plot["path"], "seconds": plot["seconds"],
                          "step": record["id"]})
    plots.sort(key=lambda item: item["seconds"])
    first_plot = plots[0]["seconds"] if plots else None
    installed_images_skipped = sum(
        record.get("installed_images_skipped", 0) for record in records)

    friction = sum(1 for record in records
                   if any(flag["code"] not in ("skipped",)
                          for flag in record["flags"]))
    failed = sum(1 for record in records if record["status"] == "failed")
    if aborted or failed:
        status = "blocked"
    elif friction:
        status = "friction"
    else:
        status = "clean"

    document = {
        "schema": SCHEMA_VERSION,
        "tool": "tools/walk_capture.py",
        "label": label,
        "walk": {
            "name": walk.name,
            "description": walk.description,
            "persona": walk.persona,
            "script": str(walk.script_path) if walk.script_path else "",
            "root": str(walk.root),
        },
        "settings": {
            "slow_seconds": settings.slow_seconds,
            "quiet_seconds": settings.quiet_seconds,
            "timeout_seconds": settings.timeout_seconds,
            "stop_on_unexpected_failure": settings.stop_on_unexpected_failure,
            "output_bytes_in_report": settings.output_bytes_in_report,
            "image_suffixes": list(settings.image_suffixes),
            "track_files": settings.track_files,
        },
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "node": platform.node(),
            "cwd": str(Path.cwd()),
        },
        "started_utc": datetime.fromtimestamp(
            started_wall, timezone.utc).isoformat(timespec="seconds"),
        "total_seconds": total_seconds,
        "status": status,
        "steps": records,
        "plots": plots,
        "time_to_first_plot_seconds": first_plot,
        "installed_images_skipped": installed_images_skipped,
        "notes": notes,
        "aborted": aborted,
    }
    (capture / "capture.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    return document


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

_STATUS_TEXT = {
    "ok": ("ok", "exit 0"),
    "expected-failure": ("expected", "failed on purpose"),
    "failed": ("failed", "unexpected failure"),
    "timeout": ("timeout", "killed after the timeout"),
    "skipped": ("skipped", "never reached"),
}

_CSS = """
:root {
  --bg: #f7f7f5; --panel: #ffffff; --ink: #1b1b1a; --muted: #5f5f5b;
  --line: #dedcd6; --ok: #1f7a4d; --warn: #a86400; --bad: #b3261e;
  --accent: #2b5f8f; --code-bg: #f2f1ec;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16171a; --panel: #1e1f23; --ink: #eceae4; --muted: #a3a29c;
    --line: #33353b; --ok: #56c98d; --warn: #e0a94a; --bad: #ef6f66;
    --accent: #7db3e0; --code-bg: #141518;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
        sans-serif;
}
main { max-width: 1080px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 40px 0 12px; letter-spacing: -0.01em; }
p.lede { color: var(--muted); margin: 0 0 6px; max-width: 70ch; }
code, pre, .mono { font-family: "Cascadia Mono", Consolas, "SF Mono",
  ui-monospace, monospace; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 24px 0 8px; }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 16px; min-width: 150px; flex: 1 1 150px;
}
.card .k { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); }
.card .v { font-size: 22px; font-weight: 600; margin-top: 2px; }
.card .n { font-size: 12px; color: var(--muted); margin-top: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); font-weight: 600; }
.scroll { overflow-x: auto; background: var(--panel);
  border: 1px solid var(--line); border-radius: 10px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 99px;
  font-size: 11px; font-weight: 600; border: 1px solid currentColor; }
.b-ok { color: var(--ok); } .b-expected { color: var(--accent); }
.b-failed, .b-timeout { color: var(--bad); } .b-skipped { color: var(--muted); }
.bar-row { display: flex; align-items: center; gap: 10px; padding: 4px 0; }
.bar-label { width: 210px; flex: none; font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; background: var(--code-bg); border-radius: 4px;
  height: 16px; position: relative; }
.bar-fill { height: 100%; border-radius: 4px; background: var(--accent);
  min-width: 2px; }
.bar-fill.f-failed, .bar-fill.f-timeout { background: var(--bad); }
.bar-fill.f-expected { background: var(--warn); }
.bar-fill.f-skipped { background: var(--line); }
.bar-time { width: 74px; flex: none; text-align: right; font-size: 12px;
  color: var(--muted); font-variant-numeric: tabular-nums; }
.step { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 16px 18px; margin: 12px 0; }
.step h3 { margin: 0 0 2px; font-size: 15px; font-weight: 600; }
.step .intent { color: var(--muted); margin: 0 0 10px; max-width: 75ch; }
.cmd { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 7px; padding: 9px 12px; font-size: 13px; overflow-x: auto;
  white-space: pre; margin: 0 0 10px; }
.meta { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px;
  color: var(--muted); margin-bottom: 8px; }
.meta b { color: var(--ink); font-weight: 600; }
details { margin-top: 8px; }
summary { cursor: pointer; font-size: 13px; color: var(--accent); }
pre.out { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 7px; padding: 10px 12px; font-size: 12.5px; overflow-x: auto;
  max-height: 460px; overflow-y: auto; margin: 8px 0 0; }
.flag { display: inline-block; margin: 2px 6px 2px 0; padding: 1px 8px;
  border-radius: 99px; font-size: 11px; font-weight: 600;
  border: 1px solid var(--warn); color: var(--warn); }
.flag.x-nonzero, .flag.x-timeout, .flag.x-not-found,
.flag.x-unexpected-success { border-color: var(--bad); color: var(--bad); }
.flag.x-skipped { border-color: var(--muted); color: var(--muted); }
footer { color: var(--muted); font-size: 12.5px; margin-top: 44px;
  border-top: 1px solid var(--line); padding-top: 14px; }
.empty { color: var(--muted); font-style: italic; }
"""


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 1:
        return "{:.2f} s".format(value)
    if value < 60:
        return "{:.1f} s".format(value)
    minutes, seconds = divmod(value, 60)
    return "{:.0f} m {:02.0f} s".format(minutes, seconds)


def _installed_images_note(document: dict[str, Any]) -> str:
    """Say out loud that images were seen and deliberately not counted.

    Without this, a walk that installed matplotlib reads as "produced no
    image files" while hundreds of PNGs sit under the walk root, and the
    next reader re-opens the same question.
    """

    skipped = document.get("installed_images_skipped", 0)
    if not skipped:
        return ""
    return ("; {} image file(s) unpacked into installed packages were not "
            "counted".format(skipped))


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def render_html(document: dict[str, Any]) -> str:
    walk = document["walk"]
    steps = document["steps"]
    settings = document["settings"]
    machine = document["machine"]

    durations = [s["seconds"] for s in steps] or [1.0]
    longest = max(max(durations), 0.001)

    friction_rows = [(s, f) for s in steps for f in s["flags"]]
    failed = [s for s in steps if s["status"] in ("failed", "timeout")]
    expected = [s for s in steps if s["status"] == "expected-failure"]
    skipped = [s for s in steps if s["status"] == "skipped"]

    out: list[str] = []
    add = out.append
    add("<title>{}</title>".format(_e(walk["name"])))
    add("<style>{}</style>".format(_CSS))
    add("<main>")
    add("<h1>{}</h1>".format(_e(walk["name"])))
    if walk["persona"]:
        add("<p class=lede><b>Who is walking:</b> {}</p>".format(
            _e(walk["persona"])))
    if walk["description"]:
        add("<p class=lede>{}</p>".format(_e(walk["description"])))
    add("<p class=lede>Every command below was executed for real, in order, "
        "on this machine. The timings, exit codes and output are "
        "measurements of that run and nothing else.</p>")

    plot_value = document.get("time_to_first_plot_seconds")
    plot_note = ("first image file written under the walk root"
                 if plot_value is not None
                 else "this walk produced no image files")
    plot_note += _installed_images_note(document)
    status_note = {
        "clean": "no step tripped a friction rule",
        "friction": "every step finished, some were painful",
        "blocked": "the user hit a wall",
    }.get(document["status"], "")
    add("<div class=cards>")
    for key, value, note in (
        ("total wall time", _fmt_seconds(document["total_seconds"]),
         "{} steps".format(len(steps))),
        ("time to first plot", _fmt_seconds(plot_value), plot_note),
        ("outcome", document["status"], status_note),
        ("friction points", str(len(friction_rows)),
         "across {} step(s)".format(
             len({s["id"] for s, _ in friction_rows}))),
        ("failures", str(len(failed)),
         "{} expected, {} never reached".format(len(expected), len(skipped))),
    ):
        add("<div class=card><div class=k>{}</div><div class=v>{}</div>"
            "<div class=n>{}</div></div>".format(_e(key), _e(value), _e(note)))
    add("</div>")

    add("<h2>Timeline</h2>")
    add("<p class=lede>Bar length is wall time. This is the shape of the "
        "wait, in the order the user experiences it.</p>")
    add("<div class=scroll style='padding:12px 16px'>")
    for step in steps:
        share = 100.0 * step["seconds"] / longest
        add("<div class=bar-row>"
            "<div class='bar-label mono' title='{title}'>{n}. {label}</div>"
            "<div class=bar-track><div class='bar-fill f-{cls}' "
            "style='width:{share:.2f}%'></div></div>"
            "<div class=bar-time>{time}</div></div>".format(
                title=_e(step["command"]), n=step["index"],
                label=_e(step["id"]), cls=_e(step["status"]),
                share=max(share, 0.5), time=_e(_fmt_seconds(step["seconds"]))))
    add("</div>")

    add("<h2>Friction</h2>")
    if friction_rows:
        add("<p class=lede>Auto-flagged from the measurements: slower than "
            "{:.0f} s, a non-zero exit, a traceback on stderr, or {:.0f} s "
            "or more with nothing printed.</p>".format(
                settings["slow_seconds"], settings["quiet_seconds"]))
        add("<div class=scroll><table><thead><tr><th>step</th><th>what</th>"
            "<th>what the user experiences</th></tr></thead><tbody>")
        for step, flag in friction_rows:
            add("<tr><td class=mono>{}. {}</td><td><span class='flag x-{}'>"
                "{}</span></td><td>{}</td></tr>".format(
                    step["index"], _e(step["id"]), _e(flag["code"]),
                    _e(flag["label"]), _e(flag["detail"])))
        add("</tbody></table></div>")
    else:
        add("<p class=empty>No step tripped a friction rule.</p>")

    add("<h2>Step by step</h2>")
    for step in steps:
        label, meaning = _STATUS_TEXT.get(
            step["status"], (step["status"], ""))
        add("<section class=step>")
        add("<h3>{}. {} <span class='badge b-{}'>{}</span></h3>".format(
            step["index"], _e(step["id"]), _e(step["status"]), _e(label)))
        if step["intent"]:
            add("<p class=intent>{}</p>".format(_e(step["intent"])))
        add("<pre class=cmd>{}</pre>".format(_e(step["command"])))
        meta = [
            "<span><b>{}</b> wall</span>".format(
                _e(_fmt_seconds(step["seconds"]))),
            "<span>exit <b>{}</b> ({})</span>".format(
                _e("--" if step["exit_code"] is None else step["exit_code"]),
                _e(meaning)),
            "<span>starts at <b>T+{}</b></span>".format(
                _e(_fmt_seconds(step["offset_seconds"]))),
            "<span>cwd <b class=mono>{}</b></span>".format(_e(step["cwd"])),
        ]
        if step["first_output_seconds"] is not None:
            meta.append("<span>first output after <b>{}</b></span>".format(
                _e(_fmt_seconds(step["first_output_seconds"]))))
        if step["files_created_count"] or step["files_modified_count"]:
            meta.append("<span><b>{}</b> file(s) created, <b>{}</b> "
                        "modified</span>".format(
                            step["files_created_count"],
                            step["files_modified_count"]))
        if step["shell"]:
            meta.append("<span><b>shell</b>: the platform shell parses this "
                        "line</span>")
        add("<div class=meta>{}</div>".format("".join(meta)))
        for flag in step["flags"]:
            add("<span class='flag x-{}'>{}</span>".format(
                _e(flag["code"]), _e(flag["label"])))
        for stream in ("stdout", "stderr"):
            text = step[stream + "_excerpt"]
            size = step[stream + "_bytes"]
            if not text.strip():
                continue
            add("<details><summary>{} ({} bytes{})</summary>"
                "<pre class=out>{}</pre></details>".format(
                    stream, size,
                    ", excerpted" if step[stream + "_truncated"] else "",
                    _e(text)))
        created = step["files_created"]
        if created:
            shown = created[:12]
            more = step["files_created_count"] - len(shown)
            add("<details><summary>files created ({})</summary>"
                "<pre class=out>{}{}</pre></details>".format(
                    step["files_created_count"], _e("\n".join(shown)),
                    _e("\n... and {} more".format(more)) if more > 0 else ""))
        add("</section>")

    if document.get("notes"):
        add("<h2>Harness notes</h2><ul>")
        for note in document["notes"]:
            add("<li>{}</li>".format(_e(note)))
        add("</ul>")

    add("<h2>Where this was captured</h2>")
    add("<div class=scroll><table><tbody>")
    for key, value in (
        ("walk script", walk["script"]),
        ("walk root", walk["root"]),
        ("machine", machine["platform"]),
        ("python", "{} ({})".format(machine["python"], machine["executable"])),
        ("started", document["started_utc"]),
        ("label", document.get("label") or "--"),
    ):
        add("<tr><th style='width:150px'>{}</th><td class=mono>{}</td>"
            "</tr>".format(_e(key), _e(value or "--")))
    add("</tbody></table></div>")
    add("<footer>Captured by <span class=mono>tools/walk_capture.py</span>. "
        "Full untruncated stdout and stderr for every step live beside "
        "<span class=mono>capture.json</span> in the capture directory."
        "</footer>")
    add("</main>")
    return "\n".join(out) + "\n"


def render_markdown(document: dict[str, Any]) -> str:
    walk = document["walk"]
    steps = document["steps"]
    lines: list[str] = []
    lines.append("# {}".format(walk["name"]))
    lines.append("")
    if walk["persona"]:
        lines.append("Who is walking: {}".format(walk["persona"]))
    if walk["description"]:
        lines.append(walk["description"])
    lines.append("")
    plot = document.get("time_to_first_plot_seconds")
    lines.append("- outcome: **{}**".format(document["status"]))
    lines.append("- total wall time: {} over {} step(s)".format(
        _fmt_seconds(document["total_seconds"]), len(steps)))
    lines.append("- time to first plot: {}{}".format(
        _fmt_seconds(plot) if plot is not None
        else "not measured (this walk produced no image files)",
        _installed_images_note(document)))
    lines.append("- captured {} on {}, python {}".format(
        document["started_utc"], document["machine"]["platform"],
        document["machine"]["python"]))
    lines.append("")
    lines.append("| # | step | exit | wall | flags |")
    lines.append("|---|------|------|------|-------|")
    for step in steps:
        flags = ", ".join(flag["label"] for flag in step["flags"]) or "--"
        lines.append("| {} | `{}` | {} | {} | {} |".format(
            step["index"], step["id"],
            "--" if step["exit_code"] is None else step["exit_code"],
            _fmt_seconds(step["seconds"]), flags))
    lines.append("")
    friction = [(s, f) for s in steps for f in s["flags"]]
    lines.append("## Friction")
    lines.append("")
    if friction:
        for step, flag in friction:
            lines.append("- **{}** ({}): {}".format(
                step["id"], flag["label"], flag["detail"]))
    else:
        lines.append("None: no step tripped a friction rule.")
    lines.append("")
    if document.get("notes"):
        lines.append("## Harness notes")
        lines.append("")
        for note in document["notes"]:
            lines.append("- {}".format(note))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(document: dict[str, Any], *, html_path: Path | None,
                  md_path: Path | None) -> list[Path]:
    written: list[Path] = []
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(document), encoding="utf-8",
                             newline="\n")
        written.append(html_path)
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(document), encoding="utf-8",
                           newline="\n")
        written.append(md_path)
    return written


def load_capture(capture_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(capture_dir)
    if path.is_dir():
        path = path / "capture.json"
    if not path.is_file():
        raise WalkRefusal(
            "no capture.json at {}: there is no recorded run to render, and "
            "rendering a report from anything else would be a report about "
            "nothing".format(path))
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA_VERSION:
        raise WalkRefusal(
            "capture {} declares schema {!r}, this harness writes and reads "
            "schema {}: the field names differ between versions, so the "
            "report would silently drop or mislabel what it shows".format(
                path, document.get("schema"), SCHEMA_VERSION))
    return document


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _default_capture_dir(walk: Walk) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", walk.name).strip("-") or "walk"
    return Path.cwd() / "walk-captures" / "{}-{}".format(safe, stamp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.walk_capture",
        description=(
            "Run a walk script for real and render what the user "
            "experienced: per-step timings, output, exit codes, file "
            "activity, an auto-flagged friction table and time to first "
            "plot."))
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", help="execute a walk script and write a capture + reports")
    run.add_argument("walk", help="walk script (.json or .toml)")
    run.add_argument("--out", help="capture directory (default: "
                                   "./walk-captures/<name>-<UTC stamp>)")
    run.add_argument("--html", help="HTML report path "
                                    "(default: <capture>/report.html)")
    run.add_argument("--md", help="Markdown summary path "
                                  "(default: <capture>/report.md)")
    run.add_argument("--root", help="override the walk root")
    run.add_argument("--label", default="",
                     help="free-text label recorded in the capture")
    run.add_argument("--slow-seconds", type=float,
                     help="override the friction threshold for slow steps")
    run.add_argument("--quiet-seconds", type=float,
                     help="override the silence threshold")
    run.add_argument("--keep-going", action="store_true",
                     help="do not stop the walk at the first unexpected "
                          "failure (records the rest of the experience)")

    report = sub.add_parser(
        "report", help="re-render the reports from an existing capture")
    report.add_argument("capture", help="capture directory or capture.json")
    report.add_argument("--html")
    report.add_argument("--md")

    validate = sub.add_parser(
        "validate", help="parse a walk script and print the plan, run nothing")
    validate.add_argument("walk")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    walk = load_walk(args.walk)
    if args.root:
        walk.root = Path(args.root).resolve()
    if args.slow_seconds is not None:
        walk.settings.slow_seconds = args.slow_seconds
    if args.quiet_seconds is not None:
        walk.settings.quiet_seconds = args.quiet_seconds
    if args.keep_going:
        walk.settings.stop_on_unexpected_failure = False

    capture_dir = Path(args.out).resolve() if args.out else _default_capture_dir(walk)
    document = run_walk(walk, capture_dir, label=args.label)
    html_path = Path(args.html) if args.html else capture_dir / "report.html"
    md_path = Path(args.md) if args.md else capture_dir / "report.md"
    written = write_reports(document, html_path=html_path, md_path=md_path)

    print("walk: {}".format(walk.name))
    print("capture: {}".format(capture_dir))
    for path in written:
        print("report: {}".format(path))
    print("outcome: {} -- {} step(s), {} total".format(
        document["status"], len(document["steps"]),
        _fmt_seconds(document["total_seconds"])))
    plot = document.get("time_to_first_plot_seconds")
    print("time to first plot: {}".format(
        _fmt_seconds(plot) if plot is not None else "no plots in this walk"))
    for step in document["steps"]:
        flags = ", ".join(flag["label"] for flag in step["flags"])
        print("  {:>2}. {:<24} {:>10}  exit {:<4} {}".format(
            step["index"], step["id"][:24],
            _fmt_seconds(step["seconds"]),
            "--" if step["exit_code"] is None else step["exit_code"],
            flags))
    return 0 if document["status"] == "clean" else 1


def _cmd_report(args: argparse.Namespace) -> int:
    document = load_capture(args.capture)
    base = Path(args.capture)
    if base.is_file():
        base = base.parent
    html_path = Path(args.html) if args.html else base / "report.html"
    md_path = Path(args.md) if args.md else base / "report.md"
    for path in write_reports(document, html_path=html_path, md_path=md_path):
        print("report: {}".format(path))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    walk = load_walk(args.walk)
    print("walk: {}".format(walk.name))
    print("root: {}".format(walk.root))
    if walk.persona:
        print("persona: {}".format(walk.persona))
    if walk.venv is not None:
        print("bootstrap: fresh venv at {}{}".format(
            walk.venv.path,
            " (system site packages)" if walk.venv.system_site_packages
            else ""))
    for step in walk.steps:
        argv = step.argv or (
            [] if step.shell else tokenize_command(step.command))
        if argv and not step.shell:
            refuse_batch_metacharacters(argv)
        shown = subprocess.list2cmdline(argv) if argv else step.command
        marks = []
        if step.expect_failure:
            marks.append("expects failure")
        if step.expect_exit_code is not None:
            marks.append("expects exit {}".format(step.expect_exit_code))
        if step.allow_failure:
            marks.append("failure does not stop the walk")
        if step.shell:
            marks.append("SHELL")
        print("  {:>2}. {:<20} {}{}".format(
            step.index, step.step_id, shown,
            "   [{}]".format("; ".join(marks)) if marks else ""))
        if step.intent:
            print("      intent: {}".format(step.intent))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "report":
            return _cmd_report(args)
        return _cmd_validate(args)
    except WalkRefusal as refusal:
        print("walk_capture refuses: {}".format(refusal), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
