"""Fail a leg when a gate whose fixtures ship in this repository skipped.

    python -m pytest -p tools.battery.no_silent_skip <files...>

Registered by default from ``tests/conftest.py``, beside
``no_silent_deselection``, for the reason that file already states: a guard
that runs only when somebody remembers to pass ``-p`` is not a default.

THE BREAKAGE THIS PREVENTS
--------------------------
``tools/battery/no_silent_deselection.py`` counts COLLECTED items -- it says
so at ``pytest_collection_finish`` and ``tools/battery/list_census.json``
records nothing else.  A ``skipif``-marked item is collected.  So a file whose
every test skips contributes its full count to the census and clears its floor
exactly as if it had passed, and the leg is green.

Measured, at this commit:

    tools/rrtmg_wrf461_oracle/lw_gate.py:23-26
        DEFAULT_FIXDIR = os.environ.get(
            "GPUWM_RRTMG_LW_FIXTURES",
            os.path.expanduser("~/.claude/jobs/fea7a141/tmp/rrtmg_lw/fixtures"))

    tests/test_rrtmg_lw_numpy.py:38  pytestmark = pytest.mark.skipif(
                                         not os.path.isdir(DEFAULT_FIXDIR), ...)

That default is one agent job's scratch directory on one machine.  It exists
in no clone, so 100% of the RRTMG longwave bitwise gates -- the only per-routine
evidence in the tree that ``setcoef``/``taumol``/``rtrnmc`` reproduce WRF
v4.6.1 -- take their skip path everywhere, while ``tests/test_rrtmg_lw_numpy.py
:92`` says of one of them "this gate must never silently skip again".  Nothing
in the repository failed, because nothing in the repository was measuring
whether a gate RAN as opposed to whether it was COLLECTED.

WHAT THIS FILE IS
-----------------
``tools/battery/must_run_gates.txt`` names the decks whose fixtures are
COMMITTED TO THIS REPOSITORY.  For those there is no environment to blame: a
skip is coverage that left.  This plugin watches the reports the run already
produced and fails the leg when a listed gate

  * produced no reports at all though the leg named its file (collected
    nothing -- ``pytest.skip(allow_module_level=True)``, an emptied module, a
    deleted file's worth of tests);
  * ran and passed NOTHING while skipping something (entirely skipped); or
  * skipped more tests than its declared ceiling.

A gate that legitimately cannot run in a fresh clone belongs in the same
file's DECLARED-SKIPPING section, which is checked by
``tests/test_must_run_gates.py`` -- so an environmental skip is a stated fact
with a reason, rather than silence.

WHAT IT COSTS
-------------
Nothing measurable.  It reads reports the run already produced, runs no test,
collects nothing extra and imports nothing but ``pathlib``.

LIMITS, STATED
--------------
* The "collected nothing" arm applies only to files named as command-line
  arguments, for the same reason the deselection guard's does: a directory
  argument does not say which files the operator expected to speak.  The
  entirely-skipped and ceiling arms apply wherever a listed file is collected.
* Under an explicit ``-k`` or ``--deselect``, and under ``--collect-only``,
  the operator is deliberately not running the gates, so nothing is checked.
* A leg that already has a failing test is already red; this guard exists to
  stop a GREEN leg hiding a skip, so it says nothing when ``testsfailed``.
"""

from __future__ import annotations

import os
import pathlib
from collections import Counter

MANIFEST_RELATIVE = "tools/battery/must_run_gates.txt"

#: Everything after this line in the manifest is the DECLARED-SKIPPING half:
#: gates that cannot run in a fresh clone, each with the reason.  They are
#: recorded so that "this gate skipped" is a statement somebody wrote rather
#: than a silence; this plugin enforces nothing about them.
SECTION_MARKER = "DECLARED-SKIPPING-BEGIN"

#: An entry may write ``skips<=N`` for a gate that skips N tests by design, or
#: ``skips<=*`` for one whose skip count is a property of the fixture data
#: rather than a constant.  A bare entry means zero.
_CEILING_PREFIX = "skips<="
_UNBOUNDED = "*"


def _rel(path: str, root: pathlib.Path) -> str:
    try:
        return pathlib.Path(path).resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return pathlib.Path(path).as_posix()


def parse_manifest(text: str) -> dict[str, tuple[int | None, str]]:
    """``{path: (ceiling, comment)}``; ``None`` is an unbounded skip count.

    Split out of the plugin so ``tests/test_must_run_gates.py`` reads the
    manifest through the same parser the leg does, rather than through a
    second one that can disagree with it.  The trailing ``#`` comment is
    returned because an unbounded entry is required to carry one.
    """

    gates: dict[str, tuple[int | None, str]] = {}
    for raw in text.splitlines():
        if SECTION_MARKER in raw:
            break
        body, _, comment = raw.partition("#")
        line = body.strip()
        if not line:
            continue
        fields = line.split()
        name = fields[0]
        path = pathlib.PurePosixPath(name)
        if (path.is_absolute() or pathlib.PureWindowsPath(name).drive
                or "\\" in name or path.as_posix() != name
                or ".." in path.parts or not name.endswith(".py")):
            raise ValueError(f"invalid required gate path: {name!r}")
        if name in gates:
            raise ValueError(f"duplicate required gate: {name}")
        ceiling: int | None = 0
        if len(fields) > 1:
            if len(fields) != 2 or not fields[1].startswith(_CEILING_PREFIX):
                raise ValueError(f"invalid skip ceiling: {line!r}")
            value = fields[1][len(_CEILING_PREFIX):]
            if value != _UNBOUNDED and not value.isdigit():
                raise ValueError(f"invalid skip ceiling: {line!r}")
            ceiling = None if value == _UNBOUNDED else int(value)
            if ceiling is None and not comment.strip():
                raise ValueError(f"unbounded skip ceiling requires a reason: {name}")
        gates[name] = (ceiling, comment.strip())
    return gates


def parse_declared_skipping(text: str) -> dict[str, str]:
    """``{path: reason}`` for the half of the manifest below the marker.

    An entry here is a claim that the gate cannot run in a fresh clone, and
    the rest of the line is the reason it cannot.  Continuation lines are
    ordinary ``#`` comments, so a long reason stays readable in the file
    without the parser needing to know about wrapping.
    """

    declared: dict[str, str] = {}
    seen_marker = False
    for raw in text.splitlines():
        if SECTION_MARKER in raw:
            seen_marker = True
            continue
        if not seen_marker:
            continue
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        path, _, reason = line.partition(" ")
        declared[path] = reason.strip()
    return declared


def _gates(root: pathlib.Path) -> dict[str, int | None]:
    """Load the required inventory; missing or invalid policy is a refusal."""
    path = root / MANIFEST_RELATIVE
    try:
        text = path.read_text(encoding="utf-8")
        gates = {name: ceiling for name, (ceiling, _) in parse_manifest(text).items()}
        if not gates:
            raise ValueError("no required gates were declared")
        return gates
    except (OSError, ValueError) as error:
        import pytest
        raise pytest.UsageError(f"required must-run manifest {path}: {error}") from error


class _Guard:
    def __init__(self) -> None:
        self.requested: set[str] = set()
        self.gates: dict[str, int | None] = {}
        self.passed: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self.seen: set[str] = set()
        self.root = pathlib.Path(os.getcwd()).resolve()
        self.inactive = False

    def pytest_cmdline_main(self, config) -> None:
        self.root = pathlib.Path(str(config.rootpath)).resolve()
        for arg in config.args:
            head = arg.split("::")[0]
            if head.endswith(".py") and (self.root / head).is_file():
                self.requested.add(_rel(head, self.root))
        self.inactive = bool(getattr(config.option, "keyword", "")
                             or getattr(config.option, "deselect", None)
                             or getattr(config.option, "collectonly", False))
        self.gates = {} if self.inactive else _gates(self.root)

    def pytest_runtest_logreport(self, report) -> None:
        """Outcomes, not collection.

        The deselection guard reads ``session.items`` because its question is
        what was COLLECTED.  This one's question is what RAN, and a skipped
        item is collected -- that difference is the whole defect -- so it has
        to read reports.  A skip lands at ``setup`` (a ``skipif`` mark or a
        skipping fixture) or at ``call`` (an in-body ``pytest.skip``); a pass
        is only ever a ``call``.  The node id rather than ``report.fspath``
        because a node id is rootdir-relative POSIX, which is the spelling the
        manifest and the command line are both written in.
        """

        rel = report.nodeid.split("::")[0]
        self.seen.add(rel)
        if report.skipped:
            self.skipped[rel] += 1
        elif report.passed and report.when == "call":
            self.passed[rel] += 1

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if self.inactive or not self.gates or session.testsfailed:
            return
        lines: list[str] = []

        mute = sorted(rel for rel in self.requested & set(self.gates)
                      if rel not in self.seen)
        if mute:
            lines += ["", "MUST-RUN GATE COLLECTED NOTHING -- these files are "
                          "on tools/battery/must_run_gates.txt and the leg "
                          "ran none of them:"]
            lines += [f"  {rel}" for rel in mute]

        dead = sorted(rel for rel in self.gates
                      if self.skipped[rel] and not self.passed[rel])
        if dead:
            lines += ["", "MUST-RUN GATE ENTIRELY SKIPPED -- these files ran "
                          "no test and skipped at least one:"]
            lines += [f"  {rel}: {self.skipped[rel]} skipped, 0 passed"
                      for rel in dead]

        over = []
        for rel, ceiling in sorted(self.gates.items()):
            if ceiling is None or rel in dead or self.skipped[rel] <= ceiling:
                continue
            over.append(f"  {rel}: {self.skipped[rel]} skipped, "
                        f"{ceiling} is the declared ceiling "
                        f"({self.skipped[rel] - ceiling} more)")
        if over:
            lines += ["", "MUST-RUN GATE SKIPPED ABOVE ITS CEILING -- these "
                          "files skipped more than the manifest declares:"]
            lines += over

        if not lines:
            return
        lines.append(
            "Every file on tools/battery/must_run_gates.txt reads fixtures "
            "COMMITTED TO THIS REPOSITORY, so there is no environment to "
            "blame for a skip.  Either restore what the gate reads, or -- if "
            "the gate genuinely cannot run in a fresh clone -- move it to the "
            "DECLARED-SKIPPING section of that file WITH THE REASON, so the "
            "skip is a statement instead of a silence.")
        print("\n".join(lines))
        import pytest

        session.exitstatus = int(pytest.ExitCode.TESTS_FAILED)


#: One registration per session, whatever route loaded this file -- the same
#: hazard ``no_silent_deselection`` records: ``tests/conftest.py`` loads this
#: BY PATH so the guard is default-on, and a leg may ALSO pass
#: ``-p tools.battery.no_silent_skip``.  Those are two module objects with two
#: ``pytest_configure`` functions, and the second registration raises
#: ``ValueError: Plugin name already registered`` as an INTERNALERROR that
#: takes the session down before a test runs.  The name is the identity, so
#: the name is the check.
PLUGIN_NAME = "no_silent_skip_guard"


def pytest_configure(config) -> None:
    if config.pluginmanager.hasplugin(PLUGIN_NAME):
        return
    guard = _Guard()
    guard.pytest_cmdline_main(config)
    config.pluginmanager.register(guard, PLUGIN_NAME)
