#!/usr/bin/env python3
"""Run cargo-mutants over the Rust a commit actually changed.

    python tools/battery/run_mutation_gate.py --since HEAD~1

from the repository root.  ``tools/battery/mutation_gates.txt`` names the
packages and says, per package, whether a surviving mutant fails the run
or is only reported; ``tools/battery/mutation_survivors.txt`` is the debt
list of survivors that already existed when the gate was first recorded.
The runner prints a matrix and one verdict line, the same shape as
``run_cargo_gates.py``, because a leg that reports only an exit status
throws away the numbers that make a regression legible.

THE BREAKAGE THIS GATE PREVENTS
    A wrong forecast NUMBER that neither crashes nor moves any hash: an
    inverted or off-by-one comparison, or a dropped unit-conversion arm,
    in the arithmetic a published product is read off.  The SHA-256
    kernel pin proves code did not CHANGE unnoticed and says nothing
    about code that changed WRONGLY; cargo_gates.txt's MIN_TESTS floors
    prove a suite still runs its tests and say nothing about whether
    those tests would fail if the arithmetic under them were reversed.
    mutation_gates.txt carries the three measured instances that opened
    this gate.

WHY THE COST FOLLOWS THE DIFF
    The workspace carries 60,713 mutants and the cheapest measured rate
    is 1.8 s each, so a whole-workspace run is over thirty hours and must
    never be a gate.  ``--since REV`` hands cargo-mutants a git diff and
    it tests only the mutants inside changed lines, which is tens of
    mutants for a normal commit and none at all for a commit that
    touches no Rust -- 65.9% of this branch's file-touches.

WHY IT COPIES THE TREE INSTEAD OF MUTATING IT
    cargo-mutants' own copy mode is unusable in this workspace: four
    crates reach grib-core by a path that escapes the workspace root
    (``../../../grib1_bridge/vendor/grib-core``, which is deliberate --
    see the comment in tools/rustwx/Cargo.toml), the tool copies only the
    workspace directory, and the baseline build then dies in 105 ms with
    "failed to read ...\\Temp\\grib1_bridge\\vendor\\grib-core\\Cargo.toml".
    That leaves ``--in-place``, and ``--in-place`` is refused together
    with ``-j``: out of the box the tool is strictly serial here.

    So this runner does the staging itself.  It copies the workspace AND
    every path dependency that escapes it into a scratch directory,
    preserving repository-relative layout so the escaping paths resolve,
    and runs ``--in-place --shard k/n`` in N independent copies.  That
    recovers the parallelism and it also means the caller's tree is never
    mutated: an interrupted ``--in-place`` run leaves mutated source
    behind, and doing that to a shared worktree in which other lanes are
    building is the failure this staging exists to prevent.  Measured:
    310 MB and about 9 s per copy.

    Do NOT "fix" the copy-mode blocker by re-vendoring grib-core into
    tools/rustwx.  tools/rustwx/Cargo.toml records that the tree already
    had two copies and that the one which lived here was the one missing
    missing_value_management, so every renderer and ingest read decoded
    the encoder's missing-value sentinels as physical data.  That is a
    guard citing a named defect and it stays.

WHY THERE IS A DEBT LIST AND NOT A CLEAN SLATE
    The same reason ``_CRLF_DEBT`` has one.  The block packages carry
    survivors today; a gate that is red on its first run gets widened
    until it means nothing.  The debt list is recorded ONCE from a full
    run, and after that a survivor the list does not already carry is a
    fresh hole and fails the run.  Entries are keyed by package, file,
    function, mutation genre and replacement text -- deliberately NOT by
    line number, so moving a function does not silently forgive its
    survivors or manufacture new ones.

EXIT STATUS IS THE VERDICT: 0 when no fresh survivor appeared in any
block package, 1 when one did, 3 when the leg could not run at all.
Those are different answers -- a leg that could not run has proven
nothing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "mutation_gates.txt"
DEBT = REPOSITORY_ROOT / "tools" / "battery" / "mutation_survivors.txt"

#: The modes the MODE column may take.  ``block`` fails the run on a
#: fresh survivor; ``report`` counts and prints and never fails.
MODES = ("block", "report")

#: The mode a per-commit invocation runs.  Named rather than spelled
#: inline so the manifest gate and the runner cannot disagree about it.
GATE_MODE = "block"

#: How many surviving mutants a `report` package lists before the run
#: prints a count instead.
REPORT_LISTING = 25

#: Exit status when the leg could not be run at all, as distinct from a
#: leg that ran and found a fresh survivor (1).  A missing cargo-mutants
#: disproves nothing.
CANNOT_RUN = 3

#: A path dependency in a Cargo.toml, e.g. ``path = "../rustwx-core"``.
_PATH_DEP = re.compile(r'^\s*(?:[\w-]+\s*=\s*)?\{?[^#\n]*?\bpath\s*=\s*"([^"]+)"')

#: The cargo-mutants outcome summaries this runner treats as a survivor.
#: ``Unviable`` did not compile and proves nothing about the tests;
#: ``Timeout`` means the mutant hung the suite, which is a detection.
SURVIVOR_SUMMARIES = ("MissedMutant",)


@dataclass(frozen=True)
class Entry:
    """One line of tools/battery/mutation_gates.txt, parsed."""

    mode: str
    workspace: str
    package: str


@dataclass(frozen=True)
class MutantKey:
    """A mutant's identity, with the line number deliberately absent.

    Line numbers move whenever anything above them is edited, so a debt
    list keyed by them would forgive a survivor that slid down the file
    and manufacture a fresh one out of the same mutation.  Package, file,
    function, genre and replacement text survive an edit that does not
    touch the mutated expression.
    """

    package: str
    file: str
    function: str
    genre: str
    replacement: str

    def spelling(self) -> str:
        return (f"{self.package} | {self.file} | {self.function} | "
                f"{self.genre} | {json.dumps(self.replacement)}")

    @classmethod
    def parse(cls, line: str, *, where: str) -> tuple[int, "MutantKey"]:
        parts = [p.strip() for p in line.split("|", 5)]
        if len(parts) != 6:
            raise ValueError(
                f"{where}: expected COUNT | PACKAGE | FILE | FUNCTION | "
                f"GENRE | REPLACEMENT, got {line!r}")
        count, package, path, function, genre, replacement = parts
        if not count.isdigit() or int(count) < 1:
            raise ValueError(
                f"{where}: COUNT {count!r} must be a positive integer -- "
                "an entry allowing zero survivors is an entry that should "
                "have been deleted")
        try:
            text = json.loads(replacement)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{where}: REPLACEMENT {replacement!r} is not a JSON "
                f"string ({exc}); it is JSON-quoted because a replacement "
                "may contain the field separator") from None
        if not isinstance(text, str):
            raise ValueError(
                f"{where}: REPLACEMENT {replacement!r} is JSON but not a "
                "string")
        return int(count), cls(package=package, file=path, function=function,
                               genre=genre, replacement=text)

    @classmethod
    def of(cls, mutant: dict) -> "MutantKey":
        function = mutant.get("function") or {}
        return cls(
            package=mutant.get("package", "?"),
            file=str(mutant.get("file", "?")).replace("\\", "/"),
            function=function.get("function_name") or "-",
            genre=mutant.get("genre", "?"),
            replacement=mutant.get("replacement", ""))


@dataclass
class Observation:
    """What one mutant did, plus where its diff was written."""

    key: MutantKey
    name: str
    summary: str


@dataclass
class PackageResult:
    entry: Entry
    tested: int = 0
    caught: int = 0
    survived: int = 0
    unviable: int = 0
    timeout: int = 0
    fresh: list[Observation] = field(default_factory=list)


def parse_manifest(text: str) -> list[Entry]:
    """Parse the manifest exactly the way this runner consumes it.

    ``tests/test_mutation_gate_manifest.py`` imports this rather than
    restating the format, so a file the suite accepts is a file the
    runner can read.
    """

    entries: list[Entry] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"tools/battery/mutation_gates.txt line {number}: expected "
                f"MODE WORKSPACE PACKAGE, got {line!r}")
        mode, workspace, package = parts
        if mode not in MODES:
            raise ValueError(
                f"tools/battery/mutation_gates.txt line {number}: mode "
                f"{mode!r} is not one of {MODES}")
        entries.append(Entry(mode=mode, workspace=workspace,
                             package=package))
    return entries


def read_manifest(path: Path = MANIFEST) -> list[Entry]:
    return parse_manifest(path.read_text(encoding="utf-8"))


def parse_debt(text: str) -> dict[MutantKey, int]:
    """Parse the allowed-survivor list.

    ``tests/test_mutation_gate_manifest.py`` imports this too.
    """

    debt: dict[MutantKey, int] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"tools/battery/mutation_survivors.txt line {number}"
        count, key = MutantKey.parse(line, where=where)
        if key in debt:
            raise ValueError(
                f"{where}: {key.spelling()} is listed twice.  Two lines for "
                "one mutant let a widening hide inside a merge; give it one "
                "line and one count.")
        debt[key] = count
    return debt


def read_debt(path: Path = DEBT) -> dict[MutantKey, int]:
    if not path.is_file():
        return {}
    return parse_debt(path.read_text(encoding="utf-8"))


def escaping_path_dependencies(workspace: Path) -> list[Path]:
    """Path dependencies that resolve OUTSIDE the workspace directory.

    Found by reading manifests rather than by naming grib-core, so a
    second escaping dependency is staged the day it is added instead of
    failing a run that nobody can explain.
    """

    manifests = [workspace / "Cargo.toml"]
    manifests += sorted(workspace.glob("crates/*/Cargo.toml"))
    outside: dict[Path, None] = {}
    for manifest in manifests:
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            match = _PATH_DEP.match(line)
            if not match:
                continue
            resolved = (manifest.parent / match.group(1)).resolve()
            if workspace in resolved.parents or resolved == workspace:
                continue
            if REPOSITORY_ROOT not in resolved.parents:
                raise RuntimeError(
                    f"{manifest} depends on {resolved}, which is outside "
                    f"the repository at {REPOSITORY_ROOT}.  This runner "
                    "stages the tree by copying repository-relative paths "
                    "and cannot reach that.")
            outside[resolved] = None
    return list(outside)


def _long(path: Path) -> str:
    """A Windows extended-length spelling of an absolute path.

    The vendored crates-io directory carries rustls-webpki's test
    fixtures, whose names run past 90 characters; under a scratch root of
    any realistic depth the copy crosses the 260-character MAX_PATH limit
    and shutil raises WinError 3 on several dozen files.  The `\\\\?\\`
    prefix is the documented way past that, and it costs nothing on a
    platform that does not need it.
    """

    if os.name != "nt":
        return str(path)
    spelling = os.path.abspath(str(path))
    if spelling.startswith("\\\\?\\"):
        return spelling
    if spelling.startswith("\\\\"):
        return "\\\\?\\UNC" + spelling[1:]
    return "\\\\?\\" + spelling


def stage_copy(workspace: Path, extras: list[Path], destination: Path) -> Path:
    """Copy the workspace and its escaping dependencies under *destination*.

    Repository-relative layout is preserved, which is the whole point:
    ``../../../grib1_bridge/vendor/grib-core`` has to resolve from the
    copy, and it only does when the copy sits at the same depth below a
    common root.
    """

    for source in [workspace] + extras:
        relative = source.relative_to(REPOSITORY_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        # `target` is skipped ONLY at the root being copied.  A blanket
        # ignore_patterns("target") also drops
        # vendor/crates-io/cc/src/target/, which is source, and the
        # baseline build then dies on "failed to calculate checksum of
        # .../cc/src/target/llvm.rs" -- a build artefact directory and a
        # module named target are not the same thing.
        root = _long(source)

        def _skip_build_artefacts(directory: str, names: list[str]) -> set:
            if os.path.abspath(directory) == os.path.abspath(root):
                return {name for name in names if name == "target"}
            return set()

        shutil.copytree(root, _long(target), symlinks=True,
                        ignore=_skip_build_artefacts)
    return destination / workspace.relative_to(REPOSITORY_ROOT)


def write_diff(workspace: Path, since: str, destination: Path) -> int:
    """Write the workspace's Rust diff against *since*; return its size.

    ``--relative`` is not optional: cargo-mutants matches the diff's
    paths against paths relative to the workspace root it is running in,
    and a diff carrying repository-relative paths silently matches
    nothing, which reads exactly like a commit that changed no Rust.

    The diff is the WORKING TREE against a revision, never a revision
    range, because the bytes staged for mutation are the working tree's.
    A range would let the runner test one state and filter by another's
    line numbers.
    """

    relative = workspace.relative_to(REPOSITORY_ROOT).as_posix()
    argv = ["git", "diff", f"--relative={relative}", "--no-color",
            "--unified=0", since, "--",
            f":(glob){relative}/**/*.rs"]
    completed = subprocess.run(argv, cwd=REPOSITORY_ROOT, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"git diff against {since!r} failed: "
            f"{completed.stderr.strip()}")
    destination.write_text(completed.stdout, encoding="utf-8", newline="\n")
    return len(completed.stdout)


def shard_argv(packages: list[str], *, diff: Path | None, output: Path,
               shard: tuple[int, int] | None, timeout: float | None,
               cargo_mutants: str) -> list[str]:
    argv = [cargo_mutants, "mutants", "--in-place", "--no-shuffle",
            "-o", str(output)]
    for package in packages:
        argv += ["-p", package]
    if diff is not None:
        argv += ["--in-diff", str(diff)]
    if shard is not None:
        index, total = shard
        argv += ["--shard", f"{index}/{total}", "--sharding", "round-robin"]
    if timeout is not None:
        argv += ["--timeout", str(timeout)]
    return argv


def run_shard(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    # Each staged copy keeps its own target directory INSIDE itself.  The
    # cargo_gates.txt runner refuses an in-tree target because a release
    # cut hashes tools/rustwx/target/release and a test leg could replace
    # the binary the cut is adjudicating; that reasoning does not reach
    # here, because this tree is a scratch copy that is deleted at the
    # end and no cut ever looks at it.  Sharing one target directory
    # between concurrent shards, on the other hand, would serialise them
    # behind cargo's build lock and undo the parallelism.
    environment.pop("CARGO_TARGET_DIR", None)
    return subprocess.run(argv, cwd=cwd, env=environment, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)


def collect(output: Path, returncode: int) -> tuple[list[Observation],
                                                    str | None]:
    """Read one shard's outcomes.json into observations.

    cargo-mutants exits non-zero when mutants SURVIVED, which is this
    runner's business to adjudicate rather than an error.  It exits
    non-zero for a baseline that would not build too, and that is an
    error -- told apart here by ``total_mutants``, because a shard that
    tested nothing and complained is a shard that proved nothing.  The
    distinction is not cosmetic: without it a broken staged copy reports
    "0 fresh survivors" and the gate passes green on a run that never
    compiled.
    """

    path = output / "mutants.out" / "outcomes.json"
    if not path.is_file():
        if returncode == 0:
            # cargo-mutants writes no outcomes at all when the diff
            # filter leaves it nothing to do ("No mutants to filter"),
            # and exits 0.  That is the common case for a commit whose
            # only Rust is inside #[cfg(test)] or a comment, and reading
            # it as a broken run would make the gate exit 3 -- proved
            # nothing -- on the cheapest commits there are.
            return [], None
        return [], (f"{path} was not written and cargo-mutants exited "
                    f"{returncode}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if returncode != 0 and not data.get("total_mutants"):
        return [], (f"cargo-mutants exited {returncode} having tested no "
                    f"mutants; see {path.parent / 'log' / 'baseline.log'}")
    observations: list[Observation] = []
    for outcome in data.get("outcomes", []):
        scenario = outcome.get("scenario")
        if not isinstance(scenario, dict) or "Mutant" not in scenario:
            continue
        mutant = scenario["Mutant"]
        observations.append(Observation(key=MutantKey.of(mutant),
                                        name=mutant.get("name", "?"),
                                        summary=outcome.get("summary", "?")))
    return observations, None


def tally(entries: list[Entry], observations: list[Observation],
          debt: dict[MutantKey, int]) -> dict[str, PackageResult]:
    results = {e.package: PackageResult(entry=e) for e in entries}
    seen: dict[MutantKey, int] = {}
    for observation in observations:
        result = results.get(observation.key.package)
        if result is None:
            continue
        result.tested += 1
        if observation.summary == "Unviable":
            result.unviable += 1
            continue
        if observation.summary == "Timeout":
            result.timeout += 1
            continue
        if observation.summary not in SURVIVOR_SUMMARIES:
            result.caught += 1
            continue
        result.survived += 1
        seen[observation.key] = seen.get(observation.key, 0) + 1
        if seen[observation.key] > debt.get(observation.key, 0):
            result.fresh.append(observation)
    return results


def write_debt(path: Path, observations: list[Observation],
               keep: dict[MutantKey, int], covered: set[str]) -> int:
    """Rewrite the debt list for *covered* packages, keeping the rest."""

    counts: dict[MutantKey, int] = {
        key: count for key, count in keep.items()
        if key.package not in covered}
    for observation in observations:
        if observation.summary not in SURVIVOR_SUMMARIES:
            continue
        if observation.key.package not in covered:
            continue
        counts[observation.key] = counts.get(observation.key, 0) + 1
    lines = [
        "# Mutants that already survived when this gate was first recorded.",
        "#",
        "# WHAT A LINE MEANS",
        "#   tools/battery/run_mutation_gate.py tested this mutation, the",
        "#   package's whole suite still passed, and the run was allowed",
        "#   anyway because the hole predates the gate.  A survivor this",
        "#   file does not carry fails the run.  Same shape and same",
        "#   reason as _CRLF_DEBT in tests/test_line_ending_stability.py:",
        "#   a gate that is red on its first run gets widened until it",
        "#   means nothing, so the existing debt is written down once and",
        "#   nothing may be added to it by accident.",
        "#",
        "#   Every line here is a place where the crate's tests would not",
        "#   notice that mutation.  It is a work list, not a permission",
        "#   slip.  Deleting a line is how a coverage hole is closed:",
        "#   write the test, watch the run stay green, remove the line.",
        "#",
        "# FORMAT",
        "#   COUNT | PACKAGE | FILE | FUNCTION | GENRE | REPLACEMENT",
        "#",
        "#   COUNT is how many identical survivors that key stands for --",
        "#   the same mutation can appear more than once in one function.",
        "#   REPLACEMENT is JSON-quoted because it may contain the field",
        "#   separator.  There is no line number on purpose: line numbers",
        "#   move when anything above them is edited, and a list keyed by",
        "#   them would forgive a survivor that slid down the file and",
        "#   manufacture a fresh one out of the same mutation.",
        "#",
        "# THE ONE FALSE POSITIVE THIS SHAPE HAS",
        "#   A function moved bodily to another file changes its key, so",
        "#   every survivor it already had reads as fresh and the run is",
        "#   red for a change that altered no behaviour.  Re-baseline",
        "#   just that package -- minutes, not the whole list:",
        "#     python tools/battery/run_mutation_gate.py --all-mutants \\",
        "#         --baseline --only <package> --jobs 8",
        "#",
        "# REGENERATE EVERYTHING (deliberately, never to make a red run",
        "# green; it also discards any note written by hand above a line):",
        "#   python tools/battery/run_mutation_gate.py --all-mutants "
        "--baseline --jobs 8",
        "",
    ]
    for key in sorted(counts, key=lambda k: (k.package, k.file, k.function,
                                             k.genre, k.replacement)):
        lines.append(f"{counts[key]} | {key.spelling()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(counts)


def _print_matrix(results: list[PackageResult]) -> None:
    header = (f"{'package':<20} {'mode':<7} {'tested':>7} {'caught':>7} "
              f"{'survived':>9} {'unviable':>9} {'fresh':>6}  verdict")
    print(header)
    print("-" * len(header))
    for result in results:
        if result.entry.mode == "block":
            verdict = "FRESH SURVIVOR" if result.fresh else "PASSED"
        else:
            verdict = "REPORTED"
        print(f"{result.entry.package:<20} {result.entry.mode:<7} "
              f"{result.tested:>7} {result.caught:>7} {result.survived:>9} "
              f"{result.unviable:>9} {len(result.fresh):>6}  {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default=None,
                        help="revision to diff the WORKING TREE against; "
                             "only mutants inside changed lines are "
                             "tested.  Use HEAD~1 for the commit just "
                             "landed, HEAD for uncommitted work.")
    parser.add_argument("--all-mutants", action="store_true",
                        help="ignore the diff and test every mutant in the "
                             "selected packages.  This is the first recording and "
                             "nightly invocation; it is measured in hours "
                             "and is not a per-commit gate.")
    parser.add_argument("--only", action="append", default=[],
                        metavar="PACKAGE",
                        help="restrict the run to this manifest package; "
                             "repeatable.  With --baseline it re-baselines "
                             "only these packages and leaves every other "
                             "debt entry alone, which is what a function "
                             "that moved wholesale needs: all its mutants "
                             "land in the diff, all its old survivors "
                             "read as fresh, and re-recording one cheap "
                             "package costs minutes.")
    parser.add_argument("--mode", default=GATE_MODE,
                        choices=list(MODES) + ["all"],
                        help=f"which manifest entries to run (default "
                             f"{GATE_MODE}, the mode a per-commit gate "
                             f"runs)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="how many staged copies to run concurrently.  "
                             "cargo-mutants refuses --in-place with --jobs, "
                             "so parallelism comes from N copies each "
                             "running one shard.  About 310 MB per copy.")
    parser.add_argument("--scratch", default=None,
                        help="directory to stage copies under (default: a "
                             "temporary directory, removed at the end)")
    parser.add_argument("--keep", action="store_true",
                        help="leave the staged copies and their "
                             "mutants.out behind for inspection")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-mutant test timeout in seconds, passed "
                             "to cargo-mutants")
    parser.add_argument("--cargo-mutants", default="cargo",
                        help="cargo executable that provides the mutants "
                             "subcommand (default: cargo on PATH)")
    parser.add_argument("--baseline", action="store_true",
                        help="rewrite tools/battery/mutation_survivors.txt "
                             "from this run's survivors.  Requires "
                             "--all-mutants: recording from a partial run "
                             "would delete debt for mutants that were "
                             "never tested.")
    parser.add_argument("--list", action="store_true",
                        help="print the selected entries and exit")
    args = parser.parse_args(argv)

    try:
        entries = read_manifest()
        debt = read_debt()
    except (OSError, ValueError) as exc:
        print(f"MUTATION GATE COULD NOT RUN: {exc}", file=sys.stderr)
        return CANNOT_RUN

    selected = [e for e in entries
                if args.mode == "all" or e.mode == args.mode]
    if args.only:
        known = {e.package for e in entries}
        unknown = sorted(set(args.only) - known)
        if unknown:
            print(f"MUTATION GATE COULD NOT RUN: --only names "
                  f"{', '.join(unknown)}, which {MANIFEST} does not list. "
                  "A typo that silently selected nothing would report a "
                  "clean run over no packages.", file=sys.stderr)
            return CANNOT_RUN
        selected = [e for e in selected if e.package in set(args.only)]
    if args.list:
        for entry in selected:
            print(f"{entry.mode}  {entry.workspace}  {entry.package}")
        return 0
    if not selected:
        print(f"MUTATION GATE COULD NOT RUN: no {args.mode} entries in "
              f"{MANIFEST}", file=sys.stderr)
        return CANNOT_RUN
    if args.baseline and not args.all_mutants:
        print("MUTATION GATE COULD NOT RUN: --baseline needs --all-mutants.  "
              "Baselining from a diff-scoped run would rewrite the debt list "
              "from the handful of mutants that diff touched and delete "
              "every other entry, which turns the ratchet into a rubber "
              "stamp.", file=sys.stderr)
        return CANNOT_RUN
    if not args.all_mutants and not args.since:
        print("MUTATION GATE COULD NOT RUN: pass --since REV or "
              "--all-mutants.  There is no default revision: a gate that "
              "guesses its own scope reports a number nobody can "
              "reproduce.", file=sys.stderr)
        return CANNOT_RUN
    if args.jobs < 1:
        print(f"MUTATION GATE COULD NOT RUN: --jobs {args.jobs} is not a "
              "positive integer", file=sys.stderr)
        return CANNOT_RUN
    if shutil.which(args.cargo_mutants) is None:
        print(f"MUTATION GATE COULD NOT RUN: {args.cargo_mutants!r} is not "
              "on PATH.  This leg does not skip -- a clean report from a "
              "box with no cargo-mutants would say the arithmetic is "
              "covered when nothing mutated it.", file=sys.stderr)
        return CANNOT_RUN

    workspaces = sorted({e.workspace for e in selected})
    if len(workspaces) != 1:
        print("MUTATION GATE COULD NOT RUN: the selected entries span "
              f"{workspaces}.  One invocation stages one workspace; run "
              "one per workspace.", file=sys.stderr)
        return CANNOT_RUN
    workspace = REPOSITORY_ROOT / workspaces[0]
    if not (workspace / "Cargo.toml").is_file():
        print(f"MUTATION GATE COULD NOT RUN: {workspaces[0]}/Cargo.toml "
              "does not exist", file=sys.stderr)
        return CANNOT_RUN

    packages = [e.package for e in selected]
    print(f"tree            {REPOSITORY_ROOT}")
    print(f"workspace       {workspaces[0]}")
    print(f"mode            {args.mode}, {len(selected)} packages: "
          f"{', '.join(packages)}")
    print(f"scope           " + ("every mutant" if args.all_mutants
                                 else f"mutants inside the diff against "
                                      f"{args.since}"))
    print(f"debt list       {len(debt)} allowed survivors")

    scratch_root = (Path(args.scratch).resolve() if args.scratch
                    else Path(tempfile.mkdtemp(prefix="mutation-gate-")))
    scratch_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        extras = escaping_path_dependencies(workspace)
        if extras:
            print("staged deps     " + ", ".join(
                str(p.relative_to(REPOSITORY_ROOT).as_posix())
                for p in extras))

        diff_path: Path | None = None
        if not args.all_mutants:
            diff_path = scratch_root / "changed.diff"
            size = write_diff(workspace, args.since, diff_path)
            if size == 0:
                print()
                print(f"MUTATION GATE PASSED  the diff against "
                      f"{args.since} changes no Rust in {workspaces[0]}; "
                      "no mutants to test")
                return 0

        copies: list[Path] = []
        for index in range(args.jobs):
            destination = scratch_root / f"shard{index}"
            copies.append(stage_copy(workspace, extras, destination))
        print(f"staged          {args.jobs} copies in "
              f"{time.monotonic() - started:.1f}s under {scratch_root}")
        print()

        outputs = [scratch_root / f"out{i}" for i in range(args.jobs)]
        plans = []
        for index, (copy, output) in enumerate(zip(copies, outputs)):
            shard = (index, args.jobs) if args.jobs > 1 else None
            local_diff = None
            if diff_path is not None:
                local_diff = diff_path
            plans.append((shard_argv(packages, diff=local_diff,
                                     output=output, shard=shard,
                                     timeout=args.timeout,
                                     cargo_mutants=args.cargo_mutants),
                          copy))

        completed: list[subprocess.CompletedProcess] = []
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.jobs) as pool:
            futures = [pool.submit(run_shard, argv, cwd)
                       for argv, cwd in plans]
            for index, future in enumerate(futures):
                result = future.result()
                completed.append(result)
                print(f"  shard {index}/{args.jobs} exit "
                      f"{result.returncode}", flush=True)

        observations: list[Observation] = []
        failures: list[str] = []
        for index, (output, result) in enumerate(zip(outputs, completed)):
            found, problem = collect(output, result.returncode)
            observations.extend(found)
            if problem is not None:
                failures.append(
                    f"shard {index}: {problem}; cargo-mutants exit "
                    f"{result.returncode}\n" +
                    "\n".join(result.stdout.strip().splitlines()[-25:]))
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            print("MUTATION GATE COULD NOT RUN: a shard produced no "
                  "outcomes.  The usual cause is a baseline that did not "
                  "build in the staged copy.", file=sys.stderr)
            return CANNOT_RUN

        if args.baseline:
            if not observations:
                print("MUTATION GATE COULD NOT RUN: the run tested no "
                      "mutants, so baselining from it would empty the debt "
                      "list and call that a clean tree.", file=sys.stderr)
                return CANNOT_RUN
            written = write_debt(DEBT, observations, debt,
                                 covered=set(packages))
            print()
            print(f"BASELINED  {DEBT} now carries {written} allowed "
                  f"survivors from {len(observations)} mutants tested")
            return 0

        results = tally(selected, observations, debt)
        ordered = [results[p] for p in packages if p in results]
        elapsed = time.monotonic() - started
        print()
        _print_matrix(ordered)
        print()

        fresh = [(r, o) for r in ordered if r.entry.mode == "block"
                 for o in r.fresh]
        reported = [(r, o) for r in ordered if r.entry.mode == "report"
                    for o in r.fresh]
        # A report package carries no debt (the debt list covers only
        # blocking packages, so an allowance can never sit against a gate
        # that cannot fire), which means every survivor prints.  Cap the
        # listing: rustwx-calc alone survives about seven hundred
        # mutations today, and seven hundred lines is a wall, not a
        # report.  The count above it is the number that matters.
        for result, observation in reported[:REPORT_LISTING]:
            print(f"    survived {result.entry.package}: {observation.name}")
        if len(reported) > REPORT_LISTING:
            print(f"    ... and {len(reported) - REPORT_LISTING} more "
                  "surviving mutants in report packages; the full list is "
                  "each shard's mutants.out/missed.txt (pass --keep)")
        for result, observation in fresh:
            print(f"--- FRESH SURVIVOR in {result.entry.package}")
            print(f"    {observation.name}")
            print(f"    key: {observation.key.spelling()}")

        tested = sum(r.tested for r in ordered)
        survived = sum(r.survived for r in ordered)
        if fresh:
            print()
            print(f"MUTATION GATE FAILED  {len(fresh)} mutation(s) changed "
                  f"the code and no test noticed, in {tested} tested "
                  f"({elapsed:.1f}s).  Each one is arithmetic a product is "
                  "read off that the suite would not defend.  Write the "
                  "test.  Adding the line to "
                  "tools/battery/mutation_survivors.txt instead is a "
                  "widening and needs a ruling, not a commit.")
            return 1
        print(f"MUTATION GATE PASSED  {tested} mutants tested, "
              f"{survived} survived and all were already on the debt list, "
              f"0 fresh ({elapsed:.1f}s)")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"MUTATION GATE COULD NOT RUN: {exc}", file=sys.stderr)
        return CANNOT_RUN
    finally:
        if args.keep:
            print(f"kept            {scratch_root}")
        else:
            shutil.rmtree(scratch_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
