#!/usr/bin/env python3
"""Run the Rust workspaces' test suites as one release-battery leg.

``tools/battery/cargo_gates.txt`` enumerates them, one package per line,
each with the expected-green contract it has to satisfy.  This is the
runner the battery invokes:

    python tools/battery/run_cargo_gates.py

from the repository root, with ``CARGO_TARGET_DIR`` pointing OUTSIDE the
tree.  It prints a matrix -- one row per entry, with the counts cargo
reported -- and ends in a single verdict line, the same shape as the
``[tiles]`` gates, because a leg that reports only an exit status throws
away the numbers that make a regression legible.

THREE REFUSALS, ALL LOUD
    * no ``CARGO_TARGET_DIR``, or one that resolves inside the
      repository.  A cut hashes the worktree-local renderer at
      ``tools/rustwx/target/release`` before it believes any leg that
      renders; a test leg sharing that directory can replace the binary
      the cut is adjudicating.  Defaulting to a scratch path would hide
      the mistake instead of stopping it, so there is no default.
    * no ``cargo`` on PATH.  This leg is not optional and does not skip:
      a battery that reports a clean leg on a box with no Rust toolchain
      is exactly the green-on-nothing failure the 2026-08-09 audit was
      called to close.
    * an entry whose workspace or package the manifest does not
      describe.  ``tests/test_cargo_gate_manifest.py`` catches that
      without a toolchain; this catches it again at the point of use.

EXIT STATUS IS THE VERDICT: 0 when every entry on the selected shard
passed its contract, 1 when any failed, 3 when the leg could not run at
all.  Those are different answers and the battery reads them
differently -- a leg that could not run has proven nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "cargo_gates.txt"

#: The shards the battery knows how to run.  ``cpu`` and ``gpu`` are the
#: shared vocabulary with tools/battery/tiles_gates.txt; ``fixtures`` is
#: this list's own, for a suite that needs a data set the repository does
#: not carry.  A fixtures entry is LISTED and not run by default, which
#: is the difference between coverage that is deferred and coverage that
#: silently vanished.
SHARDS = ("cpu", "gpu", "fixtures")

#: The shard a release cut runs.  Named rather than spelled inline so the
#: manifest gate and the runner cannot disagree about it.
BATTERY_SHARD = "cpu"

#: Cargo's per-binary summary, e.g.
#: ``test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; ...``
_RESULT = re.compile(
    r"^test result: (?P<verdict>ok|FAILED)\. "
    r"(?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored",
    re.MULTILINE)

#: Exit status when the leg could not be run at all, as distinct from a
#: leg that ran and failed (1).  A missing toolchain disproves nothing.
CANNOT_RUN = 3


@dataclass(frozen=True)
class Entry:
    """One line of the manifest, parsed."""

    shard: str
    workspace: str
    package: str
    min_tests: int
    env: tuple[str, ...]
    args: tuple[str, ...]

    @property
    def label(self) -> str:
        suffix = (" " + " ".join(self.args)) if self.args else ""
        return f"{self.package}{suffix}"

    def invocation(self) -> list[str]:
        """The exact argv, so a failing row can be re-run by hand.

        ``--locked`` and ``--offline`` are not tunable: the workspaces
        vendor their dependencies and a green leg is supposed to prove
        that vendored closure is complete.
        """

        return (["cargo", "test", "--locked", "--offline", "-p",
                 self.package] + list(self.args))


@dataclass
class Outcome:
    entry: Entry
    passed: int
    failed: int
    ignored: int
    binaries: int
    returncode: int
    seconds: float
    output: str

    @property
    def verdict(self) -> str:
        if self.returncode != 0 or self.failed:
            return "FAILED"
        if self.passed < self.entry.min_tests:
            return "UNDER-RAN"
        return "PASSED"


def parse_manifest(text: str) -> list[Entry]:
    """Parse the manifest exactly the way this runner consumes it.

    ``tests/test_cargo_gate_manifest.py`` imports this function rather
    than restating the format, so a file the suite accepts is a file the
    runner can read.
    """

    entries: list[Entry] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"tools/battery/cargo_gates.txt line {number}: expected "
                f"SHARD WORKSPACE PACKAGE MIN_TESTS [KEY=VALUE...], got "
                f"{line!r}")
        shard, workspace, package, min_tests = parts[:4]
        if shard not in SHARDS:
            raise ValueError(
                f"tools/battery/cargo_gates.txt line {number}: shard "
                f"{shard!r} is not one of {SHARDS}")
        if not min_tests.isdigit() or int(min_tests) < 1:
            raise ValueError(
                f"tools/battery/cargo_gates.txt line {number}: MIN_TESTS "
                f"{min_tests!r} must be a positive integer -- a floor of 0 "
                "is satisfied by a suite that compiled nothing")
        env: list[str] = []
        args: list[str] = []
        for token in parts[4:]:
            # KEY=VALUE is environment; anything else goes to cargo.  A
            # bare word that is neither is refused rather than guessed
            # at: `--test cloud_products` is two tokens and the second
            # one is a bare word, so it is accepted only AFTER a cargo
            # argument has opened the tail.
            if "=" in token and not token.startswith("-") and not args:
                env.append(token)
            elif token.startswith("-") or args:
                args.append(token)
            else:
                raise ValueError(
                    f"tools/battery/cargo_gates.txt line {number}: "
                    f"{token!r} is neither KEY=VALUE nor a cargo argument; "
                    "environment assignments come first, then cargo "
                    "arguments beginning with a dash")
        entries.append(Entry(shard=shard, workspace=workspace,
                             package=package, min_tests=int(min_tests),
                             env=tuple(env), args=tuple(args)))
    return entries


def read_manifest(path: Path = MANIFEST) -> list[Entry]:
    return parse_manifest(path.read_text(encoding="utf-8"))


def resolve_target_dir(explicit: str | None) -> Path:
    """The external target directory, or a refusal that says why."""

    raw = explicit or os.environ.get("CARGO_TARGET_DIR")
    if not raw:
        raise RuntimeError(
            "CARGO_TARGET_DIR is not set.  This leg builds two Cargo "
            "workspaces and it is not allowed to build them inside the "
            "tree: a release cut hashes the worktree-local renderer at "
            "tools/rustwx/target/release and refuses to assemble from a "
            "dirty source tree, and a shared target directory would let "
            "this leg replace the binary the cut is adjudicating.  Set "
            "CARGO_TARGET_DIR to a path outside the repository, or pass "
            "--target-dir.")
    target = Path(raw).resolve()
    if target == REPOSITORY_ROOT or REPOSITORY_ROOT in target.parents:
        raise RuntimeError(
            f"CARGO_TARGET_DIR={target} is inside the repository at "
            f"{REPOSITORY_ROOT}.  Point it outside the tree: the cut's "
            "renderer check and its dirty-tree refusal both read this "
            "worktree, and cargo artefacts written into it make both "
            "answers depend on which leg ran last.")
    return target


def run_entry(entry: Entry, *, target_dir: Path, cargo: str) -> Outcome:
    workspace = REPOSITORY_ROOT / entry.workspace
    if not (workspace / "Cargo.toml").is_file():
        raise RuntimeError(
            f"{entry.label}: {entry.workspace}/Cargo.toml does not exist")
    if not (workspace / ".cargo" / "config.toml").is_file():
        raise RuntimeError(
            f"{entry.label}: {entry.workspace}/.cargo/config.toml does not "
            "exist, so --offline has no vendored source to resolve against")

    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    for assignment in entry.env:
        key, _, value = assignment.partition("=")
        environment[key] = value

    argv = [cargo] + entry.invocation()[1:]
    # cwd is the WORKSPACE, never the repository root: cargo reads
    # .cargo/config.toml from the working directory upward, and that file
    # is what replaces crates.io with the vendored directory.  Invoked
    # from the root with --manifest-path the source replacement would not
    # apply and --offline would fail on a clean box for the wrong reason.
    started = time.monotonic()
    completed = subprocess.run(
        argv, cwd=workspace, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    seconds = time.monotonic() - started

    passed = failed = ignored = binaries = 0
    for match in _RESULT.finditer(completed.stdout):
        binaries += 1
        passed += int(match["passed"])
        failed += int(match["failed"])
        ignored += int(match["ignored"])
    return Outcome(entry=entry, passed=passed, failed=failed,
                   ignored=ignored, binaries=binaries,
                   returncode=completed.returncode, seconds=seconds,
                   output=completed.stdout)


def _print_matrix(outcomes: list[Outcome]) -> None:
    header = (f"{'package':<34} {'passed':>7} {'failed':>7} {'ign':>5} "
              f"{'bins':>5} {'floor':>6} {'secs':>7}  verdict")
    print(header)
    print("-" * len(header))
    for outcome in outcomes:
        print(f"{outcome.entry.label:<34} {outcome.passed:>7} "
              f"{outcome.failed:>7} {outcome.ignored:>5} "
              f"{outcome.binaries:>5} {outcome.entry.min_tests:>6} "
              f"{outcome.seconds:>7.1f}  {outcome.verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard", default=BATTERY_SHARD, choices=SHARDS,
                        help=f"which shard's entries to run (default "
                             f"{BATTERY_SHARD}, the shard a release cut "
                             f"runs)")
    parser.add_argument("--target-dir", default=None,
                        help="external CARGO_TARGET_DIR; defaults to the "
                             "environment variable, and there is no "
                             "in-tree fallback")
    parser.add_argument("--cargo", default="cargo",
                        help="cargo executable (default: cargo on PATH)")
    parser.add_argument("--list", action="store_true",
                        help="print the entries and exit without running")
    args = parser.parse_args(argv)

    try:
        entries = read_manifest()
    except (OSError, ValueError) as exc:
        print(f"CARGO GATE COULD NOT RUN: {exc}", file=sys.stderr)
        return CANNOT_RUN

    selected = [e for e in entries if e.shard == args.shard]
    if args.list:
        for entry in selected:
            tail = " ".join(entry.env + entry.args)
            print(f"{entry.shard}  {entry.workspace}  {entry.package}  "
                  f"{entry.min_tests}" + (f"  {tail}" if tail else ""))
        return 0
    if not selected:
        print(f"CARGO GATE COULD NOT RUN: no {args.shard} entries in "
              f"{MANIFEST}", file=sys.stderr)
        return CANNOT_RUN

    if shutil.which(args.cargo) is None:
        print(f"CARGO GATE COULD NOT RUN: {args.cargo!r} is not on PATH.  "
              "This leg does not skip -- a clean report from a box with no "
              "Rust toolchain would say the crates are green when nothing "
              "compiled them.", file=sys.stderr)
        return CANNOT_RUN

    try:
        target_dir = resolve_target_dir(args.target_dir)
    except RuntimeError as exc:
        print(f"CARGO GATE COULD NOT RUN: {exc}", file=sys.stderr)
        return CANNOT_RUN

    print(f"tree            {REPOSITORY_ROOT}")
    print(f"CARGO_TARGET_DIR {target_dir}")
    print(f"shard           {args.shard}, {len(selected)} entries, "
          f"cargo test --locked --offline")
    print()

    outcomes: list[Outcome] = []
    for entry in selected:
        try:
            outcome = run_entry(entry, target_dir=target_dir,
                                cargo=args.cargo)
        except RuntimeError as exc:
            print(f"CARGO GATE COULD NOT RUN: {exc}", file=sys.stderr)
            return CANNOT_RUN
        outcomes.append(outcome)
        print(f"  {outcome.entry.label:<34} {outcome.verdict:<10} "
              f"{outcome.passed} passed, {outcome.failed} failed, "
              f"{outcome.seconds:.1f}s", flush=True)

    print()
    _print_matrix(outcomes)
    print()

    bad = [o for o in outcomes if o.verdict != "PASSED"]
    for outcome in bad:
        print(f"--- {outcome.entry.label} ({outcome.verdict}, "
              f"cargo exit {outcome.returncode}) "
              f"{' '.join(outcome.entry.invocation())}")
        tail = outcome.output.strip().splitlines()[-40:]
        for line in tail:
            print(f"    {line}")

    total = sum(o.passed for o in outcomes)
    if bad:
        print(f"CARGO GATE FAILED  {len(bad)} of {len(outcomes)} entries: "
              + ", ".join(f"{o.entry.label} ({o.verdict})" for o in bad))
        return 1
    print(f"CARGO GATE PASSED  {len(outcomes)} entries, {total} tests "
          f"passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
