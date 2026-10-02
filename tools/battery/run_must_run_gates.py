"""Run the must-run oracle decks that tools/battery/must_run_gates.txt lists.

THE BREAKAGE THIS PREVENTS.  The public CI's oracles job read the list with
``mapfile -t decks < <(python -c ...)`` under bash.  On the Windows runner
Python writes its newlines as CRLF, ``mapfile -t`` strips only the LF, and
every deck arrived as ``tests/test_x.py\\r``.  pytest then reported
``ERROR: file or directory not found: tests/test_rrtmg_sw_oracle.py`` (the
carriage return does not print), exited 4 and ran no oracle at all, on 2.7.6,
2.7.7 and 2.8.0 alike, while the identical list ran 1,238 tests on Linux.

So the list never passes through a shell.  This reads it with the plugin's own
parser, refuses by name any listed file that does not exist (a missing file is
a gate that left, and it fails the job before pytest runs instead of turning
into zero tests), refuses a list shorter than ``--minimum``, and hands the
paths to pytest as arguments.  Everything after ``--`` goes to pytest
unchanged, and its exit status is this command's.

    python tools/battery/run_must_run_gates.py --minimum 20 -- -q -m "not gpu"
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = "tools/battery/must_run_gates.txt"


def _parser_module():
    """``tools/battery/no_silent_skip.py`` by path, as tests/conftest.py loads it."""
    path = ROOT / "tools" / "battery" / "no_silent_skip.py"
    spec = importlib.util.spec_from_file_location("gpuwm_must_run_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def listed_decks(root: pathlib.Path = ROOT) -> list[str]:
    """Every deck the manifest requires, in manifest order."""
    text = (root / MANIFEST).read_text(encoding="utf-8")
    return list(_parser_module().parse_manifest(text))


def missing_decks(decks: list[str], root: pathlib.Path = ROOT) -> list[str]:
    """The listed decks that are not files in this tree."""
    return [deck for deck in decks if not (root / deck).is_file()]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minimum", type=int, default=1,
                        help="refuse a manifest listing fewer decks than this")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args(argv)
    decks = listed_decks(args.root)
    missing = missing_decks(decks, args.root)
    if missing:
        print(f"{MANIFEST} lists {len(missing)} deck(s) that do not exist, so they "
              "would run zero tests:", file=sys.stderr)
        for deck in missing:
            print("  " + deck, file=sys.stderr)
        return 2
    if len(decks) < args.minimum:
        print(f"{MANIFEST} lists {len(decks)} deck(s), fewer than the {args.minimum} "
              "this leg requires", file=sys.stderr)
        return 2
    print(f"must-run decks: {len(decks)} listed, all present", flush=True)
    command = [sys.executable, "-m", "pytest", *passthrough, *decks]
    return subprocess.call(command, cwd=args.root)


if __name__ == "__main__":
    raise SystemExit(main())
