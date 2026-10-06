"""Write the generated half of the commit-message allowance.

    python tools/register_message_allowance.py --cutoff REV   # write
    python tools/register_message_allowance.py --check        # compare only

tests/test_excluded_subsystem_absent.py scans every commit after its clean
base for the excluded subsystem's names, in the message and in the author
and committer identities.  History here is forward only, so a commit that
has already merged cannot be reworded; it is registered instead.  This
program is the only writer of that registration,
tests/data/excluded_subsystem_message_allowance.json.

THE BREAKAGE THIS PREVENTS.  The 2.8.6 gate found 281 merged commits
flagged on a full-history clone, nearly all of them by an author address
from 2026-10-01 to 10-04.  Typed by hand, a list that long cannot be
reviewed, and one extra SHA in it would excuse a real leak.  Generated, it
is exactly the commits the test's own scan flags between the clean base and
a named cutoff, and the test re-derives it on every run, so a hand edit or a
silent move of the cutoff fails by name.

The scan is the test module's, loaded by path, so the definition of a
flagged commit cannot drift between the two.  Before moving the cutoff,
confirm that the public repositories still hold release snapshot commits
only (no commit of this range reachable from any public ref): the reason
this file records is true only while that holds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_PATH = REPO_ROOT / "tests" / "test_excluded_subsystem_absent.py"
GENERATOR = "tools/register_message_allowance.py"


def _load_scan():
    name = "gpuwm_message_allowance_scan"
    spec = importlib.util.spec_from_file_location(name, SCAN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode().strip()


def derive(scan, cutoff: str) -> dict:
    """The allowance document for ``cutoff``, from the scan's own rules."""

    cutoff = _git("rev-parse", "--verify", cutoff + "^{commit}")
    base = scan.CLEAN_BASE
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, cutoff],
                      cwd=REPO_ROOT).returncode != 0:
        raise SystemExit(f"refused: the clean base {base} is not an ancestor of "
                         f"{cutoff}, so the scan range would be empty or wrong")
    records = scan._commit_objects(base + ".." + cutoff)
    flagged = set(scan._flagged_commits(records)) - set(scan._MESSAGE_ALLOWANCE_PINNED)
    return {
        "schema": scan.MESSAGE_ALLOWANCE_SCHEMA,
        "generated_by": GENERATOR,
        "reason": scan.MESSAGE_ALLOWANCE_REASON,
        "clean_base": base,
        "cutoff": cutoff,
        "scanned_commits": len(records),
        "commits": sorted(flagged),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cutoff", help="the last commit to register (a merged tip)")
    mode.add_argument("--check", action="store_true",
                      help="re-derive at the recorded cutoff and compare")
    args = parser.parse_args(argv)
    scan = _load_scan()
    path = scan.MESSAGE_ALLOWANCE_FILE
    if args.check:
        recorded = json.loads(path.read_text(encoding="utf-8"))
        fresh = derive(scan, recorded["cutoff"])
        if fresh != recorded:
            print(f"{path.relative_to(REPO_ROOT)} differs from the derivation at "
                  f"{recorded['cutoff'][:12]}", file=sys.stderr)
            return 1
        print(f"ok: {len(recorded['commits'])} registered commits match the scan "
              f"over {recorded['scanned_commits']} commits to {recorded['cutoff'][:12]}")
        return 0
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))["cutoff"]
        target = _git("rev-parse", "--verify", args.cutoff + "^{commit}")
        if subprocess.run(["git", "merge-base", "--is-ancestor", previous, target],
                          cwd=REPO_ROOT).returncode != 0:
            raise SystemExit(f"refused: the recorded cutoff {previous} is not an "
                             f"ancestor of {target}; moving it sideways or back "
                             "would drop registered history")
    document = derive(scan, args.cutoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}: {len(document['commits'])} commits "
          f"of {document['scanned_commits']} to {document['cutoff'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
