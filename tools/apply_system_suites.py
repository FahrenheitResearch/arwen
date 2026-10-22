#!/usr/bin/env python3
"""Render gpuwm/data/system_suites/*.json into the physics registry.

    python tools/apply_system_suites.py            # rewrite the tracked registry
    python tools/apply_system_suites.py --check    # exit 1 if it is stale

The registry is one canonical JSON line that several builders regenerate,
so a merge that keeps another side's copy drops this block without a
textual conflict to show it.  Run this after any such merge; the test
``tests/test_system_suites.py`` fails while the block and the tables
disagree.  Only the ``system_suites`` block is written; every other byte
of the registry is carried through unchanged.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY_PATH = ROOT / "gpuwm" / "physics_registry_v2.json"


def rendered(registry_text: str) -> bytes:
    from gpuwm.physics_registry import canonical_json
    from gpuwm.system_suites import REGISTRY_BLOCK, load_tables, registry_rows

    registry = json.loads(registry_text)
    registry[REGISTRY_BLOCK] = registry_rows(load_tables())
    return (canonical_json(registry) + "\n").encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=pathlib.Path, default=REGISTRY_PATH)
    parser.add_argument("--check", action="store_true",
                        help="report whether the registry is current; write nothing")
    args = parser.parse_args(argv)
    current = args.registry.read_bytes()
    wanted = rendered(current.decode("utf-8"))
    if args.check:
        if current != wanted:
            print(f"{args.registry}: system_suites block is stale; run "
                  "tools/apply_system_suites.py")
            return 1
        print(f"{args.registry}: system_suites block is current")
        return 0
    args.registry.write_bytes(wanted)
    print(f"wrote {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
