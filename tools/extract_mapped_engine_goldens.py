"""Extract mapped-engine parity goldens from the REAL Python engine.

The goldens the parity battery holds both engines to are not authored;
they are MEASURED, by decoding the real staged agency bytes with
``gpuwm.mapped_source``'s Python engine -- the behaviour of record --
and reducing each decode to the parity digest the battery compares.

Run it when the staging tree gains a source, and re-run it (deliberately,
with the diff read) when a decode legitimately changes.  A golden that
moves without a named reason is the defect the battery exists to catch,
so this tool prints the diff summary and refuses to rewrite silently
unless ``--force`` says to.

    python tools/extract_mapped_engine_goldens.py            # all sources
    python tools/extract_mapped_engine_goldens.py --source gdas-pgrb2-0p25
    python tools/extract_mapped_engine_goldens.py --list

Two KINDS of golden, because the engine has two subcommands a bare run
reaches and they measure different work:

``decode``
    one mapping over its own bytes, reduced to per-field digests, plus
    the inspection report.  Goldens in ``tests/data/mapped_engine_goldens``.
``compose``
    what a mapped PREPARATION actually runs (``MAPPED_ROUTE_SUBCOMMAND``
    is ``compose``): the composed canonical frames, the terrain/bound
    alignment receipt and the contributing-source records.  Goldens in
    ``tests/data/mapped_engine_compose_goldens``, keyed by REGISTRY
    source id (``hrrr-prs``, ``aigfs``) rather than by decode-row name.

    python tools/extract_mapped_engine_goldens.py --kind compose
    python tools/extract_mapped_engine_goldens.py --kind compose --source aigfs

The two kinds write into different directories and are never mixed, so
adding compose coverage cannot move a decode digest.

Every golden this tool writes DECLARES THE PLATFORM THAT MEASURED IT, in
a top-level ``measured_on`` member.  A golden is a measurement of a
platform and not only of a decode: measured field by field on node-1
2026-08-18, the directly decoded fields hash identical across Windows
and Linux while the DERIVED ones -- the grid-relative wind rotation and
the relative-humidity derivation -- and the frame header hashes that
carry them do not, because the platform's libm produced them.  The
battery compares against a golden byte for byte on the platform named
there and runs the live dual-engine comparison at the same strictness
anywhere else, so a golden with no stamp cannot be scoped and is refused
by the battery rather than excused.

That also makes re-measuring on a different platform a deliberate act:
this tool names the platform change in its refusal, and ``--force`` is
what re-homes a golden to the box in front of you.

Staged bytes live under ``GPUWM_MODEL_GAUNTLET_STAGING`` (default
``~/gpuwm-model-gauntlet-staging``); a source whose bytes are absent is
reported and skipped, never written as an empty golden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    from test_mapped_engine_parity import (  # noqa: E402 - path set above
        COMPOSE_GOLDENS, COMPOSED_SOURCES, GOLDEN_PLATFORM_KEY, GOLDENS,
        STAGED_SOURCES, _composed_files, _composed_staged, _mapping_of,
        _staged, canonical, compose_golden_document, composed_recipe,
        golden_document, measuring_platform,
    )
except ImportError as error:  # pragma: no cover - wheel-install route
    raise SystemExit(
        "extract_mapped_engine_goldens runs from a gpuwm checkout only: it "
        "imports the mapped parity battery beside the goldens it writes "
        "(tests/test_mapped_engine_parity.py, tests/data/), and tests/ does "
        f"not ship in the wheel ({error})") from error


DECODE, COMPOSE = "decode", "compose"


def _platform_line(stamp) -> str:
    return f"{stamp.get('system', '?')}/{stamp.get('machine', '?')}"


def _committed_platform(path: Path):
    """The platform a golden already on disk declares, or ``None``."""

    try:
        stamp = json.loads(path.read_text(encoding="utf-8")).get(
            GOLDEN_PLATFORM_KEY)
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) and stamp.get("system") else None


def _decode_line(name: str, document) -> str:
    decode = document["decode"]
    inspection = document["inspect"]
    if decode["verdict"] == "REFUSED":
        detail = (f"decode {decode['refusal']['type']}: "
                  f"{decode['refusal']['message'][:110]}")
    else:
        detail = (f"decode {decode['frame_count']} frame(s), "
                  f"{len(decode['frames'][0]['fields'])} canonical field(s)")
    if inspection["verdict"] == "REFUSED":
        reported = f"inspect {inspection['refusal']['type']}"
    else:
        arrays = sum(len(frame["fields"])
                     for frame in inspection["inspection"]["frames"])
        reported = f"inspect {arrays} decoded array(s)"
    return f"{detail}; {reported}"


def _compose_line(name: str, document) -> str:
    if document["verdict"] == "REFUSED":
        return (f"compose {document['refusal']['type']}: "
                f"{document['refusal']['message'][:110]}")
    bindings = len(document["contributing_sources"])
    return (f"compose {document['frame_count']} frame(s), "
            f"{len(document['frames'][0]['fields'])} canonical field(s), "
            f"{bindings} cross-source binding(s), alignment "
            f"{document['alignment_receipt'].get('time_alignment', 'valid_time_exact')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_mapped_engine_goldens",
        description=(
            "Measure mapped-engine parity goldens by running the Python "
            "engine on real staged bytes."),
    )
    parser.add_argument(
        "--kind", choices=(DECODE, COMPOSE), default=DECODE,
        help=("which subcommand's goldens to measure: `decode` (default, "
              "one mapping over its own bytes) or `compose` (what a mapped "
              "preparation actually runs)"))
    parser.add_argument(
        "--source", action="append", dest="sources",
        help="extract one source; repeat for several (default: all)")
    parser.add_argument(
        "--list", action="store_true",
        help="print the registry and whether each source's bytes are staged")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite a golden whose digest changed")
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(f"golden directory (default {GOLDENS} for --kind decode, "
              f"{COMPOSE_GOLDENS} for --kind compose)"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compose = args.kind == COMPOSE
    registry = COMPOSED_SOURCES if compose else STAGED_SOURCES
    if args.output is None:
        args.output = COMPOSE_GOLDENS if compose else GOLDENS
    if args.list:
        for name in sorted(registry):
            if compose:
                primary, supplement = _composed_files(name)
                state = "staged" if _composed_staged(name) else "ABSENT"
                recipe = composed_recipe(name)
                print(f"{name:24s} {state:7s} {len(primary):4d} primary + "
                      f"{len(supplement)} supplement  "
                      f"{recipe['composition'].name}")
                continue
            entry = registry[name]
            state = "staged" if _staged(entry) else "ABSENT"
            print(f"{name:24s} {state:7s} {len(entry['files']):4d} file(s)  "
                  f"{_mapping_of(entry).name}")
        return 0

    names = sorted(args.sources or registry)
    unknown = [name for name in names if name not in registry]
    if unknown:
        print(f"unknown {args.kind} source(s): {unknown}; --list names the "
              "registry", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    here = measuring_platform()
    print(f"measuring on {_platform_line(here)}; every golden written here "
          f"is stamped with it")
    changed: list[str] = []
    skipped: list[str] = []
    for name in names:
        present = _composed_staged(name) if compose else _staged(registry[name])
        if not present:
            skipped.append(name)
            print(f"{name}: staged bytes absent, skipped")
            continue
        started = time.perf_counter()
        document = (compose_golden_document(name) if compose
                    else golden_document(name))
        elapsed = time.perf_counter() - started
        line = (_compose_line(name, document) if compose
                else _decode_line(name, document))
        print(f"{name}: {line} in {elapsed:.1f}s")

        path = args.output / f"{name}.json"
        payload = canonical(document) + "\n"
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if existing == payload:
                continue
            if not args.force:
                committed = _committed_platform(path)
                if committed is not None and committed != here:
                    # Named separately because it is a different act: the
                    # numbers are EXPECTED to move between platforms (the
                    # derived fields carry the box's libm), so re-homing a
                    # golden is a decision about which box the battery
                    # compares byte-for-byte on, not a defect report.
                    print(f"  REFUSING to rewrite {path}: it was measured on "
                          f"{_platform_line(committed)} and this is "
                          f"{_platform_line(here)}.  The battery already "
                          "runs the live dual-engine comparison here, so "
                          "nothing is blocked by leaving it alone; re-run "
                          "with --force only to MOVE the golden's home "
                          "platform to this box", file=sys.stderr)
                    changed.append(name)
                    continue
                print(f"  REFUSING to rewrite {path}: the digest moved.  "
                      "Read the diff, name the defective side, then re-run "
                      "with --force", file=sys.stderr)
                changed.append(name)
                continue
            print(f"  rewriting {path} (--force)")
        # newline="\n" explicitly: the repository stores LF, and the
        # default translation on Windows would commit a golden whose
        # every line differs from the one a POSIX box would write.
        path.write_text(payload, encoding="utf-8", newline="\n")

    if skipped:
        print(f"skipped (bytes absent): {', '.join(skipped)}")
    if changed:
        print(f"golden(s) moved without --force: {', '.join(changed)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
