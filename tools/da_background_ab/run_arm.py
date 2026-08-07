"""Run one cycling arm with pins the arm before it produced.

``tools/da_cycle_prepared.py`` binds a prepared case by three digests.
For the GFS arm those digests exist today, on disk, and can be written
into a plan.  For the HRRR arms they do not exist until the preparation
that makes the case has run -- and a plan that hard-codes them would
either be un-writable or, worse, writable with the wrong ones.

So the HRRR arms name a FILE instead of a digest: the
``public-wrapper-result.json`` the certified preparation writes, whose
``portable_bundle`` block carries exactly the three pins the forecast
stage must bind (``proof_sha256``, ``source_manifest_sha256``,
``prepared_content_sha256``).  This resolves them at launch and refuses,
loudly, if the block is absent -- which is what a preparation run
WITHOUT ``--wps-namelist`` leaves behind, and is a real mistake to make.

The assembled argument list is then pushed through
``tools.da_cycle_prepared``'s own parser before anything runs, so a
grammar error costs a message rather than a queue slot.  The pins never
reach a shell: they go straight from the JSON into ``argv``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The pins, in the order the cycle driver's flags take them.
PIN_FLAGS = (
    ("--proof-sha256", "proof_sha256"),
    ("--source-manifest-sha256", "source_manifest_sha256"),
    ("--prepared-content-sha256", "prepared_content_sha256"),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def read_pins(path: Path) -> dict[str, str]:
    """The three digests, or a refusal naming what is missing."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    bundle = document.get("portable_bundle")
    if not isinstance(bundle, dict):
        raise SystemExit(
            f"{path} carries no portable_bundle block.  That is what a "
            "preparation run WITHOUT --wps-namelist leaves behind: the "
            "native bundle exists but the three portable authorities a "
            "config-driven forecast stage binds were never published.  "
            "Re-run the preparation with --wps-namelist")
    pins = {}
    for flag, key in PIN_FLAGS:
        value = bundle.get(key)
        if not isinstance(value, str) or not _SHA256.match(value.lower()):
            raise SystemExit(
                f"{path}: portable_bundle.{key} is {value!r}, which is not "
                f"a sha256; {flag} cannot be bound from it")
        pins[key] = value.lower()
    return pins


def assemble(argv_tail: list[str], pins: dict[str, str] | None) -> list[str]:
    argv = list(argv_tail)
    if pins is not None:
        for flag, key in PIN_FLAGS:
            if flag in argv:
                raise SystemExit(
                    f"{flag} was given on the command line AND resolved "
                    "from a pins file; refusing to choose between two "
                    "statements of the same binding")
            argv.extend((flag, pins[key]))
    return argv


def validate(argv: list[str]) -> None:
    """Parse with the cycle driver's own parser, then stop before it runs."""

    import tools.da_cycle_prepared as driver

    real = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def intercept(self, args=None, namespace=None):
        real(self, args if args is not None else argv, namespace)
        raise Parsed()

    argparse.ArgumentParser.parse_args = intercept
    try:
        driver.main()
    except Parsed:
        return
    finally:
        argparse.ArgumentParser.parse_args = real
    raise SystemExit("da_cycle_prepared.main() returned without parsing")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.run_arm",
        description=__doc__.splitlines()[0])
    parser.add_argument("--pins-from", type=Path, default=None,
                        help="public-wrapper-result.json whose "
                             "portable_bundle carries the three digests")
    parser.add_argument("--validate-only", action="store_true",
                        help="parse the assembled argument list and exit; "
                             "with --pins-from absent this proves the "
                             "grammar without needing the case to exist")
    parser.add_argument("cycle_argv", nargs=argparse.REMAINDER,
                        help="-- followed by tools.da_cycle_prepared flags")
    args = parser.parse_args(argv)

    tail = list(args.cycle_argv)
    if tail and tail[0] == "--":
        tail = tail[1:]
    if not tail:
        parser.error("no cycle arguments after --")

    # A pins file that does not exist yet is the NORMAL state at
    # validation time: the preparation arm that writes it has not run.
    # Placeholder digests then prove the GRAMMAR of an argument list
    # whose values cannot exist, which is the only thing a preflight can
    # honestly claim about it.  They are unreachable without
    # --validate-only, and the resolved case is still preferred whenever
    # the file IS there.
    resolved = args.pins_from is not None and args.pins_from.is_file()
    if args.pins_from is not None and not resolved and not args.validate_only:
        read_pins(args.pins_from)                # raises, with the reason
    pins = read_pins(args.pins_from) if resolved else None
    if pins is None and args.validate_only \
            and not any(flag in tail for flag, _ in PIN_FLAGS):
        pins = {key: "0" * 64 for _, key in PIN_FLAGS}
    assembled = assemble(tail, pins)
    validate(assembled)
    if args.validate_only:
        print("cycle argument list parses ("
              + ("pins resolved" if args.pins_from is not None
                 else "placeholder pins") + ")")
        return 0

    import tools.da_cycle_prepared as driver

    sys.argv = ["tools/da_cycle_prepared.py"] + assembled
    return int(driver.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
