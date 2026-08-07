"""Prepare one HRRR case through the shipped front door, digest and all.

``gpuwm.source_cli --source hrrr`` binds the GRIB2 it reads by the
digest of their ``SHA256SUMS`` manifest.  That manifest is written by the
download step, so its digest cannot be in a plan written before the
download -- and a plan that carried a stale one would bind the wrong
bytes, which is precisely the failure the digest exists to prevent.

This computes it, from the file the download step actually wrote,
immediately before handing the assembled argument list to
``gpuwm.source_cli.main()``.  The list is validated through the CLI's
own parser and its own ``--source hrrr`` requirement check first, so a
missing flag costs a message rather than a queue slot.

Nothing about the preparation is reimplemented here.  This adds one
digest and subtracts one class of mistake.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(argv: list[str]) -> None:
    """Parse and requirement-check with ``gpuwm.source_cli``'s own code."""

    from gpuwm.source_cli import _parser, _required_hrrr_args

    args = _parser().parse_args(argv)
    if args.source != "hrrr":
        raise SystemExit(
            f"this arm prepares an HRRR case and --source is {args.source!r}")
    problems = _required_hrrr_args(args)
    if problems:
        raise SystemExit(
            "gpuwm.source_cli would refuse this argument list:\n  - "
            + "\n  - ".join(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.prepare_case",
        description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True,
                        help="the SHA256SUMS the download step wrote; its "
                             "digest becomes --source-sha256s-sha256")
    parser.add_argument("--validate-only", action="store_true",
                        help="assemble and parse the argument list, then "
                             "stop; a placeholder digest stands in when "
                             "the manifest does not exist yet")
    parser.add_argument("cli_argv", nargs=argparse.REMAINDER,
                        help="-- followed by gpuwm.source_cli flags, "
                             "WITHOUT --source-sha256s-sha256")
    args = parser.parse_args(argv)

    tail = list(args.cli_argv)
    if tail and tail[0] == "--":
        tail = tail[1:]
    if not tail:
        parser.error("no source_cli arguments after --")
    if "--source-sha256s-sha256" in tail:
        parser.error(
            "--source-sha256s-sha256 is computed here from --manifest; "
            "giving it as well would be two statements of one binding")

    if args.manifest.is_file():
        digest = sha256_file(args.manifest)
    elif args.validate_only:
        digest = "0" * 64
    else:
        raise SystemExit(
            f"{args.manifest} does not exist: the download step that "
            "writes it has not run, or did not finish")
    assembled = tail + ["--source-sha256s-sha256", digest]
    validate(assembled)
    if args.validate_only:
        print("source_cli argument list parses and satisfies every "
              "--source hrrr requirement"
              + ("" if args.manifest.is_file() else " (placeholder digest)"))
        return 0

    from gpuwm.source_cli import main as source_main

    print(f"source manifest sha256 {digest}", flush=True)
    return int(source_main(assembled) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
