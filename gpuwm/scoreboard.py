"""Forecast verification orchestration. All weather bytes are read by Rust."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import json

from gpuwm.ml_export import Binary

ABI_MARKER = ("rw_scoreboard --request JSON schema=woof.scoreboard.request.v1 "
              "method=scoreboard-v1 actions=extract,observe,score,plan,merge,interpolate,accumulate,"
              "interpolate_to_reports columns=WFSBCOL1")
BINARY = Binary(name="rw_scoreboard", env_var="GPUWM_RW_SCOREBOARD",
                subject="the forecast verification scorer")


def abi_line(path: Path) -> str | None:
    """The scorer's ``--abi`` output, or None when it does not speak the
    frozen request contract (the refusal :func:`run_request` applies)."""
    try:
        probe = subprocess.run([str(path), "--abi"], capture_output=True,
                               text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode or ABI_MARKER not in probe.stdout:
        return None
    return probe.stdout


def probe(path: Path) -> tuple[bool, str]:
    """The door's own refusal, as a doctor probe."""
    line = abi_line(path)
    if line is None:
        return False, ("--abi does not carry the frozen scoreboard request contract; "
                       + BINARY.remedy())
    return True, "--abi speaks woof.scoreboard.request.v1, scoreboard-v1, interpolate_to_reports"


def run_request(request: Path) -> int:
    """Forward an immutable request to the native data path."""
    path = BINARY.find()
    if path is None:
        print(BINARY.remedy(), file=sys.stderr)
        return 3
    stdout = abi_line(path)
    if stdout is None:
        print("The scorer does not speak the frozen scoreboard request contract. "
              + BINARY.remedy(), file=sys.stderr)
        return 3
    if json.loads(request.read_text(encoding="utf-8"))["action"] == "nightly" and "additional_actions=nightly" not in stdout:
        print("This scorer has no nightly catalog action. " + BINARY.remedy(), file=sys.stderr)
        return 3
    return subprocess.run([str(path), "--request", str(request.resolve())]).returncode


def register_cli(sub) -> None:
    parser = sub.add_parser("scoreboard", help="extract and score forecast frames against observations")
    parser.add_argument("--request", type=Path, required=True,
                        help="woof.scoreboard.request.v1 JSON request for the Rust scorer")
    parser.set_defaults(func=lambda args: run_request(args.request))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    return run_request(parser.parse_args(argv).request)


if __name__ == "__main__":
    raise SystemExit(main())
