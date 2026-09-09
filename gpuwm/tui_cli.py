"""Launch the packaged Rust terminal workspace with this Python installation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from gpuwm import bridges


TUI_NAME = "arwen-tui"
TUI_ENV = "GPUWM_TUI_BIN"
TUI_CRATE_RELATIVE = "tools/arwen-tui"


def tui_candidates() -> tuple[Path, ...]:
    """Use the native artifact ladder, including this crate's source builds."""
    filename = bridges.executable_name(TUI_NAME)
    root = Path(__file__).resolve().parent.parent
    crate = root / TUI_CRATE_RELATIVE
    override = os.environ.get(TUI_ENV)
    return (
        *((Path(override),) if override else ()),
        crate / "target" / "release" / filename,
        crate / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        bridges.packaged_bridge_dir() / filename,
        bridges.default_bridge_dir() / filename,
    )


def require_tui() -> Path:
    """Resolve, check the release pin, and check the terminal CLI contract."""
    override = os.environ.get(TUI_ENV)
    for candidate in tui_candidates():
        if candidate.is_file():
            binary = bridges.accept_resolved(candidate.resolve())
            matches, detail = bridges.bridge_abi_matches(TUI_NAME, binary)
            if matches:
                return binary
            reason = f"{binary}: {detail}"
            break
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{TUI_ENV} names a missing file: {candidate}. "
                f"Point it at {TUI_NAME}, or unset {TUI_ENV} to use the installed tools.")
    else:
        reason = "the Rust terminal workspace is not installed"
    raise FileNotFoundError(
        reason + ".\n" + bridges.artifact_remedy(
            env_var=TUI_ENV, filename=bridges.executable_name(TUI_NAME),
            subject="the Rust terminal workspace", crate_relative=TUI_CRATE_RELATIVE,
            artifact=TUI_NAME))


def tui_main(args) -> int:
    # Pass paths as individual arguments. Inherit the user's terminal and
    # working directory; the installed interpreter owns every engine job.
    try:
        command = [str(require_tui()), "--python", sys.executable]
        for name in ("config", "output", "prepared", "geog_root", "snapshot",
                     "snapshot_width", "snapshot_height", "snapshot_screen"):
            value = getattr(args, name, None)
            if value is not None:
                command.extend(("--" + name.replace("_", "-"), str(value)))
        # The terminal spawns every engine worker from this interpreter with
        # -P; PYTHONSAFEPATH covers the same ground for anything else it runs,
        # so a gpuwm/ folder in the launch directory never shadows the engine.
        environment = {**os.environ, "PYTHONSAFEPATH": "1"}
        return subprocess.run(command, check=False, shell=False, env=environment).returncode
    except (OSError, bridges.StaleBridgeError) as error:
        print(f"gpuwm tui: {error}", file=sys.stderr)
        return 2


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "tui", help="open the Rust terminal workspace",
        description="Open the terminal workspace using this installed Python environment. "
                    "Opening it starts no forecast.")
    parser.add_argument("--config", type=Path, metavar="FILE",
                        help="open an existing configuration")
    parser.add_argument("--output", type=Path, metavar="DIR",
                        help="initial output directory")
    parser.add_argument("--prepared", type=Path, metavar="DIR",
                        help="initial prepared forecast directory")
    parser.add_argument("--geog-root", type=Path, metavar="DIR",
                        help="initial WPS geography directory")
    parser.add_argument("--snapshot", type=Path, metavar="FILE.html",
                        help="write styled HTML and a .cells.json native terminal capture, then exit")
    parser.add_argument("--snapshot-width", type=int, metavar="COLUMNS",
                        help="terminal columns for --snapshot (native default 120)")
    parser.add_argument("--snapshot-height", type=int, metavar="ROWS",
                        help="terminal rows for --snapshot (native default 36)")
    parser.add_argument("--snapshot-screen", metavar="SCREEN",
                        help="screen for --snapshot: home, overview, modes, mode:ID, research:ID, scenario, "
                             "settings, logs, plots, domains, nodes, help, guide, or current")
    parser.set_defaults(func=tui_main)
