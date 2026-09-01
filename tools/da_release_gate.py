"""Hold a queue until another lane's status trail says it has finished.

Cross-lane sequencing, as a plan step rather than as a launcher's private
sleep loop.  A queue arm that declares ``needs_gpu: false`` and runs this
first is a WAIT the plan states out loud: it is visible in the plan, it
is validated with every other step, it writes the same ``.done`` marker
as any arm, and a rerun skips it.

**Why existence is not enough.**  Waiting on "the file appeared" is right
for a trail that is written once, at the end.  It is wrong for a trail
that is APPENDED to as a run progresses: the file exists from the first
line, so an existence-waiter fires immediately and the two lanes collide
exactly as if there were no gate.  So this waits for a terminal MARKER in
the content, and then for the file to stop changing -- a trail that is
still being written has not finished, whatever its last line happens to
say at the instant it was read.

**Why the marker is an argument with a documented default.**  The lane
that writes the trail owns its own vocabulary and may change it.  A
pattern hard-coded here would be a second copy of someone else's contract,
silently drifting.  The default is deliberately broad across the tokens
these trails use for "finished", and the pattern that was actually
applied is printed and recorded, so a reader can always tell what the
gate was watching for.

**Restoring a file a sibling moved.**  ``--restore-if-missing`` exists for
one specific, real hazard: two lanes waiting on the SAME handover file,
where the first to notice it renames it and the second then waits for
something that no longer exists.  Restoring it turns a permanent hang
into a delay of at most one poll for whoever is still watching.  This is
narrow and ugly and it is here because the alternative is a release that
never starts.

Nothing here kills, stops or signals another process.  It waits, and it
says what it is waiting for.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

#: Lines that are ABOUT a terminal state without BEING one.  This is the
#: whole difficulty of matching someone else's trail: the arming line of
#: the very trail this gate was written for reads
#:
#:     armed; waiting for A/B VERDICT at ...\VERDICT.txt
#:
#: which contains the word VERDICT twice and means the exact opposite of
#: finished.  A completion token is only believed on a line that is not
#: announcing an intention.
_NOT_YET = r"(?!.*\b(?:wait|waiting|waits|armed|arming|pending|queued|"
_NOT_YET += r"will|before|until|expects?|expecting|watching|polling)\b)"

#: Tokens these status trails use to mean "this lane is finished with the
#: card", accepted only on a line that is not announcing an intention.
#: Broad within that constraint on purpose -- a gate that MISSES a
#: terminal marker holds a card nobody is using until its own deadline,
#: and the per-step GPU admission check downstream catches any residual
#: overlap anyway.  A gate that fires EARLY is the expensive mistake, so
#: the intention filter above is the part that matters.
DEFAULT_TERMINAL = (
    r"(?im)^" + _NOT_YET + r".*?\b("
    r"marker[-\s]?gate\b[^\n]*\b(complete|completed|done|passed|pass|ok)\b"
    r"|gate[-\s]?complete|suite[-\s]?complete|all[-\s]green"
    r"|release[-\s]?ready|finished\b|complete[d]?\b|done\b"
    r")")

#: A trail still being appended to has not finished, whatever its last
#: line says.  This is the settle window after the marker is first seen.
DEFAULT_QUIET_SECONDS = 180.0


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def say(message: str, log: Path | None) -> None:
    line = f"{_stamp()} {message}"
    print(line, flush=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_text(path: Path) -> str:
    """The trail's content, tolerating a partial write and a BOM."""

    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except (OSError, ValueError):
        return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_release_gate",
        description=__doc__.splitlines()[0])
    parser.add_argument("--status-file", type=Path, required=True,
                        help="the other lane's status trail")
    parser.add_argument("--terminal-pattern", default=DEFAULT_TERMINAL,
                        help="regex whose match means that lane is done "
                             "with the card (default: the shared "
                             "finished-token set, printed at start)")
    parser.add_argument("--quiet-seconds", type=float,
                        default=DEFAULT_QUIET_SECONDS,
                        help="after the marker is seen, the file must stop "
                             "changing for this long; a trail still being "
                             "appended to has not finished")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--restore-if-missing", action="append", default=[],
                        metavar="SRC=DST",
                        help="if DST does not exist and SRC does, copy SRC "
                             "back to DST. For a handover file a sibling "
                             "queue renamed while another lane was still "
                             "waiting on it. Repeatable")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--on-timeout", default="proceed",
                        choices=("proceed", "refuse"),
                        help="what a deadline means: proceed (the per-step "
                             "GPU gate still protects the card) or refuse "
                             "(fail this step and the arm). Default "
                             "proceed, because a gate that never releases "
                             "holds a card nobody is using")
    args = parser.parse_args(argv)

    log = args.log
    say(f"release gate: watching {args.status_file}", log)
    say(f"release gate: terminal pattern {args.terminal_pattern!r}", log)
    say(f"release gate: settle window {args.quiet_seconds:.0f} s", log)

    # Restore first and unconditionally: whoever is waiting on that file
    # is waiting right now, and every poll it stays missing is a poll
    # they lose.
    for spec in args.restore_if_missing:
        if "=" not in spec:
            raise SystemExit(f"--restore-if-missing wants SRC=DST, got "
                             f"{spec!r}")
        src, dst = (Path(part) for part in spec.split("=", 1))
        if dst.exists():
            continue
        if not src.is_file():
            say(f"release gate: {dst.name} is missing and {src.name} is "
                "not there to restore it from", log)
            continue
        try:
            shutil.copy2(src, dst)
            say(f"release gate: restored {dst} from {src} -- another lane "
                "was waiting on it", log)
        except OSError as error:
            say(f"release gate: could not restore {dst}: {error}", log)

    pattern = re.compile(args.terminal_pattern)
    deadline = time.monotonic() + float(args.max_hours) * 3600.0
    seen_at: float | None = None
    last_size = -1
    last_mtime = -1.0

    while True:
        if not args.status_file.is_file():
            say(f"release gate: {args.status_file} does not exist yet", log)
        else:
            text = read_text(args.status_file)
            try:
                stat = args.status_file.stat()
                size, mtime = stat.st_size, stat.st_mtime
            except OSError:
                size, mtime = -1, -1.0
            match = pattern.search(text)
            if match is None:
                seen_at = None
            else:
                changed = (size != last_size or mtime != last_mtime)
                if seen_at is None or changed:
                    if seen_at is None:
                        say("release gate: terminal marker seen -- "
                            f"{match.group(0).strip()[:120]!r}; settling",
                            log)
                    else:
                        say("release gate: the trail changed after its "
                            "terminal marker; the settle window restarts",
                            log)
                    seen_at = time.monotonic()
                elif time.monotonic() - seen_at >= float(args.quiet_seconds):
                    say("release gate: that lane is finished with the card; "
                        "releasing", log)
                    return 0
            last_size, last_mtime = size, mtime

        if time.monotonic() > deadline:
            if args.on_timeout == "refuse":
                say("release gate: deadline reached and --on-timeout is "
                    "refuse; this arm will not run", log)
                return 1
            say("release gate: deadline reached; proceeding, and the "
                "per-step GPU admission check still stands between this "
                "queue and a card someone else holds", log)
            return 0
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    sys.exit(main())
