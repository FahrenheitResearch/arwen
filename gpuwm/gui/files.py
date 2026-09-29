"""Paths the GUI server may touch, and nothing else.

Every run is ``--root``/NAME or ``--root``/FOLDER/NAME.  A NEW name this
page writes is a plain name (:data:`NAME_RE`); a folder that is already
there is opened by the name it has, spaces, accents and length included,
because the engine and a terminal write those and the list shows them.  A
path is resolved (links followed) and must stay under the resolved root,
so neither ``..`` nor a link reaches outside it.  The same rule confines
files served from a run folder and from ``static/``.

Files from a run folder are data, never page code: only pictures, JSON
and plain text are shown inline, and everything else is sent as a
download, so nothing written into a run folder runs with the page's
origin.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any

from gpuwm.explain import EXPLAIN_MARK

_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\|/(?:home|Users|tmp|mnt|var|opt|root|srv|media|private)/)[^\s'\"]*")


#: The one way the page writes a moment: "2026-09-24 18:00 UTC" (the time the map viewer and Review show).
UTC_FORMAT = "%Y-%m-%d %H:%M UTC"


def utc_text(moment) -> str:
    """A UTC datetime as the page writes it."""

    return moment.strftime(UTC_FORMAT)


#: What stands in on the page for a machine path it takes out.  The words name the page server's log, so the
#: server writes the text the path came from there (:func:`scrub`).
PATH_WORDS = "(a folder named in the page server's log)"
#: Texts already written to the log, so a failed run's state, read every few seconds, is logged once.
_LOGGED: set[str] = set()
_LOGGED_KEEP = 4096
_LOGGED_LOCK = threading.Lock()


def _log_whole(text: str) -> None:
    text = text.replace(EXPLAIN_MARK, "\n").strip()
    with _LOGGED_LOCK:
        if text in _LOGGED:
            return
        if len(_LOGGED) >= _LOGGED_KEEP:
            _LOGGED.clear()
        _LOGGED.add(text)
    print(f"gpuwm gui: {text}", file=sys.stderr, flush=True)


def scrub(text: str, keep=(), *, whole: str | None = None) -> str:
    """A message with every machine path taken out; the page server's log gets the text with its paths.

    The page shows words, and the words that stand in for a path say the page server's log names it, so a
    text that loses a path is written to that log (the terminal running ``gpuwm gui``) whole, once:
    ``whole`` when the page shows only a part of a longer text.  Before this only a refused request was
    logged, and a run that failed on a missing folder sent its reader to a log that held no such folder.

    ``keep`` names folders the message is ABOUT (a refusal's scratch folder
    to make room in): a path in or under one of them stays, because a
    remedy that says "free space in (a folder named in the page server's
    log)" names nowhere to act.
    """

    text = str(text or "")
    # A folder that strips to nothing (the root) would keep every path.
    kept = [name for name in (str(folder).strip().rstrip("/\\") for folder in keep) if name]

    def replace(match) -> str:
        path = match.group(0)
        if any(path == folder or path.startswith((folder + "/", folder + "\\"))
               or path.rstrip(".,;:)") == folder for folder in kept):
            return path
        return PATH_WORDS

    shown = _PATH.sub(replace, text)
    if shown != text:
        _log_whole(str(whole or text))
    return shown


def plain_message(text: Any, keep=(), *, paragraph: bool = False) -> str | None:
    """An engine message as the page shows it: one line, before any shell comment, with no machine path.

    A message is its first line.  A remedy (``paragraph``) is its first paragraph with its lines joined:
    an engine remedy wraps across lines, and its first line alone could stop partway through the command it
    names.  A remedy written for a terminal keeps the words of its whole-line comments and drops its
    ``remedy:`` label: the GPU runtime's remedy is such a block (which CUDA major takes which extra), and
    cutting at its first ``#`` left a run that failed for want of CuPy with no remedy on the page at all.
    The engine's full text stays in the run's event log and command log, and the page server's log gets it
    whenever a path was taken out (:func:`scrub`).  ``keep`` is :func:`scrub`'s: the folders the message
    names as the place to act.
    """

    if not text:
        return None
    whole = str(text).strip()
    if paragraph:
        kept = []
        for line in re.split(r"\n\s*\n", whole)[0].splitlines():
            line = line.strip()
            line = line.lstrip("#").strip() if line.startswith("#") else line.split(" # ")[0].strip()
            if line.lower().startswith("remedy:"):
                line = line[len("remedy:"):].strip()
            kept.append(line)
        first = " ".join(" ".join(kept).split())
    else:
        first = whole.splitlines()[0].split(" # ")[0].split("# ")[0].strip()
    return scrub(first, keep, whole=whole) or None


#: The rule for a name this page CREATES (a new run, a machine, a conversation).  Never
#: the rule for opening a folder that exists: see :func:`split_run_id`.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
#: Run ids are NAME or FOLDER/NAME under the root.
SCAN_DEPTH = 2
#: The server's own state under the root (drafts); never listed as a run.
SERVER_DIR = ".arwen-gui"
JSON_LIMIT = 16 * 1024 * 1024

STATIC_DIR = Path(__file__).resolve().parent / "static"
COPY_DIR = Path(__file__).resolve().parent / "copy"

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".toml": "text/plain; charset=utf-8",
    ".geojson": "application/json; charset=utf-8",
    ".woff2": "font/woff2",
}
#: Types a run file may be shown inline as; everything else downloads.
RUN_FILE_INLINE = ("image/png", "application/json", "text/plain")


class PathRefused(ValueError):
    """A name or path the server will not touch; the message says why."""


def require_name(part: str, what: str = "name") -> str:
    if not isinstance(part, str) or not NAME_RE.match(part):
        raise PathRefused(
            f"The {what} {part!r} is not allowed. Use letters, digits, '.', '_' or '-' "
            "(up to 64, starting with a letter or digit).")
    return part


#: Names Windows opens as a device in any folder (``nul.txt`` included), whatever follows the first dot.
WINDOWS_DEVICES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in "123456789¹²³"),
                             *(f"LPT{i}" for i in "123456789¹²³")})


def windows_hazard(part: str) -> bool:
    """A name no Windows folder can have, which Windows reads as something else: a device, a stream, a drive.

    ``CON`` or ``nul.txt`` opens the device, ``run:stream`` a stream of the
    file ``run``, and ``<>"|?*`` and control characters are not names at all.
    """

    stem = part.split(".", 1)[0].rstrip(" ").upper()
    return stem in WINDOWS_DEVICES or any(char in ':<>"|?*' or ord(char) < 32 or ord(char) == 127 for char in part)


def folder_part(part: str, *, windows: bool = os.name == "nt") -> bool:
    """One segment of an existing run's id: a single folder name, never a way out of its parent.

    Anything a file system can name is allowed except what the run list
    never shows (dot folders, a render's scratch) and what would not stay
    one segment: ``.``, ``..``, a separator, a drive or a NUL.  On Windows
    a name that is a device, a stream or a drive there is refused too (no
    folder there can have one, so the list never shows one); elsewhere such
    a name is an ordinary folder and opens like any other.
    """

    from gpuwm.render_layout import is_scratch_dir

    if not part or part in (".", "..") or "/" in part or "\\" in part or chr(0) in part or is_scratch_dir(part):
        return False
    if windows and windows_hazard(part):
        return False
    pure = Path(part)
    return not pure.anchor and pure.name == part


def split_run_id(run_id: str) -> list[str]:
    """The folder names of an existing run's id (NAME or FOLDER/NAME), as the run list wrote it.

    Lookup, not creation: a folder named ``May 20 Oklahoma`` or ``München-2026``
    is listed, so it opens.  Containment under the root is checked by the
    caller once the path is joined (:func:`gpuwm.gui.runs.run_path`).
    """

    parts = str(run_id).split("/")
    if not 1 <= len(parts) <= SCAN_DEPTH or not all(folder_part(part) for part in parts):
        raise PathRefused(f"{run_id!r} is not a run in this folder (NAME or FOLDER/NAME).")
    return parts


def within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    return True


def content_type(path: Path) -> str:
    return _TYPES.get(path.suffix.lower(), "application/octet-stream")


def run_file_type(path: Path) -> tuple[str, bool]:
    """(type, inline) for a file served out of a run folder."""

    kind = content_type(path)
    if kind.startswith(RUN_FILE_INLINE):
        return kind, True
    return "application/octet-stream", False


def download_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]', "_", name) or "file"


def static_file(relative: str) -> Path:
    candidate = (STATIC_DIR / relative).resolve()
    if not within(STATIC_DIR, candidate) or not candidate.is_file():
        raise FileNotFoundError(relative)
    return candidate


def long_path(path: Path) -> Path:
    """``path`` in a form Windows opens past its 260-character limit.

    A run under a deep root holds pictures whose full path is longer
    than MAX_PATH; the listing showed them but opening one failed, so
    the page answered 404 for a picture it had just listed.  Elsewhere
    the path is returned as it is.
    """

    if os.name != "nt":
        return path
    prefix = "\\\\?\\"
    text = os.path.abspath(str(path))
    if text.startswith(prefix):
        return Path(text)
    if text.startswith("\\\\"):
        return Path(prefix + "UNC\\" + text[2:])
    return Path(prefix + text)


def safe_file(base: Path, relative: str) -> Path:
    """A file under ``base`` named by a slash path, or PathRefused."""

    parts = [part for part in str(relative).split("/") if part]
    if not parts or any(part in (".", "..") or part.startswith(".") for part in parts):
        raise PathRefused(f"{relative!r} is not a file of this run.")
    candidate = base.joinpath(*parts)
    if not within(long_path(base), long_path(candidate)):
        raise PathRefused(f"{relative!r} leads outside the run folder.")
    candidate = long_path(candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"No file {relative!r} in this run.")
    return candidate


def _finite(text: str) -> float | None:
    value = float(text)
    return value if math.isfinite(value) else None


def load_json(text: str | bytes) -> Any:
    """JSON as a run folder holds it, with every number that is not finite read as null.

    Python writes ``NaN`` and ``Infinity`` into JSON and reads them back, and
    ``1e400`` reads as infinity; a copied plan or an event holding one made
    the page's own reply invalid JSON (or a sum infinite) and the run's
    details answered 500.  Read as null, the value is unknown and the rest
    of the document still reads.
    """

    return json.loads(text, parse_float=_finite, parse_constant=lambda _name: None)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.stat().st_size > JSON_LIMIT:
            return default
        return load_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return default


def write_json(path: Path, document: Any) -> None:
    """Atomic JSON publication with LF endings."""

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


__all__ = [
    "COPY_DIR", "NAME_RE", "PathRefused", "SERVER_DIR", "STATIC_DIR",
    "content_type", "download_name", "folder_part", "load_json", "read_json", "require_name",
    "run_file_type", "safe_file", "split_run_id", "static_file", "within",
    "write_json",
]
