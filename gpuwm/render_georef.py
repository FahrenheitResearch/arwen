"""``render-georef.json``: one record per picture, in the folder the pictures are in.

``rw_wrfbatch`` writes ``<out-dir>/render-georef.json`` and merges each
launch's pictures into what the folder already holds, keyed by the flat
name it drew each picture under.  The caller then does two things the
engine knows nothing about: it launches the engine again (once per grid,
per history file, and while a forecast runs, once per committed frame),
and it moves and renames every picture into
``<domain>/<product>/<valid-day>/``.  So the record the engine leaves
names files that are about to move.

WHAT BREAKAGE THIS PREVENTS (gate law): a map that reads the manifest to
place a picture found no entry for any picture of an earlier batch or of
another grid.  The engine drops an entry whose file has left its folder,
and the placement step had just moved every earlier batch's pictures out
of the flat folder, so on the next launch they were all dropped: three
d02 frames rendered through ``gpuwm.render`` delivered 6 pictures and
listed 2.

This module does the caller's half, and the delivered folder's record is
the one it writes.  The pictures of one launch are MOVED while holding
the same ``render-georef.json.lock`` the engine takes (created
exclusively, removed on release), and in that same hold their entries
are re-keyed to where each picture was filed.  No engine launch can merge
in between, so no launch ever sees a flat entry whose picture is
mid-move.  An entry for a picture drawn again replaces the old one, every
other entry is kept while its picture is on disk, and the file is
replaced through a temporary, so a reader never sees half a manifest.

Best effort, like every receipt: a manifest that cannot be merged is
reported on stderr and never fails a render that drew its pictures.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from gpuwm.render_layout import fs_path

GEOREF_FILENAME = "render-georef.json"
GEOREF_SCHEMA = "rustwx.render-georef/v1"
LOCK_SUFFIX = ".lock"

#: The engine's own numbers (``GEOREF_LOCK_STALE``/``GEOREF_LOCK_WAIT`` in
#: ``rw-wrfbatch``), so a lock either side left behind is treated alike.
_LOCK_STALE_SECONDS = 120.0
_LOCK_WAIT_SECONDS = 300.0

#: A long placement touches its lock this often (pictures moved), so a
#: hold that is alive is never mistaken for one a dead process left.
_LOCK_TOUCH_EVERY = 100

#: Windows refuses a create over a lock that is being deleted with access
#: denied, and that passes in moments.  A create refused this long with no
#: lock in the way is a folder that refuses writes, not another render.
_WINDOWS = os.name == "nt"
_REFUSED_GRACE_SECONDS = 2.0


def read(path) -> dict[str, Any] | None:
    """A manifest of this schema, or ``None`` for anything else."""

    try:
        document = json.loads(Path(fs_path(Path(path))).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != GEOREF_SCHEMA:
        return None
    panels = document.get("panels")
    absent = document.get("without_georeference")
    if not isinstance(panels, dict) or not isinstance(absent, list):
        return None
    return document


def _key(root: Path, path: Path) -> str | None:
    try:
        return Path(os.path.abspath(path)).relative_to(
            Path(os.path.abspath(root))).as_posix()
    except ValueError:
        return None


def rekey(batch: Mapping[str, Any], *, batch_dir: Path, outdir: Path,
          moves: Mapping[Path, Path]) -> dict[str, Any]:
    """The batch's entries keyed by where each picture was filed.

    ``moves`` maps the path the engine drew a picture at to the path it
    was delivered at.  Only those entries are returned: an entry of any
    other picture is not this batch's to move (a context frame drawn only
    as an accumulation baseline, or another launch's picture sharing the
    folder).
    """

    delivered = {}
    for drawn, filed in moves.items():
        old = _key(batch_dir, Path(drawn))
        new = _key(outdir, Path(filed))
        if old is not None and new is not None:
            delivered[old] = new
    panels = {delivered[key]: value
              for key, value in (batch.get("panels") or {}).items()
              if key in delivered}
    absent = []
    for entry in batch.get("without_georeference") or []:
        if isinstance(entry, dict) and entry.get("path") in delivered:
            absent.append({**entry, "path": delivered[entry["path"]]})
    return {"schema": GEOREF_SCHEMA,
            "generated_utc": batch.get("generated_utc"),
            "panels": panels, "without_georeference": absent}


def merge(base: Mapping[str, Any] | None, batch: Mapping[str, Any], *,
          root: Path) -> dict[str, Any]:
    """``base`` with ``batch`` folded in; entries whose picture left go.

    The same rule as the engine's own merge: a path the batch drew again
    replaces its earlier entry in whichever half held it, so one picture
    is never both placed and unplaced.
    """

    panels: dict[str, Any] = dict((base or {}).get("panels") or {})
    absent: dict[str, dict] = {
        entry["path"]: entry
        for entry in (base or {}).get("without_georeference") or []
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)}
    for key, value in (batch.get("panels") or {}).items():
        absent.pop(key, None)
        panels[key] = value
    for entry in batch.get("without_georeference") or []:
        panels.pop(entry["path"], None)
        absent[entry["path"]] = entry

    def present(key: str) -> bool:
        return Path(fs_path(root / key)).is_file()

    return {
        "schema": GEOREF_SCHEMA,
        "generated_utc": (batch.get("generated_utc")
                          or (base or {}).get("generated_utc")),
        "panels": {key: panels[key] for key in sorted(panels) if present(key)},
        "without_georeference": [absent[key] for key in sorted(absent)
                                 if present(key)],
    }


def _acquire(outdir: Path) -> Path:
    """The engine's lock protocol: exclusive create, stale takeover.

    Every way round the loop sleeps and checks the deadline.  Two of them
    used to skip both -- a lock whose age could not be read, and a stale
    lock that could not be removed -- so a folder that refused writes
    held a core at full load forever and never reached the warning.  A
    create refused for permission on a POSIX folder is not contention at
    all: it is raised at once, and the caller warns and moves on.  On
    Windows the same refusal is also how a lock being deleted answers, so
    it is waited out, but only for :data:`_REFUSED_GRACE_SECONDS` while
    no lock is there: past that the folder is the refusal, and waiting
    the whole lock deadline for it cost five minutes per fold.
    """

    lock = Path(fs_path(outdir / (GEOREF_FILENAME + LOCK_SUFFIX)))
    started = time.monotonic()
    refused_since = None
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(handle)
            return lock
        except FileExistsError:
            refused_since = None
        except PermissionError:
            if not _WINDOWS:
                raise
            if os.path.lexists(lock):
                refused_since = None
            else:
                now = time.monotonic()
                if refused_since is None:
                    refused_since = now
                elif now - refused_since > _REFUSED_GRACE_SECONDS:
                    raise
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            age = None
        if age is not None and age > _LOCK_STALE_SECONDS:
            try:
                lock.unlink()
                continue
            except FileNotFoundError:
                continue
            except OSError:
                pass
        if time.monotonic() - started > _LOCK_WAIT_SECONDS:
            raise TimeoutError(f"another render held {lock} for "
                               f"{_LOCK_WAIT_SECONDS:.0f} s")
        time.sleep(0.025)


def _release(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


@contextmanager
def _locked(outdir: Path) -> Iterator[Path]:
    lock = _acquire(outdir)
    try:
        yield lock
    finally:
        _release(lock)


def _replace(target: Path, document: Mapping[str, Any]) -> None:
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    Path(fs_path(temporary)).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    for attempt in range(200):
        try:
            os.replace(fs_path(temporary), fs_path(target))
            return
        except PermissionError:
            # A reader holding the file open on Windows refuses the
            # rename for a moment; the engine retries the same way.
            if attempt == 199:
                raise
            time.sleep(0.025)


def _write_merged(outdir: Path, batch: Mapping[str, Any] | None,
                  prior: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fold ``batch`` (and ``prior``) into the folder's manifest.  Lock held."""

    target = outdir / GEOREF_FILENAME
    current = read(target)
    base = (merge(prior, current, root=outdir)
            if prior and current else (current or prior))
    document = merge(base, batch or {}, root=outdir)
    _replace(target, document)
    return document


def _warn(target: Path, error: BaseException) -> None:
    print(f"render: warning: {target} was not updated ({error}); the "
          "pictures are drawn, the map placement record for this "
          "batch is missing", file=sys.stderr)


def fold(outdir, batch: Mapping[str, Any] | None, *,
         prior: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Merge an already re-keyed ``batch`` into ``outdir``'s manifest.

    For a caller whose pictures are already where they will stay, keyed
    relative to ``outdir`` (an early render drawn in a scratch folder and
    published under the same relative paths).  ``prior`` is what the
    folder held before an engine launch; its entries are kept too.
    Returns the manifest written, or ``None`` when there was nothing to
    write or it could not be written (said on stderr).
    """

    if batch is None:
        return None
    outdir = Path(outdir)
    try:
        Path(fs_path(outdir)).mkdir(parents=True, exist_ok=True)
        with _locked(outdir):
            return _write_merged(outdir, batch, prior)
    except (OSError, TimeoutError) as error:
        _warn(outdir / GEOREF_FILENAME, error)
        return None


def file_pictures(outdir, drawn: Sequence[Path],
                  place: Callable[[Path], Path], *,
                  batch_dir=None,
                  engine: Mapping[str, Any] | None = None,
                  drawn_at: Mapping[Path, Path] | None = None,
                  prior: Mapping[str, Any] | None = None) -> list[Path]:
    """Move one launch's pictures with ``place`` and record where they went.

    The one call a render path makes once the engine has exited.  The
    lock is held across the moves and the record, so the entries the
    engine wrote under the flat names are read while those files are
    still there, and the delivered names are written before any other
    launch can look.  ``batch_dir`` is the folder the engine drew into
    (``outdir`` unless it drew in a working store); ``engine`` is its
    manifest when the caller had to read it before that store went away;
    ``drawn_at`` maps a path handed to ``place`` back to the path the
    engine drew it at, when the caller moved it once already.  ``prior``
    is what the folder held before the launch, kept in case the engine
    that ran replaced the file rather than merging into it.

    Returns where each picture now is, in order.  A record that cannot
    be written is said on stderr and the pictures are still filed.
    """

    outdir = Path(outdir)
    batch_dir = Path(batch_dir) if batch_dir is not None else outdir
    drawn = list(drawn)
    try:
        Path(fs_path(outdir)).mkdir(parents=True, exist_ok=True)
        lock = _acquire(outdir)
    except (OSError, TimeoutError) as error:
        _warn(outdir / GEOREF_FILENAME, error)
        return [place(png) for png in drawn]
    try:
        if engine is None:
            engine = read(batch_dir / GEOREF_FILENAME)
        filed = []
        for count, png in enumerate(drawn, start=1):
            filed.append(place(png))
            if count % _LOCK_TOUCH_EVERY == 0:
                try:
                    os.utime(lock)
                except OSError:
                    pass
        if engine is None and prior is None:
            return filed
        origin = drawn_at or {}
        batch = None
        if engine is not None:
            batch = rekey(engine, batch_dir=batch_dir, outdir=outdir,
                          moves={origin.get(png, png): final
                                 for png, final in zip(drawn, filed)})
        try:
            _write_merged(outdir, batch, prior)
        except OSError as error:
            _warn(outdir / GEOREF_FILENAME, error)
        return filed
    finally:
        _release(lock)


__all__ = ["GEOREF_FILENAME", "GEOREF_SCHEMA", "file_pictures", "fold",
           "merge", "read", "rekey"]
