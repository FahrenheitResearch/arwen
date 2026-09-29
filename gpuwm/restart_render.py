"""History context for pictures produced by a resumed forecast."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: A history frame's valid time, as the history writer names it.
_FRAME_VALID = re.compile(
    r"_d\d{2}_(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})$")


def _frame_valid(frame: Path) -> datetime | None:
    """The valid time a frame's name carries, or ``None``."""
    match = _FRAME_VALID.search(Path(frame).name)
    if match is None:
        return None
    day, hour, minute, second = match.groups()
    return datetime.strptime(f"{day} {hour}:{minute}:{second}",
                             "%Y-%m-%d %H:%M:%S")


def history_before_restart(checkpoint) -> tuple[Path, ...]:
    """The history frames the run saved up to its checkpoint, never later ones.

    A resumed segment writes its frames into a new folder, starting one
    history interval after its checkpoint.  Its first rainfall pictures
    (``qpf_1h``, and every window that closes there) still need the frame
    before them, which the earlier segment wrote beside the checkpoint:
    without it the first new hour of a resumed forecast drew no rainfall
    at all, while the receipt said complete.  Both history layouts are
    read, the run folder itself and its ``wrfout/`` folder.  A checkpoint
    with no saved history beside it has nothing to supply.

    This is optional context for pictures.  The forecast owns checkpoint
    validation, so a header that cannot be read here gives no context
    instead of a second refusal of the restart.
    """
    if checkpoint is None:
        return ()
    from gpuwm.io.restart import read_restart_header
    from gpuwm.resume import discover_checkpoint_sets

    checkpoint = Path(checkpoint).resolve()
    try:
        header = read_restart_header(checkpoint)
        if header.get("domain_start_time") is not None:
            start = datetime.fromisoformat(header["domain_start_time"])
            offset = (float(header.get("domain_start_ticks", 0))
                      / float(header["tick_den"]))
            cutoff = start + timedelta(
                seconds=float(header["elapsed_seconds"]) - offset)
            if cutoff.tzinfo is not None:
                cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            cutoff = next(item.valid_time
                          for item in discover_checkpoint_sets(checkpoint.parent)
                          if checkpoint in item.members.values())
    except Exception:  # noqa: BLE001 - the restore refuses a bad checkpoint by name
        return ()
    frames = set()
    for folder in (checkpoint.parent, checkpoint.parent / "wrfout"):
        try:
            candidates = list(folder.glob("wrfout_d*"))
        except OSError:
            continue
        for path in candidates:
            valid = _frame_valid(path)
            if valid is not None and valid <= cutoff and path.is_file():
                frames.add(path)
    return tuple(sorted(frames, key=lambda path: path.name))


def hour_before(frame, history) -> list[Path]:
    """The saved frames of the hour ``frame`` closes, oldest first.

    On a whole-hour frame: its grid's latest whole-hour frame in
    ``history`` before it and every saved frame of that grid after that
    one, which is what the per-frame render of an unbroken run holds when
    it draws the same frame (``LiveProducts._closing_frames``).  Empty
    for a frame between hours, and when the saved history holds no
    whole-hour frame of its grid.  Longer windows are drawn by the
    end-of-run pass over the grid's whole series.
    """
    from gpuwm.live_products import _grid_and_whole_hour

    frame = Path(frame)
    parsed = _grid_and_whole_hour(frame)
    if parsed is None or not parsed[1]:
        return []
    grid = parsed[0]
    own = []
    for path in history:
        path = Path(path)
        other = _grid_and_whole_hour(path)
        if other is not None and other[0] == grid and path.name < frame.name:
            own.append((path, other[1]))
    own.sort(key=lambda item: item[0].name)
    wholes = [path for path, whole in own if whole]
    if not wholes:
        return []
    return [path for path, _whole in own if path.name >= wholes[-1].name]
