"""Run differences: every requested product drawn as run A minus run B.

``gpuwm render --diff A B`` pairs the two runs' history frames by valid
time and hands each pair to the Rust renderer (``rw_wrfbatch
--diff-against``), which imports both, draws each product from each run's
own inputs, subtracts, and draws the difference on a zero-centred bar.
This module is orchestration only. The render door supplies its existing
native-backed metadata reader so every record is paired by its actual valid
time. Each selected pair retains the compatible records from both runs as
context for windowed products. The renderer checks the time and grid again.

A run is a folder of ``wrfout_dNN_YYYY-MM-DD_HH:MM:SS`` frames (the colon
or the Windows underscore spelling), or a list of such files.  Frames are
paired on ``(domain, valid time)``; a frame with no partner in the other
run is reported, not dropped silently.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

#: The valid time a WRF history file's own name carries.
_WRFOUT_TIME = re.compile(
    r"^wrfout_(?P<grid>d\d{2})_(?P<date>\d{4}-\d{2}-\d{2})[_T]"
    r"(?P<h>\d{2})[:_](?P<m>\d{2})[:_](?P<s>\d{2})")


def wrfout_valid_time(path) -> tuple[str, _dt.datetime] | None:
    """``(domain, valid time)`` from a wrfout NAME, or ``None``."""

    match = _WRFOUT_TIME.match(Path(path).name)
    if match is None:
        return None
    try:
        valid = _dt.datetime.strptime(
            f"{match.group('date')} {match.group('h')}:{match.group('m')}:"
            f"{match.group('s')}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return match.group("grid"), valid


def run_frames(run) -> list[Path]:
    """The history frames of one run: a folder's ``wrfout_d*`` files (in
    time order, lifecycle episodes included) or the files given."""

    if isinstance(run, (str, Path)):
        path = Path(run)
        if path.is_dir():
            from gpuwm import render_layout

            return [frame.path for frame in render_layout.history_frames(path)]
        return [path]
    return [Path(item) for item in run]


@dataclass
class FramePairing:
    """Run A's and run B's frames matched on (domain, valid time)."""

    pairs: list[tuple[str, _dt.datetime, Path, Path]] = field(default_factory=list)
    only_a: list[Path] = field(default_factory=list)
    only_b: list[Path] = field(default_factory=list)
    contexts: dict = field(default_factory=dict)
    only_a_stamps: list[tuple[str, _dt.datetime]] = field(default_factory=list)
    only_b_stamps: list[tuple[str, _dt.datetime]] = field(default_factory=list)


def _metadata_frames(frames, run, reader, domain_reader):
    indexed, groups, axes, file_axes = {}, {}, {}, {}
    for raw in frames:
        path = Path(raw)
        identity, stamps = reader(path)
        stamps = tuple(stamps)
        if not stamps or any(left >= right for left, right in zip(stamps, stamps[1:])):
            raise ValueError(f"run {run}: {path} has empty or unordered valid times")
        domain = domain_reader(path) if domain_reader is not None else None
        if domain is None:
            named = wrfout_valid_time(path)
            domain = named[0] if named is not None else None
        if domain is None:
            raise ValueError(f"run {run}: {path} declares no GRID_ID or wrfout domain")
        groups.setdefault(identity, []).append(path)
        axes.setdefault(identity, set()).update(stamps)
        file_axes[path] = stamps
        for stamp in stamps:
            key = domain, stamp
            if key in indexed:
                raise ValueError(
                    f"run {run}: {indexed[key][0]} and {path} are both {domain} "
                    f"valid {stamp:%Y-%m-%d %H:%MZ}; one run holds one frame "
                    "per domain and valid time")
            indexed[key] = path, identity
    ordered_axes = {key: tuple(sorted(stamps)) for key, stamps in axes.items()}
    for identity, paths in groups.items():
        paths.sort(key=lambda path: file_axes[path][0])
    return indexed, groups, ordered_axes, file_axes


def _index(frames: Iterable[Path], run: str) -> dict:
    index: dict = {}
    for path in frames:
        key = wrfout_valid_time(path)
        if key is None:
            raise ValueError(
                f"run {run}: {path} does not carry a valid time in its name "
                "(wrfout_dNN_YYYY-MM-DD_HH:MM:SS); pass files named that way "
                "or pair the frames yourself")
        if key in index:
            raise ValueError(
                f"run {run}: {index[key]} and {path} are both {key[0]} valid "
                f"{key[1]:%Y-%m-%d %H:%MZ}; one run holds one frame per "
                "domain and valid time")
        index[key] = path
    return index


def pair_frames_by_valid_time(a_frames: Sequence[Path],
                              b_frames: Sequence[Path], *,
                              reader: Callable | None = None,
                              domain_reader: Callable | None = None,
                              series_groups: Callable | None = None,
                              timeidx: int | None = None,
                              series: bool = False) -> FramePairing:
    """Pair two runs' frames on (domain, valid time), in time order.

    Refuses by name a frame whose name carries no valid time and a run
    holding two frames of one domain at one valid time.  Frames with no
    partner are returned in ``only_a`` / ``only_b`` so the caller can say
    which valid times were not differenced.
    """

    if reader is not None:
        return _pair_metadata_frames(a_frames, b_frames, reader, domain_reader,
                                     series_groups, timeidx, series)
    a_index = _index(a_frames, "A")
    b_index = _index(b_frames, "B")
    pairing = FramePairing()
    for key in sorted(a_index, key=lambda item: (item[1], item[0])):
        if key in b_index:
            pairing.pairs.append((key[0], key[1], a_index[key], b_index[key]))
        else:
            pairing.only_a.append(a_index[key])
    pairing.only_b = [b_index[key] for key in sorted(
        b_index, key=lambda item: (item[1], item[0])) if key not in a_index]
    return pairing


def _pair_metadata_frames(a_frames, b_frames, reader, domain_reader,
                          series_groups, timeidx, series):
    a_index, a_groups, a_axes, a_files = _metadata_frames(
        a_frames, "A", reader, domain_reader)
    b_index, b_groups, b_axes, b_files = _metadata_frames(
        b_frames, "B", reader, domain_reader)
    if series_groups is not None:
        # Use the existing series authority, including earlier overlapping
        # placements of a moving nest. Its context is imported, never drawn
        # as a second copy of the selected record.
        for frames, indexed, groups, axes, file_axes in (
                (a_frames, a_index, a_groups, a_axes, a_files),
                (b_frames, b_index, b_groups, b_axes, b_files)):
            identities = {path: identity for path, identity in indexed.values()}
            for paths, context in series_groups(frames):
                paths = [Path(path) for path in paths]
                hidden = {Path(path).resolve() for path in context}
                own = [path for path in paths if path.resolve() not in hidden]
                if not own:
                    continue
                identity = identities[own[0]]
                groups[identity] = paths
                axes[identity] = tuple(sorted({stamp for path in paths
                                               for stamp in file_axes[path]}))
    selected = set(a_index)
    if timeidx is not None:
        if timeidx < 0:
            raise ValueError("--timeidx must be non-negative")
        selected = set()
        choices = a_axes.items() if series else a_files.items()
        for owner, stamps in choices:
            if timeidx >= len(stamps):
                label = a_groups[owner][-1] if series else owner
                raise ValueError(f"run A: {label}: --timeidx {timeidx} out of "
                                 f"range; {'timeline' if series else 'file'} "
                                 f"has {len(stamps)} frame(s)")
            valid = stamps[timeidx]
            selected.update(key for key, (path, identity) in a_index.items()
                            if key[1] == valid and
                            (identity == owner if series else path == owner))
    pairing = FramePairing()
    for key in sorted(selected, key=lambda item: (item[1], item[0])):
        a_file, a_identity = a_index[key]
        if key not in b_index:
            pairing.only_a.append(a_file)
            pairing.only_a_stamps.append(key)
            continue
        b_file, b_identity = b_index[key]
        pairing.pairs.append((*key, a_file, b_file))
        pairing.contexts[key] = (a_groups[a_identity], b_groups[b_identity],
                                 a_axes[a_identity].index(key[1]))
    for key in sorted(b_index, key=lambda item: (item[1], item[0])):
        if key not in a_index:
            pairing.only_b.append(b_index[key][0])
            pairing.only_b_stamps.append(key)
    return pairing


def unpaired_notice(pairing: FramePairing) -> str | None:
    """One sentence naming the valid times only one run has, or ``None``."""

    parts = []
    for run, frames, actual in (
            ("A", pairing.only_a, pairing.only_a_stamps),
            ("B", pairing.only_b, pairing.only_b_stamps)):
        if frames:
            stamps = actual or [wrfout_valid_time(path) for path in frames]
            times = ", ".join(
                f"{domain} {stamp:%m/%d %H:%MZ}" for domain, stamp in stamps)
            parts.append(f"only run {run} has {times}")
    if not parts:
        return None
    return ("not differenced, because a difference needs both runs at one "
            "valid time: " + "; ".join(parts))


__all__ = ["FramePairing", "pair_frames_by_valid_time", "run_frames",
           "unpaired_notice", "wrfout_valid_time"]
