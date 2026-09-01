"""The anchor: one cycle boundary, on disk, in a versioned format.

The artifact this replaces is ``pickle.dump(HostDriverCheckpoint,
protocol=4)`` of live host objects.  That is a fine cache and a terrible
contract: only one Python family can read it, it does not survive a numpy
version change, it cannot be diffed, and a DA engine written in Rust --
or the renderer, or a receipt reader on another box -- cannot open it at
all.  The cycling spine needs the opposite of that, because the spine is
a SEPARATE PROCESS from both the dycore and the assimilator.  The MPAS
port pins gpuwm by SHA and re-verifies the checkout before and after
every run, so the spine cannot be a library the port imports.  What
crosses that boundary is files.

An anchor is therefore a directory, self-describing, hashed, and
committed atomically:

    <root>/anchors/anchor_<cycle:03d>_<YYYYmmdd_HHMMSS>/
      anchor.json              the manifest (:data:`ANCHOR_SCHEMA`)
      parent_prognostic.nc     rho, rho_theta, rho_u, rho_w, scalars,
                               time_seconds
      parent_derived.nc        exner and the rest of the carried
                               diagnostics -- NON-authoritative, stamped
                               with the prognostic sha they came from
      parent_seam.nc           the physics-seam mapping (optional)
      analysis_increment.nc    the DA increment in the anchor's own space
                               (optional)
      children/<grid_id>/state.nc
      COMMIT                   phase-two marker: every file with its sha

**The derived block is stamped, and that stamp is the whole point.**  A
DA increment rewrites ``rho``/``rho_theta``; the diagnostics carried
beside them then describe the pre-analysis atmosphere.  The model will
integrate that state happily and be wrong.  Every pre-existing gate on
the restart path is an IDENTITY gate -- does the restored state equal the
saved one -- and an increment must break identity by construction, so no
existing gate can see it.  :func:`gpuwm.cycle.consistency.derived_is_stale`
reads the stamp this module writes; that is the guard.

**Three-phase commit, borrowed from the one place in this program that
already gets atomicity right** (``gpuwm/ensemble/cycle.py``): everything
is written into ``anchor_...tmp/`` and fsynced; a ``COMMIT`` marker
naming every file with its sha256 is written and fsynced; only then is
the directory renamed into place.  A crash leaves a ``...tmp`` directory
that :func:`read_anchor` refuses by name -- never a half anchor that
reads like a whole one.

**Reading a directory listing to discover anchors is out of contract.**
Use :func:`latest_anchor` and :func:`anchor_for_cycle`; they only ever
return committed anchors.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from mpas_cycle_bridge import anchor_codec as codec

from gpuwm.cycle.contracts import (ANCHOR_SCHEMA, CHILD_STATES, CycleRefusal,
                                   PARENT_KINDS, TICK_HZ, ticks_to_seconds)

COMMIT_MARKER_NAME = codec.COMMIT_MARKER_NAME
MANIFEST_NAME = codec.MANIFEST_NAME
ANCHORS_DIRNAME = "anchors"

REQUIRED_PROGNOSTIC_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")

PROGNOSTIC_STEM = codec.PROGNOSTIC_STEM
DERIVED_STEM = codec.DERIVED_STEM
SEAM_STEM = codec.SEAM_STEM
INCREMENT_STEM = codec.INCREMENT_STEM

#: The anchor's ENCODING lives in ``mpas_cycle_bridge.anchor_codec``; its
#: POLICY -- required fields, legal parent kinds, the three-phase commit,
#: the refusals -- lives here.  The split is not tidiness: the MPAS
#: forecast worker runs in a process where importing gpuwm is forbidden
#: (the port pins gpuwm by commit and refuses a live one from another
#: tree), so it cannot import this module, yet it must read anchors and
#: write segments in exactly this format.  Sharing the encoding means
#: both sides run the same bytes-to-arrays code instead of a reader
#: tested against its own writer.
ARRAY_FORMAT = codec.ARRAY_FORMAT
_SUFFIX = codec.SUFFIX

_contiguous = codec.contiguous
prognostic_sha256 = codec.prognostic_sha256
_file_sha256 = codec.file_sha256
_write_arrays = codec.write_arrays
_read_arrays = codec.read_arrays
_fsync_file = codec.fsync_file


class AnchorRefusal(CycleRefusal):
    """An anchor refused to be written or read, and says what it saw."""


# --------------------------------------------------------------------------
# write-side guards


def _check_prognostic(prognostic: Mapping[str, np.ndarray],
                      *, anchor_ticks: int, cycle_index: int) -> None:
    missing = [name for name in REQUIRED_PROGNOSTIC_FIELDS
               if name not in prognostic]
    if missing:
        raise AnchorRefusal(
            "prognostic mapping is missing required fields; "
            "an anchor that cannot restart the parent is not an anchor",
            cycle_index=cycle_index, missing=missing,
            required=list(REQUIRED_PROGNOSTIC_FIELDS),
            present=sorted(prognostic))

    for name in sorted(prognostic):
        array = np.asarray(prognostic[name])
        if not np.issubdtype(array.dtype, np.floating):
            continue
        bad = ~np.isfinite(array)
        count = int(bad.sum())
        if count:
            first = int(np.flatnonzero(bad.reshape(-1))[0])
            raise AnchorRefusal(
                "prognostic field carries non-finite values; refusing to "
                "anchor a state the next cycle cannot integrate",
                cycle_index=cycle_index, field=name, count=count,
                first_flat_index=first, shape=tuple(array.shape))

    if "time_seconds" in prognostic:
        state_time = float(np.asarray(prognostic["time_seconds"]).reshape(-1)[0])
        expected = ticks_to_seconds(anchor_ticks)
        if abs(state_time - expected) > 0.5 / TICK_HZ:
            raise AnchorRefusal(
                "state time_seconds disagrees with anchor_ticks; the clock "
                "authority is the cycle clock, not the state file",
                cycle_index=cycle_index, anchor_ticks=int(anchor_ticks),
                expected_seconds=expected, state_time_seconds=state_time,
                tolerance_seconds=0.5 / TICK_HZ)


def _dimension_counts(prognostic: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Name the mesh dimensions by MEASURING them, not by assuming a layout.

    The first version of this read ``rho.shape[0]`` as the level count and
    ``rho.shape[-1]`` as the cell count, which silently encodes "the
    parent is a level-major box".  The port's own
    :class:`PrognosticState` is indeed level-major -- ``(nVertLevels,
    nCells)`` -- but a state extracted from a history file is
    cell-major, and the same code then published ``n_levels: 40962,
    n_cells: 55, n_edges: 55`` for an x1.40962 mesh that has 40,962
    cells, 55 levels and 122,880 edges.  Nothing computed was wrong --
    the hashes, the ingestion gate and the arming triple are all
    shape-agnostic -- but every future reader of that manifest was told
    three lies.

    The determination is made from the data and is not a convention:
    ``rho_w`` lives on interfaces, so it exceeds ``rho`` by exactly one
    along the VERTICAL axis and matches it along the entity axis.  That
    identifies the vertical axis with no ambiguity at any real mesh.
    ``rho_u`` then shares the vertical extent and carries edges on the
    other axis.  When the arrays do not support the determination the
    manifest says ``layout: "indeterminate"`` and leaves the counts
    ``None`` -- an absent number a reader can see beats a confident
    wrong one.
    """
    rho = np.asarray(prognostic["rho"])
    rho_w = np.asarray(prognostic["rho_w"])
    rho_u = np.asarray(prognostic["rho_u"])

    vertical = None
    if rho.ndim == 2 and rho_w.ndim == 2:
        matches = [axis for axis in (0, 1)
                   if rho_w.shape[axis] == rho.shape[axis] + 1
                   and rho_w.shape[1 - axis] == rho.shape[1 - axis]]
        if len(matches) == 1:
            vertical = matches[0]

    if vertical is None:
        return {"n_levels": None, "n_cells": None, "n_edges": None,
                "layout": "indeterminate",
                "vertical_axis": None,
                "dimension_note":
                    "rho_w does not exceed rho by one along exactly one "
                    "axis, so the vertical axis could not be measured; "
                    "counts are withheld rather than guessed",
                "shapes": {name: list(np.asarray(prognostic[name]).shape)
                           for name in ("rho", "rho_u", "rho_w")}}

    entity = 1 - vertical
    n_levels = int(rho.shape[vertical])
    n_cells = int(rho.shape[entity])
    n_edges: int | None = None
    if rho_u.ndim == 2 and rho_u.shape[vertical] == n_levels:
        n_edges = int(rho_u.shape[entity])
    return {"n_levels": n_levels, "n_cells": n_cells, "n_edges": n_edges,
            "layout": "level_major" if vertical == 0 else "cell_major",
            "vertical_axis": int(vertical)}


def _valid_time_text(valid_time: Any) -> tuple[str, str]:
    if isinstance(valid_time, datetime):
        stamp = valid_time
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (stamp.isoformat().replace("+00:00", "Z"),
                stamp.strftime("%Y%m%d_%H%M%S"))
    text = str(valid_time)
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnchorRefusal(
            "valid_time is not an ISO-8601 instant",
            valid_time=text, parse_error=str(error)) from None
    return text, stamp.strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------
# write


def write_anchor(root: str | os.PathLike[str], *, cycle_index: int,
                 anchor_ticks: int, valid_time: Any, parent_kind: str,
                 prognostic: Mapping[str, np.ndarray],
                 derived: Mapping[str, np.ndarray],
                 seam: Mapping[str, np.ndarray] | None = None,
                 mesh_id: str,
                 children: Iterable[Mapping[str, Any]] = (),
                 analysis: Mapping[str, Any] | None = None,
                 diagnostics_rebuilt: bool | None = None,
                 extra: Mapping[str, Any] | None = None) -> Path:
    """Publish one cycle boundary atomically and return its directory.

    ``children`` entries carry ``grid_id``, ``placement`` (the geographic
    request and whatever the resolver made of it), ``state`` (one of
    :data:`CHILD_STATES`) and, when the child is live, ``arrays``: the
    child's own state mapping.  ``analysis`` carries ``state``, the
    optional increment under ``arrays``, and the ingestion receipt block
    from :func:`gpuwm.cycle.ingestion.verify_ingestion` under
    ``ingestion``.
    """
    if parent_kind not in PARENT_KINDS:
        raise AnchorRefusal("unknown parent kind", parent_kind=parent_kind,
                            known=list(PARENT_KINDS))
    _check_prognostic(prognostic, anchor_ticks=int(anchor_ticks),
                      cycle_index=int(cycle_index))
    iso_time, stamp = _valid_time_text(valid_time)

    anchors = Path(root) / ANCHORS_DIRNAME
    anchors.mkdir(parents=True, exist_ok=True)
    name = f"anchor_{int(cycle_index):03d}_{stamp}"
    final = anchors / name
    staging = anchors / f"{name}.tmp"
    if final.exists():
        raise AnchorRefusal("anchor already published; the spine is "
                            "forward-only and never overwrites a boundary",
                            cycle_index=int(cycle_index), path=str(final))
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    suffix = _SUFFIX[ARRAY_FORMAT]
    written: dict[str, str] = {}

    def _emit(stem: str, mapping: Mapping[str, np.ndarray],
              subdir: Path | None = None) -> str:
        target_dir = staging if subdir is None else subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}{suffix}"
        _write_arrays(target, mapping)
        _fsync_file(target)
        relative = target.relative_to(staging).as_posix()
        written[relative] = _file_sha256(target)
        return relative

    prognostic_path = _emit(PROGNOSTIC_STEM, prognostic)
    derived_path = _emit(DERIVED_STEM, derived)
    seam_path = _emit(SEAM_STEM, seam) if seam else None

    state_sha = prognostic_sha256(prognostic)

    child_records: list[dict[str, Any]] = []
    for child in children:
        grid_id = str(child["grid_id"])
        child_state = str(child.get("state", "LIVE"))
        if child_state not in CHILD_STATES:
            raise AnchorRefusal("unknown child state", grid_id=grid_id,
                                state=child_state, known=list(CHILD_STATES))
        arrays = child.get("arrays")
        record: dict[str, Any] = {
            "grid_id": grid_id,
            "placement": dict(child.get("placement") or {}),
            "state": child_state,
            "state_sha256": None,
            "path": None,
        }
        if arrays:
            child_dir = staging / "children" / grid_id
            record["path"] = _emit("state", arrays, subdir=child_dir)
            record["state_sha256"] = prognostic_sha256(arrays)
        child_records.append(record)

    analysis_block: dict[str, Any] = {"state": "SKIPPED_NO_OBS",
                                      "increment_sha256": None,
                                      "path": None, "ingestion": None}
    if analysis is not None:
        analysis_block.update({key: value for key, value in analysis.items()
                               if key != "arrays"})
        arrays = analysis.get("arrays")
        if arrays:
            analysis_block["path"] = _emit(INCREMENT_STEM, arrays)
            analysis_block["increment_sha256"] = prognostic_sha256(arrays)
        analysis_block.setdefault("ingestion", None)

    manifest: dict[str, Any] = {
        "schema": ANCHOR_SCHEMA,
        "cycle_index": int(cycle_index),
        "anchor_ticks": int(anchor_ticks),
        "tick_hz": TICK_HZ,
        "valid_time": iso_time,
        "array_format": ARRAY_FORMAT,
        "written_at": datetime.now(timezone.utc).isoformat()
                              .replace("+00:00", "Z"),
        "parent": {
            "kind": parent_kind,
            "mesh_id": str(mesh_id),
            "prognostic_sha256": state_sha,
            "path": prognostic_path,
            **_dimension_counts(prognostic),
        },
        "derived": {
            "derived_from_sha256": state_sha,
            "path": derived_path,
            "fields": sorted(derived),
        },
        "seam": {"path": seam_path} if seam_path else None,
        "children": child_records,
        "analysis": analysis_block,
        "diagnostics_rebuilt": (None if diagnostics_rebuilt is None
                                else bool(diagnostics_rebuilt)),
    }
    if extra:
        manifest.update(dict(extra))

    manifest_path = staging / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False)
                             + "\n", encoding="utf-8")
    _fsync_file(manifest_path)
    written[MANIFEST_NAME] = _file_sha256(manifest_path)

    marker = {
        "schema": ANCHOR_SCHEMA,
        "marker": "COMMIT",
        "cycle_index": int(cycle_index),
        "array_format": ARRAY_FORMAT,
        "files": [{"path": relative, "sha256": written[relative]}
                  for relative in sorted(written)],
    }
    marker_path = staging / COMMIT_MARKER_NAME
    marker_path.write_text(json.dumps(marker, indent=2) + "\n",
                           encoding="utf-8")
    _fsync_file(marker_path)

    os.replace(staging, final)
    return final


# --------------------------------------------------------------------------
# read


@dataclass(frozen=True)
class AnchorDocument:
    """A committed anchor.  Arrays load lazily and are checked on load."""

    path: Path
    manifest: dict[str, Any]
    _marker: dict[str, str] = field(default_factory=dict, repr=False)

    # -- loaders -----------------------------------------------------------
    def _load(self, relative: str | None, *, what: str
              ) -> dict[str, np.ndarray]:
        if not relative:
            raise AnchorRefusal(f"anchor carries no {what}",
                               anchor=str(self.path), what=what,
                               cycle_index=self.manifest.get("cycle_index"))
        target = self.path / relative
        if not target.exists():
            raise AnchorRefusal("anchor member named by the manifest is "
                                "missing on disk",
                                anchor=str(self.path), member=relative)
        expected = self._marker.get(relative)
        observed = _file_sha256(target)
        if expected is not None and observed != expected:
            raise AnchorRefusal(
                "anchor member does not match the COMMIT marker; the "
                "directory was edited after publication",
                anchor=str(self.path), member=relative,
                expected_sha256=expected, observed_sha256=observed)
        return _read_arrays(target)

    def prognostic(self) -> dict[str, np.ndarray]:
        return self._load(self.manifest["parent"]["path"], what="prognostics")

    def derived(self) -> dict[str, np.ndarray]:
        return self._load(self.manifest["derived"]["path"],
                          what="derived diagnostics")

    def seam(self) -> dict[str, np.ndarray]:
        seam = self.manifest.get("seam") or {}
        return self._load(seam.get("path"), what="physics-seam state")

    def increment(self) -> dict[str, np.ndarray]:
        analysis = self.manifest.get("analysis") or {}
        return self._load(analysis.get("path"), what="analysis increment")

    def has_increment(self) -> bool:
        return bool((self.manifest.get("analysis") or {}).get("path"))

    def child_state(self, grid_id: str) -> dict[str, np.ndarray]:
        for child in self.manifest.get("children", ()):
            if child["grid_id"] == grid_id:
                return self._load(child.get("path"),
                                  what=f"state for child {grid_id}")
        raise AnchorRefusal("anchor carries no such child",
                            anchor=str(self.path), grid_id=grid_id,
                            known=[c["grid_id"]
                                   for c in self.manifest.get("children", ())])


def read_anchor(path: str | os.PathLike[str]) -> AnchorDocument:
    """Open a COMMITTED anchor, or refuse naming what is missing."""
    directory = Path(path)
    if not directory.is_dir():
        raise AnchorRefusal("anchor directory does not exist",
                            path=str(directory))
    marker_path = directory / COMMIT_MARKER_NAME
    if not marker_path.exists():
        raise AnchorRefusal(
            "anchor is not committed; its COMMIT marker is absent, which is "
            "what a crash between phase one and phase three leaves behind. "
            "Use the previous anchor, or re-run the leg that wrote this one",
            path=str(directory), marker=COMMIT_MARKER_NAME,
            present=sorted(p.name for p in directory.iterdir()))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    shas = {entry["path"]: entry["sha256"] for entry in marker.get("files", ())}

    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise AnchorRefusal("committed anchor has no manifest",
                            path=str(directory), manifest=MANIFEST_NAME)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ANCHOR_SCHEMA:
        raise AnchorRefusal("anchor manifest is a schema this reader does "
                            "not know",
                            path=str(directory),
                            observed_schema=manifest.get("schema"),
                            expected_schema=ANCHOR_SCHEMA)
    if manifest.get("array_format") not in _SUFFIX:
        raise AnchorRefusal("anchor manifest does not name its array format",
                            path=str(directory),
                            array_format=manifest.get("array_format"),
                            known=sorted(_SUFFIX))
    if manifest["array_format"] != ARRAY_FORMAT:
        raise AnchorRefusal(
            "anchor was written in an array format this process cannot "
            "read; install the writer's backend rather than guessing",
            path=str(directory), written_as=manifest["array_format"],
            readable_here=ARRAY_FORMAT)
    return AnchorDocument(path=directory, manifest=manifest, _marker=shas)


def _committed_anchors(root: str | os.PathLike[str]) -> list[Path]:
    anchors = Path(root) / ANCHORS_DIRNAME
    if not anchors.is_dir():
        return []
    return sorted((entry for entry in anchors.iterdir()
                   if entry.is_dir() and not entry.name.endswith(".tmp")
                   and (entry / COMMIT_MARKER_NAME).exists()),
                  key=lambda entry: entry.name)


def latest_anchor(root: str | os.PathLike[str]) -> Path | None:
    """The newest COMMITTED anchor, or ``None``.  The supported reader."""
    committed = _committed_anchors(root)
    return committed[-1] if committed else None


def anchor_for_cycle(root: str | os.PathLike[str],
                     cycle_index: int) -> Path | None:
    prefix = f"anchor_{int(cycle_index):03d}_"
    for entry in _committed_anchors(root):
        if entry.name.startswith(prefix):
            return entry
    return None
