"""Durable adaptive spread fields bound to an accepted ensemble generation.

The Rust array writer owns NPY payload writes. JSON describes their identity;
the enclosing ensemble manifest is published only after both fields and this
description have landed. A missing or incompatible prior is never reset on
resume, because that would change the following analysis without saying so.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from gpuwm.da.spread_repair import (
    AdaptiveInflationState, SpreadRepairController, SCHEMA as STATE_SCHEMA)

SCHEMA = "gpuwm.da.adaptive-spread-generation/v1"


class SpreadRestartError(ValueError):
    """Refuse a resume that would use the wrong inflation uncertainty."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _utc(value):
    try:
        stamp = (value if isinstance(value, datetime) else
                 datetime.fromisoformat(str(value).replace("Z", "+00:00")))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("missing timezone")
        return stamp.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError) as error:
        raise SpreadRestartError("adaptive spread time needs explicit UTC authority; an ambiguous time would resume the wrong cycle") from error


def clock_valid_time(start_time, elapsed_seconds):
    """The engine experiment's UTC clock, including its naive-UTC convention."""
    start = start_time
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return _utc(start + timedelta(seconds=float(elapsed_seconds)))


def config_identity(config, seed):
    return _digest({"config": asdict(config), "seed": int(seed)})


def fork_controller(controller):
    """A verification-only solve must not advance the carried uncertainty."""
    state = controller.state
    if state is not None:
        state = AdaptiveInflationState(state.shape.copy(), state.scale.copy(), state.cycles)
    return SpreadRepairController(controller.config, seed=controller.seed, state=state)


def _grid_shape(identity):
    return tuple(int(identity[key]) for key in ("nz", "ny", "nx"))


def capture(controller, *, identity, grid_identity_sha256, valid_time,
            elapsed_seconds, leg_number):
    """Freeze controller arrays before an asynchronous generation write."""
    if not controller.config.adaptive:
        raise SpreadRestartError("only the named adaptive policy owns a durable inflation sidecar")
    shape = _grid_shape(identity)
    if controller.state is None:
        # A first run without a usable precipitation batch still owns its
        # declared prior. This never handles a resume: restore is mandatory.
        controller.state = AdaptiveInflationState.initial(shape, controller.config)
    controller.state.validate(shape)
    record = controller.state.checkpoint()
    arrays = {}
    for key in ("shape", "scale"):
        value = record[key]
        if hasattr(value, "get"):
            value = value.get()
        arrays[key] = np.ascontiguousarray(value, dtype=np.float64)
    last = (controller.last_receipt or {}).get("analysis_valid_time")
    metadata = {
        "schema": SCHEMA, "state_schema": STATE_SCHEMA,
        "identity": dict(identity), "grid_identity_sha256": grid_identity_sha256,
        "config_sha256": config_identity(controller.config, controller.seed),
        "config": asdict(controller.config), "seed": controller.seed,
        "valid_time": _utc(valid_time),
        "analysis_valid_time": None if last is None else _utc(last),
        "elapsed_seconds": float(elapsed_seconds), "leg_number": int(leg_number),
        "cycles": record["cycles"],
    }
    return {"metadata": metadata, "arrays": arrays}


def _validate_metadata(metadata, *, identity, elapsed_seconds, leg_number):
    if not isinstance(metadata, dict):
        raise SpreadRestartError("adaptive spread metadata must describe its saved posterior, not an unrelated JSON value")
    if metadata.get("schema") != SCHEMA or metadata.get("state_schema") != STATE_SCHEMA:
        raise SpreadRestartError("adaptive spread sidecar has an unknown schema; uncertainty cannot be interpreted safely")
    if metadata.get("identity") != identity:
        raise SpreadRestartError("adaptive spread grid/prepared identity differs from its ensemble generation")
    if metadata.get("elapsed_seconds") != float(elapsed_seconds) or metadata.get("leg_number") != int(leg_number):
        raise SpreadRestartError("adaptive spread sidecar belongs to a different generation clock")
    if not math.isfinite(float(elapsed_seconds)) or elapsed_seconds < 0:
        raise SpreadRestartError("adaptive spread generation clock must be finite and nonnegative")
    if type(metadata.get("cycles")) is not int or metadata["cycles"] < 0:
        raise SpreadRestartError("adaptive spread cycle counter must be a nonnegative integer")
    grid = metadata.get("grid_identity_sha256")
    if not isinstance(grid, str) or len(grid) != 64 or any(c not in "0123456789abcdef" for c in grid):
        raise SpreadRestartError("adaptive spread needs an exact grid identity to prevent uncertainty moving between cells")
    _utc(metadata.get("valid_time"))
    last = metadata.get("analysis_valid_time")
    if last is not None and _utc(last) > _utc(metadata["valid_time"]):
        raise SpreadRestartError("adaptive spread analysis time is after its generation clock")
    if not isinstance(metadata.get("config"), dict):
        raise SpreadRestartError("adaptive spread configuration is missing; the saved prior cannot be interpreted")
    if metadata.get("config_sha256") != _digest({"config": metadata.get("config"), "seed": metadata.get("seed")}):
        raise SpreadRestartError("adaptive spread configuration digest is inconsistent")
    if metadata.get("config", {}).get("adaptive") is not True:
        raise SpreadRestartError("an inflation sidecar must declare the adaptive policy it carries")


def write_sidecar(directory, record, *, identity, elapsed_seconds, leg_number):
    from gpuwm.ingest.prepared_writer import native_writer, write_arrays
    metadata = dict(record["metadata"])
    _validate_metadata(metadata, identity=identity,
                       elapsed_seconds=elapsed_seconds, leg_number=leg_number)
    arrays = record["arrays"]
    state = AdaptiveInflationState(arrays["shape"], arrays["scale"], metadata["cycles"])
    state.validate(_grid_shape(identity))
    if any(arrays[key].dtype != np.dtype("float64") or not arrays[key].flags.c_contiguous
           for key in ("shape", "scale")):
        raise SpreadRestartError("adaptive spread arrays must preserve their contiguous float64 posterior")
    entry = native_writer()
    if entry is None:
        raise SpreadRestartError("adaptive spread persistence requires the Rust array writer; no substitute payload writer is permitted")
    folder = Path(directory) / "spread-repair"
    folder.mkdir(exist_ok=True)
    paths = {key: folder / (key + ".npy") for key in ("shape", "scale")}
    temporary = {key: folder / (key + "." + uuid.uuid4().hex + ".partial") for key in paths}
    try:
        hashes = write_arrays(entry, [(temporary[key], arrays[key]) for key in paths], workers=1)
        metadata["arrays"] = {}
        for key, digest in zip(paths, hashes, strict=True):
            os.replace(temporary[key], paths[key])
            metadata["arrays"][key] = {"file": paths[key].name,
                "file_sha256": _file_sha(paths[key]), "array_sha256": digest,
                "dtype": arrays[key].dtype.str, "shape": list(arrays[key].shape)}
        path = folder / "state.json"
        tmp = folder / ("state." + uuid.uuid4().hex + ".partial")
        tmp.write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return {"directory": "spread-repair", "metadata": "state.json", "sha256": _file_sha(path)}
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def read_sidecar(directory, manifest, *, identity, config=None, seed=None,
                 grid_identity_sha256=None, valid_time=None):
    reference = manifest.get("spread_repair")
    if reference is None:
        raise SpreadRestartError("adaptive spread resume requires its saved inflation state; resetting shape/scale would change the next analysis")
    if not isinstance(reference, dict) or reference.get("directory") != "spread-repair" or reference.get("metadata") != "state.json":
        raise SpreadRestartError("adaptive spread reference must name its owned generation sidecar")
    folder = Path(directory) / "spread-repair"
    path = folder / "state.json"
    try:
        if _file_sha(path) != reference.get("sha256"):
            raise SpreadRestartError("adaptive spread metadata hash changed; the accepted uncertainty is not intact")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        _validate_metadata(metadata, identity=identity,
            elapsed_seconds=manifest["elapsed_seconds"], leg_number=manifest["leg_number"])
        if config is not None and metadata["config_sha256"] != config_identity(config, seed):
            raise SpreadRestartError("adaptive spread configuration/seed changed; resuming would interpret a different inflation prior")
        if grid_identity_sha256 is not None and metadata["grid_identity_sha256"] != grid_identity_sha256:
            raise SpreadRestartError("adaptive spread target grid changed; uncertainty would move to different cells")
        if valid_time is not None and _utc(metadata["valid_time"]) != _utc(valid_time):
            raise SpreadRestartError("adaptive spread valid UTC differs from the restored experiment clock")
        arrays = {}
        for key in ("shape", "scale"):
            item = metadata["arrays"][key]
            if item["file"] != key + ".npy":
                raise SpreadRestartError("adaptive spread array reference leaves its declared field")
            payload = folder / item["file"]
            if _file_sha(payload) != item["file_sha256"]:
                raise SpreadRestartError(f"adaptive spread {key} hash changed; restoring a different posterior is refused")
            value = np.load(payload, allow_pickle=False)
            if value.dtype != np.dtype("float64") or value.dtype.str != item["dtype"] or list(value.shape) != item["shape"]:
                raise SpreadRestartError("adaptive spread array dtype/shape changed; uncertainty would be misinterpreted")
            arrays[key] = value
        try:
            state = AdaptiveInflationState.restore({"schema": STATE_SCHEMA,
                **arrays, "cycles": metadata["cycles"]}, _grid_shape(identity))
        except ValueError as error:
            raise SpreadRestartError(f"adaptive spread posterior is invalid and cannot be reset on resume: {error}") from error
        return state, metadata
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise SpreadRestartError("adaptive spread sidecar is missing or torn; resetting uncertainty on resume is refused") from error
