"""Retain a complete refused configuration for the existing Fit/Tile tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tomllib


RECOVERY_DIR_ENV = "GPUWM_CONFIGURATION_RECOVERY_DIR"


class MemoryAdmissionError(ValueError):
    """A complete configuration exceeded a measured or declared memory budget."""

    def __init__(self, message: str, **memory):
        super().__init__(message)
        self.memory = memory
        self.recovery = None
        self.recovery_error = None


def error_document(error: MemoryAdmissionError) -> dict:
    result = {"schema": "arwen.configuration-error.v1", "kind": "memory",
              "error": str(error), "created": False, "memory": error.memory}
    if error.recovery is not None:
        result["recovery"] = error.recovery
    if error.recovery_error is not None:
        result["recovery_error"] = error.recovery_error
    return result


def retain_final_candidate(error: MemoryAdmissionError, *, text: str,
                           requested_path: Path, stage: Path, metadata=None) -> None:
    """Attach an explicit recovery copy; never publish the requested output.

    Producers call this only after applying their final settings. The staging
    directory's earlier scaffold TOML is deliberately excluded.
    """
    if not isinstance(error, MemoryAdmissionError):
        raise TypeError("Only a typed memory admission refusal can retain a draft")
    selected = os.environ.get(RECOVERY_DIR_ENV)
    if not selected:
        return
    try:
        error.recovery = _retain(Path(selected).absolute(), text=text,
            requested_path=Path(requested_path).absolute(), stage=Path(stage),
            metadata={} if metadata is None else metadata, memory=error.memory)
    except (OSError, ValueError) as failure:
        # A disk/path failure must neither erase the real memory refusal nor
        # imply that a recovery draft was successfully retained.
        error.recovery_error = str(failure)


def _retain(directory: Path, *, text: str, requested_path: Path, stage: Path,
            metadata: dict, memory: dict) -> dict:
    from gpuwm.case_data import resolved_case_data_paths
    from gpuwm.starter_template import _tiles_wps, changes
    from gpuwm.toml_document import emit_experiment_toml

    resolved_directory = directory.resolve()
    resolved_output = requested_path.resolve()
    if resolved_directory == resolved_output or resolved_output in resolved_directory.parents:
        raise ValueError("Recovery cannot create the originally requested output path")
    if os.path.lexists(directory):
        raise FileExistsError(f"Recovery preserves the existing path: {directory}")
    original = tomllib.loads(text)
    raw = tomllib.loads(text)
    payloads, remap = {}, {}
    for source in sorted(stage.iterdir()):
        if source.name == requested_path.name:
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Recovery requires ordinary generated companions: {source}")
        name = ("draft" + source.name[len(requested_path.stem):]
                if source.name.startswith(requested_path.stem + ".") else source.name)
        if name in {"draft.toml", "recovery.json"} or name in payloads:
            raise ValueError(f"Recovery companion name conflicts: {name}")
        target = directory / name
        declared = requested_path.parent / source.name
        remap[str(declared.resolve())] = str(target)
        payload = source.read_bytes()
        if name.endswith(".namelist.wps"):
            payload = _tiles_wps(declared, target, text=payload.decode("utf-8")).encode("utf-8")
        payloads[name] = payload
    if "draft.namelist.wps" not in payloads:
        raise ValueError("The completed candidate has no WPS companion for Fit/Tile recovery")
    if "case_data" in raw:
        resolved = resolved_case_data_paths(raw["case_data"],
            base_dir=requested_path.parent, source=str(requested_path))
        for key in ("vtable", "wps_namelist"):
            if key in resolved and resolved[key] in remap:
                resolved[key] = remap[resolved[key]]
        raw["case_data"] = resolved
    if "static" in raw:
        from gpuwm.static.highres_production import parse_static_table
        config = parse_static_table(raw["static"], source=str(requested_path),
                                    base_dir=requested_path.parent)
        if config is not None:
            raw["static"]["highres"]["cache_root"] = str(config.cache_root.resolve())
    # fetch.out is owned by the fetch command's working directory.
    if raw.get("fetch", {}).get("out"):
        raw["fetch"]["out"] = str(Path(raw["fetch"]["out"]).expanduser().resolve())
    draft = ("# Memory admission refused. Review Fit domain or Tile streaming before running.\n"
             + emit_experiment_toml(raw)).encode("utf-8")
    files = [{"path": str(directory / name), "bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}
             for name, payload in sorted(payloads.items())]
    files.append({"path": str(directory / "draft.toml"), "bytes": len(draft),
                  "sha256": hashlib.sha256(draft).hexdigest()})
    receipt = {"schema": "arwen.configuration-recovery.v1", "status": "memory-refused",
               "configuration": str(directory / "draft.toml"),
               "requested_output": str(requested_path), "created": False,
               "forecast_started": False, "memory": memory,
               "original_candidate_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "config_sha256": hashlib.sha256(draft).hexdigest(),
               "path_changes": [{"field": field, "before": before, "after": after}
                                for field, before, after in changes(original, raw)],
               "metadata": metadata, "files": files}
    payloads["recovery.json"] = (json.dumps(receipt, indent=2, ensure_ascii=True,
                                          allow_nan=False, default=str) + "\n").encode("utf-8")
    if len(payloads["recovery.json"]) > 64 * 1024:
        raise ValueError("Configuration recovery receipt exceeds the interface's 64 KiB limit")
    payloads["draft.toml"] = draft  # Publish the usable draft last.
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    identity = directory.stat()
    created = []
    try:
        for name, payload in payloads.items():
            target = directory / name
            with target.open("xb") as stream:
                created.append((target, os.fstat(stream.fileno())))
                stream.write(payload)
    except BaseException:
        for target, stat in reversed(created):
            try:
                if os.path.samestat(stat, target.stat()):
                    target.unlink()
            except FileNotFoundError:
                pass
        if os.path.samestat(identity, directory.stat()):
            directory.rmdir()
        raise
    return receipt
