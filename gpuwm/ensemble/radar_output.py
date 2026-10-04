"""Retain original native radar products and retire their temporary histories."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path, PurePosixPath


def radar_enabled(inputs):
    return bool(getattr(getattr(inputs.experiment, "simulated_radar", None), "enabled", False))


def member_history_required(inputs, keep_member_files):
    # The original radar consumer reads durable volume histories while the
    # forecast runs, then drains before its original runner returns.
    return bool(keep_member_files or radar_enabled(inputs))


def _within(root, path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise ValueError("member radar artifact must be a file inside its owned output directory")
    return path


def verified_radar_products(member_directory):
    """Verify the Rust manifest and its committed artifacts before cleanup."""
    from gpuwm.output_identity import file_record
    from gpuwm.simulated_radar_config import MANIFEST_SCHEMA
    root = Path(member_directory).resolve(strict=True)
    manifest = _within(root, root / "radar" / "manifest.json")
    manifest_record = file_record(manifest)
    document = json.loads(manifest.read_bytes())
    if document.get("schema") != MANIFEST_SCHEMA or document.get("simulated") is not True:
        raise ValueError("member radar needs its original committed simulated-radar manifest")
    artifacts = [row for volume in document["volumes"]
                 for row in (*volume["files"], *volume["images"])]
    artifacts.extend(document.get("loops", ()))
    records = []
    for row in artifacts:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
            raise ValueError("member radar manifest artifact escapes its owned output directory")
        path = _within(root, root / str(relative))
        record = file_record(path)
        if record["bytes"] != row["bytes"] or record["sha256"] != row["sha256"]:
            raise ValueError("member radar artifact differs from its committed Rust manifest")
        records.append(dict(record, path=str(path.relative_to(root))))
    if file_record(manifest) != manifest_record:
        raise ValueError("member radar manifest changed during completion verification")
    return {"manifest": "radar/manifest.json", "manifest_sha256": manifest_record["sha256"],
            "artifacts": records, "volumes": len(document["volumes"]),
            "warnings": document.get("warnings", [])}


def finish_member_radar(inputs, member_directory, result, *, keep_member_files):
    """Original runners have closed their native radar queue before this seam.

    Only frame paths returned by that runner in this new member directory may
    be deleted. Failed products retain their histories for recovery.
    """
    if not radar_enabled(inputs):
        return None
    status = result.get("status") if isinstance(result, Mapping) else getattr(result, "status", None)
    if status not in (None, "PASS"):
        raise RuntimeError("failed member radar execution retains its original histories for recovery")
    from gpuwm.output_identity import file_records
    root = Path(member_directory).resolve(strict=True)
    receipt = verified_radar_products(root)
    paths = result.get("wrfout_paths", ()) if isinstance(result, Mapping) else result.wrfout_paths
    frames = []
    for original in paths:
        path = _within(root, original)
        if not path.name.startswith("wrfout_") or path.suffix == ".json":
            raise ValueError("temporary radar history must be an original member wrfout frame")
        if path not in frames:
            frames.append(path)
    records = file_records(frames)
    receipt["history_policy"] = ("keep member files" if keep_member_files else
        "original transient volume histories; deleted after native radar close and artifact verification")
    receipt["deleted_history_files"] = []
    if not keep_member_files:
        # Validate every path and all product identities before the first delete.
        for path, record in zip(frames, records):
            path.unlink()
            receipt["deleted_history_files"].append(dict(record, path=str(path.relative_to(root))))
    return receipt
