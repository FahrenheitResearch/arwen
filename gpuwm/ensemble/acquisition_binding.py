"""Original source acquisition receipts consumed by prepared member rosters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath

from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.recipes import SourceTrajectory


def manifest_trajectory(document):
    """Read the existing generic fetch envelope without rewriting its bytes."""
    row = document.get("request") if document.get("schema") == "gpuwm-fetch-route-manifest-v1" else document
    if not isinstance(row, dict) or not isinstance(row.get("source"), str):
        raise ValueError("acquisition manifest has no canonical source request")
    cycle = datetime.fromisoformat(str(row.get("cycle", "")).replace("Z", "+00:00"))
    if cycle.tzinfo is None:
        raise ValueError("acquisition source cycle needs its explicit UTC offset")
    selected = SourceTrajectory(row["source"], cycle.astimezone(timezone.utc), row.get("member") or None)
    if row is not document:
        outer = dict(row)
        outer.update({key: document[key] for key in ("source", "cycle", "member") if key in document})
        if manifest_trajectory(outer) != selected:
            raise ValueError("acquisition request conflicts with its outer source identity")
    return selected


def mapped_input_closure(acquisition, inputs, acquisition_root):
    """Close native mapped payload records against their acquisition catalog."""
    root = PurePosixPath(str(acquisition_root))
    def relative(value):
        path = PurePosixPath(str(value))
        if path.is_absolute():
            try:
                path = path.relative_to(root)
            except ValueError as error:
                raise ValueError("mapped source payload belongs to another acquisition directory") from error
        if ".." in path.parts:
            raise ValueError("mapped source payload escapes its acquisition directory")
        return str(path)
    catalog = {}
    for item in (*acquisition.get("files", ()), *acquisition.get("composed", ())):
        name = relative(item.get("relpath", item.get("name", "")))
        value = (item.get("bytes"), item.get("sha256"))
        if name in catalog and catalog[name] != value:
            raise ValueError("acquisition contains conflicting source payload records")
        catalog[name] = value
    primary = inputs.get("primary_files")
    # A native manifest that lists its payload by role is declared by the
    # preparation that reads it, in the preparation runner table.
    from gpuwm.source_cli import role_keyed_input_manifest
    role_keyed = role_keyed_input_manifest(inputs.get("schema"))
    if role_keyed is not None:
        native_source = inputs.get("source", {})
        if manifest_trajectory({"source": native_source.get("model", "").lower(),
                                "cycle": native_source.get("cycle")}) != manifest_trajectory(acquisition):
            raise ValueError("native direct source manifest names another acquisition cycle")
        primary = [{"path": item["name"], "sha256": item["sha256"],
                    "bytes": catalog.get(relative(item["name"]), (None, None))[0]}
                   for role, item in inputs.get("files", {}).items()
                   if role.startswith(role_keyed.lead_role_prefix)]
    if not isinstance(primary, list) or not primary:
        raise ValueError("mapped source manifest has no native primary payload records")
    rows = list(primary)
    for supplement in inputs.get("supplements", {}).values():
        rows.extend(supplement)
    result = []
    for item in rows:
        name = relative(item["path"])
        value = (item.get("bytes"), item.get("sha256"))
        if value[0] is None or value[1] is None or catalog.get(name) != value:
            raise ValueError("native mapped payload differs from the original acquisition catalog")
        result.append({"path": name, "bytes": value[0], "sha256": value[1]})
    return result


def verified_artifact(reference):
    """Read only the exact native artifact bytes named by an admission."""
    path = Path(reference["path"]).resolve(strict=True)
    data = path.read_bytes()
    import hashlib
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("native source artifact changed from its pinned bytes")
    return data


@dataclass(frozen=True)
class AcquiredSourceBinding:
    """Pinned acquisition bytes plus their source/native verification receipt.

    This implements the prepared member roster's existing binding protocol.
    The verification comes from native input admission or member decoding;
    reading this metadata alone is not a new payload-verification claim.
    """
    trajectory: SourceTrajectory
    path: Path
    sha256: str
    verification: dict

    def verify(self):
        path = Path(self.path).resolve(strict=True)
        if not path.is_file() or digest_file(path) != self.sha256:
            raise ValueError("source acquisition manifest changed from its pinned bytes")
        document = json.loads(path.read_bytes())
        if manifest_trajectory(document) != self.trajectory:
            raise ValueError("source acquisition belongs to another source, cycle or member")
        receipt = self.verification
        if manifest_trajectory(receipt) != self.trajectory or receipt.get("manifest_sha256") != self.sha256:
            raise ValueError("native acquisition verification belongs to another source manifest")
        if receipt.get("status") != "PASS":
            raise ValueError("source acquisition has no passing native verification")
        artifacts = receipt.get("artifacts", {})
        if not artifacts:
            raise ValueError("source acquisition verification lacks pinned native artifacts")
        contents = {name: verified_artifact(reference) for name, reference in artifacts.items()}
        if "mapped_input_manifest" in contents:
            payloads = mapped_input_closure(document, json.loads(contents["mapped_input_manifest"]),
                                           receipt["acquisition_root"])
            if "verification_receipt" not in contents:
                raise ValueError("mapped acquisition lacks its native member verification receipt")
            native = json.loads(contents["verification_receipt"])
            if (native.get("status") != "PASS" or native.get("manifest_sha256") != self.sha256
                    or manifest_trajectory(native) != self.trajectory):
                raise ValueError("native member verification does not bind this acquisition")
            def payload_key(value):
                return value["path"], value["bytes"], value["sha256"]
            if not {payload_key(value) for value in payloads} <= {
                    payload_key(value) for value in native.get("payload_files", ())}:
                raise ValueError("native verification did not hash all consumed source payloads")
            if self.trajectory.member is not None and native.get("native_member_check") != "PASS":
                raise ValueError("ensemble source lacks its actual native member verification")
            manifest_sha = artifacts["mapped_input_manifest"]["sha256"]
            if "physical_manifest" in contents:
                physical = json.loads(contents["physical_manifest"])
                if physical["source"].get("input_manifest_sha256") != manifest_sha:
                    raise ValueError("physical donor belongs to another native input manifest")
                units = json.loads(contents.get("units_qualification", b"{}"))
                if (units.get("status") != "PASS" or units.get("physical_manifest_sha256")
                        != artifacts["physical_manifest"]["sha256"]):
                    raise ValueError("physical donor lacks its pinned native units verification")
            if "cache_header" in contents:
                header = json.loads(contents["cache_header"])
                if header["identity"]["source_identity"].get("input_manifest_sha256") != manifest_sha:
                    raise ValueError("native prepared cache belongs to another source manifest")
        elif "raw_hash_manifest" in contents and "native_report" in contents:
            hashes = {}
            for line in contents["raw_hash_manifest"].decode().splitlines():
                digest, name = line.split(maxsplit=1)
                name = str(PurePosixPath(name.lstrip("*")))
                if name in hashes:
                    raise ValueError("native source checksum manifest duplicates an object")
                hashes[name] = digest
            fetched = {item["name"]: item["sha256"] for item in document["files"]
                       if item.get("role") in {"atmosphere", "soil"}}
            if not fetched or hashes != fetched:
                raise ValueError("native source checksums differ from acquired atmosphere and soil objects")
            report = json.loads(contents["native_report"])
            native = report.get("source_hash_preflight")
            if (not isinstance(native, dict) or native.get("status") != "PASS"
                    or native.get("manifest_sha256") != artifacts["raw_hash_manifest"]["sha256"]):
                raise ValueError("native preparation report does not verify these source checksums")
        else:
            raise ValueError("source acquisition has no supported native payload closure")
        if digest_file(path) != self.sha256:
            raise ValueError("source acquisition changed during verification")
        return {"trajectory_sha256": self.trajectory.identity, "source": self.trajectory.source,
            "cycle": self.trajectory.cycle.isoformat(), "member": self.trajectory.member,
            "manifest": str(path), "manifest_sha256": self.sha256, "verification": dict(receipt)}
