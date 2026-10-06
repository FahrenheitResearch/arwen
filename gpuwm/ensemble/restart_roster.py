"""Bind durable restart sets to the original ensemble member identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

CONTRACT = "gpuwm-ensemble-restart.v1"
IDENTITIES = "ensemble-identities.json"
ROSTER = "ensemble-restart.json"
INPUTS = "ensemble-inputs.json"


class CheckpointProgress:
    """Seal member-owned output state after the native checkpoint commits."""
    hosts_forecast = True

    def __init__(self, callback, *, collector, root, manifest_lock):
        self.callback, self.collector = callback, collector
        self.root, self.lock = Path(root), manifest_lock
        self.last_checkpoint = None
        self.last_attempted_checkpoint = None

    def __call__(self, **event):
        checkpoint = event.get("last_checkpoint")
        if checkpoint is not None and str(checkpoint) != self.last_attempted_checkpoint:
            self.last_attempted_checkpoint = str(checkpoint)
            with self.lock:
                save = getattr(self.collector, "save_resume", None)
                if not callable(save):
                    raise ValueError("checkpointed ensemble requires a durable diagnostic collector")
                try:
                    save()
                except TimeoutError:
                    event["ensemble_checkpoint_deferred"] = "product drain exceeded bounded checkpoint hold"
                    return self.callback(**event)
                bind_checkpoint_member(self.root, checkpoint)
                seal(self.root)
                self.last_checkpoint = str(checkpoint)
            event["ensemble_restart_roster"] = str(self.root / ROSTER)
        return self.callback(**event)

    def __getattr__(self, name):
        return getattr(self.callback, name)


def bind_checkpoint_member(root, checkpoint):
    """Stamp the original runner's member owner after its complete set commits."""
    from gpuwm.io.restart import read_restart_header, tree_restart_members
    root, checkpoint = Path(root), Path(checkpoint)
    owner = get_checkpoint_member(checkpoint)
    if owner is None:
        raise ValueError("member checkpoint has no bound ensemble identity")
    header = read_restart_header(checkpoint)
    paths = tree_restart_members(checkpoint) if header.get("domain_ids") else {1: checkpoint}
    record = {"schema": CONTRACT, "member_id": owner["memberId"], "identity_sha256": owner["memberIdentity"],
        "files": [{"path": relative_file(root, path), "sha256": digest_file(path), "bytes": path.stat().st_size}
            for path in paths.values()]}
    atomic_json(checkpoint.with_suffix(".member.json"), record)
    return record


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(record, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    pending.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    return path.relative_to(root).as_posix()


def resolve_file(root, name):
    path = (Path(root) / name).resolve()
    path.relative_to(Path(root).resolve())
    return path


def identity_record(request, member_order, member_inputs, seeds):
    from gpuwm.core.model import restart_identity_payload
    selected = dict(request.receipt())
    # A new box may have a different card count or allocator admission.
    selected.pop("member_device_ids", None)
    selected.pop("max_ordinary_members_per_device", None)
    rows = []
    for member in member_order:
        inputs = member_inputs[member]
        payload = {"member_id": member, "seed": seeds[member],
            "experiment": restart_identity_payload(inputs.experiment),
            "source": getattr(inputs, "source", None),
            "prepared_head_sha256": getattr(inputs, "prepared_head_sha256", None)}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        rows.append({**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()})
    return {"schema": CONTRACT, "request": selected, "member_order": list(member_order), "members": rows}


def verify_roster(path, identities, *, root):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != CONTRACT or record.get("identities") != identities:
        raise ValueError("ensemble restart roster has different member identities, seeds, forcing or physics")
    rows = record.get("members", [])
    if [row.get("member_id") for row in rows] != identities["member_order"]:
        raise ValueError("ensemble restart roster must name every original member exactly once in order")
    for item in record.get("files", []):
        path = resolve_file(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest_file(path) != item["sha256"]:
            raise ValueError(f"ensemble resume file is missing or hash-mismatched: {item['path']}")
    for row, identity in zip(rows, identities["members"]):
        if row.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("ensemble checkpoint was assigned to another member")
        if row.get("status") not in ("completed", "checkpoint", "pending"):
            raise ValueError("ensemble restart member has an unknown completion status")
        if row["status"] == "checkpoint":
            prefix = f"members/member-{row['member_id']:04d}/"
            if not row.get("checkpoint", "").startswith(prefix):
                raise ValueError("ensemble checkpoint path belongs to another member")
            resolve_file(root, row["checkpoint"])
        if row["status"] == "completed" and not isinstance(row.get("result"), dict):
            raise ValueError("completed ensemble member has no retained forecast result")
    return record


def get_checkpoint_member(path):
    """Return the original member owner and prepared handoff for transport."""
    path = Path(path).resolve()
    root = next((parent for parent in path.parents if (parent / IDENTITIES).is_file()), None)
    if root is None:
        return None
    identities = json.loads((root / IDENTITIES).read_text(encoding="utf-8"))
    relative = relative_file(root, path)
    identity = next((row for row in identities["members"] if
        relative.startswith(f"members/member-{row['member_id']:04d}/")), None)
    if identity is None:
        raise ValueError("checkpoint is outside every original ensemble member directory")
    descriptors = json.loads((root / INPUTS).read_text(encoding="utf-8"))
    descriptor = next(row for row in descriptors["members"] if row["member_id"] == identity["member_id"])
    collector = root / ".ensemble-resume" / "collector.json"
    files = []
    if collector.is_file():
        document = json.loads(collector.read_text(encoding="utf-8"))
        files = [collector, *(resolve_file(root, name) for name in document.get("files", ()))]
    return {"memberIndex": identities["member_order"].index(identity["member_id"]),
        "memberId": identity["member_id"], "memberIdentity": identity["identity_sha256"],
        "config": descriptor.get("experiment_config"), "wps": descriptor.get("wps_namelist"),
        "prepared": descriptor.get("prepared_root"), "root": str(root),
        "roster": str(root / ROSTER), "collectorFiles": [str(item) for item in files],
        "memberInputs": descriptors["members"]}


def relocate_inputs(root, bindings):
    """Rebind fetched preparation addresses without changing member identities."""
    root = Path(root)
    document = json.loads((root / INPUTS).read_text(encoding="utf-8"))
    for row in document["members"]:
        selected = bindings.get(row["member_id"], bindings.get(str(row["member_id"])))
        if selected is None:
            continue
        if set(selected) - {"prepared_root", "experiment_config", "wps_namelist"}:
            raise ValueError("ensemble input relocation may change only fetched preparation addresses")
        row.update({name: str(value) for name, value in selected.items()})
    atomic_json(root / INPUTS, document)
    return seal(root)


def resume_prepared_roster(args, request, *, observer=None):
    """Run the fetched member handoffs through the ordinary native front door."""
    from gpuwm import stage_cli
    from gpuwm.ensemble.door import production_run_scope
    from gpuwm.ensemble.recipe_door import member_inputs
    if getattr(args, "restart", None) is not None:
        raise ValueError("--restart-roster cannot be combined with one member's --restart")
    roster = Path(args.restart_roster).resolve()
    root = roster.parent
    if getattr(args, "outdir", None) is not None and Path(args.outdir).resolve() != root:
        raise ValueError("ensemble continuation --outdir must be the fetched restart roster directory")
    document = json.loads((root / INPUTS).read_text(encoding="utf-8"))
    prepared = {row["member_id"]: row for row in document["members"]}
    identities = json.loads((root / IDENTITIES).read_text(encoding="utf-8"))
    verify_roster(roster, identities, root=root)
    first = identities["member_order"][0]
    bundle = stage_cli.resolve_bundle(Path(prepared[first]["prepared_root"]))
    def provider(*, shared_inputs, member_id, **unused):
        return shared_inputs if member_id == first else member_inputs(shared_inputs, prepared[member_id])
    sim = stage_cli.sim_command(bundle, experiment_config=Path(prepared[first]["experiment_config"]),
        wps_namelist=(Path(prepared[first]["wps_namelist"]) if prepared[first]["wps_namelist"] else None),
        outdir=root, physics_profile=None, progress_format="jsonl", render_products=None,
        **({"devices": args.devices} if getattr(args, "devices", None) is not None else {}),
        memory_gate=not bool(getattr(args, "no_memory_gate", False)))
    if bundle["layout"] == "tree":
        from gpuwm import prepared_domain_tree_forecast as runner
    else:
        from gpuwm import prepared_single_domain_forecast as runner
    with production_run_scope(request, output_directory=root, input_provider=provider, restart_roster=roster):
        return runner.main(sim[3:], observer=observer)


def seal(root, output=None, *, collector_files=()):
    """Publish a roster only after its member checkpoints and output spool exist."""
    from gpuwm.io.restart import read_restart_header, tree_restart_members
    from gpuwm.supervisor import validate_manifest_checkpoint
    from gpuwm.resume import discover_checkpoint_sets
    root = Path(root).resolve()
    identities = json.loads((root / IDENTITIES).read_text(encoding="utf-8"))
    manifest_path = root / "ensemble-run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = set(manifest.get("members_completed", ()))
    results = {row["member_id"]: row["result"] for row in manifest.get("completed_member_results", ())}
    files = {root / IDENTITIES, manifest_path}
    if (root / INPUTS).is_file():
        files.add(root / INPUTS)
    files.update(resolve_file(root, name) for name in collector_files)
    collector_manifest = root / ".ensemble-resume" / "collector.json"
    if collector_manifest.is_file():
        document = json.loads(collector_manifest.read_text(encoding="utf-8"))
        files.add(collector_manifest)
        files.update(resolve_file(root, name) for name in document.get("files", ()))
    rows = []
    for identity in identities["members"]:
        member = identity["member_id"]
        row = {"member_id": member, "identity_sha256": identity["identity_sha256"], "status": "pending",
               "model_elapsed_seconds": 0.0}
        if member in completed:
            if member not in results:
                raise ValueError(f"completed member {member} has no durable forecast receipt")
            row.update(status="completed", result=results[member],
                       model_elapsed_seconds=float(manifest.get("requested_run_seconds", 0)))
        else:
            candidates = []
            directory = root / "members" / f"member-{member:04d}"
            for parent in {path.parent for path in directory.rglob("gpuwmrst_d*.npz")}:
                candidates.extend(discover_checkpoint_sets(parent))
            candidates.sort(key=lambda item: (item.valid_time, item.handle.stat().st_mtime_ns), reverse=True)
            for candidate in candidates:
                try:
                    header = read_restart_header(candidate.handle)
                    members = (tree_restart_members(candidate.handle) if header.get("domain_ids") else candidate.members)
                    for path in members.values():
                        validate_manifest_checkpoint(path)
                    elapsed = float(header["elapsed_seconds"])
                    binding_path = candidate.handle.with_suffix(".member.json")
                    binding = json.loads(binding_path.read_text(encoding="utf-8"))
                    if (binding.get("member_id") != member
                            or binding.get("identity_sha256") != identity["identity_sha256"]):
                        raise ValueError("checkpoint sidecar belongs to another ensemble member")
                    known = {relative_file(root, path): digest_file(path) for path in members.values()}
                    if known != {row["path"]: row["sha256"] for row in binding["files"]}:
                        raise ValueError("member checkpoint files differ from their committed owner binding")
                except (ValueError, KeyError, OSError):
                    continue
                files.update(members.values())
                files.add(binding_path)
                row.update(status="checkpoint", checkpoint=relative_file(root, candidate.handle),
                           model_elapsed_seconds=elapsed)
                break
        rows.append(row)
    record = {"schema": CONTRACT, "created_time_ns": time.time_ns(), "identities": identities, "members": rows,
        "files": [{"path": relative_file(root, path), "bytes": path.stat().st_size,
                   "sha256": digest_file(path)} for path in sorted(files)]}
    atomic_json(output or root / ROSTER, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    seal(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
