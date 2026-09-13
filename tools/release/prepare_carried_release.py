"""Emit and attach exact carried-source changes to an unpublished release packet.

The packet cannot omit carried changes or bind a report to a different engine
revision. Normalization and pair comparison stay with the selected receiver.
No publication, classification, physics merge or GPU operation happens here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tomllib

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_release_carried_channel", ROOT / "tools/carried_physics_channel.py")
channel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(channel)

VERIFICATION_SCHEMA = "arwen.carried-release-verification.v1"
ASSET_SCHEMA = "arwen.carried-release-asset.v1"


def row(path: Path) -> dict:
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(filename=path.name, bytes=path.stat().st_size, sha256=sha)


def version_at(repo: Path, revision: str) -> str:
    value = tomllib.loads(channel.git(repo, "show", revision + ":pyproject.toml").decode("utf-8"))["project"]["version"]
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]*", value) is None:
        raise channel.ChannelError("release endpoint has no explicit package version")
    return value


def artifact_name(manifest: dict) -> str:
    return f"gpuwm-carried-physics-v{manifest['old']['version']}-v{manifest['new']['version']}.json"


def summary(manifest: dict) -> dict:
    return dict(release_id=manifest["id"], old=manifest["old"], new=manifest["new"],
        scope_sha256=channel.digest(channel.canonical(manifest["scope"])),
        scoped_files=len(manifest["files"]),
        changed_engine_files=[item["path"] for item in manifest["files"] if item["changed"]])


def emit(repo: Path, old_ref: str, new_ref: str, receiver_path: Path,
         output: Path, *, old_repo: Path | None = None) -> dict:
    """Capture immutable endpoint bytes and their independently checked identity."""
    old_repo = old_repo or repo
    old = channel.GitTree(old_repo, old_ref, "resolving")
    new = channel.GitTree(repo, new_ref, "resolving")
    receiver = channel.load_receiver(receiver_path)
    manifest = channel.create_release(repo, old.commit, new.commit, receiver.channel_scope(),
        version_at(old_repo, old.commit), version_at(repo, new.commit), old_repo=old_repo)
    channel.verify_git(manifest, repo, old_repo=old_repo)
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / artifact_name(manifest)
    channel.write_new(artifact, manifest)
    verification = channel.seal(dict(schema=VERIFICATION_SCHEMA, status="PASS",
        asset=row(artifact), **summary(manifest), endpoint_versions_checked=True,
        receiver_protocol=receiver.channel_protocol(),
        qualification="All scoped bytes, additions, removals and directory members match the named immutable Git endpoints; package versions match endpoint metadata. Pair fingerprints are receiver measurements, not release-hunk hashes."))
    receipt = artifact.with_name(artifact.stem + ".verification.json")
    channel.write_new(receipt, verification)
    return dict(release=artifact, verification=receipt)


def verify_asset(artifact: Path, receipt: Path, *, revision: str, version: str,
                 repo: Path | None = None, old_repo: Path | None = None) -> dict:
    """Verify packet bytes before they become declared release attachments."""
    manifest = channel.read_json(artifact)
    channel.unpack_release(manifest)
    if manifest["new"]["commit"] != revision or manifest["new"]["version"] != version:
        raise channel.ChannelError("carried channel does not describe this release revision/version")
    if artifact.name != artifact_name(manifest):
        raise channel.ChannelError("carried channel filename disagrees with its endpoint versions")
    if receipt.name != artifact.stem + ".verification.json":
        raise channel.ChannelError("carried verification filename does not match its channel")
    proof = channel.read_json(receipt)
    channel.checked(proof, VERIFICATION_SCHEMA)
    if proof.get("status") != "PASS" or proof.get("endpoint_versions_checked") is not True:
        raise channel.ChannelError("carried channel needs a completed endpoint verification")
    if proof.get("asset") != row(artifact) or any(proof.get(k) != v for k, v in summary(manifest).items()):
        raise channel.ChannelError("carried verification does not bind these exact channel bytes")
    protocol = proof.get("receiver_protocol", {})
    if protocol.get("schema") != "arwen.carried-receiver.v1" or protocol.get("byte_api") != "named-raw-sides.v1":
        raise channel.ChannelError("carried verification has no compatible receiver protocol")
    for key in ("receiver_sha256", "rewiring_sha256", "difflib_sha256"):
        if not isinstance(protocol.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", protocol[key]) is None:
            raise channel.ChannelError("carried verification is missing an actual receiver identity")
    if repo is not None:
        channel.verify_git(manifest, repo, old_repo=old_repo)
        if (version_at(repo, revision) != version
                or version_at(old_repo or repo, manifest["old"]["commit"]) != manifest["old"]["version"]):
            raise channel.ChannelError("carried endpoint package versions changed or were mislabeled")
    return dict(schema=ASSET_SCHEMA, release_id=manifest["id"], release=row(artifact), verification=row(receipt))


def attach_to_packet(artifact: Path, receipt: Path, stage: Path, *, revision: str,
                     version: str, repo: Path, old_repo: Path | None = None) -> dict:
    fragment = verify_asset(artifact, receipt, revision=revision, version=version,
                            repo=repo, old_repo=old_repo)
    if not stage.is_dir():
        raise channel.ChannelError("create the owned packet staging directory before attachment")
    for source in (artifact, receipt):
        with source.open("rb") as inp, (stage / source.name).open("xb") as out:
            shutil.copyfileobj(inp, out, 1 << 20)
    copied = verify_asset(stage / artifact.name, stage / receipt.name, revision=revision, version=version)
    if copied != fragment:
        raise channel.ChannelError("carried channel changed while copying into the packet")
    return fragment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("emit")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--old-repo", type=Path)
    create.add_argument("--old", required=True)
    create.add_argument("--new", required=True)
    create.add_argument("--receiver", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--channel", type=Path, required=True)
    verify.add_argument("--verification", type=Path, required=True)
    verify.add_argument("--revision", required=True)
    verify.add_argument("--version", required=True)
    verify.add_argument("--repo", type=Path)
    verify.add_argument("--old-repo", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "emit":
            value = emit(args.repo, args.old, args.new, args.receiver, args.out, old_repo=args.old_repo)
        else:
            value = verify_asset(args.channel, args.verification, revision=args.revision,
                version=args.version, repo=args.repo, old_repo=args.old_repo)
        print(json.dumps(value, indent=2, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"carried release: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
