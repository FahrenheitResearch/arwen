"""Package an owned engine checkout for a detached node qualification run."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("archive", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--allow-dirty", action="store_true")
    selection.add_argument("--committed-tree", action="store_true",
                           help="read exact HEAD blobs while later-phase edits continue")
    args = parser.parse_args()
    root = args.checkout.resolve()
    if args.archive.exists():
        raise ValueError("use a new snapshot archive path")
    paths = ("gpuwm", "tools/sfire_coupled_ideal")
    dirty = git(root, "status", "--porcelain", "--", *paths)
    if dirty and not (args.allow_dirty or args.committed_tree):
        raise ValueError("commit the owned engine and harness changes before final qualification")
    if args.committed_tree:
        tip = git(root, "rev-parse", "HEAD")
        committed = subprocess.check_output(["git", "-C", str(root), "archive",
                                            "--format=tar", tip, *paths])
        blobs = {}
        modes = {}
        with tarfile.open(fileobj=io.BytesIO(committed)) as tree:
            for item in tree:
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError(f"snapshot path is not a regular tracked file: {item.name}")
                blobs[item.name] = tree.extractfile(item).read()
                modes[item.name] = 0o755 if item.mode & 0o111 else 0o644
        files = sorted(blobs)
    else:
        names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "--", *paths).splitlines()
        files = sorted({name for name in names if (root / name).is_file()})
        blobs = {name: (root / name).read_bytes() for name in files}
        modes = {name: 0o755 if name.endswith(".sh") else 0o644 for name in files}
    forbidden = [name for name in files if any(part.startswith(".env") for part in Path(name).parts)]
    if forbidden:
        raise ValueError("snapshot selection includes a forbidden environment file")
    sources = {name: hashlib.sha256(blobs[name]).hexdigest() for name in files
               if name.startswith("gpuwm/") and Path(name).suffix in (".py", ".cu", ".cuh")}
    workspace_sources = {path.relative_to(root).as_posix()
                         for path in (root / "gpuwm").rglob("*")
                         if path.is_file() and path.suffix in (".py", ".cu", ".cuh")}
    if not args.committed_tree and not workspace_sources.issubset(files):
        raise ValueError("engine contains ignored code files that are absent from the selected snapshot")
    provenance = {"schema": "sfire-source-snapshot-v1", "git_tip": git(root, "rev-parse", "HEAD"),
                  "branch": git(root, "branch", "--show-current"),
                  "selection": "committed-tree" if args.committed_tree else "workspace",
                  "dirty_selected_paths": [] if args.committed_tree else dirty.splitlines(),
                  "workspace_dirty_selected_paths": dirty.splitlines(),
                  "engine_sources_sha256": sources, "selected_files": files}
    encoded = (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w") as archive:
        for name in files:
            item = tarfile.TarInfo(name)
            item.size = len(blobs[name])
            item.mode = modes[name]
            archive.addfile(item, io.BytesIO(blobs[name]))
        item = tarfile.TarInfo("SFIRE_SOURCE_PROVENANCE.json")
        item.size = len(encoded)
        item.mode = 0o644
        archive.addfile(item, io.BytesIO(encoded))
    print(json.dumps({"archive": str(args.archive), "bytes": args.archive.stat().st_size,
                      "git_tip": provenance["git_tip"], "files": len(files),
                      "engine_source_files": len(sources),
                      "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
