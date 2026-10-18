"""Export first-parent changes over the full base and validate real git am.

Run on a sprint box. All scratch and validation paths must be lane-owned.
The full-base overlay preserves files absent from the slim source export.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def git(repo, *args, env=None, data=None):
    result = subprocess.run(["git", "-C", str(repo), *map(str, args)],
                            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    return result.stdout


def overlay_records(raw):
    records = raw.split(b"\0")
    entries = []
    for offset in range(0, len(records) - 1, 2):
        metadata, path = records[offset:offset + 2]
        _old_mode, mode, _old_blob, blob, _status = metadata[1:].split()
        if mode == b"000000":
            mode = b"0"
        entries.append(mode + b" " + blob + b"\t" + path + b"\n")
    return b"".join(entries)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--branch", default="lane/pool-combo-round4-rescore")
    ap.add_argument("--slim-base", default="base/2.8.7-slim")
    ap.add_argument("--full-base", default="4d1079249ad0cc2553a9632ac9ef267553913dd1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--validate", type=Path, required=True)
    ap.add_argument("--validation-branch", default="lane/pool-coupled-exact-chain")
    args = ap.parse_args(argv)
    repo, out, validation = args.repo.resolve(), args.out.resolve(), args.validate.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if validation.exists():
        raise ValueError("validation path must be new")
    env = dict(os.environ, GIT_INDEX_FILE=str(out / "export-index"))
    git(repo, "read-tree", args.full_base, env=env)
    previous = git(repo, "write-tree", env=env).strip().decode()
    commits = git(repo, "rev-list", "--reverse", "--first-parent",
                  args.slim_base + ".." + args.branch).decode().splitlines()
    patches, records = [], []
    for commit in commits:
        raw = git(repo, "diff-tree", "--raw", "--no-commit-id", "-r", "-z", commit + "^1", commit)
        entries = overlay_records(raw)
        if entries:
            git(repo, "update-index", "--index-info", env=env, data=entries)
        tree = git(repo, "write-tree", env=env).strip().decode()
        if tree == previous:
            continue
        patch = out / f"{len(patches) + 1:04d}.patch"
        header = git(repo, "log", "-1", "--format=email", commit)
        diff = git(repo, "diff", "--binary", "--full-index", "--no-ext-diff", previous, tree)
        patch.write_bytes(header + b"\n" + diff)
        patches.append(patch)
        records.append(dict(patch=patch.name, source_commit=commit, before_tree=previous, after_tree=tree))
        previous = tree
    git(repo, "init", validation)
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip().decode())
    (validation / ".git/objects/info/alternates").write_text(str(common / "objects") + "\n")
    for field in ("user.name", "user.email"):
        configured = subprocess.run(["git", "-C", str(repo), "config", "--get", field], capture_output=True)
        if configured.returncode == 0:
            value = configured.stdout.strip().decode()
        elif configured.returncode == 1:
            # Bundles carry commit identity but not repository configuration.
            # Preserve the desktop repository's last committer in this new,
            # owned validation repository. The shared source config is read only.
            token = "%cn" if field == "user.name" else "%ce"
            value = git(repo, "log", "-1", "--format=" + token, args.branch).strip().decode()
        else:
            raise RuntimeError(configured.stderr.decode())
        git(validation, "config", field, value)
    git(validation, "checkout", "-b", args.validation_branch, args.full_base)
    log = out / "apply.log"
    with log.open("wb") as stream:
        result = subprocess.run(["git", "-C", str(validation), "am", "-3", "--keep-cr", *map(str, patches)],
                                stdout=stream, stderr=subprocess.STDOUT)
    actual = git(validation, "rev-parse", "HEAD^{tree}").strip().decode()
    ok = result.returncode == 0 and actual == previous
    receipt = dict(passed=ok, apply_exit=result.returncode, full_base=args.full_base,
                   slim_base=git(repo, "rev-parse", args.slim_base).strip().decode(),
                   source_tip=git(repo, "rev-parse", args.branch).strip().decode(),
                   expected_tree=previous, applied_tree=actual, patches=records,
                   command=["git", "am", "-3", "--keep-cr", "*.patch"])
    (out / "application.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in
                      ("passed", "apply_exit", "source_tip", "expected_tree", "applied_tree")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
