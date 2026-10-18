"""Replay every merge-aware mail patch against the full engine base on a box.

Uses real git mailinfo and git apply --cached --3way, without creating commits,
changing branches, or checking out another source tree. Omitted base paths stay
in the full tree. The receipt records each applied patch and the final tree.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "4d1079249ad0cc2553a9632ac9ef267553913dd1"
SLIM = "3836f68c4fa77bd60ecf527bb408ee1832b8e359"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    lane = ROOT.parent.resolve()
    if lane != Path("/work/pool-verify-lanes-stack") or lane not in out.parents:
        parser.error("Run on G1 with an output directory inside this lane")
    if out.exists():
        parser.error("Use a fresh owned replay directory")
    out.mkdir()
    manifest = lane / "MANIFEST.txt"
    with manifest.open("a") as stream:
        stream.write(str(out) + "\n")
    record = {"engine_base": BASE, "slim_base": SLIM, "patches": [],
              "method": "git mailinfo then git apply --cached --3way; no commits"}
    log = (out / "git.log").open("w")

    def git(*arguments, index=None, data=None):
        env = dict(os.environ)
        if index:
            env["GIT_INDEX_FILE"] = str(index)
        log.write("COMMAND " + json.dumps(["git", *arguments]) + "\n")
        log.flush()
        result = subprocess.run(["git", *arguments], cwd=ROOT, env=env,
                                input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        log.write(result.stderr.decode("utf-8", errors="replace"))
        if result.returncode:
            raise RuntimeError(f"git {arguments[0]} exit={result.returncode}: "
                               + result.stderr.decode("utf-8", errors="replace"))
        return result.stdout

    try:
        record["lane_tip"] = git("rev-parse", "HEAD").decode().strip()
        compose, replay = out / "compose.index", out / "replay.index"
        git("read-tree", BASE, index=compose)
        git("read-tree", BASE, index=replay)
        before = git("write-tree", index=compose).decode().strip()
        commits = git("rev-list", "--reverse", "--first-parent", f"{SLIM}..HEAD").decode().split()
        for commit in commits:
            parent = git("rev-parse", f"{commit}^1").decode().strip()
            rows = git("diff-tree", "--raw", "--no-renames", "--no-commit-id",
                       "-r", parent, commit).decode().splitlines()
            entries = []
            for row in rows:
                metadata, relative = row.split("\t", 1)
                _, mode, _, blob, _ = metadata.split()
                if relative.startswith('"') or "\n" in relative:
                    raise ValueError("Quoted/control path requires explicit parsing")
                entries.append(f"{'0' if mode == '000000' else mode} {blob}\t{relative}\n")
            if entries:
                git("update-index", "--index-info", index=compose,
                    data="".join(entries).encode())
            after = git("write-tree", index=compose).decode().strip()
            if before == after:
                continue
            diff = git("diff", "--binary", "--full-index", "--no-ext-diff", before, after)
            mail = git("log", "-1", "--format=email", commit) + b"\n" + diff
            number = len(record["patches"]) + 1
            mail_path, message, parsed = (out / f"{number:04}.patch",
                                         out / "message.txt", out / "parsed.patch")
            mail_path.write_bytes(mail)
            git("mailinfo", str(message), str(parsed), data=mail)
            git("apply", "--cached", "--3way", str(parsed), index=replay)
            actual = git("write-tree", index=replay).decode().strip()
            if actual != after:
                raise RuntimeError(f"Patch {number} produces {actual}, expected {after}")
            record["patches"].append({"patch": mail_path.name, "lane_commit": commit,
                                      "sha256": hashlib.sha256(mail).hexdigest(),
                                      "bytes": len(mail), "before_tree": before,
                                      "after_tree": after, "passed": True})
            before = after
            print(f"PASS patch {number:04} {commit[:10]}", flush=True)
        full = dict(line.split("\t", 1)[::-1] for line in git("ls-tree", "-r", before).decode().splitlines())
        retained = git("ls-tree", "-r", "HEAD").decode().splitlines()
        for line in retained:
            metadata, relative = line.split("\t", 1)
            if full.get(relative) != metadata:
                raise RuntimeError(f"Full tree differs from retained lane source: {relative}")
        record.update(passed=True, full_source_tree=before,
                      retained_files_checked=len(retained))
    except Exception as error:
        record.update(passed=False, error=str(error))
    finally:
        (out / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
        log.close()
        with manifest.open("a") as stream:
            for path in out.iterdir():
                stream.write(str(path) + "\n")
    print(json.dumps({key: record.get(key) for key in
                      ("passed", "full_source_tree", "retained_files_checked", "error")}))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
