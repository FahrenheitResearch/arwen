"""Restore unchanged oracle assets omitted from the slim export, on a box.

Reads committed blobs already present in the sprint repository. No downloads
or box-to-box copies. Changed assets in the lane always win.
"""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "4d1079249ad0cc2553a9632ac9ef267553913dd1"
ROOTS = ("tests/data", "tests/fixtures", "gpuwm/data", "gpuwm-data/gpuwm_data/data",
         "evidence/p3-fortran-oracle-20260828/fixtures",
         "evidence/p3-fortran-oracle-20260828/dumps",
         "evidence/p3-cuda-verify-20260829/fixtures2",
         "evidence/p3-cuda-verify-20260829/dumps",
         "tools/rrtmg_wrf461_oracle/sw_fixtures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    listed = subprocess.check_output(["git", "ls-tree", "-r", BASE, "--", *ROOTS],
                                     cwd=ROOT, text=True)
    restored = []
    for line in listed.splitlines():
        metadata, relative = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        if kind != "blob" or mode != "100644":
            continue
        target = ROOT / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            subprocess.run(["git", "cat-file", "blob", blob], cwd=ROOT,
                           check=True, stdout=stream)
        digest = subprocess.check_output(["git", "hash-object", str(target)],
                                         cwd=ROOT, text=True).strip()
        if digest != blob:
            raise RuntimeError(f"Restored oracle blob differs: {relative}")
        restored.append({"path": relative, "blob": blob,
                         "bytes": target.stat().st_size})
        with args.manifest.open("a", encoding="utf-8") as stream:
            stream.write(str(target) + "\n")
    args.out.write_text(json.dumps({"base": BASE, "assets": restored}, indent=2)
                        + "\n", encoding="utf-8")
    print(f"restored {len(restored)} unchanged committed oracle assets")


if __name__ == "__main__":
    main()
