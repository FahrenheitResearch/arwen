"""Delete only named, manifested lane paths and record every file's size."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "MANIFEST.txt"
    # Repair an initial shell printf whose newline escape was stripped by
    # the Windows command transport. No path outside this lane is admitted.
    text = manifest.read_text().replace("n" + str(root) + "/", "\n" + str(root) + "/")
    manifest.write_text(text)
    declared = {Path(p) for p in text.splitlines() if p}
    receipt = root / args.receipt
    assert receipt.parent.resolve().is_relative_to(root)
    entries = []
    retained = root / "retained-receipts"
    retained.mkdir(exist_ok=True)
    with manifest.open("a") as stream:
        stream.write(str(retained) + "\n")
    for relative in args.paths:
        path = root / relative
        assert path.is_relative_to(root) and path != root
        assert path in declared, f"path is not manifested: {path}"
        assert path.parent.resolve().is_relative_to(root)
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or path.is_file():
            entries.append({"path": str(path), "bytes": path.lstat().st_size})
            path.unlink()
            continue
        for directory, folders, files in os.walk(path, followlinks=False):
            for name in files + [n for n in folders if (Path(directory) / n).is_symlink()]:
                child = Path(directory) / name
                entries.append({"path": str(child), "bytes": child.lstat().st_size})
                if (not child.is_symlink() and child.suffix.lower() in {".json", ".csv", ".log", ".md"}
                        and child.stat().st_size < 10_000_000):
                    destination = retained / child.relative_to(root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(child, destination)
        shutil.rmtree(path)
    receipt.write_text(json.dumps({"deleted_bytes": sum(e["bytes"] for e in entries),
                                   "files": entries}, indent=2) + "\n")
    with manifest.open("a") as stream:
        stream.write(str(receipt) + "\n")
    print(json.dumps({"deleted_bytes": sum(e["bytes"] for e in entries), "files": len(entries)}))


if __name__ == "__main__":
    main()
