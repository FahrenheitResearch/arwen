"""Delete owned replay arrays only after a flushed per-run verdict exists."""
import argparse
import csv
import json
from pathlib import Path


def prune(root):
    root = Path(root).resolve(strict=True)
    manifest = root / "MANIFEST.txt"
    if root == Path(root.anchor) or not manifest.is_file():
        raise ValueError("an owned lane root with MANIFEST.txt is required")
    paths = []
    for result in (root / "replay").glob("shard-*/results/*of48.jsonl"):
        for line in result.read_text().splitlines():
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue  # the writer can be flushing the final row
            if not all(k in v for k in ("case", "combo", "outcome")):
                continue
            for arm in ("woof", "woof-repeat"):
                directory = result.parents[1] / arm / v["case"] / v["combo"]
                if not directory.resolve().is_relative_to(root):
                    raise ValueError("verdict path leaves the owned lane root")
                for path in directory.glob("step*.bin"):
                    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                        continue
                    paths.append(path)
    log = root / "receipts" / "deleted-raw.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    deleted = []
    with manifest.open("a") as inventory, log.open("a", newline="") as stream:
        writer = csv.writer(stream)
        for path in paths:
            size = path.stat().st_size
            inventory.write(str(path) + "\n"); inventory.flush()
            path.unlink()
            writer.writerow([str(path), size, "per-run verdict recorded"]); stream.flush()
            deleted.append(dict(path=str(path), bytes=size))
    return deleted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    rows = prune(a.root)
    print(json.dumps(dict(deleted_files=len(rows), deleted_bytes=sum(r["bytes"] for r in rows))))


if __name__ == "__main__":
    main()
