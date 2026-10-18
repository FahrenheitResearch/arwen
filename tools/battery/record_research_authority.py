"""Refresh source digests after reviewing the research import/window contracts.

Product membership, window requirements and input notes are reviewed metadata.
This recorder preserves them and updates only their source authority digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def record(root: Path) -> None:
    path = root / "gpuwm/data/tui/research-diagnostics.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["authority_normalization"] != "UTF-8 with LF line endings":
        raise ValueError("unsupported authority normalization")
    for source in document["authority"]:
        payload = (root / source).read_text(encoding="utf-8").encode("utf-8")
        document["authority"][source] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"recorded {len(document['authority'])} source digests in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    record(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
