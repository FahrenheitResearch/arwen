"""Check replay determinism and explicit compiler controls on native streams."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def compare(reference, control):
    result = {}
    for source in sorted((Path(reference) / "fixtures").rglob("*.bin")):
        relative = source.relative_to(Path(reference) / "fixtures")
        target = Path(control) / "fixtures" / relative
        a, b = source.read_bytes(), target.read_bytes()
        aw, bw = np.frombuffer(a, dtype="<u4"), np.frombuffer(b, dtype="<u4")
        result[relative.as_posix()] = {"words": aw.size, "different_words": int(np.count_nonzero(aw != bw)),
                                      "reference_sha256": hashlib.sha256(a).hexdigest(),
                                      "control_sha256": hashlib.sha256(b).hexdigest()}
    return {"arrays": len(result), "words": sum(r["words"] for r in result.values()),
            "different_words": sum(r["different_words"] for r in result.values()), "details": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("control")
    parser.add_argument("receipt")
    args = parser.parse_args()
    result = compare(args.reference, args.control)
    Path(args.receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("arrays", "words", "different_words")}))
    if result["different_words"]:
        raise SystemExit(1)
