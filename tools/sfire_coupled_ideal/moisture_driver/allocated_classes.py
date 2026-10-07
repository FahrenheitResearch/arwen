"""Change only the allocated class extent in the native driver harness."""
from pathlib import Path
import argparse
import hashlib
import json


def emit(source, destination, classes):
    if classes < 5:
        raise ValueError("the native harness exercises five active moisture classes")
    original = Path(source).read_text()
    marker = "mx=6,my=4,nc=5,rx=2"
    if original.count(marker) != 1:
        raise ValueError("native moisture harness allocation declaration changed")
    text = original.replace(marker, f"mx=6,my=4,nc={classes},rx=2")
    Path(destination).write_text(text)
    return {"allocated_classes": classes, "active_class_limit": 5,
            "original_wrapper_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(text.encode()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("classes", type=int)
    args = parser.parse_args()
    print(json.dumps(emit(args.source, args.destination, args.classes)))
