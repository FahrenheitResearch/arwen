"""Read the source-pinned native WRF SFIRE fixtures and measure output words."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).parent / "fixtures"


def load(name):
    root = ROOT
    if name.startswith("corrected/"):
        root = ROOT / "corrected"
        name = name.removeprefix("corrected/")
    receipt = json.loads((root / "receipt.json").read_text())
    path = root / (name + ".npz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["cases"][name]["sha256"]:
        raise ValueError(f"SFIRE oracle corpus hash mismatch: {name}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def words(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"shape differs: {actual.shape} != {expected.shape}")
    a, b = actual.view(np.uint32), expected.view(np.uint32)
    ao = np.where(a & 0x80000000, np.uint32(0xffffffff) - a, a + np.uint32(0x80000000)).astype(np.int64)
    bo = np.where(b & 0x80000000, np.uint32(0xffffffff) - b, b + np.uint32(0x80000000)).astype(np.int64)
    finite = np.isfinite(actual) & np.isfinite(expected)
    distance = np.abs(ao - bo)
    return {"words": actual.size, "different_words": int(np.count_nonzero(a != b)),
            "max_ulp": int(np.max(distance[finite], initial=0)),
            "nonfinite_differences": int(np.count_nonzero((a != b) & ~finite))}
