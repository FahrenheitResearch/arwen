"""Fold build.sh's fixture tree into one compressed .npz plus receipts.

    python pack.py BUILD_DIR OUT_DIR

Writes OUT_DIR/gsl_drag_wrf461.npz (keys "<grid>/<case>/<name>", Fortran
index order kept), copies TOOLCHAIN.txt, SOURCES.sha256, HARNESS.sha256 and
libm-symbols.txt, records the sha256 of every raw .bin the oracle wrote
(RAW.sha256) and of the packed file (FIXTURES.sha256), and measures the two
attribution builds (WRF's -O2 flags; a correctly rounded atan2f) against the
oracle of record word for word (VARIANTS.json).  Nothing here computes a
physical value.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load_tree(root: Path) -> dict:
    kinds = {"f4": np.dtype("<f4"), "i4": np.dtype("<i4")}
    cases = {}
    for manifest in sorted(root.glob("*/*/MANIFEST.txt")):
        case = manifest.parent
        arrays = {}
        for line in manifest.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            name, kind, rank = parts[0], parts[1], int(parts[2])
            shape = tuple(int(v) for v in parts[3:3 + rank])
            raw = np.fromfile(case / f"{name}.bin", dtype=kinds[kind])
            arrays[name] = (raw.reshape(shape, order="F") if shape
                            else raw[0]).astype(kinds[kind].newbyteorder("="))
        cases[f"{case.parent.name}/{case.name}"] = arrays
    return cases


def ulp(a, b):
    def key(x):
        i = np.ascontiguousarray(x, np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(key(a) - key(b))


def main(build: Path, out: Path) -> None:
    record = load_tree(build / "fixtures")
    out.mkdir(parents=True, exist_ok=True)
    flat = {f"{case}/{name}": value for case, arrays in record.items()
            for name, value in arrays.items()}
    target = out / "gsl_drag_wrf461.npz"
    np.savez_compressed(target, **flat)
    for name in ("TOOLCHAIN.txt", "SOURCES.sha256", "HARNESS.sha256",
                 "libm-symbols.txt"):
        shutil.copy(build / "fixtures" / name, out / name)
    raw = sorted((build / "fixtures").glob("*/*/*"))
    (out / "RAW.sha256").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  "
        f"{p.relative_to(build / 'fixtures').as_posix()}\n" for p in raw))
    (out / "FIXTURES.sha256").write_text(
        f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {target.name}\n")
    variants = {}
    for variant in ("fixtures-wrfflags", "fixtures-cr-atan2"):
        other = load_tree(build / variant)
        rows = {}
        for case, arrays in record.items():
            if case.endswith("/inputs"):
                continue
            for name, value in arrays.items():
                if value.dtype != np.float32 or value.ndim == 0:
                    continue
                d = ulp(value, other[case][name])
                if d.max() > 0:
                    a = value.reshape(value.shape[0], -1)
                    b = other[case][name].reshape(value.shape[0], -1)
                    cols = np.flatnonzero(
                        (a.view(np.uint32) != b.view(np.uint32)).any(axis=1))
                    absd = np.abs(value.astype(np.float64)
                                  - other[case][name].astype(np.float64))
                    rows[f"{case}/{name}"] = {
                        "max_ulp": int(d.max()), "differ": int((d > 0).sum()),
                        "max_abs": float(np.nanmax(absd)),
                        "columns": [int(c) for c in cols]}
        variants[variant] = rows
    (out / "VARIANTS.json").write_text(json.dumps(variants, indent=1) + "\n")
    print(target, target.stat().st_size, "bytes;",
          {k: len(v) for k, v in variants.items()}, "differing fields")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
