"""Pack run_bep_bem dumps into the committed fixtures (no arithmetic).

usage: python bem_dump_to_npz.py OUT_DIR DEST_DIR

Reads each ``OUT_DIR/<variant>/dump/{data.bin,manifest.txt}`` written by
``run_bep_bem.F90`` and writes ``DEST_DIR/bep_bem_<variant>.npz`` with every
array under its dump name (``/`` spelled ``__``), in Fortran index order.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

VARIANTS = ("stock", "lcz", "gr1pv", "gr2", "long")


def load_dump(directory: Path) -> dict[str, np.ndarray]:
    raw = (directory / "data.bin").read_bytes()
    as_f4 = np.frombuffer(raw, "<f4")
    as_i4 = np.frombuffer(raw, "<i4")
    out: dict[str, np.ndarray] = {}
    for line in (directory / "manifest.txt").read_text().splitlines():
        name, kind, ndim, d1, d2, d3, d4, offset = line.split()
        shape = tuple(int(d) for d in (d1, d2, d3, d4))[:int(ndim)]
        count = int(np.prod(shape))
        start = int(offset)
        words = as_f4 if kind == "f4" else as_i4
        out[name] = words[start:start + count].reshape(shape, order="F").copy()
    return out


def main(out_dir: str, dest_dir: str) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        dump = Path(out_dir) / variant / "dump"
        arrays = load_dump(dump)
        np.savez_compressed(dest / f"bep_bem_{variant}.npz",
                            **{k.replace("/", "__"): v for k, v in arrays.items()})
        digest = hashlib.sha256((dump / "data.bin").read_bytes()).hexdigest()
        print(f"bep_bem_{variant}.npz  data.bin sha256 {digest}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
