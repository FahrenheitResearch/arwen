"""Pack run_bep_bem_regime dumps into fixtures (no arithmetic).

usage: python to_npz.py OUT_DIR DEST_DIR

Reads every ``OUT_DIR/<variant>/dump/{data.bin,manifest.txt}`` written by
``run_bep_bem_regime.F90`` and writes ``DEST_DIR/bep_bem_<variant>.npz`` with
every array under its dump name (``/`` spelled ``__``), in Fortran index
order, plus ``DEST_DIR/data-sha256sums.txt`` (the raw dumps' hashes).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.urban_wrf471_oracle.bem_dump_to_npz import load_dump


def main(out_dir: str, dest_dir: str) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    lines = []
    for dump in sorted(Path(out_dir).glob("*/dump")):
        variant = dump.parent.name
        arrays = load_dump(dump)
        np.savez_compressed(dest / f"bep_bem_{variant}.npz",
                            **{k.replace("/", "__"): v for k, v in arrays.items()})
        digest = hashlib.sha256((dump / "data.bin").read_bytes()).hexdigest()
        lines.append(f"{digest}  {variant}/dump/data.bin")
        print(f"bep_bem_{variant}.npz  data.bin sha256 {digest}")
    (dest / "data-sha256sums.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
