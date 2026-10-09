"""Does a column's WRF answer depend on where it sits in the loop?

    python -m tools.mynn_sfclay_wrf461_column_oracle.stock_position \
        RUN_COLUMNS_EXE INPUT_DIR WORK_DIR

Runs one compiled column driver (build.sh's o2, o0, wrfstock or
wrfstock_scalar ``run_columns``) on the oracle's columns as written, then
(a) with 1, 2 and 3 filler columns put in front, so every real column moves
to another index of SFCLAY1D_mynn's I loops, and (b) with every column run
alone as a one-column tile.  It prints how many output words of the real
columns changed in each case.  Option sets with spp_pbl=1 are skipped: their
pattern is a function of the column's index, so moving a column changes its
input there by design.

Why this matters: WRF's stock GNU flags vectorize some of those loops, and a
vectorized loop calls libmvec's SIMD powf on full vector lanes while the
loop remainder (and a one-column tile) calls the scalar powf.  The two can
round differently, so the same column can get a different answer depending
only on the tile it sits in.  A column kernel cannot reproduce an answer
that is not a function of the column; this tool measures whether that is
the case for a given build.  CPU only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.mynn_sfclay_wrf461_column_oracle.columns import (
    CONFIGS, FIELDS, OUTPUT_FIELDS,
)
from tools.mynn_sfclay_wrf461_column_oracle.compare import (
    read_columns, read_oracle,
)


def _write_columns(path: Path, cols: dict) -> None:
    import struct
    ncol = len(cols[FIELDS[0]])
    table = np.stack([cols[f] for f in FIELDS], axis=1).astype("<f4")
    path.write_bytes(struct.pack("<ii", ncol, len(FIELDS))
                     + table.tobytes(order="C"))


def _run(exe: Path, columns: Path, configs: Path, out: Path):
    if out.exists():
        out.unlink()
    subprocess.run([str(exe), str(columns), str(configs), str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return read_oracle(out)[1]


def measure(exe: Path, input_dir: Path, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    base_cols = read_columns(input_dir / "columns.bin")
    configs = input_dir / "configs.txt"
    base = _run(exe, input_dir / "columns.bin", configs, work / "shift0.bin")
    result = {}
    for shift in (1, 2, 3):
        # Filler: copies of the first column, so the filler itself is a
        # valid input and never trips a check.
        cols = {f: np.concatenate([np.repeat(base_cols[f][:1], shift),
                                   base_cols[f]]).astype(np.float32)
                for f in FIELDS}
        path = work / f"columns-shift{shift}.bin"
        _write_columns(path, cols)
        moved = _run(exe, path, configs, work / f"shift{shift}.bin")
        changed_words, changed_cols = 0, set()
        for (cfg, _, _, _, a), (_, _, _, _, b) in zip(base, moved):
            if CONFIGS[cfg][7]:
                continue  # the SPP pattern is indexed by column position
            for name in OUTPUT_FIELDS:
                x = a[name].view(np.uint32)
                y = b[name][shift:].view(np.uint32)
                diff = np.nonzero(x != y)[0]
                changed_words += diff.size
                changed_cols.update(int(c) for c in diff)
        result[f"shift {shift}"] = (changed_words, len(changed_cols))
    # Solo: every column run as a one-column tile.  A one-iteration loop
    # never enters a vectorized loop body, so this is the column's answer on
    # the scalar path of the same binary.
    changed_words, changed_cols = 0, set()
    for c in range(len(base_cols[FIELDS[0]])):
        cols = {f: base_cols[f][c:c + 1].astype(np.float32) for f in FIELDS}
        path = work / "columns-solo.bin"
        _write_columns(path, cols)
        solo = _run(exe, path, configs, work / "solo.bin")
        for (cfg, _, _, _, a), (_, _, _, _, b) in zip(base, solo):
            if CONFIGS[cfg][7]:
                continue
            for name in OUTPUT_FIELDS:
                if a[name][c].view(np.uint32) != b[name][0].view(np.uint32):
                    changed_words += 1
                    changed_cols.add(c)
    result["solo"] = (changed_words, len(changed_cols))
    return result


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    exe, input_dir, work = (Path(a) for a in argv)
    total = 0
    for label, (words, cols) in measure(exe, input_dir, work).items():
        print(f"{label}: {words} output words changed, in {cols} columns")
        total += words
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
