"""Pack run_myjpbl's raw outputs into the committed fixture (numpy only).

    python tools/myjpbl_wrf461_oracle/pack_fixture.py BUILD_DIR OUT.npz

Reads ``BUILD_DIR/cases.txt`` and, per case, ``out_<case>.bin`` with its
record list ``out_<case>.txt`` (one ``<name> <r|i> <n1> <n2>`` line per raw
record, in file order, after a ``#header ncol nz nstep flqi`` line).  Each
record lands in the archive as ``<case>/<name>``: a level field as a C-order
``(nz+1, ncol)`` float32 array (Fortran ``(ncol, nz+1)`` read back without a
transpose), a surface field as ``(ncol,)``, ``kpbl`` as int32.  The header
is ``<case>/meta = [ncol, nz, nstep, flqi]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def read_case(build: Path, case: str) -> dict[str, np.ndarray]:
    lines = (build / f"out_{case}.txt").read_text().splitlines()
    raw = (build / f"out_{case}.bin").read_bytes()
    head = lines[0].split()
    assert head[0] == "#header", head
    out = {"meta": np.array([int(x) for x in head[1:5]], dtype=np.int32)}
    off = 0
    for line in lines[1:]:
        name, kind, n1, n2 = line.split()
        n1, n2 = int(n1), int(n2)
        count = n1 * n2
        dtype = "<f4" if kind == "r" else "<i4"
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off).copy()
        off += 4 * count
        out[name] = a.reshape(n2, n1) if n2 > 1 else a
    assert off == len(raw), (case, off, len(raw))
    return out


def main(build_dir: str, out_path: str) -> None:
    build = Path(build_dir)
    cases = (build / "cases.txt").read_text().split()
    archive = {}
    for case in cases:
        for name, a in read_case(build, case).items():
            archive[f"{case}/{name}"] = a
    np.savez_compressed(out_path, **archive)
    print(f"packed {len(cases)} cases, {len(archive)} arrays -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
