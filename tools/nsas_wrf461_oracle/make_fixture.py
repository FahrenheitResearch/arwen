"""Pack an oracle run (cases + wrfobj outputs) into the committed fixtures.

    python tools/nsas_wrf461_oracle/make_fixture.py BUILD_DIR OUT_DIR

Writes OUT_DIR/nsas-nz{40,60}.npz, OUT_DIR/compiler.txt and
OUT_DIR/oracle-sha256sums.txt (the digests of the raw harness files the
fixtures were packed from, and of the fixtures themselves).
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

from gpuwm.verify import nsas_oracle as no


def pack(build: Path, nz: int) -> dict[str, np.ndarray]:
    _, cases = no.read_cases(build / f"cases-nz{nz}.bin")
    cols = no.flatten_columns(cases)
    out = no.read_outputs(build / f"out-nz{nz}-wrfobj.bin", nz, len(cols))
    arrays: dict[str, np.ndarray] = {}
    hdr_i = np.array([[ci, k, len(c.columns), c.family, c.mp_physics,
                       c.dx_factor_nsas, c.stepcu, c.itimestep]
                      for ci, k, c, _ in cols], dtype=np.int32)
    hdr_f = np.array([[c.dt, c.dx, c.pgcon, col.xland, col.hpbl, col.hfx,
                       col.qfx] for _, _, c, col in cols], dtype=np.float32)
    arrays["hdr_i"] = hdr_i
    arrays["hdr_f"] = hdr_f
    for name in no.IFACE_FIELDS:
        arrays[f"in.{name}"] = np.stack([col.iface[name] for *_, col in cols])
    for name in no.MASS_FIELDS:
        arrays[f"in.{name}"] = np.stack([col.mass[name] for *_, col in cols])
    for name, value in out.items():
        arrays[f"out.{name}"] = value.astype(np.float32)
    return arrays


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    build, dest = Path(argv[0]), Path(argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    lines = []
    for nz in (40, 60):
        for raw in (f"cases-nz{nz}.bin", f"out-nz{nz}-wrfobj.bin"):
            digest = hashlib.sha256((build / raw).read_bytes()).hexdigest()
            lines.append(f"{digest}  raw/{raw}")
        path = dest / f"nsas-nz{nz}.npz"
        np.savez_compressed(path, **pack(build, nz))
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                     f"{path.name}")
    (dest / "oracle-sha256sums.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    shutil.copyfile(build / "compiler.txt", dest / "compiler.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
