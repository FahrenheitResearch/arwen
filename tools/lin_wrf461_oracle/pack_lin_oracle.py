"""Pack the Lin oracle's binary input/output pair into the test fixture.

Reads lin-inputs.bin, lin-outputs.bin and lin-regimes.json from the oracle
build directory and writes one compressed .npz whose every expected value
was returned by WRF's own lin_et_al.  Provenance (WRF commit, compiler,
source hashes) travels inside the archive as a JSON string.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

IN3 = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "rho", "pii", "p", "z",
       "dz8w")
IN2 = ("dt", "ht", "rainnc", "snownc", "graupelnc")
OUT3 = ("th", "qv", "qc", "qr", "qi", "qs", "qg")
OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
        "graupelncv", "sr")


def read_inputs(path: Path):
    raw = path.read_bytes()
    ncol, nz = np.frombuffer(raw[:8], dtype="<i4")
    data = np.frombuffer(raw[8:], dtype="<f4")
    out, off = {}, 0
    for name in IN2:
        out[name] = data[off:off + ncol].copy()
        off += ncol
    for name in IN3:
        out[name] = data[off:off + ncol * nz].reshape(nz, ncol).copy()
        off += ncol * nz
    assert off == data.size, (off, data.size)
    return int(ncol), int(nz), out


def read_outputs(path: Path, ncol: int, nz: int):
    raw = path.read_bytes()
    hdr = np.frombuffer(raw[:8], dtype="<i4")
    assert tuple(hdr) == (ncol, nz), hdr
    data = np.frombuffer(raw[8:], dtype="<f4")
    out, off = {}, 0
    for name in OUT3:
        out[name] = data[off:off + ncol * nz].reshape(nz, ncol).copy()
        off += ncol * nz
    for name in OUT2:
        out[name] = data[off:off + ncol].copy()
        off += ncol
    out["refl_10cm"] = data[off:off+ncol*nz].reshape(nz,ncol).copy()
    off += ncol*nz
    assert off == data.size, (off, data.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("fixture", type=Path)
    args = ap.parse_args()
    b = args.build_dir
    ncol, nz, inputs = read_inputs(b / "lin-inputs.bin")
    expected = read_outputs(b / "lin-outputs.bin", ncol, nz)
    regimes = json.loads((b / "lin-regimes.json").read_text(encoding="utf-8"))
    sums = (b / "oracle-sha256sums.txt").read_text(encoding="utf-8")
    provenance = {
        "wrf_commit": (b / "wrf-commit.txt").read_text().strip(),
        "compiler": (b / "compiler.txt").read_text().strip(),
        "glibc": (b / "glibc.txt").read_text().strip(),
        "libm_report": (b / "libm-report.txt").read_text(),
        "sha256": sums,
        "inputs_sha256": hashlib.sha256(
            (b / "lin-inputs.bin").read_bytes()).hexdigest(),
        "outputs_sha256": hashlib.sha256(
            (b / "lin-outputs.bin").read_bytes()).hexdigest(),
    }
    arrays = {f"in_{k}": v for k, v in inputs.items()}
    arrays.update({f"out_{k}": v for k, v in expected.items()})
    arrays["regimes"] = np.array(regimes)
    arrays["provenance"] = np.array(json.dumps(provenance, indent=1))
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixture, **arrays)
    print(f"packed {ncol} columns x {nz} levels -> {args.fixture}")


if __name__ == "__main__":
    main()
