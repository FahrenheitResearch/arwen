"""Freeze the RRTMG coefficient oracle dumps into a per-array SHA manifest.

Reads the tagged big-endian dump streams written by coeffs_dump_lw /
coeffs_dump_sw (built and run by build.sh) and writes
gpuwm/data/wrf_radiation/rrtmg_coeffs_oracle_manifest.json mapping every
``module/var`` entry to its dtype, Fortran extents, and the SHA-256 of its
native little-endian, C-order byte image.  tests/test_rrtmg_coeffs.py
compares gpuwm.ingest.rrtmg_coeffs output against this manifest (and
against the raw dumps directly when they are present on disk), so the
committed manifest is the portable form of the bit-for-bit oracle gate.

Usage: python make_coeffs_manifest.py BUILD_DIR [OUTPUT_JSON]
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_dump(path: Path):
    """Parse one tagged big-endian oracle dump stream."""
    entries = {}
    order = []
    with open(path, "rb") as f:
        while True:
            raw = f.read(4)
            if not raw:
                break
            (nlen,) = struct.unpack(">i", raw)
            name = f.read(nlen).decode("ascii")
            (dtype_code,) = struct.unpack(">i", f.read(4))
            (rank,) = struct.unpack(">i", f.read(4))
            dims = struct.unpack(f">{rank}i", f.read(4 * rank)) if rank \
                else ()
            count = int(np.prod(dims)) if rank else 1
            kind = "f4" if dtype_code == 4 else "i4"
            data = np.frombuffer(f.read(4 * count), dtype=">" + kind)
            arr = data.astype(kind)
            arr = arr.reshape(dims, order="F") if rank else arr.reshape(())
            entries[name] = arr
            order.append(name)
    return entries, order


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    build_dir = Path(sys.argv[1])
    default_out = Path(__file__).resolve().parents[2] / "gpuwm" / "data" / \
        "wrf_radiation" / "rrtmg_coeffs_oracle_manifest.json"
    out_path = Path(sys.argv[2]) if len(sys.argv) == 3 else default_out

    manifest = {
        "description": (
            "Per-array SHA-256 of the WRF v4.6.1 RRTMG post-init "
            "coefficient state (raw RRTMG_*_DATA images plus every cmbgb "
            "reduction target), dumped by "
            "tools/rrtmg_wrf461_oracle/coeffs_dump_lw|sw after "
            "rrtmg_lwinit/rrtmg_swinit ran on the unmodified WRF sources. "
            "Hashes cover the native little-endian C-order byte image of "
            "each array."),
        "dumps": {},
        "arrays": {},
    }
    for side in ("lw", "sw"):
        dump_path = build_dir / f"rrtmg-coeffs-{side}.dump"
        entries, order = read_dump(dump_path)
        manifest["dumps"][f"rrtmg-coeffs-{side}.dump"] = \
            sha256_file(dump_path)
        for name in order:
            arr = entries[name]
            manifest["arrays"][name] = {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "sha256": array_sha256(arr),
            }
    out_path.write_text(json.dumps(manifest, indent=1, sort_keys=True)
                        + "\n")
    print(f"wrote {out_path} ({len(manifest['arrays'])} arrays)")


if __name__ == "__main__":
    main()
