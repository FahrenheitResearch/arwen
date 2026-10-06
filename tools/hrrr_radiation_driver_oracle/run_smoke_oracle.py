"""Generate a small source smoke fixture for numerical grading, not weather skill.

Usage: python -m tools.hrrr_radiation_driver_oracle.run_smoke_oracle BUILD_DIR OUTPUT.npz
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.hrrr_radiation_driver_oracle.run_aer3_oracle import inputs


def main():
    build = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    receipt = json.loads((build / "smoke-receipt.json").read_text(encoding="utf-8"))
    if not receipt.get("compiled"):
        raise ValueError("the source smoke oracle must be compiled before generating evidence")
    work = build / "smoke_work"
    work.mkdir(parents=True, exist_ok=True)
    data = inputs(seed=20261004, ncol=32, nz=50)
    p, t, qv = (data[key] for key in ("p", "t", "qv"))
    rho = p / (np.float32(287) * t * (np.float32(1) + np.maximum(qv, np.float32(0)) / np.float32(.622)))
    rng = np.random.default_rng(20261004)
    smoke = rng.uniform(0, 100, p.shape).astype(np.float32)
    smoke[0] = 0
    smoke[1] = 10000
    smoke[2] = 0
    smoke[2, 3] = 10000
    data["smoke_ugkg"] = smoke
    data["rho_dry"] = rho.astype(np.float32)
    ncol, nz = p.shape
    with (work / "aer3_in.bin").open("wb") as handle:
        handle.write(np.array([ncol, nz], np.int32).tobytes())
        for key in ("p", "t", "qv", "dz8w", "nwfa", "nifa"):
            handle.write(np.ascontiguousarray(data[key].T).tobytes())
        handle.write(data["ht"].tobytes())
        for key in ("smoke_ugkg", "rho_dry"):
            handle.write(np.ascontiguousarray(data[key].T).tobytes())
    subprocess.run([str(build / "oracle_aer3_smoke")], cwd=work, check=True)
    raw = np.fromfile(work / "aer3_out.bin", np.float32)
    block = ncol * nz * 14
    expected_size = 3 * block + 4 * ncol * nz + ncol
    if raw.size != expected_size:
        raise ValueError(f"source oracle output words {raw.size} != {expected_size}")
    values = {}
    for n, name in enumerate(("tauaer", "ssaaer", "asyaer")):
        values[f"out/{name}"] = raw[n * block:(n + 1) * block].reshape(14, nz, ncol).transpose(2, 1, 0).copy()
    rest = raw[3 * block:]
    values["out/taod5503d"] = rest[:ncol * nz].reshape(nz, ncol).T.copy()
    values["out/taod5502d"] = rest[ncol * nz:ncol * nz + ncol].copy()
    offset = ncol * nz + ncol
    for name in ("smoke_aod", "pm_posted_kgm3", "smoke_recovered_ugkg"):
        values[f"out/{name}"] = rest[offset:offset + ncol * nz].reshape(nz, ncol).T.copy()
        offset += ncol * nz
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{f"in/{key}": value for key, value in data.items()},
                        **values, receipt=json.dumps(receipt))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"file": str(output), "size": output.stat().st_size,
                      "sha256": digest, "purpose": "source numerical oracle, not a forecast"}))


if __name__ == "__main__":
    main()
