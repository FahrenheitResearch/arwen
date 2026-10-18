#!/usr/bin/env python3
"""WRF's SFCDIAGS against gpuwm.core.noah_sfcdiags on every float32 PSFC.

    python tools/noah_wrf461_oracle/sfcdiags_sweep.py WRF_SOURCE_ROOT WORKDIR OUT.json

Builds ``sfcdiags_sweep.F90`` against the pinned tree's
``phys/module_sf_sfcdiags.F`` and ``share/module_model_constants.F`` at -O0
(as ``build.sh`` builds the column oracle), feeds it every float32 surface
pressure in [20 kPa, 110 kPa] -- about 17.6 million columns -- with skin
temperatures cycling over 200-330 K and realistic fluxes and humidities, and
compares T2, TH2 and Q2 word for word with the kernel the driver launches
after every Noah step.  TH2 = T2*(1.E5/PSFC)**ROVCP is a libm power of the
pressure ratio, so this is an exhaustive grade of that power over the
pressures a land surface can have.

Columns whose WRF Q2 is not positive are counted apart: there the port
publishes qv(k=1) by design (``PhysicsDriver._refresh_surface_diagnostics``
documents the bound), so they are a declared divergence, not a miss.

Exits nonzero if any other word differs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
FIELDS = ("t2", "th2", "q2")


def _build(source_root: Path, work: Path) -> Path:
    defines = ["-Dwrfmodel", "-DEM_CORE=1", "-DNMM_CORE=0", "-DRWORDSIZE=4",
               "-DIWORDSIZE=4", "-DDWORDSIZE=8", "-DLWORDSIZE=4"]
    free = ["-ffree-form", "-ffree-line-length-none"]

    def run(*args):
        subprocess.run(list(args), check=True, cwd=work)

    run("gfortran", "-c", "-O0", "-cpp", *defines, *free,
        str(source_root / "share/module_model_constants.F"))
    run("gfortran", "-c", "-O0", "-cpp", *defines, *free, "-I", str(work),
        str(source_root / "phys/module_sf_sfcdiags.F"))
    run("gfortran", "-c", "-O0", *free, "-I", str(work),
        str(HERE / "sfcdiags_sweep.F90"))
    run("gfortran", "-o", "sfcdiags_sweep", "module_model_constants.o",
        "module_sf_sfcdiags.o", "sfcdiags_sweep.o")
    return work / "sfcdiags_sweep"


def sweep_columns() -> dict[str, np.ndarray]:
    """Every float32 PSFC in [20 kPa, 110 kPa], with the other inputs varied."""
    f32 = np.float32
    lo = np.array(20000.0, f32).view(np.uint32)
    hi = np.array(110000.0, f32).view(np.uint32)
    psfc = np.arange(lo, hi + 1, dtype=np.uint32).view(f32)
    n = psfc.size
    rng = np.random.default_rng(20261007)
    cols = {
        "psfc": psfc,
        "tsk": (f32(200.0) + (np.arange(n) % 1301).astype(f32) * f32(0.1)
                ).astype(f32),
        "hfx": rng.uniform(-150.0, 600.0, n).astype(f32),
        "qfx": rng.uniform(-5.0e-5, 3.0e-4, n).astype(f32),
        "qsfc": rng.uniform(1.0e-4, 3.0e-2, n).astype(f32),
        "chs2": rng.uniform(1.0e-3, 4.0e-2, n).astype(f32),
        "cqs2": rng.uniform(1.0e-3, 4.0e-2, n).astype(f32),
    }
    # Some columns below SFCDIAGS's 1.E-5 activity threshold, so both arms
    # of each test are in the sweep.
    cols["chs2"][::1000] = f32(5.0e-6)
    cols["cqs2"][::997] = f32(5.0e-6)
    return cols


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    source_root, work, out = Path(argv[1]), Path(argv[2]), Path(argv[3])
    work.mkdir(parents=True, exist_ok=True)
    exe = _build(source_root, work)
    cols = sweep_columns()
    n = cols["psfc"].size
    with open(work / "in.bin", "wb") as fh:
        np.array([n], np.int32).tofile(fh)
        for name in ("psfc", "tsk", "hfx", "qfx", "qsfc", "chs2", "cqs2"):
            cols[name].tofile(fh)
    subprocess.run([str(exe), str(work / "in.bin"), str(work / "out.bin")],
                   check=True)
    wrf = np.fromfile(work / "out.bin", dtype=np.float32).reshape(3, n)
    os.remove(work / "in.bin")
    os.remove(work / "out.bin")

    import cupy as cp

    from gpuwm.certify.compile_platform import nvrtc_build
    from gpuwm.core import constants as c
    from gpuwm.core import noah_sfcdiags
    from gpuwm.core.fp32_ulp import fp32_ulp_distance

    f32 = np.float32
    qv1 = np.full(n, f32(0.008))
    fields = {k: cp.asarray(v) for k, v in cols.items()}
    for name in FIELDS:
        fields[name] = cp.zeros(n, dtype=cp.float32)
    noah_sfcdiags.refresh(fields, cp.asarray(qv1), rd=f32(c.RD),
                          cp_air=f32(c.CP), p0=f32(c.P0), rcp=f32(c.RCP),
                          active_floor=f32(1.0e-5))
    port = {name: cp.asnumpy(fields[name]) for name in FIELDS}

    declared = wrf[2] <= 0.0
    result = {
        "strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "capability": str(cp.cuda.Device().compute_capability),
        "nvrtc": nvrtc_build(),
        "columns": int(n),
        "psfc_range_pa": [20000.0, 110000.0],
        "declared_q2_floor_columns": int(declared.sum()),
        "fields": {},
    }
    failing = 0
    for k, name in enumerate(FIELDS):
        got, want = port[name], wrf[k]
        same = got.view(np.uint32) == want.view(np.uint32)
        if name == "q2":
            same |= declared
        bad = np.nonzero(~same)[0]
        ulp = fp32_ulp_distance(got[bad], want[bad]) if bad.size else None
        result["fields"][name] = {
            "differing": int(bad.size),
            "max_ulp": int(np.max(ulp)) if bad.size else 0,
            "first": [{"psfc": float(cols["psfc"][i]), "got": float(got[i]),
                       "want": float(want[i])} for i in bad[:10]],
        }
        failing += int(bad.size)
    if declared.any():
        result["declared_q2_floor_port_is_qv1"] = bool(np.all(
            port["q2"][declared].view(np.uint32)
            == qv1[declared].view(np.uint32)))
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "fields"}))
    for name, r in result["fields"].items():
        print(f"  {name}: {r['differing']} differing, max {r['max_ulp']} ULP")
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
