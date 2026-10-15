"""Receipt for correct-mode findings WB4 and WB5: the measured numbers.

usage: python tools/correct_mode_water_budget_receipt.py OUT.json

Runs the same columns as tests/test_correct_mode_water_budget.py and writes
the budget residuals of both modes plus the exact-mode identity checks.
Needs g++ for the WSM6 CPU mirror and CuPy with a card for the GPU parts
(each part is recorded as skipped when its tool is absent).
"""
from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import test_correct_mode_water_budget as t  # noqa: E402


def wsm6_cpu(tmp: Path) -> dict:
    out = {}
    for opt in ("O0", "O2"):
        exact = t._mirror_run(tmp, t.WSM6_CU, correct=False, opt=opt)
        fixed = t._mirror_run(tmp, t.WSM6_CU, correct=True, opt=opt)
        sx = [x for x in exact if x.startswith("state,")]
        sf = [x for x in fixed if x.startswith("state,")]
        out[opt] = {
            "exact_boundary_residuals_kg_m2": t._residuals(exact, "boundary"),
            "correct_boundary_residuals_kg_m2": t._residuals(fixed, "boundary"),
            "exact_max_abs_column_residual_kg_m2": max(map(abs, t._residuals(exact, "column"))),
            "correct_max_abs_column_residual_kg_m2": max(map(abs, t._residuals(fixed, "column"))),
            "columns": len(t._residuals(exact, "column")),
            "atmosphere_and_radius_values_compared": len(sx) * 10,
            "atmosphere_and_radius_values_identical": sx == sf,
        }
    return out


def gpu() -> dict:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception as error:  # pragma: no cover - environment
        return {"skipped": f"no CUDA device: {error}"}
    _, r_exact = t._run_wsm6(cp, False)
    d_exact, _ = t._run_wsm6(cp, False)
    d_fixed, r_fixed = t._run_wsm6(cp, True)
    same = all(
        np.array_equal(cp.asnumpy(d_exact[n]).view(np.uint32), cp.asnumpy(d_fixed[n]).view(np.uint32))
        for n in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "effc", "effi", "effs"))
    rel_exact, _ = t._nt_budget(False)
    rel_fixed, _ = t._nt_budget(True)
    return {
        "device": name,
        "wsm6_columns": int(r_exact.size),
        "wsm6_exact_max_abs_residual_kg_m2": float(np.abs(r_exact).max()),
        "wsm6_correct_max_abs_residual_kg_m2": float(np.abs(r_fixed).max()),
        "wsm6_atmosphere_identical_between_modes": bool(same),
        "ntiedtke_raining_columns": int(rel_exact.size),
        "ntiedtke_exact_relative_residual": {
            "median_abs": float(np.median(np.abs(rel_exact))),
            "min": float(rel_exact.min()), "max": float(rel_exact.max())},
        "ntiedtke_correct_relative_residual": {
            "median_abs": float(np.median(np.abs(rel_fixed))),
            "min": float(rel_fixed.min()), "max": float(rel_fixed.max())},
    }


def main() -> int:
    out_path = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        receipt = {
            "schema": "gpuwm-correct-mode-water-budget-receipt/v1",
            "host": platform.node(),
            "wsm6_published_287_sha256": t.WSM6_287_SHA256,
            "wsm6_cpu_mirror": wsm6_cpu(Path(tmp)),
            "gpu": gpu(),
        }
    out_path.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
