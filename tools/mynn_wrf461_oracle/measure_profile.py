"""Record existing MYNN oracle residues while preserving every assertion.

Run from this checkout's root on a CPU host. The wrapper
returns the original ULP function's result unchanged. It records the arrays
the ordinary assertions compare and their existing per-column limits.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path


class Measurement:
    def __init__(self, out):
        self.out = out
        self.records = []
        self.modules = {}
        self.current = None

    def pytest_runtest_setup(self, item):
        module = item.module
        self.current = item.nodeid
        name = Path(module.__file__).name
        if name not in ("test_mynn_surface.py", "test_mynn_surface_coarse.py", "test_mynn_surface_water.py"):
            return
        if name in self.modules:
            return
        original = module.fp32_ulp_distance
        tables = {key: hashlib.sha256(repr(value).encode()).hexdigest()
                  for key, value in vars(module).items() if key.endswith("_ULP") and isinstance(value, dict)}
        self.modules[name] = (module, original, tables)

        def measured(actual, expected):
            residue = original(actual, expected)
            frame = inspect.currentframe().f_back
            values = frame.f_locals
            output = values.get("name")
            budget = None
            if frame.f_code.co_name == "_assert_ulp":
                budget = values["budget"]
            elif frame.f_code.co_name == "test_cpu_reference_matches_the_water_oracle":
                budget = module._budget(module.WATER_ULP, values["key"], output, len(values["rows"]))
            elif frame.f_code.co_name == "test_the_new_branches_leave_znt_and_br_bitwise_on_every_column":
                budget = 0
            if budget is not None:
                import numpy as np
                allowed = np.broadcast_to(np.asarray(budget, dtype=np.int64), residue.shape)
                self.records.append({"test": self.current, "output": output,
                                     "residue": residue.tolist(), "budget": allowed.tolist(),
                                     "over_budget": int(np.count_nonzero(residue > allowed))})
            return residue

        module.fp32_ulp_distance = measured

    def pytest_sessionfinish(self, session, exitstatus):
        import numpy as np
        fingerprints = {}
        unchanged = {}
        for name, (module, original, before) in self.modules.items():
            if hasattr(module, "_fp32_fingerprint"):
                fingerprints = module._fp32_fingerprint()
            after = {key: hashlib.sha256(repr(getattr(module, key)).encode()).hexdigest() for key in before}
            unchanged[name] = {"before": before, "after": after, "equal": before == after}
            module.fp32_ulp_distance = original
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        outcomes = ({name: len(terminal.stats.get(name, [])) for name in ("passed", "failed", "skipped", "deselected")}
                    if terminal is not None else {})
        # The C library and NumPy's AVX-512 dispatch pick the float32 arctan,
        # so a receipt names both: glibc 2.39 and 2.43 differ, and AVX-512
        # dispatch differs from either.
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as features
        except ImportError:
            features = {}
        result = {"schema": "arwen.mynn-retained-budget-measurement.v1", "process_exit": int(exitstatus),
                  "python": platform.python_version(), "numpy": np.__version__, "platform": sys.platform,
                  "libc": " ".join(platform.libc_ver()).strip() or None,
                  "avx512_dispatch": bool(features.get("X86_V4") or features.get("AVX512_SKX")),
                  "test_outcomes": outcomes,
                  "fingerprint": fingerprints, "comparisons": len(self.records),
                  "over_budget_total": sum(row["over_budget"] for row in self.records),
                  "budgets_unchanged": unchanged, "records": self.records}
        self.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        parser.error("run the measurement from this tool's repository root")
    if not (root / "gpuwm/core/mynn_surface.py").is_file():
        parser.error("the checkout has no MYNN reference implementation")
    os.environ["GPUWM_NO_LOCAL_GPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    import pytest
    return pytest.main(["-q", "-p", "no:cacheprovider", "-n", "0", "-m", "not gpu and not network",
                        "tests/test_mynn_surface.py", "tests/test_mynn_surface_coarse.py",
                        "tests/test_mynn_surface_water.py"], plugins=[Measurement(args.out)])


if __name__ == "__main__":
    raise SystemExit(main())
