"""Standalone ULP report for the UP_HELI_MAX transcription.

Prints the per-field, per-step ULP distance of the NumPy mirror (and the
CUDA kernel when a device is present) against the pinned WRF v4.6.1
cal_helicity oracle fixtures.  Asserts nothing -- the gate lives in
tests/test_uh_wrf461_parity.py.

usage: python tools/uh_wrf461_oracle/validate_uh_oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from gpuwm.core.fp32_ulp import fp32_ulp_distance  # noqa: E402
from gpuwm.core.uh_diag import mirror_up_heli_max_step_np  # noqa: E402


def _gate_module():
    # Reuse the gate module's loader without pytest plumbing (tests/ is not
    # an importable package).
    import importlib.util
    path = REPO / "tests" / "test_uh_wrf461_parity.py"
    spec = importlib.util.spec_from_file_location("uh_parity_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _gate_module()
NSTEPS, NX, NY = _MOD.NSTEPS, _MOD.NX, _MOD.NY


def _load():
    return _MOD.oracle.__wrapped__()


def main() -> int:
    static, steps = _load()
    up_heli_max = np.zeros((NY, NX), dtype=np.float32)
    print(f"fixture: {NX}x{NY} mass grid, {NSTEPS} steps")
    for s, step in enumerate(steps, start=1):
        uh, _use = mirror_up_heli_max_step_np(
            step["u"], step["v"], step["w"], step["ph"], static["phb"],
            static["msfu"], static["msfv"], static["ht"],
            static["dn"], static["dnw"], static["fnm"], static["fnp"],
            static["cf1"], static["cf2"], static["cf3"],
            static["rdx"], static["rdy"], up_heli_max)
        for name, got, want in (("uh", uh, step["uh"]),
                                ("up_heli_max", up_heli_max,
                                 step["up_heli_max"])):
            d = fp32_ulp_distance(got, want)
            print(f"  step {s} {name:12s} max_ulp={int(d.max())} "
                  f"mismatches={(d > 0).sum()}/{d.size}")
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        print("cuda: no device; mirror-only report")
        return 0
    from gpuwm.core.uh_diag import device_uh_step
    maxf = cp.zeros((NY, NX), dtype=cp.float32)
    uh_d = cp.zeros((NY, NX), dtype=cp.float32)
    use_d = cp.zeros((NY, NX), dtype=cp.float32)
    for s, step in enumerate(steps, start=1):
        device_uh_step(
            cp.asarray(step["u"]), cp.asarray(step["v"]),
            cp.asarray(step["w"]), cp.asarray(step["ph"]),
            cp.asarray(static["phb"]), cp.asarray(static["msfu"]),
            cp.asarray(static["msfv"]), cp.asarray(static["ht"]),
            cp.asarray(static["dn"]), cp.asarray(static["dnw"]),
            cp.asarray(static["fnm"]), cp.asarray(static["fnp"]),
            static["cf1"], static["cf2"], static["cf3"],
            static["rdx"], static["rdy"], uh_d, use_d, maxf)
        for name, got, want in (("uh", uh_d.get(), step["uh"]),
                                ("up_heli_max", maxf.get(),
                                 step["up_heli_max"])):
            d = fp32_ulp_distance(got, want)
            print(f"  cuda step {s} {name:12s} max_ulp={int(d.max())} "
                  f"mismatches={(d > 0).sum()}/{d.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
