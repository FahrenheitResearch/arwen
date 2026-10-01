"""Prepare real/synthetic LW inputs or save an unchanged-engine oracle."""
import argparse
import inspect
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prepare", "oracle", "compare", "frames", "device", "cpu"])
ap.add_argument("source", type=Path)
ap.add_argument("output", type=Path)
ap.add_argument("--expected-import", required=True)
args = ap.parse_args()
import gpuwm
assert args.expected_import in gpuwm.__file__, gpuwm.__file__
print("gpuwm", gpuwm.__file__, flush=True)
from gpuwm.core import rrtmg_lw as lw
from gpuwm.core.rrtmg_legacy import _lw_coeffs
args.output.mkdir(parents=True, exist_ok=True)
if args.mode == "cpu":
    import pytest
    import runpy
    tests = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "test_rrtmg_lw_speed_identity.py"))
    for name in ("test_workspace_reuses_written_profiles", "test_address_twin_preserves_float_statements", "test_every_empty_slot_has_a_write_before_read_reason"):
        tests[name]()
    patch = pytest.MonkeyPatch()
    try:
        tests["test_constants_reuse_and_invalidate_by_object_and_device"](patch)
    finally:
        patch.undo()
    print("4 CPU gates PASS", flush=True)
elif args.mode == "frames":
    import runpy
    tests = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "test_rrtmg_lw_cuda.py"))
    tests["test_gpu_local_frames"]()
    print("existing frame audit PASS", flush=True)
    from cupy.cuda import driver
    def frame(name):
        return int(driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, lw._gpu_kernel(name).ptr))
    for name in ("rlw_cldprmc", "rlw_rtrn_prol", "rlw_rtrn_march", "rlw_rtrn_accum"):
        original = frame(name)
        changed = frame(name + "_coalesced")
        print("frame", name, original, changed, flush=True)
        assert changed <= original, (name, original, changed)
elif args.mode in ("compare", "device"):
    import runpy
    import os
    os.environ["LW_SPEED_INPUTS"] = str(args.source)
    os.environ["LW_SPEED_ORACLE"] = str(args.output)
    tests = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "test_rrtmg_lw_speed_identity.py"))
    if args.mode == "device":
        tests["test_device_inputs_and_workspace_pricing"]()
        tests["test_deferred_cloud_abort_text"]()
        print("cloud abort text PASS", flush=True)
        raise SystemExit(0)
    for chunk in (1, 256, 1536, 4096):
        tests["test_saved_engine_oracle_uint32_dual_run"](chunk)
        print("uint32 dual IDENTICAL chunk", chunk, flush=True)
elif args.mode == "prepare":
    import cupy as cp
    from gpuwm.core.rrtmg_legacy_prep import lwrad_prep_batch
    from gpuwm.core.rrtmg_mcica import gpu_generate_lw_subcolumns
    keys = set(inspect.signature(lw.gpu_rrtmg_lw_batched_device).parameters) - {"C"}
    first = None
    for path in sorted(args.source.glob("lw_*.npz")):
        with np.load(path) as deck:
            kw = {k: (v.item() if v.ndim == 0 else v) for k, v in deck.items()}
        result = lwrad_prep_batch(**kw, subcolumn_generator=gpu_generate_lw_subcolumns)
        inputs = {k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v
                  for k, v in result.items() if k in keys}
        np.savez(args.output / path.name, **inputs)
        if first is None:
            first = inputs
        print("prepared", path.name, flush=True)
    mc = {"cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl"}
    for width in (1, 257, 5000):
        for kind in ("clear", "overcast", "mixed"):
            ins = {}
            for k, v in first.items():
                if not isinstance(v, np.ndarray) or v.ndim == 0:
                    ins[k] = v
                elif k in mc:
                    ins[k] = np.repeat(v[:, :1, :], width, axis=1)
                else:
                    ins[k] = np.repeat(v[:1], width, axis=0)
            ins["ncol"] = width
            ins["inflglw"] = 0
            ins["cldfmcl"].fill(0 if kind == "clear" else 1)
            ins["taucmcl"].fill(np.float32(1e-8 if kind == "clear" else 100))
            ins["tauaer"].fill(np.float32(1e-8))
            if kind == "mixed":
                ins["cldfmcl"][:, :, ::2] = 0
                ins["taucmcl"][:, :, ::3] = np.float32(1e-8)
                ins["tauaer"][:, ::3, :] = np.float32(100)
            np.savez(args.output / f"synthetic_{kind}_{width}.npz", **ins)
else:
    C = _lw_coeffs()
    for path in sorted(args.source.glob("*.npz")):
        with np.load(path) as deck:
            ins = {k: (v.item() if v.ndim == 0 else v) for k, v in deck.items()}
        out = lw.gpu_rrtmg_lw_batched(**ins, C=C, column_chunk=1536)
        again = lw.gpu_rrtmg_lw_batched(**ins, C=C, column_chunk=1536)
        for k in out:
            np.testing.assert_array_equal(out[k].view(np.uint32), again[k].view(np.uint32))
        np.savez(args.output / path.name, **out)
        print("oracle dual IDENTICAL", path.name, flush=True)
