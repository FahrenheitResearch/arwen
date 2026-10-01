"""Reproducible node-side prep verification and synchronized wall timing."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--case", help="one test function from the device prep suite")
    args = parser.parse_args()
    import gpuwm
    assert Path(gpuwm.__file__).resolve().is_relative_to(ROOT), gpuwm.__file__
    print("source:", gpuwm.__file__, flush=True)
    import numpy as np
    print("numpy:", np.__version__, flush=True)
    if not args.timing:
        import pytest
        cmd = ["tests/test_rrtmg_legacy_prep_device.py", "-x", "-q", "-s", "--noconftest"]
        if args.case:
            cmd = ["tests/test_rrtmg_legacy_prep_device.py::"+args.case,
                   "-x", "-q", "-s", "--noconftest"]
        elif args.smoke:
            cmd = ["tests/test_rrtmg_legacy_prep_device.py::test_preflight_and_numpy_minmax",
                   "tests/test_rrtmg_legacy_prep_device.py::test_all_radii_routes",
                   "-x", "-q", "-s", "--noconftest"]
        elif args.width:
            cmd = [f"tests/test_rrtmg_legacy_prep_device.py::test_synthetic_dual[variant{v}-radii{r}-{args.width}]"
                   for v in range(6) for r in range(3)] + ["-x", "-q", "-s", "--noconftest"]
        else:
            cmd += ["-k", "not synthetic"]
        return pytest.main(cmd)
    import cupy as cp
    from gpuwm.core import rrtmg_legacy_device as dev
    from gpuwm.core import rrtmg_legacy_prep as ref
    from test_rrtmg_legacy_prep_device import synthetic, side_kwargs, device_kwargs, assert_bits
    ref._PERFWAVE_DEVICE_XP = None
    base = synthetic(4096, 59)
    print("timing ncol=4096 nz=59 LW nlay=", ref.compute_lw_nlayers(60, 5000.0), flush=True)
    for sw in (False, True):
        kw = side_kwargs(base, sw)
        dkw = device_kwargs(kw)
        host = ref.swrad_prep_batch if sw else ref.lwrad_prep_batch
        device = dev.swrad_prep_batch_device if sw else dev.lwrad_prep_batch_device
        want = host(**kw)
        got = device(**dkw)
        assert_bits(got, want)
        del got, want
        samples = {}
        for name, function, inputs in (("host", host, kw), ("device", device, dkw)):
            ms = []
            for _ in range(5):
                cp.cuda.runtime.deviceSynchronize()
                start = time.perf_counter()
                result = function(**inputs)
                cp.cuda.runtime.deviceSynchronize()
                ms.append(1000*(time.perf_counter()-start))
                del result
            samples[name] = ms
        print("SW" if sw else "LW", samples,
              "median_ms", {k: float(np.median(v)) for k,v in samples.items()}, flush=True)
    print("local_frames", dev.gpu_local_frame_bytes(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
