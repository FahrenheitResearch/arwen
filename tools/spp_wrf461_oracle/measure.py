"""Measure per-element WRF SPP residuals; this is not an acceptance gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "tests")]
    import test_spp_consumers as reference
    for attr, filename in [("TURBULENCE_ORACLE", "turbulence-spp.csv"),
                           ("CONDENSATION_ORACLE", "condensation-spp.csv"),
                           ("DMP_MF_ORACLE", "dmp_mf-spp.csv"),
                           ("DRIVER_ORACLE", "driver-spp.csv")]:
        setattr(reference.pbl, attr, args.oracle / filename)
    metadata = {"python": platform.python_version(), "numpy": np.__version__,
                "backend": args.backend, "libc": platform.libc_ver()}
    if args.backend == "gpu":
        import cupy as cp
        from gpuwm.core import mynn_pbl_gpu
        from gpuwm.core.mynn_sfclay import mynn_surface_layer
        backend = "gpu", mynn_pbl_gpu, cp.asnumpy, mynn_surface_layer, cp.asarray
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        metadata.update(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                        device=device["name"].decode())
    else:
        from gpuwm.core import mynn_pbl
        backend = "cpu", mynn_pbl, np.asarray, reference.surface.mynn_surface_layer_default, np.asarray
    distances, summary = {}, {}
    def capture(label, actual, expected):
        summary[label] = {}
        for name, value in expected.items():
            distance = reference.fp32_ulp_distance(actual[name], value)
            if not np.isfinite(actual[name]).all():
                raise ValueError(f"nonfinite output {label}/{name}")
            distances[label + "/" + name] = distance.astype(np.uint32)
            summary[label][name] = int(distance.max(initial=0))
    reference._gate = capture
    for test in (reference.test_mynn_turbulence_spp_native,
                 reference.test_mynn_condensation_spp_native,
                 reference.test_mynn_mass_flux_spp_native,
                 reference.test_mynn_surface_spp_native):
        test(args.oracle, backend)
    for step in (1, 2):
        reference.test_mynn_driver_spp_native(args.oracle, backend, step)
    np.savez_compressed(args.output, **distances)
    args.output.with_suffix(".json").write_text(json.dumps(
        {"metadata": metadata, "maximum_ulp": summary,
         "compared_elements": sum(value.size for value in distances.values()),
         "nonzero_ulp_elements": sum(int(np.count_nonzero(value)) for value in distances.values())},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
