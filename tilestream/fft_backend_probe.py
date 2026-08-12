"""Do cuFFT and pocketfft agree bit for bit?  Asked, not assumed.

``gpuwm.da.perturb``'s docstring says "the filtered field itself is not
bit-identical across backends -- cuFFT and pocketfft round differently -- and
this module does not claim it is".  That is a disclaimer, not a measurement,
and the streamed execution mode turns it into a live question: a streamed
domain lives in pinned host RAM and therefore filters on the host, so if the
two backends really do differ then a resident member and a streamed member of
the same seed are DIFFERENT MEMBERS and the execution mode changed the
science.

This probe answers it by running the module's own filter both ways on the same
host white noise and counting the ULP gap.  It sweeps shape (the FFT's radix
decomposition depends on the factorisation of every axis) and dtype (float32
has 24 bits of mantissa to hide a difference in, float64 has 53).

Read the result as a measurement of THIS pairing on THIS box -- cupy version,
CUDA version, host BLAS -- and not as a property of FFTs.  That is exactly why
``PerturbationConfig.fft_host`` exists: it converts whatever this probe happens
to find into a guarantee that does not depend on the finding.
"""

from __future__ import annotations

import sys

import numpy as np

from gpuwm.da import perturb as P


def one(shape, dtype: str, *, seed: int = 4242, name: str = "theta",
        dx_km: float = 0.5, length_scale_km: float = 6.0,
        vertical_scale_levels: float = 6.0) -> dict:
    import cupy as cp

    host, _key, digest = P._white_noise(shape, seed=seed, name=name,
                                        dtype=np.float32
                                        if dtype == "float32"
                                        else np.float64)
    common = dict(seed=seed, name=name, dx_km=dx_km, dy_km=dx_km,
                  length_scale_km=length_scale_km,
                  vertical_scale_levels=vertical_scale_levels, dtype=dtype)
    a, ia = P.gaussian_random_field(shape, xp=np, **common)
    b, ib = P.gaussian_random_field(shape, xp=cp, **common)
    cp.cuda.runtime.deviceSynchronize()
    bb = np.asarray(b.get())
    assert ia["noise_sha256"] == ib["noise_sha256"] == digest
    int_kind = np.int32 if dtype == "float32" else np.int64
    scale = float(np.abs(a).max())
    gap = np.abs(a.view(int_kind).astype(np.int64)
                 - bb.view(int_kind).astype(np.int64))
    del b
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "shape": tuple(shape),
        "dtype": dtype,
        "fft_backend_host": ia["fft_backend"],
        "fft_backend_device": ib["fft_backend"],
        "identical": bool(np.array_equal(a, bb)),
        "max_abs": float(np.abs(a - bb).max()),
        "max_rel": float(np.abs(a - bb).max() / scale) if scale else 0.0,
        "max_ulps": int(gap.max()),
        "differing_cells": int(np.count_nonzero(a != bb)),
        "cells": int(a.size),
    }


SHAPES = (
    (49, 160, 192),      # the gate's own domain: 7^2, 2^5*5, 2^6*3
    (49, 161, 193),      # 7*23 and a PRIME -- Bluestein/Rader territory
    (50, 250, 200),      # the real74 ingest domain
    (49, 192, 192),
    (37, 121, 201),      # the ERA5 source grid: 37, 11^2, 3*67
    (64, 256, 256),      # all powers of two, the friendliest case there is
    (49, 500, 500),
)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    dtypes = argv or ["float64", "float32"]
    print(f"{'shape':>18} {'dtype':>9} {'identical':>10} {'max_ulps':>9} "
          f"{'max_rel':>11} {'differing':>12}")
    worst = 0
    for shape in SHAPES:
        for dtype in dtypes:
            try:
                r = one(shape, dtype)
            except Exception as exc:                    # pragma: no cover
                print(f"{str(shape):>18} {dtype:>9}  ERROR {type(exc).__name__}"
                      f": {exc}")
                continue
            worst = max(worst, r["max_ulps"])
            frac = 100.0 * r["differing_cells"] / r["cells"]
            print(f"{str(r['shape']):>18} {dtype:>9} "
                  f"{str(r['identical']):>10} {r['max_ulps']:>9} "
                  f"{r['max_rel']:>11.3e} "
                  f"{r['differing_cells']:>9} ({frac:4.1f}%)")
    print(f"\nworst ULP gap over the sweep: {worst}")
    print("0 everywhere means the two libraries happened to agree HERE; it is "
          "not a promise, which is what fft_host is for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
