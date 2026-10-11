"""Timing of one TEMPO call on a domain-sized field (replicated test column).

usage: python tempo_runtime_timing.py tempo_blob.bin tempo_ref_columns.bin NY NX [f32|f64]
"""
import sys
import time

import cupy as cp
import numpy as np

from gpuwm.core.tempo_runtime import STATE_FIELDS, TempoConfig, TempoRuntime


def main():
    blob, ref, ny, nx = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    dtype = np.float32 if (len(sys.argv) > 5 and sys.argv[5] == "f32") else np.float64
    raw = np.fromfile(ref, dtype="<f8")
    nz = int(raw[:5].view("<i8")[1])
    # scenario 2 initial state (the second scenario record)
    step = 17 * nz + 7 + 4 * nz
    n1 = int(raw[:5].view("<i8")[2])
    i2 = 5 + 17 * nz + step * n1
    init = raw[i2 + 5:i2 + 5 + 17 * nz].reshape(17, nz)
    rt = TempoRuntime(blob, nz_max=64, max_work_bytes=8 << 30)
    f = {n: cp.ascontiguousarray(cp.broadcast_to(cp.asarray(init[j], dtype=dtype)[:, None, None],
                                                 (nz, ny, nx))) for j, n in enumerate(STATE_FIELDS)}
    rt.step(f, 20.0, TempoConfig())
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = rt.step(f, 20.0, TempoConfig())
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
    ncol = ny * nx
    print(f"TEMPO_TIMING nz={nz} ny={ny} nx={nx} cols={ncol} dtype={np.dtype(dtype).name} "
          f"batch={rt.batch} s_per_call={np.median(times):.3f} us_per_col={1e6*np.median(times)/ncol:.3f} "
          f"max_refl={float(out['column']['refl10cm'].max()):.2f}")


if __name__ == "__main__":
    main()
