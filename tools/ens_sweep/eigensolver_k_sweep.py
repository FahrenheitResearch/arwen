"""Where exactly does the sm_120 batched eigensolver fall over?

The cycled sweep measured the LETKF solve at four ensemble sizes and the
shape is not a power law:

    k    solve      ratio to previous     O(k^3) would give
    10    6.3 s        --                    --
    20   10.1 s       1.60x                 8.0x
    36   71.9 s       7.12x                 5.8x
    64  109   s       1.52x                 5.6x

Mild on both sides, one step in the middle.  That is a discontinuity
between k=20 and k=36, not arithmetic, and it invalidates the O(k^3)
explanation this lane first reached for.

Another lane proposed a cuSOLVER ``syevjBatched`` cliff at k=32, then
RETRACTED it after sweeping k=16..40 on sm_89 and finding nothing.  Both
observations can be true at once if the cliff is sm_120-specific, and
nobody has swept k finely on a QUIET sm_120 -- which is what this does.

Deliberately cheap: G is small enough that even the slow side of the
cliff costs seconds, because the question is WHERE the step is, not how
expensive it gets at production point counts.  The cycled sweep already
priced that.
"""
from __future__ import annotations

import json
import sys
import time

import cupy as cp
import numpy as np

#: Fine around the suspected step, coarse away from it.
K_VALUES = (10, 16, 20, 24, 28, 30, 31, 32, 33, 34, 36, 40, 48, 64)

#: Patches per timing.  Small on purpose; see the module docstring.
PATCHES = 20_000


def letkf_like(g: int, k: int, seed: int) -> np.ndarray:
    """(R-1)I + C Yb -- the shape the LETKF actually factors."""
    rs = np.random.RandomState(seed)
    out = np.empty((g, k, k))
    for lo in range(0, g, 10_000):
        hi = min(lo + 10_000, g)
        y = rs.standard_normal((hi - lo, k, max(1, k // 2)))
        a = (k - 1) * np.eye(k) + y @ np.swapaxes(y, 1, 2)
        out[lo:hi] = 0.5 * (a + np.swapaxes(a, 1, 2))
    return np.ascontiguousarray(out)


def timed(fn, budget_s: float = 12.0) -> float:
    """Best of a few launches, never spending more than budget_s of card."""
    start = time.perf_counter()
    fn()
    cp.cuda.Stream.null.synchronize()
    warm = time.perf_counter() - start
    reps = 3 if warm < 1.0 else 1
    reps = max(1, min(reps, int(budget_s / max(warm, 1e-6))))
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def compute_apps() -> int:
    """How many processes hold this GPU -- 1 is us and nobody else."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return -1
    return len([line for line in out.splitlines() if line.strip()])


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "eigensolver-k-sweep.json"
    try:
        from gpuwm.core.jacobi_eigh import batched_eigh, supported
    except ImportError as error:
        print(f"project kernel unavailable: {error}")
        batched_eigh = supported = None

    device = cp.cuda.Device(0)
    rows = []
    apps_seen = [compute_apps()]
    for k in K_VALUES:
        host = letkf_like(PATCHES, k, seed=1000 + k)
        a = cp.asarray(host)
        row: dict = {"k": k, "patches": PATCHES}

        # cuSOLVER: cupy's batched eigh routes to syevjBatched.
        try:
            row["cusolver_s"] = round(timed(lambda: cp.linalg.eigh(a)), 4)
        except Exception as error:                    # noqa: BLE001
            row["cusolver_error"] = f"{type(error).__name__}: {error}"

        if batched_eigh is not None and supported(k, a.dtype):
            sym = 0.5 * (a + cp.swapaxes(a, 1, 2))
            try:
                row["jacobi_s"] = round(timed(lambda: batched_eigh(sym)), 4)
            except Exception as error:                # noqa: BLE001
                row["jacobi_error"] = f"{type(error).__name__}: {error}"
            del sym
        if "cusolver_s" in row and row.get("jacobi_s"):
            row["ratio_cusolver_over_jacobi"] = round(
                row["cusolver_s"] / row["jacobi_s"], 2)

        # Accuracy, once, at the step -- a fast wrong answer is not a result.
        if k == 36 and batched_eigh is not None:
            small = cp.asarray(host[:256])
            sym = 0.5 * (small + cp.swapaxes(small, 1, 2))
            w_j = cp.asnumpy(batched_eigh(sym)[0])
            w_c = np.linalg.eigvalsh(host[:256])
            row["max_eigenvalue_abs_err_vs_numpy"] = float(
                np.abs(np.sort(w_j, axis=1) - np.sort(w_c, axis=1)).max())
            del small, sym

        apps_seen.append(compute_apps())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del a, host
        cp.get_default_memory_pool().free_all_blocks()

    payload = {
        "schema": "gpuwm-da.eigensolver-k-sweep.v1",
        "device": {
            "name": device.attributes.get("Name", "unknown")
            if hasattr(device, "attributes") else "unknown",
            "compute_capability": "".join(
                str(v) for v in device.compute_capability),
        },
        "cupy_version": cp.__version__,
        "patches": PATCHES,
        "max_concurrent_gpu_apps": max(apps_seen),
        "card_exclusive": max(apps_seen) <= 1,
        "why": "the cycled sweep's solve times step between k=20 and k=36 "
               "and scale mildly on both sides, which is not O(k^3); this "
               "locates the step on a quiet sm_120",
        "rows": rows,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\n{'k':>4} {'cuSOLVER s':>11} {'jacobi s':>10} {'ratio':>8}")
    for row in rows:
        print(f"{row['k']:>4} {row.get('cusolver_s', float('nan')):>11.4f} "
              f"{row.get('jacobi_s', float('nan')):>10.4f} "
              f"{row.get('ratio_cusolver_over_jacobi', float('nan')):>8.2f}")
    print(f"\ncard_exclusive={payload['card_exclusive']} "
          f"(max concurrent GPU apps {payload['max_concurrent_gpu_apps']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
