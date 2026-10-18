#!/usr/bin/env python3
"""Wind-sector sweep: the one place gwd_opt = 1 lets a libm call choose.

    python sector_sweep.py BUILD_DIR [--per-boundary N] [--random-chunks K]
                           [--no-atanf] [--write-sample OUT.npz] [--json OUT]

bl_gwdo.F90 reads atan2 only to pick one of eight wind sectors
(bl_gwdo.F90:338-339): the sector selects which OA/OL statistic and which
grid length the drag uses.  A last-bit difference between this host's
atan2f and the kernel's ``td_atan2f`` changes an output only if it moves
``nint(fdir*(atan2 + pi))`` across a half-integer, so a column oracle can
only show such a flip did not happen on its columns.  This sweep holds the
function itself to the host's:

* ``atanf``: ``td_atanf`` against the host's ``atanf`` on EVERY float32 bit
  pattern (2**32; NaN inputs agree when both results are NaN, since the GPU
  returns its canonical NaN where x86 keeps the payload);
* ``boundary``: for each of the eight sector boundaries, N radii with
  (ubar, vbar) on the boundary ray, each word moved by up to 3 and 1 float32
  ulps;
* ``special``: every pair of signed zeros, subnormals, the smallest normal,
  1, powers of two around the 2**25 / 2**26 / 2**60 thresholds, infinities
  and NaN, plus x = 1 exactly with N random y;
* ``random``: K chunks of 2**24 uniformly random wind pairs (|wind| from
  1e-3 to 150 m/s, every direction).

The CPU side is ``libsector_sweep.so`` (built by build.sh from
sector_sweep.c): this host's ``atanf``/``atan2f`` and the sector exactly as
gfortran evaluates bl_gwdo.F90:338-339.  The GPU side is
``gpuwm.verify.gwdo_kim_oracle.sector_probe``/``atanf_probe``: the
production ``gpuwm.core.terrain_drag.module_source()`` compiled with the
production options (``MODULE_OPTIONS``) through the production NVRTC route,
with probe kernels appended that spell kernels/terrain_drag.cu's own sector
statements (checked to still be in the source).

``--write-sample`` stores the boundary pairs and every pair on which the
host's atan2f is not the correctly rounded angle (up to 65536), with the
host's words, for the GPU test that runs anywhere.  Run under the GPU mutex.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class Cpu:
    def __init__(self, build_dir: Path):
        lib = ctypes.CDLL(str(build_dir / "libsector_sweep.so"))
        f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        lib.cpu_atan2f_sector.argtypes = [f32p, f32p, f32p, i32p,
                                          ctypes.c_int64]
        lib.cpu_atanf_range.argtypes = [ctypes.c_uint32, ctypes.c_int64, f32p]
        self.lib = lib

    def atan2(self, pairs):
        n = pairs.shape[0]
        u = np.ascontiguousarray(pairs[:, 0])
        v = np.ascontiguousarray(pairs[:, 1])
        a = np.empty(n, np.float32)
        s = np.empty(n, np.int32)
        self.lib.cpu_atan2f_sector(u, v, a, s, n)
        return a, s

    def atanf(self, start, n):
        out = np.empty(n, np.float32)
        self.lib.cpu_atanf_range(start, n, out)
        return out


class Gpu:
    """The kernel's own td_atan2f / td_atanf, via gpuwm.verify.gwdo_kim_oracle."""

    def atan2(self, pairs):
        from gpuwm.verify.gwdo_kim_oracle import sector_probe
        return sector_probe(pairs)

    def atanf(self, start, n):
        from gpuwm.verify.gwdo_kim_oracle import atanf_probe
        return atanf_probe(start, n)


def _step(a: np.ndarray, k: int) -> np.ndarray:
    a = a.copy()
    direction = np.float32(np.inf) if k > 0 else np.float32(-np.inf)
    for _ in range(abs(k)):
        a = np.nextafter(a, direction)
    return a


def boundary_pairs(per_boundary: int, rng) -> np.ndarray:
    # atan2(ubar, vbar) + pi = (n + 1/2) * 2 pi / 8
    angles = np.array([(n + 0.5) * np.pi / 4.0 - np.pi for n in range(8)])
    radii = np.exp(rng.uniform(np.log(0.05), np.log(120.0), per_boundary))
    out = []
    for ang in angles:
        u = (radii * np.sin(ang)).astype(np.float32)
        v = (radii * np.cos(ang)).astype(np.float32)
        for du in range(-3, 4):
            for dv in (-1, 0, 1):
                out.append(np.stack([_step(u, du), _step(v, dv)], axis=1))
    return np.concatenate(out)


def special_pairs(n_unit: int, rng) -> np.ndarray:
    f = np.float32
    mags = [0.0, 1.4e-45, 1.0e-40, 1.17549435e-38, 2.0 ** -30, 1e-3, 0.5,
            1.0, 2.0, 80.0, 2.0 ** 25, 2.0 ** 26, 2.0 ** 27, 2.0 ** 60,
            2.0 ** 61, 3.0e38, np.inf]
    vals = [f(s * m) for m in mags for s in (1.0, -1.0)] + [f(np.nan)]
    vals = np.asarray(vals, np.float32)
    grid = np.stack(np.meshgrid(vals, vals, indexing="ij"), -1).reshape(-1, 2)
    unit = np.stack([rng.uniform(-150, 150, n_unit).astype(np.float32),
                     np.ones(n_unit, np.float32)], axis=1)
    return np.concatenate([grid, unit])


def random_pairs(n: int, rng) -> np.ndarray:
    r = np.exp(rng.uniform(np.log(1e-3), np.log(150.0), n))
    th = rng.uniform(-np.pi, np.pi, n)
    return np.stack([(r * np.sin(th)).astype(np.float32),
                     (r * np.cos(th)).astype(np.float32)], axis=1)


def grade_pairs(pairs, cpu, gpu):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    a_cpu, s_cpu = cpu.atan2(pairs)
    a_gpu, s_gpu = gpu.atan2(pairs)
    both_nan = np.isnan(a_cpu) & np.isnan(a_gpu)
    vmiss = (a_cpu.view(np.uint32) != a_gpu.view(np.uint32)) & ~both_nan
    finite = np.isfinite(a_cpu) & np.isfinite(a_gpu)
    ulp = np.asarray(fp32_ulp_distance(a_cpu[finite], a_gpu[finite]),
                     np.int64)
    smiss = s_cpu != s_gpu
    ref = np.arctan2(pairs[:, 0].astype(np.float64),
                     pairs[:, 1].astype(np.float64)).astype(np.float32)
    not_cr = finite & (a_cpu.view(np.uint32) != ref.view(np.uint32))
    row = {
        "pairs": int(pairs.shape[0]),
        "atan2_words_differ": int(vmiss.sum()),
        "atan2_max_ulp": int(ulp.max()) if ulp.size else 0,
        "sector_differs": int(smiss.sum()),
        "sector_differs_examples": [
            {"ubar": float(pairs[i, 0]), "vbar": float(pairs[i, 1]),
             "cpu_atan2": float(a_cpu[i]), "gpu_atan2": float(a_gpu[i]),
             "cpu_sector": int(s_cpu[i]), "gpu_sector": int(s_gpu[i])}
            for i in np.flatnonzero(smiss)[:10]],
        "host_atan2f_not_correctly_rounded": int(not_cr.sum()),
    }
    return row, a_cpu, s_cpu, not_cr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--per-boundary", type=int, default=1 << 18)
    ap.add_argument("--random-chunks", type=int, default=16)
    ap.add_argument("--no-atanf", action="store_true")
    ap.add_argument("--write-sample", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    cpu, gpu = Cpu(args.build_dir), Gpu()
    rng = np.random.default_rng(20261007)
    report = {}
    sample = []          # (pairs, host atan2f words, host sectors)
    pairs = boundary_pairs(args.per_boundary, rng)
    report["boundary"], a, s, _ = grade_pairs(pairs, cpu, gpu)
    sel = np.sort(rng.choice(pairs.shape[0], size=min(65536, pairs.shape[0]),
                             replace=False))
    sample.append((pairs[sel], a[sel], s[sel]))
    report["special"] = grade_pairs(special_pairs(1 << 20, rng), cpu, gpu)[0]
    total = None
    budget = 65536
    for _ in range(args.random_chunks):
        pairs = random_pairs(1 << 24, rng)
        row, a, s, not_cr = grade_pairs(pairs, cpu, gpu)
        if budget > 0:
            idx = np.flatnonzero(not_cr)[:budget]
            sample.append((pairs[idx], a[idx], s[idx]))
            budget -= idx.size
        if total is None:
            total = row
            continue
        for k in ("pairs", "atan2_words_differ", "sector_differs",
                  "host_atan2f_not_correctly_rounded"):
            total[k] += row[k]
        total["atan2_max_ulp"] = max(total["atan2_max_ulp"],
                                     row["atan2_max_ulp"])
        total["sector_differs_examples"] += row["sector_differs_examples"]
    report["random"] = total

    if not args.no_atanf:
        chunk = 1 << 27
        differ = 0
        nan_inputs = 0
        examples = []
        for c in range((1 << 32) // chunk):
            start = c * chunk
            a = cpu.atanf(start, chunk)
            b = gpu.atanf(start, chunk)
            both_nan = np.isnan(a) & np.isnan(b)
            nan_inputs += int(both_nan.sum())
            miss = (a.view(np.uint32) != b.view(np.uint32)) & ~both_nan
            differ += int(miss.sum())
            for i in np.flatnonzero(miss)[:4]:
                if len(examples) < 16:
                    examples.append({"x_bits": hex(start + int(i)),
                                     "cpu": float(a[i]), "gpu": float(b[i])})
        report["atanf_every_float32"] = {
            "inputs": 1 << 32, "words_differ": differ,
            "nan_inputs_both_nan": nan_inputs, "examples": examples}

    if args.write_sample:
        np.savez_compressed(
            args.write_sample,
            pairs=np.concatenate([p for p, _, _ in sample]),
            atan2f=np.concatenate([a for _, a, _ in sample]),
            sector=np.concatenate([s for _, _, s in sample]))

    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    ok = all(r["sector_differs"] == 0 and r["atan2_words_differ"] == 0
             for k, r in report.items() if k != "atanf_every_float32")
    ok &= report.get("atanf_every_float32", {}).get("words_differ", 0) == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
