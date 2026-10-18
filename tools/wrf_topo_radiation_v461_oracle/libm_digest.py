"""Whole-domain check of WOOF's topo-radiation libm words (device side).

    python libm_digest.py --exe BUILD/libm_digest \
        --libm NAME DIGESTS.txt [ROOT] [--libm ...] [--json OUT]   (GPU)

Compiles WOOF's own ``topo_radiation`` unit exactly as the loader composes
it (``gpuwm.core.kernels.module_source`` / ``module_options``; under
GPUWM_WRF_EXACT=1 the strict compiler applies), appends one probe kernel
that calls the unit's own ``tr_*`` words, and digests every float32 input
of each word's domain per input binade, as libm_digest.c digests a C
library's float functions (``DIGESTS.txt`` holds its output lines for one
libm; ``ROOT`` is where that libm is unpacked, empty for the box's own).

A bucket whose digests agree is bit-identical on every input in it.  For a
bucket that does not, both sides dump the bucket's 2^23 outputs and the
script states the exact differing inputs: how many, the input range, and
the largest ULP distance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WORDS = ("sinf", "cosf", "tanf", "asinf", "acosf", "atanf", "atan2f_x1000",
         "atan2f_1000_y", "expf", "powf_m1.6364", "powf_0.5",
         "atan2f_resolve_x1000", "atan2f_pairs", "atan2f_grad")

PROBE = r"""
__device__ __forceinline__ unsigned int topo_h32(unsigned int v)
{
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

__device__ __forceinline__ int topo_pair(unsigned int u, int kind, float* y,
                                         float* x)
{
    unsigned int a = topo_h32(u), b = topo_h32(u ^ 0xA5A5A5A5u);
    if (kind == 1) {
        const unsigned int ea = 107u + topo_h32(u ^ 0x1234567u) % 31u;
        const unsigned int eb = 107u + topo_h32(u ^ 0x7654321u) % 31u;
        a = (a & 0x807FFFFFu) | (ea << 23);
        b = (b & 0x807FFFFFu) | (eb << 23);
    }
    *y = __uint_as_float(a);
    *x = __uint_as_float(b);
    if (isinf(*y) || isnan(*y) || isinf(*x) || isnan(*x)) return 0;
    if (fabs((double)*y) < __dmul_rn(fabs((double)*x), 0x1p-100)) return 0;
    return 1;
}

// tr_atan2 with the exact midpoint decision taken on every input.
__device__ __forceinline__ float topo_atan2_resolved(float y, float x)
{
    const double r = atan2((double)y, (double)x);
    const float f = __double2float_rn(r);
    if (!(fabs(r) >= 0x1p-122) || isinf(y) || isinf(x) || (double)f == r)
        return f;
    return tr_atan2_resolve(y, x, r, f);
}

__device__ __forceinline__ int topo_probe_word(int word, float x, float* a)
{
    if (word == 12 || word == 13) {
        float yy, xx;
        if (!topo_pair(__float_as_uint(x), word - 12, &yy, &xx)) return 0;
        *a = tr_atan2(yy, xx);
        return 1;
    }
    if (x != x) return 0;
    const float ax = fabsf(x);
    switch (word) {
    case 0: if (ax > 64.0f) return 0; *a = tr_sin(x); return 1;
    case 1: if (ax > 64.0f) return 0; *a = tr_cos(x); return 1;
    case 2: if (ax > 64.0f) return 0; *a = tr_tan(x); return 1;
    case 3: if (ax > 1.0f) return 0; *a = tr_asin(x); return 1;
    case 4: if (ax > 1.0f) return 0; *a = tr_acos(x); return 1;
    case 5: *a = tr_atan(x); return 1;
    case 6: *a = tr_atan2(x, 1000.0f); return 1;
    case 7: *a = tr_atan2(1000.0f, x); return 1;
    case 8: if (ax > 104.0f) return 0; *a = tr_exp(x); return 1;
    case 9: if (!(x > 0.0f)) return 0; *a = tr_pow(x, -1.6364f); return 1;
    case 11: *a = topo_atan2_resolved(x, 1000.0f); return 1;
    default: if (!(x >= 0.0f)) return 0; *a = tr_pow(x, 0.5f); return 1;
    }
}

// grid (512, 64): block (b, s) digests slice s of input bucket b.
extern "C" __global__ void topo_libm_digest(int word,
                                            unsigned long long* bucket,
                                            unsigned long long* count)
{
    const unsigned long long b = blockIdx.x;
    const unsigned long long per = (1ULL << 23) / gridDim.y;
    const unsigned long long lo = (b << 23) + blockIdx.y * per;
    unsigned long long dig = 0, tot = 0;
    for (unsigned long long u = lo + threadIdx.x; u < lo + per;
         u += blockDim.x) {
        float a;
        if (!topo_probe_word(word, __uint_as_float((unsigned int)u), &a))
            continue;
        unsigned long long h = (u << 32) | __float_as_uint(a);
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        dig += h;
        tot += 1;
    }
    atomicAdd(&bucket[b], dig);
    atomicAdd(&count[b], tot);
}

extern "C" __global__ void topo_libm_dump(int word, unsigned int b,
                                          unsigned int* out)
{
    const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (1u << 23)) return;
    float a;
    out[k] = topo_probe_word(word, __uint_as_float((b << 23) + k), &a)
             ? __float_as_uint(a) : 0xFFFFFFFFu;
}
"""


def _module():
    import cupy as cp

    import gpuwm  # noqa: F401  (installs the strict compiler when asked)
    from gpuwm.core.kernels import module_options, module_source
    return cp.RawModule(code=module_source("topo_radiation") + PROBE,
                        options=module_options("topo_radiation"))


def device_buckets(mod, word_id):
    import cupy as cp
    bucket = cp.zeros(512, dtype=cp.uint64)
    count = cp.zeros(512, dtype=cp.uint64)
    mod.get_function("topo_libm_digest")((512, 64), (256,),
                                         (np.int32(word_id), bucket, count))
    b = [int(v) for v in cp.asnumpy(bucket)]
    total = sum(b) % (1 << 64)
    return int(cp.asnumpy(count).sum()), total, b


def device_dump(mod, word_id, b):
    import cupy as cp
    out = cp.empty(1 << 23, dtype=cp.uint32)
    mod.get_function("topo_libm_dump")(((1 << 23) // 256,), (256,),
                                       (np.int32(word_id), np.uint32(b), out))
    return cp.asnumpy(out)


def host_dump(exe, root, word, b):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "b.bin"
        argv = [str(exe), word, str(b), str(out)]
        if root:
            lib = Path(root) / "usr" / "lib" / "x86_64-linux-gnu"
            argv = [str(lib / "ld-linux-x86-64.so.2"), "--library-path",
                    f"{lib}:/usr/lib/x86_64-linux-gnu"] + argv
        subprocess.run(argv, check=True)
        return np.fromfile(out, dtype=np.uint32)


def _key(bits):
    i = bits.astype(np.int64)
    return np.where(i & 0x80000000, -(i & 0x7FFFFFFF), i)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--libm", nargs="+", action="append", default=[],
                    metavar="NAME DIGESTS [ROOT]")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    libms = {}
    for spec in a.libm:
        name, path = spec[0], Path(spec[1])
        root = spec[2] if len(spec) > 2 else ""
        lines = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            buckets = [int(x, 16) for x in
                       parts[-1].split("=", 1)[1].split(",")]
            lines[parts[0]] = (int(parts[1]), int(parts[2], 16), buckets)
        libms[name] = (lines, root)
    from gpuwm import wrf_exact
    mod = _module()
    report = {"strict": bool(wrf_exact.ENABLED), "words": {}}
    print(f"strict={wrf_exact.ENABLED}")
    for word_id, word in enumerate(WORDS):
        tot, dig, buckets = device_buckets(mod, word_id)
        rw = report["words"][word] = {"inputs": tot, "libms": {}}
        for name, (lines, root) in libms.items():
            htot, hdig, hb = lines[word]
            if (htot, hdig) == (tot, dig):
                rw["libms"][name] = {"identical": True, "differ": 0}
                print(f"{word:14s} {name}: identical on all {tot} inputs")
                continue
            bad = [b for b in range(512) if hb[b] != buckets[b]]
            differ, xs, worst = 0, [], 0
            dumps = {}
            for b in bad:
                dev = device_dump(mod, word_id, b)
                host = host_dump(a.exe, root, word, b)
                d = np.nonzero(dev != host)[0]
                if d.size == 0:
                    continue
                differ += int(d.size)
                inputs = ((np.uint32(b) << np.uint32(23))
                          + d.astype(np.uint32)).view(np.float32)
                xs += [float(inputs.min()), float(inputs.max())]
                both = (dev[d] != 0xFFFFFFFF) & (host[d] != 0xFFFFFFFF)
                if both.any():
                    worst = max(worst, int(np.abs(
                        _key(dev[d][both]) - _key(host[d][both])).max()))
                dumps[b] = [float(v) for v in inputs[:3]]
            rw["libms"][name] = {
                "identical": False, "differ": differ,
                "x_min": min(xs) if xs else None,
                "x_max": max(xs) if xs else None, "max_ulp": worst,
                "buckets": bad, "examples": dumps}
            print(f"{word:14s} {name}: {differ} of {tot} inputs differ, "
                  f"x in [{min(xs) if xs else 0:.6g}, "
                  f"{max(xs) if xs else 0:.6g}], max {worst} ulp")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
