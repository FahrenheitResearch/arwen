"""Engine compile path check for the TEMPO port.

WOOF compiles its kernels through CuPy/NVRTC, not nvcc.  This builds the
same headers through cupy.RawModule, replays the reference scenarios on
NREP identical columns and checks the result is bitwise identical to the
nvcc harness output (out_gpu.bin) and across replicas.

usage: python tempo_nvrtc_check.py <kernel_dir> tempo_blob.bin tempo_ref_columns.bin out_gpu.bin
"""
import re
import sys

import cupy as cp
import numpy as np

NFLD = 17


def blob_layout(header_text):
    """(name, count) in TEMPO_FIELDS order, parsed from tempo_common.cuh."""
    body = header_text.split("#define TEMPO_FIELDS(TX)", 1)[1].split("struct TempoBlob", 1)[0]
    out = []
    for m in re.finditer(r"TX\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)\)", body.replace(" ", "")):
        n = 1
        for d in m.groups()[1:]:
            n *= int(d)
        out.append((m.group(1), n))
    return out


def load_blob(path, layout):
    offs, total = {}, 0
    for name, n in layout:
        offs[name] = (total, n)
        total += n
    blob = np.zeros(total, dtype=np.float64)
    seen = set()
    with open(path, "rb") as f:
        while True:
            nm = f.read(48)
            if len(nm) < 48:
                break
            name = nm.decode().strip()
            n = int(np.frombuffer(f.read(8), dtype="<i8")[0])
            o, cnt = offs[name]
            assert cnt == n, (name, cnt, n)
            blob[o:o + n] = np.frombuffer(f.read(8 * n), dtype="<f8")
            seen.add(name)
    missing = [n for n, _ in layout if n not in seen]
    assert not missing, missing
    return blob


KERNEL = r"""
#include "tempo_column.cuh"
struct ColState { double f[17][TEMPO_NZP]; };
extern "C" __global__ void tempo_columns_nvrtc(const double* blob, TempoCtx C, int ncol, int nz,
                                              ColState* S, TempoWork* W) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncol) return;
  TempoParams P{reinterpret_cast<const TempoBlob*>(blob)};
  TempoCtx c = C;
  if (i != 0) c.first_call_main = 0;
  ColState& s = S[i];
  tempo_main_column(P, c, nz, W[i], s.f[4], s.f[5], s.f[7], s.f[6], s.f[8], s.f[9], s.f[16],
                    s.f[10], s.f[11], s.f[12], s.f[15], s.f[13], s.f[14], s.f[0], s.f[1], s.f[2], s.f[3]);
}
extern "C" __global__ void tempo_sizes(long long* out) {
  out[0] = sizeof(TempoWork); out[1] = sizeof(ColState); out[2] = sizeof(TempoCtx);
}
"""

CTX = np.dtype([("global_dt", "<f8"), ("global_inverse_dt", "<f8"), ("aerosolaware_flag", "<i4"),
                ("hailaware_flag", "<i4"), ("semi_sedi_flag", "<i4"),
                ("refl10cm_from_melting_flag", "<i4"), ("turn_off_micro_flag", "<i4"),
                ("cloud_condensation_flag", "<i4"), ("refl10cm_flag", "<i4"),
                ("re_cloud_flag", "<i4"), ("re_ice_flag", "<i4"), ("re_snow_flag", "<i4"),
                ("max_hail_diameter_flag", "<i4"), ("first_call_main", "<i4")])


def main():
    kdir, blob_path, cols_path, nvcc_out = sys.argv[1:5]
    nrep = 4096
    header = open(f"{kdir}/tempo_common.cuh").read()
    layout = blob_layout(header)
    blob = cp.asarray(load_blob(blob_path, layout))
    mod = cp.RawModule(code=KERNEL, options=("-std=c++17", f"-I{kdir}", "--fmad=false"))
    sizes = cp.zeros(3, dtype=cp.int64)
    mod.get_function("tempo_sizes")((1,), (1,), (sizes,))
    work_b, col_b, ctx_b = (int(x) for x in sizes.get())
    assert ctx_b == CTX.itemsize, (ctx_b, CTX.itemsize)
    nzp = col_b // (8 * NFLD)
    kern = mod.get_function("tempo_columns_nvrtc")

    raw = np.fromfile(cols_path, dtype="<f8")
    ref_gpu = np.fromfile(nvcc_out, dtype="<f8")
    i = 0
    o = 0
    first = 1
    all_ok = True
    S = cp.zeros((nrep, NFLD, nzp), dtype=cp.float64)
    W = cp.zeros(nrep * work_b // 8, dtype=cp.float64)
    while i < raw.size:
        hdr = raw[i:i + 5].view("<i8")
        sid, nz, nsteps = int(hdr[0]), int(hdr[1]), int(hdr[2])
        dt, semi = float(raw[i + 3]), float(raw[i + 4])
        i += 5
        init = raw[i:i + NFLD * nz].reshape(NFLD, nz)
        i += NFLD * nz
        step = NFLD * nz + 7 + 4 * nz
        i += step * nsteps
        S[...] = 0.0
        S[:, :, 1:nz + 1] = cp.asarray(init)[None]
        for it in range(nsteps):
            ctx = np.zeros(1, dtype=CTX)
            ctx["global_dt"] = dt
            ctx["global_inverse_dt"] = 1.0 / dt
            for f in ("aerosolaware_flag", "hailaware_flag", "cloud_condensation_flag",
                      "refl10cm_flag", "re_cloud_flag", "re_ice_flag", "re_snow_flag",
                      "max_hail_diameter_flag"):
                ctx[f] = 1
            ctx["semi_sedi_flag"] = 1 if semi > 0.5 else 0
            ctx["first_call_main"] = first
            first = 0
            kern(((nrep + 63) // 64,), (64,),
                 (blob, ctx, np.int32(nrep), np.int32(nz), S, W))
            state = S[0, :, 1:nz + 1].get().reshape(-1)
            want = ref_gpu[o:o + NFLD * nz]
            o += step
            if not np.array_equal(state.view("<i8"), want.view("<i8")):
                all_ok = False
        rep_equal = bool(cp.all(S == S[0:1]).get())
        print(f"scenario {sid}: replicas_bitwise_equal={rep_equal}")
        all_ok = all_ok and rep_equal
    print(f"NVRTC_MATCHES_NVCC_BITWISE {int(all_ok)}")


if __name__ == "__main__":
    main()
