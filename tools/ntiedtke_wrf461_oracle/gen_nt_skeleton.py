"""Generate the two New Tiedtke frame-probe skeletons.

Variant A: the 79 per-column arrays as function-scope locals -- what a
           straight transcription of cu_ntiedtke.F90 produces, and the
           ceiling the workspace has to beat.
Variant B: the same 79 in a global workspace laid out the way local memory
           is (lane-interleaved, one region per block), which is the shape
           gf.cu / kf.cu / ysu.cu converged on.

Both carry the SAME access pattern -- a downward sweep, an upward sweep with
a cross-array dependence, and a reduction -- so ptxas cannot dead-code any
array, and so the two differ only in WHERE the arrays live.
"""
# (name, ctype, extent) -- from tools/ nt_slots.py over cu_ntiedtke.F90
RUN = """pum1 pvm1 ztt ptte pqte pvom pvol pverv pgeo zqq pcte ztp1 zqp1
         ztu zqu zlu zlude zmfu zmfd zqsat""".split()
RUN1 = ["pgeoh"]                                   # km1
MASTER = """pmfude_rate pmfdde_rate zdpmel zmfuus zmfdus zuv2 ztenu ztenv
            ztenh zqenh zqsenh ztd zqd zmfus zmfds zmfuq zmfdq zdmfup
            zdmfdp zmful zuu zvu zud zvd zlglac""".split()
MASTER1 = ["pmflxr", "pmflxs"]                     # klevp1
MASTER_I = ["ilab"]
CALLEE = """dh dhen kup vptu vten ptu pqu plu zbuo abuoy plude zlrain
            zodetr pdmfen ztenwb zqenwb pdpmel plglac zdtdt zdqdt zdp
            zdudt zdvdt zuen zven zmfuu zmfdu zmfuv zmfdv""".split()
CALLEE_I = ["klab"]

FLOAT_KP  = RUN + MASTER + CALLEE
FLOAT_KP1 = RUN1 + MASTER1
INT_KP    = MASTER_I + CALLEE_I

SLOTS = ([(n, "float", "kp")  for n in FLOAT_KP]
         + [(n, "float", "kp1") for n in FLOAT_KP1]
         + [(n, "int",   "kp")  for n in INT_KP])
assert len(SLOTS) == 79, len(SLOTS)


def body(deref):
    """The shared access pattern.  `deref(name)` renders an array reference."""
    f = [n for n, t, _ in SLOTS if t == "float"]
    i = [n for n, t, _ in SLOTS if t == "int"]
    L = []
    # seed every array from global input -- forces materialisation
    L.append("  for (int k = 0; k < nz; ++k) {")
    for n in f:
        L.append(f"    {deref(n)}[k] = in[base_in + k] * {1.0 + f.index(n)*0.01:.4f}f;")
    for n in i:
        L.append(f"    {deref(n)}[k] = k + {i.index(n)};")
    L.append("  }")
    # downward sweep with a cross-array dependence (the updraft direction:
    # this scheme is TOP-DOWN, k=1 is the model top)
    L.append("  for (int k = 1; k < nz; ++k) {")
    for a, b in zip(f, f[1:] + f[:1]):
        L.append(f"    {deref(a)}[k] = __fmaf_rn({deref(a)}[k-1], 0.5f, "
                 f"{deref(b)}[k]);")
    L.append("  }")
    # upward sweep (the downdraft direction), integer arrays gate it
    L.append("  for (int k = nz - 2; k >= 0; --k) {")
    L.append(f"    int g = {deref(i[0])}[k] + {deref(i[1])}[k];")
    for a, b in zip(f, f[1:] + f[:1]):
        L.append(f"    {deref(a)}[k] = (g & 1) ? "
                 f"__fmaf_rn({deref(a)}[k+1], 0.25f, {deref(b)}[k]) "
                 f": {deref(a)}[k];")
    L.append("  }")
    # reduce so nothing is dead
    L.append("  float acc = 0.0f;")
    L.append("  for (int k = 0; k < nz; ++k) {")
    for n in f:
        L.append(f"    acc = __fadd_rn(acc, {deref(n)}[k]);")
    L.append("  }")
    L.append("  out[i] = acc;")
    return "\n".join(L)


HDR = """// New Tiedtke frame probe -- COMPILE ONLY, never launched.
// Not physics.  The declarations and the access pattern of the real port,
// with the arithmetic replaced by dependent FMAs, so ptxas allocates the
// same frame it would for the transcription.
#ifndef NT_KP
#define NT_KP 49
#endif
"""

# ---- variant A: everything on the per-thread stack -------------------------
a = [HDR, 'extern "C" __global__ void nt_skeleton_stack(',
     "    const float* __restrict__ in, float* __restrict__ out, int nz) {",
     "  int i = blockIdx.x * blockDim.x + threadIdx.x;",
     "  size_t base_in = (size_t)i * (size_t)NT_KP;"]
for n, t, e in SLOTS:
    a.append(f"  {t} {n}[{'NT_KP + 1' if e == 'kp1' else 'NT_KP'}];")
a.append(body(lambda n: n))
a.append("}")
open("nt_probe_stack.cu", "w", newline="\n").write("\n".join(a) + "\n")

# ---- variant B: the arrays in a lane-interleaved global workspace ----------
b = [HDR, "#define NTWS_SLOTS 79", "#define NTWS_LANES 32",
     "struct NtCol { float* p;",
     "  __device__ __forceinline__ float& operator[](int k) const {",
     "    return p[(size_t)k * (size_t)NTWS_LANES]; } };",
     "struct NtColI { int* p;",
     "  __device__ __forceinline__ int& operator[](int k) const {",
     "    return p[(size_t)k * (size_t)NTWS_LANES]; } };",
     "#define NTWS_AT(base, idx, kp) \\",
     "  (NtCol{(base) + (size_t)(idx) * (size_t)(kp) * (size_t)NTWS_LANES})",
     "#define NTWS_AT_I(base, idx, kp) \\",
     "  (NtColI{(int*)((base) + (size_t)(idx) * (size_t)(kp) "
     "* (size_t)NTWS_LANES)})",
     "#define NTWS_LANE_BASE(ws, kp) \\",
     "  ((ws) + (size_t)blockIdx.x * (size_t)NTWS_SLOTS * (size_t)(kp) \\",
     "          * (size_t)NTWS_LANES + (size_t)threadIdx.x)",
     'extern "C" __global__ void nt_skeleton_ws(',
     "    const float* __restrict__ in, float* __restrict__ out,",
     "    float* __restrict__ ws, int nz) {",
     "  int i = blockIdx.x * blockDim.x + threadIdx.x;",
     "  size_t base_in = (size_t)i * (size_t)NT_KP;",
     "  const int kp = nz + 1;",
     "  float* lane = NTWS_LANE_BASE(ws, kp);"]
for idx, (n, t, e) in enumerate(SLOTS):
    mac = "NTWS_AT_I" if t == "int" else "NTWS_AT"
    ty = "NtColI" if t == "int" else "NtCol"
    b.append(f"  {ty} {n} = {mac}(lane, {idx}, kp);")
b.append(body(lambda n: n))
b.append("}")
open("nt_probe_ws.cu", "w", newline="\n").write("\n".join(b) + "\n")
print(f"wrote nt_probe_stack.cu and nt_probe_ws.cu ({len(SLOTS)} slots)")
