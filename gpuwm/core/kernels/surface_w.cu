extern "C" __global__ void set_surface_w(
    const float* u, const float* v, const float* ht, const float* msft,
    float* w, float cf1, float cf2, float cf3, float half_rdx,
    float half_rdy, int mapped, int bx, int by, int ny, int nx) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;
    int uplane = ny * (nx + 1);
    int vplane = (ny + 1) * nx;
    int ul = j * (nx + 1) + i;
    int vb = j * nx + i;
    float uc0 = __fadd_rn(__fadd_rn(__fmul_rn(cf1, u[ul]),
                                  __fmul_rn(cf2, u[uplane + ul])),
                         __fmul_rn(cf3, u[2 * uplane + ul]));
    float uc1 = __fadd_rn(__fadd_rn(__fmul_rn(cf1, u[ul + 1]),
                                  __fmul_rn(cf2, u[uplane + ul + 1])),
                         __fmul_rn(cf3, u[2 * uplane + ul + 1]));
    float vc0 = __fadd_rn(__fadd_rn(__fmul_rn(cf1, v[vb]),
                                  __fmul_rn(cf2, v[vplane + vb])),
                         __fmul_rn(cf3, v[2 * vplane + vb]));
    float vc1 = __fadd_rn(__fadd_rn(__fmul_rn(cf1, v[vb + nx]),
                                  __fmul_rn(cf2, v[vplane + vb + nx])),
                         __fmul_rn(cf3, v[2 * vplane + vb + nx]));
    float dyn = __fsub_rn(ht[((j + 1) % ny) * nx + i], ht[col]);
    float dys = __fsub_rn(ht[col], ht[((j + ny - 1) % ny) * nx + i]);
    float dxe = __fsub_rn(ht[j * nx + (i + 1) % nx], ht[col]);
    float dxw = __fsub_rn(ht[col], ht[j * nx + (i + nx - 1) % nx]);
    // WRF module_bc_em.F clamps the outside donor to this cell. Its
    // terrain difference is zero; copying the inside slope doubles it.
    if (by && j == 0) dys = 0.0f;
    if (by && j == ny - 1) dyn = 0.0f;
    if (bx && i == 0) dxw = 0.0f;
    if (bx && i == nx - 1) dxe = 0.0f;
#if GPUWM_WRF_EXACT
    // WRF set_w_surface: msfty*.5*rdy*(...) + msftx*.5*rdx*(...), each map
    // factor taken into its own coefficient left to right; rdx = 1./dx is
    // twice half_rdx exactly.  msft is 1 on an unmapped grid.
    float msf = mapped ? msft[col] : 1.0f;
    float cy = __fmul_rn(__fmul_rn(msf, 0.5f), __fmul_rn(2.0f, half_rdy));
    float cx = __fmul_rn(__fmul_rn(msf, 0.5f), __fmul_rn(2.0f, half_rdx));
    float y = __fmul_rn(cy, __fadd_rn(__fmul_rn(dyn, vc1),
                                      __fmul_rn(dys, vc0)));
    float x = __fmul_rn(cx, __fadd_rn(__fmul_rn(dxe, uc1),
                                      __fmul_rn(dxw, uc0)));
    w[col] = __fadd_rn(y, x);
#else
    float y = __fmul_rn(half_rdy, __fadd_rn(__fmul_rn(dyn, vc1),
                                          __fmul_rn(dys, vc0)));
    float x = __fmul_rn(half_rdx, __fadd_rn(__fmul_rn(dxe, uc1),
                                          __fmul_rn(dxw, uc0)));
    float out = __fadd_rn(y, x);
    if (mapped) out = __fmul_rn(out, msft[col]);
    w[col] = out;
#endif
}
