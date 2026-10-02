// WRF v4.7.1 dyn_em/module_bc_em.F:1196-1295, cold-start fill mode.
// Every arithmetic operation rounds to FP32 without contraction.
extern "C" __global__ void wrf_cold_start_w(
    const float* u, const float* v, const float* ht,
    const float* msftx, const float* msfty, const float* znw,
    float* w, float cf1, float cf2, float cf3, float rdx, float rdy,
    int periodic_x, int periodic_y, int nz, int ny, int nx) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;
    int im1 = i == 0 ? (periodic_x ? nx - 1 : 0) : i - 1;
    int ip1 = i == nx - 1 ? (periodic_x ? 0 : nx - 1) : i + 1;
    int jm1 = j == 0 ? (periodic_y ? ny - 1 : 0) : j - 1;
    int jp1 = j == ny - 1 ? (periodic_y ? 0 : ny - 1) : j + 1;
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
    float dyn = __fsub_rn(ht[jp1 * nx + i], ht[col]);
    float dys = __fsub_rn(ht[col], ht[jm1 * nx + i]);
    float dxe = __fsub_rn(ht[j * nx + ip1], ht[col]);
    float dxw = __fsub_rn(ht[col], ht[j * nx + im1]);
    float yscale = __fmul_rn(__fmul_rn(msfty[col], 0.5f), rdy);
    float xscale = __fmul_rn(__fmul_rn(msftx[col], 0.5f), rdx);
    float y = __fmul_rn(yscale, __fadd_rn(__fmul_rn(dyn, vc1),
                                       __fmul_rn(dys, vc0)));
    float x = __fmul_rn(xscale, __fadd_rn(__fmul_rn(dxe, uc1),
                                       __fmul_rn(dxw, uc0)));
    float surface = __fadd_rn(y, x);
    w[col] = surface;
    for (int k = 1; k <= nz; ++k) {
        w[k * ny * nx + col] = __fmul_rn(__fmul_rn(surface, znw[k]), znw[k]);
    }
}
