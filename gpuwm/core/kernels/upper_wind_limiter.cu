// NOAA WRFV3.9 module_small_step_em.F:1703-1735, advance_w.
// One thread owns a column: qq is updated in descending vertical order.
extern "C" __global__
void upper_wind_limiter(float* u, float* v, const float* php,
                        const float* phb, float dts, float hdepth,
                        int base3d, int zone, int nz, int ny, int nx) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    const int j = col / nx, i = col % nx;
    if (i < zone || i >= nx-zone || j < zone || j >= ny-zone) return;
    float qq = 0.0f;
    for (int k = nz-1; k >= nz-3; --k) {
        const float uk = u[(k*ny+j)*(nx+1)+i];
        const float vk = v[(k*(ny+1)+j)*nx+i];
        qq = fmaxf(qq, __fadd_rn(__fmul_rn(uk, uk), __fmul_rn(vk, vk)));
    }
    qq = __fdiv_rn(qq, 12100.0f);
    if (qq <= 1.0f) return;
    const int top = nz*ny*nx+col;
    const float htop = __fdiv_rn(__fadd_rn(php[top], phb[base3d ? top : nz]), G);
    const float hbot = __fsub_rn(htop, hdepth);
    for (int k = nz-1; k >= 1; --k) {
        const int pos = k*ny*nx+col;
        const float hk = __fdiv_rn(__fadd_rn(php[pos], phb[base3d ? pos : k]), G);
        if (hk >= hbot && qq > 1.0f) {
            qq = sqrtf(__fmul_rn(__fsub_rn(qq, 1.0f), qq));
            float drag = __fmul_rn(__fdiv_rn(dts, 86400.0f), 2.0f);
            drag = __fmul_rn(drag, qq);
            drag = __fmul_rn(drag, gfk_pow(2.0f, __fmul_rn(0.4f, float(k-(nz-1)))));
            const float factor = __fsub_rn(1.0f, drag);
            const int up = (k*ny+j)*(nx+1)+i;
            const int vp = (k*(ny+1)+j)*nx+i;
            u[up] = __fmul_rn(u[up], factor);
            v[vp] = __fmul_rn(v[vp], factor);
        }
    }
}
