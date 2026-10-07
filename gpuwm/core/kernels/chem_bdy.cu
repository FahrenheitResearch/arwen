// WRF chem/module_input_chem_data.F:1531-2031, :2331-2357.
// One thread owns a field/level, preserving all four serial side loops.
extern "C" __global__ void chem_flow_boundaries(
    const unsigned long long* pointers, const float* u, const float* v,
    const int* has, const float* defaults,
    const float* bxs, const float* btxs, const float* bxe, const float* btxe,
    const float* bys, const float* btys, const float* bye, const float* btye,
    float dt, int nf, int nz, int ny, int nx, int w)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nf*nz) return;
    int n = tid/nz, k = tid%nz;
    float* f = reinterpret_cast<float*>(pointers[n]);
    for (int side = 0; side < 4; ++side) {
        for (int d = 0; d < w; ++d) {
            int distance = (side == 0 || side == 2) ? d : w-1-d;
            int start = side < 2 ? distance : distance+1;
            int stop = side < 2 ? nx-distance : ny-distance-1;
            for (int a = start; a < stop; ++a) {
                int i, j, si, sj;
                bool out;
                const float *b, *bt;
                size_t bi;
                if (side < 2) {
                    i = a; j = side == 0 ? d : ny-w+d;
                    si = min(max(i,w),nx-1-w);
                    sj = side == 0 ? w : ny-1-w;
                    float flux = v[((size_t)k*(ny+1)+j+(side==1))*nx+i];
                    out = side == 0 ? flux < 0.0f : flux > 0.0f;
                    b = side == 0 ? bys : bye;
                    bt = side == 0 ? btys : btye;
                    bi = ((size_t)n*nz+k)*nx+a;
                } else {
                    i = side == 2 ? d : nx-w+d; j = a;
                    si = side == 2 ? w : nx-1-w;
                    sj = min(max(j,w),ny-1-w);
                    float flux = u[((size_t)k*ny+j)*(nx+1)+i+(side==3)];
                    out = side == 2 ? flux < 0.0f : flux > 0.0f;
                    b = side == 2 ? bxs : bxe;
                    bt = side == 2 ? btxs : btxe;
                    bi = ((size_t)n*nz+k)*ny+a;
                }
                size_t dest = ((size_t)k*ny+j)*nx+i;
                if (out) f[dest] = f[((size_t)k*ny+sj)*nx+si];
                else if (has[n])
                    f[dest] = fmaxf(1.e-16f, __fadd_rn(b[bi], __fmul_rn(bt[bi],dt)));
                else f[dest] = defaults[n];
            }
        }
    }
}
