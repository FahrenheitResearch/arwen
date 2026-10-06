// Vertical low-order scalar flux of module_advect_em.F vert_order 5,
// NOAA-EMC/HRRR 40ee6058c, F:8378-8501. The pre-included pd_advection.cu
// supplies the unchanged high-order face helper. This entry overwrites
// only eta faces after pd_fluxes; order 3 never launches this module.
//
// Above face Courant 1 every build takes the fork's bounded
// semi-Lagrangian sum of the upstream cells. At |cr| <= 1 the fork takes
// the DOWNSTREAM cell (rw > 0 is downward, so its donor is cell k, and the
// fork reads cell k-1). That flux drains an empty upstream cell: the
// limiter's low-order budget goes negative there, the final-stage update
// clamps the result to zero, and the clamp manufactures mass. On a 12 h
// 3 km smoke forecast it created 65 t of smoke against 722 t emitted, and
// it does the same to every hydrometeor and chem row near zero. The
// production build therefore leaves those faces with pd_fluxes' upwind
// flux (WRF 4.7.1 flux_upwind, positive definite) and writes only the
// faces above Courant 1. The strict-arithmetic WRF verification build
// (GPUWM_WRF_EXACT_C_ADVECTION) keeps the fork's words at every face, so
// the native fork gate still measures the rest of this transcription.
extern "C" __global__
void pd_vertical_sl(const real* __restrict__ q,
                    const real* __restrict__ q0,
                    const real* __restrict__ ru,
                    const real* __restrict__ rv,
                    const real* __restrict__ rw,
                    const real* __restrict__ mut,
                    const real* __restrict__ c1h,
                    const real* __restrict__ c2h,
                    const real* __restrict__ rdnw,
                    const real* __restrict__ fnm,
                    const real* __restrict__ fnp,
                    const real* __restrict__ msft,
                    real dx, real dy, real dt,
                    real* __restrict__ fxl, real* __restrict__ fxc,
                    real* __restrict__ fyl, real* __restrict__ fyc,
                    real* __restrict__ fzl, real* __restrict__ fzc,
                    int nz, int ny, int nx, int has_msf,
                    int open_x, int open_y, int vorder)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k > nz) return;
    real fl = 0.0f, fc = 0.0f;
    if (k > 0 && k < nz) {
        real vel = rw[IDX3(k,j,i)];
        bool multi = false;
#if GPUWM_WRF_EXACT_C_ADVECTION
        fl = fmaxf(vel,0.0f)*q0[IDX3(k-1,j,i)]
           + fminf(vel,0.0f)*q0[IDX3(k,j,i)];
#endif
        if (k > 1 && k < nz-1) {
            real dz = __fdiv_rn(2.0f,rdnw[k]+rdnw[k-1]);
            real mu = c1h[k]*mut[(size_t)j*nx+i]+c2h[k];
            real cr = __fdiv_rn(__fdiv_rn(vel*dt,dz),mu);
            if (cr > 1.0f || cr < -1.0f) {
                int shift = (int)copysignf(floorf(fabsf(cr)),cr);
                int low = k == nz-2 ? -2 : -3;
                int high = k == 2 ? 2 : 3;
                shift = min(max(shift,low),high);
                real sum = 0.0f;
                int first = cr > 1.0f ? k-shift : k;
                int last = cr > 1.0f ? k-1 : k-shift-1;
                for (int level = first; level <= last; ++level)
                    sum = sum + q0[IDX3(level,j,i)];
                // deps is REAL(KIND=8) in the native routine. Preserve
                // the binary32 product and binary64 quotient round points.
                fl = __double2float_rn(__ddiv_rn((double)(vel*sum),
                                     fmax(1.0e-15,(double)fabsf(cr))));
                multi = true;
            }
        }
#if !GPUWM_WRF_EXACT_C_ADVECTION
        // Courant <= 1: pd_fluxes' upwind words stand.
        if (!multi) return;
#endif
        fc = pd_zface_half(q,vel,k,j,i,nz,ny,nx,fnm,fnp,5)-fl;
    }
    fzl[IDX3(k,j,i)] = fl;
    fzc[IDX3(k,j,i)] = fc;
}
