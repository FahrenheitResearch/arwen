// Sedimentation reference from exported base d6929cb8d.
__device__ int base_morr_sediment_nstep(
        const real* qc, const real* qr, const real* qi,
        const real* qs, const real* qg,
        const real* nc, const real* nr, const real* ni,
        const real* ns, const real* ng,
        const real* cloud_nc_for_sedimentation,
        const real* theta, const real* pii, const real* rho_fixed,
        const real* dz, int j, int i, int nz, int ny, int nx, real dt,
        real morr_ag, real morr_bg, real morr_rhog)
{
    real max_courant = 0.0f;
    // Level-outer: rho, theta, pii and dz are one load per level instead of
    // one per level per category.  The reduction is a chain of fmaxf, which
    // is exactly associative and commutative -- max rounds nothing, and
    // fmaxf(NaN, x) == x makes NaN a two-sided identity -- so regrouping it
    // by level cannot move a bit.  Each category still walks downward, so
    // its empty-level speed rebound is unchanged; the five carried pairs
    // stay in registers because the category loop is fully unrolled.
    real vm_above[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    real vn_above[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = nz - 1; k >= 0; --k) {
        size_t idx = IDX3(k, j, i);
        real rhoa = fabsf(rho_fixed[idx]);
        real temp = theta[idx] * pii[idx];
        real dzk = dz[idx];
#pragma unroll
        for (int kind = 0; kind < 5; ++kind) {
            const real* mass = kind == 0 ? qc : (kind == 1 ? qr :
                               (kind == 2 ? qi : (kind == 3 ? qs : qg)));
            const real* number = kind == 0 ? cloud_nc_for_sedimentation :
                                 (kind == 1 ? nr :
                                 (kind == 2 ? ni : (kind == 3 ? ns : ng)));
            real nn, vm, vn;
            morr_terminal_velocity(kind, fmaxf(mass[idx], 0.0f),
                                   number[idx], rhoa, temp, &nn, &vm, &vn,
                                   morr_ag, morr_bg, morr_rhog);
            if (k < nz - 1) {
                if (vm < 1.0e-10f) vm = vm_above[kind];
                if (vn < 1.0e-10f) vn = vn_above[kind];
            }
            vm_above[kind] = vm; vn_above[kind] = vn;
            max_courant = fmaxf(max_courant,
                                fmaxf(vm, vn) * dt / dzk);
        }
    }
    return max((int)(max_courant + 1.0f), 1);
}

template <int KMAX>
__device__ __forceinline__ real base_morr_sediment_pair(
        real* mass, real* number, const real* sediment_number,
        const real* theta, const real* pii,
        const real* pressure, const real* rho_fixed, const real* dz, int j, int i,
        int nz, int ny, int nx, int kind, real dt, int nstep,
        real morr_ag, real morr_bg, real morr_rhog)
{
    real qd[KMAX], nd[KMAX], nd0[KMAX];
    real vm[KMAX], vn[KMAX];
    real vm_above = 0.0f, vn_above = 0.0f;
    int ktop = -1;
    for (int k = nz - 1; k >= 0; --k) {
        size_t idx = IDX3(k, j, i);
        real temp = theta[idx] * pii[idx];
        real rhoa = fabsf(rho_fixed[idx]);
        real q = fmaxf(mass[idx], 0.0f);
        real nn;
        real nsed = sediment_number == nullptr ? number[idx]
                                                : sediment_number[idx];
        morr_terminal_velocity(kind, q, nsed, rhoa, temp,
                               &nn, &vm[k], &vn[k],
                               morr_ag, morr_bg, morr_rhog);
        if (k < nz - 1) {
            if (vm[k] < 1.0e-10f) vm[k] = vm_above;
            if (vn[k] < 1.0e-10f) vn[k] = vn_above;
        }
        vm_above = vm[k]; vn_above = vn[k];
        // Highest level this category can fall from.  The rebound above
        // makes the speeds non-zero all the way down from it, so the
        // sedimenting span is exactly [0, ktop].
        if (ktop < 0 && (vm[k] != 0.0f || vn[k] != 0.0f)) ktop = k;
        qd[k] = q * rhoa;
        // DLAM rebound changes fall speed only; flux the clipped prognostic
        // number moment itself (WRF 3376-3432).
        nd[k] = fmaxf(nsed, 0.0f) * rhoa;
        nd0[k] = nd[k];
    }
    real dts = dt / (real)nstep;
    real exported = 0.0f;
    // Above ktop both speeds are exactly zero, so every flux there is
    // 0*qd == +0 on a quantity that cannot be negative: those levels are the
    // identity and are skipped.  Entering the span with a zero inflow rather
    // than a separate top statement is bit-exact -- (0-x) == -x and
    // ((0-x)*d)/z == -((x*d)/z) -- and keeps the k==0 surface export on the
    // one path that reaches the ground, which a span top of 0 shares.
    for (int nsub = 0; ktop >= 0 && nsub < nstep; ++nsub) {
        // The update walks downward, so only the flux through the interface
        // above the current level must survive.  Carrying those two FP32
        // values preserves every per-level expression while avoiding two
        // KMAX local arrays (512 bytes/thread in the d01 specialization).
        real fm_above = 0.0f, fn_above = 0.0f;
        for (int k = ktop; k >= 0; --k) {
            size_t idx = IDX3(k, j, i);
            real fm_here = vm[k] * qd[k];
            real fn_here = vn[k] * nd[k];
            if (k == 0) exported += fm_here * dts;
            qd[k] += (fm_above - fm_here) * dts / dz[idx];
            nd[k] += (fn_above - fn_here) * dts / dz[idx];
            fm_above = fm_here;
            fn_above = fn_here;
        }
    }
    for (int k = 0; k < nz; ++k) {
        size_t idx = IDX3(k, j, i);
        real temp = theta[idx] * pii[idx];
        real rhoa = rho_fixed[idx];
        mass[idx] = fmaxf(qd[k] / rhoa, 0.0f);
        real sedimented = fmaxf(nd[k] / rhoa, 0.0f);
        if (sediment_number == nullptr) {
            number[idx] = sedimented;
        } else {
            // DUMFNC excludes the local NC3DTEN for INUM=1.  WRF adds the
            // fallout tendency (sedimented DUMFNC minus DUMFNC) to the
            // already-held local tendency; do not replace post-process NC.
            number[idx] = fmaxf(number[idx]
                                 + (nd[k] - nd0[k]) / rhoa, 0.0f);
        }
    }
    return exported;
}

template <int KMAX>
__device__ __forceinline__
void base_morrison_sediment_impl(real* __restrict__ qc,
                            real* __restrict__ qr,
                            real* __restrict__ qi,
                            real* __restrict__ qs,
                            real* __restrict__ qg,
                            real* __restrict__ nc,
                            real* __restrict__ nr,
                            real* __restrict__ ni,
                            real* __restrict__ ns,
                            real* __restrict__ ng,
                            const real* __restrict__ cloud_nc,
                            const real* __restrict__ theta,
                            const real* __restrict__ pii,
                            const real* __restrict__ pressure,
                            const real* __restrict__ rho_in,
                            const real* __restrict__ dz,
                            real* __restrict__ rainnc,
                            real* __restrict__ rainncv,
                            real* __restrict__ snownc,
                            real* __restrict__ snowncv,
                            real* __restrict__ graupelnc,
                            real* __restrict__ graupelncv,
                            real* __restrict__ sr,
                            real dt, real morr_ag, real morr_bg,
                            real morr_rhog, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;
    int nstep = base_morr_sediment_nstep(qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
                                    cloud_nc,
                                    theta, pii, rho_in, dz,
                                    j, i, nz, ny, nx, dt,
                                    morr_ag, morr_bg, morr_rhog);
    real out_c = base_morr_sediment_pair<KMAX>(qc, nc, cloud_nc, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 0, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_r = base_morr_sediment_pair<KMAX>(qr, nr, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 1, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_i = base_morr_sediment_pair<KMAX>(qi, ni, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 2, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_s = base_morr_sediment_pair<KMAX>(qs, ns, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 3, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_g = base_morr_sediment_pair<KMAX>(qg, ng, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 4, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);

    size_t sidx = (size_t)j * nx + i;
    real total = out_c + out_r + out_i + out_s + out_g;
    real snow = out_i + out_s;
    rainncv[sidx] = total;
    snowncv[sidx] = snow;
    graupelncv[sidx] = out_g;
    rainnc[sidx] += total;
    snownc[sidx] += snow;
    graupelnc[sidx] += out_g;
    sr[sidx] = (snow + out_g) / (total + 1.0e-12f);
}

#define base_MORRISON_SEDIMENT_PARAMETERS                                         \
    real* __restrict__ qc, real* __restrict__ qr,                            \
    real* __restrict__ qi, real* __restrict__ qs,                            \
    real* __restrict__ qg, real* __restrict__ nc,                            \
    real* __restrict__ nr, real* __restrict__ ni,                            \
    real* __restrict__ ns, real* __restrict__ ng,                            \
    const real* __restrict__ cloud_nc, const real* __restrict__ theta,       \
    const real* __restrict__ pii, const real* __restrict__ pressure,         \
    const real* __restrict__ rho_in, const real* __restrict__ dz,            \
    real* __restrict__ rainnc, real* __restrict__ rainncv,                    \
    real* __restrict__ snownc, real* __restrict__ snowncv,                    \
    real* __restrict__ graupelnc, real* __restrict__ graupelncv,              \
    real* __restrict__ sr, real dt, real morr_ag, real morr_bg,              \
    real morr_rhog, int nz, int ny, int nx

#define base_MORRISON_SEDIMENT_ARGUMENTS                                          \
    qc, qr, qi, qs, qg, nc, nr, ni, ns, ng, cloud_nc, theta, pii, pressure, \
    rho_in, dz, rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, \
    dt, morr_ag, morr_bg, morr_rhog, nz, ny, nx

extern "C" __global__
void base_morrison_sediment_64(base_MORRISON_SEDIMENT_PARAMETERS)
{
    base_morrison_sediment_impl<MORR_KMAX_SHALLOW>(base_MORRISON_SEDIMENT_ARGUMENTS);
}

extern "C" __global__
void base_morrison_sediment_256(base_MORRISON_SEDIMENT_PARAMETERS)
{
    base_morrison_sediment_impl<MORR_KMAX_GENERIC>(base_MORRISON_SEDIMENT_ARGUMENTS);
}

#undef base_MORRISON_SEDIMENT_ARGUMENTS
#undef base_MORRISON_SEDIMENT_PARAMETERS
