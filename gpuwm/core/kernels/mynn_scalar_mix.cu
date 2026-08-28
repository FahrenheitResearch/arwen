// WRF v4.6.1 MYNN mixscalars arms — the W4 full-admission lane (GPU half).
//
// NEW translation unit (the O2 wave of the MYNN port): the frozen
// kernels/mynn_pbl.cu (byte pin b53ab90e... in tests/test_mp8_frozen.py) is
// never edited.  This unit implements exactly what the CPU reference module
// gpuwm/core/mynn_scalar_mix.py implements — the five stock qn-family
// tridiagonal solves (module_bl_mynn.F:4654/:4695/:4736/:4778/:4820,
// nonloc=1.0 parameter at :4123) and the scalar_opt>0 DMP_mf updraft-flux
// accumulation (:6447-6456 with init :6140-6144, entrain :6213-6217, store
// :6351-6355 and the :6485-6489 limiter rescale) — consuming the plume-edge
// terms the admitted DMP_mf produces (up_w / PRE-limiter up_a / ent / rhoz /
// psig_w / plume_active / limiter_adjustment), never recomputing plume
// dynamics.  One kernel launch handles ONE species over a column batch, so
// the same two kernels serve qni/qnc/qnwfa/qnifa/qnbca, and a later wave
// can add further species on the same no-floor mirror identity.
//
// This unit is deliberately self-contained: it is NOT listed in
// gpuwm.core.kernels._EXTRA_HEADERS, so every existing module's assembled
// source stays byte-identical by construction
// (tests/test_kernel_loader_inert.py).  The pinned-arithmetic helpers below
// are verbatim copies of the mynn_pbl.cu vocabulary, for the same three
// reasons documented there: NVRTC contracts a*b+c into FMA, CuPy appends
// -ftz=true to every NVRTC compile (subnormal flush), and ptxas folds
// host-side constant expressions with the wrong tie rounding.  Inline PTX
// without the .ftz modifier is immune to all three.  Do not rewrite these as
// plain operators or as the __f*_rn intrinsics.

__device__ __forceinline__ real smx_add(real a, real b)
{
    real r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real smx_sub(real a, real b)
{
    real r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real smx_mul(real a, real b)
{
    real r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real smx_div(real a, real b)
{
    real r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

#define SMX_ADD(x, y) smx_add((x), (y))
#define SMX_SUB(x, y) smx_sub((x), (y))
#define SMX_MUL(x, y) smx_mul((x), (y))
#define SMX_DIV(x, y) smx_div((x), (y))

// Fortran comparison kept out of ptxas' folding reach, exactly as the
// frozen unit keeps mynn_gt/mynn_max2.
__device__ __forceinline__ bool smx_gt(real a, real b)
{
    unsigned int p;
    asm("{ .reg .pred %%gt; setp.gt.f32 %%gt, %1, %2; selp.u32 %0, 1, 0, "
        "%%gt; }" : "=r"(p) : "f"(a), "f"(b));
    return p != 0u;
}

__device__ __forceinline__ real smx_max2(real a, real b)
{
    return smx_gt(b, a) ? b : a;
}

// module_bl_mynn.F:5422 tridiag2, verbatim from the frozen unit's
// mynn_tridiag2_column.  x may alias d; cpw/dpw may not alias anything.
__device__ void smx_tridiag2_column(
    const real* __restrict__ a, const real* __restrict__ b,
    const real* __restrict__ c, const real* __restrict__ d,
    real* __restrict__ cpw, real* __restrict__ dpw,
    real* __restrict__ x, int n)
{
    cpw[0] = SMX_DIV(c[0], b[0]);
    dpw[0] = SMX_DIV(d[0], b[0]);
    for (int k = 1; k < n; ++k) {
        real m = SMX_SUB(b[k], SMX_MUL(cpw[k - 1], a[k]));
        cpw[k] = SMX_DIV(c[k], m);
        dpw[k] = SMX_DIV(SMX_SUB(d[k], SMX_MUL(dpw[k - 1], a[k])), m);
    }
    x[n - 1] = dpw[n - 1];
    for (int k = n - 2; k >= 0; --k)
        x[k] = SMX_SUB(dpw[k], SMX_MUL(cpw[k], x[k + 1]));
}

// module_bl_mynn.F:4121-4123 — parameter, not namelist; 1.0 in stock 4.6.1.
// Kept as a multiply so the transcription is line-for-line against the CPU
// reference (an FP32 multiply by 1.0f is exact).
#define SMX_NONLOC 1.0f

// Per-column scratch floats the solve kernel carves from its work buffer:
// khdz (nz+1) + a, b, c, d, cpw, dpw (6*nz).
#define SMX_SOLVE_SCRATCH_FLOATS(nz) ((size_t)(7) * (nz) + 1)

// ===========================================================================
// One stock qn tridiagonal solve + its tendency (module_bl_mynn.F:4654-4689;
// the :4695/:4736/:4778/:4820 blocks are the same arithmetic).  Device twin
// of gpuwm.core.mynn_scalar_mix.mix_scalar_column, except that the
// dtz/rhoinv/khdz/hdz/dzinv inputs the CPU reference consumes are rebuilt
// here from the primitive driver arrays with the exact rounded-op sequence
// mynn_tendencies uses to build them (module_bl_mynn.F:4137-4171, CPU
// reference gpuwm/core/mynn_pbl.py _mynn_tendencies_core) — each op is one
// round-to-nearest FP32 instruction, so the rebuilt arrays are bit-identical
// to the consumed ones by construction, and the probe harness measures it.
// There is NO surface-flux term in any qn RHS, no sd_aw term
// (bl_mynn_edmf_dd=0), the top boundary is the prescribed value
// d(kte)=qn(kte), and the tendency (qn2-qn)/delt has no positivity clamp
// (the clamp lines are commented out in the source).
// One thread owns one column: the tridiagonal recurrence is sequential.
// ===========================================================================
extern "C" __global__
void mynn_mix_scalar_columns(
    const real* __restrict__ qn,      // (ncol, nz)
    const real* __restrict__ dz,      // (ncol, nz)
    const real* __restrict__ rho,     // (ncol, nz)
    const real* __restrict__ dfh,     // (ncol, nz)
    const real* __restrict__ s_aw,    // (ncol, nz+1)
    const real* __restrict__ s_awqn,  // (ncol, nz+1)
    const real* __restrict__ delt_c,  // (ncol,)
    real* __restrict__ qn2,           // (ncol, nz)
    real* __restrict__ dqn,           // (ncol, nz)
    real* __restrict__ scratch,       // (ncol, SMX_SOLVE_SCRATCH_FLOATS(nz))
    int nz, int ncol)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    const size_t base = (size_t)column * nz;
    const size_t ibase = (size_t)column * (nz + 1);
    real* khdz = scratch + (size_t)column * SMX_SOLVE_SCRATCH_FLOATS(nz);
    real* a = khdz + (nz + 1);
    real* b = a + nz;
    real* c = b + nz;
    real* d = c + nz;
    real* cpw = d + nz;
    real* dpw = cpw + nz;
    const real delt = delt_c[column];

    // module_bl_mynn.F:4139-4157 rhoz/khdz, :4163-4169 stability floors —
    // the same arrays the admitted mynn_tendencies solve built (the CPU
    // reference consumes them; this rebuild is op-for-op that construction).
    real rhoz = rho[base];
    khdz[0] = SMX_MUL(rhoz, dfh[base]);
    for (int k = 1; k < nz; ++k) {
        rhoz = SMX_DIV(
            SMX_ADD(SMX_MUL(rho[base + k], dz[base + k - 1]),
                    SMX_MUL(rho[base + k - 1], dz[base + k])),
            SMX_ADD(dz[base + k - 1], dz[base + k]));
        rhoz = smx_max2(rhoz, 1.0e-4f);
        khdz[k] = SMX_MUL(rhoz, dfh[base + k]);
    }
    // rhoz(kte+1)=rhoz(kte): rhoz still holds the top mass-level value.
    khdz[nz] = SMX_MUL(rhoz, dfh[base + nz - 1]);
    for (int k = 1; k < nz - 1; ++k) {
        khdz[k] = smx_max2(khdz[k], SMX_MUL(0.5f, s_aw[ibase + k]));
        khdz[k] = smx_max2(
            khdz[k],
            -SMX_MUL(0.5f, SMX_SUB(s_aw[ibase + k], s_aw[ibase + k + 1])));
    }

    // module_bl_mynn.F:4654-4689 coefficients, hdz/dzinv rebuilt per level
    // (0.5*dtz*rhoinv and dtz*rhoinv, the shared mass-flux prefactors).
    {
        const real dtz0 = SMX_DIV(delt, dz[base]);
        const real rhoinv0 = SMX_DIV(1.0f, rho[base]);
        const real hdz0 = SMX_MUL(SMX_MUL(0.5f, dtz0), rhoinv0);
        const real dzinv0 = SMX_MUL(dtz0, rhoinv0);
        a[0] = -SMX_MUL(SMX_MUL(dtz0, khdz[0]), rhoinv0);
        b[0] = SMX_SUB(
            SMX_ADD(1.0f, SMX_MUL(SMX_MUL(dtz0, SMX_ADD(khdz[1], khdz[0])),
                                  rhoinv0)),
            SMX_MUL(SMX_MUL(hdz0, s_aw[ibase + 1]), SMX_NONLOC));
        c[0] = SMX_SUB(
            -SMX_MUL(SMX_MUL(dtz0, khdz[1]), rhoinv0),
            SMX_MUL(SMX_MUL(hdz0, s_aw[ibase + 1]), SMX_NONLOC));
        d[0] = SMX_SUB(
            qn[base],
            SMX_MUL(SMX_MUL(dzinv0, s_awqn[ibase + 1]), SMX_NONLOC));
    }
    for (int k = 1; k < nz - 1; ++k) {
        const real dtz_k = SMX_DIV(delt, dz[base + k]);
        const real rhoinv_k = SMX_DIV(1.0f, smx_max2(rho[base + k], 1.0e-4f));
        const real hdz_k = SMX_MUL(SMX_MUL(0.5f, dtz_k), rhoinv_k);
        const real dzinv_k = SMX_MUL(dtz_k, rhoinv_k);
        a[k] = SMX_ADD(
            -SMX_MUL(SMX_MUL(dtz_k, khdz[k]), rhoinv_k),
            SMX_MUL(SMX_MUL(hdz_k, s_aw[ibase + k]), SMX_NONLOC));
        b[k] = SMX_ADD(
            SMX_ADD(1.0f,
                    SMX_MUL(SMX_MUL(dtz_k, SMX_ADD(khdz[k], khdz[k + 1])),
                            rhoinv_k)),
            SMX_MUL(SMX_MUL(hdz_k, SMX_SUB(s_aw[ibase + k],
                                           s_aw[ibase + k + 1])),
                    SMX_NONLOC));
        c[k] = SMX_SUB(
            -SMX_MUL(SMX_MUL(dtz_k, khdz[k + 1]), rhoinv_k),
            SMX_MUL(SMX_MUL(hdz_k, s_aw[ibase + k + 1]), SMX_NONLOC));
        d[k] = SMX_ADD(
            qn[base + k],
            SMX_MUL(SMX_MUL(dzinv_k, SMX_SUB(s_awqn[ibase + k],
                                             s_awqn[ibase + k + 1])),
                    SMX_NONLOC));
    }
    a[nz - 1] = 0.0f;
    b[nz - 1] = 1.0f;
    c[nz - 1] = 0.0f;
    d[nz - 1] = qn[base + nz - 1];

    smx_tridiag2_column(a, b, c, d, cpw, dpw, qn2 + base, nz);
    for (int k = 0; k < nz; ++k)
        dqn[base + k] = SMX_DIV(SMX_SUB(qn2[base + k], qn[base + k]), delt);
}

// ===========================================================================
// One species' s_awqn for a column batch — device twin of
// gpuwm.core.mynn_scalar_mix.dmp_qn_flux_column.  Replays ONLY the qn plume
// lines (init :6140-6144, entrain :6213-6217, store :6351-6355, accumulate
// :6447-6456 under scalar_opt>0, limiter rescale :6485-6489) against the
// plume-edge terms the admitted DMP_mf already produced.  up_a MUST be the
// PRE-limiter plume area (:6497 scales UPA only after every s_aw* line).
// plume_active is WRF's NUP2 > 0: when the plume model bailed out the whole
// :6404-6461 block is skipped and every s_awqn stays exactly zero — the gate
// cannot be inferred from up_w alone.  One thread owns one column: the
// entrainment update is a sequential upward recurrence and the FP32
// addition order into each interface accumulator is WRF's k-outer/i-inner.
// ===========================================================================
extern "C" __global__
void mynn_dmp_qn_flux_columns(
    const real* __restrict__ qn,        // (ncol, nz)
    const real* __restrict__ dz,        // (ncol, nz)
    const real* __restrict__ zw,        // (ncol, nz+1)
    const real* __restrict__ up_w,      // (ncol, (nz+1)*nup) k-major
    const real* __restrict__ up_a,      // (ncol, (nz+1)*nup) PRE-limiter
    const real* __restrict__ ent,       // (ncol, nz*nup) k-major
    const real* __restrict__ rhoz,      // (ncol, nz)
    const real* __restrict__ psig_w,    // (ncol,)
    const int*  __restrict__ plume_active,        // (ncol,) 0/1
    const real* __restrict__ limiter_adjustment,  // (ncol,)
    real* __restrict__ s_awqn,          // (ncol, nz+1)
    real* __restrict__ up_qn_scratch,   // (ncol, (nz+1)*nup) k-major
    int nz, int nup, int ncol)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    const size_t base = (size_t)column * nz;
    const size_t ibase = (size_t)column * (nz + 1);
    const size_t pbase = (size_t)column * (size_t)(nz + 1) * nup;
    const size_t ebase = (size_t)column * (size_t)nz * nup;
    real* up_qn = up_qn_scratch + pbase;

    for (int k = 0; k <= nz; ++k) s_awqn[ibase + k] = 0.0f;
    if (plume_active[column] == 0) return;

    for (int s = 0; s < (nz + 1) * nup; ++s) up_qn[s] = 0.0f;
    for (int i = 0; i < nup; ++i) {
        // :6140-6144 surface plume value (interface interpolation).
        up_qn[i] = SMX_DIV(
            SMX_ADD(SMX_MUL(qn[base], dz[base + 1]),
                    SMX_MUL(qn[base + 1], dz[base])),
            SMX_ADD(dz[base], dz[base + 1]));
        for (int k = 1; k < nz - 1; ++k) {
            if (smx_gt(up_w[pbase + (size_t)k * nup + i], 0.0f)) {
                // :6213 EntExp, :6217 QNn, :6351-6355 store.
                real ent_exp = SMX_MUL(
                    ent[ebase + (size_t)k * nup + i],
                    SMX_SUB(zw[ibase + k + 1], zw[ibase + k]));
                up_qn[(size_t)k * nup + i] = SMX_ADD(
                    SMX_MUL(up_qn[(size_t)(k - 1) * nup + i],
                            SMX_SUB(1.0f, ent_exp)),
                    SMX_MUL(qn[base + k], ent_exp));
            }
            // else: the Fortran loop broke before storing; up_qn stays 0.
        }
    }

    // :6447-6456 — k outer, plume inner; the extra top term the Fortran
    // do k=kts,kte loop carries is an exact +0.0 (up_w(kte+1,:) unwritten).
    const real psig = psig_w[column];
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nup; ++i)
            s_awqn[ibase + k + 1] = SMX_ADD(
                s_awqn[ibase + k + 1],
                SMX_MUL(
                    SMX_MUL(
                        SMX_MUL(
                            SMX_MUL(rhoz[base + k],
                                    up_a[pbase + (size_t)k * nup + i]),
                            up_w[pbase + (size_t)k * nup + i]),
                        up_qn[(size_t)k * nup + i]),
                    psig));

    // :6485-6489 — the heat-flux limiter scales every s_awqn* alongside the
    // admitted s_aw* lines; factor exactly 1.0 when it did not fire.
    const real adj = limiter_adjustment[column];
    for (int k = 0; k <= nz; ++k)
        s_awqn[ibase + k] = SMX_MUL(s_awqn[ibase + k], adj);
}
