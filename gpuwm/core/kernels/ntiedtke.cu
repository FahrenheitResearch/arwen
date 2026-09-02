/* New Tiedtke cumulus (cu_physics = 16), one CUDA thread per mass column.
 *
 * Mirror authority: gpuwm.verify.ntiedtke_ref.np_ntiedtke_prep.
 * WRF v4.6.1 transcription anchors:
 *   phys/module_cu_ntiedtke.F
 *     391-455   cu_ntiedtke_pre_run: delt, slimsk, the interface-height
 *               accumulation, omega, and THE VERTICAL FLIP
 *   phys/physics_mmm/cu_ntiedtke.F90
 *     228-239   the scale-dependency factors scale_fac / scale_fac2
 *
 * STAGE: PREP ONLY.  cumastrn and everything under it is not here yet.  The
 * prep is graded first on purpose -- see the flip note below.
 *
 * ==========================================================================
 * THE VERTICAL IS INVERTED, and this is the only cumulus kernel where it is
 * ==========================================================================
 * WRF hands the driver a bottom-up column: k = kts is the surface.
 * cu_ntiedtke_pre_run reverses it, so cu_ntiedtke_run and everything below
 * it run TOP-DOWN -- k = 0 is the model top, k = nz-1 is the surface.  That
 * is the ECMWF convention that the whole Tiedtke line inherits.
 *
 * gf.cu and kf.cu are both bottom-up.  So this is the one structural
 * inversion in the port, and its failure mode is SILENT: an upside-down
 * column produces finite, plausible, entirely wrong numbers rather than a
 * crash or a NaN.  It is graded before any physics for exactly that reason
 * (tests/test_ntiedtke_prep_parity.py, max_ulp == 0 against
 * gpuwm/data/ntiedtke/oracle/nt-prep-levels.csv).
 *
 * ==========================================================================
 * FP CONTRACTION IS PINNED FROM THE FIRST LINE, NOT RETROFITTED
 * ==========================================================================
 * __fmaf_rn / __fmul_rn / __fadd_rn are NVIDIA-guaranteed never merged, so
 * pinning to them costs nothing.  ptxas otherwise contracts by local
 * register pressure, which means a RUNTIME BRANCH can leave two clones of
 * the same arithmetic rounding differently -- that has already failed a
 * bitwise gate once in this tree, and the fix was a branch-free shape.
 *
 * New Tiedtke is branchy by construction: ktype selects between three
 * closures INSIDE the column, and scale_fac applies on one arm while
 * scale_fac2 applies on another.  So every arithmetic expression in this
 * file is spelled in explicit rounded intrinsics from the outset, including
 * the ones that look too simple to matter.  `dot` is the live example: it is
 * a left-to-right chain of three multiplies over one add, and letting ptxas
 * fuse any pair of them moves bits.
 *
 * DIVISION: gfortran's `/` and CUDA's default -prec-div=true are both
 * correctly rounded, and __fdiv_rn is that operation named.  sqrtf likewise
 * (-prec-sqrt=true), which is why scale_fac2 = scale_fac**0.5 needs no
 * glibc mirror even though it is a real exponent -- gfortran folds the 0.5
 * case to sqrt and both sides are correctly rounded.
 *
 * LOGARITHM: scale_fac reads log(dxref/dx), and glibc's logf is NOT CUDA's.
 * gfk_log (glibc_flt32.cuh, transcribed from glibc 2.39
 * sysdeps/ieee754/flt-32/e_logf.c) is the reference's own function.  Checked
 * 2026-08-28: e_logf.c and e_logf_data.c are identical between glibc 2.35
 * and 2.39 apart from their copyright line, so the oracle built on this
 * box's 2.35 and this kernel's 2.39 transcription agree by construction.
 *
 * ==========================================================================
 * LAYOUT
 * ==========================================================================
 * Every column array is LEVEL-MAJOR, a[k * ncol + i], so the 32 threads of a
 * warp touch 32 consecutive floats at each level.  The per-thread local
 * frame is zero -- nothing here is a function-scope column array -- which
 * the compile-only probe confirms.  When cumastrn lands it brings 79 column
 * arrays that DO need somewhere to live, and that somewhere is a global
 * workspace in the kf.cu shape, not the stack: MEASURED on node-1 (RTX 5070
 * Ti, 70 SMs x 1,536, sm_120), those 79 as function-scope locals compile to
 * a 15,496 B frame at nz = 49 and reserve 1,483.9 MiB, against 0 B and
 * 0.0 MiB in the workspace shape.
 */

/* ==========================================================================
 * THE LAUNCH-GEOMETRY DESCRIPTOR, CHECKED ON THE DEVICE
 * ==========================================================================
 * The workspace is per-block and lane-interleaved: element k of slot s for
 * lane t lives at block_base + (s*nz + k)*LANES + t.  A column's slots
 * therefore belong to one (block, lane) pair FOR THE WHOLE STAGE SEQUENCE.
 *
 * If one stage were launched on a different geometry -- a different
 * threads-per-block, a different grid, a different column ordering -- that
 * column would resume on a different lane and read ANOTHER COLUMN's state.
 * Nothing would crash.  Every number would stay finite.  It is the worst
 * failure mode in this port.
 *
 * It is also the one guarantee with no analogue in the Fortran, so no
 * comparison against the reference can catch it, and this project's culture
 * is kernel performance work: cutypen is at 91 registers and someone will
 * want to re-tile it for occupancy.
 *
 * So the geometry is not documentation.  Every kernel takes the descriptor
 * it was promised, REPORTS the geometry it actually observed, and refuses
 * to compute when the two disagree -- which turns a silent cross-column
 * read into a loud parity failure.  gpuwm/core/ntiedtke.py owns the one
 * descriptor and is the only thing that launches these; a stage cannot be
 * re-tiled alone without changing the descriptor every stage reads.
 */
#define NT_STAGE_PREP     0
#define NT_STAGE_CONVERT  1
#define NT_STAGE_CUININ   2
#define NT_STAGE_CUTYPEN  3
#define NT_STAGE_MIDLEVEL 4
#define NT_STAGE_MFUB     5
#define NT_STAGE_CLOSURE  6
#define NT_STAGE_CUASCN   7
#define NT_STAGE_CUDTDQN  8
#define NT_STAGE_CUDUDVN  9
#define NT_STAGE_CUDLFSN  10
#define NT_STAGE_CUDDRAFN 11
#define NT_STAGE_CUFLXN   12
#define NT_STAGE_DEPTH    13
#define NT_STAGE_ADJUST   14
#define NT_STAGE_MRESCALE 15
#define NT_STAGE_USCALE   16
#define NT_STAGE_MPROFILE 17
#define NT_STAGE_KEDIS    18
#define NT_STAGE_POSTRUN  19
#define NT_STAGE_POSTCONV 20
#define NT_STAGE_COUNT    21

__device__ __forceinline__ bool nt_geometry_ok(
        int expect_tpb, int expect_nblocks, int stage_id,
        int *__restrict__ geom_report) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        /* what this launch ACTUALLY had, not what it was told */
        geom_report[stage_id] = (int)((gridDim.x << 16) | (blockDim.x & 0xffff));
    }
    return (int)blockDim.x == expect_tpb && (int)gridDim.x == expect_nblocks;
}

#ifndef NT_DXREF
/* GUARANTEE 4, made device-side for the same reason guarantee 6 was.
 *
 * "Launch order is the Fortran call order, on one stream" was the only
 * continuity guarantee in docs/ntiedtke/PORT-RECORD.md section 7 that outlived the
 * workspace -- thirteen kernels threading column arrays through
 * caller-allocated buffers share state exactly as hard as a lane-interleaved
 * workspace would have -- and it was still prose.
 *
 * geom_report records THAT a stage ran and UNDER WHAT TILE.  Nothing
 * recorded WHEN.  An out-of-order launch (cuflxn before the closure,
 * cududvn before cuflxn's ktopm2 overwrite, cuddrafn before cudlfsn) reads
 * a predecessor's array before that predecessor wrote it: nothing crashes,
 * every number stays finite, and a parity suite that launches the sequence
 * it was written with would not see it.
 *
 * One ticket per LAUNCH, drawn by block 0 thread 0 -- which always exists
 * and, being column 0, survives the `i >= ncol` guard.  It is taken BEFORE
 * any early return, so a stage that no column needs still records that it
 * ran and where in the order.
 */
__device__ __forceinline__ void nt_stage_ticket(
        int stage, int *__restrict__ order_report,
        int *__restrict__ ticket) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        order_report[stage] = atomicAdd(ticket, 1);
}

#define NT_DXREF 15000.0f
#endif

/* cu_ntiedtke_common:18-24 -- the Tetens/mixed-phase constants.  Parameters
 * in the reference, so they are literals here; the DERIVED ones (c2es,
 * r5alvcp, ...) come from cu_ntiedtke_init and are built per launch in
 * nt_init below, because they depend on the caller's cp/rd/rv/xlv/xlf. */
#define NT_TMELT 273.16f
#define NT_RTWAT NT_TMELT
#define NT_RTICE (NT_TMELT - 23.0f)
#define NT_C1ES  610.78f
#define NT_C3LES 17.2693882f
#define NT_C3IES 21.875f
#define NT_C4LES 35.86f
#define NT_C4IES 7.66f

/* cu_ntiedtke.F90:230-238.  The comparison is `<`, not `<=`, so dx == dxref
 * takes the ELSE arm: scale_fac(15000) = 1.1995 where the limit from below
 * is 1.06133^3 = 1.1956.  The factor is DISCONTINUOUS at the join, and the
 * coarse branch is INCREASING in dx, so it is not monotonic either (27 km
 * damps more than 15 km).  Both are transcribed, not smoothed. */
__device__ __forceinline__ void nt_scale_factors(float dx, float *sf,
                                                 float *sf2) {
    if (dx < NT_DXREF) {
        float ratio = __fdiv_rn(NT_DXREF, dx);
        float base = __fadd_rn(1.06133f, gfk_log(ratio));
        /* **3 on an INTEGER exponent: gfortran emits a multiply chain, not
         * powf.  Left to right, so (base*base)*base. */
        *sf = __fmul_rn(__fmul_rn(base, base), base);
        *sf2 = sqrtf(*sf);
    } else {
        *sf = __fadd_rn(1.0f, __fmul_rn(1.33e-5f, dx));
        *sf2 = 1.0f;
    }
}

/* module_cu_ntiedtke.F:391-455.
 *
 * Inputs are WRF order (index 0 = surface); outputs are scheme order
 * (index 0 = model top).  Half-level arrays are (nz, ncol); p8w and w are
 * (nz+1, ncol).
 */
extern "C" __global__ void ntiedtke_prep(
        const float *__restrict__ t3d,     // (nz, ncol) WRF order
        const float *__restrict__ qv3d,
        const float *__restrict__ qc3d,
        const float *__restrict__ qi3d,
        const float *__restrict__ u3d,
        const float *__restrict__ v3d,
        const float *__restrict__ pcps,
        const float *__restrict__ dz8w,
        const float *__restrict__ rho3d,
        const float *__restrict__ p8w,     // (nz+1, ncol)
        const float *__restrict__ w,       // (nz+1, ncol)
        const float *__restrict__ qvften,
        const float *__restrict__ thften,
        const float *__restrict__ xland,   // (ncol)
        const float *__restrict__ hfx,
        const float *__restrict__ qfx,
        const float *__restrict__ dx,
        float *__restrict__ prsl,          // (nz, ncol) scheme order
        float *__restrict__ ghtl,
        float *__restrict__ omg,
        float *__restrict__ tf,
        float *__restrict__ qvf,
        float *__restrict__ qcf,
        float *__restrict__ qif,
        float *__restrict__ uf,
        float *__restrict__ vf,
        float *__restrict__ qvftenz,
        float *__restrict__ thftenz,
        float *__restrict__ prsi,          // (nz+1, ncol)
        float *__restrict__ ghti,
        int *__restrict__ slimsk,          // (ncol)
        float *__restrict__ scale_fac,
        float *__restrict__ scale_fac2,
        float *__restrict__ delt_out,
        int ncol, int nz, float dt, int stepcu, int itimestep, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_PREP,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_PREP, order_report, ticket);

#define NT_IN(a, k)  (a)[(size_t)(k) * (size_t)ncol + (size_t)i]
#define NT_OUT(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    /* :396.  delt = dt*stepcu. */
    const float delt = __fmul_rn(dt, (float)stepcu);
    delt_out[i] = delt;

    /* :400-402.  slimsk = (abs(xland-2.)) -- a REAL expression assigned to
     * an INTEGER dummy, so it TRUNCATES toward zero rather than rounding.
     * xland = 2 (water) -> 0, xland = 1 (land) -> 1. */
    slimsk[i] = (int)fabsf(__fadd_rn(xland[i], -2.0f));

    nt_scale_factors(dx[i], &scale_fac[i], &scale_fac2[i]);

    /* :404-411 and the flip at :419-431 fused.
     *
     * zi[0] = 0; zi[k+1] = zi[k] + dz[k], accumulated upward, so every
     * interface carries the rounding of every layer beneath it -- the
     * accumulation ORDER is part of the answer and cannot be reassociated.
     *
     * ghti is zi reversed: ghti[nz - m] = zi[m].  Writing straight into the
     * reversed slot avoids materialising zi as a column array, which is what
     * keeps this kernel's local frame at zero.
     *
     * zl[k] = 0.5*(zi[k] + zi[k+1]) needs both ends of the layer, so the
     * previous accumulator is carried in a register.
     *
     * dot[k] = -0.5*grav*rho[k]*(w[k] + w[k+1]).  Fortran associates this
     * left to right over the parenthesised add: ((0.5*grav)*rho)*(w+w),
     * negated.  Three separate rounded multiplies -- NOT an FMA, and not
     * regrouped as 0.5*(grav*rho).
     */
    float zi_lo = 0.0f;
    NT_OUT(ghti, nz) = 0.0f;                   /* zi[0] -> ghti[nz] */
    const float half_g = __fmul_rn(0.5f, grav);

    for (int k = 0; k < nz; ++k) {
        const float zi_hi = __fadd_rn(zi_lo, NT_IN(dz8w, k));
        NT_OUT(ghti, nz - 1 - k) = zi_hi;      /* zi[k+1] -> ghti[nz-k-1] */

        const float zl = __fmul_rn(0.5f, __fadd_rn(zi_lo, zi_hi));
        const float wsum = __fadd_rn(NT_IN(w, k), NT_IN(w, k + 1));
        const float dot = -__fmul_rn(__fmul_rn(half_g, NT_IN(rho3d, k)),
                                     wsum);

        /* Half-level fields reverse over nz: out[nz-1-k] = in[k]. */
        const int o = nz - 1 - k;
        NT_OUT(ghtl, o) = zl;
        NT_OUT(omg, o)  = dot;
        NT_OUT(prsl, o) = NT_IN(pcps, k);
        NT_OUT(tf, o)   = NT_IN(t3d, k);
        NT_OUT(qvf, o)  = NT_IN(qv3d, k);
        NT_OUT(qcf, o)  = NT_IN(qc3d, k);
        NT_OUT(qif, o)  = NT_IN(qi3d, k);
        NT_OUT(uf, o)   = NT_IN(u3d, k);
        NT_OUT(vf, o)   = NT_IN(v3d, k);

        /* :449-462.  The forcing tendencies are ZEROED on the first
         * timestep rather than flipped.  Not a nicety: the nonequil closure
         * builds zcape2 entirely out of ptte/pqte, so itimestep == 1 loses
         * that term and the deep closure answers differently. */
        if (itimestep == 1) {
            NT_OUT(qvftenz, o) = 0.0f;
            NT_OUT(thftenz, o) = 0.0f;
        } else {
            NT_OUT(qvftenz, o) = NT_IN(qvften, k);
            NT_OUT(thftenz, o) = NT_IN(thften, k);
        }

        zi_lo = zi_hi;
    }

    /* Interface pressure reverses over nz+1. */
    for (int k = 0; k <= nz; ++k) {
        NT_OUT(prsi, nz - k) = NT_IN(p8w, k);
    }

#undef NT_IN
#undef NT_OUT
}

/* ==========================================================================
 * Slice 2: the conversion block, cuadjtqn (kcall = 0) and cuinin
 * ==========================================================================
 * cu_ntiedtke.F90:3542-3589 (the foe* functions), :3381-3398 (cuadjtqn's
 * kcall = 0 arm) and :1141-1215 (cuinin).
 *
 * cuinin is COLUMN-UNIVERSAL -- no ldcum, no ktype, no ierr, and its only
 * flag is loflag = .true. set unconditionally.  It runs at cumastrn:474,
 * BEFORE cutypen:490 decides the convection type, so it is graded against
 * the plain 108-column fixture with no trigger visibility.  cuadjtqn's
 * kcall = 0 arm does not even read ldflag.
 *
 * exp() here is gfk_exp (glibc_flt32.cuh), NOT CUDA's expf.  The reference
 * calls glibc's expf and the two disagree at the ULP level; the NumPy
 * mirror has to MODEL glibc by evaluating in double and rounding once,
 * while this kernel runs glibc's own algorithm and needs no model.
 *
 * All three of cuinin's working arrays (ptenh, pqenh, pqsenh) are also its
 * OUTPUTS, so they are written straight into the caller's global arrays and
 * this kernel holds no per-thread column array at all -- frame stays 0 B.
 */

/* What cu_ntiedtke_init (:100-118) derives from the caller's constants.
 * Built per thread because it is eight multiplies over kernel arguments and
 * materialising it costs less than a constant-memory round trip. */
struct NtConst {
    float cpd, rcpd, c2es, vtmpc1, r5alvcp, r5alscp, ralvdcp, ralsdcp, g;
    float rd, zrg, ralfdcp, c5les, c5ies, alv;
    /* als and alf are needed by cudtdqn (alf*plglac) and by foelhm. */
    float als, alf;
};

__device__ __forceinline__ NtConst nt_init(float cp, float rd, float rv,
                                           float xlv, float xlf, float g) {
    NtConst c;
    const float als = __fadd_rn(xlv, xlf);
    c.cpd  = cp;
    c.g    = g;
    c.rcpd = __fdiv_rn(1.0f, cp);
    c.c2es = __fdiv_rn(__fmul_rn(NT_C1ES, rd), rv);
    const float c5les = __fmul_rn(NT_C3LES, __fadd_rn(NT_TMELT, -NT_C4LES));
    const float c5ies = __fmul_rn(NT_C3IES, __fadd_rn(NT_TMELT, -NT_C4IES));
    c.r5alvcp = __fmul_rn(__fmul_rn(c5les, xlv), c.rcpd);
    c.r5alscp = __fmul_rn(__fmul_rn(c5ies, als), c.rcpd);
    c.ralvdcp = __fmul_rn(xlv, c.rcpd);
    c.ralsdcp = __fmul_rn(als, c.rcpd);
    c.vtmpc1  = __fadd_rn(__fdiv_rn(rv, rd), -1.0f);
    c.rd      = rd;
    c.zrg     = __fdiv_rn(1.0f, g);
    c.ralfdcp = __fmul_rn(xlf, c.rcpd);
    c.c5les   = c5les;
    c.c5ies   = c5ies;
    c.alv     = xlv;
    c.als     = als;
    c.alf     = xlf;
    return c;
}

/* :3542-3556.  1 over water, 0 over ice, quadratic across the ramp. */
__device__ __forceinline__ float nt_foealfa(float tt) {
    const float clamped = fmaxf(NT_RTICE, fminf(NT_RTWAT, tt));
    const float r = __fdiv_rn(__fadd_rn(clamped, -NT_RTICE),
                              __fadd_rn(NT_RTWAT, -NT_RTICE));
    return fminf(1.0f, __fmul_rn(r, r));
}

/* :3562.  The latent heat blend, alv/als by foealfa. */
__device__ __forceinline__ float nt_foelhm(float tt, const NtConst &c) {
    const float a = nt_foealfa(tt);
    return __fadd_rn(__fmul_rn(a, c.alv),
                     __fmul_rn(__fadd_rn(1.0f, -a), c.als));
}


/* :3566-3573. */
__device__ __forceinline__ float nt_foeewm(float tt, const NtConst &c) {
    const float a = nt_foealfa(tt);
    const float el = gfk_exp(__fdiv_rn(
        __fmul_rn(NT_C3LES, __fadd_rn(tt, -NT_TMELT)),
        __fadd_rn(tt, -NT_C4LES)));
    const float ei = gfk_exp(__fdiv_rn(
        __fmul_rn(NT_C3IES, __fadd_rn(tt, -NT_TMELT)),
        __fadd_rn(tt, -NT_C4IES)));
    return __fmul_rn(c.c2es, __fadd_rn(__fmul_rn(a, el),
                                       __fmul_rn(__fadd_rn(1.0f, -a), ei)));
}

/* :3576-3580. */
__device__ __forceinline__ float nt_foedem(float tt, const NtConst &c) {
    const float a = nt_foealfa(tt);
    const float dl = __fadd_rn(tt, -NT_C4LES);
    const float di = __fadd_rn(tt, -NT_C4IES);
    return __fadd_rn(
        __fmul_rn(__fmul_rn(a, c.r5alvcp),
                  __fdiv_rn(1.0f, __fmul_rn(dl, dl))),
        __fmul_rn(__fmul_rn(__fadd_rn(1.0f, -a), c.r5alscp),
                  __fdiv_rn(1.0f, __fmul_rn(di, di))));
}

/* :3583-3588. */
__device__ __forceinline__ float nt_foeldcpm(float tt, const NtConst &c) {
    const float a = nt_foealfa(tt);
    return __fadd_rn(__fmul_rn(a, c.ralvdcp),
                     __fmul_rn(__fadd_rn(1.0f, -a), c.ralsdcp));
}

/* cu_ntiedtke_run's variable conversion (:240-277).  Already TOP-DOWN --
 * this runs after the flip.  pgeoh's km1 entry is written in the scalar
 * loop BEFORE the level loop, because nothing else writes it. */
extern "C" __global__ void ntiedtke_convert(
        const float *__restrict__ tf,      // (nz, ncol) scheme order
        const float *__restrict__ qvf,
        const float *__restrict__ uf,
        const float *__restrict__ vf,
        const float *__restrict__ omg,
        const float *__restrict__ ghtl,
        const float *__restrict__ ghti,    // (nz+1, ncol)
        const float *__restrict__ prsl,
        const float *__restrict__ qvftenz,
        const float *__restrict__ thftenz,
        float *__restrict__ ztp1, float *__restrict__ zqp1,
        float *__restrict__ zqsat, float *__restrict__ pgeo,
        float *__restrict__ pgeoh,         // (nz+1, ncol)
        float *__restrict__ pum1, float *__restrict__ pvm1,
        float *__restrict__ pverv, float *__restrict__ ptte,
        float *__restrict__ pqte,
        int ncol, int nz,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CONVERT,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CONVERT, order_report, ticket);
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define A(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    A(pgeoh, nz) = __fmul_rn(c.g, A(ghti, nz));
    for (int k = 0; k < nz; ++k) {
        const float q = A(qvf, k);
        A(zqp1, k) = __fdiv_rn(q, __fadd_rn(1.0f, q));
        A(pgeo, k)  = __fmul_rn(c.g, A(ghtl, k));
        A(pgeoh, k) = __fmul_rn(c.g, A(ghti, k));
        const float t = A(tf, k);
        A(ztp1, k) = t;
        float zqs = __fdiv_rn(nt_foeewm(t, c), A(prsl, k));
        zqs = fminf(0.5f, zqs);
        const float zcor = __fdiv_rn(
            1.0f, __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zqs)));
        A(zqsat, k) = __fmul_rn(zqs, zcor);
        A(pum1, k)  = A(uf, k);
        A(pvm1, k)  = A(vf, k);
        A(pverv, k) = A(omg, k);
        A(ptte, k)  = A(thftenz, k);
        A(pqte, k)  = A(qvftenz, k);
    }
#undef A
}

/* cuadjtqn's kcall == 0 arm (:3381-3398), in place at one level.
 * Two identical Newton passes.  Never reads ldflag. */
__device__ __forceinline__ void nt_cuadjtqn0(
        float *pt, float *pq, float psp, const NtConst &c) {
    const float zqp = __fdiv_rn(1.0f, psp);
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        float zqsat = __fmul_rn(nt_foeewm(*pt, c), zqp);
        zqsat = fminf(0.5f, zqsat);
        const float zcor = __fdiv_rn(
            1.0f, __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zqsat)));
        zqsat = __fmul_rn(zqsat, zcor);
        const float den = __fadd_rn(
            1.0f, __fmul_rn(__fmul_rn(zqsat, zcor), nt_foedem(*pt, c)));
        const float zcond1 = __fdiv_rn(__fadd_rn(*pq, -zqsat), den);
        *pt = __fadd_rn(*pt, __fmul_rn(nt_foeldcpm(*pt, c), zcond1));
        *pq = __fadd_rn(*pq, -zcond1);
    }
}

/* cuinin (:1141-1215).  Fortran is 1-based and the index arithmetic is
 * load-bearing, so every loop below carries its Fortran form.
 *
 * pqsenh[0] IS NEVER WRITTEN by the reference: the jk loop starts at 2 and
 * the tail block sets only ptenh(1) and pqenh(1).  It is undefined in WRF
 * too (a cumastrn local nothing downstream reads), so it is neither
 * invented here nor graded. */
extern "C" __global__ void ntiedtke_cuinin(
        const float *__restrict__ pten,    // (nz, ncol)
        const float *__restrict__ pqen,
        const float *__restrict__ pqsen,
        const float *__restrict__ puen,
        const float *__restrict__ pven,
        const float *__restrict__ pverv,
        const float *__restrict__ pgeo,
        const float *__restrict__ paph,    // (nz+1, ncol)
        const float *__restrict__ pgeoh,   // (nz+1, ncol)
        float *__restrict__ ptenh, float *__restrict__ pqenh,
        float *__restrict__ pqsenh,
        float *__restrict__ ptu, float *__restrict__ pqu,
        float *__restrict__ ptd, float *__restrict__ pqd,
        float *__restrict__ puu, float *__restrict__ pvu,
        float *__restrict__ pud, float *__restrict__ pvd,
        float *__restrict__ plu,
        int *__restrict__ klab, int *__restrict__ klwmin,
        int ncol, int nz,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUININ,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUININ, order_report, ticket);
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define A(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    A(ptenh, 0) = 0.0f;
    A(pqenh, 0) = 0.0f;
    A(pqsenh, 0) = 0.0f;

    /* do jk = 2, klev  ->  k = 1 .. nz-1 */
    for (int k = 1; k < nz; ++k) {
        const float a = __fadd_rn(__fmul_rn(c.cpd, A(pten, k - 1)),
                                  A(pgeo, k - 1));
        const float b = __fadd_rn(__fmul_rn(c.cpd, A(pten, k)), A(pgeo, k));
        A(ptenh, k) = __fmul_rn(__fadd_rn(fmaxf(a, b), -A(pgeoh, k)), c.rcpd);
        A(pqenh, k)  = A(pqen, k - 1);
        A(pqsenh, k) = A(pqsen, k - 1);
        /* if ( jk >= klev-1 .or. jk < 2 ) cycle  ->  skip k >= nz-2 */
        if (k >= nz - 2) continue;
        float t = A(ptenh, k), q = A(pqsenh, k);
        nt_cuadjtqn0(&t, &q, A(paph, k), c);
        A(ptenh, k) = t;
        A(pqsenh, k) = q;
        const float v = __fadd_rn(fminf(A(pqen, k - 1), A(pqsen, k - 1)),
                                  __fadd_rn(q, -A(pqsen, k - 1)));
        A(pqenh, k) = fmaxf(v, 0.0f);
    }

    A(ptenh, nz - 1) = __fmul_rn(
        __fadd_rn(__fadd_rn(__fmul_rn(c.cpd, A(pten, nz - 1)),
                            A(pgeo, nz - 1)), -A(pgeoh, nz - 1)), c.rcpd);
    A(pqenh, nz - 1) = A(pqen, nz - 1);
    A(ptenh, 0) = A(pten, 0);
    A(pqenh, 0) = A(pqen, 0);

    int lwmin = nz;                 /* 1-based klev */
    float zwmax = 0.0f;

    /* do jk = klevm1, 2, -1  ->  k = nz-2 .. 1 */
    for (int k = nz - 2; k >= 1; --k) {
        const float z1 = __fadd_rn(__fmul_rn(c.cpd, A(ptenh, k)),
                                   A(pgeoh, k));
        const float z2 = __fadd_rn(__fmul_rn(c.cpd, A(ptenh, k + 1)),
                                   A(pgeoh, k + 1));
        A(ptenh, k) = __fmul_rn(__fadd_rn(fmaxf(z1, z2), -A(pgeoh, k)),
                                c.rcpd);
    }

    /* do jk = klev, 3, -1  ->  k = nz-1 .. 2; klwmin stays 1-BASED */
    for (int k = nz - 1; k >= 2; --k) {
        if (A(pverv, k) < zwmax) {
            zwmax = A(pverv, k);
            lwmin = k + 1;
        }
    }
    klwmin[i] = lwmin;

    for (int k = 0; k < nz; ++k) {
        const int ik = (k == 0) ? 0 : k - 1;   /* ik = jk-1, 1 when jk = 1 */
        A(ptu, k) = A(ptenh, k);
        A(ptd, k) = A(ptenh, k);
        A(pqu, k) = A(pqenh, k);
        A(pqd, k) = A(pqenh, k);
        A(plu, k) = 0.0f;
        A(puu, k) = A(puen, ik);
        A(pud, k) = A(puen, ik);
        A(pvu, k) = A(pven, ik);
        A(pvd, k) = A(pven, ik);
        A(klab, k) = 0;
    }
#undef A
}

/* ==========================================================================
 * Slice 3: cuadjtqn's kcall == 1 arm, and cutypen
 * ==========================================================================
 * cu_ntiedtke.F90:3324-3357 (cuadjtqn kcall == 1) and :1330-1748 (cutypen).
 *
 * cutypen is THE TRIGGER.  It decides ktype, and ktype is what selects
 * which scale factor applies downstream -- :676 scale_fac for deep,
 * :716 scale_fac2 for shallow -- so this routine is where the property
 * this whole port was bought for is actually decided.
 *
 * It assigns ktype 0, 1 and 2 ONLY.  ktype 3 (mid-level) is assigned later
 * in cubasmcn, called from cuascn:1968, and is not this routine's business.
 *
 * ==========================================================================
 * WHY EVERY EXPRESSION HERE IS SPELLED IN ROUNDED INTRINSICS
 * ==========================================================================
 * This is the branchiest routine in the scheme: two full parcel ascents
 * (shallow, then deep over ~23 candidate departure levels), each with an
 * early exit, a cloud-base refinement with two arms, and a reset that
 * rewrites the output arrays.  ptxas contracts by LOCAL REGISTER PRESSURE,
 * so a runtime branch can leave two clones of the same arithmetic rounding
 * differently -- that has already failed a bitwise gate once in this tree
 * and the fix was a branch-free shape.
 *
 * The shallow and deep ascents are near-identical in shape and differ in
 * three places (entrainment, the plu clamp, the departure-level seed).
 * They are written as ONE helper taking a `deep` flag rather than two
 * clones, precisely so ptxas cannot contract the two copies differently:
 * one body, one rounding.
 *
 * SCRATCH: the 11 per-column working arrays live in caller-provided global
 * buffers, not on the stack.  At nz = 49 they would be 2,156 B of frame,
 * over the 1,024 B default stack, and every byte of frame costs 105 KiB of
 * VRAM.  In global scratch the frame stays 0 B.
 */

#define NT_T13     (1.0f / 3.0f)
#define NT_ZDNOPRC 2.0e4f

/* cuadjtqn's kcall == 1 arm (:3324-3357).
 *
 * NOT the kcall == 0 arm with a different guard.  This one computes
 * saturation INLINE off reciprocals -- exp(c3les*(pt-tmelt)*zl) with
 * zl = 1/(pt-c4les) -- where kcall == 0 goes through foeewm and its
 * DIVISION.  A multiply by a reciprocal is not a division in float32, so
 * the two arms round differently on the same argument and neither can
 * stand in for the other.
 *
 * The second Newton pass is guarded twice: it runs only when the first
 * condensate is positive, and its result is discarded when that condensate
 * is denormal-small. */
__device__ __forceinline__ void nt_cuadjtqn1(float *pt, float *pq,
                                             float psp, const NtConst &c) {
    const float zqp = __fdiv_rn(1.0f, psp);
    float zl = __fdiv_rn(1.0f, __fadd_rn(*pt, -NT_C4LES));
    float zi = __fdiv_rn(1.0f, __fadd_rn(*pt, -NT_C4IES));
    float a = nt_foealfa(*pt);
    float el = gfk_exp(__fmul_rn(__fmul_rn(NT_C3LES,
                                           __fadd_rn(*pt, -NT_TMELT)), zl));
    float ei = gfk_exp(__fmul_rn(__fmul_rn(NT_C3IES,
                                           __fadd_rn(*pt, -NT_TMELT)), zi));
    float zqsat = __fmul_rn(c.c2es,
        __fadd_rn(__fmul_rn(a, el), __fmul_rn(__fadd_rn(1.0f, -a), ei)));
    zqsat = __fmul_rn(zqsat, zqp);
    zqsat = fminf(0.5f, zqsat);
    float zcor = __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zqsat));
    float zf = __fadd_rn(
        __fmul_rn(__fmul_rn(a, c.r5alvcp), __fmul_rn(zl, zl)),
        __fmul_rn(__fmul_rn(__fadd_rn(1.0f, -a), c.r5alscp),
                  __fmul_rn(zi, zi)));
    const float zcond = __fdiv_rn(
        __fadd_rn(__fmul_rn(*pq, __fmul_rn(zcor, zcor)),
                  -__fmul_rn(zqsat, zcor)),
        __fadd_rn(__fmul_rn(zcor, zcor), __fmul_rn(zqsat, zf)));
    if (zcond > 0.0f) {
        *pt = __fadd_rn(*pt, __fmul_rn(nt_foeldcpm(*pt, c), zcond));
        *pq = __fadd_rn(*pq, -zcond);
        zl = __fdiv_rn(1.0f, __fadd_rn(*pt, -NT_C4LES));
        zi = __fdiv_rn(1.0f, __fadd_rn(*pt, -NT_C4IES));
        a = nt_foealfa(*pt);
        el = gfk_exp(__fmul_rn(__fmul_rn(NT_C3LES,
                                         __fadd_rn(*pt, -NT_TMELT)), zl));
        ei = gfk_exp(__fmul_rn(__fmul_rn(NT_C3IES,
                                         __fadd_rn(*pt, -NT_TMELT)), zi));
        zqsat = __fmul_rn(c.c2es,
            __fadd_rn(__fmul_rn(a, el), __fmul_rn(__fadd_rn(1.0f, -a), ei)));
        zqsat = __fmul_rn(zqsat, zqp);
        zqsat = fminf(0.5f, zqsat);
        zcor = __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zqsat));
        zf = __fadd_rn(
            __fmul_rn(__fmul_rn(a, c.r5alvcp), __fmul_rn(zl, zl)),
            __fmul_rn(__fmul_rn(__fadd_rn(1.0f, -a), c.r5alscp),
                      __fmul_rn(zi, zi)));
        float zcond1 = __fdiv_rn(
            __fadd_rn(__fmul_rn(*pq, __fmul_rn(zcor, zcor)),
                      -__fmul_rn(zqsat, zcor)),
            __fadd_rn(__fmul_rn(zcor, zcor), __fmul_rn(zqsat, zf)));
        if (fabsf(zcond) < 1.0e-20f) zcond1 = 0.0f;
        *pt = __fadd_rn(*pt, __fmul_rn(nt_foeldcpm(*pt, c), zcond1));
        *pq = __fadd_rn(*pq, -zcond1);
    }
}

/* Everything below indexes 1-BASED, matching the Fortran: arrays are sized
 * nz+2 per column and index 0 is unused.  This routine is dense with
 * jk+1 / jk+2 / klev-1 / levels+1 arithmetic across three loop directions,
 * and translating each of those by hand is where a transcription silently
 * goes wrong.  The conversion happens once, at the kernel boundary. */
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

/* The exact-cloud-base block, identical in both passes (:1453-1463,
 * :1654-1685).  Picks whichever half level the LCL is nearer. */
__device__ __forceinline__ void nt_cloud_base(
        int jk, int klev, int ncol, int i,
        float *ptu, float *pqu, float *plu, float *kup, int *klab,
        const float *paph, int *kcbot, const NtConst &c) {
    const int ik = jk + 1;
    float zqsu = __fdiv_rn(nt_foeewm(NTC(ptu, ik), c), NTC(paph, ik));
    zqsu = fminf(0.5f, zqsu);
    float zcor = __fdiv_rn(1.0f, __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zqsu)));
    zqsu = __fmul_rn(zqsu, zcor);
    const float zdq = fminf(0.0f, __fadd_rn(NTC(pqu, ik), -zqsu));
    const float zalfaw = nt_foealfa(NTC(ptu, ik));
    const float dl = __fadd_rn(NTC(ptu, ik), -NT_C4LES);
    const float di = __fadd_rn(NTC(ptu, ik), -NT_C4IES);
    const float zfacw = __fdiv_rn(c.c5les, __fmul_rn(dl, dl));
    const float zfaci = __fdiv_rn(c.c5ies, __fmul_rn(di, di));
    const float zfac = __fadd_rn(__fmul_rn(zalfaw, zfacw),
                                 __fmul_rn(__fadd_rn(1.0f, -zalfaw), zfaci));
    const float zesdp = __fdiv_rn(nt_foeewm(NTC(ptu, ik), c), NTC(paph, ik));
    zcor = __fdiv_rn(1.0f, __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zesdp)));
    const float zdqsdt = __fmul_rn(__fmul_rn(zfac, zcor), zqsu);
    const float zdtdp = __fdiv_rn(__fmul_rn(c.rd, NTC(ptu, ik)),
                                  __fmul_rn(c.cpd, NTC(paph, ik)));
    const float zdp = __fdiv_rn(zdq, __fmul_rn(zdqsdt, zdtdp));
    const float zcbase = __fadd_rn(NTC(paph, ik), zdp);
    const float zpdifftop = __fadd_rn(zcbase, -NTC(paph, jk));
    const float zpdiffbot = __fadd_rn(NTC(paph, jk + 1), -zcbase);
    if (zpdifftop > zpdiffbot && NTC(kup, jk + 1) > 0.0f) {
        const int ikb = min(klev - 1, jk + 1);
        NTC(klab, ikb) = 2;
        NTC(klab, jk) = 2;
        *kcbot = ikb;
        NTC(plu, jk + 1) = 1.0e-8f;
    } else if (zpdifftop <= zpdiffbot && NTC(kup, jk) > 0.0f) {
        NTC(klab, jk) = 2;
        *kcbot = jk;
    }
}

/* ONE ascent body for both passes.  The shallow and deep loops differ in
 * exactly three places and are written as one function taking `deep` so
 * that ptxas sees one body and cannot contract two clones differently --
 * which is the specific hazard this scheme's branchiness creates. */
__device__ __forceinline__ void nt_ascent_step(
        int jk, int levels, int klev, int ncol, int i, bool deep,
        float *ptu, float *pqu, float *plu, float *dh, float *dhen,
        float *kup, float *vptu, float *vten, float *zbuo, float *abuoy,
        int *klab,
        const float *ptenh, const float *pqenh, const float *pgeo,
        const float *pgeoh, const float *paph, const float *pqsen,
        int *kcbot, int *kctop, bool *loflag, bool *lldcum,
        const NtConst &c) {
    float eta;
    if (deep) {
        const float r = __fdiv_rn(NTC(pqsen, jk), NTC(pqsen, levels));
        const float fscale = fminf(1.0f, __fmul_rn(__fmul_rn(r, r), r));
        eta = __fmul_rn(1.75e-3f, fscale);
    } else {
        eta = __fadd_rn(__fdiv_rn(0.8f, __fmul_rn(NTC(pgeo, jk), c.zrg)),
                        2.0e-4f);
    }
    const float dz = __fmul_rn(__fadd_rn(NTC(pgeoh, jk),
                                         -NTC(pgeoh, jk + 1)), c.zrg);
    const float coef = __fmul_rn(__fmul_rn(0.5f, eta), dz);
    NTC(dhen, jk) = __fadd_rn(NTC(pgeoh, jk),
                              __fmul_rn(c.cpd, NTC(ptenh, jk)));
    NTC(dh, jk) = __fdiv_rn(
        __fadd_rn(__fmul_rn(coef, __fadd_rn(NTC(dhen, jk + 1), NTC(dhen, jk))),
                  __fmul_rn(__fadd_rn(1.0f, -coef), NTC(dh, jk + 1))),
        __fadd_rn(1.0f, coef));
    NTC(pqu, jk) = __fdiv_rn(
        __fadd_rn(__fmul_rn(coef, __fadd_rn(NTC(pqenh, jk + 1),
                                            NTC(pqenh, jk))),
                  __fmul_rn(__fadd_rn(1.0f, -coef), NTC(pqu, jk + 1))),
        __fadd_rn(1.0f, coef));
    NTC(ptu, jk) = __fmul_rn(__fadd_rn(NTC(dh, jk), -NTC(pgeoh, jk)), c.rcpd);
    const float zqold = NTC(pqu, jk);

    float t = NTC(ptu, jk), q = NTC(pqu, jk);
    nt_cuadjtqn1(&t, &q, NTC(paph, jk), c);
    NTC(ptu, jk) = t;
    NTC(pqu, jk) = q;

    const float zdq = fmaxf(__fadd_rn(zqold, -NTC(pqu, jk)), 0.0f);
    NTC(plu, jk) = __fadd_rn(NTC(plu, jk + 1), zdq);
    const float zlglac = __fmul_rn(zdq,
        __fadd_rn(__fadd_rn(1.0f, -nt_foealfa(NTC(ptu, jk))),
                  -__fadd_rn(1.0f, -nt_foealfa(NTC(ptu, jk + 1)))));
    NTC(plu, jk) = deep ? __fmul_rn(0.5f, NTC(plu, jk))
                        : fminf(NTC(plu, jk), 5.0e-3f);
    NTC(dh, jk) = __fadd_rn(NTC(pgeoh, jk),
        __fmul_rn(c.cpd, __fadd_rn(NTC(ptu, jk),
                                   __fmul_rn(c.ralfdcp, zlglac))));
    NTC(vptu, jk) = __fadd_rn(
        __fmul_rn(NTC(ptu, jk),
                  __fadd_rn(__fadd_rn(1.0f, __fmul_rn(c.vtmpc1,
                                                      NTC(pqu, jk))),
                            -NTC(plu, jk))),
        __fmul_rn(c.ralfdcp, zlglac));
    NTC(vten, jk) = __fmul_rn(NTC(ptenh, jk),
        __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqenh, jk))));
    NTC(zbuo, jk) = __fdiv_rn(__fadd_rn(NTC(vptu, jk), -NTC(vten, jk)),
                              NTC(vten, jk));
    NTC(abuoy, jk) = __fmul_rn(
        __fmul_rn(__fadd_rn(NTC(zbuo, jk), NTC(zbuo, jk + 1)), 0.5f), c.g);
    const float atop1 = __fadd_rn(1.0f, -__fmul_rn(2.0f, coef));
    const float atop2 = __fmul_rn(__fmul_rn(2.0f, dz), NTC(abuoy, jk));
    const float abot = __fadd_rn(1.0f, __fmul_rn(2.0f, coef));
    NTC(kup, jk) = __fdiv_rn(
        __fadd_rn(__fmul_rn(atop1, NTC(kup, jk + 1)), atop2), abot);

    if (NTC(plu, jk) > 0.0f && NTC(klab, jk + 1) == 1) {
        nt_cloud_base(jk, klev, ncol, i, ptu, pqu, plu, kup, klab,
                      paph, kcbot, c);
    }
    if (NTC(kup, jk) < 0.0f) {
        *loflag = false;
        if (NTC(plu, jk + 1) > 0.0f) {
            *kctop = jk;
            *lldcum = true;
        } else {
            *lldcum = false;
        }
    } else {
        NTC(klab, jk) = (NTC(plu, jk) > 0.0f) ? 2 : 1;
    }
}

extern "C" __global__ void ntiedtke_cutypen(
        const float *__restrict__ pqen,    // (nz+2, ncol), 1-based
        const float *__restrict__ ptenh,
        const float *__restrict__ pqenh,
        const float *__restrict__ pgeoh,
        const float *__restrict__ paph,
        const float *__restrict__ pgeo,
        const float *__restrict__ pqsen,
        const float *__restrict__ pap,
        const float *__restrict__ pten,
        const float *__restrict__ hfx,     // (ncol)
        const float *__restrict__ qfx,
        float *__restrict__ cutu,          // (nz+2, ncol) in AND out
        float *__restrict__ cuqu,
        float *__restrict__ culu,
        int *__restrict__ culab,
        float *__restrict__ scr,           // (10, nz+2, ncol) scratch
        int *__restrict__ scr_i,           // (1, nz+2, ncol) scratch
        int *__restrict__ ldcum_o, int *__restrict__ ktype_o,
        int *__restrict__ cubot_o, int *__restrict__ cutop_o,
        int *__restrict__ kdpl_o, float *__restrict__ wbase_o,
        int ncol, int nz,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUTYPEN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUTYPEN, order_report, ticket);
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
    const int klev = nz, klevm1 = nz - 1;
    const size_t stride = (size_t)(nz + 2) * (size_t)ncol;

    float *ptu = scr + 0 * stride, *pqu = scr + 1 * stride;
    float *plu = scr + 2 * stride, *dh = scr + 3 * stride;
    float *dhen = scr + 4 * stride, *kup = scr + 5 * stride;
    float *vptu = scr + 6 * stride, *vten = scr + 7 * stride;
    float *zbuo = scr + 8 * stride, *abuoy = scr + 9 * stride;
    int *klab = scr_i;

    int kcbot = klev, kctop = klev, kdpl = klev, ktype = 0;
    int cubot = -1, cutop = -1;
    float wbase = 0.0f;
    bool ldcum = false, lldcum = false, loflag = true;

    /* ---- shallow pass (:1332-1345) --------------------------------- */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(plu, jk) = NTC(culu, jk);
        NTC(ptu, jk) = NTC(cutu, jk);
        NTC(pqu, jk) = NTC(cuqu, jk);
        NTC(klab, jk) = NTC(culab, jk);
        NTC(dh, jk) = 0.0f;  NTC(dhen, jk) = 0.0f;  NTC(kup, jk) = 0.0f;
        NTC(vptu, jk) = 0.0f; NTC(vten, jk) = 0.0f;
        NTC(zbuo, jk) = 0.0f; NTC(abuoy, jk) = 0.0f;
    }

    for (int jk = klevm1; jk >= 2; --jk) {
        if (jk == klevm1) {
            const float rho = __fdiv_rn(NTC(pap, klev),
                __fmul_rn(c.rd, __fmul_rn(NTC(pten, klev),
                    __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqen, klev))))));
            const float part1 = __fdiv_rn(
                __fmul_rn(__fmul_rn(1.5f, 0.4f), NTC(pgeo, klev)),
                __fmul_rn(rho, NTC(pten, klev)));
            const float part2 = __fadd_rn(
                -__fmul_rn(hfx[i], c.rcpd),
                -__fmul_rn(__fmul_rn(c.vtmpc1, NTC(pten, klev)), qfx[i]));
            const float root = __fadd_rn(0.001f, -__fmul_rn(part1, part2));
            if (part2 < 0.0f) {
                /* conw = 1.2*root**t13 -- a genuine cube root through powf,
                 * so gfk_pow (glibc's) and not CUDA's. */
                const float conw = __fmul_rn(1.2f, gfk_pow(root, NT_T13));
                const float deltt = fmaxf(__fdiv_rn(
                    __fmul_rn(1.5f, hfx[i]),
                    __fmul_rn(__fmul_rn(rho, c.cpd), conw)), 0.0f);
                const float deltq = fmaxf(__fdiv_rn(
                    __fmul_rn(1.5f, qfx[i]), __fmul_rn(rho, conw)), 0.0f);
                NTC(kup, klev) = __fmul_rn(0.5f, __fmul_rn(conw, conw));
                NTC(pqu, klev) = __fadd_rn(NTC(pqenh, klev), deltq);
                NTC(dhen, klev) = __fadd_rn(NTC(pgeoh, klev),
                                            __fmul_rn(NTC(ptenh, klev),
                                                      c.cpd));
                NTC(dh, klev) = __fadd_rn(NTC(dhen, klev),
                                          __fmul_rn(deltt, c.cpd));
                NTC(ptu, klev) = __fmul_rn(
                    __fadd_rn(NTC(dh, klev), -NTC(pgeoh, klev)), c.rcpd);
                NTC(vptu, klev) = __fmul_rn(NTC(ptu, klev),
                    __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqu, klev))));
                NTC(vten, klev) = __fmul_rn(NTC(ptenh, klev),
                    __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqenh, klev))));
                NTC(zbuo, klev) = __fdiv_rn(
                    __fadd_rn(NTC(vptu, klev), -NTC(vten, klev)),
                    NTC(vten, klev));
                NTC(klab, klev) = 1;
            } else {
                loflag = false;
            }
        }
        if (!loflag) break;                        /* is == 0 -> exit */
        nt_ascent_step(jk, 0, klev, ncol, i, false,
                       ptu, pqu, plu, dh, dhen, kup, vptu, vten, zbuo,
                       abuoy, klab, ptenh, pqenh, pgeo, pgeoh, paph, pqsen,
                       &kcbot, &kctop, &loflag, &lldcum, c);
    }

    if (__fadd_rn(NTC(paph, kcbot), -NTC(paph, kctop)) > NT_ZDNOPRC)
        lldcum = false;
    if (lldcum) {
        ktype = 2;  ldcum = true;
        wbase = sqrtf(fmaxf(__fmul_rn(2.0f, NTC(kup, kcbot)), 0.0f));
        cubot = kcbot;  cutop = kctop;  kdpl = klev;
    } else {
        cutop = -1;  cubot = -1;  kdpl = klev - 1;
        ldcum = false;  wbase = 0.0f;
    }
    for (int jk = klev; jk >= 1; --jk) {
        if (jk >= kctop) {
            NTC(culab, jk) = NTC(klab, jk);
            NTC(cutu, jk) = NTC(ptu, jk);
            NTC(cuqu, jk) = NTC(pqu, jk);
            NTC(culu, jk) = NTC(plu, jk);
        }
    }

    /* ---- deep pass (:1517-1746) ------------------------------------ */
    const float deltt_d = 0.2f, deltq_d = 1.0e-4f;
    bool deepflag = false;
    int itoppacel = klev;
    for (int jk = klev; jk >= 1; --jk) {
        if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 350.0e2f)
            itoppacel = jk;
    }

    for (int levels = klevm1 - 1; levels >= klev / 2 + 1; --levels) {
        for (int jk = 1; jk <= klev; ++jk) {
            NTC(plu, jk) = 0.0f; NTC(ptu, jk) = 0.0f; NTC(pqu, jk) = 0.0f;
            NTC(dh, jk) = 0.0f;  NTC(dhen, jk) = 0.0f; NTC(kup, jk) = 0.0f;
            NTC(vptu, jk) = 0.0f; NTC(vten, jk) = 0.0f;
            NTC(abuoy, jk) = 0.0f; NTC(zbuo, jk) = 0.0f;
            NTC(klab, jk) = 0;
        }
        kcbot = levels;  kctop = levels;
        lldcum = false;
        bool resetflag = false;
        loflag = (!deepflag) && (levels >= itoppacel);

        for (int jk = levels; jk >= 2; --jk) {
            if (!loflag) break;
            if (jk == levels) {
                float tmix, qmix, zmix;
                if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f) {
                    tmix = 0.0f; qmix = 0.0f; zmix = 0.0f;
                    float pmix = 0.0f;
                    for (int nk = jk + 2; nk >= jk; --nk) {
                        if (pmix < 50.0e2f) {
                            const float dp = __fadd_rn(NTC(paph, nk),
                                                       -NTC(paph, nk - 1));
                            tmix = __fadd_rn(tmix,
                                             __fmul_rn(dp, NTC(ptenh, nk)));
                            qmix = __fadd_rn(qmix,
                                             __fmul_rn(dp, NTC(pqenh, nk)));
                            zmix = __fadd_rn(zmix,
                                             __fmul_rn(dp, NTC(pgeoh, nk)));
                            pmix = __fadd_rn(pmix, dp);
                        }
                    }
                    tmix = __fdiv_rn(tmix, pmix);
                    qmix = __fdiv_rn(qmix, pmix);
                    zmix = __fdiv_rn(zmix, pmix);
                } else {
                    tmix = NTC(ptenh, jk + 1);
                    qmix = NTC(pqenh, jk + 1);
                    zmix = NTC(pgeoh, jk + 1);
                }
                NTC(pqu, jk + 1) = __fadd_rn(qmix, deltq_d);
                NTC(dhen, jk + 1) = __fadd_rn(zmix, __fmul_rn(tmix, c.cpd));
                NTC(dh, jk + 1) = __fadd_rn(NTC(dhen, jk + 1),
                                            __fmul_rn(deltt_d, c.cpd));
                NTC(ptu, jk + 1) = __fmul_rn(
                    __fadd_rn(NTC(dh, jk + 1), -NTC(pgeoh, jk + 1)), c.rcpd);
                NTC(kup, jk + 1) = 0.5f;
                NTC(klab, jk + 1) = 1;
                NTC(vptu, jk + 1) = __fmul_rn(NTC(ptu, jk + 1),
                    __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqu, jk + 1))));
                NTC(vten, jk + 1) = __fmul_rn(NTC(ptenh, jk + 1),
                    __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqenh, jk + 1))));
                NTC(zbuo, jk + 1) = __fdiv_rn(
                    __fadd_rn(NTC(vptu, jk + 1), -NTC(vten, jk + 1)),
                    NTC(vten, jk + 1));
            }
            nt_ascent_step(jk, levels, klev, ncol, i, true,
                           ptu, pqu, plu, dh, dhen, kup, vptu, vten, zbuo,
                           abuoy, klab, ptenh, pqenh, pgeo, pgeoh, paph,
                           pqsen, &kcbot, &kctop, &loflag, &lldcum, c);
        }

        if (__fadd_rn(NTC(paph, kcbot), -NTC(paph, kctop)) < NT_ZDNOPRC)
            lldcum = false;
        if (lldcum) {
            ktype = 1;  ldcum = true;  deepflag = true;
            wbase = sqrtf(fmaxf(__fmul_rn(2.0f, NTC(kup, kcbot)), 0.0f));
            cubot = kcbot;  cutop = kctop;
            kdpl = levels + 1;
            resetflag = true;
        }
        if (resetflag) {
            const int ikt = kctop, ikb = kdpl;
            for (int jk = klev; jk >= 1; --jk) {
                if (jk >= ikt && jk <= ikb) {
                    NTC(culab, jk) = NTC(klab, jk);
                    NTC(cutu, jk) = NTC(ptu, jk);
                    NTC(cuqu, jk) = NTC(pqu, jk);
                    NTC(culu, jk) = NTC(plu, jk);
                } else {
                    NTC(culab, jk) = 1;
                    NTC(cutu, jk) = NTC(ptenh, jk);
                    NTC(cuqu, jk) = NTC(pqenh, jk);
                    NTC(culu, jk) = 0.0f;
                }
                if (jk < ikt) NTC(culab, jk) = 0;
            }
        }
    }

    ldcum_o[i] = ldcum ? 1 : 0;
    ktype_o[i] = ktype;
    cubot_o[i] = cubot;
    cutop_o[i] = cutop;
    kdpl_o[i] = kdpl;
    wbase_o[i] = wbase;
}

#undef NTC

/* ==========================================================================
 * Slice 4a: cubasmcn and cuentrn
 * ==========================================================================
 * cu_ntiedtke.F90:3457-3482 (cubasmcn) and :3516-3536 (cuentrn).
 *
 * cubasmcn assigns ktype = 3 (:3480) and is therefore where the mid-level
 * arm of the scheme begins.  cuascn calls it once per level, jk = klev-1..3.
 *
 * ==========================================================================
 * THIS KERNEL MUST NOT INITIALISE ITS OUTPUTS
 * ==========================================================================
 * All THIRTEEN of cubasmcn's outputs are written only inside its guard.  A
 * column that does not trigger keeps the CALLER's value in every one of
 * them -- ktype, kcbot, klab, ptu, pqu, plu, pmfu, pmfub, pmfus, pmfuq,
 * pmful, pdmfup and plrain.
 *
 * So the reflex of zeroing outputs at kernel entry, which almost every CUDA
 * kernel does, would diverge from WRF on every non-triggering column.  On
 * this fixture that is 48 of 108 columns for cubasmcn, and the failure would
 * be QUIET: the triggering columns would still match.  Nothing here writes a
 * slot outside the guard, and the parity test covers both sides of it.
 *
 * The same applies to cuentrn, whose entire body sits inside `if (ldwork)`.
 *
 * See gpuwm/data/ntiedtke/oracle/nt-aliasing-audit.txt and section 7 of
 * docs/ntiedtke/PORT-RECORD.md for the contract this is an instance of.
 */

/* NTC is #undef'd at the end of the cutypen section; this block uses the
 * same 1-based, level-major addressing, so it is redefined rather than
 * left to leak across sections. */
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

#define NT_CMFCMIN 1.0e-10f
#define NT_CMFCMAX 1.0f

/* cubasmcn at one level (:3457-3482).  `kk` is 1-based, matching the
 * Fortran; the column arrays are 1-based too (index 0 unused), so every
 * subscript keeps its Fortran form. */
__device__ __forceinline__ void nt_cubasmcn(
        int kk, int ncol, int i,
        const float *pten, const float *pqen, const float *pqsen,
        const float *pverv, const float *pgeo, const float *pgeoh,
        int *ldcum, int *ktype, int *klab,
        /* The single plrain slot to clear -- kk+1 -- taken by ADDRESS
         * so cuascn, which keeps zlrain in a register, shares this
         * body instead of forking it. */
        float *plrain_slot,
        float *pmfu, float *pmfub, int *kcbot,
        float *ptu, float *pqu, float *plu,
        float *pmfus, float *pmfuq, float *pmful, float *pdmfup,
        const NtConst &c) {
    /* if(.not.ldcum .and. klab(kk+1) == 0) */
    if (ldcum[i] != 0 || NTC(klab, kk + 1) != 0) return;
    /* lmfmid is a parameter, .true.  pgeo*zrg is height in metres, so the
     * window below is 500 m to 10 km. */
    const float hgt = __fmul_rn(NTC(pgeo, kk), c.zrg);
    if (!(NTC(pqen, kk) > __fmul_rn(0.80f, NTC(pqsen, kk))
          && hgt > 5.0e2f && hgt < 1.0e4f)) return;

    NTC(ptu, kk + 1) = __fmul_rn(
        __fadd_rn(__fadd_rn(__fmul_rn(c.cpd, NTC(pten, kk)), NTC(pgeo, kk)),
                  -NTC(pgeoh, kk + 1)), c.rcpd);
    NTC(pqu, kk + 1) = NTC(pqen, kk);
    NTC(plu, kk + 1) = 0.0f;
    float zzzmb = fmaxf(NT_CMFCMIN, -__fmul_rn(NTC(pverv, kk), c.zrg));
    zzzmb = fminf(zzzmb, NT_CMFCMAX);
    pmfub[i] = zzzmb;
    NTC(pmfu, kk + 1) = pmfub[i];
    NTC(pmfus, kk + 1) = __fmul_rn(pmfub[i],
        __fadd_rn(__fmul_rn(c.cpd, NTC(ptu, kk + 1)), NTC(pgeoh, kk + 1)));
    NTC(pmfuq, kk + 1) = __fmul_rn(pmfub[i], NTC(pqu, kk + 1));
    NTC(pmful, kk + 1) = 0.0f;
    NTC(pdmfup, kk + 1) = 0.0f;
    kcbot[i] = kk;
    NTC(klab, kk + 1) = 1;
    *plrain_slot = 0.0f;
    ktype[i] = 3;
}

/* cuentrn at one level (:3516-3536).
 *
 * zentr is set to zero and never assigned again, so pdmfen is IDENTICALLY
 * ZERO in v4.6.1 -- the entrainment term here is dead code.  The multiply
 * is transcribed anyway, exactly as the reference performs it, so that a
 * later WRF reviving zentr breaks the parity gate rather than silently
 * changing the answer. */
__device__ __forceinline__ void nt_cuentrn(
        int kk, int ncol, int i,
        const int *kcbot, const int *ldcum, bool ldwork,
        const float *pgeoh, const float *pmfu,
        float *pdmfen, float *pdmfde, const NtConst &c) {
    if (!ldwork) return;
    *pdmfen = 0.0f;
    *pdmfde = 0.0f;
    const float zentr = 0.0f;
    if (ldcum[i] == 0) return;
    const float zdz = __fmul_rn(
        __fadd_rn(NTC(pgeoh, kk), -NTC(pgeoh, kk + 1)), c.zrg);
    const float zmf = __fmul_rn(NTC(pmfu, kk + 1), zdz);
    if (kk < kcbot[i]) {
        *pdmfen = __fmul_rn(zentr, zmf);
        *pdmfde = __fmul_rn(0.75e-4f, zmf);
    }
}

/* Drives both over the level range cuascn uses (jk = klev-1 .. 3), so the
 * per-level ordering -- cubasmcn then cuentrn at the same level -- matches
 * the reference's call sequence. */
extern "C" __global__ void ntiedtke_midlevel(
        const float *__restrict__ pten,    // (nz+2, ncol), 1-based
        const float *__restrict__ pqen,
        const float *__restrict__ pqsen,
        const float *__restrict__ pverv,
        const float *__restrict__ pgeo,
        const float *__restrict__ pgeoh,
        int *__restrict__ ldcum,           // (ncol), in AND out
        int *__restrict__ ktype,
        int *__restrict__ kcbot,
        int *__restrict__ klab,            // (nz+2, ncol), in AND out
        float *__restrict__ plrain,
        float *__restrict__ pmfu,
        float *__restrict__ pmfub,
        float *__restrict__ ptu,
        float *__restrict__ pqu,
        float *__restrict__ plu,
        float *__restrict__ pmfus,
        float *__restrict__ pmfuq,
        float *__restrict__ pmful,
        float *__restrict__ pdmfup,
        float *__restrict__ dmfen,         // (nz+2, ncol) per-level capture
        float *__restrict__ dmfde,
        int ncol, int nz,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_MIDLEVEL,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_MIDLEVEL, order_report, ticket);
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);

    for (int kk = nz - 1; kk >= 3; --kk) {
        nt_cubasmcn(kk, ncol, i, pten, pqen, pqsen, pverv, pgeo, pgeoh,
                    ldcum, ktype, klab, &NTC(plrain, kk + 1),
                    pmfu, pmfub, kcbot,
                    ptu, pqu, plu, pmfus, pmfuq, pmful, pdmfup, c);
        /* cuentrn's outputs are per-column scalars in the reference; the
         * fixture captures them per level, so they are taken by pointer
         * and stored.  ONE body, called once -- an inlined second copy
         * here would be exactly the two-clone shape ptxas can contract
         * differently. */
        float e0, d0;
        nt_cuentrn(kk, ncol, i, kcbot, ldcum, true, pgeoh, pmfu,
                   &e0, &d0, c);
        NTC(dmfen, kk) = e0;
        NTC(dmfde, kk) = d0;
    }
}

#undef NTC

/* ==========================================================================
 * Slice 4b: cumastrn's first-guess cloud-base mass flux (:500-541)
 * ==========================================================================
 * Runs BETWEEN cutypen and cuascn.  A prerequisite rather than a stage: it
 * produces pmfub, which THREE things consume --
 *
 *   cuascn      (:553 -> :1949-1952, :1992)  the whole updraft mass flux
 *   cudlfsn     (:602 -> :2469)              zmftop = -cmfdeps*pmfub
 *   the closure (:684, :698, :713, :722, :732, :745) via zmfub1
 *
 * -- and it can CLEAR ldcum (:536) for a ktype = 2 column whose PBL moist
 * static energy budget is non-positive.
 *
 * Skipping it hands cuascn pmfub = 0, and every mass-flux quantity
 * downstream is structurally zero: green, and meaningless.  That is the
 * same defect that made the cuentrn capture degenerate, and its appearing
 * twice is what identified it as a prerequisite rather than a coverage gap.
 *
 * upbl is computed here and consumed by the closure at :636-637 (ztaubl,
 * the nonequil timescale).  It is produced and graded now so the closure
 * slice inherits it proven.
 *
 * ldcum is int, not bool: it is LOGICAL(4) in the reference and this kernel
 * both reads and writes it, so the storage has to match what the harness
 * captured.
 */
#define NTM(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

extern "C" __global__ void ntiedtke_mfub(
        const float *__restrict__ ptte,    // (nz+2, ncol), 1-based
        const float *__restrict__ pqte,
        const float *__restrict__ paph,    // needs index klev+1
        const float *__restrict__ puen,
        const float *__restrict__ pven,
        const float *__restrict__ ptu,
        const float *__restrict__ pqu,
        const float *__restrict__ plu,
        const float *__restrict__ ztenh,
        const float *__restrict__ zqenh,
        const int *__restrict__ ktype,     // (ncol)
        const int *__restrict__ kcbot,
        const int *__restrict__ lndj,
        int *__restrict__ ldcum,           // (ncol), in AND out
        float *__restrict__ zdhpbl,
        float *__restrict__ upbl,
        float *__restrict__ zmfub,
        int tiedtke_closure,
        int ncol, int nz, float dt,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_MFUB,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_MFUB, order_report, ticket);
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
    const int klev = nz;

    /* :468-469.  zcons2 = 3/(g*dt); the 3 is not a typo -- zcons at :468 is
     * the same quantity with 1. */
    const float zcons2 = __fdiv_rn(3.0f, __fmul_rn(c.g, dt));

    float dhpbl = 0.0f;
    float zdqpbl = 0.0f;   /* cu6 :832-839 */
    float up = 0.0f;
    bool ld = ldcum[i] != 0;

    /* do jk = 2, klev */
    for (int jk = 2; jk <= klev; ++jk) {
        if (!(jk >= kcbot[i] && ld)) continue;
        const float dp = __fadd_rn(NTM(paph, jk + 1), -NTM(paph, jk));
        dhpbl = __fadd_rn(dhpbl, __fmul_rn(
            __fadd_rn(__fmul_rn(c.alv, NTM(pqte, jk)),
                      __fmul_rn(c.cpd, NTM(ptte, jk))), dp));
        /* module_cu_tiedtke.F:839 -- MOISTURE convergence alone, the
         * quantity cu6's deep first guess divides.  Accumulated in the
         * same loop over the same dp so the two agree cell for cell. */
        zdqpbl = __fadd_rn(zdqpbl, __fmul_rn(NTM(pqte, jk), dp));
        if (lndj[i] == 0) {
            const float u = NTM(puen, jk), v = NTM(pven, jk);
            const float wspeed = sqrtf(__fadd_rn(__fmul_rn(u, u),
                                                 __fmul_rn(v, v)));
            up = __fadd_rn(up, __fmul_rn(wspeed, dp));
        }
    }

    float mfub = 0.0f;
    if (ld) {
        const int ikb = kcbot[i];
        const float zmfmax = __fmul_rn(
            __fadd_rn(NTM(paph, ikb), -NTM(paph, ikb - 1)), zcons2);
        if (ktype[i] == 1) {
            if (tiedtke_closure) {
                /* module_cu_tiedtke.F:855-866.  cu6's deep first guess is
                 * MOISTURE CONVERGENCE, not cu16's geometric 0.1*zmfmax.
                 * Same zqumqe/zdqmin construction as the shallow arm
                 * below; the 0.01 floor is cu6 :862 and the cap :866. */
                const float zqumqe = __fadd_rn(
                    __fadd_rn(NTM(pqu, ikb), NTM(plu, ikb)), -NTM(zqenh, ikb));
                const float zdqmin = fmaxf(
                    __fmul_rn(0.01f, NTM(zqenh, ikb)), 1.0e-10f);
                if (zdqpbl > 0.0f && zqumqe > zdqmin) {
                    mfub = __fdiv_rn(zdqpbl,
                                     __fmul_rn(c.g, fmaxf(zqumqe, zdqmin)));
                    mfub = fminf(mfub, zmfmax);
                } else {
                    mfub = 0.01f;
                }
            } else {
                mfub = __fmul_rn(0.1f, zmfmax);
            }
        } else if (ktype[i] == 2) {
            const float zqumqe = __fadd_rn(
                __fadd_rn(NTM(pqu, ikb), NTM(plu, ikb)), -NTM(zqenh, ikb));
            const float zdqmin = fmaxf(
                __fmul_rn(0.01f, NTM(zqenh, ikb)), 1.0e-10f);
            float zdh = __fadd_rn(
                __fmul_rn(c.cpd, __fadd_rn(NTM(ptu, ikb), -NTM(ztenh, ikb))),
                __fmul_rn(c.alv, zqumqe));
            zdh = __fmul_rn(c.g, fmaxf(zdh, __fmul_rn(1.0e5f, zdqmin)));
            if (dhpbl > 0.0f) {
                mfub = __fdiv_rn(dhpbl, zdh);
                mfub = fminf(mfub, zmfmax);
            } else {
                mfub = __fmul_rn(0.1f, zmfmax);
                ld = false;                    /* :536 -- clears ldcum */
            }
        }
        /* ktype 0 or 3 with ldcum true is unreachable from cumastrn --
         * cutypen produces only 0/1/2 and ldcum is false whenever ktype is
         * 0 -- and the reference assigns nothing there, leaving zmfub at
         * its initialised zero.  So does this. */
    }

    zdhpbl[i] = dhpbl;
    upbl[i] = up;
    zmfub[i] = mfub;
    ldcum[i] = ld ? 1 : 0;
}

#undef NTM

/* ==========================================================================
 * Slice 5: the CAPE closure (cumastrn:620-745)
 * ==========================================================================
 * THE ARITHMETIC THE WHOLE PORT TURNS ON.  This is where scale_fac and
 * scale_fac2 are applied, and they go to DIFFERENT ktypes:
 *
 *   :676  ztau = ztauc * scale_fac        ktype == 1 (deep) only
 *   :716  zmfub1 = zmfub1 / scale_fac2    ktype == 2 (shallow) only
 *   :722  zmfub1 = zmfub                  ktype == 3, neither
 *
 * The deep arm is NOT a division by scale_fac.  zmfub1 =
 * zcape*zmfub/(zheat*ztau) is a full CAPE closure with a max(zmfub1, 0.001)
 * floor and a zmfmax cap, and it can EXCEED its own first guess -- measured
 * at 141.2% for dx = 15000.  What scales with resolution is zmfub1 through
 * 1/ztau, so the cross-resolution ratio is
 * scale_fac(15000)/scale_fac(4500) = 10.3%.
 *
 * ==========================================================================
 * KTYPE IS THE CLOSURE-TIME VALUE, NOT CUASCN'S
 * ==========================================================================
 * cumastrn:566-568 FLIPS ktype between cuascn and here -- deep becomes
 * shallow when the cloud is thinner than zdnoprc, and shallow becomes deep
 * when it is not.  Feeding this kernel cuascn's ktype runs the wrong arm.
 * That cost two rounds of debugging on the mirror; the caller must pass the
 * post-:568 value.
 *
 * ==========================================================================
 * WHAT IS UNDEFINED OUTSIDE THE DEEP BRANCH
 * ==========================================================================
 * zheat, zcape, zcape1, zcape2, ztauc, ztaubl and upbl are dimension(klon)
 * arrays in cumastrn assigned ONLY inside `if(ldcum .and. ktype==1)`.  For
 * any other column the reference leaves them at whatever they held -- the
 * pqsenh[0] class.  This kernel writes them only where the reference does,
 * so a non-deep column's slots keep the caller's values.  Do not initialise
 * them; that would diverge on 66 of 108 columns and grade green on the rest.
 *
 * ztau is a SCALAR in cumastrn, written :676 and read :684 inside one loop
 * iteration.  It never escapes, so one thread per column reproduces it.
 * (itopm2 is the scalar that LOOKS like it escapes and does not -- see
 * docs/ntiedtke/PORT-RECORD.md §9 before porting cuflxn.)
 */
#define NTK(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

extern "C" __global__ void ntiedtke_closure(
        const int *__restrict__ ldcum,      // (ncol)
        const int *__restrict__ ktype,      // CLOSURE-TIME, post :566-568
        const int *__restrict__ kcbot,
        const int *__restrict__ kctop,
        const int *__restrict__ kdpl,
        const int *__restrict__ loddraf,
        const int *__restrict__ lndj,
        const float *__restrict__ wup,
        const float *__restrict__ zmfub,
        const float *__restrict__ zdhpbl,
        const float *__restrict__ scale_fac,
        const float *__restrict__ scale_fac2,
        const float *__restrict__ pgeoh,    // (nz+2, ncol), 1-based
        const float *__restrict__ paph,
        const float *__restrict__ pap,
        const float *__restrict__ pgeo,
        const float *__restrict__ pten,
        const float *__restrict__ pqen,
        const float *__restrict__ ptenh,
        const float *__restrict__ pqenh,
        const float *__restrict__ ptu,
        const float *__restrict__ pqu,
        const float *__restrict__ plu,
        const float *__restrict__ pmfu,
        const float *__restrict__ ptd,
        const float *__restrict__ pqd,
        const float *__restrict__ ptte,
        const float *__restrict__ pqte,
        float *__restrict__ pmfd,           // (nz+2, ncol) in AND out
        float *__restrict__ pmfds,
        float *__restrict__ pmfdq,
        float *__restrict__ pdmfdp,
        float *__restrict__ pmfdde_rate,
        float *__restrict__ zheat,          // (ncol) -- deep-only outputs
        float *__restrict__ zcape,
        float *__restrict__ zcape1,
        float *__restrict__ zcape2,
        float *__restrict__ ztauc,
        float *__restrict__ ztaubl,
        float *__restrict__ ztau_o,
        float *__restrict__ upbl,           // in AND out, deep+water only
        float *__restrict__ zmfub1,
        int tiedtke_closure,
        int ncol, int nz, float dt,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CLOSURE,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CLOSURE, order_report, ticket);
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
    const int klev = nz;
    const float zcons2 = __fdiv_rn(3.0f, __fmul_rn(c.g, dt));
    const bool ld = ldcum[i] != 0;
    const int kt = ktype[i];
    const bool deep = ld && kt == 1;

    /* :622-641  timescales.  Written only on the deep arm. */
    if (deep) {
        const int ikb = kcbot[i], ikt = kctop[i];
        zmfub1[i] = zmfub[i];
        ztauc[i] = __fdiv_rn(
            __fadd_rn(NTK(pgeoh, ikt), -NTK(pgeoh, ikb)),
            __fmul_rn(__fadd_rn(2.0f, fminf(15.0f, wup[i])), c.g));
        if (lndj[i] == 0) {
            upbl[i] = __fadd_rn(2.0f, __fdiv_rn(
                upbl[i], __fadd_rn(NTK(paph, klev + 1), -NTK(paph, ikb))));
            ztaubl[i] = __fdiv_rn(
                __fadd_rn(NTK(pgeoh, ikb), -NTK(pgeoh, klev + 1)),
                __fmul_rn(c.g, upbl[i]));
            ztaubl[i] = fminf(300.0f, ztaubl[i]);
        } else {
            ztaubl[i] = ztauc[i];
        }
        zheat[i] = 0.0f; zcape[i] = 0.0f;
        zcape1[i] = 0.0f; zcape2[i] = 0.0f;
    }

    /* :644-668  the CAPE and heating integrals */
    for (int jk = 1; jk <= klev; ++jk) {
        if (deep && jk <= kcbot[i] && jk > kctop[i]) {
            const float zdz = __fadd_rn(NTK(pgeo, jk - 1), -NTK(pgeo, jk));
            const float zdp = __fadd_rn(NTK(pap, jk), -NTK(pap, jk - 1));
            zheat[i] = __fadd_rn(zheat[i], __fmul_rn(
                __fadd_rn(
                    __fdiv_rn(__fadd_rn(__fadd_rn(NTK(pten, jk - 1),
                                                  -NTK(pten, jk)),
                                        __fmul_rn(zdz, c.rcpd)),
                              NTK(ptenh, jk)),
                    __fmul_rn(c.vtmpc1, __fadd_rn(NTK(pqen, jk - 1),
                                                  -NTK(pqen, jk)))),
                __fmul_rn(c.g, __fadd_rn(NTK(pmfu, jk), NTK(pmfd, jk)))));
            zcape1[i] = __fadd_rn(zcape1[i], __fmul_rn(
                __fadd_rn(
                    __fdiv_rn(__fadd_rn(NTK(ptu, jk), -NTK(ptenh, jk)),
                              NTK(ptenh, jk)),
                    __fadd_rn(__fmul_rn(c.vtmpc1,
                                        __fadd_rn(NTK(pqu, jk),
                                                  -NTK(pqenh, jk))),
                              -NTK(plu, jk))),
                zdp));
        }
        if (deep && jk >= kcbot[i]) {
            if (__fadd_rn(NTK(paph, klev + 1), -NTK(paph, kdpl[i]))
                    < 50.0e2f) {
                const float zdp = __fadd_rn(NTK(paph, jk + 1),
                                            -NTK(paph, jk));
                zcape2[i] = __fadd_rn(zcape2[i], __fmul_rn(
                    __fmul_rn(ztaubl[i], __fadd_rn(
                        __fmul_rn(__fadd_rn(1.0f,
                                            __fmul_rn(c.vtmpc1,
                                                      NTK(pqen, jk))),
                                  NTK(ptte, jk)),
                        __fmul_rn(__fmul_rn(c.vtmpc1, NTK(pten, jk)),
                                  NTK(pqte, jk)))),
                    zdp));
            }
        }
    }

    /* :670-694  the deep closure */
    if (deep) {
        const int ikb = kcbot[i];
        float tc = ztauc[i];
        tc = fmaxf(dt, tc);
        tc = fmaxf(360.0f, tc);
        tc = fminf(10800.0f, tc);
        ztauc[i] = tc;
        /* modC.  cu16 :676 scales the adjustment time by scale_fac
         * (11.6246 at dx=4500); cu6 uses a FIXED 2400 s
         * (module_cu_tiedtke.F:105 parameter ztau = 2400.0), which is
         * precisely the scale-awareness deletion that separates the two
         * schemes' deep closures. */
        const float ztau = tiedtke_closure
            ? 2400.0f : __fmul_rn(tc, scale_fac[i]);
        ztau_o[i] = ztau;
        /* nonequil is .true. (cu_ntiedtke_common:49) */
        zcape2[i] = fmaxf(0.0f, zcape2[i]);
        zcape[i] = fmaxf(0.0f, fminf(__fadd_rn(zcape1[i], -zcape2[i]),
                                     5000.0f));
        zheat[i] = fmaxf(1.0e-4f, zheat[i]);
        float b1 = __fdiv_rn(__fmul_rn(zcape[i], zmfub[i]),
                             __fmul_rn(zheat[i], ztau));
        b1 = fmaxf(b1, 0.001f);
        const float zmfmax = __fmul_rn(
            __fadd_rn(NTK(paph, ikb), -NTK(paph, ikb - 1)), zcons2);
        zmfub1[i] = fminf(b1, zmfmax);
    }

    /* :696-720  the shallow closure -- the ONLY use of scale_fac2 */
    if (ld && kt == 2) {
        const int ikb = kcbot[i];
        float zeps;
        if (NTK(pmfd, ikb) < 0.0f && loddraf[i] != 0) {
            zeps = __fdiv_rn(-NTK(pmfd, ikb),
                             fmaxf(zmfub[i], NT_CMFCMIN));
        } else {
            zeps = 0.0f;
        }
        const float zqumqe = __fadd_rn(
            __fadd_rn(__fadd_rn(NTK(pqu, ikb), NTK(plu, ikb)),
                      -__fmul_rn(zeps, NTK(pqd, ikb))),
            -__fmul_rn(__fadd_rn(1.0f, -zeps), NTK(pqenh, ikb)));
        const float zdqmin = fmaxf(__fmul_rn(0.01f, NTK(pqenh, ikb)),
                                   NT_CMFCMIN);
        const float zmfmax = __fmul_rn(
            __fadd_rn(NTK(paph, ikb), -NTK(paph, ikb - 1)), zcons2);
        float zdh = __fadd_rn(
            __fmul_rn(c.cpd, __fadd_rn(
                __fadd_rn(NTK(ptu, ikb), -__fmul_rn(zeps, NTK(ptd, ikb))),
                -__fmul_rn(__fadd_rn(1.0f, -zeps), NTK(ptenh, ikb)))),
            __fmul_rn(c.alv, zqumqe));
        zdh = __fmul_rn(c.g, fmaxf(zdh, __fmul_rn(1.0e5f, zdqmin)));
        /* zdhpbl is an INPUT: cumastrn built it at :506-517 from CUTYPEN's
         * kcbot and ldcum, and cuascn has since changed both.  Recomputing
         * it from closure-time state is wrong on every shallow column. */
        float b1 = (zdhpbl[i] > 0.0f) ? __fdiv_rn(zdhpbl[i], zdh)
                                      : zmfub[i];
        b1 = __fdiv_rn(b1, scale_fac2[i]);
        zmfub1[i] = fminf(b1, zmfmax);
    }

    /* :722-724  mid-level takes NEITHER factor */
    if (ld && kt == 3) zmfub1[i] = zmfub[i];

    /* :726-740  scale the downdraft by zmfub1/zmfub */
    if (ld) {
        const float zfac = __fdiv_rn(zmfub1[i],
                                     fmaxf(zmfub[i], NT_CMFCMIN));
        for (int jk = 1; jk <= klev; ++jk) {
            NTK(pmfd, jk) = __fmul_rn(NTK(pmfd, jk), zfac);
            NTK(pmfds, jk) = __fmul_rn(NTK(pmfds, jk), zfac);
            NTK(pmfdq, jk) = __fmul_rn(NTK(pmfdq, jk), zfac);
            NTK(pdmfdp, jk) = __fmul_rn(NTK(pdmfdp, jk), zfac);
            NTK(pmfdde_rate, jk) = __fmul_rn(NTK(pmfdde_rate, jk), zfac);
        }
    }
}

#undef NTK


/* =====================================================================
 * Stage 7: cuascn -- the entraining/detraining updraft (:1755-2258)
 * =====================================================================
 * The largest routine in the scheme and the plume itself.  Graded against
 * nt-cuascn-out-levels.csv and nt-cuascn-surface.csv at max_ulp == 0; the
 * NumPy mirror in gpuwm/verify/ntiedtke_ref.py is the same transcription
 * and the two are graded against the same oracle rows.
 *
 * NO WORKSPACE.  cuascn declares five klon x klev locals -- zlrain, zbuo,
 * kup, zodetr, pdmfen -- and the naive port gives each a (nz+2, ncol)
 * global array, which at nz = 62 on a 372x284 domain is 81 MiB of VRAM for
 * scratch.  None of it is needed:
 *
 *   zodetr  is never assigned anywhere in the routine.  Dead.
 *   pdmfen  is written at :2050 and never read.  A local, not a dummy, so
 *           nothing downstream can observe it.  Dead.
 *   zlrain  is read only at jk+1 and jk.
 *   zbuo    is read only at jk+1 and jk.
 *   kup     is read only at jk+1 and jk.
 *
 * The loop descends one level at a time, so the last three are strictly
 * one-level lookback and live in registers -- three floats instead of
 * three arrays.  That is the difference between 0 B of scratch and 81 MiB,
 * and it is why this kernel still measures 0 B frame.
 *
 * The seeding is the part to get right.  At the top of each iteration the
 * "cur" register must hold what section 2 and section 3 left at level jk:
 * zero, except kup at cloud base, which section 3 set to 0.5*wbase^2.  The
 * "prev" register enters the loop holding level klev under the same rule.
 * cubasmcn can zero zlrain at jk+1 mid-iteration (:3480), so it takes the
 * prev register by reference rather than reading the array.
 *
 * llo3 IS A TILE-WIDE FLAG, not a per-column one: :1994 sums klab over all
 * columns and :2009 latches it true for the rest of the run.  It is passed
 * in as a kernel argument for that reason, and NOT recomputed per lane.
 * test_ntiedtke_cuascn_parity.py::test_llo3_is_true_throughout is the gate
 * that makes passing 1 exact on this fixture; a tile that broke it would
 * need a block-wide OR reduction here, and the gate fails first.
 */
#define NT_CPRCON 1.4e-3f
#define NT_RTBER  (NT_TMELT - 5.0f)

extern "C" __global__ void ntiedtke_cuascn(
        const float *__restrict__ pten,     // (nz+2, ncol), 1-based
        const float *__restrict__ pqen,
        const float *__restrict__ pqsen,
        const float *__restrict__ pgeo,
        const float *__restrict__ pgeoh,
        const float *__restrict__ pap,
        const float *__restrict__ paph,
        const float *__restrict__ pverv,
        const int *__restrict__ lndj,
        const int *__restrict__ kdpl,
        const float *__restrict__ wbase,
        float *__restrict__ ptenh,          // in AND out (:2118-2119)
        float *__restrict__ pqenh,
        float *__restrict__ ptu,
        float *__restrict__ pqu,
        float *__restrict__ plu,
        float *__restrict__ pmfu,
        float *__restrict__ pmfus,
        float *__restrict__ pmfuq,
        float *__restrict__ pmful,
        float *__restrict__ plude,
        float *__restrict__ pdmfup,
        float *__restrict__ plglac,
        float *__restrict__ pmfude_rate,
        int *__restrict__ klab,
        int *__restrict__ ldcum,            // scalars, in AND out
        int *__restrict__ ktype,
        int *__restrict__ kcbot,
        int *__restrict__ kctop,
        int *__restrict__ kctop0,
        float *__restrict__ pmfub,
        float *__restrict__ wup,
        int ncol, int klev, float ztmst,
        /* llo3 is TILE-WIDE, so it is a launch argument.  See the header. */
        int llo3,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        /* The geometry tail NtStages.launch appends, always last. */
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUASCN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUASCN, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]
    const int klevm1 = klev - 1;

    /* :1893-1899.  Each in the reference's own association.
     *
     * CORRECTED.  This said cuascn's cap is "three times looser than the
     * closure's".  It is not: zcons2 is 3/(g*dt) in all three scopes that
     * declare it and they are identical.  The factor of three separates
     * zcons2 from zcons (1/(g*dt), cumastrn:468) -- one character apart,
     * both live in cumastrn, and zcons has exactly one consumer: the
     * momentum rescale at :1000. */
    const float zcons2 = __fdiv_rn(3.0f, __fmul_rn(c.g, ztmst));
    const float zfacbuo = __fdiv_rn(0.5f, __fadd_rn(1.0f, 0.5f));
    const float zprcdgw = __fmul_rn(NT_CPRCON, c.zrg);
    const float z_cldmax = 5.0e-3f;
    const float z_cwifrac = 0.5f;
    const float z_cprc2 = 0.5f;
    const float z_cwdrag =
        __fdiv_rn(__fmul_rn(__fdiv_rn(3.0f, 8.0f), 0.506f), 0.2f);

#define LD   ldcum[i]
#define KT   ktype[i]
#define KB   kcbot[i]
#define MFUB pmfub[i]
    int kt0 = kctop0[i];
    int ktp;
    const int kdp = kdpl[i];
    const float wb = wbase[i];

    float zluold = 0.0f, wupa = 0.0f, zdpmean = 0.0f, zoentr = 0.0f;

    /* ---- 2. defaults (:1903-1937) ---------------------------------- */
    if (!LD) {
        KT = 0;
        KB = -1;
        MFUB = 0.0f;
        NTC(pqu, klev) = 0.0f;
    }
    for (int jk = 1; jk <= klev; ++jk) {
        if (jk != KB) NTC(plu, jk) = 0.0f;
        NTC(pmfu, jk) = 0.0f;   NTC(pmfus, jk) = 0.0f;
        NTC(pmfuq, jk) = 0.0f;  NTC(pmful, jk) = 0.0f;
        NTC(plude, jk) = 0.0f;  NTC(plglac, jk) = 0.0f;
        NTC(pdmfup, jk) = 0.0f; NTC(pmfude_rate, jk) = 0.0f;
        if (!LD || KT == 3) NTC(klab, jk) = 0;
        if (!LD && NTC(paph, jk) < 4.0e4f) kt0 = jk;
    }
    if (KT == 3) LD = 0;

    /* ---- 3. cloud base (:1943-1953) -------------------------------- */
    ktp = KB;
    if (LD) {
        NTC(pmfu, KB) = MFUB;
        NTC(pmfus, KB) = __fmul_rn(
            MFUB, __fadd_rn(__fmul_rn(c.cpd, NTC(ptu, KB)), NTC(pgeoh, KB)));
        NTC(pmfuq, KB) = __fmul_rn(MFUB, NTC(pqu, KB));
        NTC(pmful, KB) = __fmul_rn(MFUB, NTC(plu, KB));
    }

    /* The three one-level-lookback registers, seeded at level klev by the
     * same rule section 2 and section 3 applied to the arrays.
     *
     * kb0 is the cloud base AS SECTION 3 SAW IT, not the live kcbot.  Only
     * section 3 wrote kup, and only at that level; cubasmcn can move kcbot
     * later in the loop and must not drag the seed with it. */
    const int kb0 = LD ? KB : -1;
    float zlrain_prev = 0.0f;
    float zbuo_prev = 0.0f;
    float kup_prev = (kb0 == klev)
                   ? __fmul_rn(0.5f, __fmul_rn(wb, wb)) : 0.0f;

    float zdmfen = 0.0f, zdmfde = 0.0f;

    /* ---- 4. the ascent (:1959-2245) -------------------------------- */
    for (int jk = klevm1; jk >= 3; --jk) {
        /* cubasmcn (:1968).  It can zero zlrain at jk+1, so the register
         * goes in by reference. */
        nt_cubasmcn(jk, ncol, i, pten, pqen, pqsen, pverv, pgeo, pgeoh,
                    ldcum, ktype, klab, &zlrain_prev, pmfu, pmfub, kcbot,
                    ptu, pqu, plu, pmfus, pmfuq, pmful, pdmfup, c);

        float kup_cur = (jk == kb0)
                      ? __fmul_rn(0.5f, __fmul_rn(wb, wb)) : 0.0f;
        float zlrain_cur = 0.0f;
        float zbuo_cur = 0.0f;
        int llo1 = 0;
        float zprecip = 0.0f;

        /* :1980-2001 */
        if (NTC(klab, jk + 1) == 0) NTC(klab, jk) = 0;
        const int loflag = (LD && NTC(klab, jk + 1) == 2)
                        || (KT == 3 && NTC(klab, jk + 1) == 1);
        const float zph = NTC(paph, jk);
        if (KT == 3 && jk == KB) {
            const float zmfmax = __fmul_rn(
                __fadd_rn(NTC(paph, jk), -NTC(paph, jk - 1)), zcons2);
            if (MFUB > zmfmax) {
                const float zfac = __fdiv_rn(zmfmax, MFUB);
                NTC(pmfu, jk + 1) = __fmul_rn(NTC(pmfu, jk + 1), zfac);
                NTC(pmfus, jk + 1) = __fmul_rn(NTC(pmfus, jk + 1), zfac);
                NTC(pmfuq, jk + 1) = __fmul_rn(NTC(pmfuq, jk + 1), zfac);
                MFUB = zmfmax;
            }
            MFUB = fminf(MFUB, zmfmax);
        }

        nt_cuentrn(jk, ncol, i, kcbot, ldcum, llo3 != 0, pgeoh, pmfu,
                   &zdmfen, &zdmfde, c);

        if (!llo3) continue;

        /* :2015-2065 */
        float zqold = 0.0f;
        if (loflag) {
            zdmfde = fminf(zdmfde, __fmul_rn(0.75f, NTC(pmfu, jk + 1)));
            if (jk == KB) {
                const float r = fminf(1.0f, __fdiv_rn(NTC(pqen, jk),
                                                      NTC(pqsen, jk)));
                zoentr = __fmul_rn(
                    __fmul_rn(__fmul_rn(-1.75e-3f, __fadd_rn(r, -1.0f)),
                              __fadd_rn(NTC(pgeoh, jk),
                                        -NTC(pgeoh, jk + 1))), c.zrg);
                zoentr = __fmul_rn(fminf(0.4f, zoentr), NTC(pmfu, jk + 1));
            }
            if (jk < KB) {
                const float zmfmax = __fmul_rn(
                    __fadd_rn(NTC(paph, jk), -NTC(paph, jk - 1)), zcons2);
                const float zxs =
                    fmaxf(__fadd_rn(NTC(pmfu, jk + 1), -zmfmax), 0.0f);
                const float dpap =
                    __fadd_rn(NTC(pap, jk + 1), -NTC(pap, jk));
                wupa = __fadd_rn(wupa, __fmul_rn(kup_prev, dpap));
                /* :2036 has NO parentheses, so it associates left to
                 * right.  The wup accumulator one line up IS parenthesised.
                 * Bracketing the difference here costs 1 ULP in wup. */
                zdpmean = __fadd_rn(__fadd_rn(zdpmean, NTC(pap, jk + 1)),
                                    -NTC(pap, jk));
                zdmfen = zoentr;
                if (KT >= 2) {
                    zdmfen = __fmul_rn(2.0f, zdmfen);
                    zdmfde = zdmfen;
                }
                zdmfde = __fmul_rn(zdmfde, __fadd_rn(1.6f,
                    -fminf(1.0f, __fdiv_rn(NTC(pqen, jk),
                                           NTC(pqsen, jk)))));
                const float zmftest =
                    __fadd_rn(__fadd_rn(NTC(pmfu, jk + 1), zdmfen), -zdmfde);
                float zchange = fmaxf(__fadd_rn(zmftest, -zmfmax), 0.0f);
                const float zxe = fmaxf(__fadd_rn(zchange, -zxs), 0.0f);
                zdmfen = __fadd_rn(zdmfen, -zxe);
                zchange = __fadd_rn(zchange, -zxe);
                zdmfde = __fadd_rn(zdmfde, zchange);
            }
            /* :2050 writes pdmfen, which is a dead local.  Not stored. */
            NTC(pmfu, jk) =
                __fadd_rn(__fadd_rn(NTC(pmfu, jk + 1), zdmfen), -zdmfde);
            const float zqeen = __fmul_rn(NTC(pqenh, jk + 1), zdmfen);
            const float zseen = __fmul_rn(
                __fadd_rn(__fmul_rn(c.cpd, NTC(ptenh, jk + 1)),
                          NTC(pgeoh, jk + 1)), zdmfen);
            const float zscde = __fmul_rn(
                __fadd_rn(__fmul_rn(c.cpd, NTC(ptu, jk + 1)),
                          NTC(pgeoh, jk + 1)), zdmfde);
            const float zqude = __fmul_rn(NTC(pqu, jk + 1), zdmfde);
            NTC(plude, jk) = __fmul_rn(NTC(plu, jk + 1), zdmfde);
            const float zmfusk =
                __fadd_rn(__fadd_rn(NTC(pmfus, jk + 1), zseen), -zscde);
            const float zmfuqk =
                __fadd_rn(__fadd_rn(NTC(pmfuq, jk + 1), zqeen), -zqude);
            const float zmfulk =
                __fadd_rn(NTC(pmful, jk + 1), -NTC(plude, jk));
            const float inv =
                __fdiv_rn(1.0f, fmaxf(NT_CMFCMIN, NTC(pmfu, jk)));
            NTC(plu, jk) = __fmul_rn(zmfulk, inv);
            NTC(pqu, jk) = __fmul_rn(zmfuqk, inv);
            NTC(ptu, jk) = __fmul_rn(
                __fadd_rn(__fmul_rn(zmfusk, inv), -NTC(pgeoh, jk)), c.rcpd);
            NTC(ptu, jk) = fmaxf(100.0f, NTC(ptu, jk));
            NTC(ptu, jk) = fminf(400.0f, NTC(ptu, jk));
            zqold = NTC(pqu, jk);
            zlrain_cur = __fmul_rn(
                __fmul_rn(zlrain_prev,
                          __fadd_rn(NTC(pmfu, jk + 1), -zdmfde)), inv);
            zluold = NTC(plu, jk);
        }

        /* :2069-2075.  NOT guarded by loflag -- every column. */
        if (jk > kdp) {
            NTC(ptu, jk) = NTC(ptenh, jk);
            NTC(pqu, jk) = NTC(pqenh, jk);
            NTC(plu, jk) = 0.0f;
            zluold = NTC(plu, jk);
        }

        /* :2081-2083 */
        if (loflag) nt_cuadjtqn1(&NTC(ptu, jk), &NTC(pqu, jk), zph, c);

        const int adjusted = loflag && (NTC(pqu, jk) != zqold);

        /* :2086-2093 */
        if (adjusted) {
            NTC(plglac, jk) = __fmul_rn(NTC(plu, jk),
                __fadd_rn(__fadd_rn(1.0f, -nt_foealfa(NTC(ptu, jk))),
                          -__fadd_rn(1.0f, -nt_foealfa(NTC(ptu, jk + 1)))));
            NTC(ptu, jk) = __fadd_rn(NTC(ptu, jk),
                                     __fmul_rn(c.ralfdcp, NTC(plglac, jk)));
        }

        /* :2096-2179 */
        if (adjusted) {
            NTC(klab, jk) = 2;
            NTC(plu, jk) = __fadd_rn(__fadd_rn(NTC(plu, jk), zqold),
                                     -NTC(pqu, jk));
            const float zbc = __fmul_rn(NTC(ptu, jk), __fadd_rn(__fadd_rn(
                __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqu, jk))),
                -NTC(plu, jk + 1)), -zlrain_prev));
            const float zbe = __fmul_rn(NTC(ptenh, jk),
                __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqenh, jk))));
            zbuo_cur = __fadd_rn(zbc, -zbe);
            if (KT == 3 && NTC(klab, jk + 1) == 1) {
                if (zbuo_cur > -0.5f) {
                    LD = 1;
                    ktp = jk;
                    kup_cur = 0.5f;
                } else {
                    NTC(klab, jk) = 0;
                    NTC(pmfu, jk) = 0.0f;
                    NTC(plude, jk) = 0.0f;
                    NTC(plu, jk) = 0.0f;
                }
            }
            if (NTC(klab, jk + 1) == 2) {
                if (zbuo_cur < 0.0f) {
                    NTC(ptenh, jk) = __fmul_rn(0.5f,
                        __fadd_rn(NTC(pten, jk), NTC(pten, jk - 1)));
                    NTC(pqenh, jk) = __fmul_rn(0.5f,
                        __fadd_rn(NTC(pqen, jk), NTC(pqen, jk - 1)));
                    zbuo_cur = __fadd_rn(zbc, -__fmul_rn(NTC(ptenh, jk),
                        __fadd_rn(1.0f,
                                  __fmul_rn(c.vtmpc1, NTC(pqenh, jk)))));
                }
                const float zbuoc = __fmul_rn(__fadd_rn(
                    __fdiv_rn(zbuo_cur, __fmul_rn(NTC(ptenh, jk),
                        __fadd_rn(1.0f,
                                  __fmul_rn(c.vtmpc1, NTC(pqenh, jk))))),
                    __fdiv_rn(zbuo_prev, __fmul_rn(NTC(ptenh, jk + 1),
                        __fadd_rn(1.0f, __fmul_rn(c.vtmpc1,
                                                 NTC(pqenh, jk + 1)))))),
                    0.5f);
                const float zdkbuo = __fmul_rn(__fmul_rn(
                    __fadd_rn(NTC(pgeoh, jk), -NTC(pgeoh, jk + 1)),
                    zfacbuo), zbuoc);
                /* Both arms are the same shape on purpose: a runtime branch
                 * over differently-shaped arithmetic is what let ptxas
                 * contract two clones differently once before. */
                const float zsel = (zdmfen > 0.0f) ? zdmfen : zdmfde;
                const float zdken = fminf(1.0f, __fdiv_rn(
                    __fmul_rn(__fadd_rn(1.0f, z_cwdrag), zsel),
                    fmaxf(NT_CMFCMIN, NTC(pmfu, jk + 1))));
                kup_cur = __fdiv_rn(
                    __fadd_rn(__fmul_rn(kup_prev, __fadd_rn(1.0f, -zdken)),
                              zdkbuo),
                    __fadd_rn(1.0f, zdken));
                if (zbuo_cur < 0.0f) {
                    float zkedke =
                        __fdiv_rn(kup_cur, fmaxf(1.0e-10f, kup_prev));
                    zkedke = fmaxf(0.0f, fminf(1.0f, zkedke));
                    const float zmfun =
                        __fmul_rn(__fsqrt_rn(zkedke), NTC(pmfu, jk + 1));
                    zdmfde = fmaxf(zdmfde,
                                   __fadd_rn(NTC(pmfu, jk + 1), -zmfun));
                    NTC(plude, jk) = __fmul_rn(NTC(plu, jk + 1), zdmfde);
                    NTC(pmfu, jk) = __fadd_rn(
                        __fadd_rn(NTC(pmfu, jk + 1), zdmfen), -zdmfde);
                }
                if (zbuo_cur > -0.2f) {
                    const float rr = fminf(1.0f,
                        __fdiv_rn(NTC(pqen, jk - 1), NTC(pqsen, jk - 1)));
                    const float q3 = fminf(1.0f,
                        __fdiv_rn(NTC(pqsen, jk), NTC(pqsen, KB)));
                    zoentr = __fmul_rn(__fmul_rn(__fmul_rn(
                        __fmul_rn(1.75e-3f,
                                  __fadd_rn(0.3f, -__fadd_rn(rr, -1.0f))),
                        __fadd_rn(NTC(pgeoh, jk - 1), -NTC(pgeoh, jk))),
                        c.zrg), __fmul_rn(__fmul_rn(q3, q3), q3));
                    zoentr = __fmul_rn(fminf(0.4f, zoentr), NTC(pmfu, jk));
                } else {
                    zoentr = 0.0f;
                }
                if (jk > kdp) {
                    NTC(pmfu, jk) = NTC(pmfu, jk + 1);
                    kup_cur = 0.5f;
                }
                if (kup_cur > 0.0f && NTC(pmfu, jk) > 0.0f) {
                    ktp = jk;
                    llo1 = 1;
                } else {
                    NTC(klab, jk) = 0;
                    NTC(pmfu, jk) = 0.0f;
                    kup_cur = 0.0f;
                    zdmfde = NTC(pmfu, jk + 1);
                    NTC(plude, jk) = __fmul_rn(NTC(plu, jk + 1), zdmfde);
                }
                if (NTC(pmfu, jk + 1) > 0.0f) NTC(pmfude_rate, jk) = zdmfde;
            }
        } else if (loflag && KT == 2 && NTC(pqu, jk) == zqold) {
            NTC(klab, jk) = 0;
            NTC(pmfu, jk) = 0.0f;
            kup_cur = 0.0f;
            zdmfde = NTC(pmfu, jk + 1);
            NTC(plude, jk) = __fmul_rn(NTC(plu, jk + 1), zdmfde);
            NTC(pmfude_rate, jk) = zdmfde;
        }

        /* :2182-2216  precipitation conversion */
        if (llo1) {
            const float zdshrd = (lndj[i] == 1) ? 5.0e-4f : 3.0e-4f;
            if (NTC(plu, jk) > zdshrd) {
                const float zwu = fminf(15.0f, __fsqrt_rn(
                    __fmul_rn(2.0f, fmaxf(0.1f, kup_prev))));
                const float zprcon =
                    __fdiv_rn(zprcdgw, __fmul_rn(0.75f, zwu));
                const float zdt = fminf(NT_RTBER - NT_RTICE,
                    fmaxf(__fadd_rn(NT_RTBER, -NTC(ptu, jk)), 0.0f));
                const float zcbf =
                    __fadd_rn(1.0f, __fmul_rn(z_cprc2, __fsqrt_rn(zdt)));
                const float zzco = __fmul_rn(zprcon, zcbf);
                const float zlcrit = __fdiv_rn(zdshrd, zcbf);
                const float zdfi =
                    __fadd_rn(NTC(pgeoh, jk), -NTC(pgeoh, jk + 1));
                const float zc = __fadd_rn(NTC(plu, jk), -zluold);
                const float rq = __fdiv_rn(NTC(plu, jk), zlcrit);
                const float zarg = __fmul_rn(rq, rq);
                const float zd = (zarg < 25.0f)
                    ? __fmul_rn(__fmul_rn(zzco,
                          __fadd_rn(1.0f, -gfk_exp(-zarg))), zdfi)
                    : __fmul_rn(zzco, zdfi);
                const float zint = gfk_exp(-zd);
                float zlnew = __fadd_rn(__fmul_rn(zluold, zint),
                    __fmul_rn(__fdiv_rn(zc, zd), __fadd_rn(1.0f, -zint)));
                zlnew = fmaxf(0.0f, fminf(NTC(plu, jk), zlnew));
                zlnew = fminf(z_cldmax, zlnew);
                zprecip = fmaxf(0.0f,
                    __fadd_rn(__fadd_rn(zluold, zc), -zlnew));
                NTC(pdmfup, jk) = __fmul_rn(zprecip, NTC(pmfu, jk));
                zlrain_cur = __fadd_rn(zlrain_cur, zprecip);
                NTC(plu, jk) = zlnew;
            }
        }

        /* :2219-2236  rain fallout */
        if (llo1 && zlrain_cur > 0.0f) {
            const float zvw = __fmul_rn(21.18f, gfk_pow(zlrain_cur, 0.2f));
            const float zvi = __fmul_rn(z_cwifrac, zvw);
            const float zalfaw = nt_foealfa(NTC(ptu, jk));
            const float zvv = __fadd_rn(__fmul_rn(zalfaw, zvw),
                __fmul_rn(__fadd_rn(1.0f, -zalfaw), zvi));
            const float zrold = __fadd_rn(zlrain_cur, -zprecip);
            const float zwu = fminf(15.0f, __fsqrt_rn(
                __fmul_rn(2.0f, fmaxf(0.1f, kup_cur))));
            const float zd = __fdiv_rn(zvv, zwu);
            const float zint = gfk_exp(-zd);
            float zrnew = __fadd_rn(__fmul_rn(zrold, zint),
                __fmul_rn(__fdiv_rn(zprecip, zd), __fadd_rn(1.0f, -zint)));
            zrnew = fmaxf(0.0f, fminf(zlrain_cur, zrnew));
            zlrain_cur = zrnew;
        }

        /* :2239-2243 */
        if (loflag) {
            NTC(pmful, jk) = __fmul_rn(NTC(plu, jk), NTC(pmfu, jk));
            NTC(pmfus, jk) = __fmul_rn(
                __fadd_rn(__fmul_rn(c.cpd, NTC(ptu, jk)), NTC(pgeoh, jk)),
                NTC(pmfu, jk));
            NTC(pmfuq, jk) = __fmul_rn(NTC(pqu, jk), NTC(pmfu, jk));
        }

        kup_prev = kup_cur;
        zlrain_prev = zlrain_cur;
        zbuo_prev = zbuo_cur;
    }

    /* ---- 5. final (:2248-2256) ------------------------------------- */
    if (ktp == -1) LD = 0;
    KB = max(KB, ktp);
    if (LD) {
        wupa = fmaxf(1.0e-2f, __fdiv_rn(wupa, fmaxf(1.0f, zdpmean)));
        wupa = __fsqrt_rn(__fmul_rn(2.0f, wupa));
    }

    kctop[i] = ktp;
    kctop0[i] = kt0;
    wup[i] = wupa;

#undef LD
#undef KT
#undef KB
#undef MFUB
#undef NTC
}

/* =====================================================================
 * Stage 8: cudtdqn -- the heat and moisture tendencies (:3064-3148)
 * =====================================================================
 * Where the mass fluxes become RTHCUTEN and RQVCUTEN.
 *
 * ptent AND ptenq ARE ACCUMULATED INTO, NOT ASSIGNED (:3140-3141), and the
 * incoming array is NOT zero: measured at this routine's own entry capture,
 * non-zero on 4,428 of 5,292 rows, because cu_ntiedtke_run:273-276 seeds
 * them with the FORCING and :309-310 differences against saved copies so
 * only the convective increment escapes.  A kernel that assigned would be
 * wrong on 84% of the fixture.  See docs/ntiedtke/PORT-RECORD.md section 17.
 *
 * ktopm2 is a genuine intent(in) READ here, unlike in cuflxn where it is
 * overwritten before use -- so it IS a kernel argument.
 *
 * zdp, zdtdt and zdqdt are klon x klev locals in the reference.  zdtdt and
 * zdqdt are written in one loop and read in the next, but both loops run
 * over the same range in the same direction, so the three fuse into
 * registers.  0 B of scratch, as everywhere else in this port.
 */
extern "C" __global__ void ntiedtke_cudtdqn(
        const int *__restrict__ ldcum,
        const float *__restrict__ paph,      /* (nz+2, ncol), reads klev+1 */
        const float *__restrict__ pten,
        const float *__restrict__ plglac,
        const float *__restrict__ plude,
        const float *__restrict__ pmfus,
        const float *__restrict__ pmfds,
        const float *__restrict__ pmfuq,
        const float *__restrict__ pmfdq,
        const float *__restrict__ pmful,
        const float *__restrict__ pdmfup,
        const float *__restrict__ pdmfdp,
        const float *__restrict__ pdpmel,
        float *__restrict__ ptent,           /* ACCUMULATED, in and out */
        float *__restrict__ ptenq,
        float *__restrict__ pcte,
        int ncol, int klev, int ktopm2,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUDTDQN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUDTDQN, order_report, ticket);
    if (!ldcum[i]) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    for (int jk = ktopm2; jk <= klev; ++jk) {
        const float zdp = __fdiv_rn(
            c.g, __fadd_rn(NTC(paph, jk + 1), -NTC(paph, jk)));
        const float zalv = nt_foelhm(NTC(pten, jk), c);
        float zdtdt, zdqdt;
        if (jk < klev) {
            const float inner = __fadd_rn(__fadd_rn(__fadd_rn(
                __fadd_rn(NTC(pmful, jk + 1), -NTC(pmful, jk)),
                -NTC(plude, jk)), -NTC(pdmfup, jk)), -NTC(pdmfdp, jk));
            const float big = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(
                __fadd_rn(__fadd_rn(NTC(pmfus, jk + 1), -NTC(pmfus, jk)),
                          NTC(pmfds, jk + 1)), -NTC(pmfds, jk)),
                __fmul_rn(c.alf, NTC(plglac, jk))),
                -__fmul_rn(c.alf, NTC(pdpmel, jk))),
                -__fmul_rn(zalv, inner));
            zdtdt = __fmul_rn(__fmul_rn(zdp, c.rcpd), big);
            /* :3117-3119 chains all NINE terms left to right -- there
             * are no parentheses inside, so the last two do not group. */
            zdqdt = __fmul_rn(zdp, __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(
                __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(NTC(pmfuq, jk + 1),
                    -NTC(pmfuq, jk)), NTC(pmfdq, jk + 1)),
                    -NTC(pmfdq, jk)), NTC(pmful, jk + 1)),
                    -NTC(pmful, jk)), -NTC(plude, jk)),
                    -NTC(pdmfup, jk)), -NTC(pdmfdp, jk)));
        } else {
            const float big = __fadd_rn(__fadd_rn(
                __fadd_rn(NTC(pmfus, jk), NTC(pmfds, jk)),
                __fmul_rn(c.alf, NTC(pdpmel, jk))),
                -__fmul_rn(zalv, __fadd_rn(__fadd_rn(
                    __fadd_rn(NTC(pmful, jk), NTC(pdmfup, jk)),
                    NTC(pdmfdp, jk)), NTC(plude, jk))));
            zdtdt = -__fmul_rn(__fmul_rn(zdp, c.rcpd), big);
            zdqdt = -__fmul_rn(zdp, __fadd_rn(
                __fadd_rn(__fadd_rn(NTC(pmfuq, jk), NTC(plude, jk)),
                          NTC(pmfdq, jk)),
                __fadd_rn(__fadd_rn(NTC(pmful, jk), NTC(pdmfup, jk)),
                          NTC(pdmfdp, jk))));
        }
        /* ADD.  See the header. */
        NTC(ptent, jk) = __fadd_rn(NTC(ptent, jk), zdtdt);
        NTC(ptenq, jk) = __fadd_rn(NTC(ptenq, jk), zdqdt);
        NTC(pcte, jk) = __fmul_rn(zdp, NTC(plude, jk));
    }
#undef NTC
}

/* =====================================================================
 * Stage 9: cududvn -- the momentum tendencies (:3152-3252)
 * =====================================================================
 * The last routine, and the one that consumes puu/pvu/pud/pvd -- which
 * cuascn, cudlfsn and cuddrafn were each separately found never to write.
 * cuinin sets them, nothing between touches them, this reads them.
 *
 * ptenu/ptenv ARE ACCUMULATED INTO (:3245-3246).  Unlike cudtdqn's pair
 * they are seeded ZERO by cu_ntiedtke_run:258-259, so accumulate and
 * replace coincide here -- which is exactly why the parity test proves the
 * ADD by perturbing the seed rather than by comparing to the oracle: with a
 * zero seed an assigning kernel produces identical bytes.
 *
 * pmfu/pmfd MUST BE THE SCALED PAIR.  cumastrn:833-915 rescales into
 * zmfuus/zmfdus and it is those that reach here; the unscaled pair would be
 * wrong on exactly the columns the rescaling touched.
 *
 * THIS IS THE PORT'S FIRST AND ONLY SCRATCH ALLOCATION, and it is real
 * rather than avoidable.  The four zmf* arrays are klon x klev and are NOT
 * single-level lookback: the below-cloud taper reads zmf*[kcbot] at every
 * jk > kcbot, and the tendency loop then reads jk+1 after the taper has
 * rewritten it.  A register form would need the whole column.  Four
 * (nz+2, ncol) arrays, which the caller owns -- so the FRAME is still 0 B
 * and the cost is priced in the caller's allocation, where docs/ntiedtke/PORT-RECORD.md
 * section 13 can see it.
 */
extern "C" __global__ void ntiedtke_cududvn(
        const int *__restrict__ ldcum,
        const int *__restrict__ ktype,
        const int *__restrict__ kcbot,
        const float *__restrict__ paph,      /* reads klev+1 */
        const float *__restrict__ puen,
        const float *__restrict__ pven,
        const float *__restrict__ pmfu,      /* SCALED */
        const float *__restrict__ pmfd,      /* SCALED */
        const float *__restrict__ puu,
        const float *__restrict__ pud,
        const float *__restrict__ pvu,
        const float *__restrict__ pvd,
        float *__restrict__ ptenu,           /* ACCUMULATED, in and out */
        float *__restrict__ ptenv,
        float *__restrict__ zmfuu,           /* (nz+2, ncol) scratch */
        float *__restrict__ zmfuv,
        float *__restrict__ zmfdu,
        float *__restrict__ zmfdv,
        int ncol, int klev, int ktopm2,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUDUDVN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUDUDVN, order_report, ticket);
    if (!ldcum[i]) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]
    const int kb = kcbot[i];
    const int kt3 = (ktype[i] == 3);

    /* 1.0 the fluxes (:3191-3201) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        const int ik = jk - 1;
        NTC(zmfuu, jk) = __fmul_rn(NTC(pmfu, jk),
            __fadd_rn(NTC(puu, jk), -NTC(puen, ik)));
        NTC(zmfuv, jk) = __fmul_rn(NTC(pmfu, jk),
            __fadd_rn(NTC(pvu, jk), -NTC(pven, ik)));
        NTC(zmfdu, jk) = __fmul_rn(NTC(pmfd, jk),
            __fadd_rn(NTC(pud, jk), -NTC(puen, ik)));
        NTC(zmfdv, jk) = __fmul_rn(NTC(pmfd, jk),
            __fadd_rn(NTC(pvd, jk), -NTC(pven, ik)));
    }

    /* linear fluxes below cloud (:3203-3215) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (jk <= kb) continue;
        float zzp = __fdiv_rn(__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)),
                              __fadd_rn(NTC(paph, klev + 1), -NTC(paph, kb)));
        if (kt3) zzp = __fmul_rn(zzp, zzp);
        NTC(zmfuu, jk) = __fmul_rn(NTC(zmfuu, kb), zzp);
        NTC(zmfuv, jk) = __fmul_rn(NTC(zmfuv, kb), zzp);
        NTC(zmfdu, jk) = __fmul_rn(NTC(zmfdu, kb), zzp);
        NTC(zmfdv, jk) = __fmul_rn(NTC(zmfdv, kb), zzp);
    }

    /* 2.0 and 3.0 fused (:3219-3249).  ADD, do not assign. */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        const float zdp = __fdiv_rn(
            c.g, __fadd_rn(NTC(paph, jk + 1), -NTC(paph, jk)));
        float zdudt, zdvdt;
        if (jk < klev) {
            const int ik = jk + 1;
            zdudt = __fmul_rn(zdp, __fadd_rn(__fadd_rn(
                __fadd_rn(NTC(zmfuu, ik), -NTC(zmfuu, jk)), NTC(zmfdu, ik)),
                -NTC(zmfdu, jk)));
            zdvdt = __fmul_rn(zdp, __fadd_rn(__fadd_rn(
                __fadd_rn(NTC(zmfuv, ik), -NTC(zmfuv, jk)), NTC(zmfdv, ik)),
                -NTC(zmfdv, jk)));
        } else {
            zdudt = -__fmul_rn(zdp,
                __fadd_rn(NTC(zmfuu, jk), NTC(zmfdu, jk)));
            zdvdt = -__fmul_rn(zdp,
                __fadd_rn(NTC(zmfuv, jk), NTC(zmfdv, jk)));
        }
        NTC(ptenu, jk) = __fadd_rn(NTC(ptenu, jk), zdudt);
        NTC(ptenv, jk) = __fadd_rn(NTC(ptenv, jk), zdvdt);
    }
#undef NTC
}

/* cuadjtqn's kcall == 2 arm (:3359-3379), in place at one level.
 *
 * THE DOWNDRAFT ARM: evaporation, so both condensate steps clamp with
 * fminf(.,0) rather than fmaxf.  It computes saturation through foeewm,
 * NOT inline off reciprocals the way the kcall == 1 arm does -- the two are
 * different expressions of the same quantity and are not interchangeable at
 * max_ulp == 0.
 *
 * The `fabsf(zcond) < 1e-20` guard on the second clamp is transcribed as
 * written: it is not `zcond == 0`, and on a column where zcond is a
 * denormal the two differ.
 */
__device__ __forceinline__ void nt_cuadjtqn2(float *pt, float *pq,
                                             float psp, const NtConst &c) {
    const float zqp = __fdiv_rn(1.0f, psp);
    float zqsat = __fmul_rn(nt_foeewm(*pt, c), zqp);
    zqsat = fminf(0.5f, zqsat);
    float zcor = __fdiv_rn(1.0f, __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zqsat)));
    zqsat = __fmul_rn(zqsat, zcor);
    float zcond = __fdiv_rn(
        __fadd_rn(*pq, -zqsat),
        __fadd_rn(1.0f, __fmul_rn(__fmul_rn(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    zcond = fminf(zcond, 0.0f);
    *pt = __fadd_rn(*pt, __fmul_rn(nt_foeldcpm(*pt, c), zcond));
    *pq = __fadd_rn(*pq, -zcond);
    zqsat = __fmul_rn(nt_foeewm(*pt, c), zqp);
    zqsat = fminf(0.5f, zqsat);
    zcor = __fdiv_rn(1.0f, __fadd_rn(1.0f, -__fmul_rn(c.vtmpc1, zqsat)));
    zqsat = __fmul_rn(zqsat, zcor);
    float zcond1 = __fdiv_rn(
        __fadd_rn(*pq, -zqsat),
        __fadd_rn(1.0f, __fmul_rn(__fmul_rn(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    if (fabsf(zcond) < 1.0e-20f) zcond1 = fminf(zcond1, 0.0f);
    *pt = __fadd_rn(*pt, __fmul_rn(nt_foeldcpm(*pt, c), zcond1));
    *pq = __fadd_rn(*pq, -zcond1);
}

/* =====================================================================
 * Stage 10: cudlfsn -- the level of free sinking (:2262-2487)
 * =====================================================================
 * Where downdrafts start.
 *
 * ALL SIX LEVEL OUTPUTS ARE CLASS 2 -- ptd, pqd, pmfd, pmfds, pmfdq and
 * pdmfdp are each written at exactly ONE line, inside the LFS branch, so
 * every level the routine does not reach keeps the caller's value.  THIS
 * KERNEL MUST NOT ZERO THEM.  Zeroing outputs at entry is the reflex in
 * almost every CUDA kernel ever written and it is wrong here on every level
 * but one.  The mirror learned that by being wrong on levels 1-4 of every
 * column; see docs/ntiedtke/PORT-RECORD.md section 14.
 *
 * pud and pvd are dummies the routine never mentions -- downdraft momentum
 * is cududvn's -- so they are not arguments at all.
 *
 * ztenwb/zqenwb are klon x klev locals, but each is written, adjusted and
 * read within ONE iteration, so they are registers.  0 B frame.
 *
 * The `is == 0 cycle` at :2448 is a horizontal reduction and it is INERT:
 * ztenwb/zqenwb/zph are set for every column before it, cuadjtqn is masked
 * by the same per-column llo2 the reduction sums, and everything after is
 * inside if(llo2).  So the per-column form below is exact, and unlike
 * cuascn's llo3 that was derived rather than assumed.
 */
#define NT_CMFDEPS 0.30f

extern "C" __global__ void ntiedtke_cudlfsn(
        const int *__restrict__ ldcum,
        const int *__restrict__ kcbot,
        const int *__restrict__ kctop,
        const float *__restrict__ ptenh,
        const float *__restrict__ pqenh,
        const float *__restrict__ pten,
        const float *__restrict__ pqsen,
        const float *__restrict__ pgeo,
        const float *__restrict__ pgeoh,
        const float *__restrict__ paph,
        const float *__restrict__ ptu,
        const float *__restrict__ pqu,
        const float *__restrict__ pmfub,
        float *__restrict__ ptd,             /* CLASS 2 -- in AND out */
        float *__restrict__ pqd,
        float *__restrict__ pmfd,
        float *__restrict__ pmfds,
        float *__restrict__ pmfdq,
        float *__restrict__ pdmfdp,
        float *__restrict__ prfl,            /* UPDATED */
        int *__restrict__ kdtop,
        int *__restrict__ lddraf,
        int ncol, int klev,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUDLFSN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUDLFSN, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    /* 1. defaults (:2393-2398).  Only the two scalars; the level arrays
     * are class 2 and are deliberately left alone. */
    int ld_out = 0;
    int kdt = klev + 1;
    kdtop[i] = kdt;
    lddraf[i] = 0;
    if (!ldcum[i]) return;

    const int kb = kcbot[i];
    const int ktop = kctop[i];
    float rfl = prfl[i];

    /* 2. the level of minimum saturation moist static energy (:2422-2431) */
    int ikhsmin = klev + 1;
    float zhsmin = 1.0e8f;
    for (int jk = 3; jk <= klev - 2; ++jk) {
        const float zhsk = __fadd_rn(
            __fadd_rn(__fmul_rn(c.cpd, NTC(pten, jk)), NTC(pgeo, jk)),
            __fmul_rn(nt_foelhm(NTC(pten, jk), c), NTC(pqsen, jk)));
        if (zhsk < zhsmin) { zhsmin = zhsk; ikhsmin = jk; }
    }

    /* 2.1-2.2 the descent (:2435-2484) */
    const int ike = klev - 3;
    for (int jk = 3; jk <= ike; ++jk) {
        float ztenwb = NTC(ptenh, jk);
        float zqenwb = NTC(pqenh, jk);
        const float zph = NTC(paph, jk);
        const bool llo2 = (rfl > 0.0f) && !ld_out
                          && (jk < kb) && (jk > ktop) && (jk >= ikhsmin);
        if (!llo2) continue;

        nt_cuadjtqn2(&ztenwb, &zqenwb, zph, c);

        const float zttest = __fmul_rn(0.5f,
            __fadd_rn(NTC(ptu, jk), ztenwb));
        const float zqtest = __fmul_rn(0.5f,
            __fadd_rn(NTC(pqu, jk), zqenwb));
        const float zbuo = __fadd_rn(
            __fmul_rn(zttest, __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, zqtest))),
            -__fmul_rn(NTC(ptenh, jk),
                __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqenh, jk)))));
        const float zcond = __fadd_rn(NTC(pqenh, jk), -zqenwb);
        const float zmftop = -__fmul_rn(NT_CMFDEPS, pmfub[i]);
        if (zbuo < 0.0f
            && rfl > __fmul_rn(__fmul_rn(10.0f, zmftop), zcond)) {
            kdt = jk;
            ld_out = 1;
            NTC(ptd, jk) = zttest;
            NTC(pqd, jk) = zqtest;
            NTC(pmfd, jk) = zmftop;
            NTC(pmfds, jk) = __fmul_rn(NTC(pmfd, jk),
                __fadd_rn(__fmul_rn(c.cpd, NTC(ptd, jk)), NTC(pgeoh, jk)));
            NTC(pmfdq, jk) = __fmul_rn(NTC(pmfd, jk), NTC(pqd, jk));
            NTC(pdmfdp, jk - 1) = __fmul_rn(-0.5f,
                __fmul_rn(NTC(pmfd, jk), zcond));
            rfl = __fadd_rn(rfl, NTC(pdmfdp, jk - 1));
        }
    }

    prfl[i] = rfl;
    kdtop[i] = kdt;
    lddraf[i] = ld_out;
#undef NTC
}

/* =====================================================================
 * Stage 11: cuddrafn -- the moist downdraft descent (:2495-2721)
 * =====================================================================
 * SIX CLASS-1 DUMMIES -- prfl, ptd, pqd, pmfd, pmfds, pmfdq -- every one
 * read at jk-1 before jk is written.  They are cudlfsn's outputs and this
 * kernel reads them straight out of the same arrays, which is exactly the
 * aliasing the Fortran performs.
 *
 * paph[klev+1] IS READ, three times (:2618, :2648, :2649).  That is the
 * surface interface cuascn never touches.
 *
 * pud/pvd are never written here either.
 *
 * zoentr, zbuoy, zdmfen, zdmfde and zcond are per-column locals that persist
 * across levels -- registers.  pmfdde_rate is a real output.  0 B frame.
 */
#define NT_ENTRDD 2.0e-4f

extern "C" __global__ void ntiedtke_cuddrafn(
        const int *__restrict__ lddraf,
        const float *__restrict__ ptenh,
        const float *__restrict__ pqenh,
        const float *__restrict__ pgeo,
        const float *__restrict__ pgeoh,
        const float *__restrict__ paph,      /* reads klev+1 */
        const float *__restrict__ pmfu,
        float *__restrict__ ptd,             /* CLASS 1 -- in AND out */
        float *__restrict__ pqd,
        float *__restrict__ pmfd,
        float *__restrict__ pmfds,
        float *__restrict__ pmfdq,
        float *__restrict__ pdmfdp,
        float *__restrict__ pmfdde_rate,
        float *__restrict__ prfl,            /* UPDATED */
        int ncol, int klev,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUDDRAFN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUDDRAFN, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    float zoentr = 0.0f, zbuoy = 0.0f;
    float zdmfen = 0.0f, zdmfde = 0.0f;
    float rfl = prfl[i];
    const int ld = lddraf[i];

    /* itopde (:2616-2621).  The loop DESCENDS and the last assignment wins,
     * so itopde ends at the TOPMOST level within 60 hPa of the surface. */
    int itopde = -1;
    for (int jk = klev; jk >= 1; --jk) {
        NTC(pmfdde_rate, jk) = 0.0f;
        if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f)
            itopde = jk;
    }
    if (itopde < 0) return;          /* never set: nothing to descend into */

    for (int jk = 3; jk <= klev; ++jk) {
        const float zph = NTC(paph, jk);
        if (!(ld && NTC(pmfd, jk - 1) < 0.0f)) continue;

        const float zentr = __fmul_rn(__fmul_rn(
            __fmul_rn(NT_ENTRDD, NTC(pmfd, jk - 1)),
            __fadd_rn(NTC(pgeoh, jk - 1), -NTC(pgeoh, jk))), c.zrg);
        zdmfen = zentr;
        zdmfde = zentr;

        if (jk > itopde) {
            zdmfen = 0.0f;
            zdmfde = __fdiv_rn(
                __fmul_rn(NTC(pmfd, itopde),
                          __fadd_rn(NTC(paph, jk), -NTC(paph, jk - 1))),
                __fadd_rn(NTC(paph, klev + 1), -NTC(paph, itopde)));
        }
        if (jk <= itopde) {
            const float zdz = -__fmul_rn(
                __fadd_rn(NTC(pgeoh, jk - 1), -NTC(pgeoh, jk)), c.zrg);
            const float zzentr = __fmul_rn(__fmul_rn(zoentr, zdz),
                                           NTC(pmfd, jk - 1));
            zdmfen = __fadd_rn(zdmfen, zzentr);
            zdmfen = fmaxf(zdmfen, __fmul_rn(0.3f, NTC(pmfd, jk - 1)));
            zdmfen = fmaxf(zdmfen,
                __fadd_rn(-__fmul_rn(0.75f, NTC(pmfu, jk)),
                          -__fadd_rn(NTC(pmfd, jk - 1), -zdmfde)));
            zdmfen = fminf(zdmfen, 0.0f);
        }

        NTC(pmfd, jk) = __fadd_rn(
            __fadd_rn(NTC(pmfd, jk - 1), zdmfen), -zdmfde);
        const float zseen = __fmul_rn(
            __fadd_rn(__fmul_rn(c.cpd, NTC(ptenh, jk - 1)),
                      NTC(pgeoh, jk - 1)), zdmfen);
        const float zqeen = __fmul_rn(NTC(pqenh, jk - 1), zdmfen);
        const float zsdde = __fmul_rn(
            __fadd_rn(__fmul_rn(c.cpd, NTC(ptd, jk - 1)),
                      NTC(pgeoh, jk - 1)), zdmfde);
        const float zqdde = __fmul_rn(NTC(pqd, jk - 1), zdmfde);
        const float zmfdsk = __fadd_rn(
            __fadd_rn(NTC(pmfds, jk - 1), zseen), -zsdde);
        const float zmfdqk = __fadd_rn(
            __fadd_rn(NTC(pmfdq, jk - 1), zqeen), -zqdde);
        const float inv = __fdiv_rn(1.0f,
                                    fminf(-NT_CMFCMIN, NTC(pmfd, jk)));
        NTC(pqd, jk) = __fmul_rn(zmfdqk, inv);
        NTC(ptd, jk) = __fmul_rn(
            __fadd_rn(__fmul_rn(zmfdsk, inv), -NTC(pgeoh, jk)), c.rcpd);
        NTC(ptd, jk) = fminf(400.0f, NTC(ptd, jk));
        NTC(ptd, jk) = fmaxf(100.0f, NTC(ptd, jk));
        float zcond = NTC(pqd, jk);

        nt_cuadjtqn2(&NTC(ptd, jk), &NTC(pqd, jk), zph, c);

        zcond = __fadd_rn(zcond, -NTC(pqd, jk));
        float zbuo = __fadd_rn(
            __fmul_rn(NTC(ptd, jk),
                __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqd, jk)))),
            -__fmul_rn(NTC(ptenh, jk),
                __fadd_rn(1.0f, __fmul_rn(c.vtmpc1, NTC(pqenh, jk)))));
        if (rfl > 0.0f && NTC(pmfu, jk) > 0.0f) {
            const float zrain = __fdiv_rn(rfl, NTC(pmfu, jk));
            zbuo = __fadd_rn(zbuo, -__fmul_rn(NTC(ptd, jk), zrain));
        }
        if (zbuo >= 0.0f || rfl <= __fmul_rn(NTC(pmfd, jk), zcond)) {
            NTC(pmfd, jk) = 0.0f;
            zbuo = 0.0f;
        }
        NTC(pmfds, jk) = __fmul_rn(
            __fadd_rn(__fmul_rn(c.cpd, NTC(ptd, jk)), NTC(pgeoh, jk)),
            NTC(pmfd, jk));
        NTC(pmfdq, jk) = __fmul_rn(NTC(pqd, jk), NTC(pmfd, jk));
        const float zdmfdp = -__fmul_rn(NTC(pmfd, jk), zcond);
        NTC(pdmfdp, jk - 1) = zdmfdp;
        rfl = __fadd_rn(rfl, zdmfdp);

        /* organised entrainment for the next level down */
        float zbuoyz = __fdiv_rn(zbuo, NTC(ptenh, jk));
        zbuoyz = fminf(zbuoyz, 0.0f);
        const float zdz2 = -__fadd_rn(NTC(pgeo, jk - 1), -NTC(pgeo, jk));
        zbuoy = __fadd_rn(zbuoy, __fmul_rn(zbuoyz, zdz2));
        zoentr = __fdiv_rn(__fmul_rn(__fmul_rn(c.g, zbuoyz), 0.5f),
                           __fadd_rn(1.0f, zbuoy));
        NTC(pmfdde_rate, jk) = -zdmfde;
    }

    prfl[i] = rfl;
#undef NTC
}

/* =====================================================================
 * Stage 12: cuflxn -- the final convective fluxes (:2725-3060)
 * =====================================================================
 * Turns the mass fluxes into rain and snow: the flux-form anomalies, the
 * cloud-base taper, snow melt, and evaporation of falling precipitation.
 *
 * ktopm2 IS NOT AN ARGUMENT, and that is the point.  cumastrn:565 sets
 * itopm2 = kctop(jl) INSIDE a do-jl loop, so the value that survives is the
 * LAST column's cloud top -- a genuine horizontal leak, passed to the
 * reference as intent(inout).  But :2877 sets ktopm2 = 2 unconditionally at
 * routine top level, no line between entry and there reads it, and cuflxn
 * runs before cudtdqn and cududvn, its only other consumers.  The leaked
 * value is DEAD.  Re-derived from source rather than inherited, because the
 * sibling column-independence claim in section 2 did not survive the same
 * scrutiny.
 *
 * FOUR CLASS-1 DUMMIES -- lddraf, ktype, pmfu, pmfd -- and beyond those,
 * pmfus/pmfuq/pmfds/pmfdq/plglac/pqsen/pdmfup/pdmfdp/pmfdde_rate are all
 * rewritten IN PLACE off their incoming values.  plglac and pmfdde_rate are
 * additionally declared WITHOUT an intent attribute (:2838), which made
 * them invisible to three of the aliasing audit's four reports; the fourth
 * report exists because of them.  Nothing here may be zeroed at entry.
 *
 * pmflxr and pmflxs are klev+1 ARRAYS: the loops write jk+1 up to klev+1,
 * and that surface slot is the scheme's actual surface rain and snow flux.
 * They must persist BETWEEN the melt loop and the evaporation loop -- the
 * second reads at jk what the first wrote at (jk-1)+1 -- so they are real
 * arrays rather than registers.  They are outputs anyway.
 *
 * A THIRD zcons2.  :2857 declares zcons2 = 3/(g*dt) -- numerically equal to
 * cuascn's and different from the closure's, under the same name for the
 * third time in one file.
 */
#define NT_ZTAUMEL 18000.0f
#define NT_ZCUCOV  0.05f

extern "C" __global__ void ntiedtke_cuflxn(
        const int *__restrict__ kcbot,
        const int *__restrict__ kctop,
        const int *__restrict__ kdtop,
        const int *__restrict__ lndj,
        const float *__restrict__ pten,
        const float *__restrict__ ptenh,
        const float *__restrict__ pqenh,
        const float *__restrict__ paph,      /* reads klev+1 */
        const float *__restrict__ pap,
        const float *__restrict__ pqen,
        const float *__restrict__ pgeoh,
        int *__restrict__ ldcum,             /* scalars, in AND out */
        int *__restrict__ lddraf,
        int *__restrict__ ktype,
        float *__restrict__ pmfu,            /* CLASS 1 -- in AND out */
        float *__restrict__ pmfd,
        float *__restrict__ pmfus,
        float *__restrict__ pmfds,
        float *__restrict__ pmfuq,
        float *__restrict__ pmfdq,
        float *__restrict__ pmful,
        float *__restrict__ plude,
        float *__restrict__ plglac,          /* no intent attribute */
        float *__restrict__ pdmfup,
        float *__restrict__ pdmfdp,
        float *__restrict__ pmfdde_rate,     /* no intent attribute */
        float *__restrict__ pqsen,           /* rewritten at :2984 */
        float *__restrict__ pdpmel,
        float *__restrict__ pmflxr,          /* (nz+2, ncol), klev+1 valid */
        float *__restrict__ pmflxs,
        float *__restrict__ prain,
        int ncol, int klev, float ztmst,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_CUFLXN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_CUFLXN, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    const float zcons1a = __fdiv_rn(
        c.cpd, __fmul_rn(__fmul_rn(c.alf, c.g), NT_ZTAUMEL));
    const float zcons2 = __fdiv_rn(3.0f, __fmul_rn(c.g, ztmst));
    const float zcpecons = __fdiv_rn(5.44e-4f, c.g);

    /* 1.0 (:2865-2875) */
    float pr = 0.0f;
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int kdt = kdtop[i];
    const int ld = ldcum[i];
    int ldd = lddraf[i];
    if (!ld || kdt < ktop) ldd = 0;
    if (!ld) ktype[i] = 0;
    const int kt3 = (ktype[i] == 3);
    int idbas = klev;
    const float rhevap = (lndj[i] == 1) ? 0.7f : 0.9f;

    const int ktopm2 = 2;             /* :2877, unconditional.  See header. */

    for (int jk = ktopm2; jk <= klev; ++jk) {
        const int ikb = min(jk + 1, klev);
        NTC(pmflxr, jk) = 0.0f;
        NTC(pmflxs, jk) = 0.0f;
        NTC(pdpmel, jk) = 0.0f;
        if (ld && jk >= ktop) {
            NTC(pmfus, jk) = __fadd_rn(NTC(pmfus, jk),
                -__fmul_rn(NTC(pmfu, jk),
                    __fadd_rn(__fmul_rn(c.cpd, NTC(ptenh, jk)),
                              NTC(pgeoh, jk))));
            NTC(pmfuq, jk) = __fadd_rn(NTC(pmfuq, jk),
                -__fmul_rn(NTC(pmfu, jk), NTC(pqenh, jk)));
            NTC(plglac, jk) = __fmul_rn(NTC(pmfu, jk), NTC(plglac, jk));
            const int llddraf = ldd && (jk >= kdt);
            if (llddraf && jk >= kdt) {
                NTC(pmfds, jk) = __fadd_rn(NTC(pmfds, jk),
                    -__fmul_rn(NTC(pmfd, jk),
                        __fadd_rn(__fmul_rn(c.cpd, NTC(ptenh, jk)),
                                  NTC(pgeoh, jk))));
                NTC(pmfdq, jk) = __fadd_rn(NTC(pmfdq, jk),
                    -__fmul_rn(NTC(pmfd, jk), NTC(pqenh, jk)));
            } else {
                NTC(pmfd, jk) = 0.0f;
                NTC(pmfds, jk) = 0.0f;
                NTC(pmfdq, jk) = 0.0f;
                NTC(pdmfdp, jk - 1) = 0.0f;
            }
            if (llddraf && NTC(pmfd, jk) < 0.0f
                && fabsf(NTC(pmfd, ikb)) < 1.0e-20f) {
                idbas = jk;
            }
        } else {
            NTC(pmfu, jk) = 0.0f;  NTC(pmfd, jk) = 0.0f;
            NTC(pmfus, jk) = 0.0f; NTC(pmfds, jk) = 0.0f;
            NTC(pmfuq, jk) = 0.0f; NTC(pmfdq, jk) = 0.0f;
            NTC(pmful, jk) = 0.0f; NTC(plglac, jk) = 0.0f;
            NTC(pdmfup, jk - 1) = 0.0f;
            NTC(pdmfdp, jk - 1) = 0.0f;
            NTC(plude, jk - 1) = 0.0f;
        }
    }

    NTC(pmflxr, klev + 1) = 0.0f;
    NTC(pmflxs, klev + 1) = 0.0f;

    /* the cloud-base taper (:2926-2938) */
    if (ld) {
        const int ikb = kb;
        const int ik = ikb + 1;
        float zzp = __fdiv_rn(
            __fadd_rn(NTC(paph, klev + 1), -NTC(paph, ik)),
            __fadd_rn(NTC(paph, klev + 1), -NTC(paph, ikb)));
        if (kt3) zzp = __fmul_rn(zzp, zzp);
        NTC(pmfu, ik) = __fmul_rn(NTC(pmfu, ikb), zzp);
        NTC(pmfus, ik) = __fmul_rn(
            __fadd_rn(NTC(pmfus, ikb),
                -__fmul_rn(nt_foelhm(NTC(ptenh, ikb), c), NTC(pmful, ikb))),
            zzp);
        NTC(pmfuq, ik) = __fmul_rn(
            __fadd_rn(NTC(pmfuq, ikb), NTC(pmful, ikb)), zzp);
        NTC(pmful, ik) = 0.0f;
    }

    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (ld && jk > kb + 1) {
            const int ikb = kb + 1;
            float zzp = __fdiv_rn(
                __fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)),
                __fadd_rn(NTC(paph, klev + 1), -NTC(paph, ikb)));
            if (kt3) zzp = __fmul_rn(zzp, zzp);
            NTC(pmfu, jk) = __fmul_rn(NTC(pmfu, ikb), zzp);
            NTC(pmfus, jk) = __fmul_rn(NTC(pmfus, ikb), zzp);
            NTC(pmfuq, jk) = __fmul_rn(NTC(pmfuq, ikb), zzp);
            NTC(pmful, jk) = 0.0f;
        }
        const int ik = idbas;
        const int llddraf = ldd && (jk > ik) && (ik < klev);
        if (llddraf && ik == kb + 1) {
            float zzp = __fdiv_rn(
                __fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)),
                __fadd_rn(NTC(paph, klev + 1), -NTC(paph, ik)));
            if (kt3) zzp = __fmul_rn(zzp, zzp);
            NTC(pmfd, jk) = __fmul_rn(NTC(pmfd, ik), zzp);
            NTC(pmfds, jk) = __fmul_rn(NTC(pmfds, ik), zzp);
            NTC(pmfdq, jk) = __fmul_rn(NTC(pmfdq, ik), zzp);
            NTC(pmfdde_rate, jk) =
                -__fadd_rn(NTC(pmfd, jk - 1), -NTC(pmfd, jk));
        } else if (llddraf && ik != kb + 1 && jk == ik + 1) {
            NTC(pmfdde_rate, jk) =
                -__fadd_rn(NTC(pmfd, jk - 1), -NTC(pmfd, jk));
        }
    }

    /* 2. melting and the rain/snow split (:2975-3011) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (!(ld && jk >= ktop - 1)) continue;
        pr = __fadd_rn(pr, NTC(pdmfup, jk));
        if (NTC(pmflxs, jk) > 0.0f && NTC(pten, jk) > NT_TMELT) {
            const float zcons1 = __fmul_rn(zcons1a, __fadd_rn(1.0f,
                __fmul_rn(0.5f, __fadd_rn(NTC(pten, jk), -NT_TMELT))));
            const float zfac = __fmul_rn(zcons1,
                __fadd_rn(NTC(paph, jk + 1), -NTC(paph, jk)));
            const float zsnmlt = fminf(NTC(pmflxs, jk),
                __fmul_rn(zfac, __fadd_rn(NTC(pten, jk), -NT_TMELT)));
            NTC(pdpmel, jk) = zsnmlt;
            NTC(pqsen, jk) = __fdiv_rn(
                nt_foeewm(__fadd_rn(NTC(pten, jk),
                                    -__fdiv_rn(zsnmlt, zfac)), c),
                NTC(pap, jk));
        }
        float zalfaw = nt_foealfa(NTC(pten, jk));
        /* No liquid precipitation above the melting level. */
        if (NTC(pten, jk) < NT_TMELT && zalfaw > 0.0f) {
            NTC(plglac, jk) = __fadd_rn(NTC(plglac, jk),
                __fmul_rn(zalfaw,
                    __fadd_rn(NTC(pdmfup, jk), NTC(pdmfdp, jk))));
            zalfaw = 0.0f;
        }
        NTC(pmflxr, jk + 1) = __fadd_rn(
            __fadd_rn(NTC(pmflxr, jk),
                __fmul_rn(zalfaw,
                    __fadd_rn(NTC(pdmfup, jk), NTC(pdmfdp, jk)))),
            NTC(pdpmel, jk));
        NTC(pmflxs, jk + 1) = __fadd_rn(
            __fadd_rn(NTC(pmflxs, jk),
                __fmul_rn(__fadd_rn(1.0f, -zalfaw),
                    __fadd_rn(NTC(pdmfup, jk), NTC(pdmfdp, jk)))),
            -NTC(pdpmel, jk));
        if (__fadd_rn(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)) < 0.0f) {
            NTC(pdmfdp, jk) = -__fadd_rn(
                __fadd_rn(NTC(pmflxr, jk), NTC(pmflxs, jk)),
                NTC(pdmfup, jk));
            NTC(pmflxr, jk + 1) = 0.0f;
            NTC(pmflxs, jk + 1) = 0.0f;
            NTC(pdpmel, jk) = 0.0f;
        } else if (NTC(pmflxr, jk + 1) < 0.0f) {
            NTC(pmflxs, jk + 1) =
                __fadd_rn(NTC(pmflxs, jk + 1), NTC(pmflxr, jk + 1));
            NTC(pmflxr, jk + 1) = 0.0f;
        } else if (NTC(pmflxs, jk + 1) < 0.0f) {
            NTC(pmflxr, jk + 1) =
                __fadd_rn(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1));
            NTC(pmflxs, jk + 1) = 0.0f;
        }
    }


    /* the sub-cloud evaporation (:3012-3057).
     *
     * NAMED COVERAGE GAP: measured, 48 columns reach this guard and ZERO
     * move pdmfup, because zdrfl1 carries max(0, pqsen - pqen) and the
     * fixture's soundings are saturated below cloud base wherever the block
     * runs.  So the 0.5777 power law, zrmin and the rhevap land/sea split
     * are transcribed and graded only in the sense that their guard is
     * evaluated -- and rhevap is the ONLY path lndj takes into cuflxn, so
     * land/sea is untested here too.  See docs/ntiedtke/PORT-RECORD.md section 16.
     */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (!(ld && jk >= kb)) continue;
        const float zrfl = __fadd_rn(NTC(pmflxr, jk), NTC(pmflxs, jk));
        if (zrfl > 1.0e-20f) {
            const float zdp = __fadd_rn(NTC(paph, jk + 1), -NTC(paph, jk));
            const float zbase = __fmul_rn(
                __fdiv_rn(__fsqrt_rn(__fdiv_rn(NTC(paph, jk),
                                               NTC(paph, klev + 1))),
                          5.09e-3f),
                __fdiv_rn(zrfl, NT_ZCUCOV));
            const float zdrfl1 = __fmul_rn(
                __fmul_rn(__fmul_rn(__fmul_rn(zcpecons,
                    fmaxf(0.0f, __fadd_rn(NTC(pqsen, jk), -NTC(pqen, jk)))),
                    NT_ZCUCOV),
                    gfk_pow(zbase, 0.5777f)),
                zdp);
            float zrnew = __fadd_rn(zrfl, -zdrfl1);
            const float zrmin = __fadd_rn(zrfl,
                -__fmul_rn(__fmul_rn(NT_ZCUCOV,
                    fmaxf(0.0f, __fadd_rn(
                        __fmul_rn(rhevap, NTC(pqsen, jk)), -NTC(pqen, jk)))),
                    __fmul_rn(zcons2, zdp)));
            zrnew = fmaxf(zrnew, zrmin);
            const float zrfln = fmaxf(zrnew, 0.0f);
            const float zdrfl = fminf(0.0f, __fadd_rn(zrfln, -zrfl));
            const float zdenom = __fdiv_rn(1.0f, fmaxf(1.0e-20f,
                __fadd_rn(NTC(pmflxr, jk), NTC(pmflxs, jk))));
            float zalfaw = nt_foealfa(NTC(pten, jk));
            if (NTC(pten, jk) < NT_TMELT) zalfaw = 0.0f;
            const float zpdr = __fmul_rn(zalfaw, NTC(pdmfdp, jk));
            const float zpds =
                __fmul_rn(__fadd_rn(1.0f, -zalfaw), NTC(pdmfdp, jk));
            NTC(pmflxr, jk + 1) = __fadd_rn(
                __fadd_rn(__fadd_rn(NTC(pmflxr, jk), zpdr), NTC(pdpmel, jk)),
                __fmul_rn(__fmul_rn(zdrfl, NTC(pmflxr, jk)), zdenom));
            NTC(pmflxs, jk + 1) = __fadd_rn(
                __fadd_rn(__fadd_rn(NTC(pmflxs, jk), zpds),
                          -NTC(pdpmel, jk)),
                __fmul_rn(__fmul_rn(zdrfl, NTC(pmflxs, jk)), zdenom));
            NTC(pdmfup, jk) = __fadd_rn(NTC(pdmfup, jk), zdrfl);
            if (__fadd_rn(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)) < 0.0f) {
                NTC(pdmfup, jk) = __fadd_rn(NTC(pdmfup, jk),
                    -__fadd_rn(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)));
                NTC(pmflxr, jk + 1) = 0.0f;
                NTC(pmflxs, jk + 1) = 0.0f;
                NTC(pdpmel, jk) = 0.0f;
            } else if (NTC(pmflxr, jk + 1) < 0.0f) {
                NTC(pmflxs, jk + 1) =
                    __fadd_rn(NTC(pmflxs, jk + 1), NTC(pmflxr, jk + 1));
                NTC(pmflxr, jk + 1) = 0.0f;
            } else if (NTC(pmflxs, jk + 1) < 0.0f) {
                NTC(pmflxr, jk + 1) =
                    __fadd_rn(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1));
                NTC(pmflxs, jk + 1) = 0.0f;
            }
        } else {
            NTC(pmflxr, jk + 1) = 0.0f;
            NTC(pmflxs, jk + 1) = 0.0f;
            NTC(pdmfdp, jk) = 0.0f;
            NTC(pdpmel, jk) = 0.0f;
        }
    }

    prain[i] = pr;
    lddraf[i] = ldd;
#undef NTC
}

/* =====================================================================
 * Stage 13: the cloud-depth check -- cumastrn:562-590
 * =====================================================================
 * THE FIRST PIECE OF ORCHESTRATION, and the one the whole port exists for.
 * Thirty lines between cuascn and cudlfsn that nothing owned:
 *
 *   :566-568  THE KTYPE FLIP.  A deep column whose cloud is shallower than
 *             200 hPa becomes ktype 2; a shallow one that is deeper becomes
 *             ktype 1.  ktype selects scale_fac (deep) or scale_fac2
 *             (shallow) in the closure -- the entire reason for the port.
 *             This is the fifth failure's line: it sits between two stages
 *             that look adjacent, and feeding cuascn's ktype to the closure
 *             runs the wrong arm.  The closure kernel's header says it
 *             takes the CLOSURE-TIME ktype; THIS is what produces it.
 *   :580-588  the downdraft-array zeroing that four class-2 excuses in
 *             test_ntiedtke_aliasing_audit.py rest on.  Owning it clears
 *             those debts -- the excuses stop being conditional.
 *
 * ldcum is READ ONLY here.  ktype is flipped in place; ictop0 is set to
 * kctop for cumulus columns and left alone otherwise, which is the same
 * conditional-write discipline as everywhere else in this port.
 */
#define NT_ZDNOPRC 2.0e4f

extern "C" __global__ void ntiedtke_cloud_depth(
        const int *__restrict__ ldcum,
        const int *__restrict__ kcbot,
        const int *__restrict__ kctop,
        const float *__restrict__ paph,
        const float *__restrict__ pdmfup,
        int *__restrict__ ktype,             /* FLIPPED in place */
        int *__restrict__ ictop0,            /* conditionally written */
        float *__restrict__ prfl,
        float *__restrict__ pmfd,            /* :580-588, all zeroed */
        float *__restrict__ pmfds,
        float *__restrict__ pmfdq,
        float *__restrict__ pdmfdp,
        float *__restrict__ pdpmel,
        int ncol, int klev,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_DEPTH,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_DEPTH, order_report, ticket);

#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    if (ldcum[i]) {
        const int ikb = kcbot[i];
        const int itopm2 = kctop[i];
        const float zpbmpt = __fadd_rn(NTC(paph, ikb), -NTC(paph, itopm2));
        /* Two sequential ifs, NOT if/else: the reference lets a column
         * flipped 1 -> 2 be examined by the second test.  It cannot flip
         * back (zpbmpt cannot be both < and >= the threshold), but the
         * shape is transcribed as written. */
        if (ktype[i] == 1 && zpbmpt < NT_ZDNOPRC) ktype[i] = 2;
        if (ktype[i] == 2 && zpbmpt >= NT_ZDNOPRC) ktype[i] = 1;
        ictop0[i] = itopm2;
    }

    /* :571-577.  zrfl starts at level 1 and sums upward through klev. */
    float zrfl = NTC(pdmfup, 1);
    for (int jk = 2; jk <= klev; ++jk)
        zrfl = __fadd_rn(zrfl, NTC(pdmfup, jk));
    prfl[i] = zrfl;

    /* :580-588 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pmfd, jk) = 0.0f;
        NTC(pmfds, jk) = 0.0f;
        NTC(pmfdq, jk) = 0.0f;
        NTC(pdmfdp, jk) = 0.0f;
        NTC(pdpmel, jk) = 0.0f;
    }
#undef NTC
}

/* =====================================================================
 * Stage 14: the adjustments block -- cumastrn:833-919
 * =====================================================================
 * Between cuflxn and cudtdqn.  Five things:
 *
 *   :838-847   the DOWNDRAFT stability cap.  zmfs is the largest factor
 *              keeping |pmfd| under 0.98*pmfu at every level.  Computed in
 *              a full pass before anything is applied, so it stays a
 *              register rather than needing a second array.
 *   :849-861   apply it, and carry the precipitation the capped downdraft
 *              no longer transports into pmflxr through zmfuub -- an
 *              accumulator running DOWNWARD through the column.
 *   :863-880   entrainment-rate floors, and pdmfup recomputed from the
 *              precipitation-flux divergence.
 *   :883-892   the downdraft-top humidity guard.
 *   :896-913   the near-cloud-top humidity guard, which can REDUCE plude.
 *
 * NOT THE MOMENTUM RESCALE.  That is :996-1016, off a different zmfs
 * computed against a different limit -- and the two share the local name.
 * This port attributed one range's job to the other once already.
 *
 * pmfude_rate arrives from CUFLXN's exit, not cuascn's: the 6.5 updraft
 * rescale at :746-819 runs between them and scales it.  Assuming otherwise
 * put the mirror 1.26x low on 42 of 108 columns.
 */
extern "C" __global__ void ntiedtke_adjust(
        const int *__restrict__ ldcum,
        const int *__restrict__ loddraf,
        const int *__restrict__ idtop,
        const int *__restrict__ kctop,
        const int *__restrict__ kcbot,
        const float *__restrict__ paph,      /* reads klev+1 */
        const float *__restrict__ pqen,
        const float *__restrict__ pmfu,
        const float *__restrict__ pmfuq,
        const float *__restrict__ pmful,
        float *__restrict__ pmfd,            /* all in AND out */
        float *__restrict__ pmfds,
        float *__restrict__ pmfdq,
        float *__restrict__ plude,
        float *__restrict__ pdmfup,
        float *__restrict__ pdmfdp,
        float *__restrict__ pmfdde_rate,
        float *__restrict__ pmfude_rate,
        float *__restrict__ pmflxr,          /* klev+1 */
        float *__restrict__ pmflxs,
        float *__restrict__ prsfc,
        float *__restrict__ pssfc,
        int ncol, int klev, float ztmst,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_ADJUST,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_ADJUST, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    const int ld = ldcum[i];
    const int ldd = loddraf[i];
    const int idt = idtop[i];
    const int ktop = kctop[i];
    const int kb = kcbot[i];

    /* :834-847 */
    float zmfs = 1.0f;
    for (int jk = 2; jk <= klev; ++jk) {
        if (ldd && jk >= idt - 1) {
            const float zmfmax = __fmul_rn(NTC(pmfu, jk), 0.98f);
            if (__fadd_rn(__fadd_rn(NTC(pmfd, jk), zmfmax), 1.0e-15f)
                    < 0.0f) {
                zmfs = fminf(zmfs, __fdiv_rn(-zmfmax, NTC(pmfd, jk)));
            }
        }
    }

    /* :849-861.  zmfuub runs DOWNWARD through the column. */
    float zmfuub = 0.0f;
    for (int jk = 2; jk <= klev; ++jk) {
        if (zmfs < 1.0f && jk >= idt - 1) {
            NTC(pmfd, jk) = __fmul_rn(NTC(pmfd, jk), zmfs);
            NTC(pmfds, jk) = __fmul_rn(NTC(pmfds, jk), zmfs);
            NTC(pmfdq, jk) = __fmul_rn(NTC(pmfdq, jk), zmfs);
            NTC(pmfdde_rate, jk) = __fmul_rn(NTC(pmfdde_rate, jk), zmfs);
            zmfuub = __fadd_rn(zmfuub,
                -__fmul_rn(__fadd_rn(1.0f, -zmfs), NTC(pdmfdp, jk)));
            NTC(pmflxr, jk + 1) = __fadd_rn(NTC(pmflxr, jk + 1), zmfuub);
            NTC(pdmfdp, jk) = __fmul_rn(NTC(pdmfdp, jk), zmfs);
        }
    }

    /* :863-880 */
    for (int jk = 2; jk <= klev - 1; ++jk) {
        if (ldd && jk >= idt - 1) {
            const float zerate = __fadd_rn(
                __fadd_rn(-NTC(pmfd, jk), NTC(pmfd, jk - 1)),
                NTC(pmfdde_rate, jk));
            if (zerate < 0.0f)
                NTC(pmfdde_rate, jk) =
                    __fadd_rn(NTC(pmfdde_rate, jk), -zerate);
        }
        if (ld && jk >= ktop - 1) {
            const float zerate = __fadd_rn(
                __fadd_rn(NTC(pmfu, jk), -NTC(pmfu, jk + 1)),
                NTC(pmfude_rate, jk));
            if (zerate < 0.0f)
                NTC(pmfude_rate, jk) =
                    __fadd_rn(NTC(pmfude_rate, jk), -zerate);
            NTC(pdmfup, jk) = __fadd_rn(__fadd_rn(
                __fadd_rn(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)),
                -NTC(pmflxr, jk)), -NTC(pmflxs, jk));
            NTC(pdmfdp, jk) = 0.0f;
        }
    }

    /* :883-892  the downdraft-top humidity guard */
    if (ldd) {
        const int jk = idt;
        const int ik = min(jk + 1, klev);
        if (NTC(pmfdq, jk) < __fmul_rn(0.3f, NTC(pmfdq, ik)))
            NTC(pmfdq, jk) = __fmul_rn(0.3f, NTC(pmfdq, ik));
    }

    /* :896-913  the near-cloud-top humidity guard */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld && jk >= ktop - 1 && jk < kb) {
            const float zdz = __fdiv_rn(__fmul_rn(ztmst, c.g),
                __fadd_rn(NTC(paph, jk + 1), -NTC(paph, jk)));
            float zmfa = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(
                __fadd_rn(NTC(pmfuq, jk + 1), NTC(pmfdq, jk + 1)),
                -NTC(pmfuq, jk)), -NTC(pmfdq, jk)),
                NTC(pmful, jk + 1)), -NTC(pmful, jk)), NTC(pdmfup, jk));
            zmfa = __fmul_rn(__fadd_rn(zmfa, -NTC(plude, jk)), zdz);
            if (__fadd_rn(NTC(pqen, jk), zmfa) < 0.0f) {
                NTC(plude, jk) = __fadd_rn(NTC(plude, jk), __fdiv_rn(
                    __fmul_rn(2.0f, __fadd_rn(NTC(pqen, jk), zmfa)), zdz));
            }
            if (NTC(plude, jk) < 0.0f) NTC(plude, jk) = 0.0f;
        }
        if (!ld) NTC(pmfude_rate, jk) = 0.0f;
        if (fabsf(NTC(pmfd, jk - 1)) < 1.0e-20f)
            NTC(pmfdde_rate, jk) = 0.0f;
    }

    prsfc[i] = NTC(pmflxr, klev + 1);
    pssfc[i] = NTC(pmflxs, klev + 1);
#undef NTC
}

/* =====================================================================
 * Stage 15: the momentum mass-flux rescale -- cumastrn:996-1016
 * =====================================================================
 * THE ONLY CONSUMER OF `zcons` IN THE ENTIRE SCHEME.
 *
 * Every other mass-flux cap in New Tiedtke uses zcons2 = 3/(g*dt).  This
 * one uses zcons = 1/(g*dt) -- one character away, both declared in
 * cumastrn, three times tighter.  Getting it wrong makes the momentum
 * rescale three times too permissive, and the result is finite, plausible
 * and off by a FIXED RATIO: the least visible arithmetic error available,
 * and one that reaches f012 looking like a physics result.
 *
 * MEASURED: the cap does not bind anywhere in the fixture -- the closest
 * column reaches 0.5076 of it -- so this constant is graded only in the
 * sense that its guard is evaluated.  Named in
 * test_ntiedtke_mrescale_parity.py and in docs/ntiedtke/PORT-RECORD.md as a
 * case-table item, not an unreachable branch: 2x short, not orders.
 *
 * It produces zmfuus/zmfdus, and it is THOSE that cududvn consumes.
 *
 * The second loop runs 1..klev and assigns the UNSCALED value first, so a
 * level outside the cloud carries pmfu/pmfd through unchanged rather than
 * becoming zero.  Transcribed in that order for that reason.
 */
extern "C" __global__ void ntiedtke_momentum_rescale(
        const int *__restrict__ ldcum,
        const int *__restrict__ kctop,
        const float *__restrict__ paph,
        const float *__restrict__ pmfu,
        const float *__restrict__ pmfd,
        float *__restrict__ zmfuus,
        float *__restrict__ zmfdus,
        int ncol, int klev, float ztmst,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_MRESCALE,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_MRESCALE, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    /* zcons, NOT zcons2.  See the header. */
    const float zcons = __fdiv_rn(1.0f, __fmul_rn(c.g, ztmst));
    const int ld = ldcum[i];
    const int ktop = kctop[i];

    float zmfs = 1.0f;
    if (ld) {
        for (int jk = 2; jk <= klev; ++jk) {
            if (jk >= ktop - 1) {
                const float zmfmax = __fmul_rn(
                    __fadd_rn(NTC(paph, jk), -NTC(paph, jk - 1)), zcons);
                if (NTC(pmfu, jk) > zmfmax && jk >= ktop)
                    zmfs = fminf(zmfs, __fdiv_rn(zmfmax, NTC(pmfu, jk)));
            }
        }
    }

    for (int jk = 1; jk <= klev; ++jk) {
        NTC(zmfuus, jk) = NTC(pmfu, jk);
        NTC(zmfdus, jk) = NTC(pmfd, jk);
        if (ld && jk >= ktop - 1) {
            NTC(zmfuus, jk) = __fmul_rn(NTC(pmfu, jk), zmfs);
            NTC(zmfdus, jk) = __fmul_rn(NTC(pmfd, jk), zmfs);
        }
    }
#undef NTC
}

/* =====================================================================
 * Stage 16: the updraft rescale and two cleanups -- cumastrn:743-819
 * =====================================================================
 * WHERE THE CLOSURE'S ANSWER ACTUALLY LANDS.  :745 forms
 *
 *     zmfs = zmfub1 / max(cmfcmin, zmfub)
 *
 * and this block applies it to the whole updraft.  zmfub1 is what the CAPE
 * closure produced -- the quantity scale_fac and scale_fac2 act on, and the
 * reason this port exists.  docs/ntiedtke/PORT-RECORD.md section 9 measured its
 * retention; this is the code that spends it.
 *
 * The cap at :755-758 is the one to read twice: it tests `pmfu*zmfs >
 * zmfmax` but divides by the UNSCALED pmfu, and it re-reads its own running
 * zmfs.  Those are different once zmfs < 1, and the shape is transcribed as
 * written rather than tidied.
 *
 * The dead block at :786-802 is skipped: both its guards are `.true.`
 * parameters, asserted in test_ntiedtke_cumastrn_ownership.py.
 */
extern "C" __global__ void ntiedtke_updraft_scale(
        const int *__restrict__ loddraf,
        const int *__restrict__ kcbot,
        const int *__restrict__ kctop,
        const float *__restrict__ zmfub1,
        const float *__restrict__ zmfub,
        const float *__restrict__ paph,      /* reads klev+1 */
        int *__restrict__ ldcum,             /* 6.6 can clear these */
        int *__restrict__ ktype,
        int *__restrict__ idtop,             /* 6.7 can push this down */
        float *__restrict__ pmfu,            /* all in AND out */
        float *__restrict__ pmfus,
        float *__restrict__ pmfuq,
        float *__restrict__ pmful,
        float *__restrict__ pdmfup,
        float *__restrict__ plude,
        float *__restrict__ pmfude_rate,
        float *__restrict__ pmfd,
        float *__restrict__ pmfds,
        float *__restrict__ pmfdq,
        float *__restrict__ pdmfdp,
        float *__restrict__ pmfdde_rate,
        int ncol, int klev, float ztmst,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_USCALE,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_USCALE, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    const float zcons2 = __fdiv_rn(3.0f, __fmul_rn(c.g, ztmst));
    const int ld0 = ldcum[i];
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int ldd = loddraf[i];

    /* :743-746 */
    float zmfs = 1.0f;
    if (ld0) zmfs = __fdiv_rn(zmfub1[i], fmaxf(NT_CMFCMIN, zmfub[i]));

    /* :747-761  taper below cloud base, then cap */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld0 && jk >= ktop - 1) {
            if (jk > kb) {
                const float zdz = __fdiv_rn(
                    __fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)),
                    __fadd_rn(NTC(paph, klev + 1), -NTC(paph, kb)));
                NTC(pmfu, jk) = __fmul_rn(NTC(pmfu, kb), zdz);
            }
            const float zmfmax = __fmul_rn(
                __fadd_rn(NTC(paph, jk), -NTC(paph, jk - 1)), zcons2);
            if (__fmul_rn(NTC(pmfu, jk), zmfs) > zmfmax)
                zmfs = fminf(zmfs, __fdiv_rn(zmfmax, NTC(pmfu, jk)));
        }
    }

    /* :762-774  apply */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld0 && jk <= kb && jk >= ktop - 1) {
            NTC(pmfu, jk) = __fmul_rn(NTC(pmfu, jk), zmfs);
            NTC(pmfus, jk) = __fmul_rn(NTC(pmfus, jk), zmfs);
            NTC(pmfuq, jk) = __fmul_rn(NTC(pmfuq, jk), zmfs);
            NTC(pmful, jk) = __fmul_rn(NTC(pmful, jk), zmfs);
            NTC(pdmfup, jk) = __fmul_rn(NTC(pdmfup, jk), zmfs);
            NTC(plude, jk) = __fmul_rn(NTC(plude, jk), zmfs);
            NTC(pmfude_rate, jk) =
                __fmul_rn(NTC(pmfude_rate, jk), zmfs);
        }
    }

    /* 6.6 (:777-783) */
    if (ktype[i] == 2 && kb == ktop && kb >= klev - 1) {
        ldcum[i] = 0;
        ktype[i] = 0;
    }

    /* 6.7 (:798-818) */
    int idt = idtop[i];
    if (ldd && idt <= ktop) idt = ktop + 1;
    for (int jk = 2; jk <= klev; ++jk) {
        if (ldd) {
            if (jk < idt) {
                NTC(pmfd, jk) = 0.0f;
                NTC(pmfds, jk) = 0.0f;
                NTC(pmfdq, jk) = 0.0f;
                NTC(pmfdde_rate, jk) = 0.0f;
                NTC(pdmfdp, jk) = 0.0f;
            } else if (jk == idt) {
                NTC(pmfdde_rate, jk) = 0.0f;
            }
        }
    }
    idtop[i] = idt;
#undef NTC
}

/* =====================================================================
 * Stage 17: the updraft/downdraft momentum profiles -- cumastrn:927-995
 * =====================================================================
 * What produces the puu/pvu/pud/pvd that cududvn consumes -- and the block
 * that falsified this port's eighth wrong claim.  cuascn, cudlfsn and
 * cuddrafn genuinely never write them; the chained conclusion "so cuinin
 * sets them and nothing between touches them" was wrong, because THIS does.
 * Measured: puu differs on 1,926 of 5,292 slots between cuinin's exit and
 * cududvn's entry.
 *
 * momtrans = 2, a PARAMETER, so :943-955 -- the `if (momtrans == 1)` arm --
 * is a THIRD dead block and the pressure-gradient `else` is the live one.
 * Only the live arm is transcribed; the range is named so the omission
 * cannot read as a transcription slip.  pgcoef = 0.7.
 *
 * zuu/zvu must survive from the updraft loop to the downdraft loop --
 * :977-978 seeds zud from zuu at the SAME level -- so they stay in their
 * caller arrays rather than collapsing to registers.  They are outputs
 * anyway, so the frame is still 0 B.
 */
#define NT_PGCOEF 0.7f

extern "C" __global__ void ntiedtke_momentum_profile(
        const int *__restrict__ ldcum,
        const int *__restrict__ ktype,
        const int *__restrict__ kcbot,
        const int *__restrict__ kctop,
        const int *__restrict__ kdpl,
        const int *__restrict__ idtop,
        const float *__restrict__ puen,
        const float *__restrict__ pven,
        const float *__restrict__ pmfu,
        const float *__restrict__ pmfd,
        const float *__restrict__ pmfude_rate,
        const float *__restrict__ pmfdde_rate,
        float *__restrict__ puu,             /* read AND written */
        float *__restrict__ pvu,
        float *__restrict__ pud,
        float *__restrict__ pvd,
        int ncol, int klev,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_MPROFILE,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_MPROFILE, order_report, ticket);
    if (!ldcum[i]) return;

#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]
    const int kt = ktype[i];
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int kdp = kdpl[i];
    const int idt = idtop[i];

    /* the updraft profile (:930-971), DOWNWARD in index */
    for (int jk = klev - 1; jk >= 2; --jk) {
        const int ik = jk + 1;
        if (jk == kb && kt < 3) {
            NTC(puu, jk) = NTC(puen, kdp - 1);
            NTC(pvu, jk) = NTC(pven, kdp - 1);
        } else if (jk == kb && kt == 3) {
            NTC(puu, jk) = NTC(puen, jk - 1);
            NTC(pvu, jk) = NTC(pven, jk - 1);
        }
        if (jk < kb && jk >= ktop) {
            const float pgf_u = -__fmul_rn(
                __fmul_rn(NT_PGCOEF, 0.5f),
                __fadd_rn(
                    __fmul_rn(NTC(pmfu, ik),
                        __fadd_rn(NTC(puen, ik), -NTC(puen, jk))),
                    __fmul_rn(NTC(pmfu, jk),
                        __fadd_rn(NTC(puen, jk), -NTC(puen, jk - 1)))));
            const float pgf_v = -__fmul_rn(
                __fmul_rn(NT_PGCOEF, 0.5f),
                __fadd_rn(
                    __fmul_rn(NTC(pmfu, ik),
                        __fadd_rn(NTC(pven, ik), -NTC(pven, jk))),
                    __fmul_rn(NTC(pmfu, jk),
                        __fadd_rn(NTC(pven, jk), -NTC(pven, jk - 1)))));
            const float zerate = __fadd_rn(
                __fadd_rn(NTC(pmfu, jk), -NTC(pmfu, ik)),
                NTC(pmfude_rate, jk));
            const float zderate = NTC(pmfude_rate, jk);
            const float zmfa = __fdiv_rn(
                1.0f, fmaxf(NT_CMFCMIN, NTC(pmfu, jk)));
            NTC(puu, jk) = __fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(
                __fmul_rn(NTC(puu, ik), NTC(pmfu, ik)),
                __fmul_rn(zerate, NTC(puen, jk))),
                -__fmul_rn(zderate, NTC(puu, ik))), pgf_u), zmfa);
            NTC(pvu, jk) = __fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(
                __fmul_rn(NTC(pvu, ik), NTC(pmfu, ik)),
                __fmul_rn(zerate, NTC(pven, jk))),
                -__fmul_rn(zderate, NTC(pvu, ik))), pgf_v), zmfa);
        }
    }

    /* the downdraft profile (:972-991), UPWARD in index */
    for (int jk = 3; jk <= klev; ++jk) {
        const int ik = jk - 1;
        if (jk == idt) {
            NTC(pud, jk) = __fmul_rn(0.5f,
                __fadd_rn(NTC(puu, jk), NTC(puen, ik)));
            NTC(pvd, jk) = __fmul_rn(0.5f,
                __fadd_rn(NTC(pvu, jk), NTC(pven, ik)));
        } else if (jk > idt) {
            const float zerate = __fadd_rn(
                __fadd_rn(-NTC(pmfd, jk), NTC(pmfd, ik)),
                NTC(pmfdde_rate, jk));
            const float zmfa = __fdiv_rn(
                1.0f, fminf(-NT_CMFCMIN, NTC(pmfd, jk)));
            NTC(pud, jk) = __fmul_rn(__fadd_rn(__fadd_rn(
                __fmul_rn(NTC(pud, ik), NTC(pmfd, ik)),
                -__fmul_rn(zerate, NTC(puen, ik))),
                __fmul_rn(NTC(pmfdde_rate, jk), NTC(pud, ik))), zmfa);
            NTC(pvd, jk) = __fmul_rn(__fadd_rn(__fadd_rn(
                __fmul_rn(NTC(pvd, ik), NTC(pmfd, ik)),
                -__fmul_rn(zerate, NTC(pven, ik))),
                __fmul_rn(NTC(pmfdde_rate, jk), NTC(pvd, ik))), zmfa);
        }
    }
#undef NTC
}

/* =====================================================================
 * Stage 18: the kinetic-energy dissipation -- cumastrn:1030-1056
 * =====================================================================
 * THE LAST ARITHMETIC IN cumastrn, and the only place the momentum
 * tendency feeds BACK into the heat tendency.  cududvn has just changed
 * pvom/pvol; this measures how much kinetic energy that change removed
 * from the resolved flow and returns it as sensible heat.
 *
 * ztenu/ztenv are pvom/pvol BEFORE cududvn.  Taking that copy is
 * :1019-1024 and it is the CALLER's work, not a kernel's -- one array
 * copy, owned by the assembler.  Recorded here so the omission cannot
 * read as a missing transcription.
 *
 * zsum22 and zsum12 are COLUMN INTEGRALS.  The loop that fills them must
 * finish before the loop that divides by them starts, so unlike most of
 * this port the two passes cannot be fused -- and zuv2 is needed at every
 * level in the second pass, so it is a caller array rather than a
 * register.  It is the only value this stage needs that is not already
 * an output; the frame is still 0 B.
 *
 * ptte is ACCUMULATED into, the same add-not-assign contract as cudtdqn
 * and for the same reason (docs/ntiedtke/PORT-RECORD.md section 17).
 */
extern "C" __global__ void ntiedtke_ke_dissipation(
        const int *__restrict__ ldcum,
        const int *__restrict__ kctop,
        const float *__restrict__ paph,      /* reads klev+1 */
        const float *__restrict__ puen,
        const float *__restrict__ pven,
        const float *__restrict__ ztenu,     /* pvom BEFORE cududvn */
        const float *__restrict__ ztenv,
        const float *__restrict__ pvom,      /* pvom AFTER cududvn */
        const float *__restrict__ pvol,
        float *__restrict__ ptte,            /* ACCUMULATED, in and out */
        float *__restrict__ zuv2,            /* (nz+2, ncol) scratch */
        int ncol, int klev,
        float cp, float rd, float rv, float xlv, float xlf, float grav,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_KEDIS,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_KEDIS, order_report, ticket);

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    const int ld = ldcum[i];
    const int ktop = kctop[i];
    float zsum12 = 0.0f;
    float zsum22 = 0.0f;

    /* :1034-1048  the integrals */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(zuv2, jk) = 0.0f;
        if (ld && jk >= ktop - 1) {
            const float zdz =
                __fadd_rn(NTC(paph, jk + 1), -NTC(paph, jk));
            const float zduten =
                __fadd_rn(NTC(pvom, jk), -NTC(ztenu, jk));
            const float zdvten =
                __fadd_rn(NTC(pvol, jk), -NTC(ztenv, jk));
            NTC(zuv2, jk) = __fsqrt_rn(__fadd_rn(
                __fmul_rn(zduten, zduten), __fmul_rn(zdvten, zdvten)));
            zsum22 = __fadd_rn(zsum22, __fmul_rn(NTC(zuv2, jk), zdz));
            zsum12 = __fadd_rn(zsum12, -__fmul_rn(__fadd_rn(
                __fmul_rn(NTC(puen, jk), zduten),
                __fmul_rn(NTC(pven, jk), zdvten)), zdz));
        }
    }

    /* :1049-1056  the heating.  ADD, do not assign. */
    for (int jk = 1; jk <= klev; ++jk) {
        if (ld && jk >= ktop - 1) {
            const float ztdis = __fdiv_rn(
                __fmul_rn(__fmul_rn(c.rcpd, zsum12), NTC(zuv2, jk)),
                fmaxf(1.0e-15f, zsum22));
            NTC(ptte, jk) = __fadd_rn(NTC(ptte, jk), ztdis);
        }
    }
#undef NTC
}


/* =====================================================================
 * Stage 19: cu_ntiedtke_post_run -- module_cu_ntiedtke.F:502-527
 * =====================================================================
 * THE EIGHT FIELDS nt-levels.csv IS GRADED ON.  Until this kernel existed
 * they traced to nothing in the tree: cu_ntiedtke_run produces pt/pqv/pqc/
 * pqi/pu/pv and zprecc, and nothing turned those into rthcuten/rucuten/
 * raincv/pratec.  Phase 1's end condition names that file, so the port
 * could not have reached it however many kernels were graded.
 *
 * TWO VERTICAL CONVENTIONS IN ONE STATEMENT, and this is the only kernel
 * in the port where that is true.  exner/qv/qc/qi/t/u/v are the driver's
 * untouched WRF-order inputs, k = 1 the SURFACE, and they are the
 * reference state the tendency is measured against.  tf/qvf/qcf/qif/uf/vf
 * carry cu_ntiedtke_run's answer in SCHEME order, k = 1 the model TOP.
 * The routine pairs them by flipping and so does this: zz = klev + 1 - jk.
 *
 * ASSIGNED, NOT ACCUMULATED, and unconditionally -- unlike cudtdqn and the
 * KE dissipation there is no `if` in the loop and no add.  Every level of
 * every column is written, which is why post_run has no class-2 rows in
 * the aliasing audit despite six intent(inout) arrays.  Read off :514-524,
 * not inferred from the intent.
 *
 * NO PHYSICAL CONSTANTS.  Every other stage takes the six-member family
 * through nt_init; this one takes none, because the routine uses none.
 * Passing them anyway would make the signature lie about the dependency.
 *
 * THE ASSOCIATION IS LOAD-BEARING: (tf - t)/exner*rdelt is subtract,
 * DIVIDE, then multiply.  Folding it to (tf - t) * (rdelt/exner) is
 * algebraically identical and bitwise different, so the divide is spelled
 * with __fdiv_rn and kept in place.
 */
extern "C" __global__ void ntiedtke_post_run(
        const float *__restrict__ exner,   /* WRF order, k = 1 surface */
        const float *__restrict__ qv,
        const float *__restrict__ qc,
        const float *__restrict__ qi,
        const float *__restrict__ t,
        const float *__restrict__ u,
        const float *__restrict__ v,
        const float *__restrict__ tf,      /* scheme order, k = 1 top */
        const float *__restrict__ qvf,
        const float *__restrict__ qcf,
        const float *__restrict__ qif,
        const float *__restrict__ uf,
        const float *__restrict__ vf,
        const float *__restrict__ rn,      /* per column */
        float *__restrict__ rthcuten,      /* WRF order, all ASSIGNED */
        float *__restrict__ rqvcuten,
        float *__restrict__ rqccuten,
        float *__restrict__ rqicuten,
        float *__restrict__ rucuten,
        float *__restrict__ rvcuten,
        float *__restrict__ raincv,
        float *__restrict__ pratec,
        int ncol, int klev, int stepcu, float dt,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_POSTRUN,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_POSTRUN, order_report, ticket);

#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    /* :504-505.  stepcu is an INTEGER in the reference and Fortran
     * promotes it; the reciprocal is a separate rounding and both are
     * kept rather than folded into one constant. */
    const float fstepcu = (float)stepcu;
    const float delt = __fmul_rn(dt, fstepcu);
    const float rdelt = __fdiv_rn(1.0f, delt);

    /* :506-509.  pratec divides by ONE product, stepcu*dt -- not by
     * stepcu and then by dt, which rounds twice. */
    raincv[i] = __fdiv_rn(rn[i], fstepcu);
    pratec[i] = __fdiv_rn(rn[i], __fmul_rn(fstepcu, dt));

    /* :511-524 */
    for (int jk = 1; jk <= klev; ++jk) {
        const int zz = klev + 1 - jk;
        NTC(rthcuten, jk) = __fmul_rn(
            __fdiv_rn(__fadd_rn(NTC(tf, zz), -NTC(t, jk)), NTC(exner, jk)),
            rdelt);
        NTC(rqvcuten, jk) =
            __fmul_rn(__fadd_rn(NTC(qvf, zz), -NTC(qv, jk)), rdelt);
        NTC(rqccuten, jk) =
            __fmul_rn(__fadd_rn(NTC(qcf, zz), -NTC(qc, jk)), rdelt);
        NTC(rqicuten, jk) =
            __fmul_rn(__fadd_rn(NTC(qif, zz), -NTC(qi, jk)), rdelt);
        NTC(rucuten, jk) =
            __fmul_rn(__fadd_rn(NTC(uf, zz), -NTC(u, jk)), rdelt);
        NTC(rvcuten, jk) =
            __fmul_rn(__fadd_rn(NTC(vf, zz), -NTC(v, jk)), rdelt);
    }
#undef NTC
}


/* =====================================================================
 * Stage 20: cu_ntiedtke_run's post-conversion -- cu_ntiedtke.F90:278-320
 * =====================================================================
 * THE MISSING LINK.  cumastrn leaves TENDENCIES (ptte, pqte, pvom, pvol)
 * and a detrained condensate rate (pcte); cu_ntiedtke_post_run differences
 * updated STATE against reference state.  This block turns one into the
 * other, and without it the chain from the last cumastrn stage to the
 * eight graded fields of nt-levels.csv has a hole in it.
 *
 * THE ID IS 20 AND IT RUNS BEFORE STAGE 19.  Stage ids are labels for the
 * report array, not a sequence -- NT_CALL_ORDER in ntiedtke.py is the
 * sequence, and it has disagreed with id order since cudtdqn (8) and
 * cududvn (9) took their places.  Reading order off the id is a mistake
 * this comment exists to prevent.
 *
 * EVERYTHING HERE IS SCHEME ORDER, k = 1 the model top.  The flip back to
 * WRF order happens in post_run, not here.
 *
 * THE CONDENSATE ARM IS CONDITIONAL: `if (pcte > 0.)`, and on the false
 * arm pqc/pqi keep the values they arrived with.  That is a class-2 shape,
 * so the kernel MUST NOT zero them at entry -- the natural CUDA idiom
 * would diverge on every level that does not detrain, which is most of
 * them.  They are copied from the caller's arrays instead.
 *
 * zqp1 IS UPDATED IN PLACE and then read: pqv = zqp1/(1 - zqp1) uses the
 * NEW value.  It is an in/out array here for the same reason.
 *
 * lmfdudv guards the momentum update at :319.  It is a PARAMETER, .true.
 * at cu_ntiedtke.F90:55, so there is no runtime branch to carry.
 */
extern "C" __global__ void ntiedtke_post_conversion(
        const float *__restrict__ pcte,
        const float *__restrict__ ztp1,
        const float *__restrict__ ptte,
        const float *__restrict__ ztt,
        const float *__restrict__ pqte,
        const float *__restrict__ zqq,
        float *__restrict__ zqp1,          /* UPDATED IN PLACE, then read */
        const float *__restrict__ qcf,
        const float *__restrict__ qif,
        const float *__restrict__ uf,
        const float *__restrict__ vf,
        const float *__restrict__ pvom,
        const float *__restrict__ pvol,
        const float *__restrict__ prsfc,   /* per column */
        const float *__restrict__ pssfc,
        float *__restrict__ pqc, float *__restrict__ pqi,
        float *__restrict__ pt, float *__restrict__ pqv,
        float *__restrict__ pu, float *__restrict__ pv,
        float *__restrict__ zprecc,        /* per column */
        int ncol, int klev, float delt,
        int expect_tpb, int expect_nblocks,
        int *__restrict__ geom_report,
        int *__restrict__ order_report, int *__restrict__ ticket) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    if (!nt_geometry_ok(expect_tpb, expect_nblocks, NT_STAGE_POSTCONV,
                        geom_report)) return;
    nt_stage_ticket(NT_STAGE_POSTCONV, order_report, ticket);

#define NTC(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)i]

    /* :296-305.  fliq/fice split the detrained condensate by temperature.
     * The false arm CARRIES the incoming value; it does not zero. */
    for (int jk = 1; jk <= klev; ++jk) {
        if (NTC(pcte, jk) > 0.0f) {
            const float fliq = nt_foealfa(NTC(ztp1, jk));
            const float fice = __fadd_rn(1.0f, -fliq);
            NTC(pqc, jk) = __fadd_rn(
                NTC(qcf, jk),
                __fmul_rn(__fmul_rn(fliq, NTC(pcte, jk)), delt));
            NTC(pqi, jk) = __fadd_rn(
                NTC(qif, jk),
                __fmul_rn(__fmul_rn(fice, NTC(pcte, jk)), delt));
        } else {
            NTC(pqc, jk) = NTC(qcf, jk);
            NTC(pqi, jk) = NTC(qif, jk);
        }
    }

    /* :308-314 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pt, jk) = __fadd_rn(
            NTC(ztp1, jk),
            __fmul_rn(__fadd_rn(NTC(ptte, jk), -NTC(ztt, jk)), delt));
        const float q = __fadd_rn(
            NTC(zqp1, jk),
            __fmul_rn(__fadd_rn(NTC(pqte, jk), -NTC(zqq, jk)), delt));
        NTC(zqp1, jk) = q;
        NTC(pqv, jk) = __fdiv_rn(q, __fadd_rn(1.0f, -q));
    }

    /* :316-318.  amax1 clamps a negative flux product to zero. */
    zprecc[i] = fmaxf(0.0f, __fmul_rn(__fadd_rn(prsfc[i], pssfc[i]), delt));

    /* :319-325, lmfdudv is .true. at :55 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pu, jk) = __fadd_rn(NTC(uf, jk),
                                __fmul_rn(NTC(pvom, jk), delt));
        NTC(pv, jk) = __fadd_rn(NTC(vf, jk),
                                __fmul_rn(NTC(pvol, jk), delt));
    }
#undef NTC
}
