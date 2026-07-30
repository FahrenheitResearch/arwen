// gpuwm/core/kernels/noahmp_sflx.cu
//
// CUDA half of the parts of NOAHMP_SFLX that this lane owns outright:
// ERROR (module_sf_noahmplsm.F:1560-1737) and the three marshalling
// computations NOAHMP_SFLX performs before it calls anything -- DZSNSO
// (827-833), TROOT (837-840) and BEG_WB (844-849).
//
// Bitwise-equal to gpuwm/core/noahmp_sflx.py and to the WRF v4.6.1 oracle
// (tree d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256(phys/module_sf_noahmplsm.F) =
//  bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282).
//
// What this file deliberately does NOT do
// ---------------------------------------
// It does not compose the column.  NOAHMP_SFLX's other five statements'-worth
// of work is ATM, PHENOLOGY, PRECIP_HEAT, ENERGY and WATER, and every one of
// those already has a CUDA half of its own.  Chaining them here would mean
// either re-transcribing them -- which is the one thing a composition lane
// must not do -- or pulling five .cu files into one translation unit, which is
// blocked by exactly the obstacle noahmp_energy.cu records: several of those
// files each define their own glibc_logf / glibc_expf / glibc_powf / f_min /
// f_max, so any two of them together are duplicate definitions.  Hoisting a
// single device libm is a real and worthwhile refactor across four other
// lanes' files, and it is not this lane's to make.
//
// So the claim this file supports is exactly: **ERROR and NOAHMP_SFLX's own
// marshalling arithmetic reproduce the pinned fixtures bit for bit on the
// GPU.**  Nothing here reads a value the CPU port produced; every comparison
// in tests/test_noahmp_sflx_cuda.py is against an oracle CSV.
//
// Why these three loops and not some other three
// ----------------------------------------------
// ERROR's water block and NOAHMP_SFLX's TROOT and BEG_WB loops are dense in
// the one shape that is the dominant bitwise hazard on this device:
//
//     END_WB = END_WB + SMC(IZ) * DZSNSO(IZ) * 1000.0        ! :1699, :847
//     TROOT  = TROOT  + STC(IZ) * DZSNSO(IZ) / (-ZSOIL(N))   ! :839
//
// NVRTC defaults to --fmad=true and would contract each of those into an FMA.
// gfortran on x86-64 emits no FMA at -O0, so a contracted a*b+c is a different
// number -- and it is a different number *by more than one ULP once it has
// been accumulated over four layers*.  Every arithmetic operation below
// therefore goes through __fadd_rn / __fsub_rn / __fmul_rn / __fdiv_rn.  Do
// not "simplify" any of them back to an infix operator.
//
// There are no literal tables here, so nothing needs __constant__ memory: the
// ptxas 12.x tie-folding trap that tests/test_fp32_tie_folding_gpu.py guards
// arises from arithmetic on compile-time literal arrays, and this file has
// none.  It evaluates no transcendental at all -- neither ERROR nor the three
// marshalling loops calls one, which is the same fact stage 5 of
// build_sflx_compose.sh proves on the Fortran side with nm -u.
//
// Aborts, on a device that cannot abort
// -------------------------------------
// ERROR's three failure paths end in wrf_error_fatal, which stops the process.
// A kernel cannot, so k_sflx_error writes a status word instead:
//
//     0  no gate tripped
//     1  ABS(ERRSW)  > 0.01   ("Stop in Noah-MP")
//     2  ABS(ERRENG) > 0.01   ("Energy budget problem in NOAHMP LSM")
//     3  ABS(ERRWAT) > 0.1    ("Water budget problem in NOAHMP LSM")
//
// and returns without writing the accumulators past the point WRF would have
// died.  The host is what raises; a status of 0 is the only value that means
// the row is a fixture row.  Reporting instead of aborting is a change in
// *mechanism*, not in behaviour, and the status word is checked on every case.

#define NSOIL 4
#define NSNOW 3
#define NFULL (NSNOW + NSOIL)

// ---------------------------------------------------------------------------
// Rounding-pinned arithmetic.  Every FP32 operation in this file goes through
// one of these four; there are no infix operators on float anywhere below.
// ---------------------------------------------------------------------------
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))

__device__ __forceinline__ float f_abs(float x)
{
    // Fortran ABS on REAL(4) is a sign-bit clear, not a comparison, so it is
    // exact on every input including -0.0.
    return fabsf(x);
}

// ERROR's three thresholds, as the binary32 the Fortran literals become.
#define ERRSW_TOL  0.01f
#define ERRENG_TOL 0.01f
#define ERRWAT_TOL 0.1f

// ---------------------------------------------------------------------------
// ERROR -- module_sf_noahmplsm.F:1560-1737
// ---------------------------------------------------------------------------
//
// Scalar inputs arrive as one row of `sc`, `NSC` wide, in this order.  The
// packing is duplicated in tests/test_noahmp_sflx_cuda.py::SCALARS and the
// test asserts the two agree in length, so a field added on one side without
// the other fails rather than silently shifting every column.
#define SC_SWDOWN      0
#define SC_FSA         1
#define SC_FSR         2
#define SC_FIRA        3
#define SC_FSH         4
#define SC_FCEV        5
#define SC_FGEV        6
#define SC_FCTR        7
#define SC_SSOIL       8
#define SC_SAV         9
#define SC_SAG        10
#define SC_BEG_WB     11
#define SC_CANLIQ     12
#define SC_CANICE     13
#define SC_SNEQV      14
#define SC_WA         15
#define SC_PRCP       16
#define SC_ECAN       17
#define SC_ETRAN      18
#define SC_EDIR       19
#define SC_RUNSRF     20
#define SC_RUNSUB     21
#define SC_DT         22
#define SC_QTLDRN     23
#define SC_PAH        24
#define SC_FIRR       25
#define SC_CANHS      26
#define SC_IRMIRATE   27
#define SC_IRFIRATE   28
#define SC_ACC_DWATER 29
#define SC_ACC_PRCP   30
#define SC_ACC_ECAN   31
#define SC_ACC_ETRAN  32
#define SC_ACC_EDIR   33
#define NSC           34

// Outputs, one row of `out`, `NOUT` wide.
#define OU_ERRWAT     0
#define OU_ACC_DWATER 1
#define OU_ACC_PRCP   2
#define OU_ACC_ECAN   3
#define OU_ACC_ETRAN  4
#define OU_ACC_EDIR   5
#define OU_ERRSW      6
#define OU_ERRENG     7
#define OU_END_WB     8
#define NOUT          9

extern "C" __global__
void k_sflx_error(const float *__restrict__ sc,     // [ncase][NSC]
                  const float *__restrict__ smc,    // [ncase][NSOIL]
                  const float *__restrict__ dzsnso, // [ncase][NSOIL]
                  const int   *__restrict__ ist,    // [ncase]
                  const int   *__restrict__ calc,   // [ncase], calculate_soil
                  float *__restrict__ out,          // [ncase][NOUT]
                  int   *__restrict__ status,       // [ncase]
                  int ncase)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncase) return;

    const float *s = sc + (size_t)c * NSC;
    float *o = out + (size_t)c * NOUT;

    // Leave every output defined before the first early return, so a tripped
    // gate cannot leave the caller reading whatever the allocation held.
    for (int k = 0; k < NOUT; ++k) o[k] = 0.0f;
    status[c] = 0;

    // ERRSW = SWDOWN - (FSA + FSR)                                     :1638
    const float errsw = FSUB(s[SC_SWDOWN], FADD(s[SC_FSA], s[SC_FSR]));
    o[OU_ERRSW] = errsw;
    if (f_abs(errsw) > ERRSW_TOL) {                                  // :1641
        status[c] = 1;
        return;
    }

    // ERRENG = SAV+SAG-(FIRA+FSH+FCEV+FGEV+FCTR+SSOIL+FIRR+CANHS)+PAH  :1662
    float sink = FADD(s[SC_FIRA], s[SC_FSH]);
    sink = FADD(sink, s[SC_FCEV]);
    sink = FADD(sink, s[SC_FGEV]);
    sink = FADD(sink, s[SC_FCTR]);
    sink = FADD(sink, s[SC_SSOIL]);
    sink = FADD(sink, s[SC_FIRR]);
    sink = FADD(sink, s[SC_CANHS]);
    const float erreng =
        FADD(FSUB(FADD(s[SC_SAV], s[SC_SAG]), sink), s[SC_PAH]);
    o[OU_ERRENG] = erreng;
    if (f_abs(erreng) > ERRENG_TOL) {                                // :1665
        status[c] = 2;
        return;
    }

    float acc_dwater = s[SC_ACC_DWATER];
    float acc_prcp   = s[SC_ACC_PRCP];
    float acc_ecan   = s[SC_ACC_ECAN];
    float acc_etran  = s[SC_ACC_ETRAN];
    float acc_edir   = s[SC_ACC_EDIR];
    float errwat     = 0.0f;
    float end_wb     = 0.0f;

    if (ist[c] == 1) {                                          // :1696  soil
        const float *m = smc + (size_t)c * NSOIL;
        const float *d = dzsnso + (size_t)c * NSOIL;

        end_wb = FADD(FADD(FADD(s[SC_CANLIQ], s[SC_CANICE]), s[SC_SNEQV]),
                      s[SC_WA]);                                     // :1697
        for (int k = 0; k < NSOIL; ++k) {                            // :1698
            // END_WB + SMC(IZ) * DZSNSO(IZ) * 1000.0 -- the contraction the
            // FMUL/FADD split exists to prevent.
            end_wb = FADD(end_wb, FMUL(FMUL(m[k], d[k]), 1000.0f));  // :1699
        }

        acc_dwater = FADD(acc_dwater, FSUB(end_wb, s[SC_BEG_WB]));   // :1702
        acc_prcp   = FADD(acc_prcp,  FMUL(s[SC_PRCP],  s[SC_DT]));   // :1703
        acc_ecan   = FADD(acc_ecan,  FMUL(s[SC_ECAN],  s[SC_DT]));   // :1704
        acc_etran  = FADD(acc_etran, FMUL(s[SC_ETRAN], s[SC_DT]));   // :1705
        acc_edir   = FADD(acc_edir,  FMUL(s[SC_EDIR],  s[SC_DT]));   // :1706

        if (calc[c]) {                                               // :1708
            float inner = FADD(acc_prcp,
                               FMUL(s[SC_IRFIRATE], 1000.0f));       // :1709
            inner = FADD(inner, FMUL(s[SC_IRMIRATE], 1000.0f));
            inner = FSUB(inner, acc_ecan);
            inner = FSUB(inner, acc_etran);                          // :1710
            inner = FSUB(inner, acc_edir);
            inner = FSUB(inner, s[SC_RUNSRF]);
            inner = FSUB(inner, s[SC_RUNSUB]);
            inner = FSUB(inner, s[SC_QTLDRN]);
            errwat = FSUB(acc_dwater, inner);

            // WRF_HYDRO is not defined, so 1712-1728 is live.
            if (f_abs(errwat) > ERRWAT_TOL) {                        // :1713
                status[c] = 3;
            }
        }
    } else {                                                    // :1733  lake
        errwat = 0.0f;                                               // :1734
    }

    o[OU_ERRWAT]     = errwat;
    o[OU_ACC_DWATER] = acc_dwater;
    o[OU_ACC_PRCP]   = acc_prcp;
    o[OU_ACC_ECAN]   = acc_ecan;
    o[OU_ACC_ETRAN]  = acc_etran;
    o[OU_ACC_EDIR]   = acc_edir;
    o[OU_END_WB]     = end_wb;
}

// ---------------------------------------------------------------------------
// NOAHMP_SFLX's own marshalling -- 827-849
// ---------------------------------------------------------------------------
//
// DZSNSO, TROOT and BEG_WB, the three things NOAHMP_SFLX computes for itself
// before it calls PHENOLOGY.  Column arrays are `-NSNOW+1 .. NSOIL` flattened
// as element 0 == layer -NSNOW+1, the same convention every other Noah-MP
// kernel in this directory uses.
//
// The slots of DZSNSO below ISNOW+1 are never written by WRF -- they are
// NOAHMP_SFLX's own stack residue.  This kernel writes 0.0f there, which is
// the defined value and what gpuwm/core/noahmp_sflx.py writes;
// tests/test_noahmp_sflx.py proves no whole-column output depends on them.

extern "C" __global__
void k_sflx_marshal(const float *__restrict__ zsnso,  // [ncol][NFULL]
                    const float *__restrict__ stc,    // [ncol][NFULL]
                    const float *__restrict__ smc,    // [ncol][NSOIL]
                    const float *__restrict__ zsoil,  // [ncol][NSOIL]
                    const float *__restrict__ canliq, // [ncol]
                    const float *__restrict__ canice, // [ncol]
                    const float *__restrict__ sneqv,  // [ncol]
                    const float *__restrict__ wa,     // [ncol]
                    const int   *__restrict__ isnow,  // [ncol]
                    const int   *__restrict__ nroot,  // [ncol]
                    const int   *__restrict__ ist,    // [ncol]
                    float *__restrict__ dzsnso,       // [ncol][NFULL]
                    float *__restrict__ troot,        // [ncol]
                    float *__restrict__ beg_wb,       // [ncol]
                    int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;

    const float *zs = zsnso + (size_t)c * NFULL;
    const float *tc = stc   + (size_t)c * NFULL;
    const float *m  = smc   + (size_t)c * NSOIL;
    const float *zl = zsoil + (size_t)c * NSOIL;
    float *dz = dzsnso + (size_t)c * NFULL;

    const int is = isnow[c];

    // -- snow/soil layer thickness, 827-833 --------------------------------
    for (int k = 0; k < NFULL; ++k) dz[k] = 0.0f;
    for (int iz = is + 1; iz <= NSOIL; ++iz) {                       // :827
        const int k = iz + NSNOW - 1;
        if (iz == is + 1) {                                          // :828
            dz[k] = FSUB(0.0f, zs[k]);                               // :829
        } else {
            dz[k] = FSUB(zs[k - 1], zs[k]);                          // :831
        }
    }

    // -- root-zone temperature, 837-840 ------------------------------------
    const int nr = nroot[c];
    const float denom = FSUB(0.0f, zl[nr - 1]);
    float tr = 0.0f;                                                 // :837
    for (int iz = 1; iz <= nr; ++iz) {                               // :838
        const int k = iz + NSNOW - 1;
        // TROOT + STC(IZ)*DZSNSO(IZ)/(-ZSOIL(NROOT)): the multiply and the
        // divide are separate rounding boundaries, and the add is a third.
        tr = FADD(tr, FDIV(FMUL(tc[k], dz[k]), denom));               // :839
    }
    troot[c] = tr;

    // -- total water storage at step entry, 844-849 ------------------------
    float wb = 0.0f;
    if (ist[c] == 1) {                                               // :844
        wb = FADD(FADD(FADD(canliq[c], canice[c]), sneqv[c]), wa[c]); // :845
        for (int iz = 1; iz <= NSOIL; ++iz) {                        // :846
            const int k = iz + NSNOW - 1;
            wb = FADD(wb, FMUL(FMUL(m[iz - 1], dz[k]), 1000.0f));    // :847
        }
    }
    beg_wb[c] = wb;
}
