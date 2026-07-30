// WRF v4.6.1 Noah-MP ENERGY assembly on the device, pinned at commit
// d66e442fccc04111067e29274c9f9eaccc3cef28
// (phys/module_sf_noahmplsm.F:1741-2396).
//
// WHAT THIS KERNEL IS, AND WHAT IT IS NOT
// ---------------------------------------
// ENERGY is a composition.  Six subsystems -- THERMOPROP, RADIATION,
// VEGE_FLUX, BARE_FLUX, TSNOSOI, PHASECHANGE -- each already have a device
// port in a sibling .cu, each already at max_ulp 0 against its own fixture.
// This file is the arithmetic ENERGY *itself* owns and nothing else: the
// roughness/displacement geometry, the snow-cover fraction, the soil-water
// stress factors, the two psychrometric constants, and the tile average that
// turns a vegetated answer and a bare answer into one column.
//
// It deliberately does NOT re-run the six subsystems on the device.  Composing
// them into one translation unit is currently blocked, and by something worth
// stating rather than working around:
//
//   * noahmp_radiation.cu, noahmp_vegeflux.cu and noahmp_bareflux.cu each
//     define their own glibc_logf / glibc_expf / glibc_powf / powf_log2_inline
//     / powf_exp2_inline / f_min / f_max, so any two of them in one
//     translation unit are duplicate definitions;
//   * noahmp_thermal.cu exposes TSNOSOI and PHASECHANGE only as
//     `extern "C" __global__` entry points, with no reusable __device__ core;
//   * every one of those entry points takes its own lane's flat fixture
//     packing, not a physical argument list.
//
// Fixing that means editing four other lanes' files to hoist a single device
// libm and extract device cores.  That is a real and worthwhile refactor and
// it is not this lane's to make.  So the claim this file supports is exactly:
// **ENERGY's own arithmetic reproduces the pinned fixture bit for bit on the
// GPU**, with the subsystem results taken from the same pinned fixture.
//
// EVERY arithmetic operation goes through __fadd_rn / __fsub_rn / __fmul_rn /
// __fdiv_rn.  NVRTC defaults to --fmad=true, and contraction is the dominant
// bitwise hazard: gfortran on x86-64 emits no FMA at -O0, so a contracted
// a*b+c on the device is a different number.  The tile average is dense in
// exactly that shape (FVEG*IRG + (1-FVEG)*IRB), so do not "simplify" these
// back to infix operators.
//
// Constant tables live in __constant__ memory, never in local literal arrays:
// ptxas 12.x's constant folder does not honour round-to-nearest-even when it
// folds FP32 literals, and __fsub_rn pins the hardware rounding mode, not the
// folder.  tests/test_fp32_tie_folding_gpu.py guards that.
//
// COMPOSITION: compiled after noahmp_leaves.cu (see noahmp_energy_gpu.py).
// ENERGY needs glibc 2.39's powf and expf on the device and there must be
// exactly one transcription of those in the tree; r_pow and r_exp come from
// there.  expm1f and tanhf are new -- ENERGY's FSNO is the only TANH in
// Noah-MP -- and they are transcribed here, matching
// gpuwm/core/noahmp_libm.py statement for statement.  Keep the two in step.

#ifndef AD
#define AD(a, b) __fadd_rn((a), (b))
#define SU(a, b) __fsub_rn((a), (b))
#define MU(a, b) __fmul_rn((a), (b))
#define DV(a, b) __fdiv_rn((a), (b))
#endif

// module_sf_noahmplsm.F:204-220
#ifndef NMP_TFRZ
#define NMP_TFRZ 273.16f
#endif
#define NMPE_SB 5.67e-08f
#define NMPE_GRAV 9.80616f
#define NMPE_RW 461.269f
#define NMPE_CPAIR 1004.64f
#define NMPE_HSUB 2.8440e06f
#define NMPE_HVAP 2.5104e06f
// ENERGY's own PARAMETERs, :2028-2030
#define NMPE_MPE 1.0e-6f
#define NMPE_Z0 0.002f

#define NMPE_NSOIL 4

// ---------------------------------------------------------------------------
// glibc 2.39 expm1f / tanhf -- sysdeps/ieee754/flt-32/{s_expm1f,s_tanhf}.c
//
// FSNO = TANH(SNOWH / (SCFFAC*FMELT)) at :2072 is the only TANH in Noah-MP,
// and it is not interchangeable with an FP64 shim: over the 204,064,836 FP32
// inputs in [1e-6, 22], (float)tanh((double)x) disagrees with glibc's tanhf on
// 23.8% of them.  These are still the 1993 SunPro routines, which is why.
// ---------------------------------------------------------------------------
__constant__ float NMPE_EXPM1F_Q[5] = {
    -3.3333335072e-02f, 1.5873016091e-03f, -7.9365076090e-05f,
    4.0082177293e-06f, -2.0109921195e-07f};
__constant__ float NMPE_EXPM1F_C[6] = {
    1.0e+30f,           // huge
    1.0e-30f,           // tiny
    8.8721679688e+01f,  // o_threshold
    6.9313812256e-01f,  // ln2_hi
    9.0580006145e-06f,  // ln2_lo
    1.4426950216e+00f}; // invln2

__device__ __forceinline__ unsigned int nmpe_bits(float x)
{
    return __float_as_uint(x);
}

__device__ float nmpe_expm1f(float x)
{
    const float one = 1.0f;
    const float hugev = NMPE_EXPM1F_C[0];
    const float tinyv = NMPE_EXPM1F_C[1];
    const float ln2_hi = NMPE_EXPM1F_C[3];
    const float ln2_lo = NMPE_EXPM1F_C[4];

    unsigned int hx = nmpe_bits(x);
    unsigned int xsb = hx & 0x80000000u;
    hx &= 0x7fffffffu;
    float c = 0.0f;
    float hi, lo, t, e, hxs, hfx, r1, y;
    int k;

    if (hx >= 0x4195b844u) {                 // |x| >= 27*ln2
        if (hx >= 0x42b17218u) {             // |x| >= 88.721...
            if (hx > 0x7f800000u) return AD(x, x);
            if (hx == 0x7f800000u) return (xsb == 0u) ? x : -1.0f;
            if (x > NMPE_EXPM1F_C[2]) return MU(hugev, hugev);
        }
        if (xsb != 0u) return SU(tinyv, one);
    }

    if (hx > 0x3eb17218u) {                  // |x| > 0.5*ln2
        if (hx < 0x3F851592u) {              // |x| < 1.5*ln2
            if (xsb == 0u) { hi = SU(x, ln2_hi); lo = ln2_lo;  k = 1; }
            else           { hi = AD(x, ln2_hi); lo = -ln2_lo; k = -1; }
        } else {
            k = (int)AD(MU(NMPE_EXPM1F_C[5], x), (xsb == 0u) ? 0.5f : -0.5f);
            t = (float)k;
            hi = SU(x, MU(t, ln2_hi));       // t*ln2_hi is exact here
            lo = MU(t, ln2_lo);
        }
        x = SU(hi, lo);
        c = SU(SU(hi, x), lo);
    } else if (hx < 0x33000000u) {           // |x| < 2**-25
        t = AD(hugev, x);
        return SU(x, SU(t, AD(hugev, x)));
    } else {
        k = 0;
    }

    hfx = MU(0.5f, x);
    hxs = MU(x, hfx);
    r1 = MU(hxs, NMPE_EXPM1F_Q[4]);
    r1 = MU(hxs, AD(NMPE_EXPM1F_Q[3], r1));
    r1 = MU(hxs, AD(NMPE_EXPM1F_Q[2], r1));
    r1 = MU(hxs, AD(NMPE_EXPM1F_Q[1], r1));
    r1 = MU(hxs, AD(NMPE_EXPM1F_Q[0], r1));
    r1 = AD(one, r1);
    t = SU(3.0f, MU(r1, hfx));
    e = MU(hxs, DV(SU(r1, t), SU(6.0f, MU(x, t))));
    if (k == 0) return SU(x, SU(MU(x, e), hxs));

    e = SU(MU(x, SU(e, c)), c);
    e = SU(e, hxs);
    if (k == -1) return SU(MU(0.5f, SU(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f) return MU(-2.0f, SU(e, AD(x, 0.5f)));
        return AD(one, MU(2.0f, SU(x, e)));
    }
    if (k <= -2 || k > 56) {
        y = SU(one, SU(e, x));
        y = __uint_as_float(nmpe_bits(y) + ((unsigned int)k << 23));
        return SU(y, one);
    }
    if (k < 23) {
        t = __uint_as_float(0x3f800000u - (0x1000000u >> k));  // 1 - 2**-k
        y = SU(t, SU(e, x));
    } else {
        t = __uint_as_float(((unsigned int)(0x7f - k)) << 23); // 2**-k
        y = SU(x, AD(e, t));
        y = AD(y, one);
    }
    return __uint_as_float(nmpe_bits(y) + ((unsigned int)k << 23));
}

__device__ float nmpe_tanhf(float x)
{
    const float one = 1.0f;
    int jx = (int)nmpe_bits(x);
    int ix = jx & 0x7fffffff;
    float t, z;

    if (ix >= 0x7f800000) {
        return (jx >= 0) ? AD(DV(one, x), one) : SU(DV(one, x), one);
    }
    if (ix < 0x41b00000) {                   // |x| < 22
        if (ix == 0) return x;
        if (ix < 0x24000000) return MU(x, AD(one, x));   // |x| < 2**-55
        float ax = fabsf(x);
        if (ix >= 0x3f800000) {              // |x| >= 1
            t = nmpe_expm1f(MU(2.0f, ax));
            z = SU(one, DV(2.0f, AD(t, 2.0f)));
        } else {
            t = nmpe_expm1f(MU(-2.0f, ax));
            z = DV(-t, AD(t, 2.0f));
        }
    } else {
        z = SU(one, NMPE_EXPM1F_C[1]);
    }
    return (jx >= 0) ? z : -z;
}

// ---------------------------------------------------------------------------
// Flat slot layout.  Mirrored, name for name, in noahmp_energy_gpu.py; the
// host packs the entry state and the pinned subsystem results from
// gpuwm/data/noahmp/oracle/noahmp-energy.csv, so nothing is repacked twice.
// ---------------------------------------------------------------------------
#define E_UU 0
#define E_VV 1
#define E_ELAI 2
#define E_ESAI 3
#define E_SNOWH 4
#define E_SNEQV 5
#define E_TG 6
#define E_TV 7
#define E_SFCPRS 8
#define E_LWDN 9
#define E_FVEG 10
#define E_ZREF 11
#define E_DT 12
#define E_ACC_SSOIL 13
#define E_MFSNO 14
#define E_SCFFAC 15
#define E_Z0SNO 16
#define E_Z0MVT 17
#define E_HVT 18
#define E_EG 19
#define E_SNOW_EMIS 20
#define E_SH2O 21          // 21..24
#define E_SMCWLT 25        // 25..28
#define E_SMCREF 29        // 29..32
#define E_DZSNSO 33        // 33..36, soil layers 1..NSOIL
#define E_ZSOIL 37         // 37..40
#define E_IRC 41
#define E_IRG 42
#define E_IRB 43
#define E_SHC 44
#define E_SHG 45
#define E_SHB 46
#define E_EVC 47
#define E_EVG 48
#define E_EVB 49
#define E_TR 50
#define E_GHV 51
#define E_GHB 52
#define E_TGV 53
#define E_TGB 54
#define E_T2MV 55
#define E_T2MB 56
#define E_CHV 57
#define E_CHB 58
#define E_EAH 59
#define E_QSFC 60
#define E_Q2V 61
#define E_Q2B 62
#define E_PAHV 63
#define E_PAHG 64
#define E_PAHB 65
// TV appears twice on purpose.  The psychrometric branch at :2211 tests the
// *entry* TV, but the tile average at :2298 uses the value VEGE_FLUX wrote --
// TS = FVEG*TV + (1-FVEG)*TGB is evaluated after the call.  Feeding one slot
// for both silently makes TS wrong on every vegetated column, which is exactly
// how this kernel failed its first run on the device.
#define E_TV_POST 66
#define E_N_IN 67

#define O_FSNO 0
#define O_Z0WRF 1
#define O_BTRAN 2
#define O_BTRANI 3        // 3..6
#define O_LATHEAV 7
#define O_LATHEAG 8
#define O_FROZEN_CANOPY 9
#define O_FROZEN_GROUND 10
#define O_FIRA 11
#define O_FSH 12
#define O_FGEV 13
#define O_SSOIL 14
#define O_FCEV 15
#define O_FCTR 16
#define O_PAH 17
#define O_TG 18
#define O_T2M 19
#define O_TS 20
#define O_CH 21
#define O_Q1 22
#define O_Q2E 23
#define O_EMISSI 24
#define O_TRAD 25
#define O_ACC_SSOIL 26
#define E_N_OUT 27

// ii[0] = NROOT, ii[1] = URBAN_FLAG.  ICE is 0 and IST is 1 in the admitted
// slice, so the ICE==1 emissivity leg (:2145-2147) and the IST==2 lake legs
// (:2076-2082, :2178-2181) are not transcribed here at all.
#define E_NROOT 0
#define E_URBAN 1
#define E_N_INT 2

extern "C" __global__
void noahmp_energy_assembly(const float *in, const int *ii, float *out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float *x = in + (size_t)tid * E_N_IN;
    const int *q = ii + (size_t)tid * E_N_INT;
    float *o = out + (size_t)tid * E_N_OUT;

    const int nroot = q[E_NROOT];
    const bool urban = q[E_URBAN] != 0;

    // :2057  UR = MAX(SQRT(UU**2.0 + VV**2.0), 1.0).  The exponent is a REAL
    // literal, so gfortran emits powf; r_pow reproduces it.  (Measured: over
    // the 167,177,618 FP32 values in [1e-3,1e3] glibc's powf(x,2) is
    // bit-identical to x*x, so this one is faithful rather than load-bearing.)
    float ur = __fsqrt_rn(AD(r_pow(x[E_UU], 2.0f), r_pow(x[E_VV], 2.0f)));
    ur = (ur > 1.0f) ? ur : 1.0f;

    // :2061-2063
    const float vai = AD(x[E_ELAI], x[E_ESAI]);
    const bool veg = vai > 0.0f;

    // :2067-2073  ground snow-cover fraction [Niu and Yang, 2007]
    float fsno = 0.0f;
    if (x[E_SNOWH] > 0.0f) {
        float bdsno = DV(x[E_SNEQV], x[E_SNOWH]);
        float fmelt = r_pow(DV(bdsno, 100.0f), x[E_MFSNO]);
        fsno = nmpe_tanhf(DV(x[E_SNOWH], MU(x[E_SCFFAC], fmelt)));
    }

    // :2084  IST == 1 leg
    float z0mg = AD(MU(NMPE_Z0, SU(1.0f, fsno)), MU(x[E_Z0SNO], fsno));

    // :2089-2097
    float zpdg = x[E_SNOWH];
    float z0m, zpd;
    if (veg) {
        z0m = x[E_Z0MVT];
        zpd = MU(0.65f, x[E_HVT]);
        if (x[E_SNOWH] > zpd) zpd = x[E_SNOWH];
    } else {
        z0m = z0mg;
        zpd = zpdg;
    }

    // :2101-2106  urban override
    if (urban) {
        z0mg = x[E_Z0MVT];
        zpdg = MU(0.65f, x[E_HVT]);
        z0m = z0mg;
        zpd = zpdg;
    }

    // :2139-2147  emissivities (ICE == 0 leg)
    float emv = SU(1.0f, r_exp(DV(-AD(x[E_ELAI], x[E_ESAI]), 1.0f)));
    float emg = AD(MU(x[E_EG], SU(1.0f, fsno)), MU(x[E_SNOW_EMIS], fsno));

    // :2151-2173  BTRAN, OPT_BTR == 1 (Noah)
    float btran = 0.0f;
    float btrani[NMPE_NSOIL];
    for (int k = 0; k < NMPE_NSOIL; ++k) btrani[k] = 0.0f;
    for (int iz = 1; iz <= nroot; ++iz) {
        float gx = DV(SU(x[E_SH2O + iz - 1], x[E_SMCWLT + iz - 1]),
                      SU(x[E_SMCREF + iz - 1], x[E_SMCWLT + iz - 1]));
        gx = (gx > 0.0f) ? gx : 0.0f;
        gx = (gx < 1.0f) ? gx : 1.0f;
        float v = MU(DV(x[E_DZSNSO + iz - 1], -x[E_ZSOIL + nroot - 1]), gx);
        btrani[iz - 1] = (v > NMPE_MPE) ? v : NMPE_MPE;
        btran = AD(btran, btrani[iz - 1]);
    }
    btran = (btran > NMPE_MPE) ? btran : NMPE_MPE;
    for (int iz = 1; iz <= nroot; ++iz)
        btrani[iz - 1] = DV(btrani[iz - 1], btran);

    // :2211-2227  psychrometric constants
    float latheav, latheag;
    int frozen_canopy, frozen_ground;
    if (x[E_TV] > NMP_TFRZ) { latheav = NMPE_HVAP; frozen_canopy = 0; }
    else                    { latheav = NMPE_HSUB; frozen_canopy = 1; }
    if (x[E_TG] > NMP_TFRZ) { latheag = NMPE_HVAP; frozen_ground = 0; }
    else                    { latheag = NMPE_HSUB; frozen_ground = 1; }

    // :2282-2326  the tile average
    const float fveg = x[E_FVEG];
    const float one_m = SU(1.0f, fveg);
    const bool tile_veg = veg && (fveg > 0.0f);
    float fira, fsh, fgev, ssoil, fcev, fctr, pah, tg, t2m, ts, ch, q1, q2e;
    float z0wrf;
    if (tile_veg) {
        fira = AD(AD(MU(fveg, x[E_IRG]), MU(one_m, x[E_IRB])), x[E_IRC]);
        fsh = AD(AD(MU(fveg, x[E_SHG]), MU(one_m, x[E_SHB])), x[E_SHC]);
        fgev = AD(MU(fveg, x[E_EVG]), MU(one_m, x[E_EVB]));
        ssoil = AD(MU(fveg, x[E_GHV]), MU(one_m, x[E_GHB]));
        fcev = x[E_EVC];
        fctr = x[E_TR];
        pah = AD(AD(MU(fveg, x[E_PAHG]), MU(one_m, x[E_PAHB])), x[E_PAHV]);
        tg = AD(MU(fveg, x[E_TGV]), MU(one_m, x[E_TGB]));
        t2m = AD(MU(fveg, x[E_T2MV]), MU(one_m, x[E_T2MB]));
        ts = AD(MU(fveg, x[E_TV_POST]), MU(one_m, x[E_TGB]));
        ch = AD(MU(fveg, x[E_CHV]), MU(one_m, x[E_CHB]));
        q1 = AD(MU(fveg, DV(MU(x[E_EAH], 0.622f),
                            SU(x[E_SFCPRS], MU(0.378f, x[E_EAH])))),
                MU(one_m, x[E_QSFC]));
        q2e = AD(MU(fveg, x[E_Q2V]), MU(one_m, x[E_Q2B]));
        z0wrf = z0m;
    } else {
        fira = x[E_IRB];
        fsh = x[E_SHB];
        fgev = x[E_EVB];
        ssoil = x[E_GHB];
        tg = x[E_TGB];
        t2m = x[E_T2MB];
        fcev = 0.0f;
        fctr = 0.0f;
        pah = x[E_PAHB];
        ts = tg;
        ch = x[E_CHB];
        q1 = x[E_QSFC];
        q2e = x[E_Q2B];
        z0wrf = z0mg;
    }

    // :2321-2340
    const float fire = AD(x[E_LWDN], fira);
    float emissi = AD(MU(fveg, AD(AD(MU(emg, SU(1.0f, emv)), emv),
                                  MU(MU(emv, SU(1.0f, emv)),
                                     SU(1.0f, emg)))),
                      MU(SU(1.0f, fveg), emg));
    float trad = r_pow(DV(SU(fire, MU(SU(1.0f, emissi), x[E_LWDN])),
                          MU(emissi, NMPE_SB)), 0.25f);

    // :2350  ACC_SSOIL is the only accumulator ENERGY touches.
    float acc_ssoil = AD(x[E_ACC_SSOIL], ssoil);

    o[O_FSNO] = fsno;
    o[O_Z0WRF] = z0wrf;
    o[O_BTRAN] = btran;
    for (int k = 0; k < NMPE_NSOIL; ++k) o[O_BTRANI + k] = btrani[k];
    o[O_LATHEAV] = latheav;
    o[O_LATHEAG] = latheag;
    o[O_FROZEN_CANOPY] = (float)frozen_canopy;
    o[O_FROZEN_GROUND] = (float)frozen_ground;
    o[O_FIRA] = fira;
    o[O_FSH] = fsh;
    o[O_FGEV] = fgev;
    o[O_SSOIL] = ssoil;
    o[O_FCEV] = fcev;
    o[O_FCTR] = fctr;
    o[O_PAH] = pah;
    o[O_TG] = tg;
    o[O_T2M] = t2m;
    o[O_TS] = ts;
    o[O_CH] = ch;
    o[O_Q1] = q1;
    o[O_Q2E] = q2e;
    o[O_EMISSI] = emissi;
    o[O_TRAD] = trad;
    o[O_ACC_SSOIL] = acc_ssoil;
    (void)ur; (void)zpd; (void)zpdg; (void)latheav;
}

// ---------------------------------------------------------------------------
// Device-libm parity probe.  The CPU and CUDA transcriptions of tanhf/expm1f
// are two separate pieces of code; without this, "they match glibc" is a claim
// about one of them.
// ---------------------------------------------------------------------------
extern "C" __global__
void noahmp_energy_tanhf_probe(const float *x, float *tanh_out,
                               float *expm1_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    tanh_out[tid] = nmpe_tanhf(x[tid]);
    expm1_out[tid] = nmpe_expm1f(x[tid]);
}
