// WOOF's own correctly rounded binary64 exp, log, log10, sin, atan and pow.
//
// THE BREAKAGE THIS PREVENTS (clm-lake column oracle, 2026-10-07): the WRF
// CLM lake runs in binary64 and calls exp, log, log10, sin, atan and pow.
// CUDA's binary64 library is accurate to 1 or 2 units in the last place but
// is not correctly rounded.  In a 112-column regime campaign it returned a
// different word from the host library the native WRF oracle links for 5%
// of exp, 7% of sin, 20% of pow and 25% of atan calls, and the lake's
// sensible and ground heat flux outputs (hfx, grdflx) differed from native
// WRF in 805 words of a simulated day.  Correct rounding is the one binary64
// answer every conforming platform owes, so the lake computes these
// functions here, identically on the host and the device.
//
// Method (textbook two-phase evaluation, written for WOOF; tables and
// coefficients come from tools/lake_wrf461_oracle/f64cr_constants.py).
// exp, log, log10, atan and pow first try a table-driven phase accurate to
// about 2^-64 and keep its result when a rounding test shows the error
// bound cannot straddle a rounding boundary.  Otherwise, and always for
// sin, an accurate phase reduces the argument exactly and evaluates in
// double-double arithmetic to a relative error far below 2^-88, then
// rounds once.  exp near 0 and log near 1, where structured arguments
// (x = k 2^-53, 1 + k 2^-52) put the result on or next to a rounding
// midpoint, carry the error relative to the higher-order terms and round
// with an exact three-term sum.  Where a bound could still straddle a
// boundary the value is returned anyway and the host build counts the
// event in wcr_ambiguous_count, so a census can show it never happened.
// Every operation is an IEEE binary64 add, multiply or divide, or a fused
// multiply-add that forms an exact product error, so the result does not
// depend on contraction settings.
//
// Domain: exp, log, log10, atan and pow cover every binary64 argument,
// including the IEEE special cases.  sin reduces with a 159-bit pi/2 for
// |x| < 2^20 and uses the platform sin beyond that (the lake passes a
// latitude in radians).  pow(x, 2) is the correctly rounded product x * x;
// other exact midpoint results of pow (possible only for special x and
// integer y) cannot pass the rounding test and are counted ambiguous.
#ifndef WOOF_F64CR
#define WOOF_F64CR
#ifdef __CUDACC__
#define WCR_HD __device__ inline
#define WCR_TABLE static __device__ const
#define WCR_FMA(a, b, c) __fma_rn((a), (b), (c))
#define WCR_DIV(a, b) __ddiv_rn((a), (b))
#define WCR_RINT(a) rint(a)
#define WCR_BITS(x) __double_as_longlong(x)
#define WCR_FROM_BITS(i) __longlong_as_double(i)
#define WCR_PLATFORM_SIN(x) sin(x)
#define WCR_AMBIGUOUS()
#else
#include <cmath>
#include <cstring>
#define WCR_HD inline
#define WCR_TABLE static const
#define WCR_FMA(a, b, c) std::fma((a), (b), (c))
#define WCR_DIV(a, b) ((a) / (b))
#define WCR_RINT(a) std::rint(a)
inline long long wcr_bits(double x) {long long i; std::memcpy(&i, &x, 8); return i;}
inline double wcr_from_bits(long long i) {double x; std::memcpy(&x, &i, 8); return x;}
#define WCR_BITS(x) wcr_bits(x)
#define WCR_FROM_BITS(i) wcr_from_bits(i)
#define WCR_PLATFORM_SIN(x) std::sin(x)
inline unsigned long long wcr_ambiguous_count = 0;
#define WCR_AMBIGUOUS() (++wcr_ambiguous_count)
#endif

struct wcr_dd {double h, l;};

// CONSTANTS BEGIN (tools/lake_wrf461_oracle/f64cr_constants.py)
#define WCR_LN2_1 0x1.62e42fefa39efp-1
#define WCR_LN2_2 0x1.abc9e3b39803fp-56
#define WCR_LN2_3 0x1.7b57a079a1934p-111
#define WCR_INV_LN2 0x1.71547652b82fep+0
#define WCR_PIO2_1 0x1.921fb54442d18p+0
#define WCR_PIO2_2 0x1.1a62633145c07p-54
#define WCR_PIO2_3 (-0x1.f1976b7ed8fbcp-110)
#define WCR_INV_PIO2 0x1.45f306dc9c883p-1
#define WCR_INV_LN10_1 0x1.bcb7b1526e50ep-2
#define WCR_INV_LN10_2 0x1.95355baaafad3p-57
#define WCR_SQRT2 0x1.6a09e667f3bcdp+0
WCR_TABLE wcr_dd wcr_inv_fact[30] = {
    {0x1.0000000000000p+0,0x0p+0},
    {0x1.0000000000000p+0,0x0p+0},
    {0x1.0000000000000p-1,0x0p+0},
    {0x1.5555555555555p-3,0x1.5555555555555p-57},
    {0x1.5555555555555p-5,0x1.5555555555555p-59},
    {0x1.1111111111111p-7,0x1.1111111111111p-63},
    {0x1.6c16c16c16c17p-10,(-0x1.f49f49f49f49fp-65)},
    {0x1.a01a01a01a01ap-13,0x1.a01a01a01a01ap-73},
    {0x1.a01a01a01a01ap-16,0x1.a01a01a01a01ap-76},
    {0x1.71de3a556c734p-19,(-0x1.c154f8ddc6c00p-73)},
    {0x1.27e4fb7789f5cp-22,0x1.cbbc05b4fa99ap-76},
    {0x1.ae64567f544e4p-26,(-0x1.c062e06d1f209p-80)},
    {0x1.1eed8eff8d898p-29,(-0x1.2aec959e14c06p-83)},
    {0x1.6124613a86d09p-33,0x1.f28e0cc748ebep-87},
    {0x1.93974a8c07c9dp-37,0x1.05d6f8a2efd1fp-92},
    {0x1.ae7f3e733b81fp-41,0x1.1d8656b0ee8cbp-97},
    {0x1.ae7f3e733b81fp-45,0x1.1d8656b0ee8cbp-101},
    {0x1.952c77030ad4ap-49,0x1.ac981465ddc6cp-103},
    {0x1.6827863b97d97p-53,0x1.eec01221a8b0bp-107},
    {0x1.2f49b46814157p-57,0x1.2650f61dbdcb4p-112},
    {0x1.e542ba4020225p-62,0x1.ea72b4afe3c2fp-120},
    {0x1.71b8ef6dcf572p-66,(-0x1.d043ae40c4647p-120)},
    {0x1.0ce396db7f853p-70,(-0x1.aebcdbd20331cp-124)},
    {0x1.761b41316381ap-75,(-0x1.3423c7d91404fp-130)},
    {0x1.f2cf01972f578p-80,(-0x1.9ada5fcc1ab14p-135)},
    {0x1.3f3ccdd165fa9p-84,(-0x1.58ddadf344487p-139)},
    {0x1.88e85fc6a4e5ap-89,(-0x1.71c37ebd16540p-143)},
    {0x1.d1ab1c2dccea3p-94,0x1.054d0c78aea14p-149},
    {0x1.0a18a2635085dp-98,0x1.b9e2e28e1aa54p-153},
    {0x1.259f98b4358adp-103,0x1.eaf8c39dd9bc5p-157}
};
WCR_TABLE wcr_dd wcr_inv_odd[21] = {
    {0x1.0000000000000p+0,0x0p+0},
    {0x1.5555555555555p-2,0x1.5555555555555p-56},
    {0x1.999999999999ap-3,(-0x1.999999999999ap-57)},
    {0x1.2492492492492p-3,0x1.2492492492492p-57},
    {0x1.c71c71c71c71cp-4,0x1.c71c71c71c71cp-58},
    {0x1.745d1745d1746p-4,(-0x1.745d1745d1746p-59)},
    {0x1.3b13b13b13b14p-4,(-0x1.3b13b13b13b14p-58)},
    {0x1.1111111111111p-4,0x1.1111111111111p-60},
    {0x1.e1e1e1e1e1e1ep-5,0x1.e1e1e1e1e1e1ep-61},
    {0x1.af286bca1af28p-5,0x1.af286bca1af28p-59},
    {0x1.8618618618618p-5,0x1.8618618618618p-59},
    {0x1.642c8590b2164p-5,0x1.642c8590b2164p-60},
    {0x1.47ae147ae147bp-5,(-0x1.eb851eb851eb8p-61)},
    {0x1.2f684bda12f68p-5,0x1.2f684bda12f68p-59},
    {0x1.1a7b9611a7b96p-5,0x1.1a7b9611a7b96p-61},
    {0x1.0842108421084p-5,0x1.0842108421084p-60},
    {0x1.f07c1f07c1f08p-6,(-0x1.f07c1f07c1f08p-61)},
    {0x1.d41d41d41d41dp-6,0x1.0750750750750p-60},
    {0x1.bacf914c1bad0p-6,(-0x1.bacf914c1bad0p-60)},
    {0x1.a41a41a41a41ap-6,0x1.0690690690690p-60},
    {0x1.8f9c18f9c18fap-6,(-0x1.f3831f3831f38p-61)}
};
WCR_TABLE wcr_dd wcr_inv_int[17] = {
    {0x0p+0,0x0p+0},
    {0x1.0000000000000p+0,0x0p+0},
    {0x1.0000000000000p-1,0x0p+0},
    {0x1.5555555555555p-2,0x1.5555555555555p-56},
    {0x1.0000000000000p-2,0x0p+0},
    {0x1.999999999999ap-3,(-0x1.999999999999ap-57)},
    {0x1.5555555555555p-3,0x1.5555555555555p-57},
    {0x1.2492492492492p-3,0x1.2492492492492p-57},
    {0x1.0000000000000p-3,0x0p+0},
    {0x1.c71c71c71c71cp-4,0x1.c71c71c71c71cp-58},
    {0x1.999999999999ap-4,(-0x1.999999999999ap-58)},
    {0x1.745d1745d1746p-4,(-0x1.745d1745d1746p-59)},
    {0x1.5555555555555p-4,0x1.5555555555555p-58},
    {0x1.3b13b13b13b14p-4,(-0x1.3b13b13b13b14p-58)},
    {0x1.2492492492492p-4,0x1.2492492492492p-58},
    {0x1.1111111111111p-4,0x1.1111111111111p-60},
    {0x1.0000000000000p-4,0x0p+0}
};
WCR_TABLE wcr_dd wcr_atan_j16[17] = {
    {0x0p+0,0x0p+0},
    {0x1.ff55bb72cfdeap-5,(-0x1.c934d86d23f1dp-60)},
    {0x1.fd5ba9aac2f6ep-4,(-0x1.cd37686760c17p-59)},
    {0x1.7b97b4bce5b02p-3,0x1.347b0b4f881cap-58},
    {0x1.f5b75f92c80ddp-3,0x1.8ab6e3cf7afbdp-57},
    {0x1.362773707ebccp-2,(-0x1.963a544b672d8p-57)},
    {0x1.6f61941e4def1p-2,(-0x1.c63aae6f6e918p-56)},
    {0x1.a64eec3cc23fdp-2,(-0x1.24dec1b50b7ffp-56)},
    {0x1.dac670561bb4fp-2,0x1.a2b7f222f65e2p-56},
    {0x1.0657e94db30d0p-1,(-0x1.d5b495f6349e6p-56)},
    {0x1.1e00babdefeb4p-1,(-0x1.928df287a668fp-58)},
    {0x1.345f01cce37bbp-1,0x1.1021137c71102p-55},
    {0x1.4978fa3269ee1p-1,0x1.2419a87f2a458p-56},
    {0x1.5d58987169b18p-1,0x1.0028e4bc5e7cap-57},
    {0x1.700a7c5784634p-1,(-0x1.8c34d25aadef6p-56)},
    {0x1.819d0b7158a4dp-1,(-0x1.bf76229d3b917p-56)},
    {0x1.921fb54442d18p-1,0x1.1a62633145c07p-55}
};
#define WCR_LN2_64_1 0x1.62e42fefa39efp-7
#define WCR_LN2_64_2 0x1.abc9e3b39803fp-62
#define WCR_INV_LN2_64 0x1.71547652b82fep+6
WCR_TABLE wcr_dd wcr_exp2_j64[64] = {
    {0x1.0000000000000p+0,0x0p+0},
    {0x1.02c9a3e778061p+0,(-0x1.19083535b085dp-56)},
    {0x1.059b0d3158574p+0,0x1.d73e2a475b465p-55},
    {0x1.0874518759bc8p+0,0x1.186be4bb284ffp-57},
    {0x1.0b5586cf9890fp+0,0x1.8a62e4adc610bp-54},
    {0x1.0e3ec32d3d1a2p+0,0x1.03a1727c57b53p-59},
    {0x1.11301d0125b51p+0,(-0x1.6c51039449b3ap-54)},
    {0x1.1429aaea92de0p+0,(-0x1.32fbf9af1369ep-54)},
    {0x1.172b83c7d517bp+0,(-0x1.19041b9d78a76p-55)},
    {0x1.1a35beb6fcb75p+0,0x1.e5b4c7b4968e4p-55},
    {0x1.1d4873168b9aap+0,0x1.e016e00a2643cp-54},
    {0x1.2063b88628cd6p+0,0x1.dc775814a8495p-55},
    {0x1.2387a6e756238p+0,0x1.9b07eb6c70573p-54},
    {0x1.26b4565e27cddp+0,0x1.2bd339940e9d9p-55},
    {0x1.29e9df51fdee1p+0,0x1.612e8afad1255p-55},
    {0x1.2d285a6e4030bp+0,0x1.0024754db41d5p-54},
    {0x1.306fe0a31b715p+0,0x1.6f46ad23182e4p-55},
    {0x1.33c08b26416ffp+0,0x1.32721843659a6p-54},
    {0x1.371a7373aa9cbp+0,(-0x1.63aeabf42eae2p-54)},
    {0x1.3a7db34e59ff7p+0,(-0x1.5e436d661f5e3p-56)},
    {0x1.3dea64c123422p+0,0x1.ada0911f09ebcp-55},
    {0x1.4160a21f72e2ap+0,(-0x1.ef3691c309278p-58)},
    {0x1.44e086061892dp+0,0x1.89b7a04ef80d0p-59},
    {0x1.486a2b5c13cd0p+0,0x1.3c1a3b69062f0p-56},
    {0x1.4bfdad5362a27p+0,0x1.d4397afec42e2p-56},
    {0x1.4f9b2769d2ca7p+0,(-0x1.4b309d25957e3p-54)},
    {0x1.5342b569d4f82p+0,(-0x1.07abe1db13cadp-55)},
    {0x1.56f4736b527dap+0,0x1.9bb2c011d93adp-54},
    {0x1.5ab07dd485429p+0,0x1.6324c054647adp-54},
    {0x1.5e76f15ad2148p+0,0x1.ba6f93080e65ep-54},
    {0x1.6247eb03a5585p+0,(-0x1.383c17e40b497p-54)},
    {0x1.6623882552225p+0,(-0x1.bb60987591c34p-54)},
    {0x1.6a09e667f3bcdp+0,(-0x1.bdd3413b26456p-54)},
    {0x1.6dfb23c651a2fp+0,(-0x1.bbe3a683c88abp-57)},
    {0x1.71f75e8ec5f74p+0,(-0x1.16e4786887a99p-55)},
    {0x1.75feb564267c9p+0,(-0x1.0245957316dd3p-54)},
    {0x1.7a11473eb0187p+0,(-0x1.41577ee04992fp-55)},
    {0x1.7e2f336cf4e62p+0,0x1.05d02ba15797ep-56},
    {0x1.82589994cce13p+0,(-0x1.d4c1dd41532d8p-54)},
    {0x1.868d99b4492edp+0,(-0x1.fc6f89bd4f6bap-54)},
    {0x1.8ace5422aa0dbp+0,0x1.6e9f156864b27p-54},
    {0x1.8f1ae99157736p+0,0x1.5cc13a2e3976cp-55},
    {0x1.93737b0cdc5e5p+0,(-0x1.75fc781b57ebcp-57)},
    {0x1.97d829fde4e50p+0,(-0x1.d185b7c1b85d1p-54)},
    {0x1.9c49182a3f090p+0,0x1.c7c46b071f2bep-56},
    {0x1.a0c667b5de565p+0,(-0x1.359495d1cd533p-54)},
    {0x1.a5503b23e255dp+0,(-0x1.d2f6edb8d41e1p-54)},
    {0x1.a9e6b5579fdbfp+0,0x1.0fac90ef7fd31p-54},
    {0x1.ae89f995ad3adp+0,0x1.7a1cd345dcc81p-54},
    {0x1.b33a2b84f15fbp+0,(-0x1.2805e3084d708p-57)},
    {0x1.b7f76f2fb5e47p+0,(-0x1.5584f7e54ac3bp-56)},
    {0x1.bcc1e904bc1d2p+0,0x1.23dd07a2d9e84p-55},
    {0x1.c199bdd85529cp+0,0x1.11065895048ddp-55},
    {0x1.c67f12e57d14bp+0,0x1.2884dff483cadp-54},
    {0x1.cb720dcef9069p+0,0x1.503cbd1e949dbp-56},
    {0x1.d072d4a07897cp+0,(-0x1.cbc3743797a9cp-54)},
    {0x1.d5818dcfba487p+0,0x1.2ed02d75b3707p-55},
    {0x1.da9e603db3285p+0,0x1.c2300696db532p-54},
    {0x1.dfc97337b9b5fp+0,(-0x1.1a5cd4f184b5cp-54)},
    {0x1.e502ee78b3ff6p+0,0x1.39e8980a9cc8fp-55},
    {0x1.ea4afa2a490dap+0,(-0x1.e9c23179c2893p-54)},
    {0x1.efa1bee615a27p+0,0x1.dc7f486a4b6b0p-54},
    {0x1.f50765b6e4540p+0,0x1.9d3e12dd8a18bp-54},
    {0x1.fa7c1819e90d8p+0,0x1.74853f3a5931ep-55}
};
WCR_TABLE double wcr_log_inv[97] = {
    0x1.5555555555555p+0,
    0x1.51d07eae2f815p+0,
    0x1.4e5e0a72f0539p+0,
    0x1.4afd6a052bf5bp+0,
    0x1.47ae147ae147bp+0,
    0x1.446f86562d9fbp+0,
    0x1.4141414141414p+0,
    0x1.3e22cbce4a902p+0,
    0x1.3b13b13b13b14p+0,
    0x1.3813813813814p+0,
    0x1.3521cfb2b78c1p+0,
    0x1.323e34a2b10bfp+0,
    0x1.2f684bda12f68p+0,
    0x1.2c9fb4d812ca0p+0,
    0x1.29e4129e4129ep+0,
    0x1.27350b8812735p+0,
    0x1.2492492492492p+0,
    0x1.21fb78121fb78p+0,
    0x1.1f7047dc11f70p+0,
    0x1.1cf06ada2811dp+0,
    0x1.1a7b9611a7b96p+0,
    0x1.1811811811812p+0,
    0x1.15b1e5f75270dp+0,
    0x1.135c81135c811p+0,
    0x1.1111111111111p+0,
    0x1.0ecf56be69c90p+0,
    0x1.0c9714fbcda3bp+0,
    0x1.0a6810a6810a7p+0,
    0x1.0842108421084p+0,
    0x1.0624dd2f1a9fcp+0,
    0x1.0410410410410p+0,
    0x1.0204081020408p+0,
    0x1.0000000000000p+0,
    0x1.fc07f01fc07f0p-1,
    0x1.f81f81f81f820p-1,
    0x1.f44659e4a4271p-1,
    0x1.f07c1f07c1f08p-1,
    0x1.ecc07b301ecc0p-1,
    0x1.e9131abf0b767p-1,
    0x1.e573ac901e574p-1,
    0x1.e1e1e1e1e1e1ep-1,
    0x1.de5d6e3f8868ap-1,
    0x1.dae6076b981dbp-1,
    0x1.d77b654b82c34p-1,
    0x1.d41d41d41d41dp-1,
    0x1.d0cb58f6ec074p-1,
    0x1.cd85689039b0bp-1,
    0x1.ca4b3055ee191p-1,
    0x1.c71c71c71c71cp-1,
    0x1.c3f8f01c3f8f0p-1,
    0x1.c0e070381c0e0p-1,
    0x1.bdd2b899406f7p-1,
    0x1.bacf914c1bad0p-1,
    0x1.b7d6c3dda338bp-1,
    0x1.b4e81b4e81b4fp-1,
    0x1.b2036406c80d9p-1,
    0x1.af286bca1af28p-1,
    0x1.ac5701ac5701bp-1,
    0x1.a98ef606a63bep-1,
    0x1.a6d01a6d01a6dp-1,
    0x1.a41a41a41a41ap-1,
    0x1.a16d3f97a4b02p-1,
    0x1.9ec8e951033d9p-1,
    0x1.9c2d14ee4a102p-1,
    0x1.999999999999ap-1,
    0x1.970e4f80cb872p-1,
    0x1.948b0fcd6e9e0p-1,
    0x1.920fb49d0e229p-1,
    0x1.8f9c18f9c18fap-1,
    0x1.8d3018d3018d3p-1,
    0x1.8acb90f6bf3aap-1,
    0x1.886e5f0abb04ap-1,
    0x1.8618618618618p-1,
    0x1.83c977ab2beddp-1,
    0x1.8181818181818p-1,
    0x1.7f405fd017f40p-1,
    0x1.7d05f417d05f4p-1,
    0x1.7ad2208e0ecc3p-1,
    0x1.78a4c8178a4c8p-1,
    0x1.767dce434a9b1p-1,
    0x1.745d1745d1746p-1,
    0x1.724287f46debcp-1,
    0x1.702e05c0b8170p-1,
    0x1.6e1f76b4337c7p-1,
    0x1.6c16c16c16c17p-1,
    0x1.6a13cd1537290p-1,
    0x1.6816816816817p-1,
    0x1.661ec6a5122f9p-1,
    0x1.642c8590b2164p-1,
    0x1.623fa77016240p-1,
    0x1.6058160581606p-1,
    0x1.5e75bb8d015e7p-1,
    0x1.5c9882b931057p-1,
    0x1.5ac056b015ac0p-1,
    0x1.58ed2308158edp-1,
    0x1.571ed3c506b3ap-1,
    0x1.5555555555555p-1
};
WCR_TABLE wcr_dd wcr_log_tab[97] = {
    {(-0x1.269621134db91p-2),(-0x1.e0efadd9db02ap-56)},
    {(-0x1.1bf99635a6b95p-2),0x1.e9575c2124912p-56},
    {(-0x1.1178e8227e47ap-2),(-0x1.b8ce2d07f1cb7p-56)},
    {(-0x1.07138604d5864p-2),0x1.24e912b16ec8bp-60},
    {(-0x1.f991c6cb3b37ap-3),(-0x1.ecca0cdf30143p-58)},
    {(-0x1.e530effe71013p-3),0x1.f7627ef82f3f0p-57},
    {(-0x1.d1037f2655e7bp-3),0x1.3f3adb7b71cbcp-58},
    {(-0x1.bd087383bd8aap-3),0x1.1165504ad749ep-59},
    {(-0x1.a93ed3c8ad9e5p-3),(-0x1.bcafa9de97202p-57)},
    {(-0x1.95a5adcf70182p-3),(-0x1.8a16283fdbd1cp-57)},
    {(-0x1.823c16551a3c0p-3),(-0x1.6dcd318f4187ep-57)},
    {(-0x1.6f0128b756ab9p-3),0x1.37967087859b9p-59},
    {(-0x1.5bf406b543db0p-3),0x1.1f5b44c0df7f7p-61},
    {(-0x1.4913d8333b563p-3),0x1.0d5604930f137p-58},
    {(-0x1.365fcb0159014p-3),(-0x1.bea08d2dca256p-57)},
    {(-0x1.23d712a49c201p-3),(-0x1.51c7e9efae297p-57)},
    {(-0x1.1178e8227e47ap-3),0x1.0e63a5f01c693p-58},
    {(-0x1.fe89139dbd565p-4),0x1.ac9f4215f9394p-58},
    {(-0x1.da7276384469ep-4),(-0x1.401fa71733017p-58)},
    {(-0x1.b6ac88dad5b1dp-4),0x1.002bf768e52d0p-58},
    {(-0x1.9335e5d594988p-4),0x1.478a85704ccb7p-58},
    {(-0x1.700d30aeac0e8p-4),(-0x1.a36a677b4c8b2p-59)},
    {(-0x1.4d3115d207eacp-4),(-0x1.da7d0b1e10b2fp-60)},
    {(-0x1.2aa04a44717a1p-4),(-0x1.aea2c72d05c08p-58)},
    {(-0x1.08598b59e3a06p-4),0x1.dd7009902bf32p-58},
    {(-0x1.ccb73cdddb2d0p-5),0x1.e48fb0500efd5p-59},
    {(-0x1.894aa149fb34bp-5),0x1.2ba0b44cfaee5p-59},
    {(-0x1.466aed42de3f9p-5),0x1.9badefe942718p-60},
    {(-0x1.0415d89e74440p-5),(-0x1.c05cf1d753621p-59)},
    {(-0x1.8492528c8cac5p-6),0x1.d192d0619fa68p-60},
    {(-0x1.0205658935837p-6),(-0x1.27c8e8416e717p-60)},
    {(-0x1.010157588de69p-7),(-0x1.46662d417cecep-62)},
    {0x0p+0,0x0p+0},
    {0x1.fe02a6b106799p-8,(-0x1.e44b7e3711e7fp-67)},
    {0x1.fc0a8b0fc03c4p-7,(-0x1.83092c5964281p-62)},
    {0x1.7b91b07d5b126p-6,(-0x1.6d80ab38e9430p-62)},
    {0x1.f829b0e7832f8p-6,0x1.33e3f04f1ef25p-60},
    {0x1.39e87b9febd68p-5,(-0x1.5bfa937f551b7p-59)},
    {0x1.77458f632dcffp-5,0x1.8d3ca87b92968p-63},
    {0x1.b42dd711971b9p-5,0x1.0a34531f67db5p-59},
    {0x1.f0a30c01162a8p-5,0x1.85f325c5bbacdp-59},
    {0x1.16536eea37ae3p-4,0x1.2189705cf74cap-58},
    {0x1.341d7961bd1d0p-4,(-0x1.3599f227becbbp-58)},
    {0x1.51b073f06183cp-4,(-0x1.5b61c65e5741ap-58)},
    {0x1.6f0d28ae56b4ep-4,(-0x1.20db323097324p-59)},
    {0x1.8c345d6319b23p-4,(-0x1.294d2f5668495p-58)},
    {0x1.a926d3a4ad562p-4,(-0x1.d7a16eab1e2adp-59)},
    {0x1.c5e548f5bc743p-4,0x1.2eb0bf7c0b0d9p-59},
    {0x1.e27076e2af2eap-4,(-0x1.61578001e015ap-60)},
    {0x1.fec9131dbeabcp-4,(-0x1.5746b9981b36cp-58)},
    {0x1.0d77e7cd08e5bp-3,0x1.9a5dc5e9030adp-57},
    {0x1.1b72ad52f67a2p-3,(-0x1.fbe7ee5c69946p-57)},
    {0x1.29552f81ff521p-3,0x1.301771c407dc0p-57},
    {0x1.371fc201e8f75p-3,0x1.e6cb62af18a02p-62},
    {0x1.44d2b6ccb7d1cp-3,0x1.7d3d950f87e23p-59},
    {0x1.526e5e3a1b438p-3,(-0x1.546ff8a470d3ap-57)},
    {0x1.5ff3070a793d6p-3,(-0x1.bc60efafc6f6cp-58)},
    {0x1.6d60fe719d21bp-3,0x1.d551d97132e87p-57},
    {0x1.7ab890210d907p-3,(-0x1.1072534a57e7dp-57)},
    {0x1.87fa06520c911p-3,(-0x1.9f7fdbfa08d9ap-57)},
    {0x1.9525a9cf456b6p-3,(-0x1.26fb3e2b1d1dap-57)},
    {0x1.a23bc1fe2b561p-3,0x1.24dc46c1ea664p-57},
    {0x1.af3c94e80bff3p-3,0x1.a3398064df33ep-57},
    {0x1.bc286742d8cd4p-3,0x1.cfce744870f57p-58},
    {0x1.c8ff7c79a9a20p-3,(-0x1.4f689f8434011p-57)},
    {0x1.d5c216b4fbb94p-3,(-0x1.a37794d03657dp-58)},
    {0x1.e27076e2af2e8p-3,(-0x1.61578001e015ep-59)},
    {0x1.ef0adcbdc5935p-3,0x1.e8637950dc20dp-57},
    {0x1.fb9186d5e3e29p-3,0x1.355519b0de535p-57},
    {0x1.0402594b4d041p-2,(-0x1.08ec217a5022dp-57)},
    {0x1.0a324e27390e2p-2,0x1.bdcfde8061c03p-56},
    {0x1.1058bf9ae4ad4p-2,0x1.3f415699663ecp-63},
    {0x1.1675cababa60fp-2,0x1.ce63eab883727p-61},
    {0x1.1c898c16999fbp-2,0x1.9f1a39d500e3cp-56},
    {0x1.22941fbcf7966p-2,(-0x1.dbd7ac258a2bdp-58)},
    {0x1.2895a13de86a4p-2,0x1.7ad24c13f040fp-56},
    {0x1.2e8e2bae11d31p-2,(-0x1.1e99b72bd7bf2p-57)},
    {0x1.347dd9a987d56p-2,(-0x1.16ea62c048cfbp-56)},
    {0x1.3a64c556945eap-2,0x1.cbcd735d03424p-60},
    {0x1.404308686a7e4p-2,(-0x1.f79f6c1059cdbp-57)},
    {0x1.4618bc21c5ec2p-2,(-0x1.7a42642661c62p-61)},
    {0x1.4be5f957778a1p-2,(-0x1.4b366b609027ap-58)},
    {0x1.51aad872df82ep-2,(-0x1.d8db0a7cc1543p-56)},
    {0x1.5767717455a6cp-2,(-0x1.fb2a49af933e8p-57)},
    {0x1.5d1bdbf5809cap-2,(-0x1.7dc9c7c23801fp-56)},
    {0x1.62c82f2b9c796p-2,(-0x1.090a0dd59fe35p-58)},
    {0x1.686c81e9b14adp-2,0x1.710af840538e3p-56},
    {0x1.6e08eaa2ba1e4p-2,(-0x1.bfb1b39ca3a0fp-56)},
    {0x1.739d7f6bbd007p-2,0x1.ce24c53fad3f0p-58},
    {0x1.792a55fdd47a1p-2,0x1.f057691fe9ed7p-56},
    {0x1.7eaf83b82afc2p-2,(-0x1.698b43096b576p-59)},
    {0x1.842d1da1e8b18p-2,0x1.54ec519784677p-56},
    {0x1.89a3386c1425bp-2,0x1.2d38c40881e0bp-57},
    {0x1.8f11e873662c8p-2,0x1.f85da755a61a3p-56},
    {0x1.947941c2116fbp-2,0x1.1266e8a3e8838p-57},
    {0x1.99d958117e08ap-2,(-0x1.315b444ee1f38p-56)},
    {0x1.9f323ecbf984dp-2,(-0x1.a92e513217f58p-59)}
};
// CONSTANTS END

#define WCR_INF WCR_FROM_BITS(0x7ff0000000000000LL)

WCR_HD double wcr_fabs(double x) {return WCR_FROM_BITS(WCR_BITS(x) & 0x7fffffffffffffffLL);}
// 2^e for -1022 <= e <= 1023.
WCR_HD double wcr_pow2(int e) {return WCR_FROM_BITS((long long)(e + 1023) << 52);}

// Exact transformations: a + b = s.h + s.l, a * b = p.h + p.l.
WCR_HD wcr_dd wcr_2sum(double a, double b) {
    double s = a + b;
    double bb = s - a;
    return wcr_dd{s, (a - (s - bb)) + (b - bb)};
}
WCR_HD wcr_dd wcr_fast2sum(double a, double b) {  // |a| >= |b| or a == 0
    double s = a + b;
    return wcr_dd{s, b - (s - a)};
}
WCR_HD wcr_dd wcr_2prod(double a, double b) {
    double p = a * b;
    return wcr_dd{p, WCR_FMA(a, b, -p)};
}
WCR_HD wcr_dd wcr_neg(wcr_dd a) {return wcr_dd{-a.h, -a.l};}
WCR_HD wcr_dd wcr_sgn(wcr_dd a, int n) {return (n & 1) ? wcr_neg(a) : a;}
WCR_HD wcr_dd wcr_add(wcr_dd a, wcr_dd b) {
    wcr_dd s = wcr_2sum(a.h, b.h);
    wcr_dd t = wcr_2sum(a.l, b.l);
    s = wcr_2sum(s.h, s.l + t.h);
    return wcr_2sum(s.h, s.l + t.l);
}
// a + b without the second error-free step: relative error near 2^-104 of
// the larger operand, for sums the caller knows do not cancel.
WCR_HD wcr_dd wcr_add_nc(wcr_dd a, wcr_dd b) {
    wcr_dd s = wcr_2sum(a.h, b.h);
    return wcr_fast2sum(s.h, s.l + (a.l + b.l));
}
WCR_HD wcr_dd wcr_mul(wcr_dd a, wcr_dd b) {
    wcr_dd p = wcr_2prod(a.h, b.h);
    return wcr_fast2sum(p.h, p.l + (a.h * b.l + a.l * b.h));
}
WCR_HD wcr_dd wcr_mul_d(wcr_dd a, double b) {
    wcr_dd p = wcr_2prod(a.h, b);
    return wcr_fast2sum(p.h, p.l + a.l * b);
}
WCR_HD wcr_dd wcr_div(wcr_dd a, wcr_dd b) {
    double q1 = WCR_DIV(a.h, b.h);
    wcr_dd r = wcr_add(a, wcr_neg(wcr_mul_d(b, q1)));
    double q2 = WCR_DIV(r.h, b.h);
    r = wcr_add(r, wcr_neg(wcr_mul_d(b, q2)));
    double q3 = WCR_DIV(r.h, b.h);
    return wcr_add(wcr_fast2sum(q1, q2), wcr_dd{q3, 0.0});
}

// Round a double-double whose relative error is below rel.
WCR_HD double wcr_round(wcr_dd v, double rel) {
    double e = rel * wcr_fabs(v.h);
    double up = v.h + (v.l + e);
    double dn = v.h + (v.l - e);
    if (up != dn) {WCR_AMBIGUOUS();}
    return v.h + v.l;
}

// exp(x.h + x.l) = y * 2^k with y in [0.70, 1.42], |x.h| <= 746.
WCR_HD wcr_dd wcr_exp_core(wcr_dd x, int* k) {
    double kd = WCR_RINT(x.h * WCR_INV_LN2);
    wcr_dd p = wcr_2prod(kd, WCR_LN2_1);
    wcr_dd r = wcr_add(wcr_2sum(x.h, -p.h), wcr_dd{-p.l, 0.0});
    r = wcr_add(r, wcr_neg(wcr_2prod(kd, WCR_LN2_2)));
    r = wcr_add(r, wcr_dd{-(kd * WCR_LN2_3), 0.0});
    r = wcr_add(r, wcr_dd{x.l, 0.0});
    // expm1(r / 256) by its Taylor series (|r / 256| < 0.00136, degree 11),
    // then eight squarings in the form (1 + m)^2 = 1 + m (2 + m).
    wcr_dd s = wcr_dd{r.h * 0x1p-8, r.l * 0x1p-8};
    wcr_dd q = wcr_inv_fact[11];
    for (int n = 10; n >= 1; --n) q = wcr_add(wcr_mul(q, s), wcr_inv_fact[n]);
    wcr_dd m = wcr_mul(q, s);
    for (int i = 0; i < 8; ++i) m = wcr_mul(m, wcr_add(wcr_dd{2.0, 0.0}, m));
    *k = (int)kd;
    return wcr_add(wcr_dd{1.0, 0.0}, m);
}

// Round y * 2^k once, including onto the subnormal grid.
WCR_HD double wcr_exp_finish(wcr_dd y, int k, double rel) {
    if (k > -1022) {
        double h = wcr_round(y, rel);
        return k > 1023 ? (h * wcr_pow2(1023)) * 2.0 : h * wcr_pow2(k);
    }
    // The first product is exact; the second rounds y.h onto the 2^-1074
    // grid.  d is the exact part of y.h that rounding removed; y.l decides
    // only an exact tie of y.h, which the hardware broke to even.
    double t = (y.h * wcr_pow2(k + 600)) * 0x1p-600;
    double d = y.h - (t * 0x1p+600) * wcr_pow2(-k - 600);
    double half = wcr_pow2(-1075 - k);
    if (d == half && y.l > 0.0) t = t + 0x1p-1074;
    else if (d == -half && y.l < 0.0) t = t - 0x1p-1074;
    return t;
}

// ---- Fast phase: table-driven, relative error near 2^-67, taken whenever
// its rounding test passes (all but roughly one call in a thousand). ----

// True when v (relative error below rel) rounds to the same binary64 at both
// ends of its error interval; *out is then the correctly rounded value.
WCR_HD bool wcr_round_ok(wcr_dd v, double rel, double* out) {
    double e = rel * wcr_fabs(v.h);
    double up = v.h + (v.l + e);
    double dn = v.h + (v.l - e);
    *out = v.h + v.l;
    return up == dn;
}

// exp(x.h + x.l) = y * 2^K, y = 2^(j/64) exp(r), |r| <= ln2/128.
WCR_HD wcr_dd wcr_exp_fast(wcr_dd x, int* K) {
    double kd = WCR_RINT(x.h * WCR_INV_LN2_64);
    int kj = (int)kd;
    wcr_dd p = wcr_2prod(kd, WCR_LN2_64_1);
    double t = x.h - p.h;  // exact: x.h and p.h are within a factor of two, or kd == 0
    wcr_dd r = wcr_2sum(t, (x.l - p.l) - kd * WCR_LN2_64_2);
    double rh = r.h;
    // exp(r) - 1 - r through r^7/7!, the tail in binary64 (|tail| < 2^-15).
    double q = rh * rh * (0.5 + rh * (0x1.5555555555555p-3 + rh * (0x1.5555555555555p-5
             + rh * (0x1.1111111111111p-7 + rh * (0x1.6c16c16c16c17p-10 + rh * 0x1.a01a01a01a01ap-13)))))
             + rh * r.l;
    wcr_dd em = wcr_2sum(rh, q);
    em.l = em.l + r.l;
    wcr_dd T = wcr_exp2_j64[kj & 63];
    wcr_dd a = wcr_2prod(T.h, em.h);
    wcr_dd y = wcr_fast2sum(T.h, a.h);
    y = wcr_fast2sum(y.h, y.l + (a.l + (T.h * em.l + (T.l * em.h + T.l))));
    *K = kj >> 6;
    return y;
}

// log(x) for finite x > 0: e ln2 - log(inv_j) + log1p(m inv_j - 1).
WCR_HD wcr_dd wcr_log_fast(double x) {
    int e = 0;
    if (x < 0x1p-1022) {x = x * 0x1p+54; e = -54;}
    long long b = WCR_BITS(x);
    e += (int)((b >> 52) & 0x7ff) - 1023;
    double m = WCR_FROM_BITS((b & 0x000fffffffffffffLL) | 0x3ff0000000000000LL);
    if (m >= 1.5) {m = m * 0.5; e += 1;}
    int j = (int)WCR_RINT(m * 128.0) - 96;
    wcr_dd p = wcr_2prod(m, wcr_log_inv[j]);
    wcr_dd r = wcr_2sum(p.h - 1.0, p.l);  // p.h - 1 is exact; r is m inv_j - 1 exactly
    double rh = r.h;
    wcr_dd r2 = wcr_2prod(rh, rh);
    // log1p(r) = r - r^2/2 + r^3 (1/3 - r/4 + ... - r^7/10), |r| < 0.0053.
    double tail = rh * r2.h * (0x1.5555555555555p-2 + rh * (-0.25 + rh * (0x1.999999999999ap-3
                + rh * (-0x1.5555555555555p-3 + rh * (0x1.2492492492492p-3 + rh * (-0.125
                + rh * (0x1.c71c71c71c71cp-4 - rh * 0x1.999999999999ap-4)))))));
    wcr_dd v = wcr_2sum(rh, -0.5 * r2.h);
    v = wcr_fast2sum(v.h, v.l + (r.l + ((-0.5 * r2.l - rh * r.l) + tail)));
    wcr_dd tab = wcr_log_tab[j];
    if (e != 0) {
        double ed = (double)e;
        wcr_dd E = wcr_2prod(ed, WCR_LN2_1);
        E.l = E.l + ed * WCR_LN2_2;
        tab = wcr_add_nc(E, tab);  // |e ln2| >= 0.69 > |tab|: no cancellation
    }
    return wcr_add_nc(tab, v);     // |v| < 0.0053 < |tab| / 1.4 unless tab == 0
}

WCR_HD double wcr_scale(double h, int K) {
    return K > 1023 ? (h * wcr_pow2(1023)) * 2.0 : h * wcr_pow2(K);
}

// x + y rounded to odd: an inexact sum keeps the neighbour whose last bit
// is one.
WCR_HD double wcr_add_odd(double x, double y) {
    wcr_dd s = wcr_2sum(x, y);
    long long b = WCR_BITS(s.h);
    if (s.l != 0.0 && (b & 1) == 0) b += ((s.l > 0.0) == (s.h > 0.0)) ? 1 : -1;
    return WCR_FROM_BITS(b);
}

// a + b + c correctly rounded (the rounding-to-odd three-term sum of Boldo
// and Melquiond, 2008).
WCR_HD double wcr_rn3(double a, double b, double c) {
    wcr_dd u = wcr_2sum(b, c);
    wcr_dd t = wcr_2sum(a, u.h);
    return t.h + wcr_add_odd(t.l, u.l);
}

// exp(x) for 2^-54 <= |x| < 2^-8 as 1 + x + B, B = sum_{n>=2} x^n/n! carried
// with an error relative to B, not to the result: the arguments where
// 1 + x falls on or next to a rounding midpoint (x = k 2^-53, ...) are then
// decided by B itself.
WCR_HD double wcr_exp_near0(double x) {
    wcr_dd p = wcr_inv_fact[14];
    for (int n = 13; n >= 2; --n) p = wcr_add(wcr_mul_d(p, x), wcr_inv_fact[n]);
    wcr_dd B = wcr_mul(wcr_2prod(x, x), p);
    wcr_dd u = wcr_2sum(x, B.h);
    double w = u.l + B.l;
    double e = wcr_fabs(B.h) * 0x1p-96 + wcr_fabs(w) * 0x1p-52;
    double z = wcr_rn3(1.0, u.h, w);
    if (z != wcr_rn3(1.0, u.h, w + e) || z != wcr_rn3(1.0, u.h, w - e)) {WCR_AMBIGUOUS();}
    return z;
}

// log(x) for |x - 1| < 2^-8 as f + A, f = x - 1 (exact), A = sum_{n>=2}
// (-1)^(n+1) f^n/n, A's error relative to A.
WCR_HD double wcr_log_near1(double x) {
    double f = x - 1.0;
    wcr_dd p = wcr_sgn(wcr_inv_int[16], 17);
    for (int n = 15; n >= 2; --n) p = wcr_add(wcr_mul_d(p, f), wcr_sgn(wcr_inv_int[n], n + 1));
    wcr_dd A = wcr_mul(wcr_2prod(f, f), p);
    double e = wcr_fabs(A.h) * 0x1p-96;
    double z = wcr_rn3(f, A.h, A.l);
    if (z != wcr_rn3(f, A.h, A.l + e) || z != wcr_rn3(f, A.h, A.l - e)) {WCR_AMBIGUOUS();}
    return z;
}

WCR_HD double wcr_exp(double x) {
    if (x != x) return x + x;
    if (x > 710.0) return WCR_INF;
    if (x < -746.0) return 0.0;
    if (wcr_fabs(x) < 0x1p-54) return 1.0 + x;
    int k;
    double out;
    wcr_dd y = wcr_exp_fast(wcr_dd{x, 0.0}, &k);
    if (k > -1022 && wcr_round_ok(y, 0x1p-63, &out)) return wcr_scale(out, k);
    if (wcr_fabs(x) < 0x1p-8) return wcr_exp_near0(x);
    y = wcr_exp_core(wcr_dd{x, 0.0}, &k);
    return wcr_exp_finish(y, k, 0x1p-92);
}

// log(x) for finite x > 0, as a double-double.
WCR_HD wcr_dd wcr_log_dd(double x) {
    int e = 0;
    if (x < 0x1p-1022) {x = x * 0x1p+54; e = -54;}
    long long b = WCR_BITS(x);
    e += (int)((b >> 52) & 0x7ff) - 1023;
    double m = WCR_FROM_BITS((b & 0x000fffffffffffffLL) | 0x3ff0000000000000LL);
    if (m > WCR_SQRT2) {m = m * 0.5; e += 1;}
    // log(m) = 2 atanh(u), u = (m - 1) / (m + 1), |u| < 0.1716; m - 1 is exact.
    double f = m - 1.0;
    wcr_dd u = wcr_div(wcr_dd{f, 0.0}, wcr_2sum(2.0, f));
    wcr_dd u2 = wcr_mul(u, u);
    wcr_dd s = wcr_inv_odd[20];
    for (int n = 19; n >= 0; --n) s = wcr_add(wcr_mul(s, u2), wcr_inv_odd[n]);
    wcr_dd lm = wcr_mul(wcr_dd{2.0 * u.h, 2.0 * u.l}, s);
    if (e == 0) return lm;
    double ed = (double)e;
    wcr_dd r = wcr_add(wcr_2prod(ed, WCR_LN2_1), wcr_2prod(ed, WCR_LN2_2));
    r = wcr_add(r, wcr_dd{ed * WCR_LN2_3, 0.0});
    return wcr_add(r, lm);
}

WCR_HD double wcr_log(double x) {
    if (x != x) return x + x;
    if (x == 1.0) return 0.0;
    if (x < 0.0) return WCR_DIV(0.0, 0.0);
    if (x == 0.0) return -WCR_INF;
    if (x == WCR_INF) return x;
    double out;
    if (wcr_round_ok(wcr_log_fast(x), 0x1p-63, &out)) return out;
    if (wcr_fabs(x - 1.0) < 0x1p-8) return wcr_log_near1(x);
    return wcr_round(wcr_log_dd(x), 0x1p-92);
}

WCR_HD double wcr_log10(double x) {
    if (x != x) return x + x;
    if (x == 1.0) return 0.0;
    if (x < 0.0) return WCR_DIV(0.0, 0.0);
    if (x == 0.0) return -WCR_INF;
    if (x == WCR_INF) return x;
    double out;
    if (wcr_round_ok(wcr_mul(wcr_log_fast(x), wcr_dd{WCR_INV_LN10_1, WCR_INV_LN10_2}), 0x1p-62, &out)) return out;
    return wcr_round(wcr_mul(wcr_log_dd(x), wcr_dd{WCR_INV_LN10_1, WCR_INV_LN10_2}), 0x1p-90);
}

WCR_HD double wcr_sin(double x) {
    if (x == 0.0 || x != x) return x + x;
    if (wcr_fabs(x) == WCR_INF) return x - x;
    if (wcr_fabs(x) >= 0x1p+20) return WCR_PLATFORM_SIN(x);
    double nd = WCR_RINT(x * WCR_INV_PIO2);
    wcr_dd r = wcr_dd{x, 0.0};
    double rel = 0x1p-90;
    if (nd != 0.0) {
        wcr_dd p = wcr_2prod(nd, WCR_PIO2_1);
        r = wcr_add(wcr_2sum(x, -p.h), wcr_dd{-p.l, 0.0});
        r = wcr_add(r, wcr_neg(wcr_2prod(nd, WCR_PIO2_2)));
        r = wcr_add(r, wcr_dd{-(nd * WCR_PIO2_3), 0.0});
        // pi/2 is carried to 159 bits: widen the bound by the reduction's
        // absolute error relative to the reduced argument.
        rel = rel + WCR_DIV(wcr_fabs(nd) * 0x1p-158, wcr_fabs(r.h));
    }
    int q = ((int)nd) & 3;
    wcr_dd r2 = wcr_mul(r, r);
    wcr_dd v;
    if (q & 1) {  // cos(r) = sum (-1)^n r^2n / (2n)!
        v = wcr_sgn(wcr_inv_fact[28], 14);
        for (int n = 13; n >= 0; --n) v = wcr_add(wcr_mul(v, r2), wcr_sgn(wcr_inv_fact[2 * n], n));
    } else {      // sin(r) = r sum (-1)^n r^2n / (2n+1)!
        v = wcr_sgn(wcr_inv_fact[29], 14);
        for (int n = 13; n >= 0; --n) v = wcr_add(wcr_mul(v, r2), wcr_sgn(wcr_inv_fact[2 * n + 1], n));
        v = wcr_mul(v, r);
    }
    if (q & 2) v = wcr_neg(v);
    return wcr_round(v, rel);
}

WCR_HD double wcr_atan(double x) {
    if (x == 0.0 || x != x) return x + x;
    double ax = wcr_fabs(x);
    if (ax == WCR_INF) return x > 0.0 ? WCR_PIO2_1 : -WCR_PIO2_1;
    bool inv = ax > 1.0;
    wcr_dd z = inv ? wcr_div(wcr_dd{1.0, 0.0}, wcr_dd{ax, 0.0}) : wcr_dd{ax, 0.0};
    // atan(z) = atan(j/16) + atan(t), t = (z - j/16) / (1 + z j/16), |t| <= 1/32.
    int j = (int)WCR_RINT(z.h * 16.0);
    double c = j * 0.0625;
    wcr_dd t = z;
    if (j) t = wcr_div(wcr_add(z, wcr_dd{-c, 0.0}), wcr_add(wcr_dd{1.0, 0.0}, wcr_mul_d(z, c)));
    // Fast phase: atan(t) - t through t^13/13 in binary64 (|tail| < 2^-10 |t|).
    double th = t.h, q = th * th;
    double tail = th * q * (-0x1.5555555555555p-2 + q * (0x1.999999999999ap-3 + q * (-0x1.2492492492492p-3
                + q * (0x1.c71c71c71c71cp-4 + q * (-0x1.745d1745d1746p-4 + q * 0x1.3b13b13b13b14p-4)))));
    wcr_dd v = wcr_2sum(th, tail);
    v.l = v.l + t.l;
    if (j) v = wcr_add(wcr_atan_j16[j], v);
    if (inv) v = wcr_add(wcr_dd{WCR_PIO2_1, WCR_PIO2_2}, wcr_neg(v));
    if (x < 0.0) v = wcr_neg(v);
    double out;
    if (wcr_round_ok(v, 0x1p-62, &out)) return out;
    // Accurate phase: the whole series in double-double (t^2 <= 2^-10).
    wcr_dd t2 = wcr_mul(t, t);
    wcr_dd s = wcr_sgn(wcr_inv_odd[11], 11);
    for (int n = 10; n >= 0; --n) s = wcr_add(wcr_mul(s, t2), wcr_sgn(wcr_inv_odd[n], n));
    v = wcr_mul(s, t);
    if (j) v = wcr_add(wcr_atan_j16[j], v);
    if (inv) v = wcr_add(wcr_dd{WCR_PIO2_1, WCR_PIO2_2}, wcr_neg(v));
    if (x < 0.0) v = wcr_neg(v);
    return wcr_round(v, 0x1p-90);
}

WCR_HD bool wcr_is_int(double y) {return WCR_RINT(y) == y;}
WCR_HD bool wcr_is_odd_int(double y) {
    if (!(wcr_fabs(y) < 0x1p+53)) return false;
    return WCR_RINT(y) == y && WCR_RINT(y * 0.5) != y * 0.5;
}

WCR_HD double wcr_pow(double x, double y) {
    if (y == 0.0 || x == 1.0) return 1.0;
    // x^2 is one correctly rounded product; this also settles its exact
    // midpoints (odd 27-bit mantissas), which no error bound can.
    if (y == 2.0) return x * x;
    if (x != x || y != y) return x + y;
    double ax = wcr_fabs(x);
    if (wcr_fabs(y) == WCR_INF) {
        if (ax == 1.0) return 1.0;
        return ((ax > 1.0) == (y > 0.0)) ? WCR_INF : 0.0;
    }
    bool odd = wcr_is_odd_int(y);
    if (x == 0.0) {
        if (y > 0.0) return odd ? x : 0.0;
        return odd ? WCR_DIV(1.0, x) : WCR_INF;
    }
    if (ax == WCR_INF) {
        if (x > 0.0) return y > 0.0 ? WCR_INF : 0.0;
        if (y > 0.0) return odd ? -WCR_INF : WCR_INF;
        return odd ? -0.0 : 0.0;
    }
    double sign = 1.0;
    if (x < 0.0) {
        if (!wcr_is_int(y)) return WCR_DIV(0.0, 0.0);
        if (odd) sign = -1.0;
    }
    wcr_dd lx = wcr_log_fast(ax);
    // Decide overflow and underflow before the exact product, which a huge
    // y would turn into inf - inf.
    double t0 = lx.h * y;
    if (t0 > 710.0) return sign * WCR_INF;
    if (t0 < -746.0) return sign * 0.0;
    wcr_dd t = wcr_mul_d(lx, y);
    int k;
    double out;
    // The fast logarithm's relative error (below 2^-66) becomes an absolute
    // error of t, which is the relative error it brings to exp(t).
    wcr_dd f = wcr_exp_fast(t, &k);
    if (k > -1022 && wcr_round_ok(f, 0x1p-63 + wcr_fabs(t.h) * 0x1p-64, &out)) return sign * wcr_scale(out, k);
    t = wcr_mul_d(wcr_log_dd(ax), y);
    wcr_dd e = wcr_exp_core(t, &k);
    return sign * wcr_exp_finish(e, k, 0x1p-88);
}
#endif
