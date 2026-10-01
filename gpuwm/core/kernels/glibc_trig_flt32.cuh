// ======================================================================
// THIRD-PARTY NOTICE. Parts of this file are hand transcriptions of
// third-party work. ArWen distributes the file under the Apache License
// 2.0; the notices below belong to the transcribed parts and are kept
// here because their own licences require it. Full texts are in the
// repository NOTICE, licenses/, and beside the code in
// gpuwm/core/kernels/LICENSE-third-party.txt.
//
// Arm optimized-routines: glibc_sinf / glibc_cosf and their data tables.
// github.com/ARM-software/optimized-routines: math/sinf.c, math/cosf.c,
// math/sincosf.h, math/sincosf_data.c, published in 2018 and imported
// into glibc in August 2018; transcribed here from glibc 2.43
// sysdeps/ieee754/flt-32/s_sinf.c, s_cosf.c, s_sincosf.h,
// sincosf_poly.h and s_sincosf_data.c (x86 FMA IFUNC variant).
//
// Copyright (c) 2018-2024, Arm Limited.
// Copyright (c) 2018-2019, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Taken under the MIT branch of Arm's grant, as glibc_flt32.cuh is.
// The algorithm, constants and tables reproduce Arm's work; glibc's
// integration macros, aliases and IFUNC wrapper are not transcribed.
// Arm 5e8389113b47 (2018), MIT relicensing 11253b0b9d6b,
// current fe09d4e7ed62. The full permission notice is reproduced in
// licenses/LICENSE-Arm-optimized-routines-MIT.txt and the files above.
//
// CORE-MATH MIT transcriptions from sysdeps/ieee754/flt-32/:
// e_asinf.c and s_asincosf_data.c (src/binary32/asin/asinf.c, bc385c2);
// e_acosf.c (src/binary32/acos/acosf.c, 56dd347);
// s_atanf.c (src/binary32/atan/atanf.c, a8066a5);
// s_tanf.c (src/binary32/tan/tanf.c, 59d21d7).
// Copyright (c) 2022-2024 Alexei Sibidanov.
// Copyright (c) 2023-2024 Alexei Sibidanov.
// Full text: licenses/LICENSE-CORE-MATH-MIT.txt.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
// No SunPro FDLIBM code is used.
// ======================================================================
// Target: Ubuntu GLIBC 2.43-2ubuntu2.4, x86-64 AVX2+FMA, FE_TONEAREST.
// sinf/cosf use __sinf_fma/__cosf_fma. The other four scalar functions
// have no CPU IFUNC. Floating exceptions and errno are not device outputs.
// Arithmetic is pinned to the selected libm machine instructions, not CUDA
// trig builtins. Use -std=c++17. No FADD/FMUL macros are defined here.
#ifndef GPUWM_GLIBC_TRIG_FLT32_CUH
#define GPUWM_GLIBC_TRIG_FLT32_CUH
typedef unsigned int gt_u32;
typedef unsigned long long gt_u64;
typedef long long gt_i64;

__device__ inline float gt_d2f(double x) { return __double2float_rn(x); }
__device__ inline double gt_identity(double x) { return x; }
__device__ inline gt_u64 gt_as_u64(double x) { return (gt_u64)__double_as_longlong(x); }
__device__ inline double gt_abs(double x) {
  return __longlong_as_double((long long)(gt_as_u64(x) & 0x7fffffffffffffffULL));
}
__device__ inline float gt_absf(float x) { return __uint_as_float(__float_as_uint(x) & 0x7fffffffU); }
__device__ inline double gt_sign(double x, double y) {
  return __longlong_as_double((long long)((gt_as_u64(x) & 0x7fffffffffffffffULL) | (gt_as_u64(y) & 0x8000000000000000ULL)));
}
__device__ inline float gt_signf(float x, float y) {
  return __uint_as_float((__float_as_uint(x)&0x7fffffffU) | (__float_as_uint(y)&0x80000000U));
}
__device__ inline float gt_invalid(float x) {
  gt_u32 t=__float_as_uint(x);
  return __uint_as_float((t & 0x7fffffffU)>0x7f800000U ? t|0x00400000U : 0xffc00000U);
}
__device__ inline double gt_roundeven(double x) {
  // This helper only receives |x| < 2^28 * 2/pi. Conversion pins RN.
  return (double)__double2ll_rn(x);
}
__device__ inline float gt_tiny_fmaf(float a,float b,float x) {
  // Only called by the three tiny-input paths below, where the fused
  // correction is less than half an ULP in RN. Preserve exponent-zero
  // inputs by bits because CUDA FTZ flushes even an explicit FMA's result.
  // All normal inputs retain glibc's explicit fused operation.
  if ((__float_as_uint(x)&0x7f800000U)==0) return x;
  return __fmaf_rn(a,b,x);
}

// s_sincosf_data.c and sincosf_poly.h.
struct gt_sincos_t { double sign[4], hpi_inv, hpi, c0,c1,c2,c3,c4,s1,s2,s3; };
__device__ const gt_sincos_t gt_sincos_table[2] = {
 {{1,-1,-1,1},0x1.45F306DC9C883p+23,0x1.921FB54442D18p0,
  0x1p0,-0x1.ffffffd0c621cp-2,0x1.55553e1068f19p-5,-0x1.6c087e89a359dp-10,0x1.99343027bf8c3p-16,
  -0x1.555545995a603p-3,0x1.1107605230bc4p-7,-0x1.994eb3774cf24p-13},
 {{1,-1,-1,1},0x1.45F306DC9C883p+23,0x1.921FB54442D18p0,
  -0x1p0,0x1.ffffffd0c621cp-2,-0x1.55553e1068f19p-5,0x1.6c087e89a359dp-10,-0x1.99343027bf8c3p-16,
  -0x1.555545995a603p-3,0x1.1107605230bc4p-7,-0x1.994eb3774cf24p-13}
};
__device__ const gt_u32 gt_inv_pio4[24] = {
 0xa2,0xa2f9,0xa2f983,0xa2f9836e,0xf9836e4e,0x836e4e44,0x6e4e4415,0x4e441529,
 0x441529fc,0x1529fc27,0x29fc2757,0xfc2757d1,0x2757d1f5,0x57d1f534,0xd1f534dd,0xf534ddc0,
 0x34ddc0db,0xddc0db62,0xc0db6295,0xdb629599,0x6295993c,0x95993c43,0x993c4390,0x3c439041
};
__device__ inline float gt_sinf_poly(double x,double x2,const gt_sincos_t* p,int n) {
 if (!(n&1)) {
  double x3=__dmul_rn(x,x2);
  double s1=__fma_rn(x2,p->s3,p->s2);
  double x7=__dmul_rn(x3,x2);
  double s=__fma_rn(x3,p->s1,x);
  return gt_d2f(__fma_rn(x7,s1,s));
 }
 double x4=__dmul_rn(x2,x2);
 double c2=__fma_rn(x2,p->c4,p->c3);
 double c1=__fma_rn(x2,p->c1,p->c0);
 double x6=__dmul_rn(x4,x2);
 double c=__fma_rn(x4,p->c2,c1);
 return gt_d2f(__fma_rn(x6,c2,c));
}
__device__ inline double gt_reduce_large(gt_u32 xi,int* np) {
 const gt_u32* arr=&gt_inv_pio4[(xi>>26)&15];
 int shift=(xi>>23)&7;
 xi=((xi&0xffffffU)|0x800000U)<<shift;
 gt_u64 r0=(gt_u32)(xi*arr[0]);
 gt_u64 r1=(gt_u64)xi*arr[4], r2=(gt_u64)xi*arr[8];
 r0=((r2>>32)|(r0<<32))+r1;
 gt_u64 n=(r0+(1ULL<<61))>>62;
 r0-=n<<62;
 *np=(int)n;
 return __dmul_rn((double)(gt_i64)r0,0x1.921FB54442D18p-62);
}
__device__ inline float gt_sincos(float y,int cosine) {
 gt_u32 yi=__float_as_uint(y), top=(yi>>20)&0x7ffU;
 if (top>=0x7f8U) return gt_invalid(y);
 if (top<0x398U) return cosine ? 1.0f : y;
 double x=(double)y;
 const gt_sincos_t* p=&gt_sincos_table[0];
 int n=0, sign=0;
 if (top>=0x3f4U) {
  if (top<0x42fU) {
   double r=__dmul_rn(x,p->hpi_inv);
   n=((int)r+0x800000)>>24;
   x=__fma_rn(-(double)n,p->hpi,x);
  } else {
   x=gt_reduce_large(yi,&n);
   sign=(int)(yi>>31);
  }
  int q=(n+sign)&3;
  if (q&2) p=&gt_sincos_table[1];
  double x2=__dmul_rn(x,x);
  return gt_sinf_poly(__dmul_rn(x,gt_sincos_table[0].sign[q]),x2,p,n^cosine);
 }
 return gt_sinf_poly(x,__dmul_rn(x,x),p,cosine);
}
__device__ float glibc_sinf(float x) { return gt_sincos(x,0); }
__device__ float glibc_cosf(float x) { return gt_sincos(x,1); }

__device__ const double gt_asincos_c0[12] = {
 0x1.555555555529cp-3,0x1.333333337e0ddp-4,0x1.6db6db3b4465ep-5,
 0x1.f1c72e13ac306p-6,0x1.6e89cebe06bc4p-6,0x1.1c6dcf5289094p-6,
 0x1.c6dbbcc7c6315p-7,0x1.8f8dc2615e996p-7,0x1.a5833b7bf15e8p-8,
 0x1.43f44ace1665cp-6,-0x1.0fb17df881c73p-6,0x1.07520c026b2d6p-5
};
__device__ const double gt_asincos_c1[12] = {
 0x1.6a09e667f3bcbp+0,0x1.e2b7dddff2db9p-4,0x1.b27247ab42dbcp-6,
 0x1.02995cc4e0744p-7,0x1.5ffb0276ec8eap-9,0x1.033885a928decp-10,
 0x1.911f2be23f8c7p-12,0x1.4c3c55d2437fdp-13,0x1.af477e1d7b461p-15,
 0x1.abd6bdff67dcbp-15,-0x1.1717e86d0fa28p-16,0x1.6ff526de46023p-16
};
// s_tanf.c rbig: limb form of glibc's unsigned 128-bit multiplies.
__device__ const gt_u64 gt_tan_ipi[4] = {
 0xfe5163abdebbc562ULL,0xdb6295993c439041ULL,0xfc2757d1f534ddc0ULL,0xa2f9836e4e441529ULL
};
__device__ inline double gt_rbig(gt_u32 u,int* q) {
 gt_u64 m=(u&0x7fffffU)|0x800000U;
 gt_u64 h=__umul64hi(m,gt_tan_ipi[0]);
 gt_u64 p1l=m*gt_tan_ipi[1]+h;
 h=__umul64hi(m,gt_tan_ipi[1])+(p1l<h);
 gt_u64 p2l=m*gt_tan_ipi[2]+h;
 h=__umul64hi(m,gt_tan_ipi[2])+(p2l<h);
 gt_u64 p3l=m*gt_tan_ipi[3]+h;
 gt_u64 p3h=__umul64hi(m,gt_tan_ipi[3])+(p3l<h);
 int s=(int)((u>>23)&255)-127-23;
 gt_u64 a; gt_u32 i;
 if(s<64) { i=(gt_u32)((p3h<<s)|(p3l>>(64-s))); a=(p3l<<s)|(p2l>>(64-s)); }
 else if(s==64) { i=(gt_u32)p3l; a=p2l; }
 else { i=(gt_u32)((p3l<<(s-64))|(p2l>>(128-s))); a=(p2l<<(s-64))|(p1l>>(128-s)); }
 gt_u32 sign=0U-(u>>31);
 i-= (gt_u32)((gt_i64)a>>63);
 i=(i^sign)-sign;
 *q=(int)i;
 gt_i64 az=(gt_i64)(a^(gt_u64)(gt_i64)(int)sign);
 return __dmul_rn((double)az,0x1p-64);
}

// glibc-2.43 sysdeps/ieee754/flt-32/e_asinf.c
__device__ const double gt_asinf_b[] = {0x1.0000000000005p+0, 0x1.55557aeca105dp-3, 0x1.3314ec3db7d12p-4, 0x1.775738a5a6f92p-5, 0x1.5d5f7ce1c8538p-8, 0x1.605c6d58740fp-2, -0x1.5728b732d73c6p+1, 0x1.f152170f151ebp+3, -0x1.f962ea3ca992ep+5, 0x1.71971e17375ap+7, -0x1.860512b4ba23p+8, 0x1.26a3b8d4bdb14p+9, -0x1.36f2ea5698b51p+9, 0x1.b3d722aebfa2ep+8, -0x1.6cf89703b1289p+7, 0x1.1518af6a65e2dp+5};

__device__ static float gt_asinf_as_special(float x)
{
  gt_u32 ax = __float_as_uint(x) << 1;
  if (ax > (0xffu << 24))
    return gt_invalid(x); // Preserve the input NaN sign and payload.
  return gt_invalid(0.0);
}


__device__ static double gt_asinf_poly12(double z, const double *c)
{
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double c0 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
  double c2 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
  double c4 = __dadd_rn(c[4], __dmul_rn(z, c[5]));
  double c6 = __dadd_rn(c[6], __dmul_rn(z, c[7]));
  double c8 = __dadd_rn(c[8], __dmul_rn(z, c[9]));
  double c10 = __dadd_rn(c[10], __dmul_rn(z, c[11]));
  c0 = __dadd_rn(c0, __dmul_rn(c2, z2));
  c4 = __dadd_rn(c4, __dmul_rn(c6, z2));
  c8 = __dadd_rn(c8, __dmul_rn(z2, c10));
  c0 = __dadd_rn(c0, __dmul_rn(z4, __dadd_rn(c4, __dmul_rn(z4, c8))));
  return c0;
}


__device__ float glibc_asinf(float x)
{
  const double pi2 = 0x1.921fb54442d18p+0;
  double xs = x;
  double r;
  gt_u32 ax = __float_as_uint(x) << 1;
  if (ax > (0x7f << 24))
    return gt_asinf_as_special(x);
  if (ax < 0x7ec29000u)
  {
    if (ax < (115 << 24))
      return gt_tiny_fmaf(x, 0x1p-25f, x);
    /* table gt_asinf_b */;
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double z4 = __dmul_rn(z2, z2);
    double z8 = __dmul_rn(z4, z4);
    double z16 = __dmul_rn(z8, z8);
    r = __dmul_rn(z, __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(gt_asinf_b[0], __dmul_rn(z2, gt_asinf_b[1])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[2], __dmul_rn(z2, gt_asinf_b[3])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_asinf_b[4], __dmul_rn(z2, gt_asinf_b[5])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[6], __dmul_rn(z2, gt_asinf_b[7])))))), __dmul_rn(z16, __dadd_rn(__dadd_rn(__dadd_rn(gt_asinf_b[8], __dmul_rn(z2, gt_asinf_b[9])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[10], __dmul_rn(z2, gt_asinf_b[11])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_asinf_b[12], __dmul_rn(z2, gt_asinf_b[13])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[14], __dmul_rn(z2, gt_asinf_b[15])))))))));
    float ub = gt_d2f(r);
    float lb = gt_d2f(__dsub_rn(r, __dmul_rn(z, 0x1.efa8ebp-31)));
    if (ub == lb)
      return ub;
  }
  if (ax < (0x7eu << 24))
  {
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double c0 = gt_asinf_poly12(z2, gt_asincos_c0);
    r = __dadd_rn(z, __dmul_rn(__dmul_rn(z, z2), c0));
  }
  else
  {
    if (ax == 0x7e55688au)
      return __fadd_rn(gt_signf(0x1.75b8a2p-1f, x), gt_signf(0x1p-26f, x));
    if (ax == 0x7e107434u)
      return __fadd_rn(gt_signf(0x1.1f4b64p-1f, x), gt_signf(0x1p-26f, x));
    double bx = gt_abs(xs);
    double z = __dsub_rn(1.0, bx);
    double s = __dsqrt_rn(z);
    r = __dsub_rn(pi2, __dmul_rn(s, gt_asinf_poly12(z, gt_asincos_c1)));
    r = gt_sign(r, xs);
  }
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/e_acosf.c
__device__ const double gt_acosf_o[] = {0, 0x1.921fb54442d18p+1};
__device__ const double gt_acosf_b[] = {0x1.fffffffd9ccb8p-1, 0x1.5555c94838007p-3, 0x1.32ded4b7c20fap-4, 0x1.8566df703309ep-5, -0x1.980c959bec9a3p-6, 0x1.56fbb04998344p-1, -0x1.403d8e4c49f52p+2, 0x1.b06c3e9f311eap+4, -0x1.9ea97c4e2c21fp+6, 0x1.200b8261cc61bp+8, -0x1.2274c2799a5c7p+9, 0x1.a558a59cc19d3p+9, -0x1.aca4b6a529ffp+9, 0x1.228744703f813p+9, -0x1.d7dbb0b322228p+7, 0x1.5c2018c0c0105p+5};

__device__ static float gt_acosf_as_special(float x)
{
  const float pih = gt_d2f(0x1.921fb6p+1);
  const float pil = -0x1p-24f;
  gt_u32 t = __float_as_uint(x);
  if (t == (0x7fu << 23))
    return 0.0f;
  if (t == (0x17fu << 23))
    return __fadd_rn(pih, pil);
  gt_u32 ax = t << 1;
  if (ax > (0xffu << 24))
    return gt_invalid(x); // Preserve the input NaN sign and payload.
  return gt_invalid(0.0);
}


__device__ inline static double gt_acosf_poly12(double z, const double *c)
{
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double c0 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
  double c2 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
  double c4 = __dadd_rn(c[4], __dmul_rn(z, c[5]));
  double c6 = __dadd_rn(c[6], __dmul_rn(z, c[7]));
  double c8 = __dadd_rn(c[8], __dmul_rn(z, c[9]));
  double c10 = __dadd_rn(c[10], __dmul_rn(z, c[11]));
  c0 = __dadd_rn(c0, __dmul_rn(c2, z2));
  c4 = __dadd_rn(c4, __dmul_rn(c6, z2));
  c8 = __dadd_rn(c8, __dmul_rn(z2, c10));
  c0 = __dadd_rn(c0, __dmul_rn(z4, __dadd_rn(c4, __dmul_rn(z4, c8))));
  return c0;
}


__device__ float glibc_acosf(float x)
{
  double pi2 = 0x1.921fb54442d18p+0;
  /* table gt_acosf_o */;
  double xs = x;
  double r;
  gt_u32 t = __float_as_uint(x);
  gt_u32 ax = t << 1;
  if (ax >= (0x7f << 24))
    return gt_acosf_as_special(x);
  if (ax < 0x7ec2a1dcu)
  {
    /* table gt_acosf_b */;
    if (ax <= 0x40000000u)
      return gt_d2f(gt_identity(pi2));
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double z4 = __dmul_rn(z2, z2);
    double z8 = __dmul_rn(z4, z4);
    double z16 = __dmul_rn(z8, z8);
    r = __dmul_rn(z, __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(gt_acosf_b[0], __dmul_rn(z2, gt_acosf_b[1])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[2], __dmul_rn(z2, gt_acosf_b[3])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_acosf_b[4], __dmul_rn(z2, gt_acosf_b[5])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[6], __dmul_rn(z2, gt_acosf_b[7])))))), __dmul_rn(z16, __dadd_rn(__dadd_rn(__dadd_rn(gt_acosf_b[8], __dmul_rn(z2, gt_acosf_b[9])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[10], __dmul_rn(z2, gt_acosf_b[11])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_acosf_b[12], __dmul_rn(z2, gt_acosf_b[13])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[14], __dmul_rn(z2, gt_acosf_b[15])))))))));
    float ub = gt_d2f(__dsub_rn(0x1.921fb54574191p+0, r));
    float lb = gt_d2f(__dsub_rn(0x1.921fb543118ap+0, r));
    if (ub == lb)
      return ub;
  }
  if (ax < (0x7eu << 24))
  {
    if (t == 0x328885a3u)
      return __fadd_rn(0x1.921fb6p+0f, 0x1p-25f);
    if (t == 0x39826222u)
      return __fadd_rn(0x1.920f6ap+0f, 0x1p-25f);
    double x2 = __dmul_rn(xs, xs);
    r = __dsub_rn(__dsub_rn(pi2, xs), __dmul_rn(__dmul_rn(xs, x2), gt_acosf_poly12(x2, gt_asincos_c0)));
  }
  else
  {
    double bx = gt_abs(xs);
    double z = __dsub_rn(1.0, bx);
    double s = gt_sign(__dsqrt_rn(z), xs);
    r = __dadd_rn(gt_acosf_o[t >> 31], __dmul_rn(s, gt_acosf_poly12(z, gt_asincos_c1)));
  }
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/s_atanf.c
__device__ const double gt_atanf_cn[] = {0x1.51eccde075d67p-2, 0x1.a76bb5637f2f2p-1, 0x1.81e0eed20de88p-1, 0x1.376c8ca67d11dp-2, 0x1.aec7b69202ac6p-5, 0x1.9561899acc73ep-9, 0x1.bf9fa5b67e6p-16};
__device__ const double gt_atanf_cd[] = {0x1.51eccde075d66p-2, 0x1.dfbdd7b392d28p-1, 0x1p+0, 0x1.fd22bf0e89b54p-2, 0x1.d91ff8b576282p-4, 0x1.653ea99fc9bbp-7, 0x1.1e7fcc202340ap-12};

__device__ float glibc_atanf(float x)
{
  const double pi2 = 0x1.921fb54442d18p+0;
  gt_u32 t = __float_as_uint(x);
  int e = (t >> 23) & 0xff;
  bool gt = e >= 127;
  gt_u32 ta = t & 0x7fffffff;
  if (ta >= 0x4c700518u)
  {
    if (ta > 0x7f800000u)
      return gt_invalid(x); // Preserve the input NaN sign and payload.
    return gt_d2f(gt_sign(pi2, (double) x));
  }
  if (e < (127 - 13))
  {
    if (e < (127 - 25))
    {
      if (!(t << 1))
        return x;
      return gt_tiny_fmaf(-x, gt_absf(x), x);
    }
    return __fmaf_rn(__fmul_rn(-0x1.555556p-2f, x), __fmul_rn(x, x), x);
  }
  double z = x;
  if (gt)
    z = __ddiv_rn(1, z);
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double z8 = __dmul_rn(z4, z4);
  /* table gt_atanf_cn */;
  /* table gt_atanf_cd */;
  double cn0 = __dadd_rn(gt_atanf_cn[0], __dmul_rn(z2, gt_atanf_cn[1]));
  double cn2 = __dadd_rn(gt_atanf_cn[2], __dmul_rn(z2, gt_atanf_cn[3]));
  double cn4 = __dadd_rn(gt_atanf_cn[4], __dmul_rn(z2, gt_atanf_cn[5]));
  double cn6 = gt_atanf_cn[6];
  cn0 = __dadd_rn(cn0, __dmul_rn(z4, cn2));
  cn4 = __dadd_rn(cn4, __dmul_rn(z4, cn6));
  cn0 = __dadd_rn(cn0, __dmul_rn(z8, cn4));
  cn0 = __dmul_rn(cn0, z);
  double cd0 = __dadd_rn(gt_atanf_cd[0], __dmul_rn(z2, gt_atanf_cd[1]));
  double cd2 = __dadd_rn(gt_atanf_cd[2], __dmul_rn(z2, gt_atanf_cd[3]));
  double cd4 = __dadd_rn(gt_atanf_cd[4], __dmul_rn(z2, gt_atanf_cd[5]));
  double cd6 = gt_atanf_cd[6];
  cd0 = __dadd_rn(cd0, __dmul_rn(z4, cd2));
  cd4 = __dadd_rn(cd4, __dmul_rn(z4, cd6));
  cd0 = __dadd_rn(cd0, __dmul_rn(z8, cd4));
  double r = __ddiv_rn(cn0, cd0);
  if (!gt)
    return gt_d2f(r);
  r = __dadd_rn(__dsub_rn(gt_sign(0x1.0fdaa22168c23p-7, z), r), gt_sign(0x1.9p0, z));
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/s_tanf.c
__device__ const double gt_tanf_cn[] = {0x1.921fb54442d18p+0, -0x1.fd226e573289fp-2, 0x1.b7a60c8dac9f6p-6, -0x1.725beb40f33e5p-13};
__device__ const double gt_tanf_cd[] = {0x1p+0, -0x1.2395347fb829dp+0, 0x1.2313660f29c36p-3, -0x1.9a707ab98d1c1p-9};
__device__ const double gt_tanf_s[] = {0, 1};
__device__ const struct 
    {
      float arg;
      float rh;
      float rl;
    } gt_tanf_st[] = {{0x1.143ec4p+0f, 0x1.ddf9f6p+0f, -0x1.891d24p-52f}, {0x1.ada6aap+27f, 0x1.e80304p-3f, 0x1.419f46p-58f}, {0x1.af61dap+48f, 0x1.60d1c8p-2f, -0x1.2d6c3ap-55f}, {0x1.0088bcp+52f, 0x1.ca1edp+0f, 0x1.f6053p-53f}, {0x1.f90dfcp+72f, 0x1.597f9cp-1f, 0x1.925978p-53f}, {0x1.cc4e22p+85f, -0x1.f33584p+1f, 0x1.d7254ap-51f}, {0x1.a6ce12p+86f, -0x1.c5612ep-1f, -0x1.26c33ep-53f}, {0x1.6a0b76p+102f, -0x1.e42a1ep+0f, -0x1.1dc906p-52f}};

__device__ inline static double gt_tanf_rltl(float z, int *q)
{
  double x = z;
  double idl = __dmul_rn(-0x1.b1bbead603d8bp-32, x);
  double idh = __dmul_rn(0x1.45f306ep-1, x);
  double id = gt_roundeven(idh);
  *q = (gt_i64) id;
  return __dadd_rn(__dsub_rn(idh, id), idl);
}


__device__ float glibc_tanf(float x)
{
  gt_u32 t = __float_as_uint(x);
  int e = (t >> 23) & 0xff;
  int i;
  double z;
  if (e < (127 + 28))
  {
    if (e < 115)
    {
      if (e < 102)
        return gt_tiny_fmaf(x, gt_absf(x), x);
      float x2 = __fmul_rn(x, x);
      return __fmaf_rn(x, __fmul_rn(0x1.555556p-2f, x2), x);
    }
    z = gt_tanf_rltl(x, &i);
  }
  else
    if (e < 0xff)
    z = gt_rbig(t, &i);
  else
  {
    if (t << 9)
      return gt_invalid(x); // Preserve the input NaN sign and payload.
    return gt_invalid(x);
  }
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  /* table gt_tanf_cn */;
  /* table gt_tanf_cd */;
  /* table gt_tanf_s */;
  double n = __dadd_rn(gt_tanf_cn[0], __dmul_rn(z2, gt_tanf_cn[1]));
  double n2 = __dadd_rn(gt_tanf_cn[2], __dmul_rn(z2, gt_tanf_cn[3]));
  n = __dadd_rn(n, __dmul_rn(z4, n2));
  double d = __dadd_rn(gt_tanf_cd[0], __dmul_rn(z2, gt_tanf_cd[1]));
  double d2 = __dadd_rn(gt_tanf_cd[2], __dmul_rn(z2, gt_tanf_cd[3]));
  d = __dadd_rn(d, __dmul_rn(z4, d2));
  n = __dmul_rn(n, z);
  double s0 = gt_tanf_s[i & 1];
  double s1 = gt_tanf_s[1 - (i & 1)];
  double r1 = __ddiv_rn(__dsub_rn(__dmul_rn(n, s1), __dmul_rn(d, s0)), __dadd_rn(__dmul_rn(n, s0), __dmul_rn(d, s1)));
  gt_u64 tail = (gt_as_u64(r1) + 7) & ((~0ULL) >> 35);
  if (tail <= 14)
  {
    /* table gt_tanf_st */;
    gt_u32 ax = t & ((~0u) >> 1);
    gt_u32 sgn = t >> 31;
    for (int j = 0; j < 8; j++)
    {
      if (__float_as_uint(gt_tanf_st[j].arg) == ax)
      {
        if (sgn)
          return __fsub_rn(-gt_tanf_st[j].rh, gt_tanf_st[j].rl);
        else
          return __fadd_rn(gt_tanf_st[j].rh, gt_tanf_st[j].rl);
      }
    }

  }
  return gt_d2f(r1);
}


#endif
