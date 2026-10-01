// WRF RRTMG McICA maximum-random subcolumn cloud masks, one thread per
// (column, g-point) subcolumn.
//
// Transcribed from WRF v4.6.1 phys/module_ra_rrtmg_sw.F:
//   kissvec RNG                        lines 2008-2040
//   pmid-fraction seeding + advance    lines 1727-1744
//   cldmin=1e-20 fraction floor        lines 1692, 1717-1725
//   maximum-random overlap (icld=2)    lines 1778-1813
//   subcolumn cloud decision           lines 1941-1944, 1951-1977
// module_ra_rrtmg_lw.F carries the identical generator; WRF drives both
// with irng=0 (kissvec) and permuteseed=1 (SW) / 150 (LW).  One
// subcolumn per g-point (nsubcsw = ngptsw, line 1476), and the draw
// order matches the Fortran loops: subcolumn outer, layer inner.
//
// THE DRAW ORDER IS PRESERVED EXACTLY.  The Fortran consumes one KISS
// stream per column, g-major and layer-minor, so subcolumn g starts at
// stream position permuteseed + g*nlay.  Rather than walk there, each
// thread JUMPS: all four KISS components admit closed-form advance, and
// the host hands in the jump operators for nlay*2^j steps so a thread
// composes only the bits set in its own g.  Every draw a thread then makes
// is the same draw, in the same order, that the serial loop made.
//
//   s1  affine LCG mod 2^32          -- compose (A,C) pairs
//   s2  xorshift32, GF(2)-linear     -- 32x32 bit-matrix, applied to the
//                                       state vector (32 XORs), not composed
//   s3  MWC(18000, 2^16)             -- multiply residue by 2^-16m mod p
//   s4  MWC(30903, 2^16)                where p = a*2^16 - 1
//
// The host folds the LEVELS into one operator per subcolumn, so a thread
// applies a single operator instead of walking the set bits of its own g.
// Every level is a power of the same map, so the fold is exact in each of
// the four algebras -- affine composition mod 2^32, GF(2) matrix product,
// and modular multiplication of the two MWC residues -- and the state a
// thread lands on is the state the level-by-level walk landed on.
//
// The MWC identity is b*s' = s (mod p), so s_m = s_0 * b^-m (mod p).  That
// reconstruction is exact only while the true state is below p -- and it can
// reach exactly p+1, which aliases onto residues 0 and 1.  Those two
// residues are therefore refused and the thread walks the stream instead.
// It costs a rare thread its jump and never costs an answer.

__device__ __forceinline__ double mcica_kiss(
    unsigned int& s1, unsigned int& s2, unsigned int& s3,
    unsigned int& s4) {
  // The KISS combination; Fortran ISHFT is a logical (zero-fill) shift,
  // matching unsigned C shifts, and integer overflow wraps mod 2^32.
  s1 = 69069u * s1 + 1327217885u;
  s2 = s2 ^ (s2 << 13);
  s2 = s2 ^ (s2 >> 17);
  s2 = s2 ^ (s2 << 5);
  s3 = 18000u * (s3 & 65535u) + (s3 >> 16);
  s4 = 30903u * (s4 & 65535u) + (s4 >> 16);
  const int kiss = (int)(s1 + s2 + (s3 << 16) + s4);
  return (double)kiss * 2.328306e-10 + 0.5;
}

// Seed from the fractional Pa of the bottom four layer pressures
// (module_ra_rrtmg_sw.F:1733-1741), then advance permuteseed times
// (lines 1742-1744).
__device__ __forceinline__ void mcica_seed(
    const float* play, int col, int nlay, int permuteseed,
    unsigned int& s1, unsigned int& s2, unsigned int& s3,
    unsigned int& s4) {
  unsigned int* seeds[4] = {&s1, &s2, &s3, &s4};
  for (int n = 0; n < 4; ++n) {
    const double pmid = (double)play[col * nlay + n];
    const double frac = pmid - (double)((int)pmid);
    *seeds[n] = (unsigned int)((int)(frac * 1.0e9));
  }
  for (int i = 0; i < permuteseed; ++i) {
    mcica_kiss(s1, s2, s3, s4);
  }
}

// Apply a 32x32 GF(2) matrix (given as the images of the basis vectors) to
// the state vector: XOR the columns selected by the set bits.
__device__ __forceinline__ unsigned int mcica_gf2_apply(
    const unsigned int* mat, unsigned int v) {
  unsigned int out = 0u;
  while (v) {
    const int bit = __ffs((int)v) - 1;
    out ^= mat[bit];
    v &= v - 1u;
  }
  return out;
}

#define MCICA_P3 1179647999u
#define MCICA_P4 2025259007u

// One block per column.  Everything that depends only on the column is
// then computed once instead of ngpt times: `s_omc[k]` is `1 - cldfra[k]`
// with the cldmin floor applied, the same double the serial code formed,
// formed once for the whole block.  The layer loop also carries it -- the
// "below" of layer k IS the "cf" of layer k-1 -- so each layer costs one
// shared load rather than two float->double converts, two floor tests and
// two subtractions.  On a card whose FP64 rate is 1/64 of FP32 that
// redundancy was most of the kernel.
extern "C" __global__ void rrtmgp_mcica_maxran(
    const float* play, const float* cldfra, unsigned char* mask,
    const unsigned int* jump_s1, const unsigned int* jump_s2,
    const unsigned int* jump_s3, const unsigned int* jump_s4,
    int ncol, int nlay, int ngpt, int permuteseed) {
  const int col = blockIdx.x;
  // The WHOLE block leaves together, so the __syncthreads below is reached
  // by every thread that reaches the block at all.
  if (col >= ncol) return;

  extern __shared__ double s_omc[];
  for (int k = (int)threadIdx.x; k < nlay; k += (int)blockDim.x) {
    double cf = (double)cldfra[col * nlay + k];
    if (cf < 1.0e-20) cf = 0.0;
    s_omc[k] = 1.0 - cf;
  }
  __syncthreads();

  for (int g = (int)threadIdx.x; g < ngpt; g += (int)blockDim.x) {
    unsigned int s1, s2, s3, s4;
    mcica_seed(play, col, nlay, permuteseed, s1, s2, s3, s4);

    if (g > 0) {
      unsigned int r3 = s3 % MCICA_P3;
      unsigned int r4 = s4 % MCICA_P4;
      const unsigned int a1 = jump_s1[g * 2] * s1 + jump_s1[g * 2 + 1];
      const unsigned int a2 = mcica_gf2_apply(jump_s2 + g * 32, s2);
      r3 = (unsigned int)((unsigned long long)r3
                          * (unsigned long long)jump_s3[g] % MCICA_P3);
      r4 = (unsigned int)((unsigned long long)r4
                          * (unsigned long long)jump_s4[g] % MCICA_P4);
      if (r3 > 1u && r4 > 1u) {
        // Unambiguous: the residue IS the state.
        s1 = a1; s2 = a2; s3 = r3; s4 = r4;
      } else {
        // Aliased onto p or p+1.  Walk the stream, exactly as the Fortran
        // does, rather than guess.
        const int steps = g * nlay;
        for (int i = 0; i < steps; ++i) mcica_kiss(s1, s2, s3, s4);
      }
    }

    double cdf = 0.0;
    double omb = 0.0;  // 1 - cldfra[k-1]; only read once k > 0
    for (int k = 0; k < nlay; ++k) {
      const double draw = mcica_kiss(s1, s2, s3, s4);
      const double omc = s_omc[k];
      if (k == 0) {
        cdf = draw;
      } else if (!(cdf > omb)) {
        // Maximum-random walk: reuse the number when the layer below is
        // cloudy in this subcolumn, otherwise rescale a fresh draw
        // (lines 1803-1813, applied in place so the kept value chains).
        cdf = draw * omb;
      }
      mask[(col * nlay + k) * ngpt + g] =
          (cdf >= omc) ? (unsigned char)1 : (unsigned char)0;
      omb = omc;
    }
  }
}
