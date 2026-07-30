// WRF RRTMG McICA maximum-random subcolumn cloud masks, one thread per
// column.
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

extern "C" __global__ void rrtmgp_mcica_maxran(
    const float* play, const float* cldfra, unsigned char* mask,
    int ncol, int nlay, int ngpt, int permuteseed) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col >= ncol) return;
  // Seeds from the fractional Pa of the bottom four layer pressures
  // (module_ra_rrtmg_sw.F:1733-1741), advanced permuteseed times
  // (lines 1742-1744).
  unsigned int s1, s2, s3, s4;
  unsigned int* seeds[4] = {&s1, &s2, &s3, &s4};
  for (int n = 0; n < 4; ++n) {
    const double pmid = (double)play[col * nlay + n];
    const double frac = pmid - (double)((int)pmid);
    *seeds[n] = (unsigned int)((int)(frac * 1.0e9));
  }
  for (int i = 0; i < permuteseed; ++i) {
    mcica_kiss(s1, s2, s3, s4);
  }
  for (int g = 0; g < ngpt; ++g) {
    double cdf = 0.0;
    for (int k = 0; k < nlay; ++k) {
      const double draw = mcica_kiss(s1, s2, s3, s4);
      double cf = (double)cldfra[col * nlay + k];
      if (cf < 1.0e-20) cf = 0.0;
      if (k == 0) {
        cdf = draw;
      } else {
        double below = (double)cldfra[col * nlay + k - 1];
        if (below < 1.0e-20) below = 0.0;
        // Maximum-random walk: reuse the number when the layer below is
        // cloudy in this subcolumn, otherwise rescale a fresh draw
        // (lines 1803-1813, applied in place so the kept value chains).
        if (!(cdf > 1.0 - below)) {
          cdf = draw * (1.0 - below);
        }
      }
      mask[(col * nlay + k) * ngpt + g] =
          (cdf >= 1.0 - cf) ? (unsigned char)1 : (unsigned char)0;
    }
  }
}
