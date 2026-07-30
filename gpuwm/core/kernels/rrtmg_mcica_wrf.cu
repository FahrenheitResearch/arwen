// WRF v4.6.1 legacy RRTMG McICA subcolumn generators -- CUDA FP32 twin
// of the certified NumPy port in gpuwm/core/rrtmg_mcica.py, restricted
// to the icld=2 (maximum-random) forecast path and gated bitwise
// against the same unmodified-WRF fixtures through that port
// (tests/test_rrtmg_mcica.py).
//
// Fortran authority (comment anchors): module mcica_subcol_gen_lw in
// phys/module_ra_rrtmg_lw.F --
//   kissvec RNG                           lines 2697-2729
//   pmid-fraction seeding + warm-up       lines 2418-2438
//   cldmin=1e-20 cloud-fraction floor     lines 2383, 2408-2416
//   maximum-random overlap walk (icld=2)  lines 2469-2504
//   subcolumn cloud decision + fill       lines 2632-2665
// module_ra_rrtmg_sw.F carries the identical generator; only ngpt, the
// ngb band table and the output roster differ, all supplied by the
// host through the descriptor arrays of rmcw_fill_outputs.
//
// DISTINCT from gpuwm/core/kernels/rrtmgp_mcica.cu: that file is the
// RTE+RRTMGP-side generator whose seed arithmetic runs in double from
// pressures already in Pa.  THIS file reproduces the WRF FP32 kind_rb
// build's double-rounded seed path: play(mb) -> *100 rounded to
// float32 -> truncate -> exact fractional part -> *1e9 rounded to
// float32 -> truncate to int, transcribing the NumPy port's op
// sequence (rrtmg_mcica.py::_make_seeds / _kissvec) statement for
// statement.
//
// EVERY float arithmetic operation goes through RMCW_AD/SU/MU
// (__fadd_rn etc.): NVRTC defaults to --fmad=true and contraction is
// the dominant bitwise hazard; gfortran -O0 on x86-64 emits no FMA.
// Compile with --ftz=false: the walk product cdf*(1-cldf) is applied
// repeatedly down a column and reaches the FP32 subnormal range when
// many consecutive layers sit near cldf=1.  The host preflight
// (rrtmg_mcica.py::mcica_gpu_preflight) PROVES subnormal survival and
// the seed/kissvec chain on the live toolchain rather than assuming
// them.  No transcendentals exist on this path (the decorrelation
// exponentials belong to icld=4/5, which fail closed host-side).
//
// Integer RNG state is exact 32-bit wraparound arithmetic: unsigned
// carriers here, signed wraparound in the Fortran -- identical bit
// patterns (the NumPy port's uint32 masks document the
// correspondence), and Fortran ISHFT is a logical shift = unsigned C
// shifts.  Float->int conversions (__float2int_rz) saturate on CUDA
// where x86 wraps to INT_MIN; both conversions on the seed path are
// range-bounded for in-contract inputs (pmid < ~1.1e5 Pa physically,
// and frac*1e9 < 2^31 always because frac < 1 exactly: pmid and its
// truncation are within a factor of two, so the subtraction is exact
// by Sterbenz), hence the semantics never diverge.

#define RMCW_AD(a, b) __fadd_rn((a), (b))
#define RMCW_SU(a, b) __fsub_rn((a), (b))
#define RMCW_MU(a, b) __fmul_rn((a), (b))

#define RMCW_CLDMIN 1.0e-20f
#define RMCW_KISS_SCALE 2.328306e-10f
#define RMCW_PMID_SCALE 1.0e2f
#define RMCW_SEED_SCALE 1000000000.0f

// Output source kinds consumed by rmcw_walk_outputs (mirrored by
// rrtmg_mcica.py's _MCICA_SRC_* constants).
#define RMCW_SRC_CONST1 0  // cloudy value 1.0 (cldfmc)
#define RMCW_SRC_PERCOL 1  // cloudy value src[col, lay]
#define RMCW_SRC_BAND 2    // cloudy value src[bandoff[isub], col, lay]

// One kissvec state advance (module_ra_rrtmg_lw.F lines 2721-2725).
// The warm-up loop (lines 2433-2435) computes and discards the float
// draw; skipping the draw touches no state, so warm-up uses this alone.
__device__ __forceinline__ void rmcw_kiss_advance(
    unsigned int &s1, unsigned int &s2, unsigned int &s3,
    unsigned int &s4)
{
    s1 = 69069u * s1 + 1327217885u;
    s2 = s2 ^ (s2 << 13);
    s2 = s2 ^ (s2 >> 17);
    s2 = s2 ^ (s2 << 5);
    s3 = 18000u * (s3 & 0xFFFFu) + (s3 >> 16);
    s4 = 30903u * (s4 & 0xFFFFu) + (s4 >> 16);
}

// Advance + draw: REAL(kiss) is the signed int32 value converted
// round-nearest to float32, then one rounded multiply and one rounded
// add (line 2726; rrtmg_mcica.py::_kissvec).
__device__ __forceinline__ float rmcw_kiss_draw(
    unsigned int &s1, unsigned int &s2, unsigned int &s3,
    unsigned int &s4)
{
    rmcw_kiss_advance(s1, s2, s3, s4);
    unsigned int kiss = s1 + s2 + (s3 << 16) + s4;
    return RMCW_AD(RMCW_MU(__int2float_rn((int)kiss), RMCW_KISS_SCALE),
                   0.5f);
}

// One seed word from one bottom-layer pressure (lines 2428-2431):
// pmid = play*100 rounded once; INT() truncates; the fractional part
// is exact (Sterbenz); *1e9 rounded once; INT() truncates.
__device__ __forceinline__ unsigned int rmcw_seed_from_play(
    float play_mb)
{
    float pm = RMCW_MU(play_mb, RMCW_PMID_SCALE);
    int ipart = __float2int_rz(pm);
    float frac = RMCW_SU(pm, __int2float_rn(ipart));
    float scaled = RMCW_MU(frac, RMCW_SEED_SCALE);
    return (unsigned int)__float2int_rz(scaled);
}

// Preflight probe: x = [1e-30, 1e-10, play0..play3].  out[0] proves
// subnormal survival of __fmul_rn under the live compile options;
// out[1]/out[2] run the full seed chain plus two draws for bitwise
// comparison against the NumPy port's _make_seeds/_kissvec.
extern "C" __global__ void rmcw_probe(const float *x, float *out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    out[0] = RMCW_MU(x[0], x[1]);
    unsigned int s1 = rmcw_seed_from_play(x[2]);
    unsigned int s2 = rmcw_seed_from_play(x[3]);
    unsigned int s3 = rmcw_seed_from_play(x[4]);
    unsigned int s4 = rmcw_seed_from_play(x[5]);
    out[1] = rmcw_kiss_draw(s1, s2, s3, s4);
    out[2] = rmcw_kiss_draw(s1, s2, s3, s4);
}

// Kernel 1 -- one thread per column of the chunk: derive the column's
// four seed words from its bottom four layer-midpoint pressures,
// apply the changeSeed warm-up advances, then fill the chunk's raw
// CDF slab in the exact Fortran draw order for icld=2: subcolumn
// outer, layer inner (lines 2477-2482), one draw per (isubcol, ilev).
// All RNG state lives in registers; the per-column stream depends
// only on that column's play values and change_seed, which is what
// makes host-side chunking bitwise invisible.
//
// cdf layout (ngpt, nlay, nc): a warp handles adjacent columns, so
// the slab is written (and later read) coalesced.  The frozen
// (ngpt, ncol, nlay) output layout is materialized only by
// rmcw_walk_outputs.
extern "C" __global__ void rmcw_fill_cdf(
    int col0, int nc, int nlay, int ngpt, int change_seed,
    const float *__restrict__ play,  // (ncol_total, nlay) mb
    float *__restrict__ cdf)         // (ngpt, nlay, nc)
{
    int cc = blockDim.x * blockIdx.x + threadIdx.x;
    if (cc >= nc) return;
    const float *pcol = play + (size_t)(col0 + cc) * nlay;
    unsigned int s1 = rmcw_seed_from_play(pcol[0]);
    unsigned int s2 = rmcw_seed_from_play(pcol[1]);
    unsigned int s3 = rmcw_seed_from_play(pcol[2]);
    unsigned int s4 = rmcw_seed_from_play(pcol[3]);
    for (int i = 0; i < change_seed; ++i)
        rmcw_kiss_advance(s1, s2, s3, s4);
    size_t idx = (size_t)cc;
    for (int g = 0; g < ngpt; ++g) {
        for (int l = 0; l < nlay; ++l) {
            cdf[idx] = rmcw_kiss_draw(s1, s2, s3, s4);
            idx += (size_t)nc;
        }
    }
}

// Kernel 2 -- one thread per (subcolumn, column) pair: the icld=2
// maximum-random walk down the column (lines 2494-2504: keep the
// level-below value when it exceeded that layer's clear fraction,
// else rescale the fresh draw by it) and the >= 1-cldf cloud decision
// (line 2634), emitted as a one-byte iscloudy mask in the chunk-local
// (ngpt, nc, nlay) layout.  The cldmin floor (lines 2409-2416) is
// re-derived from raw cldfrac at each use -- a pure predicate, no
// rounding.  Splitting the mask from the output fill is pure data
// movement (the walk arithmetic is unchanged): the sequential-in-l
// walk thread writes 1 B/cell instead of 4*nout, and the fill below
// then streams the big slabs coalesced.
extern "C" __global__ void rmcw_walk_mask(
    int col0, int nc, int nlay, int ngpt,
    const float *__restrict__ cdf,      // (ngpt, nlay, nc)
    const float *__restrict__ cldfrac,  // (ncol_total, nlay)
    unsigned char *__restrict__ mask)   // (ngpt, nc, nlay)
{
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long long)ngpt * nc) return;
    int isub = (int)(t / nc);
    int cc = (int)(t % nc);
    const float *cfcol = cldfrac + (size_t)(col0 + cc) * nlay;
    const float *cdfcol = cdf + (size_t)isub * nlay * nc + cc;
    unsigned char *mcol = mask + ((size_t)isub * nc + cc) * nlay;
    float prev = 0.0f;
    for (int l = 0; l < nlay; ++l) {
        float walked = cdfcol[(size_t)l * nc];
        if (l > 0) {
            float cfb = cfcol[l - 1];
            cfb = (cfb < RMCW_CLDMIN) ? 0.0f : cfb;
            float clear = RMCW_SU(1.0f, cfb);
            walked = (prev > clear) ? prev : RMCW_MU(walked, clear);
        }
        float cfl = cfcol[l];
        cfl = (cfl < RMCW_CLDMIN) ? 0.0f : cfl;
        mcol[l] = (walked >= RMCW_SU(1.0f, cfl)) ? 1 : 0;
        prev = walked;
    }
}

// Kernel 3 -- one thread per (subcolumn, column, layer) element: the
// output fill (lines 2642-2665), written straight into the frozen
// (ngpt, ncol_total, nlay) layout at the chunk's column offset.
// Adjacent threads share a row's adjacent layers, so every mask read,
// source gather and output store is coalesced.  No float arithmetic:
// values are copied from the sources or the clear-sky constants
// exactly as the Fortran assignments do.
//
// nout <= 8 descriptor-driven outputs so the LW roster (cldf, ciwp,
// clwp, cswp, tauc; clear values 0) and the SW roster (plus ssac with
// clear value 1, asmc, fsfc) share this one audited kernel; bandoff
// carries the 0-based ngb band index per g-point for RMCW_SRC_BAND
// sources.
extern "C" __global__ void rmcw_fill_outputs(
    int ncol_total, int col0, int nc, int nlay, int ngpt,
    const unsigned char *__restrict__ mask,          // (ngpt, nc, nlay)
    int nout,
    const unsigned long long *__restrict__ outptrs,  // [nout] float*
    const unsigned long long *__restrict__ srcptrs,  // [nout] float*
    const int *__restrict__ srckind,                 // [nout]
    const float *__restrict__ clearval,              // [nout]
    const int *__restrict__ bandoff)                 // [ngpt] 0-based
{
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long long)ngpt * nc * nlay) return;
    int l = (int)(t % nlay);
    long long rest = t / nlay;
    int cc = (int)(rest % nc);
    int isub = (int)(rest / nc);
    int col = col0 + cc;
    bool cloudy = mask[t] != 0;
    size_t oidx = ((size_t)isub * ncol_total + col) * nlay + l;
    for (int j = 0; j < nout; ++j) {
        float v;
        if (cloudy) {
            int kind = srckind[j];
            const float *src = (const float *)srcptrs[j];
            if (kind == RMCW_SRC_CONST1) {
                v = 1.0f;
            } else if (kind == RMCW_SRC_PERCOL) {
                v = src[(size_t)col * nlay + l];
            } else {
                v = src[((size_t)bandoff[isub] * ncol_total + col)
                        * nlay + l];
            }
        } else {
            v = clearval[j];
        }
        ((float *)outptrs[j])[oidx] = v;
    }
}
