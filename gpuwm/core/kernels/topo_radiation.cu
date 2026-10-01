// gpuwm/core/kernels/topo_radiation.cu
//
// WRF v4.7.1 slope-dependent shortwave (slope_rad = 1) and terrain
// shadowing (topo_shading = 1), transcribed statement by statement:
//
//   topo_slope_geometry  dyn_em/start_em.F:1539-1577 (SLOPE, SLP_AZI)
//   topo_shadow_init     phys/module_radiation_driver.F:4474-4608
//                        (toposhad_init, iteration 1)
//   topo_shadow_scan     phys/module_radiation_driver.F:4610-4862
//                        (toposhad, iteration 1 on one patch)
//   topo_diffuse_frac    phys/module_radiation_driver.F:2894-2927
//                        (Ruiz-Arias split for a shortwave scheme without
//                        its own direct beam, then DIFFUSE_FRAC)
//   topo_rad_adjust      phys/module_surface_driver.F:6977-7114
//                        (TOPO_RAD_ADJ_DRVR + TOPO_RAD_ADJ)
//   topo_rad_restore     phys/module_surface_driver.F:4461-4481
//
// FP32 throughout, every operation rounded where the Fortran rounds it:
// the explicit __f*_rn intrinsics keep NVRTC from contracting a multiply
// and an add into an FMA that gfortran (no -march) never emits, and every
// transcendental is the float nearest the double result -- the oracle's
// second build links the same definitions (tools/
// wrf_topo_radiation_v471_oracle/libm_cr.c), so the two agree bit for bit
// and the stock-glibc build measures the libm seam alone.
//
// Indices are WRF's: 1-based (i, j) on the mass grid, nx by ny.  One patch
// covers the domain, as in a serial wrf.exe: toposhad's iteration-1 scan
// reads up to two cells past the domain edge, where HT carries WRF's
// specified/nested halo (set_physical_bc2d copies the edge mass value
// outward), so the shadow kernels read a height field widened by two cells
// on every side, (ny + 4) x (nx + 4), element (i, j) at [(j+1)*(nx+4)+i+1].

#define TR_W(i, j) ((size_t)((j) + 1) * (size_t)(nx + 4) + (size_t)((i) + 1))
#define TR_M(i, j) ((size_t)((j) - 1) * (size_t)nx + (size_t)((i) - 1))

// module_model_constants.F:72-73, folded by gfortran in single precision:
// piconst = 3.1415926535897932384626433 -> 0x40490FDB, DEGRAD = piconst/180.
__device__ __forceinline__ float tr_degrad() { return __int_as_float(0x3C8EFA35); }

__device__ __forceinline__ float tr_sin(float x) { return (float)sin((double)x); }
__device__ __forceinline__ float tr_cos(float x) { return (float)cos((double)x); }
__device__ __forceinline__ float tr_tan(float x) { return (float)tan((double)x); }
__device__ __forceinline__ float tr_asin(float x) { return (float)asin((double)x); }
__device__ __forceinline__ float tr_acos(float x) { return (float)acos((double)x); }
__device__ __forceinline__ float tr_atan(float x) { return (float)atan((double)x); }
__device__ __forceinline__ float tr_atan2(float y, float x)
{
    return (float)atan2((double)y, (double)x);
}
__device__ __forceinline__ float tr_exp(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float tr_pow(float x, float y)
{
    return (float)pow((double)x, (double)y);
}
// 4.*atan(1.): a constant expression gfortran folds, correctly rounded.
__device__ __forceinline__ float tr_pi() { return __fmul_rn(4.0f, tr_atan(1.0f)); }

// ---------------------------------------------------------------------------
// SLOPE and SLP_AZI from HT (start_em.F:1539-1577), non-periodic domain.
// ---------------------------------------------------------------------------
extern "C" __global__ void topo_slope_geometry(
    const float* __restrict__ ht, const float* __restrict__ msftx,
    const float* __restrict__ msfty, const float* __restrict__ sina,
    const float* __restrict__ cosa, float rdx, float rdy,
    int nx, int ny,
    float* __restrict__ slope, float* __restrict__ slp_azi)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    const int i = idx % nx + 1;
    const int j = idx / nx + 1;
    const int im1 = max(i - 1, 1);
    const int ip1 = min(i + 1, nx);
    const int jm1 = max(j - 1, 1);
    const int jp1 = min(j + 1, ny);
    // (ht(ip1)-ht(im1))*msftx*rdx/(ip1-im1), left to right
    const float hx = __fdiv_rn(__fmul_rn(__fmul_rn(
        __fsub_rn(ht[TR_M(ip1, j)], ht[TR_M(im1, j)]), msftx[TR_M(i, j)]),
        rdx), (float)(ip1 - im1));
    const float hy = __fdiv_rn(__fmul_rn(__fmul_rn(
        __fsub_rn(ht[TR_M(i, jp1)], ht[TR_M(i, jm1)]), msfty[TR_M(i, j)]),
        rdy), (float)(jp1 - jm1));
    const float pi = tr_pi();
    // atan((hx**2+hy**2)**.5): **2 is a multiply, **.5 is powf
    const float s = tr_atan(tr_pow(
        __fadd_rn(__fmul_rn(hx, hx), __fmul_rn(hy, hy)), 0.5f));
    float azi;
    float slp = s;
    if (s < 1.e-4f) {
        slp = 0.0f;
        azi = 0.0f;
    } else {
        azi = __fadd_rn(tr_atan2(hx, hy), pi);
        // Rotate slope azimuth to lat-lon grid
        const float asa = tr_asin(sina[TR_M(i, j)]);
        if (cosa[TR_M(i, j)] >= 0.0f) {
            azi = __fsub_rn(azi, asa);
        } else {
            azi = __fsub_rn(azi, __fsub_rn(pi, asa));
        }
    }
    slope[TR_M(i, j)] = slp;
    slp_azi[TR_M(i, j)] = azi;
}

// ---------------------------------------------------------------------------
// toposhad_init, iteration 1 (module_radiation_driver.F:4498-4576).
//
// ht_loc is the widened height field (a copy of widened HT), modified in
// place for a nest; ht_shad arrives holding the parent's shadow height in
// the outer two rows of a nest (spec_bdyfield, pre_radiation_driver) and
// leaves initialized.  Run with one thread per mass point; the edge checks
// of a nest read and write only their own point, so the order WRF runs
// them in cannot matter (a raised point satisfies no second test).
// ---------------------------------------------------------------------------
extern "C" __global__ void topo_shadow_init(
    float* __restrict__ ht_loc, float* __restrict__ ht_shad,
    int* __restrict__ shadowmask, int nested, int nx, int ny)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    const int i = idx % nx + 1;
    const int j = idx / nx + 1;
    int mask = 0;
    float shad = ht_shad[TR_M(i, j)];
    float loc = ht_loc[TR_W(i, j)];
    // ids = jds = 1, ide = nx + 1, jde = ny + 1
    if (!nested || (i >= 3 && i <= nx - 2 && j >= 3 && j <= ny - 2)) {
        shad = __fsub_rn(loc, 0.001f);
    }
    if (nested) {
        const bool edge = (i <= 2 || i >= nx - 1 || j <= 2 || j >= ny - 1);
        if (edge && shad > loc) {
            mask = 1;
            loc = shad;
        }
        ht_loc[TR_W(i, j)] = loc;
    }
    shadowmask[TR_M(i, j)] = mask;
    ht_shad[TR_M(i, j)] = shad;
}

// One quarter's scan (module_radiation_driver.F:4681-4755).  `along_j`
// scans rows (north/south quarters), otherwise columns (east/west).
__device__ __forceinline__ void tr_scan(
    const float* __restrict__ ht_loc, int nx, int ny, int i, int j,
    int step, bool along_j, float t, float dx, float dy, float csza,
    int gpshad, int* mask, float* shad)
{
    const float h0 = ht_loc[TR_W(i, j)];
    for (int n = 1; n <= gpshad; ++n) {
        const int k = along_j ? j + step * n : i + step * n;   // jj or ii
        float r;       // ri or rj
        int c1;        // i1 or j1
        float dxabs;
        if (along_j) {
            // ri = i + (jj-j)*tan(sol_azi)
            r = __fadd_rn((float)i, __fmul_rn((float)(k - j), t));
            c1 = (int)r;
            const float a = __fmul_rn(dy, (float)(k - j));
            const float b = __fmul_rn(dx, __fsub_rn(r, (float)i));
            dxabs = __fsqrt_rn(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b)));
        } else {
            // rj = j - (ii-i)*tan(pi/2.+sol_azi)
            r = __fsub_rn((float)j, __fmul_rn((float)(k - i), t));
            c1 = (int)r;
            const float a = __fmul_rn(dx, (float)(k - i));
            const float b = __fmul_rn(dy, __fsub_rn(r, (float)j));
            dxabs = __fsqrt_rn(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b)));
        }
        const int c2 = c1 + 1;
        const float wgt = __fsub_rn(r, (float)c1);
        // Iteration-1 bounds on one patch: ips = jps = 1, ipe = nx,
        // jpe = ny (pre_radiation_driver passes min(ipe,ide-1)).
        bool stop;
        if (along_j) {
            stop = (step > 0 ? k >= ny + 3 : k <= -2) || c1 <= -2
                   || c2 >= nx + 3;
        } else {
            stop = (step > 0 ? k >= nx + 3 : k <= -2) || c1 <= -2
                   || c2 >= ny + 3;
        }
        if (stop) return;
        const float h1 = along_j ? ht_loc[TR_W(c1, k)] : ht_loc[TR_W(k, c1)];
        const float h2 = along_j ? ht_loc[TR_W(c2, k)] : ht_loc[TR_W(k, c2)];
        // atan((wgt*h(i2)+(1.-wgt)*h(i1)-ht_loc(i,j))/dxabs)
        const float topoelev = tr_atan(__fdiv_rn(__fsub_rn(__fadd_rn(
            __fmul_rn(wgt, h2), __fmul_rn(__fsub_rn(1.0f, wgt), h1)), h0),
            dxabs));
        if (tr_sin(topoelev) >= csza) {
            *mask = 1;
            *shad = fmaxf(*shad, __fadd_rn(h0, __fmul_rn(dxabs, __fsub_rn(
                tr_tan(topoelev), tr_tan(tr_asin(csza))))));
        }
    }
}

// ---------------------------------------------------------------------------
// toposhad, iteration 1 (module_radiation_driver.F:4640-4758).
// ---------------------------------------------------------------------------
extern "C" __global__ void topo_shadow_scan(
    const float* __restrict__ ht_loc, const float* __restrict__ xlat,
    const float* __restrict__ xlong, const float* __restrict__ sina,
    const float* __restrict__ cosa, float xtime, float gmt, float radfrq,
    float declin, float dx, float dy, float shadlen, int nx, int ny,
    int* __restrict__ shadowmask, float* __restrict__ ht_shad)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    const int i = idx % nx + 1;
    const int j = idx / nx + 1;
    const float degrad = tr_degrad();
    // XT24=MOD(XTIME+RADFRQ*0.5,1440.)
    const float xt24 = fmodf(__fadd_rn(xtime, __fmul_rn(radfrq, 0.5f)),
                             1440.0f);
    const float pi = tr_pi();
    const int gpshad = (int)__fadd_rn(__fdiv_rn(shadlen, dx), 1.0f);

    int mask = shadowmask[TR_M(i, j)];
    float shad = ht_shad[TR_M(i, j)];
    // TLOCTM=GMT+XT24/60.+XLONG(i,j)/15.
    const float tloctm = __fadd_rn(__fadd_rn(gmt, __fdiv_rn(xt24, 60.0f)),
                                   __fdiv_rn(xlong[TR_M(i, j)], 15.0f));
    const float hrang = __fmul_rn(__fmul_rn(15.0f, __fsub_rn(tloctm, 12.0f)),
                                  degrad);
    const float xxlat = __fmul_rn(xlat[TR_M(i, j)], degrad);
    const float csza = __fadd_rn(__fmul_rn(tr_sin(xxlat), tr_sin(declin)),
                                 __fmul_rn(__fmul_rn(tr_cos(xxlat),
                                                     tr_cos(declin)),
                                           tr_cos(hrang)));
    if (csza < 1.e-2f) {
        // shadow mask does not need to be computed
        shadowmask[TR_M(i, j)] = 0;
        ht_shad[TR_M(i, j)] = __fsub_rn(ht_loc[TR_W(i, j)], 0.001f);
        return;
    }
    // Solar azimuth angle
    float argu = __fdiv_rn(
        __fsub_rn(__fmul_rn(csza, tr_sin(xxlat)), tr_sin(declin)),
        __fmul_rn(tr_sin(tr_acos(csza)), tr_cos(xxlat)));
    if (argu > 1.0f) argu = 1.0f;
    if (argu < -1.0f) argu = -1.0f;
    // sign(acos(argu),sin(HRANG))+pi: gfortran's SIGN is copysign
    float sol_azi = __fadd_rn(copysignf(fabsf(tr_acos(argu)), tr_sin(hrang)),
                              pi);
    if (cosa[TR_M(i, j)] >= 0.0f) {
        sol_azi = __fadd_rn(sol_azi, tr_asin(sina[TR_M(i, j)]));
    } else {
        sol_azi = __fsub_rn(__fadd_rn(sol_azi, pi),
                            tr_asin(sina[TR_M(i, j)]));
    }
    // Scan for higher surrounding topography
    const float pi_quarter = __fmul_rn(0.25f, pi);
    if (sol_azi > __fmul_rn(1.75f, pi) || sol_azi < pi_quarter) {
        // sun is in the northern quarter
        tr_scan(ht_loc, nx, ny, i, j, +1, true, tr_tan(sol_azi), dx, dy,
                csza, gpshad, &mask, &shad);
    } else if (sol_azi < __fmul_rn(0.75f, pi)) {
        // sun is in the eastern quarter
        tr_scan(ht_loc, nx, ny, i, j, +1, false,
                tr_tan(__fadd_rn(__fdiv_rn(pi, 2.0f), sol_azi)), dx, dy,
                csza, gpshad, &mask, &shad);
    } else if (sol_azi < __fmul_rn(1.25f, pi)) {
        // sun is in the southern quarter
        tr_scan(ht_loc, nx, ny, i, j, -1, true, tr_tan(sol_azi), dx, dy,
                csza, gpshad, &mask, &shad);
    } else {
        // sun is in the western quarter
        tr_scan(ht_loc, nx, ny, i, j, -1, false,
                tr_tan(__fadd_rn(__fdiv_rn(pi, 2.0f), sol_azi)), dx, dy,
                csza, gpshad, &mask, &shad);
    }
    shadowmask[TR_M(i, j)] = mask;
    ht_shad[TR_M(i, j)] = shad;
}

// ---------------------------------------------------------------------------
// SWDDIF for a shortwave scheme without its own direct beam, then
// DIFFUSE_FRAC (module_radiation_driver.F:2894-2927).  `ruiz` = 0 takes
// the scheme's SWDDIF as given (RRTMG's surface diffuse flux).  swddif
// holds the radiation call's zeroed/scheme value on entry.
// ---------------------------------------------------------------------------
extern "C" __global__ void topo_diffuse_frac(
    const float* __restrict__ coszen, const float* __restrict__ swdown,
    const float* __restrict__ ht, float* __restrict__ swddif,
    float solcon, int ruiz, int n, float* __restrict__ diffuse_frac)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float mu = coszen[idx];
    const float down = swdown[idx];
    float dif = swddif[idx];
    if (ruiz && mu > 1e-3f) {
        const float ioh = __fmul_rn(solcon, mu);          // TOA irradiance
        float kt = __fdiv_rn(down, fmaxf(ioh, 1e-3f));    // clearness index
        // airmass=exp(-ht/8434.5)/(coszen+0.50572*
        //         (asin(coszen)*57.295779513082323+6.07995)**(-1.6364))
        const float airmass = __fdiv_rn(
            tr_exp(__fdiv_rn(-ht[idx], 8434.5f)),
            __fadd_rn(mu, __fmul_rn(0.50572f, tr_pow(__fadd_rn(
                __fmul_rn(tr_asin(mu), 57.295779513082323f), 6.07995f),
                -1.6364f))));
        // kt=kt/(0.1+1.031*exp(-1.4/(0.9+(9.4/max(airmass,1e-3)))))
        kt = __fdiv_rn(kt, __fadd_rn(0.1f, __fmul_rn(1.031f, tr_exp(
            __fdiv_rn(-1.4f, __fadd_rn(0.9f, __fdiv_rn(
                9.4f, fmaxf(airmass, 1e-3f))))))));
        // kd=0.952-1.041*exp(-exp(2.300-4.702*kt))
        const float kd = __fsub_rn(0.952f, __fmul_rn(1.041f, tr_exp(
            -tr_exp(__fsub_rn(2.300f, __fmul_rn(4.702f, kt))))));
        dif = __fmul_rn(kd, down);
        swddif[idx] = dif;
    }
    float frac;
    if (down > 0.001f) {
        frac = __fdiv_rn(dif, down);
        frac = fminf(frac, 1.0f);
    } else {
        frac = 0.0f;
    }
    diffuse_frac[idx] = frac;
}

// ---------------------------------------------------------------------------
// TOPO_RAD_ADJ_DRVR + TOPO_RAD_ADJ (module_surface_driver.F:6977-7114).
// swnorm and gswsave are the driver's saves, restored by
// topo_rad_restore after the land surface has run.
// ---------------------------------------------------------------------------
extern "C" __global__ void topo_rad_adjust(
    const float* __restrict__ xlat, const float* __restrict__ coszen,
    const int* __restrict__ shadowmask,
    const float* __restrict__ diffuse_frac, const float* __restrict__ hrang,
    const float* __restrict__ slope, const float* __restrict__ slp_azi,
    float declin, int n,
    float* __restrict__ swdown, float* __restrict__ gsw,
    float* __restrict__ swnorm, float* __restrict__ gswsave)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // pi = 4.*atan(1.); degrad=pi/180.
    const float degrad = __fdiv_rn(tr_pi(), 180.0f);
    const float sw = swdown[idx];
    swnorm[idx] = sw;                                  // save
    if (!(sw > 1.E-3f)) return;                        // daytime
    const int shadow = shadowmask[idx];
    // TOPO_RAD_ADJ
    float teradj = sw;
    const float csza = coszen[idx];
    const float xxlat = __fmul_rn(xlat[idx], degrad);
    if (!(csza <= 1.E-4f)) {                           // RETURN IF NIGHT
        const float frac = diffuse_frac[idx];
        const float s = slope[idx];
        float corr;
        if (s == 0.0f || frac == 1.0f || csza <= 1.e-4f) {
            // no topographic effects when all radiation diffuse or sun
            // too close to horizon
            corr = 1.0f;
            if (shadow == 1) corr = frac;
        } else {
            const float h = hrang[idx];
            const float azi = slp_azi[idx];
            const float sin_s = tr_sin(s);
            const float cos_s = tr_cos(s);
            // cosine of zenith angle over sloping topography
            float slp = __fadd_rn(
                __fmul_rn(__fadd_rn(__fsub_rn(
                    __fmul_rn(__fmul_rn(tr_sin(xxlat), tr_cos(h)),
                              __fmul_rn(-tr_cos(azi), sin_s)),
                    __fmul_rn(tr_sin(h), __fmul_rn(tr_sin(azi), sin_s))),
                    __fmul_rn(__fmul_rn(tr_cos(xxlat), tr_cos(h)), cos_s)),
                    tr_cos(declin)),
                __fmul_rn(__fadd_rn(
                    __fmul_rn(tr_cos(xxlat), __fmul_rn(tr_cos(azi), sin_s)),
                    __fmul_rn(tr_sin(xxlat), cos_s)),
                    tr_sin(declin)));
            if (slp <= 1.E-4f) slp = 0.0f;
            // Topographic shading
            if (shadow == 1) slp = 0.0f;
            // corr_fac = diffuse_frac + (1-diffuse_frac)*csza_slp/csza
            corr = __fadd_rn(frac, __fdiv_rn(__fmul_rn(
                __fsub_rn(1.0f, frac), slp), csza));
        }
        // SWDOWN_teradj=(1.)*SWDOWN_IN*corr_fac
        teradj = __fmul_rn(__fmul_rn(1.0f, sw), corr);
    }
    gswsave[idx] = gsw[idx];                           // save
    // GSW = GSW*SWDOWN_teradj/SWDOWN
    gsw[idx] = __fdiv_rn(__fmul_rn(gsw[idx], teradj), sw);
    swdown[idx] = teradj;
}

// module_surface_driver.F:4461-4481: after the surface schemes, SWDOWN and
// GSW go back to the flat values (the history keeps them) and SWNORM
// carries the slope-affected SWDOWN.
extern "C" __global__ void topo_rad_restore(
    float* __restrict__ swdown, float* __restrict__ gsw,
    float* __restrict__ swnorm, const float* __restrict__ gswsave, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (swnorm[idx] > 1.E-3f) {                        // daytime
        const float save = swdown[idx];
        swdown[idx] = swnorm[idx];
        swnorm[idx] = save;
        gsw[idx] = gswsave[idx];
    }
}
