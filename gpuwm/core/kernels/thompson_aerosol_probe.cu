// gpuwm/core/kernels/thompson_aerosol_probe.cu
//
// Thin elementwise __global__ wrappers around every __device__ helper in
// thompson_aerosol_common.cuh, so each one can be gated POINTWISE against the
// Fortran probe tables in gpuwm/data/thompson/oracle-aero/probe-*.csv before
// any aerosol network kernel exists.
//
// This translation unit contains NO physics of its own.  Every kernel is a
// bounds check plus one helper call.  If a probe result disagrees with the
// oracle, the defect is in thompson_aerosol_common.cuh, never here.
//
// The shared header is prepended by gpuwm/core/kernels/__init__.py's
// _EXTRA_HEADERS allow-list; there is no #include of it below.

extern "C" __global__ void thompson_aa_probe_nint(
    const float* __restrict__ x, int* __restrict__ out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_nint(x[idx]);
}

extern "C" __global__ void thompson_aa_probe_nu_c(
    const float* __restrict__ nc_m3, int* __restrict__ out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_nu_c(nc_m3[idx]);
}

extern "C" __global__ void thompson_aa_probe_droplet_bin(
    const float* __restrict__ nc_m3, int* __restrict__ out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_droplet_bin(nc_m3[idx]);
}

extern "C" __global__ void thompson_aa_probe_decade_index(
    const float* __restrict__ value, int first_exponent, int table_size,
    int* __restrict__ out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_decade_index(
        value[idx], first_exponent, table_size);
}

// The DOUBLE form, promoted out of cold.cu:206-223 / warm.cu:272-289.  Both
// of its production call sites live in translation units this package does not
// own -- the rain y-intercept at (6, 37) and the graupel one at (2, 37) -- so
// without this wrapper the promoted body has no pointwise gate at all.
extern "C" __global__ void thompson_aa_probe_decade_index_double(
    const double* __restrict__ value, int first_exponent, int table_size,
    int* __restrict__ out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_decade_index_double(
        value[idx], first_exponent, table_size);
}

extern "C" __global__ void thompson_aa_probe_in_bin(
    const float* __restrict__ xni, int* __restrict__ out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_in_bin(xni[idx]);
}

extern "C" __global__ void thompson_aa_probe_inu_c_effrad(
    const float* __restrict__ nc_m3, int* __restrict__ out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_inu_c_effrad(nc_m3[idx]);
}

extern "C" __global__ void thompson_aa_probe_clamps(
    const float* __restrict__ nc_m3,
    const float* __restrict__ nwfa_m3,
    const float* __restrict__ nifa_m3,
    float* __restrict__ nc_out,
    float* __restrict__ nwfa_out,
    float* __restrict__ nifa_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    nc_out[idx] = thompson_aa_clamp_nc(nc_m3[idx]);
    nwfa_out[idx] = thompson_aa_clamp_nwfa(nwfa_m3[idx]);
    nifa_out[idx] = thompson_aa_clamp_nifa(nifa_m3[idx]);
}

extern "C" __global__ void thompson_aa_probe_saturation(
    const float* __restrict__ pressure,
    const float* __restrict__ temperature,
    float* __restrict__ rslf_out,
    float* __restrict__ rsif_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    rslf_out[idx] = thompson_rslf(pressure[idx], temperature[idx]);
    rsif_out[idx] = thompson_rsif(pressure[idx], temperature[idx]);
}

extern "C" __global__ void thompson_aa_probe_field_ab(
    const float* __restrict__ tc,
    const float* __restrict__ moment,
    float* __restrict__ a_out,
    float* __restrict__ b_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    a_out[idx] = thompson_field_a(tc[idx], moment[idx]);
    b_out[idx] = thompson_field_b(tc[idx], moment[idx]);
}

extern "C" __global__ void thompson_aa_probe_activ_ncloud(
    const float* __restrict__ temperature,
    const float* __restrict__ w,
    const float* __restrict__ nccn,
    const double* __restrict__ tnccn_act,
    float* __restrict__ out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_activ_ncloud(
        temperature[idx], w[idx], nccn[idx], tnccn_act);
}

extern "C" __global__ void thompson_aa_probe_ice_demott(
    const float* __restrict__ tempc,
    const float* __restrict__ rho,
    const float* __restrict__ nifa_m3,
    float* __restrict__ out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_ice_demott(tempc[idx], rho[idx], nifa_m3[idx]);
}

extern "C" __global__ void thompson_aa_probe_ice_koop(
    const float* __restrict__ temperature,
    const float* __restrict__ qv,
    const float* __restrict__ qvs,
    const float* __restrict__ naero,
    const float* __restrict__ dt,
    float* __restrict__ out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_ice_koop(
        temperature[idx], qv[idx], qvs[idx], naero[idx], dt[idx]);
}

extern "C" __global__ void thompson_aa_probe_eff_aero(
    const float* __restrict__ d_collector,
    const float* __restrict__ d_aerosol,
    const float* __restrict__ visc,
    const float* __restrict__ rhoa,
    const float* __restrict__ temperature,
    const int* __restrict__ species,
    float* __restrict__ out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_eff_aero(
        d_collector[idx], d_aerosol[idx], visc[idx], rhoa[idx],
        temperature[idx], species[idx]);
}

extern "C" __global__ void thompson_aa_probe_snow_number(
    const float* __restrict__ smob,
    const float* __restrict__ smoc,
    float* __restrict__ out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = thompson_aa_snow_number(smob[idx], smoc[idx]);
}

extern "C" __global__ void thompson_aa_probe_cloud_dist(
    const float* __restrict__ rc,
    const float* __restrict__ nc_per_kg,
    const float* __restrict__ rho,
    float* __restrict__ nc_out,
    int* __restrict__ nu_c_out,
    double* __restrict__ lamc_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    int nu_c = 0;
    double lamc = 0.0;
    nc_out[idx] = thompson_aa_cloud_dist(
        rc[idx], nc_per_kg[idx], rho[idx], &nu_c, &lamc);
    nu_c_out[idx] = nu_c;
    lamc_out[idx] = lamc;
}

// ---------------------------------------------------------------------------
// THE nu_c STAGING PROBE.
// ---------------------------------------------------------------------------
//
// The single most dangerous silent defect this port can carry: WRF computes
// nu_c at :1832 from the PRE-rediagnosis nc and AGAIN at :2170 from the
// POST-rediagnosis nc(k) assigned at :1840, and a kernel that reuses the
// first one stays finite, stays stable and is grossly wrong wherever the
// :1834-1838 droplet-size clamp engages.
//
// This kernel emits BOTH so a host test can assert the WRF-correct staging
// pointwise, without a network kernel and without a Fortran oracle: the
// assertion is a structural property of module_mp_thompson.F's control flow,
// not a numerical comparison.  nu_c_working_out is what every rate in the
// warm and cold networks must consume.
extern "C" __global__ void thompson_aa_probe_nu_c_staging(
    const float* __restrict__ rc,
    const float* __restrict__ nc_per_kg,
    const float* __restrict__ rho,
    float* __restrict__ nc_rediagnosed_out,
    int* __restrict__ nu_c_entry_out,
    int* __restrict__ nu_c_working_out,
    double* __restrict__ lamc_entry_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    int nu_c_entry = 0;
    double lamc_entry = 0.0;
    // :1826-1842.
    const float nc_m3 = thompson_aa_cloud_dist(
        rc[idx], nc_per_kg[idx], rho[idx], &nu_c_entry, &lamc_entry);
    nc_rediagnosed_out[idx] = nc_m3;
    nu_c_entry_out[idx] = nu_c_entry;
    // :2170, from the POST-rediagnosis nc.
    nu_c_working_out[idx] = thompson_aa_nu_c_working(nc_m3);
    lamc_entry_out[idx] = lamc_entry;
}

// The gamma-moment columns the two staging answers select.  ccg(2,nu_c) and
// ocg1(nu_c) move by more than ten orders of magnitude between nu_c = 3 and
// nu_c = 15, and the products lamc (:2173) and Dc_g (:2181) are built from
// move by 40.8x and 15.8x, which is what turns a wrong nu_c into a wrong
// droplet spectrum rather than a rounding difference.
extern "C" __global__ void thompson_aa_probe_gamma_columns(
    const int* __restrict__ nu_c,
    float* __restrict__ ccg2_out,
    float* __restrict__ ocg1_out,
    float* __restrict__ ccg3_out,
    float* __restrict__ ocg2_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    const int column = nu_c[idx];
    ccg2_out[idx] = THOMPSON_AA_CCG2[column];
    ocg1_out[idx] = THOMPSON_AA_OCG1[column];
    ccg3_out[idx] = THOMPSON_AA_CCG3[column];
    ocg2_out[idx] = THOMPSON_AA_OCG2[column];
}


// ---------------------------------------------------------------------------
// The three helpers consolidated out of cold.cu / warm.cu / sed.cu.
// ---------------------------------------------------------------------------

extern "C" __global__ void thompson_aa_probe_entry_rain_distribution(
    const float* __restrict__ qr_per_kg,
    const float* __restrict__ nr_per_kg,
    const float* __restrict__ rho,
    float* __restrict__ nr_m3_out,
    double* __restrict__ lamr_out,
    float* __restrict__ mvd_r_out,
    double* __restrict__ n0_r_out,
    int* __restrict__ l_qr_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    float nr = 0.0f;
    double lamr = 0.0;
    float mvd = 0.0f;
    double n0 = 0.0;
    const bool active = thompson_aa_entry_rain_distribution(
        qr_per_kg[idx], nr_per_kg[idx], rho[idx], &nr, &lamr, &mvd, &n0);
    nr_m3_out[idx] = nr;
    lamr_out[idx] = lamr;
    mvd_r_out[idx] = mvd;
    n0_r_out[idx] = n0;
    l_qr_out[idx] = active ? 1 : 0;
}

extern "C" __global__ void thompson_aa_probe_bound_rain_number(
    const float* __restrict__ rain_mass,
    const float* __restrict__ density,
    const float* __restrict__ nr_per_kg_in,
    float* __restrict__ nr_per_kg_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    float nr = nr_per_kg_in[idx];
    thompson_aa_bound_rain_number(rain_mass[idx], density[idx], &nr);
    nr_per_kg_out[idx] = nr;
}

extern "C" __global__ void thompson_aa_probe_bound_ice_number(
    const float* __restrict__ ice_mass,
    const float* __restrict__ density,
    const float* __restrict__ ni_per_kg_in,
    float* __restrict__ ni_per_kg_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    float ni = ni_per_kg_in[idx];
    thompson_aa_bound_ice_number(ice_mass[idx], density[idx], &ni);
    ni_per_kg_out[idx] = ni;
}

// The two mass coefficients the consolidated bounds now read by NAME where
// the deleted per-network copies spelled out a product.  Emitting both forms
// from the same translation unit is what makes "bit-identical" a measured
// fact rather than an assertion in a comment.
//   0 THOMPSON_AA_AM_R   1 3.1415926536f*1000.0f/6.0f
//   2 THOMPSON_AA_AM_I   3 3.1415926536f*890.0f/6.0f
#define THOMPSON_AA_PROBE_MASS_COEFFICIENTS 4

extern "C" __global__ void thompson_aa_probe_mass_coefficients(
    float* __restrict__ out)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0) return;
    out[0] = THOMPSON_AA_AM_R;
    out[1] = 3.1415926536f * 1000.0f / 6.0f;
    out[2] = THOMPSON_AA_AM_I;
    out[3] = 3.1415926536f * 890.0f / 6.0f;
}


// calc_effectRad, module_mp_thompson.F:5594-5699, one column entry per
// thread.  WRF's has_qc/has_qi/has_qs flags are column-wide but only gate
// loops whose bodies re-test the same per-level condition, so the elementwise
// form is exactly equivalent.  Outputs are METRES, carrying calc_effectRad's
// own clamps and background values but NOT mp_gt_driver's second clamp.
extern "C" __global__ void thompson_aa_probe_effect_rad(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc_per_kg,
    const float* __restrict__ qi,
    const float* __restrict__ ni_per_kg,
    const float* __restrict__ qs,
    float* __restrict__ effc,
    float* __restrict__ effi,
    float* __restrict__ effs,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho = 0.622f * pressure[idx]
        / (THOMPSON_AA_R_DRY * temperature[idx] * (qv[idx] + 0.622f));
    const float rc = fmaxf(THOMPSON_AA_R1, qc[idx] * rho);
    const float nc = thompson_aa_clamp_nc(nc_per_kg[idx] * rho);
    const float ri = fmaxf(THOMPSON_AA_R1, qi[idx] * rho);
    const float ni = fmaxf(THOMPSON_AA_R2, ni_per_kg[idx] * rho);
    const float rs = fmaxf(THOMPSON_AA_R1, qs[idx] * rho);

    float reqc = THOMPSON_AA_RE_QC_BG;
    float reqi = THOMPSON_AA_RE_QI_BG;
    float reqs = THOMPSON_AA_RE_QS_BG;

    if (rc > THOMPSON_AA_R1 && nc > THOMPSON_AA_R2) {
        reqc = thompson_aa_eff_rad_cloud(rc, nc);
    }
    if (ri > THOMPSON_AA_R1 && ni > THOMPSON_AA_R2) {
        reqi = thompson_aa_eff_rad_ice(ri, ni);
    }
    if (rs > THOMPSON_AA_R1) {
        reqs = thompson_aa_eff_rad_snow(rs, temperature[idx]);
    }

    effc[idx] = reqc;
    effi[idx] = reqi;
    effs[idx] = reqs;
}

// Read the __constant__ gamma-moment tables back so a host test can prove
// they are bit-identical to gpuwm.core.thompson_aerosol_contract's arrays.
// Row order is fixed and must not change: it is part of the published API.
//   0..4   cce1..cce5     5..9   ccg1..ccg5
//   10     ocg1           11     ocg2          12  g_ratio
#define THOMPSON_AA_PROBE_TABLE_ROWS 13
#define THOMPSON_AA_PROBE_TABLE_COLS 16

extern "C" __global__ void thompson_aa_probe_constant_tables(
    float* __restrict__ out)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= THOMPSON_AA_PROBE_TABLE_COLS) return;
    out[0 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCE1[idx];
    out[1 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCE2[idx];
    out[2 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCE3[idx];
    out[3 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCE4[idx];
    out[4 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCE5[idx];
    out[5 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCG1[idx];
    out[6 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCG2[idx];
    out[7 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCG3[idx];
    out[8 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCG4[idx];
    out[9 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_CCG5[idx];
    out[10 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_OCG1[idx];
    out[11 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_OCG2[idx];
    out[12 * THOMPSON_AA_PROBE_TABLE_COLS + idx] = THOMPSON_AA_G_RATIO[idx];
}
