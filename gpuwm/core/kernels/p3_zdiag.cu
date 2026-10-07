// P3 reflectivity as a PURE function of the state: the radar observation
// operator H_Z(x) for mp_physics=50/51.
//
// Compiled AFTER p3.cu in one translation unit (gpuwm/core/p3_device.py
// p3_reflectivity_module), so every helper below is the forecast port's own
// __device__ function, unchanged: p3_get_rain_dsd2, p3_impose_max_ni,
// p3_calc_bulk_rho_rime, p3_find_lt_1a and p3_access_lookup_table.  This
// file adds no Z formula of its own.
//
// What it is.  The Z half of p3_final_level (p3.cu, WRF
// module_mp_p3.F:4722-4895), replayed on LOCAL copies.  The forecast step
// evaluates that loop and updates the state in the same statements: it
// dumps sub-QSMALL rain and ice into vapour and theta, re-derives nr from
// the rain DSD limiter, imposes the total-Ni cap and clamps ni onto the
// lookup table's lambda limiters before ze_ice reads it.  Here every one of
// those values is formed in a register and nothing is written but zdbz, so
// evaluating the operator does not move the background it observes.
//
// Statement order is p3_final_level's, so on a state that has just left a
// P3 step (where those clamps are already applied) the result is the Z the
// step wrote, with the density formed from the state the operator is given
// (the step forms it from its entry temperature, :2322).
//
// Clear air.  A cell with neither rain nor ice above QSMALL keeps the two
// accumulator seeds, ze_rain = ze_ice = 1e-22 mm6 m-3 (:2286-2287), and
// reads 10*log10(2e-22*1e18) = -36.9897 dBZ.  That is the scheme's own
// zero-hydrometeor reflectivity, and it is the operator's ONE clear-air
// value in every cell.  The forecast field additionally holds the -99.0
// entry initialisation (:2278) through columns its no-hydrometeor skip
// (:3972) never diagnoses; that sentinel is not a reflectivity and the
// operator does not reproduce it, which is what gives P3 one floor.

extern "C" __global__ void p3k_reflectivity(
    const float* __restrict__ qr, const float* __restrict__ nr,
    const float* __restrict__ qi, const float* __restrict__ qir,
    const float* __restrict__ ni, const float* __restrict__ qib,
    const float* __restrict__ th, const float* __restrict__ pres,
    const float* __restrict__ itab, float* __restrict__ zdbz, int n)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    // Atmospheric variables as the step forms them (:2293-2295, :2322-2323).
    const float tm = r_pow(pres[c] * 1.0e-5f, P3_RD * P3_INV_CP);
    const float t = th[c] * tm;
    const float rho = pres[c] / (P3_RD * t);
    const float inv_rho = 1.0f / rho;
    float ze_ice = 1.0e-22f, ze_rain = 1.0e-22f;                     // :2286-2287

    const float q_r = qr[c];
    if (q_r >= P3_QSMALL) {
        float mu_r, lamr, cdistr, logn0r, nrg = nr[c];
        // the authority passes the LITERAL iSPF = 1. here (:4739)
        p3_get_rain_dsd2(&nrg, q_r, 1.0f, &mu_r, &lamr, &cdistr, &logn0r);
        float l2 = lamr * lamr;
        float lam6 = (l2 * l2) * l2;                     // :4756, integer **6
        ze_rain = rho * nrg * (mu_r + 6.0f) * (mu_r + 5.0f)
                  * (mu_r + 4.0f) * (mu_r + 3.0f) * (mu_r + 2.0f)
                  * (mu_r + 1.0f) / lam6;
        ze_rain = fmaxf(ze_rain, 1.0e-22f);
    }

    float n_i = p3_impose_max_ni(ni[c], inv_rho);
    const float q_i = qi[c];
    if (q_i >= P3_QSMALL) {
        n_i = fmaxf(n_i, P3_NSMALL);
        float rhop, qq = qir[c], bb = qib[c];
        p3_calc_bulk_rho_rime(q_i, &qq, &bb, &rhop);
        int dumi, dumjj, dumii;
        float d1, d4, d5;
        p3_find_lt_1a(q_i, n_i, qq, rhop, &dumi, &dumjj, &dumii,
                      &d1, &d4, &d5);
        float f1pr09 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                              d1, d4, d5);
        float f1pr10 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                              d1, d4, d5);
        float f1pr13 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 9,
                                              d1, d4, d5);
        n_i = fminf(n_i, f1pr09 * q_i);
        n_i = fmaxf(n_i, f1pr10 * q_i);
        ze_ice = ze_ice + 0.1892f * f1pr13 * n_i * rho;
        ze_ice = fmaxf(ze_ice, 1.0e-22f);
    }

    zdbz[c] = 10.0f * p3_log10((ze_rain + ze_ice) * 1.0e18f);
}
