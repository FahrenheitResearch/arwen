// Sun-angle albedo from NOAA-EMC/HRRR v4.1.21,
// module_radiation_driver.F:781-789,1038-1063.
// Each REAL operation is rounded independently, matching the Fortran
// oracle built without contraction.  In particular, division uses
// __fdiv_rn so constant reciprocal rewriting cannot change a word.
extern "C" __global__ void solar_albedo_update(
    long long n, int initialize,
    const float* __restrict__ albedo,
    const float* __restrict__ albbck,
    const float* __restrict__ xland,
    const float* __restrict__ snow,
    const float* __restrict__ xice,
    const int* __restrict__ ivgtyp,
    float* __restrict__ albsol,
    float* __restrict__ albbcksol,
    const float* __restrict__ coszen)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (initialize) {
        albsol[i] = albedo[i];
        albbcksol[i] = albbck[i];
    }
    if (coszen[i] > 0.0f && xland[i] < 1.5f
        && snow[i] == 0.0f && xice[i] == 0.0f) {
        // istwe = 1 for 1-5, 8, 9, 15, 18; 2 for the other MODIS classes.
        int category = ivgtyp[i];
        float d = ((category >= 1 && category <= 5) || category == 8
                   || category == 9 || category == 15 || category == 18)
                  ? 0.1f : 0.25f;
        float twice_d = __fmul_rn(2.0f, d);
        float dm = __fdiv_rn(__fadd_rn(1.0f, twice_d),
                            __fadd_rn(1.0f, __fmul_rn(twice_d, coszen[i])));
        float normalized = __fmul_rn(albbck[i], dm);
        albsol[i] = normalized;
        albbcksol[i] = normalized;
    }
    albsol[i] = fminf(albsol[i], 0.9f);
}
