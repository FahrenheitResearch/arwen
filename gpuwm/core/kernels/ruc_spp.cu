// WRF V3.9.1 module_sf_ruclsm.F:6358-6365 hydraulic parameter operator.
// WRF-derived code is covered by NOTICE and the shipped WRF licence notice.
// Separate from ruc.cu so disabled RUC retains its original translation unit.
extern "C" __global__ void ruc_spp_hydraulic(
    float* hydro, const float* pattern_spp_lsm, float* field_sf, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float original = hydro[i];
    const float pattern = pattern_spp_lsm[i];
    if (field_sf != nullptr) field_sf[i] = __fmul_rn(original, pattern);
    hydro[i] = __fmul_rn(original, __fadd_rn(1.0f, pattern));
}
