// WRF v4.8.0 module_sf_urban.F:5344-5396, ROOT_UPTAKE.
// Uses the engine's explicitly rounded binary32 arithmetic operations.
// This helper is not wired to a forecast generation yet.
#ifndef GPUWM_URBAN_ROOT_UPTAKE_480
#define GPUWM_URBAN_ROOT_UPTAKE_480
template<int NLAYER>
__device__ void u_root_uptake_480(float* sroot, const float* smc,
    float smcmax, float smcdry, float rtree, float lai_tree,
    const float* dzg, const float* zsoil, float ele, float fvg, float rw)
{
    const int nlayer = NLAYER;
    float normq[NLAYER], alphai[NLAYER], root[NLAYER], part2[NLAYER];
    if (fvg <= 0.0f || rw <= 0.0f || rtree <= 0.0f || lai_tree <= 0.0f) {
        for (int k = 0; k < nlayer; ++k) sroot[k] = 0.0f;
        return;
    }
    // The model instantiates four layers. Keep the release's separate loops
    // and each expression's operation order for every instantiation.
    for (int k = 0; k < nlayer; ++k) {
        normq[k] = FDIV(FSUB(smc[k], smcdry), FSUB(smcmax, smcdry));
        if (normq[k] > 1.0f) alphai[k] = 1.0f;
        else if (normq[k] >= 0.6f && normq[k] <= 1.0f) alphai[k] = 1.0f;
        else if (normq[k] < 0.6f && normq[k] >= 0.0f) alphai[k] = FDIV(normq[k], 0.6f);
        else alphai[k] = 0.0f;
    }
    for (int k = 0; k < nlayer; ++k) {
        root[k] = FDIV(dzg[k], -zsoil[nlayer - 1]);
    }
    float alphabar = 0.0f;
    for (int k = 0; k < nlayer; ++k) {
        alphabar = FADD(alphabar, FMUL(root[k], alphai[k]));
    }
    if (alphabar <= 1.0e-12f) {
        for (int k = 0; k < nlayer; ++k) sroot[k] = 0.0f;
        return;
    }
    for (int k = 0; k < nlayer; ++k) {
        // Preserve the source's left-to-right product, including ELE/Lv.
        float part1 = FDIV(FMUL(FMUL(FMUL(2.0f, rtree), lai_tree), FDIV(ele, 2.26e6f)), 1.0e3f);
        part2[k] = FDIV(FMUL(root[k], alphai[k]), alphabar);
        sroot[k] = FDIV(FMUL(part1, part2[k]), FMUL(fvg, rw));
    }
}
#endif
