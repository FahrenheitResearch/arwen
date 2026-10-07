// module_gocart_aerosols.F:6-75,150-264. One thread per column.
extern "C" __global__ void chem_ageing_gocart(
    const unsigned long long* fields, const int* active, const int* to,
    const double* rate, const int* coproduct_to, const float* factor,
    const float* mw, double* work, int nr, int nz, int nc, float dt) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= nc) return;
    int span = nr * nc;
    double* initial = work;
    double* value = work + span;
    double* production = work + 2 * span;
    const float mwdry = 28.966f;
    float seconds = (float)((int)dt);
    for (int k = 0; k < nz - 1; ++k) {
        int idx = k * nc + col;
        for (int r = 0; r < nr; ++r) {
            if (!active[r]) continue;
            int at = r * nc + col;
            float c = ((float*)fields[r])[idx];
            initial[at] = DMUL((double)FMUL(FDIV(c, mw[r]), mwdry), 1.e-9);
            value[at] = initial[at];
            production[at] = 0.0;
        }
        // chem_1: retain the multiply/divide even when the loss rate is zero.
        for (int r = 0; r < nr; ++r) {
            if (to[r] < 0) continue;
            int at = r * nc + col;
            double rkt = DMUL(DADD(rate[r], 0.0), (double)seconds);
            double c1 = fmax(DMUL(initial[at], exp(-rkt)), 1.e-32);
            value[at] = c1;
            production[at] = DDIV(DMUL(DSUB(initial[at], c1), rate[r]),
                                  DADD(rate[r], 0.0));
        }
        // chem_2: hydrophilic floors precede the driver's co-product.
        for (int r = 0; r < nr; ++r) {
            if (to[r] < 0) continue;
            int dest = to[r] * nc + col;
            value[dest] = fmax(DADD(initial[dest], production[r*nc+col]), 1.e-32);
        }
        // Save all target increases before adding any co-products.
        for (int r = 0; r < nr; ++r) {
            if (coproduct_to[r] < 0) continue;
            int target = to[r] * nc + col;
            production[r*nc+col] = DSUB(value[target], initial[target]);
        }
        for (int r = 0; r < nr; ++r) {
            if (coproduct_to[r] < 0) continue;
            int dest = coproduct_to[r] * nc + col;
            double tt2 = production[r*nc+col];
            value[dest] = DADD(value[dest], DMUL((double)factor[r], tt2));
        }
        for (int r = 0; r < nr; ++r) {
            if (!active[r]) continue;
            ((float*)fields[r])[idx] = __double2float_rn(
                DMUL(DMUL(DDIV(value[r*nc+col], (double)mwdry), (double)mw[r]),
                     (double)1.e9f));
        }
    }
}
