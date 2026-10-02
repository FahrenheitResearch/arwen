// portable_libm64.cuh is supplied by the library lane through EXTRA_HEADERS.
extern "C" __global__ void real_thermo(
    const double* t, const double* p, const double* q, double* out,
    unsigned int* bad, int n, int operation, double p0, double rcp,
    double rd, double rvovrd, double svp1hpa, double svp2, double svpt0,
    double svp3, double hundred, double minimum_qv, double xlv_over_rv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double ti = t[i], pi = p[i], qi = q[i], value;
    if (operation == 0) {
        value = real_mul(ti, plm_pow(real_div(p0, pi), rcp));
    } else if (operation == 1) {
        value = real_mul(ti, plm_pow(real_div(pi, p0), rcp));
    } else if (operation == 2) {
        double tm = real_mul(ti, real_add(1.0, real_mul(rvovrd, qi)));
        value = real_div(real_mul(real_mul(rd, tm),
                                  plm_pow(real_div(pi, p0), rcp)), pi);
    } else if (operation == 3 || operation == 4) {
        double expo = real_div(real_mul(svp2, real_sub(ti, svpt0)), real_sub(ti, svp3));
        double e = plm_exp(expo);
        double phpa = real_div(pi, hundred);
        if (operation == 3) {
            // NumPy clip propagates NaN and retains a signed-zero input.
            double rh = isnan(qi) ? qi : (qi < 0.0 ? 0.0 : (qi > hundred ? hundred : qi));
            double es = real_mul(real_mul(real_mul(rh, 0.01), svp1hpa), e);
            double candidate = real_div(real_mul(0.622, es), real_sub(phpa, es));
            double floored = isnan(candidate) ? candidate : (candidate > 1.0e-6 ? candidate : 1.0e-6);
            value = ti != 0.0 && isfinite(es) && es < phpa ? floored : 1.0e-6;
        } else {
            if (!isfinite(ti) || !isfinite(pi) || !isfinite(qi) || pi <= 0.0 || qi < minimum_qv)
                atomicOr(bad, 1u);
            double es = real_mul(svp1hpa, e);
            double vapor = real_div(real_mul(qi, phpa), real_add(qi, 0.622));
            value = real_div(real_mul(hundred, vapor), es);
            if (!isfinite(value)) atomicOr(bad, 2u);
        }
    } else {
        value = real_mul(hundred, plm_exp(real_mul(xlv_over_rv,
            real_sub(real_div(1.0, pi), real_div(1.0, ti)))));
    }
    out[i] = value;
}

extern "C" __global__ void real_surface_pressure(
    const double* ps, const double* zs, const double* z, const double* t,
    const double* q, double* out, unsigned int* bad, int n, double rd, double g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double tv = real_mul(t[i], real_add(1.0, real_mul(0.608, q[i])));
    if (!isfinite(tv) || tv <= 0.0) atomicOr(bad, 1u);
    double v = real_mul(ps[i], plm_exp(real_div(
        real_mul(g, real_sub(zs[i], z[i])), real_mul(rd, tv))));
    if (!isfinite(v) || v <= 0.0) atomicOr(bad, 2u);
    out[i] = v;
}

extern "C" __global__ void real_split_opt2(
    const double* phb64, const double* mu, const double* alpha,
    const float* c3f, const float* c4f, const float* c3h, const float* c4h,
    const float* dc3f, const float* dc4f, float* php,
    int ncol, int nz, float ptop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    float prev = 0.0f, mu32 = real_float(mu[i]);
    php[i] = prev;
    for (int k = 0; k < nz; ++k) {
        int a = k * ncol + i, b = a + ncol;
        float db = real_fsub(real_float(phb64[b]), real_float(phb64[a]));
        float resid = real_float(real_sub(real_sub(phb64[b], phb64[a]), real_double(db)));
        float dphb = real_fadd(db, resid);
        float target = real_float(alpha[a]);
        float pfu = real_fadd(real_fadd(real_fmul(c3f[k + 1], mu32), c4f[k + 1]), ptop);
        float phm = real_fadd(real_fadd(real_fmul(c3h[k], mu32), c4h[k]), ptop);
        float dpf = real_fadd(real_fmul(dc3f[k], mu32), dc4f[k]);
        float ratio = plm_log1pf(real_fdiv(dpf, pfu));
        float desired = real_fmul(real_fmul(target, phm), ratio);
        float centre = real_fadd(prev, real_fsub(desired, dphb));
        float candidates[3] = {
            real_nextafter32(centre, __int_as_float(0xff800000)), centre,
            real_nextafter32(centre, __int_as_float(0x7f800000))};
        float best = candidates[0];
        double best_error = 0.0;
        for (int j = 0; j < 3; ++j) {
            float delta = real_fadd(dphb, real_fsub(candidates[j], prev));
            float diagnosed = real_fdiv(real_fdiv(delta, phm), ratio);
            double error = fabs(real_sub(real_double(diagnosed), real_double(target)));
            if (j == 0 || error < best_error) { best = candidates[j]; best_error = error; }
        }
        php[b] = best;
        prev = best;
    }
}
