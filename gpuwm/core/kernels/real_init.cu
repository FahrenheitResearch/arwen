// Setup-only REAL arithmetic. Explicit rounding preserves NumPy association.
extern "C" __global__ void real_widen(
    const unsigned int* input, double* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int word = input[i];
    unsigned int exponent = (word >> 23) & 255u, mantissa = word & 0x007fffffu;
    unsigned long long bits = (unsigned long long)(word & 0x80000000u) << 32;
    if (exponent == 0u) {
        if (mantissa != 0u) {
            unsigned int leading = 31u - (unsigned int)__clz(mantissa);
            bits |= (unsigned long long)(874u + leading) << 52;
            bits |= (unsigned long long)(mantissa ^ (1u << leading)) << (52u - leading);
        }
    } else if (exponent == 255u) {
        bits |= 0x7ff0000000000000ULL | ((unsigned long long)mantissa << 29);
        if (mantissa != 0u) bits |= 0x0008000000000000ULL;
    } else {
        bits |= (unsigned long long)(exponent + 896u) << 52;
        bits |= (unsigned long long)mantissa << 29;
    }
    output[i] = __longlong_as_double((long long)bits);
}

extern "C" __global__ void real_narrow(
    const double* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = real_float(input[i]);
}

extern "C" __global__ void real_fp32_probe(
    const float* a, const float* b, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = real_fadd(a[i], b[i]);
    output[n + i] = real_fsub(a[i], b[i]);
    output[2 * n + i] = real_fmul(a[i], b[i]);
    output[3 * n + i] = real_fdiv(a[i], b[i]);
}

extern "C" __global__ void real_base_residual(
    const double* phi, float* output, int n, int ncol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float delta = real_fsub(real_float(phi[i + ncol]), real_float(phi[i]));
    output[i] = real_float(real_sub(real_sub(phi[i + ncol], phi[i]), real_double(delta)));
}

extern "C" __global__ void real_specific(
    const double* q, double* out, unsigned int* bad, int n, double lower) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = q[i];
    if (!isfinite(v) || v < lower || v >= 1.0) atomicOr(bad, 1u);
    out[i] = real_div(v, real_sub(1.0, v));
}

extern "C" __global__ void real_cap(
    const double* q, const double* p, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = p[i] < 10000.0 && q[i] > 1.0e-5 ? 3.0e-6 : q[i];
}

extern "C" __global__ void real_stagger(
    const double* p, double* out, int nz, int ny, int nx, int axis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ox = nx + (axis == 2), oy = ny + (axis == 1);
    if (i >= nz * oy * ox) return;
    int x = i % ox, y = (i / ox) % oy, k = i / (ox * oy);
    int a, b;
    if (axis == 2) {
        a = (k * ny + y) * nx + (x == 0 ? 0 : x - 1);
        b = (k * ny + y) * nx + (x == nx ? nx - 1 : x);
    } else {
        a = (k * ny + (y == 0 ? 0 : y - 1)) * nx + x;
        b = (k * ny + (y == ny ? ny - 1 : y)) * nx + x;
    }
    bool edge = axis == 2 ? (x == 0 || x == nx) : (y == 0 || y == ny);
    out[i] = edge ? p[a] : real_mul(0.5, real_add(p[a], p[b]));
}

extern "C" __global__ void real_dry_ladder(
    const double* mu, const double* c3, const double* c4, double* out,
    int ncol, int nz, double ptop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol * nz) return;
    int k = i / ncol;
    out[i] = real_add(real_add(real_mul(c3[k], mu[i % ncol]), c4[k]), ptop);
}

__device__ double real_moist_increment(double q, double rho, double dz, double g) {
    return real_mul(real_div(real_mul(real_mul(g, q), rho),
                             real_add(1.0, q)), dz);
}

extern "C" __global__ void real_integrate(
    const double* q, const double* p, const double* t, const double* z,
    const double* ps, const double* ts, const double* qs, const double* zs,
    const int* order, double* pd, double* intq, unsigned int* missing,
    int ncol, int nz, double rd, double g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    int ka = nz;
    for (int k = 0; k < nz; ++k) {
        if (p[order[k] * ncol + i] < ps[i]) { ka = k; break; }
    }
    if (ka == nz) { atomicMin(missing, (unsigned int)i); return; }
    double running = 0.0, surface = 0.0;
    pd[(nz - 1) * ncol + i] = p[order[nz - 1] * ncol + i];
    for (int k = nz - 2; k >= 0; --k) {
        int a = order[k] * ncol + i, b = order[k + 1] * ncol + i;
        double rho = real_mul(0.5, real_add(
            real_div(p[a], real_mul(rd, t[a])),
            real_div(p[b], real_mul(rd, t[b]))));
        double qb = real_mul(0.5, real_add(q[a], q[b]));
        double dz = real_sub(z[b], z[a]);
        double inc = real_moist_increment(qb, rho, dz, g);
        // The vectorized oracle adds +0 even where dz is non-positive.
        running = ka <= k ? real_add(running, dz > 0.0 ? inc : 0.0) : 0.0;
        pd[k * ncol + i] = real_sub(p[a], running);
        if (ka == k) surface = running;
    }
    int a = order[ka] * ncol + i;
    double rho = real_mul(0.5, real_add(
        real_div(ps[i], real_mul(rd, ts[i])),
        real_div(p[a], real_mul(rd, t[a]))));
    double qb = real_mul(0.5, real_add(qs[i], q[a]));
    double dz = real_sub(z[a], zs[i]);
    if (dz > 0.1) surface = real_add(surface, real_moist_increment(qb, rho, dz, g));
    for (int k = 0; k < ka; ++k)
        pd[k * ncol + i] = real_sub(p[order[k] * ncol + i], surface);
    intq[i] = surface;
}

extern "C" __global__ void real_rebalance(
    const double* q, const double* mu, const double* mub, const double* pb,
    const double* c1f, const double* c2f, const double* rdnw, const double* rdn,
    double* out, unsigned int* bad, int ncol, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    double mup = real_sub(mu[i], mub[i]);
    double qt = q[(nz - 1) * ncol + i];
    double cq = real_div(1.0, real_add(1.0, qt));
    double load = real_mul(qt, cq);
    double row = real_add(real_mul(c1f[nz], mup), real_mul(load,
        real_add(real_mul(c1f[nz], mub[i]), c2f[nz])));
    double running = real_div(real_div(real_mul(-0.5, row), rdnw[nz - 1]), cq);
    double v = real_add(pb[(nz - 1) * ncol + i], running);
    out[(nz - 1) * ncol + i] = v;
    if (!isfinite(v) || v <= 0.0) atomicOr(bad, 1u);
    for (int k = nz - 2; k >= 0; --k) {
        int kw = k + 1;
        qt = real_mul(0.5, real_add(q[k * ncol + i], q[kw * ncol + i]));
        cq = real_div(1.0, real_add(1.0, qt));
        load = real_mul(qt, cq);
        row = real_add(real_mul(c1f[kw], mup), real_mul(load,
            real_add(real_mul(c1f[kw], mub[i]), c2f[kw])));
        running = real_sub(running, real_div(real_div(row, cq), rdn[kw]));
        v = real_add(pb[k * ncol + i], running);
        out[k * ncol + i] = v;
        if (!isfinite(v) || v <= 0.0) atomicOr(bad, 1u);
    }
}

extern "C" __global__ void real_split_opt1(
    const double* phb64, const double* mu, const double* alpha,
    const double* c1h, const double* c2h, const float* dnw, const float* rdnw,
    float* php, int ncol, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    float prev = 0.0f;
    php[i] = prev;
    for (int k = 0; k < nz; ++k) {
        int a = k * ncol + i, b = a + ncol;
        float dbase = real_fsub(real_float(phb64[b]), real_float(phb64[a]));
        float resid = real_float(real_sub(real_sub(phb64[b], phb64[a]), real_double(dbase)));
        float dphb = real_fadd(dbase, resid);
        float inc = real_float(real_add(real_mul(c1h[k], mu[i]), c2h[k]));
        float target = real_float(alpha[a]);
        float desired = real_fmul(real_fmul(-dnw[k], inc), target);
        float centre = real_fadd(prev, real_fsub(desired, dphb));
        float candidates[3] = {
            real_nextafter32(centre, __int_as_float(0xff800000)), centre,
            real_nextafter32(centre, __int_as_float(0x7f800000))};
        float best = candidates[0];
        double best_error = 0.0;
        for (int j = 0; j < 3; ++j) {
            float delta = real_fadd(dphb, real_fsub(candidates[j], prev));
            float diagnosed = real_fdiv(real_fmul(-delta, rdnw[k]), inc);
            double error = fabs(real_sub(real_double(diagnosed), real_double(target)));
            if (j == 0 || error < best_error) { best = candidates[j]; best_error = error; }
        }
        php[b] = best;
        prev = best;
    }
}
