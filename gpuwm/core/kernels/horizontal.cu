// Rust horizontal_point arithmetic, rounded after each binary32 operation.
// Decode subnormals before binary64 evaluation so device FTZ cannot change bits.
__device__ __forceinline__ double hz_double(float x) {
    unsigned u = __float_as_uint(x), a = u & 0x7fffffffu;
    if (a < 0x00800000u) {
        double d = __dmul_rn((double)a, 0x1p-149);
        return u >> 31 ? -d : d;
    }
    return (double)x;
}
__device__ __forceinline__ float hz_round(double x) {
    double a = fabs(x);
    if (a > 0.0 && a < 0x1p-126) {
        unsigned m = (unsigned)rint(__dmul_rn(a, 0x1p149));
        unsigned s = __double_as_longlong(x) < 0LL ? 0x80000000u : 0u;
        return __uint_as_float(s | m);
    }
    return __double2float_rn(x);
}
__device__ __forceinline__ float hz_add(float a, float b) {
    unsigned ua = __float_as_uint(a) & 0x7fffffffu;
    unsigned ub = __float_as_uint(b) & 0x7fffffffu;
    if (ua >= 0x00800000u && ub >= 0x00800000u) {
        float r = __fadd_rn(a, b);
        if ((__float_as_uint(r) & 0x7fffffffu) >= 0x00800000u) return r;
    }
    return hz_round(__dadd_rn(hz_double(a), hz_double(b)));
}
__device__ __forceinline__ float hz_sub(float a, float b) {
    unsigned ua = __float_as_uint(a) & 0x7fffffffu;
    unsigned ub = __float_as_uint(b) & 0x7fffffffu;
    if (ua >= 0x00800000u && ub >= 0x00800000u) {
        float r = __fsub_rn(a, b);
        if ((__float_as_uint(r) & 0x7fffffffu) >= 0x00800000u) return r;
    }
    return hz_round(__dsub_rn(hz_double(a), hz_double(b)));
}
__device__ __forceinline__ float hz_mul(float a, float b) {
    unsigned ua = __float_as_uint(a) & 0x7fffffffu;
    unsigned ub = __float_as_uint(b) & 0x7fffffffu;
    if (ua >= 0x00800000u && ub >= 0x00800000u) {
        float r = __fmul_rn(a, b);
        if ((__float_as_uint(r) & 0x7fffffffu) >= 0x00800000u) return r;
    }
    return hz_round(__dmul_rn(hz_double(a), hz_double(b)));
}
__device__ __forceinline__ bool hz_zero(float a) {
    return (__float_as_uint(a) & 0x7fffffffu) == 0;
}
__device__ __forceinline__ bool hz_normal(float a) {
    unsigned e = __float_as_uint(a) & 0x7f800000u;
    return e != 0 && e != 0x7f800000u;
}
__device__ __forceinline__ float hz_oned(float x, float a, float b, float c, float d, bool host) {
    float p = hz_mul(b, c);
    unsigned ap = __float_as_uint(p) & 0x7fffffffu;
    bool product = ap >= 0x00800000u && ap <= 0x7f800000u;
    bool both = product && (host || (hz_normal(b) && hz_normal(c)));
    float q = hz_sub(1.0f, x);
    float left = hz_add(b, hz_mul(x, hz_add(hz_mul(0.5f, hz_sub(c, a)),
        hz_mul(x, hz_sub(hz_mul(0.5f, hz_add(c, a)), b)))));
    float right = hz_add(c, hz_mul(q, hz_add(hz_mul(0.5f, hz_sub(b, d)),
        hz_mul(q, hz_sub(hz_mul(0.5f, hz_add(b, d)), c)))));
    float out = 0.0f;
    if (hz_zero(x)) out = b;
    if (x == 1.0f) out = c;
    if (both) {
        if (hz_zero(a) && hz_zero(d)) out = hz_add(hz_mul(b, q), hz_mul(c, x));
        else if (hz_zero(d)) out = left;
        else if (hz_zero(a)) out = right;
        else out = hz_add(hz_mul(q, left), hz_mul(x, right));
    }
    return out;
}
__device__ __forceinline__ int hz_clamp(int x, int n) { return max(0, min(x, n - 1)); }
__device__ __forceinline__ float hz_point(const float* src, int ny, int nx,
    int iy, int ix, float fy, float fx, int method, bool host,
    long long sy, long long sx) {
    if (method == 0) return src[iy * sy + ix * sx];
    if (method == 1) {
        float q = hz_sub(1.0f, fx);
        float lower = hz_add(hz_mul(q, src[iy * sy + ix * sx]), hz_mul(fx, src[iy * sy + (ix + 1) * sx]));
        float upper = hz_add(hz_mul(q, src[(iy + 1) * sy + ix * sx]), hz_mul(fx, src[(iy + 1) * sy + (ix + 1) * sx]));
        return hz_add(hz_mul(hz_sub(1.0f, fy), lower), hz_mul(fy, upper));
    }
    float rows[4];
    for (int j = 0; j < 4; ++j) {
        float v[4];
        for (int i = 0; i < 4; ++i) {
            float a = src[hz_clamp(iy + j - 1, ny) * sy + hz_clamp(ix + i - 1, nx) * sx];
            v[i] = hz_zero(a) ? 1.0e-20f : a;
        }
        rows[j] = hz_oned(fx, v[0], v[1], v[2], v[3], host);
    }
    float r = hz_oned(fy, rows[0], rows[1], rows[2], rows[3], host);
    return r == 1.0e-20f ? 0.0f : r;
}
extern "C" __global__ void horizontal_regular(const float* src, const float* y,
    const float* x, float* out, long long count, int nt, int ny, int nx, int method,
    long long sl, long long sy, long long sx) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    int t = k % nt, iy, ix;
    float fy = 0.0f, fx = 0.0f;
    if (method == 0) {
        iy = min(__double2int_rn(hz_double(y[t])), ny - 1);
        ix = min(__double2int_rn(hz_double(x[t])), nx - 1);
    } else {
        iy = (int)floor(hz_double(y[t])); ix = (int)floor(hz_double(x[t]));
        if (method == 1) { iy = min(iy, ny - 2); ix = min(ix, nx - 2); }
        fy = hz_sub(y[t], (float)iy); fx = hz_sub(x[t], (float)ix);
    }
    out[k] = hz_point(src + (k / nt) * sl, ny, nx, iy, ix, fy, fx, method, false, sy, sx);
}
extern "C" __global__ void horizontal_rotate(const float* u, const float* v,
    const float* sina, const float* cosa, float* ou, float* ov, long long count,
    int nt, int inverse) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    int t = k % nt;
    float uc = hz_mul(u[k], cosa[t]), vs = hz_mul(v[k], sina[t]);
    float vc = hz_mul(v[k], cosa[t]), us = hz_mul(u[k], sina[t]);
    ou[k] = inverse ? hz_sub(uc, vs) : hz_add(uc, vs);
    ov[k] = inverse ? hz_add(vc, us) : hz_sub(vc, us);
}
extern "C" __global__ void horizontal_cast(const double* src, float* out, long long count) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) out[k] = hz_round(src[k]);
}
extern "C" __global__ void horizontal_divide(const float* src, float* out,
    long long count, float divisor) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    float r = __fdiv_rn(src[k], divisor);
    if ((__float_as_uint(src[k]) & 0x7fffffffu) < 0x00800000u ||
        (__float_as_uint(r) & 0x7fffffffu) < 0x00800000u)
        r = hz_round(__ddiv_rn(hz_double(src[k]), hz_double(divisor)));
    out[k] = r;
}
extern "C" __global__ void horizontal_nearest(const float* src, const bool* land,
    const float* y, const float* x, const bool* target_land, float* out,
    unsigned* missing, int nt, int ny, int nx, int surface, int radius, double fill) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    bool active = surface == 0 || (surface == 1 ? target_land[t] : !target_land[t]);
    bool desired = surface == 0 ? target_land[t] : surface == 1;
    double yf = hz_double(y[t]), xf = hz_double(x[t]), best = __longlong_as_double(0x7ff0000000000000LL);
    int cy = __double2int_rn(yf), cx = __double2int_rn(xf);
    float value = hz_round(fill);
    if (active) for (int dj = -radius; dj <= radius; ++dj) {
        int jy = cy + dj;
        if (jy < 0 || jy >= ny) continue;
        for (int di = -radius; di <= radius; ++di) {
            int ix = cx + di;
            if (ix < 0 || ix >= nx) continue;
            int cell = jy * nx + ix;
            if ((__float_as_uint(src[cell]) & 0x7f800000u) == 0x7f800000u || land[cell] != desired) continue;
            double dy = __dsub_rn(yf, (double)jy), dx = __dsub_rn(xf, (double)ix);
            double distance = __dadd_rn(__dmul_rn(dy, dy), __dmul_rn(dx, dx));
            if (distance < best) { best = distance; value = src[cell]; }
        }
    }
    out[t] = value;
    if (active && !isfinite(best)) atomicAdd(missing, 1u);
}
extern "C" __global__ void horizontal_envelope(const float* src,
    unsigned* bounds, long long count) {
    __shared__ unsigned lo[256], hi[256], bad[256];
    unsigned lower = 0xffffffffu, upper = 0u, invalid = 0u;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         k < count; k += (long long)gridDim.x * blockDim.x) {
        unsigned u = __float_as_uint(src[k]);
        invalid |= (u & 0x7f800000u) == 0x7f800000u;
        unsigned ordered = u >> 31 ? ~u : u ^ 0x80000000u;
        lower = min(lower, ordered); upper = max(upper, ordered);
    }
    int t = threadIdx.x;
    lo[t] = lower; hi[t] = upper; bad[t] = invalid;
    __syncthreads();
    for (int offset = 128; offset; offset >>= 1) {
        if (t < offset) {
            lo[t] = min(lo[t], lo[t + offset]);
            hi[t] = max(hi[t], hi[t + offset]);
            bad[t] |= bad[t + offset];
        }
        __syncthreads();
    }
    if (!t) {
        atomicMin(bounds, lo[0]); atomicMax(bounds + 1, hi[0]);
        atomicOr(bounds + 2, bad[0]);
    }
}

// Match NumPy's binary64 expression order and portable Rust libm calls.
extern "C" __global__ void horizontal_rh_water(
        const double* rh, const double* t, float* out, long long n,
        double blend_width) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = t[i];
    double ice_arg = __dsub_rn(
        __dadd_rn(__dsub_rn(9.550426, __ddiv_rn(5723.265, v)),
                  __dmul_rn(3.53068, plm_log(v))),
        __dmul_rn(0.00728332, v));
    double eis = __dmul_rn(0.01, plm_exp(ice_arg));
    double delta = __dsub_rn(v, 273.15);
    double ews = __dmul_rn(6.112, plm_exp(__ddiv_rn(
        __dmul_rn(17.67, delta), __dadd_rn(delta, 243.5))));
    double frac = __ddiv_rn(__dsub_rn(273.15, v), blend_width);
    double blended = __dadd_rn(__dmul_rn(frac, eis),
                              __dmul_rn(__dsub_rn(1.0, frac), ews));
    double r = v > 253.15 ? blended : eis;
    double converted = __dmul_rn(rh[i], __ddiv_rn(r, ews));
    out[i] = hz_round(v <= 273.15 ? converted : rh[i]);
}
