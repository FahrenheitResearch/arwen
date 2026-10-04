// THIRD-PARTY NOTICE
// WRF v4.6.1 dyn_em/module_stoch.F: spectral AR(1), rotational derivatives.
// WRF is public domain; the full notice is in stochastic.py and
// licenses/LICENSE-WRF-public-domain.txt. Philox4x32-10 follows Random123;
// its BSD notice is in licenses/LICENSE-Random123.txt and the root NOTICE.
// All floating-point divisions use explicit correctly rounded intrinsics.

__device__ __forceinline__ uint4 stoch_philox(uint4 c, unsigned long long seed) {
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = 0xD2511F53ULL * c.x;
        unsigned long long p1 = 0xCD9E8D57ULL * c.z;
        c = make_uint4((unsigned int)(p1 >> 32) ^ c.y ^ k0,
                       (unsigned int)p1,
                       (unsigned int)(p0 >> 32) ^ c.w ^ k1,
                       (unsigned int)p0);
        k0 += 0x9E3779B9U;
        k1 += 0xBB67AE85U;
    }
    return c;
}

extern "C" __global__ void stoch_rng_words(
        unsigned int* out, const unsigned int* counters,
        unsigned long long seed, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint4 c = stoch_philox(make_uint4(counters[4*i], counters[4*i+1],
                                    counters[4*i+2], counters[4*i+3]), seed);
    out[4*i] = c.x; out[4*i+1] = c.y; out[4*i+2] = c.z; out[4*i+3] = c.w;
}

__device__ float stoch_gaussian(unsigned int index, unsigned int step,
                               unsigned int stream, unsigned long long seed) {
    // WRF gauss_noise:1492-1513, polar transform with |z| < 3 rejection.
    // Every attempted pair has its own counter, independent of other cells.
    for (unsigned int attempt = 0; ; ++attempt) {
        uint4 r = stoch_philox(make_uint4(index, step, stream, attempt), seed);
        float u = __fmul_rn((float)(r.x >> 8), 0x1p-24f);
        float v = __fmul_rn((float)(r.y >> 8), 0x1p-24f);
        float x = __fsub_rn(__fmul_rn(2.0f, u), 1.0f);
        float y = __fsub_rn(__fmul_rn(2.0f, v), 1.0f);
        float radius = __fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y));
        if (!(radius > 0.0f && radius < 1.0f)) continue;
        float z = __fmul_rn(sqrtf(__fdiv_rn(__fmul_rn(-2.0f, logf(radius)), radius)), x);
        if (fabsf(z) < 3.0f) return z;
    }
}

extern "C" __global__ void stoch_spectrum_weights(
        double* log_weights, double* gamma, int nx, int ny,
        double dx, double dy, int kind, double length, double exponent,
        int minimum, int maximum) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nx * ny) return;
    int x = p % nx, y = p / nx;
    double rx = nx * dx, ry = ny * dy;
    double kx = __ddiv_rn((double)x, rx), ky = __ddiv_rn((double)y, ry);
    double rho2 = kx * kx + ky * ky;
    double rho = sqrt(rho2);
    bool band = (rho < __ddiv_rn(maximum + 0.5, rx) &&
                 rho >= __ddiv_rn(minimum - 0.5, rx)) ||
                (rho < __ddiv_rn(maximum + 0.5, ry) &&
                 rho >= __ddiv_rn(minimum - 0.5, ry));
    if (!band || (x == 0 && y == 0)) {
        log_weights[p] = -__longlong_as_double(0x7ff0000000000000LL);
        gamma[p] = kind == 0 ? -__longlong_as_double(0x7ff0000000000000LL) : 0.0;
        return;
    }
    const double pi = 3.14159265358979323846;
    if (kind == 0) {
        double logchi = -2.0 * pi * pi * length * length * rho2;
        log_weights[p] = logchi;
        gamma[p] = logchi;
    } else {
        log_weights[p] = 0.5 * exponent * log(rho2);
        gamma[p] = pow(rho2, exponent + (kind == 1 ? 1.0 : 0.0));
    }
}

extern "C" __global__ void stoch_spectrum_amplitude(
        float* amplitude, const double* log_weights, int nx, int ny,
        double shift, double f0) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nx * ny) return;
    int x = p % nx, y = p / nx;
    int qx = x <= nx / 2 ? x : nx - x;
    int qy = y <= ny / 2 ? y : ny - y;
    amplitude[p] = (float)(f0 * exp(log_weights[qy * nx + qx] - shift));
}

__device__ __forceinline__ unsigned int stoch_noise_index(
        int p, int nx, int ny, float* sign) {
    int x = p % nx, y = p / nx;
    int qx = x, qy = y;
    *sign = 1.0f;
    if ((y == 0 && x > nx / 2) || y > ny / 2) {
        qx = x == 0 ? 0 : nx - x;
        qy = y == 0 ? 0 : ny - y;
        *sign = -1.0f;
    }
    return (unsigned int)(qy * nx + qx);
}

__device__ __forceinline__ float2 stoch_ar1(
        float2 old, float alpha, float amplitude,
        float cosine, float sine, float sign) {
    float phi = __fsub_rn(1.0f, alpha);
    return make_float2(
        __fadd_rn(__fmul_rn(phi, old.x), __fmul_rn(amplitude, cosine)),
        __fadd_rn(__fmul_rn(phi, old.y), __fmul_rn(amplitude, __fmul_rn(sign, sine))));
}

extern "C" __global__ void stoch_update(
        float2* spectrum, const float* amplitude, int nx, int ny,
        float alpha, unsigned long long seed, unsigned int step,
        unsigned int stream) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nx * ny) return;
    float sign;
    unsigned int q = stoch_noise_index(p, nx, ny, &sign);
    float cosine = stoch_gaussian(2U*q, step, stream, seed);
    float sine = stoch_gaussian(2U*q+1U, step, stream, seed);
    spectrum[p] = stoch_ar1(spectrum[p], alpha, amplitude[p], cosine, sine, sign);
}

// Native WRF verification seam: prescribed innovations exercise the SAME
// conjugacy and AR(1) helpers as the runtime counter-generated path.
extern "C" __global__ void stoch_update_given_noise(
        float2* spectrum, const float* amplitude, const float2* innovations,
        int nx, int ny, float alpha) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nx * ny) return;
    float sign;
    unsigned int q = stoch_noise_index(p, nx, ny, &sign);
    float2 noise = innovations[q];
    spectrum[p] = stoch_ar1(spectrum[p], alpha, amplitude[p], noise.x, noise.y, sign);
}

extern "C" __global__ void stoch_derivative(
        float2* result, const float2* spectrum, int nx, int ny,
        float dx, float dy, int component) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nx * ny) return;
    int x = p % nx, y = p / nx;
    int k = component == 1 ? (y <= ny / 2 ? y : y - ny) :
                             (x <= nx / 2 ? x : x - nx);
    float size = component == 1 ? __fmul_rn((float)ny, dy) : __fmul_rn((float)nx, dx);
    float factor = __fmul_rn(__fdiv_rn(6.2831853071795864769f, size), (float)k);
    float2 z = spectrum[p];
    result[p] = component == 1 ?
        make_float2(__fmul_rn(factor, z.y), -__fmul_rn(factor, z.x)) :
        make_float2(-__fmul_rn(factor, z.y), __fmul_rn(factor, z.x));
}
