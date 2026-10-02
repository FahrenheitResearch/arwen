// gpuwm/core/kernels/saxpy.cu
template<typename T> struct SaxpyVector;
template<> struct SaxpyVector<float> { using type = float4; };
template<> struct SaxpyVector<double> { using type = double4; };

extern "C" __global__
void saxpy(real a, const real* x, const real* y, real* out, int n) {
    if (n <= 0) return;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x * 4ull;
    if (first >= (unsigned int)n) return;
    using Vector = typename SaxpyVector<real>::type;
    const unsigned long long addresses = (unsigned long long)x
        | (unsigned long long)y | (unsigned long long)out;
    if ((addresses & (alignof(Vector) - 1ull)) == 0ull) {
        const unsigned long long i = first + 4ull * threadIdx.x;
        if (i + 3ull < (unsigned int)n) {
            const Vector xv = ((const Vector*)x)[i / 4ull];
            const Vector yv = ((const Vector*)y)[i / 4ull];
            Vector value;
            value.x = a * xv.x + yv.x;
            value.y = a * xv.y + yv.y;
            value.z = a * xv.z + yv.z;
            value.w = a * xv.w + yv.w;
            ((Vector*)out)[i / 4ull] = value;
        } else {
            for (unsigned int j = 0; j < 4 && i + j < (unsigned int)n; ++j)
                out[i + j] = a * x[i + j] + y[i + j];
        }
    } else {
        const unsigned long long i = first + threadIdx.x;
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned long long at = i + (unsigned long long)j * blockDim.x;
            if (at < (unsigned int)n) out[at] = a * x[at] + y[at];
        }
    }
}

extern "C" __global__
void emit_g(real* out) { out[0] = G; }
