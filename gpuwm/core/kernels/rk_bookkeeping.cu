struct RKWordTable {
    unsigned long long rows[64][3];
};

#if __CUDA_ARCH__ >= 700
#define RK_TABLE_PARAM const __grid_constant__ RKWordTable
#else
#define RK_TABLE_PARAM const RKWordTable
#endif

// Copy words without floating-point conversion, including NaN payloads.
extern "C" __global__ void rk_copy_words(RK_TABLE_PARAM table) {
    const unsigned long long* row = table.rows[blockIdx.y];
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x * 4ull;
    if (first >= row[2]) return;
    const unsigned int* src = (const unsigned int*)row[0];
    unsigned int* dst = (unsigned int*)row[1];
    if (((row[0] | row[1]) & 15ull) == 0ull) {
        const unsigned long long i = first + 4ull * threadIdx.x;
        if (i + 3ull < row[2]) {
            ((uint4*)dst)[i / 4ull] = ((const uint4*)src)[i / 4ull];
        } else {
            for (unsigned int j = 0; j < 4 && i + j < row[2]; ++j)
                dst[i + j] = src[i + j];
        }
    } else {
        // Slices need only word alignment; each scalar pass stays coalesced.
        const unsigned long long i = first + threadIdx.x;
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned long long at = i + (unsigned long long)j * blockDim.x;
            if (at < row[2]) dst[at] = src[at];
        }
    }
}

extern "C" __global__ void rk_zero_words(RK_TABLE_PARAM table) {
    const unsigned long long* row = table.rows[blockIdx.y];
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x * 4ull;
    if (first >= row[2]) return;
    unsigned int* dst = (unsigned int*)row[1];
    if ((row[1] & 15ull) == 0ull) {
        const unsigned long long i = first + 4ull * threadIdx.x;
        if (i + 3ull < row[2]) {
            const uint4 zero = {0u, 0u, 0u, 0u};
            ((uint4*)dst)[i / 4ull] = zero;
        } else {
            for (unsigned int j = 0; j < 4 && i + j < row[2]; ++j)
                dst[i + j] = 0u;
        }
    } else {
        const unsigned long long i = first + threadIdx.x;
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned long long at = i + (unsigned long long)j * blockDim.x;
            if (at < row[2]) dst[at] = 0u;
        }
    }
}
