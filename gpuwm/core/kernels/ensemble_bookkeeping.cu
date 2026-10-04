// Rows hold source, destination, words, and source/destination member strides.
struct EnsembleWordTable {
    unsigned long long rows[64][5];
};

#if __CUDA_ARCH__ >= 700
#define ENSEMBLE_TABLE_PARAM const __grid_constant__ EnsembleWordTable
#else
#define ENSEMBLE_TABLE_PARAM const EnsembleWordTable
#endif

extern "C" __global__ void ensemble_copy_words(ENSEMBLE_TABLE_PARAM table) {
    const unsigned long long* row = table.rows[blockIdx.y];
    const unsigned long long member = blockIdx.z;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x * 4ull;
    if (first >= row[2]) return;
    const unsigned int* src = (const unsigned int*)row[0] + member * row[3];
    unsigned int* dst = (unsigned int*)row[1] + member * row[4];
    if ((((unsigned long long)src | (unsigned long long)dst) & 15ull) == 0ull) {
        const unsigned long long at = first + 4ull * threadIdx.x;
        if (at + 3ull < row[2]) {
            ((uint4*)dst)[at / 4ull] = ((const uint4*)src)[at / 4ull];
        } else {
            for (unsigned int q = 0; q < 4 && at + q < row[2]; ++q)
                dst[at + q] = src[at + q];
        }
    } else {
        const unsigned long long at = first + threadIdx.x;
        #pragma unroll
        for (unsigned int q = 0; q < 4; ++q) {
            const unsigned long long word = at + (unsigned long long)q * blockDim.x;
            if (word < row[2]) dst[word] = src[word];
        }
    }
}

extern "C" __global__ void ensemble_zero_words(ENSEMBLE_TABLE_PARAM table) {
    const unsigned long long* row = table.rows[blockIdx.y];
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x * 4ull;
    if (first >= row[2]) return;
    unsigned int* dst = (unsigned int*)row[1] + (unsigned long long)blockIdx.z * row[4];
    if (((unsigned long long)dst & 15ull) == 0ull) {
        const unsigned long long at = first + 4ull * threadIdx.x;
        if (at + 3ull < row[2]) {
            const uint4 zero = {0u, 0u, 0u, 0u};
            ((uint4*)dst)[at / 4ull] = zero;
        } else {
            for (unsigned int q = 0; q < 4 && at + q < row[2]; ++q)
                dst[at + q] = 0u;
        }
    } else {
        const unsigned long long at = first + threadIdx.x;
        #pragma unroll
        for (unsigned int q = 0; q < 4; ++q) {
            const unsigned long long word = at + (unsigned long long)q * blockDim.x;
            if (word < row[2]) dst[word] = 0u;
        }
    }
}

// Each destination face is disjoint from every source face. Integer words
// preserve signed zeros and NaN payloads without CuPy's overlapping-view copy.
extern "C" __global__ void ensemble_close_faces(
    unsigned int* u, unsigned int* v, unsigned long long n_u,
    unsigned long long n_v, int nx, int ny, int close_x, int close_y) {
    const unsigned long long at =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (close_x && at < n_u) {
        const unsigned long long first = at * (static_cast<unsigned long long>(nx) + 1ull);
        u[first + nx] = u[first];
    }
    if (close_y && at < n_v) {
        const unsigned long long k = at / static_cast<unsigned int>(nx);
        const unsigned long long i = at % static_cast<unsigned int>(nx);
        const unsigned long long first = k * (static_cast<unsigned long long>(ny) + 1ull) * nx + i;
        v[first + static_cast<unsigned long long>(ny) * nx] = v[first];
    }
}
