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
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row[2]) {
        const unsigned int* src = (const unsigned int*)row[0];
        unsigned int* dst = (unsigned int*)row[1];
        dst[i] = src[i];
    }
}

extern "C" __global__ void rk_zero_words(RK_TABLE_PARAM table) {
    const unsigned long long* row = table.rows[blockIdx.y];
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row[2]) {
        unsigned int* dst = (unsigned int*)row[1];
        dst[i] = 0u;
    }
}
