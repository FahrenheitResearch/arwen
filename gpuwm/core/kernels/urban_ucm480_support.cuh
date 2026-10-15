// Intrinsic spellings used by the source-order WRF 4.8 listing.
__device__ inline float u480_real(int x) {return (float)x;}
__device__ inline int u480_int(float x) {return (int)x;}
__device__ inline int u480_nint(float x) {return (int)roundf(x);}
__device__ inline int u480_mod(int x,int y) {return x%y;}
__device__ inline float u480_sign(float x,float y) {return copysignf(fabsf(x),y);}
// The constant midpoint 0.439f - 0.066f chose the odd lower word on
// the measured CUDA build. Keep operands runtime values and execute
// RN subtraction explicitly, scoped to the new generation.
#ifdef __CUDACC__
__device__ __noinline__ float u480_sub(float a,float b) {
#else
__device__ inline float u480_sub(float a,float b) {
#endif
#ifdef __CUDA_ARCH__
    return __double2float_rn(__dsub_rn((double)a,(double)b));
#else
    return __fsub_rn(a,b);
#endif
}
