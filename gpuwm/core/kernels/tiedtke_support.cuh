// WRF cu6 column storage and rounded arithmetic. The transcendental calls
// use WOOF's own routines supplied by the kernel loader.
#pragma once
#ifndef TK_TRACE
#define TK_TRACE(line, value)
#endif

__device__ inline float tk_add(float a, float b) { return __fadd_rn(a, b); }
__device__ inline float tk_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ inline float tk_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ inline float tk_div(float a, float b) { return __fdiv_rn(a, b); }
template<class T> __device__ inline T tk_min(T a, T b) { return a < b ? a : b; }
template<class T> __device__ inline T tk_max(T a, T b) { return a > b ? a : b; }
__device__ inline float tk_abs(float a) { return fabsf(a); }
__device__ inline int tk_abs(int a) { return a < 0 ? -a : a; }
template<int N> __device__ inline float tk_powi(float a) {
    if constexpr (N == 1) return a;
    else if constexpr (N % 2 == 0) {
        float b = tk_powi<N / 2>(a);
        return tk_mul(b, b);
    } else return tk_mul(a, tk_powi<N - 1>(a));
}

// Each thread is one Fortran column. Array coordinates stay one-based.
// The second dimension is vertical and the storage is level-major, so
// adjacent CUDA lanes use adjacent words at every level.
template<class T> struct TkArray {
    T* data;
    int stride;
    int first;
    int count;
    int* status;
    __device__ T& at(int index) const {
        if (index < 0 || index >= count) {
            // F:2282/2289 can write at jk-1=0 when the original cloud top
            // reaches level 3. Refuse that undefined column before its
            // tendencies can enter the model. Keep every access in bounds.
            *status = -1;
            return data[0];
        }
        return data[index * stride];
    }
    __device__ T& operator()(int i) const { return at(i - 1); }
    __device__ T& operator()(int i, int k) const {
        return at((k - 1) * first + i - 1);
    }
    __device__ TkArray<T> slice(int i) const {
        return {data + (i - 1) * stride, stride, first, count - i + 1, status};
    }
};
template<class A, class B> struct TkSame { static constexpr bool value = false; };
template<class A> struct TkSame<A, A> { static constexpr bool value = true; };

// Passed by value to each routine. A child starts after its parent's live
// arrays; sequential calls reuse the same child region. No local column
// arrays or per-thread fixed vertical ceiling are compiled.
struct TkArena {
    float* f;
    int* i;
    bool* b;
    int nf, ni, nb, stride;
    int* status;
    int* activity;
    bool force_continue;
    template<class T> __device__ TkArray<T> take(int n, int first = 1) {
        if constexpr (TkSame<T, float>::value) {
            TkArray<T> a = {f + nf * stride, stride, first, n, status};
            nf += n;
            return a;
        } else if constexpr (TkSame<T, int>::value) {
            TkArray<T> a = {i + ni * stride, stride, first, n, status};
            ni += n;
            return a;
        } else {
            TkArray<T> a = {b + nb * stride, stride, first, n, status};
            nb += n;
            return a;
        }
    }
};
