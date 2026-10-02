// NumPy keeps the first NaN operand's payload; CUDA may select the second.
// Finite arithmetic still uses the explicit round-to-nearest instruction.
__device__ __forceinline__ double real_quiet64(double a) {
    return __longlong_as_double((long long)((unsigned long long)__double_as_longlong(a)
                                          | 0x0008000000000000ULL));
}
__device__ __forceinline__ float real_quiet32(float a) {
    return __int_as_float(__float_as_int(a) | 0x00400000);
}
// Step IEEE words so FTZ cannot collapse a subnormal input or neighbor.
// Match NumPy nextafter: a NaN target takes precedence and is quieted.
__device__ __forceinline__ float real_nextafter32(float a, float b) {
    unsigned int wa = (unsigned int)__float_as_int(a);
    unsigned int wb = (unsigned int)__float_as_int(b);
    unsigned int ma = wa & 0x7fffffffu, mb = wb & 0x7fffffffu;
    if (mb > 0x7f800000u) return __int_as_float((int)(wb | 0x00400000u));
    if (ma > 0x7f800000u) return __int_as_float((int)(wa | 0x00400000u));
    // Equal zeros return b's sign; all other equal values return b unchanged.
    if (wa == wb || (ma == 0u && mb == 0u)) return b;
    if (ma == 0u) return __int_as_float((int)((wb & 0x80000000u) | 1u));
    // Opposite signs move toward zero. Equal signs compare magnitudes.
    if (((wa ^ wb) & 0x80000000u) != 0u || ma > mb) --wa;
    else ++wa;
    return __int_as_float((int)wa);
}
__device__ __forceinline__ double real_add(double a, double b) {
    if (isnan(a)) return real_quiet64(a);
    if (isnan(b)) return real_quiet64(b);
    return __dadd_rn(a, b);
}
__device__ __forceinline__ double real_sub(double a, double b) {
    if (isnan(a)) return real_quiet64(a);
    if (isnan(b)) return real_quiet64(b);
    return __dsub_rn(a, b);
}
__device__ __forceinline__ double real_mul(double a, double b) {
    if (isnan(a)) return real_quiet64(a);
    if (isnan(b)) return real_quiet64(b);
    return __dmul_rn(a, b);
}
__device__ __forceinline__ double real_div(double a, double b) {
    if (isnan(a)) return real_quiet64(a);
    if (isnan(b)) return real_quiet64(b);
    return __ddiv_rn(a, b);
}
__device__ __forceinline__ float real_float(double a);
__device__ __forceinline__ double real_double(float a) {
    unsigned int word = (unsigned int)__float_as_int(a);
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
    return __longlong_as_double((long long)bits);
}
__device__ __forceinline__ bool real_float_fallback(float a, float b, float r) {
    unsigned int ma = (unsigned int)__float_as_int(a) & 0x7fffffffu;
    unsigned int mb = (unsigned int)__float_as_int(b) & 0x7fffffffu;
    unsigned int mr = (unsigned int)__float_as_int(r) & 0x7fffffffu;
    return (ma != 0u && ma < 0x00800000u) || (mb != 0u && mb < 0x00800000u)
           || mr < 0x00800000u || isnan(r);
}
// Use native RN for ordinary values. Subnormal operands/results use the
// widened operation and a fresh FP32 rounding, avoiding sm_120's flush.
__device__ __forceinline__ float real_fadd(float a, float b) {
    if (isnan(a)) return real_quiet32(a);
    if (isnan(b)) return real_quiet32(b);
    float result = __fadd_rn(a, b);
    if (real_float_fallback(a, b, result))
        return real_float(real_add(real_double(a), real_double(b)));
    return result;
}
__device__ __forceinline__ float real_fsub(float a, float b) {
    if (isnan(a)) return real_quiet32(a);
    if (isnan(b)) return real_quiet32(b);
    float result = __fsub_rn(a, b);
    if (real_float_fallback(a, b, result))
        return real_float(real_sub(real_double(a), real_double(b)));
    return result;
}
__device__ __forceinline__ float real_fmul(float a, float b) {
    if (isnan(a)) return real_quiet32(a);
    if (isnan(b)) return real_quiet32(b);
    float result = __fmul_rn(a, b);
    if (real_float_fallback(a, b, result))
        return real_float(real_mul(real_double(a), real_double(b)));
    return result;
}
__device__ __forceinline__ float real_fdiv(float a, float b) {
    if (isnan(a)) return real_quiet32(a);
    if (isnan(b)) return real_quiet32(b);
    float result = __fdiv_rn(a, b);
    if (real_float_fallback(a, b, result))
        return real_float(real_div(real_double(a), real_double(b)));
    return result;
}
__device__ __forceinline__ float real_float(double a) {
    unsigned long long bits = (unsigned long long)__double_as_longlong(a);
    unsigned int sign = (unsigned int)((bits >> 32) & 0x80000000ULL);
    if (isnan(a)) {
        unsigned int narrowed = sign | 0x7fc00000u
                              | (unsigned int)((bits >> 29) & 0x007fffffULL);
        return __int_as_float((int)narrowed);
    }
    unsigned int exponent = (unsigned int)((bits >> 52) & 0x7ffULL);
    // Subnormal float results are rounded with integers. The conversion
    // instruction may flush them on sm_120 even with the RN spelling.
    if (exponent < 897u) {
        if (exponent < 873u) return __int_as_float((int)sign);
        unsigned long long mantissa = (bits & 0x000fffffffffffffULL) | 0x0010000000000000ULL;
        unsigned int shift = 926u - exponent;
        unsigned long long rounded = mantissa >> shift;
        unsigned long long tail = mantissa & ((1ULL << shift) - 1ULL);
        unsigned long long half = 1ULL << (shift - 1u);
        if (tail > half || (tail == half && (rounded & 1ULL))) ++rounded;
        return __int_as_float((int)(sign | (unsigned int)rounded));
    }
    return __double2float_rn(a);
}
