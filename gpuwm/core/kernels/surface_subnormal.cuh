// Zero predicates keep IEEE ordering even when a caller compiles with FTZ.
// Both zero signs compare equal. NaNs remain unordered; infinities retain
// their signs. Testing the bits prevents positive QSFC from being seeded
// as saturated water and negative previous MOL from losing its regime.
__device__ __forceinline__ bool surface_le_zero(float x) {
    unsigned int b = __float_as_uint(x), m = b & 0x7fffffffu;
    return m <= 0x7f800000u && (m == 0u || (b & 0x80000000u));
}
__device__ __forceinline__ bool surface_lt_zero(float x) {
    unsigned int b = __float_as_uint(x), m = b & 0x7fffffffu;
    return (b & 0x80000000u) && m != 0u && m <= 0x7f800000u;
}
