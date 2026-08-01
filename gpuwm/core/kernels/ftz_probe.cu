// FTZ measurement kernels.
//
// One kernel per (mechanism) so the compiled PTX for each is readable in
// isolation and one arm can be mutated without disturbing the others.  Float
// operands arrive as raw uint32 bit patterns and leave the same way: the
// conversion between double and float is itself two of the mechanisms under
// test, so it must not contaminate the other four.
//
// Nothing here decides anything.  The kernels compute; tools/ftz_receipt
// records the bits and derives the verdicts from them.

extern "C" __global__
void ftz_probe_mul_op(const unsigned int* au, const unsigned int* bu,
                      unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __int_as_float((int)au[i]);
    const float b = __int_as_float((int)bu[i]);
    const float r = a * b;
    out[i] = (unsigned int)__float_as_int(r);
}

extern "C" __global__
void ftz_probe_mul_rn(const unsigned int* au, const unsigned int* bu,
                      unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __int_as_float((int)au[i]);
    const float b = __int_as_float((int)bu[i]);
    const float r = __fmul_rn(a, b);
    out[i] = (unsigned int)__float_as_int(r);
}

extern "C" __global__
void ftz_probe_minmax(const unsigned int* au, const unsigned int* bu,
                      unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __int_as_float((int)au[i]);
    const float b = __int_as_float((int)bu[i]);
    const float r = fminf(a, b);
    out[i] = (unsigned int)__float_as_int(r);
}

// The compare arm encodes its boolean as 1.0f / 0.0f: a flushed operand
// compares equal to zero and the encoded answer changes bit pattern, which is
// what the bit table can see.
extern "C" __global__
void ftz_probe_compare(const unsigned int* au, const unsigned int* bu,
                       unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __int_as_float((int)au[i]);
    const float b = __int_as_float((int)bu[i]);
    const float r = (a != b) ? 1.0f : 0.0f;
    out[i] = (unsigned int)__float_as_int(r);
}

extern "C" __global__
void ftz_probe_d2f_rn(const double* ad, unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float r = __double2float_rn(ad[i]);
    out[i] = (unsigned int)__float_as_int(r);
}

extern "C" __global__
void ftz_probe_cast(const double* ad, unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float r = (float)ad[i];
    out[i] = (unsigned int)__float_as_int(r);
}

// ---- inline-PTX arm (route R5) -------------------------------------------
//
// Instructions written without the .ftz modifier.  NVRTC passes inline asm
// through untouched, so these bodies say exactly which instruction ran.

extern "C" __global__
void ftz_probe_ptx_mul(const unsigned int* au, const unsigned int* bu,
                       unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __int_as_float((int)au[i]);
    const float b = __int_as_float((int)bu[i]);
    float r;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    out[i] = (unsigned int)__float_as_int(r);
}

extern "C" __global__
void ftz_probe_ptx_min(const unsigned int* au, const unsigned int* bu,
                       unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __int_as_float((int)au[i]);
    const float b = __int_as_float((int)bu[i]);
    float r;
    asm volatile("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    out[i] = (unsigned int)__float_as_int(r);
}

extern "C" __global__
void ftz_probe_ptx_setp(const unsigned int* au, const unsigned int* bu,
                        unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __int_as_float((int)au[i]);
    const float b = __int_as_float((int)bu[i]);
    float r;
    asm volatile("{\n\t"
                 ".reg .pred %%p;\n\t"
                 "setp.ne.f32 %%p, %1, %2;\n\t"
                 "selp.f32 %0, 0f3F800000, 0f00000000, %%p;\n\t"
                 "}" : "=f"(r) : "f"(a), "f"(b));
    out[i] = (unsigned int)__float_as_int(r);
}

extern "C" __global__
void ftz_probe_ptx_cvt(const double* ad, unsigned int* out, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double a = ad[i];
    float r;
    asm volatile("cvt.rn.f32.f64 %0, %1;" : "=f"(r) : "d"(a));
    out[i] = (unsigned int)__float_as_int(r);
}
