// Data representation conversion at the external boundary clock's dtbc.
// Outputs are value(t), derivative(t); consumers receive dtbc=0 afterwards.
extern "C" __global__
void evaluate_linear_boundary(const real* value, const real* tendency,
                              real* out_value, real* out_tendency,
                              real seconds, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out_value[i] = value[i] + seconds * tendency[i];
    out_tendency[i] = tendency[i];
}

extern "C" __global__
void evaluate_rational_boundary(const real* value, const real* tendency,
                                const real* quadratic, const real* rate,
                                real* out_value, real* out_tendency,
                                real seconds, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // FP64 intermediates avoid cancellation while converting the FP32 input
    // coefficients; the shared dycore still consumes FP32 boundary tables.
    double t = seconds, p = tendency[i], q = quadratic[i], d = rate[i];
    double denominator = 1.0 + t*d;
    out_value[i] = (real)((double)value[i] + t*(p+t*q)/denominator);
    out_tendency[i] = (real)((p+2.0*t*q+t*t*q*d)/(denominator*denominator));
}
