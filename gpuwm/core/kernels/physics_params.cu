// Table parameter edits preserve binary64 multiplication before REAL rounding.

extern "C" __global__ void scale_physics_param_values(
    const double* values,
    const double* factors,
    const unsigned char* crop_flags,
    float* outputs,
    unsigned char* status,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const double value = values[index];
    const double factor = factors[index];
    const float result = __double2float_rn(__dmul_rn(value, factor));
    outputs[index] = result;
    unsigned char code = 0;
    if (!isfinite(value) || !isfinite(factor) || !isfinite(result)) code = 1;
    else if (result <= 0.0f) code = 2;
    else if (crop_flags[index] && result <= 0.125f) code = 3;
    status[index] = code;
}
