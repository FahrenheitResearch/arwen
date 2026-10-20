// Original ArWen replay unit conversion; hydrology remains in the prepared
// CFE output and the normal t-route MC / level-pool kernels.
extern "C" __global__ void replay_lateral(const double* depth, const double* area,
    const long long* offsets, const int* indices, float* lateral, int* status,
    int nodes, int hours, int hour, double scale) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes) return;
    double q = 0.;
    int bad = 0;
    for (long long j = offsets[node]; j < offsets[node + 1]; ++j) {
        int c = indices[j];
        double d = depth[(long long)c * hours + hour];
        if (!isfinite(d) || d < 0.) bad |= 1;
        q += d * (area[c] * scale) / 3600.;
    }
    float published = (float)q;
    if (!isfinite(q) || !isfinite(published)) bad |= 2;
    lateral[node] = bad ? nanf("") : published;
    status[node] |= bad;
}
