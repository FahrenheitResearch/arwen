// Original ArWen unit conversion and deterministic receiver gather.
// Native CFE publication convention: (Q_OUT * (areasqkm * 1000000)) / 3600.
// No hydrologic parameterization or initialization is implemented here.
// NVRTC supplies device math intrinsics; no host header search is required.

template<class Real> __device__ void route_depth(const Real* depth, const double* area,
    const long long* offsets, const int* indices, float* lateral, double* discharge,
    double* volume, int* status, int nc, int nn, double scale, double seconds,
    int accumulate, int reset) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nn) return;
    double q = 0., v = 0.; int bad = 0;
    long long first = offsets[node], last = offsets[node+1];
    if (first < 0 || last < first || last > nc) bad |= 4;
    else for (long long j=first; j<last; ++j) {
        int c = indices[j];
        if (c < 0 || c >= nc) { bad |= 4; continue; }
        double d = (double)depth[c], a = area[c];
        if (!isfinite(d) || d < 0. || !isfinite(a) || a <= 0.) { bad |= 1; continue; }
        double part = d * (a * scale);
        v += part; q += part / seconds;
    }
    // Sum actual producer volumes across the routing window. Never sample
    // just the last LSM step, which loses nine out of ten runoff increments.
    if (accumulate) {
        if (!reset) v += volume[node];
        q = v / seconds;
    }
    float published = (float)q;
    if (!isfinite(v) || !isfinite(q) || !isfinite(published)) bad |= 2;
    if (bad) { v = nan(""); q = nan(""); published = nanf(""); }
    volume[node] = v; discharge[node] = q; lateral[node] = published;
    status[node] |= bad;
}

#define ENTRY(NAME, REAL) \
extern "C" __global__ void NAME(const REAL* d, const double* a, const long long* o, \
    const int* i, float* l, double* q, double* v, int* s, int nc, int nn, double scale, double dt, int acc, int reset) { \
    route_depth(d,a,o,i,l,q,v,s,nc,nn,scale,dt,acc,reset); }
ENTRY(route_depth_f32, float)
ENTRY(route_depth_f64, double)
