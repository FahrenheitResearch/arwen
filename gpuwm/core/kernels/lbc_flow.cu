// Batched WRF share/module_bc.F flow_dep_bdy for hydrometeors.
// Y sides own corners.  Each field is independent and uses the same
// mass-flux sign, so batching changes neither arithmetic nor write order.

static __device__ __forceinline__
bool flow_frame_point(int p, int ny, int nx, int width,
                      int* j, int* i, int* distance)
{
    for (int d = 0; d < width; ++d) {
        int ni = nx - 2*d;
        int nj = max(ny - 2*d - 2, 0);
        int ring = 2*ni + 2*nj;
        if (p >= ring) {
            p -= ring;
            continue;
        }
        *distance = d;
        if (p < ni) {
            *j = d;
            *i = d + p;
        } else if (p < 2*ni) {
            *j = ny - 1 - d;
            *i = d + p - ni;
        } else if (p < 2*ni + nj) {
            *j = d + 1 + p - 2*ni;
            *i = d;
        } else {
            *j = d + 1 + p - 2*ni - nj;
            *i = nx - 1 - d;
        }
        return true;
    }
    return false;
}

static __device__ __forceinline__
real* flow_field(int field_index,
                 real* f0, real* f1, real* f2, real* f3, real* f4,
                 real* f5, real* f6, real* f7, real* f8)
{
    switch (field_index) {
    case 0: return f0;
    case 1: return f1;
    case 2: return f2;
    case 3: return f3;
    case 4: return f4;
    case 5: return f5;
    case 6: return f6;
    case 7: return f7;
    default: return f8;
    }
}

extern "C" __global__
void flow_dependent_batch(
    real* __restrict__ f0,
    real* __restrict__ f1,
    real* __restrict__ f2,
    real* __restrict__ f3,
    real* __restrict__ f4,
    real* __restrict__ f5,
    real* __restrict__ f6,
    real* __restrict__ f7,
    real* __restrict__ f8,
    const real* __restrict__ u_flux,
    const real* __restrict__ v_flux,
    int nfield, int spec_zone, int nz, int ny, int nx, int frame_count)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*frame_count) return;
    int k = tid/frame_count;
    int p = tid - k*frame_count;
    int j, i, d;
    if (!flow_frame_point(p, ny, nx, spec_zone, &j, &i, &d)) return;

    size_t destination = I3(k, j, i, ny, nx);
    size_t source;
    bool outflow;
    if (j == d) {
        int inner_i = min(max(i, spec_zone), nx - 1 - spec_zone);
        source = I3(k, spec_zone, inner_i, ny, nx);
        outflow = v_flux[I3(k, j, i, ny + 1, nx)] < 0.0f;
    } else if (j == ny - 1 - d) {
        int inner_i = min(max(i, spec_zone), nx - 1 - spec_zone);
        source = I3(k, ny - 1 - spec_zone, inner_i, ny, nx);
        outflow = v_flux[I3(k, j + 1, i, ny + 1, nx)] > 0.0f;
    } else if (i == d) {
        int inner_j = min(max(j, spec_zone), ny - 1 - spec_zone);
        source = I3(k, inner_j, spec_zone, ny, nx);
        outflow = u_flux[I3(k, j, i, ny, nx + 1)] < 0.0f;
    } else {
        int inner_j = min(max(j, spec_zone), ny - 1 - spec_zone);
        source = I3(k, inner_j, nx - 1 - spec_zone, ny, nx);
        outflow = u_flux[I3(k, j, i + 1, ny, nx + 1)] > 0.0f;
    }

    for (int n = 0; n < nfield; ++n) {
        real* field = flow_field(n, f0, f1, f2, f3, f4,
                                 f5, f6, f7, f8);
        field[destination] = outflow ? field[source] : 0.0f;
    }
}

extern "C" __global__
void zero_gradient_w(real* __restrict__ field,
                     int spec_zone, int nz, int ny, int nx, int frame_count)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*frame_count) return;
    int k = tid/frame_count;
    int p = tid - k*frame_count;
    int j, i, d;
    if (!flow_frame_point(p, ny, nx, spec_zone, &j, &i, &d)) return;

    int source_j;
    int source_i;
    if (j == d) {
        source_j = spec_zone;
        source_i = min(max(i, spec_zone), nx - 1 - spec_zone);
    } else if (j == ny - 1 - d) {
        source_j = ny - 1 - spec_zone;
        source_i = min(max(i, spec_zone), nx - 1 - spec_zone);
    } else if (i == d) {
        source_j = min(max(j, spec_zone), ny - 1 - spec_zone);
        source_i = spec_zone;
    } else {
        source_j = min(max(j, spec_zone), ny - 1 - spec_zone);
        source_i = nx - 1 - spec_zone;
    }
    field[I3(k, j, i, ny, nx)] = field[I3(k, source_j, source_i, ny, nx)];
}
