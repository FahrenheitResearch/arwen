// WRF share/module_bc.F spec_bdytend + relax_bdytend_core.
// Side layout: W/E (nz,ny,width), S/N (nz,width,nx), distance 0 outermost.
// Float64 authority mirror: gpuwm.verify.npref.np_specified_relaxation.

static __device__ __forceinline__
real lbc_value(const real* value, const real* tendency, size_t index, real dtbc)
{
    return value[index] + dtbc * tendency[index];
}

extern "C" __global__
void specified_relaxation(const real* __restrict__ field,
                          real* __restrict__ field_tend,
                          const real* __restrict__ west,
                          const real* __restrict__ west_t,
                          const real* __restrict__ east,
                          const real* __restrict__ east_t,
                          const real* __restrict__ south,
                          const real* __restrict__ south_t,
                          const real* __restrict__ north,
                          const real* __restrict__ north_t,
                          const real* __restrict__ fcx,
                          const real* __restrict__ gcx,
                          real dtbc, int width, int spec_zone, int relax_zone,
                          int apply_relax,
                          int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = nz * ny * nx;
    if (tid >= count) return;
    int k = tid / (ny * nx);
    int rem = tid - k * ny * nx;
    int j = rem / nx;
    int i = rem - j * nx;
    real result = field_tend[tid];

    // Y sides own the corners, exactly as spec_bdytend's loop order/bounds.
    int ds = j;
    int dn = ny - 1 - j;
    if (ds < spec_zone && i >= ds && i < nx - ds) {
        size_t b = ((size_t)k * width + ds) * nx + i;
        field_tend[tid] = south_t[b];
        return;
    }
    if (dn < spec_zone && i >= dn && i < nx - dn) {
        size_t b = ((size_t)k * width + dn) * nx + i;
        field_tend[tid] = north_t[b];
        return;
    }
    int dw = i;
    int de = nx - 1 - i;
    if (dw < spec_zone && j >= dw + 1 && j < ny - dw - 1) {
        size_t b = ((size_t)k * ny + j) * width + dw;
        field_tend[tid] = west_t[b];
        return;
    }
    if (de < spec_zone && j >= de + 1 && j < ny - de - 1) {
        size_t b = ((size_t)k * ny + j) * width + de;
        field_tend[tid] = east_t[b];
        return;
    }

    if (!apply_relax) return;

    // Relaxation zone.  The mutually exclusive WRF corner bounds mean at
    // most one side applies to a point.
    if (ds >= spec_zone && ds < relax_zone && i >= ds && i < nx - ds) {
        int d = ds;
        size_t b0 = ((size_t)k * width + d) * nx + i;
        size_t bm = ((size_t)k * width + d) * nx + max(i - 1, 0);
        size_t bp = ((size_t)k * width + d) * nx + min(i + 1, nx - 1);
        size_t bo = ((size_t)k * width + d - 1) * nx + i;
        size_t bi = ((size_t)k * width + d + 1) * nx + i;
        real f0 = lbc_value(south, south_t, b0, dtbc) - field[tid];
        real f1 = lbc_value(south, south_t, bm, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + max(i-1,0)];
        real f2 = lbc_value(south, south_t, bp, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + min(i+1,nx-1)];
        real f3 = lbc_value(south, south_t, bo, dtbc) - field[(size_t)k*ny*nx + (size_t)(j-1)*nx + i];
        real f4 = lbc_value(south, south_t, bi, dtbc) - field[(size_t)k*ny*nx + (size_t)(j+1)*nx + i];
        result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    } else if (dn >= spec_zone && dn < relax_zone && i >= dn && i < nx - dn) {
        int d = dn;
        size_t b0 = ((size_t)k * width + d) * nx + i;
        size_t bm = ((size_t)k * width + d) * nx + max(i - 1, 0);
        size_t bp = ((size_t)k * width + d) * nx + min(i + 1, nx - 1);
        size_t bo = ((size_t)k * width + d - 1) * nx + i;
        size_t bi = ((size_t)k * width + d + 1) * nx + i;
        real f0 = lbc_value(north, north_t, b0, dtbc) - field[tid];
        real f1 = lbc_value(north, north_t, bm, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + max(i-1,0)];
        real f2 = lbc_value(north, north_t, bp, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + min(i+1,nx-1)];
        real f3 = lbc_value(north, north_t, bo, dtbc) - field[(size_t)k*ny*nx + (size_t)(j+1)*nx + i];
        real f4 = lbc_value(north, north_t, bi, dtbc) - field[(size_t)k*ny*nx + (size_t)(j-1)*nx + i];
        result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    } else if (dw >= spec_zone && dw < relax_zone
               && j >= dw + 1 && j < ny - dw - 1) {
        int d = dw;
        size_t b0 = ((size_t)k * ny + j) * width + d;
        size_t bm = ((size_t)k * ny + j - 1) * width + d;
        size_t bp = ((size_t)k * ny + j + 1) * width + d;
        size_t bo = ((size_t)k * ny + j) * width + d - 1;
        size_t bi = ((size_t)k * ny + j) * width + d + 1;
        real f0 = lbc_value(west, west_t, b0, dtbc) - field[tid];
        real f1 = lbc_value(west, west_t, bm, dtbc) - field[(size_t)k*ny*nx + (size_t)(j-1)*nx + i];
        real f2 = lbc_value(west, west_t, bp, dtbc) - field[(size_t)k*ny*nx + (size_t)(j+1)*nx + i];
        real f3 = lbc_value(west, west_t, bo, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + i-1];
        real f4 = lbc_value(west, west_t, bi, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + i+1];
        result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    } else if (de >= spec_zone && de < relax_zone
               && j >= de + 1 && j < ny - de - 1) {
        int d = de;
        size_t b0 = ((size_t)k * ny + j) * width + d;
        size_t bm = ((size_t)k * ny + j - 1) * width + d;
        size_t bp = ((size_t)k * ny + j + 1) * width + d;
        size_t bo = ((size_t)k * ny + j) * width + d - 1;
        size_t bi = ((size_t)k * ny + j) * width + d + 1;
        real f0 = lbc_value(east, east_t, b0, dtbc) - field[tid];
        real f1 = lbc_value(east, east_t, bm, dtbc) - field[(size_t)k*ny*nx + (size_t)(j-1)*nx + i];
        real f2 = lbc_value(east, east_t, bp, dtbc) - field[(size_t)k*ny*nx + (size_t)(j+1)*nx + i];
        real f3 = lbc_value(east, east_t, bo, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + i+1];
        real f4 = lbc_value(east, east_t, bi, dtbc) - field[(size_t)k*ny*nx + (size_t)j*nx + i-1];
        result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    }
    field_tend[tid] = result;
}
