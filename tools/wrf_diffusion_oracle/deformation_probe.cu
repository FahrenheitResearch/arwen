// Diagnostic exposure of production inline tensor functions. This is appended
// to module_source("smag2d"); it contains no independent tensor transcription.
extern "C" __global__
void oracle_expose_deformation(WRF_SMAG_GRID_ARGS,
    const real* d11, const real* d22,
    real* d33, real* div, real* d13, real* d23,
    int nz, int ny, int nx, int phb3d, int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    WRF_SMAG_MAKE_GRID;
    if (i < nx && j < ny && k < nz) {
        real dw = (q.w[I3(k + 1, j, i, ny, nx)]
                    - q.w[I3(k, j, i, ny, nx)]) * wrf_rdzw(q, k, j, i);
        d33[IDX3(k,j,i)] = 2.0f * dw;
        div[IDX3(k,j,i)] = 0.5f*d11[IDX3(k,j,i)]
                         + 0.5f*d22[IDX3(k,j,i)] + dw;
    }
    if (i <= nx && j < ny && k <= nz)
        d13[I3S(k,j,i,ny,nx+1)] = wrf_defor13(q,k,j,i);
    if (i < nx && j <= ny && k <= nz)
        d23[I3S(k,j,i,ny+1,nx)] = wrf_defor23(q,k,j,i);
}
