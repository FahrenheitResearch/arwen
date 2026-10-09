// Appended to module_source("smag2d").  It launches the production device
// functions wrf_defor13 / wrf_defor23 (the D13/D23 the strict w operator
// evaluates inline) at every face; it holds no tensor arithmetic of its own.
extern "C" __global__
void oracle_expose_d13_d23(WRF_SMAG_GRID_ARGS, real* d13, real* d23,
                           int nz, int ny, int nx, int phb3d,
                           int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    WRF_SMAG_MAKE_GRID;
    if (i <= nx && j < ny && k <= nz)
        d13[I3S(k, j, i, ny, nx + 1)] = wrf_defor13(q, k, j, i);
    if (i < nx && j <= ny && k <= nz)
        d23[I3S(k, j, i, ny + 1, nx)] = wrf_defor23(q, k, j, i);
}
