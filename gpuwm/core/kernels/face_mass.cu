extern "C" __global__ void average_mass_faces(
    const float* mu, float* out, int ny, int nx, int yface) {
    int at = blockIdx.x * blockDim.x + threadIdx.x;
    int width = nx + (yface ? 0 : 1);
    int height = ny + (yface ? 1 : 0);
    if (at >= width * height) return;
    int j = at / width;
    int i = at - j * width;
    int jj = j % ny;
    int ii = i % nx;
    int other_j = yface ? (jj + ny - 1) % ny : jj;
    int other_i = yface ? ii : (ii + nx - 1) % nx;
    out[at] = __fmul_rn(0.5f, __fadd_rn(mu[jj * nx + ii],
                                      mu[other_j * nx + other_i]));
}
