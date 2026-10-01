// uwpbl.cu -- the WRF v4.7.1 UW moist-turbulence PBL (bl_pbl_physics = 9),
// one thread per column.
//
// The loader (gpuwm/core/kernels/__init__.py _EXTRA_HEADERS["uwpbl"])
// prepends, in order: glibc_flt64.cuh, uwpbl_common.cuh, uwpbl_wvsat.cuh,
// uwpbl_vdiff.cuh, uwpbl_zisocl.cuh, uwpbl_caleddy.cuh, uwpbl_eddy.cuh,
// uwpbl_driver.cuh.
// uw_camuwpbl_column (uwpbl_driver.cuh) is module_bl_camuwpbl_driver.F's
// per-column body; this file only binds the engine's arrays to it and gives
// each thread its slice of the workspace pools.
//
// Layout.  Every 3-D field is WRF-ordered (k = 1 at the bottom) and stored
// level-major with the column index fastest: element k of column c is
// base[k*ncols + c], c = j*nx + i.  Surface fields are base[c].  The pools
// are slot-major with the chunk's thread index fastest, so a warp's reads of
// one slot are 32 consecutive words.

extern "C" __global__ void uwpbl_columns(
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ th, const float* __restrict__ rho,
    const float* __restrict__ qv, const float* __restrict__ qc,
    const float* __restrict__ qi, const float* __restrict__ qnc,
    const float* __restrict__ qni, const float* __restrict__ p,
    const float* __restrict__ z, const float* __restrict__ t,
    const float* __restrict__ cldfra, const float* __restrict__ exner,
    const float* __restrict__ rthratenlw, const float* __restrict__ wsedl3d,
    const float* __restrict__ p8w, const float* __restrict__ z_at_w,
    const float* __restrict__ hfx, const float* __restrict__ qfx,
    const float* __restrict__ ust, const float* __restrict__ ht,
    float* kvm3d, float* kvh3d, float* tauresx2d, float* tauresy2d,
    float* rublten, float* rvblten, float* rthblten, float* rqvblten,
    float* rqcblten, float* rqiblten, float* rqniblten,
    float* tke_pbl, float* turbtype3d, float* smaw3d,
    float* tpert2d, float* qpert2d, float* wpert2d, float* pblh2d,
    int* kpbl2d,
    long long ncols, long long col0, int nchunk, int pool_stride, int nk,
    float dt, int itimestep,
    double* r8pool, int* i4pool, int r8cap, int i4cap, int* err)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nchunk) return;
    const long long c = col0 + tid;
    const int s = (int)ncols;
    Ws ws{r8pool + tid, i4pool + tid, pool_stride, 0, r8cap, 0, i4cap, err};
    // Views start at this column's bottom element; element k (1-based) is
    // base[(k-1)*ncols + c].  The column driver never writes an input view.
    #define UW_IN(a) UwF32View{const_cast<float*>(a) + c, s}
    #define UW_OUT(a) UwF32View{(a) + c, s}
    UwColumnIn in{UW_IN(u), UW_IN(v), UW_IN(th), UW_IN(rho), UW_IN(qv),
                  UW_IN(qc), UW_IN(qi), UW_IN(qnc), UW_IN(qni), UW_IN(p),
                  UW_IN(z), UW_IN(t), UW_IN(cldfra), UW_IN(exner),
                  UW_IN(rthratenlw), UW_IN(wsedl3d), UW_IN(p8w), UW_IN(z_at_w),
                  hfx[c], qfx[c], ust[c], ht[c], dt, itimestep};
    UwColumnOut out{UW_OUT(kvm3d), UW_OUT(kvh3d), UW_OUT(rublten),
                    UW_OUT(rvblten), UW_OUT(rthblten), UW_OUT(rqvblten),
                    UW_OUT(rqcblten), UW_OUT(rqiblten), UW_OUT(rqniblten),
                    UW_OUT(tke_pbl), UW_OUT(turbtype3d), UW_OUT(smaw3d),
                    tauresx2d + c, tauresy2d + c, tpert2d + c, qpert2d + c,
                    wpert2d + c, pblh2d + c, kpbl2d + c};
    #undef UW_IN
    #undef UW_OUT
    uw_camuwpbl_column(nk, in, out, ws);
}
