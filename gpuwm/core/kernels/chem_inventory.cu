// Anthropogenic (inventory) emission add, WRF-Chem v4.7.1 GOCART arm.
//
// SOURCE.  chem/emissions_driver.F:1597-1619, the emiss_opt == 6 block of the
// GOCART_SIMPLE case, which is the one WRF-Chem path that adds a gridded
// inventory to the GOCART species:
//
//   do k = kts, min(config_flags%kemit, kte-ksub)          ! ksub = 0 (:747)
//     conv = 4.828e-4/rho_phy(i,k,j)*dtstep/(dz8w(i,k,j)*60.)
//     chem(p_so2)  = chem(p_so2)  + emis_ant(p_e_so2)*conv           ! mol km-2 hr-1
//     chem(p_bc1)  = chem(p_bc1)  + (emis_ant(p_e_bc))*alt*dtstep/dz8w  ! ug m-2 s-1
//     ... oc1, p25, p10 like bc1; sulf like so2
//
// TABLE FORM.  No species is named here.  Each TERM is one emission entry of
// one row: the row's field, the emission frame it reads (float32, (nlev, ny,
// nx), already on the model grid in WRF's emis_ant units after the entry's
// weight), and which of the two WRF arithmetic paths applies, decided by the
// row's units (ppmv rows take the mol km-2 hr-1 path through conv, ug kg-1
// rows the alt*dtstep/dz8w path).  WRF computes conv once per level before
// its six species; so does this kernel.  A term whose frame has fewer levels
// than kemit adds nothing above them, which is WRF's add of a zero emission
// (x + 0*conv == x for the non-negative mixing ratios transport leaves).
//
// The weight multiplies the frame first (ArWen's unit conversion from the
// source's units; exactly 1 when the frame is already in WRF units, and
// x*1 is x), then WRF's own expression runs word for word.  Every product
// and sum is an explicit round-to-nearest intrinsic so NVRTC cannot contract
// a*b + c into an FMA, which gfortran -O0 never does.
//
// One thread per column; levels 0 .. min(kemit, nz)-1.

extern "C" __global__ void chem_inventory_add(
    const float* __restrict__ rho_phy,   // (nz, ny, nx) moist density, chem_prep
    const float* __restrict__ alt,       // (nz, ny, nx) inverse dry density
    const float* __restrict__ dz8w,      // (nz, ny, nx)
    const unsigned long long* __restrict__ term_field,  // float* (nz, ny, nx)
    const unsigned long long* __restrict__ term_emis,   // float* (nlev, ny, nx)
    const int* __restrict__ term_nlev,
    const int* __restrict__ term_mode,   // 0: ug m-2 s-1 path, 1: mol km-2 hr-1 path
    const float* __restrict__ term_weight,
    const int nterm,
    const int kemit,
    const float dtstep,
    const int nz,
    const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const int ktop = kemit < nz ? kemit : nz;
    for (int k = 0; k < ktop; ++k) {
        const long long idx = (long long)k * ncol + c;
        const float rho = rho_phy[idx];
        const float dz = dz8w[idx];
        const float a = alt[idx];
        // conv = 4.828e-4/rho_phy*dtstep/(dz8w*60.)
        const float conv = __fdiv_rn(__fmul_rn(__fdiv_rn(4.828e-4f, rho), dtstep),
                                     __fmul_rn(dz, 60.0f));
        for (int t = 0; t < nterm; ++t) {
            if (k >= term_nlev[t]) continue;
            float* q = reinterpret_cast<float*>(term_field[t]);
            const float* e = reinterpret_cast<const float*>(term_emis[t]);
            const float emis = __fmul_rn(e[idx], term_weight[t]);
            float add;
            if (term_mode[t] == 1) {
                add = __fmul_rn(emis, conv);
            } else {
                add = __fdiv_rn(__fmul_rn(__fmul_rn(emis, a), dtstep), dz);
            }
            q[idx] = __fadd_rn(q[idx], add);
        }
    }
}
