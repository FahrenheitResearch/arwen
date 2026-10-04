extern "C" __global__
void ruc_surface_parameters(
    const int* __restrict__ isltyp,
    const int* __restrict__ ivgtyp,
    const real* __restrict__ shdmin,
    const real* __restrict__ shdmax,
    const real* __restrict__ vegfrac,
    const real* __restrict__ znt_in,
    const real* __restrict__ lai_in,
    const int* __restrict__ ifortbl,
    const real* __restrict__ z0tbl,
    const real* __restrict__ lemitbl,
    const real* __restrict__ pctbl,
    const real* __restrict__ laitbl,
    const real* __restrict__ bb,
    const real* __restrict__ drysmc,
    const real* __restrict__ hc,
    const real* __restrict__ maxsmc,
    const real* __restrict__ refsmc,
    const real* __restrict__ satpsi,
    const real* __restrict__ satdk,
    const real* __restrict__ wltsmc,
    const real* __restrict__ qtz,
    int* __restrict__ iforest_out,
    real* __restrict__ emiss_out,
    real* __restrict__ pc_out,
    real* __restrict__ znt_out,
    real* __restrict__ lai_out,
    real* __restrict__ qwrtz_out,
    real* __restrict__ rhocs_out,
    real* __restrict__ bclh_out,
    real* __restrict__ dqm_out,
    real* __restrict__ ksat_out,
    real* __restrict__ psis_out,
    real* __restrict__ qmin_out,
    real* __restrict__ ref_out,
    real* __restrict__ wilt_out,
    int iswater, int rdlai2d, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int vegetation_index = ivgtyp[idx] - 1;
    int soil_index = isltyp[idx] - 1;
    int forest_class = ifortbl[vegetation_index];
    iforest_out[idx] = forest_class;

    real green_range = __fsub_rn(shdmax[idx], shdmin[idx]);
    real factor;
    if (green_range < 1.0f) {
        factor = 1.0f;
    } else {
        real numerator = __fsub_rn(vegfrac[idx], shdmin[idx]);
        real denominator = fmaxf(1.0f, green_range);
        real ratio = __fdiv_rn(numerator, denominator);
        factor = __fsub_rn(1.0f, fmaxf(0.0f, fminf(1.0f, ratio)));
    }

    real table_lai = laitbl[vegetation_index];
    real scaled_lai = __fmul_rn(0.8f, table_lai);
    real delta_lai = 0.0f;
    if (forest_class == 1) {
        delta_lai = fminf(0.2f, scaled_lai);
    } else if (forest_class == 2 || forest_class == 7) {
        delta_lai = fminf(0.5f, scaled_lai);
    } else if (forest_class == 3) {
        delta_lai = fminf(0.45f, scaled_lai);
    } else if (forest_class == 4) {
        delta_lai = fminf(0.75f, scaled_lai);
    } else if (forest_class == 5) {
        delta_lai = fminf(0.86f, scaled_lai);
    }

    real roughness = znt_in[idx];
    real leaf_area = lai_in[idx];
    if (ivgtyp[idx] == iswater) {
        if (!rdlai2d) leaf_area = table_lai;
    } else {
        if (!rdlai2d) {
            leaf_area = __fsub_rn(
                table_lai, __fmul_rn(delta_lai, factor));
        }
        roughness = z0tbl[vegetation_index];
        if (forest_class == 7) {
            roughness = __fsub_rn(
                roughness, __fmul_rn(0.125f, factor));
        }
    }
    emiss_out[idx] = lemitbl[vegetation_index];
    pc_out[idx] = pctbl[vegetation_index];
    znt_out[idx] = roughness;
    lai_out[idx] = leaf_area;

    qwrtz_out[idx] = 0.0f;
    rhocs_out[idx] = 0.0f;
    bclh_out[idx] = 0.0f;
    dqm_out[idx] = 0.0f;
    ksat_out[idx] = 0.0f;
    psis_out[idx] = 0.0f;
    qmin_out[idx] = 0.0f;
    ref_out[idx] = 0.0f;
    wilt_out[idx] = 0.0f;
    if (isltyp[idx] == 14) return;

    qwrtz_out[idx] = qtz[soil_index];
    rhocs_out[idx] = __fmul_rn(hc[soil_index], 1.0e6f);
    bclh_out[idx] = bb[soil_index];
    dqm_out[idx] = __fsub_rn(maxsmc[soil_index], drysmc[soil_index]);
    ksat_out[idx] = satdk[soil_index];
    psis_out[idx] = -satpsi[soil_index];
    qmin_out[idx] = drysmc[soil_index];
    ref_out[idx] = refsmc[soil_index];
    wilt_out[idx] = wltsmc[soil_index];
}