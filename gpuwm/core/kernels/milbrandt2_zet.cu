// gpuwm/core/kernels/milbrandt2_zet.cu
//
// Milbrandt-Yau (mp_physics=9) 10 cm reflectivity as a PURE function of the
// state -- H_Z(x) for the radar observation operator.
//
// WHY THIS FILE EXISTS.  module_mp_milbrandt2mom.F computes Zet inside the
// scheme's final diagnostics block (:3400-3466), and gpuwm transcribes that
// block in milbrandt2.cu's milbrandt2_diagnostics kernel.  That kernel is
// not an observation operator: the same launch clamps Q at :3336 and
// converts every number moment from #/m3 back to #/kg (:3467-3473), so
// evaluating it would move the background it is meant to observe.  The Z
// block ITSELF touches nothing -- it reads T, pres, five masses and five
// numbers and writes ZET -- so it separates cleanly, which is exactly the
// criterion gpuwm/da/obsop.py applies to mp=18 (NSSL's radardd02, a
// separate pure diagnostic) and denies to mp=50 (P3, whose Z is fused with
// a state update it cannot be lifted out of).
//
// The block is DUPLICATED here rather than shared: cupy.RawModule has no
// #include path, and milbrandt2.cu is a byte-frozen transcription whose
// compiled source, PTX and FP contraction are pinned
// (tests/test_mp8_frozen.py).  This is the same discipline the six
// aerosol-aware Thompson translation units follow.  The duplication is held
// to the original by MEASUREMENT, not by review:
// tests/test_da_obsop_milbrandt_gpu.py::
// test_the_operator_is_the_schemes_own_z_block_bitwise drives this kernel
// and the scheme's own diagnostics over the same state and asserts the two
// ZET fields are bitwise equal.
//
// TRANSCRIPTION AUTHORITY: WRF_source_v4.6.1_group/phys/
// module_mp_milbrandt2mom.F:3400-3466, read line by line.  Transcendentals
// are IEEE (log10f, never __log10f), as in milbrandt2.cu.

#define my2z_cxr ck[147]
#define my2z_cxi ck[148]
#define my2z_Gzr ck[149]
#define my2z_Gzi ck[150]
#define my2z_Gzs ck[151]
#define my2z_Gzg ck[152]
#define my2z_Gzh ck[153]

#define MY2Z_epsQ     1.0e-14f
#define MY2Z_epsN     1.0e-3f
#define MY2Z_fdielec  4.464f
#define MY2Z_zfact    1.0e+18f
#define MY2Z_minZET   (-99.0f)
#define MY2Z_TRPL     0.27316e+3f
#define MY2Z_RGASD    0.28705e+3f

// concentration_space != 0: the number moments are already per unit VOLUME
// (#/m3), the convention that holds INSIDE a scheme call between
// milbrandt2_prelim (:1228-1233) and milbrandt2_diagnostics (:3467-3473).
// concentration_space == 0: they are per unit MASS (#/kg), which is what a
// DomainState carries between steps, and each is multiplied by de here --
// the same de (:3400) the Fortran forms, so the two conventions meet at the
// scheme's own arithmetic rather than at a caller's.
extern "C" __global__
void milbrandt2_zet(
        const float* __restrict__ T,
        const float* __restrict__ QR, const float* __restrict__ QI,
        const float* __restrict__ QN, const float* __restrict__ QG,
        const float* __restrict__ QH,
        const float* __restrict__ NR, const float* __restrict__ NY,
        const float* __restrict__ NN, const float* __restrict__ NG,
        const float* __restrict__ NH,
        const float* __restrict__ pres,
        float* __restrict__ ZET,
        const float* __restrict__ ck,
        int concentration_space,
        int nz, int ny, int nx)
{
    long long n = (long long)nz * ny * nx;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float t = T[gid], pr = pres[gid];
    float de = pr / (MY2Z_RGASD * t);                        // :3400
    float tmp9 = de * de;

    float N_r = NR[gid], N_i = NY[gid];
    float N_s = NN[gid], N_g = NG[gid], N_h = NH[gid];
    if (concentration_space == 0) {
        N_r = N_r * de;
        N_i = N_i * de;
        N_s = N_s * de;
        N_g = N_g * de;
        N_h = N_h * de;
    }
    float qr = QR[gid], qi = QI[gid], qn = QN[gid];
    float qg = QG[gid], qh = QH[gid];

    float tmp1 = 0.f, tmp2 = 0.f, tmp3 = 0.f, tmp4 = 0.f, tmp5 = 0.f;
    if (qr > MY2Z_epsQ && N_r > MY2Z_epsN) tmp1 = my2z_cxr * my2z_Gzr * tmp9
                                                  * qr * qr / N_r;
    if (qi > MY2Z_epsQ && N_i > MY2Z_epsN) tmp2 = my2z_cxi * my2z_Gzi * tmp9
                                                  * qi * qi / N_i;
    if (qn > MY2Z_epsQ && N_s > MY2Z_epsN) tmp3 = my2z_cxi * my2z_Gzs * tmp9
                                                  * qn * qn / N_s;
    if (qg > MY2Z_epsQ && N_g > MY2Z_epsN) tmp4 = my2z_cxi * my2z_Gzg * tmp9
                                                  * qg * qg / N_g;
    if (qh > MY2Z_epsQ && N_h > MY2Z_epsN) tmp5 = my2z_cxi * my2z_Gzh * tmp9
                                                  * qh * qh / N_h;
    if (t > MY2Z_TRPL) {
        tmp2 = tmp2 * MY2Z_fdielec;
        tmp3 = tmp3 * MY2Z_fdielec;
        tmp4 = tmp4 * MY2Z_fdielec;
        tmp5 = tmp5 * MY2Z_fdielec;
    }
    float zet = tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
    if (zet > 0.0f) {
        zet = 10.0f * log10f(zet * MY2Z_zfact);
    } else {
        zet = MY2Z_minZET;
    }
    ZET[gid] = fmaxf(zet, MY2Z_minZET);
}
