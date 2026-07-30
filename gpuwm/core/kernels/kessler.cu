// gpuwm/core/kernels/kessler.cu
//
// Kessler warm-rain microphysics, transcribed line by line from WRF v4.6.1
// phys/module_mp_kessler.F (subroutine kessler; itself from the COMMAS
// code).  One thread per (i, j) column, k loops inside, exactly the
// Fortran's per-column structure:
//
//   1. terminal fall speed vt = 36.34*(max(0, 0.001*rho*qr))^0.1364 *
//      sqrt(rho[0]/rho) and the stable split count
//      nfall = max(1, nint(0.5 + crmax/0.75));
//   2. the time-split sedimentation loop: every split step records the
//      surface precipitation ppt = rho[0]*qr[0]*vt[0]*dtfall/rhowater
//      (RAINNC += ppt*1000 mm; RAINNCV overwritten -- last split step's
//      value, as the Fortran leaves it), then flux-upstream fallout on the
//      rdzk spacings (half-level z differences; the top level reuses the
//      spacing below and skips the 1/rho division -- the file's quirk,
//      kept verbatim), then recomputes vt / re-derives nfall from the
//      fallen field unless this was the last split step;
//   3. the production/adjustment sweep per cell: autoconversion +
//      accretion (c1 = 0.001 1/s above the FILE's threshold c2 = 0.001
//      kg/kg; accretion c3 = 2.2, exponent c4 = 0.875 -- evaluated on the
//      PRE-sedimentation qc/qr), the Teten saturation adjustment (es from
//      SVP1/SVP2/SVP3/SVPT0; the hardcoded 1004./287. pressure exponent
//      and 2.5e6/(1004.*pii) latent factor are the file's own literals and
//      are kept, while f5 uses the passed cp = CP), rain evaporation ern
//      (capped by the saturation deficit and by qr), and the coupled
//      t/qv/qc/qr update with its clamps.
//
// t is the FULL (dry) potential temperature (WRF use_theta_m = 0 prep),
// rho the dry density 1/(al+alb), pii the full-pressure Exner function,
// z/dz8w the half-level heights and layer depths from the full
// geopotential -- all per moist_physics_prep_em
// (dyn_em/module_big_step_utilities_em.F).  Mirror:
// gpuwm/verify/npref.py np_kessler_column.

// Per-thread column work arrays (local memory); the launcher rejects
// nz > KESS_KMAX.
#define KESS_KMAX 256

extern "C" __global__
void kessler_column(real* __restrict__ t,          // (nz, ny, nx) theta (K)
                    real* __restrict__ qv,         // (nz, ny, nx) kg/kg
                    real* __restrict__ qc,
                    real* __restrict__ qr,
                    const real* __restrict__ rho,  // (nz, ny, nx) dry density
                    const real* __restrict__ pii,  // (nz, ny, nx) Exner
                    const real* __restrict__ z,    // (nz, ny, nx) half-lvl z
                    const real* __restrict__ dz8w, // (nz, ny, nx) layer depth
                    real* __restrict__ rainnc,     // (ny, nx) accum rain (mm)
                    real* __restrict__ rainncv,    // (ny, nx) last-split (mm)
                    real dt, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;

    const real c1k = 0.001f, c2k = 0.001f, c3k = 2.2f, c4k = 0.875f;
    const real max_cr_sedimentation = 0.75f;
    const real f5 = SVP2 * (SVPT0 - SVP3) * XLV / CP;

    real prodk[KESS_KMAX], vt[KESS_KMAX], vtden[KESS_KMAX],
         rhok[KESS_KMAX], rdzk[KESS_KMAX];

    // --- terminal velocity and the stable time-split count
    real crmax = 0.0f;
    real rho0 = rho[IDX3(0, j, i)];
    for (int k = 0; k < nz; ++k) {
        prodk[k] = qr[IDX3(k, j, i)];
        rhok[k] = rho[IDX3(k, j, i)];
        real qrr = fmaxf(0.0f, qr[IDX3(k, j, i)] * 0.001f * rhok[k]);
        vtden[k] = sqrtf(rho0 / rhok[k]);
        vt[k] = 36.34f * powf(qrr, 0.1364f) * vtden[k];
        real rdzw = 1.0f / dz8w[IDX3(k, j, i)];
        crmax = fmaxf(vt[k] * dt * rdzw, crmax);
    }
    for (int k = 0; k < nz - 1; ++k)
        rdzk[k] = 1.0f / (z[IDX3(k + 1, j, i)] - z[IDX3(k, j, i)]);
    rdzk[nz - 1] = 1.0f / (z[IDX3(nz - 1, j, i)] - z[IDX3(nz - 2, j, i)]);

    int nfall = max(1, (int)floorf(0.5f + crmax / max_cr_sedimentation
                                   + 0.5f));                  // Fortran NINT
    real dtfall = dt / (real)nfall;
    real time_sediment = dt;

    // --- time-split sedimentation, fallout with flux upstream
    while (nfall > 0) {
        time_sediment -= dtfall;

        real ppt = rhok[0] * prodk[0] * vt[0] * dtfall / RHOWATER;
        rainncv[(size_t)j * nx + i] = ppt * 1000.0f;
        rainnc[(size_t)j * nx + i] += ppt * 1000.0f;          // mm

        // the in-place bottom-up update reads prodk[k+1] before it is
        // overwritten -- identical to the Fortran loop.
        for (int k = 0; k < nz - 1; ++k) {
            real factor = dtfall * rdzk[k] / rhok[k];
            prodk[k] = prodk[k] - factor * (rhok[k] * prodk[k] * vt[k]
                                            - rhok[k + 1] * prodk[k + 1]
                                              * vt[k + 1]);
        }
        prodk[nz - 1] = prodk[nz - 1] - (dtfall * rdzk[nz - 1])
                        * prodk[nz - 1] * vt[nz - 1];

        if (nfall > 1) {          // not the last split step: new vt, nfall
            nfall = nfall - 1;
            crmax = 0.0f;
            for (int k = 0; k < nz; ++k) {
                real qrr = fmaxf(0.0f, prodk[k] * 0.001f * rhok[k]);
                vt[k] = 36.34f * powf(qrr, 0.1364f) * vtden[k];
                real rdzw = 1.0f / dz8w[IDX3(k, j, i)];
                crmax = fmaxf(vt[k] * time_sediment * rdzw, crmax);
            }
            int nfall_new = max(1, (int)floorf(0.5f + crmax
                                               / max_cr_sedimentation
                                               + 0.5f));
            if (nfall_new != nfall) {
                nfall = nfall_new;
                dtfall = time_sediment / (real)nfall;
            }
        } else {                  // last split step: prodk is the fallen qr
            nfall = 0;
        }
    }

    // --- production of rain from qc, saturation adjustment, evaporation
    for (int k = 0; k < nz; ++k) {
        size_t idx = IDX3(k, j, i);
        real qrk = qr[idx];                     // PRE-sedimentation qr
        real qck = qc[idx];
        real qvk = qv[idx];
        real tk = t[idx];
        real piik = pii[idx];

        real factorn = 1.0f / (1.0f + c3k * dt
                               * powf(fmaxf(0.0f, qrk), c4k));
        real qrprod = qck * (1.0f - factorn)
                    + factorn * c1k * dt * fmaxf(qck - c2k, 0.0f);
        real rcgs = 0.001f * rho[idx];

        qck = fmaxf(qck - qrprod, 0.0f);
        qrk = prodk[k];                         // qr = (qr + prod - qr)
        qrk = fmaxf(qrk + qrprod, 0.0f);

        real temp = piik * tk;
        real pressure = 1.0e5f * powf(piik, 1004.0f / 287.0f);
        real gam = 2.5e6f / (1004.0f * piik);
        real es = 1000.0f * SVP1 * expf(SVP2 * (temp - SVPT0)
                                        / (temp - SVP3));
        real qvs = EP2 * es / (pressure - es);
        real prod = (qvk - qvs)
                  / (1.0f + pressure / (pressure - es) * qvs * f5
                            / ((temp - SVP3) * (temp - SVP3)));
        real ern = fminf(fminf(
            dt * (((1.6f + 124.9f * powf(rcgs * qrk, 0.2046f))
                   * powf(rcgs * qrk, 0.525f))
                  / (2.55e8f / (pressure * qvs) + 5.4e5f))
               * (fmaxf(qvs - qvk, 0.0f) / (rcgs * qvs)),
            fmaxf(-prod - qck, 0.0f)), qrk);

        real product = fmaxf(prod, -qck);
        t[idx]  = tk + gam * (product - ern);
        qv[idx] = fmaxf(qvk - product + ern, 0.0f);
        qc[idx] = qck + product;
        qr[idx] = qrk - ern;
    }
}
