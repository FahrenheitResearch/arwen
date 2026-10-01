// Finalization reference from exported base d6929cb8d.
extern "C" __global__
void base_morrison_finalize_levels(real* __restrict__ theta,
                              real* __restrict__ qv,
                              real* __restrict__ qc,
                              real* __restrict__ qr,
                              real* __restrict__ qi,
                              real* __restrict__ qs,
                              real* __restrict__ qg,
                              real* __restrict__ nc,
                              real* __restrict__ nr,
                              real* __restrict__ ni,
                              real* __restrict__ ns,
                              real* __restrict__ ng,
                              const real* __restrict__ rho_in,
                              const real* __restrict__ pii,
                              const real* __restrict__ pressure,
                              const real* __restrict__ ice_to_snow_scratch,
                              real* __restrict__ effc,
                              real* __restrict__ effi,
                              real* __restrict__ effs,
                              real* __restrict__ effr,
                              real morr_rhog,
                              int ncell)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ncell) return;
    real temp = theta[idx] * pii[idx];
    bool ice_to_snow = ice_to_snow_scratch[idx] != 0.0f;
    real rhoa = rho_in[idx];
    real qvk = qv[idx], qck = qc[idx], qrk = qr[idx];
    real qik = qi[idx], qsk = qs[idx], qgk = qg[idx];
    real nck = nc[idx], nrk = nr[idx], nik = ni[idx];
    real nsk = ns[idx], ngk = ng[idx];

    // The conversion mask already latched WRF's pre-update T3D test at
    // 3679-3689; the application does not recheck post-update T.
    if (ice_to_snow && qik > 0.0f) {
        qsk += qik; nsk += nik; qik = 0.0f; nik = 0.0f;
    }
    // XXLV/XXLS/CPM are the stale per-level values diagnosed before
    // any process update (WRF 1298-1304, reused at 3729-3849).
    real xlv = effc[idx];
    real xls = xlv + 0.3353e6f;
    real xlf = xls - xlv;
    real cpm = effi[idx];
    real ew = fminf(0.99f * pressure[idx], morr_polysvp(temp, false));
    real ei = fminf(ew, fminf(0.99f * pressure[idx], morr_polysvp(temp, true)));
    real qvs = EP2 * ew / (pressure[idx] - ew);
    real qvi = EP2 * ei / (pressure[idx] - ei);
    if (qvk / qvs < 0.9f) {
        if (qrk < 1.0e-8f) { qvk += qrk; temp -= qrk * xlv / cpm; qrk = 0.0f; }
        if (qck < 1.0e-8f) { qvk += qck; temp -= qck * xlv / cpm; qck = 0.0f; }
    }
    if (qvk / qvi < 0.9f) {
        if (qik < 1.0e-8f) { qvk += qik; temp -= qik * xls / cpm; qik = 0.0f; }
        if (qsk < 1.0e-8f) { qvk += qsk; temp -= qsk * xls / cpm; qsk = 0.0f; }
        if (qgk < 1.0e-8f) { qvk += qgk; temp -= qgk * xls / cpm; qgk = 0.0f; }
    }
    if (qck < MQSMALL) { qck = 0.0f; nck = 0.0f; }
    if (qrk < MQSMALL) { qrk = 0.0f; nrk = 0.0f; }
    if (qik < MQSMALL) { qik = 0.0f; nik = 0.0f; }
    if (qsk < MQSMALL) { qsk = 0.0f; nsk = 0.0f; }
    if (qgk < MQSMALL) { qgk = 0.0f; ngk = 0.0f; }

    if (qik >= MQSMALL && temp >= 273.15f) {
        qrk += qik; nrk += nik; temp -= qik * xlf / cpm;
        qik = 0.0f; nik = 0.0f;
    }
    if (temp <= 233.15f && qck >= MQSMALL) {
        qik += qck; nik += nck; temp += qck * xlf / cpm;
        qck = 0.0f; nck = 0.0f;
    }
    if (temp <= 233.15f && qrk >= MQSMALL) {
        qgk += qrk; ngk += nrk; temp += qrk * xlf / cpm;
        qrk = 0.0f; nrk = 0.0f;
    }

    qv[idx] = qvk;
    qc[idx] = qck; qr[idx] = qrk; qi[idx] = qik;
    qs[idx] = qsk; qg[idx] = qgk;
    nc[idx] = nck; nr[idx] = nrk; ni[idx] = nik;
    ns[idx] = nsk; ng[idx] = ngk;
    // Final PSD reconstruction rebounds LAMC from the transient updated
    // NC3D (WRF 3918-3947).  Fixed 250 cm-3 is restored only after EFFC.
    MorrMoments m = morr_bound(qc[idx], qr[idx], qi[idx], qs[idx], qg[idx],
                                rhoa, pressure[idx], temp,
                                &nc[idx], &nr[idx], &ni[idx],
                                &ns[idx], &ng[idx], false, morr_rhog);
    effc[idx] = qc[idx] >= MQSMALL
              ? (m.pg + 3.0f) / (2.0f * m.lc) * 1.0e6f : 25.0f;
    effr[idx] = qr[idx] >= MQSMALL ? 1.5e6f / m.lr : 25.0f;
    effi[idx] = qi[idx] >= MQSMALL ? 1.5e6f / m.li : 25.0f;
    effs[idx] = qs[idx] >= MQSMALL ? 1.5e6f / m.ls : 25.0f;
    ni[idx] = fminf(ni[idx], 0.3e6f / rhoa);
    nc[idx] = 250.0e6f / rhoa;
    theta[idx] = temp / pii[idx];
}
