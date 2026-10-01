"""Literal one-column WRF 4.7.1 eddy_diff transcription, 1-based lists.

Only the configured sftype='l', choice_evhc=choice_radf='maxi' path is used.
Dependencies are imported at call time so stage replay can run independently.
"""
import math

from gpuwm.core.uwpbl_constants import CPAIR, GRAVIT, RAIR, ZVIR, LATVAP, LATICE
from .fortran import fmax, fmin, zeros


def sfdiag(pver, qt, ql, sl, pi, pm, zi, cld, sfi, sfuh, sflh,
           slslope, qtslope):
    # module_cam_bl_eddy_diff.F:920-957. maxi overrides the qmin test.
    sfi[1:] = [0.0] * (pver + 1)
    sfuh[1:] = [0.0] * pver
    sflh[1:] = [0.0] * pver
    for k in range(2, pver + 1):
        sfuh[k] = cld[k]
        sflh[k] = cld[k]
        sfi[k] = 0.5 * (sflh[k-1] + fmin(sfuh[k], sflh[k-1]))
    sfi[pver+1] = sflh[pver]


def trbintd(pver, z, u, v, t, pmid, taux, tauy, ustar, rrho, s2, n2,
            ri, zi, pi, cld, qt, qv, ql, qi, sfi, sfuh, sflh, sl, slv,
            slslope, qtslope, chs, chu, cms, cmu, minpblh):
    from .wvsat import fqsatd
    latsub = LATVAP + LATICE  # init_eddy_diff: runtime addition
    rrho[1] = RAIR * t[pver] / pmid[pver]
    ustar[1] = fmax(math.sqrt(math.sqrt(taux[1]*taux[1] + tauy[1]*tauy[1]) * rrho[1]), 0.01)
    minpblh[1] = 100.0 * ustar[1]
    for k in range(1, pver + 1):
        status, es, qs, gam = fqsatd(t[k], pmid[k])
        qt[k] = qv[k] + ql[k] + qi[k]
        sl[k] = CPAIR * t[k] + GRAVIT * z[k] - LATVAP * ql[k] - latsub * qi[k]
        slv[k] = sl[k] * (1.0 + ZVIR * qt[k])
        bfact = GRAVIT / (t[k] * (1.0 + ZVIR * qv[k] - ql[k] - qi[k]))
        chu[k] = (1.0 + ZVIR * qt[k]) * bfact / CPAIR
        chs[k] = ((1.0 + (1.0 + ZVIR) * gam * CPAIR * t[k] / LATVAP) / (1.0 + gam)) * bfact / CPAIR
        cmu[k] = ZVIR * bfact * t[k]
        cms[k] = LATVAP * chs[k] - bfact * t[k]
    for a in (chu, chs, cmu, cms):
        a[pver+1] = a[pver]
    slslope[pver] = (sl[pver] - sl[pver-1]) / (pmid[pver] - pmid[pver-1])
    qtslope[pver] = (qt[pver] - qt[pver-1]) / (pmid[pver] - pmid[pver-1])
    slslope[1] = (sl[2] - sl[1]) / (pmid[2] - pmid[1])
    qtslope[1] = (qt[2] - qt[1]) / (pmid[2] - pmid[1])
    dsldp_b = slslope[1]
    dqtdp_b = qtslope[1]
    for k in range(2, pver):
        dsldp_a = dsldp_b
        dqtdp_a = dqtdp_b
        dsldp_b = (sl[k+1] - sl[k]) / (pmid[k+1] - pmid[k])
        dqtdp_b = (qt[k+1] - qt[k]) / (pmid[k+1] - pmid[k])
        product = dsldp_a * dsldp_b
        if product <= 0.0:
            slslope[k] = 0.0
        elif product > 0.0 and dsldp_a < 0.0:
            slslope[k] = fmax(dsldp_a, dsldp_b)
        elif product > 0.0 and dsldp_a > 0.0:
            slslope[k] = fmin(dsldp_a, dsldp_b)
        product = dqtdp_a * dqtdp_b
        if product <= 0.0:
            qtslope[k] = 0.0
        elif product > 0.0 and dqtdp_a < 0.0:
            qtslope[k] = fmax(dqtdp_a, dqtdp_b)
        elif product > 0.0 and dqtdp_a > 0.0:
            qtslope[k] = fmin(dqtdp_a, dqtdp_b)
    sfdiag(pver, qt, ql, sl, pi, pmid, zi, cld, sfi, sfuh, sflh, slslope, qtslope)
    for k in range(pver, 1, -1):
        km1 = k - 1
        rdz = 1.0 / (z[km1] - z[k])
        dsldz = (sl[km1] - sl[k]) * rdz
        dqtdz = (qt[km1] - qt[k]) * rdz
        chu[k] = (chu[km1] + chu[k]) * 0.5
        chs[k] = (chs[km1] + chs[k]) * 0.5
        cmu[k] = (cmu[km1] + cmu[k]) * 0.5
        cms[k] = (cms[km1] + cms[k]) * 0.5
        ch = chu[k] * (1.0 - sfi[k]) + chs[k] * sfi[k]
        cm = cmu[k] * (1.0 - sfi[k]) + cms[k] * sfi[k]
        n2[k] = ch * dsldz + cm * dqtdz
        du = u[km1] - u[k]
        dv = v[km1] - v[k]
        s2[k] = (du*du + dv*dv) * (rdz*rdz)
        s2[k] = fmax(1.e-12, s2[k])
        ri[k] = n2[k] / s2[k]
    n2[1], s2[1], ri[1] = n2[2], s2[2], ri[2]


def _retrieve(pver, slfd, qtfd, qi, z, pmid, tfd, qvfd, qlfd):
    """Lines 698-722, also exposed for independent spy replay."""
    from .wvsat import fqsatd
    latsub = LATVAP + LATICE
    for k in range(1, pver + 1):
        templ = (slfd[k] - GRAVIT*z[k]) / CPAIR
        status, es, qs, gam = fqsatd(templ, pmid[k])
        temps = templ + (qtfd[k] - qs) / (CPAIR / LATVAP + LATVAP * qs / (RAIR * (templ*templ)))
        status, es, qs, gam = fqsatd(temps, pmid[k])
        qlfd[k] = fmax(qtfd[k] - qi[k] - qs, 0.0)
        qvfd[k] = fmax(0.0, qtfd[k] - qi[k] - qlfd[k])
        tfd[k] = (slfd[k] + LATVAP * qlfd[k] + latsub * qi[k] - GRAVIT*z[k]) / CPAIR


def _relax(pver, kvm_out, kvh_out, kvm, kvh):
    for k in range(1, pver + 2):
        kvm_out[k] = 0.5 * kvm_out[k] + (1.0 - 0.5) * kvm[k]
        kvh_out[k] = 0.5 * kvh_out[k] + (1.0 - 0.5) * kvh[k]


def _error(pver, kvh, kvh_out):
    error = 0.0
    for k in range(1, pver + 1):
        delta = kvh[k] - kvh_out[k]
        error = error + delta*delta
    return math.sqrt(error / pver)


def compute_eddy_diff(pver, t, qv, ztodt, ql, qi, s, rpdel, cldn, qrl,
                      wsedl, z, zi, pmid, pi, u, v, taux, tauy, shflx, qflx,
                      wstarent, nturb, ustar, pblh, kvm_in, kvh_in, kvm_out,
                      kvh_out, kvq, cgh, cgs, tpert, qpert, wpert, tke, bprod,
                      sprod, sfi, kvinit, tauresx, tauresy, ksrftms, ipbl,
                      kpblh, wstarPBL, turbtype, sm_aw):
    from .caleddy import caleddy
    from .vdiff import compute_vdiff
    # Automatic arrays, including the diagnostic outputs passed into caleddy.
    names = ('qt sl qtfd slfd slv slslope qtslope s2 n2 ri sfuh sflh '
             'chs chu cms cmu kvh kvm kvf kbase_o ktop_o kbase_mg ktop_mg '
             'kbase_f ktop_f wet web jtbu jbbu evhc jt2slv n2ht n2hb lwp '
             'opt_depth radinvfrac radf wstar wstar3fact ebrk wbrk lbrk ricl '
             'ghcl shcl smcl ghi shi smi rii lengi wcap').split()
    a = {name: zeros(pver+1) for name in names}
    rrho, minpblh, tkes, pblhp = (zeros(1) for _ in range(4))
    ncvfin_o, ncvfin_mg, ncvfin_f = (zeros(1) for _ in range(3))
    ufd, vfd, tfd, qvfd, qlfd = (x.copy() for x in (u, v, t, qv, ql))
    # No dry fields. The eddy module selects only constituent 1.
    fieldlist = {'u': True, 'v': True, 's': True, 'q': [None, True]}
    jnk1d, jnk2d = zeros(1), zeros(pver+1)
    for iturb in range(1, nturb + 1):
        tautotx = [None, taux[1] - ksrftms[1]*ufd[pver]]
        tautoty = [None, tauy[1] - ksrftms[1]*vfd[pver]]
        trbintd(pver, z, ufd, vfd, tfd, pmid, tautotx, tautoty, ustar,
                rrho, a['s2'], a['n2'], a['ri'], zi, pi, cldn, a['qtfd'],
                qvfd, qlfd, qi, sfi, a['sfuh'], a['sflh'], a['slfd'],
                a['slv'], a['slslope'], a['qtslope'], a['chs'], a['chu'],
                a['cms'], a['cmu'], minpblh)
        if iturb == 1:
            a['qt'] = a['qtfd'].copy()
            a['sl'] = a['slfd'].copy()
        if iturb == 1:
            a['kvh'][1:] = [0.0]*(pver+1) if kvinit else kvh_in[1:]
            a['kvm'][1:] = [0.0]*(pver+1) if kvinit else kvm_in[1:]
        else:
            a['kvh'][1:] = kvh_out[1:]
            a['kvm'][1:] = kvm_out[1:]
        caleddy(pver, a['slfd'], a['qtfd'], qlfd, a['slv'], ufd, vfd, pi,
                z, zi, qflx, shflx, a['slslope'], a['qtslope'], a['chu'],
                a['chs'], a['cmu'], a['cms'], a['sfuh'], a['sflh'], a['n2'],
                a['s2'], a['ri'], rrho, pblh, ustar, a['kvh'], a['kvm'],
                kvh_out, kvm_out, tpert, qpert, qrl, a['kvf'], tke, wstarent,
                bprod, sprod, minpblh, wpert, tkes, turbtype, sm_aw,
                a['kbase_o'], a['ktop_o'], ncvfin_o, a['kbase_mg'],
                a['ktop_mg'], ncvfin_mg, a['kbase_f'], a['ktop_f'], ncvfin_f,
                *(a[n] for n in ('wet web jtbu jbbu evhc jt2slv n2ht n2hb '
                                'lwp opt_depth radinvfrac radf wstar wstar3fact '
                                'ebrk wbrk lbrk ricl ghcl shcl smcl ghi shi smi '
                                'rii lengi wcap').split()),
                pblhp, cldn, ipbl, kpblh, wsedl)
        if iturb == nturb:
            errorPBL = _error(pver, a['kvh'], kvh_out)
        if iturb > 1 and iturb < nturb:
            _relax(pver, kvm_out, kvh_out, a['kvm'], a['kvh'])
        cgh[1:], cgs[1:] = [0.0]*(pver+1), [0.0]*(pver+1)
        if iturb < nturb:
            a['slfd'][1:] = a['sl'][1:]
            a['qtfd'][1:] = a['qt'][1:]
            ufd[1:], vfd[1:] = u[1:], v[1:]
            errstring = compute_vdiff(pver, 1, pmid, pi, rpdel, t, ztodt,
                taux, tauy, shflx, qflx, 1, pver, kvh_out, kvm_out, kvh_out,
                cgs, cgh, zi, ksrftms, zeros(1), fieldlist, ufd, vfd,
                [None, a['qtfd']], a['slfd'], jnk1d, jnk1d, jnk2d, jnk1d,
                tauresx, tauresy, 0)
            # WRF does not inspect the inner errstring.
            _retrieve(pver, a['slfd'], a['qtfd'], qi, z, pmid, tfd, qvfd, qlfd)
    kvq[1:] = kvh_out[1:]
    wstarPBL[1] = fmax(0.0, a['wstar'][1]) if ipbl[1] == 1.0 else 0.0
    # outfld is inert in the oracle. Return diagnostics for stage verification.
    return {'errorPBL': errorPBL, **a}
