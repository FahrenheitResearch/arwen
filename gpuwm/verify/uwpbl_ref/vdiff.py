"""WRF diffusion_solver, non-molecular WRF_PORT path."""
import math
from gpuwm.core.uwpbl_constants import GRAVIT, RAIR
from .fortran import fmax

def vd_lu_decomp(pver, ksrf, kv, tmpi, rpdel, ztodt, cc_top, ca, cc, dnom, ze, ntop, nbot):
    for k in range(nbot-1, ntop-1, -1):
        ca[k] = kv[k+1]*tmpi[k+1]*rpdel[k]
        cc[k+1] = kv[k+1]*tmpi[k+1]*rpdel[k+1]
    ca[nbot] = 0.0
    dnom[nbot] = 1.0/(1.0+cc[nbot]+ksrf[1]*ztodt*GRAVIT*rpdel[nbot])
    ze[nbot] = cc[nbot]*dnom[nbot]
    for k in range(nbot-1, ntop, -1):
        dnom[k] = 1.0/(1.0+ca[k]+cc[k]-ca[k]*ze[k+1])
        ze[k] = cc[k]*dnom[k]
    dnom[ntop] = 1.0/(1.0+ca[ntop]+cc_top[1]-ca[ntop]*ze[ntop+1])

def vd_lu_solve(pver, q, ca, ze, dnom, ntop, nbot, cd_top):
    zf = [0.0]*(pver+1)
    zf[nbot] = q[nbot]*dnom[nbot]
    for k in range(nbot-1, ntop, -1):
        zf[k] = (q[k]+ca[k]*zf[k+1])*dnom[k]
    k = ntop
    zf[k] = (q[k]+cd_top[1]+ca[k]*zf[k+1])*dnom[k]
    q[ntop] = zf[ntop]
    for k in range(ntop+1, nbot+1):
        q[k] = zf[k]+ze[k]*q[k-1]

def compute_vdiff(pver, ncnst, pmid, pint, rpdel, t, ztodt, taux, tauy, shflx, cflx, ntop, nbot, kvh, kvm, kvq, cgs, cgh, zi, ksrftms, qmincg, fieldlist, u, v, q, dse, tautmsx, tautmsy, dtk, topflx, tauresx, tauresy, itaures):
    momentum = fieldlist['u'] or fieldlist['v']
    if momentum and not fieldlist['s']:
        return 'diffusion_solver.compute_vdiff: must diffuse s if diffusing u or v'
    rhoi, tmpi2, tmpi1, du, dv = ([0.0]*(pver+2) for _ in range(5))
    ca, cc, dnom, ze = ([0.0]*(pver+1) for _ in range(4))
    rhoi[1] = pint[1]/(RAIR*t[1])
    for k in range(2, pver+1):
        tint = 0.5*(t[k]+t[k-1])
        rhoi[k] = pint[k]/(RAIR*tint)
        grho = GRAVIT*rhoi[k]
        tmpi2[k] = ztodt*(grho*grho)/(pmid[k]-pmid[k-1])
    rhoi[pver+1] = pint[pver+1]/(RAIR*t[pver])
    rrho = RAIR*t[pver]/pmid[pver]
    tmp1 = ztodt*GRAVIT*rpdel[pver]
    zero = [None, 0.0]
    if momentum:
        du[pver+1], dv[pver+1] = -u[pver], -v[pver]
        for k in range(2, pver+1):
            du[k], dv[k] = u[k]-u[k-1], v[k]-v[k-1]
        ws = fmax(math.sqrt(math.pow(u[pver],2.0)+math.pow(v[pver],2.0)),1.0)
        tau = math.sqrt(math.pow(taux[1],2.0)+math.pow(tauy[1],2.0))
        ksrf = fmax(tau/ws,1.e-4)+ksrftms[1]
        usum, vsum = 0.0, 0.0
        for k in range(1, pver+1):
            usum = usum+(1.0/GRAVIT)*u[k]/rpdel[k]
            vsum = vsum+(1.0/GRAVIT)*v[k]/rpdel[k]
        ramda = ztodt/7200.0
        u[pver] = u[pver]+tmp1*tauresx[1]*ramda
        v[pver] = v[pver]+tmp1*tauresy[1]*ramda
        vd_lu_decomp(pver,[None,ksrf],kvm,tmpi2,rpdel,ztodt,zero,ca,cc,dnom,ze,ntop,nbot)
        vd_lu_solve(pver,u,ca,ze,dnom,ntop,nbot,zero)
        vd_lu_solve(pver,v,ca,ze,dnom,ntop,nbot,zero)
        tautmsx[1] = -(ksrftms[1]*u[pver])
        tautmsy[1] = -(ksrftms[1]*v[pver])
        usout, vsout = 0.0, 0.0
        for k in range(1, pver+1):
            usout = usout+(1.0/GRAVIT)*u[k]/rpdel[k]
            vsout = vsout+(1.0/GRAVIT)*v[k]/rpdel[k]
        tx, ty = (usout-usum)/ztodt, (vsout-vsum)/ztodt
        if itaures == 1:
            tauresx[1] = taux[1]+tautmsx[1]+tauresx[1]-tx
            tauresy[1] = tauy[1]+tautmsy[1]+tauresy[1]-ty
        tmpi1[pver+1] = 0.5*ztodt*GRAVIT*((-u[pver]+du[pver+1])*tx+(-v[pver]+dv[pver+1])*ty)
        for k in range(2,pver+1):
            dout_u, dout_v = u[k]-u[k-1], v[k]-v[k-1]
            tmpi1[k] = 0.25*tmpi2[k]*kvm[k]*(dout_u*dout_u+dout_v*dout_v+dout_u*du[k]+dout_v*dv[k])
        for k in range(1,pver+1):
            dtk[k] = (tmpi1[k+1]+tmpi1[k])*rpdel[k]
            dse[k] = dse[k]+dtk[k]
    if fieldlist['s']:
        for k in range(1,pver+1):
            dse[k] = dse[k]+ztodt*rpdel[k]*GRAVIT*(rhoi[k+1]*kvh[k+1]*cgh[k+1]-rhoi[k]*kvh[k]*cgh[k])
        dse[pver] = dse[pver]+tmp1*shflx[1]
        vd_lu_decomp(pver,zero,kvh,tmpi2,rpdel,ztodt,zero,ca,cc,dnom,ze,ntop,nbot)
        vd_lu_solve(pver,dse,ca,ze,dnom,ntop,nbot,zero)
    need_decomp = True
    for m in range(1,ncnst+1):
        if fieldlist['q'][m]:
            qtm = q[m].copy()
            for k in range(1,pver+1):
                q[m][k] = q[m][k]+ztodt*rpdel[k]*GRAVIT*(cflx[m]*rrho)*(rhoi[k+1]*kvh[k+1]*cgs[k+1]-rhoi[k]*kvh[k]*cgs[k])
            lqtst = all(q[m][k] >= qmincg[m] for k in range(1,pver+1))
            for k in range(1,pver+1):
                q[m][k] = q[m][k] if lqtst else qtm[k]
            q[m][pver] = q[m][pver]+tmp1*cflx[m]
            if need_decomp:
                vd_lu_decomp(pver,zero,kvq,tmpi2,rpdel,ztodt,zero,ca,cc,dnom,ze,ntop,nbot)
                need_decomp = False
            vd_lu_solve(pver,q[m],ca,ze,dnom,ntop,nbot,zero)
    return ''
