"""Literal binary64 UW PBL transcription. Source comments identify WRF lines."""
import math
from gpuwm.core.uwpbl_constants import CPAIR, RAIR, ZVIR, LATVAP, LATICE, GRAVIT, KARMAN, B123
from .fortran import fmax as uw_fmax, fmin as uw_fmin, F32, fint, powi

def caleddy(pver, sl, qt, ql, slv, u, v, pi, z, zi, qflx, shflx, slslope, qtslope, chu, chs, cmu, cms, sfuh, sflh, n2, s2, ri, rrho, pblh, ustar, kvh_in, kvm_in, kvh, kvm, tpert, qpert, qrlin, kvf, tke, wstarent, bprod, sprod, minpblh, wpert, tkes, turbtype_f, sm_aw, kbase_o, ktop_o, ncvfin_o, kbase_mg, ktop_mg, ncvfin_mg, kbase_f, ktop_f, ncvfin_f, wet_CL, web_CL, jtbu_CL, jbbu_CL, evhc_CL, jt2slv_CL, n2ht_CL, n2hb_CL, lwp_CL, opt_depth_CL, radinvfrac_CL, radf_CL, wstar_CL, wstar3fact_CL, ebrk, wbrk, lbrk, ricl, ghcl, shcl, smcl, gh_a, sh_a, sm_a, ri_a, leng, wcap, pblhp, cld, ipbl, kpblh, wsedl):
    ncvmax = pver
    nbot_turb = pver
    belongcv = [0] * ((pver + 1) + 1)
    belongst = [0] * ((pver + 1) + 1)
    extend = [False, False]
    extend_up = [False, False]
    extend_dn = [False, False]
    ncvfin = [0, 0]
    kbase = [0] * (ncvmax + 1)
    ktop = [0] * (ncvmax + 1)
    turbtype = [0] * ((pver + 1) + 1)
    ktopbl = [0, 0]
    bflxs = [0, 0]
    qrlw = [0] * (pver + 1)
    cldeff = [0] * (pver + 1)
    cpair = CPAIR
    rair = RAIR
    zvir = ZVIR
    latvap = LATVAP
    latice = LATICE
    g = GRAVIT
    vk = KARMAN
    b123 = B123
    latsub = latvap + latice
    ccon = (float.fromhex('0x1.1000000000000p+3') * float.fromhex('0x1.999999999999ap-4')) * vk
    for uw_k in range(1, pver + 1):  # F:1637 qrlw(:ncol,:pver) = qrlin(:ncol,:pver)
        qrlw[uw_k] = qrlin[uw_k]
    for k in range(1, pver + 1, 1):  # F:1645 do k = 1, pver
        cldeff[k] = cld[k]  # F:1650 cldeff(i,k) = cld(i,k)
    alph4exs = float.fromhex('-0x1.88240b780346ep+2')  # F:1659 alph4exs = alph4
    ghmin = float.fromhex('-0x1.c4467381d7dbfp+1')  # F:1660 ghmin    = -3.5334_r8
    for uw_k in range(1, ncvmax + 1):  # F:1677 wet_CL(i,:ncvmax)        = 0._r8
        wet_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1678 web_CL(i,:ncvmax)        = 0._r8
        web_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1679 jtbu_CL(i,:ncvmax)       = 0._r8
        jtbu_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1680 jbbu_CL(i,:ncvmax)       = 0._r8
        jbbu_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1681 evhc_CL(i,:ncvmax)       = 0._r8
        evhc_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1682 jt2slv_CL(i,:ncvmax)     = 0._r8
        jt2slv_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1683 n2ht_CL(i,:ncvmax)       = 0._r8
        n2ht_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1684 n2hb_CL(i,:ncvmax)       = 0._r8
        n2hb_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1685 lwp_CL(i,:ncvmax)        = 0._r8
        lwp_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1686 opt_depth_CL(i,:ncvmax)  = 0._r8
        opt_depth_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1687 radinvfrac_CL(i,:ncvmax) = 0._r8
        radinvfrac_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1688 radf_CL(i,:ncvmax)       = 0._r8
        radf_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1689 wstar_CL(i,:ncvmax)      = 0._r8
        wstar_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1690 wstar3fact_CL(i,:ncvmax) = 0._r8
        wstar3fact_CL[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1691 ricl(i,:ncvmax)          = 0._r8
        ricl[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1692 ghcl(i,:ncvmax)          = 0._r8
        ghcl[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1693 shcl(i,:ncvmax)          = 0._r8
        shcl[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1694 smcl(i,:ncvmax)          = 0._r8
        smcl[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1695 ebrk(i,:ncvmax)          = 0._r8
        ebrk[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1696 wbrk(i,:ncvmax)          = 0._r8
        wbrk[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, ncvmax + 1):  # F:1697 lbrk(i,:ncvmax)          = 0._r8
        lbrk[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, (pver + 1) + 1):  # F:1698 gh_a(i,:pver+1)          = 0._r8
        gh_a[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, (pver + 1) + 1):  # F:1699 sh_a(i,:pver+1)          = 0._r8
        sh_a[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, (pver + 1) + 1):  # F:1700 sm_a(i,:pver+1)          = 0._r8
        sm_a[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, (pver + 1) + 1):  # F:1701 ri_a(i,:pver+1)          = 0._r8
        ri_a[uw_k] = float.fromhex('0x0.0p+0')
    for uw_k in range(1, (pver + 1) + 1):  # F:1702 sm_aw(i,:pver+1)         = 0._r8
        sm_aw[uw_k] = float.fromhex('0x0.0p+0')
    ipbl[1] = float.fromhex('0x0.0p+0')  # F:1703 ipbl(i)                  = 0._r8
    kpblh[1] = float(pver)  # F:1704 kpblh(i)                 = real(pver,r8)
    for k in range(1, (pver + 1) + 1, 1):  # F:1713 do k = 1, pver + 1
        kvh[k] = float.fromhex('0x0.0p+0')  # F:1720 kvh(i,k) = 0._r8
        kvm[k] = float.fromhex('0x0.0p+0')  # F:1721 kvm(i,k) = 0._r8
        wcap[k] = float.fromhex('0x0.0p+0')  # F:1724 wcap(i,k) = 0._r8
        leng[k] = float.fromhex('0x0.0p+0')  # F:1725 leng(i,k) = 0._r8
        tke[k] = float.fromhex('0x0.0p+0')  # F:1726 tke(i,k)  = 0._r8
        turbtype[k] = 0  # F:1727 turbtype(i,k) = 0
    for k in range(2, pver + 1, 1):  # F:1739 do k = 2, pver
        bprod[k] = (-(kvh_in[k] * n2[k]))  # F:1741 bprod(i,k) = -kvh_in(i,k) * n2(i,k)
        sprod[k] = (kvm_in[k] * s2[k])  # F:1742 sprod(i,k) =  kvm_in(i,k) * s2(i,k)
    bprod[1] = float.fromhex('0x0.0p+0')  # F:1762 bprod(i,1) = 0._r8
    sprod[1] = float.fromhex('0x0.0p+0')  # F:1763 sprod(i,1) = 0._r8
    ch = ((chu[(pver + 1)] * (float.fromhex('0x1.0000000000000p+0') - sflh[pver])) + (chs[(pver + 1)] * sflh[pver]))  # F:1764 ch = chu(i,pver+1) * ( 1._r8 - sflh(i,pver) ) + chs(i,pver+1) * sflh(i,pver)
    cm = ((cmu[(pver + 1)] * (float.fromhex('0x1.0000000000000p+0') - sflh[pver])) + (cms[(pver + 1)] * sflh[pver]))  # F:1765 cm = cmu(i,pver+1) * ( 1._r8 - sflh(i,pver) ) + cms(i,pver+1) * sflh(i,pver)
    bflxs[1] = (((ch * shflx[1]) * rrho[1]) + ((cm * qflx[1]) * rrho[1]))  # F:1766 bflxs(i) = ch * shflx(i) * rrho(i) + cm * qflx(i) * rrho(i)
    bprod[(pver + 1)] = bflxs[1]  # F:1768 bprod(i,pver+1) = bflxs(i)
    sprod[(pver + 1)] = (powi(ustar[1], 3) / (vk * z[pver]))  # F:1772 sprod(i,pver+1) = (ustar(i)**3)/(vk*z(i,pver))
    exacol(pver, ri, bflxs, minpblh, zi, ktop, kbase, ncvfin)  # F:1787 call exacol( pcols, pver, ncol, ri, bflxs, minpblh, zi, ktop, kbase, ncvfin )
    for k in range(1, ncvmax + 1, 1):  # F:1792 do k = 1, ncvmax
        kbase_o[k] = float(kbase[k])  # F:1793 kbase_o(i,k) = real(kbase(i,k),r8)
        ktop_o[k] = float(ktop[k])  # F:1794 ktop_o(i,k)  = real(ktop(i,k),r8)
        ncvfin_o[1] = float(ncvfin[1])  # F:1795 ncvfin_o(i)  = real(ncvfin(i),r8)
    tkes[1] = math.pow(uw_fmax((((float.fromhex('0x1.7333333333333p+2') * vk) * z[pver]) * (bprod[(pver + 1)] + sprod[(pver + 1)])), float.fromhex('0x1.ad7f29abcaf48p-24')), float.fromhex('0x1.5555555555555p-1'))  # F:1822 tkes(i) = max(b1*vk*z(i,pver)*(bprod(i,pver+1)+sprod(i,pver+1)), 1.e-7_r8)**(2._r8/3._r8)
    tkes[1] = uw_fmin(tkes[1], float.fromhex('0x1.4000000000000p+4'))  # F:1823 tkes(i) = min(tkes(i), tkemax)
    tke[(pver + 1)] = tkes[1]  # F:1824 tke(i,pver+1)  = tkes(i)
    wcap[(pver + 1)] = (tkes[1] / float.fromhex('0x1.7333333333333p+2'))  # F:1825 wcap(i,pver+1) = tkes(i)/b1
    ncvsurf = 0  # F:1856 ncvsurf = 0
    if (ncvfin[1] > 0):  # F:1857 if( ncvfin(i) .gt. 0 ) then
        zisocl(pver, z, zi, n2, s2, bprod, sprod, bflxs, tkes, ncvfin, kbase, ktop, belongcv, ricl, ghcl, shcl, smcl, lbrk, wbrk, ebrk, extend, extend_up, extend_dn)  # F:1858 call zisocl( pcols  , pver     , i        , z      , zi       , n2       , s2      , bprod  , sprod    , bflxs    , tkes    , ncvfin , kbase    , ktop     , belongcv, ricl   , ghcl     , shcl     , smcl    , lbrk   , wbrk     , ebrk     , extend , extend_up, extend_dn )
        if (kbase[1] == (pver + 1)):  # F:1865 if( kbase(i,1) .eq. pver + 1 ) ncvsurf = 1
            ncvsurf = 1  # F:1865 if( kbase(i,1) .eq. pver + 1 ) ncvsurf = 1
    else:
        for uw_k in range(1, (pver + 1) + 1):  # F:1867 belongcv(i,:) = .false.
            belongcv[uw_k] = False
    for k in range(1, ncvmax + 1, 1):  # F:1873 do k = 1, ncvmax
        kbase_mg[k] = F32(float(kbase[k]))  # F:1874 kbase_mg(i,k) = real(kbase(i,k))
        ktop_mg[k] = F32(float(ktop[k]))  # F:1875 ktop_mg(i,k)  = real(ktop(i,k))
        ncvfin_mg[1] = F32(float(ncvfin[1]))  # F:1876 ncvfin_mg(i)  = real(ncvfin(i))
    ncv = 1  # F:1921 ncv  = 1
    ncvf = ncvfin[1]  # F:1922 ncvf = ncvfin(i)
    for k in range(nbot_turb, 2 - 1, -1):  # F:1926 do k = nbot_turb, ntop_turb + 1, -1
        if ((((ql[k] > float.fromhex('0x1.4f8b588e368f1p-17')) and (ql[(k - 1)] < float.fromhex('0x1.4f8b588e368f1p-17'))) and (qrlw[k] < float.fromhex('0x0.0p+0'))) and (ri[k] >= float.fromhex('0x1.851eb851eb852p-3'))):  # F:1928 if( ql(i,k) .gt. qmin .and. ql(i,k-1) .lt. qmin .and. qrlw(i,k) .lt. 0._r8 .and. ri(i,k) .ge. ricrit ) then
            if (True and belongcv[(k + 1)]):  # F:1938 if( choice_srcl .eq. 'nonamb' .and. belongcv(i,k+1) ) then
                continue  # F:1939 go to 220
            ch = (((float.fromhex('0x1.0000000000000p+0') - sfuh[k]) * chu[k]) + (sfuh[k] * chs[k]))  # F:1942 ch = ( 1._r8 - sfuh(i,k) ) * chu(i,k) + sfuh(i,k) * chs(i,k)
            cm = (((float.fromhex('0x1.0000000000000p+0') - sfuh[k]) * cmu[k]) + (sfuh[k] * cms[k]))  # F:1943 cm = ( 1._r8 - sfuh(i,k) ) * cmu(i,k) + sfuh(i,k) * cms(i,k)
            n2htsrcl = ((ch * slslope[k]) + (cm * qtslope[k]))  # F:1945 n2htsrcl = ch * slslope(i,k) + cm * qtslope(i,k)
            if (n2htsrcl <= float.fromhex('0x0.0p+0')):  # F:1947 if( n2htsrcl .le. 0._r8 ) then
                in_CL = False  # F:1955 in_CL = .false.
                while (ncv <= ncvf):  # F:1957 do while ( ncv .le. ncvf )
                    if (ktop[ncv] <= k):  # F:1958 if( ktop(i,ncv) .le. k ) then
                        if (kbase[ncv] > k):  # F:1959 if( kbase(i,ncv) .gt. k ) then
                            in_CL = True  # F:1960 in_CL = .true.
                        break  # F:1962 exit
                    else:
                        ncv = (ncv + 1)  # F:1964 ncv = ncv + 1
                if (not in_CL):  # F:1968 if( .not. in_CL ) then
                    ncvfin[1] = (ncvfin[1] + 1)  # F:1972 ncvfin(i)       =  ncvfin(i) + 1
                    ncvnew = ncvfin[1]  # F:1973 ncvnew          =  ncvfin(i)
                    ktop[ncvnew] = k  # F:1974 ktop(i,ncvnew)  =  k
                    kbase[ncvnew] = (k + 1)  # F:1975 kbase(i,ncvnew) =  k+1
                    belongcv[k] = True  # F:1976 belongcv(i,k)   = .true.
                    belongcv[(k + 1)] = True  # F:1977 belongcv(i,k+1) = .true.
                    if (k < pver):  # F:1994 if( k .lt. pver ) then
                        wbrk[ncvnew] = float.fromhex('0x0.0p+0')  # F:1996 wbrk(i,ncvnew) = 0._r8
                        ebrk[ncvnew] = float.fromhex('0x0.0p+0')  # F:1997 ebrk(i,ncvnew) = 0._r8
                        lbrk[ncvnew] = float.fromhex('0x0.0p+0')  # F:1998 lbrk(i,ncvnew) = 0._r8
                        ghcl[ncvnew] = float.fromhex('0x0.0p+0')  # F:1999 ghcl(i,ncvnew) = 0._r8
                        shcl[ncvnew] = float.fromhex('0x0.0p+0')  # F:2000 shcl(i,ncvnew) = 0._r8
                        smcl[ncvnew] = float.fromhex('0x0.0p+0')  # F:2001 smcl(i,ncvnew) = 0._r8
                        ricl[ncvnew] = float.fromhex('0x0.0p+0')  # F:2002 ricl(i,ncvnew) = 0._r8
                    else:
                        if (bflxs[1] > float.fromhex('0x0.0p+0')):  # F:2006 if( bflxs(i) .gt. 0._r8 ) then
                            ebrk[ncvnew] = tkes[1]  # F:2010 ebrk(i,ncvnew) = tkes(i)
                            lbrk[ncvnew] = z[pver]  # F:2011 lbrk(i,ncvnew) = z(i,pver)
                            wbrk[ncvnew] = (tkes[1] / float.fromhex('0x1.7333333333333p+2'))  # F:2012 wbrk(i,ncvnew) = tkes(i) / b1
                            for ks in range(1, ncvmax + 1, 1):  # F:2030 do ks = 1, ncvmax
                                pass
                            raise RuntimeError("Fortran STOP at F:2036")  # F:2036 stop
                        else:
                            ebrk[ncvnew] = float.fromhex('0x0.0p+0')  # F:2040 ebrk(i,ncvnew) = 0._r8
                            lbrk[ncvnew] = float.fromhex('0x0.0p+0')  # F:2041 lbrk(i,ncvnew) = 0._r8
                            wbrk[ncvnew] = float.fromhex('0x0.0p+0')  # F:2042 wbrk(i,ncvnew) = 0._r8
                        gg = ((((float.fromhex('0x1.0000000000000p-1') * vk) * z[pver]) * bprod[(pver + 1)]) / math.pow(tkes[1], float.fromhex('0x1.8000000000000p+0')))  # F:2056 gg = 0.5_r8 * vk * z(i,pver) * bprod(i,pver+1) / ( tkes(i)**(3._r8/2._r8) )
                        if (abs((float.fromhex('0x1.65aee631f8a09p-1') - (gg * float.fromhex('-0x1.15694467381d8p+5')))) <= float.fromhex('0x1.ad7f29abcaf48p-24')):  # F:2057 if( abs(alph5-gg*alph3) .le. 1.e-7_r8 ) then
                            gh = ghmin  # F:2060 gh = ghmin
                        else:
                            gh = (gg / (float.fromhex('0x1.65aee631f8a09p-1') - (gg * float.fromhex('-0x1.15694467381d8p+5'))))  # F:2062 gh = gg / ( alph5 - gg * alph3 )
                        gh = uw_fmin(uw_fmax(gh, ghmin), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:2066 gh = min(max(gh,ghmin),0.0233_r8)
                        ghcl[ncvnew] = gh  # F:2067 ghcl(i,ncvnew) =  gh
                        shcl[ncvnew] = uw_fmax(float.fromhex('0x0.0p+0'), (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))))  # F:2068 shcl(i,ncvnew) =  max(0._r8,alph5/(1._r8+alph3*gh))
                        smcl[ncvnew] = uw_fmax(float.fromhex('0x0.0p+0'), (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (alph4exs * gh))))  # F:2069 smcl(i,ncvnew) =  max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
                        ricl[ncvnew] = (-((smcl[ncvnew] / shcl[ncvnew]) * (bprod[(pver + 1)] / sprod[(pver + 1)])))  # F:2070 ricl(i,ncvnew) = -(smcl(i,ncvnew)/shcl(i,ncvnew))*(bprod(i,pver+1)/sprod(i,pver+1))
                        ncvsurf = ncvnew  # F:2075 ncvsurf = ncvnew
    for k in range(1, ncvmax + 1, 1):  # F:2108 do k = 1, ncvmax
        kbase_f[k] = F32(float(kbase[k]))  # F:2109 kbase_f(i,k) = real(kbase(i,k))
        ktop_f[k] = F32(float(ktop[k]))  # F:2110 ktop_f(i,k)  = real(ktop(i,k))
        ncvfin_f[1] = F32(float(ncvfin[1]))  # F:2111 ncvfin_f(i)  = real(ncvfin(i))
    ktblw = 0  # F:2134 ktblw = 0
    for ncv in range(1, ncvfin[1] + 1, 1):  # F:2135 do ncv = 1, ncvfin(i)
        kt = ktop[ncv]  # F:2137 kt = ktop(i,ncv)
        kb = kbase[ncv]  # F:2138 kb = kbase(i,ncv)
        if ((kb == (pver + 1)) and (bflxs[1] <= float.fromhex('0x0.0p+0'))):  # F:2140 if( kb .eq. (pver+1) .and. bflxs(i) .le. 0._r8 ) then
            lbulk = (zi[kt] - z[pver])  # F:2141 lbulk = zi(i,kt) - z(i,pver)
        else:
            lbulk = (zi[kt] - zi[kb])  # F:2143 lbulk = zi(i,kt) - zi(i,kb)
        for k in range(min(kb, pver), kt - 1, -1):  # F:2152 do k = min(kb,pver), kt, -1
            tunlramp = (float.fromhex('0x1.5c28f5c28f5c3p-3') * (float.fromhex('0x1.0000000000000p+0') - (float.fromhex('0x1.0000000000000p-1') * math.exp(uw_fmin(float.fromhex('0x0.0p+0'), ricl[ncv])))))  # F:2158 tunlramp = ctunl*tunl*(1._r8-(1._r8-1._r8/ctunl)*exp(min(0._r8,ricl(i,ncv))))
            tunlramp = uw_fmin(uw_fmax(tunlramp, float.fromhex('0x1.5c28f5c28f5c3p-4')), float.fromhex('0x1.5c28f5c28f5c3p-3'))  # F:2159 tunlramp = min(max(tunlramp,tunl),ctunl*tunl)
            leng[k] = math.pow((math.pow((vk * zi[k]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:2167 leng(i,k) = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            wcap[k] = ((leng[k] * leng[k]) * ((-(shcl[ncv] * n2[k])) + (smcl[ncv] * s2[k])))  # F:2172 wcap(i,k) = (leng(i,k)**2) * (-shcl(i,ncv)*n2(i,k)+smcl(i,ncv)*s2(i,k))
        if (kb < (pver + 1)):  # F:2178 if( kb .lt. pver+1 ) then
            jbzm = (z[(kb - 1)] - z[kb])  # F:2180 jbzm = z(i,kb-1) - z(i,kb)
            jbsl = (sl[(kb - 1)] - sl[kb])  # F:2181 jbsl = sl(i,kb-1) - sl(i,kb)
            jbqt = (qt[(kb - 1)] - qt[kb])  # F:2182 jbqt = qt(i,kb-1) - qt(i,kb)
            jbbu = (n2[kb] * jbzm)  # F:2183 jbbu = n2(i,kb) * jbzm
            jbbu = uw_fmax(jbbu, float.fromhex('0x1.0624dd2f1a9fcp-10'))  # F:2184 jbbu = max(jbbu,jbumin)
            jbu = (u[(kb - 1)] - u[kb])  # F:2185 jbu  = u(i,kb-1) - u(i,kb)
            jbv = (v[(kb - 1)] - v[kb])  # F:2186 jbv  = v(i,kb-1) - v(i,kb)
            ch = (((float.fromhex('0x1.0000000000000p+0') - sflh[(kb - 1)]) * chu[kb]) + (sflh[(kb - 1)] * chs[kb]))  # F:2187 ch   = (1._r8 -sflh(i,kb-1))*chu(i,kb) + sflh(i,kb-1)*chs(i,kb)
            cm = (((float.fromhex('0x1.0000000000000p+0') - sflh[(kb - 1)]) * cmu[kb]) + (sflh[(kb - 1)] * cms[kb]))  # F:2188 cm   = (1._r8 -sflh(i,kb-1))*cmu(i,kb) + sflh(i,kb-1)*cms(i,kb)
            n2hb = (((ch * jbsl) + (cm * jbqt)) / jbzm)  # F:2189 n2hb = (ch*jbsl + cm*jbqt)/jbzm
            vyb = ((n2hb * jbzm) / jbbu)  # F:2190 vyb  = n2hb*jbzm/jbbu
            vub = uw_fmin(float.fromhex('0x1.0000000000000p+0'), (((jbu * jbu) + (jbv * jbv)) / (jbbu * jbzm)))  # F:2191 vub  = min(1._r8,(jbu**2+jbv**2)/(jbbu*jbzm) )
        else:
            jbbu = float.fromhex('0x0.0p+0')  # F:2196 jbbu = 0._r8
            n2hb = float.fromhex('0x0.0p+0')  # F:2197 n2hb = 0._r8
            vyb = float.fromhex('0x0.0p+0')  # F:2198 vyb  = 0._r8
            vub = float.fromhex('0x0.0p+0')  # F:2199 vub  = 0._r8
            web = float.fromhex('0x0.0p+0')  # F:2200 web  = 0._r8
        jtzm = (z[(kt - 1)] - z[kt])  # F:2208 jtzm = z(i,kt-1) - z(i,kt)
        jtsl = (sl[(kt - 1)] - sl[kt])  # F:2209 jtsl = sl(i,kt-1) - sl(i,kt)
        jtqt = (qt[(kt - 1)] - qt[kt])  # F:2210 jtqt = qt(i,kt-1) - qt(i,kt)
        jtbu = (n2[kt] * jtzm)  # F:2211 jtbu = n2(i,kt)*jtzm
        jtbu = uw_fmax(jtbu, float.fromhex('0x1.0624dd2f1a9fcp-10'))  # F:2212 jtbu = max(jtbu,jbumin)
        jtu = (u[(kt - 1)] - u[kt])  # F:2213 jtu  = u(i,kt-1) - u(i,kt)
        jtv = (v[(kt - 1)] - v[kt])  # F:2214 jtv  = v(i,kt-1) - v(i,kt)
        ch = (((float.fromhex('0x1.0000000000000p+0') - sfuh[kt]) * chu[kt]) + (sfuh[kt] * chs[kt]))  # F:2215 ch   = (1._r8 -sfuh(i,kt))*chu(i,kt) + sfuh(i,kt)*chs(i,kt)
        cm = (((float.fromhex('0x1.0000000000000p+0') - sfuh[kt]) * cmu[kt]) + (sfuh[kt] * cms[kt]))  # F:2216 cm   = (1._r8 -sfuh(i,kt))*cmu(i,kt) + sfuh(i,kt)*cms(i,kt)
        n2ht = (((ch * jtsl) + (cm * jtqt)) / jtzm)  # F:2217 n2ht = (ch*jtsl + cm*jtqt)/jtzm
        vyt = ((n2ht * jtzm) / jtbu)  # F:2218 vyt  = n2ht*jtzm/jtbu
        vut = uw_fmin(float.fromhex('0x1.0000000000000p+0'), (((jtu * jtu) + (jtv * jtv)) / (jtbu * jtzm)))  # F:2219 vut  = min(1._r8,(jtu**2+jtv**2)/(jtbu*jtzm))
        evhc = float.fromhex('0x1.0000000000000p+0')  # F:2229 evhc   = 1._r8
        jt2slv = float.fromhex('0x0.0p+0')  # F:2230 jt2slv = 0._r8
        qleff = uw_fmax(ql[(kt - 1)], ql[kt])  # F:2253 qleff  = max( ql(i,kt-1), ql(i,kt) )
        jt2slv = (slv[max((kt - 2), 1)] - slv[kt])  # F:2254 jt2slv = slv(i,max(kt-2,1)) - slv(i,kt)
        jt2slv = uw_fmax(jt2slv, ((float.fromhex('0x1.0624dd2f1a9fcp-10') * slv[(kt - 1)]) / g))  # F:2255 jt2slv = max( jt2slv, jbumin*slv(i,kt-1)/g )
        evhc = (float.fromhex('0x1.0000000000000p+0') + (((float.fromhex('0x1.8000000000000p+4') * latvap) * qleff) / jt2slv))  # F:2256 evhc   = 1._r8 + a2l * a3l * latvap * qleff / jt2slv
        evhc = uw_fmin(evhc, float.fromhex('0x1.4000000000000p+3'))  # F:2257 evhc   = min( evhc, evhcmax )
        lwp = float.fromhex('0x0.0p+0')  # F:2268 lwp        = 0._r8
        opt_depth = float.fromhex('0x0.0p+0')  # F:2269 opt_depth  = 0._r8
        radinvfrac = float.fromhex('0x0.0p+0')  # F:2270 radinvfrac = 0._r8
        radf = float.fromhex('0x0.0p+0')  # F:2271 radf       = 0._r8
        lwp = ((ql[kt] * (pi[(kt + 1)] - pi[kt])) / g)  # F:2304 lwp         = ql(i,kt) * ( pi(i,kt+1) - pi(i,kt) ) / g
        opt_depth = (float.fromhex('0x1.3800000000000p+7') * lwp)  # F:2305 opt_depth   = 156._r8 * lwp
        radinvfrac = ((opt_depth * (float.fromhex('0x1.0000000000000p+2') + opt_depth)) / ((float.fromhex('0x1.8000000000000p+2') * (float.fromhex('0x1.0000000000000p+2') + opt_depth)) + (opt_depth * opt_depth)))  # F:2306 radinvfrac  = opt_depth * ( 4._r8 + opt_depth ) / ( 6._r8 * ( 4._r8 + opt_depth ) + opt_depth**2 )
        radf = uw_fmax((((radinvfrac * qrlw[kt]) / (pi[kt] - pi[(kt + 1)])) * (zi[kt] - zi[(kt + 1)])), float.fromhex('0x0.0p+0'))  # F:2307 radf        = max( radinvfrac * qrlw(i,kt) / ( pi(i,kt) - pi(i,kt+1) ) * ( zi(i,kt) - zi(i,kt+1) ), 0._r8 )
        lwp = ((ql[(kt - 1)] * (pi[kt] - pi[(kt - 1)])) / g)  # F:2309 lwp         = ql(i,kt-1) * ( pi(i,kt) - pi(i,kt-1) ) / g
        opt_depth = (float.fromhex('0x1.3800000000000p+7') * lwp)  # F:2310 opt_depth   = 156._r8 * lwp
        radinvfrac = ((opt_depth * (float.fromhex('0x1.0000000000000p+2') + opt_depth)) / ((float.fromhex('0x1.8000000000000p+2') * (float.fromhex('0x1.0000000000000p+2') + opt_depth)) + (opt_depth * opt_depth)))  # F:2311 radinvfrac  = opt_depth * ( 4._r8 + opt_depth ) / ( 6._r8 * ( 4._r8 + opt_depth) + opt_depth**2 )
        radf = (radf + uw_fmax((((radinvfrac * qrlw[(kt - 1)]) / (pi[(kt - 1)] - pi[kt])) * (zi[(kt - 1)] - zi[kt])), float.fromhex('0x0.0p+0')))  # F:2312 radf        = radf + max( radinvfrac * qrlw(i,kt-1) / ( pi(i,kt-1) - pi(i,kt) ) * ( zi(i,kt-1) - zi(i,kt) ), 0._r8 )
        radf = (uw_fmax(radf, float.fromhex('0x0.0p+0')) * chs[kt])  # F:2314 radf        = max( radf, 0._r8 ) * chs(i,kt)
        dzht = (zi[kt] - z[kt])  # F:2339 dzht   = zi(i,kt)  - z(i,kt)
        dzhb = (z[(kb - 1)] - zi[kb])  # F:2340 dzhb   = z(i,kb-1) - zi(i,kb)
        wstar3 = (radf * dzht)  # F:2341 wstar3 = radf * dzht
        for k in range((kt + 1), (kb - 1) + 1, 1):  # F:2342 do k = kt + 1, kb - 1
            wstar3 = (wstar3 + (bprod[k] * (z[(k - 1)] - z[k])))  # F:2343 wstar3 =  wstar3 + bprod(i,k) * ( z(i,k-1) - z(i,k) )
        if ((kb == (pver + 1)) and (bflxs[1] > float.fromhex('0x0.0p+0'))):  # F:2351 if( kb .eq. (pver+1) .and. bflxs(i) .gt. 0._r8 ) then
            wstar3 = (wstar3 + (bflxs[1] * dzhb))  # F:2352 wstar3 = wstar3 + bflxs(i) * dzhb
        wstar3 = uw_fmax((float.fromhex('0x1.4000000000000p+1') * wstar3), float.fromhex('0x0.0p+0'))  # F:2355 wstar3 = max( 2.5_r8 * wstar3, 0._r8 )
        if (wstar3 > float.fromhex('0x0.0p+0')):  # F:2408 if( wstar3 .gt. 0._r8 ) then
            cet = ((float.fromhex('0x1.999999999999ap-3') * evhc) / (jtbu * lbulk))  # F:2409 cet = a1i * evhc / ( jtbu * lbulk )
            if (kb == (pver + 1)):  # F:2410 if( kb .eq. pver + 1 ) then
                wstar3fact = uw_fmax((float.fromhex('0x1.0000000000000p+0') + ((((float.fromhex('0x1.4000000000000p+1') * cet) * n2ht) * jtzm) * dzht)), float.fromhex('0x1.0000000000000p-1'))  # F:2411 wstar3fact = max( 1._r8 + 2.5_r8 * cet * n2ht * jtzm * dzht, wstar3factcrit )
            else:
                ceb = (float.fromhex('0x1.999999999999ap-3') / (jbbu * lbulk))  # F:2413 ceb = a1i / ( jbbu * lbulk )
                wstar3fact = uw_fmax(((float.fromhex('0x1.0000000000000p+0') + ((((float.fromhex('0x1.4000000000000p+1') * cet) * n2ht) * jtzm) * dzht)) + ((((float.fromhex('0x1.4000000000000p+1') * ceb) * n2hb) * jbzm) * dzhb)), float.fromhex('0x1.0000000000000p-1'))  # F:2414 wstar3fact = max( 1._r8 + 2.5_r8 * cet * n2ht * jtzm * dzht + 2.5_r8 * ceb * n2hb * jbzm * dzhb, wstar3factcrit )
            wstar3 = (wstar3 / wstar3fact)  # F:2417 wstar3 = wstar3 / wstar3fact
        else:
            wstar3fact = float.fromhex('0x0.0p+0')  # F:2419 wstar3fact = 0._r8
            cet = float.fromhex('0x0.0p+0')  # F:2420 cet        = 0._r8
            ceb = float.fromhex('0x0.0p+0')  # F:2421 ceb        = 0._r8
        fact = ((((evhc * ((-vyt) + vut)) * dzht) + (((((-vyb) + vub) * dzhb) * leng[kb]) / leng[kt])) / lbulk)  # F:2465 fact = ( evhc * ( -vyt + vut ) * dzht + ( -vyb + vub ) * dzhb * leng(i,kb) / leng(i,kt) ) / lbulk
        trma = float.fromhex('0x1.0000000000000p+0')  # F:2475 trma = 1._r8
        trmp = (((ebrk[ncv] * (lbrk[ncv] / lbulk)) / float.fromhex('0x1.8000000000000p+1')) + float.fromhex('0x1.19799812dea11p-40'))  # F:2476 trmp = ebrk(i,ncv) * ( lbrk(i,ncv) / lbulk ) / 3._r8 + ntzero
        trmq = ((float.fromhex('0x1.7333333333333p+1') * (leng[kt] / lbulk)) * ((radf * dzht) + ((float.fromhex('0x1.999999999999ap-3') * fact) * wstar3)))  # F:2477 trmq = 0.5_r8 * b1 * ( leng(i,kt)  / lbulk ) * ( radf * dzht + a1i * fact * wstar3 )
        rmin = math.sqrt(trmp)  # F:2483 rmin  = sqrt(trmp)
        fmin = ((rmin * ((rmin * rmin) - (float.fromhex('0x1.8000000000000p+1') * trmp))) - (float.fromhex('0x1.0000000000000p+1') * trmq))  # F:2484 fmin  = rmin * ( rmin * rmin - 3._r8 * trmp ) - 2._r8 * trmq
        wstar = math.pow(wstar3, float.fromhex('0x1.5555555555555p-2'))  # F:2485 wstar = wstar3**onet
        rcrit = (float.fromhex('0x1.0000000000000p-1') * wstar)  # F:2486 rcrit = ccrit * wstar
        fcrit = ((rcrit * ((rcrit * rcrit) - (float.fromhex('0x1.8000000000000p+1') * trmp))) - (float.fromhex('0x1.0000000000000p+1') * trmq))  # F:2487 fcrit = rcrit * ( rcrit * rcrit - 3._r8 * trmp ) - 2._r8 * trmq
        noroot = (((rmin < rcrit) and (fcrit > float.fromhex('0x0.0p+0'))) or ((rmin >= rcrit) and (fmin > float.fromhex('0x0.0p+0'))))  # F:2499 noroot = ( ( rmin .lt. rcrit ) .and. ( fcrit .gt. 0._r8 ) ) .or. ( ( rmin .ge. rcrit ) .and. ( fmin  .gt. 0._r8 ) )
        if noroot:  # F:2501 if( noroot ) then
            trma = (float.fromhex('0x1.0000000000000p+0') - ((((float.fromhex('0x1.7333333333333p+2') * (leng[kt] / lbulk)) * float.fromhex('0x1.999999999999ap-3')) * fact) / float.fromhex('0x1.0000000000000p-3')))  # F:2502 trma = 1._r8 - b1 * ( leng(i,kt) / lbulk ) * a1i * fact / ccrit**3
            trma = uw_fmax(trma, float.fromhex('0x1.0000000000000p-1'))  # F:2503 trma = max( trma, 0.5_r8 )
            trmp = (trmp / trma)  # F:2504 trmp = trmp / trma
            trmq = ((((float.fromhex('0x1.7333333333333p+1') * (leng[kt] / lbulk)) * radf) * dzht) / trma)  # F:2505 trmq = 0.5_r8 * b1 * ( leng(i,kt) / lbulk ) * radf * dzht / trma
        qq = ((trmq * trmq) - powi(trmp, 3))  # F:2510 qq = trmq**2 - trmp**3
        if (qq >= float.fromhex('0x0.0p+0')):  # F:2511 if( qq .ge. 0._r8 ) then
            rootp = (math.pow((trmq + math.sqrt(qq)), float.fromhex('0x1.5555555555555p-2')) + math.pow(uw_fmax((trmq - math.sqrt(qq)), float.fromhex('0x0.0p+0')), float.fromhex('0x1.5555555555555p-2')))  # F:2512 rootp = ( trmq + sqrt(qq) )**(1._r8/3._r8) + ( max( trmq - sqrt(qq), 0._r8 ) )**(1._r8/3._r8)
        else:
            rootp = ((float.fromhex('0x1.0000000000000p+1') * math.sqrt(trmp)) * math.cos((math.acos((trmq / math.sqrt(powi(trmp, 3)))) / float.fromhex('0x1.8000000000000p+1'))))  # F:2514 rootp = 2._r8 * sqrt(trmp) * cos( acos( trmq / sqrt(trmp**3) ) / 3._r8 )
        if noroot:  # F:2520 if( noroot )  wstar3 = ( rootp / ccrit )**3
            wstar3 = powi((rootp / float.fromhex('0x1.0000000000000p-1')), 3)  # F:2520 if( noroot )  wstar3 = ( rootp / ccrit )**3
        wet = (cet * wstar3)  # F:2521 wet = cet * wstar3
        if (kb < (pver + 1)):  # F:2522 if( kb .lt. pver + 1 ) web = ceb * wstar3
            web = (ceb * wstar3)  # F:2522 if( kb .lt. pver + 1 ) web = ceb * wstar3
        ebrk[ncv] = (rootp * rootp)  # F:2553 ebrk(i,ncv) = rootp**2
        ebrk[ncv] = uw_fmin(ebrk[ncv], float.fromhex('0x1.4000000000000p+4'))  # F:2554 ebrk(i,ncv) = min(ebrk(i,ncv),tkemax)
        wbrk[ncv] = (ebrk[ncv] / float.fromhex('0x1.7333333333333p+2'))  # F:2555 wbrk(i,ncv) = ebrk(i,ncv)/b1
        if (ebrk[ncv] <= float.fromhex('0x0.0p+0')):  # F:2567 if( ebrk(i,ncv) .le. 0._r8 ) then
            belongcv[kt] = False  # F:2572 belongcv(i,kt) = .false.
            belongcv[kb] = False  # F:2573 belongcv(i,kb) = .false.
        for k in range((kb - 1), (kt + 1) - 1, -1):  # F:2591 do k = kb - 1, kt + 1, -1
            rcap = ((float.fromhex('0x1.7333333333333p+2') + (wcap[k] / wbrk[ncv])) / float.fromhex('0x1.b333333333333p+2'))  # F:2592 rcap = ( b1 * ae + wcap(i,k) / wbrk(i,ncv) ) / ( b1 * ae + 1._r8 )
            rcap = uw_fmin(uw_fmax(rcap, float.fromhex('0x1.999999999999ap-4')), float.fromhex('0x1.0000000000000p+1'))  # F:2593 rcap = min( max(rcap,rcapmin), rcapmax )
            tke[k] = (ebrk[ncv] * rcap)  # F:2594 tke(i,k) = ebrk(i,ncv) * rcap
            tke[k] = uw_fmin(tke[k], float.fromhex('0x1.4000000000000p+4'))  # F:2595 tke(i,k) = min( tke(i,k), tkemax )
            kvh[k] = ((leng[k] * math.sqrt(tke[k])) * shcl[ncv])  # F:2596 kvh(i,k) = leng(i,k) * sqrt(tke(i,k)) * shcl(i,ncv)
            kvm[k] = ((leng[k] * math.sqrt(tke[k])) * smcl[ncv])  # F:2597 kvm(i,k) = leng(i,k) * sqrt(tke(i,k)) * smcl(i,ncv)
            bprod[k] = (-(kvh[k] * n2[k]))  # F:2598 bprod(i,k) = -kvh(i,k) * n2(i,k)
            sprod[k] = (kvm[k] * s2[k])  # F:2599 sprod(i,k) =  kvm(i,k) * s2(i,k)
            turbtype[k] = 2  # F:2600 turbtype(i,k) = 2
            sm_aw[k] = (smcl[ncv] / float.fromhex('0x1.1cc63f141205cp-1'))  # F:2601 sm_aw(i,k) = smcl(i,ncv)/alph1
        kentr = (wet * jtzm)  # F:2605 kentr = wet * jtzm
        kvh[kt] = kentr  # F:2606 kvh(i,kt) = kentr
        kvm[kt] = kentr  # F:2607 kvm(i,kt) = kentr
        bprod[kt] = ((-(kentr * n2ht)) + radf)  # F:2608 bprod(i,kt) = -kentr * n2ht + radf
        sprod[kt] = (kentr * s2[kt])  # F:2609 sprod(i,kt) =  kentr * s2(i,kt)
        turbtype[kt] = 4  # F:2610 turbtype(i,kt) = 4
        trmp = float.fromhex('-0x1.b4b4b4b4b4b4bp-1')  # F:2611 trmp = -b1 * ae / ( 1._r8 + b1 * ae )
        trmq = (-(((((bprod[kt] + sprod[kt]) * float.fromhex('0x1.7333333333333p+2')) * leng[kt]) / float.fromhex('0x1.b333333333333p+2')) / math.pow(ebrk[ncv], float.fromhex('0x1.8000000000000p+0'))))  # F:2612 trmq = -(bprod(i,kt)+sprod(i,kt))*b1*leng(i,kt)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
        rcap = math.pow(compute_cubic(float.fromhex('0x0.0p+0'), trmp, trmq), float.fromhex('0x1.0000000000000p+1'))  # F:2613 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
        rcap = uw_fmin(uw_fmax(rcap, float.fromhex('0x1.999999999999ap-4')), float.fromhex('0x1.0000000000000p+1'))  # F:2614 rcap = min( max(rcap,rcapmin), rcapmax )
        tke[kt] = (ebrk[ncv] * rcap)  # F:2615 tke(i,kt)  = ebrk(i,ncv) * rcap
        tke[kt] = uw_fmin(tke[kt], float.fromhex('0x1.4000000000000p+4'))  # F:2616 tke(i,kt)  = min( tke(i,kt), tkemax )
        sm_aw[kt] = (smcl[ncv] / float.fromhex('0x1.1cc63f141205cp-1'))  # F:2617 sm_aw(i,kt) = smcl(i,ncv) / alph1
        if (kb < (pver + 1)):  # F:2625 if( kb .lt. pver + 1 ) then
            kentr = (web * jbzm)  # F:2627 kentr = web * jbzm
            if (kb != ktblw):  # F:2629 if( kb .ne. ktblw ) then
                kvh[kb] = kentr  # F:2631 kvh(i,kb) = kentr
                kvm[kb] = kentr  # F:2632 kvm(i,kb) = kentr
                bprod[kb] = (-(kvh[kb] * n2hb))  # F:2633 bprod(i,kb) = -kvh(i,kb)*n2hb
                sprod[kb] = (kvm[kb] * s2[kb])  # F:2634 sprod(i,kb) =  kvm(i,kb)*s2(i,kb)
                turbtype[kb] = 3  # F:2635 turbtype(i,kb) = 3
                trmp = float.fromhex('-0x1.b4b4b4b4b4b4bp-1')  # F:2636 trmp = -b1*ae/(1._r8+b1*ae)
                trmq = (-(((((bprod[kb] + sprod[kb]) * float.fromhex('0x1.7333333333333p+2')) * leng[kb]) / float.fromhex('0x1.b333333333333p+2')) / math.pow(ebrk[ncv], float.fromhex('0x1.8000000000000p+0'))))  # F:2637 trmq = -(bprod(i,kb)+sprod(i,kb))*b1*leng(i,kb)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
                rcap = math.pow(compute_cubic(float.fromhex('0x0.0p+0'), trmp, trmq), float.fromhex('0x1.0000000000000p+1'))  # F:2638 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
                rcap = uw_fmin(uw_fmax(rcap, float.fromhex('0x1.999999999999ap-4')), float.fromhex('0x1.0000000000000p+1'))  # F:2639 rcap = min( max(rcap,rcapmin), rcapmax )
                tke[kb] = (ebrk[ncv] * rcap)  # F:2640 tke(i,kb)  = ebrk(i,ncv) * rcap
                tke[kb] = uw_fmin(tke[kb], float.fromhex('0x1.4000000000000p+4'))  # F:2641 tke(i,kb)  = min( tke(i,kb),tkemax )
            else:
                kvh[kb] = (kvh[kb] + kentr)  # F:2645 kvh(i,kb) = kvh(i,kb) + kentr
                kvm[kb] = (kvm[kb] + kentr)  # F:2646 kvm(i,kb) = kvm(i,kb) + kentr
                dzhb5 = (z[(kb - 1)] - zi[kb])  # F:2649 dzhb5 = z(i,kb-1) - zi(i,kb)
                dzht5 = (zi[kb] - z[kb])  # F:2650 dzht5 = zi(i,kb) - z(i,kb)
                bprod[kb] = (((dzht5 * bprod[kb]) - ((dzhb5 * kentr) * n2hb)) / (dzhb5 + dzht5))  # F:2651 bprod(i,kb) = ( dzht5*bprod(i,kb) - dzhb5*kentr*n2hb )     / ( dzhb5 + dzht5 )
                sprod[kb] = (((dzht5 * sprod[kb]) + ((dzhb5 * kentr) * s2[kb])) / (dzhb5 + dzht5))  # F:2652 sprod(i,kb) = ( dzht5*sprod(i,kb) + dzhb5*kentr*s2(i,kb) ) / ( dzhb5 + dzht5 )
                trmp = float.fromhex('-0x1.b4b4b4b4b4b4bp-1')  # F:2653 trmp = -b1*ae/(1._r8+b1*ae)
                trmq = (-(((((kentr * (s2[kb] - n2hb)) * float.fromhex('0x1.7333333333333p+2')) * leng[kb]) / float.fromhex('0x1.b333333333333p+2')) / math.pow(ebrk[ncv], float.fromhex('0x1.8000000000000p+0'))))  # F:2654 trmq = -kentr*(s2(i,kb)-n2hb)*b1*leng(i,kb)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
                rcap = math.pow(compute_cubic(float.fromhex('0x0.0p+0'), trmp, trmq), float.fromhex('0x1.0000000000000p+1'))  # F:2655 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
                rcap = uw_fmin(uw_fmax(rcap, float.fromhex('0x1.999999999999ap-4')), float.fromhex('0x1.0000000000000p+1'))  # F:2656 rcap = min( max(rcap,rcapmin), rcapmax )
                tke_imsi = (ebrk[ncv] * rcap)  # F:2657 tke_imsi = ebrk(i,ncv) * rcap
                tke_imsi = uw_fmin(tke_imsi, float.fromhex('0x1.4000000000000p+4'))  # F:2658 tke_imsi = min( tke_imsi, tkemax )
                tke[kb] = (((dzht5 * tke[kb]) + (dzhb5 * tke_imsi)) / (dzhb5 + dzht5))  # F:2659 tke(i,kb)  = ( dzht5*tke(i,kb) + dzhb5*tke_imsi ) / ( dzhb5 + dzht5 )
                tke[kb] = uw_fmin(tke[kb], float.fromhex('0x1.4000000000000p+4'))  # F:2660 tke(i,kb)  = min(tke(i,kb),tkemax)
                turbtype[kb] = 5  # F:2661 turbtype(i,kb) = 5
        else:
            rcap = ((float.fromhex('0x1.7333333333333p+2') + (wcap[kb] / wbrk[ncv])) / float.fromhex('0x1.b333333333333p+2'))  # F:2671 rcap = (b1*ae + wcap(i,kb)/wbrk(i,ncv))/(b1*ae + 1._r8)
            rcap = uw_fmin(uw_fmax(rcap, float.fromhex('0x1.999999999999ap-4')), float.fromhex('0x1.0000000000000p+1'))  # F:2672 rcap = min( max(rcap,rcapmin), rcapmax )
            tke[kb] = (ebrk[ncv] * rcap)  # F:2673 tke(i,kb) = ebrk(i,ncv) * rcap
            tke[kb] = uw_fmin(tke[kb], float.fromhex('0x1.4000000000000p+4'))  # F:2674 tke(i,kb) = min( tke(i,kb),tkemax )
        sm_aw[kb] = (smcl[ncv] / float.fromhex('0x1.1cc63f141205cp-1'))  # F:2682 sm_aw(i,kb) = smcl(i,ncv)/alph1
        wcap[kt] = (((bprod[kt] + sprod[kt]) * leng[kt]) / math.sqrt(uw_fmax(tke[kt], float.fromhex('0x1.0c6f7a0b5ed8dp-20'))))  # F:2691 wcap(i,kt) = (bprod(i,kt)+sprod(i,kt))*leng(i,kt)/sqrt(max(tke(i,kt),1.e-6_r8))
        if (kb < (pver + 1)):  # F:2692 if( kb .lt. pver + 1 ) then
            wcap[kb] = (((bprod[kb] + sprod[kb]) * leng[kb]) / math.sqrt(uw_fmax(tke[kb], float.fromhex('0x1.0c6f7a0b5ed8dp-20'))))  # F:2693 wcap(i,kb) = (bprod(i,kb)+sprod(i,kb))*leng(i,kb)/sqrt(max(tke(i,kb),1.e-6_r8))
        ktblw = kt  # F:2700 ktblw = kt
        wet_CL[ncv] = wet  # F:2704 wet_CL(i,ncv)        = wet
        web_CL[ncv] = web  # F:2705 web_CL(i,ncv)        = web
        jtbu_CL[ncv] = jtbu  # F:2706 jtbu_CL(i,ncv)       = jtbu
        jbbu_CL[ncv] = jbbu  # F:2707 jbbu_CL(i,ncv)       = jbbu
        evhc_CL[ncv] = evhc  # F:2708 evhc_CL(i,ncv)       = evhc
        jt2slv_CL[ncv] = jt2slv  # F:2709 jt2slv_CL(i,ncv)     = jt2slv
        n2ht_CL[ncv] = n2ht  # F:2710 n2ht_CL(i,ncv)       = n2ht
        n2hb_CL[ncv] = n2hb  # F:2711 n2hb_CL(i,ncv)       = n2hb
        lwp_CL[ncv] = lwp  # F:2712 lwp_CL(i,ncv)        = lwp
        opt_depth_CL[ncv] = opt_depth  # F:2713 opt_depth_CL(i,ncv)  = opt_depth
        radinvfrac_CL[ncv] = radinvfrac  # F:2714 radinvfrac_CL(i,ncv) = radinvfrac
        radf_CL[ncv] = radf  # F:2715 radf_CL(i,ncv)       = radf
        wstar_CL[ncv] = wstar  # F:2716 wstar_CL(i,ncv)      = wstar
        wstar3fact_CL[ncv] = wstar3fact  # F:2717 wstar3fact_CL(i,ncv) = wstar3fact
    if (ncvsurf > 0):  # F:2728 if( ncvsurf .gt. 0 ) then
        ktopbl[1] = ktop[ncvsurf]  # F:2730 ktopbl(i) = ktop(i,ncvsurf)
        pblh[1] = zi[ktopbl[1]]  # F:2731 pblh(i)   = zi(i, ktopbl(i))
        pblhp[1] = pi[ktopbl[1]]  # F:2732 pblhp(i)  = pi(i, ktopbl(i))
        wpert[1] = uw_fmax((float.fromhex('0x1.0000000000000p+0') * math.sqrt(ebrk[ncvsurf])), float.fromhex('0x1.0c6f7a0b5ed8dp-20'))  # F:2733 wpert(i)  = max(wfac*sqrt(ebrk(i,ncvsurf)),wpertmin)
        tpert[1] = uw_fmax(((abs(((shflx[1] * rrho[1]) / cpair)) * float.fromhex('0x1.0000000000000p+0')) / wpert[1]), float.fromhex('0x0.0p+0'))  # F:2734 tpert(i)  = max(abs(shflx(i)*rrho(i)/cpair)*tfac/wpert(i),0._r8)
        qpert[1] = uw_fmax(((abs((qflx[1] * rrho[1])) * float.fromhex('0x1.0000000000000p+0')) / wpert[1]), float.fromhex('0x0.0p+0'))  # F:2735 qpert(i)  = max(abs(qflx(i)*rrho(i))*tfac/wpert(i),0._r8)
        if (bflxs[1] > float.fromhex('0x0.0p+0')):  # F:2737 if( bflxs(i) .gt. 0._r8 ) then
            turbtype[(pver + 1)] = 2  # F:2738 turbtype(i,pver+1) = 2
        else:
            turbtype[(pver + 1)] = 3  # F:2740 turbtype(i,pver+1) = 3
        ipbl[1] = float.fromhex('0x1.0000000000000p+0')  # F:2743 ipbl(i)  = 1._r8
        kpblh[1] = (ktopbl[1] - float.fromhex('0x1.0000000000000p+0'))  # F:2744 kpblh(i) = ktopbl(i) - 1._r8
    belongst[1] = False  # F:2755 belongst(i,1) = .false.
    for k in range(2, pver + 1, 1):  # F:2756 do k = 2, pver
        belongst[k] = ((ri[k] < float.fromhex('0x1.851eb851eb852p-3')) and (not belongcv[k]))  # F:2757 belongst(i,k) = ( ri(i,k) .lt. ricrit ) .and. ( .not. belongcv(i,k) )
        if (belongst[k] and (not belongst[(k - 1)])):  # F:2758 if( belongst(i,k) .and. ( .not. belongst(i,k-1) ) ) then
            kt = k  # F:2759 kt = k
        elif ((not belongst[k]) and belongst[(k - 1)]):  # F:2760 elseif( .not. belongst(i,k) .and. belongst(i,k-1) ) then
            kb = (k - 1)  # F:2761 kb = k - 1
            lbulk = (z[(kt - 1)] - z[kb])  # F:2762 lbulk = z(i,kt-1) - z(i,kb)
            for ks in range(kt, kb + 1, 1):  # F:2763 do ks = kt, kb
                tunlramp = float.fromhex('0x1.5c28f5c28f5c3p-4')  # F:2765 tunlramp = tunl
                leng[ks] = math.pow((math.pow((vk * zi[ks]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:2773 leng(i,ks) = ( (vk*zi(i,ks))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
    belongst[(pver + 1)] = (not belongcv[(pver + 1)])  # F:2788 belongst(i,pver+1) = .not. belongcv(i,pver+1)
    if belongst[(pver + 1)]:  # F:2790 if( belongst(i,pver+1) ) then
        turbtype[(pver + 1)] = 1  # F:2792 turbtype(i,pver+1) = 1
        if belongst[pver]:  # F:2794 if( belongst(i,pver) ) then
            lbulk = z[(kt - 1)]  # F:2796 lbulk = z(i,kt-1)
        else:
            kt = (pver + 1)  # F:2798 kt = pver+1
            lbulk = z[(kt - 1)]  # F:2799 lbulk = z(i,kt-1)
        ktopbl[1] = (kt - 1)  # F:2807 ktopbl(i) = kt - 1
        pblh[1] = z[ktopbl[1]]  # F:2808 pblh(i)   = z(i,ktopbl(i))
        pblhp[1] = (float.fromhex('0x1.0000000000000p-1') * (pi[ktopbl[1]] + pi[(ktopbl[1] + 1)]))  # F:2809 pblhp(i)  = 0.5_r8 * ( pi(i,ktopbl(i)) + pi(i,ktopbl(i)+1) )
        for ks in range(kt, pver + 1, 1):  # F:2814 do ks = kt, pver
            tunlramp = float.fromhex('0x1.5c28f5c28f5c3p-4')  # F:2816 tunlramp = tunl
            leng[ks] = math.pow((math.pow((vk * zi[ks]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:2824 leng(i,ks) = ( (vk*zi(i,ks))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
        wpert[1] = float.fromhex('0x0.0p+0')  # F:2834 wpert(i) = 0._r8
        tpert[1] = uw_fmax(((((shflx[1] * rrho[1]) / cpair) * float.fromhex('0x1.1000000000000p+3')) / ustar[1]), float.fromhex('0x0.0p+0'))  # F:2835 tpert(i) = max(shflx(i)*rrho(i)/cpair*fak/ustar(i),0._r8)
        qpert[1] = uw_fmax((((qflx[1] * rrho[1]) * float.fromhex('0x1.1000000000000p+3')) / ustar[1]), float.fromhex('0x0.0p+0'))  # F:2836 qpert(i) = max(qflx(i)*rrho(i)*fak/ustar(i),0._r8)
        ipbl[1] = float.fromhex('0x0.0p+0')  # F:2838 ipbl(i)  = 0._r8
        kpblh[1] = ktopbl[1]  # F:2839 kpblh(i) = ktopbl(i)
    for k in range(2, pver + 1, 1):  # F:2850 do k = 2, pver
        if belongst[k]:  # F:2852 if( belongst(i,k) ) then
            turbtype[k] = 1  # F:2854 turbtype(i,k) = 1
            trma = (((float.fromhex('-0x1.15694467381d8p+5') * alph4exs) * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * (float.fromhex('-0x1.174bc6a7ef9dbp+2') - ((alph4exs * float.fromhex('0x1.65aee631f8a09p-1')) * ri[k]))))  # F:2855 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
            trmb = (((float.fromhex('-0x1.15694467381d8p+5') + alph4exs) * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * ((-(float.fromhex('0x1.65aee631f8a09p-1') * ri[k])) + float.fromhex('0x1.1cc63f141205cp-1'))))  # F:2856 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri[k]  # F:2857 trmc = ri(i,k)
            det = uw_fmax(((trmb * trmb) - ((float.fromhex('0x1.0000000000000p+2') * trma) * trmc)), float.fromhex('0x0.0p+0'))  # F:2858 det = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            if (det < float.fromhex('0x0.0p+0')):  # F:2860 if( det .lt. 0._r8 ) then
                raise RuntimeError("Fortran STOP at F:2865")  # F:2865 stop
            gh = (((-trmb) + math.sqrt(det)) / (float.fromhex('0x1.0000000000000p+1') * trma))  # F:2867 gh = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_fmin(uw_fmax(gh, ghmin), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:2870 gh = min(max(gh,ghmin),0.0233_r8)
            sh = uw_fmax(float.fromhex('0x0.0p+0'), (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))))  # F:2871 sh = max(0._r8,alph5/(1._r8+alph3*gh))
            sm = uw_fmax(float.fromhex('0x0.0p+0'), (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (alph4exs * gh))))  # F:2872 sm = max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
            tke[k] = ((float.fromhex('0x1.7333333333333p+2') * (leng[k] * leng[k])) * ((-(sh * n2[k])) + (sm * s2[k])))  # F:2874 tke(i,k)   = b1*(leng(i,k)**2)*(-sh*n2(i,k)+sm*s2(i,k))
            tke[k] = uw_fmin(tke[k], float.fromhex('0x1.4000000000000p+4'))  # F:2875 tke(i,k)   = min(tke(i,k),tkemax)
            wcap[k] = (tke[k] / float.fromhex('0x1.7333333333333p+2'))  # F:2876 wcap(i,k)  = tke(i,k)/b1
            kvh[k] = ((leng[k] * math.sqrt(tke[k])) * sh)  # F:2877 kvh(i,k)   = leng(i,k) * sqrt(tke(i,k)) * sh
            kvm[k] = ((leng[k] * math.sqrt(tke[k])) * sm)  # F:2878 kvm(i,k)   = leng(i,k) * sqrt(tke(i,k)) * sm
            bprod[k] = (-(kvh[k] * n2[k]))  # F:2879 bprod(i,k) = -kvh(i,k) * n2(i,k)
            sprod[k] = (kvm[k] * s2[k])  # F:2880 sprod(i,k) =  kvm(i,k) * s2(i,k)
            sm_aw[k] = (sm / float.fromhex('0x1.1cc63f141205cp-1'))  # F:2882 sm_aw(i,k) = sm/alph1
    for k in range(2, pver + 1, 1):  # F:2904 do k = 2, pver
        if (((turbtype[k] == 3) or (turbtype[k] == 4)) or (turbtype[k] == 5)):  # F:2906 if( ( turbtype(i,k) .eq. 3 ) .or. ( turbtype(i,k) .eq. 4 ) .or. ( turbtype(i,k) .eq. 5 ) ) then
            trma = (((float.fromhex('-0x1.15694467381d8p+5') * alph4exs) * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * (float.fromhex('-0x1.174bc6a7ef9dbp+2') - ((alph4exs * float.fromhex('0x1.65aee631f8a09p-1')) * ri[k]))))  # F:2909 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
            trmb = (((float.fromhex('-0x1.15694467381d8p+5') + alph4exs) * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * ((-(float.fromhex('0x1.65aee631f8a09p-1') * ri[k])) + float.fromhex('0x1.1cc63f141205cp-1'))))  # F:2910 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri[k]  # F:2911 trmc = ri(i,k)
            det = uw_fmax(((trmb * trmb) - ((float.fromhex('0x1.0000000000000p+2') * trma) * trmc)), float.fromhex('0x0.0p+0'))  # F:2912 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = (((-trmb) + math.sqrt(det)) / (float.fromhex('0x1.0000000000000p+1') * trma))  # F:2913 gh   = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_fmin(uw_fmax(gh, ghmin), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:2916 gh   = min(max(gh,ghmin),0.0233_r8)
            sh = uw_fmax(float.fromhex('0x0.0p+0'), (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))))  # F:2917 sh   = max(0._r8,alph5/(1._r8+alph3*gh))
            sm = uw_fmax(float.fromhex('0x0.0p+0'), (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (alph4exs * gh))))  # F:2918 sm   = max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
            lbulk = (z[(k - 1)] - z[k])  # F:2920 lbulk = z(i,k-1) - z(i,k)
            tunlramp = float.fromhex('0x1.5c28f5c28f5c3p-4')  # F:2923 tunlramp = tunl
            leng_imsi = math.pow((math.pow((vk * zi[k]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:2931 leng_imsi = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            tke_imsi = ((float.fromhex('0x1.7333333333333p+2') * (leng_imsi * leng_imsi)) * ((-(sh * n2[k])) + (sm * s2[k])))  # F:2937 tke_imsi = b1*(leng_imsi**2)*(-sh*n2(i,k)+sm*s2(i,k))
            tke_imsi = uw_fmin(uw_fmax(tke_imsi, float.fromhex('0x0.0p+0')), float.fromhex('0x1.4000000000000p+4'))  # F:2938 tke_imsi = min(max(tke_imsi,0._r8),tkemax)
            kvh_imsi = ((leng_imsi * math.sqrt(tke_imsi)) * sh)  # F:2939 kvh_imsi = leng_imsi * sqrt(tke_imsi) * sh
            kvm_imsi = ((leng_imsi * math.sqrt(tke_imsi)) * sm)  # F:2940 kvm_imsi = leng_imsi * sqrt(tke_imsi) * sm
            if (kvh[k] < kvh_imsi):  # F:2942 if( kvh(i,k) .lt. kvh_imsi ) then
                kvh[k] = kvh_imsi  # F:2943 kvh(i,k)   =  kvh_imsi
                kvm[k] = kvm_imsi  # F:2944 kvm(i,k)   =  kvm_imsi
                leng[k] = leng_imsi  # F:2945 leng(i,k)  = leng_imsi
                tke[k] = tke_imsi  # F:2946 tke(i,k)   =  tke_imsi
                wcap[k] = (tke_imsi / float.fromhex('0x1.7333333333333p+2'))  # F:2947 wcap(i,k)  =  tke_imsi / b1
                bprod[k] = (-(kvh_imsi * n2[k]))  # F:2948 bprod(i,k) = -kvh_imsi * n2(i,k)
                sprod[k] = (kvm_imsi * s2[k])  # F:2949 sprod(i,k) =  kvm_imsi * s2(i,k)
                sm_aw[k] = (sm / float.fromhex('0x1.1cc63f141205cp-1'))  # F:2950 sm_aw(i,k) =  sm/alph1
                turbtype[k] = 1  # F:2951 turbtype(i,k) = 1
    bprod[(pver + 1)] = bflxs[1]  # F:2987 bprod(i,pver+1) = bflxs(i)
    gg = ((((float.fromhex('0x1.0000000000000p-1') * vk) * z[pver]) * bprod[(pver + 1)]) / math.pow(tkes[1], float.fromhex('0x1.8000000000000p+0')))  # F:2989 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
    if (abs((float.fromhex('0x1.65aee631f8a09p-1') - (gg * float.fromhex('-0x1.15694467381d8p+5')))) <= float.fromhex('0x1.ad7f29abcaf48p-24')):  # F:2990 if( abs(alph5-gg*alph3) .le. 1.e-7_r8 ) then
        if (bprod[(pver + 1)] > float.fromhex('0x0.0p+0')):  # F:2992 if( bprod(i,pver+1) .gt. 0._r8 ) then
            gh = float.fromhex('-0x1.c4467381d7dbfp+1')  # F:2993 gh = -3.5334_r8
        else:
            gh = ghmin  # F:2995 gh = ghmin
    else:
        gh = (gg / (float.fromhex('0x1.65aee631f8a09p-1') - (gg * float.fromhex('-0x1.15694467381d8p+5'))))  # F:2998 gh = gg/(alph5-gg*alph3)
    if (bprod[(pver + 1)] > float.fromhex('0x0.0p+0')):  # F:3002 if( bprod(i,pver+1) .gt. 0._r8 ) then
        gh = uw_fmin(uw_fmax(gh, float.fromhex('-0x1.c4467381d7dbfp+1')), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3003 gh = min(max(gh,-3.5334_r8),0.0233_r8)
    else:
        gh = uw_fmin(uw_fmax(gh, ghmin), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3005 gh = min(max(gh,ghmin),0.0233_r8)
    gh_a[(pver + 1)] = gh  # F:3008 gh_a(i,pver+1) = gh
    sh_a[(pver + 1)] = uw_fmax(float.fromhex('0x0.0p+0'), (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))))  # F:3009 sh_a(i,pver+1) = max(0._r8,alph5/(1._r8+alph3*gh))
    if (bprod[(pver + 1)] > float.fromhex('0x0.0p+0')):  # F:3010 if( bprod(i,pver+1) .gt. 0._r8 ) then
        sm_a[(pver + 1)] = uw_fmax(float.fromhex('0x0.0p+0'), (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.88240b780346ep+2') * gh))))  # F:3011 sm_a(i,pver+1) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh))
    else:
        sm_a[(pver + 1)] = uw_fmax(float.fromhex('0x0.0p+0'), (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (alph4exs * gh))))  # F:3013 sm_a(i,pver+1) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
    sm_aw[(pver + 1)] = (sm_a[(pver + 1)] / float.fromhex('0x1.1cc63f141205cp-1'))  # F:3015 sm_aw(i,pver+1) = sm_a(i,pver+1)/alph1
    ri_a[(pver + 1)] = (-((sm_a[(pver + 1)] / sh_a[(pver + 1)]) * (bprod[(pver + 1)] / sprod[(pver + 1)])))  # F:3016 ri_a(i,pver+1)  = -(sm_a(i,pver+1)/sh_a(i,pver+1))*(bprod(i,pver+1)/sprod(i,pver+1))
    for k in range(1, pver + 1, 1):  # F:3018 do k = 1, pver
        if (ri[k] < float.fromhex('0x0.0p+0')):  # F:3019 if( ri(i,k) .lt. 0._r8 ) then
            trma = ((float.fromhex('0x1.a8f03ff93f46ap+7') * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * (float.fromhex('-0x1.174bc6a7ef9dbp+2') - (float.fromhex('-0x1.11f3168d8b188p+2') * ri[k]))))  # F:3020 trma = alph3*alph4*ri(i,k) + 2._r8*b1*(alph2-alph4*alph5*ri(i,k))
            trmb = ((float.fromhex('-0x1.466dc5d638866p+5') * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * ((-(float.fromhex('0x1.65aee631f8a09p-1') * ri[k])) + float.fromhex('0x1.1cc63f141205cp-1'))))  # F:3021 trmb = (alph3+alph4)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri[k]  # F:3022 trmc = ri(i,k)
            det = uw_fmax(((trmb * trmb) - ((float.fromhex('0x1.0000000000000p+2') * trma) * trmc)), float.fromhex('0x0.0p+0'))  # F:3023 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = (((-trmb) + math.sqrt(det)) / (float.fromhex('0x1.0000000000000p+1') * trma))  # F:3024 gh   = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_fmin(uw_fmax(gh, float.fromhex('-0x1.c4467381d7dbfp+1')), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3025 gh   = min(max(gh,-3.5334_r8),0.0233_r8)
            gh_a[k] = gh  # F:3026 gh_a(i,k) = gh
            sh_a[k] = uw_fmax(float.fromhex('0x0.0p+0'), (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))))  # F:3027 sh_a(i,k) = max(0._r8,alph5/(1._r8+alph3*gh))
            sm_a[k] = uw_fmax(float.fromhex('0x0.0p+0'), (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.88240b780346ep+2') * gh))))  # F:3028 sm_a(i,k) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh))
            ri_a[k] = ri[k]  # F:3029 ri_a(i,k) = ri(i,k)
        else:
            if (ri[k] > float.fromhex('0x1.851eb851eb852p-3')):  # F:3031 if( ri(i,k) .gt. ricrit ) then
                gh_a[k] = ghmin  # F:3032 gh_a(i,k) = ghmin
                sh_a[k] = float.fromhex('0x0.0p+0')  # F:3033 sh_a(i,k) = 0._r8
                sm_a[k] = float.fromhex('0x0.0p+0')  # F:3034 sm_a(i,k) = 0._r8
                ri_a[k] = ri[k]  # F:3035 ri_a(i,k) = ri(i,k)
            else:
                trma = (((float.fromhex('-0x1.15694467381d8p+5') * alph4exs) * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * (float.fromhex('-0x1.174bc6a7ef9dbp+2') - ((alph4exs * float.fromhex('0x1.65aee631f8a09p-1')) * ri[k]))))  # F:3037 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
                trmb = (((float.fromhex('-0x1.15694467381d8p+5') + alph4exs) * ri[k]) + (float.fromhex('0x1.7333333333333p+3') * ((-(float.fromhex('0x1.65aee631f8a09p-1') * ri[k])) + float.fromhex('0x1.1cc63f141205cp-1'))))  # F:3038 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
                trmc = ri[k]  # F:3039 trmc = ri(i,k)
                det = uw_fmax(((trmb * trmb) - ((float.fromhex('0x1.0000000000000p+2') * trma) * trmc)), float.fromhex('0x0.0p+0'))  # F:3040 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
                gh = (((-trmb) + math.sqrt(det)) / (float.fromhex('0x1.0000000000000p+1') * trma))  # F:3041 gh   = (-trmb + sqrt(det))/(2._r8*trma)
                gh = uw_fmin(uw_fmax(gh, ghmin), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3042 gh   = min(max(gh,ghmin),0.0233_r8)
                gh_a[k] = gh  # F:3043 gh_a(i,k) = gh
                sh_a[k] = uw_fmax(float.fromhex('0x0.0p+0'), (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))))  # F:3044 sh_a(i,k) = max(0._r8,alph5/(1._r8+alph3*gh))
                sm_a[k] = uw_fmax(float.fromhex('0x0.0p+0'), (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (alph4exs * gh))))  # F:3045 sm_a(i,k) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
                ri_a[k] = ri[k]  # F:3046 ri_a(i,k) = ri(i,k)
    for k in range(1, (pver + 1) + 1, 1):  # F:3052 do k = 1, pver + 1
        turbtype_f[k] = F32(float(turbtype[k]))  # F:3053 turbtype_f(i,k) = real(turbtype(i,k))
    return  # F:3058 return

def exacol(pver, ri, bflxs, minpblh, zi, ktop, kbase, ncvfin):
    ncvmax = pver
    nbot_turb = pver
    riex = [0] * ((pver + 1) + 1)
    cpair = CPAIR
    rair = RAIR
    zvir = ZVIR
    latvap = LATVAP
    latice = LATICE
    g = GRAVIT
    vk = KARMAN
    b123 = B123
    latsub = latvap + latice
    ccon = (float.fromhex('0x1.1000000000000p+3') * float.fromhex('0x1.999999999999ap-4')) * vk
    ncvfin[1] = 0  # F:3114 ncvfin(i) = 0
    for ncv in range(1, ncvmax + 1, 1):  # F:3115 do ncv = 1, ncvmax
        ktop[ncv] = 0  # F:3116 ktop(i,ncv)  = 0
        kbase[ncv] = 0  # F:3117 kbase(i,ncv) = 0
    rimaxentr = float.fromhex('0x0.0p+0')  # F:3125 rimaxentr = 0._r8
    for uw_k in range(2, pver + 1):  # F:3129 riex(2:pver) = ri(i,2:pver)
        riex[uw_k] = ri[uw_k]
    riex[(pver + 1)] = (rimaxentr - bflxs[1])  # F:3134 riex(pver+1) = rimaxentr - bflxs(i)
    ncv = 0  # F:3136 ncv = 0
    k = (pver + 1)  # F:3137 k   = pver + 1
    while (k > 2):  # F:3139 do while ( k .gt. ntop_turb + 1 )
        if (riex[k] < rimaxentr):  # F:3144 if( riex(k) .lt. rimaxentr ) then
            ncv = (ncv + 1)  # F:3148 ncv = ncv + 1
            kbase[ncv] = min((k + 1), (pver + 1))  # F:3153 kbase(i,ncv) = min(k+1,pver+1)
            while ((riex[k] < rimaxentr) and (k > 2)):  # F:3157 do while( riex(k) .lt. rimaxentr .and. k .gt. ntop_turb + 1 )
                k = (k - 1)  # F:3158 k = k - 1
            ktop[ncv] = k  # F:3164 ktop(i,ncv) = k
        else:
            k = (k - 1)  # F:3170 k = k - 1
    ncvfin[1] = ncv  # F:3176 ncvfin(i) = ncv
    return  # F:3180 return

def zisocl(pver, z, zi, n2, s2, bprod, sprod, bflxs, tkes, ncvfin, kbase, ktop, belongcv, ricl, ghcl, shcl, smcl, lbrk, wbrk, ebrk, extend, extend_up, extend_dn):
    ncvmax = pver
    nbot_turb = pver
    cpair = CPAIR
    rair = RAIR
    zvir = ZVIR
    latvap = LATVAP
    latice = LATICE
    g = GRAVIT
    vk = KARMAN
    b123 = B123
    latsub = latvap + latice
    ccon = (float.fromhex('0x1.1000000000000p+3') * float.fromhex('0x1.999999999999ap-4')) * vk
    for k in range(1, ncvmax + 1, 1):  # F:3308 do k = 1, ncvmax
        ricl[k] = float.fromhex('0x0.0p+0')  # F:3309 ricl(i,k) = 0._r8
        ghcl[k] = float.fromhex('0x0.0p+0')  # F:3310 ghcl(i,k) = 0._r8
        shcl[k] = float.fromhex('0x0.0p+0')  # F:3311 shcl(i,k) = 0._r8
        smcl[k] = float.fromhex('0x0.0p+0')  # F:3312 smcl(i,k) = 0._r8
        lbrk[k] = float.fromhex('0x0.0p+0')  # F:3313 lbrk(i,k) = 0._r8
        wbrk[k] = float.fromhex('0x0.0p+0')  # F:3314 wbrk(i,k) = 0._r8
        ebrk[k] = float.fromhex('0x0.0p+0')  # F:3315 ebrk(i,k) = 0._r8
    extend[1] = False  # F:3317 extend    = .false.
    extend_up[1] = False  # F:3318 extend_up = .false.
    extend_dn[1] = False  # F:3319 extend_dn = .false.
    ncv = 1  # F:3325 ncv = 1
    while (ncv <= ncvfin[1]):  # F:3327 do while( ncv .le. ncvfin(i) )
        ncvinit = ncv  # F:3329 ncvinit = ncv
        cntu = 0  # F:3330 cntu    = 0
        cntd = 0  # F:3331 cntd    = 0
        kb = kbase[ncv]  # F:3332 kb      = kbase(i,ncv)
        kt = ktop[ncv]  # F:3333 kt      = ktop(i,ncv)
        lbulk = (zi[kt] - zi[kb])  # F:3359 lbulk      = zi(i,kt) - zi(i,kb)
        dlint_surf = float.fromhex('0x0.0p+0')  # F:3360 dlint_surf = 0._r8
        dl2n2_surf = float.fromhex('0x0.0p+0')  # F:3361 dl2n2_surf = 0._r8
        dl2s2_surf = float.fromhex('0x0.0p+0')  # F:3362 dl2s2_surf = 0._r8
        dw_surf = float.fromhex('0x0.0p+0')  # F:3363 dw_surf    = 0._r8
        if (kb == (pver + 1)):  # F:3364 if( kb .eq. pver+1 ) then
            if (bflxs[1] > float.fromhex('0x0.0p+0')):  # F:3366 if( bflxs(i) .gt. 0._r8 ) then
                gg = ((((float.fromhex('0x1.0000000000000p-1') * vk) * z[pver]) * bprod[(pver + 1)]) / math.pow(tkes[1], float.fromhex('0x1.8000000000000p+0')))  # F:3373 gg    = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                gh = (gg / (float.fromhex('0x1.65aee631f8a09p-1') - (gg * float.fromhex('-0x1.15694467381d8p+5'))))  # F:3374 gh    = gg/(alph5-gg*alph3)
                gh = uw_fmin(uw_fmax(gh, float.fromhex('-0x1.c4467381d7dbfp+1')), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3376 gh    = min(max(gh,-3.5334_r8),0.0233_r8)
                sh = (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh)))  # F:3377 sh    = alph5/(1._r8+alph3*gh)
                sm = (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.88240b780346ep+2') * gh)))  # F:3378 sm    = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
                ricll = uw_fmin((-((sm / sh) * (bprod[(pver + 1)] / sprod[(pver + 1)]))), float.fromhex('0x1.851eb851eb852p-3'))  # F:3379 ricll = min(-(sm/sh)*(bprod(i,pver+1)/sprod(i,pver+1)),ricrit)
                dlint_surf = z[pver]  # F:3387 dlint_surf = z(i,pver)
                dl2n2_surf = (-(((vk * (z[pver] * z[pver])) * bprod[(pver + 1)]) / (sh * math.sqrt(tkes[1]))))  # F:3388 dl2n2_surf = -vk*(z(i,pver)**2)*bprod(i,pver+1)/(sh*sqrt(tkes(i)))
                dl2s2_surf = (((vk * (z[pver] * z[pver])) * sprod[(pver + 1)]) / (sm * math.sqrt(tkes[1])))  # F:3389 dl2s2_surf =  vk*(z(i,pver)**2)*sprod(i,pver+1)/(sm*sqrt(tkes(i)))
                dw_surf = ((tkes[1] / float.fromhex('0x1.7333333333333p+2')) * z[pver])  # F:3390 dw_surf    = (tkes(i)/b1)*z(i,pver)
            else:
                lbulk = (zi[kt] - z[pver])  # F:3396 lbulk = zi(i,kt) - z(i,pver)
        lint = dlint_surf  # F:3406 lint = dlint_surf
        l2n2 = dl2n2_surf  # F:3407 l2n2 = dl2n2_surf
        l2s2 = dl2s2_surf  # F:3408 l2s2 = dl2s2_surf
        wint = dw_surf  # F:3409 wint = dw_surf
        l2n2 = float.fromhex('0x0.0p+0')  # F:3411 l2n2 = 0._r8
        l2s2 = float.fromhex('0x0.0p+0')  # F:3412 l2s2 = 0._r8
        if (kt < (kb - 1)):  # F:3421 if( kt .lt. kb - 1 ) then
            for k in range((kb - 1), (kt + 1) - 1, -1):  # F:3423 do k = kb - 1, kt + 1, -1
                tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3426 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                lz = math.pow((math.pow((vk * zi[k]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3434 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                dzinc = (z[(k - 1)] - z[k])  # F:3439 dzinc = z(i,k-1) - z(i,k)
                l2n2 = (l2n2 + (((lz * lz) * n2[k]) * dzinc))  # F:3440 l2n2  = l2n2 + lz*lz*n2(i,k)*dzinc
                l2s2 = (l2s2 + (((lz * lz) * s2[k]) * dzinc))  # F:3441 l2s2  = l2s2 + lz*lz*s2(i,k)*dzinc
                lint = (lint + dzinc)  # F:3442 lint  = lint + dzinc
            ricll = uw_fmin((l2n2 / uw_fmax(l2s2, float.fromhex('0x1.19799812dea11p-40'))), float.fromhex('0x1.851eb851eb852p-3'))  # F:3451 ricll = min(l2n2/max(l2s2,ntzero),ricrit)
            trma = ((float.fromhex('0x1.a8f03ff93f46ap+7') * ricll) + (float.fromhex('0x1.7333333333333p+3') * (float.fromhex('-0x1.174bc6a7ef9dbp+2') - (float.fromhex('-0x1.11f3168d8b188p+2') * ricll))))  # F:3452 trma  = alph3*alph4*ricll+2._r8*b1*(alph2-alph4*alph5*ricll)
            trmb = ((ricll * float.fromhex('-0x1.466dc5d638866p+5')) + (float.fromhex('0x1.7333333333333p+3') * ((-(float.fromhex('0x1.65aee631f8a09p-1') * ricll)) + float.fromhex('0x1.1cc63f141205cp-1'))))  # F:3453 trmb  = ricll*(alph3+alph4)+2._r8*b1*(-alph5*ricll+alph1)
            trmc = ricll  # F:3454 trmc  = ricll
            det = uw_fmax(((trmb * trmb) - ((float.fromhex('0x1.0000000000000p+2') * trma) * trmc)), float.fromhex('0x0.0p+0'))  # F:3455 det   = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = ((((-trmb) + math.sqrt(det)) / float.fromhex('0x1.0000000000000p+1')) / trma)  # F:3456 gh    = (-trmb + sqrt(det))/2._r8/trma
            gh = uw_fmin(uw_fmax(gh, float.fromhex('-0x1.c4467381d7dbfp+1')), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3458 gh    = min(max(gh,-3.5334_r8),0.0233_r8)
            sh = (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh)))  # F:3459 sh    = alph5/(1._r8+alph3*gh)
            sm = (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.88240b780346ep+2') * gh)))  # F:3460 sm    = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
            wint = ((wint - (sh * l2n2)) + (sm * l2s2))  # F:3461 wint  = wint - sh*l2n2 + sm*l2s2
        else:
            lint = dlint_surf  # F:3472 lint = dlint_surf
            l2n2 = dl2n2_surf  # F:3473 l2n2 = dl2n2_surf
            l2s2 = dl2s2_surf  # F:3474 l2s2 = dl2s2_surf
            wint = dw_surf  # F:3475 wint = dw_surf
        l2n2 = (-uw_fmin((-l2n2), ((float.fromhex('0x1.4000000000000p+4') * lint) / (float.fromhex('0x1.7333333333333p+2') * sh))))  # F:3499 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
        l2s2 = uw_fmin(l2s2, ((float.fromhex('0x1.4000000000000p+4') * lint) / (float.fromhex('0x1.7333333333333p+2') * sm)))  # F:3500 l2s2 =  min( l2s2, tkemax*lint/(b1*sm))
        extend[1] = False  # F:3530 extend = .false.
        tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3535 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
        lz = math.pow((math.pow((vk * zi[kt]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3543 lz = ( (vk*zi(i,kt))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
        dzinc = (z[(kt - 1)] - z[kt])  # F:3549 dzinc = z(i,kt-1)-z(i,kt)
        dl2n2 = (((lz * lz) * n2[kt]) * dzinc)  # F:3550 dl2n2 = lz*lz*n2(i,kt)*dzinc
        dl2s2 = (((lz * lz) * s2[kt]) * dzinc)  # F:3551 dl2s2 = lz*lz*s2(i,kt)*dzinc
        dwinc = ((-(sh * dl2n2)) + (sm * dl2s2))  # F:3552 dwinc = -sh*dl2n2 + sm*dl2s2
        while (((-dl2n2) > (-((float.fromhex('-0x1.47ae147ae147bp-5') * l2n2) / float.fromhex('0x1.0a3d70a3d70a4p+0')))) and ((kt - 1) > 1)):  # F:3563 do while ( -dl2n2 .gt. (-rinc*l2n2/(1._r8-rinc)) .and. kt-1 .gt. ntop_turb )
            lint = (lint + dzinc)  # F:3574 lint = lint + dzinc
            l2n2 = (l2n2 + dl2n2)  # F:3575 l2n2 = l2n2 + dl2n2
            l2n2 = (-uw_fmin((-l2n2), ((float.fromhex('0x1.4000000000000p+4') * lint) / (float.fromhex('0x1.7333333333333p+2') * sh))))  # F:3576 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
            l2s2 = (l2s2 + dl2s2)  # F:3577 l2s2 = l2s2 + dl2s2
            wint = (wint + dwinc)  # F:3578 wint = wint + dwinc
            kt = (kt - 1)  # F:3582 kt        = kt - 1
            extend[1] = True  # F:3583 extend    = .true.
            extend_up[1] = True  # F:3584 extend_up = .true.
            if (kt == 1):  # F:3585 if( kt .eq. ntop_turb ) then
                raise RuntimeError("Fortran STOP at F:3590")  # F:3590 stop
            ktinc = (kbase[((ncv + cntu) + 1)] - 1)  # F:3599 ktinc = kbase(i,ncv+cntu+1) - 1
            if (kt == ktinc):  # F:3601 if( kt .eq. ktinc ) then
                for k in range((kbase[((ncv + cntu) + 1)] - 1), (ktop[((ncv + cntu) + 1)] + 1) - 1, -1):  # F:3603 do k = kbase(i,ncv+cntu+1) - 1, ktop(i,ncv+cntu+1) + 1, -1
                    tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3606 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                    lz = math.pow((math.pow((vk * zi[k]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3614 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                    dzinc = (z[(k - 1)] - z[k])  # F:3620 dzinc = z(i,k-1)-z(i,k)
                    dl2n2 = (((lz * lz) * n2[k]) * dzinc)  # F:3621 dl2n2 = lz*lz*n2(i,k)*dzinc
                    dl2s2 = (((lz * lz) * s2[k]) * dzinc)  # F:3622 dl2s2 = lz*lz*s2(i,k)*dzinc
                    dwinc = ((-(sh * dl2n2)) + (sm * dl2s2))  # F:3623 dwinc = -sh*dl2n2 + sm*dl2s2
                    lint = (lint + dzinc)  # F:3625 lint = lint + dzinc
                    l2n2 = (l2n2 + dl2n2)  # F:3626 l2n2 = l2n2 + dl2n2
                    l2n2 = (-uw_fmin((-l2n2), ((float.fromhex('0x1.4000000000000p+4') * lint) / (float.fromhex('0x1.7333333333333p+2') * sh))))  # F:3627 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                    l2s2 = (l2s2 + dl2s2)  # F:3628 l2s2 = l2s2 + dl2s2
                    wint = (wint + dwinc)  # F:3629 wint = wint + dwinc
                kt = ktop[((ncv + cntu) + 1)]  # F:3633 kt        = ktop(i,ncv+cntu+1)
                ncvfin[1] = (ncvfin[1] - 1)  # F:3634 ncvfin(i) = ncvfin(i) - 1
                cntu = (cntu + 1)  # F:3635 cntu      = cntu + 1
            tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3643 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
            lz = math.pow((math.pow((vk * zi[kt]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3651 lz = ( (vk*zi(i,kt))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            dzinc = (z[(kt - 1)] - z[kt])  # F:3657 dzinc = z(i,kt-1)-z(i,kt)
            dl2n2 = (((lz * lz) * n2[kt]) * dzinc)  # F:3658 dl2n2 = lz*lz*n2(i,kt)*dzinc
            dl2s2 = (((lz * lz) * s2[kt]) * dzinc)  # F:3659 dl2s2 = lz*lz*s2(i,kt)*dzinc
            dwinc = ((-(sh * dl2n2)) + (sm * dl2s2))  # F:3660 dwinc = -sh*dl2n2 + sm*dl2s2
        if (cntu > 0):  # F:3670 if( cntu .gt. 0 ) then
            for incv in range(1, (ncvfin[1] - ncv) + 1, 1):  # F:3671 do incv = 1, ncvfin(i) - ncv
                kbase[(ncv + incv)] = kbase[((ncv + cntu) + incv)]  # F:3672 kbase(i,ncv+incv) = kbase(i,ncv+cntu+incv)
                ktop[(ncv + incv)] = ktop[((ncv + cntu) + incv)]  # F:3673 ktop(i,ncv+incv)  = ktop(i,ncv+cntu+incv)
        if (kb != (pver + 1)):  # F:3681 if( kb .ne. pver + 1 ) then
            tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3686 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
            lz = math.pow((math.pow((vk * zi[kb]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3694 lz = ( (vk*zi(i,kb))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            dzinc = (z[(kb - 1)] - z[kb])  # F:3700 dzinc = z(i,kb-1)-z(i,kb)
            dl2n2 = (((lz * lz) * n2[kb]) * dzinc)  # F:3701 dl2n2 = lz*lz*n2(i,kb)*dzinc
            dl2s2 = (((lz * lz) * s2[kb]) * dzinc)  # F:3702 dl2s2 = lz*lz*s2(i,kb)*dzinc
            dwinc = ((-(sh * dl2n2)) + (sm * dl2s2))  # F:3703 dwinc = -sh*dl2n2 + sm*dl2s2
            while (((-dl2n2) > (-((float.fromhex('-0x1.47ae147ae147bp-5') * l2n2) / float.fromhex('0x1.0a3d70a3d70a4p+0')))) and (kb != (pver + 1))):  # F:3716 do while( ( -dl2n2 .gt. (-rinc*l2n2/(1._r8-rinc)) ) .and.(kb.ne.pver+1))
                lint = (lint + dzinc)  # F:3721 lint = lint + dzinc
                l2n2 = (l2n2 + dl2n2)  # F:3722 l2n2 = l2n2 + dl2n2
                l2n2 = (-uw_fmin((-l2n2), ((float.fromhex('0x1.4000000000000p+4') * lint) / (float.fromhex('0x1.7333333333333p+2') * sh))))  # F:3723 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                l2s2 = (l2s2 + dl2s2)  # F:3724 l2s2 = l2s2 + dl2s2
                wint = (wint + dwinc)  # F:3725 wint = wint + dwinc
                kb = (kb + 1)  # F:3729 kb        =  kb + 1
                extend[1] = True  # F:3730 extend    = .true.
                extend_dn[1] = True  # F:3731 extend_dn = .true.
                kbinc = 0  # F:3744 kbinc = 0
                if (ncv > 1):  # F:3745 if( ncv .gt. 1 ) kbinc = ktop(i,ncv-1) + 1
                    kbinc = (ktop[(ncv - 1)] + 1)  # F:3745 if( ncv .gt. 1 ) kbinc = ktop(i,ncv-1) + 1
                if (kb == kbinc):  # F:3746 if( kb .eq. kbinc ) then
                    for k in range((ktop[(ncv - 1)] + 1), (kbase[(ncv - 1)] - 1) + 1, 1):  # F:3748 do k =  ktop(i,ncv-1) + 1, kbase(i,ncv-1) - 1
                        tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3751 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                        lz = math.pow((math.pow((vk * zi[k]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3759 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                        dzinc = (z[(k - 1)] - z[k])  # F:3765 dzinc = z(i,k-1)-z(i,k)
                        dl2n2 = (((lz * lz) * n2[k]) * dzinc)  # F:3766 dl2n2 = lz*lz*n2(i,k)*dzinc
                        dl2s2 = (((lz * lz) * s2[k]) * dzinc)  # F:3767 dl2s2 = lz*lz*s2(i,k)*dzinc
                        dwinc = ((-(sh * dl2n2)) + (sm * dl2s2))  # F:3768 dwinc = -sh*dl2n2 + sm*dl2s2
                        lint = (lint + dzinc)  # F:3770 lint = lint + dzinc
                        l2n2 = (l2n2 + dl2n2)  # F:3771 l2n2 = l2n2 + dl2n2
                        l2n2 = (-uw_fmin((-l2n2), ((float.fromhex('0x1.4000000000000p+4') * lint) / (float.fromhex('0x1.7333333333333p+2') * sh))))  # F:3772 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                        l2s2 = (l2s2 + dl2s2)  # F:3773 l2s2 = l2s2 + dl2s2
                        wint = (wint + dwinc)  # F:3774 wint = wint + dwinc
                    kb = kbase[(ncv - 1)]  # F:3781 kb        = kbase(i,ncv-1)
                    ncv = (ncv - 1)  # F:3782 ncv       = ncv - 1
                    ncvfin[1] = (ncvfin[1] - 1)  # F:3783 ncvfin(i) = ncvfin(i) -1
                    cntd = (cntd + 1)  # F:3784 cntd      = cntd + 1
                if (kb == (pver + 1)):  # F:3792 if( kb .eq. pver + 1 ) then
                    if (bflxs[1] > float.fromhex('0x0.0p+0')):  # F:3794 if( bflxs(i) .gt. 0._r8 ) then
                        gg = ((((float.fromhex('0x1.0000000000000p-1') * vk) * z[pver]) * bprod[(pver + 1)]) / math.pow(tkes[1], float.fromhex('0x1.8000000000000p+0')))  # F:3796 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                        gh_surf = (gg / (float.fromhex('0x1.65aee631f8a09p-1') - (gg * float.fromhex('-0x1.15694467381d8p+5'))))  # F:3797 gh_surf = gg/(alph5-gg*alph3)
                        gh_surf = uw_fmin(uw_fmax(gh_surf, float.fromhex('-0x1.c4467381d7dbfp+1')), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3799 gh_surf = min(max(gh_surf,-3.5334_r8),0.0233_r8)
                        sh_surf = (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh_surf)))  # F:3800 sh_surf = alph5/(1._r8+alph3*gh_surf)
                        sm_surf = (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh_surf)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh_surf))) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.88240b780346ep+2') * gh_surf)))  # F:3801 sm_surf = (alph1 + alph2*gh_surf)/(1._r8+alph3*gh_surf)/(1._r8+alph4*gh_surf)
                        dlint_surf = z[pver]  # F:3804 dlint_surf = z(i,pver)
                        dl2n2_surf = (-(((vk * math.pow(z[pver], float.fromhex('0x1.0000000000000p+1'))) * bprod[(pver + 1)]) / (sh_surf * math.sqrt(tkes[1]))))  # F:3805 dl2n2_surf = -vk*(z(i,pver)**2._r8)*bprod(i,pver+1)/(sh_surf*sqrt(tkes(i)))
                        dl2s2_surf = (((vk * math.pow(z[pver], float.fromhex('0x1.0000000000000p+1'))) * sprod[(pver + 1)]) / (sm_surf * math.sqrt(tkes[1])))  # F:3806 dl2s2_surf =  vk*(z(i,pver)**2._r8)*sprod(i,pver+1)/(sm_surf*sqrt(tkes(i)))
                        dw_surf = ((tkes[1] / float.fromhex('0x1.7333333333333p+2')) * z[pver])  # F:3807 dw_surf = (tkes(i)/b1)*z(i,pver)
                    else:
                        dlint_surf = float.fromhex('0x0.0p+0')  # F:3809 dlint_surf = 0._r8
                        dl2n2_surf = float.fromhex('0x0.0p+0')  # F:3810 dl2n2_surf = 0._r8
                        dl2s2_surf = float.fromhex('0x0.0p+0')  # F:3811 dl2s2_surf = 0._r8
                        dw_surf = float.fromhex('0x0.0p+0')  # F:3812 dw_surf = 0._r8
                    lint = (lint + dlint_surf)  # F:3821 lint = lint + dlint_surf
                    l2n2 = (l2n2 + dl2n2_surf)  # F:3822 l2n2 = l2n2 + dl2n2_surf
                    l2n2 = (-uw_fmin((-l2n2), ((float.fromhex('0x1.4000000000000p+4') * lint) / (float.fromhex('0x1.7333333333333p+2') * sh))))  # F:3823 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                    l2s2 = (l2s2 + dl2s2_surf)  # F:3824 l2s2 = l2s2 + dl2s2_surf
                    wint = (wint + dw_surf)  # F:3825 wint = wint + dw_surf
                else:
                    tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3830 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                    lz = math.pow((math.pow((vk * zi[kb]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3838 lz = ( (vk*zi(i,kb))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                    dzinc = (z[(kb - 1)] - z[kb])  # F:3844 dzinc = z(i,kb-1)-z(i,kb)
                    dl2n2 = (((lz * lz) * n2[kb]) * dzinc)  # F:3845 dl2n2 = lz*lz*n2(i,kb)*dzinc
                    dl2s2 = (((lz * lz) * s2[kb]) * dzinc)  # F:3846 dl2s2 = lz*lz*s2(i,kb)*dzinc
                    dwinc = ((-(sh * dl2n2)) + (sm * dl2s2))  # F:3847 dwinc = -sh*dl2n2 + sm*dl2s2
            if ((kb == (pver + 1)) and (ncv != 1)):  # F:3853 if( (kb.eq.pver+1) .and. (ncv.ne.1) ) then
                raise RuntimeError("Fortran STOP at F:3858")  # F:3858 stop
        if (cntd > 0):  # F:3869 if( cntd .gt. 0 ) then
            for incv in range(1, (ncvfin[1] - ncv) + 1, 1):  # F:3870 do incv = 1, ncvfin(i) - ncv
                kbase[(ncv + incv)] = kbase[(ncvinit + incv)]  # F:3871 kbase(i,ncv+incv) = kbase(i,ncvinit+incv)
                ktop[(ncv + incv)] = ktop[(ncvinit + incv)]  # F:3872 ktop(i,ncv+incv)  = ktop(i,ncvinit+incv)
        if (wint < float.fromhex('0x1.47ae147ae147bp-7')):  # F:3878 if( wint .lt. 0.01_r8 ) then
            wint = float.fromhex('0x1.47ae147ae147bp-7')  # F:3879 wint = 0.01_r8
        if extend[1]:  # F:3892 if( extend ) then
            ktop[ncv] = kt  # F:3894 ktop(i,ncv)  = kt
            kbase[ncv] = kb  # F:3895 kbase(i,ncv) = kb
            lbulk = (zi[kt] - zi[kb])  # F:3901 lbulk      = zi(i,kt) - zi(i,kb)
            dlint_surf = float.fromhex('0x0.0p+0')  # F:3902 dlint_surf = 0._r8
            dl2n2_surf = float.fromhex('0x0.0p+0')  # F:3903 dl2n2_surf = 0._r8
            dl2s2_surf = float.fromhex('0x0.0p+0')  # F:3904 dl2s2_surf = 0._r8
            dw_surf = float.fromhex('0x0.0p+0')  # F:3905 dw_surf    = 0._r8
            if (kb == (pver + 1)):  # F:3906 if( kb .eq. pver + 1 ) then
                if (bflxs[1] > float.fromhex('0x0.0p+0')):  # F:3907 if( bflxs(i) .gt. 0._r8 ) then
                    gg = ((((float.fromhex('0x1.0000000000000p-1') * vk) * z[pver]) * bprod[(pver + 1)]) / math.pow(tkes[1], float.fromhex('0x1.8000000000000p+0')))  # F:3909 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                    gh = (gg / (float.fromhex('0x1.65aee631f8a09p-1') - (gg * float.fromhex('-0x1.15694467381d8p+5'))))  # F:3910 gh = gg/(alph5-gg*alph3)
                    gh = uw_fmin(uw_fmax(gh, float.fromhex('-0x1.c4467381d7dbfp+1')), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3912 gh = min(max(gh,-3.5334_r8),0.0233_r8)
                    sh = (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh)))  # F:3913 sh = alph5/(1._r8+alph3*gh)
                    sm = (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.88240b780346ep+2') * gh)))  # F:3914 sm = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
                    dlint_surf = z[pver]  # F:3917 dlint_surf = z(i,pver)
                    dl2n2_surf = (-(((vk * math.pow(z[pver], float.fromhex('0x1.0000000000000p+1'))) * bprod[(pver + 1)]) / (sh * math.sqrt(tkes[1]))))  # F:3918 dl2n2_surf = -vk*(z(i,pver)**2._r8)*bprod(i,pver+1)/(sh*sqrt(tkes(i)))
                    dl2s2_surf = (((vk * math.pow(z[pver], float.fromhex('0x1.0000000000000p+1'))) * sprod[(pver + 1)]) / (sm * math.sqrt(tkes[1])))  # F:3919 dl2s2_surf =  vk*(z(i,pver)**2._r8)*sprod(i,pver+1)/(sm*sqrt(tkes(i)))
                    dw_surf = ((tkes[1] / float.fromhex('0x1.7333333333333p+2')) * z[pver])  # F:3920 dw_surf    = (tkes(i)/b1)*z(i,pver)
                else:
                    lbulk = (zi[kt] - z[pver])  # F:3922 lbulk = zi(i,kt) - z(i,pver)
            lint = dlint_surf  # F:3925 lint = dlint_surf
            l2n2 = dl2n2_surf  # F:3926 l2n2 = dl2n2_surf
            l2s2 = dl2s2_surf  # F:3927 l2s2 = dl2s2_surf
            wint = dw_surf  # F:3928 wint = dw_surf
            l2n2 = float.fromhex('0x0.0p+0')  # F:3930 l2n2 = 0._r8
            l2s2 = float.fromhex('0x0.0p+0')  # F:3931 l2s2 = 0._r8
            for k in range((kt + 1), (kb - 1) + 1, 1):  # F:3940 do k = kt + 1, kb - 1
                tunlramp = float.fromhex('0x1.051eb851eb852p-3')  # F:3942 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                lz = math.pow((math.pow((vk * zi[k]), float.fromhex('-0x1.8000000000000p+1')) + math.pow((tunlramp * lbulk), float.fromhex('-0x1.8000000000000p+1'))), float.fromhex('-0x1.5555555555555p-2'))  # F:3950 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                dzinc = (z[(k - 1)] - z[k])  # F:3955 dzinc = z(i,k-1) - z(i,k)
                lint = (lint + dzinc)  # F:3956 lint = lint + dzinc
                l2n2 = (l2n2 + (((lz * lz) * n2[k]) * dzinc))  # F:3957 l2n2 = l2n2 + lz*lz*n2(i,k)*dzinc
                l2s2 = (l2s2 + (((lz * lz) * s2[k]) * dzinc))  # F:3958 l2s2 = l2s2 + lz*lz*s2(i,k)*dzinc
            ricll = uw_fmin((l2n2 / uw_fmax(l2s2, float.fromhex('0x1.19799812dea11p-40'))), float.fromhex('0x1.851eb851eb852p-3'))  # F:3961 ricll = min(l2n2/max(l2s2,ntzero),ricrit)
            trma = ((float.fromhex('0x1.a8f03ff93f46ap+7') * ricll) + (float.fromhex('0x1.7333333333333p+3') * (float.fromhex('-0x1.174bc6a7ef9dbp+2') - (float.fromhex('-0x1.11f3168d8b188p+2') * ricll))))  # F:3962 trma = alph3*alph4*ricll+2._r8*b1*(alph2-alph4*alph5*ricll)
            trmb = ((ricll * float.fromhex('-0x1.466dc5d638866p+5')) + (float.fromhex('0x1.7333333333333p+3') * ((-(float.fromhex('0x1.65aee631f8a09p-1') * ricll)) + float.fromhex('0x1.1cc63f141205cp-1'))))  # F:3963 trmb = ricll*(alph3+alph4)+2._r8*b1*(-alph5*ricll+alph1)
            trmc = ricll  # F:3964 trmc = ricll
            det = uw_fmax(((trmb * trmb) - ((float.fromhex('0x1.0000000000000p+2') * trma) * trmc)), float.fromhex('0x0.0p+0'))  # F:3965 det = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = ((((-trmb) + math.sqrt(det)) / float.fromhex('0x1.0000000000000p+1')) / trma)  # F:3966 gh = (-trmb + sqrt(det))/2._r8/trma
            gh = uw_fmin(uw_fmax(gh, float.fromhex('-0x1.c4467381d7dbfp+1')), float.fromhex('0x1.7dbf487fcb924p-6'))  # F:3968 gh = min(max(gh,-3.5334_r8),0.0233_r8)
            sh = (float.fromhex('0x1.65aee631f8a09p-1') / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh)))  # F:3969 sh = alph5 / (1._r8+alph3*gh)
            sm = (((float.fromhex('0x1.1cc63f141205cp-1') + (float.fromhex('-0x1.174bc6a7ef9dbp+2') * gh)) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.15694467381d8p+5') * gh))) / (float.fromhex('0x1.0000000000000p+0') + (float.fromhex('-0x1.88240b780346ep+2') * gh)))  # F:3970 sm = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
            wint = uw_fmax(((wint - (sh * l2n2)) + (sm * l2s2)), float.fromhex('0x1.47ae147ae147bp-7'))  # F:3974 wint = max( wint - sh*l2n2 + sm*l2s2, 0.01_r8 )
        lbrk[ncv] = lint  # F:3982 lbrk(i,ncv) = lint
        wbrk[ncv] = (wint / lint)  # F:3983 wbrk(i,ncv) = wint/lint
        ebrk[ncv] = (float.fromhex('0x1.7333333333333p+2') * wbrk[ncv])  # F:3984 ebrk(i,ncv) = b1*wbrk(i,ncv)
        ebrk[ncv] = uw_fmin(ebrk[ncv], float.fromhex('0x1.4000000000000p+4'))  # F:3985 ebrk(i,ncv) = min(ebrk(i,ncv),tkemax)
        ricl[ncv] = ricll  # F:3986 ricl(i,ncv) = ricll
        ghcl[ncv] = gh  # F:3987 ghcl(i,ncv) = gh
        shcl[ncv] = sh  # F:3988 shcl(i,ncv) = sh
        smcl[ncv] = sm  # F:3989 smcl(i,ncv) = sm
        ncv = (ncv + 1)  # F:3996 ncv = ncv + 1
    for ncv in range((ncvfin[1] + 1), ncvmax + 1, 1):  # F:4004 do ncv = ncvfin(i) + 1, ncvmax
        ktop[ncv] = 0  # F:4005 ktop(i,ncv)  = 0
        kbase[ncv] = 0  # F:4006 kbase(i,ncv) = 0
    for k in range(1, (pver + 1) + 1, 1):  # F:4014 do k = 1, pver + 1
        belongcv[k] = False  # F:4015 belongcv(i,k) = .false.
    for ncv in range(1, ncvfin[1] + 1, 1):  # F:4018 do ncv = 1, ncvfin(i)
        for k in range(ktop[ncv], kbase[ncv] + 1, 1):  # F:4019 do k = ktop(i,ncv), kbase(i,ncv)
            belongcv[k] = True  # F:4020 belongcv(i,k) = .true.
    return  # F:4024 return

def compute_cubic(a, b, c):
    qq = (((a * a) - (float.fromhex('0x1.8000000000000p+1') * b)) / float.fromhex('0x1.2000000000000p+3'))  # F:4038 qq = (a**2-3._r8*b)/9._r8
    rr = ((((float.fromhex('0x1.0000000000000p+1') * powi(a, 3)) - ((float.fromhex('0x1.2000000000000p+3') * a) * b)) + (float.fromhex('0x1.b000000000000p+4') * c)) / float.fromhex('0x1.b000000000000p+5'))  # F:4039 rr = (2._r8*a**3 - 9._r8*a*b + 27._r8*c)/54._r8
    dd = ((rr * rr) - powi(qq, 3))  # F:4041 dd = rr**2 - qq**3
    if (dd <= float.fromhex('0x0.0p+0')):  # F:4042 if( dd .le. 0._r8 ) then
        theta = math.acos((rr / math.pow(qq, float.fromhex('0x1.8000000000000p+0'))))  # F:4043 theta = acos(rr/qq**(3._r8/2._r8))
        x1 = ((-((float.fromhex('0x1.0000000000000p+1') * math.sqrt(qq)) * math.cos((theta / float.fromhex('0x1.8000000000000p+1'))))) - (a / float.fromhex('0x1.8000000000000p+1')))  # F:4044 x1 = -2._r8*sqrt(qq)*cos(theta/3._r8) - a/3._r8
        x2 = ((-((float.fromhex('0x1.0000000000000p+1') * math.sqrt(qq)) * math.cos(((theta + float.fromhex('0x1.921fb00000000p+2')) / float.fromhex('0x1.8000000000000p+1'))))) - (a / float.fromhex('0x1.8000000000000p+1')))  # F:4045 x2 = -2._r8*sqrt(qq)*cos((theta+2._r8*3.141592)/3._r8) - a/3._r8
        x3 = ((-((float.fromhex('0x1.0000000000000p+1') * math.sqrt(qq)) * math.cos(((theta - float.fromhex('0x1.921fb00000000p+2')) / float.fromhex('0x1.8000000000000p+1'))))) - (a / float.fromhex('0x1.8000000000000p+1')))  # F:4046 x3 = -2._r8*sqrt(qq)*cos((theta-2._r8*3.141592)/3._r8) - a/3._r8
        return uw_fmax(uw_fmax(uw_fmax(x1, x2), x3), float.fromhex('0x1.47ae147ae147bp-7'))  # F:4047 compute_cubic = max(max(max(x1,x2),x3),xmin)
    else:
        if (rr >= float.fromhex('0x0.0p+0')):  # F:4050 if( rr .ge. 0._r8 ) then
            aa = (-math.pow((math.sqrt(((rr * rr) - powi(qq, 3))) + rr), float.fromhex('0x1.5555555555555p-2')))  # F:4051 aa = -(sqrt(rr**2-qq**3)+rr)**(1._r8/3._r8)
        else:
            aa = math.pow((math.sqrt(((rr * rr) - powi(qq, 3))) - rr), float.fromhex('0x1.5555555555555p-2'))  # F:4053 aa =  (sqrt(rr**2-qq**3)-rr)**(1._r8/3._r8)
        if (aa == float.fromhex('0x0.0p+0')):  # F:4055 if( aa .eq. 0._r8 ) then
            bb = float.fromhex('0x0.0p+0')  # F:4056 bb = 0._r8
        else:
            bb = (qq / aa)  # F:4058 bb = qq/aa
        return uw_fmax(((aa + bb) - (a / float.fromhex('0x1.8000000000000p+1'))), float.fromhex('0x1.47ae147ae147bp-7'))  # F:4060 compute_cubic = max((aa+bb)-a/3._r8,xmin)
