// Generated literal transcription. F: comments refer to module_cam_bl_eddy_diff.F.
// Constant expression subtrees are folded in source grouping to exact hex words.
#ifndef UWPBL_CALEDDY_CUH
#define UWPBL_CALEDDY_CUH
__device__ void uw_caleddy(int pver, V sl, V qt, V ql, V slv, V u, V v, V pi, V z, V zi, R8 qflx, R8 shflx, V slslope, V qtslope, V chu, V chs, V cmu, V cms, V sfuh, V sflh, V n2, V s2, V ri, R8 rrho, R8& pblh, R8 ustar, V kvh_in, V kvm_in, V kvh, V kvm, R8& tpert, R8& qpert, V qrlin, V kvf, V tke, bool wstarent, V bprod, V sprod, R8 minpblh, R8& wpert, R8& tkes, V turbtype_f, V sm_aw, V kbase_o, V ktop_o, R8& ncvfin_o, V kbase_mg, V ktop_mg, R8& ncvfin_mg, V kbase_f, V ktop_f, R8& ncvfin_f, V wet_CL, V web_CL, V jtbu_CL, V jbbu_CL, V evhc_CL, V jt2slv_CL, V n2ht_CL, V n2hb_CL, V lwp_CL, V opt_depth_CL, V radinvfrac_CL, V radf_CL, V wstar_CL, V wstar3fact_CL, V ebrk, V wbrk, V lbrk, V ricl, V ghcl, V shcl, V smcl, V gh_a, V sh_a, V sm_a, V ri_a, V leng, V wcap, R8& pblhp, V cld, R8& ipbl, R8& kpblh, V wsedl, Ws& ws) {
    const int ncvmax = pver;
    const int nbot_turb = pver;
    WsMark uw_mark(ws);
    VL belongcv = ws.i4((pver + 1));
    VL belongst = ws.i4((pver + 1));
    bool in_CL;
    bool extend;
    bool extend_up;
    bool extend_dn;
    int k;
    int ks;
    int ncvfin;
    int ncvf;
    int ncv;
    int ncvnew;
    int ncvsurf;
    VI kbase = ws.i4(ncvmax);
    VI ktop = ws.i4(ncvmax);
    int kb;
    int kt;
    int ktblw;
    VI turbtype = ws.i4((pver + 1));
    int ktopbl;
    R8 bflxs;
    R8 rcap;
    R8 jtzm;
    R8 jtsl;
    R8 jtqt;
    R8 jtbu;
    R8 jtu;
    R8 jtv;
    R8 jt2slv;
    R8 radf;
    R8 jbzm;
    R8 jbsl;
    R8 jbqt;
    R8 jbbu;
    R8 jbu;
    R8 jbv;
    R8 ch;
    R8 cm;
    R8 n2ht;
    R8 n2hb;
    R8 n2htsrcl;
    R8 gh;
    R8 sh;
    R8 sm;
    R8 lbulk;
    R8 dzht;
    R8 dzhb;
    R8 rootp;
    R8 evhc;
    R8 kentr;
    R8 lwp;
    R8 opt_depth;
    R8 radinvfrac;
    R8 wet;
    R8 web;
    R8 vyt;
    R8 vyb;
    R8 vut;
    R8 vub;
    R8 fact;
    R8 trma;
    R8 trmb;
    R8 trmc;
    R8 trmp;
    R8 trmq;
    R8 qq;
    R8 det;
    R8 gg;
    R8 dzhb5;
    R8 dzht5;
    V qrlw = ws.r8(pver);
    V cldeff = ws.r8(pver);
    R8 qleff;
    R8 tunlramp;
    R8 leng_imsi;
    R8 tke_imsi;
    R8 kvh_imsi;
    R8 kvm_imsi;
    R8 alph4exs;
    R8 ghmin;
    R8 sedfact;
    R8 cet;
    R8 ceb;
    R8 wstar;
    R8 wstar3;
    R8 wstar3fact;
    R8 rmin;
    R8 fmin;
    R8 rcrit;
    R8 fcrit;
    bool noroot;
    for (int uw_k = 1; uw_k <= pver; ++uw_k) { // F:1637 qrlw(:ncol,:pver) = qrlin(:ncol,:pver)
        qrlw(uw_k) = qrlin(uw_k);
    }
    for (k = 1; k <= pver; k += 1) { // F:1645 do k = 1, pver
        cldeff(k) = cld(k); // F:1650 cldeff(i,k) = cld(i,k)
    }
    alph4exs = R8(-0x1.88240b780346ep+2); // F:1659 alph4exs = alph4
    ghmin = R8(-0x1.c4467381d7dbfp+1); // F:1660 ghmin    = -3.5334_r8
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1677 wet_CL(i,:ncvmax)        = 0._r8
        wet_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1678 web_CL(i,:ncvmax)        = 0._r8
        web_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1679 jtbu_CL(i,:ncvmax)       = 0._r8
        jtbu_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1680 jbbu_CL(i,:ncvmax)       = 0._r8
        jbbu_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1681 evhc_CL(i,:ncvmax)       = 0._r8
        evhc_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1682 jt2slv_CL(i,:ncvmax)     = 0._r8
        jt2slv_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1683 n2ht_CL(i,:ncvmax)       = 0._r8
        n2ht_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1684 n2hb_CL(i,:ncvmax)       = 0._r8
        n2hb_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1685 lwp_CL(i,:ncvmax)        = 0._r8
        lwp_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1686 opt_depth_CL(i,:ncvmax)  = 0._r8
        opt_depth_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1687 radinvfrac_CL(i,:ncvmax) = 0._r8
        radinvfrac_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1688 radf_CL(i,:ncvmax)       = 0._r8
        radf_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1689 wstar_CL(i,:ncvmax)      = 0._r8
        wstar_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1690 wstar3fact_CL(i,:ncvmax) = 0._r8
        wstar3fact_CL(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1691 ricl(i,:ncvmax)          = 0._r8
        ricl(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1692 ghcl(i,:ncvmax)          = 0._r8
        ghcl(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1693 shcl(i,:ncvmax)          = 0._r8
        shcl(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1694 smcl(i,:ncvmax)          = 0._r8
        smcl(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1695 ebrk(i,:ncvmax)          = 0._r8
        ebrk(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1696 wbrk(i,:ncvmax)          = 0._r8
        wbrk(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= ncvmax; ++uw_k) { // F:1697 lbrk(i,:ncvmax)          = 0._r8
        lbrk(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1698 gh_a(i,:pver+1)          = 0._r8
        gh_a(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1699 sh_a(i,:pver+1)          = 0._r8
        sh_a(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1700 sm_a(i,:pver+1)          = 0._r8
        sm_a(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1701 ri_a(i,:pver+1)          = 0._r8
        ri_a(uw_k) = R8(0x0.0p+0);
    }
    for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1702 sm_aw(i,:pver+1)         = 0._r8
        sm_aw(uw_k) = R8(0x0.0p+0);
    }
    ipbl = R8(0x0.0p+0); // F:1703 ipbl(i)                  = 0._r8
    kpblh = uw_real(pver); // F:1704 kpblh(i)                 = real(pver,r8)
    for (k = 1; k <= (pver + 1); k += 1) { // F:1713 do k = 1, pver + 1
        kvh(k) = R8(0x0.0p+0); // F:1720 kvh(i,k) = 0._r8
        kvm(k) = R8(0x0.0p+0); // F:1721 kvm(i,k) = 0._r8
        wcap(k) = R8(0x0.0p+0); // F:1724 wcap(i,k) = 0._r8
        leng(k) = R8(0x0.0p+0); // F:1725 leng(i,k) = 0._r8
        tke(k) = R8(0x0.0p+0); // F:1726 tke(i,k)  = 0._r8
        turbtype(k) = 0; // F:1727 turbtype(i,k) = 0
    }
    for (k = 2; k <= pver; k += 1) { // F:1739 do k = 2, pver
        bprod(k) = (-(kvh_in(k) * n2(k))); // F:1741 bprod(i,k) = -kvh_in(i,k) * n2(i,k)
        sprod(k) = (kvm_in(k) * s2(k)); // F:1742 sprod(i,k) =  kvm_in(i,k) * s2(i,k)
    }
    bprod(1) = R8(0x0.0p+0); // F:1762 bprod(i,1) = 0._r8
    sprod(1) = R8(0x0.0p+0); // F:1763 sprod(i,1) = 0._r8
    ch = ((chu((pver + 1)) * (R8(0x1.0000000000000p+0) - sflh(pver))) + (chs((pver + 1)) * sflh(pver))); // F:1764 ch = chu(i,pver+1) * ( 1._r8 - sflh(i,pver) ) + chs(i,pver+1) * sflh(i,pver)
    cm = ((cmu((pver + 1)) * (R8(0x1.0000000000000p+0) - sflh(pver))) + (cms((pver + 1)) * sflh(pver))); // F:1765 cm = cmu(i,pver+1) * ( 1._r8 - sflh(i,pver) ) + cms(i,pver+1) * sflh(i,pver)
    bflxs = (((ch * shflx) * rrho) + ((cm * qflx) * rrho)); // F:1766 bflxs(i) = ch * shflx(i) * rrho(i) + cm * qflx(i) * rrho(i)
    bprod((pver + 1)) = bflxs; // F:1768 bprod(i,pver+1) = bflxs(i)
    sprod((pver + 1)) = (uw_powi(ustar, 3) / (uw_vk * z(pver))); // F:1772 sprod(i,pver+1) = (ustar(i)**3)/(vk*z(i,pver))
    uw_exacol(pver, ri, bflxs, minpblh, zi, ktop, kbase, ncvfin, ws); // F:1787 call exacol( pcols, pver, ncol, ri, bflxs, minpblh, zi, ktop, kbase, ncvfin )
    for (k = 1; k <= ncvmax; k += 1) { // F:1792 do k = 1, ncvmax
        kbase_o(k) = uw_real(kbase(k)); // F:1793 kbase_o(i,k) = real(kbase(i,k),r8)
        ktop_o(k) = uw_real(ktop(k)); // F:1794 ktop_o(i,k)  = real(ktop(i,k),r8)
        ncvfin_o = uw_real(ncvfin); // F:1795 ncvfin_o(i)  = real(ncvfin(i),r8)
    }
    tkes = uw_pow(uw_max((((R8(0x1.7333333333333p+2) * uw_vk) * z(pver)) * (bprod((pver + 1)) + sprod((pver + 1)))), R8(0x1.ad7f29abcaf48p-24)), R8(0x1.5555555555555p-1)); // F:1822 tkes(i) = max(b1*vk*z(i,pver)*(bprod(i,pver+1)+sprod(i,pver+1)), 1.e-7_r8)**(2._r8/3._r8)
    tkes = uw_min(tkes, R8(0x1.4000000000000p+4)); // F:1823 tkes(i) = min(tkes(i), tkemax)
    tke((pver + 1)) = tkes; // F:1824 tke(i,pver+1)  = tkes(i)
    wcap((pver + 1)) = (tkes / R8(0x1.7333333333333p+2)); // F:1825 wcap(i,pver+1) = tkes(i)/b1
    ncvsurf = 0; // F:1856 ncvsurf = 0
    if ((ncvfin > 0)) { // F:1857 if( ncvfin(i) .gt. 0 ) then
        uw_zisocl(pver, z, zi, n2, s2, bprod, sprod, bflxs, tkes, ncvfin, kbase, ktop, belongcv, ricl, ghcl, shcl, smcl, lbrk, wbrk, ebrk, extend, extend_up, extend_dn, ws); // F:1858 call zisocl( pcols  , pver     , i        , z      , zi       , n2       , s2      , bprod  , sprod    , bflxs    , tkes    , ncvfin , kbase    , ktop     , belongcv, ricl   , ghcl     , shcl     , smcl    , lbrk   , wbrk     , ebrk     , extend , extend_up, extend_dn )
        if ((kbase(1) == (pver + 1))) { // F:1865 if( kbase(i,1) .eq. pver + 1 ) ncvsurf = 1
            ncvsurf = 1; // F:1865 if( kbase(i,1) .eq. pver + 1 ) ncvsurf = 1
        }
    } else {
        for (int uw_k = 1; uw_k <= (pver + 1); ++uw_k) { // F:1867 belongcv(i,:) = .false.
            belongcv(uw_k) = false;
        }
    }
    for (k = 1; k <= ncvmax; k += 1) { // F:1873 do k = 1, ncvmax
        kbase_mg(k) = uw_widen((float)(kbase(k))); // F:1874 kbase_mg(i,k) = real(kbase(i,k))
        ktop_mg(k) = uw_widen((float)(ktop(k))); // F:1875 ktop_mg(i,k)  = real(ktop(i,k))
        ncvfin_mg = uw_widen((float)(ncvfin)); // F:1876 ncvfin_mg(i)  = real(ncvfin(i))
    }
    ncv = 1; // F:1921 ncv  = 1
    ncvf = ncvfin; // F:1922 ncvf = ncvfin(i)
    for (k = nbot_turb; k >= 2; k += -1) { // F:1926 do k = nbot_turb, ntop_turb + 1, -1
        if (((((ql(k) > R8(0x1.4f8b588e368f1p-17)) && (ql((k - 1)) < R8(0x1.4f8b588e368f1p-17))) && (qrlw(k) < R8(0x0.0p+0))) && (ri(k) >= R8(0x1.851eb851eb852p-3)))) { // F:1928 if( ql(i,k) .gt. qmin .and. ql(i,k-1) .lt. qmin .and. qrlw(i,k) .lt. 0._r8 .and. ri(i,k) .ge. ricrit ) then
            if ((true && belongcv((k + 1)))) { // F:1938 if( choice_srcl .eq. 'nonamb' .and. belongcv(i,k+1) ) then
                continue; // F:1939 go to 220
            }
            ch = (((R8(0x1.0000000000000p+0) - sfuh(k)) * chu(k)) + (sfuh(k) * chs(k))); // F:1942 ch = ( 1._r8 - sfuh(i,k) ) * chu(i,k) + sfuh(i,k) * chs(i,k)
            cm = (((R8(0x1.0000000000000p+0) - sfuh(k)) * cmu(k)) + (sfuh(k) * cms(k))); // F:1943 cm = ( 1._r8 - sfuh(i,k) ) * cmu(i,k) + sfuh(i,k) * cms(i,k)
            n2htsrcl = ((ch * slslope(k)) + (cm * qtslope(k))); // F:1945 n2htsrcl = ch * slslope(i,k) + cm * qtslope(i,k)
            if ((n2htsrcl <= R8(0x0.0p+0))) { // F:1947 if( n2htsrcl .le. 0._r8 ) then
                in_CL = false; // F:1955 in_CL = .false.
                while ((ncv <= ncvf)) { // F:1957 do while ( ncv .le. ncvf )
                    if ((ktop(ncv) <= k)) { // F:1958 if( ktop(i,ncv) .le. k ) then
                        if ((kbase(ncv) > k)) { // F:1959 if( kbase(i,ncv) .gt. k ) then
                            in_CL = true; // F:1960 in_CL = .true.
                        }
                        break; // F:1962 exit
                    } else {
                        ncv = (ncv + 1); // F:1964 ncv = ncv + 1
                    }
                }
                if ((!in_CL)) { // F:1968 if( .not. in_CL ) then
                    ncvfin = (ncvfin + 1); // F:1972 ncvfin(i)       =  ncvfin(i) + 1
                    ncvnew = ncvfin; // F:1973 ncvnew          =  ncvfin(i)
                    ktop(ncvnew) = k; // F:1974 ktop(i,ncvnew)  =  k
                    kbase(ncvnew) = (k + 1); // F:1975 kbase(i,ncvnew) =  k+1
                    belongcv(k) = true; // F:1976 belongcv(i,k)   = .true.
                    belongcv((k + 1)) = true; // F:1977 belongcv(i,k+1) = .true.
                    if ((k < pver)) { // F:1994 if( k .lt. pver ) then
                        wbrk(ncvnew) = R8(0x0.0p+0); // F:1996 wbrk(i,ncvnew) = 0._r8
                        ebrk(ncvnew) = R8(0x0.0p+0); // F:1997 ebrk(i,ncvnew) = 0._r8
                        lbrk(ncvnew) = R8(0x0.0p+0); // F:1998 lbrk(i,ncvnew) = 0._r8
                        ghcl(ncvnew) = R8(0x0.0p+0); // F:1999 ghcl(i,ncvnew) = 0._r8
                        shcl(ncvnew) = R8(0x0.0p+0); // F:2000 shcl(i,ncvnew) = 0._r8
                        smcl(ncvnew) = R8(0x0.0p+0); // F:2001 smcl(i,ncvnew) = 0._r8
                        ricl(ncvnew) = R8(0x0.0p+0); // F:2002 ricl(i,ncvnew) = 0._r8
                    } else {
                        if ((bflxs > R8(0x0.0p+0))) { // F:2006 if( bflxs(i) .gt. 0._r8 ) then
                            ebrk(ncvnew) = tkes; // F:2010 ebrk(i,ncvnew) = tkes(i)
                            lbrk(ncvnew) = z(pver); // F:2011 lbrk(i,ncvnew) = z(i,pver)
                            wbrk(ncvnew) = (tkes / R8(0x1.7333333333333p+2)); // F:2012 wbrk(i,ncvnew) = tkes(i) / b1
                            for (ks = 1; ks <= ncvmax; ks += 1) { // F:2030 do ks = 1, ncvmax
                                ;
                            }
                            *ws.err = 2036; return; // F:2036 stop
                        } else {
                            ebrk(ncvnew) = R8(0x0.0p+0); // F:2040 ebrk(i,ncvnew) = 0._r8
                            lbrk(ncvnew) = R8(0x0.0p+0); // F:2041 lbrk(i,ncvnew) = 0._r8
                            wbrk(ncvnew) = R8(0x0.0p+0); // F:2042 wbrk(i,ncvnew) = 0._r8
                        }
                        gg = ((((R8(0x1.0000000000000p-1) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, R8(0x1.8000000000000p+0))); // F:2056 gg = 0.5_r8 * vk * z(i,pver) * bprod(i,pver+1) / ( tkes(i)**(3._r8/2._r8) )
                        if ((uw_abs((R8(0x1.65aee631f8a09p-1) - (gg * R8(-0x1.15694467381d8p+5)))) <= R8(0x1.ad7f29abcaf48p-24))) { // F:2057 if( abs(alph5-gg*alph3) .le. 1.e-7_r8 ) then
                            gh = ghmin; // F:2060 gh = ghmin
                        } else {
                            gh = (gg / (R8(0x1.65aee631f8a09p-1) - (gg * R8(-0x1.15694467381d8p+5)))); // F:2062 gh = gg / ( alph5 - gg * alph3 )
                        }
                        gh = uw_min(uw_max(gh, ghmin), R8(0x1.7dbf487fcb924p-6)); // F:2066 gh = min(max(gh,ghmin),0.0233_r8)
                        ghcl(ncvnew) = gh; // F:2067 ghcl(i,ncvnew) =  gh
                        shcl(ncvnew) = uw_max(R8(0x0.0p+0), (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh)))); // F:2068 shcl(i,ncvnew) =  max(0._r8,alph5/(1._r8+alph3*gh))
                        smcl(ncvnew) = uw_max(R8(0x0.0p+0), (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (alph4exs * gh)))); // F:2069 smcl(i,ncvnew) =  max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
                        ricl(ncvnew) = (-((smcl(ncvnew) / shcl(ncvnew)) * (bprod((pver + 1)) / sprod((pver + 1))))); // F:2070 ricl(i,ncvnew) = -(smcl(i,ncvnew)/shcl(i,ncvnew))*(bprod(i,pver+1)/sprod(i,pver+1))
                        ncvsurf = ncvnew; // F:2075 ncvsurf = ncvnew
                    }
                }
            }
        }
    }
    for (k = 1; k <= ncvmax; k += 1) { // F:2108 do k = 1, ncvmax
        kbase_f(k) = uw_widen((float)(kbase(k))); // F:2109 kbase_f(i,k) = real(kbase(i,k))
        ktop_f(k) = uw_widen((float)(ktop(k))); // F:2110 ktop_f(i,k)  = real(ktop(i,k))
        ncvfin_f = uw_widen((float)(ncvfin)); // F:2111 ncvfin_f(i)  = real(ncvfin(i))
    }
    ktblw = 0; // F:2134 ktblw = 0
    for (ncv = 1; ncv <= ncvfin; ncv += 1) { // F:2135 do ncv = 1, ncvfin(i)
        kt = ktop(ncv); // F:2137 kt = ktop(i,ncv)
        kb = kbase(ncv); // F:2138 kb = kbase(i,ncv)
        if (((kb == (pver + 1)) && (bflxs <= R8(0x0.0p+0)))) { // F:2140 if( kb .eq. (pver+1) .and. bflxs(i) .le. 0._r8 ) then
            lbulk = (zi(kt) - z(pver)); // F:2141 lbulk = zi(i,kt) - z(i,pver)
        } else {
            lbulk = (zi(kt) - zi(kb)); // F:2143 lbulk = zi(i,kt) - zi(i,kb)
        }
        for (k = uw_imin(kb, pver); k >= kt; k += -1) { // F:2152 do k = min(kb,pver), kt, -1
            tunlramp = (R8(0x1.5c28f5c28f5c3p-3) * (R8(0x1.0000000000000p+0) - (R8(0x1.0000000000000p-1) * uw_exp(uw_min(R8(0x0.0p+0), ricl(ncv)))))); // F:2158 tunlramp = ctunl*tunl*(1._r8-(1._r8-1._r8/ctunl)*exp(min(0._r8,ricl(i,ncv))))
            tunlramp = uw_min(uw_max(tunlramp, R8(0x1.5c28f5c28f5c3p-4)), R8(0x1.5c28f5c28f5c3p-3)); // F:2159 tunlramp = min(max(tunlramp,tunl),ctunl*tunl)
            leng(k) = uw_pow((uw_pow((uw_vk * zi(k)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:2167 leng(i,k) = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            wcap(k) = (uw_sq(leng(k)) * ((-(shcl(ncv) * n2(k))) + (smcl(ncv) * s2(k)))); // F:2172 wcap(i,k) = (leng(i,k)**2) * (-shcl(i,ncv)*n2(i,k)+smcl(i,ncv)*s2(i,k))
        }
        if ((kb < (pver + 1))) { // F:2178 if( kb .lt. pver+1 ) then
            jbzm = (z((kb - 1)) - z(kb)); // F:2180 jbzm = z(i,kb-1) - z(i,kb)
            jbsl = (sl((kb - 1)) - sl(kb)); // F:2181 jbsl = sl(i,kb-1) - sl(i,kb)
            jbqt = (qt((kb - 1)) - qt(kb)); // F:2182 jbqt = qt(i,kb-1) - qt(i,kb)
            jbbu = (n2(kb) * jbzm); // F:2183 jbbu = n2(i,kb) * jbzm
            jbbu = uw_max(jbbu, R8(0x1.0624dd2f1a9fcp-10)); // F:2184 jbbu = max(jbbu,jbumin)
            jbu = (u((kb - 1)) - u(kb)); // F:2185 jbu  = u(i,kb-1) - u(i,kb)
            jbv = (v((kb - 1)) - v(kb)); // F:2186 jbv  = v(i,kb-1) - v(i,kb)
            ch = (((R8(0x1.0000000000000p+0) - sflh((kb - 1))) * chu(kb)) + (sflh((kb - 1)) * chs(kb))); // F:2187 ch   = (1._r8 -sflh(i,kb-1))*chu(i,kb) + sflh(i,kb-1)*chs(i,kb)
            cm = (((R8(0x1.0000000000000p+0) - sflh((kb - 1))) * cmu(kb)) + (sflh((kb - 1)) * cms(kb))); // F:2188 cm   = (1._r8 -sflh(i,kb-1))*cmu(i,kb) + sflh(i,kb-1)*cms(i,kb)
            n2hb = (((ch * jbsl) + (cm * jbqt)) / jbzm); // F:2189 n2hb = (ch*jbsl + cm*jbqt)/jbzm
            vyb = ((n2hb * jbzm) / jbbu); // F:2190 vyb  = n2hb*jbzm/jbbu
            vub = uw_min(R8(0x1.0000000000000p+0), ((uw_sq(jbu) + uw_sq(jbv)) / (jbbu * jbzm))); // F:2191 vub  = min(1._r8,(jbu**2+jbv**2)/(jbbu*jbzm) )
        } else {
            jbbu = R8(0x0.0p+0); // F:2196 jbbu = 0._r8
            n2hb = R8(0x0.0p+0); // F:2197 n2hb = 0._r8
            vyb = R8(0x0.0p+0); // F:2198 vyb  = 0._r8
            vub = R8(0x0.0p+0); // F:2199 vub  = 0._r8
            web = R8(0x0.0p+0); // F:2200 web  = 0._r8
        }
        jtzm = (z((kt - 1)) - z(kt)); // F:2208 jtzm = z(i,kt-1) - z(i,kt)
        jtsl = (sl((kt - 1)) - sl(kt)); // F:2209 jtsl = sl(i,kt-1) - sl(i,kt)
        jtqt = (qt((kt - 1)) - qt(kt)); // F:2210 jtqt = qt(i,kt-1) - qt(i,kt)
        jtbu = (n2(kt) * jtzm); // F:2211 jtbu = n2(i,kt)*jtzm
        jtbu = uw_max(jtbu, R8(0x1.0624dd2f1a9fcp-10)); // F:2212 jtbu = max(jtbu,jbumin)
        jtu = (u((kt - 1)) - u(kt)); // F:2213 jtu  = u(i,kt-1) - u(i,kt)
        jtv = (v((kt - 1)) - v(kt)); // F:2214 jtv  = v(i,kt-1) - v(i,kt)
        ch = (((R8(0x1.0000000000000p+0) - sfuh(kt)) * chu(kt)) + (sfuh(kt) * chs(kt))); // F:2215 ch   = (1._r8 -sfuh(i,kt))*chu(i,kt) + sfuh(i,kt)*chs(i,kt)
        cm = (((R8(0x1.0000000000000p+0) - sfuh(kt)) * cmu(kt)) + (sfuh(kt) * cms(kt))); // F:2216 cm   = (1._r8 -sfuh(i,kt))*cmu(i,kt) + sfuh(i,kt)*cms(i,kt)
        n2ht = (((ch * jtsl) + (cm * jtqt)) / jtzm); // F:2217 n2ht = (ch*jtsl + cm*jtqt)/jtzm
        vyt = ((n2ht * jtzm) / jtbu); // F:2218 vyt  = n2ht*jtzm/jtbu
        vut = uw_min(R8(0x1.0000000000000p+0), ((uw_sq(jtu) + uw_sq(jtv)) / (jtbu * jtzm))); // F:2219 vut  = min(1._r8,(jtu**2+jtv**2)/(jtbu*jtzm))
        evhc = R8(0x1.0000000000000p+0); // F:2229 evhc   = 1._r8
        jt2slv = R8(0x0.0p+0); // F:2230 jt2slv = 0._r8
        qleff = uw_max(ql((kt - 1)), ql(kt)); // F:2253 qleff  = max( ql(i,kt-1), ql(i,kt) )
        jt2slv = (slv(uw_imax((kt - 2), 1)) - slv(kt)); // F:2254 jt2slv = slv(i,max(kt-2,1)) - slv(i,kt)
        jt2slv = uw_max(jt2slv, ((R8(0x1.0624dd2f1a9fcp-10) * slv((kt - 1))) / uw_g)); // F:2255 jt2slv = max( jt2slv, jbumin*slv(i,kt-1)/g )
        evhc = (R8(0x1.0000000000000p+0) + (((R8(0x1.8000000000000p+4) * uw_latvap) * qleff) / jt2slv)); // F:2256 evhc   = 1._r8 + a2l * a3l * latvap * qleff / jt2slv
        evhc = uw_min(evhc, R8(0x1.4000000000000p+3)); // F:2257 evhc   = min( evhc, evhcmax )
        lwp = R8(0x0.0p+0); // F:2268 lwp        = 0._r8
        opt_depth = R8(0x0.0p+0); // F:2269 opt_depth  = 0._r8
        radinvfrac = R8(0x0.0p+0); // F:2270 radinvfrac = 0._r8
        radf = R8(0x0.0p+0); // F:2271 radf       = 0._r8
        lwp = ((ql(kt) * (pi((kt + 1)) - pi(kt))) / uw_g); // F:2304 lwp         = ql(i,kt) * ( pi(i,kt+1) - pi(i,kt) ) / g
        opt_depth = (R8(0x1.3800000000000p+7) * lwp); // F:2305 opt_depth   = 156._r8 * lwp
        radinvfrac = ((opt_depth * (R8(0x1.0000000000000p+2) + opt_depth)) / ((R8(0x1.8000000000000p+2) * (R8(0x1.0000000000000p+2) + opt_depth)) + uw_sq(opt_depth))); // F:2306 radinvfrac  = opt_depth * ( 4._r8 + opt_depth ) / ( 6._r8 * ( 4._r8 + opt_depth ) + opt_depth**2 )
        radf = uw_max((((radinvfrac * qrlw(kt)) / (pi(kt) - pi((kt + 1)))) * (zi(kt) - zi((kt + 1)))), R8(0x0.0p+0)); // F:2307 radf        = max( radinvfrac * qrlw(i,kt) / ( pi(i,kt) - pi(i,kt+1) ) * ( zi(i,kt) - zi(i,kt+1) ), 0._r8 )
        lwp = ((ql((kt - 1)) * (pi(kt) - pi((kt - 1)))) / uw_g); // F:2309 lwp         = ql(i,kt-1) * ( pi(i,kt) - pi(i,kt-1) ) / g
        opt_depth = (R8(0x1.3800000000000p+7) * lwp); // F:2310 opt_depth   = 156._r8 * lwp
        radinvfrac = ((opt_depth * (R8(0x1.0000000000000p+2) + opt_depth)) / ((R8(0x1.8000000000000p+2) * (R8(0x1.0000000000000p+2) + opt_depth)) + uw_sq(opt_depth))); // F:2311 radinvfrac  = opt_depth * ( 4._r8 + opt_depth ) / ( 6._r8 * ( 4._r8 + opt_depth) + opt_depth**2 )
        radf = (radf + uw_max((((radinvfrac * qrlw((kt - 1))) / (pi((kt - 1)) - pi(kt))) * (zi((kt - 1)) - zi(kt))), R8(0x0.0p+0))); // F:2312 radf        = radf + max( radinvfrac * qrlw(i,kt-1) / ( pi(i,kt-1) - pi(i,kt) ) * ( zi(i,kt-1) - zi(i,kt) ), 0._r8 )
        radf = (uw_max(radf, R8(0x0.0p+0)) * chs(kt)); // F:2314 radf        = max( radf, 0._r8 ) * chs(i,kt)
        dzht = (zi(kt) - z(kt)); // F:2339 dzht   = zi(i,kt)  - z(i,kt)
        dzhb = (z((kb - 1)) - zi(kb)); // F:2340 dzhb   = z(i,kb-1) - zi(i,kb)
        wstar3 = (radf * dzht); // F:2341 wstar3 = radf * dzht
        for (k = (kt + 1); k <= (kb - 1); k += 1) { // F:2342 do k = kt + 1, kb - 1
            wstar3 = (wstar3 + (bprod(k) * (z((k - 1)) - z(k)))); // F:2343 wstar3 =  wstar3 + bprod(i,k) * ( z(i,k-1) - z(i,k) )
        }
        if (((kb == (pver + 1)) && (bflxs > R8(0x0.0p+0)))) { // F:2351 if( kb .eq. (pver+1) .and. bflxs(i) .gt. 0._r8 ) then
            wstar3 = (wstar3 + (bflxs * dzhb)); // F:2352 wstar3 = wstar3 + bflxs(i) * dzhb
        }
        wstar3 = uw_max((R8(0x1.4000000000000p+1) * wstar3), R8(0x0.0p+0)); // F:2355 wstar3 = max( 2.5_r8 * wstar3, 0._r8 )
        if ((wstar3 > R8(0x0.0p+0))) { // F:2408 if( wstar3 .gt. 0._r8 ) then
            cet = ((R8(0x1.999999999999ap-3) * evhc) / (jtbu * lbulk)); // F:2409 cet = a1i * evhc / ( jtbu * lbulk )
            if ((kb == (pver + 1))) { // F:2410 if( kb .eq. pver + 1 ) then
                wstar3fact = uw_max((R8(0x1.0000000000000p+0) + ((((R8(0x1.4000000000000p+1) * cet) * n2ht) * jtzm) * dzht)), R8(0x1.0000000000000p-1)); // F:2411 wstar3fact = max( 1._r8 + 2.5_r8 * cet * n2ht * jtzm * dzht, wstar3factcrit )
            } else {
                ceb = (R8(0x1.999999999999ap-3) / (jbbu * lbulk)); // F:2413 ceb = a1i / ( jbbu * lbulk )
                wstar3fact = uw_max(((R8(0x1.0000000000000p+0) + ((((R8(0x1.4000000000000p+1) * cet) * n2ht) * jtzm) * dzht)) + ((((R8(0x1.4000000000000p+1) * ceb) * n2hb) * jbzm) * dzhb)), R8(0x1.0000000000000p-1)); // F:2414 wstar3fact = max( 1._r8 + 2.5_r8 * cet * n2ht * jtzm * dzht + 2.5_r8 * ceb * n2hb * jbzm * dzhb, wstar3factcrit )
            }
            wstar3 = (wstar3 / wstar3fact); // F:2417 wstar3 = wstar3 / wstar3fact
        } else {
            wstar3fact = R8(0x0.0p+0); // F:2419 wstar3fact = 0._r8
            cet = R8(0x0.0p+0); // F:2420 cet        = 0._r8
            ceb = R8(0x0.0p+0); // F:2421 ceb        = 0._r8
        }
        fact = ((((evhc * ((-vyt) + vut)) * dzht) + (((((-vyb) + vub) * dzhb) * leng(kb)) / leng(kt))) / lbulk); // F:2465 fact = ( evhc * ( -vyt + vut ) * dzht + ( -vyb + vub ) * dzhb * leng(i,kb) / leng(i,kt) ) / lbulk
        trma = R8(0x1.0000000000000p+0); // F:2475 trma = 1._r8
        trmp = (((ebrk(ncv) * (lbrk(ncv) / lbulk)) / R8(0x1.8000000000000p+1)) + R8(0x1.19799812dea11p-40)); // F:2476 trmp = ebrk(i,ncv) * ( lbrk(i,ncv) / lbulk ) / 3._r8 + ntzero
        trmq = ((R8(0x1.7333333333333p+1) * (leng(kt) / lbulk)) * ((radf * dzht) + ((R8(0x1.999999999999ap-3) * fact) * wstar3))); // F:2477 trmq = 0.5_r8 * b1 * ( leng(i,kt)  / lbulk ) * ( radf * dzht + a1i * fact * wstar3 )
        rmin = uw_sqrt(trmp); // F:2483 rmin  = sqrt(trmp)
        fmin = ((rmin * ((rmin * rmin) - (R8(0x1.8000000000000p+1) * trmp))) - (R8(0x1.0000000000000p+1) * trmq)); // F:2484 fmin  = rmin * ( rmin * rmin - 3._r8 * trmp ) - 2._r8 * trmq
        wstar = uw_pow(wstar3, R8(0x1.5555555555555p-2)); // F:2485 wstar = wstar3**onet
        rcrit = (R8(0x1.0000000000000p-1) * wstar); // F:2486 rcrit = ccrit * wstar
        fcrit = ((rcrit * ((rcrit * rcrit) - (R8(0x1.8000000000000p+1) * trmp))) - (R8(0x1.0000000000000p+1) * trmq)); // F:2487 fcrit = rcrit * ( rcrit * rcrit - 3._r8 * trmp ) - 2._r8 * trmq
        noroot = (((rmin < rcrit) && (fcrit > R8(0x0.0p+0))) || ((rmin >= rcrit) && (fmin > R8(0x0.0p+0)))); // F:2499 noroot = ( ( rmin .lt. rcrit ) .and. ( fcrit .gt. 0._r8 ) ) .or. ( ( rmin .ge. rcrit ) .and. ( fmin  .gt. 0._r8 ) )
        if (noroot) { // F:2501 if( noroot ) then
            trma = (R8(0x1.0000000000000p+0) - ((((R8(0x1.7333333333333p+2) * (leng(kt) / lbulk)) * R8(0x1.999999999999ap-3)) * fact) / R8(0x1.0000000000000p-3))); // F:2502 trma = 1._r8 - b1 * ( leng(i,kt) / lbulk ) * a1i * fact / ccrit**3
            trma = uw_max(trma, R8(0x1.0000000000000p-1)); // F:2503 trma = max( trma, 0.5_r8 )
            trmp = (trmp / trma); // F:2504 trmp = trmp / trma
            trmq = ((((R8(0x1.7333333333333p+1) * (leng(kt) / lbulk)) * radf) * dzht) / trma); // F:2505 trmq = 0.5_r8 * b1 * ( leng(i,kt) / lbulk ) * radf * dzht / trma
        }
        qq = (uw_sq(trmq) - uw_powi(trmp, 3)); // F:2510 qq = trmq**2 - trmp**3
        if ((qq >= R8(0x0.0p+0))) { // F:2511 if( qq .ge. 0._r8 ) then
            rootp = (uw_pow((trmq + uw_sqrt(qq)), R8(0x1.5555555555555p-2)) + uw_pow(uw_max((trmq - uw_sqrt(qq)), R8(0x0.0p+0)), R8(0x1.5555555555555p-2))); // F:2512 rootp = ( trmq + sqrt(qq) )**(1._r8/3._r8) + ( max( trmq - sqrt(qq), 0._r8 ) )**(1._r8/3._r8)
        } else {
            rootp = ((R8(0x1.0000000000000p+1) * uw_sqrt(trmp)) * uw_cos((uw_acos((trmq / uw_sqrt(uw_powi(trmp, 3)))) / R8(0x1.8000000000000p+1)))); // F:2514 rootp = 2._r8 * sqrt(trmp) * cos( acos( trmq / sqrt(trmp**3) ) / 3._r8 )
        }
        if (noroot) { // F:2520 if( noroot )  wstar3 = ( rootp / ccrit )**3
            wstar3 = uw_powi((rootp / R8(0x1.0000000000000p-1)), 3); // F:2520 if( noroot )  wstar3 = ( rootp / ccrit )**3
        }
        wet = (cet * wstar3); // F:2521 wet = cet * wstar3
        if ((kb < (pver + 1))) { // F:2522 if( kb .lt. pver + 1 ) web = ceb * wstar3
            web = (ceb * wstar3); // F:2522 if( kb .lt. pver + 1 ) web = ceb * wstar3
        }
        ebrk(ncv) = uw_sq(rootp); // F:2553 ebrk(i,ncv) = rootp**2
        ebrk(ncv) = uw_min(ebrk(ncv), R8(0x1.4000000000000p+4)); // F:2554 ebrk(i,ncv) = min(ebrk(i,ncv),tkemax)
        wbrk(ncv) = (ebrk(ncv) / R8(0x1.7333333333333p+2)); // F:2555 wbrk(i,ncv) = ebrk(i,ncv)/b1
        if ((ebrk(ncv) <= R8(0x0.0p+0))) { // F:2567 if( ebrk(i,ncv) .le. 0._r8 ) then
            belongcv(kt) = false; // F:2572 belongcv(i,kt) = .false.
            belongcv(kb) = false; // F:2573 belongcv(i,kb) = .false.
        }
        for (k = (kb - 1); k >= (kt + 1); k += -1) { // F:2591 do k = kb - 1, kt + 1, -1
            rcap = ((R8(0x1.7333333333333p+2) + (wcap(k) / wbrk(ncv))) / R8(0x1.b333333333333p+2)); // F:2592 rcap = ( b1 * ae + wcap(i,k) / wbrk(i,ncv) ) / ( b1 * ae + 1._r8 )
            rcap = uw_min(uw_max(rcap, R8(0x1.999999999999ap-4)), R8(0x1.0000000000000p+1)); // F:2593 rcap = min( max(rcap,rcapmin), rcapmax )
            tke(k) = (ebrk(ncv) * rcap); // F:2594 tke(i,k) = ebrk(i,ncv) * rcap
            tke(k) = uw_min(tke(k), R8(0x1.4000000000000p+4)); // F:2595 tke(i,k) = min( tke(i,k), tkemax )
            kvh(k) = ((leng(k) * uw_sqrt(tke(k))) * shcl(ncv)); // F:2596 kvh(i,k) = leng(i,k) * sqrt(tke(i,k)) * shcl(i,ncv)
            kvm(k) = ((leng(k) * uw_sqrt(tke(k))) * smcl(ncv)); // F:2597 kvm(i,k) = leng(i,k) * sqrt(tke(i,k)) * smcl(i,ncv)
            bprod(k) = (-(kvh(k) * n2(k))); // F:2598 bprod(i,k) = -kvh(i,k) * n2(i,k)
            sprod(k) = (kvm(k) * s2(k)); // F:2599 sprod(i,k) =  kvm(i,k) * s2(i,k)
            turbtype(k) = 2; // F:2600 turbtype(i,k) = 2
            sm_aw(k) = (smcl(ncv) / R8(0x1.1cc63f141205cp-1)); // F:2601 sm_aw(i,k) = smcl(i,ncv)/alph1
        }
        kentr = (wet * jtzm); // F:2605 kentr = wet * jtzm
        kvh(kt) = kentr; // F:2606 kvh(i,kt) = kentr
        kvm(kt) = kentr; // F:2607 kvm(i,kt) = kentr
        bprod(kt) = ((-(kentr * n2ht)) + radf); // F:2608 bprod(i,kt) = -kentr * n2ht + radf
        sprod(kt) = (kentr * s2(kt)); // F:2609 sprod(i,kt) =  kentr * s2(i,kt)
        turbtype(kt) = 4; // F:2610 turbtype(i,kt) = 4
        trmp = R8(-0x1.b4b4b4b4b4b4bp-1); // F:2611 trmp = -b1 * ae / ( 1._r8 + b1 * ae )
        trmq = (-(((((bprod(kt) + sprod(kt)) * R8(0x1.7333333333333p+2)) * leng(kt)) / R8(0x1.b333333333333p+2)) / uw_pow(ebrk(ncv), R8(0x1.8000000000000p+0)))); // F:2612 trmq = -(bprod(i,kt)+sprod(i,kt))*b1*leng(i,kt)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
        rcap = uw_pow(uw_compute_cubic(R8(0x0.0p+0), trmp, trmq), R8(0x1.0000000000000p+1)); // F:2613 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
        rcap = uw_min(uw_max(rcap, R8(0x1.999999999999ap-4)), R8(0x1.0000000000000p+1)); // F:2614 rcap = min( max(rcap,rcapmin), rcapmax )
        tke(kt) = (ebrk(ncv) * rcap); // F:2615 tke(i,kt)  = ebrk(i,ncv) * rcap
        tke(kt) = uw_min(tke(kt), R8(0x1.4000000000000p+4)); // F:2616 tke(i,kt)  = min( tke(i,kt), tkemax )
        sm_aw(kt) = (smcl(ncv) / R8(0x1.1cc63f141205cp-1)); // F:2617 sm_aw(i,kt) = smcl(i,ncv) / alph1
        if ((kb < (pver + 1))) { // F:2625 if( kb .lt. pver + 1 ) then
            kentr = (web * jbzm); // F:2627 kentr = web * jbzm
            if ((kb != ktblw)) { // F:2629 if( kb .ne. ktblw ) then
                kvh(kb) = kentr; // F:2631 kvh(i,kb) = kentr
                kvm(kb) = kentr; // F:2632 kvm(i,kb) = kentr
                bprod(kb) = (-(kvh(kb) * n2hb)); // F:2633 bprod(i,kb) = -kvh(i,kb)*n2hb
                sprod(kb) = (kvm(kb) * s2(kb)); // F:2634 sprod(i,kb) =  kvm(i,kb)*s2(i,kb)
                turbtype(kb) = 3; // F:2635 turbtype(i,kb) = 3
                trmp = R8(-0x1.b4b4b4b4b4b4bp-1); // F:2636 trmp = -b1*ae/(1._r8+b1*ae)
                trmq = (-(((((bprod(kb) + sprod(kb)) * R8(0x1.7333333333333p+2)) * leng(kb)) / R8(0x1.b333333333333p+2)) / uw_pow(ebrk(ncv), R8(0x1.8000000000000p+0)))); // F:2637 trmq = -(bprod(i,kb)+sprod(i,kb))*b1*leng(i,kb)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
                rcap = uw_pow(uw_compute_cubic(R8(0x0.0p+0), trmp, trmq), R8(0x1.0000000000000p+1)); // F:2638 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
                rcap = uw_min(uw_max(rcap, R8(0x1.999999999999ap-4)), R8(0x1.0000000000000p+1)); // F:2639 rcap = min( max(rcap,rcapmin), rcapmax )
                tke(kb) = (ebrk(ncv) * rcap); // F:2640 tke(i,kb)  = ebrk(i,ncv) * rcap
                tke(kb) = uw_min(tke(kb), R8(0x1.4000000000000p+4)); // F:2641 tke(i,kb)  = min( tke(i,kb),tkemax )
            } else {
                kvh(kb) = (kvh(kb) + kentr); // F:2645 kvh(i,kb) = kvh(i,kb) + kentr
                kvm(kb) = (kvm(kb) + kentr); // F:2646 kvm(i,kb) = kvm(i,kb) + kentr
                dzhb5 = (z((kb - 1)) - zi(kb)); // F:2649 dzhb5 = z(i,kb-1) - zi(i,kb)
                dzht5 = (zi(kb) - z(kb)); // F:2650 dzht5 = zi(i,kb) - z(i,kb)
                bprod(kb) = (((dzht5 * bprod(kb)) - ((dzhb5 * kentr) * n2hb)) / (dzhb5 + dzht5)); // F:2651 bprod(i,kb) = ( dzht5*bprod(i,kb) - dzhb5*kentr*n2hb )     / ( dzhb5 + dzht5 )
                sprod(kb) = (((dzht5 * sprod(kb)) + ((dzhb5 * kentr) * s2(kb))) / (dzhb5 + dzht5)); // F:2652 sprod(i,kb) = ( dzht5*sprod(i,kb) + dzhb5*kentr*s2(i,kb) ) / ( dzhb5 + dzht5 )
                trmp = R8(-0x1.b4b4b4b4b4b4bp-1); // F:2653 trmp = -b1*ae/(1._r8+b1*ae)
                trmq = (-(((((kentr * (s2(kb) - n2hb)) * R8(0x1.7333333333333p+2)) * leng(kb)) / R8(0x1.b333333333333p+2)) / uw_pow(ebrk(ncv), R8(0x1.8000000000000p+0)))); // F:2654 trmq = -kentr*(s2(i,kb)-n2hb)*b1*leng(i,kb)/(1._r8+b1*ae)/(ebrk(i,ncv)**(3._r8/2._r8))
                rcap = uw_pow(uw_compute_cubic(R8(0x0.0p+0), trmp, trmq), R8(0x1.0000000000000p+1)); // F:2655 rcap = compute_cubic(0._r8,trmp,trmq)**2._r8
                rcap = uw_min(uw_max(rcap, R8(0x1.999999999999ap-4)), R8(0x1.0000000000000p+1)); // F:2656 rcap = min( max(rcap,rcapmin), rcapmax )
                tke_imsi = (ebrk(ncv) * rcap); // F:2657 tke_imsi = ebrk(i,ncv) * rcap
                tke_imsi = uw_min(tke_imsi, R8(0x1.4000000000000p+4)); // F:2658 tke_imsi = min( tke_imsi, tkemax )
                tke(kb) = (((dzht5 * tke(kb)) + (dzhb5 * tke_imsi)) / (dzhb5 + dzht5)); // F:2659 tke(i,kb)  = ( dzht5*tke(i,kb) + dzhb5*tke_imsi ) / ( dzhb5 + dzht5 )
                tke(kb) = uw_min(tke(kb), R8(0x1.4000000000000p+4)); // F:2660 tke(i,kb)  = min(tke(i,kb),tkemax)
                turbtype(kb) = 5; // F:2661 turbtype(i,kb) = 5
            }
        } else {
            rcap = ((R8(0x1.7333333333333p+2) + (wcap(kb) / wbrk(ncv))) / R8(0x1.b333333333333p+2)); // F:2671 rcap = (b1*ae + wcap(i,kb)/wbrk(i,ncv))/(b1*ae + 1._r8)
            rcap = uw_min(uw_max(rcap, R8(0x1.999999999999ap-4)), R8(0x1.0000000000000p+1)); // F:2672 rcap = min( max(rcap,rcapmin), rcapmax )
            tke(kb) = (ebrk(ncv) * rcap); // F:2673 tke(i,kb) = ebrk(i,ncv) * rcap
            tke(kb) = uw_min(tke(kb), R8(0x1.4000000000000p+4)); // F:2674 tke(i,kb) = min( tke(i,kb),tkemax )
        }
        sm_aw(kb) = (smcl(ncv) / R8(0x1.1cc63f141205cp-1)); // F:2682 sm_aw(i,kb) = smcl(i,ncv)/alph1
        wcap(kt) = (((bprod(kt) + sprod(kt)) * leng(kt)) / uw_sqrt(uw_max(tke(kt), R8(0x1.0c6f7a0b5ed8dp-20)))); // F:2691 wcap(i,kt) = (bprod(i,kt)+sprod(i,kt))*leng(i,kt)/sqrt(max(tke(i,kt),1.e-6_r8))
        if ((kb < (pver + 1))) { // F:2692 if( kb .lt. pver + 1 ) then
            wcap(kb) = (((bprod(kb) + sprod(kb)) * leng(kb)) / uw_sqrt(uw_max(tke(kb), R8(0x1.0c6f7a0b5ed8dp-20)))); // F:2693 wcap(i,kb) = (bprod(i,kb)+sprod(i,kb))*leng(i,kb)/sqrt(max(tke(i,kb),1.e-6_r8))
        }
        ktblw = kt; // F:2700 ktblw = kt
        wet_CL(ncv) = wet; // F:2704 wet_CL(i,ncv)        = wet
        web_CL(ncv) = web; // F:2705 web_CL(i,ncv)        = web
        jtbu_CL(ncv) = jtbu; // F:2706 jtbu_CL(i,ncv)       = jtbu
        jbbu_CL(ncv) = jbbu; // F:2707 jbbu_CL(i,ncv)       = jbbu
        evhc_CL(ncv) = evhc; // F:2708 evhc_CL(i,ncv)       = evhc
        jt2slv_CL(ncv) = jt2slv; // F:2709 jt2slv_CL(i,ncv)     = jt2slv
        n2ht_CL(ncv) = n2ht; // F:2710 n2ht_CL(i,ncv)       = n2ht
        n2hb_CL(ncv) = n2hb; // F:2711 n2hb_CL(i,ncv)       = n2hb
        lwp_CL(ncv) = lwp; // F:2712 lwp_CL(i,ncv)        = lwp
        opt_depth_CL(ncv) = opt_depth; // F:2713 opt_depth_CL(i,ncv)  = opt_depth
        radinvfrac_CL(ncv) = radinvfrac; // F:2714 radinvfrac_CL(i,ncv) = radinvfrac
        radf_CL(ncv) = radf; // F:2715 radf_CL(i,ncv)       = radf
        wstar_CL(ncv) = wstar; // F:2716 wstar_CL(i,ncv)      = wstar
        wstar3fact_CL(ncv) = wstar3fact; // F:2717 wstar3fact_CL(i,ncv) = wstar3fact
    }
    if ((ncvsurf > 0)) { // F:2728 if( ncvsurf .gt. 0 ) then
        ktopbl = ktop(ncvsurf); // F:2730 ktopbl(i) = ktop(i,ncvsurf)
        pblh = zi(ktopbl); // F:2731 pblh(i)   = zi(i, ktopbl(i))
        pblhp = pi(ktopbl); // F:2732 pblhp(i)  = pi(i, ktopbl(i))
        wpert = uw_max((R8(0x1.0000000000000p+0) * uw_sqrt(ebrk(ncvsurf))), R8(0x1.0c6f7a0b5ed8dp-20)); // F:2733 wpert(i)  = max(wfac*sqrt(ebrk(i,ncvsurf)),wpertmin)
        tpert = uw_max(((uw_abs(((shflx * rrho) / uw_cpair)) * R8(0x1.0000000000000p+0)) / wpert), R8(0x0.0p+0)); // F:2734 tpert(i)  = max(abs(shflx(i)*rrho(i)/cpair)*tfac/wpert(i),0._r8)
        qpert = uw_max(((uw_abs((qflx * rrho)) * R8(0x1.0000000000000p+0)) / wpert), R8(0x0.0p+0)); // F:2735 qpert(i)  = max(abs(qflx(i)*rrho(i))*tfac/wpert(i),0._r8)
        if ((bflxs > R8(0x0.0p+0))) { // F:2737 if( bflxs(i) .gt. 0._r8 ) then
            turbtype((pver + 1)) = 2; // F:2738 turbtype(i,pver+1) = 2
        } else {
            turbtype((pver + 1)) = 3; // F:2740 turbtype(i,pver+1) = 3
        }
        ipbl = R8(0x1.0000000000000p+0); // F:2743 ipbl(i)  = 1._r8
        kpblh = (ktopbl - R8(0x1.0000000000000p+0)); // F:2744 kpblh(i) = ktopbl(i) - 1._r8
    }
    belongst(1) = false; // F:2755 belongst(i,1) = .false.
    for (k = 2; k <= pver; k += 1) { // F:2756 do k = 2, pver
        belongst(k) = ((ri(k) < R8(0x1.851eb851eb852p-3)) && (!belongcv(k))); // F:2757 belongst(i,k) = ( ri(i,k) .lt. ricrit ) .and. ( .not. belongcv(i,k) )
        if ((belongst(k) && (!belongst((k - 1))))) { // F:2758 if( belongst(i,k) .and. ( .not. belongst(i,k-1) ) ) then
            kt = k; // F:2759 kt = k
        } else if (((!belongst(k)) && belongst((k - 1)))) { // F:2760 elseif( .not. belongst(i,k) .and. belongst(i,k-1) ) then
            kb = (k - 1); // F:2761 kb = k - 1
            lbulk = (z((kt - 1)) - z(kb)); // F:2762 lbulk = z(i,kt-1) - z(i,kb)
            for (ks = kt; ks <= kb; ks += 1) { // F:2763 do ks = kt, kb
                tunlramp = R8(0x1.5c28f5c28f5c3p-4); // F:2765 tunlramp = tunl
                leng(ks) = uw_pow((uw_pow((uw_vk * zi(ks)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:2773 leng(i,ks) = ( (vk*zi(i,ks))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            }
        }
    }
    belongst((pver + 1)) = (!belongcv((pver + 1))); // F:2788 belongst(i,pver+1) = .not. belongcv(i,pver+1)
    if (belongst((pver + 1))) { // F:2790 if( belongst(i,pver+1) ) then
        turbtype((pver + 1)) = 1; // F:2792 turbtype(i,pver+1) = 1
        if (belongst(pver)) { // F:2794 if( belongst(i,pver) ) then
            lbulk = z((kt - 1)); // F:2796 lbulk = z(i,kt-1)
        } else {
            kt = (pver + 1); // F:2798 kt = pver+1
            lbulk = z((kt - 1)); // F:2799 lbulk = z(i,kt-1)
        }
        ktopbl = (kt - 1); // F:2807 ktopbl(i) = kt - 1
        pblh = z(ktopbl); // F:2808 pblh(i)   = z(i,ktopbl(i))
        pblhp = (R8(0x1.0000000000000p-1) * (pi(ktopbl) + pi((ktopbl + 1)))); // F:2809 pblhp(i)  = 0.5_r8 * ( pi(i,ktopbl(i)) + pi(i,ktopbl(i)+1) )
        for (ks = kt; ks <= pver; ks += 1) { // F:2814 do ks = kt, pver
            tunlramp = R8(0x1.5c28f5c28f5c3p-4); // F:2816 tunlramp = tunl
            leng(ks) = uw_pow((uw_pow((uw_vk * zi(ks)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:2824 leng(i,ks) = ( (vk*zi(i,ks))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
        }
        wpert = R8(0x0.0p+0); // F:2834 wpert(i) = 0._r8
        tpert = uw_max(((((shflx * rrho) / uw_cpair) * R8(0x1.1000000000000p+3)) / ustar), R8(0x0.0p+0)); // F:2835 tpert(i) = max(shflx(i)*rrho(i)/cpair*fak/ustar(i),0._r8)
        qpert = uw_max((((qflx * rrho) * R8(0x1.1000000000000p+3)) / ustar), R8(0x0.0p+0)); // F:2836 qpert(i) = max(qflx(i)*rrho(i)*fak/ustar(i),0._r8)
        ipbl = R8(0x0.0p+0); // F:2838 ipbl(i)  = 0._r8
        kpblh = ktopbl; // F:2839 kpblh(i) = ktopbl(i)
    }
    for (k = 2; k <= pver; k += 1) { // F:2850 do k = 2, pver
        if (belongst(k)) { // F:2852 if( belongst(i,k) ) then
            turbtype(k) = 1; // F:2854 turbtype(i,k) = 1
            trma = (((R8(-0x1.15694467381d8p+5) * alph4exs) * ri(k)) + (R8(0x1.7333333333333p+3) * (R8(-0x1.174bc6a7ef9dbp+2) - ((alph4exs * R8(0x1.65aee631f8a09p-1)) * ri(k))))); // F:2855 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
            trmb = (((R8(-0x1.15694467381d8p+5) + alph4exs) * ri(k)) + (R8(0x1.7333333333333p+3) * ((-(R8(0x1.65aee631f8a09p-1) * ri(k))) + R8(0x1.1cc63f141205cp-1)))); // F:2856 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri(k); // F:2857 trmc = ri(i,k)
            det = uw_max(((trmb * trmb) - ((R8(0x1.0000000000000p+2) * trma) * trmc)), R8(0x0.0p+0)); // F:2858 det = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            if ((det < R8(0x0.0p+0))) { // F:2860 if( det .lt. 0._r8 ) then
                *ws.err = 2865; return; // F:2865 stop
            }
            gh = (((-trmb) + uw_sqrt(det)) / (R8(0x1.0000000000000p+1) * trma)); // F:2867 gh = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_min(uw_max(gh, ghmin), R8(0x1.7dbf487fcb924p-6)); // F:2870 gh = min(max(gh,ghmin),0.0233_r8)
            sh = uw_max(R8(0x0.0p+0), (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh)))); // F:2871 sh = max(0._r8,alph5/(1._r8+alph3*gh))
            sm = uw_max(R8(0x0.0p+0), (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (alph4exs * gh)))); // F:2872 sm = max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
            tke(k) = ((R8(0x1.7333333333333p+2) * uw_sq(leng(k))) * ((-(sh * n2(k))) + (sm * s2(k)))); // F:2874 tke(i,k)   = b1*(leng(i,k)**2)*(-sh*n2(i,k)+sm*s2(i,k))
            tke(k) = uw_min(tke(k), R8(0x1.4000000000000p+4)); // F:2875 tke(i,k)   = min(tke(i,k),tkemax)
            wcap(k) = (tke(k) / R8(0x1.7333333333333p+2)); // F:2876 wcap(i,k)  = tke(i,k)/b1
            kvh(k) = ((leng(k) * uw_sqrt(tke(k))) * sh); // F:2877 kvh(i,k)   = leng(i,k) * sqrt(tke(i,k)) * sh
            kvm(k) = ((leng(k) * uw_sqrt(tke(k))) * sm); // F:2878 kvm(i,k)   = leng(i,k) * sqrt(tke(i,k)) * sm
            bprod(k) = (-(kvh(k) * n2(k))); // F:2879 bprod(i,k) = -kvh(i,k) * n2(i,k)
            sprod(k) = (kvm(k) * s2(k)); // F:2880 sprod(i,k) =  kvm(i,k) * s2(i,k)
            sm_aw(k) = (sm / R8(0x1.1cc63f141205cp-1)); // F:2882 sm_aw(i,k) = sm/alph1
        }
    }
    for (k = 2; k <= pver; k += 1) { // F:2904 do k = 2, pver
        if ((((turbtype(k) == 3) || (turbtype(k) == 4)) || (turbtype(k) == 5))) { // F:2906 if( ( turbtype(i,k) .eq. 3 ) .or. ( turbtype(i,k) .eq. 4 ) .or. ( turbtype(i,k) .eq. 5 ) ) then
            trma = (((R8(-0x1.15694467381d8p+5) * alph4exs) * ri(k)) + (R8(0x1.7333333333333p+3) * (R8(-0x1.174bc6a7ef9dbp+2) - ((alph4exs * R8(0x1.65aee631f8a09p-1)) * ri(k))))); // F:2909 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
            trmb = (((R8(-0x1.15694467381d8p+5) + alph4exs) * ri(k)) + (R8(0x1.7333333333333p+3) * ((-(R8(0x1.65aee631f8a09p-1) * ri(k))) + R8(0x1.1cc63f141205cp-1)))); // F:2910 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri(k); // F:2911 trmc = ri(i,k)
            det = uw_max(((trmb * trmb) - ((R8(0x1.0000000000000p+2) * trma) * trmc)), R8(0x0.0p+0)); // F:2912 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = (((-trmb) + uw_sqrt(det)) / (R8(0x1.0000000000000p+1) * trma)); // F:2913 gh   = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_min(uw_max(gh, ghmin), R8(0x1.7dbf487fcb924p-6)); // F:2916 gh   = min(max(gh,ghmin),0.0233_r8)
            sh = uw_max(R8(0x0.0p+0), (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh)))); // F:2917 sh   = max(0._r8,alph5/(1._r8+alph3*gh))
            sm = uw_max(R8(0x0.0p+0), (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (alph4exs * gh)))); // F:2918 sm   = max(0._r8,(alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
            lbulk = (z((k - 1)) - z(k)); // F:2920 lbulk = z(i,k-1) - z(i,k)
            tunlramp = R8(0x1.5c28f5c28f5c3p-4); // F:2923 tunlramp = tunl
            leng_imsi = uw_pow((uw_pow((uw_vk * zi(k)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:2931 leng_imsi = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            tke_imsi = ((R8(0x1.7333333333333p+2) * uw_sq(leng_imsi)) * ((-(sh * n2(k))) + (sm * s2(k)))); // F:2937 tke_imsi = b1*(leng_imsi**2)*(-sh*n2(i,k)+sm*s2(i,k))
            tke_imsi = uw_min(uw_max(tke_imsi, R8(0x0.0p+0)), R8(0x1.4000000000000p+4)); // F:2938 tke_imsi = min(max(tke_imsi,0._r8),tkemax)
            kvh_imsi = ((leng_imsi * uw_sqrt(tke_imsi)) * sh); // F:2939 kvh_imsi = leng_imsi * sqrt(tke_imsi) * sh
            kvm_imsi = ((leng_imsi * uw_sqrt(tke_imsi)) * sm); // F:2940 kvm_imsi = leng_imsi * sqrt(tke_imsi) * sm
            if ((kvh(k) < kvh_imsi)) { // F:2942 if( kvh(i,k) .lt. kvh_imsi ) then
                kvh(k) = kvh_imsi; // F:2943 kvh(i,k)   =  kvh_imsi
                kvm(k) = kvm_imsi; // F:2944 kvm(i,k)   =  kvm_imsi
                leng(k) = leng_imsi; // F:2945 leng(i,k)  = leng_imsi
                tke(k) = tke_imsi; // F:2946 tke(i,k)   =  tke_imsi
                wcap(k) = (tke_imsi / R8(0x1.7333333333333p+2)); // F:2947 wcap(i,k)  =  tke_imsi / b1
                bprod(k) = (-(kvh_imsi * n2(k))); // F:2948 bprod(i,k) = -kvh_imsi * n2(i,k)
                sprod(k) = (kvm_imsi * s2(k)); // F:2949 sprod(i,k) =  kvm_imsi * s2(i,k)
                sm_aw(k) = (sm / R8(0x1.1cc63f141205cp-1)); // F:2950 sm_aw(i,k) =  sm/alph1
                turbtype(k) = 1; // F:2951 turbtype(i,k) = 1
            }
        }
    }
    bprod((pver + 1)) = bflxs; // F:2987 bprod(i,pver+1) = bflxs(i)
    gg = ((((R8(0x1.0000000000000p-1) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, R8(0x1.8000000000000p+0))); // F:2989 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
    if ((uw_abs((R8(0x1.65aee631f8a09p-1) - (gg * R8(-0x1.15694467381d8p+5)))) <= R8(0x1.ad7f29abcaf48p-24))) { // F:2990 if( abs(alph5-gg*alph3) .le. 1.e-7_r8 ) then
        if ((bprod((pver + 1)) > R8(0x0.0p+0))) { // F:2992 if( bprod(i,pver+1) .gt. 0._r8 ) then
            gh = R8(-0x1.c4467381d7dbfp+1); // F:2993 gh = -3.5334_r8
        } else {
            gh = ghmin; // F:2995 gh = ghmin
        }
    } else {
        gh = (gg / (R8(0x1.65aee631f8a09p-1) - (gg * R8(-0x1.15694467381d8p+5)))); // F:2998 gh = gg/(alph5-gg*alph3)
    }
    if ((bprod((pver + 1)) > R8(0x0.0p+0))) { // F:3002 if( bprod(i,pver+1) .gt. 0._r8 ) then
        gh = uw_min(uw_max(gh, R8(-0x1.c4467381d7dbfp+1)), R8(0x1.7dbf487fcb924p-6)); // F:3003 gh = min(max(gh,-3.5334_r8),0.0233_r8)
    } else {
        gh = uw_min(uw_max(gh, ghmin), R8(0x1.7dbf487fcb924p-6)); // F:3005 gh = min(max(gh,ghmin),0.0233_r8)
    }
    gh_a((pver + 1)) = gh; // F:3008 gh_a(i,pver+1) = gh
    sh_a((pver + 1)) = uw_max(R8(0x0.0p+0), (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh)))); // F:3009 sh_a(i,pver+1) = max(0._r8,alph5/(1._r8+alph3*gh))
    if ((bprod((pver + 1)) > R8(0x0.0p+0))) { // F:3010 if( bprod(i,pver+1) .gt. 0._r8 ) then
        sm_a((pver + 1)) = uw_max(R8(0x0.0p+0), (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (R8(-0x1.88240b780346ep+2) * gh)))); // F:3011 sm_a(i,pver+1) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh))
    } else {
        sm_a((pver + 1)) = uw_max(R8(0x0.0p+0), (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (alph4exs * gh)))); // F:3013 sm_a(i,pver+1) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
    }
    sm_aw((pver + 1)) = (sm_a((pver + 1)) / R8(0x1.1cc63f141205cp-1)); // F:3015 sm_aw(i,pver+1) = sm_a(i,pver+1)/alph1
    ri_a((pver + 1)) = (-((sm_a((pver + 1)) / sh_a((pver + 1))) * (bprod((pver + 1)) / sprod((pver + 1))))); // F:3016 ri_a(i,pver+1)  = -(sm_a(i,pver+1)/sh_a(i,pver+1))*(bprod(i,pver+1)/sprod(i,pver+1))
    for (k = 1; k <= pver; k += 1) { // F:3018 do k = 1, pver
        if ((ri(k) < R8(0x0.0p+0))) { // F:3019 if( ri(i,k) .lt. 0._r8 ) then
            trma = ((R8(0x1.a8f03ff93f46ap+7) * ri(k)) + (R8(0x1.7333333333333p+3) * (R8(-0x1.174bc6a7ef9dbp+2) - (R8(-0x1.11f3168d8b188p+2) * ri(k))))); // F:3020 trma = alph3*alph4*ri(i,k) + 2._r8*b1*(alph2-alph4*alph5*ri(i,k))
            trmb = ((R8(-0x1.466dc5d638866p+5) * ri(k)) + (R8(0x1.7333333333333p+3) * ((-(R8(0x1.65aee631f8a09p-1) * ri(k))) + R8(0x1.1cc63f141205cp-1)))); // F:3021 trmb = (alph3+alph4)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
            trmc = ri(k); // F:3022 trmc = ri(i,k)
            det = uw_max(((trmb * trmb) - ((R8(0x1.0000000000000p+2) * trma) * trmc)), R8(0x0.0p+0)); // F:3023 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = (((-trmb) + uw_sqrt(det)) / (R8(0x1.0000000000000p+1) * trma)); // F:3024 gh   = (-trmb + sqrt(det))/(2._r8*trma)
            gh = uw_min(uw_max(gh, R8(-0x1.c4467381d7dbfp+1)), R8(0x1.7dbf487fcb924p-6)); // F:3025 gh   = min(max(gh,-3.5334_r8),0.0233_r8)
            gh_a(k) = gh; // F:3026 gh_a(i,k) = gh
            sh_a(k) = uw_max(R8(0x0.0p+0), (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh)))); // F:3027 sh_a(i,k) = max(0._r8,alph5/(1._r8+alph3*gh))
            sm_a(k) = uw_max(R8(0x0.0p+0), (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (R8(-0x1.88240b780346ep+2) * gh)))); // F:3028 sm_a(i,k) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh))
            ri_a(k) = ri(k); // F:3029 ri_a(i,k) = ri(i,k)
        } else {
            if ((ri(k) > R8(0x1.851eb851eb852p-3))) { // F:3031 if( ri(i,k) .gt. ricrit ) then
                gh_a(k) = ghmin; // F:3032 gh_a(i,k) = ghmin
                sh_a(k) = R8(0x0.0p+0); // F:3033 sh_a(i,k) = 0._r8
                sm_a(k) = R8(0x0.0p+0); // F:3034 sm_a(i,k) = 0._r8
                ri_a(k) = ri(k); // F:3035 ri_a(i,k) = ri(i,k)
            } else {
                trma = (((R8(-0x1.15694467381d8p+5) * alph4exs) * ri(k)) + (R8(0x1.7333333333333p+3) * (R8(-0x1.174bc6a7ef9dbp+2) - ((alph4exs * R8(0x1.65aee631f8a09p-1)) * ri(k))))); // F:3037 trma = alph3*alph4exs*ri(i,k) + 2._r8*b1*(alph2-alph4exs*alph5*ri(i,k))
                trmb = (((R8(-0x1.15694467381d8p+5) + alph4exs) * ri(k)) + (R8(0x1.7333333333333p+3) * ((-(R8(0x1.65aee631f8a09p-1) * ri(k))) + R8(0x1.1cc63f141205cp-1)))); // F:3038 trmb = (alph3+alph4exs)*ri(i,k) + 2._r8*b1*(-alph5*ri(i,k)+alph1)
                trmc = ri(k); // F:3039 trmc = ri(i,k)
                det = uw_max(((trmb * trmb) - ((R8(0x1.0000000000000p+2) * trma) * trmc)), R8(0x0.0p+0)); // F:3040 det  = max(trmb*trmb-4._r8*trma*trmc,0._r8)
                gh = (((-trmb) + uw_sqrt(det)) / (R8(0x1.0000000000000p+1) * trma)); // F:3041 gh   = (-trmb + sqrt(det))/(2._r8*trma)
                gh = uw_min(uw_max(gh, ghmin), R8(0x1.7dbf487fcb924p-6)); // F:3042 gh   = min(max(gh,ghmin),0.0233_r8)
                gh_a(k) = gh; // F:3043 gh_a(i,k) = gh
                sh_a(k) = uw_max(R8(0x0.0p+0), (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh)))); // F:3044 sh_a(i,k) = max(0._r8,alph5/(1._r8+alph3*gh))
                sm_a(k) = uw_max(R8(0x0.0p+0), (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (alph4exs * gh)))); // F:3045 sm_a(i,k) = max(0._r8,(alph1+alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4exs*gh))
                ri_a(k) = ri(k); // F:3046 ri_a(i,k) = ri(i,k)
            }
        }
    }
    for (k = 1; k <= (pver + 1); k += 1) { // F:3052 do k = 1, pver + 1
        turbtype_f(k) = uw_widen((float)(turbtype(k))); // F:3053 turbtype_f(i,k) = real(turbtype(i,k))
    }
    return; // F:3058 return
}
#endif
