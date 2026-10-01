// WRF 4.7.1 eddy_diff, configured sftype=l, choice_evhc=choice_radf=maxi.
// Concatenate after common, saturation, vdiff, and caleddy. No includes.
// The column driver is in uwpbl_driver.cuh, concatenated after this header.
#ifndef UWPBL_EDDY_CUH
#define UWPBL_EDDY_CUH

__device__ void uw_sfdiag(int pver, V qt, V ql, V sl, V pi, V pm, V zi,
    V cld, V sfi, V sfuh, V sflh, V slslope, V qtslope) {
    for (int k=1; k<=pver+1; ++k) sfi(k)=R8(0.0);
    for (int k=1; k<=pver; ++k) { sfuh(k)=R8(0.0); sflh(k)=R8(0.0); }
    // The maxi branch overwrites the preceding ql<qmin test unconditionally.
    for (int k=2; k<=pver; ++k) {
        sfuh(k)=cld(k); sflh(k)=cld(k);
        sfi(k)=R8(0.5)*(sflh(k-1)+uw_min(sfuh(k),sflh(k-1)));
    }
    sfi(pver+1)=sflh(pver);
}

__device__ void uw_trbintd(int pver, V z, V u, V v, V t, V pmid,
    R8 taux, R8 tauy, R8& ustar, R8& rrho, V s2, V n2, V ri, V zi, V pi,
    V cld, V qt, V qv, V ql, V qi, V sfi, V sfuh, V sflh, V sl, V slv,
    V slslope, V qtslope, V chs, V chu, V cms, V cmu, R8& minpblh) {
    R8 latsub=R8(UW_LATVAP)+R8(UW_LATICE);
    rrho=R8(UW_RAIR)*t(pver)/pmid(pver);
    ustar=uw_max(uw_sqrt(uw_sqrt(uw_sq(taux)+uw_sq(tauy))*rrho),R8(0.01));
    minpblh=R8(100.0)*ustar;
    for (int k=1; k<=pver; ++k) {
        R8 es,qs,gam;
        int status=uw_fqsatd(t(k),pmid(k),es,qs,gam);
        qt(k)=qv(k)+ql(k)+qi(k);
        sl(k)=R8(UW_CPAIR)*t(k)+R8(UW_GRAVIT)*z(k)-R8(UW_LATVAP)*ql(k)-latsub*qi(k);
        slv(k)=sl(k)*(R8(1.0)+R8(UW_ZVIR)*qt(k));
        R8 bfact=R8(UW_GRAVIT)/(t(k)*(R8(1.0)+R8(UW_ZVIR)*qv(k)-ql(k)-qi(k)));
        chu(k)=(R8(1.0)+R8(UW_ZVIR)*qt(k))*bfact/R8(UW_CPAIR);
        chs(k)=((R8(1.0)+(R8(1.0)+R8(UW_ZVIR))*gam*R8(UW_CPAIR)*t(k)/R8(UW_LATVAP))/(R8(1.0)+gam))*bfact/R8(UW_CPAIR);
        cmu(k)=R8(UW_ZVIR)*bfact*t(k);
        cms(k)=R8(UW_LATVAP)*chs(k)-bfact*t(k);
    }
    chu(pver+1)=chu(pver); chs(pver+1)=chs(pver);
    cmu(pver+1)=cmu(pver); cms(pver+1)=cms(pver);
    slslope(pver)=(sl(pver)-sl(pver-1))/(pmid(pver)-pmid(pver-1));
    qtslope(pver)=(qt(pver)-qt(pver-1))/(pmid(pver)-pmid(pver-1));
    slslope(1)=(sl(2)-sl(1))/(pmid(2)-pmid(1));
    qtslope(1)=(qt(2)-qt(1))/(pmid(2)-pmid(1));
    R8 dsldp_b=slslope(1),dqtdp_b=qtslope(1);
    for (int k=2; k<=pver-1; ++k) {
        R8 dsldp_a=dsldp_b,dqtdp_a=dqtdp_b;
        dsldp_b=(sl(k+1)-sl(k))/(pmid(k+1)-pmid(k));
        dqtdp_b=(qt(k+1)-qt(k))/(pmid(k+1)-pmid(k));
        R8 product=dsldp_a*dsldp_b;
        if (product<=R8(0.0)) slslope(k)=R8(0.0);
        else if (product>R8(0.0) && dsldp_a<R8(0.0)) slslope(k)=uw_max(dsldp_a,dsldp_b);
        else if (product>R8(0.0) && dsldp_a>R8(0.0)) slslope(k)=uw_min(dsldp_a,dsldp_b);
        product=dqtdp_a*dqtdp_b;
        if (product<=R8(0.0)) qtslope(k)=R8(0.0);
        else if (product>R8(0.0) && dqtdp_a<R8(0.0)) qtslope(k)=uw_max(dqtdp_a,dqtdp_b);
        else if (product>R8(0.0) && dqtdp_a>R8(0.0)) qtslope(k)=uw_min(dqtdp_a,dqtdp_b);
    }
    uw_sfdiag(pver,qt,ql,sl,pi,pmid,zi,cld,sfi,sfuh,sflh,slslope,qtslope);
    for (int k=pver; k>=2; --k) {
        int km1=k-1;
        R8 rdz=R8(1.0)/(z(km1)-z(k));
        R8 dsldz=(sl(km1)-sl(k))*rdz,dqtdz=(qt(km1)-qt(k))*rdz;
        chu(k)=(chu(km1)+chu(k))*R8(0.5);
        chs(k)=(chs(km1)+chs(k))*R8(0.5);
        cmu(k)=(cmu(km1)+cmu(k))*R8(0.5);
        cms(k)=(cms(km1)+cms(k))*R8(0.5);
        R8 ch=chu(k)*(R8(1.0)-sfi(k))+chs(k)*sfi(k);
        R8 cm=cmu(k)*(R8(1.0)-sfi(k))+cms(k)*sfi(k);
        n2(k)=ch*dsldz+cm*dqtdz;
        s2(k)=(uw_sq(u(km1)-u(k))+uw_sq(v(km1)-v(k)))*uw_sq(rdz);
        s2(k)=uw_max(R8(1.e-12),s2(k));
        ri(k)=n2(k)/s2(k);
    }
    n2(1)=n2(2); s2(1)=s2(2); ri(1)=ri(2);
}

__device__ void uw_eddy_retrieve(int pver, V slfd, V qtfd, V qi, V z,
    V pmid, V tfd, V qvfd, V qlfd) {
    R8 latsub=R8(UW_LATVAP)+R8(UW_LATICE);
    for (int k=1; k<=pver; ++k) {
        R8 es,qs,gam;
        R8 templ=(slfd(k)-R8(UW_GRAVIT)*z(k))/R8(UW_CPAIR);
        int status=uw_fqsatd(templ,pmid(k),es,qs,gam);
        R8 temps=templ+(qtfd(k)-qs)/(R8(UW_CPAIR)/R8(UW_LATVAP)+R8(UW_LATVAP)*qs/(R8(UW_RAIR)*uw_sq(templ)));
        status=uw_fqsatd(temps,pmid(k),es,qs,gam);
        qlfd(k)=uw_max(qtfd(k)-qi(k)-qs,R8(0.0));
        qvfd(k)=uw_max(R8(0.0),qtfd(k)-qi(k)-qlfd(k));
        tfd(k)=(slfd(k)+R8(UW_LATVAP)*qlfd(k)+latsub*qi(k)-R8(UW_GRAVIT)*z(k))/R8(UW_CPAIR);
    }
}

__device__ void uw_eddy_relax(int pver, V kvm_out, V kvh_out, V kvm, V kvh) {
    for (int k=1; k<=pver+1; ++k) {
        // 1-lambda is a constant subexpression folded to exactly 0.5.
        kvm_out(k)=R8(0.5)*kvm_out(k)+R8(0.5)*kvm(k);
        kvh_out(k)=R8(0.5)*kvh_out(k)+R8(0.5)*kvh(k);
    }
}
__device__ R8 uw_eddy_error(int pver, V kvh, V kvh_out) {
    R8 error=R8(0.0);
    for (int k=1; k<=pver; ++k) error=error+uw_sq(kvh(k)-kvh_out(k));
    return uw_sqrt(error/uw_real(pver));
}

__device__ void uw_compute_eddy_diff(int pver,
    V t, V qv, R8 ztodt, V ql, V qi, V s, V rpdel, V cldn, V qrl, V wsedl,
    V z, V zi, V pmid, V pi, V u, V v, R8 taux, R8 tauy, R8 shflx, R8 qflx,
    bool wstarent, int nturb, R8& ustar, R8& pblh, V kvm_in, V kvh_in,
    V kvm_out, V kvh_out, V kvq, V cgh, V cgs, R8& tpert, R8& qpert,
    R8& wpert, V tke, V bprod, V sprod, V sfi, bool kvinit,
    R8& tauresx, R8& tauresy, R8 ksrftms, R8& ipbl, R8& kpblh,
    R8& wstarPBL, V turbtype, V sm_aw, Ws& ws) {
    WsMark mark(ws);
    // ncvmax=pver. Diagnostic arrays have their declared Fortran lengths.
    V ufd=ws.r8(pver),vfd=ws.r8(pver),tfd=ws.r8(pver),qvfd=ws.r8(pver),qlfd=ws.r8(pver);
    V qt=ws.r8(pver),sl=ws.r8(pver),qtfd=ws.r8(pver),slfd=ws.r8(pver),slv=ws.r8(pver);
    V slslope=ws.r8(pver),qtslope=ws.r8(pver),s2=ws.r8(pver),n2=ws.r8(pver),ri=ws.r8(pver);
    V sfuh=ws.r8(pver),sflh=ws.r8(pver);
    V chs=ws.r8(pver+1),chu=ws.r8(pver+1),cms=ws.r8(pver+1),cmu=ws.r8(pver+1);
    V kvh=ws.r8(pver+1),kvm=ws.r8(pver+1),kvf=ws.r8(pver+1);
    V kbase_o=ws.r8(pver),ktop_o=ws.r8(pver),kbase_mg=ws.r8(pver),ktop_mg=ws.r8(pver);
    V kbase_f=ws.r8(pver),ktop_f=ws.r8(pver);
    V wet=ws.r8(pver),web=ws.r8(pver),jtbu=ws.r8(pver),jbbu=ws.r8(pver),evhc=ws.r8(pver);
    V jt2slv=ws.r8(pver),n2ht=ws.r8(pver),n2hb=ws.r8(pver),lwp=ws.r8(pver),opt_depth=ws.r8(pver);
    V radinvfrac=ws.r8(pver),radf=ws.r8(pver),wstar=ws.r8(pver),wstar3fact=ws.r8(pver);
    V ebrk=ws.r8(pver),wbrk=ws.r8(pver),lbrk=ws.r8(pver),ricl=ws.r8(pver);
    V ghcl=ws.r8(pver),shcl=ws.r8(pver),smcl=ws.r8(pver);
    V ghi=ws.r8(pver+1),shi=ws.r8(pver+1),smi=ws.r8(pver+1),rii=ws.r8(pver+1),lengi=ws.r8(pver+1),wcap=ws.r8(pver+1);
    V jnk2d=ws.r8(pver+1);
    R8 rrho,minpblh,tkes,pblhp,ncvfin_o,ncvfin_mg,ncvfin_f,jnk1d,errorPBL;
    R8 cflx[1]={qflx},qmincg[1]={R8(0.0)};
    UwVdiffFields fl{true,true,true,{true,false,false,false,false}};
    for (int k=1; k<=pver; ++k) {
        ufd(k)=u(k); vfd(k)=v(k); tfd(k)=t(k); qvfd(k)=qv(k); qlfd(k)=ql(k);
    }
    for (int iturb=1; iturb<=nturb; ++iturb) {
        R8 tautotx=taux-ksrftms*ufd(pver),tautoty=tauy-ksrftms*vfd(pver);
        uw_trbintd(pver,z,ufd,vfd,tfd,pmid,tautotx,tautoty,ustar,rrho,s2,n2,ri,zi,pi,cldn,qtfd,qvfd,qlfd,qi,sfi,sfuh,sflh,slfd,slv,slslope,qtslope,chs,chu,cms,cmu,minpblh);
        if (iturb==1) for (int k=1; k<=pver; ++k) { qt(k)=qtfd(k); sl(k)=slfd(k); }
        for (int k=1; k<=pver+1; ++k) {
            if (iturb==1) { kvh(k)=kvinit?R8(0.0):kvh_in(k); kvm(k)=kvinit?R8(0.0):kvm_in(k); }
            else { kvh(k)=kvh_out(k); kvm(k)=kvm_out(k); }
        }
        uw_caleddy(pver,slfd,qtfd,qlfd,slv,ufd,vfd,pi,z,zi,qflx,shflx,
            slslope,qtslope,chu,chs,cmu,cms,sfuh,sflh,n2,s2,ri,rrho,pblh,ustar,
            kvh,kvm,kvh_out,kvm_out,tpert,qpert,qrl,kvf,tke,wstarent,bprod,sprod,
            minpblh,wpert,tkes,turbtype,sm_aw,kbase_o,ktop_o,ncvfin_o,kbase_mg,
            ktop_mg,ncvfin_mg,kbase_f,ktop_f,ncvfin_f,wet,web,jtbu,jbbu,evhc,
            jt2slv,n2ht,n2hb,lwp,opt_depth,radinvfrac,radf,wstar,wstar3fact,
            ebrk,wbrk,lbrk,ricl,ghcl,shcl,smcl,ghi,shi,smi,rii,lengi,wcap,
            pblhp,cldn,ipbl,kpblh,wsedl,ws);
        if (iturb==nturb) errorPBL=uw_eddy_error(pver,kvh,kvh_out);
        if (iturb>1 && iturb<nturb) uw_eddy_relax(pver,kvm_out,kvh_out,kvm,kvh);
        for (int k=1; k<=pver+1; ++k) { cgh(k)=R8(0.0); cgs(k)=R8(0.0); }
        if (iturb<nturb) {
            for (int k=1; k<=pver; ++k) { slfd(k)=sl(k); qtfd(k)=qt(k); ufd(k)=u(k); vfd(k)=v(k); }
            V q[1]={qtfd}; int errflag;
            uw_compute_vdiff(pver,1,pmid,pi,rpdel,t,ztodt,taux,tauy,shflx,cflx,
                1,pver,kvh_out,kvm_out,kvh_out,cgs,cgh,zi,ksrftms,qmincg,fl,
                ufd,vfd,q,slfd,jnk1d,jnk1d,jnk2d,jnk1d,errflag,tauresx,tauresy,0,ws);
            uw_eddy_retrieve(pver,slfd,qtfd,qi,z,pmid,tfd,qvfd,qlfd);
        }
    }
    for (int k=1; k<=pver+1; ++k) kvq(k)=kvh_out(k);
    wstarPBL=ipbl==R8(1.0)?uw_max(R8(0.0),wstar(1)):R8(0.0);
}
#endif
