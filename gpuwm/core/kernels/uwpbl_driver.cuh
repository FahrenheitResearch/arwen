// Column ABI: UwF32View.p points to this column's bottom element; s is the
// element stride. Views are 1-based WRF order (bottom first), nk mass levels,
// nk+1 interfaces. The enclosing kernel supplies UwColumnIn and UwColumnOut.
// Surface inputs are float values. Surface outputs are pointers to this
// column's single float. kvm3d/kvh3d and tauresx2d/tauresy2d are inout.
// Ws pools must include all nested eddy/caleddy/vdiff allocations. Overflow
// is signalled by ws.err. The lead chooses capacity for its integrated lanes.
// Concatenate after uwpbl_eddy.cuh, before the enclosing __global__ kernel.
#ifndef UWPBL_DRIVER_CUH
#define UWPBL_DRIVER_CUH
struct UwF32View {
    float* p; int s;
    __device__ __forceinline__ float& operator()(int k) const {
        return p[(long long)(k-1)*s];
    }
};
struct UwColumnIn {
    UwF32View u,v,th,rho,qv,qc,qi,qnc,qni,p,z,t,cldfra,exner,rthratenlw,wsedl3d,p8w,z_at_w;
    float hfx,qfx,ust,ht,dt; int itimestep;
};
struct UwColumnOut {
    UwF32View kvm3d,kvh3d,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqniblten;
    UwF32View tke_pbl,turbtype3d,smaw3d;
    float *tauresx2d,*tauresy2d,*tpert2d,*qpert2d,*wpert2d,*pblh2d;
    int *kpbl2d; // INTEGER in WRF, module_bl_camuwpbl_driver.F:208
};
__device__ void uw_camuwpbl_column(int nk, const UwColumnIn& in,
    const UwColumnOut& out, Ws& ws) {
    WsMark mark(ws);
    bool first=in.itimestep==1;
    R8 ztodt=uw_widen(in.dt),rztodt=R8(1.0)/ztodt;
    R8 phis=uw_widen(in.ht)*R8(UW_GRAVIT);
    V u=ws.r8(nk),v=ws.r8(nk),pmid=ws.r8(nk),pdel=ws.r8(nk),rpdel=ws.r8(nk);
    V z=ws.r8(nk),t=ws.r8(nk),s=ws.r8(nk),qrl=ws.r8(nk),wsedl=ws.r8(nk),exner=ws.r8(nk),cldn=ws.r8(nk);
    V pdeldry=ws.r8(nk),rpdeldry=ws.r8(nk),pmiddry=ws.r8(nk),pintdry=ws.r8(nk+1);
    V cloud[5],cloudtnd[5];
    for (int m=0; m<5; ++m) { cloud[m]=ws.r8(nk); cloudtnd[m]=ws.r8(nk); }
    V pi=ws.r8(nk+1),zi=ws.r8(nk+1),kvq=ws.r8(nk+1),cgh=ws.r8(nk+1),cgs=ws.r8(nk+1);
    V kvh=ws.r8(nk+1),kvm=ws.r8(nk+1),kvh_in=ws.r8(nk+1),kvm_in=ws.r8(nk+1);
    for (int k=1; k<=nk; ++k) {
        int kflip=nk-k+1;
        for (int m=0; m<5; ++m) { cloud[m](kflip)=R8(-999888777.0); cloudtnd[m](kflip)=R8(-999888777.0); }
        u(kflip)=uw_widen(in.u(k)); v(kflip)=uw_widen(in.v(k));
        pmid(kflip)=uw_widen(in.p(k));
        R8 dp=uw_widen(__fsub_rn(in.p8w(k),in.p8w(k+1)));
        pdel(kflip)=dp; rpdel(kflip)=R8(1.0)/dp;
        z(kflip)=uw_widen(__fsub_rn(in.z(k),in.ht));
        t(kflip)=uw_widen(in.t(k));
        s(kflip)=R8(UW_CPAIR)*t(kflip)+R8(UW_GRAVIT)*z(kflip)+phis;
        qrl(kflip)=uw_widen(__fmul_rn(in.rthratenlw(k),in.exner(k)))*R8(UW_CPAIR)*dp;
        wsedl(kflip)=uw_widen(in.wsedl3d(k));
        R8 multFrc=R8(1.0)/(R8(1.0)+uw_widen(in.qv(k)));
        cloud[0](kflip)=uw_max(uw_widen(in.qv(k))*multFrc,R8(1.e-30));
        cloud[1](kflip)=uw_widen(in.qc(k))*multFrc;
        cloud[2](kflip)=uw_widen(in.qi(k))*multFrc;
        cloud[3](kflip)=uw_widen(in.qnc(k))*multFrc;
        cloud[4](kflip)=uw_widen(in.qni(k))*multFrc;
        exner(kflip)=uw_widen(in.exner(k)); cldn(kflip)=uw_widen(in.cldfra(k));
        pdeldry(kflip)=pdel(kflip)*(R8(1.0)-cloud[0](kflip));
        rpdeldry(kflip)=R8(1.0)/pdeldry(kflip);
    }
    for (int k=1; k<=nk+1; ++k) {
        int kflip=nk-k+2;
        pi(kflip)=uw_widen(in.p8w(k)); zi(kflip)=uw_widen(__fsub_rn(in.z_at_w(k),in.ht));
        kvq(kflip)=R8(0.0); cgh(kflip)=R8(0.0); cgs(kflip)=R8(0.0);
        if (first) { out.kvm3d(k)=uw_narrow(R8(0.0)); out.kvh3d(k)=uw_narrow(R8(0.0)); }
        kvh(kflip)=uw_widen(out.kvh3d(k)); kvm(kflip)=uw_widen(out.kvm3d(k));
    }
    pintdry(1)=pi(1);
    for (int k=1; k<=nk; ++k) {
        pintdry(k+1)=pintdry(k)+pdeldry(k);
        pmiddry(k)=(pintdry(k+1)+pintdry(k))*R8(0.5);
    }
    R8 shflx=uw_widen(in.hfx);
    R8 sgh=R8(0.0),landfrac=R8(0.0);
    R8 uMean=uw_widen(__fsqrt_rn(__fadd_rn(__fmul_rn(in.u(1),in.u(1)),__fmul_rn(in.v(1),in.v(1)))));
    R8 tauFac=uw_widen(__fmul_rn(__fmul_rn(in.rho(1),in.ust),in.ust))/uMean;
    R8 taux=-(tauFac*uw_widen(in.u(1))),tauy=-(tauFac*uw_widen(in.v(1)));
    if (first) { *out.tauresx2d=uw_narrow(R8(0.0)); *out.tauresy2d=uw_narrow(R8(0.0)); }
    R8 tauresx=uw_widen(*out.tauresx2d),tauresy=uw_widen(*out.tauresy2d);
    R8 ksrftms=R8(0.0),tautotx=taux,tautoty=tauy;
    R8 cflx[5]={R8(0.0),R8(0.0),R8(0.0),R8(0.0),R8(0.0)};
    cflx[0]=uw_widen(in.qfx);
    R8 ustar=R8(0.0),pblh=R8(0.0),ipbl=R8(0.0),kpblh=R8(0.0),wstarPBL=R8(0.0);
    R8 tpert,qpert,wpert;
    V tke=ws.r8(nk+1),bprod=ws.r8(nk+1),sprod=ws.r8(nk+1),sfi=ws.r8(nk+1),turbtype=ws.r8(nk+1),smaw=ws.r8(nk+1);
    for (int k=1; k<=nk+1; ++k) { kvm_in(k)=kvm(k); kvh_in(k)=kvh(k); }
    uw_compute_eddy_diff(nk,t,cloud[0],ztodt,cloud[1],cloud[2],s,rpdel,cldn,qrl,wsedl,
        z,zi,pmid,pi,u,v,taux,tauy,shflx,cflx[0],true,5,ustar,pblh,kvm_in,kvh_in,
        kvm,kvh,kvq,cgh,cgs,tpert,qpert,wpert,tke,bprod,sprod,sfi,first,tauresx,
        tauresy,ksrftms,ipbl,kpblh,wstarPBL,turbtype,smaw,ws);
    for (int k=1; k<=nk+1; ++k) {
        int kflip=nk-k+2;
        out.kvh3d(k)=uw_narrow(kvh(kflip)); out.kvm3d(k)=uw_narrow(kvm(kflip));
    }
    V stnd=ws.r8(nk),utnd=ws.r8(nk),vtnd=ws.r8(nk),sl_pre=ws.r8(nk),qt_pre=ws.r8(nk);
    V tem2=ws.r8(nk),ftem=ws.r8(nk),ftem_pre=ws.r8(nk),dtk=ws.r8(nk);
    for (int k=1; k<=nk; ++k) {
        for (int m=0; m<5; ++m) cloudtnd[m](k)=cloud[m](k);
        stnd(k)=s(k); utnd(k)=u(k); vtnd(k)=v(k);
        sl_pre(k)=stnd(k)-R8(UW_LATVAP)*cloudtnd[1](k)-(R8(UW_LATVAP)+R8(UW_LATICE))*cloudtnd[2](k);
        qt_pre(k)=cloudtnd[0](k)+cloudtnd[1](k)+cloudtnd[2](k);
    }
    uw_aqsat(t,pmid,tem2,ftem,1,nk);
    for (int k=1; k<=nk; ++k) ftem_pre(k)=cloud[0](k)/ftem(k)*R8(100.0);
    R8 qmincg[5]={R8(UW_QMIN_Q),R8(0.0),R8(0.0),R8(0.0),R8(0.0)};
    UwVdiffFields fl{true,true,true,{true,true,true,true,true}};
    R8 tautmsx,tautmsy,topflx; int errflag;
    uw_compute_vdiff(nk,5,pmid,pi,rpdel,t,ztodt,taux,tauy,shflx,cflx,1,nk,
        kvh,kvm,kvq,cgs,cgh,zi,ksrftms,qmincg,fl,utnd,vtnd,cloudtnd,stnd,
        tautmsx,tautmsy,dtk,topflx,errflag,tauresx,tauresy,1,ws);
    if (errflag) { *ws.err=2; return; } // enclosing kernel reports WRF fatal
    *out.tauresx2d=uw_narrow(tauresx); *out.tauresy2d=uw_narrow(tauresy);
    V slten=ws.r8(nk),qtten=ws.r8(nk),taft=ws.r8(nk);
    for (int k=1; k<=nk; ++k) {
        R8 sl=stnd(k)-R8(UW_LATVAP)*cloudtnd[1](k)-(R8(UW_LATVAP)+R8(UW_LATICE))*cloudtnd[2](k);
        R8 qt=cloudtnd[0](k)+cloudtnd[1](k)+cloudtnd[2](k);
        slten(k)=(sl-sl_pre(k))*rztodt; qtten(k)=(qt-qt_pre(k))*rztodt;
        stnd(k)=(stnd(k)-s(k))*rztodt; utnd(k)=(utnd(k)-u(k))*rztodt; vtnd(k)=(vtnd(k)-v(k))*rztodt;
        for (int m=0; m<5; ++m) cloudtnd[m](k)=(cloudtnd[m](k)-cloud[m](k))*rztodt;
        R8 qv_aft=cloud[0](k)+cloudtnd[0](k)*ztodt;
        R8 ql_aft=cloud[1](k)+cloudtnd[1](k)*ztodt;
        R8 qi_aft=cloud[2](k)+cloudtnd[2](k)*ztodt;
        R8 s_aft=s(k)+stnd(k)*ztodt;
        taft(k)=(s_aft-R8(UW_GRAVIT)*z(k))/R8(UW_CPAIR);
        R8 u_aft=u(k)+utnd(k)*ztodt,v_aft=v(k)+vtnd(k)*ztodt;
    }
    uw_aqsat(taft,pmid,tem2,ftem,1,nk);
    for (int k=1; k<=nk; ++k) {
        R8 ftem_aft=(cloud[0](k)+cloudtnd[0](k)*ztodt)/ftem(k)*R8(100.0);
        R8 tten=(taft(k)-t(k))*rztodt,rhten=(ftem_aft-ftem_pre(k))*rztodt;
    }
    for (int k=1; k<=nk; ++k) {
        int kflip=nk-k+1;
        out.rublten(k)=uw_narrow(utnd(kflip)); out.rvblten(k)=uw_narrow(vtnd(kflip));
        out.rthblten(k)=uw_narrow(stnd(kflip)/R8(UW_CPAIR)/exner(kflip));
        R8 multFrc=R8(1.0)+uw_widen(in.qv(k));
        out.rqvblten(k)=uw_narrow(cloudtnd[0](kflip)*multFrc*multFrc);
        out.rqcblten(k)=uw_narrow(cloudtnd[1](kflip)*multFrc);
        out.rqiblten(k)=uw_narrow(cloudtnd[2](kflip)*multFrc);
        out.rqniblten(k)=uw_narrow(cloudtnd[4](kflip)*multFrc);
    }
    for (int k=1; k<=nk+1; ++k) {
        int kflip=nk-k+2;
        out.tke_pbl(k)=uw_narrow(tke(kflip)); out.turbtype3d(k)=uw_narrow(turbtype(kflip)); out.smaw3d(k)=uw_narrow(smaw(kflip));
    }
    *out.kpbl2d=nk-uw_int(kpblh)+1;
    *out.pblh2d=uw_narrow(pblh); *out.tpert2d=uw_narrow(tpert);
    *out.qpert2d=uw_narrow(qpert); *out.wpert2d=uw_narrow(wpert);
}
#endif
