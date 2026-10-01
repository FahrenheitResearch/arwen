// WRF diffusion_solver: non-molecular WRF_PORT path. Views are 1-based.
#ifndef UW_VDIFF_CUH
#define UW_VDIFF_CUH
struct UwVdiffFields { bool u,v,s; bool q[5]; };
__device__ void uw_vd_lu_decomp(int pver,R8 ksrf,V kv,V tmpi,V rpdel,R8 ztodt,R8 cc_top,V ca,V cc,V dnom,V ze,int ntop,int nbot) {
 for(int k=nbot-1;k>=ntop;--k) {ca(k)=kv(k+1)*tmpi(k+1)*rpdel(k);cc(k+1)=kv(k+1)*tmpi(k+1)*rpdel(k+1);}
 ca(nbot)=R8(0);
 dnom(nbot)=R8(1)/(R8(1)+cc(nbot)+ksrf*ztodt*R8(UW_GRAVIT)*rpdel(nbot));
 ze(nbot)=cc(nbot)*dnom(nbot);
 for(int k=nbot-1;k>=ntop+1;--k) {dnom(k)=R8(1)/(R8(1)+ca(k)+cc(k)-ca(k)*ze(k+1));ze(k)=cc(k)*dnom(k);}
 dnom(ntop)=R8(1)/(R8(1)+ca(ntop)+cc_top-ca(ntop)*ze(ntop+1));
}
__device__ void uw_vd_lu_solve(int pver,V q,V ca,V ze,V dnom,int ntop,int nbot,R8 cd_top) {
 // zf is stored in q: each original value is consumed before being replaced.
 // This preserves each arithmetic operation of the automatic zf array.
 q(nbot)=q(nbot)*dnom(nbot);
 for(int k=nbot-1;k>=ntop+1;--k)q(k)=(q(k)+ca(k)*q(k+1))*dnom(k);
 q(ntop)=(q(ntop)+cd_top+ca(ntop)*q(ntop+1))*dnom(ntop);
 for(int k=ntop+1;k<=nbot;++k)q(k)=q(k)+ze(k)*q(k-1);
}
__device__ void uw_compute_vdiff(int pver,int ncnst,V pmid,V pint,V rpdel,V t,R8 ztodt,R8 taux,R8 tauy,R8 shflx,const R8* cflx,int ntop,int nbot,V kvh,V kvm,V kvq,V cgs,V cgh,V zi,R8 ksrftms,const R8* qmincg,const UwVdiffFields& fl,V u,V v,V* q,V dse,R8& tautmsx,R8& tautmsy,V dtk,R8& topflx,int& errflag,R8& tauresx,R8& tauresy,int itaures,Ws& ws) {
 errflag=0;bool momentum=fl.u||fl.v;
 if(momentum&&!fl.s) {errflag=1;return;}
 WsMark mark(ws);
 V rhoi=ws.r8(pver+1),tmpi2=ws.r8(pver+1),tmpi1=ws.r8(pver+1),du=ws.r8(pver+1),dv=ws.r8(pver+1);
 V ca=ws.r8(pver),cc=ws.r8(pver),dnom=ws.r8(pver),ze=ws.r8(pver),qtm=ws.r8(pver);
 rhoi(1)=pint(1)/(R8(UW_RAIR)*t(1));
 for(int k=2;k<=pver;++k) {R8 tint=R8(0.5)*(t(k)+t(k-1));rhoi(k)=pint(k)/(R8(UW_RAIR)*tint);tmpi2(k)=ztodt*uw_sq(R8(UW_GRAVIT)*rhoi(k))/(pmid(k)-pmid(k-1));}
 rhoi(pver+1)=pint(pver+1)/(R8(UW_RAIR)*t(pver));
 R8 rrho=R8(UW_RAIR)*t(pver)/pmid(pver),tmp1=ztodt*R8(UW_GRAVIT)*rpdel(pver);
 if(momentum) {
  du(1)=R8(0);dv(1)=R8(0);du(pver+1)=-u(pver);dv(pver+1)=-v(pver);
  for(int k=2;k<=pver;++k){du(k)=u(k)-u(k-1);dv(k)=v(k)-v(k-1);}
  R8 speed=uw_max(uw_sqrt(uw_pow(u(pver),R8(2))+uw_pow(v(pver),R8(2))),R8(1));
  R8 tau=uw_sqrt(uw_pow(taux,R8(2))+uw_pow(tauy,R8(2)));
  R8 ksrf=uw_max(tau/speed,R8(1.e-4))+ksrftms;
  R8 usum=R8(0),vsum=R8(0);
  for(int k=1;k<=pver;++k){usum=usum+(R8(1)/R8(UW_GRAVIT))*u(k)/rpdel(k);vsum=vsum+(R8(1)/R8(UW_GRAVIT))*v(k)/rpdel(k);}
  R8 ramda=ztodt/R8(7200);
  u(pver)=u(pver)+tmp1*tauresx*ramda;v(pver)=v(pver)+tmp1*tauresy*ramda;
  uw_vd_lu_decomp(pver,ksrf,kvm,tmpi2,rpdel,ztodt,R8(0),ca,cc,dnom,ze,ntop,nbot);
  uw_vd_lu_solve(pver,u,ca,ze,dnom,ntop,nbot,R8(0));uw_vd_lu_solve(pver,v,ca,ze,dnom,ntop,nbot,R8(0));
  tautmsx=-(ksrftms*u(pver));tautmsy=-(ksrftms*v(pver));
  R8 usout=R8(0),vsout=R8(0);
  for(int k=1;k<=pver;++k){usout=usout+(R8(1)/R8(UW_GRAVIT))*u(k)/rpdel(k);vsout=vsout+(R8(1)/R8(UW_GRAVIT))*v(k)/rpdel(k);}
  R8 tx=(usout-usum)/ztodt,ty=(vsout-vsum)/ztodt;
  if(itaures==1){tauresx=taux+tautmsx+tauresx-tx;tauresy=tauy+tautmsy+tauresy-ty;}
  tmpi1(1)=R8(0);
  tmpi1(pver+1)=R8(0.5)*ztodt*R8(UW_GRAVIT)*((-u(pver)+du(pver+1))*tx+(-v(pver)+dv(pver+1))*ty);
  for(int k=2;k<=pver;++k){R8 dout_u=u(k)-u(k-1),dout_v=v(k)-v(k-1);tmpi1(k)=R8(0.25)*tmpi2(k)*kvm(k)*(uw_sq(dout_u)+uw_sq(dout_v)+dout_u*du(k)+dout_v*dv(k));}
  for(int k=1;k<=pver;++k){dtk(k)=(tmpi1(k+1)+tmpi1(k))*rpdel(k);dse(k)=dse(k)+dtk(k);}
 }
 if(fl.s){
  for(int k=1;k<=pver;++k)dse(k)=dse(k)+ztodt*rpdel(k)*R8(UW_GRAVIT)*(rhoi(k+1)*kvh(k+1)*cgh(k+1)-rhoi(k)*kvh(k)*cgh(k));
  dse(pver)=dse(pver)+tmp1*shflx;
  uw_vd_lu_decomp(pver,R8(0),kvh,tmpi2,rpdel,ztodt,R8(0),ca,cc,dnom,ze,ntop,nbot);
  uw_vd_lu_solve(pver,dse,ca,ze,dnom,ntop,nbot,R8(0));
 }
 bool need_decomp=true;
 for(int m=1;m<=ncnst;++m)if(fl.q[m-1]){
  for(int k=1;k<=pver;++k)qtm(k)=q[m-1](k);
  for(int k=1;k<=pver;++k)q[m-1](k)=q[m-1](k)+ztodt*rpdel(k)*R8(UW_GRAVIT)*(cflx[m-1]*rrho)*(rhoi(k+1)*kvh(k+1)*cgs(k+1)-rhoi(k)*kvh(k)*cgs(k));
  bool lqtst=true;for(int k=1;k<=pver;++k)if(!(q[m-1](k)>=qmincg[m-1]))lqtst=false;
  for(int k=1;k<=pver;++k)if(!lqtst)q[m-1](k)=qtm(k);
  q[m-1](pver)=q[m-1](pver)+tmp1*cflx[m-1];
  if(need_decomp){uw_vd_lu_decomp(pver,R8(0),kvq,tmpi2,rpdel,ztodt,R8(0),ca,cc,dnom,ze,ntop,nbot);need_decomp=false;}
  uw_vd_lu_solve(pver,q[m-1],ca,ze,dnom,ntop,nbot,R8(0));
 }
}
#endif
