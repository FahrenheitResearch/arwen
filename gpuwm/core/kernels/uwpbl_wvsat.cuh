// WRF module_cam_wv_saturation.F:82-98,235-298,512-639,757-774.
#ifndef UW_WVSAT_CUH
#define UW_WVSAT_CUH
__device__ R8 uw_estblf(R8 td) {
 R8 e=uw_max(uw_min(td,R8(375.16)),R8(173.16));
 int i=uw_int(e-R8(173.16))+1; R8 ai=uw_aint(e-R8(173.16));
 return (R8(173.16)+ai-e+R8(1))*R8(UW_ESTBL[i-1])-(R8(173.16)+ai-e)*R8(UW_ESTBL[i]);
}
__device__ void uw_aqsat(V t,V p,V es,V qs,int kstart,int kend) {
 R8 omeps=R8(1)-R8(UW_EPSILO);
 for(int k=kstart;k<=kend;++k) {
  es(k)=uw_estblf(t(k)); qs(k)=R8(UW_EPSILO)*es(k)/(p(k)-omeps*es(k));
  qs(k)=uw_min(R8(1),qs(k));
  if(qs(k)<R8(0)) {qs(k)=R8(1);es(k)=p(k);}
 }
}
__device__ int uw_fqsatd(R8 t,R8 p,R8& es,R8& qs,R8& gam) {
 R8 omeps=R8(1)-R8(UW_EPSILO);
 es=uw_estblf(t);qs=R8(UW_EPSILO)*es/(p-omeps*es);qs=uw_min(R8(1),qs);
 if(qs<R8(0)) {qs=R8(1);es=p;}
 R8 trinv=R8(1)/R8(20),tc=t-R8(UW_TMELT);
 bool lflg=tc>=R8(-20)&&tc<R8(0);
 R8 weight=uw_min(-(tc*trinv),R8(1));
 R8 hlatsb=R8(UW_LATVAP)+weight*R8(UW_LATICE);
 R8 hlatvp=R8(UW_LATVAP)-R8(2369)*tc;
 R8 hltalt=t<R8(UW_TMELT)?hlatsb:hlatvp;
 R8 tterm=lflg?R8(5.04469588506e-01)+tc*(R8(-5.47288442819)+tc*(R8(-3.67471858735e-01)+tc*(R8(-8.95963532403e-03)+tc*R8(-7.78053686625e-05)))):R8(0);
 R8 desdt=hltalt*es/(R8(UW_RH2O)*t*t)+tterm*trinv;
 gam=hltalt*qs*p*desdt/(R8(UW_CPAIR)*es*(p-omeps*es));
 if(qs==R8(1))gam=R8(0);return 1;
}
#endif
