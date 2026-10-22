// NOAA-EMC/HRRR 40ee6058, phys/module_diag_gsd.F:724-813.
extern "C" __global__ void precipitation_buckets(
 const real* rainc,const real* rain,const real* snow,const real* graup,
 const real* t2,real* pc,real* pn,real* sn,real* gn,int count,int reset) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
 real c=rainc[i];
 pc[i]=fmaxf(0.,(reset?0.:pc[i])+c);
 pn[i]=fmaxf(0.,(reset?0.:pn[i])+rain[i]);
 sn[i]=fmaxf(0.,(reset?0.:sn[i])+snow[i]+(t2[i]<273.15f?c:0.f));
 gn[i]=fmaxf(0.,(reset?0.:gn[i])+graup[i]);
}
