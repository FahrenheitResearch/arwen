// Apache-2.0. Aerosol increment follows the RTE-RRTMGP BSD-3-Clause
// optical-property moments. Cloud extinction alone is McICA masked.
// One thread per g-point/layer/column; all spectra retain model-top zeros.
extern "C" __global__ void chem_radiation_finalize(
 const float *gas_tau, const float *gas_ssa,
 const float *cloud_tau, const float *cloud_ssa, const float *cloud_g,
 const float *aer_tau, const float *aer_ssa, const float *aer_g,
 const int *bands, const unsigned char *mask,
 float *out_tau, float *out_ssa, float *out_g,
 int n, int ng, int nb, int sw, int have_mask, int clear) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n) return;
 int b=(i/ng)*nb+bands[i%ng];
 float tc=clear?0.f:cloud_tau[b];
 if(have_mask && !clear) tc*=float(mask[i]);
 float ta=aer_tau[b], wa=aer_ssa[b];
 if(!sw) {
  out_tau[i]=gas_tau[i]+tc*(1.f-cloud_ssa[b])+ta*(1.f-wa);
  return;
 }
 float t=gas_tau[i]+tc+ta;
 float sc=gas_tau[i]*gas_ssa[i]+tc*cloud_ssa[b]+ta*wa;
 float sg=tc*cloud_ssa[b]*cloud_g[b]+ta*wa*aer_g[b];
 const float floor=3.f*1.17549435e-38f;
 float w=sc/fmaxf(floor,t), g=sg/fmaxf(floor,sc);
 // Delta scaling is applied after ALL constituents have been mixed.
 float f=g*g, wf=w*f;
 out_tau[i]=(1.f-wf)*t;
 out_ssa[i]=(w-wf)/fmaxf(floor,1.f-wf);
 out_g[i]=(g-f)/fmaxf(floor,1.f-f);
}
