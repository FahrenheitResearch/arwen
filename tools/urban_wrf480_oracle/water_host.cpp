// Native test entry for the CUDA-source root-aware water solve.
#include "../../gpuwm/core/kernels/urban_ucm_host.cpp"
#include "../../gpuwm/core/kernels/urban_ground_water_480.cuh"
extern "C" void water_host(float* out, const float* smcp, const float* zsoil,
    const float* et, const float* root, const float* v)
{
    UcmHostRounding rounding;
    float rh[4]={}, a[4]={}, b[4]={}, c[4]={};
    float r1=0, r2=0, r3=0, drip=0, cmc=0;
    float smc[4]={smcp[0],smcp[1],smcp[2],smcp[3]};
    u_srt_480(rh,v[9],et,smcp,v[12],zsoil,v[6],v[5],v[2],v[3],
              r1,r2,v[0],v[4],root,a,b,c);
    for(int k=0;k<3;++k) {out[k]=rh[k];out[3+k]=a[k];out[6+k]=b[k];out[9+k]=c[k];}
    out[20]=r1;out[21]=r2;
    u_smflx_480(smcp,smc,v[11],cmc,v[0],v[1],zsoil,v[2],v[3],v[4],v[5],v[6],
                v[7],v[8],r1,r2,r3,v[9],v[10],et,root,drip);
    for(int k=0;k<3;++k)out[12+k]=smc[k];
    out[15]=cmc;out[16]=r1;out[17]=r2;out[18]=r3;out[19]=drip;
}
