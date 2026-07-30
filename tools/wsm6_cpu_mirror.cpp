// CPU execution harness for kernels/wsm6.cu.  This exercises the exact same
// float implementation without creating a CUDA context; it is a verification
// tool, not a second science implementation.
#define WSM6_CPU_MIRROR
#include "../gpuwm/core/kernels/wsm6.cu"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    constexpr int nz=8, ny=1, nx=1;
    int scenario=(argc>1)?std::atoi(argv[1]):0;
    int nsteps=(argc>2)?std::atoi(argv[2]):1;
    float dt=(argc>3)?std::strtof(argv[3],nullptr):30.0f;
    if(nsteps<1 || !(dt>0.0f)) return 2;
    float theta[nz],qv[nz],qc[nz],qi[nz],qr[nz],qs[nz],qg[nz];
    float den[nz],p[nz],pii[nz],dz[nz],effc[nz],effi[nz],effs[nz];
    for(int k=0;k<nz;++k){
        float tk=(scenario==0)?(286.0f-0.7f*k)
                :(scenario==1)?(269.0f-1.8f*k):(280.0f-2.0f*k);
        p[k]=96000.0f-7000.0f*k;
        den[k]=p[k]/(287.0f*tk);
        pii[k]=powf(p[k]/100000.0f,287.0f/1004.5f);
        theta[k]=tk/pii[k]; dz[k]=500.0f+25.0f*k;
        qv[k]=(scenario==0)?(0.011f-0.0007f*k)
             :(scenario==1)?(0.0038f-0.00025f*k):(0.0045f-0.0002f*k);
        qc[k]=(scenario==0)?(7.0e-4f+2.0e-5f*k)
             :(scenario==1)?2.0e-5f:1.0e-4f;
        // Keep both oracle cases free of pre-existing precipitating species.
        // Scenario 1 therefore isolates cold-cloud source terms; a separate
        // sedimentation case exercises WRF's PLM remap.
        qi[k]=(scenario>=2)?(2.0e-5f+1.0e-6f*k):0.0f;
        qr[k]=(scenario>=2)?(2.0e-4f+1.0e-5f*k):0.0f;
        qs[k]=(scenario>=2)?(1.2e-4f+8.0e-6f*k):0.0f;
        qg[k]=(scenario>=2)?(5.0e-5f+5.0e-6f*k):0.0f;
        effc[k]=2.5f;effi[k]=5.0f;effs[k]=10.0f;
    }
    float rain=0,raincv=0,snow=0,snowcv=0,graup=0,graupcv=0,sr=0;
    for(int n=0;n<nsteps;++n)
        wsm6_column_impl(theta,qv,qc,qi,qr,qs,qg,den,p,pii,dz,
            &rain,&raincv,&snow,&snowcv,&graup,&graupcv,&sr,
            effc,effi,effs,dt,(scenario==3)?1:0,nz,1,0);
    std::printf("surface %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
        rain,raincv,snow,snowcv,graup,graupcv,sr);
    for(int k=0;k<nz;++k)std::printf(
        "%d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
        k,theta[k]*pii[k],qv[k],qc[k],qi[k],qr[k],qs[k],qg[k],
        effc[k],effi[k],effs[k]);
    return 0;
}
