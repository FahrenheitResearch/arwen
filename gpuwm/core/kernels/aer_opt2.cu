// WRF 4.6.1 module_ra_aerosol.F, SW only, without optional 3D AOD input.
// WOOF owns the float32 math helpers supplied by the shared kernel loader.
// Explicit rounded operations preserve WRF statement order in both modes.
__device__ float a2_add(float a,float b){return __fadd_rn(a,b);}
__device__ float a2_sub(float a,float b){return __fsub_rn(a,b);}
__device__ float a2_mul(float a,float b){return __fmul_rn(a,b);}
__device__ float a2_div(float a,float b){return __fdiv_rn(a,b);}
__device__ float a2_rh(float p,float t,float q){
    float tc=a2_sub(t,273.15f),rv=fmaxf(0.f,q);
    float es=a2_mul(6.112f,gfk_exp(a2_div(a2_mul(17.6f,tc),a2_add(tc,243.5f))));
    float e=a2_div(a2_mul(a2_mul(0.01f,rv),p),a2_add(rv,0.62197f));
    return fminf(99.f,fmaxf(0.f,a2_div(a2_mul(100.f,e),es)));
}
__device__ float a2_lagrange(float rh,const float *nodes,const float *lut,int type,int nb,int b,float scale,bool scaled){
    int ii=1;
    while(ii<=8 && rh>nodes[ii-1]) ++ii;
    int imin=max(1,ii-3),imax=min(8,ii+2);
    float result=0.f;
    for(int jj=imin;jj<=imax;++jj){
        float lj=1.f;
        for(int k=imin;k<=imax;++k)
            if(k!=jj) lj=a2_div(a2_mul(lj,a2_sub(rh,nodes[k-1])),a2_sub(nodes[jj-1],nodes[k-1]));
        float term=a2_mul(lj,lut[((type-1)*8+jj-1)*nb+b]);
        if(scaled) term=a2_mul(term,scale);
        result=a2_add(result,term);
    }
    return result;
}
__device__ float a2_spectral(float aod,float alpha,float lo,float hi){
    alpha=fminf(2.5f,fmaxf(-0.3f,alpha));
    float rate;
    if(alpha<0.999f || alpha>1.001f){
        float e=a2_sub(1.f,alpha);
        rate=a2_div(a2_mul(gfk_pow(0.55f,alpha),a2_sub(gfk_pow(hi,e),gfk_pow(lo,e))),a2_mul(e,a2_sub(hi,lo)));
    } else {
        rate=a2_mul(a2_div(0.55f,a2_sub(hi,lo)),gfk_log(a2_div(hi,lo)));
    }
    return a2_mul(aod,rate);
}
extern "C" __global__ void aer_opt2_columns(
 int nc,int nz,int nl,int nb,int type,int oaod,int oang,int ossa,int oasy,
 const float *p,const float *t,const float *q,const float *dz,const float *ht,const float *input,
 const float *raod,const float *slut,const float *alut,const float *low,const float *high,const float *nodes,
 float *tau,float *ssa,float *asy,float *aod3d,float *diag){
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc) return;
    int base=c*nz;
    float aod=input[c*4],alpha=input[c*4+1],sv=input[c*4+2],av=input[c*4+3];
    float rh0=a2_rh(p[base],t[base],q[base]);
    float z0=a2_add(ht[c],a2_mul(0.5f,dz[base])),ztop=z0;
    for(int k=1;k<nz;++k) ztop=a2_add(ztop,a2_mul(0.5f,a2_add(dz[base+k-1],dz[base+k])));
    float htoa=a2_add(ztop,a2_mul(0.5f,dz[base+nz-1]));
    float den=a2_mul(2500.f,a2_sub(gfk_exp(a2_div(-ht[c],2500.f)),gfk_exp(a2_div(-htoa,2500.f))));
    float xy=0.f,xx=0.f,sdiag=0.f,adiag=0.f;
    for(int b=0;b<nb;++b){
        float total=(oang==3 ? a2_lagrange(rh0,nodes,raod,type,nb,b,aod,true):a2_spectral(aod,alpha,low[b],high[b]));
        int rmax=(nb==14 ? 10:8);
        if(oang==3 && aod>0.f && b>=7 && b<=rmax){
            float x=gfk_log(a2_div(a2_mul(0.5f,a2_add(low[b],high[b])),0.55f));
            xy=a2_add(xy,a2_mul(x,gfk_log(a2_div(total,aod))));
            xx=a2_add(xx,a2_mul(x,x));
        }
        float scale=a2_div(total,den),z=z0;
        for(int k=0;k<nl;++k){
            int idx=(c*nb+b)*nl+k;
            if(k>=nz){tau[idx]=0.f;ssa[idx]=1.f;asy[idx]=0.f;continue;}
            if(k>0) z=a2_add(z,a2_mul(0.5f,a2_add(dz[base+k-1],dz[base+k])));
            float rh=a2_rh(p[base+k],t[base+k],q[base+k]);
            tau[idx]=a2_mul(a2_mul(scale,dz[base+k]),gfk_exp(a2_div(-z,2500.f)));
            ssa[idx]=(ossa==3 ? a2_lagrange(rh,nodes,slut,type,nb,b,1.f,false):sv);
            asy[idx]=(oasy==3 ? a2_lagrange(rh,nodes,alut,type,nb,b,1.f,false):av);
            if(b==9) aod3d[base+k]=tau[idx];
            if(k==0 && b>=7 && b<=rmax){sdiag=a2_add(sdiag,ssa[idx]);adiag=a2_add(adiag,asy[idx]);}
        }
    }
    float navg=(nb==14 ? 4.f:2.f);
    diag[c*4]=aod;
    diag[c*4+1]=(oang==3 ? (aod==0.f ? 0.f : (nb==14 ? a2_div(-a2_div(xy,navg),a2_div(xx,navg)):a2_div(-xy,xx))):(oang==1 ? fminf(2.5f,fmaxf(-0.3f,alpha)):alpha));
    diag[c*4+2]=(ossa==3 ? a2_div(sdiag,navg):sv);
    diag[c*4+3]=(oasy==3 ? a2_div(adiag,navg):av);
}

extern "C" __global__ void aer_opt2_normal_numerator(
 int n,const float *tau,const float *ssa,const float *asy,int *bad){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    // Decode bits so a subnormal operand cannot be flushed by conversion.
    unsigned int it=__float_as_uint(tau[i]),is=__float_as_uint(ssa[i]),ia=__float_as_uint(asy[i]);
    double td=((it&0x7f800000u)==0u)?(double)(it&0x7fffffu)*0x1p-149:(double)tau[i];
    double sd=((is&0x7f800000u)==0u)?(double)(is&0x7fffffu)*0x1p-149:(double)ssa[i];
    double ad=((ia&0x7f800000u)==0u)?(double)(ia&0x7fffffu)*0x1p-149:fabs((double)asy[i]);
    double product=fabs(td*sd*ad);
    if(product>0. && product<0x1p-126) atomicExch(bad,1);
}
