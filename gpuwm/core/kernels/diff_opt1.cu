// WRF v4.7.1 model-coordinate horizontal diffusion.
// module_big_step_utilities_em.F:2715-3060. Explicit REAL staging keeps
// coordinate fluxes separate from the unchanged diff_opt=2 kernels.
__device__ __forceinline__ float d1add(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float d1sub(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float d1mul(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d1four(float a,float b,float c,float d){
    return d1mul(0.25f,d1add(d1add(d1add(a,b),c),d));
}
__device__ __forceinline__ int d1index(int n,int size,int boundary){
    return boundary ? max(0,min(n,size-1)) : (n%size+size)%size;
}
struct Diff1 {
    const float *field,*km,*mu,*c1,*c2,*base,*msft,*msfu,*msfv;
    int nz,ny,nx,stag,bx,by,perturb;
    __device__ float mass(int k,int j,int i) const {
        return d1add(d1mul(c1[k],mu[d1index(j,ny,by)*nx+d1index(i,nx,bx)]),c2[k]);
    }
    __device__ float kval(int k,int j,int i) const {
        return km[(max(0,min(k,nz-1))*ny+d1index(j,ny,by))*nx+d1index(i,nx,bx)];
    }
    __device__ float fval(int k,int j,int i) const {
        int ii=stag==1 ? (bx ? max(0,min(i,nx)) : d1index(i,nx,0)) : d1index(i,nx,bx);
        int jj=stag==2 ? (by ? max(0,min(j,ny)) : d1index(j,ny,0)) : d1index(j,ny,by);
        return field[(k*(ny+(stag==2))+jj)*(nx+(stag==1))+ii];
    }
    __device__ float bval(int k,int j,int i) const {
        return base[(k*ny+d1index(j,ny,by))*nx+d1index(i,nx,bx)];
    }
    __device__ float map(int stagger,int j,int i) const {
        int width=nx+(stagger==1);
        int ii=stagger==1 && bx ? max(0,min(i,nx)) : d1index(i,nx,bx);
        int jj=stagger==2 && by ? max(0,min(j,ny)) : d1index(j,ny,by);
        return (stagger==1?msfu:stagger==2?msfv:msft)[jj*width+ii];
    }
    __device__ float delta(int k,int j1,int i1,int j0,int i0) const {
        float v=d1sub(fval(k,j1,i1),fval(k,j0,i0));
        if(perturb) v=d1add(d1sub(v,bval(k,j1,i1)),bval(k,j0,i0));
        return v;
    }
};

extern "C" __global__ void wrf_diff_opt1_horizontal(
    const float* field,const float* km,const float* mu,const float* c1,const float* c2,
    const float* base,const float* msft,const float* msfu,const float* msfv,
    float rdx,float rdy,float* tendency,int nz,int ny,int nx,
    int stag,int bx,int by,int perturb)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
    int nxs=nx+(stag==1),nys=ny+(stag==2),nzs=nz+(stag==3);
    if(i>=nxs||j>=nys||k>=nzs||(stag==3&&(k==0||k==nz)))return;
    if(bx&&(i==0||i>=nx-(stag!=1)))return;
    if(by&&(j==0||j>=ny-(stag!=2)))return;
    Diff1 q={field,km,mu,c1,c2,base,msft,msfu,msfv,nz,ny,nx,stag,bx,by,perturb};
    float xm,xp,ym,yp;
    if(stag==1){
        xm=d1mul(d1mul(q.mass(k,j,i-1),q.kval(k,j,i-1)),rdx);
        xp=d1mul(d1mul(q.mass(k,j,i),q.kval(k,j,i)),rdx);
        ym=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j-1,i-1),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k,j-1,i-1),q.kval(k,j,i-1))),rdy);
        yp=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j+1,i),q.mass(k,j+1,i-1),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j+1,i),q.kval(k,j+1,i-1),q.kval(k,j,i-1))),rdy);
    }else if(stag==2){
        xm=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j-1,i-1),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k,j-1,i-1),q.kval(k,j,i-1))),rdx);
        xp=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j-1,i+1),q.mass(k,j,i+1)),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k,j-1,i+1),q.kval(k,j,i+1))),rdx);
        // This WRF v4.7.1 coordinate-v meridional row omits dry mass.
        ym=d1mul(q.kval(k,j-1,i),rdy);
        yp=d1mul(q.kval(k,j,i),rdy);
    }else if(stag==3){
        xm=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j,i-1),q.mass(k,j,i),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j,i-1),q.kval(k-1,j,i),q.kval(k-1,j,i-1))),rdx);
        xp=d1mul(d1mul(d1four(q.mass(k,j,i+1),q.mass(k,j,i),q.mass(k,j,i+1),q.mass(k,j,i)),
                      d1four(q.kval(k,j,i+1),q.kval(k,j,i),q.kval(k-1,j,i+1),q.kval(k-1,j,i))),rdx);
        ym=d1mul(d1mul(d1mul(q.map(2,j,i),__fdiv_rn(1.0f,q.map(2,j,i))),
                      d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j,i),q.mass(k,j-1,i))),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k-1,j,i),q.kval(k-1,j-1,i)));
        ym=d1mul(ym,rdy);
        yp=d1mul(d1mul(d1mul(q.map(2,j+1,i),__fdiv_rn(1.0f,q.map(2,j+1,i))),
                      d1four(q.mass(k,j+1,i),q.mass(k,j,i),q.mass(k,j+1,i),q.mass(k,j,i))),
                      d1four(q.kval(k,j+1,i),q.kval(k,j,i),q.kval(k-1,j+1,i),q.kval(k-1,j,i)));
        yp=d1mul(yp,rdy);
    }else{
        xm=d1mul(d1mul(d1mul(d1mul(0.5f,d1add(q.kval(k,j,i),q.kval(k,j,i-1))),0.5f),
                      d1add(q.mass(k,j,i),q.mass(k,j,i-1))),rdx);
        xp=d1mul(d1mul(d1mul(d1mul(0.5f,d1add(q.kval(k,j,i+1),q.kval(k,j,i))),0.5f),
                      d1add(q.mass(k,j,i+1),q.mass(k,j,i))),rdx);
        ym=d1mul(d1mul(d1mul(d1mul(d1mul(q.map(2,j,i),__fdiv_rn(1.0f,q.map(2,j,i))),0.5f),
                      d1add(q.kval(k,j,i),q.kval(k,j-1,i))),0.5f),
                      d1add(q.mass(k,j,i),q.mass(k,j-1,i)));
        ym=d1mul(ym,rdy);
        yp=d1mul(d1mul(d1mul(d1mul(d1mul(q.map(2,j+1,i),__fdiv_rn(1.0f,q.map(2,j+1,i))),0.5f),
                      d1add(q.kval(k,j+1,i),q.kval(k,j,i))),0.5f),
                      d1add(q.mass(k,j+1,i),q.mass(k,j,i)));
        yp=d1mul(yp,rdy);
    }
    float map=q.map(stag==3?0:stag,j,i),map2=d1mul(map,map);
    float tx=d1mul(d1mul(map2,rdx),d1sub(d1mul(xp,q.delta(k,j,i+1,j,i)),d1mul(xm,q.delta(k,j,i,j,i-1))));
    float ty=d1mul(d1mul(map2,rdy),d1sub(d1mul(yp,q.delta(k,j+1,i,j,i)),d1mul(ym,q.delta(k,j,i,j-1,i))));
    int idx=(k*nys+j)*nxs+i;
    tendency[idx]=d1add(tendency[idx],d1add(tx,ty));
}

// module_diffusion_em.F:1934-2044, diff_opt=1 has no slope reduction.
extern "C" __global__ void wrf_diff_opt1_km4(const float* d11,const float* d22,
    const float* d12,const float* msft,float dx,float dy,float cs,float pr,
    float* km,float* kh,int nz,int ny,int nx,int bx,int by)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
    if(i>=nx||j>=ny||k>=nz)return;
    int idx=(k*ny+j)*nx+i;
    if((bx&&(i==0||i==nx-1))||(by&&(j==0||j==ny-1))){km[idx]=kh[idx]=0.0f;return;}
    int ip=d1index(i+1,nx,bx),jp=d1index(j+1,ny,by);
    float d=d1sub(d11[idx],d22[idx]);
    float cross=d1four(d12[idx],d12[(k*ny+jp)*nx+i],d12[(k*ny+j)*nx+ip],d12[(k*ny+jp)*nx+ip]);
    float strain=sqrtf(d1add(d1mul(0.25f,d1mul(d,d)),d1mul(cross,cross)));
    float map=msft[j*nx+i],length=sqrtf(__fdiv_rn(d1mul(__fdiv_rn(dx,map),dy),map));
    float value=fminf(d1mul(d1mul(d1mul(d1mul(cs,cs),length),length),strain),d1mul(10.0f,length));
    km[idx]=value;kh[idx]=__fdiv_rn(value,pr);
}
