// chem/module_gocart_seasalt.F:154-167, process constants, REAL literals widened.
__device__ const double ss_c_old[5]={(double)1.373f,(double)3.2f,(double).057f,(double)1.05f,(double)1.190f};
__device__ const double ss_b_old[2]={(double).380f,(double).650f};
__device__ const double ss_dr=.05, ss_frh=2., ss_theta=30.f;
// module_data_gocart_seas.F:6, REAL literal widened into REAL*8.
__device__ const double ss_pi=(double)3.141592653559f;
// The pinned source sets old=.TRUE., new=.FALSE.; theta is unused in this arm.
__device__ const bool ss_old=true, ss_new=false;
extern "C" __global__ void chem_seasalt_offsets(int rows,const double* ra,
    const double* rb,int* offset) {
    if(blockIdx.x || threadIdx.x) return;
    offset[0]=0;
    for(int n=0;n<rows;n++) {
        int nr=(int)DADD(DDIV(DSUB(DMUL(rb[n],ss_frh),DMUL(ra[n],ss_frh)),ss_dr),(double).001f);
        offset[n+1]=offset[n]+nr;
    }
}
// chem/module_gocart_seasalt.F:225-269. The old arm is a compile-time constant.
extern "C" __global__ void chem_seasalt_table(int rows,const double* ra,
    const double* rb,const double* density,const int* offset,double* table,float dt) {
    if(blockIdx.x || threadIdx.x) return;
    for(int n=0;n<rows;n++) {
        double r=DMUL(ra[n],ss_frh);
        for(int ir=offset[n];ir<offset[n+1];ir++) {
            double rw=DADD(r,DMUL(ss_dr,(double).5f)); r=DADD(r,ss_dr);
            double b=DDIV(DSUB(ss_b_old[0],log10(rw)),ss_b_old[1]);
            double dfn=DMUL(DMUL(DDIV(ss_c_old[0],pow(rw,3.)),DADD(1.f,DMUL(ss_c_old[2],pow(rw,ss_c_old[3])))),pow(10.,DMUL(ss_c_old[4],exp(-DMUL(b,b)))));
            double rd=DMUL(DDIV(rw,ss_frh),1.e-6);
            double dfm=DMUL((double)(4.f/3.f),ss_pi);
            dfm=DMUL(dfm,DMUL(rd,DMUL(rd,rd)));
            dfm=DMUL(dfm,density[n]); dfm=DMUL(dfm,ss_frh); dfm=DMUL(dfm,dfn);
            dfm=DMUL(dfm,ss_dr); table[ir]=DMUL(dfm,dt);
        }
    }
}
extern "C" __global__ void chem_seasalt(int rows,int nc,
    const unsigned long long* species,const unsigned long long* emissions,
    const int* offset,const double* table,const float* xland,const float* z,
    const float* p,const float* dz,const float* u10,const float* v10,
    const float* u,const float* v,float dx,float gravity) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc || !(xland[c]>1.5f && z[c]<1.e-3f)) return;
    double wind=sqrtf(FADD(FMUL(u10[c],u10[c]),FMUL(v10[c],v10[c])));
    if(dz[c]<12.f) wind=sqrtf(FADD(FMUL(u[c],u[c]),FMUL(v[c],v[c])));
    double airmass=FDIV(FMUL(FMUL(-FSUB(p[nc+c],p[c]),dx),dx),gravity);
    double dxy=FMUL(dx,dx);
    for(int n=0;n<rows;n++) {
        float* q=(float*)species[n]; float* e=(float*)emissions[n];
        double tc=DMUL(q[c],1.e-9),bems=0.;
        for(int ir=offset[n];ir<offset[n+1];ir++) {
            double src=DMUL(DMUL(table[ir],dxy),pow(wind,ss_c_old[1]));
            if(src<0.) src=0.; tc=DADD(tc,DDIV(src,airmass)); bems=DADD(bems,src);
        }
        q[c]=__double2float_rn(DMUL(tc,1.e9)); e[c]=__double2float_rn(bems);
    }
}
