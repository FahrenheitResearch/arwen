// Native firebrand arithmetic preserves the source's mixed REAL/DOUBLE tree.
struct sfire_brand {float mass,diam,effd,temp,tvel;};
__device__ float sfire_brand_sqrt(float x) {
    return x<0.0f ? __int_as_float(0xffc00000) : __fsqrt_rn(x);
}
__device__ float sfire_brand_pow4(float x) {float a=x*x;return a*a;}
__device__ sfire_brand sfire_brand_property(float d,float e,float t,float v,float density,int original=0) {
    sfire_brand p={0.0f,d,e,t,v};
    float a=density*3.1415927410125732421875f;
    p.mass=(float)__ddiv_rn((double)a*glibc_pow(__ddiv_rn((double)e,1000.0),original?2.0:3.0),6.0);
    return p;
}
__device__ void sfire_brand_burnout(sfire_brand &p,float dt,float pres,float aird,float temp,float wind,
                                    float density,float density_char) {
    p.mass=fmaxf(p.mass,1.1920928955078125e-7f);
    float aird2=(float)__ddiv_rn(1000.0*(double)pres,(double)(287.149993896484375f*p.temp));
    float vel=fabsf(wind-p.tvel);
    float pdia=(float)__ddiv_rn((double)p.diam,1000.0);
    float pedia=(float)__ddiv_rn((double)p.effd,1000.0);
    float dmvc=(float)__ddiv_rn((double)0.001458000042475759983f*glibc_pow((double)temp,1.5),
                               (double)(temp+110.40000152587890625f));
    float dmvc2=(float)__ddiv_rn((double)0.001458000042475759983f*glibc_pow((double)p.temp,1.5),
                                (double)(p.temp+110.40000152587890625f));
    float gama=(float)__ddiv_rn((double)(__fdiv_rn(dmvc,aird)+__fdiv_rn(dmvc2,aird2)),2.0);
    float reyn=__fdiv_rn(vel*pdia,gama);
    float rr=gfk_pow(reyn,0.5f);
    float pr=gfk_pow(0.699999988079071044921875f,0.3333333432674407958984375f);
    float nuss=(float)(2.0+0.6*(double)rr*(double)pr);
    float hbar=__fdiv_rn(nuss*27.0f,pdia);
    float beta=(float)((double)4.79999982871959219e-7f+
        (double)4.79999982871959219e-7f*(0.276*(double)rr*(double)pr));
    float parta=sfire_brand_pow4(pdia);
    float partb=(float)(1.7320508075688772935*(double)(beta*beta)*(double)(dt*dt));
    float pdia_new4=parta-partb;
    p.diam=(float)glibc_pow((double)pdia_new4,0.25);
    float p_effd2=pedia*pedia-beta*dt;
    p.effd=sfire_brand_sqrt(p_effd2);
    p.mass=(float)__ddiv_rn((double)(((density*3.1415927410125732421875f)*p_effd2)*p.effd),6.0);
    float fbvol=__fdiv_rn(p.mass,density);
    float qcon=hbar*(p.temp-temp);
    float qrad=(5.67e-5f*0.89999997615814208984375f)*
               (sfire_brand_pow4(p.temp)-sfire_brand_pow4(temp));
    float pratio=__fdiv_rn(p.effd,p.diam);
    float cpmix=(float)((double)(pratio*1466.0f)+(1.0-(double)pratio)*712.0);
    float partc=(float)__ddiv_rn(6.0,(double)((p.mass*cpmix)*p.diam));
    float dtemp=((dt*fbvol)*partc)*(qcon+qrad);
    p.temp=p.temp-dtemp;
    p.diam=(float)(1000.0*(double)p.diam);
    p.effd=(float)(1000.0*(double)p.effd);
}
__device__ void sfire_brand_termvel(sfire_brand &p,float &h,float dt,float pres,float aird,
                                   float density,float density_char) {
    float aird2=(float)__ddiv_rn(1000.0*(double)pres,(double)(287.149993896484375f*p.temp));
    float pdia=(float)__ddiv_rn((double)p.diam,1000.0);
    float ratio=__fdiv_rn(p.effd,p.diam);
    float parta=((ratio*density+(1.0f-ratio)*density_char)*pdia)*9.80615997314453125f;
    float partb=(float)(3.0*__ddiv_rn((double)(aird+aird2),2.0)*(double)0.449999988079071044921875f);
    ratio=__fdiv_rn(parta,partb);
    if(!isnan(ratio))ratio=fmaxf(fminf(3.40282346638528859811704183484516925e38f,ratio),1.17549435082228750796873653722224568e-38f);
    float vt=sfire_brand_sqrt(ratio);
    h=(float)fmax(0.0,(double)(h-vt*fabsf(dt)));
    p.tvel=vt;
}
__device__ sfire_brand sfire_brand_read(const float *a,int n,int q) {
    return {a[q],a[n+q],a[2*n+q],a[3*n+q],a[4*n+q]};
}
__device__ void sfire_brand_write(float *a,int n,int q,sfire_brand p) {
    // The raw native helper emits the scalar host invalid-operation word.
    // Driver removal handles fully burnt particles before deposition.
    float nan=__int_as_float(0xffc00000);
    if(isnan(p.mass))p.mass=nan;if(isnan(p.diam))p.diam=nan;
    if(isnan(p.effd))p.effd=nan;if(isnan(p.temp))p.temp=nan;
    if(isnan(p.tvel))p.tvel=nan;
    a[q]=p.mass;a[n+q]=p.diam;a[2*n+q]=p.effd;a[3*n+q]=p.temp;a[4*n+q]=p.tvel;
}
extern "C" __global__ void sfire_spotting_property(const float *input,float *out,int n,float density,int original) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;
    if(q<n)sfire_brand_write(out,n,q,sfire_brand_property(input[q],input[n+q],input[2*n+q],input[3*n+q],density,original));
}
extern "C" __global__ void sfire_spotting_physics(float *prop,float *height,const float *pres,
 const float *aird,const float *temp,const float *wind,int n,int mode,float dt,float density,float density_char) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;
    if(q>=n)return;
    sfire_brand p=sfire_brand_read(prop,n,q);
    if(mode!=1)sfire_brand_burnout(p,dt,pres[q],aird[q],temp[q],wind[q],density,density_char);
    if(mode!=0)sfire_brand_termvel(p,height[q],dt,pres[q],aird[q],density,density_char);
    sfire_brand_write(prop,n,q,p);
}

__device__ float sfire_brand_interp2(float x,float y,float a,float b,float c,float d) {
    int i=(int)floorf(x),j=(int)floorf(y);
    float da=fabsf((float(i+1)-x)*(float(j)-y));
    float db=fabsf((float(i+1)-x)*(float(j+1)-y));
    float dc=fabsf((float(i)-x)*(float(j+1)-y));
    float dd=fabsf((float(i)-x)*(float(j)-y));
    return __fdiv_rn(((b*da+a*db)+c*dc)+d*dd,((da+db)+dc)+dd);
}
__device__ float sfire_brand_interp3(float x,float y,float z,
 float a,float b,float c,float d,float e,float f,float g,float h) {
    int k=max(1,(int)floorf(z));
    float bot=z-float(k),top=float(k+1)-z;
    float lo=sfire_brand_interp2(x,y,a,b,c,d),hi=sfire_brand_interp2(x,y,e,f,g,h);
    return hi*bot+lo*top;
}
__device__ float sfire_brand_field(const float *a,int i,int j,int k,int nx,int ny,int ml,int mb) {
    return a[((k-1)*ny+j-mb)*nx+i-ml];
}
__device__ float sfire_brand_xy(const float *a,float x,float y,int k,int nx,int ny,int ml,int mb) {
    int i=(int)floorf(x),j=(int)floorf(y);
    return sfire_brand_interp2(x,y,
       sfire_brand_field(a,i,j,k,nx,ny,ml,mb),sfire_brand_field(a,i,j+1,k,nx,ny,ml,mb),
       sfire_brand_field(a,i+1,j,k,nx,ny,ml,mb),sfire_brand_field(a,i+1,j+1,k,nx,ny,ml,mb));
}
__device__ float sfire_brand_grid3(const float *a,float x,float y,float z,
 int i,int j,int k,int nx,int ny,int ml,int mb) {
    return sfire_brand_interp3(x,y,z,
       sfire_brand_field(a,i,j,k,nx,ny,ml,mb),sfire_brand_field(a,i,j+1,k,nx,ny,ml,mb),
       sfire_brand_field(a,i+1,j,k,nx,ny,ml,mb),sfire_brand_field(a,i+1,j+1,k,nx,ny,ml,mb),
       sfire_brand_field(a,i,j,k+1,nx,ny,ml,mb),sfire_brand_field(a,i,j+1,k+1,nx,ny,ml,mb),
       sfire_brand_field(a,i+1,j,k+1,nx,ny,ml,mb),sfire_brand_field(a,i+1,j+1,k+1,nx,ny,ml,mb));
}
__device__ float sfire_brand_hgt2k(float x,float y,float h,const float *z,
 int nx,int ny,int nz,int ml,int mb,int &valid) {
    float x0=x-0.5f,y0=y-0.5f;
    int i=(int)floorf(x0),j=(int)floorf(y0),k=0;
    if(i<ml||i+1>=ml+nx||j<mb||j+1>=mb+ny){valid=0;return 0.0f;}
    float best=3.4028234663852886e38f;
    for(int kk=1;kk<=nz;++kk) {
        float d=h-sfire_brand_field(z,i,j,kk,nx,ny,ml,mb);
        if(d>=0.0f && d<best){best=d;k=kk;}
    }
    if(k<1||k>=nz){valid=0;return 0.0f;}
    float lo=sfire_brand_xy(z,x0,y0,k,nx,ny,ml,mb);
    float hi=sfire_brand_xy(z,x0,y0,k+1,nx,ny,ml,mb);
    return float(k)+__fdiv_rn(h-lo,hi-lo);
}
__device__ void sfire_brand_uvw(float x,float y,float z,const float *u,const float *v,const float *w,
 int nx,int ny,int ml,int mb,float &ru,float &rv,float &rw) {
    float x0=x-0.5f,y0=y-0.5f,z0=z-0.5f;
    int i=(int)floorf(x),j=(int)floorf(y),k=max(1,(int)floorf(z));
    int i0=(int)floorf(x0),j0=(int)floorf(y0),k0=max(1,(int)floorf(z0));
    ru=sfire_brand_grid3(u,x,y0,z0,i,j0,k0,nx,ny,ml,mb);
    rv=sfire_brand_grid3(v,x0,y,z0,i0,j,k0,nx,ny,ml,mb);
    // W interpolates X/Z planes and then interpolates along Y.
    rw=sfire_brand_interp3(x0,z,y,
      sfire_brand_field(w,i0,j,k,nx,ny,ml,mb),sfire_brand_field(w,i0,j,k+1,nx,ny,ml,mb),
      sfire_brand_field(w,i0+1,j,k,nx,ny,ml,mb),sfire_brand_field(w,i0+1,j,k+1,nx,ny,ml,mb),
      sfire_brand_field(w,i0,j+1,k,nx,ny,ml,mb),sfire_brand_field(w,i0,j+1,k+1,nx,ny,ml,mb),
      sfire_brand_field(w,i0+1,j+1,k,nx,ny,ml,mb),sfire_brand_field(w,i0+1,j+1,k+1,nx,ny,ml,mb));
}
__device__ void sfire_brand_met(float x,float y,float z,const float *p,const float *t,const float *d,
 int nx,int ny,int ml,int mb,int ihs,int ihe,int jhs,int jhe,float &lp,float &lt,float &ld) {
    float x0=x-0.5f,y0=y-0.5f;
    int i=min(max((int)floorf(x0),ihs),ihe),j=min(max((int)floorf(y0),jhs),jhe),k=(int)floorf(z);
    lp=sfire_brand_grid3(p,x0,y0,z,i,j,k,nx,ny,ml,mb);
    lt=sfire_brand_grid3(t,x0,y0,z,i,j,k,nx,ny,ml,mb);
    float zh=z-0.5f;k=max(1,(int)floorf(zh));
    ld=sfire_brand_grid3(d,x0,y0,zh,i,j,k,nx,ny,ml,mb);
    ld=(float)((double)ld*1000.0);
}
extern "C" __global__ void sfire_spotting_interp_box(const float *coordinates,const float *corners,
 float *out,int n) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=n)return;
    out[q]=sfire_brand_interp3(coordinates[q],coordinates[n+q],coordinates[2*n+q],
      corners[q],corners[n+q],corners[2*n+q],corners[3*n+q],corners[4*n+q],
      corners[5*n+q],corners[6*n+q],corners[7*n+q]);
}
extern "C" __global__ void sfire_spotting_sample(const float *coordinates,const float *u,const float *v,
 const float *w,const float *p,const float *t,const float *d,const float *z,float *out,int *status,
 int n,int nx,int ny,int nz,int ml,int mb,int ihs,int ihe,int jhs,int jhe) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=n)return;
    float x=coordinates[q],y=coordinates[n+q],h=coordinates[2*n+q];
    int valid=1;float zp=sfire_brand_hgt2k(x,y,h,z,nx,ny,nz,ml,mb,valid);
    if(!valid){status[q]=1;return;}
    out[q]=zp;
    sfire_brand_uvw(x,y,zp,u,v,w,nx,ny,ml,mb,out[n+q],out[2*n+q],out[3*n+q]);
    sfire_brand_met(x,y,zp,p,t,d,nx,ny,ml,mb,ihs,ihe,jhs,jhe,out[4*n+q],out[5*n+q],out[6*n+q]);
}
__device__ float sfire_brand_metric(const float *a,float x,float y,int nx,int ml,int mb) {
    return a[((int)floorf(y)-mb)*nx+(int)floorf(x)-ml];
}
extern "C" __global__ void sfire_spotting_advect(const float *coordinates,const int *life,float *prop,
 const float *u,const float *v,const float *w,const float *p,const float *t,const float *d,const float *z,
 const float *mx,const float *my,float *out,int *status,int n,int nx,int ny,int nz,int ml,int mb,
 int is_,int ie,int js,int je,int momentum,int original_cleanup,float dt,float land,float density,float density_char) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=n)return;
    float x=coordinates[q],y=coordinates[n+q],h=coordinates[2*n+q];
    if(h<=land){out[q]=x;out[n+q]=y;out[2*n+q]=h;return;}
    int valid=1;float zk=sfire_brand_hgt2k(x,y,h,z,nx,ny,nz,ml,mb,valid);
    if(!valid){out[q]=out[n+q]=out[2*n+q]=0.0f;status[q]=1;return;}
    float mfx=sfire_brand_metric(mx,x,y,nx,ml,mb),mfy=sfire_brand_metric(my,x,y,nx,ml,mb);
    float xm0=__fdiv_rn(x,mfx),ym0=__fdiv_rn(y,mfy);
    float ru,rv,rw;
    sfire_brand_uvw(x,y,zk,u,v,w,nx,ny,ml,mb,ru,rv,rw);
    float xm1=ru*dt+xm0,ym1=rv*dt+ym0,hm1=rw*dt+h;
    float x1=xm1*mfx,y1=ym1*mfy;
    if(fabsf(x1-x)>2.0f||fabsf(y1-y)>2.0f){out[q]=out[n+q]=out[2*n+q]=0.0f;status[q]=1;return;}
    float z1=sfire_brand_hgt2k(x1,y1,hm1,z,nx,ny,nz,ml,mb,valid);
    if(!valid){out[q]=out[n+q]=out[2*n+q]=0.0f;status[q]=1;return;}
    float wp=rw,xout=x1,yout=y1,hout=hm1,zout=z1;
    int ihs=is_-4,ihe=ie-1+4,jhs=js-4,jhe=je-1+4;
    if(!((int)floorf(x1-0.5f)<ihs||(int)floorf(x1-0.5f)+1>ihe||
         (int)floorf(y1-0.5f)<jhs||(int)floorf(y1-0.5f)+1>jhe)) {
        sfire_brand_uvw(x1,y1,z1,u,v,w,nx,ny,ml,mb,ru,rv,rw);
        float xm2=ru*dt+xm0,ym2=rv*dt+ym0,hm2=rw*dt+h;
        float x2=xm2*sfire_brand_metric(mx,x1,y1,nx,ml,mb),y2=ym2*sfire_brand_metric(my,x1,y1,nx,ml,mb);
        if(fabsf(x2-x1)>2.0f||fabsf(y2-y1)>2.0f){out[q]=out[n+q]=out[2*n+q]=0.0f;status[q]=1;return;}
        bool outside=(int)floorf(x2-0.5f)<ihs||(int)floorf(x2-0.5f)+1>ihe||
                     (int)floorf(y2-0.5f)<jhs||(int)floorf(y2-0.5f)+1>jhe;
        float z2=sfire_brand_hgt2k(outside?x1:x2,outside?y1:y2,hm2,z,nx,ny,nz,ml,mb,valid);
        if(!valid){out[q]=out[n+q]=out[2*n+q]=0.0f;status[q]=1;return;}
        xout=(x1+x2)*0.5f;yout=(y1+y2)*0.5f;hout=(hm1+hm2)*0.5f;zout=(z1+z2)*0.5f;wp=(rw+wp)*0.5f;
    }
    sfire_brand b=sfire_brand_read(prop,n,q);
    if(!(hout>0.0f)||!isfinite(hout)) {
        bool invalid=!isfinite(hout);
        sfire_brand zero={0.0f,0.0f,0.0f,0.0f,0.0f};sfire_brand_write(prop,n,q,zero);
        hout=0.0f;
        if(invalid)xout=yout=0.0f;
    } else if(life[q]>=momentum) {
        float lp,lt,ld;
        sfire_brand_met(xout,yout,zout,p,t,d,nx,ny,ml,mb,ihs,ihe,jhs,jhe,lp,lt,ld);
        sfire_brand_burnout(b,dt,lp,ld,lt,wp,density,density_char);
        bool burnt=!isfinite(b.mass)||!isfinite(b.diam)||!isfinite(b.effd)||!isfinite(b.temp)||
                    b.mass<=0.0f||b.diam<=0.0f||b.effd<=0.0f;
        if(burnt && !original_cleanup) {
            b={0.0f,0.0f,0.0f,0.0f,0.0f};xout=yout=hout=0.0f;status[q]=2;
        } else {
            sfire_brand_termvel(b,hout,dt,lp,ld,density,density_char);
            if(isnan(b.tvel)){b={0.0f,0.0f,0.0f,0.0f,0.0f};hout=0.0f;}
        }
        sfire_brand_write(prop,n,q,b);
    }
    out[q]=xout;out[n+q]=yout;out[2*n+q]=hout;
}
__device__ unsigned long long sfire_brand_rotate(unsigned long long x,int n){return (x<<n)|(x>>(64-n));}
__device__ unsigned long long sfire_brand_random(unsigned long long *s) {
    unsigned long long r=sfire_brand_rotate(s[1]*5ULL,7)*9ULL,t=s[1]<<17;
    s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=sfire_brand_rotate(s[3],45);
    return r;
}
extern "C" __global__ void sfire_spotting_release_heights(float *coordinates,float *prop,float *random,
 int points,int levels,int seed,float land) {
    if(blockIdx.x||threadIdx.x)return;
    int n=points*levels;
    if(!n)return;
    if(seed>0) {
        unsigned int words[8];
        // A small release repeats its seed pattern to supply all eight GNU
        // seed words. The original reallocates too few words and aborts.
        for(int k=0;k<8;++k) {
            int a=k%n,p=a/levels,lev=a%levels+1;
            float f=((coordinates[p]*coordinates[n+p])*float(seed))*float(lev);
            words[k]=(unsigned)__float2int_rz(f);
        }
        const unsigned long long keys[4]={0xbd0c5b6e50c2df49ULL,0xd46061cd46e1df38ULL,0xbb4f4d4ed6103544ULL,0x114a583d0756ad39ULL};
        unsigned long long s[4];
        for(int k=0;k<4;++k)s[k]=(((unsigned long long)words[6-2*k]<<32)|words[7-2*k])^keys[k];
        for(int q=0;q<n;++q){unsigned h=(unsigned)(sfire_brand_random(s)>>32);random[q]=float(h&0xffffff00u)*0x1p-32f;}
    }
    float minimum=1.0f+land;
    for(int level=1;level<levels;++level)for(int p=0;p<points;++p) {
        int q=level*points+p;
        coordinates[q]=coordinates[p];coordinates[n+q]=coordinates[n+p];
        float fraction=float(levels-level-1)*__fdiv_rn(1.0f,float(levels-1));
        float low=coordinates[2*n+p]>=minimum?minimum:1.0f;
        if(seed>0)fraction=random[q];
        coordinates[2*n+q]=(coordinates[2*n+p]-low)*fraction+low;
        sfire_brand_write(prop,n,q,sfire_brand_read(prop,n,p));
    }
}
extern "C" __global__ void sfire_spotting_generate(float *coordinates,float *prop,int *ident,
 float *release,float *rprop,const int *rlife,const int *rsrc,int *control,
 int capacity,int nrel,int imported,int source_prefix) {
    if(blockIdx.x||threadIdx.x)return;
    int active=0,count=0;
    for(int q=0;q<capacity;++q)if(ident[q]>0)++active;
    for(int q=0;q<nrel;++q)if((int)release[q]>0 && (int)release[nrel+q]>0)++count;
    control[1]=active;
    if(!count||active+count>capacity)return;
    for(int q=0;q<count;++q) {
        int p=active+q;
        if(ident[p]!=0||(int)release[q]==0||(int)release[nrel+q]==0){control[2]=1;return;}
        if((long long)control[0]+10>=2147483647LL)control[0]=0;
        ident[p]=++control[0];
        ident[capacity+p]=imported?rsrc[q]:source_prefix+ident[p];
        ident[2*capacity+p]=imported?rlife[q]:0;
        for(int k=0;k<3;++k)coordinates[k*capacity+p]=release[k*nrel+q];
        sfire_brand_write(prop,capacity,p,sfire_brand_read(rprop,nrel,q));
    }
    for(int q=0;q<3*nrel;++q)release[q]=0.0f;
    control[1]=active+count;
}
extern "C" __global__ void sfire_spotting_order(const float *a,float *out,int n,int order) {
    if(blockIdx.x||threadIdx.x||!n)return;
    int diff=n-1,best=0;
    for(int i=0;i<n;++i) {
        int rank=0;for(int j=0;j<n;++j)if(a[j]>a[i])++rank;
        int error=abs(order-rank);
        if(error<=diff)best=i;diff=min(diff,error);if(diff<=2)break;
    }
    out[0]=a[best];
}
extern "C" __global__ void sfire_spotting_prepare(const float *ph,const float *phb,const float *pp,const float *pb,
 const float *th,const float *moist,const float *al,const float *alb,const float *msfx,const float *msfy,
 const float *mu,const float *c1h,const float *c2h,const float *dnw,const float *fnm,const float *fnp,
 float *height,float *pressure,float *theta,float *density,float *mx,float *my,float *p8w,
 int ncol,int nz,int nm,int theta_m,int theta_total,float ptop,float rdx,float rdy) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=ncol)return;
    int volume=ncol*nz;
    for(int k=0;k<nz;++k) {
        int p=k*ncol+q;
        height[p]=__fdiv_rn(phb[p]+ph[p],9.80615997314453125f);
        theta[p]=density[p]=p8w[p]=0.0f;
    }
    mx[q]=rdx*msfx[q];my[q]=rdy*msfy[q];
    pressure[(nz-1)*ncol+q]=ptop;
    for(int k=nz-2;k>=0;--k) {
        int p=k*ncol+q;
        float total=0.0f;
        for(int m=0;m<nm;++m)total=total+moist[m*volume+p];
        pressure[p]=pressure[p+ncol]-((1.0f+total)*(c1h[k]*mu[q]+c2h[k]))*dnw[k];
        density[p]=__fdiv_rn(1.0f,al[p]+alb[p])*(1.0f+moist[p]);
    }
    for(int k=1;k<nz-1;++k) {
        int p=k*ncol+q;
        float thi=theta_total?th[p]:th[p]+300.0f,thl=theta_total?th[p-ncol]:th[p-ncol]+300.0f;
        if(theta_m) {
            float ratio=__fdiv_rn(461.600006103515625f,287.149993896484375f);
            thi=__fdiv_rn(thi,1.0f+ratio*moist[p]);
            thl=__fdiv_rn(thl,1.0f+ratio*moist[p-ncol]);
        }
        theta[p]=fnm[k]*thi+thl*fnp[k];
        p8w[p]=fnm[k]*(pp[p]+pb[p])+(pp[p-ncol]+pb[p-ncol])*fnp[k];
    }
    float z1=0.5f*(height[q]+height[ncol+q]),z2=0.5f*(height[ncol+q]+height[2*ncol+q]);
    float weight=z1-z2,rounded=weight*100.0f;
    int nearzero=rounded>=0.0f?(int)floorf(rounded+0.5f):(int)ceilf(rounded-0.5f);
    if(nearzero==0)weight=1.0f;
    weight=__fdiv_rn(height[q]-z2,weight);
    float thi=theta_total?th[q]:th[q]+300.0f,thn=theta_total?th[ncol+q]:th[ncol+q]+300.0f;
    if(theta_m) {
        float ratio=__fdiv_rn(461.600006103515625f,287.149993896484375f);
        thi=__fdiv_rn(thi,1.0f+ratio*moist[q]);thn=__fdiv_rn(thn,1.0f+ratio*moist[ncol+q]);
    }
    theta[q]=weight*thi+(1.0f-weight)*thn;
    p8w[q]=weight*(pp[q]+pb[q])+(1.0f-weight)*(pp[ncol+q]+pb[ncol+q]);
    float ground=height[q];for(int k=0;k<nz;++k)height[k*ncol+q]=height[k*ncol+q]-ground;
}
extern "C" __global__ void sfire_spotting_pack_atmosphere(const float *u,const float *v,const float *w,
 const float *ph,const float *phb,const float *pp,const float *pb,const float *thp,const float *thb,
 const float *moist,const float *al,const float *alb,const float *msf,const float *mup,const float *mub,
 float *ou,float *ov,float *ow,float *oph,float *ophb,float *op,float *opb,float *oth,float *omoist,
 float *oal,float *oalb,float *omsf,float *omu,int nx,int ny,int nz,int nm,int profile) {
    int q=blockDim.x*blockIdx.x+threadIdx.x,px=nx+8,py=ny+8,plane=px*py;
    if(q>=(nz+1)*plane)return;
    int k=q/plane,h=q%plane,i=h%px-4,j=h/px-4;
    int x=min(max(i,0),nx-1),y=min(max(j,0),ny-1),p=y*nx+x,n=nx*ny,r=k*n+p;
    ow[q]=w[r];oph[q]=ph[r];ophb[q]=phb[(profile&1)?k:r];
    ou[q]=ov[q]=op[q]=opb[q]=oth[q]=oal[q]=oalb[q]=0.0f;
    if(k<nz) {
        ou[q]=u[k*ny*(nx+1)+y*(nx+1)+min(max(i,0),nx)];
        ov[q]=v[k*(ny+1)*nx+min(max(j,0),ny)*nx+x];
        op[q]=pp[r];opb[q]=pb[(profile&2)?k:r];
        oth[q]=thp[r]+thb[(profile&4)?k:r];
        oal[q]=al[r];oalb[q]=alb[(profile&8)?k:r];
    }
    for(int m=0;m<nm;++m)omoist[m*(nz+1)*plane+q]=k<nz?moist[m*nz*n+r]:0.0f;
    if(k==0){omsf[h]=msf[p];omu[h]=mup[p]+mub[p];}
}
__device__ int sfire_brand_fuel(int c,int crosswalk) {
    if(c>=1 && c<=13)return c;
    if(!crosswalk) {
        if(c>=101&&c<=109)return c-101+15;if(c>=121&&c<=124)return c-121+24;
        if(c>=141&&c<=149)return c-141+28;if(c>=161&&c<=165)return c-161+37;
        if(c>=181&&c<=189)return c-181+42;if(c>=201&&c<=204)return c-201+51;
    } else {
        if(c==101||c==104||c==107)return 1;if(c==102||(c>=121&&c<=124))return 2;
        if(c==103||c==105||c==106||c==108||c==109)return 3;
        if(c==145||c==147)return 4;if(c==142)return 5;if(c==141||c==146)return 6;
        if(c==143||c==144||c==148||c==149)return 7;
        if(c==181||c==183||c==184||c==187)return 8;
        if(c==182||c==186||c==188||c==189)return 9;
        if(c>=161&&c<=165)return 10;if(c==185||c==201)return 11;
        if(c==202)return 12;if(c==203||c==204)return 13;
    }
    return 14;
}
__device__ float sfire_brand_order_masked(const float *a,int n,float lo,int count,int order) {
    int diff=count-1,best=-1;
    for(int i=0;i<n;++i)if(a[i]>lo) {
        int rank=0;for(int j=0;j<n;++j)if(a[j]>lo&&a[j]>a[i])++rank;
        int error=abs(order-rank);
        if(error<=diff)best=i;diff=min(diff,error);if(diff<=2)break;
    }
    return best<0?0.0f:a[best];
}
extern "C" __global__ void sfire_spotting_select(const float *burn,const float *fgi,float *potential,
 float *release,float *prop,int *gen,int *control,int fx,int fy,int ax,int ay,int sr_x,int sr_y,
 int limit,int levels,int capacity,int fgi_stride,float maxheight,float diam,float effd,float temp,float tvel,float density) {
    if(blockIdx.x||threadIdx.x)return;
    int n=fx*fy,aw=ax+8;
    for(int q=0;q<n;++q)potential[q]=(1.0f*burn[q])*fgi[(q/fx)*fgi_stride+q%fx];
    float low=1.0e-6f;int count=0;
    for(int q=0;q<n;++q)if(potential[q]>low)++count;
    for(int step=0;step<2 && count>limit;++step) {
        low=(float)(10.0*(double)low);count=0;
        for(int q=0;q<n;++q)if(potential[q]>low)++count;
    }
    float median=0.0f;
    if(count>limit) {
        low=sfire_brand_order_masked(potential,n,low,count,limit);count=0;
        for(int q=0;q<n;++q)if(potential[q]>low)++count;
        if(count>0)median=sfire_brand_order_masked(potential,n,low,count,__float2int_rz(__fdiv_rn(float(count),2.0f)));
    } else if(count>0)median=sfire_brand_order_masked(potential,n,low,count,__float2int_rz(0.5f*float(count)));
    float threshold=0.0f;
    if(count>0) {
        float mean=__fdiv_rn(median*float(count)+0.0f,float(count));
        if(count>limit){float ratio=__fdiv_rn(float(limit),float(count));threshold=__fdiv_rn(1.0f-ratio,ratio)*mean;}
    }
    int points=0;
    for(int q=0;q<n;++q)if(potential[q]>low&&(threshold==0.0f||potential[q]>=threshold)) {
        int i=q%fx+1,j=q/fx+1;
        float x=1.0f+__fdiv_rn(float(i)-1.0f,float(sr_x));x=x+__fdiv_rn(0.5f,float(sr_x));
        float y=1.0f+__fdiv_rn(float(j)-1.0f,float(sr_y));y=y+__fdiv_rn(0.5f,float(sr_y));
        if(capacity>0) {
            release[points]=x;release[capacity+points]=y;release[2*capacity+points]=maxheight;
            sfire_brand_write(prop,capacity,points,sfire_brand_property(diam,effd,temp,tvel,density));
            ++gen[((int)y+3)*aw+(int)x+3];
        }
        ++points;
    }
    control[0]=points;control[1]=count;
}
extern "C" __global__ void sfire_spotting_copy_plane(const float *source,float *dest,
 int nx,int ny,int stride,int x0,int y0) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=nx*ny)return;
    dest[q]=source[(q/nx+y0)*stride+q%nx+x0];
}
extern "C" __global__ void sfire_spotting_remove_deposit(float *coordinates,float *prop,int *ident,
 const int *status,float *delta,int capacity,int ax,int ay,int maxlife,float land) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=capacity||ident[q]<=0)return;
    int age=++ident[2*capacity+q];
    float x=coordinates[q],y=coordinates[capacity+q],h=coordinates[2*capacity+q];
    sfire_brand p=sfire_brand_read(prop,capacity,q);
    if(isnan(p.mass)||isnan(p.diam)||isnan(p.effd)||isnan(p.temp)||isnan(p.tvel))prop[2*capacity+q]=0.0f;
    bool remove=status[q]!=0 ||(int)floorf(x-0.5f)<1||(int)floorf(y-0.5f)<1||
       (int)floorf(x-0.5f)+1>ax+1||(int)floorf(y-0.5f)+1>ay+1||age>maxlife||
       (prop[2*capacity+q]<=1.1754943508222875e-38f&&h>land);
    if(remove){ident[q]=0;return;}
    int i=(int)floorf(x),j=(int)floorf(y);
    if(i>=1&&i<=ax&&j>=1&&j<=ay&&h<=land) {
        atomicAdd(delta+(j+3)*(ax+8)+i+3,1.0f);ident[q]=0;return;
    }
    if(i<1||i>ax||j<1||j>ay)ident[q]=0;
}
extern "C" __global__ void sfire_spotting_compact(float *coordinates,float *prop,int *ident,int *control,int capacity) {
    if(blockIdx.x||threadIdx.x)return;
    int active=0;
    for(int q=0;q<capacity;++q)if(ident[q]>0) {
        if(active!=q) {
            for(int k=0;k<3;++k){coordinates[k*capacity+active]=coordinates[k*capacity+q];ident[k*capacity+active]=ident[k*capacity+q];}
            for(int k=0;k<5;++k)prop[k*capacity+active]=prop[k*capacity+q];
        }
        ++active;
    }
    for(int q=active;q<capacity;++q) {
        for(int k=0;k<3;++k){coordinates[k*capacity+q]=0.0f;ident[k*capacity+q]=0;}
        for(int k=0;k<5;++k)prop[k*capacity+q]=0.0f;
    }
    control[1]=active;
}
extern "C" __global__ void sfire_spotting_add_counts(float *all,float *hist,const float *delta,int n) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q<n){all[q]=all[q]+delta[q];hist[q]=hist[q]+delta[q];}
}
extern "C" __global__ void sfire_spotting_history(const float *fine_area,const float *fgi,const float *fmc,
 const float *cat,float *area,float *risk,float *hist,int *mask,float *fraction,float *likelihood,
 int ax,int ay,int rx,int ry,int crosswalk,int area_stride,int fgi_stride,int fmc_stride,int cat_stride) {
    if(blockIdx.x||threadIdx.x)return;
    int aw=ax+8,ah=ay+8,fx=ax*rx;
    for(int q=0;q<aw*ah;++q){fraction[q]=likelihood[q]=0.0f;}
    int landed=0;
    for(int j=0;j<ay;++j)for(int i=0;i<ax;++i) {
        int p=(j+4)*aw+i+4;float sum=0.0f;
        for(int y=0;y<ry;++y)for(int x=0;x<rx;++x)sum=sum+fine_area[(j*ry+y)*area_stride+i*rx+x];
        area[p]=sum;
        if(hist[p]>0.0f)++landed;
    }
    if(!landed)return;
    landed=0;
    for(int j=0;j<ay;++j)for(int i=0;i<ax;++i) {
        int p=(j+4)*aw+i+4;bool keep=hist[p]>0.0f&&area[p]==0.0f;
        mask[p]=keep?1:0;if(!keep)hist[p]=0.0f;else ++landed;
    }
    if(!landed)return;
    for(int q=0;q<aw*ah;++q)risk[q]=0.0f;
    int total=0;float maximum=0.0f;
    for(int j=0;j<ay;++j)for(int i=0;i<ax;++i) {
        int p=(j+4)*aw+i+4;float sum=0.0f;
        for(int y=0;y<ry;++y)for(int x=0;x<rx;++x) {
            int fy=j*ry+y,fi=i*rx+x,c=sfire_brand_fuel((int)cat[fy*cat_stride+fi],crosswalk);
            float value=c==14?0.0f:1.0f-fminf(1.0f,__fdiv_rn(fmc[fy*fmc_stride+fi],fgi[fy*fgi_stride+fi]));
            sum=sum+value;
        }
        risk[p]=sum;if(mask[p])total+=(int)hist[p];
    }
    for(int j=0;j<ay;++j)for(int i=0;i<ax;++i) {
        int p=(j+4)*aw+i+4;if(!mask[p])continue;
        fraction[p]=__fdiv_rn(float((int)hist[p]),float(total));
        likelihood[p]=fraction[p]*risk[p];maximum=fmaxf(maximum,likelihood[p]);
    }
    for(int j=0;j<ay;++j)for(int i=0;i<ax;++i){int p=(j+4)*aw+i+4;if(mask[p])likelihood[p]=maximum>0.0f?__fdiv_rn(likelihood[p],maximum):0.0f;}
}
__device__ int sfire_brand_neighbor(float x,float y,int xl,int xh,int yl,int yh) {
    int i=(int)floorf(x),j=(int)floorf(y);
    bool left=i<xl,right=i>xh,up=j>yh,down=j<yl;
    if(left&&up)return 4;if(right&&up)return 5;if(left&&down)return 6;if(right&&down)return 7;
    if(left)return 0;if(right)return 1;if(up)return 2;if(down)return 3;return -1;
}
extern "C" __global__ void sfire_spotting_pack_neighbors(const float *coordinates,const float *prop,
 const int *ident,const int *mask,const int *neighbors,float *real_packet,int *int_packet,int *counts,
 int n,int xl,int xh,int yl,int yh) {
    if(blockIdx.x||threadIdx.x)return;
    int cursor=0;
    for(int edge=0;edge<8;++edge) {
        counts[edge]=0;
        if(neighbors[edge]<0)continue;
        for(int q=0;q<n;++q)if(mask[q] && sfire_brand_neighbor(coordinates[q],coordinates[n+q],xl,xh,yl,yh)==edge) {
            for(int k=0;k<3;++k){real_packet[k*n+cursor]=coordinates[k*n+q];int_packet[k*n+cursor]=ident[k*n+q];}
            for(int k=0;k<5;++k)real_packet[(k+3)*n+cursor]=prop[k*n+q];
            ++cursor;++counts[edge];
        }
    }
}
extern "C" __global__ void sfire_spotting_particle_blocks(const float *coordinates,int *groups,
 int n,int nx,int ny,int bx,int by) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=n)return;
    int i=min(max((int)floorf(coordinates[q])-1,0),nx-1);
    int j=min(max((int)floorf(coordinates[n+q])-1,0),ny-1);
    groups[q]=(j/by)*((nx+bx-1)/bx)+i/bx;
}
