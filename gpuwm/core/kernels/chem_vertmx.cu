// WRF chem/module_vertmx_wrf.F:6-198, dry_dep_driver.F:681-803.
//
// One thread per column.  Only the Thomas solver's two recurrences (q, f)
// are kept per level in the thread's frame; every other column quantity --
// the floored mixing ratio, dry density, the heights, ekmfull with its
// floors, the coeffs and the rlhside row -- is recomputed from the global
// inputs where it is used, by the same float32 operations in the same order,
// so each word is the one the fully materialized form gave (the oracle parity
// tests hold the kernel to WRF bit for bit).  Materializing twelve nz-long
// arrays made the frame 2,144 B at nz = 59, a 292 MiB launch-time local
// memory reservation on an RTX 5090; two arrays keep it under the default
// per-thread stack.
// The launcher specializes CHEM_NZ to the column's level count
// (gpuwm/core/chem_vertmx.py); the unspecialized bound is what a standalone
// compile of this file reads, so its frame is the ceiling the preflight's
// level-specialized row is checked against.
#ifndef CHEM_NZ
#define CHEM_NZ 256
#endif
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))

struct VertmxColumn {
    const float *chem, *alt, *zw, *z, *exch;
    int c, n, nz;
    float floor;
    // ekmfull floors (dry_dep_driver.F:701-727): anthropogenic CO > 0 raises
    // kts+1:kts+10 to 1; CO > 200 or a PM2.5 pair/single above 8.19e-4*200
    // (all gated on CO presence and sf_urban_physics == 0), or fire CO > 0,
    // raise kts+1:kte/2 to 2.
    bool raise1, raise2;

    __device__ float phi(int k) const {
        return fmaxf(floor, chem[k * n + c]);
    }
    __device__ float rho(int k) const {
        return D(1.f, alt[k * n + c]);
    }
    __device__ float zf(int k) const {          // interface heights, 0..nz
        return S(zw[k * n + c], zw[c]);
    }
    __device__ float zz(int k) const {          // mass-level heights
        return S(z[k * n + c], zw[c]);
    }
    __device__ float ek(int k) const {          // ekmfull, 0..nz
        if (k == 0 || k == nz) return 0.f;
        float e = fmaxf(1.e-6f, exch[k * n + c]);
        if (raise1 && k <= 10) e = fmaxf(e, 1.f);
        if (raise2 && k < nz / 2) e = fmaxf(e, 2.f);
        return e;
    }
    __device__ float a(int k) const {           // coeffs' a_coeff, k >= 1
        return D(M(.5f, A(rho(k), rho(k - 1))), S(zz(k), zz(k - 1)));
    }
    __device__ float b(int k) const {           // coeffs' b_coeff
        return D(1.f, M(rho(k), S(zf(k + 1), zf(k))));
    }
};

extern "C" __global__ void chem_vertmx(
 float *chem,const float *alt,const float *zw,const float *z,const float *dz,
 const float *exch,const float *vd,float dt,float floor,int gas,float *accum,
 const float *anth,const float *fire,const float *pm1,const float *pm2,const float *pm,
 int ha,int hf,int hp,int hs,int urban,int nz,int n,
 float *mixed_out,float *ek_out,float *dd_out,int debug) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=n)return;
 float f[CHEM_NZ],q[CHEM_NZ];
 VertmxColumn col;
 col.chem=chem;col.alt=alt;col.zw=zw;col.z=z;col.exch=exch;
 col.c=c;col.n=n;col.nz=nz;col.floor=floor;
 col.raise1=false;col.raise2=false;
 if(ha && urban==0) {
  if(anth[c]>0.f)col.raise1=true;
  if(anth[c]>200.f || (hp && A(pm1[c],pm2[c])>M(8.19e-4f,200.f)) ||
                         (hs && pm[c]>M(8.19e-4f,200.f)))
   col.raise2=true;
 }
 if(hf && fire[c]>0.f)col.raise2=true;
 // rlhside and the tridiag forward sweep, one row at a time.
 for(int k=0;k<nz;k++) {
  float a1=k?M(col.a(k),col.ek(k)):0.f;
  float a2=k<nz-1?M(col.a(k+1),col.ek(k+1)):0.f;
  float inv=D(1.f,M(dt,col.b(k)));
  float rk=col.rho(k);
  float loss=k==0?A(M(vd[c],rk),a2):(k==nz-1?a1:A(a1,a2));
  float pk=col.phi(k);
  float center=M(S(inv,M(.25f,loss)),pk);
  float gain=k==0?M(.25f,M(a2,col.phi(k+1))):
             (k==nz-1?M(.25f,M(a1,col.phi(k-1))):
              M(.25f,A(M(a1,col.phi(k-1)),M(a2,col.phi(k+1)))));
  float rhs=A(center,gain);
  float l1=k?M(-.75f,a1):0.f;
  float l2=A(inv,M(.75f,loss));
  float l3=k<nz-1?M(-.75f,a2):0.f;
  if(k==0) {
   q[0]=D(-l3,l2);f[0]=D(rhs,l2);
  } else {
   float p=D(1.f,A(l2,M(l1,q[k-1])));
   q[k]=M(-l3,p);f[k]=M(S(rhs,M(l1,f[k-1])),p);
  }
 }
 for(int k=nz-2;k>=0;k--)f[k]=A(f[k],M(q[k],f[k+1]));
 float old=0.f,newmass=0.f;
 for(int k=0;k<nz-1;k++) {
  float rk=col.rho(k);
  float fac=gas?M(D(M(M(1.e-6f,rk),1.f),M(28.966f,1.e-3f)),dz[k*n+c]):M(rk,dz[k*n+c]);
  old=A(old,M(col.phi(k),fac));newmass=A(newmass,M(fmaxf(floor,f[k]),fac));
  chem[k*n+c]=fmaxf(floor,f[k]);
 }
 float dd=fmaxf(0.f,S(old,newmass));accum[c]=A(accum[c],dd);
 if(debug) {
  for(int k=0;k<nz;k++) {mixed_out[k*n+c]=f[k];ek_out[k*n+c]=col.ek(k);}
  ek_out[nz*n+c]=col.ek(nz);dd_out[c]=dd;
 }
}
#undef A
#undef S
#undef M
#undef D
