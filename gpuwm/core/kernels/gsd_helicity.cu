// cal_helicity in NOAA-EMC/HRRR 40ee6058 module_diffusion_em.F:6068-6816.
// Terrain/map metric helpers are shared with the stock transcription.
extern "C" __global__ void gsd_vorticity(
 const real* u,const real* v,const real* ph,const real* phb,int phb3d,
 const real* msfu,const real* msfv,const real* dn,const real* dnw,
 const real* fnm,const real* fnp,real cf1,real cf2,real cf3,
 real rdx,real rdy,real* rv,int nz,int ny,int nx) {
 int idx=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny; if(idx>=nz*n)return;
 int k=idx/n,c=idx-k*n,cy=c/nx,cx=c%nx;
 // Boundary values are the inward corner's value, as the fork copies.
 cy=max(cy,1);cx=max(cx,1);
 real cft2=-__fdiv_rn(__fmul_rn(.5f,dnw[nz-1]),dn[nz-1]);
 real cft1=__fsub_rn(1.f,cft2);
 real mm=__fmul_rn(__fmul_rn(.25f,__fadd_rn(msfu[(cy-1)*(nx+1)+cx],msfu[cy*(nx+1)+cx])),
                  __fadd_rn(msfv[cy*nx+cx-1],msfv[cy*nx+cx]));
 rv[idx]=uh_rvort(u,v,ph,phb,phb3d,msfu,msfv,fnm,fnp,cf1,cf2,cf3,
                 cft1,cft2,rdx,rdy,mm,k,cy,cx,nz,ny,nx);
}
extern "C" __global__ void gsd_uh_columns(
 const real* w,const real* ph,const real* phb,int phb3d,const real* ht,
 const real* rv,real* uh,real* usecol,real lower,real upper,int nz,int ny,int nx) {
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny;if(c>=n)return;
 int cy=c/nx,cx=c%nx;real sum=0.f,use=1.f;
 if(cx==0||cy==0){uh[c]=0.f;usecol[c]=1.f;return;}
 real zl=uh_agl(ph,phb,phb3d,ht,0,cy,cx,ny,nx);
 for(int k=0;k<nz;k++){
  real zu=uh_agl(ph,phb,phb3d,ht,k+1,cy,cx,ny,nx);
  if(zl>=lower&&zu<=upper){
   real wa=uh_wavg(w,k,cy,cx,ny,nx);
   real wb=k+1<nz?uh_wavg(w,k+1,cy,cx,ny,nx):0.f;
   real rb=k+1<nz?rv[(k+1)*n+c]:0.f;
   if(wa>0.f&&wb>0.f){
    real pair=__fmul_rn(__fadd_rn(__fmul_rn(wa,rv[k*n+c]),__fmul_rn(wb,rb)),.5f);
    sum=__fadd_rn(sum,__fmul_rn(pair,__fsub_rn(zu,zl)));
   }else{use=0.f;sum=0.f;}
  }
  zl=zu;
 }
 uh[c]=sum;usecol[c]=use;
}
extern "C" __global__ void gsd_uh_fold(const real* uh,const real* use,
 real* maximum,real* minimum,int ny,int nx){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nx*ny)return;
 int y=c/nx,x=c%nx;if(x<1||x>nx-2||y<1||y>ny-2)return;
 real edge=__fadd_rn(uh[c+1],uh[c-1]);
 edge=__fadd_rn(edge,uh[c+nx]);edge=__fadd_rn(edge,uh[c-nx]);
 real corner=__fadd_rn(uh[c+nx+1],uh[c-nx+1]);
 corner=__fadd_rn(corner,uh[c+nx-1]);corner=__fadd_rn(corner,uh[c-nx-1]);
 real val=__fadd_rn(__fadd_rn(__fmul_rn(.25f,uh[c]),__fmul_rn(.125f,edge)),__fmul_rn(.0625f,corner));
 if(use[c]!=0.f){if(val>maximum[c])maximum[c]=val;if(val<minimum[c])minimum[c]=val;}
}
extern "C" __global__ void gsd_vorticity_fold(const real* rv,const real* ph,
 const real* phb,int phb3d,const real* ht,real* maximum,real* max01,
 real* total_steps,int idump,int nz,int ny,int nx){
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny;if(c>=n)return;
 int y=c/nx,x=c%nx;
 // cal_helicity uses the integer timestep modulo INT(reset*60/dt),
 // even when adaptive dt changes. This is separate from the hourly
 // curr_secs reset in module_diag_gsd, which follows this call.
 real old=((int)total_steps[c]%idump==0)?0.f:maximum[c];
 total_steps[c]=__fadd_rn(total_steps[c],1.f);
 // Preserve the fork's overwrite ordering: compare every level with the
 // PREVIOUS timestep's 0-2 km value, including for the 0-1 km field.
 for(int k=1;k<nz;k++){
  real z=__fsub_rn(__fmul_rn(.5f,__fadd_rn(uh_z_at_w(ph,phb,phb3d,k,y,x,ny,nx),
                 uh_z_at_w(ph,phb,phb3d,k+1,y,x,ny,nx))),ht[c]);
  if(z<2001.f)maximum[c]=fmaxf(rv[k*n+c],old);
  if(z<1001.f)max01[c]=fmaxf(rv[k*n+c],old);
 }
}
