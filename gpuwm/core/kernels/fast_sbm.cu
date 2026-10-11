// FAST_SBM tile and column launches from the shared host/device driver.
// One block per column, one active thread. All neighbor inputs are frozen
// by the preparation launch before the column launch begins.
#include "glibc_flt32.cuh"
#include "glibc_flt64.cuh"
#include "fast_sbm_common.cuh"
#include "fast_sbm_coll.cuh"
#include "fast_sbm_aux.cuh"
#include "fast_sbm_nucl.cuh"
#include "fast_sbm_core.cuh"
#include "fast_sbm_column.cuh"
#include "fast_sbm_driver.cuh"

template <class R>
__device__ void fsbm_launch(int itimestep, R dt, R dx, R dy, const int* dims,
 const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, R* work, int* kz, double* tally,
 int* status, int mode) {
 FsbmTab<R> T;
 T.xl = reinterpret_cast<const R*>(tp[0]);
 T.xi = reinterpret_cast<const R*>(tp[1]);
 T.xs = reinterpret_cast<const R*>(tp[2]);
 T.xg = reinterpret_cast<const R*>(tp[3]);
 T.xh = reinterpret_cast<const R*>(tp[4]);
 T.vr1 = reinterpret_cast<const R*>(tp[5]);
 T.vr2 = reinterpret_cast<const R*>(tp[6]);
 T.vr3 = reinterpret_cast<const R*>(tp[7]);
 T.vr4 = reinterpret_cast<const R*>(tp[8]);
 T.vr5 = reinterpret_cast<const R*>(tp[9]);
 T.rlec = reinterpret_cast<const R*>(tp[10]);
 T.riec = reinterpret_cast<const R*>(tp[11]);
 T.rsec = reinterpret_cast<const R*>(tp[12]);
 T.rgec = reinterpret_cast<const R*>(tp[13]);
 T.rhec = reinterpret_cast<const R*>(tp[14]);
 T.ro1bl = reinterpret_cast<const R*>(tp[15]);
 T.ro2bl = reinterpret_cast<const R*>(tp[16]);
 T.ro3bl = reinterpret_cast<const R*>(tp[17]);
 T.ro4bl = reinterpret_cast<const R*>(tp[18]);
 T.ro5bl = reinterpret_cast<const R*>(tp[19]);
 T.ywll = reinterpret_cast<const R*>(tp[20]);
 T.ywls = reinterpret_cast<const R*>(tp[21]);
 T.ywlg = reinterpret_cast<const R*>(tp[22]);
 T.ywlh = reinterpret_cast<const R*>(tp[23]);
 T.ywss = reinterpret_cast<const R*>(tp[24]);
 T.pkij = reinterpret_cast<const R*>(tp[25]);
 T.qkj = reinterpret_cast<const R*>(tp[26]);
 T.xl_mg = reinterpret_cast<const double*>(tp[27]);
 T.xs_mg = reinterpret_cast<const double*>(tp[28]);
 T.xg_mg = reinterpret_cast<const double*>(tp[29]);
 T.xh_mg = reinterpret_cast<const double*>(tp[30]);
 T.ima = reinterpret_cast<const int*>(tp[31]);
 T.chucm = reinterpret_cast<const double*>(tp[32]);
 T.ecoalmassm = reinterpret_cast<const double*>(tp[33]);
 T.prob = reinterpret_cast<const double*>(tp[34]);
 T.gain_var_new = reinterpret_cast<const double*>(tp[35]);
 T.dropradii = reinterpret_cast<const R*>(tp[36]);
 T.fccnr_con = reinterpret_cast<const double*>(tp[37]);
 T.fccnr_mar = reinterpret_cast<const double*>(tp[38]);
 T.rccn = reinterpret_cast<const R*>(tp[39]);
 T.ikr_spon_break = ikr;
 FsbmFlags F;
 F.hail = flags[0];
 F.rainnc_own_column = flags[1];
 F.rainnc_bottom_outflow = flags[2];
 F.nucleation_water_sink = flags[3];
 F.dtcond_carry = flags[4];
 F.breakup_retry_cap = flags[5];
 F.cloud_base_first_level = flags[6];
 F.fall_speed_own_bin = flags[7];
 F.snow_clip_qi_plus_qs = flags[8];
 F.melt_exponential = flags[9];
 F.fusion_heat_334 = flags[10];
 F.homfrz_any_saturation = flags[11];
 F.dsupice_w_own_point = flags[12];
 F.dsupice_wind_own_point = flags[13];
 F.supsat_rederived = flags[14];
 F.sedimentation_cell_depth = flags[15];
 F.collision_mass_conserved = flags[16];
 State<R> S = {reinterpret_cast<R*>(ap[0]), reinterpret_cast<R*>(ap[1]), reinterpret_cast<R*>(ap[2]), reinterpret_cast<R*>(ap[3]), reinterpret_cast<R*>(ap[4]), reinterpret_cast<R*>(ap[5]), reinterpret_cast<R*>(ap[6]), reinterpret_cast<R*>(ap[7]), reinterpret_cast<R*>(ap[8]), reinterpret_cast<R*>(ap[9]), reinterpret_cast<R*>(ap[10]), reinterpret_cast<R*>(ap[11]), reinterpret_cast<R*>(ap[12]), reinterpret_cast<R*>(ap[13]), reinterpret_cast<R*>(ap[14]), reinterpret_cast<R*>(ap[15]), reinterpret_cast<R*>(ap[16]), reinterpret_cast<R*>(ap[17]), reinterpret_cast<R*>(ap[18]), reinterpret_cast<R*>(ap[19]), reinterpret_cast<R*>(ap[20]), reinterpret_cast<R*>(ap[21]), reinterpret_cast<R*>(ap[22]), reinterpret_cast<R*>(ap[23]), reinterpret_cast<R*>(ap[24]), reinterpret_cast<R*>(ap[25]), reinterpret_cast<R*>(ap[26]), reinterpret_cast<R*>(ap[27]), reinterpret_cast<R*>(ap[28]), reinterpret_cast<R*>(ap[29])};
 long n = (long)(dims[7]-dims[6]+1)*(dims[11]-dims[10]+1)*(dims[9]-dims[8]+1);
 int nz = dims[17]-dims[16]+1;
 int nc = (dims[13]-dims[12]+1)*(dims[15]-dims[14]+1);
 // One column per THREAD: the column-local modes (1, 3, 4) may be launched
 // with several threads per block (fast_sbm_cuda.COLUMN_BLOCK); one thread
 // per block left 31 of every warp's 32 lanes idle. Each column's arithmetic
 // is unchanged, so the result is byte-identical to the one-thread launch.
 const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
 // Preparation is column-local. Freeze the halo fields first, then
 // convert only each active column's spectrum. Separate launches make
 // every neighbor read follow the completed halo write, without races.
 if (mode == 3 || mode == 4) {
   long ni=dims[7]-dims[6]+1, nk=dims[11]-dims[10]+1;
   auto ix=[&](int i,int k,int j)->long {
     return (long)(i-dims[6])+(long)(k-dims[10])*ni+(long)(j-dims[8])*ni*nk;
   };
   int i,j;
   if (mode == 3) {
     int lo_i=dims[12]-1>1?dims[12]-1:1;
     int lo_j=dims[14]-1>1?dims[14]-1:1;
     int hi_i=dims[1]-1<dims[13]+1?dims[1]-1:dims[13]+1;
     int hi_j=dims[3]-1<dims[15]+1?dims[3]-1:dims[15]+1;
     int width=hi_i-lo_i+1;
     if (gid >= (long)width*(hi_j-lo_j+1)) return;
     i=lo_i+(int)(gid%width); j=lo_j+(int)(gid/width);
     R z_full=(R)0.;
     for (int k=dims[16];k<=dims[17];++k) {
       long o=ix(i,k,j);
       work[o]=S.p[o]*(R)10.;
       work[n+o]=S.rho[o]*(R)0.001;
       work[2*n+o]=z_full+(R)0.5*S.dz8w[o]*(R)100;
       z_full=z_full+S.dz8w[o]*(R)100.;
       R tho=itimestep==1?S.th[o]:S.th_old[o];
       work[5*n+o]=itimestep==1?S.qv[o]:S.qv_old[o];
       work[3*n+o]=S.th[o]*S.pii[o];
       work[4*n+o]=tho*S.pii[o];
     }
   } else {
     if (gid >= nc) return;
     i=dims[12]+(int)(gid%(dims[13]-dims[12]+1));
     j=dims[14]+(int)(gid/(dims[13]-dims[12]+1));
     for (int k=dims[16];k<=dims[17];++k) {
       long o=ix(i,k,j);
       fsbm_entry_point<R>(T,F.hail,S.chem+n+o,n,S.rho[o]);
       if (itimestep==1)
         fsbm_aerosol_init_point<R>(T,S.chem+n+o,n,work[n+o],
                                   work[n+ix(i,1,j)],
                                   S.xland[(long)(i-dims[6])+(long)(j-dims[8])*ni]);
     }
   }
   return;
 }
 if (mode == 1 && gid >= nc) return;
 int col = mode == 1 ? (int)gid : 0;
 int d[18]; for (int j=0;j<18;++j) d[j]=dims[j];
 if (mode == 1) {
   int i = dims[12] + col % (dims[13]-dims[12]+1);
   int j = dims[14] + col / (dims[13]-dims[12]+1);
   d[12]=d[13]=i; d[14]=d[15]=j;
 }
 R* sub = work + 7*n + 2*col*(nz+2);
 FsbmScratch<R> X = {work, work+n, work+2*n, work+3*n, work+4*n,
                     work+5*n, work+6*n, sub, sub+nz+2, kz+col};
 double* out = tally + 17*col;
 status[col] = fsbm_tile<R>(T,F,itimestep,dt,dx,dy,d,S,X,out,mode);
}
extern "C" __global__ void fsbm_tile_f32(int step, float dt, float dx, float dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, float* work, int* kz, double* tally, int* status) {
 if (threadIdx.x != 0) return;
 fsbm_launch<float>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,0);
}
extern "C" __global__ void fsbm_prepare_f32(int step, float dt, float dx, float dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, float* work, int* kz, double* tally, int* status) {
 if (threadIdx.x != 0) return;
 fsbm_launch<float>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,2);
}
extern "C" __global__ void fsbm_columns_f32(int step, float dt, float dx, float dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, float* work, int* kz, double* tally, int* status) {
 fsbm_launch<float>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,1);
}
extern "C" __global__ void fsbm_tile_f64(int step, double dt, double dx, double dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, double* work, int* kz, double* tally, int* status) {
 if (threadIdx.x != 0) return;
 fsbm_launch<double>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,0);
}
extern "C" __global__ void fsbm_prepare_f64(int step, double dt, double dx, double dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, double* work, int* kz, double* tally, int* status) {
 if (threadIdx.x != 0) return;
 fsbm_launch<double>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,2);
}
extern "C" __global__ void fsbm_columns_f64(int step, double dt, double dx, double dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, double* work, int* kz, double* tally, int* status) {
 fsbm_launch<double>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,1);
}

extern "C" __global__ void fsbm_grid_fields_f32(int step, float dt, float dx, float dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, float* work, int* kz, double* tally, int* status) {
 fsbm_launch<float>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,3);
}

extern "C" __global__ void fsbm_prepare_columns_f32(int step, float dt, float dx, float dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, float* work, int* kz, double* tally, int* status) {
 fsbm_launch<float>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,4);
}

extern "C" __global__ void fsbm_grid_fields_f64(int step, double dt, double dx, double dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, double* work, int* kz, double* tally, int* status) {
 fsbm_launch<double>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,3);
}

extern "C" __global__ void fsbm_prepare_columns_f64(int step, double dt, double dx, double dy,
 const int* dims, const unsigned long long* tp, int ikr, const int* flags,
 const unsigned long long* ap, double* work, int* kz, double* tally, int* status) {
 fsbm_launch<double>(step,dt,dx,dy,dims,tp,ikr,flags,ap,work,kz,tally,status,4);
}
