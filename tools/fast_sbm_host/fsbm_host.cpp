// Host build of WOOF's WRF mp_physics = 30 port: the same headers the
// device unit compiles (gpuwm/core/kernels/fast_sbm_*.cuh), compiled by a
// host C++ compiler against the platform libm, with a driver that runs
// FAST_SBM over a WRF tile in WRF's own memory layout and loop order.
//
// This is the float64 (and float32) verification path: it is graded
// against WRF's Fortran (tests/data/fast_sbm_oracle) and is never part of
// a forecast.  Build: tools/fast_sbm_host/build.sh.  C ABI below.

#include <cstdlib>
#include <cstring>
#include <vector>

#include "fast_sbm_common.cuh"
#include "fast_sbm_coll.cuh"
#include "fast_sbm_aux.cuh"
#include "fast_sbm_nucl.cuh"
#include "fast_sbm_core.cuh"
#include "fast_sbm_column.cuh"
#include "fast_sbm_init.cuh"
#include "fast_sbm_driver.cuh"

namespace {

// The 27 tables as read, in this order (the C ABI's table vector).
enum {
  TB_XL, TB_XI, TB_XS, TB_XG, TB_XH, TB_VR1, TB_VR2, TB_VR3, TB_VR4, TB_VR5,
  TB_RLEC, TB_RIEC, TB_RSEC, TB_RGEC, TB_RHEC, TB_RO1BL, TB_RO2BL, TB_RO3BL, TB_RO4BL, TB_RO5BL,
  TB_YWLL, TB_YWLS, TB_YWLG, TB_YWLH, TB_YWSS, TB_PKIJ, TB_QKJ, TB_N
};
const int tb_size[TB_N] = {33, 99, 33, 33, 33, 33, 99, 33, 33, 33, 33, 99, 33, 33, 33,
                           33, 99, 33, 33, 33, 3267, 3267, 3267, 3267, 3267, 5832, 324};

template <class R> struct Handle {
  std::vector<R> tab[TB_N];
  FsbmDerivedHost<R> d;
  FsbmTab<R> t;
  FsbmFlags f;
};

template <class R> Handle<R>* make_handle(const R* const* tabs, const int* flags) {
  Handle<R>* h = new Handle<R>();
  for (int i = 0; i < TB_N; ++i) h->tab[i].assign(tabs[i], tabs[i] + tb_size[i]);
  std::memcpy(&h->f, flags, sizeof(FsbmFlags));
  fsbm_derive<R>(h->tab[TB_XL].data(), h->tab[TB_XS].data(), h->tab[TB_XG].data(),
                 h->tab[TB_XH].data(), h->tab[TB_VR1].data(), h->d);
  FsbmTab<R>& t = h->t;
  t.xl = h->tab[TB_XL].data(); t.xi = h->tab[TB_XI].data(); t.xs = h->tab[TB_XS].data();
  t.xg = h->tab[TB_XG].data(); t.xh = h->tab[TB_XH].data();
  t.vr1 = h->tab[TB_VR1].data(); t.vr2 = h->tab[TB_VR2].data(); t.vr3 = h->tab[TB_VR3].data();
  t.vr4 = h->tab[TB_VR4].data(); t.vr5 = h->tab[TB_VR5].data();
  t.rlec = h->tab[TB_RLEC].data(); t.riec = h->tab[TB_RIEC].data(); t.rsec = h->tab[TB_RSEC].data();
  t.rgec = h->tab[TB_RGEC].data(); t.rhec = h->tab[TB_RHEC].data();
  t.ro1bl = h->tab[TB_RO1BL].data(); t.ro2bl = h->tab[TB_RO2BL].data();
  t.ro3bl = h->tab[TB_RO3BL].data(); t.ro4bl = h->tab[TB_RO4BL].data();
  t.ro5bl = h->tab[TB_RO5BL].data();
  t.ywll = h->tab[TB_YWLL].data(); t.ywls = h->tab[TB_YWLS].data(); t.ywlg = h->tab[TB_YWLG].data();
  t.ywlh = h->tab[TB_YWLH].data(); t.ywss = h->tab[TB_YWSS].data();
  t.pkij = h->tab[TB_PKIJ].data(); t.qkj = h->tab[TB_QKJ].data();
  t.xl_mg = h->d.xl_mg; t.xs_mg = h->d.xs_mg; t.xg_mg = h->d.xg_mg; t.xh_mg = h->d.xh_mg;
  t.ima = h->d.ima; t.chucm = h->d.chucm;
  t.ecoalmassm = h->f.fall_speed_own_bin ? h->d.ecoal_own : h->d.ecoal_wrf;
  t.prob = h->d.prob; t.gain_var_new = h->d.gain_var_new;
  t.dropradii = h->d.dropradii; t.fccnr_con = h->d.fccnr_con; t.fccnr_mar = h->d.fccnr_mar;
  t.rccn = h->d.rccn; t.ikr_spon_break = h->d.ikr_spon_break;
  return h;
}

template <class R>
int host_fast_sbm(const Handle<R>& H, int itimestep, R dt, R dx, R dy, const int* dims,
                  const State<R>& S, double* tally, double* ledger=nullptr) {
  long n = (long)(dims[7]-dims[6]+1)*(dims[11]-dims[10]+1)*(dims[9]-dims[8]+1);
  int nz = dims[17]-dims[16]+1;
  int nc = (dims[13]-dims[12]+1)*(dims[15]-dims[14]+1);
  std::vector<R> work(7*n + 2*(nz+2), R(0));
  std::vector<int> kz(nc, 0);
  R* p = work.data();
  FsbmScratch<R> x = {p, p+n, p+2*n, p+3*n, p+4*n, p+5*n, p+6*n,
                      p+7*n, p+7*n+nz+2, kz.data()};
  return fsbm_tile<R>(H.t, H.f, itimestep, dt, dx, dy, dims, S, x, tally, 0, ledger);
}

template <class R> void derived_out(const Handle<R>& H, double* out) {
  // xl_mg, xs_mg, xg_mg, xh_mg (4x33), chucm (1089), ima (1089, as double),
  // ecoal_wrf (1089), ecoal_own (1089), prob (33), gain_var_new (1089),
  // dropradii (33), fccnr_con (43), fccnr_mar (43), xccn (43), rccn (43),
  // ikr_spon_break (1)
  const FsbmDerivedHost<R>& d = H.d;
  double* p = out;
  for (int i = 0; i < 33; ++i) *p++ = d.xl_mg[i];
  for (int i = 0; i < 33; ++i) *p++ = d.xs_mg[i];
  for (int i = 0; i < 33; ++i) *p++ = d.xg_mg[i];
  for (int i = 0; i < 33; ++i) *p++ = d.xh_mg[i];
  for (int i = 0; i < 1089; ++i) *p++ = d.chucm[i];
  for (int i = 0; i < 1089; ++i) *p++ = (double)d.ima[i];
  for (int i = 0; i < 1089; ++i) *p++ = d.ecoal_wrf[i];
  for (int i = 0; i < 1089; ++i) *p++ = d.ecoal_own[i];
  for (int i = 0; i < 33; ++i) *p++ = d.prob[i];
  for (int i = 0; i < 1089; ++i) *p++ = d.gain_var_new[i];
  for (int i = 0; i < 33; ++i) *p++ = (double)d.dropradii[i];
  for (int i = 0; i < 43; ++i) *p++ = d.fccnr_con[i];
  for (int i = 0; i < 43; ++i) *p++ = d.fccnr_mar[i];
  for (int i = 0; i < 43; ++i) *p++ = (double)d.xccn[i];
  for (int i = 0; i < 43; ++i) *p++ = (double)d.rccn[i];
  *p++ = (double)d.ikr_spon_break;
}

}  // namespace

extern "C" {

int fsbm_host_abi_version() { return 1; }
int fsbm_host_flag_count() { return (int)(sizeof(FsbmFlags) / sizeof(int)); }
int fsbm_host_derived_count() { return 33 * 4 + 1089 * 4 + 33 + 1089 + 33 + 43 * 4 + 1; }

void* fsbm_host_new_f64(const double* const* tabs, const int* flags) { return make_handle<double>(tabs, flags); }
void* fsbm_host_new_f32(const float* const* tabs, const int* flags) { return make_handle<float>(tabs, flags); }
void fsbm_host_free_f64(void* h) { delete (Handle<double>*)h; }
void fsbm_host_free_f32(void* h) { delete (Handle<float>*)h; }
void fsbm_host_derived_f64(void* h, double* out) { derived_out<double>(*(Handle<double>*)h, out); }
void fsbm_host_derived_f32(void* h, double* out) { derived_out<float>(*(Handle<float>*)h, out); }

// arrays: the 30 pointers of State in declaration order
int fsbm_host_call_f64(void* h, int itimestep, double dt, double dx, double dy, const int* dims,
                       double* const* arrays, double* tally) {
  State<double> S;
  std::memcpy(&S, arrays, sizeof(S));
  return host_fast_sbm<double>(*(Handle<double>*)h, itimestep, dt, dx, dy, dims, S, tally);
}
int fsbm_host_call_f32(void* h, int itimestep, float dt, float dx, float dy, const int* dims,
                       float* const* arrays, double* tally) {
  State<float> S;
  std::memcpy(&S, arrays, sizeof(S));
  return host_fast_sbm<float>(*(Handle<float>*)h, itimestep, dt, dx, dy, dims, S, tally);
}


int fsbm_host_ledger_f64(void* h, int itimestep, double dt, double dx, double dy,
 const int* dims, double* const* arrays, double* tally, double* ledger) {
  State<double> S; std::memcpy(&S,arrays,sizeof(S));
  return host_fast_sbm<double>(*(Handle<double>*)h,itimestep,dt,dx,dy,dims,S,tally,ledger);
}

}  // extern "C"
