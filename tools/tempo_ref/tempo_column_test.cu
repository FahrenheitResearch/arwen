// TEMPO port column harness (lane tempo-mp).
// Loads TEMPO's own tempo_init state (tempo_blob.bin from tempo_ref_driver),
// replays the reference scenarios (tempo_ref_columns.bin) through the port
// on the CPU (float64, host build of the same headers) and on the GPU
// (one thread per column), and writes both in the reference record layout:
//   out_cpu.bin, out_gpu.bin
// GPU reproducibility: every scenario runs on NREP identical columns, twice;
// the harness reports whether all replicas and both runs are bitwise equal.
//
// build: nvcc -O2 -std=c++17 -arch=sm_120 -fmad=false -Xcompiler -ffp-contract=off
//        -I ../../gpuwm/core/kernels/tempo tempo_column_test.cu -o tempo_column_test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>

#include "tempo_column.cuh"

#define NFLD 17
struct ColState {
  double f[NFLD][TEMPO_NZP];  // t p w dz qv qc qr qi qs qg ni nr nc nwfa nifa ng qb
};
enum { F_T, F_P, F_W, F_DZ, F_QV, F_QC, F_QR, F_QI, F_QS, F_QG, F_NI, F_NR, F_NC, F_NWFA, F_NIFA, F_NG, F_QB };

TEMPO_HDN inline void run_one(const TempoParams& P, const TempoCtx& C, int nz, TempoWork& W, ColState& S) {
  tempo_main_column(P, C, nz, W, S.f[F_QV], S.f[F_QC], S.f[F_QI], S.f[F_QR], S.f[F_QS], S.f[F_QG],
                    S.f[F_QB], S.f[F_NI], S.f[F_NR], S.f[F_NC], S.f[F_NG], S.f[F_NWFA], S.f[F_NIFA],
                    S.f[F_T], S.f[F_P], S.f[F_W], S.f[F_DZ]);
}

__global__ void tempo_columns_kernel(const TempoBlob* L, TempoCtx C, int ncol, int nz,
                                     ColState* S, TempoWork* W) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncol) return;
  TempoParams P{L};
  TempoCtx c = C;
  if (i != 0) c.first_call_main = 0;  // Fortran SAVE flag: true only for the very first column
  run_one(P, c, nz, W[i], S[i]);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct FieldInfo { const char* name; size_t off; size_t n; };
static std::vector<FieldInfo> field_table() {
  std::vector<FieldInfo> v;
#define TX(name, d1, d2, d3, d4, d5) \
  v.push_back({#name, offsetof(TempoBlob, name) / sizeof(double), (size_t)(d1) * (d2) * (d3) * (d4) * (d5)});
  TEMPO_FIELDS(TX)
#undef TX
  return v;
}

static TempoBlob* load_blob(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  TempoBlob* b = (TempoBlob*)calloc(1, sizeof(TempoBlob));
  double* base = (double*)b;
  auto ft = field_table();
  std::map<std::string, FieldInfo> m;
  for (auto& fi : ft) m[fi.name] = fi;
  std::map<std::string, int> seen;
  char nm[49];
  int64_t n;
  while (fread(nm, 1, 48, f) == 48) {
    nm[48] = 0;
    for (int i = 47; i >= 0 && nm[i] == ' '; --i) nm[i] = 0;
    if (fread(&n, 8, 1, f) != 1) break;
    auto it = m.find(nm);
    if (it == m.end()) { fprintf(stderr, "blob: unknown field %s\n", nm); exit(1); }
    if ((size_t)n != it->second.n) { fprintf(stderr, "blob: %s count %lld != %zu\n", nm, (long long)n, it->second.n); exit(1); }
    if (fread(base + it->second.off, 8, n, f) != (size_t)n) { fprintf(stderr, "blob: short %s\n", nm); exit(1); }
    seen[nm] = 1;
  }
  fclose(f);
  for (auto& fi : ft)
    if (!seen.count(fi.name)) { fprintf(stderr, "blob: missing field %s\n", fi.name); exit(1); }
  fprintf(stderr, "blob: %zu fields, %.1f MB\n", ft.size(), sizeof(TempoBlob) / 1e6);
  return b;
}

struct Scenario {
  int64_t sid, nz, nsteps;
  double dt, semi;
  std::vector<double> init;  // NFLD*nz
  long ref_offset;           // file offset of step records
};

static void read_state(FILE* f, std::vector<double>& v, int nz) {
  v.resize((size_t)NFLD * nz);
  if (fread(v.data(), 8, v.size(), f) != v.size()) { fprintf(stderr, "short state\n"); exit(1); }
}

static std::vector<Scenario> read_scenarios(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  std::vector<Scenario> out;
  while (true) {
    Scenario s;
    if (fread(&s.sid, 8, 1, f) != 1) break;
    if (fread(&s.nz, 8, 1, f) != 1 || fread(&s.nsteps, 8, 1, f) != 1 || fread(&s.dt, 8, 1, f) != 1 ||
        fread(&s.semi, 8, 1, f) != 1) { fprintf(stderr, "short header\n"); exit(1); }
    read_state(f, s.init, (int)s.nz);
    s.ref_offset = ftell(f);
    long step_bytes = (long)8 * (NFLD * s.nz + 7 + 4 * s.nz);
    fseek(f, step_bytes * s.nsteps, SEEK_CUR);
    out.push_back(s);
  }
  fclose(f);
  return out;
}

static void set_state(ColState& S, const Scenario& s) {
  memset(&S, 0, sizeof(S));
  int nz = (int)s.nz;
  for (int j = 0; j < NFLD; ++j)
    for (int k = 1; k <= nz; ++k) S.f[j][k] = s.init[(size_t)j * nz + (k - 1)];
}

static void write_step(FILE* f, const ColState& S, const TempoWork& W, int nz) {
  for (int j = 0; j < NFLD; ++j) fwrite(&S.f[j][1], 8, nz, f);
  double mh_col = 0.0;
  for (int k = 1; k <= nz; ++k) mh_col = (k == 1 || W.max_hail_diameter[k] > mh_col) ? W.max_hail_diameter[k] : mh_col;
  double sc[7] = {W.rain_precip, W.ice_liquid_equiv_precip, W.snow_liquid_equiv_precip,
                  W.graupel_liquid_equiv_precip, W.frz_rain_precip, W.max_hail_diameter[1], mh_col};
  fwrite(sc, 8, 7, f);
  fwrite(&W.refl10cm[1], 8, nz, f);
  fwrite(&W.re_cloud[1], 8, nz, f);
  fwrite(&W.re_ice[1], 8, nz, f);
  fwrite(&W.re_snow[1], 8, nz, f);
}

static TempoCtx make_ctx(const Scenario& s, int first) {
  TempoCtx C;
  memset(&C, 0, sizeof(C));
  C.global_dt = s.dt;
  C.global_inverse_dt = 1.0 / s.dt;
  C.aerosolaware_flag = 1;
  C.hailaware_flag = 1;
  C.semi_sedi_flag = s.semi > 0.5;
  C.refl10cm_from_melting_flag = 0;
  C.turn_off_micro_flag = 0;
  C.cloud_condensation_flag = 1;
  C.refl10cm_flag = 1;
  C.re_cloud_flag = 1;
  C.re_ice_flag = 1;
  C.re_snow_flag = 1;
  C.max_hail_diameter_flag = 1;
  C.first_call_main = first;
  return C;
}

int main(int argc, char** argv) {
  const char* blob_path = argc > 1 ? argv[1] : "tempo_blob.bin";
  const char* cols_path = argc > 2 ? argv[2] : "tempo_ref_columns.bin";
  const int NREP = argc > 3 ? atoi(argv[3]) : 4096;
  TempoBlob* hb = load_blob(blob_path);
  TempoParams P{hb};
  auto scen = read_scenarios(cols_path);
  fprintf(stderr, "scenarios: %zu, sizeof(TempoWork) %.1f KB\n", scen.size(), sizeof(TempoWork) / 1e3);

  // ---------------- CPU float64 reference ----------------
  {
    const char* pert = getenv("TEMPO_PERTURB");  // relative perturbation of initial qg (sensitivity probe)
    FILE* fo = fopen(pert ? "out_cpu_pert.bin" : "out_cpu.bin", "wb");
    TempoWork* W = (TempoWork*)calloc(1, sizeof(TempoWork));
    ColState S;
    int first = 1;
    for (auto& s : scen) {
      set_state(S, s);
      if (pert)
        for (int k = 1; k <= (int)s.nz; ++k) S.f[F_QG][k] *= (1.0 + atof(pert));
      for (int it = 1; it <= s.nsteps; ++it) {
        TempoCtx C = make_ctx(s, first);
        first = 0;
        run_one(P, C, (int)s.nz, *W, S);
        write_step(fo, S, *W, (int)s.nz);
      }
    }
    fclose(fo);
    free(W);
    fprintf(stderr, "cpu: done\n");
  }

  // ---------------- GPU ----------------
  if (NREP <= 0) return 0;
  TempoBlob* db = nullptr;
  CK(cudaMalloc(&db, sizeof(TempoBlob)));
  CK(cudaMemcpy(db, hb, sizeof(TempoBlob), cudaMemcpyHostToDevice));
  ColState* dS = nullptr;
  TempoWork* dW = nullptr;
  CK(cudaMalloc(&dS, sizeof(ColState) * NREP));
  CK(cudaMalloc(&dW, sizeof(TempoWork) * NREP));
  CK(cudaMemset(dW, 0, sizeof(TempoWork) * NREP));
  CK(cudaMemset(dS, 0, sizeof(ColState) * NREP));
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
  std::vector<ColState> hS(NREP);
  std::vector<ColState> first_run_final;  // per scenario final state of run A (replica 0)
  bool all_rep_equal = true, run_equal = true;
  double kernel_ms_total = 0.0;
  long steps_total = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int run = 0; run < 2; ++run) {
    FILE* fo = run == 0 ? fopen("out_gpu.bin", "wb") : nullptr;
    int first = 1;
    for (size_t si = 0; si < scen.size(); ++si) {
      auto& s = scen[si];
      for (int r = 0; r < NREP; ++r) set_state(hS[r], s);
      CK(cudaMemcpy(dS, hS.data(), sizeof(ColState) * NREP, cudaMemcpyHostToDevice));
      for (int it = 1; it <= s.nsteps; ++it) {
        TempoCtx C = make_ctx(s, first);
        first = 0;
        int threads = 64;
        int blocks = (NREP + threads - 1) / threads;
        CK(cudaEventRecord(e0));
        tempo_columns_kernel<<<blocks, threads>>>(db, C, NREP, (int)s.nz, dS, dW);
        CK(cudaEventRecord(e1));
        CK(cudaGetLastError());
        CK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        kernel_ms_total += ms;
        steps_total += 1;
        if (fo) {
          ColState S0;
          TempoWork* W0 = (TempoWork*)malloc(sizeof(TempoWork));
          CK(cudaMemcpy(&S0, dS, sizeof(ColState), cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(W0, dW, sizeof(TempoWork), cudaMemcpyDeviceToHost));
          write_step(fo, S0, *W0, (int)s.nz);
          free(W0);
        }
      }
      CK(cudaMemcpy(hS.data(), dS, sizeof(ColState) * NREP, cudaMemcpyDeviceToHost));
      for (int r = 1; r < NREP; ++r)
        if (memcmp(&hS[r], &hS[0], sizeof(ColState)) != 0) { all_rep_equal = false; break; }
      if (run == 0) first_run_final.push_back(hS[0]);
      else if (memcmp(&hS[0], &first_run_final[si], sizeof(ColState)) != 0) run_equal = false;
    }
    if (fo) fclose(fo);
  }
  printf("GPU_REPLICAS_BITWISE_EQUAL %d\n", all_rep_equal ? 1 : 0);
  printf("GPU_RUN_TO_RUN_BITWISE_EQUAL %d\n", run_equal ? 1 : 0);
  printf("GPU_COLUMNS %d KERNEL_MS_PER_STEP %.3f US_PER_COLUMN_STEP %.4f\n", NREP,
         kernel_ms_total / steps_total, 1000.0 * kernel_ms_total / steps_total / NREP);
  cudaFree(db);
  cudaFree(dS);
  cudaFree(dW);
  free(hb);
  return 0;
}
