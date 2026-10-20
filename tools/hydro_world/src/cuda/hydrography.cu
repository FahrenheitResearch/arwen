// HYDRO-WORLD DEM hydrography: the per-cell passes on the GPU.
//
// Every result here is bit-identical to the CPU path in flow.rs (tested by
// `rw_hydro_world hydrography-compare`):
// - fill: the filled surface is the unique minimax surface (each cell becomes
//   the lowest possible maximum elevation over all 8-connected paths to an
//   outlet: a box-edge cell or a sea cell). Priority-Flood (CPU) and this
//   monotone relaxation both compute exactly that surface, and only copy
//   DEM values, so the floats are equal.
// - D8: the same f32 drop, the same f64 distances (uploaded per row from the
//   host), the same strict comparison in the same neighbour order.
// - flats (Barnes, Lehman and Mulla 2014): the flat label, the distance to
//   lower terrain and the distance from higher terrain are unique
//   breadth-first quantities; relaxation reaches the same integers.
// - accumulation: integer cell areas, so the sum does not depend on order.
//
// Relaxation kernels work on 32 x 32 tiles in shared memory until the tile is
// stable, and the host repeats over tiles whose neighbourhood changed.
// Compiled with NVRTC defaults (no fast math, no flush to zero).

#define TS 32
#define BY 8
#define NT (TS * BY)
#define UNDRAINED 255
#define OUTC 8
#define SEAC 9
#define MAXU 0xffffffffu

__constant__ int DRc[8] = {0, -1, -1, -1, 0, 1, 1, 1};
__constant__ int DCc[8] = {1, 1, 0, -1, -1, -1, 0, 1};

__device__ __forceinline__ bool block_should_run(const unsigned int* act_in) {
    __shared__ int run;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int a = 0;
        int tx = blockIdx.x, ty = blockIdx.y, nxt = gridDim.x, nyt = gridDim.y;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int u = tx + dx, v = ty + dy;
                if (u >= 0 && v >= 0 && u < nxt && v < nyt && act_in[v * nxt + u]) a = 1;
            }
        run = a;
    }
    __syncthreads();
    return run != 0;
}

// Outlets (box edge, sea) keep their elevation; every other cell starts at +inf.
extern "C" __global__ void fill_init(const float* __restrict__ z, const unsigned char* __restrict__ sea,
                                     float* __restrict__ w, unsigned char* __restrict__ fixed, int ny, int nx) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)ny * (unsigned int)nx) return;
    int r = i / nx, c = i % nx;
    bool f = sea[i] || r == 0 || c == 0 || r == ny - 1 || c == nx - 1;
    fixed[i] = f;
    w[i] = f ? z[i] : __int_as_float(0x7f800000);
}

// w = max(z, min over the 8 neighbours of w), repeated inside the tile until stable.
extern "C" __global__ void fill_relax(const float* __restrict__ z, const unsigned char* __restrict__ fixed, float* w,
                                      int ny, int nx, const unsigned int* act_in, unsigned int* act_out,
                                      unsigned int* changed) {
    if (!block_should_run(act_in)) return;
    __shared__ float s[TS + 2][TS + 3];
    __shared__ float sz[TS][TS];
    __shared__ unsigned char sf[TS][TS];
    __shared__ int ch_iter, ch_any;
    const float INF = __int_as_float(0x7f800000);
    int r0 = blockIdx.y * TS, c0 = blockIdx.x * TS;
    int tid = threadIdx.y * TS + threadIdx.x;
    for (int k = tid; k < (TS + 2) * (TS + 2); k += NT) {
        int lr = k / (TS + 2), lc = k % (TS + 2);
        int r = r0 + lr - 1, c = c0 + lc - 1;
        s[lr][lc] = (r >= 0 && c >= 0 && r < ny && c < nx) ? w[(unsigned int)r * nx + c] : INF;
    }
    for (int k = tid; k < TS * TS; k += NT) {
        int lr = k / TS, lc = k % TS;
        int r = r0 + lr, c = c0 + lc;
        bool in = r < ny && c < nx;
        sz[lr][lc] = in ? z[(unsigned int)r * nx + c] : 0.0f;
        sf[lr][lc] = in ? fixed[(unsigned int)r * nx + c] : 1;
    }
    if (tid == 0) ch_any = 0;
    for (;;) {
        __syncthreads();
        if (tid == 0) ch_iter = 0;
        __syncthreads();
        for (int lr = threadIdx.y; lr < TS; lr += BY) {
            int lc = threadIdx.x;
            if (sf[lr][lc]) continue;
            float m = INF;
            for (int k = 0; k < 8; k++) {
                float x = s[lr + 1 + DRc[k]][lc + 1 + DCc[k]];
                if (x < m) m = x;
            }
            float zc = sz[lr][lc];
            float v = m > zc ? m : zc;
            if (v < s[lr + 1][lc + 1]) {
                s[lr + 1][lc + 1] = v;
                ch_iter = 1;
            }
        }
        __syncthreads();
        if (!ch_iter) break;
        if (tid == 0) ch_any = 1;
    }
    __syncthreads();
    if (!ch_any) return;
    for (int k = tid; k < TS * TS; k += NT) {
        int lr = k / TS, lc = k % TS;
        int r = r0 + lr, c = c0 + lc;
        if (r < ny && c < nx) w[(unsigned int)r * nx + c] = s[lr + 1][lc + 1];
    }
    if (tid == 0) {
        act_out[blockIdx.y * gridDim.x + blockIdx.x] = 1;
        *changed = 1;
    }
}

// Cells still at +inf have no path to an outlet; cells above their DEM value were raised.
extern "C" __global__ void fill_check(const float* __restrict__ z, const float* __restrict__ w, unsigned int n,
                                      unsigned int* unreached, unsigned int* raised) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (isinf(w[i]) && w[i] > 0.0f) atomicAdd(unreached, 1u);
    if (__float_as_uint(w[i]) != __float_as_uint(z[i])) atomicAdd(raised, 1u);
}

// D8 steepest descent; dist holds (dx, diagonal, dy) per row in metres.
extern "C" __global__ void d8(const float* __restrict__ f, const unsigned char* __restrict__ sea,
                              const double* __restrict__ dist, unsigned char* __restrict__ dir, int ny, int nx) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)ny * (unsigned int)nx) return;
    if (sea[i]) {
        dir[i] = SEAC;
        return;
    }
    int r = i / nx, c = i % nx;
    double dx = dist[3 * r], dg = dist[3 * r + 1], dy = dist[3 * r + 2];
    double d[8] = {dx, dg, dy, dg, dx, dg, dy, dg};
    int best = -1;
    double bs = 0.0;
    bool touches_sea = false;
    float fi = f[i];
    for (int k = 0; k < 8; k++) {
        int nr = r + DRc[k], nc = c + DCc[k];
        if (nr < 0 || nc < 0 || nr >= ny || nc >= nx) continue;
        unsigned int n = (unsigned int)nr * nx + nc;
        touches_sea |= sea[n] != 0;
        float df = fi - f[n];
        double drop = (double)df;
        if (drop > 0.0) {
            double sl = drop / d[k];
            if (sl > bs) {
                bs = sl;
                best = k;
            }
        }
    }
    bool border = r == 0 || c == 0 || r == ny - 1 || c == nx - 1;
    if (best >= 0) {
        unsigned int n = (unsigned int)(r + DRc[best]) * nx + (c + DCc[best]);
        dir[i] = sea[n] ? OUTC : (unsigned char)best;
    } else {
        dir[i] = (border || touches_sea) ? OUTC : UNDRAINED;
    }
}

// Per undrained cell: in-flat neighbour bits (undrained, same filled elevation),
// flat label seed (own index), distance seeds next to lower terrain (an equal
// neighbour that already drains) and next to higher terrain.
extern "C" __global__ void flat_prep(const float* __restrict__ f, const unsigned char* __restrict__ dir,
                                     unsigned char* __restrict__ nm, unsigned int* __restrict__ lab,
                                     unsigned int* __restrict__ dl, unsigned int* __restrict__ dh, int ny, int nx,
                                     unsigned int* undrained) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)ny * (unsigned int)nx) return;
    if (dir[i] != UNDRAINED) {
        nm[i] = 0;
        lab[i] = MAXU;
        dl[i] = MAXU;
        dh[i] = MAXU;
        return;
    }
    atomicAdd(undrained, 1u);
    int r = i / nx, c = i % nx;
    float e = f[i];
    unsigned char m = 0;
    bool low = false, high = false;
    for (int k = 0; k < 8; k++) {
        int nr = r + DRc[k], nc = c + DCc[k];
        if (nr < 0 || nc < 0 || nr >= ny || nc >= nx) continue;
        unsigned int n = (unsigned int)nr * nx + nc;
        float fn = f[n];
        if (fn == e) {
            if (dir[n] == UNDRAINED) m |= (unsigned char)(1u << k);
            else low = true;
        }
        if (fn > e) high = true;
    }
    nm[i] = m;
    lab[i] = i;
    dl[i] = low ? 1u : MAXU;
    dh[i] = high ? 1u : MAXU;
}

// v = min(v, v[neighbour] + inc) over the in-flat neighbours, inside the tile until stable.
// inc 0: connected-component label (minimum index); inc 1: breadth-first distance.
extern "C" __global__ void relax_u32(unsigned int* v, const unsigned char* __restrict__ nm, unsigned int inc, int ny,
                                     int nx, const unsigned int* act_in, unsigned int* act_out,
                                     unsigned int* changed) {
    if (!block_should_run(act_in)) return;
    __shared__ unsigned int s[TS + 2][TS + 3];
    __shared__ unsigned char sm[TS][TS];
    __shared__ int ch_iter, ch_any, any_mask;
    int r0 = blockIdx.y * TS, c0 = blockIdx.x * TS;
    int tid = threadIdx.y * TS + threadIdx.x;
    if (tid == 0) any_mask = 0;
    __syncthreads();
    for (int k = tid; k < TS * TS; k += NT) {
        int lr = k / TS, lc = k % TS;
        int r = r0 + lr, c = c0 + lc;
        unsigned char m = (r < ny && c < nx) ? nm[(unsigned int)r * nx + c] : 0;
        sm[lr][lc] = m;
        if (m) any_mask = 1;
    }
    __syncthreads();
    if (!any_mask) return;
    for (int k = tid; k < (TS + 2) * (TS + 2); k += NT) {
        int lr = k / (TS + 2), lc = k % (TS + 2);
        int r = r0 + lr - 1, c = c0 + lc - 1;
        s[lr][lc] = (r >= 0 && c >= 0 && r < ny && c < nx) ? v[(unsigned int)r * nx + c] : MAXU;
    }
    if (tid == 0) ch_any = 0;
    for (;;) {
        __syncthreads();
        if (tid == 0) ch_iter = 0;
        __syncthreads();
        for (int lr = threadIdx.y; lr < TS; lr += BY) {
            int lc = threadIdx.x;
            unsigned char m = sm[lr][lc];
            if (!m) continue;
            unsigned int best = s[lr + 1][lc + 1];
            for (int k = 0; k < 8; k++) {
                if (!((m >> k) & 1)) continue;
                unsigned int x = s[lr + 1 + DRc[k]][lc + 1 + DCc[k]];
                if (x == MAXU) continue;
                unsigned int cand = x + inc;
                if (cand < best) best = cand;
            }
            if (best < s[lr + 1][lc + 1]) {
                s[lr + 1][lc + 1] = best;
                ch_iter = 1;
            }
        }
        __syncthreads();
        if (!ch_iter) break;
        if (tid == 0) ch_any = 1;
    }
    __syncthreads();
    if (!ch_any) return;
    for (int k = tid; k < TS * TS; k += NT) {
        int lr = k / TS, lc = k % TS;
        int r = r0 + lr, c = c0 + lc;
        if (r < ny && c < nx && sm[lr][lc]) v[(unsigned int)r * nx + c] = s[lr + 1][lc + 1];
    }
    if (tid == 0) {
        act_out[blockIdx.y * gridDim.x + blockIdx.x] = 1;
        *changed = 1;
    }
}

// Per flat: cell count and the largest distance from higher terrain; flats with
// a cell that cannot reach an outlet at its own elevation are counted.
extern "C" __global__ void flat_stats(const unsigned char* __restrict__ dir, const unsigned int* __restrict__ lab,
                                      const unsigned int* __restrict__ dl, const unsigned int* __restrict__ dh,
                                      unsigned int n, unsigned int* cnt, unsigned int* hmax, unsigned int* nflats,
                                      unsigned int* no_outlet, unsigned int* first_bad) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || dir[i] != UNDRAINED) return;
    unsigned int l = lab[i];
    atomicAdd(&cnt[l], 1u);
    if (dh[i] != MAXU) atomicMax(&hmax[l], dh[i]);
    if (l == i) atomicAdd(nflats, 1u);
    if (dl[i] == MAXU) {
        atomicAdd(no_outlet, 1u);
        atomicMin(first_bad, i);
    }
}

extern "C" __global__ void flat_largest(const unsigned int* __restrict__ lab, const unsigned int* __restrict__ cnt,
                                        unsigned int n, unsigned int* largest) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || lab[i] != i) return;
    atomicMax(largest, cnt[i]);
}

__device__ __forceinline__ unsigned long long flat_mask(const unsigned int* lab, const unsigned int* dl,
                                                        const unsigned int* dh, const unsigned int* hmax,
                                                        unsigned int j) {
    unsigned long long away = dh[j] == MAXU ? 0ull : (unsigned long long)(hmax[lab[j]] - dh[j]);
    return 2ull * (unsigned long long)dl[j] + away;
}

// Direction of every undrained cell: towards the in-flat neighbour with the
// smallest lower mask, or an equal-elevation neighbour that already drains.
// Reads the first-pass directions (dir), writes out (a copy of dir).
extern "C" __global__ void flat_dir(const float* __restrict__ f, const unsigned char* __restrict__ dir,
                                    const unsigned char* __restrict__ nm, const unsigned int* __restrict__ lab,
                                    const unsigned int* __restrict__ dl, const unsigned int* __restrict__ dh,
                                    const unsigned int* __restrict__ hmax, unsigned char* __restrict__ out, int ny,
                                    int nx, unsigned int* no_lower, unsigned int* first_bad) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)ny * (unsigned int)nx || dir[i] != UNDRAINED) return;
    int r = i / nx, c = i % nx;
    float e = f[i];
    unsigned char m = nm[i];
    unsigned long long mj = flat_mask(lab, dl, dh, hmax, i);
    unsigned long long bv = 0xffffffffffffffffull;
    int bk = -1;
    for (int k = 0; k < 8; k++) {
        int nr = r + DRc[k], nc = c + DCc[k];
        if (nr < 0 || nc < 0 || nr >= ny || nc >= nx) continue;
        unsigned int n = (unsigned int)nr * nx + nc;
        if (f[n] != e) continue;
        unsigned long long cand;
        if ((m >> k) & 1) {
            unsigned long long mm = flat_mask(lab, dl, dh, hmax, n);
            if (!(mm < mj)) continue;
            cand = mm;
        } else if (dir[n] != UNDRAINED) {
            cand = 0ull;
        } else {
            continue;
        }
        if (bk < 0 || cand < bv) {
            bv = cand;
            bk = k;
        }
    }
    if (bk < 0) {
        atomicAdd(no_lower, 1u);
        atomicMin(first_bad, i);
        return;
    }
    unsigned int n = (unsigned int)(r + DRc[bk]) * nx + (c + DCc[bk]);
    out[i] = dir[n] == SEAC ? OUTC : (unsigned char)bk;
}

// Accumulation: in-degree, then one thread per source cell walks downstream;
// at a confluence only the last arriving thread continues (integer atomics,
// so the sums are exact and order-free).
extern "C" __global__ void acc_init(const unsigned char* __restrict__ dir, const unsigned long long* __restrict__ area,
                                    unsigned long long* __restrict__ acc, unsigned int* __restrict__ indeg,
                                    unsigned char* __restrict__ source, int ny, int nx, unsigned int* land) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)ny * (unsigned int)nx) return;
    int r = i / nx, c = i % nx;
    unsigned int d = 0;
    for (int k = 0; k < 8; k++) {
        int nr = r + DRc[k], nc = c + DCc[k];
        if (nr < 0 || nc < 0 || nr >= ny || nc >= nx) continue;
        if (dir[(unsigned int)nr * nx + nc] == (unsigned char)((k + 4) % 8)) d++;
    }
    indeg[i] = d;
    bool sea = dir[i] == SEAC;
    // Sources are fixed here: during the walk other threads decrement indeg.
    source[i] = !sea && d == 0;
    acc[i] = sea ? 0ull : area[r];
    if (!sea) atomicAdd(land, 1u);
}

extern "C" __global__ void acc_walk(const unsigned char* __restrict__ dir, unsigned long long* acc,
                                    unsigned int* indeg, const unsigned char* __restrict__ source, int ny, int nx,
                                    unsigned int* processed, unsigned int* starts, unsigned int* longest) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)ny * (unsigned int)nx) return;
    if (!source[i]) return;
    atomicAdd(starts, 1u);
    unsigned int cur = i, steps = 0;
    for (;;) {
        steps++;
        atomicAdd(processed, 1u);
        unsigned char d = dir[cur];
        if (d >= 8) break;
        int r = cur / nx, c = cur % nx;
        int nr = r + DRc[d], nc = c + DCc[d];
        if (nr < 0 || nc < 0 || nr >= ny || nc >= nx) break;
        unsigned int nxt = (unsigned int)nr * nx + nc;
        unsigned long long carry = atomicAdd(&acc[cur], 0ull);
        atomicAdd(&acc[nxt], carry);
        __threadfence();
        if (atomicSub(&indeg[nxt], 1u) != 1u) break;
        __threadfence();
        cur = nxt;
    }
    atomicMax(longest, steps);
}
