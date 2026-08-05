// Batched symmetric eigendecomposition: one thread block per matrix,
// two-sided cyclic Jacobi in shared memory.
//
// Solves A = U diag(w) U^T for a stack of small REAL SYMMETRIC matrices,
// with the eigenvalues ascending and the eigenvectors in the columns of U --
// the same output contract as numpy.linalg.eigh / cupy.linalg.eigh, plus a
// sign convention those two do not have (see "Canonical form" below).
//
// Why Jacobi rather than the tridiagonal-plus-QL path a general library takes
// -------------------------------------------------------------------------
// The caller is gpuwm/da/letkf.py, which factors ONE matrix per analysis
// gridpoint: k x k with k the ensemble size (10 today), of which there are a
// few hundred thousand per analysis cycle.  That workload is the opposite of
// the one LAPACK is shaped for.  Householder tridiagonalisation is a sequence
// of rank-2 updates with a serial dependence down the diagonal, and the
// implicitly shifted QL iteration that follows it has a data-dependent shift
// per step; both are the right answer for one large matrix on one core and
// neither has anything for a thread block to do.  A Jacobi sweep, by
// contrast, splits into k/2 rotations on DISJOINT index pairs that can all be
// applied at once, so a block of threads is busy for the whole factorisation
// and the entire k x k problem stays in shared memory from load to store.
//
// Two properties beyond parallelism decided it:
//
//   * Demmel and Veselic (1992), *Jacobi's method is more accurate than QR*,
//     SIMAX 13, 1204-1245: on a symmetric POSITIVE DEFINITE matrix scaled to
//     unit diagonal, Jacobi computes every eigenvalue to a relative accuracy
//     that depends on the condition number of the SCALED matrix, not of A.
//     QR-based methods only promise absolute accuracy proportional to
//     ||A||.  The matrix here -- (R-1)I/rho + C Yb, Hunt et al. step 5 -- is
//     positive definite by construction, so the stronger bound is the one
//     that applies.
//   * A matrix that is already diagonal is a FIXED POINT, exactly.  Every
//     rotation is skipped by the threshold below, U comes back the literal
//     identity and w the literal diagonal, bit for bit.  No library
//     eigensolver promises that, and gpuwm/da/letkf.py's inactive-gridpoint
//     argument (its module docstring) is written around the absence of the
//     promise.  Here it is a property of the algorithm.
//
// Ordering
// --------
// The pairs are visited in the round-robin ("chess tournament") ordering:
// index 0 is held fixed while the other M-1 indices rotate, which emits all
// M(M-1)/2 pairs in M-1 rounds of M/2 mutually disjoint pairs.  The ordering
// is FIXED, compiled in, and part of this kernel's contract -- results are
// bitwise reproducible run to run, and a different ordering would diagonalise
// the same matrix to a different rounding.
//
// Odd k is padded to M = k+1 with a unit diagonal and exact zeros off it.
// The padding row and column are never touched: a rotation is only applied
// when |a_pq| clears the threshold, an exact zero never does, and neither a
// left nor a right rotation between two real indices writes into the padding
// line.  It stays exactly e_M with eigenvalue exactly 1 and is dropped on the
// way out.
//
// Canonical form
// --------------
// An eigendecomposition is not unique -- eigenvectors carry an arbitrary
// sign, and a degenerate eigenvalue admits any orthonormal basis of its
// eigenspace -- so this kernel pins what it can and documents the rest:
//
//   * eigenvalues ASCENDING; ties broken by the source column index, so the
//     permutation is a total order and not merely a sort;
//   * each eigenvector scaled so its LARGEST-MAGNITUDE component is
//     positive, ties in magnitude broken by the lowest row index.
//
// What that does NOT fix is the basis chosen inside a degenerate eigenspace,
// which no convention can: it is a property of the matrix, not of the
// output format.  Callers that need a convention-free comparison should
// compare the matrix functions U f(w) U^T, which are unique.  That is
// precisely what letkf.py consumes -- A^-1 and A^-1/2 -- and it is the
// invariant tests/test_jacobi_eigh_gpu.py asserts.
//
// Status word, per matrix
// -----------------------
//    > 0   converged, in that many sweeps
//     -1   still rotating at the sweep cap: NOT an answer, refuse it
//     -2   non-finite entry on input; w and U are filled with NaN

#ifndef JACOBI_K
#define JACOBI_K 10
#endif

// The worked size: k rounded up to even, so the round-robin ordering has a
// partner for every index.
#define JACOBI_M (JACOBI_K + (JACOBI_K & 1))

#ifndef JACOBI_TPB
#define JACOBI_TPB 32
#endif

#ifndef JACOBI_MPB
#define JACOBI_MPB 8
#endif

#ifndef JACOBI_SWEEPS
#define JACOBI_SWEEPS 40
#endif

#ifndef JACOBI_REAL_BYTES
#define JACOBI_REAL_BYTES 8
#endif

#if JACOBI_REAL_BYTES == 8
typedef double jreal;
#define JACOBI_EPS 2.220446049250313e-16
#else
typedef float jreal;
#define JACOBI_EPS 1.1920928955078125e-07f
#endif

// One matrix per warp needs only a warp-wide barrier, and a warp that
// finishes early cannot strand a sibling.  Wider tiers put one matrix on the
// whole block, so every thread runs the same number of sweeps and the block
// barrier is equally safe.
#if JACOBI_TPB == 32
#define JACOBI_SYNC() __syncwarp()
#else
#define JACOBI_SYNC() __syncthreads()
#endif

static __device__ __forceinline__ double jacobi_quiet_nan()
{
    return __longlong_as_double(0x7ff8000000000000LL);
}

// The round-robin pairing.  Index 0 is fixed; 1..M-1 rotate by one per round.
// Returned normalised so p < q, which is what makes the visiting order -- and
// therefore the rounding -- a property of the algorithm rather than of an
// index accident.
static __device__ __forceinline__ void jacobi_pair(
    int sweep_round, int pair_index, int *p_out, int *q_out)
{
    const int a = (pair_index == 0)
        ? 0
        : ((pair_index - 1 + sweep_round) % (JACOBI_M - 1) + 1);
    const int b = (JACOBI_M - 2 - pair_index + sweep_round)
        % (JACOBI_M - 1) + 1;
    *p_out = a < b ? a : b;
    *q_out = a < b ? b : a;
}

extern "C" __global__ void jacobi_eigh_batched(
    const jreal *__restrict__ a_in,
    jreal *__restrict__ w_out,
    jreal *__restrict__ v_out,
    int *__restrict__ status_out,
    const long long n_matrices)
{
    extern __shared__ unsigned char jacobi_smem[];

#if JACOBI_TPB == 32
    const int slot = threadIdx.y;
#else
    const int slot = 0;
#endif
    const long long mat = (long long)blockIdx.x * JACOBI_MPB + slot;
    const int tid = threadIdx.x;

    // Shared partition, per matrix in the block:
    //   work[M*M]  the matrix being diagonalised
    //   vecs[M*M]  the accumulated rotations
    //   cosv,sinv  one rotation per concurrent pair
    //   diag[M]    eigenvalues before the sort
    //   sign[M]    the canonical sign per output column
    // then, after every matrix's real block, the integer scratch:
    //   perm[M]    the ascending permutation
    //   flag[1]    "a rotation happened" / "input was not finite"
    const int reals_per_matrix = 2 * JACOBI_M * JACOBI_M + 3 * JACOBI_M;
    jreal *shared_reals = (jreal *)jacobi_smem;
    jreal *work = shared_reals + (long long)slot * reals_per_matrix;
    jreal *vecs = work + JACOBI_M * JACOBI_M;
    jreal *cosv = vecs + JACOBI_M * JACOBI_M;
    jreal *sinv = cosv + JACOBI_M / 2;
    jreal *diag = sinv + JACOBI_M / 2;
    jreal *sign = diag + JACOBI_M;

    int *shared_ints =
        (int *)(shared_reals + (long long)JACOBI_MPB * reals_per_matrix);
    int *perm = shared_ints + (long long)slot * (JACOBI_M + 1);
    int *flag = perm + JACOBI_M;

    if (mat >= n_matrices) {
        return;
    }

    const jreal *src = a_in + mat * (long long)JACOBI_K * JACOBI_K;

    // ---- load, padding an odd k with an isolated unit diagonal ----------
    if (tid == 0) {
        flag[0] = 0;
    }
    JACOBI_SYNC();
    for (int idx = tid; idx < JACOBI_M * JACOBI_M; idx += JACOBI_TPB) {
        const int r = idx / JACOBI_M;
        const int c = idx - r * JACOBI_M;
        jreal value;
        if (r < JACOBI_K && c < JACOBI_K) {
            value = src[(long long)r * JACOBI_K + c];
            if (!isfinite(value)) {
                flag[0] = 1;
            }
        } else {
            value = (r == c) ? (jreal)1 : (jreal)0;
        }
        work[idx] = value;
        vecs[idx] = (r == c) ? (jreal)1 : (jreal)0;
    }
    JACOBI_SYNC();
    if (flag[0]) {
        // Refused, not repaired.  The caller raises; nothing downstream may
        // mistake a NaN for an analysis.
        if (tid == 0) {
            status_out[mat] = -2;
        }
        for (int i = tid; i < JACOBI_K; i += JACOBI_TPB) {
            w_out[mat * (long long)JACOBI_K + i] = (jreal)jacobi_quiet_nan();
        }
        for (int i = tid; i < JACOBI_K * JACOBI_K; i += JACOBI_TPB) {
            v_out[mat * (long long)JACOBI_K * JACOBI_K + i] =
                (jreal)jacobi_quiet_nan();
        }
        return;
    }

    // ---- sweep until no pair in a whole sweep clears the threshold ------
    int sweeps = 0;
    int converged = 0;
    for (int sweep = 0; sweep < JACOBI_SWEEPS; ++sweep) {
        if (tid == 0) {
            flag[0] = 0;
        }
        JACOBI_SYNC();

        for (int sweep_round = 0; sweep_round < JACOBI_M - 1; ++sweep_round) {
            // (a) the M/2 rotations of this round, all on disjoint pairs
            for (int i = tid; i < JACOBI_M / 2; i += JACOBI_TPB) {
                int p, q;
                jacobi_pair(sweep_round, i, &p, &q);
                const jreal app = work[p * JACOBI_M + p];
                const jreal aqq = work[q * JACOBI_M + q];
                const jreal apq = work[p * JACOBI_M + q];
                jreal cs = (jreal)1;
                jreal sn = (jreal)0;
                // Demmel-Veselic: measure the off-diagonal against the
                // GEOMETRIC MEAN of its two diagonals, not against ||A||.
                // That is what keeps the relative accuracy of a small
                // eigenvalue of a positive definite matrix, and it makes an
                // exactly diagonal input a fixed point.
                const jreal threshold =
                    (jreal)JACOBI_EPS * sqrt(fabs(app) * fabs(aqq));
                if (fabs(apq) > threshold) {
                    // Golub and Van Loan, Algorithm 8.4.1 (sym.schur2): the
                    // root of smaller magnitude, so |theta| <= pi/4 and the
                    // rotation never mixes more than it must.
                    const jreal tau = (aqq - app) / (apq + apq);
                    const jreal root = sqrt((jreal)1 + tau * tau);
                    const jreal t = (tau >= (jreal)0)
                        ? ((jreal)1 / (tau + root))
                        : ((jreal)(-1) / (root - tau));
                    cs = (jreal)1 / sqrt((jreal)1 + t * t);
                    sn = t * cs;
                    flag[0] = 1;
                }
                cosv[i] = cs;
                sinv[i] = sn;
            }
            JACOBI_SYNC();

            // (b) left rotation, A <- J^T A.  Pairs own disjoint ROWS.
            for (int idx = tid; idx < (JACOBI_M / 2) * JACOBI_M;
                 idx += JACOBI_TPB) {
                const int i = idx / JACOBI_M;
                const int j = idx - i * JACOBI_M;
                int p, q;
                jacobi_pair(sweep_round, i, &p, &q);
                const jreal cs = cosv[i];
                const jreal sn = sinv[i];
                const jreal ap = work[p * JACOBI_M + j];
                const jreal aq = work[q * JACOBI_M + j];
                work[p * JACOBI_M + j] = cs * ap - sn * aq;
                work[q * JACOBI_M + j] = sn * ap + cs * aq;
            }
            JACOBI_SYNC();

            // (c) right rotation, A <- A J, and the same on the accumulated
            //     eigenvectors.  Pairs own disjoint COLUMNS.
            for (int idx = tid; idx < (JACOBI_M / 2) * JACOBI_M;
                 idx += JACOBI_TPB) {
                const int i = idx / JACOBI_M;
                const int r = idx - i * JACOBI_M;
                int p, q;
                jacobi_pair(sweep_round, i, &p, &q);
                const jreal cs = cosv[i];
                const jreal sn = sinv[i];
                const jreal ap = work[r * JACOBI_M + p];
                const jreal aq = work[r * JACOBI_M + q];
                work[r * JACOBI_M + p] = cs * ap - sn * aq;
                work[r * JACOBI_M + q] = sn * ap + cs * aq;
                const jreal vp = vecs[r * JACOBI_M + p];
                const jreal vq = vecs[r * JACOBI_M + q];
                vecs[r * JACOBI_M + p] = cs * vp - sn * vq;
                vecs[r * JACOBI_M + q] = sn * vp + cs * vq;
            }
            JACOBI_SYNC();

            // (d) the rotation was constructed to annihilate (p, q); write
            //     the zero it was aiming for instead of the few ulp it
            //     actually landed on.  Without this the threshold test above
            //     can keep firing on rounding noise and the sweep loop never
            //     reports convergence.
            for (int i = tid; i < JACOBI_M / 2; i += JACOBI_TPB) {
                int p, q;
                jacobi_pair(sweep_round, i, &p, &q);
                work[p * JACOBI_M + q] = (jreal)0;
                work[q * JACOBI_M + p] = (jreal)0;
            }
            JACOBI_SYNC();
        }

        // A congruence by an orthogonal matrix is symmetric analytically; in
        // floating point the left and right passes see different intermediate
        // values and the two triangles drift apart by an ulp or so per round.
        // Averaging them once per sweep keeps the rotation formula -- which
        // reads only the upper triangle -- describing the matrix that is
        // actually there.  Each (r, c) with r < c is owned by exactly one
        // thread, which writes both copies, so there is no read/write race.
        for (int idx = tid; idx < JACOBI_M * JACOBI_M; idx += JACOBI_TPB) {
            const int r = idx / JACOBI_M;
            const int c = idx - r * JACOBI_M;
            if (r < c) {
                const jreal mean = (work[r * JACOBI_M + c]
                                    + work[c * JACOBI_M + r]) * (jreal)0.5;
                work[r * JACOBI_M + c] = mean;
                work[c * JACOBI_M + r] = mean;
            }
        }
        JACOBI_SYNC();

        sweeps = sweep + 1;
        if (flag[0] == 0) {
            converged = 1;
            break;
        }
        JACOBI_SYNC();
    }

    // ---- ascending order, ties broken by source column -----------------
    for (int i = tid; i < JACOBI_K; i += JACOBI_TPB) {
        diag[i] = work[i * JACOBI_M + i];
        perm[i] = i;
    }
    JACOBI_SYNC();
    // Odd-even transposition: a sorting network, so the comparisons and their
    // order are fixed by K alone and the permutation is reproducible.
    for (int phase = 0; phase < JACOBI_K; ++phase) {
        const int parity = phase & 1;
        for (int i = tid; i < (JACOBI_K + 1) / 2; i += JACOBI_TPB) {
            const int a = 2 * i + parity;
            const int b = a + 1;
            if (b < JACOBI_K) {
                const int pa = perm[a];
                const int pb = perm[b];
                const jreal ka = diag[pa];
                const jreal kb = diag[pb];
                if (kb < ka || (kb == ka && pb < pa)) {
                    perm[a] = pb;
                    perm[b] = pa;
                }
            }
        }
        JACOBI_SYNC();
    }

    // ---- canonical sign: largest-magnitude component positive ----------
    for (int c = tid; c < JACOBI_K; c += JACOBI_TPB) {
        const int sc = perm[c];
        int best = 0;
        jreal best_magnitude = fabs(vecs[sc]);
        for (int r = 1; r < JACOBI_K; ++r) {
            const jreal magnitude = fabs(vecs[r * JACOBI_M + sc]);
            if (magnitude > best_magnitude) {
                best_magnitude = magnitude;
                best = r;
            }
        }
        sign[c] = (vecs[best * JACOBI_M + sc] < (jreal)0)
            ? (jreal)(-1) : (jreal)1;
    }
    JACOBI_SYNC();

    for (int i = tid; i < JACOBI_K; i += JACOBI_TPB) {
        w_out[mat * (long long)JACOBI_K + i] = diag[perm[i]];
    }
    for (int idx = tid; idx < JACOBI_K * JACOBI_K; idx += JACOBI_TPB) {
        const int r = idx / JACOBI_K;
        const int c = idx - r * JACOBI_K;
        v_out[mat * (long long)JACOBI_K * JACOBI_K + idx] =
            sign[c] * vecs[r * JACOBI_M + perm[c]];
    }
    if (tid == 0) {
        status_out[mat] = converged ? sweeps : -1;
    }
}
