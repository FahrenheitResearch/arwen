// The integration-health reduction over ONE TILE'S INTERIOR, folded into the
// DOMAIN's coordinates.  A separate translation unit from health.cu on
// purpose: that file's source string is pinned by
// tests/test_mp8_frozen.py::test_every_frozen_kernel_module_is_unchanged,
// and an identical source string is what keeps its PTX, register allocation
// and FP contraction identical.  Only the PARTIAL record is produced here;
// health.cu's health_final folds it, unchanged, so the two paths cannot
// drift apart in how a record becomes a report.
//
// WHY THIS EXISTS
// ---------------
// Under [tiles] store = "host" the domain lives in a pinned host store
// and gpuwm.core.streaming.attach fills it with `pinned_copy`, which COPIES.
// From that instant the prepared DomainState the run loop still holds is a
// snapshot of t = 0 that no sweep ever writes again -- and
// gpuwm.runtime.integrate_prepared_case hands exactly that state to
// stability_report every dynamics substep.  So the NaN guard, the w_max
// monitor and the CFL monitor all observe a corpse: the snapshot is healthy,
// nan_free stays true forever, w_max freezes at its initial value, and a
// domain that went non-finite in the store completes "successfully" and
// writes a checkpoint recording that it did.  A silently disarmed NaN guard
// is indistinguishable from a well-behaved forecast.
//
// The defect is not the reduction.  It is a max/OR fold and those are
// associative, so it can be folded per tile; the only question was which
// memory it reads.  A tile buffer holds real domain data exactly once per
// sweep -- after its step, before its interior is scattered back -- and tile
// interiors PARTITION the domain (tilestream.spec.validate_plan proves it),
// so one record per tile through health_final reproduces the whole-domain
// record EXACTLY:
//
//   * max is associative and exact over the non-negative abs values;
//   * the NaN classes are a bitwise OR;
//   * the emitted w argmax is a DOMAIN flat index, so "lowest flat index
//     wins ties" resolves the same way whatever order the tiles are swept
//     in.  A tile-local index would have made the reported location depend
//     on the tiling, which is precisely the class of defect this project
//     keeps finding;
//   * boundary versus free interior is classified against the DOMAIN's
//     extents and the DOMAIN's spec_bdy_width.  A tile seam is not a
//     boundary, and a tile knows its own offset, so this needs no
//     information a tile cannot see -- the observers are foldable, not
//     structural.
//
// WHAT IT MAY NOT DO, AND WHY THE WINDOW IS NOT OPTIONAL
// ------------------------------------------------------
// It may not reduce over the whole BUFFER.  The halo is at least the
// per-step dependency radius (tilestream.harness.halo_radius), so the
// INTERIOR is bit-exact -- but the halo itself was stepped with insufficient
// neighbours and is progressively wrong towards its outer ring, which is why
// the sweep discards it.  Folding it in would let a tile's discarded halo
// raise a NaN the domain never had.  A false alarm on the run's only safety
// gate is no better than a missing one, and it would be worse to debug.
//
// u carries one extra COLUMN on the tile that owns the domain's closing face
// (TileSpec.owns_x_alias) and none on any other, which is what makes the
// tiles' u interiors add up to the domain's nx+1 columns exactly once.  Pass
// it as u_extra; getting it wrong drops or double-counts a column of the
// x-momentum maximum, silently.

#define HEALTH_TILE_THREADS 256
#define HEALTH_TILE_FIELDS 9

static __device__ __forceinline__
void tile_update_max(real value, unsigned long long index, real& current,
                     unsigned long long& current_index)
{
    if (value > current || (value == current && index < current_index)) {
        current = value;
        current_index = index;
    }
}

extern "C" __global__
void health_partial_tile(const real* __restrict__ u,
                         const real* __restrict__ w,
                         const real* __restrict__ thp,
                         const real* __restrict__ ph,
                         const real* __restrict__ phb,
                         real* __restrict__ partial,
                         int block_offset,
                         int bny, int bnx,     // buffer mass extents
                         int jb, int ib,       // interior origin in buffer
                         int iny, int inx,     // interior mass extents
                         int jd, int id,       // interior origin in domain
                         int dny, int dnx,     // domain mass extents
                         int nz, int u_extra, int phb_full,
                         int have_geopotential, int width, real gravity)
{
    real umax = 0.0f, wmax = 0.0f, thmax = 0.0f;
    real edge = 0.0f, interior = 0.0f, vertical_rate = 0.0f;
    unsigned long long windex = 0xffffffffffffffffull;
    unsigned mask = 0u;
    unsigned long long start = ((unsigned long long)blockIdx.x * blockDim.x
                                + threadIdx.x);
    unsigned long long stride = ((unsigned long long)gridDim.x * blockDim.x);

    const unsigned long long bplane = (unsigned long long)bny * bnx;
    const unsigned long long uplane = (unsigned long long)bny * (bnx + 1);
    const unsigned long long area = (unsigned long long)iny * inx;
    const unsigned long long uarea = (unsigned long long)iny * (inx + u_extra);

    // |u| over the interior columns, plus the closing face on the one tile
    // that owns it.  No index is kept: stability_report reports u_max only.
    for (unsigned long long t = start; t < (unsigned long long)nz * uarea;
         t += stride) {
        unsigned long long k = t / uarea;
        unsigned long long r = t - k * uarea;
        unsigned long long jj = r / (unsigned long long)(inx + u_extra);
        unsigned long long ii = r - jj * (unsigned long long)(inx + u_extra);
        real value = fabsf(u[k * uplane
                             + (unsigned long long)(jb + jj) * (bnx + 1)
                             + (unsigned long long)(ib + ii)]);
        if (isnan(value)) mask |= 1u;
        else if (value > umax) umax = value;
    }

    // |w| over the interior columns at every one of the nz+1 faces, with the
    // DOMAIN flat index and the DOMAIN's boundary classification.
    for (unsigned long long t = start;
         t < (unsigned long long)(nz + 1) * area; t += stride) {
        unsigned long long k = t / area;
        unsigned long long r = t - k * area;
        unsigned long long jj = r / (unsigned long long)inx;
        unsigned long long ii = r - jj * (unsigned long long)inx;
        real value = fabsf(w[k * bplane
                             + (unsigned long long)(jb + jj) * bnx
                             + (unsigned long long)(ib + ii)]);
        int j = jd + (int)jj, i = id + (int)ii;
        unsigned long long dindex = (k * (unsigned long long)dny
                                     + (unsigned long long)j)
                                    * (unsigned long long)dnx
                                    + (unsigned long long)i;
        bool boundary = false;
        if (width > 0) {
            boundary = (j < width || j >= dny - width
                        || i < width || i >= dnx - width);
        }
        if (isnan(value)) {
            mask |= 2u;
            if (width > 0) mask |= boundary ? 8u : 16u;
        } else {
            tile_update_max(value, dindex, wmax, windex);
            if (width > 0) {
                if (boundary) edge = value > edge ? value : edge;
                else interior = value > interior ? value : interior;
            }
        }
    }

    // |theta'| over the interior mass cells.
    for (unsigned long long t = start; t < (unsigned long long)nz * area;
         t += stride) {
        unsigned long long k = t / area;
        unsigned long long r = t - k * area;
        unsigned long long jj = r / (unsigned long long)inx;
        unsigned long long ii = r - jj * (unsigned long long)inx;
        real value = fabsf(thp[k * bplane
                               + (unsigned long long)(jb + jj) * bnx
                               + (unsigned long long)(ib + ii)]);
        if (isnan(value)) mask |= 4u;
        else if (value > thmax) thmax = value;
    }

    // One mass cell owns dz between its own lower/upper geopotential faces
    // and the w value on its upper face, exactly as health_partial pairs
    // them: reducing the ratio preserves co-location, so a strong upper
    // updraft can never be paired with an unrelated thin surface layer.
    if (have_geopotential) {
        for (unsigned long long t = start; t < (unsigned long long)nz * area;
             t += stride) {
            unsigned long long k = t / area;
            unsigned long long r = t - k * area;
            unsigned long long jj = r / (unsigned long long)inx;
            unsigned long long ii = r - jj * (unsigned long long)inx;
            unsigned long long bix = k * bplane
                                     + (unsigned long long)(jb + jj) * bnx
                                     + (unsigned long long)(ib + ii);
            unsigned long long upper = bix + bplane;
            real base_lower = phb_full ? phb[bix] : phb[k];
            real base_upper = phb_full ? phb[upper] : phb[k + 1];
            real dz = ((ph[upper] + base_upper) - (ph[bix] + base_lower))
                      / gravity;
            real speed = fabsf(w[upper]);
            if (!isfinite(dz) || dz <= 0.0f || !isfinite(speed)) {
                mask |= 32u;
            } else {
                real rate = speed / dz;
                vertical_rate = rate > vertical_rate ? rate : vertical_rate;
            }
        }
    }

    __shared__ real values[6][HEALTH_TILE_THREADS];
    __shared__ unsigned long long indices[HEALTH_TILE_THREADS];
    __shared__ unsigned masks[HEALTH_TILE_THREADS];
    int lane = threadIdx.x;
    values[0][lane] = umax;
    values[1][lane] = wmax;
    values[2][lane] = thmax;
    values[3][lane] = edge;
    values[4][lane] = interior;
    values[5][lane] = vertical_rate;
    indices[lane] = windex;
    masks[lane] = mask;
    __syncthreads();

    for (int offset = HEALTH_TILE_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            values[0][lane] = values[0][lane + offset] > values[0][lane]
                              ? values[0][lane + offset] : values[0][lane];
            tile_update_max(values[1][lane + offset], indices[lane + offset],
                            values[1][lane], indices[lane]);
            values[2][lane] = values[2][lane + offset] > values[2][lane]
                              ? values[2][lane + offset] : values[2][lane];
            values[3][lane] = values[3][lane + offset] > values[3][lane]
                              ? values[3][lane + offset] : values[3][lane];
            values[4][lane] = values[4][lane + offset] > values[4][lane]
                              ? values[4][lane + offset] : values[4][lane];
            values[5][lane] = values[5][lane + offset] > values[5][lane]
                              ? values[5][lane + offset] : values[5][lane];
            masks[lane] |= masks[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        size_t out = (size_t)(block_offset + blockIdx.x) * HEALTH_TILE_FIELDS;
        for (int field = 0; field < 6; ++field)
            partial[out + field] = values[field][0];
        partial[out + 6] = __uint_as_float((unsigned)indices[0]);
        partial[out + 7] = __uint_as_float((unsigned)(indices[0] >> 32));
        partial[out + 8] = __uint_as_float(masks[0]);
    }
}
