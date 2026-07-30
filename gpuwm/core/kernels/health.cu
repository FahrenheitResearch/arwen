// Combined integration-health reductions.  Two device launches produce one
// compact host record for u/w/theta maxima, w argmax, and real74 boundary /
// interior w maxima.  Max comparisons are exact for non-negative FP32 abs
// values; the lowest 64-bit flat index wins ties, matching argmax.  The
// compact float record stores that index as two exact uint32 payload words.
// Partial records are [u, w, theta, edge-w, interior-w, index-lo, index-hi,
// mask]; the final record drops the mask and keeps the first seven fields.

#define HEALTH_THREADS 256
#define HEALTH_FIELDS 8

static __device__ __forceinline__
void update_max(real value, unsigned long long index, real& current,
                unsigned long long& current_index)
{
    if (value > current || (value == current && index < current_index)) {
        current = value;
        current_index = index;
    }
}

extern "C" __global__
void health_partial(const real* __restrict__ u,
                    const real* __restrict__ w,
                    const real* __restrict__ thp,
                    real* __restrict__ partial,
                    unsigned long long nu, unsigned long long nw,
                    unsigned long long nth,
                    int width, int ny, int nx)
{
    real umax = 0.0f, wmax = 0.0f, thmax = 0.0f;
    real edge = 0.0f, interior = 0.0f;
    unsigned long long windex = 0xffffffffffffffffull;
    unsigned mask = 0u;
    unsigned long long start = ((unsigned long long)blockIdx.x * blockDim.x
                                + threadIdx.x);
    unsigned long long stride = ((unsigned long long)gridDim.x * blockDim.x);

    for (unsigned long long ix = start; ix < nu; ix += stride) {
        real value = fabsf(u[ix]);
        if (isnan(value)) mask |= 1u;
        else if (value > umax) umax = value;
    }
    for (unsigned long long ix = start; ix < nw; ix += stride) {
        real value = fabsf(w[ix]);
        bool boundary = false;
        if (width > 0) {
            unsigned long long c = ix % ((unsigned long long)ny * nx);
            int j = (int)(c / nx), i = (int)(c - (unsigned long long)j * nx);
            boundary = (j < width || j >= ny - width
                        || i < width || i >= nx - width);
        }
        if (isnan(value)) {
            mask |= 2u;
            if (width > 0) mask |= boundary ? 8u : 16u;
        } else {
            update_max(value, ix, wmax, windex);
            if (width > 0) {
                if (boundary) edge = value > edge ? value : edge;
                else interior = value > interior ? value : interior;
            }
        }
    }
    for (unsigned long long ix = start; ix < nth; ix += stride) {
        real value = fabsf(thp[ix]);
        if (isnan(value)) mask |= 4u;
        else if (value > thmax) thmax = value;
    }

    __shared__ real values[5][HEALTH_THREADS];
    __shared__ unsigned long long indices[HEALTH_THREADS];
    __shared__ unsigned masks[HEALTH_THREADS];
    int lane = threadIdx.x;
    values[0][lane] = umax;
    values[1][lane] = wmax;
    values[2][lane] = thmax;
    values[3][lane] = edge;
    values[4][lane] = interior;
    indices[lane] = windex;
    masks[lane] = mask;
    __syncthreads();

    for (int offset = HEALTH_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            values[0][lane] = values[0][lane + offset] > values[0][lane]
                              ? values[0][lane + offset] : values[0][lane];
            update_max(values[1][lane + offset], indices[lane + offset],
                       values[1][lane], indices[lane]);
            values[2][lane] = values[2][lane + offset] > values[2][lane]
                              ? values[2][lane + offset] : values[2][lane];
            values[3][lane] = values[3][lane + offset] > values[3][lane]
                              ? values[3][lane + offset] : values[3][lane];
            values[4][lane] = values[4][lane + offset] > values[4][lane]
                              ? values[4][lane + offset] : values[4][lane];
            masks[lane] |= masks[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        size_t out = (size_t)blockIdx.x * HEALTH_FIELDS;
        for (int field = 0; field < 5; ++field)
            partial[out + field] = values[field][0];
        partial[out + 5] = __uint_as_float((unsigned)indices[0]);
        partial[out + 6] = __uint_as_float((unsigned)(indices[0] >> 32));
        partial[out + 7] = __uint_as_float(masks[0]);
    }
}

extern "C" __global__
void health_final(const real* __restrict__ partial,
                  real* __restrict__ result, int nblocks)
{
    real maxima[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    unsigned long long windex = 0xffffffffffffffffull;
    unsigned mask = 0u;
    int lane = threadIdx.x;
    for (int block = lane; block < nblocks; block += HEALTH_THREADS) {
        size_t src = (size_t)block * HEALTH_FIELDS;
        maxima[0] = partial[src] > maxima[0] ? partial[src] : maxima[0];
        unsigned long long index = __float_as_uint(partial[src + 5]);
        index |= ((unsigned long long)__float_as_uint(partial[src + 6])) << 32;
        update_max(partial[src + 1], index, maxima[1], windex);
        for (int field = 2; field < 5; ++field)
            maxima[field] = partial[src + field] > maxima[field]
                            ? partial[src + field] : maxima[field];
        mask |= __float_as_uint(partial[src + 7]);
    }

    __shared__ real values[5][HEALTH_THREADS];
    __shared__ unsigned long long indices[HEALTH_THREADS];
    __shared__ unsigned masks[HEALTH_THREADS];
    for (int field = 0; field < 5; ++field)
        values[field][lane] = maxima[field];
    indices[lane] = windex;
    masks[lane] = mask;
    __syncthreads();

    for (int offset = HEALTH_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            values[0][lane] = values[0][lane + offset] > values[0][lane]
                              ? values[0][lane + offset] : values[0][lane];
            update_max(values[1][lane + offset], indices[lane + offset],
                       values[1][lane], indices[lane]);
            values[2][lane] = values[2][lane + offset] > values[2][lane]
                              ? values[2][lane + offset] : values[2][lane];
            values[3][lane] = values[3][lane + offset] > values[3][lane]
                              ? values[3][lane + offset] : values[3][lane];
            values[4][lane] = values[4][lane + offset] > values[4][lane]
                              ? values[4][lane + offset] : values[4][lane];
            masks[lane] |= masks[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        result[0] = (masks[0] & 1u) ? nanf("") : values[0][0];
        result[1] = (masks[0] & 2u) ? nanf("") : values[1][0];
        result[2] = (masks[0] & 4u) ? nanf("") : values[2][0];
        result[3] = (masks[0] & 8u) ? nanf("") : values[3][0];
        result[4] = (masks[0] & 16u) ? nanf("") : values[4][0];
        result[5] = __uint_as_float((unsigned)indices[0]);
        result[6] = __uint_as_float((unsigned)(indices[0] >> 32));
    }
}

// -------------------------------------------------------------------------
// Phase-5 full mutable-state health gate.
//
// Pointer/size/status metadata is represented as pairs of uint32 words in
// the model's sanctioned FP32 scratch pool.  One block owns each descriptor;
// all descriptors run in this single kernel launch.  The compact result is
// [uint64 ORed-status, uint64 packed-first-bad], where packed-first-bad is
// (field_id << 48) | flat_index.  atomicMin therefore gives deterministic
// field-first, then index-first attribution independent of block scheduling.
// -------------------------------------------------------------------------

#define VALIDATE_LOWER       1u
#define VALIDATE_UPPER       2u
#define VALIDATE_STRICT_LOW  4u
#define VALIDATE_ADD_AUX     8u
#define VALIDATE_AUX_LEVEL  16u
#define VALIDATE_INT32      32u

static __device__ __forceinline__
unsigned long long words_u64(const unsigned* words, int index)
{
    int offset = 2 * index;
    return ((unsigned long long)words[offset]
            | ((unsigned long long)words[offset + 1] << 32));
}

extern "C" __global__
void validate_full_state(
        const unsigned* __restrict__ field_pointer_words,
        const unsigned* __restrict__ auxiliary_pointer_words,
        const unsigned* __restrict__ field_size_words,
        const real* __restrict__ bounds,
        const unsigned* __restrict__ rule_flags,
        const unsigned* __restrict__ plane_sizes,
        const unsigned* __restrict__ status_bit_words,
        unsigned long long* __restrict__ result,
        int nfields)
{
    int field = blockIdx.x;
    if (field >= nfields) return;
    unsigned long long values_address = words_u64(
        field_pointer_words, field);
    const real* values = reinterpret_cast<const real*>(values_address);
    const int* integer_values = reinterpret_cast<const int*>(values_address);
    const real* auxiliary = reinterpret_cast<const real*>(
        words_u64(auxiliary_pointer_words, field));
    unsigned long long size = words_u64(field_size_words, field);
    unsigned flags = rule_flags[field];
    real lower = bounds[2 * field];
    real upper = bounds[2 * field + 1];
    bool failed = false;
    unsigned long long first = 0xffffffffffffffffull;
    for (unsigned long long index = threadIdx.x; index < size;
         index += blockDim.x) {
        real value = ((flags & VALIDATE_INT32)
                      ? (real)integer_values[index] : values[index]);
        if (flags & VALIDATE_ADD_AUX) {
            unsigned long long aux_index = index;
            if (flags & VALIDATE_AUX_LEVEL)
                aux_index = index / (unsigned long long)plane_sizes[field];
            value += auxiliary[aux_index];
        }
        bool bad = !isfinite(value);
        if (flags & VALIDATE_LOWER)
            bad |= ((flags & VALIDATE_STRICT_LOW)
                    ? value <= lower : value < lower);
        if (flags & VALIDATE_UPPER)
            bad |= value > upper;
        if (bad) {
            failed = true;
            first = index < first ? index : first;
        }
    }
    __shared__ unsigned long long first_by_thread[HEALTH_THREADS];
    __shared__ unsigned failed_by_thread[HEALTH_THREADS];
    int lane = threadIdx.x;
    first_by_thread[lane] = first;
    failed_by_thread[lane] = failed ? 1u : 0u;
    __syncthreads();
    for (int offset = HEALTH_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            unsigned long long other = first_by_thread[lane + offset];
            first_by_thread[lane] = other < first_by_thread[lane]
                                    ? other : first_by_thread[lane];
            failed_by_thread[lane] |= failed_by_thread[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && failed_by_thread[0]) {
        atomicOr(result, words_u64(status_bit_words, field));
        unsigned long long packed = ((unsigned long long)field << 48)
                                    | first_by_thread[0];
        atomicMin(result + 1, packed);
    }
}
