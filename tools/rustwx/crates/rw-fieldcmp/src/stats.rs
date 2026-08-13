//! The judge's arithmetic, written so a reader can check it against the
//! reference implementation line by line.
//!
//! This crate is a verification instrument, so its summary statistics must
//! agree with the reference judge's rather than merely be defensible.  Four
//! choices here exist only for that reason and are called out where they are
//! made:
//!
//! * `tree_sum_by` reproduces the block-of-128, eight-accumulator summation
//!   the reference stack's kernel uses instead of a naive running total, so a
//!   mean over a quarter-million cells lands on the same bits rather than
//!   merely the same first ten digits;
//! * `REDUCTION_BUFFER` reproduces the *buffering around* that kernel.  A
//!   whole-array reduction is a sequence of trees, not one tree, and past
//!   about a million elements the difference is visible in the last bit;
//! * [`percentile_linear`] reproduces the reference `linear` quantile: the
//!   same virtual-index expression, the same out-of-range index clamping, and
//!   the same two-sided interpolation that switches formula at gamma 0.5;
//! * [`accumulation_sum_f32`] reproduces a *single-precision* accumulation,
//!   because the reference judge sums the accumulation fields in the on-disk
//!   dtype rather than promoting first.  It is the one place where the
//!   reference is the less accurate of the two, so the f64 value is carried
//!   beside it rather than instead of it.

use rayon::prelude::*;

/// Block size at which the reference summation stops recursing and switches
/// to eight interleaved accumulators.
const PAIRWISE_BLOCK: usize = 128;

/// Elements the reference stack reduces per pass over a large array.
///
/// The pairwise tree below is what the reference's summation kernel
/// implements, but it is not what a whole-array reduction runs: the reduction
/// hands its kernel one buffer of elements at a time and adds each buffer's
/// total into a running scalar, so an array longer than one buffer is summed
/// as a *sequence* of trees rather than as one tree.  Above about a million
/// elements the two arrangements part company in the last bit or two, which
/// is exactly the size the paired-run metrics work at.
///
/// This value was measured against the reference rather than read off a
/// constant: on real fields and on random data, at sizes from four thousand
/// to two and a quarter million, in both precisions, summing in runs of 8192
/// reproduces the reference's answer at every size tested, and the single
/// tree does not.  It is the reference stack's default reduction buffer, and
/// it is a *settable* default -- a caller that changes it changes the last
/// bit of every large sum, in the reference and therefore here too.
const REDUCTION_BUFFER: usize = 8192;

/// Blocks below which computing the block totals concurrently costs more than
/// it saves.
const PARALLEL_BLOCK_FLOOR: usize = 8;

/// One buffer's worth of the reference's pairwise tree, in double precision.
///
/// Under 8 elements it is a plain running total; up to one block it keeps
/// eight accumulators and combines them as a balanced tree; above that it
/// splits at the largest multiple of 8 not past the midpoint and recurses.
fn tree_sum_by<F>(start: usize, n: usize, value: &F) -> f64
where
    F: Fn(usize) -> f64,
{
    if n < 8 {
        let mut total = 0.0f64;
        for offset in 0..n {
            total += value(start + offset);
        }
        total
    } else if n <= PAIRWISE_BLOCK {
        let mut acc = [
            value(start),
            value(start + 1),
            value(start + 2),
            value(start + 3),
            value(start + 4),
            value(start + 5),
            value(start + 6),
            value(start + 7),
        ];
        let whole = n - (n % 8);
        let mut i = 8usize;
        while i < whole {
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot += value(start + i + k);
            }
            i += 8;
        }
        let mut total =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        while i < n {
            total += value(start + i);
            i += 1;
        }
        total
    } else {
        let mut half = n / 2;
        half -= half % 8;
        tree_sum_by(start, half, value) + tree_sum_by(start + half, n - half, value)
    }
}

/// Sum `count` values produced on demand, the way the reference reduces an
/// array: one pairwise tree per [`REDUCTION_BUFFER`] elements, and the block
/// totals added into a running scalar in order.
///
/// The reference forms its big sums over a temporary -- `np.sum(d * d)`
/// materialises the squares before summing them -- and the reduction is laid
/// over that temporary's layout.  Generating each element from its index
/// instead reproduces the same blocks in the same order without building the
/// temporary, which is what keeps a difference-of-differences over ten
/// million cells from needing its own copy of them.
///
/// The block totals are computed concurrently and then added in order, which
/// is a rearrangement of *when* the work happens and not of which additions
/// happen: the answer is the same bits as the serial pass.
pub fn pairwise_sum_by<F>(count: usize, value: &F) -> f64
where
    F: Fn(usize) -> f64 + Sync,
{
    let blocks = count.div_ceil(REDUCTION_BUFFER);
    if blocks <= 1 {
        return tree_sum_by(0, count, value);
    }
    let block_total = |block: usize| {
        let start = block * REDUCTION_BUFFER;
        tree_sum_by(start, REDUCTION_BUFFER.min(count - start), value)
    };
    let totals: Vec<f64> = if blocks >= PARALLEL_BLOCK_FLOOR {
        (0..blocks).into_par_iter().map(block_total).collect()
    } else {
        (0..blocks).map(block_total).collect()
    };
    let mut total = 0.0f64;
    for subtotal in totals {
        total += subtotal;
    }
    total
}

/// Sum a slice the way the reference reduces one.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    pairwise_sum_by(values.len(), &|index| values[index])
}

/// The same reduction carried out in single precision, for parity with a
/// reference that sums accumulation fields in their on-disk dtype.
///
/// The inputs arrive as `f64` because the reader promotes on read, but each
/// one is an exactly-representable `f32`, so narrowing back is lossless and
/// only the accumulation order and precision are being reproduced.  The
/// buffer length is a count of elements, not of bytes, so it is the same
/// 8192 here as in double precision -- measured, not assumed.
pub fn accumulation_sum_f32(values: &[f64]) -> f32 {
    fn tree(values: &[f64]) -> f32 {
        let n = values.len();
        if n < 8 {
            let mut total = 0.0f32;
            for &value in values {
                total += value as f32;
            }
            total
        } else if n <= PAIRWISE_BLOCK {
            let mut acc = [
                values[0] as f32,
                values[1] as f32,
                values[2] as f32,
                values[3] as f32,
                values[4] as f32,
                values[5] as f32,
                values[6] as f32,
                values[7] as f32,
            ];
            let whole = n - (n % 8);
            let mut i = 8usize;
            while i < whole {
                for (k, slot) in acc.iter_mut().enumerate() {
                    *slot += values[i + k] as f32;
                }
                i += 8;
            }
            let mut total =
                ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            while i < n {
                total += values[i] as f32;
                i += 1;
            }
            total
        } else {
            let mut half = n / 2;
            half -= half % 8;
            tree(&values[..half]) + tree(&values[half..])
        }
    }
    let mut total = 0.0f32;
    for block in values.chunks(REDUCTION_BUFFER) {
        total += tree(block);
    }
    total
}

/// Arithmetic mean over the flattened field.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(values) / values.len() as f64
}

/// Maximum that propagates a non-finite operand rather than skipping it, so
/// a field that has gone bad reports as bad instead of reporting its largest
/// surviving cell.
pub fn propagating_max(values: &[f64]) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for &value in values {
        if value.is_nan() {
            return f64::NAN;
        }
        if value > best {
            best = value;
        }
    }
    best
}

/// Sort ascending with a total order, which places any NaN last exactly as
/// the reference sort does.
pub fn sorted_ascending(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    sorted
}

/// Which two order statistics a quantile is read from, and how far between
/// them it sits.
///
/// Splitting the index arithmetic out of the lookup is what lets the same
/// definition serve both access paths: the readable one that indexes a fully
/// sorted array, and the fast one that selects only the handful of order
/// statistics the three quantiles actually touch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantilePlan {
    pub low: usize,
    pub high: usize,
    pub gamma: f64,
}

/// Plotting-position parameters of the reference `linear` quantile method.
/// Both being one is what makes the virtual index fall on `q * (n - 1)`.
const QUANTILE_ALPHA: f64 = 1.0;
const QUANTILE_BETA: f64 = 1.0;

/// Work out where a `linear`-method quantile falls in a field of `count`
/// values.  `percent` is on 0..=100.
pub fn quantile_plan(count: usize, percent: f64) -> QuantilePlan {
    let n = count as f64;
    let quantile = percent / 100.0;
    // n*q + (alpha + q*(1 - alpha - beta)) - 1, evaluated in that order.
    let virtual_index = n * quantile
        + (QUANTILE_ALPHA + quantile * (1.0 - QUANTILE_ALPHA - QUANTILE_BETA))
        - 1.0;

    // Out-of-range virtual indices collapse onto one end.  The reference
    // marks the top end with -1 and lets it wrap to the last element, and
    // computes gamma against that marker rather than the wrapped index;
    // both ends are degenerate, so gamma has no effect on the result there.
    let previous = if virtual_index.is_nan() || virtual_index >= n - 1.0 {
        -1.0
    } else if virtual_index < 0.0 {
        0.0
    } else {
        virtual_index.floor()
    };
    let next = if previous < 0.0 || virtual_index < 0.0 {
        previous
    } else {
        previous + 1.0
    };

    let resolve = |index: f64| -> usize {
        if index < 0.0 {
            (n + index) as usize
        } else {
            index as usize
        }
    };
    QuantilePlan {
        low: resolve(previous),
        high: resolve(next),
        gamma: virtual_index - previous,
    }
}

/// Interpolate between two order statistics.
///
/// The formula switches at gamma 0.5 so the result is anchored to whichever
/// endpoint it is nearer, which is what keeps a quantile that lands exactly
/// on a data point returning that point rather than a rounded neighbour.
pub fn quantile_interpolate(low: f64, high: f64, gamma: f64) -> f64 {
    let span = high - low;
    if gamma >= 0.5 {
        high - span * (1.0 - gamma)
    } else {
        low + span * gamma
    }
}

/// The reference `linear` quantile, evaluated on an already-sorted slice.
pub fn percentile_linear(sorted: &[f64], percent: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let plan = quantile_plan(sorted.len(), percent);
    quantile_interpolate(sorted[plan.low], sorted[plan.high], plan.gamma)
}

/// Pick out the values at `indices` (strictly ascending) by repeated
/// selection, permuting `values` in the process.
///
/// A full sort would answer the same question, but the judge only ever asks
/// for six order statistics out of a quarter-million, and selection is
/// linear where sorting is not.
pub fn order_statistics(values: &mut [f64], indices: &[usize]) -> Vec<f64> {
    let mut found = Vec::with_capacity(indices.len());
    let mut consumed = 0usize;
    let mut rest: &mut [f64] = values;
    for &index in indices {
        let local = index - consumed;
        let (_, at, tail) = rest.select_nth_unstable_by(local, |a, b| a.total_cmp(b));
        found.push(*at);
        rest = tail;
        consumed = index + 1;
    }
    found
}

/// The five-number summary the judge prints for every named field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub mean: f64,
    pub p10: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
}

impl Summary {
    /// Summarise a flattened field, permuting it in the process.
    ///
    /// The mean and the maximum are taken first, while the field is still in
    /// its original order; the three percentiles then share one round of
    /// selection over the six order statistics they need between them.
    pub fn of_in_place(values: &mut [f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: f64::NAN,
                p10: f64::NAN,
                p50: f64::NAN,
                p99: f64::NAN,
                max: f64::NAN,
            };
        }
        let mean = mean(values);
        let max = propagating_max(values);

        let plans = [
            quantile_plan(values.len(), 10.0),
            quantile_plan(values.len(), 50.0),
            quantile_plan(values.len(), 99.0),
        ];
        let mut wanted: Vec<usize> = plans.iter().flat_map(|p| [p.low, p.high]).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let found = order_statistics(values, &wanted);
        let at = |index: usize| found[wanted.binary_search(&index).expect("planned index")];
        let value = |plan: &QuantilePlan| {
            quantile_interpolate(at(plan.low), at(plan.high), plan.gamma)
        };

        Self {
            mean,
            p10: value(&plans[0]),
            p50: value(&plans[1]),
            p99: value(&plans[2]),
            max,
        }
    }

    /// Summarise without disturbing the caller's field.
    pub fn of(values: &[f64]) -> Self {
        let mut scratch = values.to_vec();
        Self::of_in_place(&mut scratch)
    }

    /// Element-wise difference of two summaries, left minus right.
    pub fn minus(&self, other: &Summary) -> Summary {
        Summary {
            mean: self.mean - other.mean,
            p10: self.p10 - other.p10,
            p50: self.p50 - other.p50,
            p99: self.p99 - other.p99,
            max: self.max - other.max,
        }
    }
}

/// Pooled RMSE over samples that need not be the same size.
///
/// Squares and element counts accumulate across the samples and the square
/// root is taken once, so a loud window is not averaged away by a run of
/// quiet ones.  The two-level shape is the reference's and is load-bearing:
/// each sample's squares are summed pairwise, and the per-sample totals are
/// then added into one running total in sample order.  Summing the samples
/// pairwise as well would be the more accurate arrangement and the wrong
/// answer for an instrument whose job is to agree.
#[derive(Debug, Clone, Copy, Default)]
pub struct PooledRmse {
    total: f64,
    count: usize,
}

impl PooledRmse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one sample's pairwise sum of squares and its element count.
    pub fn push(&mut self, sum_of_squares: f64, count: usize) {
        self.total += sum_of_squares;
        self.count += count;
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// The pooled root mean square, or `None` if no sample was ever pushed --
    /// which the reference treats as an error rather than as a zero.
    pub fn finish(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some((self.total / self.count as f64).sqrt())
        }
    }
}

/// Count of values that are neither finite nor merely large: the judge does
/// not refuse on them, but it must never report a table that hides them.
pub fn nonfinite_count(values: &[f64]) -> usize {
    values.iter().filter(|v| !v.is_finite()).count()
}

/// Column block a composite worker takes at a time.  Sized so its slice of
/// the output plane stays in cache while the worker sweeps every level.
const COMPOSITE_BLOCK: usize = 8192;

/// Fold one level of a volume into a running per-column maximum.
///
/// NaN wins over any value, so a column that contains one bad cell reports
/// as bad rather than reporting its largest good cell.
pub fn fold_column_max(composite: &mut [f64], level: &[f64]) {
    for (slot, &value) in composite.iter_mut().zip(level) {
        if slot.is_nan() {
            continue;
        }
        if value.is_nan() {
            *slot = f64::NAN;
        } else if value > *slot {
            *slot = value;
        }
    }
}

/// Reduce a `[level, y, x]` volume to its per-column maximum, the composite
/// of a reflectivity field.  Propagates NaN per column.
///
/// The output plane is cut into blocks and each block sweeps the whole level
/// axis on its own worker.  A volume is the largest thing this crate reads,
/// so the reduction over it is worth spreading rather than leaving on the
/// thread that happened to read it.
pub fn column_max(volume: &[f64], levels: usize, plane: usize) -> Vec<f64> {
    let mut composite = vec![f64::NEG_INFINITY; plane];
    composite
        .par_chunks_mut(COMPOSITE_BLOCK)
        .enumerate()
        .for_each(|(block, out)| {
            let start = block * COMPOSITE_BLOCK;
            let width = out.len();
            for level in 0..levels {
                let base = level * plane + start;
                fold_column_max(out, &volume[base..base + width]);
            }
        });
    composite
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle values produced by the reference stack on
    /// `numpy.arange(37, dtype=numpy.float32).astype(numpy.float64) * 0.37`.
    fn ramp() -> Vec<f64> {
        (0..37).map(|i| (i as f32 as f64) * 0.37).collect()
    }

    #[test]
    fn pairwise_sum_matches_the_naive_sum_on_an_exact_ramp() {
        // Powers of two are summed exactly under any ordering, so this test
        // pins the traversal rather than the rounding.
        let values: Vec<f64> = (0..1000).map(|i| ((i % 8) as f64) * 0.5).collect();
        let naive: f64 = values.iter().sum();
        assert_eq!(pairwise_sum(&values), naive);
    }

    #[test]
    fn pairwise_sum_beats_the_naive_sum_where_they_differ() {
        // A long run of small addends after one large one: the naive total
        // loses the tail, the pairwise total keeps most of it.
        let mut values = vec![1.0e8f64];
        values.extend(std::iter::repeat_n(1.0f64, 4096));
        let naive = values.iter().fold(0.0f64, |a, b| a + b);
        let exact = 1.0e8 + 4096.0;
        assert!((pairwise_sum(&values) - exact).abs() <= (naive - exact).abs());
    }

    /// The reduction is a sequence of trees, not one tree, and past one
    /// buffer the two disagree.  This is the guard on the finding: if someone
    /// "simplifies" the buffering away, the sums silently drift from the
    /// reference by a bit or two on every array the metrics actually use.
    #[test]
    fn a_long_array_is_summed_as_a_sequence_of_trees_not_one_tree() {
        fn one_tree(values: &[f64]) -> f64 {
            let n = values.len();
            if n <= PAIRWISE_BLOCK {
                return pairwise_sum(values);
            }
            let mut half = n / 2;
            half -= half % 8;
            one_tree(&values[..half]) + one_tree(&values[half..])
        }
        // Comparable magnitudes with noisy low bits: the arrangement of the
        // additions is then visible in the result, where a run of rapidly
        // shrinking terms would hide it under the leading one.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64) / ((1u64 << 53) as f64) + 1.0
        };
        let values: Vec<f64> = (0..1_000_003).map(|_| next()).collect();

        // Inside one buffer the two arrangements are the same arrangement,
        // which is why the difference stayed invisible until the metrics
        // started summing millions of cells.
        let short = &values[..REDUCTION_BUFFER];
        assert_eq!(pairwise_sum(short), one_tree(short));

        // Past it they part company.  Whether a given length happens to land
        // on the same double is luck, so the guard is that some length does
        // not -- not that a particular one does.
        let lengths = [8193usize, 12_345, 24_713, 100_000, 1_000_003];
        assert!(
            lengths
                .iter()
                .any(|&n| pairwise_sum(&values[..n]) != one_tree(&values[..n])),
            "the buffered reduction is indistinguishable from one tree, so \
             either the buffer length is wrong or it has been optimised away"
        );
    }

    /// The mapped and the slice forms must be the same reduction, or a
    /// difference-of-differences would be summed differently from a plain
    /// field and only one of them could match the reference.
    #[test]
    fn the_mapped_and_slice_reductions_agree_at_every_length() {
        let values: Vec<f64> = (0..2 * REDUCTION_BUFFER + 61)
            .map(|i| ((i * 7919) % 1000) as f64 * 0.001 + 1.0 / (i as f64 + 1.0))
            .collect();
        for count in [0usize, 1, 7, 8, 9, 128, 129, 8191, 8192, 8193, 16384, 16445] {
            let slice = &values[..count];
            assert_eq!(
                pairwise_sum(slice),
                pairwise_sum_by(count, &|index| values[index]),
                "at n={count}"
            );
        }
    }

    /// Concurrency must not be visible in the answer, only in the clock.
    #[test]
    fn the_block_totals_add_up_the_same_however_many_workers_ran() {
        let values: Vec<f64> = (0..40 * REDUCTION_BUFFER)
            .map(|i| 1.0 / (i as f64 + 1.0))
            .collect();
        let concurrent = pairwise_sum(&values);
        let mut serial = 0.0f64;
        for block in values.chunks(REDUCTION_BUFFER) {
            serial += tree_sum_by(0, block.len(), &|index| block[index]);
        }
        assert_eq!(concurrent, serial);
    }

    #[test]
    fn percentile_interpolates_between_neighbours() {
        let sorted = sorted_ascending(&[0.0, 1.0, 2.0, 3.0]);
        // virtual index for p50 over four points is 1.5, halfway between 1
        // and 2, which is where the interpolation formula switches branch.
        assert_eq!(percentile_linear(&sorted, 50.0), 1.5);
        assert_eq!(percentile_linear(&sorted, 0.0), 0.0);
        assert_eq!(percentile_linear(&sorted, 100.0), 3.0);
    }

    #[test]
    fn percentile_on_the_ramp_matches_the_reference() {
        let sorted = sorted_ascending(&ramp());
        // Oracle: numpy.percentile(a, [10, 50, 99]) on the same ramp.
        assert!((percentile_linear(&sorted, 10.0) - 1.3320000000000003).abs() < 1e-15);
        assert!((percentile_linear(&sorted, 50.0) - 6.66).abs() < 1e-15);
        assert!((percentile_linear(&sorted, 99.0) - 13.1868).abs() < 1e-13);
    }

    /// The selection path and the sorted path must agree exactly, on field
    /// sizes either side of every branch in the index arithmetic.  This is
    /// the guard on the optimisation: the readable implementation stays the
    /// definition, and the fast one has to match it bit for bit.
    #[test]
    fn selection_and_sorting_agree_on_every_field_size() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 400.0 - 200.0
        };
        for count in [1usize, 2, 3, 4, 7, 8, 9, 100, 101, 999, 5000, 50_000] {
            let values: Vec<f64> = (0..count).map(|_| next()).collect();
            let sorted = sorted_ascending(&values);
            let fast = Summary::of(&values);
            assert_eq!(fast.p10, percentile_linear(&sorted, 10.0), "p10 at n={count}");
            assert_eq!(fast.p50, percentile_linear(&sorted, 50.0), "p50 at n={count}");
            assert_eq!(fast.p99, percentile_linear(&sorted, 99.0), "p99 at n={count}");
            assert_eq!(fast.max, *sorted.last().expect("non-empty"), "max at n={count}");
        }
    }

    /// Summarising must not leave the caller's field permuted, or the
    /// domain sums taken afterwards would be summing a different ordering
    /// than the reference does.
    #[test]
    fn summarising_leaves_the_callers_field_in_order() {
        let values: Vec<f64> = (0..1000).map(|i| ((i * 37) % 1000) as f64).collect();
        let before = values.clone();
        let _ = Summary::of(&values);
        assert_eq!(values, before);
    }

    #[test]
    fn summary_of_a_constant_field_is_that_constant() {
        let values = vec![7.25f64; 5000];
        let summary = Summary::of(&values);
        assert_eq!(summary.mean, 7.25);
        assert_eq!(summary.p10, 7.25);
        assert_eq!(summary.p50, 7.25);
        assert_eq!(summary.p99, 7.25);
        assert_eq!(summary.max, 7.25);
    }

    #[test]
    fn max_propagates_a_bad_cell_instead_of_hiding_it() {
        let values = vec![1.0, 2.0, f64::NAN, 3.0];
        assert!(propagating_max(&values).is_nan());
        assert_eq!(nonfinite_count(&values), 1);
    }

    #[test]
    fn column_max_composites_over_the_level_axis() {
        // Two levels of a 2x2 plane; the composite takes the larger cell.
        let volume = vec![1.0, 9.0, 3.0, 4.0, 5.0, 2.0, 7.0, 0.0];
        assert_eq!(column_max(&volume, 2, 4), vec![5.0, 9.0, 7.0, 4.0]);
    }

    #[test]
    fn single_precision_accumulation_reproduces_its_own_loss() {
        // 200_000 copies of 0.1 exceed f32's ability to keep the tail; the
        // f64 sum and the f32 sum must therefore disagree, which is exactly
        // the deviation this function exists to reproduce.
        let values = vec![0.1f32 as f64; 200_000];
        // 0.1 is not representable in binary, so the exact total is
        // 200_000 * (the f32 nearest 0.1), which the wide sum reproduces to
        // within a rounding of itself.
        let exact = 200_000.0 * (0.1f32 as f64);
        let wide = pairwise_sum(&values);
        let narrow = accumulation_sum_f32(&values) as f64;
        assert!((wide - exact).abs() < 1.0e-9);
        assert!(narrow != wide);
        assert!((narrow - wide).abs() / wide < 1.0e-5);
    }
}
