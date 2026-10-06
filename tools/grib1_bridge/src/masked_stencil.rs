//! The native HRRR route's land-only bilinear stencil: soil temperature and
//! soil moisture from HRRR's own land, in float64 weights, and its apply.
//!
//! This is the arithmetic `gpuwm.ingest.hrrr._build_masked_bilinear_stencil`
//! and `_CpuMaskedBilinearStencil.apply` ran as single-core NumPy (and a
//! SciPy k-d tree for a donor past the radius), moved here statement for
//! statement.  The NumPy builder is kept as a test oracle only
//! (`gpuwm/verify/hrrr_stencil_oracle.py`), and every index, every weight
//! and every number of the report must equal it byte for byte:
//!
//! * the corner weights, their renormalisation and the weight sums keep
//!   NumPy's operand order, and nothing here fuses a multiply and an add;
//! * a squared distance is `dx * dx + dy * dy` in float64, as NumPy's
//!   `array ** 2` computes it;
//! * the donor inside the radius is the first strictly nearer cell in the
//!   scan order rows then columns, and the donor past it the nearest valid
//!   cell of the window, ties to the lowest row and then the lowest column
//!   (the k-d tree only proposed candidates there; the choice was this
//!   comparison);
//! * every output element depends only on its own target, the report's
//!   counts are integer sums and its extremes are minima and maxima, so no
//!   result depends on the worker count or the schedule.

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};

/// `fallback_radius` is negative.
pub const ERR_RADIUS: i32 = 20;
/// `closed_edges` names an edge that is not a window edge.
pub const ERR_EDGES: i32 = 21;
/// A target's bilinear corners leave the source window.
pub const ERR_OFF_WINDOW: i32 = 22;
/// Some target has no donor the window can vouch for; the unresolved
/// targets and the worst distance are written.
pub const ERR_UNRESOLVED: i32 = 23;

pub const EDGE_WEST: u32 = 1;
pub const EDGE_EAST: u32 = 2;
pub const EDGE_SOUTH: u32 = 4;
pub const EDGE_NORTH: u32 = 8;

/// Report count slots, in this order.
pub const COUNT_SOURCE_VALID: usize = 0;
pub const COUNT_TARGET_APPLY: usize = 1;
pub const COUNT_DIRECT: usize = 2;
pub const COUNT_RENORMALIZED: usize = 3;
pub const COUNT_FALLBACK: usize = 4;
pub const COUNT_CROSS_SURFACE: usize = 5;
pub const COUNT_NEGATIVE_WEIGHT: usize = 6;
pub const COUNT_DISTANT: usize = 7;
pub const COUNT_UNRESOLVED: usize = 8;
pub const COUNT_LISTED: usize = 9;
pub const COUNT_SLOTS: usize = 10;

/// Report real slots: the farthest donor, the least nonzero raw support
/// (NaN when no target is direct), the least and greatest weight sum, the
/// worst unresolved distance (NaN when none).
pub const REAL_MAX_DISTANCE: usize = 0;
pub const REAL_MIN_RAW_SUPPORT: usize = 1;
pub const REAL_SUM_MIN: usize = 2;
pub const REAL_SUM_MAX: usize = 3;
pub const REAL_WORST_DISTANCE: usize = 4;
pub const REAL_SLOTS: usize = 5;

/// Flag slots: the weights are finite, non-negative and sum to one within
/// 2e-15; the window holds a valid cell.
pub const FLAG_WEIGHTS_CONVEX: usize = 0;
pub const FLAG_ANY_VALID: usize = 1;
pub const FLAG_SLOTS: usize = 2;

/// Targets per unit of parallel work, pulled one at a time so a chunk of
/// expensive donor searches does not hold the others up.
const CHUNK: usize = 2048;

/// A fallback target's state.
const STATE_NONE: u8 = 0;
const STATE_WITHIN: u8 = 1;
const STATE_BEYOND: u8 = 2;
const STATE_UNRESOLVED: u8 = 3;

struct Window<'a> {
    valid: &'a [u8],
    ny: usize,
    nx: usize,
}

impl Window<'_> {
    #[inline]
    fn valid(&self, row: i64, column: i64) -> bool {
        row >= 0
            && column >= 0
            && (row as u64) < self.ny as u64
            && (column as u64) < self.nx as u64
            && self.valid[row as usize * self.nx + column as usize] != 0
    }
}

#[inline]
fn distance2(column: i64, row: i64, x: f64, y: f64) -> f64 {
    let dx = column as f64 - x;
    let dy = row as f64 - y;
    dx * dx + dy * dy
}

/// The radius scan: the first strictly nearer valid cell within the disk,
/// rows then columns, as the NumPy scan's offset loops meet them.
fn radius_donor(w: &Window, x: f64, y: f64, radius: i64) -> Option<(i64, i64, f64)> {
    let center_x = (x + 0.5).floor() as i64;
    let center_y = (y + 0.5).floor() as i64;
    let limit = (radius * radius) as f64;
    let mut best: Option<(i64, i64, f64)> = None;
    let mut best_d2 = f64::INFINITY;
    for offset_y in -radius..=radius {
        let row = center_y + offset_y;
        for offset_x in -radius..=radius {
            let column = center_x + offset_x;
            if !w.valid(row, column) {
                continue;
            }
            let d2 = distance2(column, row, x, y);
            if d2 <= limit && d2 < best_d2 {
                best_d2 = d2;
                best = Some((row, column, d2));
            }
        }
    }
    best
}

/// The nearest valid cell of the window, ties to the lowest row and then
/// the lowest column: square rings around the nearest cell, stopped once
/// no farther ring can hold a cell as near as the best found.
fn nearest_valid(w: &Window, x: f64, y: f64) -> Option<(i64, i64, f64)> {
    let center_x = (x + 0.5).floor() as i64;
    let center_y = (y + 0.5).floor() as i64;
    let span = w.ny.max(w.nx) as i64 + 1;
    let mut best: Option<(i64, i64, f64)> = None;
    let consider = |row: i64, column: i64, best: &mut Option<(i64, i64, f64)>| {
        if !w.valid(row, column) {
            return;
        }
        let d2 = distance2(column, row, x, y);
        let better = match *best {
            None => true,
            Some((best_row, best_column, best_d2)) => {
                d2 < best_d2 || (d2 == best_d2 && (row, column) < (best_row, best_column))
            }
        };
        if better {
            *best = Some((row, column, d2));
        }
    };
    for ring in 0..=span {
        if ring == 0 {
            consider(center_y, center_x, &mut best);
        } else {
            for column in center_x - ring..=center_x + ring {
                consider(center_y - ring, column, &mut best);
                consider(center_y + ring, column, &mut best);
            }
            for row in center_y - ring + 1..=center_y + ring - 1 {
                consider(row, center_x - ring, &mut best);
                consider(row, center_x + ring, &mut best);
            }
        }
        if let Some((_, _, best_d2)) = best {
            // Every cell of a farther ring is at least ring + 0.5 away,
            // because the target lies within half a cell of the centre.
            let bound = ring as f64 + 0.5 - 1.0e-9;
            if bound * bound > best_d2 * (1.0 + 1.0e-12) {
                break;
            }
        }
    }
    best
}

/// How far a point sees before a source cell could lie outside the window
/// (`_window_reach`): infinity when every edge is closed.
#[inline]
fn window_reach(x: f64, y: f64, ny: usize, nx: usize, closed: u32) -> f64 {
    let mut reach = f64::INFINITY;
    for (edge, gap) in [
        (EDGE_WEST, x + 1.0),
        (EDGE_EAST, nx as f64 - x),
        (EDGE_SOUTH, y + 1.0),
        (EDGE_NORTH, ny as f64 - y),
    ] {
        if closed & edge == 0 && gap < reach {
            reach = gap;
        }
    }
    reach
}

#[derive(Clone)]
struct Stats {
    target_apply: u64,
    direct: u64,
    renormalized: u64,
    fallback: u64,
    cross_surface: u64,
    negative: u64,
    min_raw: f64,
    sum_min: f64,
    sum_max: f64,
    convex: bool,
}

impl Stats {
    fn new() -> Self {
        Stats {
            target_apply: 0,
            direct: 0,
            renormalized: 0,
            fallback: 0,
            cross_surface: 0,
            negative: 0,
            min_raw: f64::INFINITY,
            sum_min: f64::INFINITY,
            sum_max: f64::NEG_INFINITY,
            convex: true,
        }
    }

    fn merge(&mut self, other: &Stats) {
        self.target_apply += other.target_apply;
        self.direct += other.direct;
        self.renormalized += other.renormalized;
        self.fallback += other.fallback;
        self.cross_surface += other.cross_surface;
        self.negative += other.negative;
        self.min_raw = self.min_raw.min(other.min_raw);
        self.sum_min = self.sum_min.min(other.sum_min);
        self.sum_max = self.sum_max.max(other.sum_max);
        self.convex &= other.convex;
    }
}

struct Build<'a> {
    window: Window<'a>,
    x: &'a [f64],
    y: &'a [f64],
    apply: &'a [u8],
    radius: i64,
    closed: u32,
    any_valid: bool,
}

/// Per-target outputs of the parallel pass.
struct TargetOut<'a> {
    indices_y: &'a mut [i32],
    indices_x: &'a mut [i32],
    weights: &'a mut [f32],
    state: &'a mut [u8],
    donor_d2: &'a mut [f64],
    reach: &'a mut [f64],
}

/// The lower bilinear corner of a coordinate on an axis of `n` cells.
///
/// A coordinate exactly on the last cell (`p == n - 1`, `n >= 2`) is
/// spelled as the cell before with a unit fraction: its whole weight lands
/// on its own cell and its zero-weight partner stays inside the window.
/// Every other coordinate floors exactly as before, so every input this
/// entry accepted before builds the same stencil.  Breakage it removes: a
/// target that IS the source grid (the native grid copied index for index,
/// `gpuwm.ingest.hrrr._snap_to_native_lattice`) puts its last column and
/// row on `n - 1`, whose floor partner `n` is outside, and the whole soil
/// stencil of that grid was refused as leaving the window.
#[inline]
fn lower_corner(p: f64, n: usize) -> f64 {
    let floor = p.floor();
    if n >= 2 && p == (n - 1) as f64 {
        floor - 1.0
    } else {
        floor
    }
}

fn build_target(b: &Build, t: usize, out: &mut TargetOut, k: usize, stats: &mut Stats) {
    let x = b.x[t];
    let y = b.y[t];
    let x0 = lower_corner(x, b.window.nx) as i64;
    let y0 = lower_corner(y, b.window.ny) as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let mut index_x = [x0, x1, x0, x1];
    let mut index_y = [y0, y0, y1, y1];
    let mut weight = [
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ];
    for corner in 0..4 {
        let valid = f64::from(u8::from(b.window.valid(index_y[corner], index_x[corner])));
        weight[corner] *= valid;
    }
    let raw = ((weight[0] + weight[1]) + weight[2]) + weight[3];
    let apply = b.apply[t] != 0;
    let direct = apply && raw > 0.0;
    out.state[k] = STATE_NONE;
    if direct {
        for value in weight.iter_mut() {
            *value /= raw;
        }
        stats.direct += 1;
        if raw < 1.0 - 1.0e-12 {
            stats.renormalized += 1;
        }
        if raw < stats.min_raw {
            stats.min_raw = raw;
        }
    }
    if apply {
        stats.target_apply += 1;
    }
    if apply && !direct {
        stats.fallback += 1;
        let reach = window_reach(x, y, b.window.ny, b.window.nx, b.closed);
        out.reach[k] = reach;
        let within = radius_donor(&b.window, x, y, b.radius);
        let nearest = if within.is_none() && b.any_valid {
            nearest_valid(&b.window, x, y)
        } else {
            None
        };
        let donor = match within {
            Some(found) => {
                out.state[k] = STATE_WITHIN;
                Some(found)
            }
            None => match nearest {
                Some(found) if found.2 < reach * reach => {
                    out.state[k] = STATE_BEYOND;
                    Some(found)
                }
                Some(found) => {
                    out.state[k] = STATE_UNRESOLVED;
                    out.donor_d2[k] = found.2;
                    None
                }
                None => {
                    out.state[k] = STATE_UNRESOLVED;
                    out.donor_d2[k] = f64::NAN;
                    None
                }
            },
        };
        if let Some((row, column, d2)) = donor {
            index_x = [column; 4];
            index_y = [row; 4];
            weight = [1.0, 0.0, 0.0, 0.0];
            out.donor_d2[k] = d2;
        }
    }
    if !apply {
        weight = [1.0, 0.0, 0.0, 0.0];
    }
    let sum = ((weight[0] + weight[1]) + weight[2]) + weight[3];
    if sum < stats.sum_min {
        stats.sum_min = sum;
    }
    if sum > stats.sum_max {
        stats.sum_max = sum;
    }
    // np.allclose(sums, 1.0, rtol=0.0, atol=2.0e-15) and finite weights.
    if !((sum - 1.0).abs() <= 2.0e-15) {
        stats.convex = false;
    }
    for corner in 0..4 {
        let value = weight[corner];
        if !value.is_finite() {
            stats.convex = false;
        }
        if value < 0.0 {
            stats.negative += 1;
            stats.convex = false;
        }
        if value > 0.0 && apply && !b.window.valid(index_y[corner], index_x[corner]) {
            stats.cross_surface += 1;
        }
        out.indices_y[corner] = index_y[corner] as i32;
        out.indices_x[corner] = index_x[corner] as i32;
        out.weights[corner] = value as f32;
    }
}

/// Build the stencil of every target.
///
/// `x`/`y` are the targets' zero-based fractional source coordinates,
/// `source_valid` the `ny x nx` donor mask, `target_apply` the targets
/// that take a donor.  `closed_edges` is a mask of `EDGE_*` bits naming
/// the window edges that are edges of the whole source grid;
/// `edges_unknown` nonzero reports a name that is not an edge (refused
/// after the finiteness and radius checks, as the NumPy builder orders
/// them).  Outputs: `indices_y`, `indices_x` (int32) and `weights`
/// (float32), each `4 x ntarget` corner-major; `counts`, `reals` and
/// `flags` in the slot orders above; `histogram` (`nhistogram` bins) the
/// donors per ceiling distance in cells; the first `listed` donors farther
/// than `distant_cells` in target order (`distant_target` flat indices,
/// `distant_source` row and column pairs, `distant_distance`,
/// `distant_reach`, infinity for an unlimited reach); on `ERR_UNRESOLVED`
/// the unresolved targets' flat indices in `unresolved` (capacity
/// `ntarget`) and the worst distance.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer its
/// dimensions imply, and outputs must not overlap inputs.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_masked_bilinear_stencil_f64(
    x: *const f64,
    y: *const f64,
    target_apply: *const u8,
    ntarget: usize,
    source_valid: *const u8,
    ny: usize,
    nx: usize,
    fallback_radius: i64,
    closed_edges: u32,
    edges_unknown: i32,
    distant_cells: f64,
    listed: usize,
    indices_y: *mut i32,
    indices_x: *mut i32,
    weights: *mut f32,
    counts: *mut u64,
    reals: *mut f64,
    flags: *mut i32,
    histogram: *mut u64,
    nhistogram: usize,
    distant_target: *mut u64,
    distant_source: *mut i64,
    distant_distance: *mut f64,
    distant_reach: *mut f64,
    unresolved: *mut u64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if x.is_null()
            || y.is_null()
            || target_apply.is_null()
            || source_valid.is_null()
            || indices_y.is_null()
            || indices_x.is_null()
            || weights.is_null()
            || counts.is_null()
            || reals.is_null()
            || flags.is_null()
            || histogram.is_null()
            || unresolved.is_null()
            || (listed > 0
                && (distant_target.is_null()
                    || distant_source.is_null()
                    || distant_distance.is_null()
                    || distant_reach.is_null()))
        {
            return ERR_NULL;
        }
        if ntarget == 0 || ny == 0 || nx == 0 || workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) if value < i32::MAX as usize => value,
            _ => return ERR_DIMENSION,
        };
        if ntarget.checked_mul(4).is_none() || nhistogram < ny + nx + 2 {
            return ERR_DIMENSION;
        }
        let xs = std::slice::from_raw_parts(x, ntarget);
        let ys = std::slice::from_raw_parts(y, ntarget);
        if xs.iter().chain(ys.iter()).any(|value| !value.is_finite()) {
            return ERR_NONFINITE;
        }
        if fallback_radius < 0 {
            return ERR_RADIUS;
        }
        if edges_unknown != 0 {
            return ERR_EDGES;
        }
        let top_x = nx as f64;
        let top_y = ny as f64;
        if xs
            .iter()
            .zip(ys.iter())
            .any(|(&px, &py)| {
                let (x0, y0) = (lower_corner(px, nx), lower_corner(py, ny));
                x0 < 0.0 || x0 + 1.0 >= top_x || y0 < 0.0 || y0 + 1.0 >= top_y
            })
        {
            return ERR_OFF_WINDOW;
        }
        let valid = std::slice::from_raw_parts(source_valid, cells);
        let apply = std::slice::from_raw_parts(target_apply, ntarget);
        let valid_count = valid.iter().filter(|&&v| v != 0).count() as u64;
        let build = Build {
            window: Window { valid, ny, nx },
            x: xs,
            y: ys,
            apply,
            radius: fallback_radius,
            closed: closed_edges,
            any_valid: valid_count > 0,
        };
        let out_y = std::slice::from_raw_parts_mut(indices_y, 4 * ntarget);
        let out_x = std::slice::from_raw_parts_mut(indices_x, 4 * ntarget);
        let out_w = std::slice::from_raw_parts_mut(weights, 4 * ntarget);
        let mut state = vec![STATE_NONE; ntarget];
        let mut donor_d2 = vec![0.0f64; ntarget];
        let mut reach = vec![0.0f64; ntarget];
        let mut stencil_y = vec![0i32; 4 * ntarget];
        let mut stencil_x = vec![0i32; 4 * ntarget];
        let mut stencil_w = vec![0.0f32; 4 * ntarget];
        let stats = {
            // Target-major scratch (four corners per target), so each
            // chunk of targets owns one contiguous piece of every buffer.
            let pieces = std::sync::Mutex::new(
                stencil_y
                    .chunks_mut(4 * CHUNK)
                    .zip(stencil_x.chunks_mut(4 * CHUNK))
                    .zip(stencil_w.chunks_mut(4 * CHUNK))
                    .zip(state.chunks_mut(CHUNK))
                    .zip(donor_d2.chunks_mut(CHUNK))
                    .zip(reach.chunks_mut(CHUNK))
                    .enumerate(),
            );
            let count = workers.max(1).min(ntarget.div_ceil(CHUNK));
            let run = || {
                let mut local = Stats::new();
                loop {
                    let next = pieces.lock().unwrap_or_else(|p| p.into_inner()).next();
                    let Some((chunk, (((((sy, sx), sw), st), dd), rr))) = next else {
                        break;
                    };
                    let start = chunk * CHUNK;
                    for k in 0..st.len() {
                        let mut out = TargetOut {
                            indices_y: &mut sy[4 * k..4 * k + 4],
                            indices_x: &mut sx[4 * k..4 * k + 4],
                            weights: &mut sw[4 * k..4 * k + 4],
                            state: &mut st[k..k + 1],
                            donor_d2: &mut dd[k..k + 1],
                            reach: &mut rr[k..k + 1],
                        };
                        build_target(&build, start + k, &mut out, 0, &mut local);
                    }
                }
                local
            };
            let mut stats = Stats::new();
            if count <= 1 {
                stats.merge(&run());
            } else {
                let partials: Vec<Stats> = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..count).map(|_| scope.spawn(&run)).collect();
                    handles.into_iter().map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))).collect()
                });
                for part in &partials {
                    stats.merge(part);
                }
            }
            stats
        };
        {
            // Corner-major, as the NumPy builder stacks its four corners.
            for t in 0..ntarget {
                for corner in 0..4 {
                    out_y[corner * ntarget + t] = stencil_y[4 * t + corner];
                    out_x[corner * ntarget + t] = stencil_x[4 * t + corner];
                    out_w[corner * ntarget + t] = stencil_w[4 * t + corner];
                }
            }
            let counts = std::slice::from_raw_parts_mut(counts, COUNT_SLOTS);
            let reals = std::slice::from_raw_parts_mut(reals, REAL_SLOTS);
            let flags = std::slice::from_raw_parts_mut(flags, FLAG_SLOTS);
            let histogram = std::slice::from_raw_parts_mut(histogram, nhistogram);
            histogram.iter_mut().for_each(|bin| *bin = 0);
            counts.iter_mut().for_each(|slot| *slot = 0);
            counts[COUNT_SOURCE_VALID] = valid_count;
            counts[COUNT_TARGET_APPLY] = stats.target_apply;
            counts[COUNT_DIRECT] = stats.direct;
            counts[COUNT_RENORMALIZED] = stats.renormalized;
            counts[COUNT_FALLBACK] = stats.fallback;
            counts[COUNT_CROSS_SURFACE] = stats.cross_surface;
            counts[COUNT_NEGATIVE_WEIGHT] = stats.negative;
            reals[REAL_MIN_RAW_SUPPORT] = if stats.direct > 0 { stats.min_raw } else { f64::NAN };
            reals[REAL_SUM_MIN] = stats.sum_min;
            reals[REAL_SUM_MAX] = stats.sum_max;
            reals[REAL_MAX_DISTANCE] = 0.0;
            reals[REAL_WORST_DISTANCE] = f64::NAN;
            flags[FLAG_WEIGHTS_CONVEX] = i32::from(stats.convex);
            flags[FLAG_ANY_VALID] = i32::from(build.any_valid);

            // The fallback targets in target order: refusal first, as the
            // NumPy builder raises before it reports.
            let unresolved = std::slice::from_raw_parts_mut(unresolved, ntarget);
            let mut unresolved_count = 0usize;
            let mut worst = f64::NEG_INFINITY;
            for t in 0..ntarget {
                if state[t] == STATE_UNRESOLVED {
                    unresolved[unresolved_count] = t as u64;
                    unresolved_count += 1;
                    if donor_d2[t] > worst {
                        worst = donor_d2[t];
                    }
                }
            }
            if unresolved_count > 0 {
                counts[COUNT_UNRESOLVED] = unresolved_count as u64;
                if build.any_valid {
                    reals[REAL_WORST_DISTANCE] = worst.sqrt();
                }
                return ERR_UNRESOLVED;
            }
            let mut listed_count = 0usize;
            let mut distant_count = 0u64;
            let mut max_distance = f64::NEG_INFINITY;
            for t in 0..ntarget {
                if state[t] != STATE_WITHIN && state[t] != STATE_BEYOND {
                    continue;
                }
                let distance = donor_d2[t].sqrt();
                if distance > max_distance {
                    max_distance = distance;
                }
                let bin = distance.ceil() as usize;
                if bin >= nhistogram {
                    return ERR_DIMENSION;
                }
                histogram[bin] += 1;
                if distance > distant_cells {
                    if listed_count < listed {
                        *distant_target.add(listed_count) = t as u64;
                        *distant_source.add(2 * listed_count) = i64::from(out_y[t]);
                        *distant_source.add(2 * listed_count + 1) = i64::from(out_x[t]);
                        *distant_distance.add(listed_count) = distance;
                        *distant_reach.add(listed_count) = reach[t];
                        listed_count += 1;
                    }
                    distant_count += 1;
                }
            }
            if stats.fallback > 0 {
                reals[REAL_MAX_DISTANCE] = max_distance;
            }
            counts[COUNT_DISTANT] = distant_count;
            counts[COUNT_LISTED] = listed_count as u64;
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// Apply a built stencil to every layer of a float32 field: per target
/// `((v0 * w0 + v1 * w1) + v2 * w2) + v3 * w3` in float32, as the NumPy
/// apply sums its four corner terms, then, where `select` is given, the
/// fill on every unselected target (`fill_mode` 1: `fill_scalar`, 2: the
/// per-target `fill_array`, the same for every layer).
///
/// Every number it produces is NumPy's, bit for bit, and it produces NaN
/// exactly where NumPy does.  Which of two NaNs survives when both meet
/// is not matched: NumPy's own answer depends on whether the element falls
/// in its vector loop's body or its tail (the body keeps the first
/// operand's, the tail the second's) and on the CPU's vector width, so the
/// same inputs give it either sign.  The route admits no NaN: a source
/// with a non-finite soil value is refused before mapping.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer its
/// dimensions imply (`select` and `fill_array` may be null when unused),
/// and `output` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_masked_stencil_apply_f32(
    field: *const f32,
    nlayer: usize,
    ny: usize,
    nx: usize,
    indices_y: *const i32,
    indices_x: *const i32,
    weights: *const f32,
    ntarget: usize,
    select: *const u8,
    fill_mode: i32,
    fill_scalar: f32,
    fill_array: *const f32,
    output: *mut f32,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if field.is_null()
            || indices_y.is_null()
            || indices_x.is_null()
            || weights.is_null()
            || output.is_null()
            || (fill_mode != 0 && select.is_null())
            || (fill_mode == 2 && fill_array.is_null())
        {
            return ERR_NULL;
        }
        if !(0..=2).contains(&fill_mode) || nlayer == 0 || ny == 0 || nx == 0 || ntarget == 0 || workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        if nlayer.checked_mul(cells).is_none() || nlayer.checked_mul(ntarget).is_none() {
            return ERR_DIMENSION;
        }
        let iy = std::slice::from_raw_parts(indices_y, 4 * ntarget);
        let ix = std::slice::from_raw_parts(indices_x, 4 * ntarget);
        let w = std::slice::from_raw_parts(weights, 4 * ntarget);
        if iy.iter().any(|&v| v < 0 || v as usize >= ny) || ix.iter().any(|&v| v < 0 || v as usize >= nx) {
            return ERR_DIMENSION;
        }
        let fields = std::slice::from_raw_parts(field, nlayer * cells);
        let out = std::slice::from_raw_parts_mut(output, nlayer * ntarget);
        let chosen = if fill_mode != 0 {
            Some(std::slice::from_raw_parts(select, ntarget))
        } else {
            None
        };
        let fills = if fill_mode == 2 {
            Some(std::slice::from_raw_parts(fill_array, ntarget))
        } else {
            None
        };
        let body = |start: usize, piece: &mut [f32]| {
            for (offset, slot) in piece.iter_mut().enumerate() {
                let flat = start + offset;
                let layer = flat / ntarget;
                let t = flat % ntarget;
                if let Some(chosen) = chosen {
                    if chosen[t] == 0 {
                        *slot = match fills {
                            Some(values) => values[t],
                            None => fill_scalar,
                        };
                        continue;
                    }
                }
                let plane = &fields[layer * cells..(layer + 1) * cells];
                let term = |corner: usize| {
                    let at = corner * ntarget + t;
                    plane[iy[at] as usize * nx + ix[at] as usize] * w[at]
                };
                *slot = ((term(0) + term(1)) + term(2)) + term(3);
            }
        };
        let length = nlayer * ntarget;
        let count = workers.max(1).min(length.div_ceil(CHUNK));
        if count <= 1 {
            body(0, out);
        } else {
            let pieces = std::sync::Mutex::new(out.chunks_mut(CHUNK).enumerate());
            std::thread::scope(|scope| {
                for _ in 0..count {
                    scope.spawn(|| loop {
                        let next = pieces.lock().unwrap_or_else(|p| p.into_inner()).next();
                        let Some((chunk, piece)) = next else {
                            break;
                        };
                        body(chunk * CHUNK, piece);
                    });
                }
            });
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::type_complexity)]
    fn build(
        x: &[f64],
        y: &[f64],
        apply: &[u8],
        valid: &[u8],
        ny: usize,
        nx: usize,
        radius: i64,
        closed: u32,
        workers: usize,
    ) -> (i32, Vec<i32>, Vec<i32>, Vec<f32>, Vec<u64>, Vec<f64>, Vec<u64>) {
        let n = x.len();
        let mut iy = vec![0i32; 4 * n];
        let mut ix = vec![0i32; 4 * n];
        let mut w = vec![0f32; 4 * n];
        let mut counts = vec![0u64; COUNT_SLOTS];
        let mut reals = vec![0f64; REAL_SLOTS];
        let mut flags = vec![0i32; FLAG_SLOTS];
        let mut hist = vec![0u64; ny + nx + 2];
        let mut dt = vec![0u64; 8];
        let mut ds = vec![0i64; 16];
        let mut dd = vec![0f64; 8];
        let mut dr = vec![0f64; 8];
        let mut un = vec![0u64; n];
        let code = unsafe {
            gpuwm_masked_bilinear_stencil_f64(
                x.as_ptr(), y.as_ptr(), apply.as_ptr(), n, valid.as_ptr(), ny, nx, radius, closed, 0, 8.0, 8,
                iy.as_mut_ptr(), ix.as_mut_ptr(), w.as_mut_ptr(), counts.as_mut_ptr(), reals.as_mut_ptr(),
                flags.as_mut_ptr(), hist.as_mut_ptr(), hist.len(), dt.as_mut_ptr(), ds.as_mut_ptr(),
                dd.as_mut_ptr(), dr.as_mut_ptr(), un.as_mut_ptr(), workers,
            )
        };
        (code, iy, ix, w, counts, reals, un)
    }

    #[test]
    fn a_target_with_no_valid_corner_takes_the_first_nearest_cell() {
        let (ny, nx) = (12usize, 12usize);
        let mut valid = vec![0u8; ny * nx];
        valid[3 * nx + 9] = 1;
        valid[9 * nx + 3] = 1;
        // Equidistant from (6, 6): the lower row is scanned first.
        let (code, iy, ix, w, counts, _, _) = build(&[6.0], &[6.0], &[1], &valid, ny, nx, 8, 0, 1);
        assert_eq!(code, OK);
        assert_eq!((iy[0], ix[0]), (3, 9));
        assert_eq!(w[0], 1.0);
        assert_eq!(counts[COUNT_FALLBACK], 1);
    }

    #[test]
    fn past_the_radius_needs_the_window_to_vouch() {
        let (ny, nx) = (40usize, 40usize);
        let mut valid = vec![0u8; ny * nx];
        valid[20 * nx + 38] = 1;
        let (code, _, _, _, counts, reals, un) = build(&[5.2], &[20.0], &[1], &valid, ny, nx, 2, 0, 1);
        assert_eq!(code, ERR_UNRESOLVED);
        assert_eq!(counts[COUNT_UNRESOLVED], 1);
        assert_eq!(un[0], 0);
        assert!((reals[REAL_WORST_DISTANCE] - 32.8).abs() < 1e-9);
        let all = EDGE_WEST | EDGE_EAST | EDGE_SOUTH | EDGE_NORTH;
        let (code, iy, ix, _, _, _, _) = build(&[5.2], &[20.0], &[1], &valid, ny, nx, 2, all, 1);
        assert_eq!(code, OK);
        assert_eq!((iy[0], ix[0]), (20, 38));
    }

    #[test]
    fn worker_count_moves_nothing() {
        let (ny, nx) = (60usize, 70usize);
        let mut state = 7u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let valid: Vec<u8> = (0..ny * nx).map(|_| u8::from(next() < 0.3)).collect();
        let n = 9000;
        let x: Vec<f64> = (0..n).map(|_| next() * (nx as f64 - 1.0001)).collect();
        let y: Vec<f64> = (0..n).map(|_| next() * (ny as f64 - 1.0001)).collect();
        let apply: Vec<u8> = (0..n).map(|_| u8::from(next() < 0.6)).collect();
        let one = build(&x, &y, &apply, &valid, ny, nx, 8, 0, 1);
        for workers in [2usize, 3, 7, 64] {
            let many = build(&x, &y, &apply, &valid, ny, nx, 8, 0, workers);
            assert_eq!(one.0, many.0);
            assert_eq!(one.1, many.1);
            assert_eq!(one.2, many.2);
            assert!(one.3.iter().zip(many.3.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_eq!(one.4, many.4);
            assert!(one.5.iter().zip(many.5.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn apply_sums_corner_terms_in_order_and_fills_unselected() {
        let field = [1.0f32, 2.0, 3.0, 4.0];
        let iy = [0i32, 0, 1, 1];
        let ix = [0i32, 1, 0, 1];
        let w = [0.25f32, 0.25, 0.25, 0.25];
        let mut out = [0f32; 1];
        let code = unsafe {
            gpuwm_masked_stencil_apply_f32(
                field.as_ptr(), 1, 2, 2, iy.as_ptr(), ix.as_ptr(), w.as_ptr(), 1, std::ptr::null(), 0, 0.0,
                std::ptr::null(), out.as_mut_ptr(), 1,
            )
        };
        assert_eq!(code, OK);
        assert_eq!(out[0], 2.5);
        let select = [0u8];
        let code = unsafe {
            gpuwm_masked_stencil_apply_f32(
                field.as_ptr(), 1, 2, 2, iy.as_ptr(), ix.as_ptr(), w.as_ptr(), 1, select.as_ptr(), 1, 7.0,
                std::ptr::null(), out.as_mut_ptr(), 1,
            )
        };
        assert_eq!(code, OK);
        assert_eq!(out[0], 7.0);
    }

    #[test]
    fn a_target_on_the_last_cell_takes_its_own_cell_whole() {
        // A 3 x 4 window, every target on a whole cell including the last
        // column and row: each takes its own cell with weight 1 (the last
        // one as the cell before with a unit fraction), nothing leaves the
        // window, and one step past the last cell is still refused.
        let (ny, nx) = (3usize, 4usize);
        let valid = vec![1u8; ny * nx];
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for row in 0..ny {
            for column in 0..nx {
                xs.push(column as f64);
                ys.push(row as f64);
            }
        }
        let n = xs.len();
        let apply = vec![1u8; n];
        let (code, iy, ix, w, counts, _, _) = build(&xs, &ys, &apply, &valid, ny, nx, 2, 0, 1);
        assert_eq!(code, OK);
        assert_eq!(counts[COUNT_DIRECT], n as u64);
        for t in 0..n {
            let mut own = 0.0f32;
            for corner in 0..4 {
                let (row, column, weight) = (iy[corner * n + t], ix[corner * n + t], w[corner * n + t]);
                assert!((0..ny as i32).contains(&row) && (0..nx as i32).contains(&column));
                if (row as f64, column as f64) == (ys[t], xs[t]) {
                    own += weight;
                } else {
                    assert_eq!(weight, 0.0);
                }
            }
            assert_eq!(own, 1.0);
        }
        let (code, ..) = build(&[3.5], &[1.0], &[1], &valid, ny, nx, 2, 0, 1);
        assert_eq!(code, ERR_OFF_WINDOW);
        let (code, ..) = build(&[1.0], &[2.000001], &[1], &valid, ny, nx, 2, 0, 1);
        assert_eq!(code, ERR_OFF_WINDOW);
    }

    #[test]
    fn no_fused_multiply_add_in_this_module() {
        let source = include_str!("masked_stencil.rs");
        let fused = concat!("mul", "_add(");
        assert!(!source.contains(fused), "a fused multiply-add entered the stencil");
    }
}
