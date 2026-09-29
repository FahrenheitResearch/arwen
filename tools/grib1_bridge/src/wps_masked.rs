//! WPS metgrid's masked-field chain in float64: soil moisture and soil
//! temperature, snow, skin temperature and sea ice.
//!
//! This is the arithmetic `gpuwm.ingest.horiz` used to run as single-core
//! NumPy, moved here statement for statement.  The NumPy transcription is
//! kept as a test oracle only (`gpuwm/verify/wps_masked_oracle.py`), and
//! every output value and every repair count must equal it byte for byte:
//!
//! * every scalar expression keeps NumPy's operand order and association,
//!   and nothing here fuses a multiply and an add;
//! * the search's squared distance is `dx * dx + dy * dy`, as the
//!   shared-walk `_wps_search` computes it, and its walk is taken once
//!   per start cell as there ([`walk`]);
//! * a target is answered when its value is finite, never merely when it
//!   is not NaN, so an average that overflows falls through exactly as it
//!   does in NumPy;
//! * every output element depends only on its own target, the source
//!   arrays and per-call scalars, and every count is an integer sum, so
//!   the result cannot depend on the worker count or the schedule.
//!
//! Operator references are to WPS v4.6.0 `metgrid/src/interp_module.F`.

use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

/// A target coordinate outside the source grid (same code and meaning as
/// `wif_ffi::ERR_TARGET_OFF_GRID`).
pub const ERR_TARGET_OFF_GRID: i32 = 10;
/// A chain operator this library does not know was reached by a target
/// that was still waiting; `unknown_op` receives its chain position.
pub const ERR_UNKNOWN_OPERATOR: i32 = 11;

pub const OP_SIXTEEN_PT: u8 = 0;
pub const OP_FOUR_PT: u8 = 1;
pub const OP_AVERAGE_4PT: u8 = 2;
pub const OP_WT_AVERAGE_4PT: u8 = 3;
pub const OP_WT_AVERAGE_16PT: u8 = 4;
pub const OP_SEARCH: u8 = 5;

pub const MODE_PLAIN: i32 = 0;
pub const MODE_LAND: i32 = 1;
pub const MODE_SKIN: i32 = 2;

/// Count slots per layer, in this order.
pub const COUNT_SIXTEEN_PT_OUTSIDE_RANGE: usize = 0;
pub const COUNT_SEARCH: usize = 1;
pub const COUNT_SEARCH_PAST_UNUSABLE: usize = 2;
pub const COUNT_FILL: usize = 3;
pub const COUNT_SOURCE_OUTSIDE_RANGE: usize = 4;
pub const COUNT_SOURCE_ROUNDOFF_AT_BOUND: usize = 5;
pub const COUNT_OTHER_SURFACE: usize = 6;
pub const COUNT_RECOVERED: usize = 7;
pub const COUNT_SLOTS: usize = 8;
/// Unit scan slots per layer: land cells, carried (finite), inside range.
pub const SCAN_SLOTS: usize = 3;

/// `interp_opts` search depth cap (interp_module.F:267).
const SEARCH_DEPTH: i64 = 1200;
/// `gpuwm.ingest.horiz._DONOR_SPAN_TOLERANCE`.
const DONOR_SPAN_TOLERANCE: f64 = 1.0e-9;
/// `gpuwm.ingest.horiz.SOURCE_ROUNDOFF_FRACTION`.
const SOURCE_ROUNDOFF_FRACTION: f64 = 0.01;
/// Targets per unit of parallel work.  Workers pull chunks one at a time,
/// so a chunk full of expensive searches does not hold the others up.
const CHUNK: usize = 4096;

/// `np.maximum(0.0, value)` for a value that is never NaN.
#[inline]
fn numpy_maximum_zero(value: f64) -> f64 {
    if 0.0 >= value {
        0.0
    } else {
        value
    }
}

#[inline]
fn clip_index(value: i64, high: usize) -> usize {
    if value < 0 {
        0
    } else if value as u64 > high as u64 {
        high
    } else {
        value as usize
    }
}

/// Metgrid `oned` (interp_module.F), as `_wps_oned` evaluates it: every
/// candidate is computed and one is selected, with WPS's exact zero-value
/// special cases.
#[inline]
pub fn oned(x: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    let mut result = 0.0f64;
    if x == 0.0 {
        result = b;
    }
    if x == 1.0 {
        result = c;
    }
    let parab_b = b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b));
    let parab_c = c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c));
    let linear = b * (1.0 - x) + c * x;
    let both = (1.0 - x) * parab_b + x * parab_c;
    let inner = if a == 0.0 && d == 0.0 {
        linear
    } else if a != 0.0 {
        if d != 0.0 {
            both
        } else {
            parab_b
        }
    } else {
        parab_c
    };
    if b * c != 0.0 {
        inner
    } else {
        result
    }
}

/// The per-call source arrays: which cells are donors, and their values
/// with every non-donor neutralised to 0 so it never enters a product.
struct Prologue {
    usable: Vec<bool>,
    safe: Vec<f64>,
    /// `source_valid & ~usable`: cells of the right surface whose value is
    /// missing or outside the range.
    unusable_valid: Vec<bool>,
    /// Donor-surface cells whose finite value lies outside the range by
    /// more than packing roundoff (None without a range).
    outside: Option<Vec<bool>>,
    any_usable: bool,
    any_outside: bool,
}

fn run_rows<F>(ny: usize, nx: usize, workers: usize, body: F)
where
    F: Fn(usize, usize) + Sync,
{
    let length = ny * nx;
    if length == 0 {
        return;
    }
    let count = workers.max(1).min(ny.max(1));
    if count <= 1 {
        body(0, length);
        return;
    }
    std::thread::scope(|scope| {
        for (start, stop) in crate::worker_ranges(ny, count) {
            let body = &body;
            scope.spawn(move || body(start * nx, stop * nx));
        }
    });
}

fn prologue(
    field: &[f64],
    valid: &[bool],
    range: Option<(f64, f64)>,
    ny: usize,
    nx: usize,
    workers: usize,
) -> Prologue {
    let length = ny * nx;
    let mut usable = vec![false; length];
    let mut safe = vec![0.0f64; length];
    let mut unusable_valid = vec![false; length];
    let mut outside = range.map(|_| vec![false; length]);
    let donor_bounds = range.map(|(low, high)| {
        let slack = SOURCE_ROUNDOFF_FRACTION * (high - low);
        (low - slack, high + slack)
    });
    {
        let usable_address = usable.as_mut_ptr() as usize;
        let safe_address = safe.as_mut_ptr() as usize;
        let unusable_address = unusable_valid.as_mut_ptr() as usize;
        let outside_address = outside.as_mut().map(|v| v.as_mut_ptr() as usize);
        run_rows(ny, nx, workers, |start, stop| {
            // Each worker owns the disjoint cell range [start, stop).
            let usable_ptr = usable_address as *mut bool;
            let safe_ptr = safe_address as *mut f64;
            let unusable_ptr = unusable_address as *mut bool;
            let outside_ptr = outside_address.map(|a| a as *mut bool);
            for index in start..stop {
                let value = field[index];
                let mut ok = valid[index] && value.is_finite();
                if let Some((low, high)) = donor_bounds {
                    let in_range = value.is_finite() && value >= low && value <= high;
                    let out = ok && !in_range;
                    ok = ok && !out;
                    unsafe { *outside_ptr.unwrap().add(index) = out };
                }
                unsafe {
                    *usable_ptr.add(index) = ok;
                    *safe_ptr.add(index) = if ok { value } else { 0.0 };
                    *unusable_ptr.add(index) = valid[index] && !ok;
                }
            }
        });
    }
    let any_usable = usable.iter().any(|&v| v);
    let any_outside = outside.as_ref().map_or(false, |v| v.iter().any(|&o| o));
    Prologue {
        usable,
        safe,
        unusable_valid,
        outside,
        any_usable,
        any_outside,
    }
}

struct Call<'a> {
    ny: usize,
    nx: usize,
    pro: &'a Prologue,
    yy: &'a [f64],
    xx: &'a [f64],
    chain: &'a [u8],
    range: Option<(f64, f64)>,
}

impl Call<'_> {
    #[inline]
    fn at(&self, j: usize, i: usize) -> usize {
        j * self.nx + i
    }
}

/// Per-worker counts.  The chain-position counters exist because NumPy
/// ASSIGNS the `sixteen_pt` and search counts at the position that made
/// them rather than accumulating them.
#[derive(Clone)]
struct Stats {
    outside: Vec<u64>,
    beyond: Vec<u64>,
    produced: Vec<u64>,
    past: Vec<u64>,
    fill: u64,
    roundoff: u64,
    unknown: Option<usize>,
}

impl Stats {
    fn new(nchain: usize) -> Self {
        Stats {
            outside: vec![0; nchain],
            beyond: vec![0; nchain],
            produced: vec![0; nchain],
            past: vec![0; nchain],
            fill: 0,
            roundoff: 0,
            unknown: None,
        }
    }

    fn absorb(&mut self, other: &Stats) {
        for pos in 0..self.outside.len() {
            self.outside[pos] += other.outside[pos];
            self.beyond[pos] += other.beyond[pos];
            self.produced[pos] += other.produced[pos];
            self.past[pos] += other.past[pos];
        }
        self.fill += other.fill;
        self.roundoff += other.roundoff;
        self.unknown = match (self.unknown, other.unknown) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, None) => a,
            (None, b) => b,
        };
    }
}

/// Metgrid `sixteen_pt` (interp_module.F:1227-1332); NaN falls through.
fn sixteen_pt(c: &Call, yy: f64, xx: f64) -> f64 {
    let i = (xx + 1.0e-5).floor() as i64;
    let j = (yy + 1.0e-5).floor() as i64;
    let xf = xx - i as f64;
    let yf = yy - j as f64;
    if xf.abs() <= 1.0e-4 && yf.abs() <= 1.0e-4 {
        let index = c.at(clip_index(j, c.ny - 1), clip_index(i, c.nx - 1));
        return if c.pro.usable[index] {
            c.pro.safe[index]
        } else {
            f64::NAN
        };
    }
    let mut stencil = [[0.0f64; 4]; 4];
    let mut all_ok = true;
    for (k, column) in stencil.iter_mut().enumerate() {
        let kk = clip_index(i + (k as i64 - 1), c.nx - 1);
        for (l, cell) in column.iter_mut().enumerate() {
            let ll = clip_index(j + (l as i64 - 1), c.ny - 1);
            let index = c.at(ll, kk);
            let value = c.pro.safe[index];
            all_ok &= c.pro.usable[index];
            *cell = if value == 0.0 { 1.0e-20 } else { value };
        }
    }
    let row = |l: usize| oned(xf, stencil[0][l], stencil[1][l], stencil[2][l], stencil[3][l]);
    let (a, b, cc, d) = (row(0), row(1), row(2), row(3));
    let mut value = oned(yf, a, b, cc, d);
    if value == 1.0e-20 {
        value = 0.0;
    }
    if all_ok {
        value
    } else {
        f64::NAN
    }
}

/// The least and greatest source value in a target's `sixteen_pt` stencil.
fn donor_span(c: &Call, yy: f64, xx: f64) -> (f64, f64) {
    let i = (xx + 1.0e-5).floor() as i64;
    let j = (yy + 1.0e-5).floor() as i64;
    let mut least = f64::INFINITY;
    let mut greatest = f64::NEG_INFINITY;
    for k in 0..4i64 {
        let kk = clip_index(i + (k - 1), c.nx - 1);
        for m in 0..4i64 {
            let value = c.pro.safe[c.at(clip_index(j + (m - 1), c.ny - 1), kk)];
            if value < least {
                least = value;
            }
            if value > greatest {
                greatest = value;
            }
        }
    }
    (least, greatest)
}

struct Corners {
    fx: usize,
    cx: usize,
    fy: usize,
    cy: usize,
}

#[inline]
fn corners(c: &Call, yy: f64, xx: f64) -> Corners {
    Corners {
        fx: clip_index(xx.floor() as i64, c.nx - 1),
        cx: clip_index(xx.ceil() as i64, c.nx - 1),
        fy: clip_index(yy.floor() as i64, c.ny - 1),
        cy: clip_index(yy.ceil() as i64, c.ny - 1),
    }
}

/// Metgrid `four_pt` (interp_module.F:1099-1169) or `average_4pt`
/// (:691-732); NaN falls through.
fn four_pt(c: &Call, yy: f64, xx: f64, average: bool) -> f64 {
    let Corners { fx, cx, fy, cy } = corners(c, yy, xx);
    let (iff, ifc, icf, icc) = (c.at(fy, fx), c.at(cy, fx), c.at(fy, cx), c.at(cy, cx));
    let safe = &c.pro.safe;
    let usable = &c.pro.usable;
    let (v_ff, v_fc, v_cf, v_cc) = (safe[iff], safe[ifc], safe[icf], safe[icc]);
    if average {
        let weight = |ok: bool| if ok { 1.0f64 } else { 0.0f64 };
        let (w_ff, w_fc, w_cf, w_cc) = (
            weight(usable[iff]),
            weight(usable[ifc]),
            weight(usable[icf]),
            weight(usable[icc]),
        );
        let wsum = w_ff + w_fc + w_cf + w_cc;
        if wsum > 0.0 {
            return (w_ff * v_ff + w_fc * v_fc + w_cf * v_cf + w_cc * v_cc) / wsum;
        }
        return f64::NAN;
    }
    let all_ok = usable[iff] && usable[ifc] && usable[icf] && usable[icc];
    if !all_ok {
        return f64::NAN;
    }
    let (fxf, cxf, fyf, cyf) = (fx as f64, cx as f64, fy as f64, cy as f64);
    let x_int = fx == cx;
    let y_int = fy == cy;
    if x_int {
        if y_int {
            v_ff
        } else {
            v_ff * (cyf - yy) + v_fc * (yy - fyf)
        }
    } else if y_int {
        v_ff * (cxf - xx) + v_cf * (xx - fxf)
    } else {
        (yy - fyf) * (v_fc * (cxf - xx) + v_cc * (xx - fxf))
            + (cyf - yy) * (v_ff * (cxf - xx) + v_cf * (xx - fxf))
    }
}

/// Metgrid `wt_average_4pt` (interp_module.F:776-779) or
/// `wt_average_16pt` (:993-1024); NaN falls through.  The 16-point form
/// refuses a stencil that would leave the array instead of clamping it.
fn wt_average(c: &Call, yy: f64, xx: f64, sixteen: bool) -> f64 {
    let safe = &c.pro.safe;
    let usable = &c.pro.usable;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    if sixteen {
        let fx = xx.floor() as i64;
        let fy = yy.floor() as i64;
        let nx = c.nx as i64;
        let ny = c.ny as i64;
        if !(fx >= 1 && fx <= nx - 3 && fy >= 1 && fy <= ny - 3) {
            return f64::NAN;
        }
        for dx in [-1i64, 0, 1, 2] {
            for dy in [-1i64, 0, 1, 2] {
                let ii = fx + dx;
                let jj = fy + dy;
                let ex = xx - ii as f64;
                let ey = yy - jj as f64;
                let index = c.at(jj as usize, ii as usize);
                let mut w = numpy_maximum_zero(2.0 - (ex * ex + ey * ey).sqrt());
                if !usable[index] {
                    w = 0.0;
                }
                num += w * safe[index];
                den += w;
            }
        }
    } else {
        let Corners { fx, cx, fy, cy } = corners(c, yy, xx);
        for (ii, jj) in [(fx, fy), (fx, cy), (cx, fy), (cx, cy)] {
            let ex = xx - ii as f64;
            let ey = yy - jj as f64;
            let index = c.at(jj, ii);
            let mut w = numpy_maximum_zero(1.0 - (ex * ex + ey * ey).sqrt());
            if !usable[index] {
                w = 0.0;
            }
            num += w * safe[index];
            den += w;
        }
    }
    if den > 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

/// One worker's reusable search state: a generation-stamped visited map,
/// the queue, and the walks already taken from each start cell, allocated
/// when the worker first reaches `search`.
struct SearchScratch {
    visited: Vec<u32>,
    stamp: u32,
    queue: VecDeque<(i64, i64, i64)>,
    /// Start cell -> the cells its walk compares, in comparison order
    /// (None when no usable cell is reachable).  One call has one usable
    /// mask, so a walk depends on its start cell alone.
    walks: HashMap<usize, Option<Box<[u32]>>>,
    walk_cells: usize,
}

/// Candidate cells one worker keeps cached before it starts over.  A
/// memory bound only: a walk taken again gives the same cells.
const WALK_CACHE_CELLS: usize = 1 << 22;

impl SearchScratch {
    fn new(cells: usize) -> Self {
        SearchScratch {
            visited: vec![0; cells],
            stamp: 0,
            queue: VecDeque::new(),
            walks: HashMap::new(),
            walk_cells: 0,
        }
    }

    fn next_stamp(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.visited.iter_mut().for_each(|v| *v = 0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.stamp
    }
}

/// The cells metgrid's search compares from one start cell, in the order
/// it compares them: the first usable cell dequeued, then every usable
/// cell still in the queue (`_wps_search_candidates`).
///
/// Four-connected FIFO walk from the start cell: expansion stops once the
/// first usable point is DEQUEUED (that iteration still enqueues its
/// neighbours), neighbour order is x-1, x+1, y-1, y+1, and the depth
/// counter is WRF's in-place `qdata%depth` mutation, capped at 1200.  The
/// walk depends on the start cell, the grid and the usable mask only.
fn walk(c: &Call, ix: i64, jy: i64, scratch: &mut SearchScratch) -> Option<Box<[u32]>> {
    let nx = c.nx as i64;
    let ny = c.ny as i64;
    let stamp = scratch.next_stamp();
    let at = |i: i64, j: i64| (j * nx + i) as usize;
    scratch.queue.clear();
    scratch.queue.push_back((ix, jy, 0));
    scratch.visited[at(ix, jy)] = stamp;
    let usable = &c.pro.usable;
    let mut found: Option<usize> = None;
    while found.is_none() {
        let Some((i, j, depth)) = scratch.queue.pop_front() else {
            break;
        };
        let mut dd = depth;
        for (ni, nj) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] {
            if ni >= 0 && ni < nx && nj >= 0 && nj < ny && dd < SEARCH_DEPTH && scratch.visited[at(ni, nj)] != stamp
            {
                dd += 1;
                scratch.queue.push_back((ni, nj, dd));
                scratch.visited[at(ni, nj)] = stamp;
            }
        }
        if usable[at(i, j)] {
            found = Some(at(i, j));
        }
    }
    let first = found?;
    let mut cells = vec![first as u32];
    cells.extend(
        scratch
            .queue
            .iter()
            .map(|&(i, j, _)| at(i, j))
            .filter(|&flat| usable[flat])
            .map(|flat| flat as u32),
    );
    Some(cells.into_boxed_slice())
}

/// Metgrid `search_extrap` for one target (interp_module.F:484-607), as
/// `_wps_search` evaluates it: the walk from `NINT(xx), NINT(yy)`
/// ([`walk`], taken once per start cell), then a first-minimum over the
/// walk's cells of `dx * dx + dy * dy`, so the first found wins a tie and
/// never-enqueued points never win.
fn search(c: &Call, yy: f64, xx: f64, scratch: &mut SearchScratch) -> f64 {
    let nx = c.nx as i64;
    let ny = c.ny as i64;
    let ix = (xx + 0.5).floor() as i64;
    let jy = (yy + 0.5).floor() as i64;
    if ix < 0 || ix >= nx || jy < 0 || jy >= ny {
        return f64::NAN;
    }
    let start = (jy * nx + ix) as usize;
    if !scratch.walks.contains_key(&start) {
        let cells = walk(c, ix, jy, scratch);
        let size = cells.as_ref().map_or(1, |w| w.len());
        if scratch.walk_cells + size > WALK_CACHE_CELLS {
            scratch.walks.clear();
            scratch.walk_cells = 0;
        }
        scratch.walk_cells += size;
        scratch.walks.insert(start, cells);
    }
    let Some(cells) = &scratch.walks[&start] else {
        return f64::NAN;
    };
    let distance = |flat: u32| {
        let flat = flat as usize;
        let dx = (flat % c.nx) as f64 - xx;
        let dy = (flat / c.nx) as f64 - yy;
        dx * dx + dy * dy
    };
    let mut best = cells[0];
    let mut best_d2 = distance(best);
    for &flat in &cells[1..] {
        let d2 = distance(flat);
        if d2 < best_d2 {
            best_d2 = d2;
            best = flat;
        }
    }
    c.pro.safe[best as usize]
}

/// Whether a cell of the right surface with a missing or out-of-range
/// value lies within two source cells of the target (the reach of every
/// operator before `search`).
fn unusable_within_reach(c: &Call, yy: f64, xx: f64) -> bool {
    let base_row = yy.floor() as i64;
    let base_column = xx.floor() as i64;
    let ny = c.ny as i64;
    let nx = c.nx as i64;
    for dy in [-1i64, 0, 1, 2] {
        for dx in [-1i64, 0, 1, 2] {
            let jj = base_row + dy;
            let ii = base_column + dx;
            if jj >= 0 && jj < ny && ii >= 0 && ii < nx {
                let ex = xx - ii as f64;
                let ey = yy - jj as f64;
                if ex * ex + ey * ey < 4.0 && c.pro.unusable_valid[c.at(jj as usize, ii as usize)] {
                    return true;
                }
            }
        }
    }
    false
}

/// The chain for one target: the first finite answer, or None.  Unknown
/// operators are recorded (by chain position) when the target reaches
/// them, exactly when NumPy would raise.
fn chain_target(
    c: &Call,
    t: usize,
    scratch: &mut Option<SearchScratch>,
    stats: &mut Stats,
) -> Option<f64> {
    let yy = c.yy[t];
    let xx = c.xx[t];
    for (pos, &op) in c.chain.iter().enumerate() {
        let got = match op {
            OP_SIXTEEN_PT => {
                let mut value = sixteen_pt(c, yy, xx);
                if let Some((low, high)) = c.range {
                    if value.is_finite() && (value < low || value > high) {
                        stats.outside[pos] += 1;
                        let (least, greatest) = donor_span(c, yy, xx);
                        let swing = DONOR_SPAN_TOLERANCE * (high - low);
                        if value < least - swing || value > greatest + swing {
                            stats.beyond[pos] += 1;
                            value = f64::NAN;
                        }
                    }
                }
                value
            }
            OP_FOUR_PT => four_pt(c, yy, xx, false),
            OP_AVERAGE_4PT => four_pt(c, yy, xx, true),
            OP_WT_AVERAGE_4PT => wt_average(c, yy, xx, false),
            OP_WT_AVERAGE_16PT => wt_average(c, yy, xx, true),
            OP_SEARCH => {
                let value = if c.pro.any_usable {
                    let work = scratch.get_or_insert_with(|| SearchScratch::new(c.ny * c.nx));
                    search(c, yy, xx, work)
                } else {
                    f64::NAN
                };
                if value.is_finite() {
                    stats.produced[pos] += 1;
                    if unusable_within_reach(c, yy, xx) {
                        stats.past[pos] += 1;
                    }
                }
                value
            }
            _ => {
                stats.unknown = Some(stats.unknown.map_or(pos, |p| p.min(pos)));
                return None;
            }
        };
        if got.is_finite() {
            return Some(got);
        }
    }
    None
}

/// Why a call did not produce its answer.
#[derive(Debug, PartialEq)]
pub enum ChainError {
    UnknownOperator(usize),
}

/// One `wps_masked_field_interpolate` call: values for every target (the
/// fill where a target is inactive or unanswered) and the six counts.
#[allow(clippy::too_many_arguments)]
fn run_call(
    field: &[f64],
    ny: usize,
    nx: usize,
    donors: &[bool],
    yy: &[f64],
    xx: &[f64],
    active: &[bool],
    chain: &[u8],
    range: Option<(f64, f64)>,
    fill: f64,
    workers: usize,
) -> Result<(Vec<f64>, [u64; 6]), ChainError> {
    let pro = prologue(field, donors, range, ny, nx, workers);
    let ntarget = yy.len();
    let mut counts = [0u64; 6];
    let any_active = active.iter().any(|&a| a);
    if range.is_some() && any_active && pro.any_outside {
        // Only the source cells a target stencil can reach are this
        // domain's business.
        let mut row_min = f64::INFINITY;
        let mut row_max = f64::NEG_INFINITY;
        let mut column_min = f64::INFINITY;
        let mut column_max = f64::NEG_INFINITY;
        for t in 0..ntarget {
            if active[t] {
                row_min = row_min.min(yy[t]);
                row_max = row_max.max(yy[t]);
                column_min = column_min.min(xx[t]);
                column_max = column_max.max(xx[t]);
            }
        }
        let j0 = (row_min.floor() as i64 - 1).max(0) as usize;
        let j1 = ((row_max.floor() as i64 + 3).min(ny as i64)).max(0) as usize;
        let i0 = (column_min.floor() as i64 - 1).max(0) as usize;
        let i1 = ((column_max.floor() as i64 + 3).min(nx as i64)).max(0) as usize;
        let outside = pro.outside.as_ref().expect("range implies outside mask");
        let mut total = 0u64;
        for j in j0..j1 {
            if i0 < i1 {
                total += outside[j * nx + i0..j * nx + i1].iter().filter(|&&o| o).count() as u64;
            }
        }
        counts[COUNT_SOURCE_OUTSIDE_RANGE] = total;
    }
    let mut output = vec![fill; ntarget];
    let call = Call {
        ny,
        nx,
        pro: &pro,
        yy,
        xx,
        chain,
        range,
    };
    let mut stats = Stats::new(chain.len());
    if any_active && ntarget > 0 {
        let pieces = Mutex::new(output.chunks_mut(CHUNK).enumerate());
        let count = workers.max(1).min(ntarget.div_ceil(CHUNK));
        let run = |call: &Call| {
            let mut local = Stats::new(chain.len());
            let mut scratch: Option<SearchScratch> = None;
            loop {
                let next = pieces.lock().unwrap_or_else(|p| p.into_inner()).next();
                let Some((chunk, slice)) = next else {
                    break;
                };
                let start = chunk * CHUNK;
                for (offset, slot) in slice.iter_mut().enumerate() {
                    let t = start + offset;
                    if !active[t] {
                        continue;
                    }
                    match chain_target(call, t, &mut scratch, &mut local) {
                        Some(mut value) => {
                            if let Some((low, high)) = range {
                                if value < low || value > high {
                                    local.roundoff += 1;
                                    value = if value < low { low } else { high };
                                }
                            }
                            *slot = value;
                        }
                        None => local.fill += 1,
                    }
                }
            }
            local
        };
        if count <= 1 {
            stats = run(&call);
        } else {
            let partials: Vec<Stats> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..count)
                    .map(|_| {
                        let run = &run;
                        let call = &call;
                        scope.spawn(move || run(call))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                    .collect()
            });
            for partial in &partials {
                stats.absorb(partial);
            }
        }
    }
    if let Some(pos) = stats.unknown {
        return Err(ChainError::UnknownOperator(pos));
    }
    // NumPy assigns these at the chain position that made them; a later
    // position that made none leaves the earlier assignment standing.
    let mut sixteen = 0u64;
    let mut searched = 0u64;
    let mut past = 0u64;
    for (pos, &op) in chain.iter().enumerate() {
        if op == OP_SIXTEEN_PT && stats.outside[pos] > 0 {
            sixteen = stats.beyond[pos];
        }
        if op == OP_SEARCH && stats.produced[pos] > 0 {
            past = stats.past[pos];
            searched = stats.produced[pos] - stats.past[pos];
        }
    }
    counts[COUNT_SIXTEEN_PT_OUTSIDE_RANGE] = sixteen;
    counts[COUNT_SEARCH] = searched;
    counts[COUNT_SEARCH_PAST_UNUSABLE] = past;
    counts[COUNT_FILL] = stats.fill;
    if range.is_some() {
        counts[COUNT_SOURCE_ROUNDOFF_AT_BOUND] = stats.roundoff;
    }
    Ok((output, counts))
}

/// The layer's result and its eight count slots.
type LayerResult = Result<(Vec<f64>, [u64; COUNT_SLOTS]), ChainError>;

fn add_counts(into: &mut [u64; 6], from: &[u64; 6]) {
    for (slot, value) in into.iter_mut().zip(from.iter()) {
        *slot += value;
    }
}

/// `_land_pass_with_fractional_second_chance`: the WPS land pass, then the
/// source's fractional land where the binarized flag has none, then the
/// fill.  Returns values that are NaN where nothing answered (the caller
/// applies its own fill), the six counts with `fill` recounted after both
/// passes, and the recovered count.
#[allow(clippy::too_many_arguments)]
fn land_pass(
    field: &[f64],
    ny: usize,
    nx: usize,
    land: &[bool],
    partial: &[bool],
    yy: &[f64],
    xx: &[f64],
    active: &[bool],
    chain: &[u8],
    range: Option<(f64, f64)>,
    workers: usize,
) -> Result<(Vec<f64>, [u64; 6], u64), ChainError> {
    let (mut values, mut passes) =
        run_call(field, ny, nx, land, yy, xx, active, chain, range, f64::NAN, workers)?;
    let mut recovered = 0u64;
    if !land.iter().any(|&v| v) && active.iter().any(|&v| v) && partial.iter().any(|&v| v) {
        let starved: Vec<bool> = (0..values.len())
            .map(|t| active[t] && !values[t].is_finite())
            .collect();
        let (second, counts) =
            run_call(field, ny, nx, partial, yy, xx, &starved, chain, range, f64::NAN, workers)?;
        add_counts(&mut passes, &counts);
        for t in 0..values.len() {
            if starved[t] && second[t].is_finite() {
                values[t] = second[t];
                recovered += 1;
            }
        }
    }
    passes[COUNT_FILL] = (0..values.len())
        .filter(|&t| active[t] && !values[t].is_finite())
        .count() as u64;
    Ok((values, passes, recovered))
}

fn finish(values: Vec<f64>, fill: f64) -> Vec<f64> {
    values
        .into_iter()
        .map(|v| if v.is_finite() { v } else { fill })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    field: &[f64],
    ny: usize,
    nx: usize,
    donors: &[bool],
    partial: &[bool],
    yy: &[f64],
    xx: &[f64],
    target: &[bool],
    chain: &[u8],
    mode: i32,
    range: Option<(f64, f64)>,
    fill: f64,
    workers: usize,
) -> LayerResult {
    let mut counts = [0u64; COUNT_SLOTS];
    match mode {
        MODE_PLAIN => {
            let (values, six) = run_call(field, ny, nx, donors, yy, xx, target, chain, range, fill, workers)?;
            counts[..6].copy_from_slice(&six);
            Ok((values, counts))
        }
        MODE_LAND => {
            let (values, six, recovered) =
                land_pass(field, ny, nx, donors, partial, yy, xx, target, chain, range, workers)?;
            counts[..6].copy_from_slice(&six);
            counts[COUNT_RECOVERED] = recovered;
            Ok((finish(values, fill), counts))
        }
        _ => {
            // `_skin_temperature_on_both_surfaces` (METGRID.TBL masked=both).
            let (land_part, six, recovered) =
                land_pass(field, ny, nx, donors, partial, yy, xx, target, chain, range, workers)?;
            let water: Vec<bool> = donors.iter().map(|&v| !v).collect();
            let water_targets: Vec<bool> = target.iter().map(|&v| !v).collect();
            let (water_part, _) =
                run_call(field, ny, nx, &water, yy, xx, &water_targets, chain, range, f64::NAN, workers)?;
            let mut combined: Vec<f64> = (0..target.len())
                .map(|t| if target[t] { land_part[t] } else { water_part[t] })
                .collect();
            // Both starved sets come from `combined` before either is filled.
            let starved_water: Vec<bool> = (0..target.len())
                .map(|t| !target[t] && !combined[t].is_finite())
                .collect();
            let starved_land: Vec<bool> = (0..target.len())
                .map(|t| target[t] && !combined[t].is_finite())
                .collect();
            let mut other_surface = 0u64;
            for (starved, surface) in [(&starved_water, donors), (&starved_land, &water[..])] {
                if !starved.iter().any(|&v| v) {
                    continue;
                }
                let (answer, _) =
                    run_call(field, ny, nx, surface, yy, xx, starved, chain, range, f64::NAN, workers)?;
                for t in 0..combined.len() {
                    if starved[t] && answer[t].is_finite() {
                        combined[t] = answer[t];
                        other_surface += 1;
                    }
                }
            }
            counts[..6].copy_from_slice(&six);
            counts[COUNT_FILL] = (0..target.len())
                .filter(|&t| target[t] && !combined[t].is_finite())
                .count() as u64;
            counts[COUNT_OTHER_SURFACE] = other_surface;
            counts[COUNT_RECOVERED] = recovered;
            Ok((finish(combined, fill), counts))
        }
    }
}

unsafe fn mask(pointer: *const u8, length: usize) -> Vec<bool> {
    std::slice::from_raw_parts(pointer, length)
        .iter()
        .map(|&v| v != 0)
        .collect()
}

/// The masked chain over every layer of one field.
///
/// `mode` 0 is one plain call (`donors` = source_valid, `target_mask` =
/// target_active), 1 the land pass with its fractional second chance
/// (`donors` = binarized land, `partial_donors` = fractional land,
/// `target_mask` = target_active), 2 skin temperature on both surfaces
/// (`target_mask` = target land).  `counts` receives, per layer, the eight
/// slots `sixteen_pt_outside_range, search, search_past_unusable, fill,
/// source_outside_range, source_roundoff_at_bound, other_surface,
/// recovered`.  On `ERR_UNKNOWN_OPERATOR`, `unknown_op` holds the chain
/// position the first waiting target reached.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer its
/// dimensions imply (`partial_donors` may be null in mode 0, `chain` when
/// `nchain` is 0).  Output buffers must not overlap inputs.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_wps_masked_chain_f64(
    source: *const f64,
    donors: *const u8,
    partial_donors: *const u8,
    target_y: *const f64,
    target_x: *const f64,
    target_mask: *const u8,
    chain: *const u8,
    nchain: usize,
    mode: i32,
    fill_value: f64,
    has_range: i32,
    low: f64,
    high: f64,
    output: *mut f64,
    counts: *mut u64,
    unknown_op: *mut u64,
    nlayer: usize,
    ny: usize,
    nx: usize,
    ntarget: usize,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null()
            || donors.is_null()
            || output.is_null()
            || counts.is_null()
            || unknown_op.is_null()
            || (ntarget > 0 && (target_y.is_null() || target_x.is_null() || target_mask.is_null()))
            || (nchain > 0 && chain.is_null())
            || (mode != MODE_PLAIN && partial_donors.is_null())
        {
            return ERR_NULL;
        }
        if !(MODE_PLAIN..=MODE_SKIN).contains(&mode) || nlayer == 0 || ny == 0 || nx == 0 || workers == 0 {
            return ERR_DIMENSION;
        }
        let range = if has_range != 0 {
            if !(low < high) {
                return ERR_DIMENSION;
            }
            Some((low, high))
        } else {
            None
        };
        let cells = match ny.checked_mul(nx) {
            // The search caches its walks as 32-bit cell indices.
            Some(value) if value <= u32::MAX as usize => value,
            _ => return ERR_DIMENSION,
        };
        if nlayer.checked_mul(cells).is_none() || nlayer.checked_mul(ntarget).is_none() {
            return ERR_DIMENSION;
        }
        let (yy, xx) = if ntarget > 0 {
            (
                std::slice::from_raw_parts(target_y, ntarget),
                std::slice::from_raw_parts(target_x, ntarget),
            )
        } else {
            (&[][..], &[][..])
        };
        let top_y = (ny - 1) as f64;
        let top_x = (nx - 1) as f64;
        if yy
            .iter()
            .zip(xx.iter())
            .any(|(&y, &x)| !(y >= 0.0 && y <= top_y && x >= 0.0 && x <= top_x))
        {
            return ERR_TARGET_OFF_GRID;
        }
        let target = if ntarget > 0 { mask(target_mask, ntarget) } else { Vec::new() };
        let donor_mask = mask(donors, cells);
        let partial_mask = if mode == MODE_PLAIN {
            Vec::new()
        } else {
            mask(partial_donors, cells)
        };
        let chain_ops = if nchain > 0 {
            std::slice::from_raw_parts(chain, nchain)
        } else {
            &[][..]
        };
        let fields = std::slice::from_raw_parts(source, nlayer * cells);
        let out = std::slice::from_raw_parts_mut(output, nlayer * ntarget);
        let tallies = std::slice::from_raw_parts_mut(counts, nlayer * COUNT_SLOTS);
        for layer in 0..nlayer {
            let field = &fields[layer * cells..(layer + 1) * cells];
            match run_layer(
                field,
                ny,
                nx,
                &donor_mask,
                &partial_mask,
                yy,
                xx,
                &target,
                chain_ops,
                mode,
                range,
                fill_value,
                workers,
            ) {
                Ok((values, slots)) => {
                    out[layer * ntarget..(layer + 1) * ntarget].copy_from_slice(&values);
                    tallies[layer * COUNT_SLOTS..(layer + 1) * COUNT_SLOTS].copy_from_slice(&slots);
                }
                Err(ChainError::UnknownOperator(position)) => {
                    *unknown_op = position as u64;
                    return ERR_UNKNOWN_OPERATOR;
                }
            }
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// The unit check's numbers for every layer of a bounded land field.
///
/// Donors are the binarized land when the source has any, otherwise its
/// fractional land.  Per layer, `stats_counts` receives the donor cell
/// count, how many carry a finite value and how many of those lie inside
/// `low..high` widened by packing roundoff; `stats_span` the least and
/// greatest finite donor value (+inf and -inf when none is finite).
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer its
/// dimensions imply.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_wps_land_unit_scan_f64(
    source: *const f64,
    land: *const u8,
    partial: *const u8,
    nlayer: usize,
    ny: usize,
    nx: usize,
    low: f64,
    high: f64,
    stats_counts: *mut u64,
    stats_span: *mut f64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || land.is_null() || partial.is_null() || stats_counts.is_null() || stats_span.is_null() {
            return ERR_NULL;
        }
        if nlayer == 0 || ny == 0 || nx == 0 || workers == 0 || !(low < high) {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        if nlayer.checked_mul(cells).is_none() {
            return ERR_DIMENSION;
        }
        let land_mask = mask(land, cells);
        let donors = if land_mask.iter().any(|&v| v) {
            land_mask
        } else {
            mask(partial, cells)
        };
        let slack = SOURCE_ROUNDOFF_FRACTION * (high - low);
        let (lowest, highest) = (low - slack, high + slack);
        let fields = std::slice::from_raw_parts(source, nlayer * cells);
        let counts = std::slice::from_raw_parts_mut(stats_counts, nlayer * SCAN_SLOTS);
        let spans = std::slice::from_raw_parts_mut(stats_span, nlayer * 2);
        let land_cells = donors.iter().filter(|&&v| v).count() as u64;
        for layer in 0..nlayer {
            let field = &fields[layer * cells..(layer + 1) * cells];
            let scan = |start: usize, stop: usize| {
                let mut carried = 0u64;
                let mut inside = 0u64;
                let mut least = f64::INFINITY;
                let mut greatest = f64::NEG_INFINITY;
                for index in start..stop {
                    if !donors[index] {
                        continue;
                    }
                    let value = field[index];
                    if value.is_finite() {
                        carried += 1;
                        if value >= lowest && value <= highest {
                            inside += 1;
                        }
                        if value < least {
                            least = value;
                        }
                        if value > greatest {
                            greatest = value;
                        }
                    }
                }
                (carried, inside, least, greatest)
            };
            let count = workers.min(ny);
            let partials: Vec<(u64, u64, f64, f64)> = if count <= 1 {
                vec![scan(0, cells)]
            } else {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = crate::worker_ranges(ny, count)
                        .into_iter()
                        .map(|(start, stop)| {
                            let scan = &scan;
                            scope.spawn(move || scan(start * nx, stop * nx))
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                        .collect()
                })
            };
            let mut carried = 0u64;
            let mut inside = 0u64;
            let mut least = f64::INFINITY;
            let mut greatest = f64::NEG_INFINITY;
            for (c, i, l, g) in partials {
                carried += c;
                inside += i;
                if l < least {
                    least = l;
                }
                if g > greatest {
                    greatest = g;
                }
            }
            counts[layer * SCAN_SLOTS] = land_cells;
            counts[layer * SCAN_SLOTS + 1] = carried;
            counts[layer * SCAN_SLOTS + 2] = inside;
            spans[layer * 2] = least;
            spans[layer * 2 + 1] = greatest;
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn chain_call(
        field: &[f64],
        ny: usize,
        nx: usize,
        donors: &[u8],
        yy: &[f64],
        xx: &[f64],
        target: &[u8],
        chain: &[u8],
        mode: i32,
        fill: f64,
        range: Option<(f64, f64)>,
        workers: usize,
    ) -> (i32, Vec<f64>, Vec<u64>, u64) {
        let partial = vec![0u8; ny * nx];
        let mut output = vec![0.0f64; yy.len()];
        let mut counts = vec![0u64; COUNT_SLOTS];
        let mut unknown = u64::MAX;
        let (has_range, low, high) = match range {
            Some((l, h)) => (1, l, h),
            None => (0, 0.0, 1.0),
        };
        let code = unsafe {
            gpuwm_wps_masked_chain_f64(
                field.as_ptr(),
                donors.as_ptr(),
                partial.as_ptr(),
                yy.as_ptr(),
                xx.as_ptr(),
                target.as_ptr(),
                chain.as_ptr(),
                chain.len(),
                mode,
                fill,
                has_range,
                low,
                high,
                output.as_mut_ptr(),
                counts.as_mut_ptr(),
                &mut unknown,
                1,
                ny,
                nx,
                yy.len(),
                workers,
            )
        };
        (code, output, counts, unknown)
    }

    #[test]
    fn oned_branches_follow_the_zero_value_rules() {
        // b * c == 0 collapses to 0 unless x is exactly 0 or 1.
        assert_eq!(oned(0.5, 1.0, 0.0, 2.0, 3.0), 0.0);
        assert_eq!(oned(0.0, 1.0, 0.0, 2.0, 3.0), 0.0);
        assert_eq!(oned(1.0, 1.0, 0.0, 2.0, 3.0), 2.0);
        // a == d == 0 is linear.
        assert_eq!(oned(0.25, 0.0, 1.0, 2.0, 0.0), 1.0 * 0.75 + 2.0 * 0.25);
        // Only a == 0: the parabola through b, c, d.
        let x = 0.5f64;
        let expected = 2.0 + (1.0 - x) * (0.5 * (1.0 - 4.0) + (1.0 - x) * (0.5 * (1.0 + 4.0) - 2.0));
        assert_eq!(oned(x, 0.0, 1.0, 2.0, 4.0).to_bits(), expected.to_bits());
    }

    #[test]
    fn search_is_queue_limited_not_global_nearest() {
        // tests/test_horiz.py's counterexample: target (4.49, 4.49), 11 at
        // (x=0, y=4) on the x-first arm (d2 20.4) and 22 at (x=7, y=6)
        // (d2 8.58, not yet enqueued when 11 is dequeued).  WPS returns 11.
        let (ny, nx) = (10usize, 10usize);
        let mut field = vec![0.0f64; ny * nx];
        let mut donors = vec![0u8; ny * nx];
        field[4 * nx] = 11.0;
        donors[4 * nx] = 1;
        field[6 * nx + 7] = 22.0;
        donors[6 * nx + 7] = 1;
        let (code, out, counts, _) = chain_call(
            &field, ny, nx, &donors, &[4.49], &[4.49], &[1], &[OP_SEARCH], MODE_PLAIN, -999.0, None, 1,
        );
        assert_eq!(code, OK);
        assert_eq!(out[0], 11.0);
        assert_eq!(counts[COUNT_SEARCH], 1);
    }

    #[test]
    fn depth_counter_stops_the_search_like_wrf() {
        // One usable cell 1300 columns away is beyond the 1200 depth cap.
        let (ny, nx) = (1usize, 1400usize);
        let mut field = vec![0.0f64; nx];
        let mut donors = vec![0u8; nx];
        field[1300] = 5.0;
        donors[1300] = 1;
        let (code, out, counts, _) = chain_call(
            &field, ny, nx, &donors, &[0.0], &[0.0], &[1], &[OP_SEARCH], MODE_PLAIN, 7.0, None, 1,
        );
        assert_eq!(code, OK);
        assert_eq!(out[0], 7.0);
        assert_eq!(counts[COUNT_FILL], 1);
    }

    #[test]
    fn an_unknown_operator_is_reported_only_when_reached() {
        let field = vec![1.0f64; 16];
        let donors = vec![1u8; 16];
        let (code, out, _, _) = chain_call(
            &field, 4, 4, &donors, &[1.5], &[1.5], &[1], &[OP_FOUR_PT, 99], MODE_PLAIN, 0.0, None, 1,
        );
        assert_eq!(code, OK);
        assert_eq!(out[0], 1.0);
        let none = vec![0u8; 16];
        let (code, _, _, unknown) = chain_call(
            &field, 4, 4, &none, &[1.5], &[1.5], &[1], &[OP_FOUR_PT, 99], MODE_PLAIN, 0.0, None, 1,
        );
        assert_eq!(code, ERR_UNKNOWN_OPERATOR);
        assert_eq!(unknown, 1);
    }

    #[test]
    fn worker_count_does_not_move_a_bit() {
        let (ny, nx) = (37usize, 53usize);
        let field: Vec<f64> = (0..ny * nx)
            .map(|k| ((k * 7919) % 1009) as f64 / 1009.0 - 0.004)
            .collect();
        let donors: Vec<u8> = (0..ny * nx).map(|k| u8::from((k * 31) % 7 < 3)).collect();
        let ntarget = 9000usize;
        let yy: Vec<f64> = (0..ntarget).map(|t| ((t * 13) % 3601) as f64 / 3600.0 * (ny - 1) as f64).collect();
        let xx: Vec<f64> = (0..ntarget).map(|t| ((t * 29) % 5201) as f64 / 5200.0 * (nx - 1) as f64).collect();
        let target: Vec<u8> = (0..ntarget).map(|t| u8::from(t % 5 != 0)).collect();
        let chain = [OP_SIXTEEN_PT, OP_FOUR_PT, OP_WT_AVERAGE_4PT, OP_WT_AVERAGE_16PT, OP_SEARCH];
        let serial = chain_call(&field, ny, nx, &donors, &yy, &xx, &target, &chain, MODE_PLAIN, 1.0, Some((0.0, 1.0)), 1);
        assert_eq!(serial.0, OK);
        for workers in [2usize, 3, 7, 64] {
            let parallel =
                chain_call(&field, ny, nx, &donors, &yy, &xx, &target, &chain, MODE_PLAIN, 1.0, Some((0.0, 1.0)), workers);
            assert_eq!(parallel.0, OK);
            assert!(serial.1.iter().zip(parallel.1.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_eq!(serial.2, parallel.2, "counts moved at {workers} workers");
        }
    }

    #[test]
    fn no_fused_multiply_add_in_this_module() {
        // Byte identity with NumPy needs every product rounded before its
        // sum; a fused operation rounds once and moves bits.
        let source = include_str!("wps_masked.rs");
        let fused = concat!("mul", "_add(");
        assert!(!source.contains(fused), "a fused multiply-add entered the masked chain");
    }
}
