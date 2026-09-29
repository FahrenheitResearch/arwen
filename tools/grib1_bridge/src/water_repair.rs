//! The water-temperature repairs and the per-body assembly in float64.
//!
//! Two single-core NumPy data steps of
//! `gpuwm.ingest.water_temperature`, moved here statement for statement:
//!
//! * `_fill_missing_water_temperature` with `_label_boxes`,
//!   `_propagate_into`, `_edge_cells`, `_donor_index` and `_nearest_donor`:
//!   the box repairs of water cells whose provider left no admissible
//!   temperature (`gpuwm_water_repair_f64`);
//! * the per-body loop of `assemble_water_temperature`, which built
//!   `labels == label`, the source donor mask, the blend and the fill over
//!   the whole domain once per water body (`gpuwm_water_bodies_f64`).
//!
//! The NumPy code is kept as a test oracle only
//! (`gpuwm/verify/water_blend_oracle.py`), and every output must equal it
//! byte for byte:
//!
//! * a ring sweep adds its eight neighbours in NumPy's order, adding `0.0`
//!   for a neighbour that is not a seed, and divides the sum by the count;
//! * a donor search compares integer squared distances and keeps the first
//!   strictly nearer donor in row-major order (`np.argmin`);
//! * the blend and the four-neighbour fill are those of `water_blend.rs`,
//!   evaluated on the cells of one body, which is all NumPy read of them;
//! * bodies are independent where NumPy's loop let them be (their own
//!   holes, their own donors) and run in order where one body's fill can
//!   seed the next (the skin repairs), so no result depends on the worker
//!   count.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::water_blend::ERR_CORNER;
use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

/// A water-body label is negative, or above the number of bodies declared.
pub const ERR_LABEL: i32 = 32;

/// The eight neighbours in the order `_propagate_into` sums them.
const RING: [(i64, i64); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

fn admissible(value: f64, minimum: f64, maximum: f64) -> bool {
    value.is_finite() && value >= minimum && value <= maximum
}

/// `f(0..count)` on `workers` threads pulling one index at a time; the
/// results come back in index order.
fn parallel_map<R: Send, F>(count: usize, workers: usize, f: F) -> Vec<R>
where
    F: Fn(usize) -> R + Sync,
{
    let threads = workers.max(1).min(count);
    if threads <= 1 {
        return (0..count).map(&f).collect();
    }
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = (0..count).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= count {
                    break;
                }
                let value = f(index);
                *slots[index].lock().unwrap_or_else(|p| p.into_inner()) = Some(value);
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.into_inner().unwrap_or_else(|p| p.into_inner()).expect("every index ran"))
        .collect()
}

/// A body's box: rows `j0..j1` and columns `i0..i1`, end exclusive.
#[derive(Clone, Copy)]
struct Window {
    j0: usize,
    j1: usize,
    i0: usize,
    i1: usize,
}

impl Window {
    fn height(&self) -> usize {
        self.j1 - self.j0
    }
    fn width(&self) -> usize {
        self.i1 - self.i0
    }
    fn global(&self, local: usize, nx: usize) -> usize {
        let w = self.width();
        (self.j0 + local / w) * nx + self.i0 + local % w
    }
}

/// `_edge_cells` of a `(height, width)` mask: cells with an 8-neighbour
/// outside the mask or off the grid.
fn edge_cells(mask: &[bool], height: usize, width: usize) -> Vec<bool> {
    let mut edge = vec![false; mask.len()];
    for j in 0..height {
        for i in 0..width {
            let k = j * width + i;
            if !mask[k] {
                continue;
            }
            let mut interior = true;
            for &(dj, di) in RING.iter() {
                let nj = j as i64 + dj;
                let ni = i as i64 + di;
                if nj < 0 || ni < 0 || nj >= height as i64 || ni >= width as i64 || !mask[nj as usize * width + ni as usize] {
                    interior = false;
                    break;
                }
            }
            edge[k] = !interior;
        }
    }
    edge
}

/// `_donor_index`: the row-major rows and columns of the donors' edge.
fn donor_index(donors: &[bool], ny: usize, nx: usize) -> (Vec<i64>, Vec<i64>) {
    let edge = edge_cells(donors, ny, nx);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for (k, &on) in edge.iter().enumerate() {
        if on {
            rows.push((k / nx) as i64);
            cols.push((k % nx) as i64);
        }
    }
    (rows, cols)
}

/// `_nearest_donor`: the donor nearest any cell of `body` (a mask over
/// `window`), or `None`.
fn nearest_donor(body: &[bool], window: Window, index: &(Vec<i64>, Vec<i64>), ny: usize, nx: usize) -> Option<usize> {
    let (donor_rows, donor_cols) = index;
    if donor_rows.is_empty() {
        return None;
    }
    let edge = edge_cells(body, window.height(), window.width());
    let mut edge_rows = Vec::new();
    let mut edge_cols = Vec::new();
    for (k, &on) in edge.iter().enumerate() {
        if on {
            edge_rows.push((window.j0 + k / window.width()) as i64);
            edge_cols.push((window.i0 + k % window.width()) as i64);
        }
    }
    let j0 = *edge_rows.iter().min()?;
    let j1 = *edge_rows.iter().max()?;
    let i0 = *edge_cols.iter().min()?;
    let i1 = *edge_cols.iter().max()?;
    let last_y = ny as i64 - 1;
    let last_x = nx as i64 - 1;
    let mut radius: i64 = 8;
    loop {
        let wj0 = 0i64.max(j0 - radius);
        let wj1 = last_y.min(j1 + radius);
        let wi0 = 0i64.max(i0 - radius);
        let wi1 = last_x.min(i1 + radius);
        let whole = wj0 == 0 && wi0 == 0 && wj1 == last_y && wi1 == last_x;
        let lo = donor_rows.partition_point(|&r| r < wj0);
        let hi = donor_rows.partition_point(|&r| r <= wj1);
        let mut best: Option<(i64, usize)> = None;
        for d in lo..hi {
            let c = donor_cols[d];
            if c < wi0 || c > wi1 {
                continue;
            }
            let r = donor_rows[d];
            let mut squared = i64::MAX;
            for e in 0..edge_rows.len() {
                let dy = r - edge_rows[e];
                let dx = c - edge_cols[e];
                let value = dy * dy + dx * dx;
                if value < squared {
                    squared = value;
                }
            }
            if best.map_or(true, |(b, _)| squared < b) {
                best = Some((squared, d));
            }
        }
        match best {
            Some((squared, d)) => {
                if whole || squared < (radius + 1) * (radius + 1) {
                    return Some(donor_rows[d] as usize * nx + donor_cols[d] as usize);
                }
            }
            None => {
                if whole {
                    return None;
                }
            }
        }
        radius *= 2;
    }
}

/// `_propagate_into` on `window` of `values`: closes `holes` ring by ring
/// from their 8-neighbours in `have` (masks over the window).  Returns the
/// filled cells as `(global cell, value)` in the order written, and the
/// window mask of holes no seed reached.
fn propagate_into(values: &[f64], nx: usize, window: Window, holes: &[bool], have: &[bool]) -> (Vec<(usize, f64)>, Vec<bool>) {
    let height = window.height();
    let width = window.width();
    let mut local: Vec<f64> = (0..height * width).map(|k| values[window.global(k, nx)]).collect();
    let mut open = holes.to_vec();
    let mut seeded: Vec<bool> = (0..open.len()).map(|k| have[k] && !open[k]).collect();
    let mut waiting: Vec<usize> = (0..open.len()).filter(|&k| open[k]).collect();
    let mut written = Vec::new();
    while !waiting.is_empty() {
        let mut ready: Vec<(usize, f64)> = Vec::new();
        for &k in &waiting {
            let j = (k / width) as i64;
            let i = (k % width) as i64;
            let mut total = 0.0f64;
            let mut count = 0.0f64;
            for &(dj, di) in RING.iter() {
                let nj = j + dj;
                let ni = i + di;
                if nj < 0 || ni < 0 || nj >= height as i64 || ni >= width as i64 {
                    total += 0.0;
                    count += 0.0;
                    continue;
                }
                let n = nj as usize * width + ni as usize;
                total += if seeded[n] { local[n] } else { 0.0 };
                count += if seeded[n] { 1.0 } else { 0.0 };
            }
            if count > 0.0 {
                ready.push((k, total / count));
            }
        }
        if ready.is_empty() {
            break;
        }
        for &(k, value) in &ready {
            local[k] = value;
            open[k] = false;
            seeded[k] = true;
            written.push((window.global(k, nx), value));
        }
        waiting.retain(|&k| open[k]);
    }
    (written, open)
}

/// `_label_boxes`: each label's bounding box widened by one cell, for
/// every label `0..=largest`.
fn label_boxes(labels: &[i32], ny: usize, nx: usize, largest: usize) -> Vec<Window> {
    let mut j0 = vec![ny as i64; largest + 1];
    let mut j1 = vec![-1i64; largest + 1];
    let mut i0 = vec![nx as i64; largest + 1];
    let mut i1 = vec![-1i64; largest + 1];
    for (k, &label) in labels.iter().enumerate() {
        if label == 0 {
            continue;
        }
        let l = label as usize;
        let (j, i) = ((k / nx) as i64, (k % nx) as i64);
        j0[l] = j0[l].min(j);
        j1[l] = j1[l].max(j);
        i0[l] = i0[l].min(i);
        i1[l] = i1[l].max(i);
    }
    (0..=largest)
        .map(|l| Window {
            j0: 0i64.max(j0[l] - 1) as usize,
            j1: (ny as i64).min(j1[l] + 2).max(0) as usize,
            i0: 0i64.max(i0[l] - 1) as usize,
            i1: (nx as i64).min(i1[l] + 2).max(0) as usize,
        })
        .collect()
}

fn body_mask(labels: &[i32], nx: usize, window: Window, label: i32) -> Vec<bool> {
    (0..window.height() * window.width()).map(|k| labels[window.global(k, nx)] == label).collect()
}

/// `_fill_missing_water_temperature`, in place on `values` and `source`.
///
/// `counts` receives the own-body, nearest-water and surrounding-skin
/// tallies; `filled` marks the bad cells that end admissible.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_water_repair_f64(
    values: *mut f64,
    source: *mut i8,
    water: *const u8,
    labels: *const i32,
    ny: usize,
    nx: usize,
    minimum: f64,
    maximum: f64,
    nearest_water_code: i8,
    surrounding_skin_code: i8,
    counts: *mut u64,
    filled: *mut u8,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if counts.is_null() {
            return ERR_NULL;
        }
        if workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let counts = std::slice::from_raw_parts_mut(counts, 3);
        counts.fill(0);
        if cells == 0 {
            return OK;
        }
        if values.is_null() || source.is_null() || water.is_null() || labels.is_null() || filled.is_null() {
            return ERR_NULL;
        }
        let values = std::slice::from_raw_parts_mut(values, cells);
        let source = std::slice::from_raw_parts_mut(source, cells);
        let water = std::slice::from_raw_parts(water, cells);
        let labels = std::slice::from_raw_parts(labels, cells);
        let filled = std::slice::from_raw_parts_mut(filled, cells);
        if labels.iter().any(|&l| l < 0) {
            return ERR_LABEL;
        }
        let bad: Vec<bool> = (0..cells).map(|k| water[k] != 0 && !admissible(values[k], minimum, maximum)).collect();
        filled.fill(0);
        if !bad.iter().any(|&b| b) {
            return OK;
        }
        let donors: Vec<bool> = (0..cells).map(|k| water[k] != 0 && !bad[k]).collect();
        let largest = labels.iter().copied().max().unwrap_or(0) as usize;
        let mut is_wanted = vec![false; largest + 1];
        for k in 0..cells {
            if bad[k] && labels[k] != 0 {
                is_wanted[labels[k] as usize] = true;
            }
        }
        let wanted: Vec<i32> = (1..=largest).filter(|&l| is_wanted[l]).map(|l| l as i32).collect();
        let boxes = label_boxes(labels, ny, nx, largest);

        // 1. each body from its own admissible water; bodies read and write
        //    only their own cells, so they run side by side.
        let own: Vec<Option<(Vec<usize>, Vec<(usize, f64)>)>> = {
            let snapshot: &[f64] = values;
            parallel_map(wanted.len(), workers, |n| {
                let label = wanted[n];
                let window = boxes[label as usize];
                let body = body_mask(labels, nx, window, label);
                let mut holes = vec![false; body.len()];
                let mut have = vec![false; body.len()];
                let mut hole_cells = Vec::new();
                for k in 0..body.len() {
                    let g = window.global(k, nx);
                    holes[k] = body[k] && bad[g];
                    have[k] = body[k] && donors[g];
                    if holes[k] {
                        hole_cells.push(g);
                    }
                }
                if !have.iter().any(|&h| h) {
                    return None;
                }
                let (written, _) = propagate_into(snapshot, nx, window, &holes, &have);
                Some((hole_cells, written))
            })
        };
        let mut whole_bodies = Vec::new();
        for (n, result) in own.into_iter().enumerate() {
            match result {
                None => whole_bodies.push(wanted[n]),
                Some((hole_cells, written)) => {
                    for (g, value) in written {
                        values[g] = value;
                    }
                    for &g in &hole_cells {
                        source[g] = nearest_water_code;
                    }
                    counts[0] += hole_cells.len() as u64;
                }
            }
        }

        // 2. a body with none takes the nearest admissible water, one value.
        let mut skin_bodies = Vec::new();
        if !whole_bodies.is_empty() {
            let index = donor_index(&donors, ny, nx);
            let nearest: Vec<Option<usize>> = parallel_map(whole_bodies.len(), workers, |n| {
                let label = whole_bodies[n];
                let window = boxes[label as usize];
                nearest_donor(&body_mask(labels, nx, window, label), window, &index, ny, nx)
            });
            for (n, cell) in nearest.into_iter().enumerate() {
                let label = whole_bodies[n];
                let Some(cell) = cell else {
                    skin_bodies.push(label);
                    continue;
                };
                let window = boxes[label as usize];
                let donor = values[cell];
                for k in 0..window.height() * window.width() {
                    let g = window.global(k, nx);
                    if labels[g] == label {
                        values[g] = donor;
                        source[g] = nearest_water_code;
                        counts[1] += 1;
                    }
                }
            }
        }

        // 3. the skin around a body, in label order: a body filled here
        //    can seed the next one's shore.
        let mut stranded: Vec<(i32, Vec<bool>)> = Vec::new();
        for &label in &skin_bodies {
            let window = boxes[label as usize];
            let body = body_mask(labels, nx, window, label);
            let have: Vec<bool> = (0..body.len())
                .map(|k| admissible(values[window.global(k, nx)], minimum, maximum) && !body[k])
                .collect();
            let (written, left) = propagate_into(values, nx, window, &body, &have);
            for (g, value) in written {
                values[g] = value;
            }
            for k in 0..body.len() {
                if body[k] && !left[k] {
                    source[window.global(k, nx)] = surrounding_skin_code;
                    counts[2] += 1;
                }
            }
            if left.iter().any(|&l| l) {
                stranded.push((label, left));
            }
        }
        if !stranded.is_empty() {
            let admissible_now: Vec<bool> = values.iter().map(|&v| admissible(v, minimum, maximum)).collect();
            let anywhere = donor_index(&admissible_now, ny, nx);
            for (label, left) in &stranded {
                let window = boxes[*label as usize];
                let Some(cell) = nearest_donor(left, window, &anywhere, ny, nx) else {
                    continue;
                };
                let donor = values[cell];
                for k in 0..left.len() {
                    if left[k] {
                        let g = window.global(k, nx);
                        values[g] = donor;
                        source[g] = surrounding_skin_code;
                        counts[2] += 1;
                    }
                }
            }
        }
        for k in 0..cells {
            filled[k] = (bad[k] && admissible(values[k], minimum, maximum)) as u8;
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// Stats slots per body in `gpuwm_water_bodies_f64`'s `stats`.
pub const BODY_STAT_SLOTS: usize = 7;

struct SharedOut {
    values: *mut f64,
    source: *mut i8,
}
unsafe impl Sync for SharedOut {}
unsafe impl Send for SharedOut {}

/// The per-body loop of `assemble_water_temperature`, for bodies
/// `1..=nlabels` (`lake_class[label]` nonzero for a lake).
///
/// `values` and `source` hold the caller's starting fields (mapped skin
/// and land) and receive each body's cells.  Per body, `stats` receives
/// `[cells, donors, provider, analysis cells, component-skin cells,
/// lake-water cells, lake fallback cells]` (provider 1 analysis, 2 skin,
/// 3 lake water) and `coverage` the analysis coverage.  `listed` receives
/// up to `max_listed` `(row, column)` pairs of lake fallback cells in body
/// order, and `listed_count` their number.
///
/// The source arrays (`sst`, `owner`, the corners) are null when the
/// caller has no source analysis; `lake_water` is null when it has no
/// lake-model state.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_water_bodies_f64(
    labels: *const i32,
    ny: usize,
    nx: usize,
    nlabels: usize,
    lake_class: *const u8,
    skin: *const f64,
    lake_water: *const f64,
    sst: *const f64,
    owner: *const i32,
    sny: usize,
    snx: usize,
    corner_y: *const i64,
    corner_x: *const i64,
    weight: *const f64,
    ncorner: usize,
    denominator_floor: f64,
    min_coverage: f64,
    minimum: f64,
    maximum: f64,
    max_sweeps: usize,
    codes: *const i8,
    values: *mut f64,
    source: *mut i8,
    stats: *mut u64,
    coverage: *mut f64,
    listed: *mut i64,
    max_listed: usize,
    listed_count: *mut u64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        if labels.is_null() || lake_class.is_null() || skin.is_null() || codes.is_null() || values.is_null() || source.is_null() || stats.is_null() || coverage.is_null() || listed_count.is_null() || (max_listed > 0 && listed.is_null()) {
            return ERR_NULL;
        }
        let have_source = !sst.is_null();
        if have_source && (owner.is_null() || (ncorner > 0 && (corner_y.is_null() || corner_x.is_null() || weight.is_null()))) {
            return ERR_NULL;
        }
        let labels = std::slice::from_raw_parts(labels, cells);
        let lake_class = std::slice::from_raw_parts(lake_class, nlabels + 1);
        let skin = std::slice::from_raw_parts(skin, cells);
        let lake_water = if lake_water.is_null() { None } else { Some(std::slice::from_raw_parts(lake_water, cells)) };
        let codes = std::slice::from_raw_parts(codes, 3);
        let (analysis_code, skin_code, lake_code) = (codes[0], codes[1], codes[2]);
        let stats = std::slice::from_raw_parts_mut(stats, (nlabels + 1) * BODY_STAT_SLOTS);
        let coverage = std::slice::from_raw_parts_mut(coverage, nlabels + 1);
        stats.fill(0);
        coverage.fill(0.0);
        *listed_count = 0;
        if labels.iter().any(|&l| l < 0 || l as usize > nlabels) {
            return ERR_LABEL;
        }
        let scells = if have_source {
            match sny.checked_mul(snx) {
                Some(value) if value > 0 => value,
                _ => return ERR_DIMENSION,
            }
        } else {
            0
        };
        let (sst, owner) = if have_source {
            (std::slice::from_raw_parts(sst, scells), std::slice::from_raw_parts(owner, scells))
        } else {
            (&[][..], &[][..])
        };
        let stacked = if have_source {
            match ncorner.checked_mul(cells) {
                Some(value) => value,
                None => return ERR_DIMENSION,
            }
        } else {
            0
        };
        let (cy, cx, w): (&[i64], &[i64], &[f64]) = if stacked == 0 {
            (&[], &[], &[])
        } else {
            (
                std::slice::from_raw_parts(corner_y, stacked),
                std::slice::from_raw_parts(corner_x, stacked),
                std::slice::from_raw_parts(weight, stacked),
            )
        };
        if cy.iter().any(|&v| v < 0 || v as u64 >= sny as u64) || cx.iter().any(|&v| v < 0 || v as u64 >= snx as u64) {
            return ERR_CORNER;
        }

        // Every body's cells in row-major order, in one pass.
        let mut offsets = vec![0usize; nlabels + 2];
        for &l in labels {
            offsets[l as usize + 1] += 1;
        }
        for l in 0..=nlabels {
            offsets[l + 1] += offsets[l];
        }
        let mut members = vec![0usize; cells];
        let mut cursor = offsets.clone();
        for (k, &l) in labels.iter().enumerate() {
            members[cursor[l as usize]] = k;
            cursor[l as usize] += 1;
        }
        // Donors per body: the source cells a body owns with an admissible
        // analysis, counted once rather than masked once per body.
        let mut donor_count = vec![0u64; nlabels + 1];
        if have_source {
            for s in 0..scells {
                let o = owner[s];
                if o > 0 && (o as usize) <= nlabels && admissible(sst[s], minimum, maximum) {
                    donor_count[o as usize] += 1;
                }
            }
        }

        let out = SharedOut { values, source };
        let out = &out;
        let body_stats: Vec<([u64; BODY_STAT_SLOTS], f64)> = parallel_map(nlabels, workers, |n| {
            let label = (n + 1) as i32;
            let body = &members[offsets[n + 1]..offsets[n + 2]];
            let mut row = [0u64; BODY_STAT_SLOTS];
            let count = body.len();
            row[0] = count as u64;
            if count == 0 {
                return (row, 0.0);
            }
            let write = |g: usize, value: f64, code: i8| unsafe {
                // Bodies are disjoint, so no two workers write one cell.
                *out.values.add(g) = value;
                *out.source.add(g) = code;
            };
            let mut chosen: Option<Vec<f64>> = None;
            let mut body_coverage = 0.0f64;
            let donors = donor_count[label as usize];
            if have_source {
                row[1] = donors;
                if donors > 0 {
                    let donor = |cell: usize| owner[cell] == label && admissible(sst[cell], minimum, maximum);
                    let estimate: Vec<f64> = body
                        .iter()
                        .map(|&t| {
                            let mut numerator = 0.0f64;
                            let mut denominator = 0.0f64;
                            for corner in 0..ncorner {
                                let at = corner * cells + t;
                                let cell = cy[at] as usize * snx + cx[at] as usize;
                                let present_donor = donor(cell);
                                let present = if present_donor { 1.0f64 } else { 0.0f64 };
                                let safe = if present_donor { sst[cell] } else { 0.0f64 };
                                let term = w[at] * present;
                                numerator += term * safe;
                                denominator += term;
                            }
                            if denominator > denominator_floor {
                                numerator / denominator
                            } else {
                                f64::NAN
                            }
                        })
                        .collect();
                    let covered = estimate.iter().filter(|v| v.is_finite()).count();
                    body_coverage = covered as f64 / count as f64;
                    if body_coverage >= min_coverage {
                        chosen = Some(fill_within_body(estimate, body, labels, label, ny, nx, max_sweeps));
                    }
                }
            }
            let provided_possible = lake_class[label as usize] != 0 && lake_water.is_some();
            let lake_any = provided_possible && body.iter().any(|&t| lake_water.unwrap()[t].is_finite());
            match &chosen {
                None if lake_any => {
                    let lw = lake_water.unwrap();
                    for &t in body {
                        if lw[t].is_finite() {
                            write(t, lw[t], lake_code);
                            row[5] += 1;
                        } else {
                            write(t, skin[t], skin_code);
                            row[4] += 1;
                        }
                    }
                    row[2] = 3;
                }
                None => {
                    for &t in body {
                        write(t, skin[t], skin_code);
                    }
                    row[4] = count as u64;
                    row[2] = 2;
                }
                Some(filled) => {
                    for (n, &t) in body.iter().enumerate() {
                        if filled[n].is_finite() {
                            write(t, filled[n], analysis_code);
                            row[3] += 1;
                        } else {
                            write(t, skin[t], skin_code);
                            row[4] += 1;
                        }
                    }
                    row[2] = 1;
                }
            }
            if chosen.is_none() && provided_possible {
                let lw = lake_water.unwrap();
                row[6] = body.iter().filter(|&&t| !lw[t].is_finite()).count() as u64;
            }
            (row, body_coverage)
        });
        for (n, (row, value)) in body_stats.iter().enumerate() {
            let l = n + 1;
            stats[l * BODY_STAT_SLOTS..(l + 1) * BODY_STAT_SLOTS].copy_from_slice(row);
            coverage[l] = *value;
        }
        // The lake fallback cells, first ones in body order.
        let mut taken = 0usize;
        if let Some(lw) = lake_water {
            let listed = if max_listed > 0 { std::slice::from_raw_parts_mut(listed, 2 * max_listed) } else { &mut [][..] };
            'bodies: for n in 0..nlabels {
                if taken >= max_listed {
                    break;
                }
                let (row, _) = &body_stats[n];
                if row[2] == 1 || lake_class[n + 1] == 0 || row[6] == 0 {
                    continue;
                }
                for &t in &members[offsets[n + 1]..offsets[n + 2]] {
                    if !lw[t].is_finite() {
                        listed[2 * taken] = (t / nx) as i64;
                        listed[2 * taken + 1] = (t % nx) as i64;
                        taken += 1;
                        if taken >= max_listed {
                            break 'bodies;
                        }
                    }
                }
            }
        }
        *listed_count = taken as u64;
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// `_fill_within_component` of one body: `estimate` is the body's values
/// in the row-major order of `body`; NaN cells take the mean of their
/// finite four-neighbours in the body, sweep by sweep, as
/// `gpuwm_component_fill_f64` does over the whole grid (an in-grid
/// neighbour outside the body adds `0.0`).
fn fill_within_body(estimate: Vec<f64>, body: &[usize], labels: &[i32], label: i32, ny: usize, nx: usize, max_sweeps: usize) -> Vec<f64> {
    let mut current = estimate;
    // Where each body cell sits in `current`, found by binary search on
    // the row-major member list.
    let position = |g: usize| -> Option<usize> {
        if labels[g] != label {
            return None;
        }
        body.binary_search(&g).ok()
    };
    for _ in 0..max_sweeps {
        let waiting: Vec<usize> = (0..body.len()).filter(|&n| current[n].is_nan()).collect();
        if waiting.is_empty() {
            break;
        }
        if !current.iter().any(|v| v.is_finite()) {
            break;
        }
        let mut ready: Vec<(usize, f64)> = Vec::new();
        for &n in &waiting {
            let k = body[n];
            let j = k / nx;
            let i = k % nx;
            let mut accumulated = 0.0f64;
            let mut count = 0.0f64;
            let mut add = |g: usize| {
                let value = position(g).map(|p| current[p]).filter(|v| v.is_finite());
                accumulated += value.unwrap_or(0.0);
                count += if value.is_some() { 1.0 } else { 0.0 };
            };
            if j >= 1 {
                add(k - nx);
            }
            if j + 1 < ny {
                add(k + nx);
            }
            if i >= 1 {
                add(k - 1);
            }
            if i + 1 < nx {
                add(k + 1);
            }
            if count > 0.0 {
                ready.push((n, accumulated / count));
            }
        }
        if ready.is_empty() {
            break;
        }
        for (n, value) in ready {
            current[n] = value;
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hole_closes_from_its_own_body() {
        let (ny, nx) = (3usize, 3usize);
        let mut values = vec![280.0, 281.0, 282.0, 283.0, 0.0, 285.0, 286.0, 287.0, 288.0];
        let mut source = vec![1i8; 9];
        let water = vec![1u8; 9];
        let labels = vec![1i32; 9];
        let mut counts = [0u64; 3];
        let mut filled = vec![0u8; 9];
        let code = unsafe {
            gpuwm_water_repair_f64(values.as_mut_ptr(), source.as_mut_ptr(), water.as_ptr(), labels.as_ptr(), ny, nx, 170.0, 400.0, 5, 6, counts.as_mut_ptr(), filled.as_mut_ptr(), 3)
        };
        assert_eq!(code, OK);
        let want = (280.0 + 281.0 + 282.0 + 283.0 + 285.0 + 286.0 + 287.0 + 288.0) / 8.0;
        assert_eq!(values[4], want);
        assert_eq!(source[4], 5);
        assert_eq!(counts, [1, 0, 0]);
        assert_eq!(filled[4], 1);
    }

    #[test]
    fn a_negative_label_is_refused() {
        let mut values = vec![0.0; 4];
        let mut source = vec![0i8; 4];
        let water = vec![1u8; 4];
        let labels = vec![1, -1, 1, 1];
        let mut counts = [0u64; 3];
        let mut filled = vec![0u8; 4];
        let code = unsafe {
            gpuwm_water_repair_f64(values.as_mut_ptr(), source.as_mut_ptr(), water.as_ptr(), labels.as_ptr(), 2, 2, 170.0, 400.0, 5, 6, counts.as_mut_ptr(), filled.as_mut_ptr(), 1)
        };
        assert_eq!(code, ERR_LABEL);
    }

    #[test]
    fn nearest_donor_takes_the_first_of_a_tie() {
        let (ny, nx) = (30usize, 30usize);
        let mut donors = vec![false; ny * nx];
        donors[15 * nx + 2] = true;
        donors[15 * nx + 28] = true;
        let index = donor_index(&donors, ny, nx);
        let window = Window { j0: 14, j1: 17, i0: 14, i1: 17 };
        let mut body = vec![false; 9];
        body[4] = true;
        assert_eq!(nearest_donor(&body, window, &index, ny, nx), Some(15 * nx + 2));
    }
}
