//! The lake skin search and the water-temperature blends in float64.
//!
//! These are the arithmetic of four single-core NumPy data steps, moved
//! here statement for statement:
//!
//! * `gpuwm.ingest.horiz._nearest_finite_source_water`, the per-lake
//!   search for the nearest source water cell (`gpuwm_lake_water_nearest_f64`);
//! * `gpuwm.ingest.water_temperature.normalized_masked_bilinear`, the
//!   bilinear blend renormalised over the donors that exist
//!   (`gpuwm_masked_bilinear_blend_f64`);
//! * `gpuwm.ingest.water_temperature._fill_within_component`, the
//!   four-neighbour sweep that closes a water body's holes from its own
//!   cells (`gpuwm_component_fill_f64`);
//! * the corner blend of `gpuwm.ingest.water_overlay.masked_bilinear_sample`
//!   (`gpuwm_overlay_bilinear_sample_f64`);
//! * `gpuwm.ingest.water_temperature._label_components`, the 8-connected
//!   labelling of water bodies, a per-cell Python loop
//!   (`gpuwm_label_components_8`).
//!
//! The NumPy code is kept as a test oracle only
//! (`gpuwm/verify/water_blend_oracle.py`), and every output must equal it
//! byte for byte:
//!
//! * every sum keeps NumPy's operand order, a term NumPy adds as `0.0`
//!   is added here as `0.0` (so `-0.0` sums round the same way), and
//!   nothing fuses a multiply and an add;
//! * a squared distance is `dy * dy + dx * dx`, as NumPy's `array ** 2`
//!   computes it, while the search's stopping bound is squared with the C
//!   library's `pow`, as Python's `float ** 2` computes it;
//! * the nearest cell is the first strictly nearer one in row-major order
//!   (`np.argmin`);
//! * every output element depends only on its own target (the sweep reads
//!   the previous sweep's field), and the only cross-target values are
//!   yes-or-no answers, so no result depends on the worker count.

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};

/// A lake target's search found no water in the whole source window.
pub const ERR_NO_WATER: i32 = 30;
/// A corner index lies outside the source.
pub const ERR_CORNER: i32 = 31;

/// Targets per unit of parallel work, pulled one at a time.
const CHUNK: usize = 2048;

extern "C" {
    #[link_name = "pow"]
    fn c_pow(x: f64, y: f64) -> f64;
}

/// `value ** 2` as CPython computes it for a float: the C library's `pow`.
/// The exponent passes through `black_box` so the compiler cannot rewrite
/// the call as `value * value`, which differs from `pow` in the last bit
/// for some values.
fn python_square(value: f64) -> f64 {
    unsafe { c_pow(value, std::hint::black_box(2.0)) }
}

/// Run `body(start, piece)` over `out` in chunks pulled by `workers` threads.
fn run_chunks<T: Send, F>(out: &mut [T], workers: usize, body: F)
where
    F: Fn(usize, &mut [T]) + Sync,
{
    let length = out.len();
    if length == 0 {
        return;
    }
    let count = workers.max(1).min(length.div_ceil(CHUNK));
    if count <= 1 {
        body(0, out);
        return;
    }
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

/// `_nearest_finite_source_water` for one target, or `None` when the whole
/// source holds no water.
pub fn lake_nearest(skin: &[f64], water: &[u8], ny: usize, nx: usize, y: f64, x: f64) -> Option<f64> {
    let last_y = ny as i64 - 1;
    let last_x = nx as i64 - 1;
    let mut radius = 8.0f64;
    loop {
        let j0 = 0i64.max((y - radius).ceil() as i64);
        let j1 = last_y.min((y + radius).floor() as i64);
        let i0 = 0i64.max((x - radius).ceil() as i64);
        let i1 = last_x.min((x + radius).floor() as i64);
        let mut best = f64::INFINITY;
        let mut at: Option<usize> = None;
        let mut row = j0;
        while row <= j1 {
            let dy = row as f64 - y;
            let base = row as usize * nx;
            let mut col = i0;
            while col <= i1 {
                let flat = base + col as usize;
                if water[flat] != 0 {
                    let dx = col as f64 - x;
                    let distance = dy * dy + dx * dx;
                    if at.is_none() || distance < best {
                        best = distance;
                        at = Some(flat);
                    }
                }
                col += 1;
            }
            row += 1;
        }
        if let Some(flat) = at {
            // Python's list, in its order, and min() keeps the first least.
            let mut bound: Option<f64> = None;
            let mut consider = |value: f64| {
                bound = Some(match bound {
                    Some(current) if !(value < current) => current,
                    _ => value,
                });
            };
            if j0 > 0 {
                consider(y - (j0 - 1) as f64);
            }
            if j1 < last_y {
                consider((j1 + 1) as f64 - y);
            }
            if i0 > 0 {
                consider(x - (i0 - 1) as f64);
            }
            if i1 < last_x {
                consider((i1 + 1) as f64 - x);
            }
            match bound {
                None => return Some(skin[flat]),
                Some(limit) if best < python_square(limit) => return Some(skin[flat]),
                _ => {}
            }
        }
        if j0 == 0 && j1 == last_y && i0 == 0 && i1 == last_x {
            return None;
        }
        radius *= 2.0;
    }
}

/// The nearest source water cell's skin temperature for every lake target.
///
/// `skin` and `water` are `(ny, nx)`; `target_y`/`target_x` the targets'
/// zero-based fractional source coordinates.  A target whose search finds
/// no water anywhere is refused with [`ERR_NO_WATER`].
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_lake_water_nearest_f64(
    skin: *const f64,
    water: *const u8,
    ny: usize,
    nx: usize,
    target_y: *const f64,
    target_x: *const f64,
    ntarget: usize,
    output: *mut f64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if ntarget == 0 {
            return OK;
        }
        if skin.is_null() || water.is_null() || target_y.is_null() || target_x.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if ny == 0 || nx == 0 || workers == 0 || ny > i64::MAX as usize / 2 || nx > i64::MAX as usize / 2 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let skin = std::slice::from_raw_parts(skin, cells);
        let water = std::slice::from_raw_parts(water, cells);
        let ty = std::slice::from_raw_parts(target_y, ntarget);
        let tx = std::slice::from_raw_parts(target_x, ntarget);
        if ty.iter().chain(tx.iter()).any(|value| !value.is_finite()) {
            return ERR_NONFINITE;
        }
        let out = std::slice::from_raw_parts_mut(output, ntarget);
        let lost = std::sync::atomic::AtomicBool::new(false);
        run_chunks(out, workers, |start, piece| {
            for (offset, slot) in piece.iter_mut().enumerate() {
                let t = start + offset;
                match lake_nearest(skin, water, ny, nx, ty[t], tx[t]) {
                    Some(value) => *slot = value,
                    None => {
                        *slot = f64::NAN;
                        lost.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        });
        if lost.load(std::sync::atomic::Ordering::Relaxed) {
            ERR_NO_WATER
        } else {
            OK
        }
    }))
    .unwrap_or(ERR_PANIC)
}

/// `normalized_masked_bilinear`: per target, over `ncorner` corners in
/// order, `numerator += weight * present * safe` and
/// `denominator += weight * present`; the value is their quotient where
/// the denominator exceeds `denominator_floor`, NaN elsewhere.
///
/// `corner_y`, `corner_x` and `weight` are `(ncorner, ntarget)`.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_masked_bilinear_blend_f64(
    field: *const f64,
    donors: *const u8,
    ny: usize,
    nx: usize,
    corner_y: *const i64,
    corner_x: *const i64,
    weight: *const f64,
    ncorner: usize,
    ntarget: usize,
    denominator_floor: f64,
    output: *mut f64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if ntarget == 0 {
            return OK;
        }
        if field.is_null() || donors.is_null() || output.is_null() || (ncorner > 0 && (corner_y.is_null() || corner_x.is_null() || weight.is_null())) {
            return ERR_NULL;
        }
        if ny == 0 || nx == 0 || workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let stacked = match ncorner.checked_mul(ntarget) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let field = std::slice::from_raw_parts(field, cells);
        let donors = std::slice::from_raw_parts(donors, cells);
        let (cy, cx, w): (&[i64], &[i64], &[f64]) = if stacked == 0 {
            (&[], &[], &[])
        } else {
            (
                std::slice::from_raw_parts(corner_y, stacked),
                std::slice::from_raw_parts(corner_x, stacked),
                std::slice::from_raw_parts(weight, stacked),
            )
        };
        if cy.iter().any(|&v| v < 0 || v as u64 >= ny as u64) || cx.iter().any(|&v| v < 0 || v as u64 >= nx as u64) {
            return ERR_CORNER;
        }
        let out = std::slice::from_raw_parts_mut(output, ntarget);
        run_chunks(out, workers, |start, piece| {
            for (offset, slot) in piece.iter_mut().enumerate() {
                let t = start + offset;
                let mut numerator = 0.0f64;
                let mut denominator = 0.0f64;
                for corner in 0..ncorner {
                    let at = corner * ntarget + t;
                    let cell = cy[at] as usize * nx + cx[at] as usize;
                    let donor = donors[cell] != 0;
                    let present = if donor { 1.0f64 } else { 0.0f64 };
                    let safe = if donor { field[cell] } else { 0.0f64 };
                    let term = w[at] * present;
                    numerator += term * safe;
                    denominator += term;
                }
                *slot = if denominator > denominator_floor {
                    numerator / denominator
                } else {
                    f64::NAN
                };
            }
        });
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// `_fill_within_component`, in place on `values` `(ny, nx)`, whose cells
/// outside `component` the caller has already set to NaN.  Writes the
/// number of sweeps that changed a cell to `sweeps_run` when not null.
///
/// Each sweep gives every NaN component cell with a finite four-neighbour
/// the mean of those neighbours, read from the previous sweep, adding the
/// neighbours in NumPy's order (above, below, left, right) and adding
/// `0.0` for an in-grid neighbour that is not finite component water.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_component_fill_f64(
    values: *mut f64,
    component: *const u8,
    ny: usize,
    nx: usize,
    max_sweeps: usize,
    sweeps_run: *mut u64,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if values.is_null() || component.is_null() {
            return ERR_NULL;
        }
        if workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let out = std::slice::from_raw_parts_mut(values, cells);
        let component = std::slice::from_raw_parts(component, cells);
        let mut changed = 0u64;
        // Each cell's value for this sweep and whether it is ready.
        let mut next = vec![(0.0f64, false); cells];
        for _ in 0..max_sweeps {
            let holes = (0..cells).any(|k| component[k] != 0 && out[k].is_nan());
            if !holes {
                break;
            }
            let have = |k: usize| component[k] != 0 && out[k].is_finite();
            if !(0..cells).any(have) {
                break;
            }
            let current: &[f64] = out;
            run_chunks(&mut next, workers, |start, piece| {
                for (offset, slot) in piece.iter_mut().enumerate() {
                    let k = start + offset;
                    *slot = (0.0, false);
                    if !(component[k] != 0 && current[k].is_nan()) {
                        continue;
                    }
                    let j = k / nx;
                    let i = k % nx;
                    let mut accumulated = 0.0f64;
                    let mut count = 0.0f64;
                    let mut add = |n: usize| {
                        let seeded = component[n] != 0 && current[n].is_finite();
                        accumulated += if seeded { current[n] } else { 0.0 };
                        count += if seeded { 1.0 } else { 0.0 };
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
                        *slot = (accumulated / count, true);
                    }
                }
            });
            let mut any_ready = false;
            for k in 0..cells {
                if next[k].1 {
                    any_ready = true;
                    out[k] = next[k].0;
                }
            }
            if !any_ready {
                break;
            }
            changed += 1;
        }
        if !sweeps_run.is_null() {
            *sweeps_run = changed;
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// The corner blend of `masked_bilinear_sample`: per target, the corners
/// `(y0, x0)`, `(y0, x0 + 1)`, `(y0 + 1, x0)`, `(y0 + 1, x0 + 1)` with the
/// weights `(1 - fy)(1 - fx)`, `(1 - fy) fx`, `fy (1 - fx)`, `fy fx`; a
/// valid corner adds its weight to `total` and `weight * value` to the
/// sum.  `covered` is `inside and total > 0`; the value is `sum / total`
/// where covered, NaN elsewhere.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_overlay_bilinear_sample_f64(
    temperature: *const f64,
    valid: *const u8,
    ny: usize,
    nx: usize,
    y0: *const i64,
    x0: *const i64,
    fy: *const f64,
    fx: *const f64,
    inside: *const u8,
    ntarget: usize,
    output: *mut f64,
    covered: *mut u8,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if ntarget == 0 {
            return OK;
        }
        if temperature.is_null() || valid.is_null() || y0.is_null() || x0.is_null() || fy.is_null() || fx.is_null() || inside.is_null() || output.is_null() || covered.is_null() {
            return ERR_NULL;
        }
        if ny < 2 || nx < 2 || workers == 0 {
            return ERR_DIMENSION;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let temperature = std::slice::from_raw_parts(temperature, cells);
        let valid = std::slice::from_raw_parts(valid, cells);
        let y0 = std::slice::from_raw_parts(y0, ntarget);
        let x0 = std::slice::from_raw_parts(x0, ntarget);
        let fy = std::slice::from_raw_parts(fy, ntarget);
        let fx = std::slice::from_raw_parts(fx, ntarget);
        let inside = std::slice::from_raw_parts(inside, ntarget);
        if y0.iter().any(|&v| v < 0 || v as u64 + 1 >= ny as u64) || x0.iter().any(|&v| v < 0 || v as u64 + 1 >= nx as u64) {
            return ERR_CORNER;
        }
        let out = std::slice::from_raw_parts_mut(output, ntarget);
        let cover = std::slice::from_raw_parts_mut(covered, ntarget);
        // One pass writes the value and the flag of each target.
        let mut both: Vec<(f64, u8)> = vec![(f64::NAN, 0); ntarget];
        run_chunks(&mut both, workers, |start, piece| {
            for (offset, slot) in piece.iter_mut().enumerate() {
                let t = start + offset;
                let (a, b) = (fy[t], fx[t]);
                let base = y0[t] as usize * nx + x0[t] as usize;
                let corners = [
                    (base, (1.0 - a) * (1.0 - b)),
                    (base + 1, (1.0 - a) * b),
                    (base + nx, a * (1.0 - b)),
                    (base + nx + 1, a * b),
                ];
                let mut total = 0.0f64;
                let mut accumulated = 0.0f64;
                for (cell, w) in corners {
                    let ok = valid[cell] != 0;
                    let contribution = if ok { w } else { 0.0 };
                    total += contribution;
                    accumulated += contribution * if ok { temperature[cell] } else { 0.0 };
                }
                let is_covered = inside[t] != 0 && total > 0.0;
                *slot = if is_covered {
                    (accumulated / total, 1)
                } else {
                    (f64::NAN, 0)
                };
            }
        });
        for (t, (value, flag)) in both.into_iter().enumerate() {
            out[t] = value;
            cover[t] = flag;
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// `_label_components`: the 8-connected components of `mask` `(ny, nx)`,
/// labelled 1, 2, ... in the row-major order of each component's first
/// cell (0 off the mask), and their number.
///
/// Those labels are a function of the components alone, so this two-pass
/// union-find (the oracle's own method) gives the oracle's labels exactly.
/// The scan is sequential by nature and runs on one thread; it replaces a
/// per-cell Python loop.
///
/// # Safety
/// Every pointer must reference a buffer of the stated length.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_label_components_8(
    mask: *const u8,
    ny: usize,
    nx: usize,
    labels: *mut i32,
    count: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if count.is_null() || ((ny > 0 && nx > 0) && (mask.is_null() || labels.is_null())) {
            return ERR_NULL;
        }
        let cells = match ny.checked_mul(nx) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        if cells == 0 {
            *count = 0;
            return OK;
        }
        if cells > i32::MAX as usize {
            return ERR_DIMENSION;
        }
        let mask = std::slice::from_raw_parts(mask, cells);
        let out = std::slice::from_raw_parts_mut(labels, cells);
        let mut provisional = vec![0u32; cells];
        let mut parent: Vec<u32> = vec![0];
        fn find(parent: &mut [u32], mut a: u32) -> u32 {
            let mut root = a;
            while parent[root as usize] != root {
                root = parent[root as usize];
            }
            while parent[a as usize] != root {
                let next = parent[a as usize];
                parent[a as usize] = root;
                a = next;
            }
            root
        }
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if mask[k] == 0 {
                    continue;
                }
                let mut neighbours = [0u32; 4];
                let mut n = 0;
                let mut take = |label: u32| {
                    if label != 0 {
                        neighbours[n] = label;
                        n += 1;
                    }
                };
                if j > 0 {
                    take(provisional[k - nx]);
                    if i > 0 {
                        take(provisional[k - nx - 1]);
                    }
                    if i + 1 < nx {
                        take(provisional[k - nx + 1]);
                    }
                }
                if i > 0 {
                    take(provisional[k - 1]);
                }
                if n == 0 {
                    let label = parent.len() as u32;
                    parent.push(label);
                    provisional[k] = label;
                } else {
                    let smallest = *neighbours[..n].iter().min().unwrap();
                    provisional[k] = smallest;
                    for &other in &neighbours[..n] {
                        let ra = find(&mut parent, smallest);
                        let rb = find(&mut parent, other);
                        if ra != rb {
                            parent[ra.max(rb) as usize] = ra.min(rb);
                        }
                    }
                }
            }
        }
        let mut remap = vec![0i32; parent.len()];
        let mut next = 0i32;
        for k in 0..cells {
            let label = provisional[k];
            if label == 0 {
                out[k] = 0;
                continue;
            }
            let root = find(&mut parent, label) as usize;
            if remap[root] == 0 {
                next += 1;
                remap[root] = next;
            }
            out[k] = remap[root];
        }
        *count = next as u64;
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_square_is_the_c_library_pow() {
        for value in [0.0, 0.5, 1.5, 3.25, 7.999999999, 1e-3] {
            assert_eq!(python_square(value).to_bits(), unsafe { c_pow(value, 2.0) }.to_bits());
        }
    }

    #[test]
    fn lake_nearest_takes_the_first_of_a_tie_after_widening() {
        let (ny, nx) = (40usize, 40usize);
        let mut water = vec![0u8; ny * nx];
        let skin: Vec<f64> = (0..ny * nx).map(|k| k as f64).collect();
        water[20 * nx + 30] = 1;
        water[20 * nx + 10] = 1;
        // Equidistant at 10 cells: widening past 8 finds both; row-major
        // order picks column 10.
        let got = lake_nearest(&skin, &water, ny, nx, 20.0, 20.0).unwrap();
        assert_eq!(got, (20 * nx + 10) as f64);
    }

    #[test]
    fn lake_nearest_without_water_is_none() {
        let water = vec![0u8; 9];
        let skin = vec![1.0; 9];
        assert!(lake_nearest(&skin, &water, 3, 3, 1.0, 1.0).is_none());
    }

    #[test]
    fn outputs_do_not_change_with_worker_count() {
        let (ny, nx) = (50usize, 70usize);
        let mut water = vec![0u8; ny * nx];
        let mut skin = vec![0.0; ny * nx];
        for k in 0..ny * nx {
            water[k] = ((k * 7919) % 13 == 0) as u8;
            skin[k] = 270.0 + (k % 97) as f64 * 0.37;
        }
        let n = 5000;
        let ty: Vec<f64> = (0..n).map(|k| (k * 31 % 4900) as f64 / 100.0).collect();
        let tx: Vec<f64> = (0..n).map(|k| (k * 17 % 6900) as f64 / 100.0).collect();
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        unsafe {
            assert_eq!(gpuwm_lake_water_nearest_f64(skin.as_ptr(), water.as_ptr(), ny, nx, ty.as_ptr(), tx.as_ptr(), n, a.as_mut_ptr(), 1), OK);
            assert_eq!(gpuwm_lake_water_nearest_f64(skin.as_ptr(), water.as_ptr(), ny, nx, ty.as_ptr(), tx.as_ptr(), n, b.as_mut_ptr(), 7), OK);
        }
        assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn component_fill_closes_a_hole_from_its_neighbours() {
        let (ny, nx) = (3usize, 3usize);
        let nan = f64::NAN;
        let mut values = vec![nan, 2.0, nan, 4.0, nan, 6.0, nan, 8.0, nan];
        let component = vec![1u8; 9];
        let mut sweeps = 0u64;
        unsafe {
            assert_eq!(gpuwm_component_fill_f64(values.as_mut_ptr(), component.as_ptr(), ny, nx, 1000, &mut sweeps, 3), OK);
        }
        assert_eq!(values[4], (((0.0 + 2.0) + 8.0) + 4.0 + 6.0) / 4.0);
        assert!(values.iter().all(|v| v.is_finite()));
        assert_eq!(sweeps, 1);
    }

    #[test]
    fn components_are_eight_connected_and_numbered_by_first_cell() {
        // A diagonal pinch joins; a U closes late; a lone cell is its own.
        let mask: Vec<u8> = vec![
            1, 0, 0, 1, 0, //
            0, 1, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 0, 0, 0, 1, //
            1, 0, 0, 0, 0, //
        ];
        let mut labels = vec![-1i32; 25];
        let mut count = 0u64;
        unsafe {
            assert_eq!(gpuwm_label_components_8(mask.as_ptr(), 5, 5, labels.as_mut_ptr(), &mut count), OK);
        }
        assert_eq!(count, 2);
        assert_eq!(labels[0], 1);
        assert_eq!(labels[3], 1);
        assert_eq!(labels[18], 0);
        assert_eq!(labels[19], 1);
        assert_eq!(labels[20], 2);
    }

    #[test]
    fn blend_refuses_a_corner_off_the_source() {
        let field = vec![1.0; 4];
        let donors = vec![1u8; 4];
        let cy = vec![0i64, 2];
        let cx = vec![0i64, 0];
        let w = vec![0.5, 0.5];
        let mut out = vec![0.0; 1];
        let code = unsafe { gpuwm_masked_bilinear_blend_f64(field.as_ptr(), donors.as_ptr(), 2, 2, cy.as_ptr(), cx.as_ptr(), w.as_ptr(), 2, 1, 1e-6, out.as_mut_ptr(), 1) };
        assert_eq!(code, ERR_CORNER);
    }
}
