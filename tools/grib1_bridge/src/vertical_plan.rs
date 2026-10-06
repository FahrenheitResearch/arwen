//! Immutable WRF column geometry shared by repeated field and mask replays.
//! Field arithmetic calls the original Lagrange expression without factoring.
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicI32, Ordering};
use crate::{lagrange, parallel, ERR_DIMENSION, ERR_INTERPOLATION_WINDOW,
    ERR_NONFINITE, ERR_NULL, ERR_PANIC, ERR_PRESSURE_ORDER,
    ERR_SURFACE_BRACKET, ERR_TARGET_ABOVE_TOP, OK, TOP_COLOCATION_RTOL};

const SURFACE: u32 = u32::MAX;
/// An optional plan was unavailable or its immutable input bytes changed.
/// The driver keeps the original checked interpolation road for this result.
pub const PLAN_UNAVAILABLE: i32 = 126;

pub struct VerticalPlan {
    nsource: usize, ntarget: usize, ncolumn: usize,
    ascending: bool,
    source: Vec<f32>, surface: Vec<f32>, target: Vec<f32>,
    indices: Vec<u32>, x: Vec<f32>, lengths: Vec<u32>, deepest: Vec<f32>,
    lower: Vec<u32>, target_x: Vec<f32>, target_pressure: Vec<f32>,
}

fn allocate<T: Clone + Default>(length: usize) -> Result<Vec<T>, i32> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| PLAN_UNAVAILABLE)?;
    values.resize(length, T::default());
    Ok(values)
}

fn planned_bytes(nsource: usize, ntarget: usize, ncolumn: usize) -> Option<usize> {
    // Three exact input snapshots; assembled indices/logs; per-column
    // metadata; target brackets/logs/pressures. Every retained byte is priced.
    let source = nsource.checked_mul(ncolumn)?;
    let target = ntarget.checked_mul(ncolumn)?;
    let assembled = nsource.checked_add(1)?.checked_mul(ncolumn)?;
    source.checked_mul(4)?.checked_add(target.checked_mul(4)?)?
        .checked_add(ncolumn.checked_mul(4)?)?
        .checked_add(assembled.checked_mul(8)?)?
        .checked_add(ncolumn.checked_mul(8)?)?
        .checked_add(target.checked_mul(12)?)?
        .checked_add(std::mem::size_of::<VerticalPlan>())
}

#[derive(Clone, Copy)]
struct GeometryOutput {
    indices: usize, x: usize, lengths: usize, deepest: usize,
    lower: usize, target_x: usize, target_pressure: usize,
}

#[allow(clippy::too_many_arguments)]
unsafe fn column_geometry(source: &[f32], surface: &[f32], target: &[f32],
                   n: usize, ntarget: usize, nc: usize, output: GeometryOutput,
                   column: usize, logp: bool, force: usize, zap: f32,
                   ascending: bool,
                   indices: &mut Vec<u32>, pressure: &mut Vec<f32>) -> Result<(), i32> {
    let source_index = |level: usize| if ascending { n - 1 - level } else { level };
    let source_pressure = |level: usize| source[source_index(level) * nc + column];
    let psfc = surface[column];
    if !psfc.is_finite() || psfc <= 0.0 { return Err(ERR_NONFINITE); }
    let mut previous = f32::INFINITY;
    let mut first_above = None;
    for level in 0..n {
        let p = source_pressure(level);
        if !p.is_finite() || p <= 0.0 { return Err(ERR_NONFINITE); }
        if level > 0 && p >= previous { return Err(ERR_PRESSURE_ORDER); }
        previous = p;
        if first_above.is_none() && p < psfc { first_above = Some(level); }
    }
    let first_above = first_above.ok_or(ERR_SURFACE_BRACKET)?;
    indices.clear(); pressure.clear();
    if first_above > 0 {
        for level in 0..first_above {
            indices.push(source_index(level) as u32);
            pressure.push(source_pressure(level));
        }
        if pressure[pressure.len() - 1] - psfc < zap {
            pressure.pop(); indices.pop();
        }
        pressure.push(psfc); indices.push(SURFACE);
        let mut next = first_above;
        if force > 0 {
            let pforce = target[(force - 1) * nc + column];
            for level in first_above..n {
                if source_pressure(level) <= pforce {
                    next = level; break;
                }
            }
        }
        let start = if pressure[pressure.len() - 1]
            - source_pressure(next) < zap { next + 1 } else { next };
        for level in start..n {
            pressure.push(source_pressure(level)); indices.push(source_index(level) as u32);
        }
    } else {
        pressure.push(psfc); indices.push(SURFACE);
        let mut next = 0;
        if force > 0 {
            let pforce = target[(force - 1) * nc + column];
            for level in 0..n {
                if source_pressure(level) <= pforce { next = level; break; }
            }
        }
        for level in next..n {
            let p = source_pressure(level);
            if pressure[pressure.len() - 1] - p < zap && level < n - 1 { continue; }
            pressure.push(p); indices.push(source_index(level) as u32);
        }
    }
    if pressure.len() < 2 { return Err(ERR_INTERPOLATION_WINDOW); }
    let base = column * (n + 1);
    let count = pressure.len();
    *(output.lengths as *mut u32).add(column) = count as u32;
    *(output.deepest as *mut f32).add(column) = pressure[0];
    for index in 0..count {
        *(output.indices as *mut u32).add(base + index) = indices[index];
        *(output.x as *mut f32).add(base + index) =
            if logp { pressure[index].ln() } else { pressure[index] };
    }
    let x = std::slice::from_raw_parts((output.x as *const f32).add(base), count);
    for level in 0..ntarget {
        let mut p = target[level * nc + column];
        if !p.is_finite() || p <= 0.0 { return Err(ERR_NONFINITE); }
        let top = pressure[count - 1];
        if p < top {
            if top - p > TOP_COLOCATION_RTOL * top.abs() {
                return Err(ERR_TARGET_ABOVE_TOP);
            }
            p = top;
        }
        let tx = if logp { p.ln() } else { p };
        let found = (0..count - 1).find(|&lower|
            (tx - x[lower]) * (tx - x[lower + 1]) <= 0.0);
        let slot = column * ntarget + level;
        *(output.lower as *mut u32).add(slot) = match found {
            Some(lower) => lower as u32,
            None if p > pressure[0] => SURFACE,
            None => return Err(ERR_TARGET_ABOVE_TOP),
        };
        *(output.target_x as *mut f32).add(slot) = tx;
        *(output.target_pressure as *mut f32).add(slot) = p;
    }
    Ok(())
}

/// Make immutable geometry under a caller's bounded memory allowance.
///
/// # Safety
/// All input buffers have the declared dimensions and remain unchanged
/// for this call. `out` and `bytes` address writable disjoint result slots.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wrf_vertical_plan_new(
    source: *const f32, surface: *const f32, target: *const f32,
    nsource: usize, ntarget: usize, ncolumn: usize, logp: i32,
    force: usize, zap: f32, workers: usize, budget: usize,
    out: *mut *mut VerticalPlan, bytes: *mut usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || surface.is_null() || target.is_null()
            || out.is_null() || bytes.is_null() { return ERR_NULL; }
        *out = std::ptr::null_mut(); *bytes = 0;
        if nsource < 2 || ntarget == 0 || ncolumn == 0 || workers == 0
            || force > ntarget || !(0..=1).contains(&logp)
            || !zap.is_finite() || zap < 0.0 { return ERR_DIMENSION; }
        if nsource >= SURFACE as usize { return PLAN_UNAVAILABLE; }
        let required = match planned_bytes(nsource, ntarget, ncolumn).and_then(|size|
            (nsource + 1).checked_mul(8)?.checked_mul(
                workers.min(ncolumn).min(parallel::resources::available_workers()))?
                .checked_add(size)) {
            Some(size) => size, None => return PLAN_UNAVAILABLE,
        };
        *bytes = required;
        if required > budget { return PLAN_UNAVAILABLE; }
        let build = || -> Result<VerticalPlan, i32> {
            let ns = nsource * ncolumn;
            let nt = ntarget * ncolumn;
            let na = (nsource + 1) * ncolumn;
            let mut plan = VerticalPlan {
                nsource, ntarget, ncolumn, ascending: false,
                source: allocate(ns)?, surface: allocate(ncolumn)?, target: allocate(nt)?,
                indices: allocate(na)?, x: allocate(na)?, lengths: allocate(ncolumn)?,
                deepest: allocate(ncolumn)?, lower: allocate(nt)?, target_x: allocate(nt)?,
                target_pressure: allocate(nt)?,
            };
            plan.source.copy_from_slice(std::slice::from_raw_parts(source, ns));
            // Keep the caller's original bytes. Every column must follow
            // the same strict orientation; logical descending assembly
            // maps directly to original source-plane indices in Rust.
            plan.ascending = plan.source[ncolumn] > plan.source[0];
            plan.surface.copy_from_slice(std::slice::from_raw_parts(surface, ncolumn));
            plan.target.copy_from_slice(std::slice::from_raw_parts(target, nt));
            // A task writes only its columns in every retained flat array.
            let output = GeometryOutput {
                indices: plan.indices.as_mut_ptr() as usize,
                x: plan.x.as_mut_ptr() as usize,
                lengths: plan.lengths.as_mut_ptr() as usize,
                deepest: plan.deepest.as_mut_ptr() as usize,
                lower: plan.lower.as_mut_ptr() as usize,
                target_x: plan.target_x.as_mut_ptr() as usize,
                target_pressure: plan.target_pressure.as_mut_ptr() as usize,
            };
            let source = &plan.source;
            let surface = &plan.surface;
            let target = &plan.target;
            let ascending = plan.ascending;
            let error = AtomicI32::new(OK);
            parallel::run_ranges(ncolumn, workers, |start, stop| {
                let mut indices = Vec::with_capacity(nsource + 1);
                let mut pressure = Vec::with_capacity(nsource + 1);
                for column in start..stop {
                    if error.load(Ordering::Relaxed) != OK { break; }
                    // The helper uses indexed slots owned by this column.
                    let result = column_geometry(source, surface, target,
                        nsource, ntarget, ncolumn, output, column, logp != 0,
                        force, zap, ascending, &mut indices, &mut pressure);
                    if let Err(code) = result {
                        error.compare_exchange(OK, code, Ordering::Relaxed, Ordering::Relaxed).ok();
                    }
                }
            });
            let code = error.load(Ordering::Relaxed);
            if code != OK { return Err(code); }
            Ok(plan)
        };
        match build() {
            Ok(plan) => { *out = Box::into_raw(Box::new(plan)); OK }
            Err(code) => code,
        }
    })).unwrap_or(ERR_PANIC)
}

/// Release only a plan returned by `gpuwm_wrf_vertical_plan_new`.
/// # Safety
/// The handle is live, not in use and is released exactly once.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wrf_vertical_plan_free(plan: *mut VerticalPlan) {
    if !plan.is_null() { drop(Box::from_raw(plan)); }
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Apply checked field bytes with immutable geometry and original arithmetic.
/// # Safety
/// The live handle, all declared input buffers and output remain disjoint.
/// Every input stays unchanged until the call returns. Outputs have
/// `ntarget*ncolumn` elements. The plan may be shared by read-only callers.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wrf_vertical_plan_apply(
    plan: *const VerticalPlan, field: *const f32, surface_field: *const f32,
    source: *const f32, surface: *const f32, target: *const f32,
    output: *mut f32, extrap_temperature: i32, vboundb: usize, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if plan.is_null() || field.is_null() || surface_field.is_null()
            || source.is_null() || surface.is_null() || target.is_null()
            || output.is_null() { return ERR_NULL; }
        if workers == 0 || !(0..=1).contains(&extrap_temperature) { return ERR_DIMENSION; }
        let plan = &*plan;
        let ns = plan.nsource * plan.ncolumn;
        let nt = plan.ntarget * plan.ncolumn;
        if !same(&plan.source, std::slice::from_raw_parts(source, ns))
            || !same(&plan.surface, std::slice::from_raw_parts(surface, plan.ncolumn))
            || !same(&plan.target, std::slice::from_raw_parts(target, nt)) {
            return PLAN_UNAVAILABLE;
        }
        let field = std::slice::from_raw_parts(field, ns);
        let sf = std::slice::from_raw_parts(surface_field, plan.ncolumn);
        let output_address = output as usize;
        let error = AtomicI32::new(OK);
        parallel::run_ranges(plan.ncolumn, workers, |start, stop| {
            let mut y = Vec::with_capacity(plan.nsource + 1);
            for column in start..stop {
                if error.load(Ordering::Relaxed) != OK { break; }
                let mut evaluate = || -> Result<(), i32> {
                    if !sf[column].is_finite() || (0..plan.nsource).any(|level|
                        !field[level * plan.ncolumn + column].is_finite()) {
                        return Err(ERR_NONFINITE);
                    }
                    let base = column * (plan.nsource + 1);
                    let count = plan.lengths[column] as usize;
                    y.clear();
                    for &index in &plan.indices[base..base + count] {
                        y.push(if index == SURFACE { sf[column] }
                               else { field[index as usize * plan.ncolumn + column] });
                    }
                    let x = &plan.x[base..base + count];
                    for level in 0..plan.ntarget {
                        let slot = column * plan.ntarget + level;
                        let lower = plan.lower[slot];
                        let result = if lower != SURFACE {
                            let lower = lower as usize;
                            let tx = plan.target_x[slot];
                            if level + 1 >= 1 + vboundb {
                                let upper = lower + 2 <= count - 1;
                                let previous = lower >= 1;
                                if upper && previous {
                                    0.5 * (lagrange(&x[lower..lower + 3], &y[lower..lower + 3], 2, tx)
                                        + lagrange(&x[lower - 1..lower + 2], &y[lower - 1..lower + 2], 2, tx))
                                } else if upper {
                                    lagrange(&x[lower..lower + 3], &y[lower..lower + 3], 2, tx)
                                } else if previous {
                                    lagrange(&x[lower - 1..lower + 2], &y[lower - 1..lower + 2], 2, tx)
                                } else { return Err(ERR_INTERPOLATION_WINDOW); }
                            } else { lagrange(&x[lower..lower + 2], &y[lower..lower + 2], 1, tx) }
                        } else if extrap_temperature != 0 {
                            let pressure = plan.target_pressure[slot];
                            let deepest = plan.deepest[column];
                            let t1 = y[0] * (deepest / 100000.0).powf(0.2857143);
                            let average_pressure = 0.5 * (pressure + deepest);
                            let dhdp = 11880.516 * 0.1902632 * (average_pressure / 100.0).powf(0.1902632 - 1.0);
                            let dt = dhdp * ((pressure - deepest) / 100.0) * 0.0065;
                            (t1 + dt) * (100000.0 / pressure).powf(0.2857143)
                        } else { y[0] };
                        if !result.is_finite() { return Err(ERR_NONFINITE); }
                        *(output_address as *mut f32).add(level * plan.ncolumn + column) = result;
                    }
                    Ok(())
                };
                if let Err(code) = evaluate() {
                    error.compare_exchange(OK, code, Ordering::Relaxed, Ordering::Relaxed).ok();
                }
            }
        });
        error.load(Ordering::Relaxed)
    })).unwrap_or(ERR_PANIC)
}
