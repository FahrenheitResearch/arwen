//! `merge_contributions`: one or more radars' accumulators into the
//! observation fields.
//!
//! Reflectivity merges across radars (sums and counts added, maxima of
//! maxima) onto the whole domain; radial velocity never merges and keeps a
//! leading radar axis whose planes cover each radar's own window, padded
//! to the widest (the v2 windowed layout).  Contributions are visited in
//! the caller's order, so every merged cell is formed from the same
//! addends in the same order as the reference's whole-array additions.

use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::request::{Arrays, Params};
use crate::{Result, SuperobFailure};

/// Floor on `||sum b_i|| / n` (`MIN_BEAM_COHERENCE`).
pub const MIN_BEAM_COHERENCE: f64 = 0.5;

#[derive(Debug, Deserialize)]
pub struct ContributionIn {
    pub window: [usize; 4],
    pub z_linear_sum: usize,
    pub z_count: usize,
    pub z0_count: usize,
    pub z_sum_dbz: usize,
    pub z_sumsq_dbz: usize,
    pub z_max_dbz: usize,
    pub vr_sum: usize,
    pub vr_sumsq: usize,
    pub vr_count: usize,
    pub beam_east: usize,
    pub beam_north: usize,
    pub beam_up: usize,
    pub vr_rejected: usize,
}

#[derive(Debug, Deserialize)]
pub struct MergeOutputsIn {
    pub z_obs: usize,
    pub z_mask: usize,
    pub z_err: usize,
    pub z_max: usize,
    pub z_mean: usize,
    pub z_count: usize,
    pub z0_mask: usize,
    pub z0_count: usize,
    pub z0_err: usize,
    pub vr_obs: usize,
    pub vr_mask: usize,
    pub vr_err: usize,
    pub vr_count: usize,
    pub vr_rejected: usize,
    pub vr_beam_east: usize,
    pub vr_beam_north: usize,
    pub vr_beam_up: usize,
    pub vr_beam_coherence: usize,
}

#[derive(Debug, Deserialize)]
pub struct MergeRequest {
    pub params: Params,
    pub z_reduce: String,
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
    pub max_nj: usize,
    pub max_ni: usize,
    pub contributions: Vec<ContributionIn>,
    pub outputs: MergeOutputsIn,
}

#[inline]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() { a } else { b }
}

struct Contribution<'a> {
    j0: usize,
    i0: usize,
    nj: usize,
    ni: usize,
    z_linear_sum: &'a [f64],
    z_count: &'a [i64],
    z0_count: &'a [i64],
    z_sum_dbz: &'a [f64],
    z_sumsq_dbz: &'a [f64],
    z_max_dbz: &'a [f64],
    vr_sum: &'a [f64],
    vr_sumsq: &'a [f64],
    vr_count: &'a [i64],
    beam_east: &'a [f64],
    beam_north: &'a [f64],
    beam_up: &'a [f64],
    vr_rejected: &'a [i64],
}

pub fn merge(request: &MergeRequest, arrays: &Arrays) -> Result<Value> {
    let params = &request.params;
    params.validate()?;
    let reduce_max = match request.z_reduce.as_str() {
        "max" => true,
        "mean" => false,
        other => {
            return Err(SuperobFailure::request(format!(
                "z_reduce must be 'max' or 'mean', got {other:?}"
            )));
        }
    };
    if request.contributions.is_empty() {
        return Err(SuperobFailure::request("no radar contributions to merge"));
    }
    let (nz, ny, nx) = (request.nz, request.ny, request.nx);
    let plane = ny * nx;
    let cells = nz * plane;
    let mut contributions = Vec::with_capacity(request.contributions.len());
    for (index, c) in request.contributions.iter().enumerate() {
        let [j0, j1, i0, i1] = c.window;
        if j1 < j0 || i1 < i0 || j1 >= ny || i1 >= nx {
            return Err(SuperobFailure::request(format!(
                "contribution {index} window j {j0}..{j1}, i {i0}..{i1} does not fit a grid of ({nz}, {ny}, {nx})"
            )));
        }
        let (nj, ni) = (j1 - j0 + 1, i1 - i0 + 1);
        if nj > request.max_nj || ni > request.max_ni {
            return Err(SuperobFailure::request("a window exceeds the padded plane"));
        }
        let size = nz * nj * ni;
        let contribution = Contribution {
            j0,
            i0,
            nj,
            ni,
            z_linear_sum: arrays.f64s(c.z_linear_sum, "z_linear_sum")?,
            z_count: arrays.i64s(c.z_count, "z_count")?,
            z0_count: arrays.i64s(c.z0_count, "z0_count")?,
            z_sum_dbz: arrays.f64s(c.z_sum_dbz, "z_sum_dbz")?,
            z_sumsq_dbz: arrays.f64s(c.z_sumsq_dbz, "z_sumsq_dbz")?,
            z_max_dbz: arrays.f64s(c.z_max_dbz, "z_max_dbz")?,
            vr_sum: arrays.f64s(c.vr_sum, "vr_sum")?,
            vr_sumsq: arrays.f64s(c.vr_sumsq, "vr_sumsq")?,
            vr_count: arrays.i64s(c.vr_count, "vr_count")?,
            beam_east: arrays.f64s(c.beam_east, "beam_east")?,
            beam_north: arrays.f64s(c.beam_north, "beam_north")?,
            beam_up: arrays.f64s(c.beam_up, "beam_up")?,
            vr_rejected: arrays.i64s(c.vr_rejected, "vr_rejected")?,
        };
        let lens = [
            contribution.z_linear_sum.len(), contribution.z_count.len(),
            contribution.z0_count.len(), contribution.z_sum_dbz.len(),
            contribution.z_sumsq_dbz.len(), contribution.z_max_dbz.len(),
            contribution.vr_sum.len(), contribution.vr_sumsq.len(),
            contribution.vr_count.len(), contribution.beam_east.len(),
            contribution.beam_north.len(), contribution.beam_up.len(),
            contribution.vr_rejected.len(),
        ];
        if lens.iter().any(|&len| len != size) {
            return Err(SuperobFailure::request(format!(
                "contribution {index}: every accumulator must hold {size} cells"
            )));
        }
        contributions.push(contribution);
    }

    let o = &request.outputs;
    arrays.check_disjoint(&[
        o.z_obs, o.z_mask, o.z_err, o.z_max, o.z_mean, o.z_count, o.z0_mask,
        o.z0_count, o.z0_err, o.vr_obs, o.vr_mask, o.vr_err, o.vr_count,
        o.vr_rejected, o.vr_beam_east, o.vr_beam_north, o.vr_beam_up,
        o.vr_beam_coherence,
    ])?;
    let z_obs = arrays.f64s_mut(o.z_obs, "z_obs")?;
    let z_mask = arrays.i8s_mut(o.z_mask, "z_mask")?;
    let z_err = arrays.f64s_mut(o.z_err, "z_err")?;
    let z_max = arrays.f64s_mut(o.z_max, "z_max")?;
    let z_mean = arrays.f64s_mut(o.z_mean, "z_mean")?;
    let z_count = arrays.i32s_mut(o.z_count, "z_count")?;
    let z0_mask = arrays.i8s_mut(o.z0_mask, "z0_mask")?;
    let z0_count = arrays.i32s_mut(o.z0_count, "z0_count")?;
    let z0_err = arrays.f64s_mut(o.z0_err, "z0_err")?;
    for len in [
        z_obs.len(), z_mask.len(), z_err.len(), z_max.len(), z_mean.len(),
        z_count.len(), z0_mask.len(), z0_count.len(), z0_err.len(),
    ] {
        if len != cells {
            return Err(SuperobFailure::request(format!(
                "reflectivity outputs must each hold {cells} cells"
            )));
        }
    }

    // --- reflectivity: compose the windows, in contribution order ---
    //
    // The output buffers double as the accumulators so a continental grid
    // needs no second set of whole-domain arrays: z_mean holds the linear
    // sum, z_err the dBZ sum, z0_err the dBZ sum of squares, z_max the
    // maximum.  The counts are summed in int32 with wrap-around, which is
    // the int64 sum followed by the reference's `astype(np.int32)`.
    z_mean.fill(0.0);
    z_err.fill(0.0);
    z0_err.fill(0.0);
    z_max.fill(f64::NEG_INFINITY);
    z_count.fill(0);
    z0_count.fill(0);
    for c in &contributions {
        let window_plane = c.nj * c.ni;
        let slabs = z_mean
            .par_chunks_mut(plane)
            .zip(z_err.par_chunks_mut(plane))
            .zip(z0_err.par_chunks_mut(plane))
            .zip(z_max.par_chunks_mut(plane))
            .zip(z_count.par_chunks_mut(plane))
            .zip(z0_count.par_chunks_mut(plane))
            .enumerate();
        slabs.for_each(|(k, (((((lin, sum), sumsq), mx), count), count0))| {
            for jl in 0..c.nj {
                for il in 0..c.ni {
                    let src = k * window_plane + jl * c.ni + il;
                    let dst = (c.j0 + jl) * nx + (c.i0 + il);
                    lin[dst] += c.z_linear_sum[src];
                    count[dst] = count[dst].wrapping_add(c.z_count[src] as i32);
                    count0[dst] = count0[dst].wrapping_add(c.z0_count[src] as i32);
                    sum[dst] += c.z_sum_dbz[src];
                    sumsq[dst] += c.z_sumsq_dbz[src];
                    mx[dst] = np_maximum(mx[dst], c.z_max_dbz[src]);
                }
            }
        });
    }
    let base_sq = params.z_error_base_dbz * params.z_error_base_dbz;
    let floor = params.z_error_floor_dbz;
    let min_gates = params.clear_air_min_gates;
    let clear_err = params.clear_air_error_dbz;
    z_obs
        .par_iter_mut()
        .zip(z_mask.par_iter_mut())
        .zip(z_err.par_iter_mut())
        .zip(z_max.par_iter_mut())
        .zip(z_mean.par_iter_mut())
        .zip(z_count.par_iter())
        .zip(z0_mask.par_iter_mut())
        .zip(z0_count.par_iter())
        .zip(z0_err.par_iter_mut())
        .for_each(|((((((((obs, mask), err), mx), mean), &count), mask0), &count0), err0)| {
            let lin = *mean;
            let sum = *err;
            let sumsq = *err0;
            let count = count as i64;
            let divisor = count.max(1) as f64;
            let has_z = count > 0;
            let z_mean_value = if has_z { 10.0 * (np_maximum(lin, 1e-30) / divisor).log10() } else { 0.0 };
            let mean_dbz = if has_z { sum / divisor } else { 0.0 };
            let variance = if count > 1 {
                np_maximum(sumsq / divisor - mean_dbz * mean_dbz, 0.0)
            } else {
                0.0
            };
            let z_max_value = if has_z { *mx } else { 0.0 };
            *mx = z_max_value;
            *mean = z_mean_value;
            *obs = if has_z { if reduce_max { z_max_value } else { z_mean_value } } else { 0.0 };
            *err = if has_z { np_maximum(floor, (base_sq / divisor + variance).sqrt()) } else { 0.0 };
            *mask = has_z as i8;
            let has_z0 = (count0 as i64 as f64) >= min_gates && !has_z;
            *mask0 = has_z0 as i8;
            *err0 = if has_z0 { clear_err } else { 0.0 };
        });

    // --- radial velocity: one padded plane per radar, never merged ---
    let n_radar = contributions.len();
    let vplane = nz * request.max_nj * request.max_ni;
    let vcells = n_radar * vplane;
    let vr_obs = arrays.f64s_mut(o.vr_obs, "vr_obs")?;
    let vr_mask = arrays.i8s_mut(o.vr_mask, "vr_mask")?;
    let vr_err = arrays.f64s_mut(o.vr_err, "vr_err")?;
    let vr_count = arrays.i32s_mut(o.vr_count, "vr_count")?;
    let vr_rejected = arrays.i32s_mut(o.vr_rejected, "vr_rejected")?;
    let beam_east = arrays.f64s_mut(o.vr_beam_east, "vr_beam_east")?;
    let beam_north = arrays.f64s_mut(o.vr_beam_north, "vr_beam_north")?;
    let beam_up = arrays.f64s_mut(o.vr_beam_up, "vr_beam_up")?;
    let coherence = arrays.f64s_mut(o.vr_beam_coherence, "vr_beam_coherence")?;
    for len in [
        vr_obs.len(), vr_mask.len(), vr_err.len(), vr_count.len(), vr_rejected.len(),
        beam_east.len(), beam_north.len(), beam_up.len(), coherence.len(),
    ] {
        if len != vcells {
            return Err(SuperobFailure::request(format!(
                "velocity outputs must each hold {vcells} cells"
            )));
        }
    }
    let vbase_sq = params.vr_error_base_ms * params.vr_error_base_ms;
    let vfloor = params.vr_error_floor_ms;
    let (max_nj, max_ni) = (request.max_nj, request.max_ni);
    vr_obs
        .par_chunks_mut(vplane)
        .zip(vr_mask.par_chunks_mut(vplane))
        .zip(vr_err.par_chunks_mut(vplane))
        .zip(vr_count.par_chunks_mut(vplane))
        .zip(vr_rejected.par_chunks_mut(vplane))
        .zip(beam_east.par_chunks_mut(vplane))
        .zip(beam_north.par_chunks_mut(vplane))
        .zip(beam_up.par_chunks_mut(vplane))
        .zip(coherence.par_chunks_mut(vplane))
        .zip(contributions.par_iter())
        .for_each(|(((((((((obs, mask), err), count_out), rejected), be), bn), bu), coh), c)| {
            obs.fill(0.0);
            mask.fill(0);
            err.fill(0.0);
            count_out.fill(0);
            rejected.fill(0);
            be.fill(0.0);
            bn.fill(0.0);
            bu.fill(0.0);
            coh.fill(0.0);
            for k in 0..nz {
                for jl in 0..c.nj {
                    for il in 0..c.ni {
                        let src = (k * c.nj + jl) * c.ni + il;
                        let dst = (k * max_nj + jl) * max_ni + il;
                        let count = c.vr_count[src];
                        let (e, n, u) = (c.beam_east[src], c.beam_north[src], c.beam_up[src]);
                        let norm = (e * e + n * n + u * u).sqrt();
                        let has_vr = count > 0 && norm > 0.0;
                        let divisor = count.max(1) as f64;
                        let mean = if has_vr { c.vr_sum[src] / divisor } else { 0.0 };
                        let variance = if count > 1 {
                            np_maximum(c.vr_sumsq[src] / divisor - mean * mean, 0.0)
                        } else {
                            0.0
                        };
                        let coherent = if count > 0 { norm / divisor } else { 0.0 };
                        let beamless = count > 0 && !has_vr;
                        let beam_divisor = if count > 0 { count as f64 } else { 1.0 };
                        let mut out_count = if has_vr { count as i32 } else { 0 };
                        let mut out_rejected =
                            (c.vr_rejected[src] + if beamless { count } else { 0 }) as i32;
                        let mut out_obs = if has_vr { mean } else { 0.0 };
                        let mut out_err = if has_vr {
                            np_maximum(vfloor, (vbase_sq / divisor + variance).sqrt())
                        } else {
                            0.0
                        };
                        let mut out_mask = has_vr as i8;
                        let (mut oe, mut on, mut ou) = if has_vr {
                            (e / beam_divisor, n / beam_divisor, u / beam_divisor)
                        } else {
                            (0.0, 0.0, 0.0)
                        };
                        let out_coherence = if has_vr { coherent } else { 0.0 };
                        if has_vr && coherent < MIN_BEAM_COHERENCE {
                            out_mask = 0;
                            out_rejected = out_rejected.wrapping_add(out_count);
                            out_count = 0;
                            out_obs = 0.0;
                            out_err = 0.0;
                            oe = 0.0;
                            on = 0.0;
                            ou = 0.0;
                        }
                        obs[dst] = out_obs;
                        mask[dst] = out_mask;
                        err[dst] = out_err;
                        count_out[dst] = out_count;
                        rejected[dst] = out_rejected;
                        be[dst] = oe;
                        bn[dst] = on;
                        bu[dst] = ou;
                        coh[dst] = out_coherence;
                    }
                }
            }
        });
    Ok(json!({"ok": true}))
}
