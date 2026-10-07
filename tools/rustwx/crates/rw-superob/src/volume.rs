//! `superob_volume`: one radar volume onto its reach window of the grid.
//!
//! Accumulators only; the reduction to observations happens once, at the
//! merge.  The order of every reduction is the reference's: sweeps in pack
//! order, moments in pack order, radials ascending, gates ascending -- the
//! sequence `numpy.add.at` applies the Python stage's additions in, so a
//! cell's floating-point sum is formed from the same addends in the same
//! order.  Gate geometry, which is order-free, runs in parallel by radial.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::ccqc::{self, Masker, PlaneRef, Plan, SweepPlanInput};
use crate::geometry::{Propagation, Site};
use crate::grid::GridView;
use crate::request::{Arrays, CcParams, DealiasUse, GridIn, Params, de_f64, de_opt_f64};
use crate::{Result, SuperobFailure};

pub const REFLECTIVITY: &str = "REF";
pub const VELOCITY: &str = "VEL";
pub const CORRELATION: &str = "RHO";

const MEASURED: u8 = 0;
const BELOW_THRESHOLD: u8 = 1;
const RANGE_FOLDED: u8 = 2;
const NOT_COLLECTED: u8 = 3;
const NODATA: u8 = 4;
const SENTINEL_AMBIGUOUS: u8 = 5;

#[derive(Debug, Deserialize)]
pub struct SiteIn {
    pub id: String,
    #[serde(deserialize_with = "de_f64")]
    pub lat_deg: f64,
    #[serde(deserialize_with = "de_f64")]
    pub lon_deg: f64,
    #[serde(deserialize_with = "de_f64")]
    pub alt_m: f64,
}

#[derive(Debug, Deserialize)]
pub struct MomentIn {
    pub product: String,
    pub gate_count: usize,
    /// The data plane's second dimension, which must equal `gate_count`.
    pub columns: usize,
    #[serde(deserialize_with = "de_f64")]
    pub first_gate_range_m: f64,
    #[serde(deserialize_with = "de_f64")]
    pub gate_size_m: f64,
    pub data: usize,
    #[serde(default)]
    pub censor: Option<usize>,
    /// The unfolded velocity plane (float64, raw where unresolved) and the
    /// unfolder's resolved mask, when dealiasing ran on this sweep.
    #[serde(default)]
    pub dealiased: Option<usize>,
    #[serde(default)]
    pub resolved: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SweepIn {
    pub sweep_index: i64,
    #[serde(deserialize_with = "de_f64")]
    pub elevation_angle_deg: f64,
    /// The pack's scalar Nyquist, or null when the sweep reported none.
    #[serde(deserialize_with = "de_opt_f64")]
    pub nyquist_velocity_ms: Option<f64>,
    pub nyquist_radials_disagree: bool,
    pub azimuth: usize,
    pub elevation: usize,
    pub moments: Vec<MomentIn>,
}

#[derive(Debug, Deserialize)]
pub struct VolumeOutputsIn {
    pub z_linear_sum: usize,
    pub z_count: usize,
    pub z0_count: usize,
    pub z_max_dbz: usize,
    pub z_sumsq_dbz: usize,
    pub z_sum_dbz: usize,
    pub vr_sum: usize,
    pub vr_sumsq: usize,
    pub vr_count: usize,
    pub vr_min: usize,
    pub vr_max: usize,
    pub beam_east: usize,
    pub beam_north: usize,
    pub beam_up: usize,
    pub nyquist_min: usize,
    pub vr_rejected: usize,
}

impl VolumeOutputsIn {
    fn indices(&self) -> Vec<usize> {
        vec![
            self.z_linear_sum, self.z_count, self.z0_count, self.z_max_dbz,
            self.z_sumsq_dbz, self.z_sum_dbz, self.vr_sum, self.vr_sumsq,
            self.vr_count, self.vr_min, self.vr_max, self.beam_east,
            self.beam_north, self.beam_up, self.nyquist_min, self.vr_rejected,
        ]
    }
}

#[derive(Debug, Deserialize)]
pub struct VolumeRequest {
    pub params: Params,
    #[serde(default)]
    pub dealias: Option<DealiasUse>,
    #[serde(default)]
    pub cc_qc: Option<CcParams>,
    pub clear_air_from_censor: bool,
    /// The pack can mint the two ODIM-only censor states, so their
    /// counters are armed at zero.
    pub odim_census: bool,
    pub site: SiteIn,
    pub grid: GridIn,
    /// Inclusive `(j0, j1, i0, i1)` this radar's accumulators cover.
    pub window: [i64; 4],
    pub sweeps: Vec<SweepIn>,
    pub outputs: VolumeOutputsIn,
}

/// `SuperobCounts`, minus the census.
#[derive(Debug, Default, Serialize)]
pub struct Counts {
    pub gates_considered: i64,
    pub gates_out_of_grid: i64,
    pub gates_out_of_column: i64,
    pub gates_below_floor: i64,
    pub gates_nonfinite: i64,
    pub gates_beyond_range: i64,
    pub sweeps_used: i64,
    pub sweeps_skipped_elevation: i64,
    pub velocity_gates_rejected_nyquist: i64,
    pub velocity_gates_rejected_no_nyquist: i64,
    pub velocity_cells_rejected_spread: i64,
    pub sweeps_without_nyquist: i64,
    pub sweeps_with_implausible_nyquist: i64,
    pub sweeps_with_nyquist_disagreement: i64,
    pub velocity_gate_pairs_tested: i64,
    pub velocity_fold_boundaries: i64,
    pub velocity_radials_fold_suspect: i64,
    pub velocity_sweeps_fold_suspect: i64,
    pub velocity_gates_rejected_shear: i64,
    pub cc_sweeps_masked: i64,
    pub cc_sweeps_paired_companion: i64,
    pub cc_sweeps_without_rho: i64,
    pub cc_sweeps_without_ref: i64,
    pub cc_gates_tested: i64,
    pub cc_gates_rho_missing: i64,
    pub cc_velocity_gates_rejected: i64,
    pub cc_reflectivity_gates_rejected: i64,
    pub cc_velocity_gates_rejected_shielded_z: i64,
    pub cc_velocity_gates_exempt_tds_fringe: i64,
    pub cc_couplet_seed_gates: i64,
    pub cc_velocity_tds_rho_below_floor: i64,
    pub cc_velocity_tds_below_reflectivity: i64,
    pub cc_velocity_tds_at_or_above_shield: i64,
    pub cc_velocity_tds_no_couplet_nearby: i64,
}

/// `CensorCounts`.
#[derive(Debug, Default, Serialize)]
pub struct Census {
    pub reflectivity_measured: i64,
    pub reflectivity_below_threshold: i64,
    pub reflectivity_range_folded: i64,
    pub reflectivity_not_collected: i64,
    pub clear_air_gates_admitted: i64,
    pub range_folded_gates_refused: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reflectivity_nodata: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reflectivity_sentinel_ambiguous: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentinel_ambiguous_gates_refused: Option<i64>,
}

/// The accumulators, borrowed from the caller's window-shaped buffers.
struct Accumulators<'a> {
    z_linear_sum: &'a mut [f64],
    z_count: &'a mut [i64],
    z0_count: &'a mut [i64],
    z_max_dbz: &'a mut [f64],
    z_sumsq_dbz: &'a mut [f64],
    z_sum_dbz: &'a mut [f64],
    vr_sum: &'a mut [f64],
    vr_sumsq: &'a mut [f64],
    vr_count: &'a mut [i64],
    vr_min: &'a mut [f64],
    vr_max: &'a mut [f64],
    beam_east: &'a mut [f64],
    beam_north: &'a mut [f64],
    beam_up: &'a mut [f64],
    nyquist_min: &'a mut [f64],
    vr_rejected: &'a mut [i64],
}

/// `np.maximum` / `np.minimum` on two floats (the scalar loop's rule).
#[inline]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() { a } else { b }
}

#[inline]
fn np_minimum(a: f64, b: f64) -> f64 {
    if a <= b || a.is_nan() { a } else { b }
}

/// Where one gate went.
#[derive(Clone, Copy)]
enum Place {
    Unusable,
    OffGrid,
    OutOfColumn,
    Outside { j: i64, i: i64 },
    Cell { flat: usize, east: f64, north: f64, up: f64 },
}

/// `_believable_nyquist`.
fn believable(reported: Option<f64>, params: &Params) -> Option<f64> {
    let value = reported?;
    if !value.is_finite() || !(params.nyquist_min_ms <= value && value <= params.nyquist_max_ms) {
        return None;
    }
    Some(value)
}

struct MomentView<'a> {
    product: &'a str,
    gate_count: usize,
    first_gate_range_m: f64,
    gate_size_m: f64,
    data: &'a [f64],
    censor: Option<&'a [u8]>,
    dealiased: Option<&'a [f64]>,
    resolved: Option<&'a [u8]>,
}

struct SweepView<'a> {
    input: &'a SweepIn,
    azimuth: &'a [f64],
    elevation: &'a [f64],
    moments: Vec<MomentView<'a>>,
}

impl<'a> SweepView<'a> {
    fn plane(&self, product: &str) -> Option<PlaneRef<'a>> {
        self.moments.iter().find(|m| m.product == product).map(|m| PlaneRef {
            data: m.data,
            rows: self.azimuth.len(),
            gate_count: m.gate_count,
            first_gate_range_m: m.first_gate_range_m,
            gate_size_m: m.gate_size_m,
        })
    }
}

fn load_sweeps<'a>(request: &'a VolumeRequest, arrays: &Arrays<'a>) -> Result<Vec<SweepView<'a>>> {
    let mut out = Vec::with_capacity(request.sweeps.len());
    for sweep in &request.sweeps {
        let azimuth = arrays.f64s(sweep.azimuth, "azimuth")?;
        let elevation = arrays.f64s(sweep.elevation, "elevation")?;
        let radials = azimuth.len();
        if elevation.len() != radials {
            return Err(SuperobFailure::request(format!(
                "sweep {}: {} elevations for {} radials",
                sweep.sweep_index,
                elevation.len(),
                radials
            )));
        }
        let mut moments = Vec::with_capacity(sweep.moments.len());
        for moment in &sweep.moments {
            if moment.columns != moment.gate_count {
                return Err(SuperobFailure::request(format!(
                    "sweep {} {}: plane has {} columns for gate_count {}",
                    sweep.sweep_index, moment.product, moment.columns, moment.gate_count
                )));
            }
            let cells = radials * moment.gate_count;
            let data = arrays.f64s(moment.data, "moment data")?;
            if data.len() != cells {
                return Err(SuperobFailure::request(format!(
                    "sweep {} {}: {} values for {} radials x {} gates",
                    sweep.sweep_index, moment.product, data.len(), radials, moment.gate_count
                )));
            }
            let censor = match moment.censor {
                Some(index) => Some(arrays.u8s(index, "censor plane")?),
                None => None,
            };
            let dealiased = match moment.dealiased {
                Some(index) => Some(arrays.f64s(index, "dealiased velocity")?),
                None => None,
            };
            let resolved = match moment.resolved {
                Some(index) => Some(arrays.u8s(index, "resolved mask")?),
                None => None,
            };
            for (name, len) in [
                ("censor plane", censor.map(|c| c.len())),
                ("dealiased velocity", dealiased.map(|d| d.len())),
                ("resolved mask", resolved.map(|r| r.len())),
            ] {
                if let Some(len) = len {
                    if len != cells {
                        return Err(SuperobFailure::request(format!(
                            "sweep {} {}: {name} holds {len} values for {cells} gates",
                            sweep.sweep_index, moment.product
                        )));
                    }
                }
            }
            if dealiased.is_some() != resolved.is_some() {
                return Err(SuperobFailure::request(
                    "a dealiased plane travels with its resolved mask",
                ));
            }
            moments.push(MomentView {
                product: &moment.product,
                gate_count: moment.gate_count,
                first_gate_range_m: moment.first_gate_range_m,
                gate_size_m: moment.gate_size_m,
                data,
                censor,
                dealiased,
                resolved,
            });
        }
        out.push(SweepView { input: sweep, azimuth, elevation, moments });
    }
    Ok(out)
}

/// Run one volume.  Returns the JSON the seam hands back.
pub fn superob_volume(request: &VolumeRequest, arrays: &Arrays) -> Result<Value> {
    let params = &request.params;
    params.validate()?;
    if let Some(cc) = &request.cc_qc {
        cc.validate()?;
    }
    let grid_in = &request.grid;
    let z_w = arrays.f64s(grid_in.z_w, "z_w")?;
    let grid = GridView::new(
        grid_in.spec.to_spec()?,
        grid_in.translations.clone(),
        grid_in.nx,
        grid_in.ny,
        grid_in.nz,
        z_w,
    )?;
    let [j0, j1, i0, i1] = request.window;
    if j0 < 0 || i0 < 0 || j1 < j0 || i1 < i0 {
        return Err(SuperobFailure::request(format!(
            "window j[{j0}..{j1}] i[{i0}..{i1}] is not a box"
        )));
    }
    let nj = (j1 - j0 + 1) as usize;
    let ni = (i1 - i0 + 1) as usize;
    let nz = grid.nz;
    let cells = nz * nj * ni;

    let outputs = &request.outputs;
    arrays.check_disjoint(&outputs.indices())?;
    let mut acc = Accumulators {
        z_linear_sum: arrays.f64s_mut(outputs.z_linear_sum, "z_linear_sum")?,
        z_count: arrays.i64s_mut(outputs.z_count, "z_count")?,
        z0_count: arrays.i64s_mut(outputs.z0_count, "z0_count")?,
        z_max_dbz: arrays.f64s_mut(outputs.z_max_dbz, "z_max_dbz")?,
        z_sumsq_dbz: arrays.f64s_mut(outputs.z_sumsq_dbz, "z_sumsq_dbz")?,
        z_sum_dbz: arrays.f64s_mut(outputs.z_sum_dbz, "z_sum_dbz")?,
        vr_sum: arrays.f64s_mut(outputs.vr_sum, "vr_sum")?,
        vr_sumsq: arrays.f64s_mut(outputs.vr_sumsq, "vr_sumsq")?,
        vr_count: arrays.i64s_mut(outputs.vr_count, "vr_count")?,
        vr_min: arrays.f64s_mut(outputs.vr_min, "vr_min")?,
        vr_max: arrays.f64s_mut(outputs.vr_max, "vr_max")?,
        beam_east: arrays.f64s_mut(outputs.beam_east, "beam_east")?,
        beam_north: arrays.f64s_mut(outputs.beam_north, "beam_north")?,
        beam_up: arrays.f64s_mut(outputs.beam_up, "beam_up")?,
        nyquist_min: arrays.f64s_mut(outputs.nyquist_min, "nyquist_min")?,
        vr_rejected: arrays.i64s_mut(outputs.vr_rejected, "vr_rejected")?,
    };
    {
        let lens = [
            acc.z_linear_sum.len(), acc.z_count.len(), acc.z0_count.len(),
            acc.z_max_dbz.len(), acc.z_sumsq_dbz.len(), acc.z_sum_dbz.len(),
            acc.vr_sum.len(), acc.vr_sumsq.len(), acc.vr_count.len(),
            acc.vr_min.len(), acc.vr_max.len(), acc.beam_east.len(),
            acc.beam_north.len(), acc.beam_up.len(), acc.nyquist_min.len(),
            acc.vr_rejected.len(),
        ];
        if lens.iter().any(|&len| len != cells) {
            return Err(SuperobFailure::request(format!(
                "accumulators must each hold {cells} cells ({nz} x {nj} x {ni})"
            )));
        }
    }
    // The identities of every reduction.
    for slice in [
        &mut *acc.z_linear_sum, &mut *acc.z_sumsq_dbz, &mut *acc.z_sum_dbz,
        &mut *acc.vr_sum, &mut *acc.vr_sumsq, &mut *acc.beam_east,
        &mut *acc.beam_north, &mut *acc.beam_up,
    ] {
        slice.fill(0.0);
    }
    for slice in [&mut *acc.z_count, &mut *acc.z0_count, &mut *acc.vr_count, &mut *acc.vr_rejected] {
        slice.fill(0);
    }
    acc.z_max_dbz.fill(f64::NEG_INFINITY);
    acc.vr_max.fill(f64::NEG_INFINITY);
    acc.vr_min.fill(f64::INFINITY);
    acc.nyquist_min.fill(f64::INFINITY);

    let sweeps = load_sweeps(request, arrays)?;
    let site = Site {
        lat_deg: request.site.lat_deg,
        lon_deg: request.site.lon_deg,
        alt_m: request.site.alt_m,
    };
    let propagation = Propagation::new(site, params.earth_radius_m, params.refraction_factor);
    let mut counts = Counts::default();
    let mut census = if request.clear_air_from_censor {
        let mut census = Census::default();
        if request.odim_census {
            census.reflectivity_nodata = Some(0);
            census.reflectivity_sentinel_ambiguous = Some(0);
            census.sentinel_ambiguous_gates_refused = Some(0);
        }
        Some(census)
    } else {
        None
    };

    // CC QC plans span the volume: a sweep skipped for elevation can still
    // lend its RHO plane to the cut beside it.
    let mut plans: Option<Vec<Plan>> = request.cc_qc.as_ref().map(|cc| {
        let inputs: Vec<SweepPlanInput> = sweeps
            .iter()
            .map(|s| SweepPlanInput {
                elevation_angle_deg: s.input.elevation_angle_deg,
                sweep_index: s.input.sweep_index,
                azimuth: s.azimuth,
                reflectivity: s.plane(REFLECTIVITY),
                velocity: s.plane(VELOCITY),
                correlation: s.plane(CORRELATION),
            })
            .collect();
        ccqc::build_plans(&inputs, cc)
    });

    let max_range_m = params.max_range_km * 1000.0;
    let mut skipped_products: std::collections::BTreeSet<String> = Default::default();
    let mut sweep_records: Vec<Value> = Vec::new();
    let mut gates_refused_at_grid: i64 = 0;
    let window = Window { j0, i0, nj, ni };

    for (position, sweep) in sweeps.iter().enumerate() {
        if sweep.input.elevation_angle_deg > params.max_elevation_deg {
            counts.sweeps_skipped_elevation += 1;
            continue;
        }
        let nyquist = believable(sweep.input.nyquist_velocity_ms, params);
        if sweep.input.nyquist_velocity_ms.is_none() {
            counts.sweeps_without_nyquist += 1;
        } else if nyquist.is_none() {
            counts.sweeps_with_implausible_nyquist += 1;
        }
        if sweep.input.nyquist_radials_disagree {
            counts.sweeps_with_nyquist_disagreement += 1;
        }
        counts.sweeps_used += 1;
        let mut fold = FoldStats::default();
        let plan = plans.as_mut().map(|p| &mut p[position]);
        let mut plan = plan;
        if let Some(plan) = plan.as_deref() {
            if plan.masker.is_some() {
                counts.cc_sweeps_masked += 1;
                if plan.reason == ccqc::REASON_COMPANION {
                    counts.cc_sweeps_paired_companion += 1;
                }
            } else if plan.reason == ccqc::REASON_NO_RHO {
                counts.cc_sweeps_without_rho += 1;
            } else if plan.reason == ccqc::REASON_NO_REF {
                counts.cc_sweeps_without_ref += 1;
            }
        }

        for moment in &sweep.moments {
            let is_ref = moment.product == REFLECTIVITY;
            let is_vel = moment.product == VELOCITY;
            if !is_ref && !is_vel {
                skipped_products.insert(moment.product.to_string());
                continue;
            }
            let masker = plan.as_deref_mut().and_then(|p| p.masker.as_mut());
            process_moment(
                &MomentJob {
                    sweep,
                    moment,
                    is_vel,
                    nyquist,
                    params,
                    dealias: request.dealias.as_ref(),
                    grid: &grid,
                    propagation: &propagation,
                    window,
                    max_range_m,
                    site_id: &request.site.id,
                },
                masker,
                &mut acc,
                &mut counts,
                census.as_mut(),
                &mut fold,
                &mut gates_refused_at_grid,
            )?;
        }

        let has_velocity = sweep.moments.iter().any(|m| m.product == VELOCITY);
        if has_velocity {
            counts.velocity_gate_pairs_tested += fold.pairs;
            counts.velocity_fold_boundaries += fold.boundaries;
            counts.velocity_radials_fold_suspect += fold.suspect_radials;
            if fold.boundaries != 0 {
                counts.velocity_sweeps_fold_suspect += 1;
            }
        }
        let mut record = Map::new();
        record.insert("position".into(), json!(position));
        record.insert(
            "fold".into(),
            if has_velocity {
                json!({
                    "gate_pairs_tested": fold.pairs,
                    "fold_boundaries": fold.boundaries,
                    "radials_fold_suspect": fold.suspect_radials,
                    "gates_rejected_shear": fold.shear_rejected,
                })
            } else {
                Value::Null
            },
        );
        if let Some(plan) = plan {
            let mut cc = Map::new();
            cc.insert("applied".into(), json!(plan.masker.is_some()));
            cc.insert("reason".into(), json!(plan.reason));
            if let Some(masker) = &plan.masker {
                counts.cc_gates_tested += masker.gates_tested.values().sum::<i64>();
                counts.cc_gates_rho_missing += masker.gates_rho_missing.values().sum::<i64>();
                counts.cc_velocity_gates_rejected_shielded_z +=
                    masker.gates_dropped_shielded_z.get(VELOCITY).copied().unwrap_or(0);
                counts.cc_couplet_seed_gates += masker.couplet_seed_gates;
                if let Some(turned) = masker.gates_tds_turned_away.get(VELOCITY) {
                    let get = |k: &str| turned.get(k).copied().unwrap_or(0);
                    counts.cc_velocity_tds_rho_below_floor += get(ccqc::TDS_RHO_BELOW_FLOOR);
                    counts.cc_velocity_tds_below_reflectivity += get(ccqc::TDS_BELOW_Z_BAND);
                    counts.cc_velocity_tds_at_or_above_shield += get(ccqc::TDS_ABOVE_Z_BAND);
                    counts.cc_velocity_tds_no_couplet_nearby += get(ccqc::TDS_NO_ROTATION);
                }
                cc.insert("rho_companion_sweep_index".into(), json!(masker.companion_sweep_index));
                cc.insert("gates_tested".into(), json!(masker.gates_tested));
                cc.insert("gates_rho_missing".into(), json!(masker.gates_rho_missing));
                cc.insert("gates_dropped".into(), json!(masker.gates_dropped));
                cc.insert("gates_dropped_reason".into(), json!(masker.gates_dropped_reason));
                cc.insert("gates_dropped_shielded_z".into(), json!(masker.gates_dropped_shielded_z));
                cc.insert("gates_exempt_tds_fringe".into(), json!(masker.gates_exempt_tds_fringe));
                cc.insert("gates_tds_turned_away".into(), json!(masker.gates_tds_turned_away));
                cc.insert("couplet_seed_gates".into(), json!(masker.couplet_seed_gates));
            }
            record.insert("cc".into(), Value::Object(cc));
        }
        sweep_records.push(Value::Object(record));
    }

    // A fold caught inside one cell: drop it whole.
    let mut folded_cells = 0i64;
    for cell in 0..cells {
        let count = acc.vr_count[cell];
        let spread = if count > 0 { acc.vr_max[cell] - acc.vr_min[cell] } else { 0.0 };
        let nyq = acc.nyquist_min[cell];
        if count > 1 && nyq.is_finite() && spread > params.nyquist_spread_fraction * nyq {
            folded_cells += 1;
            acc.vr_rejected[cell] += count;
            acc.vr_sum[cell] = 0.0;
            acc.vr_sumsq[cell] = 0.0;
            acc.beam_east[cell] = 0.0;
            acc.beam_north[cell] = 0.0;
            acc.beam_up[cell] = 0.0;
            acc.vr_count[cell] = 0;
        }
    }
    counts.velocity_cells_rejected_spread = folded_cells;

    Ok(json!({
        "ok": true,
        "counts": counts,
        "censor": census,
        "sweeps": sweep_records,
        "skipped_products": skipped_products,
        "gates_refused_at_grid": gates_refused_at_grid,
    }))
}

#[derive(Clone, Copy)]
struct Window {
    j0: i64,
    i0: i64,
    nj: usize,
    ni: usize,
}

#[derive(Default)]
struct FoldStats {
    pairs: i64,
    boundaries: i64,
    suspect_radials: i64,
    shear_rejected: i64,
}

struct MomentJob<'a, 'b> {
    sweep: &'b SweepView<'a>,
    moment: &'b MomentView<'a>,
    is_vel: bool,
    nyquist: Option<f64>,
    params: &'b Params,
    dealias: Option<&'b DealiasUse>,
    grid: &'b GridView<'b>,
    propagation: &'b Propagation,
    window: Window,
    max_range_m: f64,
    site_id: &'b str,
}

#[allow(clippy::too_many_arguments)]
fn process_moment(
    job: &MomentJob,
    masker: Option<&mut Masker>,
    acc: &mut Accumulators,
    counts: &mut Counts,
    mut census: Option<&mut Census>,
    fold: &mut FoldStats,
    gates_refused_at_grid: &mut i64,
) -> Result<()> {
    let moment = job.moment;
    let sweep = job.sweep;
    let params = job.params;
    let radials = sweep.azimuth.len();
    let gate_count = moment.gate_count;

    // slant_range_m: first + size * arange(gate_count)
    let columns: Vec<usize> = (0..gate_count)
        .filter(|&g| moment.first_gate_range_m + moment.gate_size_m * (g as f64) <= job.max_range_m)
        .collect();
    counts.gates_beyond_range += ((gate_count - columns.len()) * radials) as i64;
    if columns.is_empty() {
        return Ok(());
    }
    let ranges: Vec<f64> = columns
        .iter()
        .map(|&g| moment.first_gate_range_m + moment.gate_size_m * (g as f64))
        .collect();
    let n = columns.len();
    let total = radials * n;

    let dealiased = if job.is_vel { moment.dealiased.zip(moment.resolved) } else { None };
    let mut values = vec![0.0f64; total];
    let mut resolved: Option<Vec<bool>> = dealiased.map(|_| vec![false; total]);
    for r in 0..radials {
        for (k, &g) in columns.iter().enumerate() {
            let source = r * gate_count + g;
            values[r * n + k] = match dealiased {
                Some((plane, _)) => plane[source],
                None => moment.data[source],
            };
            if let (Some((_, mask)), Some(out)) = (dealiased, resolved.as_mut()) {
                out[r * n + k] = mask[source] != 0;
            }
        }
    }

    // --- the decoder's reason for each NaN ---
    let mut clear_flag = vec![false; total];
    if let (Some(codes), Some(census)) = (moment.censor, census.as_deref_mut()) {
        for r in 0..radials {
            for &g in &columns {
                let code = codes[r * gate_count + g];
                if code == RANGE_FOLDED {
                    census.range_folded_gates_refused += 1;
                }
                if let Some(refused) = census.sentinel_ambiguous_gates_refused.as_mut() {
                    if code == SENTINEL_AMBIGUOUS {
                        *refused += 1;
                    }
                }
                if !job.is_vel {
                    match code {
                        MEASURED => census.reflectivity_measured += 1,
                        BELOW_THRESHOLD => census.reflectivity_below_threshold += 1,
                        RANGE_FOLDED => census.reflectivity_range_folded += 1,
                        NOT_COLLECTED => census.reflectivity_not_collected += 1,
                        _ => {}
                    }
                    if let Some(nodata) = census.reflectivity_nodata.as_mut() {
                        if code == NODATA {
                            *nodata += 1;
                        }
                    }
                    if let Some(ambiguous) = census.reflectivity_sentinel_ambiguous.as_mut() {
                        if code == SENTINEL_AMBIGUOUS {
                            *ambiguous += 1;
                        }
                    }
                }
            }
        }
        if !job.is_vel {
            // Equality with ONE code, never "not an echo": range-folded
            // (2), no-data (4) and sentinel-ambiguous (5) are never clear.
            for r in 0..radials {
                for (k, &g) in columns.iter().enumerate() {
                    clear_flag[r * n + k] = codes[r * gate_count + g] == BELOW_THRESHOLD;
                }
            }
        }
    }
    // --- CC QC, before the shear scan sees the plane ---
    if let Some(masker) = masker {
        let drop = masker.drop_mask(moment.product, job.is_vel, &ranges);
        let finite_before: Vec<bool> = values.iter().map(|v| v.is_finite()).collect();
        let exempt = masker.count_block(moment.product, &finite_before, radials > 0);
        if job.is_vel {
            counts.cc_velocity_gates_exempt_tds_fringe += exempt;
        }
        let hits: Vec<bool> = (0..total).map(|k| drop[k] && finite_before[k]).collect();
        if hits.iter().any(|&h| h) {
            for k in 0..total {
                if hits[k] {
                    values[k] = f64::NAN;
                }
            }
            let dropped = masker.count_dropped(moment.product, &hits);
            if job.is_vel {
                counts.cc_velocity_gates_rejected += dropped;
            } else {
                counts.cc_reflectivity_gates_rejected += dropped;
            }
        }
    }

    // --- gate-to-gate shear scan on the (unfolded, CC-masked) plane ---
    let mut fold_flags = vec![false; total];
    if job.is_vel {
        if let Some(nyq) = job.nyquist {
            let limit = params.shear_fold_fraction * 2.0 * nyq;
            for r in 0..radials {
                let row = &values[r * n..(r + 1) * n];
                let flags = &mut fold_flags[r * n..(r + 1) * n];
                let mut any = false;
                for k in 0..n.saturating_sub(1) {
                    let delta = (row[k + 1] - row[k]).abs();
                    if delta.is_finite() {
                        fold.pairs += 1;
                        if delta > limit {
                            fold.boundaries += 1;
                            flags[k] = true;
                            flags[k + 1] = true;
                            any = true;
                        }
                    }
                }
                if any {
                    fold.suspect_radials += 1;
                }
            }
        }
    }

    // --- geometry and placement, parallel by radial (order-free) ---
    let grid = job.grid;
    let window = job.window;
    let places: Vec<Place> = (0..radials)
        .into_par_iter()
        .flat_map_iter(|r| {
            let radial = job
                .propagation
                .radial(sweep.azimuth[r], sweep.elevation[r]);
            let values = &values;
            let clear_flag = &clear_flag;
            let ranges = &ranges;
            (0..n).map(move |k| {
                let v = values[r * n + k];
                if !(v.is_finite() || clear_flag[r * n + k]) {
                    return Place::Unusable;
                }
                let gate = job.propagation.gate(&radial, ranges[k]);
                let (fi, fj) = grid.mass_index(gate.lat, gate.lon);
                let i = crate::num::rint_index(fi);
                let j = crate::num::rint_index(fj);
                if !grid.inside(i, j) {
                    return Place::OffGrid;
                }
                let level = grid.level_index(i as usize, j as usize, gate.height);
                if level < 0 {
                    return Place::OutOfColumn;
                }
                let jl = j - window.j0;
                let il = i - window.i0;
                if jl < 0 || il < 0 || jl as usize >= window.nj || il as usize >= window.ni {
                    return Place::Outside { j, i };
                }
                Place::Cell {
                    flat: (level as usize * window.nj + jl as usize) * window.ni + il as usize,
                    east: gate.east,
                    north: gate.north,
                    up: gate.up,
                }
            })
        })
        .collect();

    // --- accumulate, sequentially, in gate order ---
    counts.gates_considered += total as i64;
    counts.gates_nonfinite += values.iter().filter(|v| !v.is_finite()).count() as i64;
    if let Some(Place::Outside { .. }) = places.iter().find(|p| matches!(p, Place::Outside { .. })) {
        let (mut jmin, mut jmax, mut imin, mut imax) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
        for place in &places {
            if let Place::Outside { j, i } = place {
                jmin = jmin.min(*j);
                jmax = jmax.max(*j);
                imin = imin.min(*i);
                imax = imax.max(*i);
            }
        }
        let j1 = window.j0 + window.nj as i64 - 1;
        let i1 = window.i0 + window.ni as i64 - 1;
        return Err(SuperobFailure::window(format!(
            "{}: gates fell outside the computed reach window j[{}..{}] i[{}..{}] \
             (saw j[{jmin}..{jmax}] i[{imin}..{imax}] beyond it) -- the window \
             under-covers this radar's gates and observations would have been lost.  \
             This is a bug in horizontal_window(), not a data problem.",
            job.site_id, window.j0, j1, window.i0, i1
        )));
    }
    let mut clear_admitted = 0i64;
    for (index, place) in places.iter().enumerate() {
        match *place {
            Place::Unusable | Place::Outside { .. } => {}
            Place::OffGrid => counts.gates_out_of_grid += 1,
            Place::OutOfColumn => counts.gates_out_of_column += 1,
            Place::Cell { flat, east, north, up } => {
                let value = values[index];
                if !job.is_vel {
                    let clear = clear_flag[index];
                    let echo = value >= params.min_reflectivity_dbz;
                    if !echo && !clear {
                        counts.gates_below_floor += 1;
                    }
                    if clear {
                        clear_admitted += 1;
                    }
                    if !echo {
                        acc.z0_count[flat] += 1;
                    } else {
                        acc.z_linear_sum[flat] += 10.0f64.powf(value / 10.0);
                        acc.z_sum_dbz[flat] += value;
                        acc.z_sumsq_dbz[flat] += value * value;
                        acc.z_count[flat] += 1;
                        acc.z_max_dbz[flat] = np_maximum(acc.z_max_dbz[flat], value);
                    }
                    continue;
                }
                let Some(nyq) = job.nyquist else {
                    counts.velocity_gates_rejected_no_nyquist += 1;
                    acc.vr_rejected[flat] += 1;
                    continue;
                };
                let keep_resolved = match &resolved {
                    Some(mask) => {
                        if !mask[index] {
                            *gates_refused_at_grid += 1;
                        }
                        mask[index]
                    }
                    None => true,
                };
                let within_bound = match (job.dealias, &resolved) {
                    (Some(d), Some(_)) if d.keep_beyond_reject_fraction => value.abs() <= d.max_speed_ms,
                    _ => value.abs() <= params.nyquist_reject_fraction * nyq,
                };
                let within = within_bound && keep_resolved;
                if keep_resolved && !within {
                    counts.velocity_gates_rejected_nyquist += 1;
                }
                let flanking = fold_flags[index];
                if within && flanking {
                    counts.velocity_gates_rejected_shear += 1;
                    fold.shear_rejected += 1;
                }
                let keep = within && !flanking;
                if !keep {
                    acc.vr_rejected[flat] += 1;
                    continue;
                }
                acc.vr_sum[flat] += value;
                acc.vr_sumsq[flat] += value * value;
                acc.vr_count[flat] += 1;
                acc.vr_min[flat] = np_minimum(acc.vr_min[flat], value);
                acc.vr_max[flat] = np_maximum(acc.vr_max[flat], value);
                acc.beam_east[flat] += east;
                acc.beam_north[flat] += north;
                acc.beam_up[flat] += up;
                acc.nyquist_min[flat] = np_minimum(acc.nyquist_min[flat], nyq);
            }
        }
    }
    if let Some(census) = census {
        // Only reflectivity ever carries a clear flag.
        census.clear_air_gates_admitted += clear_admitted;
    }
    Ok(())
}
