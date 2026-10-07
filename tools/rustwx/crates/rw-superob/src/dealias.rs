//! The region-global dealiaser for one sweep, driven natively.
//!
//! Transcribed from `gpuwm/obs/dealias_region.py::dealias_sweep_region`
//! and `_dealias_by_sector` / `_reconcile_sectors`.  The solver itself is
//! the vendored `region-global-dealias` crate, linked here as a Rust
//! library and called through the very C entry points (`bw_dealias`,
//! `bw_dealias_rift_v1`) the Python seam calls through ctypes, so the two
//! routes run one solver.
//!
//! What moved into Rust is the part that used to be Python data-path work:
//! the per-sweep bookkeeping, the physical speed bound, and the mixed-
//! Nyquist path -- one solve per constant-Nyquist sector with every other
//! radial masked, then the sectors reconciled by whole multiples of their
//! own interval.  A mixed-Nyquist sweep is therefore never refused here,
//! and the reconciliation runs in a fixed order (sectors ascending, pairs
//! in azimuth order, candidate shifts -2..2 in that order).

use std::collections::BTreeMap;

use region_global_dealias::{
    BW_RIFT_API_VERSION, BwRiftOptionsV1, BwRiftStatsV1, BwStats, bw_dealias, bw_dealias_rift_v1,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::request::{Arrays, de_f64, de_opt_f64};
use crate::{Result, SuperobFailure};

pub const STATE_REJECTED: i8 = 0;
pub const STATE_UNCHANGED: i8 = 1;
pub const STATE_UNFOLDED: i8 = 2;
pub const REASON_NONE: i8 = 0;
pub const REASON_NONFINITE: i8 = 1;
pub const REASON_NO_NYQUIST: i8 = 2;
pub const REASON_SPEED: i8 = 6;

/// `REASON_NAMES` minus none/nonfinite/disabled, the keys of `rejected`.
const REJECTED_NAMES: [&str; 6] = [
    "no_nyquist",
    "unresolved",
    "conflict",
    "fold_out_of_range",
    "speed_out_of_range",
    "reference_departure",
];

const RIFT_REASON_NAMES: [(u32, &str); 13] = [
    (1 << 0, "residue_trigger"),
    (1 << 1, "branch_unstable"),
    (1 << 2, "temporal_anchor"),
    (1 << 3, "vertical_anchor"),
    (1 << 4, "environmental_anchor"),
    (1 << 5, "caller_anchor"),
    (1 << 6, "vortex_proposal"),
    (1 << 7, "fusion_accepted"),
    (1 << 8, "conflicting_references"),
    (1 << 9, "low_coverage"),
    (1 << 10, "abstained"),
    (1 << 11, "budget_exceeded"),
    (1 << 12, "nyquist_transition"),
];

#[derive(Debug, Deserialize)]
pub struct DealiasRequest {
    pub velocity: usize,
    pub rows: usize,
    pub gates: usize,
    /// float64 azimuths, exactly as the caller holds them.
    pub azimuth: usize,
    #[serde(deserialize_with = "de_opt_f64")]
    pub nyquist: Option<f64>,
    #[serde(default)]
    pub nyquist_by_radial: Option<usize>,
    pub nyquist_radials_disagree: bool,
    #[serde(deserialize_with = "de_f64")]
    pub max_speed_ms: f64,
    pub refinement: bool,
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub first_gate_m: Option<f64>,
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub gate_spacing_m: Option<f64>,
    pub outputs: DealiasOutputs,
}

#[derive(Debug, Deserialize)]
pub struct DealiasOutputs {
    pub velocity: usize,
    pub state: usize,
    pub reason: usize,
    pub fold: usize,
}

/// The four planes one solve produces.
struct Planes {
    output: Vec<f64>,
    state: Vec<i8>,
    reason: Vec<i8>,
    fold: Vec<i16>,
}

impl Planes {
    fn new(total: usize) -> Self {
        Self {
            output: vec![f64::NAN; total],
            state: vec![STATE_REJECTED; total],
            reason: vec![REASON_NONFINITE; total],
            fold: vec![0; total],
        }
    }
}

struct Sweep<'a> {
    velocity: &'a [f64],
    azimuth: &'a [f64],
    rows: usize,
    gates: usize,
    max_speed_ms: f64,
    refinement: bool,
    first_gate_m: Option<f64>,
    gate_spacing_m: Option<f64>,
}

/// Run one sweep; writes the planes and returns the stats.
pub fn dealias_region(request: &DealiasRequest, arrays: &Arrays) -> Result<Value> {
    let total = request.rows * request.gates;
    let velocity = arrays.f64s(request.velocity, "velocity")?;
    let azimuth = arrays.f64s(request.azimuth, "azimuth")?;
    if velocity.len() != total || azimuth.len() != request.rows {
        return Err(SuperobFailure::request(
            "velocity must be rows x gates and azimuth one value per radial",
        ));
    }
    let by_radial = match request.nyquist_by_radial {
        Some(index) => Some(arrays.f64s(index, "nyquist_by_radial")?),
        None => None,
    };
    let o = &request.outputs;
    arrays.check_disjoint(&[o.velocity, o.state, o.reason, o.fold])?;
    let out_velocity = arrays.f64s_mut(o.velocity, "velocity out")?;
    let out_state = arrays.i8s_mut(o.state, "state out")?;
    let out_reason = arrays.i8s_mut(o.reason, "reason out")?;
    let out_fold = arrays.i16s_mut(o.fold, "fold out")?;
    if [out_velocity.len(), out_state.len(), out_reason.len(), out_fold.len()]
        .iter()
        .any(|&len| len != total)
    {
        return Err(SuperobFailure::request("every output plane is rows x gates"));
    }
    let sweep = Sweep {
        velocity,
        azimuth,
        rows: request.rows,
        gates: request.gates,
        max_speed_ms: request.max_speed_ms,
        refinement: request.refinement,
        first_gate_m: request.first_gate_m,
        gate_spacing_m: request.gate_spacing_m,
    };
    let (planes, stats) =
        solve(&sweep, request.nyquist, by_radial, request.nyquist_radials_disagree)?;
    out_velocity.copy_from_slice(&planes.output);
    out_state.copy_from_slice(&planes.state);
    out_reason.copy_from_slice(&planes.reason);
    out_fold.copy_from_slice(&planes.fold);
    Ok(json!({"ok": true, "stats": stats}))
}

fn base_stats(finite: usize, nyquist: Option<f64>) -> Map<String, Value> {
    let mut stats = Map::new();
    stats.insert("engine".into(), json!("region-global"));
    stats.insert("gates_finite".into(), json!(finite));
    stats.insert("gates_unchanged".into(), json!(0));
    stats.insert("gates_unfolded".into(), json!(0));
    stats.insert("gates_rejected".into(), json!(0));
    let rejected: Map<String, Value> =
        REJECTED_NAMES.iter().map(|name| (name.to_string(), json!(0))).collect();
    stats.insert("rejected".into(), Value::Object(rejected));
    stats.insert("fold_histogram".into(), json!([]));
    stats.insert(
        "reference".into(),
        json!({"external_supplied": false, "bands": 0, "bands_valid": 0}),
    );
    stats.insert("nyquist_ms".into(), json!(nyquist));
    stats
}

fn set_rejected(stats: &mut Map<String, Value>, name: &str, value: i64) {
    if let Some(Value::Object(rejected)) = stats.get_mut("rejected") {
        rejected.insert(name.into(), json!(value));
    }
}

fn get_rejected(stats: &Map<String, Value>, name: &str) -> i64 {
    stats
        .get("rejected")
        .and_then(|r| r.get(name))
        .and_then(Value::as_i64)
        .unwrap_or(0)
}

/// `np.unique` of the believable per-radial values (sorted ascending).
fn distinct(values: &[f64], believable: &[bool]) -> Vec<f64> {
    let mut out: Vec<f64> = values
        .iter()
        .zip(believable)
        .filter(|(_, b)| **b)
        .map(|(v, _)| *v)
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out.dedup_by(|a, b| a == b);
    out
}

/// A sorted `{fold: count}` histogram, as `[[fold, count], ...]`.
fn histogram(folds: impl Iterator<Item = i16>) -> Value {
    let mut counts: BTreeMap<i16, i64> = BTreeMap::new();
    for fold in folds {
        *counts.entry(fold).or_insert(0) += 1;
    }
    json!(counts.into_iter().map(|(f, c)| json!([f, c])).collect::<Vec<_>>())
}

fn solve(
    sweep: &Sweep,
    nyquist: Option<f64>,
    by_radial: Option<&[f64]>,
    disagree: bool,
) -> Result<(Planes, Map<String, Value>)> {
    let (rows, gates) = (sweep.rows, sweep.gates);
    let finite_all: Vec<bool> = sweep.velocity.iter().map(|v| v.is_finite()).collect();
    let mut stats = base_stats(finite_all.iter().filter(|&&f| f).count(), nyquist);

    // ---- a nonuniform sweep: one solve per constant-Nyquist sector ----
    if let Some(row_nyquist) = by_radial {
        if row_nyquist.len() == rows {
            let believable: Vec<bool> =
                row_nyquist.iter().map(|v| v.is_finite() && *v > 0.0).collect();
            let sectors = distinct(row_nyquist, &believable);
            if sectors.len() > 1 {
                return by_sector(sweep, row_nyquist, &believable, &sectors, stats);
            }
        }
    }

    // _uniform_nyquist_or_refuse
    let (believable_row, distinct_row): (Vec<bool>, Option<Vec<f64>>) = match by_radial {
        None => {
            if disagree {
                return Err(SuperobFailure {
                    kind: "dealias_params",
                    message: "this sweep reports nyquist_radials_disagree=True and carries \
                              no nyquist_velocity_ms_by_radial, so the only Nyquist \
                              available is the sweep summary -- and unfolding a nonuniform \
                              cut in one interval mis-corrects every radial above it by a \
                              whole fold difference, finite and plausible and undetectable \
                              downstream.  Re-decode the volume with a decoder that emits \
                              the per-radial Nyquist array, or run this volume without \
                              dealiasing"
                        .into(),
                });
            }
            (vec![true; rows], None)
        }
        Some(row_nyquist) => {
            if row_nyquist.len() != rows {
                return Err(SuperobFailure::request(format!(
                    "nyquist_by_radial has {} entries, the sweep has {rows} radials",
                    row_nyquist.len()
                )));
            }
            let believable: Vec<bool> =
                row_nyquist.iter().map(|v| v.is_finite() && *v > 0.0).collect();
            let values = distinct(row_nyquist, &believable);
            if disagree {
                // A uniform per-radial array that still reports disagreement:
                // the reference refuses this (it never reaches the sector path
                // with fewer than two distinct values).
                return Err(SuperobFailure {
                    kind: "dealias_params",
                    message: format!(
                        "engine='region-global' decides every fold in ONE Nyquist interval \
                         for the whole sweep and this sweep reports its radials disagree \
                         while carrying {values:?} m/s"
                    ),
                });
            }
            (believable, Some(values))
        }
    };
    stats.insert("nyquist_by_radial".into(), json!(by_radial.is_some()));
    stats.insert(
        "nyquist_radials_no_value".into(),
        json!(believable_row.iter().filter(|b| !**b).count()),
    );
    if let Some(values) = &distinct_row {
        if !values.is_empty() {
            stats.insert("nyquist_distinct".into(), json!(values));
        }
    }

    let mut planes = Planes::new(rows * gates);
    let mut finite = finite_all.clone();
    if !believable_row.iter().all(|&b| b) {
        let mut refused = 0i64;
        for r in 0..rows {
            if believable_row[r] {
                continue;
            }
            for g in 0..gates {
                let k = r * gates + g;
                if finite[k] {
                    planes.reason[k] = REASON_NO_NYQUIST;
                    finite[k] = false;
                    refused += 1;
                }
            }
        }
        set_rejected(&mut stats, "no_nyquist", refused);
        if !finite.iter().any(|&f| f) {
            stats.insert("gates_rejected".into(), json!(refused));
            return Ok((planes, stats));
        }
    }

    let nyquist = match nyquist {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => {
            let offered = finite.iter().filter(|&&f| f).count() as i64;
            for k in 0..finite.len() {
                if finite[k] {
                    planes.reason[k] = REASON_NO_NYQUIST;
                }
            }
            stats.insert("gates_rejected".into(), json!(offered));
            set_rejected(&mut stats, "no_nyquist", offered);
            return Ok((planes, stats));
        }
    };
    if !finite.iter().any(|&f| f) {
        stats.insert(
            "native".into(),
            json!({"gates_total": rows * gates, "gates_finite": 0, "gates_modified": 0,
                   "max_abs_fold": 0, "wraps": null,
                   "skipped": "no finite gate in this sweep"}),
        );
        return Ok((planes, stats));
    }

    let interval = 2.0 * nyquist;
    let velocity32: Vec<f32> = sweep.velocity.iter().map(|&v| v as f32).collect();
    let azimuth32: Vec<f32> = sweep.azimuth.iter().map(|&a| a as f32).collect();
    let rays = vec![nyquist as f32; rows];
    let mut unfolded = vec![0f32; rows * gates];
    let mut native = if sweep.refinement {
        let (Some(first), Some(spacing)) = (sweep.first_gate_m, sweep.gate_spacing_m) else {
            return Err(SuperobFailure::request(
                "the region-global refinement pass needs the sweep's physical gate geometry \
                 (first_gate_m, gate_spacing_m)",
            ));
        };
        rift(&velocity32, &azimuth32, &rays, rows, gates, first, spacing, &mut unfolded)?
    } else {
        let mut bw = BwStats::default();
        let code = unsafe {
            bw_dealias(
                velocity32.as_ptr(),
                azimuth32.as_ptr(),
                rays.as_ptr(),
                rows,
                gates,
                unfolded.as_mut_ptr(),
                &mut bw,
            )
        };
        if code != 0 {
            return Err(SuperobFailure::request(format!("bw_dealias returned {code}")));
        }
        let mut native = Map::new();
        native.insert("gates_total".into(), json!(bw.gates_total));
        native.insert("gates_finite".into(), json!(bw.gates_finite));
        native.insert("gates_modified".into(), json!(bw.gates_modified));
        native.insert("max_abs_fold".into(), json!(bw.max_abs_fold));
        native.insert("wraps".into(), json!(bw.wraps != 0));
        native
    };
    native.insert("refinement".into(), json!(sweep.refinement));
    stats.insert("native".into(), Value::Object(native));

    // Rounding the ratio recovers the integer the engine applied; a drift
    // from a whole number means the library no longer honours its contract.
    let mut worst = 0.0f64;
    for k in 0..rows * gates {
        if !finite[k] {
            continue;
        }
        let ratio = (unfolded[k] as f64 - sweep.velocity[k]) / interval;
        let fold = ratio.round_ties_even();
        worst = worst.max((ratio - fold).abs());
        let fold = fold as i16;
        planes.state[k] = if fold != 0 { STATE_UNFOLDED } else { STATE_UNCHANGED };
        planes.reason[k] = REASON_NONE;
        planes.fold[k] = fold;
        planes.output[k] = unfolded[k] as f64;
    }
    if worst > 1e-3 {
        return Err(SuperobFailure {
            kind: "region_dealias",
            message: format!(
                "the linked region-global solver moved a gate by {worst:.6} of a Nyquist \
                 interval away from a whole number; its contract is whole intervals only"
            ),
        });
    }

    // ---- the physical bound ----
    let mut beyond = vec![false; rows * gates];
    for k in 0..rows * gates {
        let v = planes.output[k];
        if v.is_finite() && v.abs() > sweep.max_speed_ms {
            beyond[k] = true;
            planes.state[k] = STATE_REJECTED;
            planes.reason[k] = REASON_SPEED;
            planes.output[k] = f64::NAN;
            planes.fold[k] = 0;
        }
    }
    stats.insert("max_speed_ms".into(), json!(sweep.max_speed_ms));
    let (mut unchanged, mut unfolded_n, mut rejected) = (0i64, 0i64, 0i64);
    for k in 0..rows * gates {
        if finite[k] {
            match planes.state[k] {
                STATE_UNCHANGED => unchanged += 1,
                STATE_UNFOLDED => unfolded_n += 1,
                _ => rejected += 1,
            }
        }
    }
    let no_nyquist = get_rejected(&stats, "no_nyquist");
    stats.insert("gates_unchanged".into(), json!(unchanged));
    stats.insert("gates_unfolded".into(), json!(unfolded_n));
    stats.insert("gates_rejected".into(), json!(rejected + no_nyquist));
    set_rejected(
        &mut stats,
        "speed_out_of_range",
        beyond.iter().filter(|&&b| b).count() as i64,
    );
    stats.insert(
        "fold_histogram".into(),
        histogram((0..rows * gates).filter(|&k| finite[k] && !beyond[k]).map(|k| planes.fold[k])),
    );
    Ok((planes, stats))
}

#[allow(clippy::too_many_arguments)]
fn rift(
    velocity: &[f32],
    azimuth: &[f32],
    rays: &[f32],
    rows: usize,
    gates: usize,
    first_gate_m: f64,
    gate_spacing_m: f64,
    out: &mut [f32],
) -> Result<Map<String, Value>> {
    if BW_RIFT_API_VERSION != 1 {
        return Err(SuperobFailure::request("linked solver speaks another refinement API"));
    }
    let options = BwRiftOptionsV1 {
        flags: 0,
        max_abs_fold: 4,
        max_rois: 4,
        max_roi_gates: 65_536,
        max_total_roi_gates: 0,
        min_confidence: 160,
        first_gate_m: first_gate_m as f32,
        gate_spacing_m: gate_spacing_m as f32,
        ..BwRiftOptionsV1::default()
    };
    let total = rows * gates;
    let mut folds = vec![0i8; total];
    let mut confidence = vec![0u8; total];
    let mut reasons = vec![0u16; total];
    let mut stats = BwRiftStatsV1::default();
    let code = unsafe {
        bw_dealias_rift_v1(
            velocity.as_ptr(),
            azimuth.as_ptr(),
            rays.as_ptr(),
            rows,
            gates,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &options,
            out.as_mut_ptr(),
            folds.as_mut_ptr(),
            confidence.as_mut_ptr(),
            reasons.as_mut_ptr(),
            &mut stats,
        )
    };
    if code != 0 {
        return Err(SuperobFailure::request(format!("bw_dealias_rift_v1 returned {code}")));
    }
    let names = |mask: u32| -> Vec<&str> {
        RIFT_REASON_NAMES.iter().filter(|(bit, _)| mask & bit != 0).map(|(_, n)| *n).collect()
    };
    let mut native = Map::new();
    native.insert("gates_total".into(), json!(stats.gates_total));
    native.insert("gates_finite".into(), json!(stats.gates_finite));
    native.insert("gates_modified".into(), json!(stats.gates_modified));
    native.insert("max_abs_fold".into(), json!(stats.max_abs_fold));
    native.insert("wraps".into(), json!(stats.wraps != 0));
    native.insert("rois_detected".into(), json!(stats.rois_detected));
    native.insert("rois_solved".into(), json!(stats.rois_solved));
    native.insert("rois_accepted".into(), json!(stats.rois_accepted));
    native.insert("gates_refined".into(), json!(stats.gates_refined));
    native.insert("gates_ambiguous".into(), json!(stats.gates_ambiguous));
    native.insert("budget_aborts".into(), json!(stats.budget_aborts));
    native.insert("reasons".into(), json!(names(stats.reason_flags)));
    native.insert("abstained".into(), json!(names(stats.abstain_flags)));
    native.insert("confidence_max".into(), json!(confidence.iter().copied().max().unwrap_or(0)));
    Ok(native)
}

/// `_dealias_by_sector`.
fn by_sector(
    sweep: &Sweep,
    row_nyquist: &[f64],
    believable: &[bool],
    sectors: &[f64],
    mut stats: Map<String, Value>,
) -> Result<(Planes, Map<String, Value>)> {
    let (rows, gates) = (sweep.rows, sweep.gates);
    let total = rows * gates;
    let finite: Vec<bool> = sweep.velocity.iter().map(|v| v.is_finite()).collect();
    let mut planes = Planes::new(total);
    stats.insert("nyquist_by_radial".into(), json!(true));
    stats.insert("nyquist_distinct".into(), json!(sectors));
    stats.insert(
        "nyquist_radials_no_value".into(),
        json!(believable.iter().filter(|b| !**b).count()),
    );
    stats.insert("nyquist_ms".into(), json!(sectors[0]));
    let mut unknown = vec![false; total];
    let mut unknown_count = 0i64;
    for r in 0..rows {
        if believable[r] {
            continue;
        }
        for g in 0..gates {
            let k = r * gates + g;
            if finite[k] {
                unknown[k] = true;
                planes.reason[k] = REASON_NO_NYQUIST;
                unknown_count += 1;
            }
        }
    }
    set_rejected(&mut stats, "no_nyquist", unknown_count);
    let mut sector_records = Vec::new();
    let mut native_sectors = Vec::new();
    let mut max_speed = Value::Null;
    for &value in sectors {
        let sel: Vec<bool> = (0..rows).map(|r| believable[r] && row_nyquist[r] == value).collect();
        let masked: Vec<f64> = (0..total)
            .map(|k| if sel[k / gates] { sweep.velocity[k] } else { f64::NAN })
            .collect();
        let by_row: Vec<f64> = sel.iter().map(|&s| if s { value } else { f64::NAN }).collect();
        let part_sweep = Sweep { velocity: &masked, ..*sweep };
        let (part, pst) = solve(&part_sweep, Some(value), Some(&by_row), false)?;
        for r in 0..rows {
            if !sel[r] {
                continue;
            }
            for g in 0..gates {
                let k = r * gates + g;
                planes.output[k] = part.output[k];
                planes.state[k] = part.state[k];
                planes.reason[k] = part.reason[k];
                planes.fold[k] = part.fold[k];
            }
        }
        // Every other rejection name the sector reported is folded in; the
        // speed count is recomputed below, after the reconciliation.
        for name in REJECTED_NAMES {
            if name == "no_nyquist" {
                continue;
            }
            let add = get_rejected(&pst, name);
            let have = get_rejected(&stats, name);
            set_rejected(&mut stats, name, have + add);
        }
        sector_records.push(json!({
            "nyquist_ms": value,
            "radials": sel.iter().filter(|&&s| s).count(),
            "gates_unfolded": pst.get("gates_unfolded").cloned().unwrap_or(json!(0)),
        }));
        max_speed = pst.get("max_speed_ms").cloned().unwrap_or(Value::Null);
        native_sectors.push(pst.get("native").cloned().unwrap_or(Value::Null));
    }
    stats.insert("nyquist_sectors".into(), json!(sector_records));
    stats.insert("max_speed_ms".into(), max_speed);
    stats.insert("native_sectors".into(), json!(native_sectors));

    let shifts = reconcile(&planes.output, sweep.azimuth, row_nyquist, believable, sectors, gates);
    for &(value, k) in &shifts {
        if k == 0 {
            continue;
        }
        let step = k as f64 * 2.0 * value;
        for r in 0..rows {
            if !(believable[r] && row_nyquist[r] == value) {
                continue;
            }
            for g in 0..gates {
                let idx = r * gates + g;
                if planes.state[idx] != STATE_REJECTED {
                    planes.output[idx] += step;
                    planes.fold[idx] = planes.fold[idx].wrapping_add(k as i16);
                    planes.state[idx] =
                        if planes.fold[idx] != 0 { STATE_UNFOLDED } else { STATE_UNCHANGED };
                }
            }
        }
    }
    let bound = sweep.max_speed_ms;
    for k in 0..total {
        if planes.state[k] != STATE_REJECTED && planes.output[k].abs() > bound {
            planes.output[k] = f64::NAN;
            planes.fold[k] = 0;
            planes.state[k] = STATE_REJECTED;
            planes.reason[k] = REASON_SPEED;
        }
    }
    stats.insert(
        "sector_shifts".into(),
        json!(shifts.iter().map(|(v, k)| json!([v, k])).collect::<Vec<_>>()),
    );
    let (mut unchanged, mut unfolded, mut rejected) = (0i64, 0i64, 0i64);
    let mut speed = 0i64;
    for k in 0..total {
        if finite[k] && planes.reason[k] == REASON_SPEED {
            speed += 1;
        }
        if !(finite[k] && believable[k / gates]) {
            continue;
        }
        match planes.state[k] {
            STATE_UNCHANGED => unchanged += 1,
            STATE_UNFOLDED => unfolded += 1,
            _ => rejected += 1,
        }
    }
    set_rejected(&mut stats, "speed_out_of_range", speed);
    stats.insert("gates_unchanged".into(), json!(unchanged));
    stats.insert("gates_unfolded".into(), json!(unfolded));
    stats.insert("gates_rejected".into(), json!(rejected + unknown_count));
    stats.insert(
        "fold_histogram".into(),
        histogram(
            (0..total)
                .filter(|&k| finite[k] && believable[k / gates] && planes.state[k] != STATE_REJECTED)
                .map(|k| planes.fold[k]),
        ),
    );
    let _ = unknown;
    Ok((planes, stats))
}

/// numpy's float64 pairwise summation (`pairwise_sum_DOUBLE`), so a sum of
/// many terms rounds exactly as `ndarray.sum()` does.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.median` of finite values (sorted; the mean of the middle two).
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (0.0 + values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// `_reconcile_sectors`: `[(nyquist, shift)]` in the reference's dict order.
fn reconcile(
    output: &[f64],
    azimuth: &[f64],
    row_nyquist: &[f64],
    believable: &[bool],
    sectors: &[f64],
    gates: usize,
) -> Vec<(f64, i64)> {
    let rows = azimuth.len();
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (azimuth[a], azimuth[b]);
        match (x.is_nan(), y.is_nan()) {
            (false, false) => x.partial_cmp(&y).unwrap(),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (true, true) => std::cmp::Ordering::Equal,
        }
    });
    let order: Vec<usize> = order.into_iter().filter(|&r| believable[r]).collect();
    let n = order.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        let (a, b) = (order[i], order[(i + 1) % n]);
        if row_nyquist[a] != row_nyquist[b] {
            pairs.push((a, b));
        }
    }
    let mut both: Vec<(usize, usize)> = pairs.clone();
    both.extend(pairs.iter().map(|&(p, q)| (q, p)));
    let size = |v: f64| (0..rows).filter(|&r| believable[r] && row_nyquist[r] == v).count();
    // max by (size, -v): the largest sector, ties to the smaller interval;
    // Python's max keeps the first of equal keys.
    let mut reference = sectors[0];
    for &v in &sectors[1..] {
        let (sv, sr) = (size(v), size(reference));
        if sv > sr || (sv == sr && -v > -reference) {
            reference = v;
        }
    }
    let row = |r: usize| &output[r * gates..(r + 1) * gates];

    let propagate = |ref_shift: i64| -> Vec<(f64, i64)> {
        let mut shift: Vec<(f64, i64)> = vec![(reference, ref_shift)];
        let find = |shift: &Vec<(f64, i64)>, v: f64| shift.iter().find(|(s, _)| *s == v).map(|(_, k)| *k);
        for _ in 0..sectors.len() {
            for &(a, b) in &both {
                let (sa, sb) = (row_nyquist[a], row_nyquist[b]);
                let Some(ka) = find(&shift, sa) else { continue };
                if find(&shift, sb).is_some() {
                    continue;
                }
                let offset = ka as f64 * 2.0 * sa;
                let diffs: Vec<f64> = row(a)
                    .iter()
                    .zip(row(b))
                    .map(|(&va, &vb)| (va + offset, vb))
                    .filter(|(va, vb)| va.is_finite() && vb.is_finite())
                    .map(|(va, vb)| va - vb)
                    .collect();
                if diffs.len() < 3 {
                    continue;
                }
                let k = (median(diffs) / (2.0 * sb)).round_ties_even() as i64;
                shift.push((sb, k));
            }
        }
        for &v in sectors {
            if find(&shift, v).is_none() {
                shift.push((v, 0));
            }
        }
        shift
    };
    let sweep_mean = |shift: &Vec<(f64, i64)>| -> f64 {
        let mut total = 0.0f64;
        let mut count = 0i64;
        for &(v, k) in shift {
            let offset = k as f64 * 2.0 * v;
            let mut block = Vec::new();
            for r in 0..rows {
                if believable[r] && row_nyquist[r] == v {
                    for &x in row(r) {
                        if x.is_finite() {
                            block.push(x + offset);
                        }
                    }
                }
            }
            total += pairwise_sum(&block);
            count += block.len() as i64;
        }
        if count > 0 { (total / count as f64).abs() } else { 0.0 }
    };
    let mut bins: Vec<f64> = (0..rows)
        .filter(|&r| believable[r])
        .map(|r| (azimuth[r] / 10.0).floor())
        .collect();
    bins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    bins.dedup();
    let candidates: &[i64] = if bins.len() >= 27 { &[-2, -1, 0, 1, 2] } else { &[0] };
    let mut best: Option<(Vec<(f64, i64)>, f64)> = None;
    for &candidate in candidates {
        let shift = propagate(candidate);
        let mean = sweep_mean(&shift);
        if best.as_ref().is_none_or(|(_, m)| mean < *m) {
            best = Some((shift, mean));
        }
    }
    best.map(|(s, _)| s).unwrap_or_default()
}
