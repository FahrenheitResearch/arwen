//! Opt-in, bounded wrapped-domain vortex proposal for RIFT-VDA.
//!
//! This module is invoked only by the opt-in RIFT path in [`crate::solver`]. A
//! caller supplies the current region solution plus an *unsigned* gate-resolution
//! local-cut mask.  The fit uses raw wrapped velocity only; the incumbent and
//! local mask are used solely to decide whether the fitted absolute branch is
//! safe to fuse.  On every rejected or over-budget input the returned velocity
//! is bit-for-bit the supplied baseline.
//!
//! The implementation is dependency-free and has fixed work budgets: at most
//! eight centers, 128 coarse fits, 27 bounded refinements, four full fits, and
//! one binary cut over at most 4,096 gates.

use std::collections::VecDeque;

pub const MAX_ROI_GATES: usize = 4_096;
pub const MAX_LOCAL_GATES: usize = 1_024;
pub const MAX_SIGNED_SEEDS: usize = 256;
pub const MAX_CENTER_SEEDS: usize = 8;
pub const MAX_ABS_FOLD: i8 = 4;

const CENTER_TANGENT_OFFSET_KM: f64 = 2.0;
const CENTER_RADIAL_OFFSETS_KM: [f64; 4] = [-5.0, -3.0, -1.0, 1.0];
const RADIUS_BANK_KM: [f64; 8] = [0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];
const AMPLITUDE_BANK_N: [f64; 2] = [0.8, 3.6];
const BRANCH_IRLS_STEPS: usize = 3;
const REFINE_OFFSET_KM: f64 = 0.75;
const REFINE_RADIUS_SCALE: [f64; 3] = [0.75, 1.0, 1.25];
const REFINE_PARENT_LIMIT: usize = 1;
const FULL_CANDIDATE_LIMIT: usize = 4;
const FIT_RADIUS_KM: f64 = 6.0;

const SIGNED_EDGE_N: f64 = -1.75;
const SEED_RAY_REACH: usize = 3;
const SEED_GATE_REACH: usize = 3;
const MIN_SEED_EDGES: usize = 3;
const SUPPORT_RAY_PAD: usize = 3;
const SUPPORT_GATE_PAD: usize = 6;
const SUPPORT_VELOCITY_N: f64 = 0.35;
const MIN_SUPPORT_PER_SIGN: usize = 15;

const HUBER_SIGMA_N: f64 = 0.12;
const HUBER_DELTA: f64 = 1.5;
const MIN_MODEL_IMPROVEMENT: f64 = 0.08;
const MIN_SIDE_COVERAGE: usize = 15;
const MIN_PROPOSAL_GATES: usize = 6;
const MAX_PROPOSAL_FRACTION: f64 = 0.45;
const MIN_STABILITY_IOU: f64 = 0.45;
const MIN_LOCAL_MODEL_IOU: f64 = 0.50;
const MIN_LOCAL_BRANCH_CONSENSUS: f64 = 0.90;
const MIN_SIGNED_NEGATIVE_FRACTION: f64 = 0.75;
const MIN_VORTEX_AMPLITUDE_N: f64 = 1.0;
const MAX_OPPOSITE_BRANCH_FRACTION: f64 = 0.10;

const FUSION_FOLD_PENALTY: f64 = 0.35;
const FUSION_LOCAL_CUT_CUE: f64 = 10.0;
const FUSION_PAIR_WEIGHT: f64 = 0.8;
const FUSION_PAIR_SCALE_N: f64 = 0.70;
const HARD_CAPACITY: f64 = 1.0e9;
const FLOW_EPSILON: f64 = 1.0e-10;

pub const ABSTAIN_INVALID_INPUT: u32 = 1 << 0;
pub const ABSTAIN_BUDGET: u32 = 1 << 1;
pub const ABSTAIN_NO_SIGNED_SUPPORT: u32 = 1 << 2;
pub const ABSTAIN_NO_FIT: u32 = 1 << 3;
pub const ABSTAIN_LOW_IMPROVEMENT: u32 = 1 << 4;
pub const ABSTAIN_LOW_COVERAGE: u32 = 1 << 5;
pub const ABSTAIN_LOW_LOCAL_OVERLAP: u32 = 1 << 6;
pub const ABSTAIN_WRONG_BRANCH: u32 = 1 << 7;
pub const ABSTAIN_LOW_BRANCH_CONSENSUS: u32 = 1 << 8;
pub const ABSTAIN_OPPOSITE_BRANCH: u32 = 1 << 9;
pub const ABSTAIN_UNSTABLE: u32 = 1 << 10;
pub const ABSTAIN_BROAD_PROPOSAL: u32 = 1 << 11;
pub const ABSTAIN_FUSION_ENERGY: u32 = 1 << 12;
pub const ABSTAIN_NYQUIST_TRANSITION: u32 = 1 << 13;

/// Inputs for one already-localized proposal-fusion ROI.
///
/// `local_cut_mask` is deliberately unsigned.  The vortex fit must determine
/// the absolute fold direction independently. `first_gate_m` is the center
/// range of gate zero in this ROI; callers must supply real radar geometry.
pub struct Input<'a> {
    pub observed: &'a [f32],
    pub baseline: &'a [f32],
    pub local_cut_mask: &'a [u8],
    pub azimuth_deg: &'a [f32],
    pub nyquist_mps: &'a [f32],
    pub rows: usize,
    pub gates: usize,
    pub first_gate_m: f32,
    pub gate_spacing_m: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitSummary {
    pub center_east_km: f64,
    pub center_north_km: f64,
    pub radius_km: f64,
    pub amplitude_mps: f64,
    pub env_east_mps: f64,
    pub env_north_mps: f64,
    pub wrapped_cost: f64,
    pub background_cost: f64,
    pub improvement_fraction: f64,
    pub fit_gates: usize,
    pub negative_side_gates: usize,
    pub positive_side_gates: usize,
    pub proposal_gates: usize,
    pub proposal_components: usize,
    pub proposal_largest_component: usize,
    pub signed_edge_count: usize,
    pub signed_negative_branch_edges: usize,
    pub signed_negative_branch_fraction: f64,
    pub signed_mean_sensitivity: f64,
    pub local_cut_intersection: usize,
    pub local_cut_union: usize,
    pub local_cut_iou: f64,
    pub local_branch_delta: i8,
    pub local_branch_consensus: f64,
    pub opposite_branch_fraction: f64,
    pub stability_iou: f64,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub velocity: Vec<f32>,
    pub accepted_mask: Vec<u8>,
    /// Complete raw-model proposal, expressed relative to the baseline fold.
    pub model_fold_delta: Vec<i8>,
    pub fit: Option<FitSummary>,
    pub accepted: bool,
    pub abstain_flags: u32,
    pub signed_seed_edges: usize,
    pub support_windows: usize,
    pub center_seeds: usize,
    pub coarse_fits: usize,
    pub refined_fits: usize,
    pub full_fits: usize,
    pub fusion_energy_delta: f64,
}

impl Outcome {
    fn baseline(input: &Input<'_>, flags: u32) -> Self {
        Self {
            velocity: input.baseline.to_vec(),
            accepted_mask: vec![0; input.baseline.len()],
            model_fold_delta: vec![0; input.baseline.len()],
            fit: None,
            accepted: false,
            abstain_flags: flags,
            signed_seed_edges: 0,
            support_windows: 0,
            center_seeds: 0,
            coarse_fits: 0,
            refined_fits: 0,
            full_fits: 0,
            fusion_energy_delta: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
struct SupportWindow {
    center_east_km: f64,
    center_north_km: f64,
}

#[derive(Clone)]
struct Candidate {
    fit: FitSummary,
    prediction: Vec<f64>,
    proposal_fold: Vec<i8>,
    changed: Vec<bool>,
}

#[derive(Clone, Copy)]
struct CenterSeed {
    east_km: f64,
    north_km: f64,
}

struct Geometry {
    east: Vec<f64>,
    north: Vec<f64>,
    sin_az: Vec<f64>,
    cos_az: Vec<f64>,
}

/// Fit, validate, and fuse one unsigned local proposal.
pub fn propose_and_fuse(input: &Input<'_>) -> Outcome {
    let Some(total) = validate(input) else {
        return Outcome::baseline(input, ABSTAIN_INVALID_INPUT);
    };
    let local_count = input
        .local_cut_mask
        .iter()
        .zip(input.observed)
        .filter(|(marked, value)| **marked != 0 && value.is_finite())
        .count();
    if total > MAX_ROI_GATES || local_count > MAX_LOCAL_GATES {
        return Outcome::baseline(input, ABSTAIN_BUDGET);
    }
    if local_count < MIN_PROPOSAL_GATES {
        return Outcome::baseline(input, ABSTAIN_NO_SIGNED_SUPPORT);
    }

    // The fixed WLS amplitude bank is normalized by one Nyquist value. Until
    // the fitter itself is zoned, never substitute a first/median value across
    // a transition: conservatively abstain. Exact equality is intentional;
    // even a near-mixed per-ray interval is not scalar noise. All fold
    // assignment and fusion helpers below nevertheless use each ray's N.
    let nyquist = input.nyquist_mps[0] as f64;
    if input
        .nyquist_mps
        .iter()
        .any(|value| value.to_bits() != input.nyquist_mps[0].to_bits())
    {
        return Outcome::baseline(input, ABSTAIN_NYQUIST_TRANSITION);
    }
    let geometry = make_geometry(input);
    let (supports, signed_seed_edges, overflowed) = signed_supports(input, &geometry, nyquist);
    if overflowed {
        return Outcome::baseline(input, ABSTAIN_BUDGET);
    }
    if supports.is_empty() {
        let mut outcome = Outcome::baseline(input, ABSTAIN_NO_SIGNED_SUPPORT);
        outcome.signed_seed_edges = signed_seed_edges;
        return outcome;
    }
    let seeds = center_seeds(&supports);
    if seeds.is_empty() {
        let mut outcome = Outcome::baseline(input, ABSTAIN_NO_SIGNED_SUPPORT);
        outcome.signed_seed_edges = signed_seed_edges;
        outcome.support_windows = supports.len();
        return outcome;
    }

    let fit_run = fit_vortex(input, &geometry, nyquist, &seeds);
    let mut outcome = Outcome::baseline(input, ABSTAIN_NO_FIT);
    outcome.signed_seed_edges = signed_seed_edges;
    outcome.support_windows = supports.len();
    outcome.center_seeds = seeds.len();
    outcome.coarse_fits = fit_run.coarse_count;
    outcome.refined_fits = fit_run.refined_count;
    outcome.full_fits = fit_run.full_count;
    let Some(mut best) = fit_run.best else {
        return outcome;
    };

    let reasons = acceptance_flags(input, nyquist, &best.fit);
    outcome.model_fold_delta = baseline_relative_folds(input, nyquist, &best.proposal_fold);
    outcome.fit = Some(best.fit.clone());
    if reasons != 0 {
        outcome.abstain_flags = reasons;
        return outcome;
    }

    let fusion = fuse(input, nyquist, &best);
    outcome.fusion_energy_delta = fusion.energy_delta;
    if fusion.energy_delta >= 0.0 || fusion.changed == 0 {
        outcome.abstain_flags = ABSTAIN_FUSION_ENERGY;
        return outcome;
    }
    outcome.velocity = fusion.velocity;
    outcome.accepted_mask = fusion.mask;
    outcome.accepted = true;
    outcome.abstain_flags = 0;
    // Drop the largest temporary vectors before returning in debug builds.
    best.prediction.clear();
    best.proposal_fold.clear();
    best.changed.clear();
    outcome
}

fn validate(input: &Input<'_>) -> Option<usize> {
    let total = input.rows.checked_mul(input.gates)?;
    if input.rows < 2
        || input.gates < 2
        || total == 0
        || input.observed.len() != total
        || input.baseline.len() != total
        || input.local_cut_mask.len() != total
        || input.azimuth_deg.len() != input.rows
        || input.nyquist_mps.len() != input.rows
        || !input.first_gate_m.is_finite()
        || input.first_gate_m < 0.0
        || !input.gate_spacing_m.is_finite()
        || input.gate_spacing_m <= 0.0
    {
        return None;
    }
    if input
        .nyquist_mps
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return None;
    }
    Some(total)
}

fn make_geometry(input: &Input<'_>) -> Geometry {
    let total = input.rows * input.gates;
    let mut east = vec![0.0; total];
    let mut north = vec![0.0; total];
    let mut sin_az = vec![0.0; total];
    let mut cos_az = vec![0.0; total];
    for row in 0..input.rows {
        let radians = (input.azimuth_deg[row] as f64).to_radians();
        let sine = radians.sin();
        let cosine = radians.cos();
        for gate in 0..input.gates {
            let index = row * input.gates + gate;
            let range =
                (input.first_gate_m as f64 + input.gate_spacing_m as f64 * gate as f64) / 1_000.0;
            east[index] = sine * range;
            north[index] = cosine * range;
            sin_az[index] = sine;
            cos_az[index] = cosine;
        }
    }
    Geometry {
        east,
        north,
        sin_az,
        cos_az,
    }
}

fn wrapped_azimuth_delta(first: f32, second: f32) -> f64 {
    ((second as f64 - first as f64 + 180.0).rem_euclid(360.0)) - 180.0
}

fn is_canonical_forward_edge(first: f32, second: f32) -> bool {
    let delta = wrapped_azimuth_delta(first, second);
    delta > 0.0 && delta < 2.0
}

fn signed_supports(
    input: &Input<'_>,
    geometry: &Geometry,
    nyquist: f64,
) -> (Vec<SupportWindow>, usize, bool) {
    let mut seeds = Vec::new();
    for row in 0..input.rows {
        let next = (row + 1) % input.rows;
        if !is_canonical_forward_edge(input.azimuth_deg[row], input.azimuth_deg[next]) {
            continue;
        }
        for gate in 0..input.gates {
            let first = input.baseline[row * input.gates + gate];
            let second = input.baseline[next * input.gates + gate];
            if first.is_finite()
                && second.is_finite()
                && (second as f64 - first as f64) < SIGNED_EDGE_N * nyquist
            {
                seeds.push((row, gate));
                if seeds.len() > MAX_SIGNED_SEEDS {
                    return (Vec::new(), seeds.len(), true);
                }
            }
        }
    }
    let seed_count = seeds.len();
    let mut assigned = vec![false; seeds.len()];
    let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
    for start in 0..seeds.len() {
        if assigned[start] {
            continue;
        }
        assigned[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut group = Vec::new();
        while let Some(current) = queue.pop_front() {
            let point = seeds[current];
            group.push(point);
            for candidate in 0..seeds.len() {
                if assigned[candidate] {
                    continue;
                }
                let other = seeds[candidate];
                let ray_direct = point.0.abs_diff(other.0);
                let ray_distance = ray_direct.min(input.rows - ray_direct);
                if ray_distance <= SEED_RAY_REACH && point.1.abs_diff(other.1) <= SEED_GATE_REACH {
                    assigned[candidate] = true;
                    queue.push_back(candidate);
                }
            }
        }
        group.sort_unstable();
        groups.push(group);
    }
    groups.sort_by_key(|group| group[0]);

    let mut supports = Vec::new();
    for group in groups {
        if group.len() < MIN_SEED_EDGES {
            continue;
        }
        let min_row = group.iter().map(|point| point.0).min().unwrap();
        let max_row = group.iter().map(|point| point.0).max().unwrap();
        if max_row - min_row > input.rows / 2 {
            continue;
        }
        let min_gate = group.iter().map(|point| point.1).min().unwrap();
        let max_gate = group.iter().map(|point| point.1).max().unwrap();
        let row_start = min_row.saturating_sub(SUPPORT_RAY_PAD);
        let row_end = (max_row + 2 + SUPPORT_RAY_PAD).min(input.rows);
        let gate_start = min_gate.saturating_sub(SUPPORT_GATE_PAD);
        let gate_end = (max_gate + 1 + SUPPORT_GATE_PAD).min(input.gates);
        let mut negative = 0usize;
        let mut positive = 0usize;
        let mut overlaps_local = false;
        for row in row_start..row_end {
            for gate in gate_start..gate_end {
                let index = row * input.gates + gate;
                let value = input.baseline[index];
                if !value.is_finite() {
                    continue;
                }
                negative += usize::from(value as f64 <= -SUPPORT_VELOCITY_N * nyquist);
                positive += usize::from(value as f64 >= SUPPORT_VELOCITY_N * nyquist);
                overlaps_local |= input.local_cut_mask[index] != 0;
            }
        }
        if negative < MIN_SUPPORT_PER_SIGN || positive < MIN_SUPPORT_PER_SIGN || !overlaps_local {
            continue;
        }
        let mut east_sum = 0.0;
        let mut north_sum = 0.0;
        let mut endpoints = 0usize;
        for &(row, gate) in &group {
            for endpoint_row in [row, (row + 1) % input.rows] {
                let index = endpoint_row * input.gates + gate;
                east_sum += geometry.east[index];
                north_sum += geometry.north[index];
                endpoints += 1;
            }
        }
        supports.push(SupportWindow {
            center_east_km: east_sum / endpoints as f64,
            center_north_km: north_sum / endpoints as f64,
        });
        if supports.len() == MAX_CENTER_SEEDS / CENTER_RADIAL_OFFSETS_KM.len() {
            break;
        }
    }
    (supports, seed_count, false)
}

fn center_seeds(supports: &[SupportWindow]) -> Vec<CenterSeed> {
    let mut seeds = Vec::new();
    for support in supports {
        let bearing = support.center_east_km.atan2(support.center_north_km);
        let radial = [bearing.sin(), bearing.cos()];
        let clockwise = [bearing.cos(), -bearing.sin()];
        for radial_offset in CENTER_RADIAL_OFFSETS_KM {
            seeds.push(CenterSeed {
                east_km: support.center_east_km
                    + radial_offset * radial[0]
                    + CENTER_TANGENT_OFFSET_KM * clockwise[0],
                north_km: support.center_north_km
                    + radial_offset * radial[1]
                    + CENTER_TANGENT_OFFSET_KM * clockwise[1],
            });
            if seeds.len() == MAX_CENTER_SEEDS {
                return seeds;
            }
        }
    }
    seeds
}

struct FitRun {
    best: Option<Candidate>,
    coarse_count: usize,
    refined_count: usize,
    full_count: usize,
}

fn fit_vortex(
    input: &Input<'_>,
    geometry: &Geometry,
    nyquist: f64,
    seeds: &[CenterSeed],
) -> FitRun {
    let mut coarse = Vec::new();
    for seed in seeds {
        let distance = center_distances(geometry, *seed);
        for radius in RADIUS_BANK_KM {
            let sensitivity = radius_sensitivity(geometry, *seed, radius, &distance);
            for amplitude_n in AMPLITUDE_BANK_N {
                if let Some(mut candidate) = fit_fixed(
                    input,
                    geometry,
                    &distance,
                    &sensitivity,
                    nyquist,
                    *seed,
                    radius,
                    amplitude_n * nyquist,
                    true,
                ) {
                    annotate(input, geometry, nyquist, &mut candidate, false);
                    // Only the summary participates in coarse ranking. Keeping
                    // three full-ROI buffers for every bank entry turns the
                    // bounded search into tens of megabytes of live memory.
                    candidate.prediction = Vec::new();
                    candidate.proposal_fold = Vec::new();
                    candidate.changed = Vec::new();
                    coarse.push(candidate);
                }
            }
        }
    }
    let coarse_count = coarse.len();
    coarse.sort_by(|first, second| candidate_cmp(nyquist, first, second));

    let mut refined = Vec::new();
    let mut seen = Vec::<(i64, i64, i64, i64)>::new();
    for parent in coarse.iter().take(REFINE_PARENT_LIMIT) {
        let fit = &parent.fit;
        let bearing = fit.center_east_km.atan2(fit.center_north_km);
        let radial = [bearing.sin(), bearing.cos()];
        let tangent = [bearing.cos(), -bearing.sin()];
        for radial_step in -1..=1 {
            for tangent_step in -1..=1 {
                let center = CenterSeed {
                    east_km: fit.center_east_km
                        + REFINE_OFFSET_KM
                            * (radial_step as f64 * radial[0] + tangent_step as f64 * tangent[0]),
                    north_km: fit.center_north_km
                        + REFINE_OFFSET_KM
                            * (radial_step as f64 * radial[1] + tangent_step as f64 * tangent[1]),
                };
                let distance = center_distances(geometry, center);
                for scale in REFINE_RADIUS_SCALE {
                    let radius = (fit.radius_km * scale).clamp(0.25, 4.5);
                    let key = (
                        (center.east_km * 10_000.0).round() as i64,
                        (center.north_km * 10_000.0).round() as i64,
                        (radius * 10_000.0).round() as i64,
                        (fit.amplitude_mps * 10_000.0).round() as i64,
                    );
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.push(key);
                    let sensitivity = radius_sensitivity(geometry, center, radius, &distance);
                    if let Some(mut candidate) = fit_fixed(
                        input,
                        geometry,
                        &distance,
                        &sensitivity,
                        nyquist,
                        center,
                        radius,
                        fit.amplitude_mps,
                        true,
                    ) {
                        annotate(input, geometry, nyquist, &mut candidate, false);
                        candidate.prediction = Vec::new();
                        candidate.proposal_fold = Vec::new();
                        candidate.changed = Vec::new();
                        refined.push(candidate);
                    }
                }
            }
        }
    }
    let refined_count = refined.len();
    refined.sort_by(|first, second| candidate_cmp(nyquist, first, second));

    let mut full = Vec::new();
    for candidate in refined.iter().take(FULL_CANDIDATE_LIMIT) {
        let fit = &candidate.fit;
        let center = CenterSeed {
            east_km: fit.center_east_km,
            north_km: fit.center_north_km,
        };
        let distance = center_distances(geometry, center);
        let sensitivity = radius_sensitivity(geometry, center, fit.radius_km, &distance);
        if let Some(mut evaluated) = fit_fixed(
            input,
            geometry,
            &distance,
            &sensitivity,
            nyquist,
            center,
            fit.radius_km,
            fit.amplitude_mps,
            false,
        ) {
            annotate(input, geometry, nyquist, &mut evaluated, true);
            full.push(evaluated);
        }
    }
    full.sort_by(|first, second| candidate_cmp(nyquist, first, second));
    let full_count = full.len();
    if !full.is_empty() {
        let stability = if full.len() > 1 {
            mask_iou(&full[0].changed, &full[1].changed)
        } else {
            0.0
        };
        full[0].fit.stability_iou = stability;
    }
    FitRun {
        best: full.into_iter().next(),
        coarse_count,
        refined_count,
        full_count,
    }
}

fn candidate_cmp(nyquist: f64, first: &Candidate, second: &Candidate) -> std::cmp::Ordering {
    let first_physical = candidate_is_physical(nyquist, &first.fit);
    let second_physical = candidate_is_physical(nyquist, &second.fit);
    second_physical
        .cmp(&first_physical)
        .then_with(|| second.fit.local_cut_iou.total_cmp(&first.fit.local_cut_iou))
        .then_with(|| {
            second
                .fit
                .local_branch_consensus
                .total_cmp(&first.fit.local_branch_consensus)
        })
        .then_with(|| first.fit.wrapped_cost.total_cmp(&second.fit.wrapped_cost))
        .then_with(|| {
            second
                .fit
                .improvement_fraction
                .total_cmp(&first.fit.improvement_fraction)
        })
        .then_with(|| first.fit.proposal_gates.cmp(&second.fit.proposal_gates))
        .then_with(|| first.fit.radius_km.total_cmp(&second.fit.radius_km))
        .then_with(|| {
            first
                .fit
                .center_east_km
                .total_cmp(&second.fit.center_east_km)
        })
        .then_with(|| {
            first
                .fit
                .center_north_km
                .total_cmp(&second.fit.center_north_km)
        })
}

fn candidate_is_physical(nyquist: f64, fit: &FitSummary) -> bool {
    fit.improvement_fraction >= MIN_MODEL_IMPROVEMENT
        && fit.amplitude_mps >= MIN_VORTEX_AMPLITUDE_N * nyquist
        && fit.signed_edge_count > 0
        && fit.signed_negative_branch_fraction >= MIN_SIGNED_NEGATIVE_FRACTION
        && fit.signed_mean_sensitivity < -0.05
        && fit.local_branch_delta == -1
        && fit.local_branch_consensus >= 0.75
        && fit.opposite_branch_fraction <= 0.20
}

// These slices and scalar controls are kept separate to make every hot-loop
// dependency explicit and avoid allocating a candidate-input object.
#[allow(clippy::too_many_arguments)]
fn fit_fixed(
    input: &Input<'_>,
    geometry: &Geometry,
    distance: &[f64],
    sensitivity: &[f64],
    nyquist: f64,
    center: CenterSeed,
    radius: f64,
    amplitude_seed: f64,
    decimate: bool,
) -> Option<Candidate> {
    let total = input.rows * input.gates;
    debug_assert_eq!(distance.len(), total);
    debug_assert_eq!(sensitivity.len(), total);
    let sample: Vec<usize> = (0..total)
        .filter(|index| {
            let row = index / input.gates;
            let gate = index % input.gates;
            input.observed[*index].is_finite()
                && input.baseline[*index].is_finite()
                && (!decimate || (row.is_multiple_of(2) && gate.is_multiple_of(2)))
        })
        .collect();
    if sample.len() < 40 {
        return None;
    }
    let env = initial_environment(input, geometry, &sample)?;
    let mut beta = [amplitude_seed, env[0], env[1]];
    let sigma = HUBER_SIGMA_N * nyquist;
    for _ in 0..BRANCH_IRLS_STEPS {
        let mut normal = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for &index in &sample {
            let design = [
                sensitivity[index],
                geometry.sin_az[index],
                geometry.cos_az[index],
            ];
            let prediction = dot3(design, beta);
            let observed = input.observed[index] as f64;
            let period = 2.0 * input.nyquist_mps[index / input.gates] as f64;
            let fold = ((prediction - observed) / period)
                .round()
                .clamp(-(MAX_ABS_FOLD as f64), MAX_ABS_FOLD as f64);
            let target = observed + period * fold;
            let residual = target - prediction;
            let geometry_weight = 1.0 / (1.0 + (distance[index] / FIT_RADIUS_KM).powi(6));
            let huber_weight = (HUBER_DELTA * sigma / residual.abs().max(1.0e-6)).min(1.0);
            let weight = geometry_weight * huber_weight;
            for row in 0..3 {
                rhs[row] += weight * design[row] * target;
                for column in 0..3 {
                    normal[row][column] += weight * design[row] * design[column];
                }
            }
        }
        let scale = ((normal[0][0] + normal[1][1] + normal[2][2]) / 3.0).max(1.0);
        let ridge = [1.0e-5 * scale, 2.0e-3 * scale, 2.0e-3 * scale];
        for index in 0..3 {
            normal[index][index] += ridge[index];
            rhs[index] += ridge[index] * beta[index];
        }
        beta = solve3(normal, rhs)?;
        beta[0] = beta[0].clamp(0.25 * nyquist, 5.0 * nyquist);
    }

    let mut prediction = vec![f64::NAN; total];
    let mut proposal_fold = vec![0i8; total];
    let changed = vec![false; total];
    for index in 0..total {
        if !input.observed[index].is_finite() {
            continue;
        }
        prediction[index] = beta[0] * sensitivity[index]
            + beta[1] * geometry.sin_az[index]
            + beta[2] * geometry.cos_az[index];
        let period = 2.0 * input.nyquist_mps[index / input.gates] as f64;
        proposal_fold[index] = (((prediction[index] - input.observed[index] as f64) / period)
            .round()
            .clamp(-(MAX_ABS_FOLD as f64), MAX_ABS_FOLD as f64))
            as i8;
    }

    let mut vortex_loss = 0.0;
    let mut background_loss = 0.0;
    let mut weight_sum = 0.0;
    for &index in &sample {
        let weight = 1.0 / (1.0 + (distance[index] / FIT_RADIUS_KM).powi(6));
        let observed = input.observed[index] as f64;
        let period = 2.0 * input.nyquist_mps[index / input.gates] as f64;
        let vortex_residual = wrap_period(prediction[index] - observed, period);
        let background = beta[1] * geometry.sin_az[index] + beta[2] * geometry.cos_az[index];
        let background_residual = wrap_period(background - observed, period);
        vortex_loss += weight * huber_loss(vortex_residual, sigma);
        background_loss += weight * huber_loss(background_residual, sigma);
        weight_sum += weight;
    }
    let wrapped_cost = vortex_loss / weight_sum.max(1.0e-9);
    let background_cost = background_loss / weight_sum.max(1.0e-9);
    let improvement = (background_cost - wrapped_cost) / background_cost.max(1.0e-9);

    let mut negative_side = 0usize;
    let mut positive_side = 0usize;
    let side_radius = (2.5 * radius).max(1.5);
    for &index in &sample {
        if distance[index] <= side_radius {
            negative_side += usize::from(sensitivity[index] <= -0.2);
            positive_side += usize::from(sensitivity[index] >= 0.2);
        }
    }
    let fit = FitSummary {
        center_east_km: center.east_km,
        center_north_km: center.north_km,
        radius_km: radius,
        amplitude_mps: beta[0],
        env_east_mps: beta[1],
        env_north_mps: beta[2],
        wrapped_cost,
        background_cost,
        improvement_fraction: improvement,
        fit_gates: sample.len(),
        negative_side_gates: negative_side,
        positive_side_gates: positive_side,
        ..FitSummary::default()
    };
    Some(Candidate {
        fit,
        prediction,
        proposal_fold,
        changed,
    })
}

fn center_distances(geometry: &Geometry, center: CenterSeed) -> Vec<f64> {
    geometry
        .east
        .iter()
        .zip(&geometry.north)
        .map(|(&east, &north)| (east - center.east_km).hypot(north - center.north_km))
        .collect()
}

fn radius_sensitivity(
    geometry: &Geometry,
    center: CenterSeed,
    radius: f64,
    distance: &[f64],
) -> Vec<f64> {
    (0..distance.len())
        .map(|index| {
            vortex_sensitivity(
                geometry.east[index] - center.east_km,
                geometry.north[index] - center.north_km,
                distance[index],
                radius,
                geometry.sin_az[index],
                geometry.cos_az[index],
            )
        })
        .collect()
}

fn initial_environment(
    input: &Input<'_>,
    geometry: &Geometry,
    sample: &[usize],
) -> Option<[f64; 2]> {
    let mut normal = [[0.0; 2]; 2];
    let mut rhs = [0.0; 2];
    for &index in sample {
        let design = [geometry.sin_az[index], geometry.cos_az[index]];
        let target = input.baseline[index] as f64;
        for row in 0..2 {
            rhs[row] += design[row] * target;
            for column in 0..2 {
                normal[row][column] += design[row] * design[column];
            }
        }
    }
    let ridge = sample.len() as f64 * 1.0e-3;
    normal[0][0] += ridge;
    normal[1][1] += ridge;
    solve2(normal, rhs)
}

fn annotate(
    input: &Input<'_>,
    geometry: &Geometry,
    nyquist: f64,
    candidate: &mut Candidate,
    summarize_components: bool,
) {
    let total = input.rows * input.gates;
    let mut local_count = 0usize;
    let mut local_histogram = [0usize; 17];
    let mut changed_count = 0usize;
    let mut local_intersection = 0usize;
    let mut local_union = 0usize;
    let mut local_opposite = 0usize;
    let mut local_changed = 0usize;
    let mut max_abs_proposal = 0i8;
    for index in 0..total {
        if !input.observed[index].is_finite() || !input.baseline[index].is_finite() {
            continue;
        }
        let period = 2.0 * input.nyquist_mps[index / input.gates] as f64;
        let baseline_fold =
            ((input.baseline[index] as f64 - input.observed[index] as f64) / period).round() as i16;
        let delta = candidate.proposal_fold[index] as i16 - baseline_fold;
        candidate.changed[index] = delta != 0;
        max_abs_proposal = max_abs_proposal.max(candidate.proposal_fold[index].abs());
        let local = input.local_cut_mask[index] != 0;
        changed_count += usize::from(delta != 0);
        local_intersection += usize::from(local && delta != 0);
        local_union += usize::from(local || delta != 0);
        if local {
            local_count += 1;
            if delta != 0 {
                let bucket = (delta + 8).clamp(0, 16) as usize;
                local_histogram[bucket] += 1;
                local_changed += 1;
                local_opposite += usize::from(delta > 0);
            }
        }
    }
    let (dominant_bucket, dominant_count) = local_histogram
        .iter()
        .enumerate()
        .filter(|(bucket, _)| *bucket != 8)
        .max_by(|(left_bucket, left_count), (right_bucket, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| {
                    let left = (*left_bucket as i16 - 8).abs();
                    let right = (*right_bucket as i16 - 8).abs();
                    right.cmp(&left)
                })
                .then_with(|| right_bucket.cmp(left_bucket))
        })
        .map(|(bucket, count)| (bucket, *count))
        .unwrap_or((8, 0));
    candidate.fit.local_branch_delta = (dominant_bucket as i16 - 8) as i8;
    candidate.fit.local_branch_consensus = if local_count > 0 {
        dominant_count as f64 / local_count as f64
    } else {
        0.0
    };
    candidate.fit.local_cut_intersection = local_intersection;
    candidate.fit.local_cut_union = local_union;
    candidate.fit.local_cut_iou = if local_union > 0 {
        local_intersection as f64 / local_union as f64
    } else {
        1.0
    };
    candidate.fit.opposite_branch_fraction = if local_changed > 0 {
        local_opposite as f64 / local_changed as f64
    } else {
        0.0
    };
    candidate.fit.proposal_gates = changed_count;

    if summarize_components {
        let (components, largest) = component_summary(&candidate.changed, input.rows, input.gates);
        candidate.fit.proposal_components = components;
        candidate.fit.proposal_largest_component = largest;
    }

    let mut signed_count = 0usize;
    let mut signed_negative = 0usize;
    let mut sensitivity_sum = 0.0;
    for row in 0..input.rows.saturating_sub(1) {
        if !is_canonical_forward_edge(input.azimuth_deg[row], input.azimuth_deg[row + 1]) {
            continue;
        }
        for gate in 0..input.gates {
            let index = row * input.gates + gate;
            let next = index + input.gates;
            let first = input.baseline[index];
            let second = input.baseline[next];
            if !first.is_finite()
                || !second.is_finite()
                || (second as f64 - first as f64) >= SIGNED_EDGE_N * nyquist
            {
                continue;
            }
            signed_count += 1;
            let period = 2.0 * input.nyquist_mps[row] as f64;
            let baseline_fold =
                ((first as f64 - input.observed[index] as f64) / period).round() as i16;
            let delta = candidate.proposal_fold[index] as i16 - baseline_fold;
            signed_negative += usize::from(delta == -1);
            let dx = geometry.east[index] - candidate.fit.center_east_km;
            let dy = geometry.north[index] - candidate.fit.center_north_km;
            let distance = dx.hypot(dy);
            sensitivity_sum += vortex_sensitivity(
                dx,
                dy,
                distance,
                candidate.fit.radius_km,
                geometry.sin_az[index],
                geometry.cos_az[index],
            );
        }
    }
    candidate.fit.signed_edge_count = signed_count;
    candidate.fit.signed_negative_branch_edges = signed_negative;
    candidate.fit.signed_negative_branch_fraction = if signed_count > 0 {
        signed_negative as f64 / signed_count as f64
    } else {
        0.0
    };
    candidate.fit.signed_mean_sensitivity = if signed_count > 0 {
        sensitivity_sum / signed_count as f64
    } else {
        0.0
    };
    let _ = max_abs_proposal;
}

fn acceptance_flags(input: &Input<'_>, nyquist: f64, fit: &FitSummary) -> u32 {
    let mut flags = 0u32;
    if fit.improvement_fraction < MIN_MODEL_IMPROVEMENT
        || fit.amplitude_mps < MIN_VORTEX_AMPLITUDE_N * nyquist
    {
        flags |= ABSTAIN_LOW_IMPROVEMENT;
    }
    if fit.negative_side_gates.min(fit.positive_side_gates) < MIN_SIDE_COVERAGE {
        flags |= ABSTAIN_LOW_COVERAGE;
    }
    if fit.local_cut_iou < MIN_LOCAL_MODEL_IOU {
        flags |= ABSTAIN_LOW_LOCAL_OVERLAP;
    }
    if fit.signed_edge_count == 0
        || fit.signed_negative_branch_fraction < MIN_SIGNED_NEGATIVE_FRACTION
        || fit.signed_mean_sensitivity >= -0.05
        || fit.local_branch_delta != -1
    {
        flags |= ABSTAIN_WRONG_BRANCH;
    }
    if fit.local_branch_consensus < MIN_LOCAL_BRANCH_CONSENSUS {
        flags |= ABSTAIN_LOW_BRANCH_CONSENSUS;
    }
    if fit.opposite_branch_fraction > MAX_OPPOSITE_BRANCH_FRACTION {
        flags |= ABSTAIN_OPPOSITE_BRANCH;
    }
    if fit.stability_iou < MIN_STABILITY_IOU {
        flags |= ABSTAIN_UNSTABLE;
    }
    let valid = input
        .observed
        .iter()
        .filter(|value| value.is_finite())
        .count();
    if fit.proposal_gates < MIN_PROPOSAL_GATES
        || fit.proposal_gates as f64 > MAX_PROPOSAL_FRACTION * valid.max(1) as f64
    {
        flags |= ABSTAIN_BROAD_PROPOSAL;
    }
    flags
}

fn baseline_relative_folds(input: &Input<'_>, nyquist: f64, proposal: &[i8]) -> Vec<i8> {
    let _ = nyquist;
    input
        .observed
        .iter()
        .zip(input.baseline)
        .zip(proposal)
        .enumerate()
        .map(|(index, ((&observed, &baseline), &fold))| {
            if observed.is_finite() && baseline.is_finite() {
                let period = 2.0 * input.nyquist_mps[index / input.gates] as f64;
                let base = ((baseline as f64 - observed as f64) / period).round() as i16;
                (fold as i16 - base).clamp(i8::MIN as i16, i8::MAX as i16) as i8
            } else {
                0
            }
        })
        .collect()
}

struct FusionResult {
    velocity: Vec<f32>,
    mask: Vec<u8>,
    energy_delta: f64,
    changed: usize,
}

fn fuse(input: &Input<'_>, nyquist: f64, candidate: &Candidate) -> FusionResult {
    let total = input.rows * input.gates;
    let sigma = HUBER_SIGMA_N * nyquist;
    let delta = candidate.fit.local_branch_delta;
    let mut proposed = input.baseline.to_vec();
    let mut d0 = vec![0.0; total];
    let mut d1 = vec![0.0; total];
    let mut finite = vec![false; total];
    for index in 0..total {
        finite[index] = input.observed[index].is_finite()
            && input.baseline[index].is_finite()
            && candidate.prediction[index].is_finite();
        if !finite[index] {
            continue;
        }
        let local = input.local_cut_mask[index] != 0;
        let period = 2.0 * input.nyquist_mps[index / input.gates] as f64;
        if local {
            proposed[index] = input.baseline[index] + (period as f32) * delta as f32;
        }
        d0[index] = huber_loss(
            input.baseline[index] as f64 - candidate.prediction[index],
            sigma,
        ) + if local { FUSION_LOCAL_CUT_CUE } else { 0.0 };
        d1[index] = huber_loss(proposed[index] as f64 - candidate.prediction[index], sigma)
            + if local {
                FUSION_FOLD_PENALTY * (delta as f64).abs()
            } else {
                0.0
            };
    }

    let mut node_of = vec![usize::MAX; total];
    let coordinates: Vec<usize> = (0..total).filter(|index| finite[*index]).collect();
    for (node, &index) in coordinates.iter().enumerate() {
        node_of[index] = node;
    }
    let source = coordinates.len();
    let sink = source + 1;
    let mut graph = FlowGraph::new(sink + 1);
    for (node, &index) in coordinates.iter().enumerate() {
        let row = index / input.gates;
        let gate = index % input.gates;
        let boundary = row == 0 || gate == 0 || row + 1 == input.rows || gate + 1 == input.gates;
        graph.add_edge(
            source,
            node,
            d1[index] + if boundary { HARD_CAPACITY } else { 0.0 },
        );
        graph.add_edge(node, sink, d0[index]);
    }
    for &index in &coordinates {
        let row = index / input.gates;
        let gate = index % input.gates;
        for next in [
            (row + 1 < input.rows).then_some(index + input.gates),
            (gate + 1 < input.gates).then_some(index + 1),
        ]
        .into_iter()
        .flatten()
        {
            if !finite[next] {
                continue;
            }
            let edge_period =
                input.nyquist_mps[row].min(input.nyquist_mps[next / input.gates]) as f64 * 2.0;
            let gradient = wrap_period(
                input.observed[index] as f64 - input.observed[next] as f64,
                edge_period,
            )
            .abs();
            let weight =
                FUSION_PAIR_WEIGHT * (-(gradient / (FUSION_PAIR_SCALE_N * nyquist)).powi(2)).exp();
            graph.add_undirected(node_of[index], node_of[next], weight);
        }
    }
    graph.max_flow(source, sink);
    let source_set = graph.source_set(source);
    let mut mask = vec![0u8; total];
    let mut velocity = input.baseline.to_vec();
    for (node, &index) in coordinates.iter().enumerate() {
        if !source_set[node] && input.local_cut_mask[index] != 0 {
            mask[index] = 1;
            velocity[index] = proposed[index];
        }
    }
    let unary_before: f64 = coordinates.iter().map(|index| d0[*index]).sum();
    let unary_after: f64 = coordinates
        .iter()
        .map(|index| {
            if mask[*index] != 0 {
                d1[*index]
            } else {
                d0[*index]
            }
        })
        .sum();
    let mut pair_after = 0.0;
    for &index in &coordinates {
        let row = index / input.gates;
        let gate = index % input.gates;
        for next in [
            (row + 1 < input.rows).then_some(index + input.gates),
            (gate + 1 < input.gates).then_some(index + 1),
        ]
        .into_iter()
        .flatten()
        {
            if finite[next] && mask[index] != mask[next] {
                let edge_period =
                    input.nyquist_mps[row].min(input.nyquist_mps[next / input.gates]) as f64 * 2.0;
                let gradient = wrap_period(
                    input.observed[index] as f64 - input.observed[next] as f64,
                    edge_period,
                )
                .abs();
                pair_after += FUSION_PAIR_WEIGHT
                    * (-(gradient / (FUSION_PAIR_SCALE_N * nyquist)).powi(2)).exp();
            }
        }
    }
    let changed = mask.iter().filter(|value| **value != 0).count();
    FusionResult {
        velocity,
        mask,
        energy_delta: unary_after + pair_after - unary_before,
        changed,
    }
}

fn vortex_sensitivity(
    dx: f64,
    dy: f64,
    distance: f64,
    radius: f64,
    sin_az: f64,
    cos_az: f64,
) -> f64 {
    if distance < 1.0e-6 {
        return 0.0;
    }
    let q = distance / radius;
    let profile = 2.0_f64.sqrt() * q / (1.0 + q.powi(4)).sqrt();
    let tangential_east = -dy / distance;
    let tangential_north = dx / distance;
    profile * (tangential_east * sin_az + tangential_north * cos_az)
}

fn wrap_period(value: f64, period: f64) -> f64 {
    (value + 0.5 * period).rem_euclid(period) - 0.5 * period
}

fn huber_loss(residual: f64, sigma: f64) -> f64 {
    let normalized = residual.abs() / sigma;
    if normalized <= HUBER_DELTA {
        0.5 * normalized * normalized
    } else {
        HUBER_DELTA * (normalized - 0.5 * HUBER_DELTA)
    }
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn solve2(matrix: [[f64; 2]; 2], rhs: [f64; 2]) -> Option<[f64; 2]> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if !determinant.is_finite() || determinant.abs() < 1.0e-12 {
        return None;
    }
    Some([
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ])
}

#[allow(clippy::needless_range_loop)]
fn solve3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for column in 0..3 {
        let pivot = (column..3).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if !matrix[pivot][column].is_finite() || matrix[pivot][column].abs() < 1.0e-12 {
            return None;
        }
        if pivot != column {
            matrix.swap(pivot, column);
            rhs.swap(pivot, column);
        }
        let divisor = matrix[column][column];
        for item in column..3 {
            matrix[column][item] /= divisor;
        }
        rhs[column] /= divisor;
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for item in column..3 {
                matrix[row][item] -= factor * matrix[column][item];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    rhs.iter().all(|value| value.is_finite()).then_some(rhs)
}

fn mask_iou(first: &[bool], second: &[bool]) -> f64 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (&left, &right) in first.iter().zip(second) {
        intersection += usize::from(left && right);
        union += usize::from(left || right);
    }
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn component_summary(mask: &[bool], rows: usize, gates: usize) -> (usize, usize) {
    let mut seen = vec![false; mask.len()];
    let mut components = 0usize;
    let mut largest = 0usize;
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        components += 1;
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut size = 0usize;
        while let Some(index) = queue.pop_front() {
            size += 1;
            let row = index / gates;
            let gate = index % gates;
            for next in [
                (row > 0).then(|| index - gates),
                (row + 1 < rows).then_some(index + gates),
                (gate > 0).then(|| index - 1),
                (gate + 1 < gates).then_some(index + 1),
            ]
            .into_iter()
            .flatten()
            {
                if mask[next] && !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
        largest = largest.max(size);
    }
    (components, largest)
}

#[derive(Clone, Copy)]
struct FlowEdge {
    target: usize,
    reverse: usize,
    capacity: f64,
}

struct FlowGraph {
    graph: Vec<Vec<FlowEdge>>,
}

impl FlowGraph {
    fn new(nodes: usize) -> Self {
        Self {
            graph: vec![Vec::new(); nodes],
        }
    }

    fn add_edge(&mut self, source: usize, target: usize, capacity: f64) {
        let source_reverse = self.graph[target].len();
        let target_reverse = self.graph[source].len();
        self.graph[source].push(FlowEdge {
            target,
            reverse: source_reverse,
            capacity,
        });
        self.graph[target].push(FlowEdge {
            target: source,
            reverse: target_reverse,
            capacity: 0.0,
        });
    }

    fn add_undirected(&mut self, first: usize, second: usize, capacity: f64) {
        self.add_edge(first, second, capacity);
        self.add_edge(second, first, capacity);
    }

    fn max_flow(&mut self, source: usize, sink: usize) {
        let nodes = self.graph.len();
        let mut height = vec![0usize; nodes];
        let mut excess = vec![0.0f64; nodes];
        let mut cursor = vec![0usize; nodes];
        let mut queued = vec![false; nodes];
        let mut active = VecDeque::new();
        height[source] = nodes;
        for edge_index in 0..self.graph[source].len() {
            let edge = self.graph[source][edge_index];
            if edge.capacity <= FLOW_EPSILON {
                continue;
            }
            self.graph[source][edge_index].capacity = 0.0;
            self.graph[edge.target][edge.reverse].capacity += edge.capacity;
            excess[source] -= edge.capacity;
            excess[edge.target] += edge.capacity;
            if edge.target != sink && !queued[edge.target] {
                queued[edge.target] = true;
                active.push_back(edge.target);
            }
        }
        while let Some(node) = active.pop_front() {
            queued[node] = false;
            while excess[node] > FLOW_EPSILON {
                if cursor[node] == self.graph[node].len() {
                    let Some(minimum) = self.graph[node]
                        .iter()
                        .filter(|edge| edge.capacity > FLOW_EPSILON)
                        .map(|edge| height[edge.target])
                        .min()
                    else {
                        break;
                    };
                    height[node] = minimum.saturating_add(1);
                    cursor[node] = 0;
                    continue;
                }
                let edge_index = cursor[node];
                let edge = self.graph[node][edge_index];
                if edge.capacity <= FLOW_EPSILON || height[node] != height[edge.target] + 1 {
                    cursor[node] += 1;
                    continue;
                }
                let pushed = excess[node].min(edge.capacity);
                let target_was_inactive = excess[edge.target] <= FLOW_EPSILON;
                self.graph[node][edge_index].capacity -= pushed;
                self.graph[edge.target][edge.reverse].capacity += pushed;
                excess[node] -= pushed;
                excess[edge.target] += pushed;
                if edge.target != source
                    && edge.target != sink
                    && target_was_inactive
                    && !queued[edge.target]
                {
                    queued[edge.target] = true;
                    active.push_back(edge.target);
                }
            }
            if excess[node] > FLOW_EPSILON && !queued[node] {
                queued[node] = true;
                active.push_back(node);
            }
        }
    }

    fn source_set(&self, source: usize) -> Vec<bool> {
        let mut reached = vec![false; self.graph.len()];
        reached[source] = true;
        let mut queue = VecDeque::from([source]);
        while let Some(node) = queue.pop_front() {
            for edge in &self.graph[node] {
                if edge.capacity > FLOW_EPSILON && !reached[edge.target] {
                    reached[edge.target] = true;
                    queue.push_back(edge.target);
                }
            }
        }
        reached
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    struct Case {
        name: &'static str,
        rows: usize,
        gates: usize,
        global_gate_offset: usize,
        expected_changed: usize,
        local_runs: &'static [(usize, usize, usize)],
        observed: &'static [u8],
        azimuth: &'static [u8],
        nyquist: &'static [u8],
        expected: &'static [u8],
    }

    const CASES: &[Case] = &[
        Case {
            name: "El Reno 2013",
            rows: 25,
            gates: 37,
            global_gate_offset: 204,
            expected_changed: 137,
            local_runs: &[
                (4, 18, 21),
                (5, 18, 22),
                (6, 15, 22),
                (7, 13, 22),
                (8, 7, 10),
                (8, 12, 26),
                (8, 29, 35),
                (9, 6, 22),
                (9, 29, 36),
                (10, 6, 20),
                (10, 29, 34),
                (11, 6, 20),
                (11, 29, 32),
                (12, 6, 20),
                (13, 8, 20),
                (14, 9, 11),
                (14, 12, 15),
                (15, 9, 10),
            ],
            observed: include_bytes!("../test/fixtures/rift/el-reno-2013/observed.f32"),
            azimuth: include_bytes!("../test/fixtures/rift/el-reno-2013/azimuth.f32"),
            nyquist: include_bytes!("../test/fixtures/rift/el-reno-2013/nyquist.f32"),
            expected: include_bytes!("../test/fixtures/rift/el-reno-2013/expected.f32"),
        },
        Case {
            name: "Tuscaloosa 2011",
            rows: 34,
            gates: 49,
            global_gate_offset: 139,
            expected_changed: 78,
            local_runs: &[
                (5, 33, 35),
                (6, 26, 29),
                (6, 32, 34),
                (7, 23, 33),
                (8, 22, 32),
                (9, 22, 36),
                (10, 21, 36),
                (11, 23, 35),
                (12, 24, 29),
                (12, 31, 33),
                (13, 24, 27),
            ],
            observed: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/observed.f32"),
            azimuth: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/azimuth.f32"),
            nyquist: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/nyquist.f32"),
            expected: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/expected.f32"),
        },
    ];

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn bits_equal(left: &[f32], right: &[f32]) -> bool {
        left.iter()
            .zip(right)
            .all(|(&first, &second)| first.to_bits() == second.to_bits())
    }

    type ReviewedInput = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<f32>);

    fn reviewed_input(case: &Case) -> ReviewedInput {
        let observed = f32s(case.observed);
        let azimuth = f32s(case.azimuth);
        let nyquist = f32s(case.nyquist);
        let expected = f32s(case.expected);
        let mut baseline = expected.clone();
        let mut local = vec![0u8; observed.len()];
        for &(row, gate_start, gate_end) in case.local_runs {
            for gate in gate_start..gate_end {
                let index = row * case.gates + gate;
                local[index] = 1;
                baseline[index] += 2.0 * nyquist[row];
            }
        }
        (observed, azimuth, nyquist, baseline, local, expected)
    }

    fn run_case(case: &Case) -> Outcome {
        let (observed, azimuth, nyquist, baseline, local, _) = reviewed_input(case);
        propose_and_fuse(&Input {
            observed: &observed,
            baseline: &baseline,
            local_cut_mask: &local,
            azimuth_deg: &azimuth,
            nyquist_mps: &nyquist,
            rows: case.rows,
            gates: case.gates,
            first_gate_m: 2_125.0 + case.global_gate_offset as f32 * 250.0,
            gate_spacing_m: 250.0,
        })
    }

    #[test]
    fn reviewed_crops_are_exact_and_bounded() {
        for case in CASES {
            let (_, _, _, _, _, expected) = reviewed_input(case);
            let started = Instant::now();
            let result = run_case(case);
            let elapsed = started.elapsed();
            eprintln!(
                "{}: {:?}, coarse/refined/full={}/{}/{}, model IoU={:.4}, gain={:.4}, branch={:?}/{:.4}, signed={:.4}, stable={:.4}, opposite={:.4}",
                case.name,
                elapsed,
                result.coarse_fits,
                result.refined_fits,
                result.full_fits,
                result.fit.as_ref().map_or(0.0, |fit| fit.local_cut_iou),
                result
                    .fit
                    .as_ref()
                    .map_or(0.0, |fit| fit.improvement_fraction),
                result.fit.as_ref().map(|fit| fit.local_branch_delta),
                result
                    .fit
                    .as_ref()
                    .map_or(0.0, |fit| fit.local_branch_consensus),
                result
                    .fit
                    .as_ref()
                    .map_or(0.0, |fit| fit.signed_negative_branch_fraction),
                result.fit.as_ref().map_or(0.0, |fit| fit.stability_iou),
                result
                    .fit
                    .as_ref()
                    .map_or(0.0, |fit| fit.opposite_branch_fraction),
            );
            assert!(
                result.accepted,
                "{} abstained: {:#x}",
                case.name, result.abstain_flags
            );
            assert_eq!(
                result
                    .accepted_mask
                    .iter()
                    .filter(|value| **value != 0)
                    .count(),
                case.expected_changed,
                "{} mask size",
                case.name,
            );
            assert!(
                result
                    .velocity
                    .iter()
                    .zip(&expected)
                    .all(|(&actual, &wanted)| actual.to_bits() == wanted.to_bits()),
                "{} did not reproduce the reviewed crop bit-for-bit",
                case.name,
            );
            assert!(result.center_seeds <= MAX_CENTER_SEEDS);
            assert!(
                result.coarse_fits
                    <= MAX_CENTER_SEEDS * RADIUS_BANK_KM.len() * AMPLITUDE_BANK_N.len()
            );
            assert!(result.refined_fits <= REFINE_PARENT_LIMIT * 9 * REFINE_RADIUS_SCALE.len());
            assert!(result.full_fits <= FULL_CANDIDATE_LIMIT);
            assert!(
                elapsed.as_secs_f32() < 5.0,
                "{} pathological test runtime",
                case.name
            );
        }
    }

    #[test]
    fn repeat_run_is_bit_deterministic() {
        for case in CASES {
            let first = run_case(case);
            let second = run_case(case);
            assert_eq!(first.accepted, second.accepted, "{} status", case.name);
            assert_eq!(
                first.accepted_mask, second.accepted_mask,
                "{} mask",
                case.name
            );
            assert_eq!(
                first.model_fold_delta, second.model_fold_delta,
                "{} folds",
                case.name
            );
            assert_eq!(
                first
                    .velocity
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                second
                    .velocity
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{} velocity",
                case.name,
            );
            assert_eq!(first.fit, second.fit, "{} fit", case.name);
        }
    }

    #[test]
    fn missing_independent_shape_abstains_without_harm() {
        let case = &CASES[0];
        let (observed, azimuth, nyquist, baseline, _, _) = reviewed_input(case);
        let local = vec![0u8; observed.len()];
        let result = propose_and_fuse(&Input {
            observed: &observed,
            baseline: &baseline,
            local_cut_mask: &local,
            azimuth_deg: &azimuth,
            nyquist_mps: &nyquist,
            rows: case.rows,
            gates: case.gates,
            first_gate_m: 2_125.0 + case.global_gate_offset as f32 * 250.0,
            gate_spacing_m: 250.0,
        });
        assert!(!result.accepted);
        assert_ne!(result.abstain_flags, 0);
        assert!(
            result
                .velocity
                .iter()
                .zip(&baseline)
                .all(|(&actual, &current)| actual.to_bits() == current.to_bits())
        );
    }

    #[test]
    fn over_budget_roi_abstains_before_allocating_fit_bank() {
        let rows = 65;
        let gates = 65;
        let total = rows * gates;
        let observed = vec![0.0f32; total];
        let baseline = observed.clone();
        let local = vec![1u8; total];
        let azimuth: Vec<f32> = (0..rows).map(|row| row as f32 * 0.5).collect();
        let nyquist = vec![30.0; rows];
        let result = propose_and_fuse(&Input {
            observed: &observed,
            baseline: &baseline,
            local_cut_mask: &local,
            azimuth_deg: &azimuth,
            nyquist_mps: &nyquist,
            rows,
            gates,
            first_gate_m: 1_000.0,
            gate_spacing_m: 250.0,
        });
        assert!(!result.accepted);
        assert_eq!(result.abstain_flags, ABSTAIN_BUDGET);
        assert_eq!(result.velocity, baseline);
        assert_eq!(result.coarse_fits, 0);
    }

    #[test]
    fn invalid_geometry_and_mixed_nyquist_are_safe_abstentions() {
        let case = &CASES[1];
        let (observed, azimuth, nyquist, baseline, local, _) = reviewed_input(case);
        let bad_geometry = propose_and_fuse(&Input {
            observed: &observed,
            baseline: &baseline,
            local_cut_mask: &local,
            azimuth_deg: &azimuth,
            nyquist_mps: &nyquist,
            rows: case.rows,
            gates: case.gates,
            first_gate_m: f32::NAN,
            gate_spacing_m: 250.0,
        });
        assert_eq!(bad_geometry.abstain_flags, ABSTAIN_INVALID_INPUT);
        assert!(bits_equal(&bad_geometry.velocity, &baseline));

        let zero_first_gate = propose_and_fuse(&Input {
            observed: &observed,
            baseline: &baseline,
            local_cut_mask: &local,
            azimuth_deg: &azimuth,
            nyquist_mps: &nyquist,
            rows: case.rows,
            gates: case.gates,
            first_gate_m: 0.0,
            gate_spacing_m: 250.0,
        });
        assert_eq!(zero_first_gate.abstain_flags & ABSTAIN_INVALID_INPUT, 0);

        let mut mixed = nyquist.clone();
        mixed[case.rows / 2] = f32::from_bits(mixed[case.rows / 2].to_bits() + 1);
        let transition = propose_and_fuse(&Input {
            observed: &observed,
            baseline: &baseline,
            local_cut_mask: &local,
            azimuth_deg: &azimuth,
            nyquist_mps: &mixed,
            rows: case.rows,
            gates: case.gates,
            first_gate_m: 2_125.0 + case.global_gate_offset as f32 * 250.0,
            gate_spacing_m: 250.0,
        });
        assert_eq!(transition.abstain_flags, ABSTAIN_NYQUIST_TRANSITION);
        assert!(bits_equal(&transition.velocity, &baseline));
        assert_eq!(transition.coarse_fits, 0);
    }

    #[test]
    fn signed_edge_geometry_is_strictly_forward_and_below_two_degrees() {
        assert!(is_canonical_forward_edge(0.0, 0.5));
        assert!(is_canonical_forward_edge(359.5, 0.5));
        assert!(!is_canonical_forward_edge(0.5, 0.0));
        assert!(!is_canonical_forward_edge(0.0, 0.0));
        assert!(!is_canonical_forward_edge(0.0, 2.0));
        assert!(!is_canonical_forward_edge(f32::NAN, 0.5));
    }

    #[test]
    fn final_signed_edge_annotation_rejects_reversed_ray_order() {
        fn signed_count(azimuth: &[f32; 2]) -> usize {
            let observed = [30.0, 30.0, -30.0, -30.0];
            let baseline = observed;
            let local = [1, 0, 0, 0];
            let nyquist = [30.0, 30.0];
            let input = Input {
                observed: &observed,
                baseline: &baseline,
                local_cut_mask: &local,
                azimuth_deg: azimuth,
                nyquist_mps: &nyquist,
                rows: 2,
                gates: 2,
                first_gate_m: 0.0,
                gate_spacing_m: 250.0,
            };
            let geometry = make_geometry(&input);
            let mut candidate = Candidate {
                fit: FitSummary {
                    radius_km: 1.0,
                    ..FitSummary::default()
                },
                prediction: vec![0.0; 4],
                proposal_fold: vec![-1, 0, 0, 0],
                changed: vec![false; 4],
            };
            annotate(&input, &geometry, 30.0, &mut candidate, false);
            candidate.fit.signed_edge_count
        }

        assert_eq!(signed_count(&[0.0, 0.5]), 2);
        assert_eq!(signed_count(&[359.5, 0.5]), 2);
        assert_eq!(signed_count(&[0.5, 0.0]), 0);
        assert_eq!(signed_count(&[0.0, 2.0]), 0);
    }
}
