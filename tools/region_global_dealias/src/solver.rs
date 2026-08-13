//! Region-based velocity dealiasing for a single radar sweep.
//!
//! # Attribution and licence
//!
//! A Rust port of Py-ART's `dealias_region_based` per-sweep core
//! (<https://github.com/ARM-DOE/pyart>, `pyart/correct/region_dealias.py`).
//! Copyright (c) 2013, UChicago Argonne, LLC. All rights reserved.
//! Licensed BSD-3-Clause, which applies to this file in addition to this
//! project's own MIT or Apache-2.0 terms. The verbatim upstream notice -
//! including the U.S. DOE government-rights preamble and its warranty
//! disclaimer - is kept at `PYART-LICENSE.txt`; see also
//! `THIRD-PARTY-NOTICES.md`.
//!
//! **This is modified software and must not be confused with the Py-ART
//! distributed by ANL.** It is a partial translation of one solver, and it is
//! altered:
//!
//! - [`MAX_EXTRA_INTERVALS`] bounds an interval count that Python's
//!   arbitrary-precision integers do not need to bound.
//! - Regions are labelled by a union-find raster pass rather than by a flood
//!   fill per velocity interval, which walked the whole sweep once per interval
//!   and probed all four neighbours of every gate it took. The regions and
//!   their numbering are unchanged - see [`find_regions`].
//! - The merge loop selects the strongest edge with a lazy-deletion max-heap
//!   and tracks shared neighbours with an epoch counter, rather than rescanning
//!   the edge list and clearing an array on every merge. Both were quadratic in
//!   the region count. Merged-away edges are likewise retired in place rather
//!   than spliced out of the two node lists holding them, which was linear in
//!   the merged-into node's degree on every merge. Selection is unchanged -
//!   verified edge-for-edge against the original on 86 real sweeps.
//!
//! Neither UChicago Argonne, LLC, Argonne National Laboratory, the U.S.
//! Government, nor the Py-ART contributors endorse this port. Cite Helmus &
//! Collis 2016, *J. Open Res. Softw.* 4(1) e25,
//! <https://doi.org/10.5334/jors.119>.
//!
//! # Model
//!
//! Everything here works on flat, row-major arrays - `rows * gates` velocities
//! in ray order - so the crate has no dependencies and no notion of a radar
//! file format. A non-finite velocity means "no data".

use std::collections::BinaryHeap;

#[path = "local_couplet.rs"]
mod local_couplet;
#[path = "rift_vortex.rs"]
mod rift_vortex;

const INTERVAL_SPLITS: usize = 3;
const SKIP_BETWEEN_RAYS: usize = 100;
const SKIP_ALONG_RAY: usize = 100;
/// Weight stamped on an edge once it has been merged away. Live weights are
/// gate counts, so they are always >= 1 and only ever grow; any negative weight
/// means retired, which is what every `weight < 0` test below reads.
const RETIRED: i32 = -999;
/// Furthest a ray-to-ray gap can reach: the adjacent ray plus the skips.
const MAX_RAY_REACH: isize = SKIP_BETWEEN_RAYS as isize + 1;
/// Ray index standing for "this gate has had no data on any ray in reach".
/// Halved so that `row - NO_PREVIOUS_RAY` cannot overflow.
const NO_PREVIOUS_RAY: isize = isize::MIN / 2;
/// Bound on the extra velocity intervals [`interval_limits`] will add per side
/// for data outside ±Nyquist. See the comment at its use site: this only ever
/// engages on non-physical input, and keeps `count` far from `i32` overflow.
const MAX_EXTRA_INTERVALS: i32 = 512;

/// Whether ray order closes a full 360 degree sweep, so the last ray is
/// azimuthally adjacent to the first. True for any normal full PPI.
///
/// Ray spacing is estimated as `360 / rows`, so this assumes the rays span the
/// circle. A real sector at native ray spacing is classified correctly (a
/// 90 degree sweep at 0.5 degree spacing is 180 rays and reads as open), but a
/// synthetic sweep with a handful of rays crammed into a narrow arc can read as
/// closed.
pub fn sweep_wraps(azimuths: &[f32]) -> bool {
    let rows = azimuths.len();
    if rows < 8 {
        return false;
    }
    let (Some(first), Some(last)) = (azimuths.first(), azimuths.last()) else {
        return false;
    };
    if !first.is_finite() || !last.is_finite() {
        return false;
    }
    let gap = (first - last)
        .rem_euclid(360.0)
        .min((last - first).rem_euclid(360.0));
    let typical = 360.0 / rows as f32;
    gap <= 3.0 * typical
}

/// Median of the usable per-ray Nyquist velocities, or `None` when no ray
/// carries one. Used to fill in rays whose own value is missing.
pub fn median_nyquist(nyq: &[f32]) -> Option<f32> {
    let mut values: Vec<f32> = nyq
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

/// Per-ray Nyquist with the sweep median substituted wherever a ray's own value
/// is missing or non-physical. Rays left `NaN` (no usable value anywhere on the
/// sweep) pass through the solver unfolded.
pub fn resolve_nyquist(nyq: &[f32], rows: usize) -> Vec<f32> {
    let fallback = median_nyquist(nyq);
    (0..rows)
        .map(|row| {
            nyq.get(row)
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .or(fallback)
                .unwrap_or(f32::NAN)
        })
        .collect()
}

/// Unfold one sweep, returning the velocity field with folds applied.
///
/// `observed` is `rows * gates` velocities in row-major ray order, `nyq` and
/// `azimuths` are one value per ray. Non-finite velocities pass through
/// untouched. Unlike the fold integers, the returned velocities are exact
/// `f32` - no fixed-point quantization is applied.
pub fn dealias_sweep(
    observed: &[f32],
    nyq: &[f32],
    rows: usize,
    gates: usize,
    azimuths: &[f32],
) -> Vec<f32> {
    let resolved = resolve_nyquist(nyq, rows);
    let folds = region_folds(observed, &resolved, rows, gates, sweep_wraps(azimuths));
    apply_folds(observed, &resolved, &folds, rows, gates)
}

pub const RIFT_API_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceKind {
    Caller = 0,
    Temporal = 1,
    Vertical = 2,
    Environmental = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceField<'a> {
    pub velocity: &'a [f32],
    pub quality: Option<&'a [u8]>,
    pub kind: ReferenceKind,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RiftContext<'a> {
    pub references: &'a [ReferenceField<'a>],
    pub reflectivity: Option<&'a [f32]>,
    pub spectrum_width: Option<&'a [f32]>,
    pub rho_hv: Option<&'a [f32]>,
}

#[derive(Clone, Copy, Debug)]
pub struct RiftOptions {
    pub max_abs_fold: u8,
    pub max_rois: u8,
    pub max_roi_gates: u32,
    pub max_total_roi_gates: u32,
    pub min_confidence: u8,
    /// Center range of gate zero, in metres. Required by the physically scaled
    /// automatic wrapped-vortex path; ignored for reference-only refinement.
    pub first_gate_m: f32,
    /// Range spacing between adjacent gates, in metres.
    pub gate_spacing_m: f32,
    /// Run the single-sweep ambiguity/vortex rescue. Disable this when only
    /// caller/temporal/vertical/environmental proposals should be considered.
    pub automatic_single_sweep: bool,
}

impl Default for RiftOptions {
    fn default() -> Self {
        Self {
            max_abs_fold: 4,
            max_rois: 4,
            max_roi_gates: 65_536,
            max_total_roi_gates: 0,
            min_confidence: 160,
            first_gate_m: f32::NAN,
            gate_spacing_m: f32::NAN,
            automatic_single_sweep: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiftError {
    Dimensions,
    Length,
    Limit,
    Reference,
}

impl std::fmt::Display for RiftError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Dimensions => "rows and gates must describe a non-empty sweep",
            Self::Length => "an input array does not match the sweep dimensions",
            Self::Limit => "a RIFT option exceeds its supported safety limit",
            Self::Reference => "a reference field is invalid",
        })
    }
}

impl std::error::Error for RiftError {}

pub const RIFT_REASON_RESIDUE_TRIGGER: u16 = 1 << 0;
pub const RIFT_REASON_BRANCH_UNSTABLE: u16 = 1 << 1;
pub const RIFT_REASON_TEMPORAL_ANCHOR: u16 = 1 << 2;
pub const RIFT_REASON_VERTICAL_ANCHOR: u16 = 1 << 3;
pub const RIFT_REASON_ENVIRONMENTAL_ANCHOR: u16 = 1 << 4;
pub const RIFT_REASON_CALLER_ANCHOR: u16 = 1 << 5;
pub const RIFT_REASON_VORTEX_PROPOSAL: u16 = 1 << 6;
pub const RIFT_REASON_FUSION_ACCEPTED: u16 = 1 << 7;
pub const RIFT_REASON_CONFLICTING_REFERENCES: u16 = 1 << 8;
pub const RIFT_REASON_LOW_COVERAGE: u16 = 1 << 9;
pub const RIFT_REASON_ABSTAINED: u16 = 1 << 10;
pub const RIFT_REASON_BUDGET_EXCEEDED: u16 = 1 << 11;
pub const RIFT_REASON_NYQUIST_TRANSITION: u16 = 1 << 12;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RiftStats {
    pub rois_detected: u32,
    pub rois_solved: u32,
    pub rois_accepted: u32,
    pub gates_refined: u32,
    pub gates_ambiguous: u32,
    pub budget_aborts: u32,
}

/// RIFT output. Unchanged baseline gates have zero confidence and reason bits;
/// confidence describes only an accepted gate-level refinement.
#[derive(Clone, Debug)]
pub struct RiftResult {
    pub velocity: Vec<f32>,
    pub folds: Vec<i8>,
    pub confidence: Vec<u8>,
    pub reasons: Vec<u16>,
    pub stats: RiftStats,
}

/// Region-Initialized Fold Tracking (RIFT-VDA).
///
/// The legacy [`dealias_sweep`] remains region-only and byte-compatible. This
/// opt-in path retains that result as its baseline, then fuses up to four
/// contextual proposals and an aggressively gated single-sweep rescue.
pub fn dealias_sweep_rift(
    observed: &[f32],
    nyq: &[f32],
    rows: usize,
    gates: usize,
    azimuths: &[f32],
    context: &RiftContext<'_>,
    options: RiftOptions,
) -> Result<RiftResult, RiftError> {
    let Some(total) = rows.checked_mul(gates).filter(|&total| total > 0) else {
        return Err(RiftError::Dimensions);
    };
    if observed.len() != total || azimuths.len() != rows {
        return Err(RiftError::Length);
    }
    if !(1..=8).contains(&options.max_abs_fold)
        || !(1..=16).contains(&options.max_rois)
        || options.max_roi_gates == 0
        || options.max_roi_gates > 262_144
        || options.max_total_roi_gates > 262_144
        || context.references.len() > 4
        || (options.automatic_single_sweep
            && (!options.first_gate_m.is_finite()
                || options.first_gate_m < 0.0
                || !options.gate_spacing_m.is_finite()
                || options.gate_spacing_m <= 0.0))
    {
        return Err(RiftError::Limit);
    }
    for field in [context.reflectivity, context.spectrum_width, context.rho_hv]
        .into_iter()
        .flatten()
    {
        if field.len() != total {
            return Err(RiftError::Length);
        }
    }
    for reference in context.references {
        if reference.velocity.len() != total
            || reference
                .quality
                .is_some_and(|quality| quality.len() != total)
        {
            return Err(RiftError::Reference);
        }
    }

    let resolved = resolve_nyquist(nyq, rows);
    let mut folds = region_folds(observed, &resolved, rows, gates, sweep_wraps(azimuths));
    let mut confidence = vec![0u8; total];
    let mut reasons = vec![0u16; total];
    let mut stats = RiftStats::default();

    // References that choose different integer branches at the same gate are
    // genuinely contradictory evidence.  Do not let call order decide which
    // one wins: exclude that gate from every contextual proposal and report
    // the abstention explicitly.
    let reference_conflicts = conflicting_reference_gates(
        observed,
        &resolved,
        gates,
        context.references,
        options.max_abs_fold,
    );
    for (reason, &conflict) in reasons.iter_mut().zip(&reference_conflicts) {
        if conflict {
            *reason |= RIFT_REASON_CONFLICTING_REFERENCES | RIFT_REASON_ABSTAINED;
        }
    }

    // Contextual reference fusion is the first-class proposal path. It is
    // implemented independently of the single-sweep trigger so a trusted
    // temporal/caller anchor can split an otherwise indistinguishable region.
    let mut ordered_references: Vec<&ReferenceField<'_>> = context.references.iter().collect();
    ordered_references.sort_by_key(|reference| match reference.kind {
        ReferenceKind::Caller => 0,
        ReferenceKind::Temporal => 1,
        ReferenceKind::Vertical => 2,
        ReferenceKind::Environmental => 3,
    });
    for reference in ordered_references {
        let diagnostics = local_couplet::refine_reference(
            observed,
            &resolved,
            rows,
            gates,
            &mut folds,
            reference.velocity,
            reference.quality,
            reference.kind,
            options,
            &reference_conflicts,
            &mut confidence,
            &mut reasons,
        );
        stats.rois_detected = stats
            .rois_detected
            .saturating_add(diagnostics.candidates_detected);
        stats.rois_solved = stats
            .rois_solved
            .saturating_add(diagnostics.candidates_authorized);
        stats.rois_accepted = stats
            .rois_accepted
            .saturating_add(diagnostics.components_accepted);
        stats.gates_refined = stats
            .gates_refined
            .saturating_add(diagnostics.gates_refined);
        stats.budget_aborts = stats
            .budget_aborts
            .saturating_add(u32::from(diagnostics.abstain_flags != 0));
    }

    if options.automatic_single_sweep {
        if let Some(local_nyquist) = uniform_nyquist(&resolved) {
            let local = local_couplet::refine_folds(
                observed,
                local_nyquist,
                azimuths,
                rows,
                gates,
                &mut folds,
                options,
                &mut confidence,
                &mut reasons,
            );
            stats.rois_detected = stats
                .rois_detected
                .saturating_add(local.candidates_detected);
            stats.rois_solved = stats
                .rois_solved
                .saturating_add(local.candidates_authorized);
            stats.rois_accepted = stats
                .rois_accepted
                .saturating_add(local.components_accepted);
            stats.gates_refined = stats.gates_refined.saturating_add(local.gates_refined);
            stats.budget_aborts = stats
                .budget_aborts
                .saturating_add(u32::from(local.abstain_flags != 0));
        } else {
            // The automatic graph cut and vortex fit currently require one
            // exact fold interval. Keep the region baseline and make this
            // conservative sweep-level abstention visible to every caller.
            for (reason, value) in reasons.iter_mut().zip(observed) {
                if value.is_finite() {
                    *reason |= RIFT_REASON_NYQUIST_TRANSITION | RIFT_REASON_ABSTAINED;
                }
            }
        }
    }

    let mut compact_folds = Vec::with_capacity(total);
    for &fold in &folds {
        compact_folds.push(i8::try_from(fold).map_err(|_| RiftError::Limit)?);
    }
    stats.gates_ambiguous =
        u32::try_from(reasons.iter().filter(|&&reason| reason != 0).count()).unwrap_or(u32::MAX);
    // Count refined gates once even when more than one compatible context
    // source contributed to the same final branch.
    stats.gates_refined =
        u32::try_from(confidence.iter().filter(|&&value| value != 0).count()).unwrap_or(u32::MAX);
    Ok(RiftResult {
        velocity: apply_folds(observed, &resolved, &folds, rows, gates),
        folds: compact_folds,
        confidence,
        reasons,
        stats,
    })
}

fn conflicting_reference_gates(
    observed: &[f32],
    nyquist: &[f32],
    gates: usize,
    references: &[ReferenceField<'_>],
    max_abs_fold: u8,
) -> Vec<bool> {
    let mut proposal = vec![None; observed.len()];
    let mut conflicts = vec![false; observed.len()];
    for reference in references {
        for index in 0..observed.len() {
            if conflicts[index] || reference.quality.is_some_and(|quality| quality[index] == 0) {
                continue;
            }
            let row = index / gates;
            let n = nyquist.get(row).copied().unwrap_or(f32::NAN);
            let observed_value = observed[index];
            let reference_value = reference.velocity[index];
            if !observed_value.is_finite()
                || !reference_value.is_finite()
                || !n.is_finite()
                || n <= 0.0
            {
                continue;
            }
            let fold = ((reference_value - observed_value) / (2.0 * n))
                .round_ties_even()
                .clamp(-f32::from(max_abs_fold), f32::from(max_abs_fold))
                as i32;
            match proposal[index] {
                Some(previous) if previous != fold => conflicts[index] = true,
                None => proposal[index] = Some(fold),
                _ => {}
            }
        }
    }
    conflicts
}

fn apply_folds(
    observed: &[f32],
    resolved_nyquist: &[f32],
    folds: &[i32],
    rows: usize,
    gates: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; observed.len()];
    for row in 0..rows {
        let n = resolved_nyquist.get(row).copied().unwrap_or(f32::NAN);
        for gate in 0..gates {
            let index = row * gates + gate;
            let Some(&value) = observed.get(index) else {
                continue;
            };
            if !value.is_finite() {
                continue;
            }
            out[index] = if n.is_finite() && n > 0.0 {
                value + 2.0 * n * folds[index] as f32
            } else {
                value
            };
        }
    }
    out
}

/// Phase timings in milliseconds for `examples/profile.rs`:
/// `[interval_limits, find_regions, edge_sum_and_count, merge_loop]`, plus
/// the region and edge counts.
///
/// Behind a feature so it stays out of the default public surface; it exists
/// to keep optimization work honest, not for callers.
#[cfg(feature = "profiling")]
pub fn profile_phases(
    observed: &[f32],
    nyq: &[f32],
    rows: usize,
    gates: usize,
    wraps: bool,
) -> ([f64; 4], usize, usize) {
    use std::time::Instant;
    let nvel = sweep_nyquist(nyq).unwrap();
    let nyquist_interval = 2.0 * nvel;

    let t = Instant::now();
    let limits = interval_limits(nvel, observed);
    let t_limits = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (labels, region_sizes) = find_regions(observed, rows, gates, &limits);
    let t_regions = t.elapsed().as_secs_f64() * 1000.0;
    let nfeatures = region_sizes.len().saturating_sub(1);

    let t = Instant::now();
    let edges = edge_sum_and_count(
        &labels,
        observed,
        rows,
        gates,
        wraps,
        nyquist_interval,
        nfeatures,
    );
    let t_edges = t.elapsed().as_secs_f64() * 1000.0;
    let nedges = edges.len();

    let t = Instant::now();
    let mut regions = RegionTracker::new(region_sizes);
    let mut edge_tracker = EdgeTracker::new(edges, nfeatures + 1);
    while let Some((node1, node2, diff, edge_number)) = edge_tracker.pop_edge() {
        let mut rdiff = round_ties_even_to_i32(diff);
        let node1_size = regions.get_node_size(node1);
        let node2_size = regions.get_node_size(node2);
        let (base_node, merge_node) = if node1_size > node2_size {
            (node1, node2)
        } else {
            rdiff = -rdiff;
            (node2, node1)
        };
        if rdiff != 0 {
            regions.unwrap_node(merge_node, rdiff);
            edge_tracker.unwrap_node(merge_node, rdiff);
        }
        regions.merge_nodes(base_node, merge_node);
        edge_tracker.merge_nodes(base_node, merge_node, edge_number);
    }
    let t_merge = t.elapsed().as_secs_f64() * 1000.0;

    ([t_limits, t_regions, t_edges, t_merge], nfeatures, nedges)
}

/// Integer Nyquist fold per gate (0 where unknown or no data).
///
/// `nyq` must already be resolved per ray (see [`resolve_nyquist`]).
pub fn region_folds(
    observed: &[f32],
    nyq: &[f32],
    rows: usize,
    gates: usize,
    wraps: bool,
) -> Vec<i32> {
    let total = rows.saturating_mul(gates);
    // Every bail-out below leaves the whole sweep unfolded; the array is only
    // built for real once the labels are in hand, so the common path fills it
    // exactly once.
    let unfolded = || vec![0i32; total];
    if rows == 0 || gates == 0 || observed.len() != total {
        return unfolded();
    }

    let nvel = sweep_nyquist(nyq).unwrap_or(f32::NAN);
    if !nvel.is_finite() || nvel <= 0.0 {
        return unfolded();
    }
    let nyquist_interval = 2.0 * nvel;
    let limits = interval_limits(nvel, observed);
    let (labels, region_sizes) = find_regions(observed, rows, gates, &limits);
    let nfeatures = region_sizes.len().saturating_sub(1);
    if nfeatures < 2 {
        return unfolded();
    }

    let edges = edge_sum_and_count(
        &labels,
        observed,
        rows,
        gates,
        wraps,
        nyquist_interval,
        nfeatures,
    );
    if edges.is_empty() {
        return unfolded();
    }

    let mut regions = RegionTracker::new(region_sizes);
    let mut edge_tracker = EdgeTracker::new(edges, nfeatures + 1);
    while let Some((node1, node2, diff, edge_number)) = edge_tracker.pop_edge() {
        let mut rdiff = round_ties_even_to_i32(diff);
        let node1_size = regions.get_node_size(node1);
        let node2_size = regions.get_node_size(node2);
        let (base_node, merge_node) = if node1_size > node2_size {
            (node1, node2)
        } else {
            rdiff = -rdiff;
            (node2, node1)
        };
        if rdiff != 0 {
            regions.unwrap_node(merge_node, rdiff);
            edge_tracker.unwrap_node(merge_node, rdiff);
        }
        regions.merge_nodes(base_node, merge_node);
        edge_tracker.merge_nodes(base_node, merge_node, edge_number);
    }

    let gates_dealiased: u64 = regions
        .node_size
        .iter()
        .skip(1)
        .map(|&value| value as u64)
        .sum();
    if gates_dealiased > 0 {
        let total_folds: i64 = regions
            .original_region_sizes
            .iter()
            .enumerate()
            .skip(1)
            .map(|(region, &size)| size as i64 * regions.unwrap_number[region] as i64)
            .sum();
        let sweep_offset = round_ties_even_to_i32(total_folds as f64 / gates_dealiased as f64);
        if sweep_offset != 0 {
            for unwrap in &mut regions.unwrap_number {
                *unwrap -= sweep_offset;
            }
        }
    }

    // Label 0 is the background: it joins no node, so nothing ever unwrapped it,
    // and the sweep-wide correction above is not its to carry. Pinning it back
    // to zero turns the output into a plain gather with no per-gate branch.
    regions.unwrap_number[0] = 0;
    labels
        .iter()
        .map(|&label| regions.unwrap_number[label as usize])
        .collect()
}

fn sweep_nyquist(nyq: &[f32]) -> Option<f32> {
    nyq.iter()
        .copied()
        .find(|value| value.is_finite() && *value > 0.0)
}

/// Local couplet refinement currently assumes one exact fold interval across
/// its graph-cut window. Even a small per-ray difference is a transition, not
/// scalar noise: the automatic path must abstain until a variable-interval cut
/// has its own regression coverage.
fn uniform_nyquist(nyq: &[f32]) -> Option<f32> {
    let first = sweep_nyquist(nyq)?;
    nyq.iter()
        .copied()
        .all(|value| value.is_finite() && value > 0.0 && value.to_bits() == first.to_bits())
        .then_some(first)
}

fn interval_limits(nyquist: f32, observed: &[f32]) -> Vec<f32> {
    let interval = (2.0 * nyquist) / INTERVAL_SPLITS as f32;
    let mut add_start = 0i32;
    let mut add_end = 0i32;
    let (min_value, max_value) = finite_extent(observed);
    // A finite minimum is exactly the "some gate had data" flag: every finite
    // value is below the +inf the reduction starts from.
    if min_value.is_finite() && (max_value > nyquist || min_value < -nyquist) {
        // Py-ART computes both bounds inside this one combined condition (so a
        // sweep aliased on only one side still evaluates the other, possibly
        // negative, term) — matched deliberately for bit-parity.
        //
        // Unlike Python's arbitrary-precision `int`, these casts are i32, and
        // the sum below overflows for absurd inputs (a corrupt feed carrying
        // 1e30 m/s saturates both terms at i32::MAX). Clamp instead: MAX_EXTRA
        // admits ~170 folds per side at the default 3 splits, versus the ±5 the
        // region engines cap folds at, so no physically meaningful sweep is
        // affected and the limit vector stays a small allocation.
        add_start = clamp_extra_intervals(((max_value - nyquist) / interval).ceil());
        add_end = clamp_extra_intervals((-(min_value + nyquist) / interval).ceil());
    }
    let start = -nyquist - add_start as f32 * interval;
    let end = nyquist + add_end as f32 * interval;
    let count = INTERVAL_SPLITS as i32 + 1 + add_start + add_end;
    if count <= 1 {
        return vec![start, end];
    }
    (0..count)
        .map(|i| start + (end - start) * i as f32 / (count - 1) as f32)
        .collect()
}

/// Smallest and largest finite value in the field, as `(+inf, -inf)` when there
/// is none.
///
/// This reads every gate on the sweep, so its shape matters. Reducing the
/// floats directly leaves one scalar compare per gate however it is written:
/// IEEE min/max is not a vectorizable reduction unless the compiler can rule
/// NaN out, which it cannot here. Reducing the bit patterns as signed integers
/// is - the transform below is strictly monotone over finite values, so the
/// integer extremes decode back to the float extremes, and non-finite gates map
/// to the identity element exactly as the `is_finite` filter used to skip them.
/// Worth the indirection at roughly 3x on a full sweep.
///
/// `-0.0` and `0.0` sort apart under this key where they compare equal as
/// floats, so which one comes back can differ. That is invisible to the only
/// caller, which does nothing with the result but add ±Nyquist to it.
fn finite_extent(observed: &[f32]) -> (f32, f32) {
    let mut low = i32::MAX;
    let mut high = i32::MIN;
    for &value in observed {
        let key = order_key(value.to_bits());
        let finite = value.is_finite();
        low = low.min(if finite { key } else { i32::MAX });
        high = high.max(if finite { key } else { i32::MIN });
    }
    // Neither sentinel is reachable from finite input: both decode to a NaN.
    (
        if low == i32::MAX {
            f32::INFINITY
        } else {
            f32::from_bits(order_key(low as u32) as u32)
        },
        if high == i32::MIN {
            f32::NEG_INFINITY
        } else {
            f32::from_bits(order_key(high as u32) as u32)
        },
    )
}

/// Float bits reordered so signed integer comparison matches float comparison:
/// invert everything below the sign bit for negatives, leave positives alone.
/// Self-inverse, so the same call decodes.
#[inline(always)]
fn order_key(bits: u32) -> i32 {
    let raw = bits as i32;
    raw ^ (((raw >> 31) as u32 >> 1) as i32)
}

/// A `ceil`ed extra-interval count, bounded so `count` cannot overflow.
///
/// Rust's float-to-int casts saturate rather than wrap, so ±inf land on
/// `i32::{MAX,MIN}` and NaN lands on 0 before the clamp bounds them.
fn clamp_extra_intervals(value: f32) -> i32 {
    (value as i32).clamp(-MAX_EXTRA_INTERVALS, MAX_EXTRA_INTERVALS)
}

/// One provisional region: its union-find link, its gate count, and the
/// velocity interval it belongs to.
#[derive(Clone, Copy, Default)]
struct Provisional {
    /// Union-find parent, never larger than the label's own index.
    parent: i32,
    size: u32,
    interval: u16,
}

/// Labels the 4-connected regions of every velocity interval, returning the
/// per-gate label (0 = no region) and the gate count of each label.
///
/// Py-ART flood-fills one interval at a time: a full scan of the sweep per
/// interval, and every gate it takes is stacked and probed on all four sides.
/// This is the union-find equivalent, which sees each gate once. A single pass
/// in gate order looks only at the gate to the west and the gate on the previous
/// ray, hands out provisional labels, and records the pairs that turn out to be
/// one region; a second pass rewrites the provisional labels into final ones.
///
/// The numbering is observable downstream and comes out identical. Py-ART
/// numbers regions by (interval, first gate in ray order). Provisional labels
/// are minted in that same gate order and a merge always keeps the smaller of
/// the two, so the label a region survives under is the one minted at its first
/// gate — and numbering the survivors by interval, then in ascending order,
/// is Py-ART's numbering.
fn find_regions(
    observed: &[f32],
    rows: usize,
    gates: usize,
    limits: &[f32],
) -> (Vec<i32>, Vec<u32>) {
    let total = rows * gates;
    // Lets the bounds checks below see one length.
    let observed = &observed[..total];
    let mut labels = vec![0i32; total];

    // Provisional labels, with 0 standing for "no region". One array, not three:
    // the pass allocates once and the resolution reads all three fields of a
    // label together.
    let mut regions = vec![Provisional::default()];
    // Interval of each gate of the previous ray. Zeroed, so the first ray reads
    // "no region" above it, which is what an edge ray should see.
    let mut north_ids = vec![0u16; gates];
    // Gates hold a label for a stretch at a time, so their sizes are totalled a
    // stretch at a time: incrementing one counter per gate would serialise the
    // whole pass on store-to-load forwarding.
    let mut run_label = 0usize;
    let mut run_len = 0u32;

    let mut idx = 0;
    for _ in 0..rows {
        let mut west_id = 0u16;
        let mut west_label = 0i32;
        for slot in north_ids.iter_mut() {
            let id = interval_id(observed[idx], limits);
            let north_id = *slot;
            *slot = id;
            // A non-zero id on the previous ray means that gate was labelled,
            // so `idx` is at least one ray in and the read below is in bounds.
            debug_assert!(north_id == 0 || idx >= gates);
            let label = if id == 0 {
                0
            } else if west_id == id {
                if north_id == id {
                    let north = labels[idx - gates];
                    if north == west_label {
                        west_label
                    } else {
                        unite(&mut regions, west_label, north)
                    }
                } else {
                    west_label
                }
            } else if north_id == id {
                labels[idx - gates]
            } else {
                let fresh = regions.len() as i32;
                regions.push(Provisional {
                    parent: fresh,
                    size: 0,
                    interval: id,
                });
                fresh
            };
            labels[idx] = label;

            if label as usize == run_label {
                run_len += 1;
            } else {
                regions[run_label].size += run_len;
                run_label = label as usize;
                run_len = 1;
            }
            west_id = id;
            west_label = label;
            idx += 1;
        }
    }
    regions[run_label].size += run_len;
    // Dead from here, and freeing it before the resolution allocates rather
    // than on the way out keeps the sweep's peak heap where the allocator will
    // hand the same pages back next sweep instead of returning them to the
    // kernel — worth about a tenth of the solver on the parity corpus.
    drop(north_ids);

    resolve_labels(labels, &mut regions)
}

/// Turns the provisional labelling into Py-ART's, rewriting `labels` in place
/// and returning the gate count of each final label.
fn resolve_labels(mut labels: Vec<i32>, regions: &mut [Provisional]) -> (Vec<i32>, Vec<u32>) {
    // Sizes settle onto the roots. A label's parent is always smaller than it,
    // so one sweep from the top carries every chain down, however long.
    for label in (1..regions.len()).rev() {
        let up = regions[label].parent as usize;
        if up != label {
            let carried = regions[label].size;
            regions[up].size += carried;
        }
    }

    // Final numbering is (interval, first gate), and the roots are already in
    // first-gate order, so it is a counting sort on the interval alone.
    let bins = regions.iter().map(|r| r.interval).max().unwrap_or(0) as usize;
    let mut next = vec![0i32; bins + 1];
    for label in 1..regions.len() {
        if regions[label].parent == label as i32 {
            next[regions[label].interval as usize] += 1;
        }
    }
    let mut running = 1;
    for slot in &mut next {
        let start = running;
        running += *slot;
        *slot = start;
    }

    // The final label overwrites the union-find link it is derived from, which
    // saves an array the size of the provisional labelling. A label's parent is
    // always smaller than it, so by the time a label is read as a parent it
    // already holds its final number.
    let mut sizes = vec![0u32; running as usize];
    for label in 1..regions.len() {
        let up = regions[label].parent as usize;
        regions[label].parent = if up == label {
            let slot = &mut next[regions[label].interval as usize];
            sizes[*slot as usize] = regions[label].size;
            *slot += 1;
            *slot - 1
        } else {
            regions[up].parent
        };
    }

    // regions[0].parent == 0 keeps unlabelled gates unlabelled without a branch.
    for label in labels.iter_mut() {
        *label = regions[*label as usize].parent;
    }
    (labels, sizes)
}

/// Merges two provisional labels' regions, returning the survivor. Always the
/// smaller root, so a region keeps the label minted at its first gate.
fn unite(regions: &mut [Provisional], a: i32, b: i32) -> i32 {
    let mut kept = root(regions, a);
    let mut merged = root(regions, b);
    if kept > merged {
        std::mem::swap(&mut kept, &mut merged);
    }
    regions[merged as usize].parent = kept;
    kept
}

/// Follows a provisional label to its root, halving the path on the way so the
/// next walk is shorter. Halving keeps every parent below its child, which the
/// resolution sweeps rely on.
fn root(regions: &mut [Provisional], mut label: i32) -> i32 {
    loop {
        let up = regions[label as usize].parent;
        if up == label {
            return label;
        }
        let above = regions[up as usize].parent;
        regions[label as usize].parent = above;
        label = above;
    }
}

/// Which velocity interval a gate falls in, numbered from 1, or 0 for none.
///
/// The intervals are disjoint, so two gates are in the same one exactly when
/// these agree — the only question the pass asks of a neighbour.
#[inline]
fn interval_id(value: f32, limits: &[f32]) -> u16 {
    if !value.is_finite() {
        return 0;
    }
    for (interval, pair) in limits.windows(2).enumerate() {
        if pair[0] <= value && value < pair[1] {
            // `interval_limits` caps the count far below u16, see MAX_EXTRA_INTERVALS.
            return interval as u16 + 1;
        }
    }
    0
}

#[derive(Clone, Copy, Default)]
struct EdgeAccum {
    count: u32,
    vel_sum: f64,
    nvel_sum: f64,
}

/// Accumulator keyed on a packed `(label, neighbour)` pair.
///
/// This is the hottest structure in the solver — a lookup for every adjacency
/// that crosses from one region into a lower-numbered one, so millions per
/// tilt. An ordered map spends ~log2(regions) tuple comparisons on each one;
/// this is one multiply and a short linear probe.
///
/// Iteration order differs from an ordered map's, which is fine: the only
/// consumer drains it and sorts by `(neighbour, label)`, and those keys are
/// unique, so the sort is a total order and the result is identical either way.
struct EdgeMap {
    /// `0` marks an empty slot. Real keys pack two labels that are both >= 1,
    /// so a real key can never be zero.
    keys: Vec<u64>,
    vals: Vec<EdgeAccum>,
    mask: usize,
    len: usize,
}

impl EdgeMap {
    fn with_capacity(slots: usize) -> Self {
        let n = slots.next_power_of_two().max(1024);
        Self {
            keys: vec![0; n],
            vals: vec![EdgeAccum::default(); n],
            mask: n - 1,
            len: 0,
        }
    }

    #[inline(always)]
    fn pack(label: i32, neighbor: i32) -> u64 {
        ((label as u32 as u64) << 32) | (neighbor as u32 as u64)
    }

    /// Fibonacci hashing: multiply by the golden-ratio constant and take the
    /// high bits, which mixes both packed halves into the slot index.
    #[inline(always)]
    fn slot(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & self.mask
    }

    #[inline]
    fn entry(&mut self, key: u64) -> &mut EdgeAccum {
        // Keep the load factor under 70%; linear probing degrades sharply above.
        if (self.len + 1) * 10 >= self.keys.len() * 7 {
            self.grow();
        }
        let mut i = self.slot(key);
        loop {
            let existing = self.keys[i];
            if existing == key {
                return &mut self.vals[i];
            }
            if existing == 0 {
                self.keys[i] = key;
                self.len += 1;
                return &mut self.vals[i];
            }
            i = (i + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let n = self.keys.len() * 2;
        let old_keys = std::mem::replace(&mut self.keys, vec![0; n]);
        let old_vals = std::mem::replace(&mut self.vals, vec![EdgeAccum::default(); n]);
        self.mask = n - 1;
        for (key, val) in old_keys.into_iter().zip(old_vals) {
            if key == 0 {
                continue;
            }
            let mut i = self.slot(key);
            while self.keys[i] != 0 {
                i = (i + 1) & self.mask;
            }
            self.keys[i] = key;
            self.vals[i] = val;
        }
    }

    /// Occupied entries as `((label, neighbour), accum)`, in unspecified order.
    fn into_entries(self) -> Vec<((i32, i32), EdgeAccum)> {
        let mut out = Vec::with_capacity(self.len);
        for (key, val) in self.keys.into_iter().zip(self.vals) {
            if key == 0 {
                continue;
            }
            out.push((((key >> 32) as u32 as i32, key as u32 as i32), val));
        }
        out
    }
}

#[derive(Clone, Copy)]
struct EdgeState {
    alpha: usize,
    beta: usize,
    sum_diff: f64,
    weight: i32,
}

fn edge_sum_and_count(
    labels: &[i32],
    observed: &[f32],
    rows: usize,
    gates: usize,
    wraps: bool,
    nyquist_interval: f32,
    nfeatures: usize,
) -> Vec<EdgeState> {
    // Regions border about twice as many others as there are regions, so this
    // usually lands one power of two clear of the 70% load factor; a field that
    // beats the estimate just grows the map once.
    let mut acc = EdgeMap::with_capacity(nfeatures * 3);

    // Stand-in for the ray beyond an open sweep's first and last row, so the
    // hot loop can read the adjacent ray without first testing whether one
    // exists. All-zero reads as "no data", which is what the gap scan concludes
    // at an open edge anyway, and the velocities are never reached because a
    // zero label short-circuits first.
    let absent_labels = vec![0i32; gates];
    let absent_vel = vec![0f32; gates];

    // Nearest labelled ray at or above each gate, as a *signed* ray index so a
    // closed sweep can seed it with the rays that precede ray 0. Every labelled
    // gate writes its own ray here, which is all the backward gap scan ever
    // needs, so looking back costs a subtraction instead of a walk.
    let mut prev_ray = vec![NO_PREVIOUS_RAY; gates];
    if wraps {
        // A gap reaches back at most SKIP_BETWEEN_RAYS + 1 rays, and a sweep
        // shorter than that folds onto itself, so one turn is always enough.
        let seed = (SKIP_BETWEEN_RAYS + 1).min(rows);
        for row in rows - seed..rows {
            for (gate, &label) in labels[row * gates..][..gates].iter().enumerate() {
                if label != 0 {
                    prev_ray[gate] = row as isize - rows as isize;
                }
            }
        }
    }

    for row in 0..rows {
        let base = row * gates;
        let cur = &labels[base..base + gates];
        let cur_vel = &observed[base..base + gates];
        // Row-major means the two rays either side and this one are the only
        // memory the fast path touches, and all three are walked in step.
        let (up_labels, up_vel) = ray(labels, observed, gates, base, -1, wraps);
        let (down_labels, down_vel) = ray(labels, observed, gates, base, 1, wraps);
        let up_labels = up_labels.unwrap_or(&absent_labels);
        let up_vel = up_vel.unwrap_or(&absent_vel);
        let down_labels = down_labels.unwrap_or(&absent_labels);
        let down_vel = down_vel.unwrap_or(&absent_vel);

        // Nearest labelled gate at or before the one being visited, the ray-wise
        // twin of `prev_ray`.
        let mut prev_gate = None;

        for gate in 0..gates {
            let label = cur[gate];
            if label == 0 {
                continue;
            }
            let vel = cur_vel[gate];

            // Each direction takes the adjacent cell when it carries data - by
            // far the common case, since regions are contiguous. The two
            // backward directions then fall back to what has already been seen;
            // only the two forward ones have to scan the gap.
            let neighbor = up_labels[gate];
            if neighbor != 0 {
                add_directed_edge(&mut acc, label, neighbor, vel, up_vel[gate]);
            } else if row as isize - prev_ray[gate] <= MAX_RAY_REACH {
                let n = prev_ray[gate].rem_euclid(rows as isize) as usize * gates + gate;
                add_directed_edge(&mut acc, label, labels[n], vel, observed[n]);
            }

            let neighbor = down_labels[gate];
            if neighbor != 0 {
                add_directed_edge(&mut acc, label, neighbor, vel, down_vel[gate]);
            } else if let Some(n) = next_ray_across_gap(labels, gates, base + gate, wraps) {
                add_directed_edge(&mut acc, label, labels[n], vel, observed[n]);
            }

            if gate > 0 && cur[gate - 1] != 0 {
                add_directed_edge(&mut acc, label, cur[gate - 1], vel, cur_vel[gate - 1]);
            } else if let Some(n) = prev_gate.filter(|n| gate - n <= SKIP_ALONG_RAY + 1) {
                add_directed_edge(&mut acc, label, cur[n], vel, cur_vel[n]);
            }

            if gate + 1 < gates && cur[gate + 1] != 0 {
                add_directed_edge(&mut acc, label, cur[gate + 1], vel, cur_vel[gate + 1]);
            } else if let Some(n) = next_gate_across_gap(cur, gate) {
                add_directed_edge(&mut acc, label, cur[n], vel, cur_vel[n]);
            }

            prev_gate = Some(gate);
            prev_ray[gate] = row as isize;
        }
    }

    let mut edges: Vec<EdgeState> = acc
        .into_entries()
        .into_iter()
        .filter(|(_, edge)| edge.count != 0)
        .map(|((label, neighbor), edge)| EdgeState {
            alpha: label as usize,
            beta: neighbor as usize,
            sum_diff: (edge.vel_sum - edge.nvel_sum) / f64::from(nyquist_interval),
            weight: edge.count as i32,
        })
        .collect();
    sort_edges_by_endpoints(&mut edges);
    edges
}

/// Order the edge list by `(beta, alpha)`, restoring a deterministic sequence
/// regardless of how the accumulator laid its entries out. The keys are unique,
/// so this is a total order.
///
/// Two stable counting passes rather than a comparison sort: both fields are
/// dense region labels, so the histograms are small and cheap, and each edge
/// moves once at the end instead of on every partition pass. Least significant
/// key (`alpha`) first, as an LSD radix sort requires.
fn sort_edges_by_endpoints(edges: &mut Vec<EdgeState>) {
    // Every surviving edge has `alpha > beta`, so the largest alpha bounds both
    // fields and one histogram buffer serves both passes.
    let Some(bound) = edges.iter().map(|edge| edge.alpha).max() else {
        return;
    };
    let mut counts = vec![0u32; bound + 1];
    let mut by_alpha = vec![0u32; edges.len()];
    let mut by_beta = vec![0u32; edges.len()];

    histogram(&mut counts, edges.iter().map(|edge| edge.alpha));
    for (index, edge) in edges.iter().enumerate() {
        by_alpha[take_slot(&mut counts, edge.alpha)] = index as u32;
    }

    histogram(&mut counts, edges.iter().map(|edge| edge.beta));
    for &index in &by_alpha {
        by_beta[take_slot(&mut counts, edges[index as usize].beta)] = index;
    }

    *edges = by_beta.iter().map(|&index| edges[index as usize]).collect();
}

/// Rewrites `counts` as the starting output offset of each key.
fn histogram(counts: &mut [u32], keys: impl Iterator<Item = usize>) {
    counts.fill(0);
    for key in keys {
        counts[key] += 1;
    }
    let mut running = 0;
    for slot in counts.iter_mut() {
        let occurrences = *slot;
        *slot = running;
        running += occurrences;
    }
}

/// The next free output slot for `key`, advancing it. Stable as long as the
/// caller visits the input in order.
#[inline]
fn take_slot(counts: &mut [u32], key: usize) -> usize {
    let slot = counts[key] as usize;
    counts[key] += 1;
    slot
}

#[inline]
fn add_directed_edge(acc: &mut EdgeMap, label: i32, neighbor: i32, vel: f32, nvel: f32) {
    // Adjacency is symmetric - every pair is seen once from each side - and the
    // drain above keeps only the `label > neighbour` half, so accumulating the
    // mirror is pure waste. Dropping it here costs nothing: the surviving half
    // sees the same contributions in the same order, so its sums are bit-identical.
    if neighbor >= label || neighbor == 0 {
        return;
    }
    let entry = acc.entry(EdgeMap::pack(label, neighbor));
    entry.count += 1;
    entry.vel_sum += f64::from(vel);
    entry.nvel_sum += f64::from(nvel);
}

/// The ray `step` away from the one starting at `base`, as label and velocity
/// slices, or `None` where an open sweep has no such ray.
fn ray<'a>(
    labels: &'a [i32],
    observed: &'a [f32],
    gates: usize,
    base: usize,
    step: isize,
    wraps: bool,
) -> (Option<&'a [i32]>, Option<&'a [f32]>) {
    let total = labels.len();
    let next = if step < 0 {
        match (base >= gates, wraps) {
            (true, _) => base - gates,
            (false, true) => total - gates,
            (false, false) => return (None, None),
        }
    } else {
        match (base + gates < total, wraps) {
            (true, _) => base + gates,
            (false, true) => 0,
            (false, false) => return (None, None),
        }
    };
    (
        Some(&labels[next..next + gates]),
        Some(&observed[next..next + gates]),
    )
}

/// Flat index of the next labelled gate across a no-data gap on a following
/// ray, at most [`MAX_RAY_REACH`] rays on.
///
/// Walking flat indices rather than `(row, gate)` pairs keeps the scan
/// multiply-free; the reachability is the original's - the adjacent ray, then
/// up to `SKIP_BETWEEN_RAYS` more - and an open sweep's last ray ends the walk
/// rather than folding it round to the first.
fn next_ray_across_gap(labels: &[i32], gates: usize, idx: usize, wraps: bool) -> Option<usize> {
    let total = labels.len();
    let mut check = idx;
    for _ in 0..=SKIP_BETWEEN_RAYS {
        check += gates;
        if check >= total {
            if !wraps {
                return None;
            }
            check -= total;
        }
        if labels[check] != 0 {
            return Some(check);
        }
    }
    None
}

/// Index of the next labelled gate across a no-data gap further out along
/// `ray`, at most `SKIP_ALONG_RAY + 1` gates on. A ray never wraps: it ends at
/// the radar and at maximum range.
fn next_gate_across_gap(ray: &[i32], gate: usize) -> Option<usize> {
    let last = (gate + SKIP_ALONG_RAY + 1).min(ray.len() - 1);
    ray[gate + 1..=last]
        .iter()
        .position(|&l| l != 0)
        .map(|n| gate + 1 + n)
}

struct RegionTracker {
    node_size: Vec<u32>,
    original_region_sizes: Vec<u32>,
    /// The regions a node owns, as an intrusive singly-linked chain: node `n`
    /// runs from `chain_head[n]` along `next_region` until [`RegionTracker::END`].
    /// A node starts out owning the one region it was made from, and nodes and
    /// regions share an index space, so no per-node vector is needed — merging
    /// is a splice of two chains rather than a copy, and building the tracker
    /// costs one allocation instead of one per region.
    ///
    /// The only reader adds a constant to each region's unwrap count, so chain
    /// order is not observable.
    next_region: Vec<u32>,
    chain_head: Vec<u32>,
    chain_tail: Vec<u32>,
    unwrap_number: Vec<i32>,
}

impl RegionTracker {
    const END: u32 = u32::MAX;

    fn new(region_sizes: Vec<u32>) -> Self {
        let nregions = region_sizes.len();
        Self {
            node_size: region_sizes.clone(),
            original_region_sizes: region_sizes,
            next_region: vec![Self::END; nregions],
            chain_head: (0..nregions as u32).collect(),
            chain_tail: (0..nregions as u32).collect(),
            unwrap_number: vec![0; nregions],
        }
    }

    fn merge_nodes(&mut self, node_a: usize, node_b: usize) {
        let tail_a = self.chain_tail[node_a] as usize;
        self.next_region[tail_a] = self.chain_head[node_b];
        self.chain_tail[node_a] = self.chain_tail[node_b];
        self.node_size[node_a] += self.node_size[node_b];
        self.node_size[node_b] = 0;
    }

    fn unwrap_node(&mut self, node: usize, nwrap: i32) {
        if nwrap == 0 {
            return;
        }
        let mut region = self.chain_head[node];
        while region != Self::END {
            self.unwrap_number[region as usize] += nwrap;
            region = self.next_region[region as usize];
        }
    }

    fn get_node_size(&self, node: usize) -> u32 {
        self.node_size[node]
    }
}

/// The base node's neighbour map: `common[n].epoch == epoch` means node `n`
/// already borders the current base node along edge `common[n].edge`.
///
/// `epoch` replaces a `bool` flag whose reset used to clear the whole array on
/// every base-node change; bumping the counter is the same invalidation in
/// O(1). Both halves live in one 8-byte cell because the rebuild below touches
/// them together, once per edge, at a random index — two arrays meant two cache
/// misses for what fits in one.
#[derive(Clone, Copy)]
struct Common {
    epoch: u32,
    edge: u32,
}

struct EdgeTracker {
    edges: Vec<EdgeState>,
    edges_in_node: Vec<Vec<u32>>,
    /// Max-heap of [`EdgeTracker::heap_key`]s, so the strongest edge pops first
    /// and ties resolve to the lowest index — exactly the edge the original
    /// full-list scan selected. Entries are never removed on update; a pop whose
    /// recorded weight no longer matches the edge is stale and is discarded
    /// ([`EdgeTracker::pop_edge`]).
    heap: BinaryHeap<u64>,
    common: Vec<Common>,
    /// Starts at 1 so the `epoch: 0` cells above read as "not a neighbour", and
    /// rises once per base-node change — at most once per merge, so it cannot
    /// come close to wrapping.
    epoch: u32,
    last_base_node: Option<usize>,
}

impl EdgeTracker {
    fn new(edges: Vec<EdgeState>, nnodes: usize) -> Self {
        let mut edges_in_node = vec![Vec::new(); nnodes];
        let mut heap = BinaryHeap::with_capacity(edges.len());
        for (edge_index, edge) in edges.iter().enumerate() {
            edges_in_node[edge.alpha].push(edge_index as u32);
            edges_in_node[edge.beta].push(edge_index as u32);
            heap.push(Self::heap_key(edge.weight, edge_index));
        }
        Self {
            edges,
            edges_in_node,
            heap,
            common: vec![Common { epoch: 0, edge: 0 }; nnodes],
            epoch: 1,
            last_base_node: None,
        }
    }

    /// Weight and edge index in one `u64`, ordered exactly as the
    /// `(weight, Reverse(index))` pair it replaces: weight in the high half, and
    /// the index complemented in the low half so the smaller index compares
    /// greater. One word rather than two keeps twice as much of the heap in
    /// cache, and every sift step becomes a single integer comparison.
    ///
    /// Only live weights are ever pushed, and those are gate counts that fit an
    /// `i32` by construction, so the high half never sees a sign bit.
    #[inline]
    fn heap_key(weight: i32, index: usize) -> u64 {
        debug_assert!(weight > 0 && index <= u32::MAX as usize);
        ((weight as u32 as u64) << 32) | (u32::MAX - index as u32) as u64
    }

    /// The strongest surviving edge, or `None` once every edge is retired.
    ///
    /// Selection is identical to the original linear scan (max weight, lowest
    /// index on a tie); only the search is different. Weights are monotone —
    /// they rise in [`EdgeTracker::combine_edges`] and otherwise drop to the
    /// retired sentinel, which is negative and so matches no key — meaning a
    /// stale entry can never be mistaken for a live one and no edge is ever
    /// visited twice at the same weight.
    fn pop_edge(&mut self) -> Option<(usize, usize, f64, usize)> {
        while let Some(key) = self.heap.pop() {
            let weight = (key >> 32) as i32;
            let index = (u32::MAX - key as u32) as usize;
            let edge = &self.edges[index];
            if edge.weight != weight {
                continue;
            }
            return Some((
                edge.alpha,
                edge.beta,
                edge.sum_diff / f64::from(weight),
                index,
            ));
        }
        None
    }

    /// Fold `merge_node` into `base_node`, `foo_edge` being the edge between
    /// them that was just selected.
    ///
    /// Retired edges are left sitting in their nodes' lists rather than spliced
    /// out. Splicing meant a `Vec::remove` — a scan for the edge plus a memmove
    /// of the tail — and the base node is the sweep's growing blob, so on a busy
    /// tilt that one line walked tens of millions of entries per sweep. Every
    /// full traversal below drops the retirees it passes, so a list never
    /// carries more stale entries than the merges that created them, and the
    /// live entries keep exactly the order the splicing version left them in.
    fn merge_nodes(&mut self, base_node: usize, merge_node: usize, foo_edge: usize) {
        self.edges[foo_edge].weight = RETIRED;
        self.common[merge_node].epoch = 0;

        if self.last_base_node != Some(base_node) {
            self.epoch += 1;
            // Moved out and back so the walk can take `&mut self`, and compacted
            // in place: this is the one pass that sees all of base's edges.
            let mut edges_in_base = std::mem::take(&mut self.edges_in_node[base_node]);
            let mut kept = 0;
            for index in 0..edges_in_base.len() {
                let edge_num = edges_in_base[index] as usize;
                if self.edges[edge_num].weight < 0 {
                    continue;
                }
                if self.edges[edge_num].beta == base_node {
                    self.reverse_edge_direction(edge_num);
                }
                debug_assert_eq!(self.edges[edge_num].alpha, base_node);
                let neighbor = self.edges[edge_num].beta;
                self.common[neighbor] = Common {
                    epoch: self.epoch,
                    edge: edge_num as u32,
                };
                edges_in_base[kept] = edge_num as u32;
                kept += 1;
            }
            edges_in_base.truncate(kept);
            self.edges_in_node[base_node] = edges_in_base;
        }

        // Emptying merge's list here is what the old trailing `mem::take` did;
        // taking it up front also gives the loop a snapshot to walk while it
        // appends the survivors to base.
        let edges_in_merge = std::mem::take(&mut self.edges_in_node[merge_node]);
        for &edge_num in &edges_in_merge {
            let edge_num = edge_num as usize;
            if self.edges[edge_num].weight < 0 {
                continue;
            }
            if self.edges[edge_num].beta == merge_node {
                self.reverse_edge_direction(edge_num);
            }
            debug_assert_eq!(self.edges[edge_num].alpha, merge_node);
            self.edges[edge_num].alpha = base_node;
            let neighbor = self.edges[edge_num].beta;
            if self.common[neighbor].epoch == self.epoch {
                let base_edge_num = self.common[neighbor].edge as usize;
                self.combine_edges(base_edge_num, edge_num);
            } else {
                self.common[neighbor] = Common {
                    epoch: self.epoch,
                    edge: edge_num as u32,
                };
                self.edges_in_node[base_node].push(edge_num as u32);
            }
        }
        self.last_base_node = Some(base_node);
    }

    fn combine_edges(&mut self, base_edge: usize, merge_edge: usize) {
        self.edges[base_edge].weight += self.edges[merge_edge].weight;
        self.edges[merge_edge].weight = RETIRED;
        self.edges[base_edge].sum_diff += self.edges[merge_edge].sum_diff;
        // The only place an edge's weight rises, so the only place the heap
        // needs a fresh entry. The superseded one is discarded when it pops.
        self.heap
            .push(Self::heap_key(self.edges[base_edge].weight, base_edge));
    }

    fn reverse_edge_direction(&mut self, edge: usize) {
        let edge = &mut self.edges[edge];
        std::mem::swap(&mut edge.alpha, &mut edge.beta);
        edge.sum_diff = -edge.sum_diff;
    }

    fn unwrap_node(&mut self, node: usize, nwrap: i32) {
        if nwrap == 0 {
            return;
        }
        for &edge_index in &self.edges_in_node[node] {
            let edge = &mut self.edges[edge_index as usize];
            if edge.weight < 0 {
                continue;
            }
            let delta = edge.weight * nwrap;
            if node == edge.alpha {
                edge.sum_diff += f64::from(delta);
            } else {
                debug_assert_eq!(node, edge.beta);
                edge.sum_diff -= f64::from(delta);
            }
        }
    }
}

fn round_ties_even_to_i32(value: f64) -> i32 {
    value.round_ties_even() as i32
}
#[cfg(test)]
mod tests {
    use super::*;

    /// In-Nyquist data must produce exactly Py-ART's default split: four limits
    /// spanning ±Nyquist, with no extra intervals on either side.
    #[test]
    fn in_nyquist_data_uses_the_default_splits() {
        let limits = interval_limits(24.0, &[-20.0, 0.0, 18.0, f32::NAN]);
        assert_eq!(limits.len(), INTERVAL_SPLITS + 1);
        assert!((limits[0] + 24.0).abs() < 1e-4, "{limits:?}");
        assert!((limits[limits.len() - 1] - 24.0).abs() < 1e-4, "{limits:?}");
    }

    /// Data outside ±Nyquist adds intervals so the extra branches get their own
    /// regions.
    ///
    /// Note which bound each term widens: upstream Py-ART derives `add_start`
    /// (which extends the NEGATIVE end) from the MAXIMUM velocity, and
    /// `add_end` from the minimum — the two are crossed. This port mirrors that
    /// deliberately, so the limits are not guaranteed to span the observed
    /// range. Raw radar data is inside ±Nyquist by construction, so this path
    /// only engages on pre-processed input.
    #[test]
    fn out_of_nyquist_data_adds_intervals() {
        let limits = interval_limits(24.0, &[-60.0, 0.0, 55.0]);
        assert_eq!(limits.len(), INTERVAL_SPLITS + 1 + 2 + 3);
        // add_start = ceil((55 - 24) / 16) = 2  ->  start = -24 - 2*16 = -56
        // add_end   = ceil((60 - 24) / 16) = 3  ->  end   =  24 + 3*16 =  72
        assert!((limits[0] + 56.0).abs() < 1e-3, "{:?}", limits[0]);
        let last = limits[limits.len() - 1];
        assert!((last - 72.0).abs() < 1e-3, "{last:?}");
    }

    /// Regression: a corrupt feed carrying non-physical velocities used to
    /// saturate both interval counts at `i32::MAX`, overflowing their sum (a
    /// debug panic, and a multi-gigabyte allocation in release). Python's
    /// arbitrary-precision `int` cannot hit this, so it is ours to guard.
    #[test]
    fn non_physical_velocities_stay_bounded() {
        for extreme in [1e30f32, -1e30, f32::MAX, -f32::MAX] {
            let limits = interval_limits(24.0, &[extreme, 0.0]);
            assert!(
                limits.len() <= (INTERVAL_SPLITS as i32 + 1 + 2 * MAX_EXTRA_INTERVALS) as usize,
                "{extreme} produced {} limits",
                limits.len()
            );
        }
    }

    /// A vanishingly small Nyquist makes the interval width tiny, which is the
    /// other route to an unbounded limit count.
    #[test]
    fn tiny_nyquist_stays_bounded() {
        let limits = interval_limits(1e-6, &[-100.0, 100.0]);
        assert!(limits.len() <= (INTERVAL_SPLITS as i32 + 1 + 2 * MAX_EXTRA_INTERVALS) as usize);
    }

    /// The bit-pattern reduction must agree with the obvious float one, NaN and
    /// infinity exclusions included.
    #[test]
    fn finite_extent_matches_a_float_reduction() {
        let field = [
            f32::NAN,
            -0.0,
            0.0,
            f32::INFINITY,
            -18.5,
            f32::NEG_INFINITY,
            1e30,
            -1e30,
            24.0,
            f32::MIN_POSITIVE,
        ];
        let (low, high) = finite_extent(&field);
        assert_eq!(low, -1e30);
        assert_eq!(high, 1e30);

        let (low, high) = finite_extent(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        assert_eq!((low, high), (f32::INFINITY, f32::NEG_INFINITY));
        assert_eq!(finite_extent(&[]), (f32::INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn all_no_data_leaves_the_default_splits() {
        let limits = interval_limits(24.0, &[f32::NAN, f32::NAN]);
        assert_eq!(limits.len(), INTERVAL_SPLITS + 1);
    }

    #[test]
    fn automatic_nyquist_requires_exactly_one_interval() {
        assert_eq!(uniform_nyquist(&[25.0, 25.0, 25.0]), Some(25.0));

        let adjacent_float = f32::from_bits(25.0f32.to_bits() + 1);
        assert_eq!(uniform_nyquist(&[25.0, adjacent_float, 25.0]), None);
        assert_eq!(uniform_nyquist(&[25.0, f32::NAN, 25.0]), None);
    }

    #[test]
    fn automatic_mixed_nyquist_abstention_is_reported_per_finite_gate() {
        let rows = 4;
        let gates = 4;
        let mut observed = vec![0.0; rows * gates];
        observed[3] = f32::NAN;
        let mut nyquist = vec![25.0; rows];
        nyquist[2] = f32::from_bits(25.0f32.to_bits() + 1);
        let azimuth = [0.0, 0.5, 1.0, 1.5];

        let result = dealias_sweep_rift(
            &observed,
            &nyquist,
            rows,
            gates,
            &azimuth,
            &RiftContext::default(),
            RiftOptions {
                first_gate_m: 0.0,
                gate_spacing_m: 250.0,
                automatic_single_sweep: true,
                ..RiftOptions::default()
            },
        )
        .expect("mixed Nyquist is a safe abstention, not an input error");

        let expected = RIFT_REASON_NYQUIST_TRANSITION | RIFT_REASON_ABSTAINED;
        for (&value, &reason) in observed.iter().zip(&result.reasons) {
            assert_eq!(reason, if value.is_finite() { expected } else { 0 });
        }
        assert_eq!(result.stats.gates_refined, 0);
    }
}
