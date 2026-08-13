//! Conservative local fold correction for compact, intense velocity couplets.
//!
//! The region merge intentionally favors large, globally coherent branches.
//! A compact tornado couplet can need the opposite fold on only part of one of
//! those large regions.  This pass first requires a strong, correctly oriented
//! azimuthal signature, then minimizes total variation in a small gate window
//! with a binary graph cut. It is deliberately authorization-gated: ordinary
//! shear and noisy seams should normally stop before model fitting, and every
//! rejected candidate leaves the original fold field unchanged.

use std::collections::VecDeque;

const SHEAR_FRACTION: f32 = 0.9;
const SIGNED_SEED_FRACTION: f32 = 1.75;
const LOBE_FRACTION: f32 = 0.35;
const MINIMUM_SHEAR_ENDPOINTS: usize = 40;
const MINIMUM_SIGNED_SEEDS: usize = 3;
const MINIMUM_LOBE_GATES: usize = 15;
const DETECTOR_DILATION: usize = 2;
const ROI_PADDING: usize = 6;
const VORTEX_CONTEXT_PADDING: usize = 2;
const SEED_CLUSTER_RADIUS: usize = 3;
const SUPPORT_RAY_BEFORE: usize = 3;
const SUPPORT_RAY_AFTER: usize = 4;
const SUPPORT_GATE_RADIUS: usize = 6;
const MAXIMUM_FORWARD_AZIMUTH_GAP: f32 = 2.0;
const SHIFT_PENALTY_FRACTION: f32 = 0.01;
const MAXIMUM_CANDIDATE_AREA: usize = 4096;
const MAXIMUM_CANDIDATES: usize = 4;
const AUTOMATIC_CONFIDENCE: u8 = 220;
const PIN_CAPACITY: f64 = 1.0e12;
const FLOW_EPSILON: f64 = 1.0e-10;

pub(super) const REASON_SIGNED_SHEAR: u32 = 1 << 0;
pub(super) const REASON_LOCAL_PROPOSAL: u32 = 1 << 1;
pub(super) const REASON_FUSION_ACCEPTED: u32 = 1 << 2;
pub(super) const ABSTAINED_BUDGET: u32 = 1 << 0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Diagnostics {
    pub candidates_detected: u32,
    pub candidates_authorized: u32,
    pub components_accepted: u32,
    pub gates_refined: u32,
    pub reason_flags: u32,
    pub abstain_flags: u32,
}

#[derive(Clone, Copy, Debug)]
struct Rectangle {
    row_start: usize,
    row_end: usize,
    gate_start: usize,
    gate_end: usize,
}

impl Rectangle {
    fn overlaps(self, other: Self) -> bool {
        self.row_start < other.row_end
            && other.row_start < self.row_end
            && self.gate_start < other.gate_end
            && other.gate_start < self.gate_end
    }
}

#[derive(Clone, Copy)]
struct FlowEdge {
    target: usize,
    reverse: usize,
    capacity: f64,
}

struct Dinic {
    graph: Vec<Vec<FlowEdge>>,
}

impl Dinic {
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

    fn global_relabel(&self, source: usize, sink: usize, height: &mut [usize]) {
        let nodes = self.graph.len();
        height.fill(nodes + 1);
        height[sink] = 0;
        let mut queue = VecDeque::from([sink]);
        while let Some(node) = queue.pop_front() {
            for edge in &self.graph[node] {
                let neighbor = edge.target;
                let reverse_capacity = self.graph[neighbor][edge.reverse].capacity;
                if neighbor != source
                    && height[neighbor] == nodes + 1
                    && reverse_capacity > FLOW_EPSILON
                {
                    height[neighbor] = height[node] + 1;
                    queue.push_back(neighbor);
                }
            }
        }
        height[source] = nodes;
    }

    /// Deterministic FIFO push-relabel. Unlike recursive Dinic DFS this cannot
    /// exhaust the small WebAssembly call stack on a large but still in-budget
    /// RIFT ROI.
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
        // Starting from exact residual distances to the sink avoids the
        // one-level-at-a-time wave that dominates large grid cuts.
        self.global_relabel(source, sink, &mut height);

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

// The arguments intentionally mirror the flat solver inputs so this hot path
// does not allocate or build an intermediate sweep object.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine_folds(
    observed: &[f32],
    nyquist: f32,
    azimuths: &[f32],
    rows: usize,
    gates: usize,
    folds: &mut [i32],
    options: super::RiftOptions,
    confidence: &mut [u8],
    reasons: &mut [u16],
) -> Diagnostics {
    let mut diagnostics = Diagnostics::default();
    if rows < 2
        || gates < 2
        || azimuths.len() < rows
        || folds.len() != observed.len()
        || confidence.len() != observed.len()
        || reasons.len() != observed.len()
    {
        return diagnostics;
    }
    let interval = 2.0 * nyquist;
    let supports =
        signed_couplet_supports(observed, folds, interval, nyquist, azimuths, rows, gates);
    if supports.is_empty() {
        return diagnostics;
    }
    diagnostics.reason_flags |= REASON_SIGNED_SHEAR;
    for support in &supports {
        for row in support.row_start..support.row_end {
            for gate in support.gate_start..support.gate_end {
                reasons[row * gates + gate] |= super::RIFT_REASON_BRANCH_UNSTABLE;
            }
        }
    }
    let mut solved: Vec<f32> = observed
        .iter()
        .zip(folds.iter())
        .map(|(&value, &fold)| {
            if value.is_finite() {
                value + interval * fold as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    let candidates = shear_candidates(&solved, nyquist, rows, gates);
    diagnostics.candidates_detected = u32::try_from(candidates.len()).unwrap_or(u32::MAX);
    let roi_limit = usize::from(options.max_rois).min(MAXIMUM_CANDIDATES);
    let per_roi_budget = (options.max_roi_gates as usize)
        .min(MAXIMUM_CANDIDATE_AREA)
        .min(super::rift_vortex::MAX_ROI_GATES);
    let total_budget = if options.max_total_roi_gates == 0 {
        observed
            .len()
            .min(MAXIMUM_CANDIDATES.saturating_mul(MAXIMUM_CANDIDATE_AREA))
    } else {
        options.max_total_roi_gates as usize
    };
    if candidates.len() > roi_limit {
        diagnostics.abstain_flags |= ABSTAINED_BUDGET;
        for candidate in candidates.iter().skip(roi_limit) {
            mark_budget_abstention(*candidate, gates, reasons);
        }
    }
    let mut budget_used = 0usize;
    for candidate in candidates.into_iter().take(roi_limit) {
        if !supports.iter().any(|support| candidate.overlaps(*support)) {
            continue;
        }
        let Some(vortex_roi) = expand_vortex_context(candidate, rows, gates, per_roi_budget) else {
            diagnostics.abstain_flags |= ABSTAINED_BUDGET;
            mark_budget_abstention(candidate, gates, reasons);
            continue;
        };
        let area = rectangle_area(vortex_roi);
        if budget_used.saturating_add(area) > total_budget {
            diagnostics.abstain_flags |= ABSTAINED_BUDGET;
            mark_budget_abstention(vortex_roi, gates, reasons);
            continue;
        }
        budget_used += area;
        diagnostics.candidates_authorized += 1;
        diagnostics.reason_flags |= REASON_LOCAL_PROPOSAL;
        // The signed increasing-azimuth trigger is specifically cyclonic:
        // it authorizes the negative local branch. The vortex fit still has
        // to select that branch independently before any result is applied.
        let (selected, energy_delta) = graph_cut(&solved, nyquist, gates, candidate, -1);
        if energy_delta >= -FLOW_EPSILON {
            continue;
        }
        let roi_rows = vortex_roi.row_end - vortex_roi.row_start;
        let roi_gates = vortex_roi.gate_end - vortex_roi.gate_start;
        let mut roi_observed = Vec::with_capacity(roi_rows * roi_gates);
        let mut roi_baseline = Vec::with_capacity(roi_rows * roi_gates);
        let mut roi_local_mask = Vec::with_capacity(roi_rows * roi_gates);
        let mut roi_azimuth = Vec::with_capacity(roi_rows);
        let mut roi_nyquist = Vec::with_capacity(roi_rows);
        let mut selected_mask = vec![false; observed.len()];
        for &index in &selected {
            selected_mask[index] = true;
        }
        for (row, &azimuth) in azimuths
            .iter()
            .enumerate()
            .take(vortex_roi.row_end)
            .skip(vortex_roi.row_start)
        {
            roi_azimuth.push(azimuth);
            roi_nyquist.push(nyquist);
            for gate in vortex_roi.gate_start..vortex_roi.gate_end {
                let index = row * gates + gate;
                roi_observed.push(observed[index]);
                roi_baseline.push(solved[index]);
                roi_local_mask.push(u8::from(selected_mask[index]));
            }
        }
        let vortex = super::rift_vortex::propose_and_fuse(&super::rift_vortex::Input {
            observed: &roi_observed,
            baseline: &roi_baseline,
            local_cut_mask: &roi_local_mask,
            azimuth_deg: &roi_azimuth,
            nyquist_mps: &roi_nyquist,
            rows: roi_rows,
            gates: roi_gates,
            first_gate_m: options.first_gate_m
                + vortex_roi.gate_start as f32 * options.gate_spacing_m,
            gate_spacing_m: options.gate_spacing_m,
        });
        if !vortex.accepted {
            for row in vortex_roi.row_start..vortex_roi.row_end {
                for gate in vortex_roi.gate_start..vortex_roi.gate_end {
                    let local =
                        (row - vortex_roi.row_start) * roi_gates + gate - vortex_roi.gate_start;
                    if roi_local_mask[local] != 0 {
                        let index = row * gates + gate;
                        reasons[index] |= super::RIFT_REASON_BRANCH_UNSTABLE
                            | super::RIFT_REASON_VORTEX_PROPOSAL
                            | super::RIFT_REASON_ABSTAINED;
                    }
                }
            }
            continue;
        }
        if AUTOMATIC_CONFIDENCE < options.min_confidence {
            for row in vortex_roi.row_start..vortex_roi.row_end {
                for gate in vortex_roi.gate_start..vortex_roi.gate_end {
                    let local =
                        (row - vortex_roi.row_start) * roi_gates + gate - vortex_roi.gate_start;
                    if vortex.accepted_mask[local] != 0 {
                        let index = row * gates + gate;
                        reasons[index] |= super::RIFT_REASON_BRANCH_UNSTABLE
                            | super::RIFT_REASON_VORTEX_PROPOSAL
                            | super::RIFT_REASON_ABSTAINED;
                    }
                }
            }
            continue;
        }
        let accepted = vortex
            .accepted_mask
            .iter()
            .filter(|&&accepted| accepted != 0)
            .count();
        diagnostics.components_accepted += 1;
        diagnostics.gates_refined = diagnostics
            .gates_refined
            .saturating_add(u32::try_from(accepted).unwrap_or(u32::MAX));
        diagnostics.reason_flags |= REASON_FUSION_ACCEPTED;
        for row in vortex_roi.row_start..vortex_roi.row_end {
            for gate in vortex_roi.gate_start..vortex_roi.gate_end {
                let local = (row - vortex_roi.row_start) * roi_gates + gate - vortex_roi.gate_start;
                if vortex.accepted_mask[local] == 0 {
                    continue;
                }
                let index = row * gates + gate;
                let n = nyquist;
                folds[index] = ((vortex.velocity[local] - observed[index]) / (2.0 * n))
                    .round_ties_even() as i32;
                solved[index] = vortex.velocity[local];
                confidence[index] = confidence[index].max(AUTOMATIC_CONFIDENCE);
                reasons[index] |= super::RIFT_REASON_BRANCH_UNSTABLE
                    | super::RIFT_REASON_VORTEX_PROPOSAL
                    | super::RIFT_REASON_FUSION_ACCEPTED;
            }
        }
    }
    diagnostics
}

fn rectangle_area(rectangle: Rectangle) -> usize {
    (rectangle.row_end - rectangle.row_start)
        .saturating_mul(rectangle.gate_end - rectangle.gate_start)
}

fn mark_budget_abstention(candidate: Rectangle, gates: usize, reasons: &mut [u16]) {
    for row in candidate.row_start..candidate.row_end {
        for gate in candidate.gate_start..candidate.gate_end {
            let index = row * gates + gate;
            reasons[index] |= super::RIFT_REASON_BUDGET_EXCEEDED | super::RIFT_REASON_ABSTAINED;
        }
    }
}

fn expand_vortex_context(
    candidate: Rectangle,
    rows: usize,
    gates: usize,
    max_area: usize,
) -> Option<Rectangle> {
    for padding in (0..=VORTEX_CONTEXT_PADDING).rev() {
        let expanded = Rectangle {
            row_start: candidate.row_start.saturating_sub(padding),
            row_end: (candidate.row_end + padding).min(rows),
            gate_start: candidate.gate_start.saturating_sub(padding),
            gate_end: (candidate.gate_end + padding).min(gates),
        };
        if rectangle_area(expanded) <= max_area {
            return Some(expanded);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(super) fn refine_reference(
    observed: &[f32],
    nyquist: &[f32],
    rows: usize,
    gates: usize,
    folds: &mut [i32],
    reference: &[f32],
    quality: Option<&[u8]>,
    kind: super::ReferenceKind,
    options: super::RiftOptions,
    conflicts: &[bool],
    confidence: &mut [u8],
    reasons: &mut [u16],
) -> Diagnostics {
    let mut diagnostics = Diagnostics::default();
    let total = observed.len();
    let anchor_reason = match kind {
        super::ReferenceKind::Caller => super::RIFT_REASON_CALLER_ANCHOR,
        super::ReferenceKind::Temporal => super::RIFT_REASON_TEMPORAL_ANCHOR,
        super::ReferenceKind::Vertical => super::RIFT_REASON_VERTICAL_ANCHOR,
        super::ReferenceKind::Environmental => super::RIFT_REASON_ENVIRONMENTAL_ANCHOR,
    };
    let mut proposal = folds.to_vec();
    let mut changed = vec![false; total];
    for (row, &n) in nyquist.iter().take(rows).enumerate() {
        if !n.is_finite() || n <= 0.0 {
            continue;
        }
        let interval = 2.0 * n;
        for gate in 0..gates {
            let index = row * gates + gate;
            if conflicts.get(index).copied().unwrap_or(true) {
                continue;
            }
            let observed_value = observed[index];
            let reference_value = reference[index];
            if !observed_value.is_finite() || !reference_value.is_finite() {
                continue;
            }
            let proposed = ((reference_value - observed_value) / interval)
                .round_ties_even()
                .clamp(
                    -i32::from(options.max_abs_fold) as f32,
                    i32::from(options.max_abs_fold) as f32,
                ) as i32;
            proposal[index] = proposed;
            changed[index] = proposed != folds[index];
        }
    }

    let valid = observed.iter().filter(|value| value.is_finite()).count();
    let total_budget = if options.max_total_roi_gates == 0 {
        (valid / 10).min(262_144)
    } else {
        options.max_total_roi_gates as usize
    };
    let candidates = proposal_components(&changed, rows, gates);
    diagnostics.candidates_detected = u32::try_from(candidates.len()).unwrap_or(u32::MAX);
    let mut budget_used = 0usize;
    for candidate in candidates.into_iter().take(options.max_rois as usize) {
        let area = (candidate.row_end - candidate.row_start)
            .saturating_mul(candidate.gate_end - candidate.gate_start);
        if area > options.max_roi_gates as usize || budget_used.saturating_add(area) > total_budget
        {
            diagnostics.abstain_flags |= ABSTAINED_BUDGET;
            for row in candidate.row_start..candidate.row_end {
                for gate in candidate.gate_start..candidate.gate_end {
                    let index = row * gates + gate;
                    if changed[index] {
                        reasons[index] |= anchor_reason
                            | super::RIFT_REASON_BUDGET_EXCEEDED
                            | super::RIFT_REASON_ABSTAINED;
                    }
                }
            }
            continue;
        }
        budget_used += area;
        diagnostics.candidates_authorized += 1;
        let (selected, energy_delta, normalized_gain) = graph_cut_reference(
            observed, nyquist, folds, &proposal, reference, quality, gates, candidate,
        );
        if selected.is_empty() || energy_delta >= -FLOW_EPSILON {
            continue;
        }
        let candidate_confidence = (normalized_gain * 255.0).round().clamp(0.0, 255.0) as u8;
        if candidate_confidence < options.min_confidence {
            for &index in &selected {
                reasons[index] |= anchor_reason | super::RIFT_REASON_ABSTAINED;
            }
            continue;
        }
        diagnostics.components_accepted += 1;
        diagnostics.gates_refined = diagnostics
            .gates_refined
            .saturating_add(u32::try_from(selected.len()).unwrap_or(u32::MAX));
        diagnostics.reason_flags |= REASON_FUSION_ACCEPTED;
        for &index in &selected {
            folds[index] = proposal[index];
            confidence[index] = confidence[index].max(candidate_confidence);
            reasons[index] |= anchor_reason | super::RIFT_REASON_FUSION_ACCEPTED;
        }
    }
    diagnostics
}

fn proposal_components(mask: &[bool], rows: usize, gates: usize) -> Vec<Rectangle> {
    const MINIMUM_COMPONENT_GATES: usize = 6;
    const BUFFER: usize = 6;
    let mut visited = vec![false; mask.len()];
    let mut components = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut size = 0usize;
        let mut row_min = start / gates;
        let mut row_max = row_min;
        let mut gate_min = start % gates;
        let mut gate_max = gate_min;
        while let Some(index) = queue.pop_front() {
            size += 1;
            let row = index / gates;
            let gate = index % gates;
            row_min = row_min.min(row);
            row_max = row_max.max(row);
            gate_min = gate_min.min(gate);
            gate_max = gate_max.max(gate);
            if row > 0 {
                visit(mask, &mut visited, index - gates, &mut queue);
            }
            if row + 1 < rows {
                visit(mask, &mut visited, index + gates, &mut queue);
            }
            if gate > 0 {
                visit(mask, &mut visited, index - 1, &mut queue);
            }
            if gate + 1 < gates {
                visit(mask, &mut visited, index + 1, &mut queue);
            }
        }
        if size >= MINIMUM_COMPONENT_GATES {
            components.push(Rectangle {
                row_start: row_min.saturating_sub(BUFFER),
                row_end: (row_max + BUFFER + 1).min(rows),
                gate_start: gate_min.saturating_sub(BUFFER),
                gate_end: (gate_max + BUFFER + 1).min(gates),
            });
        }
    }
    components.sort_by_key(|rectangle| {
        std::cmp::Reverse(
            (rectangle.row_end - rectangle.row_start) * (rectangle.gate_end - rectangle.gate_start),
        )
    });
    components
}

#[allow(clippy::too_many_arguments)]
fn graph_cut_reference(
    observed: &[f32],
    nyquist: &[f32],
    baseline: &[i32],
    proposal: &[i32],
    reference: &[f32],
    quality: Option<&[u8]>,
    gates: usize,
    roi: Rectangle,
) -> (Vec<usize>, f64, f64) {
    let roi_gates = roi.gate_end - roi.gate_start;
    let mut node_at = vec![usize::MAX; (roi.row_end - roi.row_start) * roi_gates];
    let mut indices = Vec::new();
    for row in roi.row_start..roi.row_end {
        for gate in roi.gate_start..roi.gate_end {
            let index = row * gates + gate;
            if observed[index].is_finite() && nyquist[row].is_finite() && nyquist[row] > 0.0 {
                node_at[(row - roi.row_start) * roi_gates + gate - roi.gate_start] = indices.len();
                indices.push(index);
            }
        }
    }
    if indices.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    let mut keep_cost = vec![0.0f64; indices.len()];
    let mut proposal_cost = vec![0.0f64; indices.len()];
    for (node, &index) in indices.iter().enumerate() {
        let row = index / gates;
        let n = nyquist[row];
        let interval = 2.0 * n;
        let observed_value = observed[index];
        let reference_value = reference[index];
        if reference_value.is_finite() {
            let weight = f64::from(quality.map_or(255, |values| values[index])) / 255.0;
            let baseline_value = observed_value + interval * baseline[index] as f32;
            let proposal_value = observed_value + interval * proposal[index] as f32;
            keep_cost[node] += weight * huber(baseline_value - reference_value, 0.5 * n);
            proposal_cost[node] += weight * huber(proposal_value - reference_value, 0.5 * n);
        }
        proposal_cost[node] +=
            f64::from(0.02 * n) * f64::from((proposal[index] - baseline[index]).abs());
        let row_edge = row == roi.row_start || row + 1 == roi.row_end;
        let gate = index % gates;
        if row_edge || gate == roi.gate_start || gate + 1 == roi.gate_end {
            proposal_cost[node] += PIN_CAPACITY;
        }
    }

    let source = indices.len();
    let sink = source + 1;
    let mut graph = Dinic::new(sink + 1);
    for node in 0..indices.len() {
        if proposal_cost[node] > 0.0 {
            graph.add_edge(source, node, proposal_cost[node]);
        }
        if keep_cost[node] > 0.0 {
            graph.add_edge(node, sink, keep_cost[node]);
        }
    }
    let mut pairwise = Vec::new();
    for row in roi.row_start..roi.row_end {
        for gate in roi.gate_start..roi.gate_end {
            let local = (row - roi.row_start) * roi_gates + gate - roi.gate_start;
            let first_node = node_at[local];
            if first_node == usize::MAX {
                continue;
            }
            let first_index = row * gates + gate;
            for (next_row, next_gate) in [(row + 1, gate), (row, gate + 1)] {
                if next_row >= roi.row_end || next_gate >= roi.gate_end {
                    continue;
                }
                let next_local =
                    (next_row - roi.row_start) * roi_gates + next_gate - roi.gate_start;
                let second_node = node_at[next_local];
                if second_node == usize::MAX {
                    continue;
                }
                let second_index = next_row * gates + next_gate;
                let local_nyquist = nyquist[row].min(nyquist[next_row]);
                let mut weight = 0.12 * local_nyquist;
                let raw_difference = (observed[first_index] - observed[second_index]).abs();
                if raw_difference > 0.7 * local_nyquist {
                    weight *= 0.25;
                }
                if proposal[first_index] - baseline[first_index]
                    != proposal[second_index] - baseline[second_index]
                {
                    weight *= 0.35;
                }
                let weight = f64::from(weight);
                graph.add_undirected(first_node, second_node, weight);
                pairwise.push((first_node, second_node, weight));
            }
        }
    }
    graph.max_flow(source, sink);
    let source_set = graph.source_set(source);
    let selected_nodes: Vec<usize> = (0..indices.len())
        .filter(|&node| !source_set[node])
        .collect();
    let selected: Vec<usize> = selected_nodes.iter().map(|&node| indices[node]).collect();
    let mut energy_delta = 0.0;
    let mut selected_mask = vec![false; indices.len()];
    for &node in &selected_nodes {
        selected_mask[node] = true;
        energy_delta += proposal_cost[node] - keep_cost[node];
    }
    for &(first, second, weight) in &pairwise {
        if selected_mask[first] != selected_mask[second] {
            energy_delta += weight;
        }
    }
    let nyquist_scale = selected
        .iter()
        .map(|&index| f64::from(nyquist[index / gates]))
        .sum::<f64>()
        .max(1.0);
    let normalized_gain = (-energy_delta / nyquist_scale).max(0.0);
    (selected, energy_delta, normalized_gain)
}

fn huber(error: f32, transition: f32) -> f64 {
    let magnitude = f64::from(error.abs());
    let transition = f64::from(transition.max(1.0e-3));
    if magnitude <= transition {
        0.5 * magnitude * magnitude / transition
    } else {
        magnitude - 0.5 * transition
    }
}

fn signed_couplet_supports(
    observed: &[f32],
    folds: &[i32],
    interval: f32,
    nyquist: f32,
    azimuths: &[f32],
    rows: usize,
    gates: usize,
) -> Vec<Rectangle> {
    let mut seeds = vec![false; observed.len()];
    for row in 0..rows - 1 {
        let azimuth_delta = (azimuths[row + 1] - azimuths[row]).rem_euclid(360.0);
        if !(0.0 < azimuth_delta && azimuth_delta < MAXIMUM_FORWARD_AZIMUTH_GAP) {
            continue;
        }
        let first = row * gates;
        let second = (row + 1) * gates;
        for gate in 0..gates {
            let upper = solved_value(observed, folds, interval, first + gate);
            let lower = solved_value(observed, folds, interval, second + gate);
            if upper.is_finite()
                && lower.is_finite()
                && lower - upper < -SIGNED_SEED_FRACTION * nyquist
            {
                seeds[first + gate] = true;
            }
        }
    }

    let mut consumed = vec![false; seeds.len()];
    let mut supports = Vec::new();
    for start in 0..seeds.len() {
        if !seeds[start] || consumed[start] {
            continue;
        }
        consumed[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut members = Vec::new();
        while let Some(current) = queue.pop_front() {
            let row = current / gates;
            let gate = current % gates;
            members.push((row, gate));
            let row_start = row.saturating_sub(SEED_CLUSTER_RADIUS);
            let row_end = (row + SEED_CLUSTER_RADIUS + 1).min(rows - 1);
            let gate_start = gate.saturating_sub(SEED_CLUSTER_RADIUS);
            let gate_end = (gate + SEED_CLUSTER_RADIUS + 1).min(gates);
            for other_row in row_start..row_end {
                for other_gate in gate_start..gate_end {
                    let other = other_row * gates + other_gate;
                    if seeds[other] && !consumed[other] {
                        consumed[other] = true;
                        queue.push_back(other);
                    }
                }
            }
        }
        if members.len() < MINIMUM_SIGNED_SEEDS {
            continue;
        }
        let row_min = members.iter().map(|&(row, _)| row).min().unwrap();
        let row_max = members.iter().map(|&(row, _)| row).max().unwrap();
        let gate_min = members.iter().map(|&(_, gate)| gate).min().unwrap();
        let gate_max = members.iter().map(|&(_, gate)| gate).max().unwrap();
        let support = Rectangle {
            row_start: row_min.saturating_sub(SUPPORT_RAY_BEFORE),
            row_end: (row_max + SUPPORT_RAY_AFTER + 1).min(rows),
            gate_start: gate_min.saturating_sub(SUPPORT_GATE_RADIUS),
            gate_end: (gate_max + SUPPORT_GATE_RADIUS + 1).min(gates),
        };
        let mut negative = 0;
        let mut positive = 0;
        for row in support.row_start..support.row_end {
            for gate in support.gate_start..support.gate_end {
                let value = solved_value(observed, folds, interval, row * gates + gate);
                negative += usize::from(value.is_finite() && value <= -LOBE_FRACTION * nyquist);
                positive += usize::from(value.is_finite() && value >= LOBE_FRACTION * nyquist);
            }
        }
        if negative >= MINIMUM_LOBE_GATES && positive >= MINIMUM_LOBE_GATES {
            supports.push(support);
        }
    }
    supports
}

#[inline(always)]
fn solved_value(observed: &[f32], folds: &[i32], interval: f32, index: usize) -> f32 {
    let value = observed[index];
    if value.is_finite() {
        value + interval * folds[index] as f32
    } else {
        f32::NAN
    }
}

fn shear_candidates(values: &[f32], nyquist: f32, rows: usize, gates: usize) -> Vec<Rectangle> {
    let mut endpoints = vec![false; values.len()];
    let threshold = SHEAR_FRACTION * nyquist;
    for row in 0..rows {
        for gate in 0..gates {
            let index = row * gates + gate;
            let value = values[index];
            if !value.is_finite() {
                continue;
            }
            if gate + 1 < gates {
                let next = values[index + 1];
                if next.is_finite() && (next - value).abs() > threshold {
                    endpoints[index] = true;
                    endpoints[index + 1] = true;
                }
            }
            if row + 1 < rows {
                let next = values[index + gates];
                if next.is_finite() && (next - value).abs() > threshold {
                    endpoints[index] = true;
                    endpoints[index + gates] = true;
                }
            }
        }
    }
    let mut mask = endpoints.clone();
    for _ in 0..DETECTOR_DILATION {
        let previous = mask.clone();
        for row in 0..rows {
            for gate in 0..gates {
                let index = row * gates + gate;
                if previous[index]
                    || (row > 0 && previous[index - gates])
                    || (row + 1 < rows && previous[index + gates])
                    || (gate > 0 && previous[index - 1])
                    || (gate + 1 < gates && previous[index + 1])
                {
                    mask[index] = true;
                }
            }
        }
    }

    let mut visited = vec![false; values.len()];
    let mut candidates = Vec::new();
    for start in 0..values.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut endpoint_count = 0;
        let mut row_min = start / gates;
        let mut row_max = row_min;
        let mut gate_min = start % gates;
        let mut gate_max = gate_min;
        while let Some(index) = queue.pop_front() {
            let row = index / gates;
            let gate = index % gates;
            endpoint_count += usize::from(endpoints[index]);
            row_min = row_min.min(row);
            row_max = row_max.max(row);
            gate_min = gate_min.min(gate);
            gate_max = gate_max.max(gate);
            if row > 0 {
                visit(mask.as_slice(), &mut visited, index - gates, &mut queue);
            }
            if row + 1 < rows {
                visit(mask.as_slice(), &mut visited, index + gates, &mut queue);
            }
            if gate > 0 {
                visit(mask.as_slice(), &mut visited, index - 1, &mut queue);
            }
            if gate + 1 < gates {
                visit(mask.as_slice(), &mut visited, index + 1, &mut queue);
            }
        }
        if endpoint_count >= MINIMUM_SHEAR_ENDPOINTS {
            candidates.push(Rectangle {
                row_start: row_min.saturating_sub(ROI_PADDING),
                row_end: (row_max + ROI_PADDING + 1).min(rows),
                gate_start: gate_min.saturating_sub(ROI_PADDING),
                gate_end: (gate_max + ROI_PADDING + 1).min(gates),
            });
        }
    }
    candidates
}

fn visit(mask: &[bool], visited: &mut [bool], index: usize, queue: &mut VecDeque<usize>) {
    if mask[index] && !visited[index] {
        visited[index] = true;
        queue.push_back(index);
    }
}

fn graph_cut(
    values: &[f32],
    nyquist: f32,
    gates: usize,
    roi: Rectangle,
    direction: i32,
) -> (Vec<usize>, f64) {
    let mut node_at =
        vec![usize::MAX; (roi.row_end - roi.row_start) * (roi.gate_end - roi.gate_start)];
    let roi_gates = roi.gate_end - roi.gate_start;
    let mut indices = Vec::new();
    for row in roi.row_start..roi.row_end {
        for gate in roi.gate_start..roi.gate_end {
            let index = row * gates + gate;
            if values[index].is_finite() {
                let local = (row - roi.row_start) * roi_gates + gate - roi.gate_start;
                node_at[local] = indices.len();
                indices.push(index);
            }
        }
    }
    if indices.is_empty() {
        return (Vec::new(), 0.0);
    }
    let source = indices.len();
    let sink = source + 1;
    let mut graph = Dinic::new(sink + 1);
    let mut linear = vec![f64::from(SHIFT_PENALTY_FRACTION * nyquist); indices.len()];
    let shift = direction as f32 * 2.0 * nyquist;

    for row in roi.row_start..roi.row_end {
        for gate in roi.gate_start..roi.gate_end {
            let local = (row - roi.row_start) * roi_gates + gate - roi.gate_start;
            let first_node = node_at[local];
            if first_node == usize::MAX {
                continue;
            }
            let first = values[row * gates + gate];
            for (next_row, next_gate) in [(row + 1, gate), (row, gate + 1)] {
                if next_row >= roi.row_end || next_gate >= roi.gate_end {
                    continue;
                }
                let next_local =
                    (next_row - roi.row_start) * roi_gates + next_gate - roi.gate_start;
                let second_node = node_at[next_local];
                if second_node == usize::MAX {
                    continue;
                }
                let difference = first - values[next_row * gates + next_gate];
                let same = f64::from(difference.abs());
                let first_shifted = f64::from((difference + shift).abs());
                let second_shifted = f64::from((difference - shift).abs());
                let weight = ((first_shifted + second_shifted - 2.0 * same) * 0.5).max(0.0);
                linear[first_node] += first_shifted - same - weight;
                linear[second_node] += second_shifted - same - weight;
                if weight > 0.0 {
                    graph.add_undirected(first_node, second_node, weight);
                }
            }
        }
    }

    for (node, &index) in indices.iter().enumerate() {
        let row = index / gates;
        let gate = index % gates;
        let mut keep_cost = 0.0;
        let mut shift_cost = 0.0;
        if linear[node] >= 0.0 {
            shift_cost += linear[node];
        } else {
            keep_cost -= linear[node];
        }
        if row == roi.row_start
            || row + 1 == roi.row_end
            || gate == roi.gate_start
            || gate + 1 == roi.gate_end
        {
            shift_cost += PIN_CAPACITY;
        }
        if shift_cost > 0.0 {
            graph.add_edge(source, node, shift_cost);
        }
        if keep_cost > 0.0 {
            graph.add_edge(node, sink, keep_cost);
        }
    }
    graph.max_flow(source, sink);
    let source_set = graph.source_set(source);
    let selected: Vec<usize> = indices
        .into_iter()
        .enumerate()
        .filter_map(|(node, index)| (!source_set[node]).then_some(index))
        .collect();
    let energy_delta = cut_energy_delta(values, &selected, roi, shift, nyquist, gates);
    (selected, energy_delta)
}

fn cut_energy_delta(
    values: &[f32],
    selected: &[usize],
    roi: Rectangle,
    shift: f32,
    nyquist: f32,
    gates: usize,
) -> f64 {
    let roi_gates = roi.gate_end - roi.gate_start;
    let mut shifted = vec![false; (roi.row_end - roi.row_start) * roi_gates];
    for &index in selected {
        let row = index / gates;
        let gate = index % gates;
        shifted[(row - roi.row_start) * roi_gates + gate - roi.gate_start] = true;
    }
    let mut delta = f64::from(SHIFT_PENALTY_FRACTION * nyquist) * selected.len() as f64;
    for row in roi.row_start..roi.row_end {
        for gate in roi.gate_start..roi.gate_end {
            let index = row * gates + gate;
            let first = values[index];
            if !first.is_finite() {
                continue;
            }
            let local = (row - roi.row_start) * roi_gates + gate - roi.gate_start;
            let corrected_first = first + if shifted[local] { shift } else { 0.0 };
            for (next_row, next_gate) in [(row + 1, gate), (row, gate + 1)] {
                if next_row >= roi.row_end || next_gate >= roi.gate_end {
                    continue;
                }
                let second = values[next_row * gates + next_gate];
                if !second.is_finite() {
                    continue;
                }
                let next_local =
                    (next_row - roi.row_start) * roi_gates + next_gate - roi.gate_start;
                let corrected_second = second + if shifted[next_local] { shift } else { 0.0 };
                delta +=
                    f64::from((corrected_first - corrected_second).abs() - (first - second).abs());
            }
        }
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_finds_the_cheaper_partition() {
        let mut flow = Dinic::new(4);
        flow.add_edge(2, 0, 1.0);
        flow.add_edge(0, 3, 8.0);
        flow.add_edge(2, 1, 8.0);
        flow.add_edge(1, 3, 1.0);
        flow.add_undirected(0, 1, 2.0);
        flow.max_flow(2, 3);
        let source = flow.source_set(2);
        assert!(!source[0]);
        assert!(source[1]);
    }

    #[test]
    fn push_relabel_matches_exhaustive_small_graph_cuts() {
        const VARIABLES: usize = 5;
        const SOURCE: usize = VARIABLES;
        const SINK: usize = SOURCE + 1;
        let mut state = 0x51f1_5eed_u64;

        for _case in 0..128 {
            let mut next_capacity = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 60) + 1) as f64
            };
            let mut flow = Dinic::new(SINK + 1);
            let mut arcs = Vec::new();
            for node in 0..VARIABLES {
                let source_capacity = next_capacity();
                let sink_capacity = next_capacity();
                flow.add_edge(SOURCE, node, source_capacity);
                flow.add_edge(node, SINK, sink_capacity);
                arcs.push((SOURCE, node, source_capacity));
                arcs.push((node, SINK, sink_capacity));
            }
            for first in 0..VARIABLES {
                for second in first + 1..VARIABLES {
                    let capacity = next_capacity();
                    flow.add_undirected(first, second, capacity);
                    arcs.push((first, second, capacity));
                    arcs.push((second, first, capacity));
                }
            }

            let cut_cost = |mask: usize| {
                let on_source_side =
                    |node: usize| node == SOURCE || (node < VARIABLES && mask & (1 << node) != 0);
                arcs.iter()
                    .filter(|&&(from, to, _)| on_source_side(from) && !on_source_side(to))
                    .map(|&(_, _, capacity)| capacity)
                    .sum::<f64>()
            };
            let expected = (0..1usize << VARIABLES)
                .map(cut_cost)
                .fold(f64::INFINITY, f64::min);

            flow.max_flow(SOURCE, SINK);
            let source_set = flow.source_set(SOURCE);
            let actual_mask = (0..VARIABLES).fold(0usize, |mask, node| {
                mask | (usize::from(source_set[node]) << node)
            });
            assert_eq!(cut_cost(actual_mask), expected);
        }
    }
}
