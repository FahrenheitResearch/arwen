//! Storm cell identification using SCIT-style multi-threshold watershed segmentation.
//!
//! Based on the NSSL Storm Cell Identification and Tracking (SCIT) algorithm:
//! 1. Cascade through reflectivity thresholds from high to low (60, 55, 50, 45, 40, 35 dBZ)
//! 2. At each level, find connected components via union-find
//! 3. High-threshold cores become cell centroids; lower thresholds define the cell envelope
//! 4. Range-dependent minimum size filters out noise at long range
//! 5. Reflectivity-weighted centroids for accurate positioning

use std::collections::HashSet;

use crate::level2::Level2Sweep;
use crate::products::RadarProduct;

/// Threshold cascade from high to low (dBZ). Each level finds cores within the
/// previous level's envelope. This is what separates embedded cells within a
/// larger convective mass.
const THRESHOLDS: [f32; 6] = [60.0, 55.0, 50.0, 45.0, 40.0, 35.0];

/// Minimum number of gates at each threshold level to count as a core.
/// Higher thresholds need fewer gates (strong cores are small).
const MIN_GATES_BY_LEVEL: [usize; 6] = [4, 6, 10, 15, 20, 30];

/// Range-dependent gate scaling: beyond this range (km), double the min gate count.
const RANGE_SCALE_KM: f32 = 150.0;

/// Maximum azimuth gap (degrees) for adjacency.
const AZ_ADJACENCY_DEG: f32 = 2.0;

/// Maximum range gap (gates) for adjacency.
const RANGE_ADJACENCY_GATES: usize = 3;

/// Minimum range to consider (km) — filters ground clutter.
const MIN_RANGE_KM: f32 = 10.0;

/// Maximum range to consider (km) — beyond this, beam broadening degrades detection.
const MAX_RANGE_KM: f32 = 230.0;

/// A detected storm cell from a single sweep.
#[derive(Debug, Clone)]
pub struct StormCell {
    /// Cell label (1-based, sorted by max_reflectivity descending).
    pub label: usize,
    /// Reflectivity-weighted centroid azimuth (degrees).
    pub centroid_azimuth: f32,
    /// Reflectivity-weighted centroid range (km).
    pub centroid_range_km: f32,
    /// Centroid latitude (if site location known).
    pub lat: f64,
    /// Centroid longitude (if site location known).
    pub lon: f64,
    /// Maximum reflectivity in the cell (dBZ).
    pub max_reflectivity: f32,
    /// Mean reflectivity in the cell (dBZ).
    pub mean_reflectivity: f32,
    /// Number of gates in the cell.
    pub gate_count: usize,
    /// Approximate area (km²).
    pub area_km2: f32,
    /// Azimuth extent (degrees).
    pub az_extent: f32,
    /// Range extent (km).
    pub range_extent_km: f32,
    /// Min azimuth.
    pub az_min: f32,
    /// Max azimuth.
    pub az_max: f32,
    /// Min range (km).
    pub range_min_km: f32,
    /// Max range (km).
    pub range_max_km: f32,
    /// Elevation angle of the sweep.
    pub elevation: f32,
    /// Highest threshold this cell was detected at (dBZ) — indicates core intensity.
    pub core_threshold: f32,
}

/// A gate with its reflectivity value and coordinates.
struct Gate {
    radial_idx: usize,
    gate_idx: usize,
    azimuth: f32,
    range_km: f32,
    reflectivity: f32,
}

/// Union-Find data structure for connected component labeling.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// A cell core found at a high threshold.
struct CellCore {
    /// Reflectivity-weighted centroid azimuth (degrees).
    centroid_az: f32,
    /// Reflectivity-weighted centroid range (km).
    centroid_range: f32,
    /// Max reflectivity.
    max_ref: f32,
    /// Threshold at which this core was first detected.
    detection_threshold: f32,
    /// Gate indices (into the lowest-threshold gate list) that belong to this cell.
    gate_indices: HashSet<usize>,
}

/// Identify storm cells using multi-threshold watershed segmentation.
///
/// Returns cells sorted by max_reflectivity descending (C1 = strongest).
pub fn identify_cells(
    sweep: &Level2Sweep,
    site_lat: Option<f64>,
    site_lon: Option<f64>,
) -> Vec<StormCell> {
    // Collect ALL gates with reflectivity data
    let mut all_gates: Vec<Gate> = Vec::new();

    for (ri, radial) in sweep.radials.iter().enumerate() {
        let ref_moment = match radial.moments.iter()
            .find(|m| m.product == RadarProduct::Reflectivity) {
            Some(m) => m,
            None => continue,
        };

        let gate_size_km = ref_moment.gate_size as f32 / 1000.0;
        let first_gate_km = ref_moment.first_gate_range as f32 / 1000.0;

        for (gi, &val) in ref_moment.data.iter().enumerate() {
            if val.is_nan() || val < THRESHOLDS[THRESHOLDS.len() - 1] {
                continue;
            }
            let range_km = first_gate_km + gi as f32 * gate_size_km;
            if range_km < MIN_RANGE_KM || range_km > MAX_RANGE_KM {
                continue;
            }
            all_gates.push(Gate {
                radial_idx: ri,
                gate_idx: gi,
                azimuth: radial.azimuth,
                range_km,
                reflectivity: val,
            });
        }
    }

    if all_gates.is_empty() {
        return Vec::new();
    }

    let num_radials = sweep.radials.len();

    // Build gate lookup for adjacency
    let mut gate_map: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::with_capacity(all_gates.len());
    for (i, g) in all_gates.iter().enumerate() {
        gate_map.insert((g.radial_idx, g.gate_idx), i);
    }

    // --- Multi-threshold cascade ---
    // Start from highest threshold, find cores, then grow them at lower thresholds
    let mut cell_cores: Vec<CellCore> = Vec::new();

    for (level, &threshold) in THRESHOLDS.iter().enumerate() {
        // Get gates at this threshold
        let level_gates: Vec<usize> = all_gates.iter()
            .enumerate()
            .filter(|(_, g)| g.reflectivity >= threshold)
            .map(|(i, _)| i)
            .collect();

        if level_gates.is_empty() {
            continue;
        }

        // Build UF for this threshold's connected components
        let n = level_gates.len();
        let mut uf = UnionFind::new(n);

        // Index: gate_map_idx -> position in level_gates
        let mut level_lookup: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(n);
        for (li, &gi) in level_gates.iter().enumerate() {
            level_lookup.insert(gi, li);
        }

        for li in 0..n {
            let gi = level_gates[li];
            let g = &all_gates[gi];
            let ri = g.radial_idx;
            let gidx = g.gate_idx;

            // Range-adjacent on same radial
            for dg in 1..=RANGE_ADJACENCY_GATES {
                if let Some(&neighbor_gi) = gate_map.get(&(ri, gidx + dg)) {
                    if all_gates[neighbor_gi].reflectivity >= threshold {
                        if let Some(&neighbor_li) = level_lookup.get(&neighbor_gi) {
                            uf.union(li, neighbor_li);
                        }
                    }
                }
            }

            // Azimuthally adjacent
            for dr in 1..=3usize {
                for sign in &[1i32, -1i32] {
                    let adj_ri = ((ri as i32 + sign * dr as i32).rem_euclid(num_radials as i32)) as usize;
                    let az1 = g.azimuth;
                    let az2 = sweep.radials[adj_ri].azimuth;
                    if az_distance(az1, az2) > AZ_ADJACENCY_DEG {
                        continue;
                    }

                    for dg in -(RANGE_ADJACENCY_GATES as i32)..=(RANGE_ADJACENCY_GATES as i32) {
                        let target_gi = gidx as i32 + dg;
                        if target_gi < 0 { continue; }
                        if let Some(&neighbor_gi) = gate_map.get(&(adj_ri, target_gi as usize)) {
                            if all_gates[neighbor_gi].reflectivity >= threshold {
                                if let Some(&neighbor_li) = level_lookup.get(&neighbor_gi) {
                                    uf.union(li, neighbor_li);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Collect components
        let mut components: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for li in 0..n {
            let root = uf.find(li);
            components.entry(root).or_default().push(level_gates[li]);
        }

        // Range-dependent minimum gate count
        let base_min = MIN_GATES_BY_LEVEL[level];

        for (_root, gate_indices) in &components {
            let avg_range: f32 = gate_indices.iter()
                .map(|&gi| all_gates[gi].range_km)
                .sum::<f32>() / gate_indices.len() as f32;

            let min_gates = if avg_range > RANGE_SCALE_KM {
                base_min * 2
            } else {
                base_min
            };

            if gate_indices.len() < min_gates {
                continue;
            }

            // Check if this component overlaps with an existing core (from a higher threshold)
            // If it does, expand that core. If not, create a new core.
            let mut merged = false;
            for core in &mut cell_cores {
                let overlap = gate_indices.iter().any(|&gi| core.gate_indices.contains(&gi));
                if overlap {
                    // Expand with O(1) insert per gate (HashSet deduplicates)
                    for &gi in gate_indices {
                        core.gate_indices.insert(gi);
                        if all_gates[gi].reflectivity > core.max_ref {
                            core.max_ref = all_gates[gi].reflectivity;
                        }
                    }
                    merged = true;
                    break;
                }
            }

            if !merged {
                // New core — compute centroid
                let max_ref = gate_indices.iter()
                    .map(|&gi| all_gates[gi].reflectivity)
                    .fold(f32::NEG_INFINITY, f32::max);

                cell_cores.push(CellCore {
                    centroid_az: 0.0,   // computed later
                    centroid_range: 0.0, // computed later
                    max_ref,
                    detection_threshold: threshold,
                    gate_indices: gate_indices.iter().copied().collect(),
                });
            }
        }
    }

    // --- Post-process: merge cores that became too close after expansion ---
    // Two cores whose reflectivity-weighted centroids are within 10 km get merged
    // (keep the stronger one)
    let merge_dist_km = 10.0f32;
    loop {
        let mut merged_any = false;
        // Compute centroids for all cores
        for core in &mut cell_cores {
            compute_core_centroid(core, &all_gates);
        }

        let n = cell_cores.len();
        let mut to_merge: Option<(usize, usize)> = None;
        'outer: for i in 0..n {
            for j in (i + 1)..n {
                let daz = az_distance(cell_cores[i].centroid_az, cell_cores[j].centroid_az)
                    .to_radians() as f64;
                let dr = (cell_cores[i].centroid_range - cell_cores[j].centroid_range) as f64;
                let avg_r = ((cell_cores[i].centroid_range + cell_cores[j].centroid_range) / 2.0) as f64;
                let dist = ((avg_r * daz).powi(2) + dr.powi(2)).sqrt() as f32;
                if dist < merge_dist_km {
                    to_merge = Some((i, j));
                    break 'outer;
                }
            }
        }

        if let Some((i, j)) = to_merge {
            // Merge j into i (keep whichever has higher max_ref)
            let (keep, remove) = if cell_cores[i].max_ref >= cell_cores[j].max_ref {
                (i, j)
            } else {
                (j, i)
            };
            let removed_gates = cell_cores[remove].gate_indices.clone();
            let removed_thresh = cell_cores[remove].detection_threshold;
            cell_cores[keep].gate_indices.extend(removed_gates);
            if removed_thresh > cell_cores[keep].detection_threshold {
                cell_cores[keep].detection_threshold = removed_thresh;
            }
            cell_cores.remove(remove);
            merged_any = true;
        }

        if !merged_any { break; }
    }

    // --- Build final StormCell structs ---
    let gate_size_km = sweep.radials.first()
        .and_then(|r| r.moments.iter().find(|m| m.product == RadarProduct::Reflectivity))
        .map(|m| m.gate_size as f32 / 1000.0)
        .unwrap_or(0.25);
    let az_spacing_rad = sweep.radials.first()
        .map(|r| r.azimuth_spacing.to_radians())
        .unwrap_or(1.0f32.to_radians());

    let mut cells: Vec<StormCell> = Vec::new();

    for core in &mut cell_cores {
        compute_core_centroid(core, &all_gates);

        let count = core.gate_indices.len();
        let centroid_az = core.centroid_az;
        let centroid_range = core.centroid_range;

        // Statistics
        let mut sum_ref: f64 = 0.0;
        let mut max_ref: f32 = f32::NEG_INFINITY;
        let mut az_min: f32 = 360.0;
        let mut az_max: f32 = 0.0;
        let mut range_min: f32 = f32::MAX;
        let mut range_max: f32 = 0.0;

        for &gi in &core.gate_indices {
            let g = &all_gates[gi];
            sum_ref += g.reflectivity as f64;
            if g.reflectivity > max_ref { max_ref = g.reflectivity; }
            if g.azimuth < az_min { az_min = g.azimuth; }
            if g.azimuth > az_max { az_max = g.azimuth; }
            if g.range_km < range_min { range_min = g.range_km; }
            if g.range_km > range_max { range_max = g.range_km; }
        }

        let mean_ref = (sum_ref / count as f64) as f32;
        let avg_range = centroid_range;
        let gate_area = gate_size_km * (avg_range * az_spacing_rad);
        let area_km2 = gate_area * count as f32;
        let az_extent = az_distance(az_min, az_max);
        let range_extent = range_max - range_min;

        let (lat, lon) = if let (Some(slat), Some(slon)) = (site_lat, site_lon) {
            let az_rad = (centroid_az as f64).to_radians();
            let la = slat + (centroid_range as f64 * az_rad.cos()) / 111.139;
            let lo = slon + (centroid_range as f64 * az_rad.sin())
                / (111.139 * slat.to_radians().cos());
            ((la * 1000.0).round() / 1000.0, (lo * 1000.0).round() / 1000.0)
        } else {
            (0.0, 0.0)
        };

        cells.push(StormCell {
            label: 0,
            centroid_azimuth: (centroid_az * 10.0).round() / 10.0,
            centroid_range_km: (centroid_range * 10.0).round() / 10.0,
            lat,
            lon,
            max_reflectivity: (max_ref * 10.0).round() / 10.0,
            mean_reflectivity: (mean_ref * 10.0).round() / 10.0,
            gate_count: count,
            area_km2: (area_km2 * 10.0).round() / 10.0,
            az_extent: (az_extent * 10.0).round() / 10.0,
            range_extent_km: (range_extent * 10.0).round() / 10.0,
            az_min,
            az_max,
            range_min_km: range_min,
            range_max_km: range_max,
            elevation: sweep.elevation_angle,
            core_threshold: core.detection_threshold,
        });
    }

    // Sort by max reflectivity descending, assign labels
    cells.sort_by(|a, b| b.max_reflectivity.partial_cmp(&a.max_reflectivity)
        .unwrap_or(std::cmp::Ordering::Equal));
    for (i, cell) in cells.iter_mut().enumerate() {
        cell.label = i + 1;
    }

    cells
}

/// Compute reflectivity-weighted centroid for a core.
fn compute_core_centroid(core: &mut CellCore, gates: &[Gate]) {
    let mut sum_az_x: f64 = 0.0;
    let mut sum_az_y: f64 = 0.0;
    let mut sum_range: f64 = 0.0;
    let mut sum_weight: f64 = 0.0;

    for &gi in &core.gate_indices {
        let g = &gates[gi];
        // Weight by reflectivity squared — emphasizes the core
        let w = (g.reflectivity as f64).powi(2);
        let az_rad = (g.azimuth as f64).to_radians();
        sum_az_x += az_rad.cos() * w;
        sum_az_y += az_rad.sin() * w;
        sum_range += g.range_km as f64 * w;
        sum_weight += w;
    }

    if sum_weight > 0.0 {
        core.centroid_az = (sum_az_y.atan2(sum_az_x).to_degrees() as f32 + 360.0) % 360.0;
        core.centroid_range = (sum_range / sum_weight) as f32;
    }
}

/// Circular azimuth distance in degrees.
fn az_distance(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    if d > 180.0 { 360.0 - d } else { d }
}

/// Associate mesocyclone detections with the nearest cell.
///
/// Returns the cell label (1-based) for each detection, or 0 if no cell is within
/// `max_distance_km`.
pub fn associate_mesos_with_cells(
    cells: &[StormCell],
    meso_azimuths: &[f32],
    meso_ranges_km: &[f32],
    max_distance_km: f32,
) -> Vec<usize> {
    meso_azimuths.iter().zip(meso_ranges_km.iter()).map(|(&maz, &mrng)| {
        let mut best_label = 0usize;
        let mut best_dist = max_distance_km;

        for cell in cells {
            let daz = az_distance(maz, cell.centroid_azimuth).to_radians() as f64;
            let dr = (mrng - cell.centroid_range_km) as f64;
            let avg_range = ((mrng + cell.centroid_range_km) / 2.0) as f64;
            let dist = ((avg_range * daz).powi(2) + dr.powi(2)).sqrt() as f32;

            if dist < best_dist {
                best_dist = dist;
                best_label = cell.label;
            }
        }

        best_label
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level2::{Level2Sweep, RadialData, MomentData};

    fn make_test_sweep() -> Level2Sweep {
        let mut radials = Vec::new();
        for ri in 0..360 {
            let azimuth = ri as f32;
            let mut ref_data = vec![f32::NAN; 500];

            // Strong cell: az 175-185, gates 100-130, peak 60 dBZ
            if (175..=185).contains(&ri) {
                for gi in 100..130 {
                    let dist_from_center = ((gi as f32 - 115.0).abs() + (ri as f32 - 180.0).abs()) / 2.0;
                    ref_data[gi] = (60.0 - dist_from_center * 1.5).max(30.0);
                }
            }

            // Weaker cell: az 90-100, gates 200-220, peak 48 dBZ
            if (90..=100).contains(&ri) {
                for gi in 200..220 {
                    let dist_from_center = ((gi as f32 - 210.0).abs() + (ri as f32 - 95.0).abs()) / 2.0;
                    ref_data[gi] = (48.0 - dist_from_center * 1.5).max(30.0);
                }
            }

            radials.push(RadialData {
                azimuth,
                elevation: 0.5,
                azimuth_spacing: 1.0,
                nyquist_velocity: None,
                radial_status: 1,
                moments: vec![MomentData {
                    product: RadarProduct::Reflectivity,
                    gate_count: 500,
                    first_gate_range: 2125,
                    gate_size: 250,
                    data: ref_data,
                }],
            });
        }

        Level2Sweep {
            elevation_number: 1,
            elevation_angle: 0.5,
            nyquist_velocity: None,
            sweep_index: 0,
            start_status: 3,
            end_status: 2,
            cut_sector: 0,
            radials,
        }
    }

    #[test]
    fn test_identify_cells_two_distinct() {
        let sweep = make_test_sweep();
        let cells = identify_cells(&sweep, Some(35.0), Some(-97.0));

        assert!(cells.len() >= 2, "should find at least two distinct cells, got {}", cells.len());
        assert_eq!(cells[0].label, 1);
        // Strongest cell should have higher max ref
        assert!(cells[0].max_reflectivity > cells[1].max_reflectivity);
        // They should be at different positions
        assert!(az_distance(cells[0].centroid_azimuth, cells[1].centroid_azimuth) > 10.0);
    }

    #[test]
    fn test_no_cells_below_threshold() {
        let mut sweep = make_test_sweep();
        for radial in &mut sweep.radials {
            for moment in &mut radial.moments {
                for val in &mut moment.data {
                    if !val.is_nan() {
                        *val = 20.0;
                    }
                }
            }
        }
        let cells = identify_cells(&sweep, None, None);
        assert!(cells.is_empty());
    }

    #[test]
    fn test_core_threshold_reflects_intensity() {
        let sweep = make_test_sweep();
        let cells = identify_cells(&sweep, None, None);
        // Strongest cell (peak 60 dBZ) should have been detected at 55+ dBZ threshold
        assert!(cells[0].core_threshold >= 50.0,
            "strongest cell core_threshold should be >= 50, got {}", cells[0].core_threshold);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(3));
        assert_ne!(uf.find(0), uf.find(4));
    }

    #[test]
    fn test_az_distance() {
        assert!((az_distance(350.0, 10.0) - 20.0).abs() < 0.01);
        assert!((az_distance(10.0, 350.0) - 20.0).abs() < 0.01);
        assert!((az_distance(90.0, 180.0) - 90.0).abs() < 0.01);
    }
}
