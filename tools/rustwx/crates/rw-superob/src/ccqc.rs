//! Correlation-coefficient QC, transcribed from `gpuwm/obs/cc_qc.py`.
//!
//! The rule is per moment: reflectivity is compound (low RhoHV AND
//! reflectivity below the shield), velocity is unshielded, and the
//! debris-signature fringe beside a clustered couplet keeps its velocity.
//! The rulings and their reasoning live in the Python module; this file
//! reproduces its masks and every counter it publishes.

use std::collections::BTreeMap;

use crate::num::py_mod;
use crate::request::CcParams;

pub const REASON_COLOCATED: &str = "colocated";
pub const REASON_COMPANION: &str = "companion";
pub const REASON_NO_RHO: &str = "no-rho";
pub const REASON_NO_REF: &str = "no-ref";
pub const REASON_NO_TARGET: &str = "no-target";

pub const DROP_LOW_RHO: &str = "low-rho";
pub const DROP_LOW_RHO_BELOW_SHIELD: &str = "low-rho-below-shield";
pub const DROP_RHO_FLOOR: &str = "rho-floor";

pub const TDS_RHO_BELOW_FLOOR: &str = "rho-below-debris-floor";
pub const TDS_BELOW_Z_BAND: &str = "below-debris-reflectivity";
pub const TDS_ABOVE_Z_BAND: &str = "at-or-above-shield";
pub const TDS_NO_ROTATION: &str = "no-couplet-nearby";

/// One moment plane as the masker samples it.
#[derive(Clone, Copy)]
pub struct PlaneRef<'a> {
    pub data: &'a [f64],
    pub rows: usize,
    pub gate_count: usize,
    pub first_gate_range_m: f64,
    pub gate_size_m: f64,
}

impl PlaneRef<'_> {
    /// `rint((ranges - first) / size)` for each target range, and whether
    /// it lands inside this plane's gates.
    fn gate_index(&self, ranges: &[f64]) -> Vec<Option<usize>> {
        ranges
            .iter()
            .map(|&r| {
                let index = crate::num::rint_index((r - self.first_gate_range_m) / self.gate_size_m);
                if index >= 0 && (index as u64) < self.gate_count as u64 {
                    Some(index as usize)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// A sweep as plan building sees it.
pub struct SweepPlanInput<'a> {
    pub elevation_angle_deg: f64,
    pub sweep_index: i64,
    pub azimuth: &'a [f64],
    pub reflectivity: Option<PlaneRef<'a>>,
    pub velocity: Option<PlaneRef<'a>>,
    pub correlation: Option<PlaneRef<'a>>,
}

/// `_nearest_azimuth_rows`: per target radial, the nearest source radial
/// by circular azimuth, or -1 beyond `tolerance_deg`.
pub fn nearest_azimuth_rows(target_az: &[f64], source_az: &[f64], tolerance_deg: f64) -> Vec<i64> {
    let source: Vec<f64> = source_az.iter().map(|&a| py_mod(a, 360.0)).collect();
    if source.is_empty() {
        return vec![-1; target_az.len()];
    }
    // np.argsort(kind="stable"): NaN sorts last.
    let mut order: Vec<usize> = (0..source.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (source[a], source[b]);
        match (x.is_nan(), y.is_nan()) {
            (false, false) => x.partial_cmp(&y).unwrap(),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (true, true) => std::cmp::Ordering::Equal,
        }
    });
    let n = order.len();
    let mut ext_az = Vec::with_capacity(n + 2);
    let mut ext_rows = Vec::with_capacity(n + 2);
    ext_az.push(source[order[n - 1]] - 360.0);
    ext_rows.push(order[n - 1]);
    for &row in &order {
        ext_az.push(source[row]);
        ext_rows.push(row);
    }
    ext_az.push(source[order[0]] + 360.0);
    ext_rows.push(order[0]);
    let last = ext_az.len() as i64 - 1;
    target_az
        .iter()
        .map(|&t| {
            let t = py_mod(t, 360.0);
            // searchsorted, side="left", in numpy's NaN-last order:
            // a < b, or b is NaN and a is not.
            let pos = ext_az
                .partition_point(|&v| v < t || (t.is_nan() && !v.is_nan()))
                as i64;
            let left = (pos - 1).clamp(0, last) as usize;
            let right = pos.clamp(0, last) as usize;
            let take_right = (ext_az[right] - t).abs() < (t - ext_az[left]).abs();
            let chosen = if take_right { right } else { left };
            if (ext_az[chosen] - t).abs() > tolerance_deg {
                -1
            } else {
                ext_rows[chosen] as i64
            }
        })
        .collect()
}

/// `_box_count`: True cells in the `(2h+1) x (2w+1)` box about each cell,
/// radials wrapping (by index, so a box taller than the sweep revisits
/// rows exactly as the reference does), gates not.
pub fn box_count(mask: &[bool], rows: usize, cols: usize, half_rows: usize, half_cols: usize) -> Vec<i64> {
    let mut out = vec![0i64; rows * cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    // Row-wise windowed counts over the gate axis.
    let mut row_counts = vec![0i64; rows * cols];
    for r in 0..rows {
        let mut prefix = vec![0i64; cols + 1];
        for c in 0..cols {
            prefix[c + 1] = prefix[c] + mask[r * cols + c] as i64;
        }
        for c in 0..cols {
            let lo = c.saturating_sub(half_cols);
            let hi = (c + half_cols + 1).min(cols);
            row_counts[r * cols + c] = prefix[hi] - prefix[lo];
        }
    }
    for r in 0..rows {
        for dr in 0..(2 * half_rows + 1) {
            let source = ((r + rows * (half_rows / rows + 1) + dr) - half_rows) % rows;
            let base = source * cols;
            for c in 0..cols {
                out[r * cols + c] += row_counts[base + c];
            }
        }
    }
    out
}

/// `couplet_seed_mask`.
pub fn couplet_seed_mask(velocity: &PlaneRef, azimuth: &[f64], params: &CcParams) -> Vec<bool> {
    let rows = velocity.rows;
    let cols = velocity.gate_count;
    let mut seeds = vec![false; rows * cols];
    if rows < 2 {
        return seeds;
    }
    let lobe = params.tds_couplet_lobe_ms;
    let dv = params.tds_couplet_delta_v_ms;
    let mut pair = vec![false; rows * cols];
    for r in 0..rows {
        let next = (r + 1) % rows;
        let gap = py_mod(azimuth[next] - azimuth[r], 360.0);
        let adjacent = gap > 0.0 && gap <= params.tds_couplet_max_azimuth_gap_deg;
        if !adjacent {
            continue;
        }
        for c in 0..cols {
            let v = velocity.data[r * cols + c];
            let n = velocity.data[next * cols + c];
            pair[r * cols + c] = v.is_finite()
                && n.is_finite()
                && v * n < 0.0
                && v.abs() >= lobe
                && n.abs() >= lobe
                && (v - n).abs() >= dv;
        }
    }
    for r in 0..rows {
        let previous = (r + rows - 1) % rows;
        for c in 0..cols {
            seeds[r * cols + c] = pair[r * cols + c] || pair[previous * cols + c];
        }
    }
    seeds
}

/// Per-moment state the counters need, kept from that moment's one
/// `drop_mask` call per sweep.
#[derive(Default)]
struct MomentMasks {
    /// `(reason, mask)` in the reference's insertion order.
    reasons: Vec<(&'static str, Vec<bool>)>,
    shielded: Vec<bool>,
    exempt: Option<Vec<bool>>,
    turned_away: Option<Vec<(&'static str, Vec<bool>)>>,
}

/// `CcSweepMasker`.
pub struct Masker<'a> {
    rho: PlaneRef<'a>,
    rho_rows: Vec<i64>,
    reflectivity: PlaneRef<'a>,
    velocity: Option<PlaneRef<'a>>,
    azimuth: &'a [f64],
    params: CcParams,
    pub reason: &'static str,
    pub companion_sweep_index: Option<i64>,
    rotation_plane: Option<(Vec<bool>, usize, usize)>,
    masks: BTreeMap<String, MomentMasks>,
    pub gates_tested: BTreeMap<String, i64>,
    pub gates_rho_missing: BTreeMap<String, i64>,
    pub gates_dropped: BTreeMap<String, i64>,
    pub gates_dropped_reason: BTreeMap<String, BTreeMap<String, i64>>,
    pub gates_dropped_shielded_z: BTreeMap<String, i64>,
    pub gates_exempt_tds_fringe: BTreeMap<String, i64>,
    pub gates_tds_turned_away: BTreeMap<String, BTreeMap<String, i64>>,
    pub couplet_seed_gates: i64,
}

/// `CcSweepPlan`.
pub struct Plan<'a> {
    pub reason: &'static str,
    pub masker: Option<Masker<'a>>,
}

/// `build_cc_plans`: one plan per sweep, in sweep order.
pub fn build_plans<'a>(sweeps: &[SweepPlanInput<'a>], params: &CcParams) -> Vec<Plan<'a>> {
    let mut plans = Vec::with_capacity(sweeps.len());
    for (position, sweep) in sweeps.iter().enumerate() {
        if sweep.reflectivity.is_none() && sweep.velocity.is_none() {
            plans.push(Plan { reason: REASON_NO_TARGET, masker: None });
            continue;
        }
        let Some(reflectivity) = sweep.reflectivity else {
            plans.push(Plan { reason: REASON_NO_REF, masker: None });
            continue;
        };
        let make = |rho: PlaneRef<'a>, rows: Vec<i64>, reason, companion| Masker {
            rho,
            rho_rows: rows,
            reflectivity,
            velocity: sweep.velocity,
            azimuth: sweep.azimuth,
            params: params.clone(),
            reason,
            companion_sweep_index: companion,
            rotation_plane: None,
            masks: BTreeMap::new(),
            gates_tested: BTreeMap::new(),
            gates_rho_missing: BTreeMap::new(),
            gates_dropped: BTreeMap::new(),
            gates_dropped_reason: BTreeMap::new(),
            gates_dropped_shielded_z: BTreeMap::new(),
            gates_exempt_tds_fringe: BTreeMap::new(),
            gates_tds_turned_away: BTreeMap::new(),
            couplet_seed_gates: 0,
        };
        if let Some(rho) = sweep.correlation {
            let rows = (0..rho.rows as i64).collect();
            plans.push(Plan {
                reason: REASON_COLOCATED,
                masker: Some(make(rho, rows, REASON_COLOCATED, None)),
            });
            continue;
        }
        let mut companion = None;
        if params.pair_companion_sweeps {
            for offset in [-1i64, 1] {
                let neighbour = position as i64 + offset;
                if neighbour < 0 || neighbour >= sweeps.len() as i64 {
                    continue;
                }
                let neighbour = &sweeps[neighbour as usize];
                let Some(near) = neighbour.correlation else { continue };
                let apart = (neighbour.elevation_angle_deg - sweep.elevation_angle_deg).abs();
                if apart <= params.companion_elevation_tolerance_deg {
                    companion = Some((neighbour, near));
                    break;
                }
            }
        }
        let Some((neighbour, near)) = companion else {
            plans.push(Plan { reason: REASON_NO_RHO, masker: None });
            continue;
        };
        let rows = nearest_azimuth_rows(
            sweep.azimuth,
            neighbour.azimuth,
            params.companion_azimuth_tolerance_deg,
        );
        plans.push(Plan {
            reason: REASON_COMPANION,
            masker: Some(make(near, rows, REASON_COMPANION, Some(neighbour.sweep_index))),
        });
    }
    plans
}

impl Masker<'_> {
    /// `_sample`: values at (rows, nearest gate by slant range), NaN where
    /// the row is unmatched or the range is beyond the plane.
    fn sample(plane: &PlaneRef, ranges: &[f64], rows: &[i64]) -> Vec<f64> {
        let index = plane.gate_index(ranges);
        let n = ranges.len();
        let mut out = vec![f64::NAN; rows.len() * n];
        if plane.rows == 0 {
            return out;
        }
        for (t, &row) in rows.iter().enumerate() {
            if row < 0 {
                continue;
            }
            let row = (row as usize).min(plane.rows - 1);
            for (k, gate) in index.iter().enumerate() {
                if let Some(g) = gate {
                    out[t * n + k] = plane.data[row * plane.gate_count + g];
                }
            }
        }
        out
    }

    /// `_rotation_at`: the rotation-association plane of this sweep's own
    /// velocity, sampled at `ranges`; built lazily on first need.
    fn rotation_at(&mut self, ranges: &[f64]) -> Vec<bool> {
        let target_rows = self.reflectivity.rows;
        let n = ranges.len();
        let Some(velocity) = self.velocity else {
            return vec![false; target_rows * n];
        };
        if self.rotation_plane.is_none() {
            let rows = velocity.rows;
            let cols = velocity.gate_count;
            let seeds = couplet_seed_mask(&velocity, self.azimuth, &self.params);
            let radius = self.params.tds_couplet_cluster_radius as usize;
            let counts = box_count(&seeds, rows, cols, radius, radius);
            let clustered: Vec<bool> = seeds
                .iter()
                .zip(&counts)
                .map(|(&s, &c)| s && c >= self.params.tds_couplet_min_seeds)
                .collect();
            self.couplet_seed_gates = clustered.iter().filter(|&&c| c).count() as i64;
            let plane = if !clustered.iter().any(|&c| c) {
                clustered
            } else {
                box_count(
                    &clustered,
                    rows,
                    cols,
                    self.params.tds_rotation_radius_radials as usize,
                    self.params.tds_rotation_radius_gates as usize,
                )
                .into_iter()
                .map(|c| c > 0)
                .collect()
            };
            self.rotation_plane = Some((plane, rows, cols));
        }
        let (plane, plane_rows, plane_cols) = self.rotation_plane.as_ref().unwrap();
        let index = velocity.gate_index(ranges);
        let rows = (*plane_rows).min(target_rows);
        let mut sampled = vec![false; target_rows * n];
        for r in 0..rows {
            for (k, gate) in index.iter().enumerate() {
                if let Some(g) = gate {
                    sampled[r * n + k] = plane[r * plane_cols + g];
                }
            }
        }
        sampled
    }

    /// `drop_mask`: `(radials, len(ranges))`, true where the gate drops.
    pub fn drop_mask(&mut self, product: &str, velocity_product: bool, ranges: &[f64]) -> Vec<bool> {
        let p = self.params.clone();
        let rho = Self::sample(&self.rho, ranges, &self.rho_rows);
        let identity: Vec<i64> = (0..self.reflectivity.rows as i64).collect();
        let z = Self::sample(&self.reflectivity, ranges, &identity);
        let cells = rho.len().min(z.len());
        let has_rho: Vec<bool> = rho.iter().map(|v| v.is_finite()).collect();
        let shielded: Vec<bool> = z.iter().map(|&v| v >= p.ref_shield_dbz).collect();
        let mut entry = MomentMasks::default();
        let (mut primary, primary_reason) = if velocity_product {
            let threshold = p.velocity_threshold();
            let primary: Vec<bool> = (0..cells).map(|k| has_rho[k] && rho[k] < threshold).collect();
            (primary, DROP_LOW_RHO)
        } else {
            let primary: Vec<bool> = (0..cells)
                .map(|k| has_rho[k] && rho[k] < p.rho_min && !shielded[k])
                .collect();
            (primary, DROP_LOW_RHO_BELOW_SHIELD)
        };
        if velocity_product && p.tds_fringe_exempt {
            primary = self.apply_fringe_exemption(&mut entry, primary, &rho, &z, ranges);
        }
        let mut drop = primary.clone();
        entry.reasons.push((primary_reason, primary));
        if let Some(floor) = p.rho_floor {
            let floor_only: Vec<bool> = (0..cells)
                .map(|k| has_rho[k] && rho[k] < floor && !entry.reasons[0].1[k])
                .collect();
            for k in 0..cells {
                drop[k] = drop[k] || floor_only[k];
            }
            entry.reasons.push((DROP_RHO_FLOOR, floor_only));
        }
        entry.shielded = shielded;
        let tested = has_rho.iter().filter(|&&h| h).count() as i64;
        *self.gates_tested.entry(product.to_string()).or_insert(0) += tested;
        *self.gates_rho_missing.entry(product.to_string()).or_insert(0) +=
            has_rho.len() as i64 - tested;
        self.masks.insert(product.to_string(), entry);
        drop
    }

    fn apply_fringe_exemption(
        &mut self,
        entry: &mut MomentMasks,
        would_drop: Vec<bool>,
        rho: &[f64],
        z: &[f64],
        ranges: &[f64],
    ) -> Vec<bool> {
        let p = &self.params;
        let cells = would_drop.len();
        let above_floor: Vec<bool> = (0..cells).map(|k| would_drop[k] && rho[k] >= p.tds_rho_floor).collect();
        let in_band: Vec<bool> = (0..cells).map(|k| above_floor[k] && z[k] >= p.tds_ref_min_dbz).collect();
        let fringe: Vec<bool> = (0..cells).map(|k| in_band[k] && z[k] < p.ref_shield_dbz).collect();
        let rotation = if fringe.iter().any(|&f| f) {
            self.rotation_at(ranges)
        } else {
            vec![false; cells]
        };
        let exempt: Vec<bool> = (0..cells).map(|k| fringe[k] && rotation[k]).collect();
        entry.turned_away = Some(vec![
            (TDS_RHO_BELOW_FLOOR, (0..cells).map(|k| would_drop[k] && !above_floor[k]).collect()),
            (TDS_BELOW_Z_BAND, (0..cells).map(|k| above_floor[k] && !in_band[k]).collect()),
            (TDS_ABOVE_Z_BAND, (0..cells).map(|k| in_band[k] && !fringe[k]).collect()),
            (TDS_NO_ROTATION, (0..cells).map(|k| fringe[k] && !rotation[k]).collect()),
        ]);
        let surviving = (0..cells).map(|k| would_drop[k] && !exempt[k]).collect();
        entry.exempt = Some(exempt);
        surviving
    }

    /// `count_block` over a whole moment at once (the reference's per-block
    /// sums add to this): `finite` is the moment's finite-before-masking
    /// plane.  Returns the gates exempted.  `touched` is false when the
    /// moment had no radials, in which case the reference never called it.
    pub fn count_block(&mut self, product: &str, finite: &[bool], touched: bool) -> i64 {
        if !touched {
            return 0;
        }
        let Some(entry) = self.masks.get(product) else { return 0 };
        if let Some(turned) = &entry.turned_away {
            let per = self.gates_tds_turned_away.entry(product.to_string()).or_default();
            for (criterion, mask) in turned {
                let n = finite.iter().zip(mask).filter(|(f, m)| **f && **m).count() as i64;
                *per.entry(criterion.to_string()).or_insert(0) += n;
            }
        }
        let Some(exempt) = &entry.exempt else { return 0 };
        let saved = finite.iter().zip(exempt).filter(|(f, e)| **f && **e).count() as i64;
        *self.gates_exempt_tds_fringe.entry(product.to_string()).or_insert(0) += saved;
        saved
    }

    /// `count_dropped` over a whole moment: `hits` is dropped-and-finite.
    /// The reference only calls it for blocks with at least one hit, so
    /// the keys appear exactly when the moment's total is non-zero.
    pub fn count_dropped(&mut self, product: &str, hits: &[bool]) -> i64 {
        let dropped = hits.iter().filter(|&&h| h).count() as i64;
        if dropped == 0 {
            return 0;
        }
        *self.gates_dropped.entry(product.to_string()).or_insert(0) += dropped;
        let entry = self.masks.get(product).expect("drop_mask ran for this moment");
        let per = self.gates_dropped_reason.entry(product.to_string()).or_default();
        for (reason, mask) in &entry.reasons {
            let n = hits.iter().zip(mask).filter(|(h, m)| **h && **m).count() as i64;
            *per.entry(reason.to_string()).or_insert(0) += n;
        }
        let shielded = hits.iter().zip(&entry.shielded).filter(|(h, s)| **h && **s).count() as i64;
        *self.gates_dropped_shielded_z.entry(product.to_string()).or_insert(0) += shielded;
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_count_wraps_rows_and_clips_gates() {
        // 3 x 4, one true cell at (0, 0)
        let mut mask = vec![false; 12];
        mask[0] = true;
        let counts = box_count(&mask, 3, 4, 1, 1);
        // rows 2, 0, 1 all see row 0 within one radial (wrap); gates 0..=1
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[2 * 4], 1);
        assert_eq!(counts[4], 1);
        // a box taller than the sweep revisits rows: 3 rows, half 3 -> 7 offsets
        let counts = box_count(&mask, 3, 4, 3, 0);
        assert_eq!(counts[0], 3); // offsets -3, 0, 3 land on row 0
    }

    #[test]
    fn nearest_rows_are_circular() {
        let rows = nearest_azimuth_rows(&[359.9, 0.2, 180.0], &[0.0, 90.0, 270.0], 1.0);
        assert_eq!(rows, vec![0, 0, -1]);
    }
}
