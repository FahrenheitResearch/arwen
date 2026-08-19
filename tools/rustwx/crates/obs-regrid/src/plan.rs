//! Plan construction and application: the Rust half of
//! `gpuwm/verify/obs/regrid.py`.
//!
//! The plan-as-data discipline is seeded from `rustwx-regrid::plan` in
//! Drew's consolidated Rust -- a remap between two fixed grids is a
//! fixed mapping, built once and applied many times, and applying it
//! writes into a caller-owned buffer.  The OPERATORS are gpuwm's, and
//! they are not the ones that crate carries:
//!
//! * `nearest` is scattered-point (curvilinear observation swaths, not
//!   a regular lat/lon spec), so it is a k-d tree over unit vectors
//!   rather than an index arithmetic shortcut, and its bound is a
//!   chord on the unit sphere rather than haversine kilometres.
//! * `cell_average` is a REVERSE assignment -- each source cell is
//!   given to its nearest destination centre, then each destination
//!   cell averages what landed on it -- not an area-overlap
//!   conservative remap.
//! * validity is an explicit boolean field remapped WITH the values,
//!   not a NaN sentinel and a missing policy.  A destination cell built
//!   only from invalid sources is invalid; missing does not become zero
//!   at a grid change.

use crate::error::RegridError;
use crate::geometry::{arc_from_chord, chord_from_arc, unit_vectors};
use crate::kdtree::KdTree;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Every destination cell takes the value of the nearest source
    /// cell centre.  `source_index` is per DESTINATION cell.
    Nearest,
    /// Every destination cell takes the mean of the source cells whose
    /// nearest destination centre is this one.  `source_index` is per
    /// SOURCE cell, and -1 means "assigned to no destination".
    CellAverage,
}

impl Method {
    pub fn from_code(code: u32) -> Result<Self, RegridError> {
        match code {
            0 => Ok(Method::Nearest),
            1 => Ok(Method::CellAverage),
            other => Err(RegridError::InvalidOptions(format!(
                "unknown remap method code {other}; expected 0 (nearest) or \
                 1 (cell_average)"
            ))),
        }
    }
}

/// A fixed integer mapping from one grid to another.
#[derive(Clone, Debug)]
pub struct RegridPlan {
    pub method: Method,
    /// Per destination cell for [`Method::Nearest`], per source cell for
    /// [`Method::CellAverage`].
    pub source_index: Vec<i64>,
    /// The destination mask the distance bound leaves usable, in either
    /// case.
    pub reachable: Vec<bool>,
    pub destination_shape: (usize, usize),
    pub source_shape: (usize, usize),
    pub max_distance_m: f64,
    pub max_used_distance_m: f64,
}

fn cells(shape: (usize, usize)) -> usize {
    shape.0 * shape.1
}

/// Compute the remap once, for reuse across every arm of a case.
pub fn build_plan(
    method: Method,
    source_latitude: &[f64],
    source_longitude: &[f64],
    source_shape: (usize, usize),
    destination_latitude: &[f64],
    destination_longitude: &[f64],
    destination_shape: (usize, usize),
    max_distance_m: f64,
) -> Result<RegridPlan, RegridError> {
    if source_latitude.len() != cells(source_shape)
        || source_longitude.len() != cells(source_shape)
    {
        return Err(RegridError::InvalidGrid(String::from(
            "the source latitude/longitude arrays do not fill the source shape",
        )));
    }
    if destination_latitude.len() != cells(destination_shape)
        || destination_longitude.len() != cells(destination_shape)
    {
        return Err(RegridError::InvalidGrid(String::from(
            "the destination latitude/longitude arrays do not fill the \
             destination shape",
        )));
    }
    let source_points = unit_vectors(source_latitude, source_longitude)?;
    let destination_points = unit_vectors(destination_latitude, destination_longitude)?;
    let bound = chord_from_arc(max_distance_m)?;
    // scipy squares the bound once and compares squared distances; the
    // rounding of that single multiply is part of the predicate, so it
    // happens here rather than inside the tree.
    let bound_squared = bound * bound;

    match method {
        Method::Nearest => {
            let tree = KdTree::build(source_points);
            let mut source_index = vec![0i64; destination_points.len()];
            let mut reachable = vec![false; destination_points.len()];
            let mut largest_squared = f64::NEG_INFINITY;
            for (slot, query) in destination_points.iter().enumerate() {
                if let Some(found) = tree.nearest(*query, bound_squared) {
                    source_index[slot] = found.index as i64;
                    reachable[slot] = true;
                    if found.distance_squared > largest_squared {
                        largest_squared = found.distance_squared;
                    }
                }
            }
            let max_used_distance_m = if largest_squared.is_finite() {
                arc_from_chord(largest_squared.sqrt())
            } else {
                0.0
            };
            Ok(RegridPlan {
                method,
                source_index,
                reachable,
                destination_shape,
                source_shape,
                max_distance_m,
                max_used_distance_m,
            })
        }
        Method::CellAverage => {
            let destination_count = destination_points.len();
            let tree = KdTree::build(destination_points);
            let mut source_index = vec![-1i64; source_points.len()];
            let mut reachable = vec![false; destination_count];
            let mut largest_squared = f64::NEG_INFINITY;
            for (slot, query) in source_points.iter().enumerate() {
                if let Some(found) = tree.nearest(*query, bound_squared) {
                    source_index[slot] = found.index as i64;
                    reachable[found.index] = true;
                    if found.distance_squared > largest_squared {
                        largest_squared = found.distance_squared;
                    }
                }
            }
            let max_used_distance_m = if largest_squared.is_finite() {
                arc_from_chord(largest_squared.sqrt())
            } else {
                0.0
            };
            Ok(RegridPlan {
                method,
                source_index,
                reachable,
                destination_shape,
                source_shape,
                max_distance_m,
                max_used_distance_m,
            })
        }
    }
}

/// Remap a field and its validity together.
///
/// Values under a false destination mask are zero and must not be read;
/// the mask is the answer to "is there an observation here", and the
/// scorer asks it.
///
/// `out_values` and `out_valid` are caller-owned and sized to the
/// destination grid, which is what lets the ctypes seam write straight
/// into preallocated numpy buffers.
pub fn apply_plan(
    method: Method,
    source_index: &[i64],
    reachable: &[bool],
    source_shape: (usize, usize),
    destination_shape: (usize, usize),
    values: &[f64],
    valid: &[bool],
    out_values: &mut [f64],
    out_valid: &mut [bool],
) -> Result<(), RegridError> {
    let source_cells = cells(source_shape);
    let destination_cells = cells(destination_shape);
    if values.len() != source_cells || valid.len() != source_cells {
        return Err(RegridError::ShapeMismatch(format!(
            "field length {} does not match the plan's source grid \
             ({}, {})",
            values.len(),
            source_shape.0,
            source_shape.1
        )));
    }
    if reachable.len() != destination_cells
        || out_values.len() != destination_cells
        || out_valid.len() != destination_cells
    {
        return Err(RegridError::ShapeMismatch(String::from(
            "the plan's reachability mask and the output buffers must fill \
             the destination grid",
        )));
    }

    match method {
        Method::Nearest => {
            if source_index.len() != destination_cells {
                return Err(RegridError::ShapeMismatch(String::from(
                    "a nearest plan indexes one source cell per DESTINATION \
                     cell",
                )));
            }
            for slot in 0..destination_cells {
                let picked = source_index[slot];
                if picked < 0 || picked as usize >= source_cells {
                    return Err(RegridError::ShapeMismatch(format!(
                        "the plan points destination cell {slot} at source \
                         cell {picked}, which is outside the source grid"
                    )));
                }
                let picked = picked as usize;
                let is_valid = valid[picked] && reachable[slot];
                out_valid[slot] = is_valid;
                out_values[slot] = if is_valid { values[picked] } else { 0.0 };
            }
            Ok(())
        }
        Method::CellAverage => {
            if source_index.len() != source_cells {
                return Err(RegridError::ShapeMismatch(String::from(
                    "a cell_average plan indexes one destination cell per \
                     SOURCE cell",
                )));
            }
            let mut totals = vec![0.0f64; destination_cells];
            let mut counts = vec![0i64; destination_cells];
            // Ascending source flat order, because that is the order
            // `numpy.add.at` accumulates in and float64 addition is not
            // associative: a different order is a different sum in the
            // last bits, and the parity contract on this port is bitwise.
            // This loop is deliberately NOT parallel for the same reason.
            for slot in 0..source_cells {
                let target = source_index[slot];
                if target < 0 || !valid[slot] {
                    continue;
                }
                let target = target as usize;
                if target >= destination_cells {
                    return Err(RegridError::ShapeMismatch(format!(
                        "the plan points source cell {slot} at destination \
                         cell {target}, which is outside the destination grid"
                    )));
                }
                totals[target] += values[slot];
                counts[target] += 1;
            }
            for slot in 0..destination_cells {
                if counts[slot] > 0 {
                    out_valid[slot] = true;
                    out_values[slot] = totals[slot] / counts[slot] as f64;
                } else {
                    out_valid[slot] = false;
                    out_values[slot] = 0.0;
                }
            }
            Ok(())
        }
    }
}

/// The count the plan's receipt reports.
pub fn unreachable_destination_cells(reachable: &[bool]) -> usize {
    reachable.iter().filter(|value| !**value).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(ny: usize, nx: usize, lat0: f64, lon0: f64, step: f64) -> (Vec<f64>, Vec<f64>) {
        let mut lat = Vec::with_capacity(ny * nx);
        let mut lon = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            for i in 0..nx {
                lat.push(lat0 + j as f64 * step);
                lon.push(lon0 + i as f64 * step);
            }
        }
        (lat, lon)
    }

    #[test]
    fn nearest_onto_the_same_grid_is_the_identity() {
        let (lat, lon) = ramp(4, 5, 35.0, -100.0, 0.1);
        let plan = build_plan(
            Method::Nearest,
            &lat,
            &lon,
            (4, 5),
            &lat,
            &lon,
            (4, 5),
            50_000.0,
        )
        .unwrap();
        assert!(plan.reachable.iter().all(|&value| value));
        let values: Vec<f64> = (0..20).map(|v| v as f64 * 1.5).collect();
        let valid = vec![true; 20];
        let mut out_values = vec![0.0; 20];
        let mut out_valid = vec![false; 20];
        apply_plan(
            Method::Nearest,
            &plan.source_index,
            &plan.reachable,
            (4, 5),
            (4, 5),
            &values,
            &valid,
            &mut out_values,
            &mut out_valid,
        )
        .unwrap();
        assert_eq!(out_values, values);
        assert!(out_valid.iter().all(|&value| value));
        assert_eq!(plan.max_used_distance_m, 0.0);
    }

    #[test]
    fn a_destination_outside_the_bound_is_invalid_not_borrowed() {
        // One source point in Kansas, one destination beside it and one
        // a thousand kilometres away.  Without the bound the far cell
        // silently borrows the near observation and gets scored.
        let plan = build_plan(
            Method::Nearest,
            &[38.0],
            &[-98.0],
            (1, 1),
            &[38.0, 38.0],
            &[-98.0, -85.0],
            (1, 2),
            50_000.0,
        )
        .unwrap();
        assert_eq!(plan.reachable, vec![true, false]);
        let mut out_values = vec![9.0; 2];
        let mut out_valid = vec![true; 2];
        apply_plan(
            Method::Nearest,
            &plan.source_index,
            &plan.reachable,
            (1, 1),
            (1, 2),
            &[7.5],
            &[true],
            &mut out_values,
            &mut out_valid,
        )
        .unwrap();
        assert_eq!(out_values, vec![7.5, 0.0]);
        assert_eq!(out_valid, vec![true, false]);
        assert_eq!(unreachable_destination_cells(&plan.reachable), 1);
    }

    #[test]
    fn an_invalid_source_does_not_become_a_confident_zero() {
        let plan = build_plan(
            Method::Nearest,
            &[38.0, 38.1],
            &[-98.0, -98.0],
            (2, 1),
            &[38.0, 38.1],
            &[-98.0, -98.0],
            (2, 1),
            50_000.0,
        )
        .unwrap();
        let mut out_values = vec![0.0; 2];
        let mut out_valid = vec![false; 2];
        apply_plan(
            Method::Nearest,
            &plan.source_index,
            &plan.reachable,
            (2, 1),
            (2, 1),
            &[7.5, 3.25],
            &[true, false],
            &mut out_values,
            &mut out_valid,
        )
        .unwrap();
        assert_eq!(out_valid, vec![true, false]);
        assert_eq!(out_values, vec![7.5, 0.0]);
    }

    #[test]
    fn cell_average_takes_the_mean_of_what_landed_in_it() {
        // Four source cells inside one destination cell's footprint.
        let (source_lat, source_lon) = ramp(2, 2, 38.0, -98.0, 0.01);
        let plan = build_plan(
            Method::CellAverage,
            &source_lat,
            &source_lon,
            (2, 2),
            &[38.005],
            &[-97.995],
            (1, 1),
            50_000.0,
        )
        .unwrap();
        assert_eq!(plan.source_index, vec![0, 0, 0, 0]);
        assert_eq!(plan.reachable, vec![true]);
        let mut out_values = vec![0.0; 1];
        let mut out_valid = vec![false; 1];
        apply_plan(
            Method::CellAverage,
            &plan.source_index,
            &plan.reachable,
            (2, 2),
            (1, 1),
            &[1.0, 2.0, 3.0, 4.0],
            &[true; 4],
            &mut out_values,
            &mut out_valid,
        )
        .unwrap();
        assert_eq!(out_values, vec![2.5]);
        assert_eq!(out_valid, vec![true]);
    }

    #[test]
    fn cell_average_ignores_invalid_contributors_and_marks_empty_cells() {
        let (source_lat, source_lon) = ramp(2, 2, 38.0, -98.0, 0.01);
        let plan = build_plan(
            Method::CellAverage,
            &source_lat,
            &source_lon,
            (2, 2),
            &[38.005, 39.5],
            &[-97.995, -97.995],
            (1, 2),
            50_000.0,
        )
        .unwrap();
        let mut out_values = vec![0.0; 2];
        let mut out_valid = vec![true; 2];
        apply_plan(
            Method::CellAverage,
            &plan.source_index,
            &plan.reachable,
            (2, 2),
            (1, 2),
            &[1.0, 2.0, 3.0, 100.0],
            &[true, true, true, false],
            &mut out_values,
            &mut out_valid,
        )
        .unwrap();
        assert_eq!(out_values, vec![2.0, 0.0]);
        assert_eq!(out_valid, vec![true, false]);
    }

    #[test]
    fn a_field_that_does_not_match_the_plan_is_refused_by_name() {
        let plan = build_plan(
            Method::Nearest,
            &[38.0],
            &[-98.0],
            (1, 1),
            &[38.0],
            &[-98.0],
            (1, 1),
            50_000.0,
        )
        .unwrap();
        let mut out_values = vec![0.0; 1];
        let mut out_valid = vec![false; 1];
        let error = apply_plan(
            Method::Nearest,
            &plan.source_index,
            &plan.reachable,
            (1, 1),
            (1, 1),
            &[1.0, 2.0],
            &[true, true],
            &mut out_values,
            &mut out_valid,
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not match the plan"));
    }

    #[test]
    fn an_unknown_method_code_is_refused_by_name() {
        let error = Method::from_code(7).unwrap_err();
        assert!(error.to_string().contains("unknown remap method code 7"));
    }

    #[test]
    fn a_plan_that_reaches_nothing_reports_a_zero_used_distance() {
        let plan = build_plan(
            Method::Nearest,
            &[38.0],
            &[-98.0],
            (1, 1),
            &[-38.0],
            &[98.0],
            (1, 1),
            1_000.0,
        )
        .unwrap();
        assert_eq!(plan.reachable, vec![false]);
        assert_eq!(plan.max_used_distance_m, 0.0);
    }
}
