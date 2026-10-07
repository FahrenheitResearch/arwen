//! Plan construction and application: the Rust half of
//! `gpuwm/verify/obs/regrid.py`.
//!
//! The plan-as-data discipline is seeded from `rustwx-regrid::plan` in
//! the project's consolidated Rust -- a remap between two fixed grids is a
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
    /// Reverse assignment of extensive quantities. Empty cells are valid zero.
    CellSum,
    /// Row-major n by n partition centres in a regular latitude/longitude cell.
    CellSumSplit { n: usize },
}

impl Method {
    pub fn from_code(code: u32) -> Result<Self, RegridError> {
        match code {
            0 => Ok(Method::Nearest),
            1 => Ok(Method::CellAverage),
            2 => Ok(Method::CellSum),
            other => Err(RegridError::InvalidOptions(format!(
                "unknown remap method code {other}; expected 0 (nearest) or \
                 1 (cell_average) or 2 (cell_sum)"
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

pub(crate) fn split_count(method: Method) -> Result<usize, RegridError> {
    let n = match method { Method::CellSumSplit { n } => n, _ => 1 };
    if n == 0 || n.checked_mul(n).is_none() {
        return Err(RegridError::InvalidOptions("cell-sum split n must be positive and its square fit usize; otherwise sub-point weights are undefined".into()));
    }
    Ok(n)
}

/// One caller array and one temporary Rust array coexist at the seam.
/// Bound indices to 4 GiB each before either allocation, preventing a full
/// continental fine-grid partition from exhausting host memory.
pub const MAX_SUM_INDEX_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Coordinates may be stored as float32. Tolerance admits their quantization,
/// but not an irregular grid whose partition centres would be misplaced.
pub(crate) fn regular_spacing(lat: &[f64], lon: &[f64], shape: (usize, usize)) -> Result<(f64, f64), RegridError> {
    let refuse = || RegridError::InvalidGrid("cell_sum_split refuses an irregular source grid: regular 2-D latitude/longitude spacing is required to locate sub-points".into());
    if shape.0 < 2 || shape.1 < 2 { return Err(refuse()); }
    let dy = (lat[(shape.0-1)*shape.1] - lat[0]) / (shape.0-1) as f64;
    let dx = (lon[shape.1-1] - lon[0]) / (shape.1-1) as f64;
    if dy == 0.0 || dx == 0.0 { return Err(refuse()); }
    for j in 0..shape.0 { for i in 0..shape.1 {
        let k = j*shape.1+i;
        let expected_lat = lat[0]+j as f64*dy;
        let expected_lon = lon[0]+i as f64*dx;
        // One coordinate rounding plus rounded endpoints used for spacing.
        // Two stored-f32 ULPs admit that quantization, not a curvilinear grid.
        let tolerance = |v: f64| {
            let stored = v.abs() as f32;
            2.0 * (f32::from_bits(stored.to_bits()+1) as f64 - stored as f64) + 1e-12
        };
        if !lat[k].is_finite() || !lon[k].is_finite()
            || (lat[k] - expected_lat).abs() > tolerance(expected_lat)
            || (lon[k] - expected_lon).abs() > tolerance(expected_lon) { return Err(refuse()); }
    }}
    Ok((dy, dx))
}

/// Serial source-major accumulation. Receipt order: finite source mass,
/// remapped mass, unreachable valid mass, masked finite mass.
/// Invalid contributors mark every destination they touch invalid; valid mass
/// still accumulates there. No contributor means a confident emission zero.
pub fn apply_sum(method: Method, index: &[i64], values: &[f64], valid: &[bool],
                 out: &mut [f64], out_valid: &mut [bool]) -> Result<[f64; 4], RegridError> {
    if !matches!(method, Method::CellSum | Method::CellSumSplit { .. }) {
        return Err(RegridError::InvalidOptions("apply_sum requires a cell-sum method; an observation plan has a different index direction".into()));
    }
    let n = split_count(method)?;
    let parts = n*n;
    if valid.len() != values.len() || out.len() != out_valid.len()
        || values.len().checked_mul(parts) != Some(index.len()) {
        return Err(RegridError::ShapeMismatch("cell-sum buffers do not match source/sub-point/destination counts; refusing out-of-bounds mass assignment".into()));
    }
    out.fill(0.0); out_valid.fill(true);
    let mut receipt = [0.0; 4];
    for source in 0..values.len() {
        let value = values[source];
        if value.is_finite() { receipt[0] += value; }
        let usable = valid[source] && value.is_finite();
        if !usable && value.is_finite() { receipt[3] += value; }
        let piece = value / parts as f64;
        for sub in 0..parts {
            let target = index[source*parts+sub];
            if target < -1 || target >= out.len() as i64 {
                return Err(RegridError::ShapeMismatch("cell-sum destination index is outside the grid; refusing out-of-bounds mass assignment".into()));
            }
            if target == -1 { if usable { receipt[2] += piece; } }
            else if usable { out[target as usize] += piece; }
            else { out_valid[target as usize] = false; }
        }
    }
    for value in out { receipt[1] += *value; }
    Ok(receipt)
}

/// Apply a cell-sum plan to a fire-INTENSITY field (fire radiative power):
/// every destination cell touched by at least one of a source cell's
/// sub-points receives that source cell's WHOLE value, once, and values from
/// different source cells landing in one destination still add.
///
/// Why this is not [`apply_sum`].  Fire power drives plume rise through the
/// fire's size (Freitas burnt area is proportional to FRP), so partitioning a
/// 3 km pixel's FRP into n*n pieces on a finer grid turns one fire into n*n
/// fires of 1/n^2 the power and lowers every plume (at 750 m, n = 5: 25 fires
/// of 4 % power).  The emitted MASS is partitioned by [`apply_sum`]; the power
/// each column's plume sees is the pixel's.  With n = 1 the two operators are
/// the same sum.  The destination total is therefore NOT the source total when
/// n > 1: receipt[1] is the touched total, reported, never compared.
///
/// Receipt: finite source total, touched destination total, unreachable
/// (valid sources none of whose sub-points reached a destination), masked.
pub fn apply_touch_sum(method: Method, index: &[i64], values: &[f64], valid: &[bool],
                       out: &mut [f64], out_valid: &mut [bool]) -> Result<[f64; 4], RegridError> {
    if !matches!(method, Method::CellSum | Method::CellSumSplit { .. }) {
        return Err(RegridError::InvalidOptions("apply_touch_sum requires a cell-sum method; an observation plan has a different index direction".into()));
    }
    let n = split_count(method)?;
    let parts = n * n;
    if valid.len() != values.len() || out.len() != out_valid.len()
        || values.len().checked_mul(parts) != Some(index.len()) {
        return Err(RegridError::ShapeMismatch("cell-sum buffers do not match source/sub-point/destination counts; refusing out-of-bounds assignment".into()));
    }
    out.fill(0.0);
    out_valid.fill(true);
    let mut receipt = [0.0; 4];
    let mut touched: Vec<usize> = Vec::with_capacity(parts);
    for source in 0..values.len() {
        let value = values[source];
        if value.is_finite() {
            receipt[0] += value;
        }
        let usable = valid[source] && value.is_finite();
        if !usable && value.is_finite() {
            receipt[3] += value;
        }
        touched.clear();
        for sub in 0..parts {
            let target = index[source * parts + sub];
            if target < -1 || target >= out.len() as i64 {
                return Err(RegridError::ShapeMismatch("cell-sum destination index is outside the grid; refusing out-of-bounds assignment".into()));
            }
            if target >= 0 && !touched.contains(&(target as usize)) {
                touched.push(target as usize);
            }
        }
        if touched.is_empty() {
            if usable {
                receipt[2] += value;
            }
            continue;
        }
        // Destinations in first-touch (row-major sub-point) order: the sum
        // into each destination is still ascending source order, serial.
        for &target in &touched {
            if usable {
                out[target] += value;
            } else {
                out_valid[target] = false;
            }
        }
    }
    for value in out.iter() {
        receipt[1] += *value;
    }
    Ok(receipt)
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
    if matches!(method, Method::CellSum | Method::CellSumSplit { .. }) {
        for (lat, lon) in source_latitude.iter().zip(source_longitude)
            .chain(destination_latitude.iter().zip(destination_longitude)) {
            if !lat.is_finite() || lat.abs() > 90.0 || !lon.is_finite() {
                return Err(RegridError::InvalidGrid("cell-sum coordinates must be finite with latitude in [-90,90]; invalid geometry would assign emission mass to wrong cells".into()));
            }
        }
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
        Method::CellAverage | Method::CellSum | Method::CellSumSplit { .. } => {
            let n = split_count(method)?;
            let spacing = if matches!(method, Method::CellSumSplit { .. }) {
                Some(regular_spacing(source_latitude, source_longitude, source_shape)?)
            } else { None };
            let destination_count = destination_points.len();
            let tree = KdTree::build(destination_points);
            let index_count = source_points.len().checked_mul(n * n).ok_or_else(||
                RegridError::InvalidOptions("cell-sum split plan size overflows; cannot allocate destination indices".into()))?;
            if matches!(method, Method::CellSum | Method::CellSumSplit { .. })
                && index_count > MAX_SUM_INDEX_BYTES / 8 {
                return Err(RegridError::InvalidOptions("cell-sum destination indices exceed 4 GiB per buffer; caller and Rust copies would exhaust the supported host-memory budget".into()));
            }
            let mut source_index = vec![-1i64; index_count];
            let mut reachable = vec![false; destination_count];
            let mut largest_squared = f64::NEG_INFINITY;
            for (source, centre) in source_points.iter().enumerate() {
              // Arc displacement cannot exceed the latitude plus longitude
              // displacement. Reject a whole far cell before partitioning it.
              if let Some((dy, dx)) = spacing {
                  let radius = crate::geometry::EARTH_RADIUS_M * (dy.abs()+dx.abs()).to_radians()/2.0;
                  let expanded = chord_from_arc((max_distance_m+radius).min(std::f64::consts::PI*crate::geometry::EARTH_RADIUS_M))?;
                  if tree.nearest(*centre, expanded*expanded).is_none() { continue; }
              }
              for sub in 0..n*n {
                let query = if let Some((dy, dx)) = spacing {
                    let lat = source_latitude[source] + ((sub / n) as f64 + 0.5 - n as f64 / 2.0) * dy / n as f64;
                    let lon = source_longitude[source] + ((sub % n) as f64 + 0.5 - n as f64 / 2.0) * dx / n as f64;
                    unit_vectors(&[lat], &[lon])?[0]
                } else { *centre };
                if let Some(found) = tree.nearest(query, bound_squared) {
                    source_index[source*n*n + sub] = found.index as i64;
                    reachable[found.index] = true;
                    if found.distance_squared > largest_squared {
                        largest_squared = found.distance_squared;
                    }
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
        Method::CellSum | Method::CellSumSplit { .. } => {
            Err(RegridError::InvalidOptions("cell-sum callers must use apply_sum and retain its receipt; apply_plan cannot report unreachable emission mass".into()))
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

    #[test]
    fn cell_sum_identity_empty_invalid_and_unreachable_mass() {
        let (lat, lon) = ramp(2, 2, 30.0, 100.0, 0.03);
        let plan = build_plan(Method::CellSum, &lat, &lon, (2,2),
                              &[30.0,30.0], &[100.0,101.0], (1,2), 100.0).unwrap();
        let mut out = [0.0;2]; let mut valid = [false;2];
        let receipt = apply_sum(Method::CellSum, &plan.source_index,
                                &[8.0,4.0,2.0,1.0], &[true;4], &mut out, &mut valid).unwrap();
        assert_eq!(out, [8.0,0.0]); assert_eq!(valid, [true,true]);
        assert_eq!(receipt, [15.0,8.0,7.0,0.0]);
        let receipt = apply_sum(Method::CellSum, &[0,0,1,-1],
                                &[8.0,4.0,2.0,1.0], &[true,false,true,true], &mut out, &mut valid).unwrap();
        assert_eq!(out, [8.0,2.0]); assert_eq!(valid, [false,true]);
        assert_eq!(receipt, [15.0,10.0,1.0,4.0]);
        let plan = build_plan(Method::CellSum, &lat, &lon, (2,2), &lat, &lon, (2,2), 1.0).unwrap();
        let mut out = [0.0;4]; let mut valid = [false;4];
        assert_eq!(apply_sum(Method::CellSum, &plan.source_index, &[1.0,2.0,3.0,4.0], &[true;4], &mut out, &mut valid).unwrap(), [10.0,10.0,0.0,0.0]);
        assert_eq!(out, [1.0,2.0,3.0,4.0]);
    }

    #[test]
    fn cell_sum_split_three_km_onto_750m_and_500m() {
        let (lat, lon) = ramp(2,2,0.0,100.0,0.027);
        for n in [4,6] {
            let step = 0.027/n as f64;
            let (dlat, dlon) = ramp(2*n,2*n,-0.0135+step/2.0,100.0-0.0135+step/2.0,step);
            let method = Method::CellSumSplit { n };
            let plan = build_plan(method,&lat,&lon,(2,2),&dlat,&dlon,(2*n,2*n),20.0).unwrap();
            let mut out = vec![0.0;4*n*n]; let mut valid = vec![false;out.len()];
            let receipt = apply_sum(method,&plan.source_index,&[36.0;4],&[true;4],&mut out,&mut valid).unwrap();
            assert_eq!(receipt,[144.0,144.0,0.0,0.0]);
            assert!(out.iter().all(|v| *v == 36.0/(n*n) as f64));
            assert!(valid.iter().all(|v| *v));
        }
    }

    #[test]
    fn touch_sum_gives_each_touched_cell_the_whole_pixel_and_still_adds_pixels() {
        // One 3 km pixel onto a 2 x 2 fine grid with n = 2: every fine cell
        // is touched once and sees the pixel's whole power, where the
        // partitioning sum would give each a quarter.
        let index = [0i64, 1, 2, 3];
        let mut out = [0.0; 4];
        let mut valid = [false; 4];
        let receipt = apply_touch_sum(Method::CellSumSplit { n: 2 }, &index, &[40.0], &[true],
                                      &mut out, &mut valid).unwrap();
        assert_eq!(out, [40.0; 4]);
        assert_eq!(receipt, [40.0, 160.0, 0.0, 0.0]);
        // Two pixels whose sub-points share destination 1 add there; a
        // pixel whose sub-points all miss is unreachable; a pixel touching
        // one destination twice counts once.
        let index = [0i64, 1, 1, 1, -1, -1, -1, -1, 1, 2, 3, 3];
        let mut out = [0.0; 4];
        let mut valid = [false; 4];
        let receipt = apply_touch_sum(Method::CellSumSplit { n: 2 }, &index, &[5.0, 7.0, 11.0],
                                      &[true, true, true], &mut out, &mut valid).unwrap();
        assert_eq!(out, [5.0, 16.0, 11.0, 11.0]);
        assert_eq!(receipt, [23.0, 43.0, 7.0, 0.0]);
        // With n = 1 the touch sum is the partitioning sum.
        let index = [0i64, 0, 1];
        let values = [1.5, 2.5, 4.0];
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        let mut va = [false; 2];
        let mut vb = [false; 2];
        let ra = apply_touch_sum(Method::CellSum, &index, &values, &[true; 3], &mut a, &mut va).unwrap();
        let rb = apply_sum(Method::CellSum, &index, &values, &[true; 3], &mut b, &mut vb).unwrap();
        assert_eq!(a, b);
        assert_eq!(ra, rb);
        assert_eq!(va, vb);
    }

    #[test]
    fn cell_sum_split_refuses_irregular_grid_and_reports_partial_mass() {
        let (mut lat, lon) = ramp(2,2,30.0,100.0,0.03);
        lat[3] += 0.001;
        assert!(build_plan(Method::CellSumSplit { n:2 },&lat,&lon,(2,2),&[30.0],&[100.0],(1,1),100.0).unwrap_err().to_string().contains("irregular source grid"));
        let mut out = [0.0]; let mut valid = [false];
        let receipt = apply_sum(Method::CellSumSplit { n:2 },&[0,-1,0,-1],&[8.0],&[true],&mut out,&mut valid).unwrap();
        assert_eq!(receipt,[8.0,4.0,4.0,0.0]);
    }

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
