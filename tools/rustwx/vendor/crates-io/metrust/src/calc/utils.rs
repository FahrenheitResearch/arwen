//! Meteorological utility functions.
//!
//! General-purpose helpers for direction conversion, interpolation,
//! curve crossing detection, and nearest-neighbor resampling.

// ─────────────────────────────────────────────
// Direction / angle conversion
// ─────────────────────────────────────────────

/// 8-point compass rose in clockwise order starting from North (0 degrees).
const DIRECTIONS_8: [&str; 8] = [
    "N", "NE", "E", "SE", "S", "SW", "W", "NW",
];

/// 16-point compass rose in clockwise order starting from North (0 degrees).
const DIRECTIONS: [&str; 16] = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
];

/// 32-point compass rose in clockwise order starting from North (0 degrees).
const DIRECTIONS_32: [&str; 32] = [
    "N", "NbE", "NNE", "NEbN",
    "NE", "NEbE", "ENE", "EbN",
    "E", "EbS", "ESE", "SEbE",
    "SE", "SEbS", "SSE", "SbE",
    "S", "SbW", "SSW", "SWbS",
    "SW", "SWbW", "WSW", "WbS",
    "W", "WbN", "WNW", "NWbW",
    "NW", "NWbN", "NNW", "NbW",
];

/// 8-point full word compass names.
const DIRECTIONS_8_FULL: [&str; 8] = [
    "North", "Northeast", "East", "Southeast",
    "South", "Southwest", "West", "Northwest",
];

/// 16-point full word compass names.
const DIRECTIONS_16_FULL: [&str; 16] = [
    "North", "North-Northeast", "Northeast", "East-Northeast",
    "East", "East-Southeast", "Southeast", "South-Southeast",
    "South", "South-Southwest", "Southwest", "West-Southwest",
    "West", "West-Northwest", "Northwest", "North-Northwest",
];

/// 32-point full word compass names.
const DIRECTIONS_32_FULL: [&str; 32] = [
    "North", "North by East", "North-Northeast", "Northeast by North",
    "Northeast", "Northeast by East", "East-Northeast", "East by North",
    "East", "East by South", "East-Southeast", "Southeast by East",
    "Southeast", "Southeast by South", "South-Southeast", "South by East",
    "South", "South by West", "South-Southwest", "Southwest by South",
    "Southwest", "Southwest by West", "West-Southwest", "West by South",
    "West", "West by North", "West-Northwest", "Northwest by West",
    "Northwest", "Northwest by North", "North-Northwest", "North by West",
];

/// Convert a meteorological angle (degrees clockwise from north) to a
/// cardinal direction string.
///
/// # Arguments
/// * `angle` - Angle in degrees clockwise from north
/// * `level` - Number of compass points: 8, 16 (default), or 32
/// * `full` - If true, return full word names (e.g. "North" instead of "N")
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::angle_to_direction_ext;
/// assert_eq!(angle_to_direction_ext(0.0, 16, false), "N");
/// assert_eq!(angle_to_direction_ext(90.0, 8, true), "East");
/// assert_eq!(angle_to_direction_ext(225.0, 16, false), "SW");
/// assert_eq!(angle_to_direction_ext(11.25, 32, false), "NbE");
/// ```
pub fn angle_to_direction_ext(angle: f64, level: u32, full: bool) -> &'static str {
    let a = ((angle % 360.0) + 360.0) % 360.0;
    match (level, full) {
        (8, false) => {
            let step = 360.0 / 8.0; // 45.0
            let idx = ((a + step / 2.0) / step) as usize % 8;
            DIRECTIONS_8[idx]
        }
        (8, true) => {
            let step = 360.0 / 8.0;
            let idx = ((a + step / 2.0) / step) as usize % 8;
            DIRECTIONS_8_FULL[idx]
        }
        (32, false) => {
            let step = 360.0 / 32.0; // 11.25
            let idx = ((a + step / 2.0) / step) as usize % 32;
            DIRECTIONS_32[idx]
        }
        (32, true) => {
            let step = 360.0 / 32.0;
            let idx = ((a + step / 2.0) / step) as usize % 32;
            DIRECTIONS_32_FULL[idx]
        }
        (_, true) => {
            // default 16-point full
            let idx = ((a + 11.25) / 22.5) as usize % 16;
            DIRECTIONS_16_FULL[idx]
        }
        _ => {
            // default 16-point abbreviation
            let idx = ((a + 11.25) / 22.5) as usize % 16;
            DIRECTIONS[idx]
        }
    }
}

/// Convert a meteorological angle (degrees clockwise from north) to a
/// 16-point cardinal direction string.
///
/// The angle is normalised to [0, 360) before binning into 22.5-degree
/// sectors.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::angle_to_direction;
/// assert_eq!(angle_to_direction(0.0), "N");
/// assert_eq!(angle_to_direction(90.0), "E");
/// assert_eq!(angle_to_direction(225.0), "SW");
/// assert_eq!(angle_to_direction(359.0), "N");
/// ```
pub fn angle_to_direction(angle: f64) -> &'static str {
    angle_to_direction_ext(angle, 16, false)
}

/// Parse a cardinal direction string into degrees (meteorological convention).
///
/// Accepts the 16-point compass rose used by [`angle_to_direction`].
/// Case-insensitive.  Returns `None` for unrecognised strings.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::parse_angle;
/// assert_eq!(parse_angle("N"), Some(0.0));
/// assert_eq!(parse_angle("sw"), Some(225.0));
/// assert_eq!(parse_angle("bogus"), None);
/// ```
pub fn parse_angle(dir: &str) -> Option<f64> {
    let upper = dir.to_uppercase();
    DIRECTIONS.iter().position(|&d| d == upper).map(|i| i as f64 * 22.5)
}

// ─────────────────────────────────────────────
// Interpolation helpers
// ─────────────────────────────────────────────

/// Find the two indices in `values` that bracket `target`.
///
/// Searches for the first pair `(i, i+1)` where `target` lies between
/// `values[i]` and `values[i+1]` (inclusive of endpoints). Works for
/// both monotonically increasing and decreasing sequences.
///
/// Returns `None` if `values` has fewer than two elements or `target` is
/// outside the range of the data.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::find_bounding_indices;
/// let v = vec![1.0, 3.0, 5.0, 7.0, 9.0];
/// assert_eq!(find_bounding_indices(&v, 4.0), Some((1, 2)));
/// assert_eq!(find_bounding_indices(&v, 1.0), Some((0, 1)));
/// assert_eq!(find_bounding_indices(&v, 0.0), None);
/// ```
pub fn find_bounding_indices(values: &[f64], target: f64) -> Option<(usize, usize)> {
    if values.len() < 2 {
        return None;
    }
    for i in 0..values.len() - 1 {
        let (lo, hi) = if values[i] <= values[i + 1] {
            (values[i], values[i + 1])
        } else {
            (values[i + 1], values[i])
        };
        if target >= lo && target <= hi {
            return Some((i, i + 1));
        }
    }
    None
}

/// Find the index nearest to the point where two series cross.
///
/// Given a common x-axis and two y-series, locates where
/// `y1[i] - y2[i]` changes sign and returns the index of the crossing
/// point that is closest to zero difference.
///
/// Returns `None` if no crossing is found or inputs are too short.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::nearest_intersection_idx;
/// let x  = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y1 = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
/// assert_eq!(nearest_intersection_idx(&x, &y1, &y2), Some(2));
/// ```
pub fn nearest_intersection_idx(x: &[f64], y1: &[f64], y2: &[f64]) -> Option<usize> {
    let n = x.len().min(y1.len()).min(y2.len());
    if n < 2 {
        return None;
    }

    let diff: Vec<f64> = (0..n).map(|i| y1[i] - y2[i]).collect();

    let mut best_idx: Option<usize> = None;
    let mut best_abs = f64::INFINITY;

    for i in 0..n - 1 {
        // Check for a sign change (or zero crossing)
        if diff[i] * diff[i + 1] <= 0.0 {
            // Pick whichever endpoint is closer to zero
            let (idx, abs_val) = if diff[i].abs() <= diff[i + 1].abs() {
                (i, diff[i].abs())
            } else {
                (i + 1, diff[i + 1].abs())
            };
            if abs_val < best_abs {
                best_abs = abs_val;
                best_idx = Some(idx);
            }
        }
    }

    best_idx
}

// ─────────────────────────────────────────────
// Resampling
// ─────────────────────────────────────────────

/// Nearest-neighbour 1-D resampling.
///
/// For each value in `x`, finds the closest point in `xp` and returns the
/// corresponding value from `fp`. This is the 1-D equivalent of
/// `scipy.interpolate.interp1d(kind='nearest')`.
///
/// `xp` and `fp` must have the same length and `xp` should be sorted in
/// ascending order for correct results.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::resample_nn_1d;
/// let xp = vec![0.0, 1.0, 2.0, 3.0];
/// let fp = vec![10.0, 20.0, 30.0, 40.0];
/// let x  = vec![0.4, 1.6, 2.9];
/// let result = resample_nn_1d(&x, &xp, &fp);
/// assert_eq!(result, vec![10.0, 30.0, 40.0]);
/// ```
pub fn resample_nn_1d(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let np = xp.len().min(fp.len());
    x.iter()
        .map(|&xi| {
            if np == 0 {
                return f64::NAN;
            }
            let mut best_idx = 0;
            let mut best_dist = (xi - xp[0]).abs();
            for j in 1..np {
                let d = (xi - xp[j]).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = j;
                }
            }
            fp[best_idx]
        })
        .collect()
}

// ─────────────────────────────────────────────
// Peak detection
// ─────────────────────────────────────────────

/// Find peaks (maxima) or troughs (minima) in a 1-D array, filtered by IQR.
///
/// A local extremum is any point that is greater (or less, for troughs) than
/// both its neighbours.  Only those extrema that stand out by at least
/// `iqr_ratio * IQR` above (or below) the median are kept.
///
/// This matches MetPy's `find_peaks` logic: local extrema are identified,
/// then filtered by comparing their value against `median + iqr_ratio * IQR`
/// for maxima (or `median - iqr_ratio * IQR` for minima).
///
/// Returns the indices of the qualifying extrema in ascending order.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::find_peaks;
/// let data = vec![0.0, 5.0, 1.0, 3.0, 0.0, 10.0, 0.0];
/// let peaks = find_peaks(&data, true, 0.0);
/// assert!(peaks.contains(&1));
/// assert!(peaks.contains(&5));
/// ```
pub fn find_peaks(data: &[f64], maxima: bool, iqr_ratio: f64) -> Vec<usize> {
    let n = data.len();
    if n < 3 {
        return Vec::new();
    }

    // Find all local extrema.
    let mut candidates: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if maxima {
            if data[i] > data[i - 1] && data[i] > data[i + 1] {
                candidates.push(i);
            }
        } else {
            if data[i] < data[i - 1] && data[i] < data[i + 1] {
                candidates.push(i);
            }
        }
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    // Compute median and IQR from the full data (only finite values).
    let mut sorted: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    let median = if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    };
    let q1 = sorted[len / 4];
    let q3 = sorted[3 * len / 4];
    let iqr = q3 - q1;

    // Filter: keep peaks that exceed median + iqr_ratio * IQR.
    let threshold = if maxima {
        median + iqr_ratio * iqr
    } else {
        median - iqr_ratio * iqr
    };

    candidates
        .into_iter()
        .filter(|&i| {
            if maxima {
                data[i] >= threshold
            } else {
                data[i] <= threshold
            }
        })
        .collect()
}

/// Topological persistence-based peak detection.
///
/// Uses the 0-dimensional persistent homology approach to rank peaks (or
/// troughs) by their "persistence" -- the height difference between a peak
/// and the higher of the two saddle points that bound it.  Peaks with
/// greater persistence are more prominent features.
///
/// Returns `(index, persistence)` pairs sorted by descending persistence.
///
/// When `maxima` is false, the data is negated internally so that troughs
/// are detected instead.
///
/// # Algorithm
///
/// 1. Sort data values in decreasing order (for maxima).
/// 2. Process points in that order, using a union-find structure:
///    - If a point has no already-processed neighbours, create a new component.
///    - If one neighbour is processed, add to that component.
///    - If two neighbours are processed and they belong to different components,
///      merge: the younger component dies with persistence = peak_of_younger - current_value.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::peak_persistence;
/// let data = vec![0.0, 10.0, 0.0, 5.0, 0.0];
/// let peaks = peak_persistence(&data, true);
/// // The tallest peak at index 1 (value 10) should have the highest persistence.
/// assert_eq!(peaks[0].0, 1);
/// ```
pub fn peak_persistence(data: &[f64], maxima: bool) -> Vec<(usize, f64)> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    // If looking for minima, negate the data.
    let values: Vec<f64> = if maxima {
        data.to_vec()
    } else {
        data.iter().map(|&x| -x).collect()
    };

    // Sort indices by decreasing value.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find with path compression.
    let mut parent: Vec<usize> = vec![usize::MAX; n];
    let mut birth: Vec<f64> = vec![0.0; n]; // birth value of each component
    let mut processed = vec![false; n];
    let mut result: Vec<(usize, f64)> = Vec::new();

    // Find root of component.
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for &idx in &order {
        // Create a new component born at this point.
        parent[idx] = idx;
        birth[idx] = values[idx];
        processed[idx] = true;

        // Check left and right neighbours.
        let neighbors: Vec<usize> = {
            let mut nb = Vec::new();
            if idx > 0 && processed[idx - 1] {
                nb.push(idx - 1);
            }
            if idx + 1 < n && processed[idx + 1] {
                nb.push(idx + 1);
            }
            nb
        };

        // Collect unique component roots of neighbours.
        let mut roots: Vec<usize> = neighbors.iter().map(|&nb| find(&mut parent, nb)).collect();
        roots.sort_unstable();
        roots.dedup();

        if roots.is_empty() {
            // New isolated component -- nothing to do.
            continue;
        }

        // Merge with all neighbouring components.
        // Keep the oldest (highest birth value) component alive; younger ones die.
        // Include the current point's own component in the merge set.
        roots.push(idx);
        roots.sort_unstable();
        roots.dedup();

        // Recompute roots after adding idx.
        let mut comp_roots: Vec<usize> = roots.iter().map(|&r| find(&mut parent, r)).collect();
        comp_roots.sort_unstable();
        comp_roots.dedup();

        if comp_roots.len() <= 1 {
            // Only one component -- just union if needed.
            let r = comp_roots[0];
            parent[idx] = r;
            continue;
        }

        // Find the oldest component (highest birth value).
        let oldest = *comp_roots
            .iter()
            .max_by(|&&a, &&b| birth[a].partial_cmp(&birth[b]).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Kill younger components.
        for &root in &comp_roots {
            if root != oldest {
                let persistence = birth[root] - values[idx];
                result.push((root, persistence));
                parent[root] = oldest;
            }
        }
    }

    // The last surviving component (the global maximum) gets infinite persistence.
    // Find it and add it.
    if n > 0 {
        // Find the root of the last surviving component.
        let global_root = find(&mut parent, order[0]);
        let persistence = birth[global_root] - values[*order.last().unwrap()];
        result.push((global_root, persistence));
    }

    // Sort by descending persistence.
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ─────────────────────────────────────────────
// Radar coordinate conversion
// ─────────────────────────────────────────────

/// Convert radar azimuth/range coordinates to latitude/longitude.
///
/// Uses the great circle (spherical earth) forward projection to compute
/// the geographic coordinates of radar gates given their azimuth and range
/// from a radar site.
///
/// # Arguments
///
/// * `azimuths` - Azimuth angles in degrees clockwise from north
/// * `ranges` - Range values in meters from the radar
/// * `center_lat` - Radar site latitude in degrees
/// * `center_lon` - Radar site longitude in degrees
///
/// # Returns
///
/// `(latitudes, longitudes)` - Flattened arrays of length
/// `azimuths.len() * ranges.len()` in row-major order (azimuth varies
/// slowest), with values in degrees.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::azimuth_range_to_lat_lon;
/// let az = vec![0.0]; // due north
/// let rng = vec![0.0]; // at the radar
/// let (lats, lons) = azimuth_range_to_lat_lon(&az, &rng, 35.0, -97.0);
/// assert!((lats[0] - 35.0).abs() < 1e-6);
/// assert!((lons[0] + 97.0).abs() < 1e-6);
/// ```
pub fn azimuth_range_to_lat_lon(
    azimuths: &[f64],
    ranges: &[f64],
    center_lat: f64,
    center_lon: f64,
) -> (Vec<f64>, Vec<f64>) {
    const EARTH_RADIUS: f64 = 6371228.0; // meters, same as MetPy

    let n_az = azimuths.len();
    let n_rng = ranges.len();
    let n_total = n_az * n_rng;
    let mut lats = Vec::with_capacity(n_total);
    let mut lons = Vec::with_capacity(n_total);

    let lat0 = center_lat.to_radians();
    let lon0 = center_lon.to_radians();

    for &az_deg in azimuths {
        let az = az_deg.to_radians();
        for &rng in ranges {
            let angular_dist = rng / EARTH_RADIUS;
            let lat = (lat0.sin() * angular_dist.cos()
                + lat0.cos() * angular_dist.sin() * az.cos())
            .asin();
            let lon = lon0
                + (az.sin() * angular_dist.sin() * lat0.cos())
                    .atan2(angular_dist.cos() - lat0.sin() * lat.sin());
            lats.push(lat.to_degrees());
            lons.push(lon.to_degrees());
        }
    }

    (lats, lons)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── angle_to_direction ──

    #[test]
    fn test_cardinal_directions() {
        assert_eq!(angle_to_direction(0.0), "N");
        assert_eq!(angle_to_direction(90.0), "E");
        assert_eq!(angle_to_direction(180.0), "S");
        assert_eq!(angle_to_direction(270.0), "W");
    }

    #[test]
    fn test_intercardinal_directions() {
        assert_eq!(angle_to_direction(45.0), "NE");
        assert_eq!(angle_to_direction(135.0), "SE");
        assert_eq!(angle_to_direction(225.0), "SW");
        assert_eq!(angle_to_direction(315.0), "NW");
    }

    #[test]
    fn test_secondary_intercardinals() {
        assert_eq!(angle_to_direction(22.5), "NNE");
        assert_eq!(angle_to_direction(67.5), "ENE");
        assert_eq!(angle_to_direction(202.5), "SSW");
    }

    #[test]
    fn test_angle_wrapping() {
        assert_eq!(angle_to_direction(360.0), "N");
        assert_eq!(angle_to_direction(720.0), "N");
        assert_eq!(angle_to_direction(-90.0), "W");
    }

    #[test]
    fn test_boundary_angles() {
        // 11.25 is the boundary between N and NNE
        assert_eq!(angle_to_direction(11.24), "N");
        assert_eq!(angle_to_direction(11.26), "NNE");
    }

    // ── parse_angle ──

    #[test]
    fn test_parse_angle_all_directions() {
        for (i, &dir) in DIRECTIONS.iter().enumerate() {
            let expected = i as f64 * 22.5;
            assert_eq!(parse_angle(dir), Some(expected), "failed for {}", dir);
        }
    }

    #[test]
    fn test_parse_angle_case_insensitive() {
        assert_eq!(parse_angle("n"), Some(0.0));
        assert_eq!(parse_angle("Sw"), Some(225.0));
        assert_eq!(parse_angle("nnw"), Some(337.5));
    }

    #[test]
    fn test_parse_angle_invalid() {
        assert_eq!(parse_angle("X"), None);
        assert_eq!(parse_angle(""), None);
        assert_eq!(parse_angle("north"), None);
    }

    #[test]
    fn test_roundtrip_angle_direction() {
        for angle in [0.0, 22.5, 45.0, 90.0, 180.0, 270.0, 315.0] {
            let dir = angle_to_direction(angle);
            let parsed = parse_angle(dir).unwrap();
            assert!(
                (parsed - angle).abs() < 1e-10,
                "roundtrip failed for {angle}: got {dir} -> {parsed}"
            );
        }
    }

    // ── find_bounding_indices ──

    #[test]
    fn test_bounding_indices_basic() {
        let v = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(find_bounding_indices(&v, 4.0), Some((1, 2)));
        assert_eq!(find_bounding_indices(&v, 6.0), Some((2, 3)));
    }

    #[test]
    fn test_bounding_indices_on_point() {
        let v = vec![1.0, 3.0, 5.0];
        assert_eq!(find_bounding_indices(&v, 3.0), Some((0, 1)));
    }

    #[test]
    fn test_bounding_indices_endpoints() {
        let v = vec![1.0, 3.0, 5.0];
        assert_eq!(find_bounding_indices(&v, 1.0), Some((0, 1)));
        assert_eq!(find_bounding_indices(&v, 5.0), Some((1, 2)));
    }

    #[test]
    fn test_bounding_indices_outside() {
        let v = vec![1.0, 3.0, 5.0];
        assert_eq!(find_bounding_indices(&v, 0.0), None);
        assert_eq!(find_bounding_indices(&v, 6.0), None);
    }

    #[test]
    fn test_bounding_indices_decreasing() {
        let v = vec![9.0, 7.0, 5.0, 3.0, 1.0];
        assert_eq!(find_bounding_indices(&v, 4.0), Some((2, 3)));
    }

    #[test]
    fn test_bounding_indices_short() {
        assert_eq!(find_bounding_indices(&[], 1.0), None);
        assert_eq!(find_bounding_indices(&[5.0], 5.0), None);
    }

    // ── nearest_intersection_idx ──

    #[test]
    fn test_intersection_basic() {
        let x  = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y1 = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        assert_eq!(nearest_intersection_idx(&x, &y1, &y2), Some(2));
    }

    #[test]
    fn test_intersection_no_crossing() {
        let x  = vec![0.0, 1.0, 2.0];
        let y1 = vec![5.0, 6.0, 7.0];
        let y2 = vec![1.0, 2.0, 3.0];
        assert_eq!(nearest_intersection_idx(&x, &y1, &y2), None);
    }

    #[test]
    fn test_intersection_at_endpoints() {
        let x  = vec![0.0, 1.0, 2.0];
        let y1 = vec![0.0, 1.0, 2.0];
        let y2 = vec![0.0, 2.0, 4.0];
        // diff = [0, -1, -2], crossing at index 0 (diff==0)
        assert_eq!(nearest_intersection_idx(&x, &y1, &y2), Some(0));
    }

    #[test]
    fn test_intersection_short_input() {
        assert_eq!(nearest_intersection_idx(&[1.0], &[1.0], &[1.0]), None);
        let empty: &[f64] = &[];
        assert_eq!(nearest_intersection_idx(empty, empty, empty), None);
    }

    // ── resample_nn_1d ──

    #[test]
    fn test_resample_exact_points() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0, 30.0, 40.0];
        let x  = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(resample_nn_1d(&x, &xp, &fp), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_resample_midpoints() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0, 30.0, 40.0];
        let x  = vec![0.4, 1.6, 2.9];
        let result = resample_nn_1d(&x, &xp, &fp);
        // 0.4 -> nearest 0.0 -> 10.0
        // 1.6 -> nearest 2.0 -> 30.0
        // 2.9 -> nearest 3.0 -> 40.0
        assert_eq!(result, vec![10.0, 30.0, 40.0]);
    }

    #[test]
    fn test_resample_single_point() {
        let xp = vec![5.0];
        let fp = vec![100.0];
        let x  = vec![0.0, 5.0, 10.0];
        let result = resample_nn_1d(&x, &xp, &fp);
        assert_eq!(result, vec![100.0, 100.0, 100.0]);
    }

    #[test]
    fn test_resample_empty_source() {
        let result = resample_nn_1d(&[1.0, 2.0], &[], &[]);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    // ── angle_to_direction_ext ──

    #[test]
    fn test_direction_8_point() {
        assert_eq!(angle_to_direction_ext(0.0, 8, false), "N");
        assert_eq!(angle_to_direction_ext(45.0, 8, false), "NE");
        assert_eq!(angle_to_direction_ext(90.0, 8, false), "E");
        assert_eq!(angle_to_direction_ext(180.0, 8, false), "S");
        assert_eq!(angle_to_direction_ext(270.0, 8, false), "W");
    }

    #[test]
    fn test_direction_32_point() {
        assert_eq!(angle_to_direction_ext(0.0, 32, false), "N");
        assert_eq!(angle_to_direction_ext(11.25, 32, false), "NbE");
        assert_eq!(angle_to_direction_ext(22.5, 32, false), "NNE");
        assert_eq!(angle_to_direction_ext(90.0, 32, false), "E");
    }

    #[test]
    fn test_direction_full_names() {
        assert_eq!(angle_to_direction_ext(0.0, 8, true), "North");
        assert_eq!(angle_to_direction_ext(90.0, 8, true), "East");
        assert_eq!(angle_to_direction_ext(180.0, 8, true), "South");
        assert_eq!(angle_to_direction_ext(225.0, 8, true), "Southwest");
    }

    #[test]
    fn test_direction_16_full_names() {
        assert_eq!(angle_to_direction_ext(0.0, 16, true), "North");
        assert_eq!(angle_to_direction_ext(22.5, 16, true), "North-Northeast");
        assert_eq!(angle_to_direction_ext(90.0, 16, true), "East");
        assert_eq!(angle_to_direction_ext(225.0, 16, true), "Southwest");
    }

    #[test]
    fn test_direction_32_full_names() {
        assert_eq!(angle_to_direction_ext(0.0, 32, true), "North");
        assert_eq!(angle_to_direction_ext(11.25, 32, true), "North by East");
        assert_eq!(angle_to_direction_ext(90.0, 32, true), "East");
    }

    // ── find_peaks ──

    #[test]
    fn test_find_peaks_basic_maxima() {
        let data = vec![0.0, 5.0, 1.0, 3.0, 0.0, 10.0, 0.0];
        let peaks = find_peaks(&data, true, 0.0);
        assert!(peaks.contains(&1), "Expected index 1 in peaks: {:?}", peaks);
        assert!(peaks.contains(&3), "Expected index 3 in peaks: {:?}", peaks);
        assert!(peaks.contains(&5), "Expected index 5 in peaks: {:?}", peaks);
    }

    #[test]
    fn test_find_peaks_iqr_filter() {
        // With a higher IQR ratio, only the tallest peak should remain.
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 10.0, 0.0];
        let peaks = find_peaks(&data, true, 2.0);
        assert!(peaks.contains(&5), "Expected index 5 in peaks: {:?}", peaks);
        // Small peaks should be filtered out.
        assert!(!peaks.contains(&1), "Did not expect index 1 in peaks: {:?}", peaks);
    }

    #[test]
    fn test_find_peaks_troughs() {
        let data = vec![5.0, 1.0, 5.0, 3.0, 5.0, 0.0, 5.0];
        let peaks = find_peaks(&data, false, 0.0);
        assert!(peaks.contains(&1), "Expected index 1 in troughs: {:?}", peaks);
        assert!(peaks.contains(&5), "Expected index 5 in troughs: {:?}", peaks);
    }

    #[test]
    fn test_find_peaks_empty() {
        let peaks = find_peaks(&[], true, 0.0);
        assert!(peaks.is_empty());
        let peaks = find_peaks(&[1.0, 2.0], true, 0.0);
        assert!(peaks.is_empty());
    }

    // ── peak_persistence ──

    #[test]
    fn test_peak_persistence_single_peak() {
        let data = vec![0.0, 5.0, 0.0];
        let peaks = peak_persistence(&data, true);
        assert!(!peaks.is_empty());
        // The single peak at index 1 should be the most persistent.
        assert_eq!(peaks[0].0, 1);
        assert!((peaks[0].1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_persistence_two_peaks() {
        let data = vec![0.0, 10.0, 0.0, 5.0, 0.0];
        let peaks = peak_persistence(&data, true);
        // The taller peak (index 1, value 10) should be most persistent.
        assert!(peaks.len() >= 2, "Expected at least 2 peaks, got {:?}", peaks);
        assert_eq!(peaks[0].0, 1, "Most persistent peak should be at index 1");
        assert!(peaks[0].1 > peaks[1].1, "First peak should have higher persistence");
    }

    #[test]
    fn test_peak_persistence_minima() {
        let data = vec![10.0, 0.0, 10.0, 5.0, 10.0];
        let peaks = peak_persistence(&data, false);
        assert!(!peaks.is_empty());
        // The deepest trough at index 1 (value 0) should be most persistent.
        assert_eq!(peaks[0].0, 1);
    }

    #[test]
    fn test_peak_persistence_empty() {
        let peaks = peak_persistence(&[], true);
        assert!(peaks.is_empty());
    }

    // ── azimuth_range_to_lat_lon ──

    #[test]
    fn test_az_range_at_origin() {
        let az = vec![0.0];
        let rng = vec![0.0];
        let (lats, lons) = azimuth_range_to_lat_lon(&az, &rng, 35.0, -97.0);
        assert_eq!(lats.len(), 1);
        assert!((lats[0] - 35.0).abs() < 1e-10);
        assert!((lons[0] + 97.0).abs() < 1e-10);
    }

    #[test]
    fn test_az_range_due_north() {
        // 1 degree of latitude ~ 111.19 km
        let az = vec![0.0]; // due north
        let rng = vec![111_194.0]; // ~1 degree of latitude in meters
        let (lats, lons) = azimuth_range_to_lat_lon(&az, &rng, 0.0, 0.0);
        assert!((lats[0] - 1.0).abs() < 0.01, "Expected lat ~ 1.0, got {}", lats[0]);
        assert!(lons[0].abs() < 0.01, "Expected lon ~ 0.0, got {}", lons[0]);
    }

    #[test]
    fn test_az_range_due_east() {
        // Due east at equator, 1 degree ~ 111.19 km
        let az = vec![90.0]; // due east
        let rng = vec![111_194.0];
        let (lats, lons) = azimuth_range_to_lat_lon(&az, &rng, 0.0, 0.0);
        assert!(lats[0].abs() < 0.01, "Expected lat ~ 0, got {}", lats[0]);
        assert!((lons[0] - 1.0).abs() < 0.01, "Expected lon ~ 1.0, got {}", lons[0]);
    }

    #[test]
    fn test_az_range_output_shape() {
        let az = vec![0.0, 90.0, 180.0];
        let rng = vec![1000.0, 2000.0, 3000.0, 4000.0];
        let (lats, lons) = azimuth_range_to_lat_lon(&az, &rng, 35.0, -97.0);
        assert_eq!(lats.len(), 12); // 3 * 4
        assert_eq!(lons.len(), 12);
    }
}
