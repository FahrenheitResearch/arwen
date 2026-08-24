//! Nearest-cell resample weights: the one-time, per-mesh half of the bridge.
//!
//! Method: nearest cell centre on the sphere, by 3-D chord distance through a
//! k-d tree over unit vectors. Nearest-cell is a deliberate choice, not a
//! shortcut. A resampled value is exactly the value the model carried in that
//! cell, so extremes survive the transfer unchanged: a 70 dBZ core stays
//! 70 dBZ, a 35 m/s gust stays 35 m/s. Every smoothing scheme, inverse
//! distance included, pulls maxima down and pushes minima up, which is the
//! wrong trade for products whose whole point is the tail of the
//! distribution.
//!
//! Nothing here is conservative and nothing produced through this path
//! supports an accumulation-budget claim.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{MpasError, MpasResult};
use crate::window::{Window, MPAS_EARTH_RADIUS_M};

pub const WEIGHTS_SCHEMA: &str = "mpas-port.nearest-cell-render-weights/v1";
pub const RESAMPLE_METHOD: &str = "nearest-cell (spherical k-d tree over unit vectors)";

const RAD: f64 = std::f64::consts::PI / 180.0;

/// How far past pi/2 a mesh latitude may sit and still be a pole cell.
///
/// MPAS writes `latCell` as float32, and a mesh with cells at the poles stores
/// pi/2 as the nearest float32, which is 4.4e-8 rad *above* pi/2. A tolerance
/// tighter than that refuses every polar global mesh, x1.40962 included. One
/// float32 ulp at 1.57 is 1.19e-7 rad, so the bound below is a few ulps: it
/// admits a rounded pole and nothing else. 1e-6 rad is 5.7e-5 degrees, about
/// 6 m on the ground, so no genuinely out-of-range mesh slips through.
const POLE_TOLERANCE_RAD: f64 = std::f64::consts::FRAC_PI_2 + 1.0e-6;

/// How a mesh's angular units were established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleUnits {
    /// The file declared radians in a `units` attribute.
    Declared,
    /// The file declared nothing and radians were established from the range
    /// the values actually span. Recorded in the emitted frame's provenance.
    InferredFromRange,
}

impl AngleUnits {
    pub fn label(self) -> &'static str {
        match self {
            AngleUnits::Declared => "declared-radians",
            AngleUnits::InferredFromRange => "inferred-radians-from-range",
        }
    }
}

/// Decide whether `latCell`/`lonCell` are radians.
///
/// A declared unit is obeyed: radians pass, anything else is refused, exactly
/// as before. The new case is an *absent* `units` attribute, which the CUDA
/// port's history writer leaves off. Absence is not a licence -- it is
/// resolved against the numbers, and only when they cannot be degrees:
/// a degree-valued mesh reaches |lat| = 90, which is 57x the radian bound of
/// pi/2, so the two interpretations do not overlap for any mesh that spans
/// more than 1.57 degrees of latitude or 6.28 of longitude.
///
/// The residual ambiguity is named rather than hidden: a *degree*-valued mesh
/// confined to roughly 1.57 deg of the equator and 6.28 deg of the prime
/// meridian -- a patch of the Gulf of Guinea -- would be read as radians.
/// No MPAS mesh in this program is regional, let alone that one, and the
/// emitted frame records `inferred-radians-from-range` so a reader can see
/// which branch was taken.
pub fn resolve_angle_units(
    declared: &str,
    max_abs_lat: f64,
    max_abs_lon: f64,
) -> MpasResult<AngleUnits> {
    let units = declared.trim().to_ascii_lowercase();
    if matches!(units.as_str(), "rad" | "radian" | "radians") {
        return Ok(AngleUnits::Declared);
    }
    if !units.is_empty() {
        return Err(MpasError::Refusal(format!(
            "mesh latCell/lonCell units '{declared}' are not radians"
        )));
    }
    const LAT_LIMIT: f64 = std::f64::consts::FRAC_PI_2 + 1.0e-6;
    const LON_LIMIT: f64 = 2.0 * std::f64::consts::PI + 1.0e-6;
    if max_abs_lat <= LAT_LIMIT && max_abs_lon <= LON_LIMIT {
        Ok(AngleUnits::InferredFromRange)
    } else {
        Err(MpasError::Refusal(format!(
            "mesh latCell/lonCell declare no units and the values they span \
             (max |lat| {max_abs_lat}, max |lon| {max_abs_lon}) are not radians \
             (radians bound them by {LAT_LIMIT} and {LON_LIMIT})"
        )))
    }
}

/// Cell centres read from a grid, static, init or history file.
#[derive(Debug, Clone)]
pub struct MeshCoordinates {
    pub latitude_degrees: Vec<f64>,
    pub longitude_degrees: Vec<f64>,
    pub source_path: PathBuf,
    pub source_sha256: String,
    pub angle_units: AngleUnits,
}

impl MeshCoordinates {
    pub fn n_cells(&self) -> usize {
        self.latitude_degrees.len()
    }
}

/// Read `latCell`/`lonCell` and bind the file by digest.
///
/// Longitudes are normalised to `[-180, 180)`. A rotated mesh -- the
/// `grid_rotate` output the port pins -- stores post-rotation earth
/// coordinates directly, so no de-rotation is applied or wanted: the cell
/// centre in the file is where that cell is on the Earth. The converter
/// cross-checks this against the history file's own coordinates before it
/// resamples anything.
pub fn read_mesh_coordinates(path: &Path, expect_sha256: Option<&str>) -> MpasResult<MeshCoordinates> {
    let digest = crate::sha256_file(path)?;
    if let Some(want) = expect_sha256 {
        if digest != want {
            return Err(MpasError::Refusal(format!(
                "mesh {} is sha256 {digest}, expected {want}",
                path.display()
            )));
        }
    }
    let file = netcrust::File::open(path)?;
    let mut declared: Vec<String> = Vec::new();
    for name in ["latCell", "lonCell"] {
        let variable = file.variable(name).ok_or_else(|| {
            MpasError::Refusal(format!("mesh {} has no {name}", path.display()))
        })?;
        declared.push(
            variable
                .attribute("units")
                .and_then(|a| a.as_string())
                .unwrap_or_default()
                .to_string(),
        );
    }
    let latitude = file.read_f64("latCell")?;
    let longitude = file.read_f64("lonCell")?;
    let max_abs_lat = latitude.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_abs_lon = longitude.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut angle_units = AngleUnits::Declared;
    for (name, text) in ["latCell", "lonCell"].iter().zip(declared.iter()) {
        let resolved = resolve_angle_units(text, max_abs_lat, max_abs_lon).map_err(|e| {
            MpasError::Refusal(format!("mesh {} {name}: {e}", path.display()))
        })?;
        if resolved == AngleUnits::InferredFromRange {
            angle_units = AngleUnits::InferredFromRange;
        }
    }
    if angle_units == AngleUnits::InferredFromRange {
        eprintln!(
            "rw_mpas_convert: advisory: mesh {} declares no units on latCell/lonCell; \
             radians established from the range (max |lat| {max_abs_lat:.6}, max |lon| \
             {max_abs_lon:.6}) and recorded as {} in the emitted frame",
            path.display(),
            AngleUnits::InferredFromRange.label()
        );
    }
    if latitude.len() != longitude.len() || latitude.is_empty() {
        return Err(MpasError::Refusal(
            "mesh latCell/lonCell shapes disagree".to_string(),
        ));
    }
    if latitude.iter().chain(longitude.iter()).any(|v| !v.is_finite()) {
        return Err(MpasError::Refusal(
            "mesh coordinates are not finite".to_string(),
        ));
    }
    if max_abs_lat > POLE_TOLERANCE_RAD {
        return Err(MpasError::Refusal(format!(
            "mesh latitude leaves [-pi/2, pi/2]: max |lat| is {max_abs_lat} rad,              which is {:e} rad past the pole (tolerance {:e})",
            max_abs_lat - std::f64::consts::FRAC_PI_2,
            POLE_TOLERANCE_RAD - std::f64::consts::FRAC_PI_2
        )));
    }
    let deg = 180.0 / std::f64::consts::PI;
    Ok(MeshCoordinates {
        latitude_degrees: latitude.iter().map(|v| v * deg).collect(),
        longitude_degrees: longitude
            .iter()
            .map(|v| (v * deg + 180.0).rem_euclid(360.0) - 180.0)
            .collect(),
        source_path: path.to_path_buf(),
        source_sha256: digest,
        angle_units,
    })
}

fn unit_vector(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let lat = lat_deg * RAD;
    let lon = lon_deg * RAD;
    let cos_lat = lat.cos();
    [cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin()]
}

// ---------------------------------------------------------------------------
// A 3-D k-d tree over the mesh's unit vectors.
// ---------------------------------------------------------------------------

/// Balanced k-d tree built by repeated median selection. Nodes are stored
/// implicitly: `order[lo..hi]` is a subtree and its median is its root, so
/// there is one permutation vector and no per-node allocation.
struct KdTree {
    points: Vec<[f64; 3]>,
    order: Vec<u32>,
    /// Split axis chosen for each subtree root, indexed by the median slot.
    axis: Vec<u8>,
}

/// Split `order[lo..hi]` at its median along the widest axis of that
/// subtree's bounding box -- widest, because it is what keeps the tree
/// shallow on a mesh with a refined region.
///
/// The median selection is the standard library's `select_nth_unstable_by`.
/// A hand-rolled quickselect is the obvious thing to write here and the wrong
/// thing to trust: the first version of this function looped forever whenever
/// a partition landed with every element on one side, which a mesh with many
/// coincident coordinates reaches immediately.
fn build_range(
    points: &[[f64; 3]],
    order: &mut [u32],
    axis_of: &mut [u8],
    lo: usize,
    hi: usize,
) {
    if hi <= lo {
        return;
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &idx in &order[lo..hi] {
        let p = points[idx as usize];
        for k in 0..3 {
            if p[k] < min[k] {
                min[k] = p[k];
            }
            if p[k] > max[k] {
                max[k] = p[k];
            }
        }
    }
    let mut axis = 0usize;
    let mut widest = max[0] - min[0];
    for k in 1..3 {
        if max[k] - min[k] > widest {
            widest = max[k] - min[k];
            axis = k;
        }
    }
    let mid = lo + (hi - lo) / 2;
    order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
        points[a as usize][axis].total_cmp(&points[b as usize][axis])
    });
    axis_of[mid] = axis as u8;
    build_range(points, order, axis_of, lo, mid);
    build_range(points, order, axis_of, mid + 1, hi);
}

impl KdTree {
    fn build(points: Vec<[f64; 3]>) -> KdTree {
        let n = points.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut axis = vec![0u8; n];
        build_range(&points, &mut order, &mut axis, 0, n);
        KdTree {
            points,
            order,
            axis,
        }
    }

    /// Index of the nearest stored point to `target`, by squared chord
    /// distance, with the distance.
    fn nearest(&self, target: [f64; 3]) -> (u32, f64) {
        let mut best = (u32::MAX, f64::INFINITY);
        self.search(0, self.order.len(), target, &mut best);
        best
    }

    fn search(&self, lo: usize, hi: usize, target: [f64; 3], best: &mut (u32, f64)) {
        if hi <= lo {
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let idx = self.order[mid];
        let p = self.points[idx as usize];
        let d = (p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2) + (p[2] - target[2]).powi(2);
        if d < best.1 {
            *best = (idx, d);
        }
        let axis = self.axis[mid] as usize;
        let delta = target[axis] - p[axis];
        let (near_lo, near_hi, far_lo, far_hi) = if delta < 0.0 {
            (lo, mid, mid + 1, hi)
        } else {
            (mid + 1, hi, lo, mid)
        };
        self.search(near_lo, near_hi, target, best);
        if delta * delta < best.1 {
            self.search(far_lo, far_hi, target, best);
        }
    }
}

/// One nearest-cell index map for one mesh and one window.
#[derive(Debug, Clone)]
pub struct NearestCellWeights {
    pub window_name: String,
    pub window: Window,
    pub target_latitude: Vec<f64>,
    pub target_longitude: Vec<f64>,
    pub cell_index: Vec<i32>,
    pub distance_km: Vec<f32>,
    pub mesh_sha256: String,
    pub mesh_path: String,
    pub mesh_angle_units: AngleUnits,
    pub n_cells: usize,
    pub ny: usize,
    pub nx: usize,
    pub weights_sha256: String,
    pub build_seconds: f64,
}

impl NearestCellWeights {
    pub fn shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    pub fn max_distance_km(&self) -> f32 {
        self.distance_km.iter().copied().fold(f32::MIN, f32::max)
    }

    pub fn mean_distance_km(&self) -> f64 {
        self.distance_km.iter().map(|&v| v as f64).sum::<f64>() / self.distance_km.len() as f64
    }

    /// Gather a cell-indexed field onto the target window.
    ///
    /// The result is exactly the source array's values, reordered: no
    /// arithmetic happens, so every bit pattern in the output came straight
    /// from the model field. `values` is `(n_cells, levels)` row-major, and
    /// the output is `(levels, ny, nx)` -- the WRF layout.
    pub fn gather(&self, values: &[f32], levels: usize) -> MpasResult<Vec<f32>> {
        if values.len() != self.n_cells * levels {
            return Err(MpasError::Refusal(format!(
                "field carries {} value(s); mesh has {} cells x {levels} level(s)",
                values.len(),
                self.n_cells
            )));
        }
        let points = self.cell_index.len();
        let mut out = vec![0.0f32; levels * points];
        for level in 0..levels {
            let target = &mut out[level * points..(level + 1) * points];
            for (slot, &cell) in self.cell_index.iter().enumerate() {
                target[slot] = values[cell as usize * levels + level];
            }
        }
        Ok(out)
    }
}

fn weights_digest(
    window_name: &str,
    spec_compact: &str,
    mesh_sha256: &str,
    cell_index: &[i32],
) -> String {
    let mut spec_hash = Sha256::new();
    spec_hash.update(spec_compact.as_bytes());
    let spec_digest = hex(&spec_hash.finalize());

    let mut digest = Sha256::new();
    digest.update(WEIGHTS_SCHEMA.as_bytes());
    digest.update([0u8]);
    digest.update(RESAMPLE_METHOD.as_bytes());
    digest.update([0u8]);
    digest.update(window_name.as_bytes());
    digest.update([0u8]);
    digest.update(spec_digest.as_bytes());
    digest.update([0u8]);
    digest.update(mesh_sha256.as_bytes());
    digest.update([0u8]);
    let mut bytes = Vec::with_capacity(cell_index.len() * 4);
    for &v in cell_index {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    digest.update(&bytes);
    hex(&digest.finalize())
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Build one nearest-cell index map. No cache is consulted or written.
pub fn build_weights(
    mesh: &MeshCoordinates,
    window_name: &str,
    window: &Window,
) -> MpasResult<NearestCellWeights> {
    let started = std::time::Instant::now();
    let (target_latitude, target_longitude) = window.coordinates()?;
    let (ny, nx) = window.shape()?;
    if target_latitude.len() != ny * nx {
        return Err(MpasError::Refusal(
            "window coordinates do not match its shape".to_string(),
        ));
    }

    let cells: Vec<[f64; 3]> = (0..mesh.n_cells())
        .map(|i| unit_vector(mesh.latitude_degrees[i], mesh.longitude_degrees[i]))
        .collect();
    let tree = KdTree::build(cells);

    let mut cell_index = vec![0i32; ny * nx];
    let mut distance_km = vec![0.0f32; ny * nx];
    for slot in 0..ny * nx {
        let target = unit_vector(target_latitude[slot], target_longitude[slot]);
        let (idx, chord_squared) = tree.nearest(target);
        cell_index[slot] = idx as i32;
        // Chord to great-circle, on the mesh's physical radius.
        let chord = chord_squared.sqrt();
        let great_circle = 2.0 * (chord / 2.0).clamp(0.0, 1.0).asin() * MPAS_EARTH_RADIUS_M / 1000.0;
        distance_km[slot] = great_circle as f32;
    }
    if cell_index
        .iter()
        .any(|&i| i < 0 || i as usize >= mesh.n_cells())
    {
        return Err(MpasError::Refusal(
            "nearest-cell index left the mesh".to_string(),
        ));
    }

    let spec_compact = window.spec_json_compact();
    Ok(NearestCellWeights {
        window_name: window_name.to_string(),
        window: window.clone(),
        target_latitude,
        target_longitude,
        cell_index: cell_index.clone(),
        distance_km,
        mesh_sha256: mesh.source_sha256.clone(),
        mesh_path: mesh.source_path.display().to_string(),
        mesh_angle_units: mesh.angle_units,
        n_cells: mesh.n_cells(),
        ny,
        nx,
        weights_sha256: weights_digest(
            window_name,
            &spec_compact,
            &mesh.source_sha256,
            &cell_index,
        ),
        build_seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The k-d tree must agree with brute force on every query, or the whole
    /// map is quietly wrong in a way no eye would catch.
    #[test]
    fn the_kd_tree_agrees_with_brute_force() {
        let mut state = 0x2026_0812_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let cells: Vec<[f64; 3]> = (0..4000)
            .map(|_| {
                let lat = next() * 180.0 - 90.0;
                let lon = next() * 360.0 - 180.0;
                unit_vector(lat, lon)
            })
            .collect();
        let tree = KdTree::build(cells.clone());
        for _ in 0..2000 {
            let target = unit_vector(next() * 180.0 - 90.0, next() * 360.0 - 180.0);
            let (got, got_d) = tree.nearest(target);
            let mut best = (usize::MAX, f64::INFINITY);
            for (i, p) in cells.iter().enumerate() {
                let d = (p[0] - target[0]).powi(2)
                    + (p[1] - target[1]).powi(2)
                    + (p[2] - target[2]).powi(2);
                if d < best.1 {
                    best = (i, d);
                }
            }
            assert_eq!(got as usize, best.0, "index differs");
            assert_eq!(got_d, best.1, "distance differs");
        }
    }

    #[test]
    fn a_gather_reorders_without_arithmetic() {
        let weights = NearestCellWeights {
            window_name: "t".into(),
            window: Window::named("focus").unwrap(),
            target_latitude: vec![0.0; 4],
            target_longitude: vec![0.0; 4],
            cell_index: vec![2, 0, 1, 2],
            distance_km: vec![0.0; 4],
            mesh_sha256: String::new(),
            mesh_path: String::new(),
            mesh_angle_units: AngleUnits::Declared,
            n_cells: 3,
            ny: 2,
            nx: 2,
            weights_sha256: String::new(),
            build_seconds: 0.0,
        };
        // Awkward bit patterns: a value that survives a gather bit-for-bit
        // cannot be hiding behind a benign-looking number.
        let values = vec![
            f32::from_bits(0x7f7f_ffff),
            -0.0,
            f32::from_bits(0x0080_0000),
            1.0,
            2.0,
            3.0,
        ];
        let out = weights.gather(&values, 2).unwrap();
        // level 0 then level 1, each (ny, nx)
        assert_eq!(out[0].to_bits(), values[4].to_bits());
        assert_eq!(out[1].to_bits(), values[0].to_bits());
        assert_eq!(out[2].to_bits(), values[2].to_bits());
        assert_eq!(out[3].to_bits(), values[4].to_bits());
        assert_eq!(out[4].to_bits(), values[5].to_bits());
        assert_eq!(out[5].to_bits(), values[1].to_bits());
    }

    #[test]
    fn the_pole_tolerance_admits_a_float32_pi_over_two() {
        // The exact value a float32 mesh stores for the north pole.
        let f32_pole = (std::f64::consts::FRAC_PI_2 as f32) as f64;
        assert!(
            f32_pole > std::f64::consts::FRAC_PI_2,
            "float32 pi/2 must round up for this test to mean anything"
        );
        assert!(
            f32_pole <= POLE_TOLERANCE_RAD,
            "float32 pi/2 ({f32_pole}) must be inside the pole tolerance ({POLE_TOLERANCE_RAD})"
        );
        // And the tolerance is nowhere near admitting degrees.
        assert!(90.0 > POLE_TOLERANCE_RAD);
        // A milliradian past the pole -- 6 km -- is still refused.
        assert!(std::f64::consts::FRAC_PI_2 + 1.0e-3 > POLE_TOLERANCE_RAD);
    }

    #[test]
    fn declared_radians_are_taken_as_declared() {
        for text in ["rad", "radian", "radians", " RADIANS "] {
            assert_eq!(
                resolve_angle_units(text, 1.5, 3.1).unwrap(),
                AngleUnits::Declared,
                "{text}"
            );
        }
    }

    #[test]
    fn declared_degrees_are_still_refused_whatever_the_range() {
        // The guard that mattered before must not have been weakened: a file
        // that says degrees is refused even when its numbers would pass the
        // range test on their own.
        let err = resolve_angle_units("degrees", 1.5, 3.1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not radians"), "{err}");
        for text in ["degrees", "degree", "deg", "degrees_north", "m"] {
            assert!(resolve_angle_units(text, 0.1, 0.1).is_err(), "{text}");
        }
    }

    #[test]
    fn absent_units_with_radian_range_are_inferred() {
        // pi/2 and pi exactly: the x1.40962 global mesh's own extremes.
        assert_eq!(
            resolve_angle_units("", std::f64::consts::FRAC_PI_2, std::f64::consts::PI).unwrap(),
            AngleUnits::InferredFromRange
        );
    }

    #[test]
    fn absent_units_with_degree_range_are_refused() {
        // A degree-valued global mesh reaches |lat| = 90, which is 57x the
        // radian bound. There is no overlap to get wrong.
        let err = resolve_angle_units("", 90.0, 180.0).unwrap_err().to_string();
        assert!(err.contains("declare no units"), "{err}");
        assert!(err.contains("90"), "{err}");
        // Latitude alone is enough to refuse, even with a small longitude span.
        assert!(resolve_angle_units("", 41.5, 1.0).is_err());
        // And longitude alone is enough, even with a small latitude span.
        assert!(resolve_angle_units("", 0.5, 96.0).is_err());
    }

    #[test]
    fn absent_units_reject_the_boundary_by_a_hair() {
        // Just outside pi/2 must fail; just inside must pass. This pins the
        // limit rather than trusting it.
        assert!(resolve_angle_units("", std::f64::consts::FRAC_PI_2 + 1.0e-3, 1.0).is_err());
        assert!(resolve_angle_units("", std::f64::consts::FRAC_PI_2 - 1.0e-3, 1.0).is_ok());
    }

    #[test]
    fn a_field_of_the_wrong_length_is_refused() {
        let weights = NearestCellWeights {
            window_name: "t".into(),
            window: Window::named("focus").unwrap(),
            target_latitude: vec![0.0; 1],
            target_longitude: vec![0.0; 1],
            cell_index: vec![0],
            distance_km: vec![0.0],
            mesh_sha256: String::new(),
            mesh_path: String::new(),
            mesh_angle_units: AngleUnits::Declared,
            n_cells: 3,
            ny: 1,
            nx: 1,
            weights_sha256: String::new(),
            build_seconds: 0.0,
        };
        let err = weights.gather(&[1.0, 2.0], 1).unwrap_err().to_string();
        assert!(err.contains("mesh has 3 cells"), "{err}");
    }
}
