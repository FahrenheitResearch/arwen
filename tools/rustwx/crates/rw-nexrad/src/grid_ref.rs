//! `rw_nexrad grid-ref`: many Level-II volumes onto one model grid, as one
//! reflectivity product per call, in the convention NOAA's radar latent
//! heating (`ref2tten`) reads.
//!
//! The whole data path is here: decode, beam geometry, projection, binning
//! and reduction.  Python writes the grid descriptor, names the volumes and
//! reads the result back; it does no arithmetic on a gate.
//!
//! ```text
//! rw_nexrad grid-ref --grid GRID.json --volumes V1,V2,... --out PREFIX
//!                    [--reduce mean|max] [--max-range-km 250]
//!                    [--max-elevation-deg 20] [--threads N]
//!                    [--window-end TIME [--max-age-s 900]] [--out-data FILE]
//! ```
//!
//! **Geometry** is a port of `gpuwm/obs/geometry.py` (`beam_geometry`,
//! `great_circle_point`): the 4/3 effective-earth straight ray over WRF's
//! 6370 km sphere, the same expressions in the same evaluation order.  The
//! horizontal placement is `TargetGrid.mass_index` (the grid projection's
//! `latlon_to_ij` minus one, rounded half-to-even like `numpy.rint`) through
//! the `static-fields` crate, which is the byte-parity port of
//! `gpuwm/static/projection.py`.  The vertical placement is
//! `TargetGrid.level_index`: the layer whose `z_w` interfaces bracket the gate
//! height in that gate's own column, and nothing below the terrain surface or
//! at or above the model top.
//!
//! **QC** is declared, simple and stated in the receipt:
//! * a range-folded gate (censor code 2) is never an observation, and any
//!   other censor code that is neither a measurement nor below-threshold is
//!   dropped the same way;
//! * a measured gate whose co-located RhoHV is below [`NONMET_RHO_MAX`] while
//!   its reflectivity is below [`NONMET_SHIELD_DBZ`] is dropped as
//!   non-meteorological.  Where the radial carries no RhoHV at that range,
//!   nothing is dropped.  This is NOT the Python superob's CC QC (no split-cut
//!   companion pairing, no 0.95 / 35 dBZ thresholds); it is a declared simple
//!   filter and the receipt says so.
//!
//! **Reduction** is per model cell over every gate of every radar:
//! `mean` is the mean of linear Z over echo gates, back to dBZ; `max` is the
//! largest echo dBZ.  An echo gate is a finite measured reflectivity that
//! survived QC (there is no dBZ floor).  A cell with at least one
//! below-threshold gate and no echo gate is "observed, no echo".
//!
//! **Determinism**: gate placement runs in parallel, accumulation does not.
//! Each volume accumulates its gates in (sweep, radial, gate) order into its
//! own window, and the windows are added into the domain in a fixed order
//! (site, volume start, file name).  The output bytes therefore do not depend
//! on the thread count or on the order the volumes were named in.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use static_fields::projection::{GridSpec, ProjectedGrid, ProjectionKind};
use wx_radar::level2::{censor, Level2File, Level2Sweep};
use wx_radar::products::RadarProduct;

use crate::decode::{agree_sweep_layouts, cut_complete, cut_instants, resolve_site_from};
use crate::pack;
use crate::s3::{boxed_error, hex_sha256, iso8601, iso8601_ms, parse_time, parse_volume_key};

/// The grid descriptor Python writes (`gpuwm/obs/radar_tten_grid.py`).
pub const GRID_SCHEMA: &str = "gpuwm-obs.radar-tten-grid.v1";
/// The receipt this subcommand writes beside the product.
pub const REF_SCHEMA: &str = "gpuwm-obs.radar-tten-ref.v1";

/// ref2tten's "the radar looked and saw no echo" value.
pub const OBSERVED_NO_ECHO: f32 = -99.0;
/// ref2tten's "no radar sampled this cell" value.
pub const NO_COVERAGE: f32 = -99999.0;

/// WRF's sphere, `gpuwm.static.projection.EARTH_RADIUS_M`.
pub const EARTH_RADIUS_M: f64 = 6_370_000.0;
/// `gpuwm.obs.geometry.REFRACTION_FACTOR`.
pub const REFRACTION_FACTOR: f64 = 4.0 / 3.0;
/// `horizontal_window`'s margin, `superob._WINDOW_MARGIN_CELLS`.
const WINDOW_MARGIN_CELLS: f64 = 2.0;

/// Declared non-meteorological filter: RhoHV strictly below this ...
pub const NONMET_RHO_MAX: f32 = 0.80;
/// ... while reflectivity is strictly below this, dBZ.
pub const NONMET_SHIELD_DBZ: f32 = 45.0;

const REF: RadarProduct = RadarProduct::Reflectivity;
const RHO: RadarProduct = RadarProduct::CorrelationCoefficient;
const DEG: f64 = std::f64::consts::PI / 180.0;

pub const QC_STATEMENT: &str = "declared simple filter, not the Python superob CC QC: \
drop range-folded gates (censor code 2) and any other non-measured code except \
below-threshold; drop a measured gate whose co-located RhoHV (same radial, nearest \
RhoHV gate) is finite and < 0.80 while its reflectivity is < 45 dBZ; no dBZ floor";

// ---------------------------------------------------------------------------
// geometry: gpuwm/obs/geometry.py, transcribed
// ---------------------------------------------------------------------------

/// `beam_geometry`: gate height MSL and surface arc for one gate.
///
/// Same expressions, same association, so the float64 arithmetic matches the
/// NumPy path wherever the transcendental functions agree.
pub fn beam_geometry(slant_range_m: f64, elevation_deg: f64, radar_alt_m: f64) -> (f64, f64) {
    let effective_radius = EARTH_RADIUS_M * REFRACTION_FACTOR;
    let radius = effective_radius + radar_alt_m;
    let sin_el = (elevation_deg * DEG).sin();
    let cos_el = (elevation_deg * DEG).cos();
    let slant_from_centre = (slant_range_m * slant_range_m
        + radius * radius
        + 2.0 * slant_range_m * radius * sin_el)
        .sqrt();
    let height_above_antenna = slant_from_centre - radius;
    let height_msl = height_above_antenna + radar_alt_m;
    let centre_angle = (slant_range_m * cos_el / slant_from_centre)
        .clamp(-1.0, 1.0)
        .asin();
    (height_msl, centre_angle * effective_radius)
}

/// `great_circle_point`: (lat, lon) degrees at a bearing and surface arc.
pub fn great_circle_point(lat_deg: f64, lon_deg: f64, azimuth_deg: f64, arc_m: f64) -> (f64, f64) {
    let delta = arc_m / EARTH_RADIUS_M;
    let lat0 = lat_deg * DEG;
    let lon0 = lon_deg * DEG;
    let az = azimuth_deg * DEG;
    let sin_lat = (lat0.sin() * delta.cos() + lat0.cos() * delta.sin() * az.cos()).clamp(-1.0, 1.0);
    let lat = sin_lat.asin();
    let lon = lon0
        + (az.sin() * delta.sin() * lat0.cos()).atan2(delta.cos() - lat0.sin() * sin_lat);
    // Python's `%` with a positive divisor is the Euclidean remainder.
    let lon_out = (lon / DEG + 180.0).rem_euclid(360.0) - 180.0;
    (lat / DEG, lon_out)
}

/// One gate's position: (lat, lon, height MSL).
pub fn gate_location(
    site_lat_deg: f64,
    site_lon_deg: f64,
    site_alt_m: f64,
    azimuth_deg: f64,
    slant_range_m: f64,
    elevation_deg: f64,
) -> (f64, f64, f64) {
    let (height, arc) = beam_geometry(slant_range_m, elevation_deg, site_alt_m);
    let (lat, lon) = great_circle_point(site_lat_deg, site_lon_deg, azimuth_deg, arc);
    (lat, lon, height)
}

// ---------------------------------------------------------------------------
// the model grid
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ArrayRef {
    file: String,
    dtype: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct GridFile {
    schema: String,
    identity_sha256: String,
    #[serde(default)]
    name: Option<String>,
    nx: usize,
    ny: usize,
    nz: usize,
    projection: GridSpec,
    z_w: ArrayRef,
}

/// A model grid as this stage needs it: the projection, and every column's
/// layer-interface heights.
pub struct ModelGrid {
    pub name: String,
    pub identity_sha256: String,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx_m: f64,
    pub dy_m: f64,
    pub projection: ProjectedGrid,
    /// `(ny * nx)` columns of `nz + 1` heights, column after column.
    z_cols: Vec<f64>,
    /// Mass-point latitude/longitude, row-major, for reach windows only.
    lat: Vec<f64>,
    lon: Vec<f64>,
}

impl ModelGrid {
    /// Build from a projection spec and a C-ordered `(nz + 1, ny, nx)` `z_w`.
    pub fn new(
        name: String,
        identity_sha256: String,
        spec: GridSpec,
        nx: usize,
        ny: usize,
        nz: usize,
        z_w: &[f64],
    ) -> Result<Self, String> {
        if spec.kind == ProjectionKind::Rows {
            return Err(
                "grid-ref places gates on WPS projections (lambert, mercator, polar); a \
                 `rows` grid has no WPS mass-point registration to round a gate onto"
                    .to_string(),
            );
        }
        if spec.e_we != nx as i64 + 1 || spec.e_sn != ny as i64 + 1 {
            return Err(format!(
                "grid projection declares e_we={} e_sn={}, which is not the {nx} x {ny} mass \
                 grid the descriptor states (e_we must be nx+1, e_sn ny+1)",
                spec.e_we, spec.e_sn
            ));
        }
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(format!("grid has an empty axis: nx={nx} ny={ny} nz={nz}"));
        }
        let cells = nx * ny;
        if z_w.len() != (nz + 1) * cells {
            return Err(format!(
                "z_w holds {} values, expected (nz+1)*ny*nx = {}",
                z_w.len(),
                (nz + 1) * cells
            ));
        }
        let dx_m = spec.dx;
        let dy_m = spec.dy;
        let projection = ProjectedGrid::new(spec).map_err(|e| e.to_string())?;
        let levels = nz + 1;
        let mut z_cols = vec![0.0f64; levels * cells];
        z_cols
            .par_chunks_mut(levels)
            .enumerate()
            .for_each(|(cell, column)| {
                for (k, slot) in column.iter_mut().enumerate() {
                    *slot = z_w[k * cells + cell];
                }
            });
        // The same refusals `TargetGrid.__post_init__` makes.
        let bad = z_cols.par_chunks(levels).position_any(|column| {
            column.iter().any(|z| !z.is_finite()) || column.windows(2).any(|w| w[1] <= w[0])
        });
        if let Some(cell) = bad {
            return Err(format!(
                "z_w column (j={}, i={}) is non-finite or does not increase monotonically with level",
                cell / nx,
                cell % nx
            ));
        }
        let mut lat = vec![0.0f64; cells];
        let mut lon = vec![0.0f64; cells];
        lat.par_iter_mut()
            .zip(lon.par_iter_mut())
            .enumerate()
            .for_each(|(cell, (la, lo))| {
                let (a, b) = projection
                    .ij_to_latlon((cell % nx) as f64 + 1.0, (cell / nx) as f64 + 1.0);
                *la = a;
                *lo = b;
            });
        Ok(Self {
            name,
            identity_sha256,
            nx,
            ny,
            nz,
            dx_m,
            dy_m,
            projection,
            z_cols,
            lat,
            lon,
        })
    }

    /// Read the descriptor Python wrote and the raw `z_w` it names.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read grid descriptor {}: {e}", path.display()))?;
        let file: GridFile = serde_json::from_str(&text)
            .map_err(|e| format!("grid descriptor {} is not a {GRID_SCHEMA} document: {e}", path.display()))?;
        if file.schema != GRID_SCHEMA {
            return Err(format!(
                "grid descriptor {} declares schema {:?}, expected {GRID_SCHEMA:?}",
                path.display(),
                file.schema
            ));
        }
        if file.z_w.dtype != "<f8" || file.z_w.shape != [file.nz + 1, file.ny, file.nx] {
            return Err(format!(
                "grid z_w must be <f8 of shape [nz+1, ny, nx] = [{}, {}, {}], got {} {:?}",
                file.nz + 1,
                file.ny,
                file.nx,
                file.z_w.dtype,
                file.z_w.shape
            ));
        }
        let z_path = {
            let named = PathBuf::from(&file.z_w.file);
            if named.is_absolute() {
                named
            } else {
                path.parent().unwrap_or(Path::new(".")).join(named)
            }
        };
        let bytes = std::fs::read(&z_path)
            .map_err(|e| format!("cannot read grid z_w {}: {e}", z_path.display()))?;
        let digest = hex_sha256(&bytes);
        if digest != file.z_w.sha256 {
            return Err(format!(
                "grid z_w {} has sha256 {digest}, the descriptor names {}; refusing a column \
                 structure that is not the one the grid identity was taken over",
                z_path.display(),
                file.z_w.sha256
            ));
        }
        if bytes.len() % 8 != 0 {
            return Err(format!("grid z_w {} is not whole float64 words", z_path.display()));
        }
        let z_w: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|w| f64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]))
            .collect();
        Self::new(
            file.name.unwrap_or_else(|| "target".to_string()),
            file.identity_sha256,
            file.projection,
            file.nx,
            file.ny,
            file.nz,
            &z_w,
        )
    }

    /// Mass cell `cell = j * nx + i`: its latitude and longitude, degrees,
    /// on this grid's projection.  Shared with `grid-composite`, which maps
    /// every model column onto a 2D source lattice through these.
    pub fn cell_latlon(&self, cell: usize) -> (f64, f64) {
        (self.lat[cell], self.lon[cell])
    }

    /// Column (i, j)'s `nz + 1` layer-interface heights MSL, bottom up.
    /// `column_z_w(i, j)[0]` is the terrain surface.
    pub fn column_z_w(&self, i: usize, j: usize) -> &[f64] {
        let levels = self.nz + 1;
        let start = (j * self.nx + i) * levels;
        &self.z_cols[start..start + levels]
    }

    /// `TargetGrid.mass_index`: zero-based fractional (i, j).
    pub fn mass_index(&self, lat: f64, lon: f64) -> (f64, f64) {
        let (x, y) = self.projection.latlon_to_ij(lat, lon);
        (x - 1.0, y - 1.0)
    }

    /// Nearest mass cell, `numpy.rint` then `TargetGrid.inside`.
    pub fn nearest_cell(&self, lat: f64, lon: f64) -> Option<(usize, usize)> {
        let (x, y) = self.mass_index(lat, lon);
        let i = x.round_ties_even();
        let j = y.round_ties_even();
        if i >= 0.0 && i < self.nx as f64 && j >= 0.0 && j < self.ny as f64 {
            Some((i as usize, j as usize))
        } else {
            None
        }
    }

    /// `TargetGrid.level_index` for one gate in column (i, j).
    pub fn level(&self, i: usize, j: usize, height_msl_m: f64) -> Option<usize> {
        let levels = self.nz + 1;
        let start = (j * self.nx + i) * levels;
        let column = &self.z_cols[start..start + levels];
        if !(height_msl_m >= column[0] && height_msl_m < column[self.nz]) {
            return None;
        }
        Some(column[1..].partition_point(|&z| z <= height_msl_m))
    }

    /// `superob.horizontal_window`: the inclusive (j0, j1, i0, i1) box of
    /// cells a radar at (lat, lon) can reach.  Haversine on the same sphere.
    pub fn reach_window(&self, lat_deg: f64, lon_deg: f64, max_range_km: f64) -> (usize, usize, usize, usize) {
        let reach_m = max_range_km * 1000.0 + WINDOW_MARGIN_CELLS * self.dx_m.max(self.dy_m);
        let site_lat = lat_deg.to_radians();
        let site_lon = lon_deg.to_radians();
        let (mut j0, mut j1, mut i0, mut i1) = (usize::MAX, 0usize, usize::MAX, 0usize);
        for j in 0..self.ny {
            for i in 0..self.nx {
                let cell = j * self.nx + i;
                let lat = self.lat[cell].to_radians();
                let lon = self.lon[cell].to_radians();
                let a = ((lat - site_lat) / 2.0).sin().powi(2)
                    + site_lat.cos() * lat.cos() * ((lon - site_lon) / 2.0).sin().powi(2);
                let distance = 2.0 * EARTH_RADIUS_M * a.clamp(0.0, 1.0).sqrt().asin();
                if distance <= reach_m {
                    j0 = j0.min(j);
                    j1 = j1.max(j);
                    i0 = i0.min(i);
                    i1 = i1.max(i);
                }
            }
        }
        if j0 == usize::MAX {
            (0, 0, 0, 0)
        } else {
            (j0, j1, i0, i1)
        }
    }
}

// ---------------------------------------------------------------------------
// accumulation
// ---------------------------------------------------------------------------

/// How every gate of a volume was spent.  Each considered gate lands in
/// exactly one of the `echo_gates`, `clear_gates` or `dropped_*` counters.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct GateCounts {
    pub sweeps_used: u64,
    pub sweeps_skipped_elevation: u64,
    pub sweeps_without_ref: u64,
    pub radials_without_ref: u64,
    /// Reflectivity gates the radials declared beyond `--max-range-km`.
    pub gates_beyond_range: u64,
    /// Reflectivity gates inside the range ceiling: the population below.
    pub gates_considered: u64,
    pub echo_gates: u64,
    pub clear_gates: u64,
    pub dropped_range_folded: u64,
    pub dropped_other_censor: u64,
    pub dropped_nonfinite_measured: u64,
    pub dropped_nonmet_rho: u64,
    pub dropped_out_of_grid: u64,
    pub dropped_out_of_column: u64,
    /// Measured gates whose radial carried RhoHV at that range.
    pub gates_with_rho: u64,
}

impl GateCounts {
    fn add(&mut self, o: &GateCounts) {
        self.sweeps_used += o.sweeps_used;
        self.sweeps_skipped_elevation += o.sweeps_skipped_elevation;
        self.sweeps_without_ref += o.sweeps_without_ref;
        self.radials_without_ref += o.radials_without_ref;
        self.gates_beyond_range += o.gates_beyond_range;
        self.gates_considered += o.gates_considered;
        self.echo_gates += o.echo_gates;
        self.clear_gates += o.clear_gates;
        self.dropped_range_folded += o.dropped_range_folded;
        self.dropped_other_censor += o.dropped_other_censor;
        self.dropped_nonfinite_measured += o.dropped_nonfinite_measured;
        self.dropped_nonmet_rho += o.dropped_nonmet_rho;
        self.dropped_out_of_grid += o.dropped_out_of_grid;
        self.dropped_out_of_column += o.dropped_out_of_column;
        self.gates_with_rho += o.gates_with_rho;
    }

    /// Every considered gate is accounted for exactly once.
    pub fn balances(&self) -> bool {
        self.gates_considered
            == self.echo_gates
                + self.clear_gates
                + self.dropped_range_folded
                + self.dropped_other_censor
                + self.dropped_nonfinite_measured
                + self.dropped_nonmet_rho
                + self.dropped_out_of_grid
                + self.dropped_out_of_column
    }
}

/// One radar's accumulators over the cells it can reach.
pub struct Window {
    pub j0: usize,
    pub i0: usize,
    pub nj: usize,
    pub ni: usize,
    pub nz: usize,
    pub linear_sum: Vec<f64>,
    pub echo_count: Vec<u32>,
    pub max_dbz: Vec<f32>,
    pub clear_count: Vec<u32>,
}

impl Window {
    fn new(nz: usize, (j0, j1, i0, i1): (usize, usize, usize, usize)) -> Self {
        let nj = j1 - j0 + 1;
        let ni = i1 - i0 + 1;
        let n = nz * nj * ni;
        Self {
            j0,
            i0,
            nj,
            ni,
            nz,
            linear_sum: vec![0.0; n],
            echo_count: vec![0; n],
            max_dbz: vec![f32::NEG_INFINITY; n],
            clear_count: vec![0; n],
        }
    }

    fn local(&self, k: usize, j: usize, i: usize) -> Option<usize> {
        if j < self.j0 || i < self.i0 {
            return None;
        }
        let (jl, il) = (j - self.j0, i - self.i0);
        if jl >= self.nj || il >= self.ni {
            return None;
        }
        Some((k * self.nj + jl) * self.ni + il)
    }
}

/// One placed gate: its cell inside the volume's window, and its dBZ
/// (NaN for a below-threshold gate).
type Placement = (u32, f32);

#[derive(Debug, Clone, Copy)]
pub struct GridParams {
    pub max_range_km: f64,
    pub max_elevation_deg: f64,
}

/// Place one sweep's reflectivity gates.  Radials are placed in parallel
/// and returned in radial order, so the caller's accumulation order is the
/// gate order whatever the thread count.
fn place_sweep(
    sweep: &Level2Sweep,
    site: (f64, f64, f64),
    grid: &ModelGrid,
    window: &Window,
    params: &GridParams,
) -> Result<(Vec<Vec<Placement>>, GateCounts), String> {
    let mut counts = GateCounts::default();
    let (layouts, _trimmed) = agree_sweep_layouts(sweep, &[REF, RHO], params.max_range_km)?;
    let Some(ref_layout) = layouts.iter().find(|l| l.product == REF).cloned() else {
        counts.sweeps_without_ref += 1;
        return Ok((Vec::new(), counts));
    };
    let rho_layout = layouts.iter().find(|l| l.product == RHO).cloned();
    counts.sweeps_used += 1;
    let max_range_m = params.max_range_km * 1000.0;
    let (site_lat, site_lon, site_alt) = (site.0, site.1, site.2);

    let per_radial: Vec<Result<(Vec<Placement>, GateCounts), String>> = sweep
        .radials
        .par_iter()
        .map(|radial| {
            let mut c = GateCounts::default();
            let mut out: Vec<Placement> = Vec::new();
            let Some(moment) = radial.moments.iter().find(|m| m.product == REF) else {
                c.radials_without_ref += 1;
                return Ok((out, c));
            };
            let rho = rho_layout.as_ref().and_then(|layout| {
                radial
                    .moments
                    .iter()
                    .find(|m| m.product == RHO)
                    .map(|m| (m, layout.gates))
            });
            c.gates_beyond_range += (ref_layout.declared_gates - ref_layout.gates) as u64;
            let azimuth = radial.azimuth as f64;
            let elevation = radial.elevation as f64;
            for gate in 0..ref_layout.gates {
                let range = ref_layout.first_gate_range_m + ref_layout.gate_size_m * gate as f64;
                if range > max_range_m {
                    c.gates_beyond_range += 1;
                    continue;
                }
                c.gates_considered += 1;
                let value = moment.data[gate];
                let code = moment.censor[gate];
                let echo = match code {
                    censor::MEASURED => {
                        if !value.is_finite() {
                            c.dropped_nonfinite_measured += 1;
                            continue;
                        }
                        true
                    }
                    censor::BELOW_THRESHOLD => false,
                    censor::RANGE_FOLDED => {
                        c.dropped_range_folded += 1;
                        continue;
                    }
                    _ => {
                        c.dropped_other_censor += 1;
                        continue;
                    }
                };
                if echo {
                    if let Some((rho_moment, rho_gates)) = rho {
                        let first = rho_moment.first_gate_range as f64;
                        let size = rho_moment.gate_size as f64;
                        if size > 0.0 {
                            let index = ((range - first) / size).round();
                            if index >= 0.0 && (index as usize) < rho_gates {
                                let rho_value = rho_moment.data[index as usize];
                                if rho_value.is_finite() {
                                    c.gates_with_rho += 1;
                                    if rho_value < NONMET_RHO_MAX && value < NONMET_SHIELD_DBZ {
                                        c.dropped_nonmet_rho += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
                let (lat, lon, height) =
                    gate_location(site_lat, site_lon, site_alt, azimuth, range, elevation);
                let Some((i, j)) = grid.nearest_cell(lat, lon) else {
                    c.dropped_out_of_grid += 1;
                    continue;
                };
                let Some(k) = grid.level(i, j, height) else {
                    c.dropped_out_of_column += 1;
                    continue;
                };
                let Some(local) = window.local(k, j, i) else {
                    return Err(format!(
                        "a gate fell outside the computed reach window j[{}..{}] i[{}..{}] at \
                         (j={j}, i={i}); the window under-covers this radar and gates would be \
                         lost. This is a bug in reach_window(), not a data problem",
                        window.j0,
                        window.j0 + window.nj - 1,
                        window.i0,
                        window.i0 + window.ni - 1
                    ));
                };
                if echo {
                    c.echo_gates += 1;
                    out.push((local as u32, value));
                } else {
                    c.clear_gates += 1;
                    out.push((local as u32, f32::NAN));
                }
            }
            Ok((out, c))
        })
        .collect();

    let mut placements = Vec::with_capacity(per_radial.len());
    for result in per_radial {
        let (radial, c) = result?;
        counts.add(&c);
        placements.push(radial);
    }
    Ok((placements, counts))
}

/// Grid one parsed volume into its own reach window.
pub fn grid_volume(
    file: &Level2File,
    site: (f64, f64, f64),
    grid: &ModelGrid,
    params: &GridParams,
) -> Result<(Window, GateCounts), String> {
    let mut window = Window::new(grid.nz, grid.reach_window(site.0, site.1, params.max_range_km));
    let mut counts = GateCounts::default();
    for sweep in &file.sweeps {
        if sweep.elevation_angle as f64 > params.max_elevation_deg {
            counts.sweeps_skipped_elevation += 1;
            continue;
        }
        if sweep.radials.is_empty() {
            continue;
        }
        let (placements, c) = place_sweep(sweep, site, grid, &window, params)?;
        counts.add(&c);
        for radial in placements {
            for (cell, dbz) in radial {
                let cell = cell as usize;
                if dbz.is_nan() {
                    window.clear_count[cell] += 1;
                } else {
                    window.linear_sum[cell] += 10f64.powf(dbz as f64 / 10.0);
                    window.echo_count[cell] += 1;
                    if dbz > window.max_dbz[cell] {
                        window.max_dbz[cell] = dbz;
                    }
                }
            }
        }
    }
    Ok((window, counts))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Reduce {
    Mean,
    Max,
}

/// The domain product, accumulated from windows in the order given.
pub struct Domain {
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
    linear_sum: Vec<f64>,
    echo_count: Vec<u32>,
    max_dbz: Vec<f32>,
    clear_count: Vec<u32>,
}

impl Domain {
    pub fn new(nz: usize, ny: usize, nx: usize) -> Self {
        let n = nz * ny * nx;
        Self {
            nz,
            ny,
            nx,
            linear_sum: vec![0.0; n],
            echo_count: vec![0; n],
            max_dbz: vec![f32::NEG_INFINITY; n],
            clear_count: vec![0; n],
        }
    }

    pub fn add(&mut self, w: &Window) {
        for k in 0..w.nz {
            for jl in 0..w.nj {
                let row = (k * self.ny + (w.j0 + jl)) * self.nx + w.i0;
                let wrow = (k * w.nj + jl) * w.ni;
                for il in 0..w.ni {
                    let (d, s) = (row + il, wrow + il);
                    self.linear_sum[d] += w.linear_sum[s];
                    self.echo_count[d] += w.echo_count[s];
                    if w.max_dbz[s] > self.max_dbz[d] {
                        self.max_dbz[d] = w.max_dbz[s];
                    }
                    self.clear_count[d] += w.clear_count[s];
                }
            }
        }
    }

    /// The ref2tten field and its cell census (echo, clear, no coverage).
    pub fn reduce(&self, reduce: Reduce) -> (Vec<f32>, [u64; 3]) {
        let mut census = [0u64; 3];
        let field: Vec<f32> = (0..self.linear_sum.len())
            .map(|cell| {
                if self.echo_count[cell] > 0 {
                    census[0] += 1;
                    match reduce {
                        Reduce::Max => self.max_dbz[cell],
                        Reduce::Mean => (10.0
                            * (self.linear_sum[cell].max(1e-30) / self.echo_count[cell] as f64)
                                .log10()) as f32,
                    }
                } else if self.clear_count[cell] > 0 {
                    census[1] += 1;
                    OBSERVED_NO_ECHO
                } else {
                    census[2] += 1;
                    NO_COVERAGE
                }
            })
            .collect();
        (field, census)
    }
}

// ---------------------------------------------------------------------------
// volumes
// ---------------------------------------------------------------------------

/// A decoded volume and the facts its receipt row needs.
pub struct DecodedVolume {
    pub path: PathBuf,
    pub file_name: String,
    pub bytes: usize,
    pub sha256: String,
    pub file: Level2File,
    pub site_id: String,
    pub site_lat: f64,
    pub site_lon: f64,
    pub site_alt: f64,
    pub site_source: String,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub complete: bool,
    pub sweeps_in_volume: usize,
    pub sweeps_incomplete: usize,
    pub key_time: Option<String>,
    pub decode_seconds: f64,
}

/// Read, frame-check, strictly parse and date one volume: the same path
/// `rw_nexrad decode` takes, without writing a pack.
pub fn decode_volume(path: &Path) -> Result<DecodedVolume, String> {
    let started = Instant::now();
    let raw = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (stream, _framing) = pack::read_volume(&raw).map_err(|e| e.to_string())?;
    let file = Level2File::parse_strict(&stream)?;
    pack::validate_decoded(&file).map_err(|e| e.to_string())?;
    let site = resolve_site_from(&file.station_id, None, file.vol_site.as_ref())
        .map_err(|e| e.to_string())?;
    let mut start: Option<DateTime<Utc>> = None;
    let mut end: Option<DateTime<Utc>> = None;
    let mut sweeps_incomplete = 0usize;
    for sweep in &file.sweeps {
        let (first, last) = cut_instants(sweep)?;
        start = Some(start.map_or(first, |v| v.min(first)));
        end = Some(end.map_or(last, |v| v.max(last)));
        if !cut_complete(sweep) {
            sweeps_incomplete += 1;
        }
    }
    let complete = sweeps_incomplete == 0
        && file.sweeps.first().is_some_and(|s| s.start_status == 3)
        && file.sweeps.last().is_some_and(|s| s.end_status == 4);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    let key_time = parse_volume_key(&file_name).map(|k| iso8601(k.valid_time));
    let sweeps_in_volume = file.sweeps.len();
    let (start, end) = match (start, end) {
        (Some(s), Some(e)) => (s, e),
        _ => return Err("volume carried no datable sweep".to_string()),
    };
    // Keep only what gridding reads; the other moments are the bulk of a
    // decoded volume and many volumes are in flight at once.
    let mut file = file;
    for sweep in &mut file.sweeps {
        for radial in &mut sweep.radials {
            radial.moments.retain(|m| m.product == REF || m.product == RHO);
        }
    }
    Ok(DecodedVolume {
        path: path.to_path_buf(),
        file_name,
        bytes: raw.len(),
        sha256: hex_sha256(&raw),
        site_id: site.id,
        site_lat: site.lat_deg,
        site_lon: site.lon_deg,
        site_alt: site.alt_m,
        site_source: site.source,
        file,
        start,
        end,
        complete,
        sweeps_in_volume,
        sweeps_incomplete,
        key_time,
        decode_seconds: started.elapsed().as_secs_f64(),
    })
}

/// The causal rule `tools/da_recent_observations.causal_volume` applies:
/// a complete measured roster that ended at or before the window end and
/// at most `max_age_s` before it.
pub fn causal_refusal(volume: &DecodedVolume, window_end: DateTime<Utc>, max_age_s: f64) -> Option<String> {
    if !volume.complete || volume.sweeps_incomplete != 0 || volume.sweeps_in_volume == 0 {
        return Some("radar volume is not a complete measured sweep roster".to_string());
    }
    if volume.end > window_end {
        return Some(format!(
            "radar volume finishes at {} after the window end {}",
            iso8601_ms(volume.end),
            iso8601(window_end)
        ));
    }
    let age = (window_end - volume.end).num_milliseconds() as f64 / 1000.0;
    if age > max_age_s {
        return Some(format!("radar measured end is stale ({age:.1} s > {max_age_s} s)"));
    }
    None
}

#[derive(Serialize)]
struct VolumeRow {
    site: String,
    site_lat_deg: f64,
    site_lon_deg: f64,
    site_alt_m: f64,
    site_source: String,
    key: String,
    path: String,
    bytes: usize,
    sha256: String,
    key_time: Option<String>,
    volume_start: String,
    volume_end: String,
    complete: bool,
    window: [usize; 4],
    counts: GateCounts,
    decode_seconds: f64,
    grid_seconds: f64,
}

#[derive(Serialize)]
struct RefusedRow {
    path: String,
    site: Option<String>,
    reason: String,
}

/// Everything one grid-ref call needs.
pub struct GridRefRequest {
    pub grid: PathBuf,
    pub volumes: Vec<PathBuf>,
    pub out_prefix: PathBuf,
    /// Where the float32 field goes; `PREFIX.ref.f32` when not given.
    pub out_data: Option<PathBuf>,
    pub reduce: Reduce,
    pub params: GridParams,
    pub threads: usize,
    pub window_end: Option<DateTime<Utc>>,
    pub max_age_s: f64,
}

struct GridRefResult {
    field: Vec<f32>,
    census: [u64; 3],
    rows: Vec<VolumeRow>,
    refused: Vec<RefusedRow>,
    totals: GateCounts,
    seconds: serde_json::Map<String, serde_json::Value>,
}

fn volume_group_key(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    match parse_volume_key(&name) {
        Some(key) => key.site,
        // An unnamed volume is its own group.
        None => format!("?{}", path.display()),
    }
}

/// Decode and grid every volume (or, with a window end, the newest causal
/// volume per site), then merge in a fixed order.
fn run_volumes(request: &GridRefRequest, grid: &ModelGrid) -> Result<GridRefResult, String> {
    let mut seconds = serde_json::Map::new();
    // Groups: one per site under a window end (newest key first), else one
    // per volume.
    let mut groups: Vec<(String, Vec<PathBuf>)> = Vec::new();
    if request.window_end.is_some() {
        for path in &request.volumes {
            let key = volume_group_key(path);
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, list)) => list.push(path.clone()),
                None => groups.push((key, vec![path.clone()])),
            }
        }
        for (_, list) in &mut groups {
            list.sort_by(|a, b| {
                let ka = parse_volume_key(&a.file_name().unwrap_or_default().to_string_lossy())
                    .map(|k| k.valid_time);
                let kb = parse_volume_key(&b.file_name().unwrap_or_default().to_string_lossy())
                    .map(|k| k.valid_time);
                kb.cmp(&ka).then_with(|| b.cmp(a))
            });
        }
    } else {
        for path in &request.volumes {
            groups.push((volume_group_key(path), vec![path.clone()]));
        }
    }

    let started = Instant::now();
    type GroupOut = (Option<(DecodedVolume, Window, GateCounts, f64)>, Vec<RefusedRow>);
    let outcomes: Vec<Result<GroupOut, String>> = groups
        .par_iter()
        .map(|(_, paths)| {
            let mut refused = Vec::new();
            for path in paths {
                let decoded = match decode_volume(path) {
                    Ok(v) => v,
                    Err(reason) => {
                        refused.push(RefusedRow {
                            path: path.display().to_string(),
                            site: None,
                            reason: format!("decode: {reason}"),
                        });
                        continue;
                    }
                };
                if let Some(end) = request.window_end {
                    if let Some(reason) = causal_refusal(&decoded, end, request.max_age_s) {
                        refused.push(RefusedRow {
                            path: path.display().to_string(),
                            site: Some(decoded.site_id.clone()),
                            reason,
                        });
                        continue;
                    }
                }
                let t0 = Instant::now();
                let site = (decoded.site_lat, decoded.site_lon, decoded.site_alt);
                match grid_volume(&decoded.file, site, grid, &request.params) {
                    Ok((window, counts)) => {
                        let mut decoded = decoded;
                        // The gates are in the window now.
                        decoded.file.sweeps.clear();
                        return Ok((Some((decoded, window, counts, t0.elapsed().as_secs_f64())), refused));
                    }
                    Err(reason) => {
                        if reason.contains("bug in reach_window") {
                            return Err(reason);
                        }
                        refused.push(RefusedRow {
                            path: path.display().to_string(),
                            site: Some(decoded.site_id.clone()),
                            reason: format!("grid: {reason}"),
                        });
                    }
                }
            }
            Ok((None, refused))
        })
        .collect();
    seconds.insert("decode_and_grid_wall".into(), round3(started.elapsed().as_secs_f64()).into());

    let mut used = Vec::new();
    let mut refused = Vec::new();
    for outcome in outcomes {
        let (volume, mut rows) = outcome?;
        refused.append(&mut rows);
        if let Some(v) = volume {
            used.push(v);
        }
    }
    // The fixed merge order: site, measured start, file name.
    used.sort_by(|a, b| {
        a.0.site_id
            .cmp(&b.0.site_id)
            .then(a.0.start.cmp(&b.0.start))
            .then(a.0.file_name.cmp(&b.0.file_name))
    });

    let t0 = Instant::now();
    let mut domain = Domain::new(grid.nz, grid.ny, grid.nx);
    let mut totals = GateCounts::default();
    let mut rows = Vec::new();
    let mut decode_sum = 0.0;
    let mut grid_sum = 0.0;
    for (volume, window, counts, grid_seconds) in &used {
        domain.add(window);
        totals.add(counts);
        decode_sum += volume.decode_seconds;
        grid_sum += grid_seconds;
        rows.push(VolumeRow {
            site: volume.site_id.clone(),
            site_lat_deg: volume.site_lat,
            site_lon_deg: volume.site_lon,
            site_alt_m: volume.site_alt,
            site_source: volume.site_source.clone(),
            key: volume.file_name.clone(),
            path: volume.path.display().to_string(),
            bytes: volume.bytes,
            sha256: volume.sha256.clone(),
            key_time: volume.key_time.clone(),
            volume_start: iso8601_ms(volume.start),
            volume_end: iso8601_ms(volume.end),
            complete: volume.complete,
            window: [window.j0, window.j0 + window.nj - 1, window.i0, window.i0 + window.ni - 1],
            counts: counts.clone(),
            decode_seconds: round3(volume.decode_seconds),
            grid_seconds: round3(*grid_seconds),
        });
    }
    seconds.insert("decode_sum".into(), round3(decode_sum).into());
    seconds.insert("grid_sum".into(), round3(grid_sum).into());
    let (field, census) = domain.reduce(request.reduce);
    seconds.insert("merge_reduce".into(), round3(t0.elapsed().as_secs_f64()).into());
    Ok(GridRefResult {
        field,
        census,
        rows,
        refused,
        totals,
        seconds,
    })
}

/// Seconds rounded to the millisecond, as every receipt states them.
pub fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

/// Write `bytes` to a sibling temporary file, then rename it over `path`,
/// so a reader never sees a half-written product.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!(
        "{}.tmp{}",
        path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default(),
        std::process::id()
    ));
    std::fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("cannot publish {}: {e}", path.display()))
}

/// `prefix` with `suffix` appended to its last component.
pub fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut name = prefix.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

/// Run one grid-ref call and return its receipt as JSON text.
pub fn grid_ref(request: &GridRefRequest) -> Result<String, Box<dyn Error>> {
    let all = Instant::now();
    if request.volumes.is_empty() {
        return Err(boxed_error("--volumes named no volume"));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(request.threads)
        .build()
        .map_err(|e| boxed_error(format!("cannot build a {}-thread pool: {e}", request.threads)))?;
    let t0 = Instant::now();
    let grid = pool.install(|| ModelGrid::load(&request.grid)).map_err(boxed_error)?;
    let load_grid = t0.elapsed().as_secs_f64();
    let mut result = pool.install(|| run_volumes(request, &grid)).map_err(boxed_error)?;
    if result.rows.is_empty() {
        let why = result
            .refused
            .iter()
            .take(5)
            .map(|r| format!("{}: {}", r.path, r.reason))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(boxed_error(format!(
            "no volume survived to be gridded ({} refused). {why}",
            result.refused.len()
        )));
    }
    if !result.totals.balances() {
        return Err(boxed_error(format!(
            "gate accounting does not balance: {:?}",
            result.totals
        )));
    }

    let t0 = Instant::now();
    let mut bytes = Vec::with_capacity(result.field.len() * 4);
    for value in &result.field {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let data_path = request
        .out_data
        .clone()
        .unwrap_or_else(|| with_suffix(&request.out_prefix, ".ref.f32"));
    let receipt_path = with_suffix(&request.out_prefix, ".json");
    for path in [&data_path, &receipt_path] {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| boxed_error(format!("cannot create {}: {e}", parent.display())))?;
            }
        }
    }
    // The receipt names the data file relative to itself when they share a
    // directory, so a product directory can be moved whole.
    let data_name = if data_path.parent() == receipt_path.parent() {
        data_path.file_name().map(|n| n.to_string_lossy().to_string())
    } else {
        Some(data_path.display().to_string())
    };
    write_atomic(&data_path, &bytes).map_err(boxed_error)?;
    let data_sha = hex_sha256(&bytes);
    result.seconds.insert("write".into(), round3(t0.elapsed().as_secs_f64()).into());
    result.seconds.insert("load_grid".into(), round3(load_grid).into());
    result.seconds.insert("total".into(), round3(all.elapsed().as_secs_f64()).into());

    let receipt = serde_json::json!({
        "schema": REF_SCHEMA,
        "status": "READY",
        "product": "radar reflectivity on the model grid, ref2tten convention",
        "convention": {
            "dtype": "<f4",
            "order": "C",
            "shape": [grid.nz, grid.ny, grid.nx],
            "dims": ["bottom_top", "south_north", "west_east"],
            "echo": "dBZ",
            "observed_no_echo": OBSERVED_NO_ECHO,
            "no_coverage": NO_COVERAGE,
        },
        "shape": [grid.nz, grid.ny, grid.nx],
        "data": {
            "file": data_name,
            "bytes": bytes.len(),
            "sha256": data_sha,
        },
        "grid": {
            "name": grid.name,
            "identity_sha256": grid.identity_sha256,
            "descriptor": request.grid.display().to_string(),
            "map_proj": format!("{:?}", grid.projection.spec.kind).to_lowercase(),
            "nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
        },
        "reduce": request.reduce,
        "params": {
            "max_range_km": request.params.max_range_km,
            "max_elevation_deg": request.params.max_elevation_deg,
            "earth_radius_m": EARTH_RADIUS_M,
            "refraction_factor": REFRACTION_FACTOR,
            "horizontal": "nearest mass cell: rint(latlon_to_ij - 1), as TargetGrid.mass_index",
            "vertical": "model layer whose z_w interfaces bracket the gate height, as TargetGrid.level_index",
        },
        "qc": {
            "statement": QC_STATEMENT,
            "nonmet_rho_max": NONMET_RHO_MAX,
            "nonmet_shield_dbz": NONMET_SHIELD_DBZ,
            "clear_air": "a cell with >= 1 below-threshold (censor code 1) gate and no echo gate",
        },
        "window": request.window_end.map(|end| serde_json::json!({
            "end": iso8601(end),
            "max_age_s": request.max_age_s,
            "rule": "per site, the newest volume (by key time) that is complete and whose measured end is at or before the window end and at most max_age_s before it",
        })),
        "counts": {
            "volumes_used": result.rows.len(),
            "volumes_refused": result.refused.len(),
            "echo_cells": result.census[0],
            "clear_cells": result.census[1],
            "no_coverage_cells": result.census[2],
            "gates_used": result.totals.echo_gates + result.totals.clear_gates,
            "gates": result.totals,
        },
        "volumes": result.rows,
        "refused_volumes": result.refused,
        "threads": request.threads,
        "seconds": result.seconds,
        "binary": format!("rw_nexrad {}", env!("CARGO_PKG_VERSION")),
    });
    let text = format!("{}\n", serde_json::to_string_pretty(&receipt)?);
    write_atomic(&receipt_path, text.as_bytes()).map_err(boxed_error)?;
    Ok(text)
}

pub const GRID_REF_USAGE: &str = "\
usage: rw_nexrad grid-ref --grid GRID.json --volumes V1,V2,... --out PREFIX [OPTIONS]

  Decode raw Level-II volumes and grid their reflectivity, all radars together,
  onto a model grid in NOAA's ref2tten convention: float32 C-order (nz, ny, nx)
  written to PREFIX.ref.f32, echo cells in dBZ, observed-no-echo -99.0, no
  coverage -99999.0, plus the receipt PREFIX.json (gpuwm-obs.radar-tten-ref.v1).

  --grid FILE             gpuwm-obs.radar-tten-grid.v1 descriptor (written by
                          gpuwm/obs/radar_tten_grid.py) naming the raw z_w file
  --volumes LIST          comma-separated Level-II volume files
  --volume-list FILE      one volume path per line (added to --volumes)
  --out PREFIX            output prefix: PREFIX.json is the receipt
  --out-data FILE         the float32 field (default PREFIX.ref.f32)
  --reduce mean|max       mean of linear Z (default) or max dBZ per cell
  --max-range-km KM       slant-range ceiling (default 250)
  --max-elevation-deg DEG drop sweeps above this elevation (default 20)
  --threads N             worker threads (default: available cores, at most 64)
  --window-end TIME       causal mode: per site use only the newest complete
                          volume whose measured end is at or before TIME
  --max-age-s S           causal mode staleness bound (default 900)
";

/// Parse the grid-ref arguments (everything after the subcommand).
pub fn parse_request(args: &[String]) -> Result<GridRefRequest, Box<dyn Error>> {
    let mut grid = None;
    let mut volumes: Vec<PathBuf> = Vec::new();
    let mut out = None;
    let mut out_data = None;
    let mut reduce = Reduce::Mean;
    let mut max_range_km = 250.0;
    let mut max_elevation_deg = 20.0;
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(64);
    let mut window_end = None;
    let mut max_age_s = 900.0;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let mut value = || -> Result<String, Box<dyn Error>> {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| boxed_error(format!("{flag} needs a value")))
        };
        let positive = |raw: &str, name: &str| -> Result<f64, Box<dyn Error>> {
            match raw.parse::<f64>() {
                Ok(v) if v.is_finite() && v > 0.0 => Ok(v),
                _ => Err(boxed_error(format!("{name} must be a finite positive number, got {raw:?}"))),
            }
        };
        match flag {
            "--grid" => grid = Some(PathBuf::from(value()?)),
            "--volumes" => volumes.extend(
                value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from),
            ),
            "--volume-list" => {
                let list = value()?;
                let text = std::fs::read_to_string(&list)
                    .map_err(|e| boxed_error(format!("cannot read --volume-list {list}: {e}")))?;
                volumes.extend(
                    text.lines()
                        .map(str::trim)
                        .filter(|s| !s.is_empty() && !s.starts_with('#'))
                        .map(PathBuf::from),
                );
            }
            "--out" => out = Some(PathBuf::from(value()?)),
            "--out-data" => out_data = Some(PathBuf::from(value()?)),
            "--reduce" => {
                reduce = match value()?.as_str() {
                    "mean" => Reduce::Mean,
                    "max" => Reduce::Max,
                    other => return Err(boxed_error(format!("--reduce must be mean or max, got {other:?}"))),
                }
            }
            "--max-range-km" => max_range_km = positive(&value()?, "--max-range-km")?,
            "--max-elevation-deg" => max_elevation_deg = positive(&value()?, "--max-elevation-deg")?,
            "--threads" => {
                let raw = value()?;
                threads = match raw.parse::<usize>() {
                    Ok(n) if n > 0 => n,
                    _ => return Err(boxed_error(format!("--threads must be a positive count, got {raw:?}"))),
                }
            }
            "--window-end" => window_end = Some(parse_time(&value()?)?),
            "--max-age-s" => max_age_s = positive(&value()?, "--max-age-s")?,
            "--help" | "-h" => return Err(boxed_error(GRID_REF_USAGE)),
            other => return Err(boxed_error(format!("unknown grid-ref option {other:?}\n\n{GRID_REF_USAGE}"))),
        }
        index += 1;
    }
    Ok(GridRefRequest {
        grid: grid.ok_or_else(|| boxed_error("--grid is required"))?,
        volumes,
        out_prefix: out.ok_or_else(|| boxed_error("--out is required (the output prefix)"))?,
        out_data,
        reduce,
        params: GridParams {
            max_range_km,
            max_elevation_deg,
        },
        threads,
        window_end,
        max_age_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wx_radar::level2::{MomentData, RadialData};

    // -- geometry, against gpuwm/obs/geometry.py ---------------------------

    #[test]
    fn beam_geometry_matches_the_python_transcription() {
        // (slant range m, elevation deg, alt m) -> (height MSL m, arc m), from
        // gpuwm.obs.geometry.beam_geometry on box K (numpy float64).
        for &(r, el, alt, h, arc) in PY_BEAM {
            let (hh, aa) = beam_geometry(r, el, alt);
            assert!((hh - h).abs() <= 1e-6, "height {r} {el}: {hh} vs {h}");
            assert!((aa - arc).abs() <= 1e-6, "arc {r} {el}: {aa} vs {arc}");
        }
    }

    #[test]
    fn gate_locations_match_the_python_transcription() {
        for &(lat0, lon0, alt, az, r, el, lat, lon, h) in PY_GATES {
            let (la, lo, hh) = gate_location(lat0, lon0, alt, az, r, el);
            assert!((la - lat).abs() <= 1e-10, "lat {az} {r}: {la} vs {lat}");
            assert!((lo - lon).abs() <= 1e-10, "lon {az} {r}: {lo} vs {lon}");
            assert!((hh - h).abs() <= 1e-6, "height {az} {r}: {hh} vs {h}");
        }
    }

    // -- projection, against TargetGrid.mass_index on the CONUS grid -------

    fn conus_spec() -> GridSpec {
        serde_json::from_str(PY_CONUS_SPEC).expect("spec")
    }

    #[test]
    fn mass_index_matches_target_grid_on_the_conus_grid() {
        let spec = conus_spec();
        let nx = (spec.e_we - 1) as usize;
        let ny = (spec.e_sn - 1) as usize;
        let z_w: Vec<f64> = (0..3).flat_map(|k| std::iter::repeat(k as f64 * 1000.0).take(nx * ny)).collect();
        let grid = ModelGrid::new("conus".into(), "x".into(), spec, nx, ny, 2, &z_w).unwrap();
        for &(lat, lon, i, j) in PY_MASS_INDEX {
            let (x, y) = grid.mass_index(lat, lon);
            assert!((x - i).abs() <= 1e-9 && (y - j).abs() <= 1e-9, "({lat}, {lon}): ({x}, {y}) vs ({i}, {j})");
        }
    }

    // -- vertical placement --------------------------------------------------

    fn flat_grid(nx: usize, ny: usize, z_levels: &[f64]) -> ModelGrid {
        let spec = GridSpec {
            kind: ProjectionKind::Lambert,
            ref_lat: 35.3,
            ref_lon: -97.3,
            truelat1: 30.0,
            truelat2: 60.0,
            stand_lon: -97.3,
            dx: 3000.0,
            dy: 3000.0,
            e_we: nx as i64 + 1,
            e_sn: ny as i64 + 1,
            known_x: (nx as f64 + 1.0) / 2.0,
            known_y: (ny as f64 + 1.0) / 2.0,
            moad_cen_lat: 35.3,
            moad_cen_lon: -97.3,
            lat_deg: Vec::new(),
            lon0_deg: 0.0,
            dlon_deg: 0.0,
        };
        let nz = z_levels.len() - 1;
        let z_w: Vec<f64> = z_levels
            .iter()
            .flat_map(|z| std::iter::repeat(*z).take(nx * ny))
            .collect();
        ModelGrid::new("test".into(), "id".into(), spec, nx, ny, nz, &z_w).unwrap()
    }

    #[test]
    fn level_index_brackets_like_target_grid() {
        let grid = flat_grid(4, 3, &[300.0, 500.0, 1000.0, 2000.0]);
        assert_eq!(grid.level(0, 0, 299.9), None, "below terrain");
        assert_eq!(grid.level(0, 0, 300.0), Some(0));
        assert_eq!(grid.level(0, 0, 499.9), Some(0));
        assert_eq!(grid.level(0, 0, 500.0), Some(1), "an interface belongs to the layer above");
        assert_eq!(grid.level(3, 2, 1999.9), Some(2));
        assert_eq!(grid.level(3, 2, 2000.0), None, "the model top is outside");
        assert_eq!(grid.level(3, 2, f64::NAN), None);
    }

    #[test]
    fn a_column_that_does_not_increase_is_refused() {
        let spec = flat_grid(2, 2, &[0.0, 1.0]).projection.spec.clone();
        let z_w = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, -1.0, 5.0];
        let err = ModelGrid::new("t".into(), "i".into(), spec, 2, 2, 1, &z_w).err().unwrap();
        assert!(err.contains("monotonically"), "{err}");
    }

    // -- reduction and determinism -----------------------------------------

    fn ref_moment(values: &[f32], codes: &[u8]) -> MomentData {
        MomentData {
            product: REF,
            gate_count: values.len() as u16,
            first_gate_range: 2125,
            gate_size: 250,
            data: values.to_vec(),
            censor: codes.to_vec(),
        }
    }

    fn rho_moment(values: &[f32]) -> MomentData {
        MomentData {
            product: RHO,
            gate_count: values.len() as u16,
            first_gate_range: 2125,
            gate_size: 250,
            data: values.to_vec(),
            censor: vec![censor::MEASURED; values.len()],
        }
    }

    /// A synthetic volume: `radials` radials per sweep at the given
    /// elevations, every gate a pseudo-random mix of echo, clear,
    /// range-folded, and low-RhoHV clutter.
    fn synthetic_volume(seed: u32, radials: usize, gates: usize, elevations: &[f32]) -> Level2File {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let sweeps = elevations
            .iter()
            .enumerate()
            .map(|(index, &elevation)| Level2Sweep {
                elevation_number: index as u8 + 1,
                elevation_angle: elevation,
                nyquist_velocity: None,
                nyquist_radials_disagree: false,
                sweep_index: index as u16,
                start_status: if index == 0 { 3 } else { 0 },
                end_status: if index + 1 == elevations.len() { 4 } else { 2 },
                cut_sector: 0,
                radials: (0..radials)
                    .map(|row| {
                        let mut values = Vec::with_capacity(gates);
                        let mut codes = Vec::with_capacity(gates);
                        let mut rho = Vec::with_capacity(gates);
                        for _ in 0..gates {
                            let draw = next();
                            match draw % 4 {
                                0 | 1 => {
                                    values.push((draw % 7000) as f32 / 100.0 - 10.0);
                                    codes.push(censor::MEASURED);
                                }
                                2 => {
                                    values.push(f32::NAN);
                                    codes.push(censor::BELOW_THRESHOLD);
                                }
                                _ => {
                                    values.push(f32::NAN);
                                    codes.push(censor::RANGE_FOLDED);
                                }
                            }
                            rho.push(if (draw >> 8) % 5 == 0 { 0.5 } else { 0.99 });
                        }
                        RadialData {
                            azimuth: (row as f32 + 0.5) * 360.0 / radials as f32,
                            elevation,
                            azimuth_spacing: 360.0 / radials as f32,
                            nyquist_velocity: None,
                            radial_status: if row == 0 { 0 } else { 1 },
                            collection_time_ms: 72_000_000 + row as u32,
                            collection_date: 20_000,
                            moments: vec![ref_moment(&values, &codes), rho_moment(&rho)],
                        }
                    })
                    .collect(),
            })
            .collect();
        Level2File {
            station_id: "KTLX".into(),
            volume_date: 20_000,
            volume_time: 72_000_000,
            vol_site: None,
            sweeps,
        }
    }

    fn product(threads: usize, order: &[usize], reduce: Reduce) -> (Vec<f32>, [u64; 3], GateCounts) {
        let grid = flat_grid(60, 50, &[300.0, 800.0, 1500.0, 3000.0, 6000.0, 12000.0]);
        let params = GridParams { max_range_km: 80.0, max_elevation_deg: 20.0 };
        let sites = [(35.3, -97.3, 390.0), (35.6, -97.0, 420.0), (35.0, -97.6, 350.0)];
        let volumes: Vec<Level2File> = (0..3)
            .map(|v| synthetic_volume(v as u32 + 7, 90, 300, &[0.5, 1.5, 3.0, 25.0]))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let mut domain = Domain::new(grid.nz, grid.ny, grid.nx);
            let mut totals = GateCounts::default();
            let windows: Vec<(Window, GateCounts)> = order
                .par_iter()
                .map(|&v| grid_volume(&volumes[v], sites[v], &grid, &params).unwrap())
                .collect();
            // Merge in a fixed (site) order, as run_volumes does.
            let mut indexed: Vec<(usize, &(Window, GateCounts))> =
                order.iter().copied().zip(windows.iter()).collect();
            indexed.sort_by_key(|(v, _)| *v);
            for (_, (window, counts)) in indexed {
                domain.add(window);
                totals.add(counts);
            }
            let (field, census) = domain.reduce(reduce);
            (field, census, totals)
        })
    }

    #[test]
    fn the_product_is_byte_identical_across_thread_counts_and_volume_order() {
        for reduce in [Reduce::Mean, Reduce::Max] {
            let (a, census_a, counts_a) = product(1, &[0, 1, 2], reduce);
            let (b, census_b, counts_b) = product(8, &[2, 0, 1], reduce);
            let bits = |f: &[f32]| f.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "{reduce:?}");
            assert_eq!(census_a, census_b);
            assert_eq!(counts_a, counts_b);
            assert!(counts_a.balances(), "{counts_a:?}");
            assert!(census_a[0] > 0 && census_a[1] > 0 && census_a[2] > 0, "{census_a:?}");
            assert!(counts_a.dropped_range_folded > 0 && counts_a.dropped_nonmet_rho > 0);
            assert_eq!(counts_a.sweeps_skipped_elevation, 3, "the 25 deg sweep of each volume");
        }
    }

    #[test]
    fn mean_is_linear_z_and_max_is_max_and_clear_needs_no_echo() {
        let grid = flat_grid(40, 40, &[0.0, 20000.0]);
        let params = GridParams { max_range_km: 3.0, max_elevation_deg: 20.0 };
        // Four gates 2125..2875 m on one radial: 10 dBZ, 20 dBZ, below
        // threshold, range folded.  All four land in one tall 3 km cell or
        // its neighbour; the arithmetic is checked on whichever holds echo.
        let radial = RadialData {
            azimuth: 0.0,
            elevation: 0.5,
            azimuth_spacing: 1.0,
            nyquist_velocity: None,
            radial_status: 3,
            collection_time_ms: 0,
            collection_date: 1,
            moments: vec![
                ref_moment(
                    &[10.0, 20.0, f32::NAN, f32::NAN],
                    &[censor::MEASURED, censor::MEASURED, censor::BELOW_THRESHOLD, censor::RANGE_FOLDED],
                ),
                rho_moment(&[0.99, 0.5, 0.99, 0.99]),
            ],
        };
        let file = Level2File {
            station_id: "KTLX".into(),
            volume_date: 1,
            volume_time: 0,
            vol_site: None,
            sweeps: vec![Level2Sweep {
                elevation_number: 1,
                elevation_angle: 0.5,
                nyquist_velocity: None,
                nyquist_radials_disagree: false,
                sweep_index: 0,
                start_status: 3,
                end_status: 4,
                cut_sector: 0,
                radials: vec![radial],
            }],
        };
        let site = (35.3, -97.3, 390.0);
        let (window, counts) = grid_volume(&file, site, &grid, &params).unwrap();
        // The 20 dBZ gate has RhoHV 0.5 under the 45 dBZ shield: clutter.
        assert_eq!(counts.dropped_nonmet_rho, 1);
        assert_eq!(counts.dropped_range_folded, 1);
        assert_eq!(counts.echo_gates, 1);
        assert_eq!(counts.clear_gates, 1);
        assert!(counts.balances());
        let mut domain = Domain::new(grid.nz, grid.ny, grid.nx);
        domain.add(&window);
        let (mean, census) = domain.reduce(Reduce::Mean);
        let (max, _) = domain.reduce(Reduce::Max);
        assert_eq!(census[0], 1);
        let echo: Vec<f32> = mean.iter().copied().filter(|v| *v > -90.0).collect();
        assert_eq!(echo.len(), 1);
        assert!((echo[0] - 10.0).abs() < 1e-5, "{echo:?}");
        assert!(max.iter().any(|v| *v == 10.0));
        // A clear gate in a cell that also holds echo does not make it clear:
        // here they are in different cells or the same one; either way the
        // census accounts every cell once.
        assert_eq!(census.iter().sum::<u64>() as usize, mean.len());

        // Two echoes in one cell: mean is the mean of linear Z.
        let mut w = Window::new(1, (0, 0, 0, 0));
        w.linear_sum[0] = 10f64.powf(1.0) + 10f64.powf(3.0);
        w.echo_count[0] = 2;
        w.max_dbz[0] = 30.0;
        w.clear_count[0] = 5;
        let mut d = Domain::new(1, 1, 1);
        d.add(&w);
        let (m, c) = d.reduce(Reduce::Mean);
        assert!((m[0] as f64 - 10.0 * (505.0f64).log10()).abs() < 1e-5);
        assert_eq!(c, [1, 0, 0], "echo wins over clear in the same cell");
        let (x, _) = d.reduce(Reduce::Max);
        assert_eq!(x[0], 30.0);
        let mut clear = Window::new(1, (0, 0, 0, 0));
        clear.clear_count[0] = 1;
        let mut d = Domain::new(1, 1, 1);
        d.add(&clear);
        assert_eq!(d.reduce(Reduce::Mean).0[0], OBSERVED_NO_ECHO);
        assert_eq!(Domain::new(1, 1, 1).reduce(Reduce::Max).0[0], NO_COVERAGE);
    }

    #[test]
    fn grid_ref_options_parse_and_refuse() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let request = parse_request(&args(&[
            "--grid", "g.json", "--volumes", "a,b", "--out", "o", "--reduce", "max", "--threads", "3",
            "--window-end", "2026-10-01T20:00:00Z",
        ]))
        .unwrap();
        assert_eq!(request.volumes.len(), 2);
        assert_eq!(request.reduce, Reduce::Max);
        assert_eq!(request.threads, 3);
        assert_eq!(request.params.max_range_km, 250.0);
        assert!(request.window_end.is_some());
        assert!(parse_request(&args(&["--grid", "g", "--out", "o", "--reduce", "median"])).is_err());
        assert!(parse_request(&args(&["--volumes", "a", "--out", "o"])).is_err());
        assert!(parse_request(&args(&["--grid", "g", "--out", "o", "--threads", "0"])).is_err());
    }

    // Reference values computed by the Python implementation on box K.
    include!("grid_ref_python_fixtures.rs");
}
